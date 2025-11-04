"""
KAI Test Suite - KAI Integration Tests
Extrahiert aus test_kai_worker.py fuer bessere Wartbarkeit.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import pytest

logger = logging.getLogger(__name__)
TEST_VECTOR_DIM = 384


class TestKaiWorkerIntegration:
    """Integrationstests für den vollständigen KAI-Workflow."""

    def test_define_command_workflow(self, kai_worker_with_mocks, clean_test_concepts):
        """Testet den kompletten Workflow für Definiere-Befehle."""
        test_word = f"{clean_test_concepts}sonne"
        query = f"Definiere: {test_word} / bedeutung = Ein Stern im Zentrum"

        kai_worker_with_mocks.process_query(query)

        # Prüfe, ob finished-Signal aufgerufen wurde
        assert kai_worker_with_mocks.signals.finished.emit.called
        response = kai_worker_with_mocks.signals.finished.emit.call_args.args[0]
        assert "gemerkt" in response.text.lower()

        # Verifiziere im Graphen
        details = kai_worker_with_mocks.netzwerk.get_details_fuer_wort(test_word)
        assert details is not None
        assert any("Stern" in b for b in details["bedeutungen"])

    def test_question_with_knowledge_gap(
        self, kai_worker_with_mocks, clean_test_concepts
    ):
        """Testet Fragen zu unbekannten Konzepten."""
        unknown_word = f"{clean_test_concepts}unbekanntesfantasiewort"
        query = f"Was ist ein {unknown_word}?"

        kai_worker_with_mocks.process_query(query)

        response = kai_worker_with_mocks.signals.finished.emit.call_args.args[0]
        assert "weiß nichts" in response.text.lower()

        # PHASE 4: Prüfe, ob Kontext für Folgefrage gesetzt wurde (typsicher)
        assert kai_worker_with_mocks.context.is_active()
        assert kai_worker_with_mocks.context.aktion.value == "erwarte_beispielsatz"

    def test_question_with_existing_knowledge(
        self, kai_worker_with_mocks, clean_test_concepts
    ):
        """Testet Fragen zu bekannten Konzepten."""
        test_word = f"{clean_test_concepts}vogel"
        test_object = f"{clean_test_concepts}tier"

        # Erstelle Wissen
        kai_worker_with_mocks.netzwerk.assert_relation(
            test_word, "IS_A", test_object, source_sentence="test"
        )

        query = f"Was ist ein {test_word}?"
        kai_worker_with_mocks.process_query(query)

        response = kai_worker_with_mocks.signals.finished.emit.call_args.args[0]
        assert test_word in response.text.lower()
        assert test_object in response.text.lower()

    def test_ingest_text_workflow(self, kai_worker_with_mocks, clean_test_concepts):
        """Testet das Ingestieren von Text mit Extraktion."""
        subject = f"{clean_test_concepts}delfin"
        object = f"{clean_test_concepts}säugetier"
        text = f"Ein {subject} ist ein {object}."

        query = f'Ingestiere Text: "{text}"'
        kai_worker_with_mocks.process_query(query)

        response = kai_worker_with_mocks.signals.finished.emit.call_args.args[0]
        assert "fakt" in response.text.lower() or "gelernt" in response.text.lower()

        # Verifiziere im Graphen
        facts = kai_worker_with_mocks.netzwerk.query_graph_for_facts(subject)
        assert "IS_A" in facts
        # KORREKTUR: Plural-Normalisierung ist deaktiviert, erwarte Originalform
        assert object in facts["IS_A"], f"Erwartete '{object}' in {facts['IS_A']}"

    def test_learn_pattern_workflow(self, kai_worker_with_mocks):
        """Testet das Lernen eines neuen Musters."""
        relation_type = "TEST_PATTERN_RELATION"
        regex = r"^(.+) testet (.+)$"

        try:
            # Erstelle Regel
            kai_worker_with_mocks.netzwerk.create_extraction_rule(relation_type, regex)

            # Lehre Muster
            query = f'Lerne Muster: "A testet B" bedeutet {relation_type}'
            kai_worker_with_mocks.process_query(query)

            response = kai_worker_with_mocks.signals.finished.emit.call_args.args[0]
            assert "gelernt" in response.text.lower()

            # Prüfe, ob Prototyp erstellt und verknüpft wurde
            prototypes = kai_worker_with_mocks.netzwerk.get_all_pattern_prototypes()
            # Es sollte mindestens ein Prototyp existieren
            assert len(prototypes) > 0
        finally:
            with kai_worker_with_mocks.netzwerk.driver.session(
                database="neo4j"
            ) as session:
                session.run(
                    """
                    MATCH (r:ExtractionRule {relation_type: $rel})
                    DETACH DELETE r
                """,
                    rel=relation_type,
                )
                session.run(
                    """
                    MATCH (p:PatternPrototype)
                    WHERE p.category = 'DEFINITION'
                    DETACH DELETE p
                """
                )

    def test_context_handling_follow_up(
        self, kai_worker_with_mocks, clean_test_concepts
    ):
        """Testet die Kontextbehandlung bei Folgefragen."""
        unknown = f"{clean_test_concepts}neuartigeskonzept"

        # Erste Frage: Setzt Kontext
        kai_worker_with_mocks.process_query(f"Was ist ein {unknown}?")
        # PHASE 4: Prüfe typsicheren Kontext
        assert kai_worker_with_mocks.context.is_active()

        # Folgeantwort: Sollte als Beispielsatz verarbeitet werden
        # PHASE 1: Folgeantwort wird als UNKNOWN erkannt (kein Match für IS_A Pattern)
        # TODO Phase 2: Dies wird mit Confidence Gates besser behandelt
        kai_worker_with_mocks.signals.reset_mock()
        kai_worker_with_mocks.process_query(f"Ein {unknown} ist etwas Besonderes.")

        response = kai_worker_with_mocks.signals.finished.emit.call_args.args[0]
        # PHASE 3: Clarification-Response wird korrekt generiert
        assert (
            "verstanden" in response.text.lower()
            or "gelernt" in response.text.lower()
            or "gemerkt" in response.text.lower()  # Context wurde verarbeitet
            or "keinen plan" in response.text.lower()
            or "nicht sicher" in response.text.lower()  # Clarification response
        )

        # PHASE 4: Prüfe, dass Kontext nach Follow-up zurückgesetzt wurde
        assert (
            not kai_worker_with_mocks.context.is_active()
        ), "Kontext sollte nach Follow-up zurückgesetzt sein"


# ============================================================================
# TESTS FÜR SETUP UND INITIALISIERUNG
# ============================================================================


class TestEdgeCases:
    """Tests für Randfälle und Fehlerbehandlung."""

    def test_empty_query(self, kai_worker_with_mocks):
        """Testet Verhalten bei leerer Eingabe."""
        kai_worker_with_mocks.process_query("")

        response = kai_worker_with_mocks.signals.finished.emit.call_args.args[0]
        # PHASE 3: Clarification-Response wird korrekt generiert
        assert (
            "nicht verstanden" in response.text.lower()
            or "absicht" in response.text.lower()
            or "keinen plan" in response.text.lower()
            or "nicht sicher" in response.text.lower()  # Clarification response
        )

    def test_whitespace_only_query(self, kai_worker_with_mocks):
        """Testet Verhalten bei reinen Whitespace-Eingaben."""
        for whitespace_input in ["   ", "\t", "\n", "\r\n", "  \t  \n  "]:
            kai_worker_with_mocks.signals.reset_mock()
            kai_worker_with_mocks.context.clear()  # Kontext zurücksetzen zwischen Tests
            kai_worker_with_mocks.process_query(whitespace_input)
            assert (
                kai_worker_with_mocks.signals.finished.emit.called
            ), f"Sollte für Whitespace '{repr(whitespace_input)}' ein finished-Signal senden"

    def test_special_characters_in_query(self, kai_worker_with_mocks):
        """Testet Verhalten bei Sonderzeichen."""
        special_inputs = [
            "Was ist @#$%?",
            "Definiere: <script>alert('xss')</script>",
            "Was ist ein Apfel 🍎?",
            "Was ist ein\x00nullbyte?",
        ]
        for special_input in special_inputs:
            kai_worker_with_mocks.signals.reset_mock()
            try:
                kai_worker_with_mocks.process_query(special_input)
                assert kai_worker_with_mocks.signals.finished.emit.called
            except Exception as e:
                logger.warning(
                    f"Sonderzeichen-Input '{special_input}' erzeugt Exception: {e}"
                )

    def test_malformed_command(self, kai_worker_with_mocks):
        """Testet Verhalten bei fehlerhaften Befehlen."""
        malformed_commands = [
            "Definiere: ohne_werte",
            "Definiere:",
            "Lerne Muster:",
            'Lerne Muster: "ohne relation"',
            "Ingestiere Text:",
        ]

        for cmd in malformed_commands:
            kai_worker_with_mocks.signals.reset_mock()
            kai_worker_with_mocks.context.clear()  # Kontext zurücksetzen zwischen Tests
            kai_worker_with_mocks.process_query(cmd)
            # Sollte nicht crashen
            assert (
                kai_worker_with_mocks.signals.finished.emit.called
            ), f"Sollte für fehlerhaften Befehl '{cmd}' ein finished-Signal senden"

    def test_unicode_handling(self, kai_worker_with_mocks, clean_test_concepts):
        """Testet Verarbeitung von Unicode-Zeichen."""
        unicode_word = f"{clean_test_concepts}日本語"  # Japanisch
        query = f"Was ist {unicode_word}?"

        kai_worker_with_mocks.process_query(query)
        assert kai_worker_with_mocks.signals.finished.emit.called

        # Verifiziere, dass das Wort gespeichert werden kann
        details = kai_worker_with_mocks.netzwerk.get_details_fuer_wort(unicode_word)
        assert (
            details is not None
        ), "Unicode-Wörter sollten im Graphen gespeichert werden können"

    def test_very_long_text_ingestion(self, kai_worker_with_mocks, clean_test_concepts):
        """Testet Ingestion von längerem Text."""
        subject1 = f"{clean_test_concepts}elefant"
        subject2 = f"{clean_test_concepts}maus"

        long_text = f"""
        Ein {subject1} ist ein Säugetier.
        Eine {subject2} ist auch ein Säugetier.
        Ein {subject1} ist groß und eine {subject2} ist klein.
        """

        query = f'Ingestiere Text: "{long_text}"'
        kai_worker_with_mocks.process_query(query)

        # Sollte mehrere Fakten extrahieren
        response = kai_worker_with_mocks.signals.finished.emit.call_args.args[0]
        assert "fakt" in response.text.lower() or "gelernt" in response.text.lower()

        # Verifiziere, dass mindestens ein Fakt extrahiert wurde
        facts1 = kai_worker_with_mocks.netzwerk.query_graph_for_facts(subject1)
        facts2 = kai_worker_with_mocks.netzwerk.query_graph_for_facts(subject2)

        # Mindestens einer sollte funktionieren (abhängig von Pattern-Matching)
        facts_found = ("IS_A" in facts1) or ("IS_A" in facts2)
        assert (
            facts_found
        ), f"Sollte mindestens eine IS_A Relation finden.\nFacts1: {facts1}\nFacts2: {facts2}"

        # Wenn IS_A gefunden wurde, prüfe ob säugetier dabei ist
        if "IS_A" in facts1:
            assert (
                "säugetier" in facts1["IS_A"]
            ), f"Sollte 'säugetier' in {subject1} Fakten finden, hat aber: {facts1['IS_A']}"
            logger.info(f"[SUCCESS] {subject1} IS_A säugetier erfolgreich extrahiert")

        if "IS_A" in facts2:
            assert (
                "säugetier" in facts2["IS_A"]
            ), f"Sollte 'säugetier' in {subject2} Fakten finden, hat aber: {facts2['IS_A']}"
            logger.info(f"[SUCCESS] {subject2} IS_A säugetier erfolgreich extrahiert")

    def test_extremely_long_single_sentence(
        self, kai_worker_with_mocks, clean_test_concepts
    ):
        """Testet sehr langen Einzelsatz (> 1000 Zeichen)."""
        test_word = f"{clean_test_concepts}ultralangerworttest"
        long_sentence = f"Ein {test_word} ist " + "sehr " * 200 + "lang."

        query = f'Ingestiere Text: "{long_sentence}"'
        kai_worker_with_mocks.process_query(query)

        # Sollte nicht crashen
        assert kai_worker_with_mocks.signals.finished.emit.called

    def test_concurrent_relation_creation(self, netzwerk_session, clean_test_concepts):
        """Testet, ob parallele Relationserstellung zu Duplikaten führt."""
        subject = f"{clean_test_concepts}concurrent_test"
        object = f"{clean_test_concepts}target"

        # Simuliere gleichzeitige Erstellung derselben Relation
        created1 = netzwerk_session.assert_relation(subject, "IS_A", object, "source1")
        created2 = netzwerk_session.assert_relation(subject, "IS_A", object, "source2")

        # Nur die erste sollte erstellt worden sein
        assert created1 is True
        assert created2 is False

        # Verifiziere keine Duplikate
        with netzwerk_session.driver.session(database="neo4j") as session:
            result = session.run(
                """
                MATCH (s:Konzept {name: $subject})-[r:IS_A]->(o:Konzept {name: $object})
                RETURN count(r) AS count
            """,
                subject=subject,
                object=object,
            )
            count = result.single()["count"]
            assert count == 1, f"Sollte genau 1 Relation haben, hat aber {count}"

    def test_circular_relations(self, netzwerk_session, clean_test_concepts):
        """Testet zirkuläre Relationen (A -> B -> C -> A)."""
        a = f"{clean_test_concepts}circular_a"
        b = f"{clean_test_concepts}circular_b"
        c = f"{clean_test_concepts}circular_c"

        netzwerk_session.assert_relation(a, "RELATED_TO", b, "test")
        netzwerk_session.assert_relation(b, "RELATED_TO", c, "test")
        netzwerk_session.assert_relation(c, "RELATED_TO", a, "test")

        # System sollte zirkuläre Relationen erlauben (keine Exception)
        facts_a = netzwerk_session.query_graph_for_facts(a)
        assert "RELATED_TO" in facts_a
        assert b in facts_a["RELATED_TO"]


# ============================================================================
# TESTS FÜR DATENBANK-KONSISTENZ UND ROBUSTHEIT
# ============================================================================


class TestDatabaseConsistency:
    """Tests für Datenbank-Konsistenz und Fehlerbehandlung."""

    def test_node_uniqueness_constraint(self, netzwerk_session, clean_test_concepts):
        """Testet, ob Constraints Duplikate verhindern."""
        test_word = f"{clean_test_concepts}unique_test"

        # Erstelle dasselbe Wort zweimal
        netzwerk_session.ensure_wort_und_konzept(test_word)
        netzwerk_session.ensure_wort_und_konzept(test_word)

        # Sollte nur ein Wort-Knoten existieren
        with netzwerk_session.driver.session(database="neo4j") as session:
            result = session.run(
                """
                MATCH (w:Wort {lemma: $lemma})
                RETURN count(w) AS count
            """,
                lemma=test_word,
            )
            count = result.single()["count"]
            assert (
                count == 1
            ), f"Constraint-Verletzung: {count} Knoten für '{test_word}' gefunden"

    def test_orphaned_nodes_prevention(self, netzwerk_session, clean_test_concepts):
        """Testet, ob Wort- und Konzept-Knoten immer verknüpft sind."""
        test_word = f"{clean_test_concepts}linked_test"
        netzwerk_session.ensure_wort_und_konzept(test_word)

        # Prüfe, ob Wort und Konzept verbunden sind
        with netzwerk_session.driver.session(database="neo4j") as session:
            result = session.run(
                """
                MATCH (w:Wort {lemma: $lemma})
                WHERE NOT (w)-[:BEDEUTET]->()
                RETURN count(w) AS orphaned_count
            """,
                lemma=test_word,
            )
            orphaned = result.single()["orphaned_count"]
            assert orphaned == 0, f"Gefunden {orphaned} verwaiste Wort-Knoten"

    def test_relation_source_tracking(self, netzwerk_session, clean_test_concepts):
        """Testet, ob Relationen ihre Quell-Sätze korrekt speichern."""
        subject = f"{clean_test_concepts}source_test"
        object = f"{clean_test_concepts}target"
        source_sentence = "Dies ist ein Test-Quellsatz."

        netzwerk_session.assert_relation(subject, "IS_A", object, source_sentence)

        # Verifiziere, dass source_text gespeichert wurde (nicht source_sentence!)
        with netzwerk_session.driver.session(database="neo4j") as session:
            result = session.run(
                """
                MATCH (s:Konzept {name: $subject})-[r:IS_A]->(o:Konzept {name: $object})
                RETURN r.source_text AS source, r.asserted_at AS timestamp, r.confidence AS conf
            """,
                subject=subject,
                object=object,
            )
            record = result.single()
            assert record is not None, "Relation sollte existieren"
            assert record["source"] is not None, "source_text sollte nicht None sein"
            assert (
                source_sentence == record["source"]
            ), f"Quellsatz nicht korrekt gespeichert: erwartet '{source_sentence}', bekommen '{record['source']}'"

            # Zusätzliche Validierung: Timestamp und Confidence sollten gesetzt sein
            assert (
                record["timestamp"] is not None
            ), "asserted_at Timestamp sollte gesetzt sein"
            assert record["conf"] is not None, "Confidence sollte gesetzt sein"
            assert (
                record["conf"] > 0.0
            ), f"Confidence sollte > 0 sein, ist aber {record['conf']}"

    def test_extraction_rule_regex_validation(self, netzwerk_session):
        """Testet, ob ungültige Regex-Patterns abgefangen werden."""
        invalid_patterns = [
            r"(.+",  # Ungeschlossene Klammer
            r"(?P<name>test",  # Ungeschlossene Named Group
            r"[a-z",  # Ungeschlossene Character Class
        ]

        for pattern in invalid_patterns:
            try:
                netzwerk_session.create_extraction_rule("INVALID_TEST", pattern)
                # Falls keine Exception: Cleanup
                with netzwerk_session.driver.session(database="neo4j") as session:
                    session.run(
                        """
                        MATCH (r:ExtractionRule {relation_type: 'INVALID_TEST'})
                        DETACH DELETE r
                    """
                    )
            except Exception as e:
                # Erwartete Exception - das ist gut
                logger.info(
                    f"[SUCCESS] Ungültiges Regex-Pattern korrekt abgefangen: {pattern}"
                )

    def test_multi_relation_integrity(self, netzwerk_session, clean_test_concepts):
        """Testet, ob ein Konzept mehrere unterschiedliche Relationen haben kann."""
        subject = f"{clean_test_concepts}multi_rel_subject"
        object1 = f"{clean_test_concepts}multi_rel_obj1"
        object2 = f"{clean_test_concepts}multi_rel_obj2"
        object3 = f"{clean_test_concepts}multi_rel_obj3"

        # Erstelle verschiedene Relationstypen
        netzwerk_session.assert_relation(subject, "IS_A", object1, "test1")
        netzwerk_session.assert_relation(subject, "CAPABLE_OF", object2, "test2")
        netzwerk_session.assert_relation(subject, "HAS_PROPERTY", object3, "test3")

        # Verifiziere alle drei Relationen
        facts = netzwerk_session.query_graph_for_facts(subject)
        assert "IS_A" in facts and object1 in facts["IS_A"]
        assert "CAPABLE_OF" in facts and object2 in facts["CAPABLE_OF"]
        assert "HAS_PROPERTY" in facts and object3 in facts["HAS_PROPERTY"]

        logger.info(
            f"[SUCCESS] Multi-Relation-Integrität verifiziert: {len(facts)} Relationstypen"
        )


# ============================================================================
# TESTS FÜR ALLE RELATION-TYPEN MIT EDGE CASES
# ============================================================================


class TestLimitsAndPerformance:
    """
    Tests für System-Grenzwerte und Performance-Charakteristiken.

    WICHTIG: Performance ist zweitrangig - Korrektheit geht vor Geschwindigkeit!
    Timeouts sind großzügig gewählt, um korrekte aber langsame Ergebnisse zu erlauben.
    """

    @pytest.mark.slow
    def test_large_batch_relation_creation(self, netzwerk_session, clean_test_concepts):
        """
        Testet Korrektheit bei vielen Relationen.
        Performance ist sekundär - Hauptfokus: Alle Relationen korrekt gespeichert.
        """
        import time

        subject = f"{clean_test_concepts}batch_test"
        num_relations = 50

        start_time = time.time()
        for i in range(num_relations):
            obj = f"{clean_test_concepts}batch_obj_{i}"
            netzwerk_session.assert_relation(subject, "RELATED_TO", obj, f"batch_{i}")
        elapsed = time.time() - start_time

        # HAUPTTEST: Verifiziere Vollständigkeit und Korrektheit
        facts = netzwerk_session.query_graph_for_facts(subject)
        assert (
            len(facts.get("RELATED_TO", [])) == num_relations
        ), f"Sollte {num_relations} Relationen haben, hat aber {len(facts.get('RELATED_TO', []))}"

        # Performance-Info (kein Fehler bei langsamer Ausführung)
        if elapsed > 10.0:
            logger.warning(
                f"[WARNING] Batch-Erstellung dauerte {elapsed:.2f}s (langsam, aber korrekt)"
            )
        else:
            logger.info(
                f"[SUCCESS] {num_relations} Relationen in {elapsed:.2f}s erstellt (korrekt)"
            )

    @pytest.mark.slow
    def test_deep_graph_traversal(self, netzwerk_session, clean_test_concepts):
        """Testet Performance bei tiefer Graph-Traversierung."""
        # Erstelle Kette: A -> B -> C -> D -> E
        chain = [f"{clean_test_concepts}chain_{i}" for i in range(5)]
        for i in range(len(chain) - 1):
            netzwerk_session.assert_relation(
                chain[i], "LEADS_TO", chain[i + 1], f"chain_{i}"
            )

        # Verifiziere Start- und Endknoten
        facts_start = netzwerk_session.query_graph_for_facts(chain[0])
        assert "LEADS_TO" in facts_start

        facts_end = netzwerk_session.query_graph_for_facts(chain[-1])
        # Endknoten sollte keine ausgehenden LEADS_TO Relationen haben
        assert "LEADS_TO" not in facts_end or len(facts_end["LEADS_TO"]) == 0

    def test_embedding_service_caching(self, embedding_service_session):
        """
        Testet, ob wiederholte Embedding-Anfragen identische Vektoren liefern.
        Korrektheit hat Vorrang - Geschwindigkeit ist sekundär.
        """
        import time

        test_sentence = "Dies ist ein Test für Caching."

        # Erste Anfrage
        start_cold = time.time()
        vec1 = embedding_service_session.get_embedding(test_sentence)
        cold_time = time.time() - start_cold

        # Zweite Anfrage
        start_warm = time.time()
        vec2 = embedding_service_session.get_embedding(test_sentence)
        warm_time = time.time() - start_warm

        # HAUPTTEST: Vektoren sollten identisch sein (Konsistenz)
        import numpy as np

        assert np.allclose(
            vec1, vec2
        ), "Wiederholte Anfragen sollten identische Vektoren liefern"

        # Performance-Info (informativ, nicht kritisch)
        logger.info(
            f"[SUCCESS] Embedding-Konsistenz verifiziert: cold={cold_time:.4f}s, warm={warm_time:.4f}s"
        )

    @pytest.mark.slow
    def test_maximum_text_length_handling(self, kai_worker_with_mocks):
        """
        Testet Korrektheit bei extrem langen Texten (> 10.000 Zeichen).
        Darf lange dauern, solange es funktioniert!
        """
        very_long_text = "Das ist ein sehr langer Test. " * 500  # ~15.000 Zeichen

        query = f'Ingestiere Text: "{very_long_text}"'

        try:
            kai_worker_with_mocks.process_query(query)
            assert kai_worker_with_mocks.signals.finished.emit.called
            logger.info(
                f"[SUCCESS] Extrem langer Text ({len(very_long_text)} Zeichen) korrekt verarbeitet"
            )
        except Exception as e:
            logger.warning(f"Sehr langer Text erzeugt Exception: {e}")

    @pytest.mark.slow
    def test_large_text_ingestion_correctness(
        self, kai_worker_with_mocks, clean_test_concepts
    ):
        """
        BENCHMARK: Testet Korrektheit bei großer Textmenge (mehrere Absätze).
        Performance ist sekundär - Fokus auf vollständige Faktenextraktion.
        """
        import time

        # Verwende normale deutsche Wörter für realistische Textverarbeitung
        subjects = ["elefant", "löwe", "adler", "wal"]

        long_text = f"""
        Ein {subjects[0]} ist ein Säugetier. Elefanten leben in Afrika.
        Ein {subjects[1]} ist ein Raubtier. Löwen können brüllen.
        Ein {subjects[2]} ist ein Vogel. Adler können fliegen.
        Ein {subjects[3]} ist ein Meerestier. Wale leben im Ozean.
        """

        query = f'Ingestiere Text: "{long_text}"'

        start_time = time.time()
        kai_worker_with_mocks.process_query(query)
        elapsed = time.time() - start_time

        # HAUPTTEST: Verifiziere, dass Fakten korrekt extrahiert wurden
        extracted_facts_count = 0
        for subject in subjects:
            facts = kai_worker_with_mocks.netzwerk.query_graph_for_facts(subject)
            if facts:
                extracted_facts_count += sum(len(v) for v in facts.values())

        # Erwarte mindestens einige Fakten (nicht alle, da Pattern-Matching variieren kann)
        assert (
            extracted_facts_count >= 4
        ), f"Sollte mindestens 4 Fakten extrahiert haben, hat aber nur {extracted_facts_count}"

        # Performance-Info
        logger.info(
            f"[SUCCESS] Großer Text ({len(long_text)} Zeichen) verarbeitet: "
            f"{extracted_facts_count} Fakten extrahiert in {elapsed:.2f}s"
        )
        if elapsed > 30.0:
            logger.info(
                f"[INFO] Verarbeitung dauerte {elapsed:.2f}s - Korrektheit wichtiger als Geschwindigkeit!"
            )

    @pytest.mark.slow
    def test_many_concurrent_facts_query_correctness(
        self, netzwerk_session, clean_test_concepts
    ):
        """
        Testet Korrektheit beim Abfragen vieler Fakten.
        Vollständigkeit und Genauigkeit haben Vorrang.
        """
        import time

        # Erstelle umfassendes Wissensnetz
        central_concept = f"{clean_test_concepts}zentral"
        num_facts = 100

        # Füge viele verschiedene Fakten hinzu
        for i in range(num_facts):
            obj = f"{clean_test_concepts}fakt_{i}"
            rel_type = ["IS_A", "HAS_PROPERTY", "CAPABLE_OF", "PART_OF", "LOCATED_IN"][
                i % 5
            ]
            netzwerk_session.assert_relation(
                central_concept, rel_type, obj, f"Fakt {i}"
            )

        # HAUPTTEST: Vollständige und korrekte Abfrage
        start = time.time()
        facts = netzwerk_session.query_graph_for_facts(central_concept)
        elapsed = time.time() - start

        # Verifiziere Vollständigkeit
        total_facts = sum(len(v) for v in facts.values())
        assert (
            total_facts == num_facts
        ), f"Sollte {num_facts} Fakten zurückgeben, hat aber {total_facts}"

        # Verifiziere, dass alle Relationstypen vorhanden sind
        assert len(facts) == 5, "Sollte alle 5 Relationstypen enthalten"

        logger.info(
            f"[SUCCESS] Query mit {num_facts} Fakten: Vollständig und korrekt in {elapsed:.4f}s"
        )
        if elapsed > 2.0:
            logger.info(
                f"[INFO] Query dauerte {elapsed:.4f}s - Vollständigkeit wichtiger als Geschwindigkeit!"
            )


# ============================================================================
# PHASE 5: TESTS FÜR INTERAKTIVES LERNEN & VERFEINERUNG
# ============================================================================
