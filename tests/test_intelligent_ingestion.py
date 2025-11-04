"""
KAI Test Suite - Intelligent Ingestion Tests
Extrahiert aus test_kai_worker.py fuer bessere Wartbarkeit.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import pytest

logger = logging.getLogger(__name__)
TEST_VECTOR_DIM = 384


class TestIntelligentIngestion:
    """Tests für Phase 6: Vector-basierte Ingestion und Synonym-Erweiterung."""

    def test_get_rule_for_prototype_exists(
        self, netzwerk_session, embedding_service_session
    ):
        """
        PHASE 6: Testet ob get_rule_for_prototype() korrekt funktioniert.
        Dies ist die Brücke zwischen Prototypen und Extraktionsregeln.
        """
        from component_8_prototype_matcher import PrototypingEngine

        # Setup: Erstelle Regel und Prototyp
        relation_type = "TEST_GET_RULE"
        regex = r"^(.+) testet_rule (.+)$"

        try:
            # Erstelle Regel
            success = netzwerk_session.create_extraction_rule(relation_type, regex)
            assert success, "Regel konnte nicht erstellt werden"

            # Erstelle Prototyp und verknüpfe
            engine = PrototypingEngine(netzwerk_session, embedding_service_session)
            test_vector = embedding_service_session.get_embedding("A testet_rule B")
            prototype_id = engine.process_vector(test_vector, "DEFINITION")

            # Verknüpfe Prototyp mit Regel
            link_success = netzwerk_session.link_prototype_to_rule(
                prototype_id, relation_type
            )
            assert link_success, "Verknüpfung fehlgeschlagen"

            # TEST: get_rule_for_prototype() sollte die Regel finden
            rule = netzwerk_session.get_rule_for_prototype(prototype_id)

            assert rule is not None, "get_rule_for_prototype() gab None zurück"
            assert rule["relation_type"] == relation_type
            assert rule["regex_pattern"] == regex

            logger.info(
                f"[SUCCESS] get_rule_for_prototype() findet Regel: {relation_type}"
            )

        finally:
            # Cleanup
            with netzwerk_session.driver.session(database="neo4j") as session:
                session.run(
                    """
                    MATCH (r:ExtractionRule {relation_type: $rel})
                    DETACH DELETE r
                """,
                    rel=relation_type,
                )
                session.run(
                    """
                    MATCH (p:PatternPrototype {category: 'DEFINITION'})
                    DETACH DELETE p
                """
                )

    def test_intelligent_ingestion_uses_prototypes(
        self, kai_worker_with_mocks, clean_test_concepts
    ):
        """
        PHASE 6: Testet ob intelligente Ingestion Prototypen statt Brute-Force nutzt.
        """
        # Setup: Lehre KAI ein Muster
        relation_type = "TEST_INTELLIGENT"
        regex = r"^(.+) ist_intelligent (.+)$"

        try:
            # Erstelle Regel
            kai_worker_with_mocks.netzwerk.create_extraction_rule(relation_type, regex)

            # Lehre Muster
            learn_query = f'Lerne Muster: "Ein Hund ist_intelligent ein Tier" bedeutet {relation_type}'
            kai_worker_with_mocks.process_query(learn_query)

            # Reset Signals
            kai_worker_with_mocks.signals.reset_mock()

            # Jetzt ingestiere ähnlichen Satz
            subject = f"{clean_test_concepts}katze_intelligent"
            obj = f"{clean_test_concepts}haustier_intelligent"
            text = f"Eine {subject} ist_intelligent ein {obj}."

            ingest_query = f'Ingestiere Text: "{text}"'
            kai_worker_with_mocks.process_query(ingest_query)

            # Verifiziere, dass Fakt extrahiert wurde
            facts = kai_worker_with_mocks.netzwerk.query_graph_for_facts(subject)

            assert (
                relation_type in facts
            ), f"Relation {relation_type} nicht gefunden. Facts: {facts}"
            assert obj in facts[relation_type], f"{obj} nicht in {facts[relation_type]}"

            logger.info("[SUCCESS] Intelligente Ingestion nutzt gelerntes Muster")

        finally:
            # Cleanup
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
                    WHERE p.category IN ['DEFINITION', 'TEST_INTELLIGENT']
                    DETACH DELETE p
                """
                )

    def test_query_facts_with_synonyms_merges_knowledge(
        self, netzwerk_session, clean_test_concepts
    ):
        """
        PHASE 5.3 (Aktion 3): Testet query_facts_with_synonyms().
        Fakten über Synonyme sollten zusammengeführt werden.
        """
        word1 = f"{clean_test_concepts}auto_syn"
        word2 = f"{clean_test_concepts}pkw_syn"
        word3 = f"{clean_test_concepts}wagen_syn"

        # Erstelle Synonyme
        netzwerk_session.add_information_zu_wort(word1, "synonym", word2)
        netzwerk_session.add_information_zu_wort(word1, "synonym", word3)

        # Füge verschiedene Fakten zu verschiedenen Synonymen hinzu
        netzwerk_session.assert_relation(
            word1, "IS_A", f"{clean_test_concepts}fahrzeug", "test1"
        )
        netzwerk_session.assert_relation(
            word2, "HAS_PROPERTY", f"{clean_test_concepts}schnell", "test2"
        )
        netzwerk_session.assert_relation(
            word3, "CAPABLE_OF", f"{clean_test_concepts}fahren", "test3"
        )

        # TEST: query_facts_with_synonyms() sollte ALLE Fakten zurückgeben
        result = netzwerk_session.query_facts_with_synonyms(word1)

        assert result is not None
        assert result["primary_topic"] == word1
        assert set(result["synonyms"]) == {word2, word3}

        facts = result["facts"]
        assert "IS_A" in facts
        assert "HAS_PROPERTY" in facts
        assert "CAPABLE_OF" in facts

        # Verifiziere, dass Fakten von allen Synonymen kommen
        assert f"{clean_test_concepts}fahrzeug" in facts["IS_A"]
        assert f"{clean_test_concepts}schnell" in facts["HAS_PROPERTY"]
        assert f"{clean_test_concepts}fahren" in facts["CAPABLE_OF"]

        logger.info(
            f"[SUCCESS] Synonym-Fakten zusammengeführt: {len(facts)} Relationstypen"
        )

    def test_answer_generation_uses_synonym_facts(
        self, kai_worker_with_mocks, clean_test_concepts
    ):
        """
        PHASE 5.3 (Aktion 3): Testet ob Antwortgenerierung Synonym-Fakten nutzt.
        Verwendet deutsche Wörter ohne problematische Endungen.
        """
        word1 = f"{clean_test_concepts}hund_syn"
        word2 = f"{clean_test_concepts}koeter_syn"

        # Erstelle Synonym
        kai_worker_with_mocks.netzwerk.add_information_zu_wort(word1, "synonym", word2)

        # Füge Fakten zu beiden hinzu
        kai_worker_with_mocks.netzwerk.assert_relation(
            word1, "IS_A", f"{clean_test_concepts}tier_syn", "test1"
        )
        kai_worker_with_mocks.netzwerk.assert_relation(
            word2, "CAPABLE_OF", f"{clean_test_concepts}bellen_syn", "test2"
        )

        # Frage nach word1
        query = f"Was ist ein {word1}?"
        kai_worker_with_mocks.process_query(query)

        response = kai_worker_with_mocks.signals.finished.emit.call_args.args[0]

        # Antwort sollte BEIDE Fakten enthalten (von word1 UND word2)
        response_lower = response.text.lower()

        assert (
            "tier_syn" in response_lower
        ), f"Sollte IS_A Fakt enthalten. Response: {response.text}"
        assert (
            "bellen_syn" in response_lower
        ), f"Sollte CAPABLE_OF Fakt von Synonym enthalten. Response: {response.text}"

        # Sollte auch Synonym erwähnen
        assert (
            word2 in response_lower
        ), f"Sollte Synonym erwähnen. Response: {response.text}"

        logger.info("[SUCCESS] Antwortgenerierung nutzt Synonym-erweiterte Fakten")

    def test_synonym_bidirectionality(self, netzwerk_session, clean_test_concepts):
        """
        PHASE 5.3 (Aktion 3): Testet bidirektionale Synonym-Suche.
        A -> Synonymgruppe <- B bedeutet: Suche nach A findet B, und umgekehrt.
        """
        word_a = f"{clean_test_concepts}bidir_a"
        word_b = f"{clean_test_concepts}bidir_b"

        # Erstelle Synonym A -> B
        netzwerk_session.add_information_zu_wort(word_a, "synonym", word_b)

        # TEST 1: Suche von A sollte B finden
        result_a = netzwerk_session.query_facts_with_synonyms(word_a)
        assert (
            word_b in result_a["synonyms"]
        ), f"Suche von A sollte B finden. Synonyms: {result_a['synonyms']}"

        # TEST 2: Suche von B sollte A finden (bidirektional!)
        result_b = netzwerk_session.query_facts_with_synonyms(word_b)
        assert (
            word_a in result_b["synonyms"]
        ), f"Suche von B sollte A finden. Synonyms: {result_b['synonyms']}"

        logger.info("[SUCCESS] Synonym-Suche ist bidirektional")

    @pytest.mark.slow
    def test_ingestion_performance_with_learned_patterns(
        self, kai_worker_with_mocks, clean_test_concepts
    ):
        """
        PHASE 6: Testet Korrektheit der Ingestion mit gelernten Mustern.
        Geschwindigkeit ist sekundär - Hauptfokus: Richtige Regel wird verwendet.
        """
        import time

        # Setup: Erstelle mehrere Regeln
        relations = ["TEST_PERF_1", "TEST_PERF_2", "TEST_PERF_3"]
        for rel in relations:
            kai_worker_with_mocks.netzwerk.create_extraction_rule(
                rel, r"^(.+) perf_test (.+)$"
            )

        try:
            # Lehre Muster für eine spezifische Regel
            learn_query = 'Lerne Muster: "A perf_test B" bedeutet TEST_PERF_1'
            kai_worker_with_mocks.process_query(learn_query)

            # Ingestiere ähnlichen Text
            text = f"Ein {clean_test_concepts}perf_subject perf_test {clean_test_concepts}perf_object."

            start = time.time()
            kai_worker_with_mocks.signals.reset_mock()
            kai_worker_with_mocks.process_query(f'Ingestiere Text: "{text}"')
            elapsed = time.time() - start

            # HAUPTTEST: Verifiziere Korrektheit (richtige Regel verwendet)
            facts = kai_worker_with_mocks.netzwerk.query_graph_for_facts(
                f"{clean_test_concepts}perf_subject"
            )
            assert (
                "TEST_PERF_1" in facts
            ), "Sollte korrekte Regel (TEST_PERF_1) verwendet haben"

            # Performance-Info
            logger.info(
                f"[SUCCESS] Intelligente Ingestion: Korrekte Regel verwendet in {elapsed:.4f}s"
            )
            if elapsed > 5.0:
                logger.info(
                    f"[INFO] Dauerte {elapsed:.4f}s - Korrektheit wichtiger als Geschwindigkeit!"
                )

        finally:
            # Cleanup
            with kai_worker_with_mocks.netzwerk.driver.session(
                database="neo4j"
            ) as session:
                for rel in relations:
                    session.run(
                        """
                        MATCH (r:ExtractionRule {relation_type: $rel})
                        DETACH DELETE r
                    """,
                        rel=rel,
                    )
                session.run(
                    """
                    MATCH (p:PatternPrototype)
                    WHERE p.category = 'DEFINITION'
                    DETACH DELETE p
                """
                )


# ============================================================================
# TESTS FÜR NEUE NLP-FEATURES (FUZZY MATCHING, ERWEITERTE FRAGEN)
# ============================================================================
