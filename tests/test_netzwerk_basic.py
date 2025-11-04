"""
KAI Test Suite - Netzwerk Basic Tests
Basis-Tests aus test_kai_worker.py extrahiert.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

logger = logging.getLogger(__name__)
TEST_VECTOR_DIM = 384


class TestKonzeptNetzwerk:
    """Tests für die grundlegenden Netzwerk-Operationen."""

    def test_connection_established(self, netzwerk_session):
        """Prüft, ob die DB-Verbindung besteht."""
        assert netzwerk_session.driver is not None
        netzwerk_session.driver.verify_connectivity()

    def test_ensure_wort_und_konzept(self, netzwerk_session, clean_test_concepts):
        """Testet das Erstellen von Wort-Konzept-Paaren mit Edge Cases."""
        test_word = f"{clean_test_concepts}apfel"
        netzwerk_session.ensure_wort_und_konzept(test_word)

        with netzwerk_session.driver.session(database="neo4j") as session:
            result = session.run(
                """
                MATCH (w:Wort {lemma: $lemma})-[:BEDEUTET]->(k:Konzept {name: $lemma})
                RETURN count(*) AS count
            """,
                lemma=test_word,
            )
            assert result.single()["count"] == 1

        # Edge Case 1: Idempotenz - zweimaliges Erstellen sollte nicht duplizieren
        netzwerk_session.ensure_wort_und_konzept(test_word)
        with netzwerk_session.driver.session(database="neo4j") as session:
            result = session.run(
                """
                MATCH (w:Wort {lemma: $lemma})-[:BEDEUTET]->(k:Konzept {name: $lemma})
                RETURN count(*) AS count
            """,
                lemma=test_word,
            )
            assert (
                result.single()["count"] == 1
            ), "Doppeltes Erstellen sollte keine Duplikate erzeugen"

        # Edge Case 2: Leeres Wort sollte Fehler vermeiden
        try:
            netzwerk_session.ensure_wort_und_konzept("")
        except Exception as e:
            logger.warning(f"Leeres Wort erzeugt erwartete Exception: {e}")

        # Edge Case 3: Sehr langes Wort
        long_word = f"{clean_test_concepts}{'x' * 500}"
        netzwerk_session.ensure_wort_und_konzept(long_word)
        with netzwerk_session.driver.session(database="neo4j") as session:
            result = session.run(
                """
                MATCH (w:Wort {lemma: $lemma})
                RETURN count(*) AS count
            """,
                lemma=long_word,
            )
            assert (
                result.single()["count"] == 1
            ), "Sehr lange Wörter sollten gespeichert werden können"

    def test_add_bedeutung(self, netzwerk_session, clean_test_concepts):
        """Testet das Hinzufügen einer Bedeutung."""
        test_word = f"{clean_test_concepts}birne"
        bedeutung = "Eine süße Frucht"

        result = netzwerk_session.add_information_zu_wort(
            test_word, "bedeutung", bedeutung
        )
        assert result.get("created") is True

        details = netzwerk_session.get_details_fuer_wort(test_word)
        assert bedeutung in details["bedeutungen"]

    def test_add_synonym(self, netzwerk_session, clean_test_concepts):
        """Testet das Erstellen von Synonym-Beziehungen."""
        word1 = f"{clean_test_concepts}auto"
        word2 = f"{clean_test_concepts}wagen"

        result = netzwerk_session.add_information_zu_wort(word1, "synonym", word2)
        assert result.get("created") is True

        details = netzwerk_session.get_details_fuer_wort(word1)
        assert word2 in details["synonyme"]

    def test_create_extraction_rule_persists(self, netzwerk_session):
        """
        KRITISCHER TEST: Stellt sicher, dass Extraktionsregeln
        tatsächlich in der DB gespeichert werden.
        """
        relation_type = "TEST_PERSIST_RELATION"
        regex = r"^(.+) persistiert (.+)$"

        try:
            # Erstelle Regel
            netzwerk_session.create_extraction_rule(relation_type, regex)

            # Sofortige Verifikation innerhalb derselben Session
            with netzwerk_session.driver.session(database="neo4j") as session:
                result = session.run(
                    """
                    MATCH (r:ExtractionRule {relation_type: $rel})
                    RETURN r.regex_pattern AS pattern, r.created_at AS created
                """,
                    rel=relation_type,
                )
                record = result.single()

                assert (
                    record is not None
                ), f"Regel '{relation_type}' wurde NICHT in DB gespeichert!"
                assert (
                    record["pattern"] == regex
                ), f"Regex stimmt nicht überein. Erwartet: {regex}, Gefunden: {record['pattern']}"
                assert record["created"] is not None, "created_at Timestamp fehlt"

            # Verifiziere via get_all_extraction_rules()
            rules = netzwerk_session.get_all_extraction_rules()
            test_rule = next(
                (r for r in rules if r["relation_type"] == relation_type), None
            )
            assert (
                test_rule is not None
            ), f"Regel '{relation_type}' nicht via get_all_extraction_rules() auffindbar"
            assert test_rule["regex_pattern"] == regex

            logger.info(
                f"[SUCCESS] Regel '{relation_type}' erfolgreich persistiert und verifiziert"
            )
        finally:
            with netzwerk_session.driver.session(database="neo4j") as session:
                session.run(
                    """
                    MATCH (r:ExtractionRule {relation_type: $rel})
                    DETACH DELETE r
                """,
                    rel=relation_type,
                )

    def test_create_extraction_rule(self, netzwerk_session):
        """Testet das Erstellen von Extraktionsregeln."""
        relation_type = "TEST_RELATION"
        regex = r"^(.+) testet (.+)$"

        try:
            netzwerk_session.create_extraction_rule(relation_type, regex)

            rules = netzwerk_session.get_all_extraction_rules()
            test_rule = next(
                (r for r in rules if r["relation_type"] == relation_type), None
            )
            assert test_rule is not None
            assert test_rule["regex_pattern"] == regex
        finally:
            with netzwerk_session.driver.session(database="neo4j") as session:
                session.run(
                    """
                    MATCH (r:ExtractionRule {relation_type: $rel})
                    DETACH DELETE r
                """,
                    rel=relation_type,
                )

    def test_assert_relation(self, netzwerk_session, clean_test_concepts):
        """Testet das Erstellen von Relationen zwischen Konzepten mit Edge Cases."""
        subject = f"{clean_test_concepts}hund"
        object = f"{clean_test_concepts}tier"

        created = netzwerk_session.assert_relation(
            subject, "IS_A", object, "Test-Quelle"
        )
        assert created is True

        # Zweiter Aufruf sollte False zurückgeben (bereits vorhanden)
        created_again = netzwerk_session.assert_relation(
            subject, "IS_A", object, "Test-Quelle"
        )
        assert created_again is False

        # Edge Case 1: Selbst-Relation (subject == object)
        created_self = netzwerk_session.assert_relation(
            subject, "RELATED_TO", subject, "Selbstbezug"
        )
        # System sollte dies erlauben (z.B. für rekursive Definitionen)
        assert created_self is True

        # Edge Case 2: Verschiedene Relationstypen zwischen gleichen Konzepten
        created_new_rel = netzwerk_session.assert_relation(
            subject, "CAPABLE_OF", object, "Andere Relation"
        )
        assert created_new_rel is True

        # Edge Case 3: Gleiche Relation mit unterschiedlicher Source sollte nicht duplizieren
        created_same_rel_diff_source = netzwerk_session.assert_relation(
            subject, "IS_A", object, "Andere Quelle"
        )
        assert created_same_rel_diff_source is False

        # Verifiziere, dass nur eine IS_A Relation existiert
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
            assert count == 1, f"Sollte genau 1 IS_A Relation haben, hat aber {count}"

    def test_query_graph_for_facts(self, netzwerk_session, clean_test_concepts):
        """Testet das Abfragen von Fakten aus dem Graphen."""
        subject = f"{clean_test_concepts}katze"
        object1 = f"{clean_test_concepts}haustier"
        object2 = f"{clean_test_concepts}miauen"

        netzwerk_session.assert_relation(
            subject, "IS_A", object1, source_sentence="test"
        )
        netzwerk_session.assert_relation(
            subject, "CAPABLE_OF", object2, source_sentence="test"
        )

        facts = netzwerk_session.query_graph_for_facts(subject)
        assert "IS_A" in facts
        assert object1 in facts["IS_A"]
        assert "CAPABLE_OF" in facts
        assert object2 in facts["CAPABLE_OF"]


# ============================================================================
# TESTS FÜR EMBEDDING SERVICE (component_11_embedding_service.py)
# ============================================================================
