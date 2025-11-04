"""
Erweiterte Tests für component_1_netzwerk.py um Coverage von 69% auf 75-80% zu erhöhen.

Diese Tests decken ab:
- Driver Setter
- Close Methode
- Edge Cases
- Fehlerbehandlung
- Weniger häufig getestete Methoden
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import MagicMock


class TestKonzeptNetzwerkCore:
    """Tests für grundlegende Netzwerk-Funktionalität."""

    def test_driver_setter(self, netzwerk_session):
        """Testet Driver Setter und Propagierung zu Sub-Modulen."""
        # Erstelle Mock Driver
        mock_driver = MagicMock()

        # Setze Driver
        netzwerk_session.driver = mock_driver

        # Verifiziere, dass alle Sub-Module den neuen Driver haben
        assert netzwerk_session._core.driver == mock_driver
        assert netzwerk_session._patterns.driver == mock_driver
        assert netzwerk_session._memory.driver == mock_driver
        assert netzwerk_session._word_usage.driver == mock_driver
        assert netzwerk_session._feedback.driver == mock_driver

    def test_close_connection(self, netzwerk_session):
        """Testet close() Methode."""
        # Close sollte ohne Fehler funktionieren
        netzwerk_session.close()

        # Nach close sollte keine Verbindung mehr bestehen
        with pytest.raises(Exception):
            netzwerk_session.driver.verify_connectivity()

    def test_set_wort_attribut(self, netzwerk_session, clean_test_concepts):
        """Testet set_wort_attribut Methode."""
        test_word = f"{clean_test_concepts}testattribut"

        # Erstelle Wort
        netzwerk_session.ensure_wort_und_konzept(test_word)

        # Setze Attribut
        netzwerk_session.set_wort_attribut(test_word, "test_attr", "test_value")

        # Verifiziere Attribut
        with netzwerk_session.driver.session(database="neo4j") as session:
            result = session.run(
                """
                MATCH (w:Wort {lemma: $lemma})
                RETURN w.test_attr AS attr
            """,
                lemma=test_word,
            )
            record = result.single()
            assert record is not None
            assert record["attr"] == "test_value"

    def test_query_graph_for_facts_empty(self, netzwerk_session, clean_test_concepts):
        """Testet query_graph_for_facts für unbekanntes Topic."""
        unknown_word = f"{clean_test_concepts}niemalserstellteswort"

        facts = netzwerk_session.query_graph_for_facts(unknown_word)

        # Sollte leeres Dictionary oder Dictionary mit leeren Listen zurückgeben
        assert isinstance(facts, dict)
        assert len(facts) == 0 or all(len(v) == 0 for v in facts.values())

    def test_query_graph_for_facts_multiple_relations(
        self, netzwerk_session, clean_test_concepts
    ):
        """Testet query_graph_for_facts mit mehreren Relations-Typen."""
        subject = f"{clean_test_concepts}multirel"
        obj1 = f"{clean_test_concepts}obj1"
        obj2 = f"{clean_test_concepts}obj2"
        obj3 = f"{clean_test_concepts}obj3"

        # Erstelle verschiedene Relations
        netzwerk_session.assert_relation(subject, "IS_A", obj1, "test")
        netzwerk_session.assert_relation(subject, "HAS_PROPERTY", obj2, "test")
        netzwerk_session.assert_relation(subject, "LOCATED_IN", obj3, "test")

        facts = netzwerk_session.query_graph_for_facts(subject)

        # Verifiziere alle Relations
        assert "IS_A" in facts
        assert "HAS_PROPERTY" in facts
        assert "LOCATED_IN" in facts
        assert obj1 in facts["IS_A"]
        assert obj2 in facts["HAS_PROPERTY"]
        assert obj3 in facts["LOCATED_IN"]

    def test_assert_relation_idempotent(self, netzwerk_session, clean_test_concepts):
        """Testet Idempotenz von assert_relation."""
        subject = f"{clean_test_concepts}idempotent"
        obj = f"{clean_test_concepts}object"

        # Erstelle Relation zweimal
        result1 = netzwerk_session.assert_relation(subject, "IS_A", obj, "test")
        result2 = netzwerk_session.assert_relation(subject, "IS_A", obj, "test")

        # Beide sollten erfolgreich sein (idempotent)
        assert result1 is True
        assert result2 is True

        # Verifiziere, dass nur eine Relation existiert
        with netzwerk_session.driver.session(database="neo4j") as session:
            result = session.run(
                """
                MATCH (k1:Konzept {name: $subject})-[r:IS_A]->(k2:Konzept {name: $obj})
                RETURN count(r) AS count
            """,
                subject=subject,
                obj=obj,
            )
            assert result.single()["count"] == 1


class TestKonzeptNetzwerkPatterns:
    """Tests für Pattern-bezogene Methoden."""

    def test_get_all_extraction_rules_empty(self, netzwerk_session):
        """Testet get_all_extraction_rules wenn keine Regeln existieren."""
        # Lösche alle Test-Regeln
        with netzwerk_session.driver.session(database="neo4j") as session:
            session.run(
                """
                MATCH (r:ExtractionRule)
                WHERE r.relation_type STARTS WITH 'TEST_'
                DETACH DELETE r
            """
            )

        rules = netzwerk_session.get_all_extraction_rules()

        # Sollte Liste zurückgeben (kann leer sein oder nur Standard-Regeln enthalten)
        assert isinstance(rules, list)

    def test_create_extraction_rule_with_invalid_regex(self, netzwerk_session):
        """Testet create_extraction_rule mit ungültigem Regex."""
        import re

        relation_type = "TEST_INVALID_REGEX"
        invalid_regex = r"^(.+) invalid [regex$"  # Ungültiger Regex

        # Sollte Exception werfen oder False zurückgeben
        try:
            result = netzwerk_session.create_extraction_rule(
                relation_type, invalid_regex
            )
            # Wenn keine Exception, sollte False sein
            assert result is False
        except re.error:
            # re.error ist akzeptabel
            pass

    def test_get_all_pattern_prototypes(self, netzwerk_session):
        """Testet get_all_pattern_prototypes Methode."""
        prototypes = netzwerk_session.get_all_pattern_prototypes()

        # Sollte Liste zurückgeben
        assert isinstance(prototypes, list)

    def test_link_prototype_to_rule(self, netzwerk_session):
        """Testet link_prototype_to_rule Methode."""
        # Erstelle Test-Regel
        relation_type = "TEST_LINK_RULE"
        regex = r"^(.+) linkt (.+)$"
        netzwerk_session.create_extraction_rule(relation_type, regex)

        # Erstelle Prototyp (via direktem Cypher)
        with netzwerk_session.driver.session(database="neo4j") as session:
            result = session.run(
                """
                CREATE (p:PatternPrototype {
                    prototype_id: randomUUID(),
                    category: 'TEST',
                    vector: [0.0] * 384,
                    created_at: datetime()
                })
                RETURN p.prototype_id AS id
            """
            )
            prototype_id = result.single()["id"]

        # Linke Prototyp mit Regel
        success = netzwerk_session.link_prototype_to_rule(prototype_id, relation_type)

        assert success is True

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
                MATCH (p:PatternPrototype {prototype_id: $id})
                DETACH DELETE p
            """,
                id=prototype_id,
            )


class TestKonzeptNetzwerkMemory:
    """Tests für Memory-bezogene Methoden."""

    def test_store_learning_episode(self, netzwerk_session, clean_test_concepts):
        """Testet store_learning_episode Methode."""
        test_fact = f"{clean_test_concepts}episodisches_lernen"

        episode_id = netzwerk_session.store_learning_episode(
            source_text="Test-Episode",
            facts_learned=[test_fact],
            extraction_method="TEST_METHOD",
        )

        assert episode_id is not None
        assert isinstance(episode_id, str)

    def test_get_learning_episodes(self, netzwerk_session):
        """Testet get_learning_episodes Methode."""
        episodes = netzwerk_session.get_learning_episodes(limit=10)

        assert isinstance(episodes, list)

    def test_store_inference_episode(self, netzwerk_session, clean_test_concepts):
        """Testet store_inference_episode Methode."""
        test_query = f"{clean_test_concepts}inference_test"

        episode_id = netzwerk_session.store_inference_episode(
            query=test_query,
            result="Test result",
            inference_path=["step1", "step2"],
            confidence=0.9,
        )

        assert episode_id is not None
        assert isinstance(episode_id, str)

    def test_get_inference_episodes(self, netzwerk_session):
        """Testet get_inference_episodes Methode."""
        episodes = netzwerk_session.get_inference_episodes(limit=10)

        assert isinstance(episodes, list)


class TestKonzeptNetzwerkWordUsage:
    """Tests für Word Usage Tracking."""

    def test_track_word_usage(self, netzwerk_session, clean_test_concepts):
        """Testet track_word_usage Methode."""
        test_word = f"{clean_test_concepts}tracked"
        test_context = "Test context for tracking"

        netzwerk_session.track_word_usage(test_word, test_context)

        # Verifiziere, dass Usage getrackt wurde
        with netzwerk_session.driver.session(database="neo4j") as session:
            result = session.run(
                """
                MATCH (w:Wort {lemma: $lemma})-[:USED_IN]->(u:Usage)
                RETURN count(u) AS count
            """,
                lemma=test_word,
            )
            count = result.single()["count"]
            assert count >= 1

    def test_get_word_usage_contexts(self, netzwerk_session, clean_test_concepts):
        """Testet get_word_usage_contexts Methode."""
        test_word = f"{clean_test_concepts}usagecontext"
        test_context = "Context for retrieval test"

        # Track usage
        netzwerk_session.track_word_usage(test_word, test_context)

        # Retrieve contexts
        contexts = netzwerk_session.get_word_usage_contexts(test_word, limit=10)

        assert isinstance(contexts, list)
        assert any(test_context in ctx for ctx in contexts)


class TestKonzeptNetzwerkFeedback:
    """Tests für Feedback System."""

    def test_record_typo_feedback(self, netzwerk_session, clean_test_concepts):
        """Testet record_typo_feedback Methode."""
        typo = f"{clean_test_concepts}typo"
        correction = f"{clean_test_concepts}correct"

        netzwerk_session.record_typo_feedback(
            typo=typo, correction=correction, accepted=True, confidence=0.85
        )

        # Verifiziere, dass Feedback gespeichert wurde
        with netzwerk_session.driver.session(database="neo4j") as session:
            result = session.run(
                """
                MATCH (f:TypoFeedback {typo: $typo, correction: $correction})
                RETURN count(f) AS count
            """,
                typo=typo,
                correction=correction,
            )
            count = result.single()["count"]
            assert count >= 1

    def test_get_typo_feedback(self, netzwerk_session, clean_test_concepts):
        """Testet get_typo_feedback Methode."""
        typo = f"{clean_test_concepts}getfeedback"
        correction = f"{clean_test_concepts}correct"

        # Record feedback
        netzwerk_session.record_typo_feedback(
            typo=typo, correction=correction, accepted=False, confidence=0.75
        )

        # Retrieve feedback
        feedback = netzwerk_session.get_typo_feedback(typo, correction)

        assert feedback is not None
        assert feedback["typo"] == typo
        assert feedback["correction"] == correction


class TestKonzeptNetzwerkEdgeCases:
    """Tests für Edge Cases und Error Handling."""

    def test_empty_subject_relation(self, netzwerk_session):
        """Testet Verhalten mit leerem Subject."""
        try:
            result = netzwerk_session.assert_relation("", "IS_A", "object", "test")
            # Sollte entweder False oder Exception sein
            assert result is False
        except Exception:
            # Exception ist akzeptabel
            pass

    def test_empty_object_relation(self, netzwerk_session, clean_test_concepts):
        """Testet Verhalten mit leerem Object."""
        subject = f"{clean_test_concepts}emptyobj"
        try:
            result = netzwerk_session.assert_relation(subject, "IS_A", "", "test")
            # Sollte entweder False oder Exception sein
            assert result is False
        except Exception:
            # Exception ist akzeptabel
            pass

    def test_special_characters_in_lemma(self, netzwerk_session, clean_test_concepts):
        """Testet Verhalten mit Sonderzeichen in Lemma."""
        special_word = f"{clean_test_concepts}test@#$%"

        # Sollte ohne Fehler funktionieren
        netzwerk_session.ensure_wort_und_konzept(special_word)

        details = netzwerk_session.get_details_fuer_wort(special_word)
        assert details is not None

    def test_unicode_in_lemma(self, netzwerk_session, clean_test_concepts):
        """Testet Verhalten mit Unicode-Zeichen."""
        unicode_word = f"{clean_test_concepts}tëst_ñämé_日本"

        # Sollte ohne Fehler funktionieren
        netzwerk_session.ensure_wort_und_konzept(unicode_word)

        details = netzwerk_session.get_details_fuer_wort(unicode_word)
        assert details is not None

    def test_very_long_source_sentence(self, netzwerk_session, clean_test_concepts):
        """Testet Verhalten mit sehr langem source_sentence."""
        subject = f"{clean_test_concepts}longsource"
        obj = f"{clean_test_concepts}obj"
        long_sentence = "x" * 10000  # 10k characters

        result = netzwerk_session.assert_relation(subject, "IS_A", obj, long_sentence)

        assert result is True
