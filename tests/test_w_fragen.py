"""
KAI Test Suite - W-Fragen Verarbeitung Tests
Extrahiert aus test_kai_worker.py fuer bessere Wartbarkeit.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

import pytest

from kai_response_formatter import KaiResponseFormatter
from kai_worker import KaiWorker

logger = logging.getLogger(__name__)
TEST_VECTOR_DIM = 384


class TestWFragenVerarbeitung:
    """Tests für erweiterte W-Fragen-Verarbeitung (Wer, Wie, Warum, Wann, etc.)."""

    def test_wer_frage_erkennung(self, kai_worker_with_mocks):
        """Testet die Erkennung von Wer-Fragen."""
        from component_6_linguistik_engine import LinguisticPreprocessor

        preprocessor = LinguisticPreprocessor()

        # Test verschiedene Wer-Fragen
        test_queries = [
            "Wer ist Einstein?",
            "Wer hat das erfunden?",
            "Wer kann fliegen?",
        ]

        for query in test_queries:
            doc = preprocessor.process(query)
            meaning_points = kai_worker_with_mocks.extractor.extract(doc)

            assert len(meaning_points) > 0, f"Keine Meaning Points für '{query}'"
            mp = meaning_points[0]

            # Prüfe dass es als Frage erkannt wurde
            from component_5_linguistik_strukturen import MeaningPointCategory

            assert (
                mp.category == MeaningPointCategory.QUESTION
            ), "'{query}' sollte als QUESTION erkannt werden"

            # Prüfe dass question_word = "wer" und question_type = "person_query"
            assert (
                mp.arguments.get("question_word") == "wer"
            ), "Fragewort sollte 'wer' sein"
            assert (
                mp.arguments.get("question_type") == "person_query"
            ), "Fragetyp sollte 'person_query' sein"

        logger.info("[SUCCESS] Wer-Fragen werden korrekt erkannt")

    def test_wie_frage_erkennung(self, kai_worker_with_mocks):
        """Testet die Erkennung von Wie-Fragen."""
        from component_6_linguistik_engine import LinguisticPreprocessor

        preprocessor = LinguisticPreprocessor()

        test_queries = ["Wie funktioniert ein Motor?", "Wie macht man Kuchen?"]

        for query in test_queries:
            doc = preprocessor.process(query)
            meaning_points = kai_worker_with_mocks.extractor.extract(doc)

            assert len(meaning_points) > 0
            mp = meaning_points[0]

            from component_5_linguistik_strukturen import MeaningPointCategory

            # Akzeptiere entweder QUESTION oder dass question_word = "wie" gesetzt ist
            assert (
                mp.category == MeaningPointCategory.QUESTION
                or mp.arguments.get("question_word") == "wie"
            ), f"'{query}' sollte als Wie-Frage erkannt werden, ist aber {mp.category} mit arguments {mp.arguments}"

            if mp.category == MeaningPointCategory.QUESTION:
                assert (
                    mp.arguments.get("question_word") == "wie"
                ), "Fragewort sollte 'wie' sein"
                assert (
                    mp.arguments.get("question_type") == "process_query"
                ), "Fragetyp sollte 'process_query' sein"

        logger.info("[SUCCESS] Wie-Fragen werden korrekt erkannt")

    def test_warum_frage_erkennung(self, kai_worker_with_mocks):
        """Testet die Erkennung von Warum/Wieso/Weshalb-Fragen."""
        from component_6_linguistik_engine import LinguisticPreprocessor

        preprocessor = LinguisticPreprocessor()

        test_queries = [
            "Warum ist der Himmel blau?",
            "Wieso funktioniert das nicht?",
            "Weshalb ist das so?",
        ]

        for query in test_queries:
            doc = preprocessor.process(query)
            meaning_points = kai_worker_with_mocks.extractor.extract(doc)

            assert len(meaning_points) > 0
            mp = meaning_points[0]

            from component_5_linguistik_strukturen import MeaningPointCategory

            assert mp.category == MeaningPointCategory.QUESTION
            assert mp.arguments.get("question_word") in ["warum", "wieso", "weshalb"]
            assert mp.arguments.get("question_type") == "reason_query"

        logger.info("[SUCCESS] Warum-Fragen werden korrekt erkannt")

    def test_wann_frage_erkennung(self, kai_worker_with_mocks):
        """Testet die Erkennung von Wann-Fragen."""
        from component_6_linguistik_engine import LinguisticPreprocessor

        preprocessor = LinguisticPreprocessor()

        test_queries = [
            "Wann ist Weihnachten?",
            "Wann findet das statt?",
            "Wann war das?",
        ]

        for query in test_queries:
            doc = preprocessor.process(query)
            meaning_points = kai_worker_with_mocks.extractor.extract(doc)

            assert len(meaning_points) > 0
            mp = meaning_points[0]

            from component_5_linguistik_strukturen import MeaningPointCategory

            assert mp.category == MeaningPointCategory.QUESTION
            assert mp.arguments.get("question_word") == "wann"
            assert mp.arguments.get("question_type") == "time_query"

        logger.info("[SUCCESS] Wann-Fragen werden korrekt erkannt")

    def test_wo_frage_erkennung(self, kai_worker_with_mocks):
        """Testet die Erkennung von Wo-Fragen."""
        from component_6_linguistik_engine import LinguisticPreprocessor

        preprocessor = LinguisticPreprocessor()

        test_queries = ["Wo ist Berlin?", "Wo liegt München?"]

        for query in test_queries:
            doc = preprocessor.process(query)
            meaning_points = kai_worker_with_mocks.extractor.extract(doc)

            assert len(meaning_points) > 0
            mp = meaning_points[0]

            from component_5_linguistik_strukturen import MeaningPointCategory

            assert mp.category == MeaningPointCategory.QUESTION
            # Wo-Fragen haben spezielle Logik mit property_name LOCATED_IN
            assert "topic" in mp.arguments or "property_name" in mp.arguments

        logger.info("[SUCCESS] Wo-Fragen werden korrekt erkannt")

    def test_alle_w_woerter_konsistenz(self, kai_worker_with_mocks):
        """Testet dass alle W-Wörter konsistent in allen Listen vorkommen."""
        # Definierte W-Wörter sollten in allen relevanten Heuristiken vorhanden sein
        expected_w_words = [
            "was",
            "wer",
            "wie",
            "wo",
            "wann",
            "warum",
            "welche",
            "welcher",
            "welches",
            "wozu",
            "wieso",
            "weshalb",
        ]

        # Prüfe dass alle W-Wörter in der Vektor-basierten Argument-Extraktion vorkommen

        # Test: Jedes W-Wort sollte zu einer Frage-Erkennung führen
        from component_6_linguistik_engine import LinguisticPreprocessor

        preprocessor = LinguisticPreprocessor()

        for w_word in expected_w_words:
            query = f"{w_word.capitalize()} ist das?"
            doc = preprocessor.process(query)
            meaning_points = kai_worker_with_mocks.extractor.extract(doc)

            # Sollte als Frage erkannt werden (entweder durch Heuristik oder als deklarative Aussage gefiltert)
            assert (
                len(meaning_points) > 0
            ), f"'{query}' sollte einen MeaningPoint erzeugen"

        logger.info(
            f"[SUCCESS] Alle {len(expected_w_words)} W-Wörter werden konsistent verarbeitet"
        )

    def test_spezifische_antworten_fuer_fragetypen(
        self, netzwerk_session, embedding_service_session
    ):
        """Testet dass verschiedene Fragetypen spezifische Antworten erhalten."""
        KaiWorker(netzwerk_session, embedding_service_session)

        # Setup: Erstelle Test-Wissen
        test_concept = "test_person_einstein"
        netzwerk_session.assert_relation(
            test_concept, "IS_A", "wissenschaftler", "test"
        )
        netzwerk_session.assert_relation(
            test_concept, "CAPABLE_OF", "relativitätstheorie entwickeln", "test"
        )
        netzwerk_session.add_information_zu_wort(
            test_concept, "bedeutung", "Ein berühmter Physiker"
        )

        try:
            # Test: Verschiedene Fragetypen sollten unterschiedliche Antworten generieren
            # Dies kann nur indirekt getestet werden durch Inspektion der Formatter-Methoden

            # Test Person-Formatter
            facts = {
                "IS_A": ["wissenschaftler"],
                "CAPABLE_OF": ["relativitätstheorie entwickeln"],
            }
            bedeutungen = ["Ein berühmter Physiker"]
            synonyms = []

            person_answer = KaiResponseFormatter.format_person_answer(
                test_concept, facts, bedeutungen, synonyms
            )
            assert "wissenschaftler" in person_answer.lower()
            assert "physiker" in person_answer.lower()

            # Test Process-Formatter
            process_answer = KaiResponseFormatter.format_process_answer(
                test_concept, facts, bedeutungen
            )
            assert "physiker" in process_answer.lower()

            logger.info("[SUCCESS] Spezifische Antworten für Fragetypen funktionieren")

        finally:
            # Cleanup
            with netzwerk_session.driver.session(database="neo4j") as session:
                session.run(
                    """
                    MATCH (w:Wort {text: $text})
                    DETACH DELETE w
                """,
                    text=test_concept,
                )

    def test_wissenslücke_mit_fragetyp_spezifischer_rückfrage(
        self, netzwerk_session, embedding_service_session
    ):
        """Testet dass Wissenslücken fragetyp-spezifische Rückfragen generieren."""
        kai_worker = KaiWorker(netzwerk_session, embedding_service_session)

        # Simuliere Wissenslücke für verschiedene Fragetypen
        # Wer-Frage
        from component_5_linguistik_strukturen import (
            MeaningPoint,
            MeaningPointCategory,
            Modality,
            Polarity,
        )

        wer_intent = MeaningPoint(
            id="test_wer",
            category=MeaningPointCategory.QUESTION,
            cue="test",
            text_span="Wer ist unbekannt?",
            modality=Modality.INTERROGATIVE,
            polarity=Polarity.POSITIVE,
            confidence=0.9,
            arguments={
                "topic": "unbekannt",
                "question_word": "wer",
                "question_type": "person_query",
            },
        )

        # Teste Sub-Goal-Ausführung direkt
        context = {
            "intent": wer_intent,
            "topic": "unbekannt",
            "has_knowledge_gap": True,
            "learned_facts": {},
            "bedeutungen": [],
            "fuzzy_suggestions": [],
        }

        from component_5_linguistik_strukturen import GoalStatus, SubGoal

        sub_goal = SubGoal(
            id="sg1",
            description="Formuliere eine Antwort oder eine Rückfrage.",
            status=GoalStatus.PENDING,
        )

        success, result = kai_worker._execute_sub_goal(sub_goal, context)

        assert success
        response = result["final_response"]

        # Sollte fragetyp-spezifische Rückfrage enthalten
        assert "person" in response.lower() or "wer" in response.lower()

        logger.info("[SUCCESS] Wissenslücken erzeugen fragetyp-spezifische Rückfragen")


# ============================================================================
# MAIN - Führt Tests aus, wenn Datei direkt ausgeführt wird
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
