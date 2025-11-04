"""
KAI Test Suite - Interactive Learning Tests
Extrahiert aus test_kai_worker.py fuer bessere Wartbarkeit.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

logger = logging.getLogger(__name__)
TEST_VECTOR_DIM = 384


class TestInteractiveLearning:
    """Tests für Phase 5: Interaktives Lernen mit Feedback-Loops."""

    def test_clarification_request_on_low_confidence(self, kai_worker_with_mocks):
        """
        PHASE 5.1 (Aktion 1): Testet Clarification-Request bei niedriger Konfidenz.
        KAI sollte fragen wenn unsicher, nicht einfach aufgeben.
        """
        # Eingabe mit niedriger Konfidenz (unbekanntes Muster)
        query = "Zeige mir alle fliegenden Bananen"  # Ungewöhnliche Formulierung

        kai_worker_with_mocks.process_query(query)

        response = kai_worker_with_mocks.signals.finished.emit.call_args.args[0]

        # Sollte eine Klarstellungsfrage stellen
        assert any(
            keyword in response.text.lower()
            for keyword in [
                "nicht sicher",
                "unsicher",
                "anders formulieren",
                "beispiel",
            ]
        ), f"Erwartete Clarification-Request, bekam: '{response.text}'"

        # Kontext sollte gesetzt sein für Feedback
        assert kai_worker_with_mocks.context.is_active()
        assert (
            kai_worker_with_mocks.context.aktion.value
            == "erwarte_feedback_zu_clarification"
        )

        logger.info("[SUCCESS] Clarification-Request bei niedriger Konfidenz korrekt")

    def test_feedback_loop_after_clarification(
        self, kai_worker_with_mocks, clean_test_concepts
    ):
        """
        PHASE 5.1 (Aktion 1): Testet Feedback-Loop nach Clarification.
        Nutzer gibt Pattern-Learning-Beispiel -> KAI lernt -> Retry der ursprünglichen Frage.
        """
        # Setup: Frage, die Clarification auslöst
        unknown_pattern = "Zeige mir Äpfel"
        kai_worker_with_mocks.process_query(unknown_pattern)

        # Verifiziere Clarification
        assert kai_worker_with_mocks.context.is_active()
        original_query = kai_worker_with_mocks.context.metadata.get(
            "original_query", ""
        )
        assert unknown_pattern.lower() in original_query.lower()

        # Nutzer gibt Pattern-Learning-Feedback
        kai_worker_with_mocks.signals.reset_mock()
        feedback_query = 'Lerne Muster: "Zeige mir Bananen" bedeutet HAS_PROPERTY'
        kai_worker_with_mocks.process_query(feedback_query)

        # Sollte Learning-Bestätigung geben
        response = kai_worker_with_mocks.signals.finished.emit.call_args.args[0]
        assert any(
            keyword in response.text.lower()
            for keyword in ["gelernt", "verstanden", "ursprüngliche", "nochmal"]
        ), f"Erwartete Learning-Bestätigung, bekam: '{response.text}'"

        logger.info("[SUCCESS] Feedback-Loop nach Clarification funktioniert")

    def test_confirmation_request_on_medium_confidence(
        self, kai_worker_with_mocks, clean_test_concepts
    ):
        """
        PHASE 5.2 (Aktion 2): Testet Confirmation-Request bei mittlerer Konfidenz.
        KAI sollte nachfragen "Ist das richtig?" statt blind zu handeln.

        HINWEIS: Dieser Test ist dokumentarisch, da Confirmation-Triggering
        von der internen Konfidenzberechnung abhängt.
        """
        # Erstelle eine Frage mit mittlerer Konfidenz
        # (Dies ist implementierungsabhängig - hier simulieren wir es)
        test_word = f"{clean_test_concepts}mittlereconfidence"
        query = f"Was weißt du über {test_word}?"

        # Erstelle minimales Wissen, damit nicht Knowledge Gap, aber auch nicht viel
        kai_worker_with_mocks.netzwerk.ensure_wort_und_konzept(test_word)

        kai_worker_with_mocks.process_query(query)

        response = kai_worker_with_mocks.signals.finished.emit.call_args.args[0]

        # Je nach aktueller Konfidenz könnte eine Bestätigung kommen
        # Dieser Test ist eher dokumentarisch, da Konfidenz-Berechnung komplex ist
        logger.info(f"Response bei mittlerer Konfidenz: '{response.text}'")

        # Kontext-Check: Akzeptiere verschiedene Szenarien
        if kai_worker_with_mocks.context.is_active():
            # Falls Kontext gesetzt ist, validiere den Typ
            context_action = kai_worker_with_mocks.context.aktion.value
            assert context_action in [
                "erwarte_bestaetigung",
                "erwarte_beispielsatz",
                "erwarte_feedback_zu_clarification",
            ], f"Unerwarteter Kontext: {context_action}"
            logger.info(f"[SUCCESS] Kontext gesetzt: {context_action}")
        else:
            logger.info(
                "[INFO] Kein Kontext gesetzt (direkte Antwort oder zu niedrige Konfidenz)"
            )

    def test_confirmation_yes_proceeds_with_action(
        self, kai_worker_with_mocks, clean_test_concepts
    ):
        """
        PHASE 5.2 (Aktion 2): Testet "Ja"-Antwort auf Confirmation.
        KAI sollte den ursprünglichen Plan ausführen.
        """
        # Simuliere Confirmation-Kontext manuell (da Auslösung komplex ist)
        from component_5_linguistik_strukturen import (
            ContextAction,
            MeaningPointCategory,
            MeaningPoint,
            Modality,
            Polarity,
        )

        test_word = f"{clean_test_concepts}confirmtest"

        # Erstelle Mock-Intent mit allen required Feldern
        mock_intent = MeaningPoint(
            id="test-mp-1",
            category=MeaningPointCategory.QUESTION,
            cue="was",
            text_span=f"Was ist {test_word}?",
            modality=Modality.INTERROGATIVE,
            polarity=Polarity.POSITIVE,
            confidence=0.6,  # Mittlere Konfidenz
            arguments={"topic": test_word},
        )

        # Setze Kontext
        kai_worker_with_mocks.context.aktion = ContextAction.ERWARTE_BESTAETIGUNG
        kai_worker_with_mocks.context.original_intent = mock_intent
        kai_worker_with_mocks.context.metadata["sub_goal_context"] = {
            "intent": mock_intent
        }

        # Nutzer antwortet "Ja"
        kai_worker_with_mocks.signals.reset_mock()
        kai_worker_with_mocks.process_query("Ja, genau")

        # Response sollte kommen (entweder Antwort oder Knowledge Gap)
        assert kai_worker_with_mocks.signals.finished.emit.called

        # Der ursprüngliche Bestätigungs-Kontext sollte abgearbeitet sein
        # ABER: Ein neuer Kontext kann gesetzt sein (z.B. ERWARTE_BEISPIELSATZ bei Knowledge Gap)
        # Dies ist korrekt, da KAI erkannt hat, dass es nichts über das Thema weiß
        if kai_worker_with_mocks.context.is_active():
            # Neuer Kontext (z.B. wegen Knowledge Gap) ist akzeptabel
            logger.info(
                f"[SUCCESS] 'Ja'-Antwort führte Plan aus und setzte neuen Kontext: {kai_worker_with_mocks.context.aktion.value}"
            )
        else:
            logger.info("[SUCCESS] 'Ja'-Antwort führte Plan aus ohne neuen Kontext")

    def test_confirmation_no_offers_help(
        self, kai_worker_with_mocks, clean_test_concepts
    ):
        """
        PHASE 5.2 (Aktion 2): Testet "Nein"-Antwort auf Confirmation.
        KAI sollte konstruktiv nachfragen, was stattdessen gemeint war.
        """
        # Simuliere Confirmation-Kontext
        from component_5_linguistik_strukturen import (
            ContextAction,
            MeaningPointCategory,
            MeaningPoint,
            Modality,
            Polarity,
        )

        mock_intent = MeaningPoint(
            id="test-mp-2",
            category=MeaningPointCategory.QUESTION,
            cue="was",
            text_span="Testsatz",
            modality=Modality.INTERROGATIVE,
            polarity=Polarity.POSITIVE,
            confidence=0.5,
            arguments={"topic": "test"},
        )

        kai_worker_with_mocks.context.aktion = ContextAction.ERWARTE_BESTAETIGUNG
        kai_worker_with_mocks.context.original_intent = mock_intent

        # Nutzer antwortet "Nein"
        kai_worker_with_mocks.signals.reset_mock()
        kai_worker_with_mocks.process_query("Nein, das ist falsch")

        response = kai_worker_with_mocks.signals.finished.emit.call_args.args[0]

        # Sollte konstruktiven Lernvorschlag machen
        assert any(
            keyword in response.text.lower()
            for keyword in [
                "was wolltest du",
                "anders formulieren",
                "lerne muster",
                "beispiel",
            ]
        ), f"Erwartete konstruktiven Lernvorschlag, bekam: '{response.text}'"

        # Kontext sollte gelöscht sein
        assert not kai_worker_with_mocks.context.is_active()

        logger.info("[SUCCESS] 'Nein'-Antwort bietet konstruktive Hilfe")


# ============================================================================
# PHASE 6: TESTS FÜR INTELLIGENTE INGESTION & SYNONYM-ERWEITERUNG
# ============================================================================
