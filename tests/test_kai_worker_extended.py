"""
Erweiterte Tests für kai_worker.py um Coverage von 66% auf 75-80% zu erhöhen.

Diese Tests decken ab:
- Initialisierungs-Fehler-Handling
- Command Suggestion Pfade
- Pattern Recognition Pfade
- Auto-Korrekturen
- Edge Cases und Error Paths
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import MagicMock, patch
from kai_worker import KaiWorker, KaiResponse
from kai_exceptions import Neo4jConnectionError, EmbeddingError


class TestKaiWorkerInitialization:
    """Tests für Initialisierungs-Szenarien."""

    def test_initialization_with_neo4j_error(self):
        """Testet graceful degradation bei Neo4j Connection Error."""
        with patch("kai_worker.KonzeptNetzwerk") as mock_netzwerk_class:
            mock_netzwerk = MagicMock()
            mock_netzwerk_class.return_value = mock_netzwerk

            with patch("kai_worker.LinguisticPreprocessor") as mock_preprocessor:
                mock_preprocessor.side_effect = Neo4jConnectionError(
                    "Konnte keine Verbindung zu Neo4j herstellen"
                )

                embedding_service = MagicMock()
                worker = KaiWorker(mock_netzwerk, embedding_service)

                # Worker sollte nicht initialisiert sein
                assert not worker.is_initialized_successfully
                assert worker.initialization_error_message is not None
                assert (
                    "Neo4j" in worker.initialization_error_message
                    or "Datenbank" in worker.initialization_error_message
                )

    def test_initialization_with_embedding_error(self):
        """Testet graceful degradation bei Embedding Service Error."""
        with patch("kai_worker.KonzeptNetzwerk") as mock_netzwerk_class:
            mock_netzwerk = MagicMock()
            mock_netzwerk_class.return_value = mock_netzwerk

            with patch("kai_worker.LinguisticPreprocessor"):
                with patch("kai_worker.Engine") as mock_engine:
                    mock_engine.side_effect = EmbeddingError(
                        "Embedding Model konnte nicht geladen werden"
                    )

                    embedding_service = MagicMock()
                    worker = KaiWorker(mock_netzwerk, embedding_service)

                    # Worker sollte nicht initialisiert sein
                    assert not worker.is_initialized_successfully
                    assert worker.initialization_error_message is not None

    def test_initialization_with_generic_error(self):
        """Testet graceful degradation bei generischem Fehler."""
        with patch("kai_worker.KonzeptNetzwerk") as mock_netzwerk_class:
            mock_netzwerk = MagicMock()
            mock_netzwerk_class.return_value = mock_netzwerk

            with patch("kai_worker.LinguisticPreprocessor") as mock_preprocessor:
                mock_preprocessor.side_effect = RuntimeError("Unerwarteter Fehler")

                embedding_service = MagicMock()
                worker = KaiWorker(mock_netzwerk, embedding_service)

                # Worker sollte nicht initialisiert sein
                assert not worker.is_initialized_successfully
                assert (
                    "unerwarteter Fehler" in worker.initialization_error_message.lower()
                )

    def test_process_query_with_failed_initialization(self):
        """Testet process_query wenn Initialisierung fehlgeschlagen ist."""
        with patch("kai_worker.KonzeptNetzwerk") as mock_netzwerk_class:
            mock_netzwerk = MagicMock()
            mock_netzwerk_class.return_value = mock_netzwerk

            with patch("kai_worker.LinguisticPreprocessor") as mock_preprocessor:
                mock_preprocessor.side_effect = RuntimeError("Init failed")

                embedding_service = MagicMock()
                worker = KaiWorker(mock_netzwerk, embedding_service)

                # Mock signals
                worker.signals.finished.emit = MagicMock()

                # Process query sollte Error zurückgeben
                worker.process_query("Test query")

                # Finished signal sollte mit Fehlermeldung emittiert werden
                assert worker.signals.finished.emit.called
                response = worker.signals.finished.emit.call_args[0][0]
                assert "ERROR" in response.text or "Fehler" in response.text


class TestKaiWorkerCommandSuggestions:
    """Tests für Command Suggestion Funktionalität."""

    def test_command_suggestion_high_confidence(self, kai_worker_with_mocks):
        """Testet Command Suggestion mit hoher Confidence (>= 0.7)."""
        # Mock command suggester mit hoher Confidence
        kai_worker_with_mocks.command_suggester.suggest_command = MagicMock(
            return_value={
                "original": "ingestire text:",
                "suggestion": "ingestiere text:",
                "full_suggestion": 'ingestiere text: "..."',
                "description": "Verarbeitet einen Text und extrahiert Fakten",
                "example": 'ingestiere text: "Ein Hund ist ein Tier"',
                "confidence": 0.85,
            }
        )

        kai_worker_with_mocks.process_query("ingestire text: Test")

        # Finished signal sollte mit Vorschlag emittiert werden
        assert kai_worker_with_mocks.signals.finished.emit.called
        response = kai_worker_with_mocks.signals.finished.emit.call_args[0][0]
        assert "Meintest du" in response.text
        assert "ingestiere text:" in response.text

        # Kontext sollte gesetzt sein
        assert kai_worker_with_mocks.context.is_active()
        assert (
            kai_worker_with_mocks.context.aktion.value == "erwarte_befehl_bestaetigung"
        )

    def test_command_suggestion_low_confidence(self, kai_worker_with_mocks):
        """Testet Command Suggestion mit niedriger Confidence (< 0.7)."""
        # Mock command suggester mit niedriger Confidence
        kai_worker_with_mocks.command_suggester.suggest_command = MagicMock(
            return_value={
                "original": "was ist das",
                "suggestion": None,
                "confidence": 0.3,
            }
        )

        # Process Query sollte normal weiterlaufen (nicht bei Command Suggestion stoppen)
        kai_worker_with_mocks.process_query("was ist das?")

        # Finished signal sollte emittiert werden, aber nicht mit Command Suggestion
        assert kai_worker_with_mocks.signals.finished.emit.called
        response = kai_worker_with_mocks.signals.finished.emit.call_args[0][0]
        assert "Meintest du" not in response.text


class TestKaiWorkerPatternRecognition:
    """Tests für Pattern Recognition Funktionalität."""

    def test_pattern_recognition_with_typo_clarification(self, kai_worker_with_mocks):
        """Testet Pattern Recognition mit Tippfehler-Rückfrage."""
        if not kai_worker_with_mocks.pattern_orchestrator:
            pytest.skip("Pattern Orchestrator nicht aktiviert")

        # Mock pattern orchestrator mit User Clarification
        kai_worker_with_mocks.pattern_orchestrator.process_input = MagicMock(
            return_value={
                "needs_user_clarification": True,
                "corrected_text": "Ein Hund ist ein Tier",
                "typo_corrections": [
                    {
                        "original": "Hudn",
                        "correction": "Hund",
                        "confidence": 0.75,
                        "decision": "needs_clarification",
                    }
                ],
            }
        )

        kai_worker_with_mocks.process_query("Ein Hudn ist ein Tier")

        # Finished signal sollte mit Rückfrage emittiert werden
        assert kai_worker_with_mocks.signals.finished.emit.called
        response = kai_worker_with_mocks.signals.finished.emit.call_args[0][0]
        assert "Hudn" in response.text or "Hund" in response.text

        # Kontext sollte gesetzt sein
        assert kai_worker_with_mocks.context.is_active()
        assert kai_worker_with_mocks.context.aktion.value == "erwarte_typo_klarstellung"

    def test_pattern_recognition_with_auto_correction(self, kai_worker_with_mocks):
        """Testet Pattern Recognition mit Auto-Korrektur."""
        if not kai_worker_with_mocks.pattern_orchestrator:
            pytest.skip("Pattern Orchestrator nicht aktiviert")

        # Mock pattern orchestrator mit Auto-Korrektur
        kai_worker_with_mocks.pattern_orchestrator.process_input = MagicMock(
            return_value={
                "needs_user_clarification": False,
                "corrected_text": "Ein Hund ist ein Tier",
                "typo_corrections": [
                    {
                        "original": "Hudn",
                        "correction": "Hund",
                        "confidence": 0.95,
                        "decision": "auto_corrected",
                    }
                ],
            }
        )

        kai_worker_with_mocks.process_query("Ein Hudn ist ein Tier")

        # Query sollte normal verarbeitet werden mit korrigiertem Text
        assert kai_worker_with_mocks.signals.finished.emit.called

    def test_pattern_recognition_disabled(self, kai_worker_with_mocks):
        """Testet Verarbeitung wenn Pattern Recognition deaktiviert ist."""
        # Deaktiviere Pattern Orchestrator
        kai_worker_with_mocks.pattern_orchestrator = None

        kai_worker_with_mocks.process_query("Test query")

        # Query sollte normal verarbeitet werden
        assert kai_worker_with_mocks.signals.finished.emit.called


class TestKaiWorkerContextHandling:
    """Tests für erweiterte Context-Handling Szenarien."""

    def test_context_command_confirmation_yes(self, kai_worker_with_mocks):
        """Testet Bestätigung eines Command Suggestions."""
        # Setze Kontext für Command Bestätigung
        kai_worker_with_mocks.context.set_action("erwarte_befehl_bestaetigung")
        kai_worker_with_mocks.context.set_data(
            "command_suggestion",
            {
                "full_suggestion": 'ingestiere text: "Test"',
                "suggestion": "ingestiere text:",
            },
        )
        kai_worker_with_mocks.context.set_data("original_query", "ingestire text: Test")

        # Benutzer bestätigt mit "Ja"
        kai_worker_with_mocks.process_query("Ja")

        # Context Manager sollte den vorgeschlagenen Befehl ausführen
        assert kai_worker_with_mocks.signals.finished.emit.called

    def test_context_typo_clarification_yes(self, kai_worker_with_mocks):
        """Testet Bestätigung einer Tippfehler-Korrektur."""
        # Setze Kontext für Typo Klarstellung
        kai_worker_with_mocks.context.set_action("erwarte_typo_klarstellung")
        kai_worker_with_mocks.context.set_data(
            "pattern_result", {"corrected_text": "Ein Hund ist ein Tier"}
        )
        kai_worker_with_mocks.context.set_data(
            "original_query", "Ein Hudn ist ein Tier"
        )

        # Benutzer bestätigt mit "Ja"
        kai_worker_with_mocks.process_query("Ja")

        # Context Manager sollte den korrigierten Text verarbeiten
        assert kai_worker_with_mocks.signals.finished.emit.called


class TestKaiWorkerExecutePlan:
    """Tests für execute_plan Methode."""

    def test_execute_plan_with_empty_subgoals(self, kai_worker_with_mocks):
        """Testet execute_plan mit leerem Plan."""
        from component_5_linguistik_strukturen import MainGoal, GoalType

        plan = MainGoal(
            description="Test Plan", goal_type=GoalType.GENERIC_RESPONSE, sub_goals=[]
        )

        result = kai_worker_with_mocks.execute_plan(plan)

        # Plan sollte erfolgreich sein, aber nichts tun
        assert result is not None

    def test_execute_plan_with_exception_in_subgoal(self, kai_worker_with_mocks):
        """Testet execute_plan wenn ein SubGoal eine Exception wirft."""
        from component_5_linguistik_strukturen import (
            MainGoal,
            SubGoal,
            GoalType,
            StrategyType,
        )

        # Mock SubGoalExecutor um Exception zu werfen
        kai_worker_with_mocks.sub_goal_executor.execute = MagicMock(
            side_effect=RuntimeError("SubGoal failed")
        )

        plan = MainGoal(
            description="Test Plan with Error",
            goal_type=GoalType.ANSWER_QUESTION,
            sub_goals=[
                SubGoal(description="Failing subgoal", strategy=StrategyType.QUESTION)
            ],
        )

        result = kai_worker_with_mocks.execute_plan(plan)

        # Plan sollte Fehler behandeln
        assert result is not None


class TestKaiWorkerHelperMethods:
    """Tests für Helper-Methoden."""

    def test_create_typo_clarification(self, kai_worker_with_mocks):
        """Testet _create_typo_clarification Methode."""
        pattern_result = {
            "typo_corrections": [
                {"original": "Hudn", "correction": "Hund", "confidence": 0.75}
            ],
            "corrected_text": "Ein Hund ist ein Tier",
        }

        response = kai_worker_with_mocks._create_typo_clarification(pattern_result)

        assert isinstance(response, KaiResponse)
        assert "Hudn" in response.text or "Hund" in response.text

    def test_ingest_text_callback(self, kai_worker_with_mocks):
        """Testet _ingest_text_callback Delegierung."""
        # Mock ingestion handler
        kai_worker_with_mocks.ingestion_handler.ingest_text = MagicMock(
            return_value={
                "facts_created": 5,
                "learned_patterns": 2,
                "fallback_patterns": 1,
            }
        )

        result = kai_worker_with_mocks._ingest_text_callback("Test text")

        assert result["facts_created"] == 5
        assert result["learned_patterns"] == 2
        assert result["fallback_patterns"] == 1
        kai_worker_with_mocks.ingestion_handler.ingest_text.assert_called_once_with(
            "Test text"
        )
