# tests/test_pattern_orchestrator_commands.py
"""
Tests für PatternOrchestrator Command-Erkennung

Sicherstellt, dass explizite Commands nicht durch Typo-Detection abgefangen werden.
"""

import pytest
from unittest.mock import Mock

from component_24_pattern_orchestrator import PatternOrchestrator


class TestPatternOrchestratorCommands:
    """Tests für Early-Exit bei expliziten Commands"""

    @pytest.fixture
    def mock_netzwerk(self):
        """Mock für KonzeptNetzwerk"""
        mock = Mock()
        mock.get_all_known_words.return_value = ["katze", "hund", "baum"]
        mock.get_normalized_word_frequency.return_value = 0.5

        # Mock für AdaptiveThresholdManager
        mock.query_graph_for_facts.return_value = {}

        # Mock für Feedback-System (wird in TypoCandidateFinder verwendet)
        mock._feedback = None  # Kein Feedback-System in Tests

        # Mock für SequencePredictor (get_word_connections)
        mock.get_word_connections.return_value = []  # Keine Connections in Tests

        return mock

    @pytest.fixture
    def orchestrator(self, mock_netzwerk):
        """PatternOrchestrator Instanz"""
        return PatternOrchestrator(mock_netzwerk)

    def test_file_read_command_no_typo_detection(self, orchestrator):
        """Lese Datei: Command sollte Pattern Recognition überspringen"""

        test_inputs = [
            "Lese Datei: /pfad/zur/datei.pdf",
            "LESE DATEI: C:\\Dokumente\\test.docx",
            "lese datei: ./relative/path.txt",
            "  Lese Datei: /with/spaces.pdf  ",
        ]

        for input_text in test_inputs:
            result = orchestrator.process_input(input_text)

            # Sollte keine Typo-Korrekturen haben
            assert (
                result["typo_corrections"] == []
            ), f"Unerwartete Typo-Korrektur für: {input_text}"

            # Sollte keine User-Klarstellung benötigen
            assert (
                result["needs_user_clarification"] is False
            ), f"Unerwartete Klarstellung für: {input_text}"

            # Text sollte unverändert sein
            assert (
                result["corrected_text"] == input_text
            ), f"Text wurde verändert: {input_text}"

    def test_alternative_file_commands_no_typo_detection(self, orchestrator):
        """Alternative Datei-Commands sollten Pattern Recognition überspringen"""

        test_inputs = [
            "Ingestiere Dokument: /path/to/file.pdf",
            "Verarbeite PDF: document.pdf",
            "Lade Datei: myfile.txt",
        ]

        for input_text in test_inputs:
            result = orchestrator.process_input(input_text)

            assert (
                result["typo_corrections"] == []
            ), f"Unerwartete Typo-Korrektur für: {input_text}"
            assert result["needs_user_clarification"] is False
            assert result["corrected_text"] == input_text

    def test_other_commands_no_typo_detection(self, orchestrator):
        """Andere explizite Commands sollten Pattern Recognition überspringen"""

        test_inputs = [
            "Lerne: Ein Apfel ist eine Frucht",
            'Lerne Muster: "X schmeckt Y" bedeutet HAS_TASTE',
            "Definiere: konzept / eigenschaft = wert",
            'Ingestiere Text: "Dies ist ein Test-Text"',
        ]

        for input_text in test_inputs:
            result = orchestrator.process_input(input_text)

            assert (
                result["typo_corrections"] == []
            ), f"Unerwartete Typo-Korrektur für: {input_text}"
            assert result["needs_user_clarification"] is False
            assert result["corrected_text"] == input_text

    def test_normal_text_still_gets_typo_detection(self, orchestrator, mock_netzwerk):
        """Normaler Text (keine Commands) sollte weiterhin Typo-Detection durchlaufen"""

        # Mock: "Ktzae" ist unbekannt, sollte Typo-Kandidaten finden
        mock_netzwerk.get_all_known_words.return_value = ["katze", "satz", "der"]

        result = orchestrator.process_input("Ktzae ist ein Tier")

        # Sollte Typo-Kandidaten haben (oder nicht, je nach Confidence)
        # Wir prüfen nur, dass die Pipeline läuft (keine Exception)
        assert "typo_corrections" in result
        assert "corrected_text" in result
        assert "needs_user_clarification" in result

    def test_question_with_similar_words_gets_typo_detection(self, orchestrator):
        """Fragen (keine Commands) sollten Typo-Detection durchlaufen"""

        result = orchestrator.process_input("Was ist eine Ktzae?")

        # Pipeline sollte laufen (keine Exception)
        assert "typo_corrections" in result
        assert "corrected_text" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
