# tests/test_meaning_extractor_file_commands.py
"""
Tests für Datei-Command-Erkennung im MeaningExtractor (Phase 2).

Testet die Pattern-Matching-Logik für:
- "lese datei:"
- "ingestiere dokument:"
- "verarbeite pdf:"
- "lade datei:"
"""

import pytest
from unittest.mock import MagicMock

from component_7_meaning_extractor import MeaningPointExtractor
from component_5_linguistik_strukturen import MeaningPointCategory, Modality
from component_6_linguistik_engine import LinguisticPreprocessor
from component_11_embedding_service import EmbeddingService


@pytest.fixture
def mock_embedding_service():
    """Mock für EmbeddingService."""
    mock = MagicMock(spec=EmbeddingService)
    mock.is_available.return_value = (
        False  # Deaktiviere Vektor-Matching für Command-Tests
    )
    return mock


@pytest.fixture
def preprocessor():
    """Echter Preprocessor für spaCy-Integration."""
    return LinguisticPreprocessor()


@pytest.fixture
def extractor(mock_embedding_service, preprocessor):
    """MeaningPointExtractor mit Mock-Dependencies."""
    return MeaningPointExtractor(
        embedding_service=mock_embedding_service,
        preprocessor=preprocessor,
        prototyping_engine=None,
    )


class TestFileCommandDetection:
    """Tests für Datei-Command-Erkennung."""

    def test_lese_datei_command(self, extractor, preprocessor):
        """Test: 'lese datei:' Command wird erkannt."""
        text = "lese datei: C:\\Users\\test\\document.txt"
        doc = preprocessor.process(text)

        mps = extractor.extract(doc)

        assert len(mps) == 1
        mp = mps[0]
        assert mp.category == MeaningPointCategory.COMMAND
        assert mp.cue == "lese datei:"
        assert mp.modality == Modality.IMPERATIVE
        assert mp.confidence == 0.95
        assert mp.arguments["command"] == "read_file"
        assert mp.arguments["file_path"] == "C:\\Users\\test\\document.txt"

    def test_ingestiere_dokument_command(self, extractor, preprocessor):
        """Test: 'ingestiere dokument:' Command wird erkannt."""
        text = "ingestiere dokument: /home/user/notes.txt"
        doc = preprocessor.process(text)

        mps = extractor.extract(doc)

        assert len(mps) == 1
        mp = mps[0]
        assert mp.category == MeaningPointCategory.COMMAND
        assert mp.cue == "ingestiere dokument:"
        assert mp.modality == Modality.IMPERATIVE
        assert mp.confidence == 0.95
        assert mp.arguments["command"] == "ingest_document"
        assert mp.arguments["file_path"] == "/home/user/notes.txt"

    def test_verarbeite_pdf_command(self, extractor, preprocessor):
        """Test: 'verarbeite pdf:' Command wird erkannt."""
        text = "verarbeite pdf: ./documents/paper.pdf"
        doc = preprocessor.process(text)

        mps = extractor.extract(doc)

        assert len(mps) == 1
        mp = mps[0]
        assert mp.category == MeaningPointCategory.COMMAND
        assert mp.cue == "verarbeite pdf:"
        assert mp.modality == Modality.IMPERATIVE
        assert mp.confidence == 0.95
        assert mp.arguments["command"] == "process_pdf"
        assert mp.arguments["file_path"] == "./documents/paper.pdf"

    def test_lade_datei_command(self, extractor, preprocessor):
        """Test: 'lade datei:' Command wird erkannt."""
        text = "lade datei: ../data/input.csv"
        doc = preprocessor.process(text)

        mps = extractor.extract(doc)

        assert len(mps) == 1
        mp = mps[0]
        assert mp.category == MeaningPointCategory.COMMAND
        assert mp.cue == "lade datei:"
        assert mp.modality == Modality.IMPERATIVE
        assert mp.confidence == 0.95
        assert mp.arguments["command"] == "load_file"
        assert mp.arguments["file_path"] == "../data/input.csv"

    def test_case_insensitive_matching(self, extractor, preprocessor):
        """Test: Commands sind case-insensitive."""
        test_cases = [
            "LESE DATEI: test.txt",
            "Lese Datei: test.txt",
            "lEsE dAtEi: test.txt",
            "INGESTIERE DOKUMENT: test.txt",
            "Verarbeite PDF: test.pdf",
            "Lade Datei: test.csv",
        ]

        for text in test_cases:
            doc = preprocessor.process(text)
            mps = extractor.extract(doc)

            assert len(mps) == 1, f"Failed for: {text}"
            assert mps[0].category == MeaningPointCategory.COMMAND
            assert mps[0].confidence == 0.95

    def test_file_path_with_spaces(self, extractor, preprocessor):
        """Test: Dateipfade mit Leerzeichen werden korrekt extrahiert."""
        text = "lese datei: C:\\Users\\John Doe\\My Documents\\notes.txt"
        doc = preprocessor.process(text)

        mps = extractor.extract(doc)

        assert len(mps) == 1
        mp = mps[0]
        assert (
            mp.arguments["file_path"] == "C:\\Users\\John Doe\\My Documents\\notes.txt"
        )

    def test_file_path_with_special_chars(self, extractor, preprocessor):
        """Test: Dateipfade mit Sonderzeichen."""
        text = "ingestiere dokument: /tmp/test-file_v2.0_(final).txt"
        doc = preprocessor.process(text)

        mps = extractor.extract(doc)

        assert len(mps) == 1
        mp = mps[0]
        assert mp.arguments["file_path"] == "/tmp/test-file_v2.0_(final).txt"

    def test_relative_paths(self, extractor, preprocessor):
        """Test: Relative Pfade werden korrekt erkannt."""
        test_cases = [
            ("lese datei: ./test.txt", "./test.txt"),
            ("lese datei: ../parent/file.txt", "../parent/file.txt"),
            ("lese datei: subfolder/file.txt", "subfolder/file.txt"),
        ]

        for text, expected_path in test_cases:
            doc = preprocessor.process(text)
            mps = extractor.extract(doc)

            assert len(mps) == 1
            assert mps[0].arguments["file_path"] == expected_path

    def test_whitespace_handling(self, extractor, preprocessor):
        """Test: Whitespace wird korrekt behandelt."""
        test_cases = [
            "lese datei:test.txt",  # Kein Space nach Doppelpunkt
            "lese datei:  test.txt",  # Mehrere Spaces
            "  lese datei: test.txt  ",  # Leading/Trailing Whitespace
        ]

        for text in test_cases:
            doc = preprocessor.process(text)
            mps = extractor.extract(doc)

            assert len(mps) == 1, f"Failed for: {text}"
            assert mps[0].category == MeaningPointCategory.COMMAND
            # Pfad sollte getrimmt sein
            assert mps[0].arguments["file_path"].strip() == "test.txt"

    def test_no_false_positives(self, extractor, preprocessor):
        """Test: Ähnliche aber nicht-passende Inputs erzeugen keine Datei-Commands."""
        test_cases = [
            "lese die Datei test.txt",  # Falsche Syntax (kein Doppelpunkt)
            "ingestiere text: test",  # Falscher Command-Typ
            "verarbeite: test.pdf",  # Fehlendes "pdf"
            "datei lesen: test.txt",  # Falsche Reihenfolge
        ]

        for text in test_cases:
            doc = preprocessor.process(text)
            mps = extractor.extract(doc)

            # Sollte entweder nicht COMMAND sein oder einen anderen Command-Typ haben
            assert len(mps) >= 1
            if mps[0].category == MeaningPointCategory.COMMAND:
                # Falls es ein Command ist, sollte es kein Datei-Command sein
                assert mps[0].arguments.get("command") not in [
                    "read_file",
                    "ingest_document",
                    "process_pdf",
                    "load_file",
                ], f"False positive for: {text}"

    def test_text_span_preservation(self, extractor, preprocessor):
        """Test: Original-Text wird in text_span gespeichert."""
        text = "lese datei: test.txt"
        doc = preprocessor.process(text)

        mps = extractor.extract(doc)

        assert len(mps) == 1
        assert mps[0].text_span == text

    def test_confidence_score(self, extractor, preprocessor):
        """Test: Confidence für Datei-Commands ist 0.95."""
        test_cases = [
            "lese datei: test.txt",
            "ingestiere dokument: test.txt",
            "verarbeite pdf: test.pdf",
            "lade datei: test.csv",
        ]

        for text in test_cases:
            doc = preprocessor.process(text)
            mps = extractor.extract(doc)

            assert len(mps) == 1
            assert mps[0].confidence == 0.95, f"Wrong confidence for: {text}"

    def test_windows_paths(self, extractor, preprocessor):
        """Test: Windows-Pfade mit Backslashes."""
        text = "lese datei: C:\\Program Files\\App\\data.json"
        doc = preprocessor.process(text)

        mps = extractor.extract(doc)

        assert len(mps) == 1
        assert mps[0].arguments["file_path"] == "C:\\Program Files\\App\\data.json"

    def test_unix_paths(self, extractor, preprocessor):
        """Test: Unix-Pfade mit Forward-Slashes."""
        text = "ingestiere dokument: /var/log/application.log"
        doc = preprocessor.process(text)

        mps = extractor.extract(doc)

        assert len(mps) == 1
        assert mps[0].arguments["file_path"] == "/var/log/application.log"

    def test_file_extensions(self, extractor, preprocessor):
        """Test: Verschiedene Dateierweiterungen werden unterstützt."""
        test_cases = [
            ("lese datei: document.txt", ".txt"),
            ("lese datei: data.json", ".json"),
            ("verarbeite pdf: paper.pdf", ".pdf"),
            ("lade datei: spreadsheet.xlsx", ".xlsx"),
            ("ingestiere dokument: archive.zip", ".zip"),
            ("lese datei: noextension", ""),  # Keine Extension
        ]

        for text, expected_ext in test_cases:
            doc = preprocessor.process(text)
            mps = extractor.extract(doc)

            assert len(mps) == 1
            file_path = mps[0].arguments["file_path"]
            actual_ext = file_path[file_path.rfind(".") :] if "." in file_path else ""
            assert actual_ext == expected_ext, f"Failed for: {text}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
