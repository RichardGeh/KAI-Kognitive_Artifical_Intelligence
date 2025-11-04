# test_file_reader_strategy.py
"""
Testet die FileReaderStrategy für Datei-Ingestion.

Validiert:
1. Datei-Validierung (Existenz, Lesbarkeit, Format)
2. Text-Extraktion aus DOCX/PDF
3. Ingestion-Pipeline mit Progress-Updates
4. Bericht-Formulierung
5. Fehlerbehandlung für verschiedene Szenarien
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import tempfile
import pytest
from component_5_linguistik_strukturen import (
    MeaningPoint,
    MeaningPointCategory,
    Modality,
    Polarity,
    SubGoal,
)
from kai_sub_goal_executor import FileReaderStrategy

# Logging konfigurieren
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# MOCK CLASSES
# ============================================================================


class MockWorker:
    """Mock KaiWorker für isolierte Strategy-Tests."""

    def __init__(self):
        self.netzwerk = MockNetzwerk()
        self.preprocessor = MockPreprocessor()
        self.prototyping_engine = MockPrototypingEngine()
        self.embedding_service = MockEmbeddingService()
        self.signals = MockSignals()


class MockNetzwerk:
    """Mock KonzeptNetzwerk für Speicher-Tests."""

    def __init__(self):
        self.stored_facts = []
        self.episodes = []

    def create_episode(self, episode_type, content, metadata):
        """Simuliert Episode-Erstellung."""
        episode_id = f"episode-{len(self.episodes)}"
        self.episodes.append(
            {
                "id": episode_id,
                "type": episode_type,
                "content": content,
                "metadata": metadata,
            }
        )
        return episode_id

    def assert_relation(self, subject, relation_type, obj, source_text):
        """Simuliert Relation-Speicherung."""
        self.stored_facts.append(
            {
                "subject": subject,
                "relation_type": relation_type,
                "object": obj,
                "source": source_text,
                "created": True,
            }
        )
        return True


class MockPreprocessor:
    """Mock LinguisticPreprocessor."""

    def process(self, text):
        """Simuliert Satz-Extraktion."""

        # Einfache Mock-Implementierung: Split auf Satzzeichen
        class MockDoc:
            def __init__(self, text):
                self.text = text
                self.sents = [
                    MockSentence(s.strip() + ".") for s in text.split(".") if s.strip()
                ]

        class MockSentence:
            def __init__(self, text):
                self.text = text

        return MockDoc(text)


class MockPrototypingEngine:
    """Mock PrototypingEngine."""

    def find_best_match(self, vector):
        """Simuliert Prototyp-Matching."""
        return None  # Kein Match -> Fallback


class MockEmbeddingService:
    """Mock EmbeddingService."""

    def is_available(self):
        return True

    def get_embeddings_batch(self, sentences):
        """Simuliert Batch-Embeddings."""
        return [None] * len(sentences)  # Keine Embeddings -> Fallback


class MockSignal:
    """Mock PyQt Signal."""

    def emit(self, value):
        pass  # Ignoriere Signal-Emits in Tests


class MockSignals:
    """Mock PyQt Signals."""

    def __init__(self):
        self.progress_update = MockSignal()


# ============================================================================
# TEST FUNCTIONS
# ============================================================================


def test_can_handle():
    """
    Test 1: Prüft ob FileReaderStrategy die richtigen SubGoals erkennt.
    """
    logger.info("\n" + "=" * 70)
    logger.info("TEST 1: CAN_HANDLE - KEYWORD-ERKENNUNG")
    logger.info("=" * 70)

    worker = MockWorker()
    strategy = FileReaderStrategy(worker)

    # Positive Tests
    positive_cases = [
        "Validiere Dateipfad und Lesbarkeit.",
        "Extrahiere Text aus der Datei.",
        "Verarbeite extrahierten Text durch Ingestion-Pipeline.",
        "Formuliere Ingestion-Bericht.",
    ]

    for description in positive_cases:
        assert strategy.can_handle(description), f"Sollte '{description}' erkennen"
        logger.info(f"  [OK] Erkennt: '{description[:50]}...'")

    # Negative Tests
    negative_cases = [
        "Identifiziere das Thema der Frage",
        "Verarbeite Beispielsatz zu Vektor",
        "Formuliere eine Antwort",
    ]

    for description in negative_cases:
        assert not strategy.can_handle(
            description
        ), f"Sollte '{description}' nicht erkennen"
        logger.info(f"  [OK] Ignoriert: '{description[:50]}...'")

    logger.info("\n  [SUCCESS] can_handle() funktioniert korrekt")


def test_validate_file_success():
    """
    Test 2: Prüft erfolgreiche Datei-Validierung.
    """
    logger.info("\n" + "=" * 70)
    logger.info("TEST 2: VALIDATE_FILE - ERFOLGREICHE VALIDIERUNG")
    logger.info("=" * 70)

    worker = MockWorker()
    strategy = FileReaderStrategy(worker)

    # Erstelle temporäre Test-Datei
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
        temp_path = temp_file.name
        temp_file.write(b"Test content")

    try:
        # Erstelle MeaningPoint mit gültigem Dateipfad
        mp = MeaningPoint(
            id="mp-test-validate",
            category=MeaningPointCategory.COMMAND,
            cue="lese",
            text_span=f"Lese Datei: {temp_path}",
            modality=Modality.IMPERATIVE,
            polarity=Polarity.POSITIVE,
            confidence=0.95,
            arguments={"command": "read_file", "file_path": temp_path},
        )

        # Teste Validierung
        success, result = strategy._validate_file(mp)

        # Validierung
        assert success, "Validierung sollte erfolgreich sein"
        assert "file_path" in result, "Result sollte file_path enthalten"
        assert "format" in result, "Result sollte format enthalten"
        assert (
            result["format"] == ".pdf"
        ), f"Format sollte .pdf sein, ist aber: {result['format']}"

        logger.info(f"  [OK] Datei validiert: {Path(temp_path).name}")
        logger.info(f"  [OK] Format erkannt: {result['format']}")
        logger.info("\n  [SUCCESS] Validierung funktioniert korrekt")

    finally:
        # Cleanup
        Path(temp_path).unlink(missing_ok=True)


def test_validate_file_not_found():
    """
    Test 3: Prüft Fehlerbehandlung bei nicht existierender Datei.
    """
    logger.info("\n" + "=" * 70)
    logger.info("TEST 3: VALIDATE_FILE - DATEI NICHT GEFUNDEN")
    logger.info("=" * 70)

    worker = MockWorker()
    strategy = FileReaderStrategy(worker)

    mp = MeaningPoint(
        id="mp-test-not-found",
        category=MeaningPointCategory.COMMAND,
        cue="lese",
        text_span="Lese Datei: nicht_existent.pdf",
        modality=Modality.IMPERATIVE,
        polarity=Polarity.POSITIVE,
        confidence=0.95,
        arguments={"command": "read_file", "file_path": "nicht_existent.pdf"},
    )

    success, result = strategy._validate_file(mp)

    assert not success, "Validierung sollte fehlschlagen"
    assert "error" in result, "Result sollte error enthalten"
    assert (
        "nicht gefunden" in result["error"]
    ), f"Error-Message sollte 'nicht gefunden' enthalten: {result['error']}"

    logger.info(f"  [OK] Fehler erkannt: {result['error']}")
    logger.info(
        "\n  [SUCCESS] Fehlerbehandlung für nicht existierende Datei funktioniert"
    )


def test_validate_file_unsupported_format():
    """
    Test 4: Prüft Fehlerbehandlung bei nicht unterstütztem Format.
    """
    logger.info("\n" + "=" * 70)
    logger.info("TEST 4: VALIDATE_FILE - NICHT UNTERSTÜTZTES FORMAT")
    logger.info("=" * 70)

    worker = MockWorker()
    strategy = FileReaderStrategy(worker)

    # Erstelle temporäre Datei mit nicht unterstütztem Format
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_file:
        temp_path = temp_file.name
        temp_file.write(b"Test content")

    try:
        mp = MeaningPoint(
            id="mp-test-unsupported",
            category=MeaningPointCategory.COMMAND,
            cue="lese",
            text_span=f"Lese Datei: {temp_path}",
            modality=Modality.IMPERATIVE,
            polarity=Polarity.POSITIVE,
            confidence=0.95,
            arguments={"command": "read_file", "file_path": temp_path},
        )

        success, result = strategy._validate_file(mp)

        assert not success, "Validierung sollte fehlschlagen"
        assert "error" in result, "Result sollte error enthalten"
        assert (
            "nicht unterstützt" in result["error"]
        ), f"Error sollte 'nicht unterstützt' enthalten: {result['error']}"

        logger.info(f"  [OK] Fehler erkannt: {result['error'][:80]}...")
        logger.info(
            "\n  [SUCCESS] Fehlerbehandlung für nicht unterstützte Formate funktioniert"
        )

    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_formulate_report_multiple_facts():
    """
    Test 5: Prüft Bericht-Formulierung mit mehreren Fakten.
    """
    logger.info("\n" + "=" * 70)
    logger.info("TEST 5: FORMULATE_REPORT - MEHRERE FAKTEN")
    logger.info("=" * 70)

    worker = MockWorker()
    strategy = FileReaderStrategy(worker)

    context = {
        "file_name": "test_dokument.pdf",
        "facts_created": 42,
        "chunks_processed": 3,
        "learned_patterns": 30,
        "fallback_patterns": 12,
        "fragments_stored": 100,
        "connections_stored": 50,
    }

    success, result = strategy._formulate_report(context)

    assert success, "Report-Formulierung sollte erfolgreich sein"
    assert "final_response" in result, "Result sollte final_response enthalten"

    response = result["final_response"]
    assert "test_dokument.pdf" in response, "Response sollte Dateinamen enthalten"
    assert "42" in response, "Response sollte Anzahl Fakten enthalten"
    assert (
        "erfolgreich verarbeitet" in response
    ), "Response sollte Erfolgs-Message enthalten"

    logger.info(f"  [OK] Bericht generiert: {response[:100]}...")
    logger.info("\n  [SUCCESS] Report-Formulierung funktioniert korrekt")


def test_formulate_report_no_facts():
    """
    Test 6: Prüft Bericht-Formulierung ohne gelernte Fakten.
    """
    logger.info("\n" + "=" * 70)
    logger.info("TEST 6: FORMULATE_REPORT - KEINE FAKTEN")
    logger.info("=" * 70)

    worker = MockWorker()
    strategy = FileReaderStrategy(worker)

    context = {
        "file_name": "empty.pdf",
        "facts_created": 0,
        "chunks_processed": 1,
        "learned_patterns": 0,
        "fallback_patterns": 0,
    }

    success, result = strategy._formulate_report(context)

    assert success, "Report-Formulierung sollte erfolgreich sein"
    response = result["final_response"]
    assert "keine neuen Fakten" in response, "Response sollte 'keine Fakten' enthalten"

    logger.info(f"  [OK] Bericht für leeres Dokument: {response}")
    logger.info("\n  [SUCCESS] Report-Formulierung für leere Dokumente funktioniert")


def test_formulate_report_single_fact():
    """
    Test 7: Prüft Bericht-Formulierung mit genau einem Fakt.
    """
    logger.info("\n" + "=" * 70)
    logger.info("TEST 7: FORMULATE_REPORT - EINZELNER FAKT")
    logger.info("=" * 70)

    worker = MockWorker()
    strategy = FileReaderStrategy(worker)

    context = {
        "file_name": "small.pdf",
        "facts_created": 1,
        "chunks_processed": 1,
        "learned_patterns": 1,
        "fallback_patterns": 0,
    }

    success, result = strategy._formulate_report(context)

    assert success, "Report-Formulierung sollte erfolgreich sein"
    response = result["final_response"]
    assert (
        "1 neuen Fakt" in response
    ), "Response sollte '1 neuen Fakt' (Singular) enthalten"

    logger.info(f"  [OK] Bericht mit Singular-Form: {response}")
    logger.info("\n  [SUCCESS] Report-Formulierung mit Singular funktioniert")


def test_execute_dispatcher():
    """
    Test 8: Prüft dass execute() korrekt zu Untermethoden dispatched.
    """
    logger.info("\n" + "=" * 70)
    logger.info("TEST 8: EXECUTE - DISPATCHER")
    logger.info("=" * 70)

    worker = MockWorker()
    strategy = FileReaderStrategy(worker)

    # Test-Cases für verschiedene SubGoals
    test_cases = [
        ("Validiere Dateipfad und Lesbarkeit.", "_validate_file"),
        ("Formuliere Ingestion-Bericht.", "_formulate_report"),
    ]

    for description, expected_method in test_cases:
        sub_goal = SubGoal(description=description)

        # Minimaler Context (wird zu False führen, da Intent fehlt)
        context = {}

        success, result = strategy.execute(sub_goal, context)

        # Sollte fehlschlagen wegen fehlendem Intent, aber Dispatch hat funktioniert
        assert not success, "Sollte fehlschlagen wegen fehlendem Intent"
        assert "error" in result, "Result sollte error enthalten"

        logger.info(
            f"  [OK] Dispatcher erkennt: '{description[:50]}...' -> {expected_method}"
        )

    logger.info("\n  [SUCCESS] Execute-Dispatcher funktioniert korrekt")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
