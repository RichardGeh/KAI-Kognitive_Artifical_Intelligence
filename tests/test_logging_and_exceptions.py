"""
test_logging_and_exceptions.py

Testsuite für das neue Logging-System und die Exception-Hierarchie.
Verifiziert:
- Strukturiertes Logging mit Kontext-Informationen
- Performance-Tracking
- Exception-Wrapping und -Verkettung
- Log-Datei-Erstellung
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import time
from pathlib import Path

import pytest

from component_15_logging_config import (
    DEFAULT_LOG_FILE,
    ERROR_LOG_FILE,
    LOG_DIR,
    PERFORMANCE_LOG_FILE,
    PerformanceLogger,
    WindowsSafeRotatingFileHandler,
    get_logger,
    setup_logging,
)
from kai_exceptions import (
    ConceptNotFoundError,
    ExtractionRuleError,
    KAIException,
    Neo4jConnectionError,
    Neo4jQueryError,
    ParsingError,
    wrap_exception,
)


class TestLoggingSystem:
    """Tests für das zentrale Logging-System"""

    def setup_method(self):
        """Setup vor jedem Test - initialisiere Logging"""
        setup_logging(console_level=logging.DEBUG, file_level=logging.DEBUG)
        self.logger = get_logger("test_logger")

    def test_logger_creation(self):
        """Test: Logger kann erstellt werden"""
        logger = get_logger("test_component")
        assert logger is not None
        assert logger.logger.name == "test_component"

    def test_structured_logging_with_context(self):
        """Test: Logging mit strukturierten Extra-Informationen"""
        # Teste verschiedene Log-Levels mit Kontext
        self.logger.debug("Debug-Nachricht", extra={"test_param": "debug_value"})
        self.logger.info("Info-Nachricht", extra={"user_query": "test query"})
        self.logger.warning("Warnung", extra={"threshold": 15.0})
        self.logger.error("Fehler", extra={"component": "test", "operation": "test_op"})

        # Keine Assertions nötig - Test erfolgreich wenn keine Exception

    def test_log_files_created(self):
        """Test: Log-Dateien werden erstellt"""
        # Logge eine Nachricht, um Datei-Erstellung zu triggern
        self.logger.info("Test-Nachricht für Datei-Erstellung")

        # Verifiziere dass Log-Verzeichnis existiert
        assert LOG_DIR.exists()
        assert LOG_DIR.is_dir()

        # Verifiziere dass Haupt-Log-Datei existiert
        assert DEFAULT_LOG_FILE.exists()

    def test_error_log_separation(self):
        """Test: Error-Logs werden separat gespeichert"""
        self.logger.error("Test-Fehler für Error-Log")

        # Warte kurz damit Logs geschrieben werden
        time.sleep(0.1)

        # Verifiziere dass Error-Log-Datei existiert
        assert ERROR_LOG_FILE.exists()

    def test_windows_safe_handler_used(self):
        """Test: WindowsSafeRotatingFileHandler wird verwendet"""
        # Verifiziere dass alle File-Handler die Windows-sichere Variante sind
        root_logger = logging.getLogger()
        file_handlers = [
            h
            for h in root_logger.handlers
            if isinstance(h, logging.handlers.RotatingFileHandler)
        ]

        # Mindestens ein Handler sollte existieren
        assert len(file_handlers) > 0

        # Alle sollten WindowsSafeRotatingFileHandler sein
        for handler in file_handlers:
            assert isinstance(
                handler, WindowsSafeRotatingFileHandler
            ), f"Handler {handler} ist kein WindowsSafeRotatingFileHandler"

    def test_windows_safe_handler_rotation_error_handling(self):
        """Test: Handler behandelt Rotation-Fehler graceful"""
        import os
        import tempfile

        # Erstelle temporäres Log-File
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".log") as f:
            temp_log = f.name

        try:
            # Erstelle Handler mit sehr kleiner Größe (2 KB)
            handler = WindowsSafeRotatingFileHandler(
                temp_log, maxBytes=2048, backupCount=2, encoding="utf-8"
            )
            handler.setFormatter(logging.Formatter("%(message)s"))

            # Erstelle temporären Logger
            test_logger = logging.getLogger("test_rotation")
            test_logger.addHandler(handler)
            test_logger.setLevel(logging.INFO)

            # Schreibe genug Logs, um Rotation zu triggern (2KB)
            for i in range(100):
                test_logger.info(f"Test-Nachricht {i} " + "X" * 50)

            # Verifiziere dass Rotation-Error-Counter existiert
            assert hasattr(handler, "rotation_errors")
            assert isinstance(handler.rotation_errors, int)

            # Handler sollte weiterhin funktionieren (keine Exception)
            test_logger.info("Nach Rotation")

            # Cleanup
            test_logger.removeHandler(handler)
            handler.close()

        finally:
            # Cleanup temp files
            if os.path.exists(temp_log):
                os.remove(temp_log)
            for i in range(1, 3):
                backup = f"{temp_log}.{i}"
                if os.path.exists(backup):
                    os.remove(backup)

    @pytest.mark.slow
    def test_performance_logger_context_manager(self):
        """Test: PerformanceLogger tracked Ausführungszeit"""
        test_operation = "test_operation"

        with PerformanceLogger(self.logger, test_operation, param1="value1"):
            # Simuliere Operation
            time.sleep(0.05)  # 50ms

        # Performance-Log-Datei sollte existieren
        assert PERFORMANCE_LOG_FILE.exists()

    @pytest.mark.slow
    def test_performance_logger_with_exception(self):
        """Test: PerformanceLogger loggt auch bei Exceptions"""
        test_operation = "failing_operation"

        with pytest.raises(ValueError):
            with PerformanceLogger(self.logger, test_operation, param="test"):
                raise ValueError("Test-Exception")

        # Exception sollte propagiert werden, aber Performance-Log erstellt

    def test_log_exception_method(self):
        """Test: log_exception() loggt vollständigen Traceback"""
        try:
            # Erzeuge verschachtelte Exception
            try:
                int("invalid")
            except ValueError as e:
                raise TypeError("Wrapper-Exception") from e
        except TypeError as e:
            self.logger.log_exception(
                e, message="Test-Exception-Logging", context_param="test"
            )

        # Verifiziere dass Error-Log geschrieben wurde
        assert ERROR_LOG_FILE.exists()


class TestExceptionHierarchy:
    """Tests für die KAI-Exception-Hierarchie"""

    def test_base_exception_creation(self):
        """Test: Basis-Exception kann erstellt werden"""
        exc = KAIException("Test-Nachricht", context={"key": "value"})
        assert str(exc) == "Test-Nachricht | Context: key=value"
        assert exc.context == {"key": "value"}

    def test_exception_with_original_exception(self):
        """Test: Exception-Verkettung funktioniert"""
        original = ValueError("Original-Fehler")
        exc = KAIException("Wrapper-Fehler", original_exception=original)

        assert "Caused by: ValueError: Original-Fehler" in str(exc)
        assert exc.original_exception is original

    def test_neo4j_connection_error(self):
        """Test: Neo4jConnectionError mit Kontext"""
        exc = Neo4jConnectionError(
            "Verbindung fehlgeschlagen",
            context={"uri": "bolt://localhost:7687", "user": "neo4j"},
        )

        assert "Verbindung fehlgeschlagen" in str(exc)
        assert exc.context["uri"] == "bolt://localhost:7687"

    def test_neo4j_query_error_with_query(self):
        """Test: Neo4jQueryError speichert Query-Informationen"""
        exc = Neo4jQueryError(
            "Query fehlgeschlagen", query="MATCH (n) RETURN n", parameters={"limit": 10}
        )

        assert exc.context["query"] == "MATCH (n) RETURN n"
        assert exc.context["parameters"] == {"limit": 10}

    def test_concept_not_found_error(self):
        """Test: ConceptNotFoundError mit Konzept-Name"""
        exc = ConceptNotFoundError(
            "Konzept nicht gefunden", concept_name="unbekanntes_wort"
        )

        assert exc.context["concept_name"] == "unbekanntes_wort"

    def test_extraction_rule_error(self):
        """Test: ExtractionRuleError mit Pattern-Informationen"""
        exc = ExtractionRuleError(
            "Ungültiges Pattern", rule_id="test_rule", pattern=r"^(.+)$"
        )

        assert exc.context["rule_id"] == "test_rule"
        assert exc.context["pattern"] == r"^(.+)$"

    def test_exception_wrapping(self):
        """Test: wrap_exception() wandelt generische Exceptions um"""
        try:
            raise ValueError("Original-Fehler")
        except ValueError as e:
            wrapped = wrap_exception(
                e, ParsingError, "Parsing fehlgeschlagen", input_text="invalid input"
            )

            assert isinstance(wrapped, ParsingError)
            assert isinstance(wrapped, KAIException)
            assert wrapped.original_exception is e
            assert wrapped.context["input_text"] == "invalid input"
            assert "Original-Fehler" in str(wrapped)

    def test_exception_inheritance(self):
        """Test: Exception-Hierarchie ist korrekt"""
        # Neo4jConnectionError ist eine DatabaseException
        exc = Neo4jConnectionError("Test")
        assert isinstance(exc, KAIException)

        # ConceptNotFoundError ist eine KnowledgeException
        exc2 = ConceptNotFoundError("Test")
        assert isinstance(exc2, KAIException)

    def test_exception_repr(self):
        """Test: __repr__() liefert brauchbare Darstellung"""
        exc = KAIException("Test", context={"key": "value"})
        repr_str = repr(exc)

        assert "KAIException" in repr_str
        assert "Test" in repr_str
        assert "key" in repr_str


class TestIntegrationLoggingAndExceptions:
    """Integrationstests: Logging + Exceptions zusammen"""

    def setup_method(self):
        setup_logging(console_level=logging.DEBUG)
        self.logger = get_logger("integration_test")

    def test_log_kai_exception(self):
        """Test: KAI-Exceptions werden korrekt geloggt"""
        try:
            raise Neo4jQueryError(
                "Test-Query-Fehler", query="MATCH (n) RETURN n", parameters={"limit": 5}
            )
        except Neo4jQueryError as e:
            self.logger.log_exception(e, "Test-Exception-Handling", component="test")

        # Error-Log sollte erstellt worden sein
        assert ERROR_LOG_FILE.exists()

    @pytest.mark.slow
    def test_performance_tracking_with_exceptions(self):
        """Test: Performance-Tracking funktioniert auch bei Fehlern"""
        operation_name = "failing_db_operation"

        try:
            with PerformanceLogger(self.logger, operation_name, query="test"):
                time.sleep(0.01)
                raise Neo4jQueryError("Simulierter DB-Fehler", query="TEST")
        except Neo4jQueryError:
            pass  # Exception wird erwartet

        # Performance-Log sollte trotzdem geschrieben worden sein
        assert PERFORMANCE_LOG_FILE.exists()

    def test_wrapped_exception_logging(self):
        """Test: Gewrappte Exceptions behalten Original-Kontext"""
        try:
            try:
                # Simuliere Parsing-Fehler
                raise ValueError("Ungültiger Wert: 'xyz'")
            except ValueError as e:
                raise wrap_exception(
                    e,
                    ParsingError,
                    "Konnte Eingabe nicht parsen",
                    input_text="xyz",
                    expected_type="int",
                )
        except ParsingError as e:
            self.logger.log_exception(e, "Parsing fehlgeschlagen")

            # Verifiziere Kontext-Informationen
            assert e.context["input_text"] == "xyz"
            assert e.context["expected_type"] == "int"
            assert isinstance(e.original_exception, ValueError)


# Performance-Benchmark (optional, nicht automatisch ausgeführt)
@pytest.mark.skip(reason="Performance-Benchmark, manuell ausführen")
def test_logging_performance_benchmark():
    """Benchmark: Logging-Performance bei vielen Nachrichten"""
    setup_logging()
    logger = get_logger("benchmark")

    iterations = 1000
    start_time = time.time()

    for i in range(iterations):
        logger.info(f"Benchmark-Nachricht {i}", extra={"iteration": i, "data": "test"})

    duration = time.time() - start_time
    avg_per_log = (duration / iterations) * 1000  # in ms

    print(f"\nLogging-Performance:")
    print(f"  {iterations} Logs in {duration:.3f}s")
    print(f"  Durchschnitt: {avg_per_log:.3f}ms pro Log-Eintrag")

    # Performance-Ziel: < 1ms pro Log-Eintrag
    assert avg_per_log < 1.0, f"Logging zu langsam: {avg_per_log:.3f}ms"


if __name__ == "__main__":
    # Führe Tests aus
    pytest.main([__file__, "-v", "--tb=short"])
