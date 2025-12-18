# conftest.py
"""
Gemeinsame Pytest-Fixtures für alle KAI Tests.
Diese Fixtures werden automatisch von pytest geladen und sind in allen Test-Dateien verfügbar.
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from unittest.mock import MagicMock

import pytest

from component_1_netzwerk import KonzeptNetzwerk
from component_11_embedding_service import EmbeddingService

# Note: kai_worker import only needed for specific tests
# from kai_worker import KaiWorker

# Semantische Embeddings haben 384 Dimensionen (vorher 8D Featurizer)
TEST_VECTOR_DIM = 384
logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def netzwerk_session():
    """Stellt eine DB-Verbindung für die gesamte Test-Session bereit."""
    nw = KonzeptNetzwerk()
    assert nw.driver is not None, "DB-Verbindung fehlgeschlagen"

    # Verifiziere Constraints
    with nw.driver.session(database="neo4j") as session:
        result = session.run("SHOW CONSTRAINTS")
        constraints = [record["name"] for record in result]
        logger.info(f"Vorhandene Constraints: {constraints}")

    yield nw
    nw.close()


@pytest.fixture(scope="session")
def embedding_service_session():
    """Stellt einen echten Embedding-Service bereit."""
    service = EmbeddingService()
    assert service.is_available(), "Embedding-Service nicht verfügbar"
    return service


@pytest.fixture(scope="function")
def clean_test_concepts(netzwerk_session):
    """
    Stellt sicher, dass Test-Konzepte vor und nach jedem Test gelöscht werden.
    Verwendet ein Präfix für Testdaten.
    """
    test_prefix = "test_"

    def cleanup():
        with netzwerk_session.driver.session(database="neo4j") as session:
            # Lösche alle Knoten, die mit test_ beginnen
            session.run(
                """
                MATCH (n)
                WHERE n.name STARTS WITH $prefix OR n.lemma STARTS WITH $prefix
                DETACH DELETE n
            """,
                prefix=test_prefix,
            )
            logger.info(
                f"[CLEANUP] Alle Test-Konzepte mit Praefix '{test_prefix}' geloescht."
            )

    cleanup()  # Vor dem Test
    yield test_prefix
    cleanup()  # Nach dem Test


@pytest.fixture(scope="function")
def kai_worker_with_mocks(netzwerk_session):
    """Erstellt einen KaiWorker mit Mock-Signals für Tests."""
    # Import here to avoid circular dependency
    from kai_worker import KaiWorker

    embedding_service = EmbeddingService()
    worker = KaiWorker(netzwerk_session, embedding_service)
    worker.signals = MagicMock()
    assert (
        worker.is_initialized_successfully
    ), "KAI Worker Initialisierung fehlgeschlagen"
    return worker


# ============================================================================
# SCENARIO TESTING FIXTURES (for integration_scenarios/)
# ============================================================================


@pytest.fixture(scope="session")
def scenario_mode():
    """
    Enable scenario testing mode with extended timeouts and comprehensive logging.
    This fixture is automatically available to all scenario tests.
    """
    import logging

    # Enable debug logging for scenario tests
    logging.getLogger().setLevel(logging.DEBUG)
    logger.info("Scenario testing mode enabled")
    return True


@pytest.fixture(scope="function")
def kai_worker_scenario_mode(
    netzwerk_session, embedding_service_session, scenario_mode
):
    """
    KAI worker configured specifically for scenario testing.

    Includes:
    - Extended timeouts (1 hour default)
    - Comprehensive logging enabled
    - Deeper reasoning allowed (max_reasoning_depth=20)
    - All reasoning engines enabled
    """
    from kai_worker import KaiWorker

    worker = KaiWorker(netzwerk_session, embedding_service_session)
    worker.signals = MagicMock()

    # Configure for scenario testing
    if hasattr(worker, "enable_comprehensive_logging"):
        worker.enable_comprehensive_logging = True

    if hasattr(worker, "max_reasoning_depth"):
        worker.max_reasoning_depth = 20  # Allow deeper reasoning

    if hasattr(worker, "timeout_seconds"):
        worker.timeout_seconds = 3600  # 1 hour timeout

    logger.info("KAI worker configured for scenario testing mode")
    return worker


@pytest.fixture(scope="function")
def scenario_timeout():
    """
    Return timeout value for scenario tests (in seconds).
    Can be overridden by SCENARIO_TIMEOUT environment variable.
    """
    import os

    return int(os.environ.get("SCENARIO_TIMEOUT", "3600"))  # Default 1 hour


@pytest.fixture(scope="function")
def progress_reporter(request):
    """
    Progress reporter for long-running scenario tests.
    Automatically configured with test name.
    """
    from tests.integration_scenarios.utils.progress_reporter import ProgressReporter

    test_name = request.node.name
    reporter = ProgressReporter(test_name=test_name, total_steps=10)
    return reporter


@pytest.fixture(scope="function")
def scenario_logger(request, tmp_path):
    """
    Comprehensive logger for scenario test execution.
    Logs are saved to tests/integration_scenarios/results/ directory.
    """
    from tests.integration_scenarios.utils.result_logger import ScenarioLogger

    test_name = request.node.name
    # Create results directory
    results_dir = Path(__file__).parent / "integration_scenarios" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    logger_instance = ScenarioLogger(scenario_name=test_name, output_dir=results_dir)
    return logger_instance


@pytest.fixture(scope="function")
def confidence_tracker():
    """
    Confidence calibration tracker for evaluating confidence vs. correctness.
    """
    from tests.integration_scenarios.utils.confidence_calibrator import (
        ConfidenceCalibrationTracker,
    )

    tracker = ConfidenceCalibrationTracker()
    return tracker
