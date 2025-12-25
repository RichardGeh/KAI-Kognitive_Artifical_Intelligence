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

import pytest

from component_1_netzwerk import KonzeptNetzwerk
from component_11_embedding_service import EmbeddingService

# Note: kai_worker import only needed for specific tests
# from kai_worker import KaiWorker

# Semantische Embeddings haben 384 Dimensionen (vorher 8D Featurizer)
TEST_VECTOR_DIM = 384
logger = logging.getLogger(__name__)


# ============================================================================
# MOCK SIGNAL CLASSES (must be defined BEFORE fixtures that use them)
# ============================================================================


class MockSignal:
    """
    Mock Qt signal that stores emitted values and calls callbacks.

    Provides full signal-like behavior for testing without requiring Qt:
    - connect(callback): Register a callback to be called on emit
    - disconnect(callback): Remove a registered callback
    - emit(*args): Call all registered callbacks with arguments
    - last_emit: The most recent emit arguments (tuple)
    - emit_history: List of all emit argument tuples
    """

    def __init__(self):
        self.callbacks = []
        self.last_emit = None
        self.emit_history = []

    def connect(self, callback):
        """Register a callback to be called when signal emits."""
        if callback not in self.callbacks:
            self.callbacks.append(callback)

    def disconnect(self, callback=None):
        """
        Remove a callback from the signal.

        If callback is None, removes all callbacks (Qt-like behavior).
        """
        if callback is None:
            self.callbacks.clear()
        elif callback in self.callbacks:
            self.callbacks.remove(callback)

    def emit(self, *args, **kwargs):
        """
        Emit the signal with given arguments.

        Stores args in last_emit and emit_history, then calls all
        registered callbacks. Callback errors are logged but don't
        prevent other callbacks from being called.
        """
        self.last_emit = args
        self.emit_history.append(args)
        for callback in self.callbacks:
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.warning(f"MockSignal callback error: {e}")

    def clear_history(self):
        """Clear emit history and last_emit."""
        self.last_emit = None
        self.emit_history.clear()


class MockSignals:
    """
    Complete mock signals object with ALL KaiWorker signals.

    This class provides mock implementations of all signals used by
    KaiWorker, enabling testing without Qt event loop.

    Signal Categories:
    - Core response: finished, response_ready, error
    - Proof tree: proof_tree_updated, reasoning_step
    - Goal tracking: set_main_goal, clear_goals, add_sub_goal,
                     update_sub_goal_status, update_subgoal
    - UI updates: inner_picture_update, preview_confirmation_needed
    - Progress: progress, status_update
    """

    def __init__(self):
        # Core response signals
        self.finished = MockSignal()
        self.response_ready = MockSignal()
        self.error = MockSignal()

        # Proof tree signals
        self.proof_tree_updated = MockSignal()
        self.reasoning_step = MockSignal()

        # Goal tracking signals
        self.set_main_goal = MockSignal()
        self.clear_goals = MockSignal()
        self.add_sub_goal = MockSignal()
        self.update_sub_goal_status = MockSignal()
        self.update_subgoal = MockSignal()

        # UI update signals
        self.inner_picture_update = MockSignal()
        self.preview_confirmation_needed = MockSignal()

        # Progress and status signals
        self.progress = MockSignal()
        self.status_update = MockSignal()

    def reset_all(self):
        """Reset all signal histories (useful between test cases)."""
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, MockSignal):
                attr.clear_history()


# ============================================================================
# SESSION AND FUNCTION FIXTURES
# ============================================================================


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
def kai_worker_with_mocks(netzwerk_session, embedding_service_session):
    """
    Erstellt einen KaiWorker mit Mock-Signals fuer Tests.

    Uses MockSignals class (defined above) to provide signal-like
    behavior without requiring Qt event loop.
    """
    # Import here to avoid circular dependency at module load time
    from kai_worker import KaiWorker

    worker = KaiWorker(netzwerk_session, embedding_service_session)

    # Use MockSignals class defined above (no import needed)
    worker.signals = MockSignals()

    assert (
        worker.is_initialized_successfully
    ), "KAI Worker Initialisierung fehlgeschlagen"
    return worker


# ============================================================================
# SCENARIO TESTING FIXTURES (for integration_scenarios/)
# ============================================================================
# Note: MockSignal and MockSignals classes are defined above in
# "MOCK SIGNAL CLASSES" section to ensure they're available to all fixtures.


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
    - Full MockSignals with callback support
    """
    from kai_worker import KaiWorker

    worker = KaiWorker(netzwerk_session, embedding_service_session)
    worker.signals = MockSignals()

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


# ============================================================================
# PATTERN DISCOVERY SYSTEM FIXTURES
# ============================================================================


@pytest.fixture(scope="function")
def clean_pattern_data(netzwerk_session):
    """
    Clean up all Pattern Discovery nodes before and after tests.
    Removes: Utterance, Token, Pattern, PatternItem, Slot, AllowedLemma nodes.
    """

    def cleanup():
        with netzwerk_session.driver.session(database="neo4j") as session:
            session.run(
                """
                MATCH (n) WHERE n:Utterance OR n:Token OR n:Pattern
                    OR n:PatternItem OR n:Slot OR n:AllowedLemma
                DETACH DELETE n
            """
            )
            logger.info("[CLEANUP] All Pattern Discovery nodes deleted.")

    cleanup()  # Before test
    yield
    cleanup()  # After test


@pytest.fixture(scope="function")
def utterance_manager(netzwerk_session, clean_pattern_data):
    """Provides UtteranceManager instance with clean database."""
    from component_1_netzwerk_utterances import UtteranceManager

    return UtteranceManager(netzwerk_session.driver)


@pytest.fixture(scope="function")
def pattern_manager(netzwerk_session, clean_pattern_data):
    """Provides PatternManager instance with clean database."""
    from component_1_netzwerk_patterns import PatternManager

    return PatternManager(netzwerk_session.driver)


@pytest.fixture(scope="session")
def embedding_384d():
    """Valid 384-dimensional embedding vector."""
    return [0.1] * 384


@pytest.fixture(scope="function")
def sample_utterance_ids(utterance_manager, embedding_384d):
    """Create 10 sample utterances and return their IDs."""
    utterance_ids = []
    for i in range(10):
        uid = utterance_manager.create_utterance(
            text=f"Test utterance number {i}",
            embedding=embedding_384d,
            user_id=f"user_{i}",
        )
        utterance_ids.append(uid)
    return utterance_ids


@pytest.fixture(scope="session")
def sample_utterances_json():
    """Sample utterance data with tokens (from test scenarios)."""
    return {
        "simple_question": {
            "text": "Was ist ein Hund?",
            "tokens": [
                {"surface": "Was", "lemma": "was", "pos": "PRON", "idx": 0},
                {"surface": "ist", "lemma": "sein", "pos": "AUX", "idx": 1},
                {"surface": "ein", "lemma": "ein", "pos": "DET", "idx": 2},
                {"surface": "Hund", "lemma": "hund", "pos": "NOUN", "idx": 3},
            ],
        },
        "complex_declarative": {
            "text": "Der schnelle braune Fuchs springt ueber den faulen Hund.",
            "tokens": [
                {"surface": "Der", "lemma": "der", "pos": "DET", "idx": 0},
                {"surface": "schnelle", "lemma": "schnell", "pos": "ADJ", "idx": 1},
                {"surface": "braune", "lemma": "braun", "pos": "ADJ", "idx": 2},
                {"surface": "Fuchs", "lemma": "fuchs", "pos": "NOUN", "idx": 3},
                {"surface": "springt", "lemma": "springen", "pos": "VERB", "idx": 4},
                {"surface": "ueber", "lemma": "ueber", "pos": "ADP", "idx": 5},
                {"surface": "den", "lemma": "der", "pos": "DET", "idx": 6},
                {"surface": "faulen", "lemma": "faul", "pos": "ADJ", "idx": 7},
                {"surface": "Hund", "lemma": "hund", "pos": "NOUN", "idx": 8},
            ],
        },
    }


@pytest.fixture(scope="function")
def pattern_matcher(netzwerk_session, clean_pattern_data):
    """Provides TemplatePatternMatcher instance with clean database."""
    from component_61_pattern_matcher import TemplatePatternMatcher

    return TemplatePatternMatcher(netzwerk_session)


@pytest.fixture(scope="function")
def pattern_bootstrap(netzwerk_session):
    """Provides pattern bootstrap functions."""
    from component_61_pattern_bootstrap import (
        check_bootstrap_status,
        load_seed_templates,
    )

    return {
        "load_seed_templates": lambda path: load_seed_templates(netzwerk_session, path),
        "check_bootstrap_status": lambda: check_bootstrap_status(netzwerk_session),
    }


@pytest.fixture(scope="session")
def test_templates_yml_path(tmp_path_factory):
    """Create minimal test_templates.yml for fast tests.

    NOTE: LITERAL values use lemmas (not surface forms) to match
    pattern matcher's anchor token extraction which uses token lemmas.
    """
    import yaml

    templates_dir = tmp_path_factory.mktemp("templates")
    test_yaml_path = templates_dir / "test_templates.yml"

    data = {
        "templates": [
            {
                "id": "test_wh_simple",
                "name": "Test WH Simple",
                "category": "Question",
                "pattern": [
                    {"kind": "SLOT", "slot_type": "WH_WORD", "allowed": ["was", "wer"]},
                    {"kind": "LITERAL", "value": "sein"},  # Lemma of "ist"
                    {"kind": "SLOT", "slot_type": "SUBJECT", "allowed": []},
                    {"kind": "LITERAL", "value": "?"},
                ],
            },
            {
                "id": "test_statement",
                "name": "Test Statement",
                "category": "Statement",
                "pattern": [
                    {"kind": "SLOT", "slot_type": "SUBJECT", "allowed": []},
                    {"kind": "LITERAL", "value": "sein"},  # Lemma of "ist"
                    {"kind": "SLOT", "slot_type": "PREDICATE", "allowed": []},
                ],
            },
            {
                "id": "test_command",
                "name": "Test Command",
                "category": "Command",
                "pattern": [
                    {"kind": "LITERAL", "value": "zeigen"},  # Lemma of "zeige"
                    {"kind": "SLOT", "slot_type": "OBJECT", "allowed": []},
                ],
            },
        ]
    }

    with open(test_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True)

    return str(test_yaml_path)


@pytest.fixture(scope="session")
def sample_german_utterances():
    """Sample German utterances for tests."""
    return [
        "Was ist ein Hund?",
        "Wer ist das?",
        "Wie viele Hunde gibt es?",
        "Wo ist der Park?",
        "Wann ist das Treffen?",
        "Ein Hund ist ein Tier.",
        "Der Himmel ist blau.",
        "Zeige mir alle Hunde",
        "Lerne: Ein Vogel kann fliegen",
        "Was bedeutet Hund?",
    ]


@pytest.fixture(scope="function")
def preprocessor_mock():
    """Mock preprocessor for token creation tests."""
    from unittest.mock import MagicMock

    import spacy

    # Try to load real spaCy model, fallback to mock
    try:
        nlp = spacy.load("de_core_news_sm")
        return nlp
    except OSError:
        # Fallback to mock if spaCy not available
        mock = MagicMock()
        mock_doc = MagicMock()
        mock_doc.__iter__ = lambda self: iter([])
        mock_doc.__len__ = lambda self: 0
        mock.return_value = mock_doc
        return mock


# ============================================================================
# NEO4J MAINTENANCE FIXTURES
# ============================================================================


@pytest.fixture(scope="function")
def neo4j_cleanup(netzwerk_session):
    """
    Cleanup Neo4j test data before and after each test.

    Uses the neo4j_maintenance module for efficient batch deletion.
    Only cleans up nodes with 'test_' prefix to preserve real data.
    """
    from scripts.neo4j_maintenance import clear_test_data

    # Cleanup before test
    clear_test_data(netzwerk_session.driver, prefix="test_")

    yield

    # Cleanup after test
    clear_test_data(netzwerk_session.driver, prefix="test_")


@pytest.fixture(scope="session")
def neo4j_health_check():
    """
    Verify Neo4j connection at session start.

    If connection fails, provides helpful error message about starting
    Neo4j Desktop manually.
    """
    from scripts.neo4j_maintenance import check_connection

    if not check_connection():
        pytest.fail(
            "Neo4j is not responding!\n"
            "Please ensure Neo4j Desktop is running and the database is started.\n"
            "Manual steps:\n"
            "  1. Open Neo4j Desktop\n"
            "  2. Start the database\n"
            "  3. Re-run tests"
        )

    return True
