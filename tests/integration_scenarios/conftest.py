"""
tests/integration_scenarios/conftest.py

Scenario-specific pytest fixtures for integration testing.
Provides fixtures for KAI worker, logging, progress tracking, and confidence tracking.

NO Unicode characters - ASCII only (see CLAUDE.md encoding rules).

NOTE: This module imports MockSignals from tests.conftest to avoid duplication.
The MockSignals class provides all 15+ signals that KaiWorker requires.
"""

import json
import os
from pathlib import Path

import pytest

from tests.conftest import MockSignals
from tests.integration_scenarios.utils.confidence_calibrator import (
    ConfidenceCalibrationTracker,
)
from tests.integration_scenarios.utils.progress_reporter import ProgressReporter
from tests.integration_scenarios.utils.result_logger import ScenarioLogger


@pytest.fixture(scope="session")
def scenario_output_dir():
    """Create and return output directory for scenario test results"""
    output_dir = Path("tests/integration_scenarios/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture(scope="function")
def kai_with_full_instrumentation(netzwerk_session, embedding_service_session):
    """
    KAI worker with full instrumentation for scenario testing.

    Includes:
    - All reasoning engines enabled
    - Comprehensive logging
    - ProofTree tracking
    - Confidence tracking
    - Memory tracking

    Uses MockSignals from tests.conftest which provides all 15+ signals
    that KaiWorker requires (response_ready, proof_tree_updated, reasoning_step,
    finished, error, set_main_goal, clear_goals, add_sub_goal, update_sub_goal_status,
    update_subgoal, inner_picture_update, preview_confirmation_needed, progress,
    status_update).
    """
    from kai_worker import KaiWorker

    worker = KaiWorker(netzwerk_session, embedding_service_session)

    # Use complete MockSignals from tests.conftest (15+ signals)
    worker.signals = MockSignals()

    # Enable verbose logging if method exists
    if hasattr(worker, "enable_comprehensive_logging"):
        worker.enable_comprehensive_logging()

    return worker


@pytest.fixture(scope="function")
def scenario_logger(scenario_output_dir):
    """Create a scenario logger for the current test"""
    test_name = os.environ.get("PYTEST_CURRENT_TEST", "unknown_test")
    # Clean up test name
    test_name = test_name.split("::")[-1].split(" ")[0]
    return ScenarioLogger(test_name, scenario_output_dir)


@pytest.fixture(scope="function")
def progress_reporter():
    """Create a progress reporter for long-running tests"""
    test_name = os.environ.get("PYTEST_CURRENT_TEST", "unknown_test")
    # Clean up test name
    test_name = test_name.split("::")[-1].split(" ")[0]
    return ProgressReporter(test_name, total_steps=10)  # Override in tests


@pytest.fixture(scope="function")
def confidence_tracker():
    """Create a confidence calibration tracker"""
    return ConfidenceCalibrationTracker()


@pytest.fixture(scope="session")
def baseline_scores(scenario_output_dir):
    """Load or create baseline scores for comparison"""
    baseline_file = scenario_output_dir / "baseline_scores.json"
    if baseline_file.exists():
        with open(baseline_file, encoding="utf-8") as f:
            return json.load(f)
    return {}


@pytest.fixture(scope="function", autouse=True)
def clean_graph_for_scenario(netzwerk_session):
    """
    Ensure clean graph state for each scenario test.
    Clears all data except production rules, base dictionary, and extraction rules.
    Then adds essential ExtractionRules for testing.
    """
    # Before test - clean graph (preserve essential rules)
    with netzwerk_session.driver.session(database="neo4j") as session:
        # Clear all data except production rules, base dictionary, and extraction rules
        session.run(
            """
            MATCH (n)
            WHERE NOT n:ProductionRule AND NOT n:BaseDictionary AND NOT n:ExtractionRule
            DETACH DELETE n
        """
        )

    # Add essential ExtractionRules for testing
    # NOTE: relation_type has UNIQUE constraint, so only ONE pattern per relation type
    # Patterns must use EXACTLY 2 capture groups: (subject) and (object)
    essential_rules = [
        # CAPABLE_OF: "Ein X bellt/fliegt/kann Y" -> X CAPABLE_OF verb
        # Matches: "Ein Hund bellt", "Ein Vogel fliegt", "Ein Hund kann bellen"
        (
            "CAPABLE_OF",
            r"^(?:ein(?:e)?\s+)?(.+?)\s+(bellt|fliegt|schwimmt|laeuft|springt|singt|schlaeft|frisst|trinkt|bellen|fliegen|kann\s+\w+)\.?$",
        ),
        # IS_A: "X ist ein Y" -> X IS_A Y
        ("IS_A", r"^(.+?)\s+ist\s+(?:ein(?:e)?\s+)?(.+)\.?$"),
        # HAS_PROPERTY: "X mag Y" -> X HAS_PROPERTY Y
        ("HAS_PROPERTY", r"^(.+?)\s+mag\s+(.+)\.?$"),
        # PART_OF: "X hat ein Y"
        ("PART_OF", r"^(.+?)\s+hat\s+(?:ein(?:e)?\s+)?(.+)\.?$"),
    ]

    for relation_type, pattern in essential_rules:
        netzwerk_session.create_extraction_rule(relation_type, pattern)

    yield

    # After test - clean graph again (preserve essential rules)
    with netzwerk_session.driver.session(database="neo4j") as session:
        session.run(
            """
            MATCH (n)
            WHERE NOT n:ProductionRule AND NOT n:BaseDictionary AND NOT n:ExtractionRule
            DETACH DELETE n
        """
        )


@pytest.fixture(scope="function")
def scenario_timeout():
    """Return timeout value for scenario tests (in seconds)"""
    # Can be overridden by environment variable
    return int(os.environ.get("SCENARIO_TIMEOUT", "3600"))  # Default 1 hour


@pytest.fixture(scope="function")
def kai_worker_scenario_mode(
    netzwerk_session, embedding_service_session, scenario_timeout
):
    """
    KAI worker configured specifically for scenario testing.

    Includes:
    - Extended timeouts (configurable via SCENARIO_TIMEOUT env var)
    - Comprehensive logging enabled
    - Deeper reasoning allowed (max_reasoning_depth=20)

    Uses MockSignals from tests.conftest which provides all 15+ signals
    that KaiWorker requires.
    """
    from kai_worker import KaiWorker

    worker = KaiWorker(netzwerk_session, embedding_service_session)

    # Use complete MockSignals from tests.conftest (15+ signals)
    worker.signals = MockSignals()

    # Configure for scenario testing
    if hasattr(worker, "enable_comprehensive_logging"):
        worker.enable_comprehensive_logging = True

    if hasattr(worker, "max_reasoning_depth"):
        worker.max_reasoning_depth = 20  # Allow deeper reasoning

    if hasattr(worker, "timeout_seconds"):
        worker.timeout_seconds = scenario_timeout

    return worker


@pytest.fixture(scope="session")
def scenario_mode():
    """Enable scenario testing mode with extended timeouts and logging"""
    import logging

    logging.getLogger().setLevel(logging.DEBUG)
    return True


@pytest.fixture(scope="function", autouse=True)
def disable_word_usage_tracking():
    """
    Disable word usage tracking for integration scenario tests.
    This significantly improves test performance by avoiding slow
    UsageContext database operations.
    """
    from kai_config import get_config

    config = get_config()

    # Store original value
    original_value = config.get("word_usage_tracking", True)

    # Disable for test
    config._config["word_usage_tracking"] = False

    yield

    # Restore original value
    config._config["word_usage_tracking"] = original_value
