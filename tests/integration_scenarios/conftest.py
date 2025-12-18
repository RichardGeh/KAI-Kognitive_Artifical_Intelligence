"""
tests/integration_scenarios/conftest.py

Scenario-specific pytest fixtures for integration testing.
Provides fixtures for KAI worker, logging, progress tracking, and confidence tracking.

NO Unicode characters - ASCII only (see CLAUDE.md encoding rules).
"""

import json
import os
from pathlib import Path

import pytest

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
    """
    from unittest.mock import Mock

    from kai_worker import KaiWorker

    worker = KaiWorker(netzwerk_session, embedding_service_session)

    # Create mock signals that actually call connected callbacks
    class MockSignal:
        def __init__(self):
            self.callbacks = []

        def connect(self, callback):
            self.callbacks.append(callback)

        def emit(self, *args, **kwargs):
            for callback in self.callbacks:
                callback(*args, **kwargs)

    # Create mock signals object with callable signal attributes
    signals_mock = Mock()
    signals_mock.response_ready = MockSignal()
    signals_mock.proof_tree_updated = MockSignal()
    signals_mock.reasoning_step = MockSignal()
    signals_mock.finished = MockSignal()
    signals_mock.error = MockSignal()

    worker.signals = signals_mock

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
    Clears all data except production rules and base dictionary.
    """
    # Before test - clean graph
    with netzwerk_session.driver.session(database="neo4j") as session:
        # Clear all data except production rules and base dictionary
        session.run(
            """
            MATCH (n)
            WHERE NOT n:ProductionRule AND NOT n:BaseDictionary
            DETACH DELETE n
        """
        )

    yield

    # After test - clean graph again
    with netzwerk_session.driver.session(database="neo4j") as session:
        session.run(
            """
            MATCH (n)
            WHERE NOT n:ProductionRule AND NOT n:BaseDictionary
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
    Includes extended timeouts, comprehensive logging, and deeper reasoning.
    """
    from unittest.mock import Mock

    from kai_worker import KaiWorker

    worker = KaiWorker(netzwerk_session, embedding_service_session)

    # Create mock signals that actually call connected callbacks
    class MockSignal:
        def __init__(self):
            self.callbacks = []

        def connect(self, callback):
            self.callbacks.append(callback)

        def emit(self, *args, **kwargs):
            for callback in self.callbacks:
                callback(*args, **kwargs)

    # Create mock signals object with callable signal attributes
    signals_mock = Mock()
    signals_mock.response_ready = MockSignal()
    signals_mock.proof_tree_updated = MockSignal()
    signals_mock.reasoning_step = MockSignal()
    signals_mock.finished = MockSignal()
    signals_mock.error = MockSignal()

    worker.signals = signals_mock

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
