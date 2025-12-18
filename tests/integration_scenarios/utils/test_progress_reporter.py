"""
tests/integration_scenarios/utils/test_progress_reporter.py

Unit tests for progress_reporter.py module.
Tests ProgressReporter class for long-running test progress tracking.

NO Unicode characters - ASCII only (see CLAUDE.md encoding rules).
"""

import time

from tests.integration_scenarios.utils.progress_reporter import ProgressReporter


class TestProgressReporter:
    """Test ProgressReporter class"""

    def test_initialization(self):
        """Test reporter initialization"""
        reporter = ProgressReporter("test_scenario", 10)
        assert reporter.test_name == "test_scenario"
        assert reporter.total_steps == 10
        assert reporter.current_step == 0
        assert reporter.start_time is None
        assert reporter.step_logs == []

    def test_start(self, capsys):
        """Test starting progress tracking"""
        reporter = ProgressReporter("test_scenario", 5)
        reporter.start()

        assert reporter.start_time is not None
        assert reporter.current_step == 0

        # Check output
        captured = capsys.readouterr()
        assert "Starting test" in captured.out
        assert "test_scenario" in captured.out

    def test_update_auto_increment(self, capsys):
        """Test progress update with auto-increment"""
        reporter = ProgressReporter("test_scenario", 5)
        reporter.start()

        reporter.update("Step 1")
        assert reporter.current_step == 1

        reporter.update("Step 2")
        assert reporter.current_step == 2

        # Check output
        captured = capsys.readouterr()
        assert "Step 1" in captured.out
        assert "20.0%" in captured.out  # 1/5 = 20%

    def test_update_explicit_percentage(self, capsys):
        """Test progress update with explicit percentage"""
        reporter = ProgressReporter("test_scenario", 10)
        reporter.start()

        reporter.update("Custom step", percentage=50.0)
        assert reporter.current_step == 5  # 50% of 10

        # Check output
        captured = capsys.readouterr()
        assert "50.0%" in captured.out
        assert "Custom step" in captured.out

    def test_update_without_start_calls_start(self, capsys):
        """Test that update calls start if not started"""
        reporter = ProgressReporter("test_scenario", 5)
        reporter.update("First step")

        assert reporter.start_time is not None
        assert reporter.current_step == 1

    def test_log_substep(self, capsys):
        """Test substep logging"""
        reporter = ProgressReporter("test_scenario", 5)
        reporter.log_substep("Sub-action performed")

        captured = capsys.readouterr()
        assert "SUBSTEP" in captured.out
        assert "Sub-action performed" in captured.out

    def test_complete_success(self, capsys):
        """Test completion with success"""
        reporter = ProgressReporter("test_scenario", 5)
        reporter.start()
        reporter.update("Step 1")
        reporter.complete(success=True)

        captured = capsys.readouterr()
        assert "PASSED" in captured.out
        assert "test_scenario" in captured.out

    def test_complete_failure(self, capsys):
        """Test completion with failure"""
        reporter = ProgressReporter("test_scenario", 5)
        reporter.start()
        reporter.update("Step 1")
        reporter.complete(success=False)

        captured = capsys.readouterr()
        assert "FAILED" in captured.out

    def test_complete_without_start(self):
        """Test that complete handles missing start gracefully"""
        reporter = ProgressReporter("test_scenario", 5)
        reporter.complete(success=True)
        # Should not raise exception

    def test_get_eta_calculating(self):
        """Test ETA calculation before any steps"""
        reporter = ProgressReporter("test_scenario", 10)
        reporter.start()
        eta = reporter.get_eta()

        assert eta == "calculating..."

    def test_get_eta_complete(self):
        """Test ETA when all steps done"""
        reporter = ProgressReporter("test_scenario", 5)
        reporter.start()
        for i in range(5):
            reporter.update(f"Step {i+1}")

        eta = reporter.get_eta()
        assert eta == "complete"

    def test_get_eta_with_progress(self):
        """Test ETA calculation with some progress"""
        reporter = ProgressReporter("test_scenario", 10)
        reporter.start()
        time.sleep(0.1)  # Small delay
        reporter.update("Step 1")

        eta = reporter.get_eta()
        # Should return some time estimate (not "calculating..." or "complete")
        assert eta != "calculating..."
        assert eta != "complete"
        assert "s" in eta or "m" in eta or "h" in eta

    def test_format_duration_seconds(self):
        """Test duration formatting - seconds"""
        reporter = ProgressReporter("test", 1)
        formatted = reporter._format_duration(45.0)
        assert formatted == "45s"

    def test_format_duration_minutes(self):
        """Test duration formatting - minutes"""
        reporter = ProgressReporter("test", 1)
        formatted = reporter._format_duration(125.0)  # 2m 5s
        assert "2m" in formatted
        assert "5s" in formatted

    def test_format_duration_hours(self):
        """Test duration formatting - hours"""
        reporter = ProgressReporter("test", 1)
        formatted = reporter._format_duration(7325.0)  # 2h 2m
        assert "2h" in formatted
        assert "2m" in formatted

    def test_get_step_summary(self):
        """Test step summary generation"""
        reporter = ProgressReporter("test_scenario", 5)
        reporter.start()
        reporter.update("Step 1")
        reporter.update("Step 2")

        summary = reporter.get_step_summary()
        assert "Step Summary" in summary
        assert "Step 1" in summary
        assert "Step 2" in summary

    def test_get_step_summary_empty(self):
        """Test step summary with no steps"""
        reporter = ProgressReporter("test_scenario", 5)
        summary = reporter.get_step_summary()

        assert "No steps completed" in summary

    def test_get_detailed_timing(self):
        """Test detailed timing information"""
        reporter = ProgressReporter("test_scenario", 3)
        reporter.start()
        time.sleep(0.05)
        reporter.update("Step 1")
        time.sleep(0.05)
        reporter.update("Step 2")

        timing = reporter.get_detailed_timing()
        assert "Detailed Timing" in timing
        assert "Step 1" in timing
        assert "Step 2" in timing

    def test_get_detailed_timing_no_data(self):
        """Test detailed timing with no data"""
        reporter = ProgressReporter("test_scenario", 5)
        timing = reporter.get_detailed_timing()

        assert "No timing data" in timing

    def test_reset(self):
        """Test reporter reset"""
        reporter = ProgressReporter("test_scenario", 5)
        reporter.start()
        reporter.update("Step 1")
        reporter.update("Step 2")

        # Reset
        reporter.reset()

        assert reporter.current_step == 0
        assert reporter.start_time is None
        assert reporter.step_logs == []

    def test_multiple_updates_preserve_order(self):
        """Test that multiple updates preserve order"""
        reporter = ProgressReporter("test_scenario", 5)
        reporter.start()

        steps = ["First", "Second", "Third"]
        for step in steps:
            reporter.update(step)

        assert len(reporter.step_logs) == 3
        assert reporter.step_logs[0]["name"] == "First"
        assert reporter.step_logs[1]["name"] == "Second"
        assert reporter.step_logs[2]["name"] == "Third"

    def test_step_logs_contain_metadata(self):
        """Test that step logs contain all necessary metadata"""
        reporter = ProgressReporter("test_scenario", 5)
        reporter.start()
        reporter.update("Test step")

        log = reporter.step_logs[0]
        assert "step" in log
        assert "name" in log
        assert "time" in log
        assert "percentage" in log

        assert log["name"] == "Test step"
        assert log["step"] == 1

    def test_progress_tracking_workflow(self, capsys):
        """Test complete progress tracking workflow"""
        reporter = ProgressReporter("Integration Test", 5)

        # Start
        reporter.start()

        # Update steps
        reporter.update("Initializing", percentage=0)
        reporter.log_substep("Loading dependencies")
        reporter.update("Processing", percentage=40)
        reporter.log_substep("Running main logic")
        reporter.update("Analyzing", percentage=70)
        reporter.update("Finalizing", percentage=90)
        reporter.update("Complete", percentage=100)

        # Complete
        reporter.complete(success=True)

        # Check all steps logged
        assert len(reporter.step_logs) == 5

        # Check output contains expected elements
        captured = capsys.readouterr()
        assert "Starting test" in captured.out
        assert "Integration Test" in captured.out
        assert "SUBSTEP" in captured.out
        assert "PASSED" in captured.out
