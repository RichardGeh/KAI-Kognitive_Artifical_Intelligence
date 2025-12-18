"""
tests/integration_scenarios/utils/progress_reporter.py

Provide visible progress updates for long-running tests.
Reports test progress with ETA and status updates.

NO Unicode characters - ASCII only (see CLAUDE.md encoding rules).
"""

import time
from typing import Optional


class ProgressReporter:
    """Reports test progress with ETA and status updates"""

    def __init__(self, test_name: str, total_steps: int):
        """
        Initialize progress reporter.

        Args:
            test_name: Name of the test being run
            total_steps: Total number of steps expected
        """
        self.test_name = test_name
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time: Optional[float] = None
        self.step_logs = []

    def start(self):
        """Start progress tracking"""
        self.start_time = time.time()
        self.current_step = 0
        self.step_logs = []
        print(f"\n[PROGRESS] Starting test: {self.test_name}")
        print(f"[PROGRESS] Total steps: {self.total_steps}")

    def update(self, step_name: str, percentage: Optional[float] = None):
        """
        Update progress with current step.

        Prints:
        [PROGRESS] Test: logic_puzzle_extreme | Step 3/10 (30%) | ETA: 15m 23s | Current: SAT solving...

        Args:
            step_name: Description of current step
            percentage: Optional explicit percentage (0-100), otherwise auto-calculated
        """
        if self.start_time is None:
            self.start()

        # Increment step counter if using automatic calculation
        if percentage is None:
            self.current_step += 1
            percentage = (self.current_step / self.total_steps) * 100.0
        else:
            # Use provided percentage to estimate step
            self.current_step = int((percentage / 100.0) * self.total_steps)

        # Log this step
        current_time = time.time()
        self.step_logs.append(
            {
                "step": self.current_step,
                "name": step_name,
                "time": current_time,
                "percentage": percentage,
            }
        )

        # Calculate ETA
        eta_str = self.get_eta()

        # Print progress update
        print(
            f"[PROGRESS] Test: {self.test_name} | "
            f"Step {self.current_step}/{self.total_steps} ({percentage:.1f}%) | "
            f"ETA: {eta_str} | "
            f"Current: {step_name}"
        )

    def log_substep(self, message: str):
        """
        Log a substep without updating overall progress.

        Args:
            message: Substep message
        """
        print(f"[SUBSTEP]   -> {message}")

    def complete(self, success: bool):
        """
        Mark test as complete.

        Args:
            success: Whether the test succeeded
        """
        if self.start_time is None:
            return

        total_time = time.time() - self.start_time
        time_str = self._format_duration(total_time)

        status = "PASSED" if success else "FAILED"
        print(f"\n[PROGRESS] Test {status}: {self.test_name}")
        print(f"[PROGRESS] Total time: {time_str}")
        print(f"[PROGRESS] Steps completed: {self.current_step}/{self.total_steps}\n")

    def get_eta(self) -> str:
        """
        Calculate and format estimated time remaining.

        Returns:
            Formatted ETA string (e.g., "15m 23s" or "2h 15m")
        """
        if self.start_time is None or self.current_step == 0:
            return "calculating..."

        elapsed = time.time() - self.start_time
        steps_done = self.current_step
        steps_remaining = self.total_steps - steps_done

        if steps_remaining <= 0:
            return "complete"

        # Calculate average time per step
        avg_time_per_step = elapsed / steps_done

        # Estimate remaining time
        eta_seconds = avg_time_per_step * steps_remaining

        return self._format_duration(eta_seconds)

    def _format_duration(self, seconds: float) -> str:
        """
        Format duration in human-readable format.

        Args:
            seconds: Duration in seconds

        Returns:
            Formatted string (e.g., "15m 23s", "2h 15m", "45s")
        """
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            return f"{hours}h {minutes}m"

    def get_step_summary(self) -> str:
        """
        Get summary of all completed steps.

        Returns:
            Multi-line string with step summary
        """
        if not self.step_logs:
            return "No steps completed yet"

        lines = ["Step Summary:"]
        for log in self.step_logs:
            step_num = log["step"]
            step_name = log["name"]
            percentage = log["percentage"]
            lines.append(f"  {step_num}. {step_name} ({percentage:.1f}%)")

        return "\n".join(lines)

    def get_detailed_timing(self) -> str:
        """
        Get detailed timing information for each step.

        Returns:
            Multi-line string with timing details
        """
        if not self.step_logs or self.start_time is None:
            return "No timing data available"

        lines = ["Detailed Timing:"]
        prev_time = self.start_time

        for log in self.step_logs:
            step_num = log["step"]
            step_name = log["name"]
            step_time = log["time"]
            duration = step_time - prev_time
            duration_str = self._format_duration(duration)

            lines.append(f"  {step_num}. {step_name}: {duration_str}")
            prev_time = step_time

        return "\n".join(lines)

    def reset(self):
        """Reset progress reporter for reuse"""
        self.current_step = 0
        self.start_time = None
        self.step_logs = []
