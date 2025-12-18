"""
tests/integration_scenarios/combined_scenarios/very_hard/test_temporal_reasoning_dialogue.py

Multi-turn dialogue with temporal reasoning (birth years, age calculations).
Requires episodic memory, arithmetic reasoning, and temporal inference.

Expected Reasoning: temporal, arithmetic, episodic_memory, comparison, inference
Success Criteria:
- Reasoning Quality >= 30% (multi-strategy coordination)
- Confidence Calibration >= 30%
- Correctness >= 30%
- Overall >= 30%
"""

import re
from typing import Dict

from tests.integration_scenarios.utils.scenario_base import ScenarioTestBase


class TestTemporalReasoningDialogue(ScenarioTestBase):
    """Test: Temporal Reasoning Dialogue - Very Hard Combined Scenario"""

    # REQUIRED: Class attributes
    DIFFICULTY = "very_hard"
    DOMAIN = "combined_scenarios"
    TIMEOUT_SECONDS = 3600  # 1 hour

    # REQUIRED: Scoring weights
    REASONING_QUALITY_WEIGHT = 0.5
    CONFIDENCE_CALIBRATION_WEIGHT = 0.2
    CORRECTNESS_WEIGHT = 0.3

    def test_temporal_reasoning_dialogue(
        self,
        kai_worker_scenario_mode,
        progress_reporter,
        scenario_logger,
        confidence_tracker,
    ):
        """
        Test: Multi-turn dialogue with temporal reasoning.

        Scenario:
        - Learn birth years for multiple people
        - Calculate ages and age differences
        - Answer temporal queries (who is older, when event happened)
        - Requires episodic memory + arithmetic + temporal reasoning
        """

        # Setup progress tracking
        progress_reporter.total_steps = 5
        progress_reporter.start()

        # Define test input (German language, ASCII only)
        input_text = """
        Mehrere Fakten ueber Personen und Zeitpunkte:

        1. Anna wurde 1985 geboren.
        2. Bernd wurde 1990 geboren.
        3. Clara wurde 1982 geboren.
        4. Daniel ist 3 Jahre juenger als Bernd.
        5. Elena ist doppelt so alt wie der Altersunterschied zwischen Anna und Clara.

        Aktuelles Jahr: 2024

        Fragen:
        1. Wer ist die aelteste Person?
        2. Wie alt ist Daniel im Jahr 2024?
        3. Wie alt ist Elena im Jahr 2024?
        4. Wer war im Jahr 2000 bereits volljaehrig (18 Jahre)?
        5. In welchem Jahr wird Bernd 40 Jahre alt?

        Beantworte alle fuenf Fragen mit Begruendung.
        Zeige die Berechnungsschritte.
        """

        # Define expected outputs
        # Calculations:
        # - Anna: 2024 - 1985 = 39 Jahre
        # - Bernd: 2024 - 1990 = 34 Jahre
        # - Clara: 2024 - 1982 = 42 Jahre (oldest)
        # - Daniel: 3 Jahre juenger als Bernd -> 34 - 3 = 31 Jahre
        # - Altersunterschied Anna-Clara: |39 - 42| = 3 Jahre
        # - Elena: 2 * 3 = 6 Jahre
        # - Im Jahr 2000: Anna=15, Bernd=10, Clara=18 (only Clara)
        # - Bernd 40 Jahre: 1990 + 40 = 2030
        expected_outputs = {
            "oldest": "Clara",
            "daniel_age": 31,
            "elena_age": 6,
            "adult_in_2000": "Clara",
            "bernd_40_year": 2030,
        }

        # Execute using BASE CLASS method
        result = self.run_scenario(
            input_text=input_text,
            expected_outputs=expected_outputs,
            context={},
            kai_worker=kai_worker_scenario_mode,
            logger=scenario_logger,
            progress_reporter=progress_reporter,
            confidence_tracker=confidence_tracker,
        )

        # Assertions on ScenarioResult object
        assert (
            result.overall_score >= 30
        ), f"Overall score too low: {result.overall_score:.1f}% (expected >= 30%)"

        assert (
            result.correctness_score >= 25
        ), f"Correctness too low: {result.correctness_score:.1f}%"

        assert (
            result.reasoning_quality_score >= 30
        ), f"Reasoning quality too low: {result.reasoning_quality_score:.1f}%"

        # Domain-specific assertions - check for multiple strategies
        temporal_strategies = [
            "temporal",
            "arithmetic",
            "episodic",
            "memory",
            "comparison",
            "inference",
            "calculation",
        ]
        found_strategy = any(
            any(ts in s.lower() for ts in temporal_strategies)
            for s in result.strategies_used
        )

        assert (
            found_strategy
        ), f"Expected temporal/arithmetic strategy, got: {result.strategies_used}"

        # Check multiple strategies used (combined scenario)
        assert (
            len(result.strategies_used) >= 2
        ), f"Expected multiple strategies, got: {result.strategies_used}"

        # Check proof tree depth
        assert (
            6 <= result.proof_tree_depth <= 25
        ), f"ProofTree depth {result.proof_tree_depth} outside expected range 6-25"

        # Performance assertion
        assert (
            result.execution_time_ms < 3600000
        ), f"Too slow: {result.execution_time_ms}ms (expected <1 hour)"

        # Logging
        log_file = scenario_logger.save_logs()
        print(f"\n[INFO] Detailed logs saved to: {log_file}")
        print(f"[INFO] Overall Score: {result.overall_score:.1f}/100")
        print(
            f"[INFO] Breakdown: Reasoning={result.reasoning_quality_score:.1f}, "
            f"Confidence={result.confidence_calibration_score:.1f}, "
            f"Correctness={result.correctness_score:.1f}"
        )

        if result.overall_score < 50:
            print("[WEAKNESS] Identified issues:")
            for weakness in result.identified_weaknesses:
                print(f"  - {weakness}")

        if result.improvement_suggestions:
            print("[SUGGESTION] Improvements:")
            for suggestion in result.improvement_suggestions:
                print(f"  - {suggestion}")

        # Final check
        assert result.passed, f"Test failed: {result.error or 'Score below threshold'}"

    def score_correctness(
        self, actual: str, expected: Dict, allow_partial: bool = True
    ) -> float:
        """
        Custom correctness scoring for temporal reasoning dialogue.

        Partial credit for:
        - Oldest = Clara (20 points)
        - Daniel age = 31 (20 points)
        - Elena age = 6 (20 points)
        - Adult in 2000 = Clara (20 points)
        - Bernd 40 in year 2030 (20 points)
        """
        score = 0.0

        # Check oldest person
        oldest = expected.get("oldest", "")
        if oldest:
            pattern = rf"\b{oldest}\b.*\baelt|aelt.*\b{oldest}\b"
            if re.search(pattern, actual, re.IGNORECASE):
                score += 20.0

        # Check Daniel's age (31)
        daniel_age = expected.get("daniel_age", 0)
        if daniel_age and re.search(
            rf"\bDaniel\b.*\b{daniel_age}\b|\b{daniel_age}\b.*\bDaniel\b",
            actual,
        ):
            score += 20.0

        # Check Elena's age (6)
        elena_age = expected.get("elena_age", 0)
        if elena_age and re.search(
            rf"\bElena\b.*\b{elena_age}\b|\b{elena_age}\b.*\bElena\b",
            actual,
        ):
            score += 20.0

        # Check adult in 2000 (Clara)
        adult_2000 = expected.get("adult_in_2000", "")
        if adult_2000 and re.search(
            rf"\b{adult_2000}\b.*\b2000\b|\b2000\b.*\b{adult_2000}\b",
            actual,
        ):
            score += 20.0

        # Check Bernd 40 in 2030
        bernd_40_year = expected.get("bernd_40_year", 0)
        if bernd_40_year and re.search(
            rf"\bBernd\b.*\b{bernd_40_year}\b|\b{bernd_40_year}\b.*\bBernd\b",
            actual,
        ):
            score += 20.0

        return score
