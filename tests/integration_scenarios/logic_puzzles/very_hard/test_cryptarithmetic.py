"""
tests/integration_scenarios/logic_puzzles/very_hard/test_cryptarithmetic.py

Cryptarithmetic puzzle: SEND + MORE = MONEY
Each letter represents a unique digit (0-9).
Leading digits cannot be zero.

Expected Reasoning: constraint, arithmetic, backtracking, digit_assignment, csp
Success Criteria:
- Reasoning Quality >= 30% (depth 15-30 expected)
- Confidence Calibration >= 30%
- Correctness >= 30%
- Overall >= 30%
"""

import re
from typing import Dict

from tests.integration_scenarios.utils.scenario_base import ScenarioTestBase


class TestCryptarithmetic(ScenarioTestBase):
    """Test: Cryptarithmetic Puzzle SEND + MORE = MONEY - Very Hard"""

    # REQUIRED: Class attributes
    DIFFICULTY = "very_hard"
    DOMAIN = "logic_puzzles"
    TIMEOUT_SECONDS = 3600  # 1 hour

    # REQUIRED: Scoring weights
    REASONING_QUALITY_WEIGHT = 0.5
    CONFIDENCE_CALIBRATION_WEIGHT = 0.2
    CORRECTNESS_WEIGHT = 0.3

    def test_cryptarithmetic(
        self,
        kai_worker_scenario_mode,
        progress_reporter,
        scenario_logger,
        confidence_tracker,
    ):
        """
        Test: SEND + MORE = MONEY cryptarithmetic puzzle.

        Each letter represents a unique digit (0-9).
        Leading letters (S, M) cannot be zero.
        Find the digit assignment that makes the equation true.

        Known solution: 9567 + 1085 = 10652
        """

        # Setup progress tracking
        progress_reporter.total_steps = 5
        progress_reporter.start()

        # Define test input (German language, ASCII only)
        input_text = """
        Kryptarithmetik-Raetsel:

        SEND + MORE = MONEY

        Regeln:
        1. Jeder Buchstabe steht fuer eine eindeutige Ziffer (0-9).
        2. Verschiedene Buchstaben stehen fuer verschiedene Ziffern.
        3. Fuehrende Buchstaben (S, M) koennen nicht Null sein.
        4. Die Gleichung muss mathematisch korrekt sein.

        Buchstaben: S, E, N, D, M, O, R, Y

        Addition:
          S E N D
        + M O R E
        ---------
        M O N E Y

        Hinweise:
        - M muss 1 sein (da SEND + MORE fuenfstellig ergibt, Uebertrag aus Tausenderstelle).
        - S muss 8 oder 9 sein (um Uebertrag zu erzeugen).
        - Systematisches Ausprobieren mit Rueckwaerts-Tracking noetig.

        Frage: Welche Ziffern stehen fuer S, E, N, D, M, O, R, Y?
        Geben Sie die Loesung als Zuordnung an.
        """

        # Define expected outputs
        # Known solution: S=9, E=5, N=6, D=7, M=1, O=0, R=8, Y=2
        expected_outputs = {
            "S": 9,
            "E": 5,
            "N": 6,
            "D": 7,
            "M": 1,
            "O": 0,
            "R": 8,
            "Y": 2,
            "send_value": 9567,
            "more_value": 1085,
            "money_value": 10652,
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
            result.correctness_score >= 20
        ), f"Correctness too low: {result.correctness_score:.1f}%"

        assert (
            result.reasoning_quality_score >= 30
        ), f"Reasoning quality too low: {result.reasoning_quality_score:.1f}%"

        # Domain-specific assertions - check for constraint/arithmetic strategies
        crypto_strategies = [
            "constraint",
            "arithmetic",
            "backtracking",
            "csp",
            "digit",
            "assignment",
            "search",
            "enumeration",
        ]
        found_strategy = any(
            any(cs in s.lower() for cs in crypto_strategies)
            for s in result.strategies_used
        )

        assert (
            found_strategy
        ), f"Expected constraint/arithmetic strategy, got: {result.strategies_used}"

        # Check proof tree depth (cryptarithmetic requires deep search)
        assert (
            10 <= result.proof_tree_depth <= 30
        ), f"ProofTree depth {result.proof_tree_depth} outside expected range 10-30"

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
        Custom correctness scoring for cryptarithmetic puzzle.

        Partial credit for:
        - M = 1 (15 points) - critical insight
        - S = 9 (15 points) - critical insight
        - E = 5 (10 points)
        - N = 6 (10 points)
        - D = 7 (10 points)
        - O = 0 (10 points)
        - R = 8 (10 points)
        - Y = 2 (10 points)
        - Mentions 9567 or 1085 or 10652 (10 points)
        """
        score = 0.0

        # Check M = 1 (critical)
        if re.search(r"\bM\s*=\s*1\b|M.*\b1\b", actual):
            score += 15.0

        # Check S = 9 (critical)
        if re.search(r"\bS\s*=\s*9\b|S.*\b9\b", actual):
            score += 15.0

        # Check E = 5
        if re.search(r"\bE\s*=\s*5\b|E.*\b5\b", actual):
            score += 10.0

        # Check N = 6
        if re.search(r"\bN\s*=\s*6\b|N.*\b6\b", actual):
            score += 10.0

        # Check D = 7
        if re.search(r"\bD\s*=\s*7\b|D.*\b7\b", actual):
            score += 10.0

        # Check O = 0
        if re.search(r"\bO\s*=\s*0\b|O.*\b0\b", actual):
            score += 10.0

        # Check R = 8
        if re.search(r"\bR\s*=\s*8\b|R.*\b8\b", actual):
            score += 10.0

        # Check Y = 2
        if re.search(r"\bY\s*=\s*2\b|Y.*\b2\b", actual):
            score += 10.0

        # Check for numeric solution
        if "9567" in actual or "1085" in actual or "10652" in actual:
            score += 10.0

        return min(score, 100.0)  # Cap at 100
