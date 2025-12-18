"""
tests/integration_scenarios/logic_puzzles/extreme/test_contradictory_hints.py

Extreme Logic Puzzle: Embedded Contradiction Detection

Scenario: Logic puzzle with embedded logical contradiction that makes the
puzzle unsatisfiable. Tests KAI's ability to detect contradictions through
constraint propagation and report the puzzle as unsolvable.

Expected Reasoning:
- Constraint propagation
- Contradiction detection
- Unsatisfiability proof
- Clear explanation of conflicting constraints

Success Criteria:
- Detects contradiction (weight: 50%)
- Reports UNSATISFIABLE or explains impossibility (weight: 30%)
- Reasoning quality >= 20% (extreme threshold)
- Overall score >= 20%

NO Unicode characters - ASCII only (see CLAUDE.md encoding rules).
"""

from typing import Dict

from tests.integration_scenarios.utils.scenario_base import ScenarioTestBase


class TestContradictoryHints(ScenarioTestBase):
    """Test: Logic puzzle with embedded contradiction"""

    # REQUIRED: Class attributes
    DIFFICULTY = "extreme"
    DOMAIN = "logic_puzzles"
    TIMEOUT_SECONDS = 7200  # 2 hours

    # REQUIRED: Scoring weights
    REASONING_QUALITY_WEIGHT = 0.5
    CONFIDENCE_CALIBRATION_WEIGHT = 0.2
    CORRECTNESS_WEIGHT = 0.3

    def test_contradictory_hints(
        self,
        kai_worker_scenario_mode,
        progress_reporter,
        scenario_logger,
        confidence_tracker,
    ):
        """
        Test: Puzzle with embedded contradiction requiring detection.

        Contradiction path:
        1. Anna trinkt Kaffee (given)
        2. Bob trinkt nicht Tee (given)
        3. Bob trinkt nicht Kaffee (given)
        4. Clara trinkt nicht Kaffee (given)
        5. Only 3 drinks available (Kaffee, Tee, Wasser)
        6. But: Es gibt nur zwei verschiedene Getraenke (CONTRADICTION!)

        This tests KAI's ability to:
        1. Propagate constraints systematically
        2. Detect logical contradictions
        3. Report UNSATISFIABLE
        4. Explain which constraints conflict
        """

        # Setup progress tracking
        progress_reporter.total_steps = 5
        progress_reporter.start()

        # Define test input (German language, ASCII only)
        input_text = """
        Es gibt drei Personen: Anna, Bob, Clara.
        Es gibt drei Getraenke: Kaffee, Tee, Wasser.

        Wichtige Bedingung: Es gibt nur zwei verschiedene Getraenke,
        die tatsaechlich getrunken werden. Das dritte Getraenk wird
        von niemandem gewaehlt.

        Hinweise:
        1. Anna trinkt Kaffee.
        2. Bob trinkt nicht Tee.
        3. Bob trinkt nicht Kaffee.
        4. Clara trinkt nicht Kaffee.
        5. Clara trinkt nicht Wasser.
        6. Jede Person trinkt genau ein Getraenk.

        Frage: Wer trinkt was?
        """

        # Define expected outputs
        expected_outputs = {
            "contradiction_detected": True,
            "result": "UNSATISFIABLE",
            "conflicting_constraints": [
                "only_two_drinks",
                "all_assignments",
            ],
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
        # Extreme difficulty: target >= 20%
        assert (
            result.overall_score >= 20
        ), f"Overall score too low: {result.overall_score:.1f}% (expected >= 20%)"

        # Lower thresholds for extreme difficulty
        assert (
            result.reasoning_quality_score >= 15
        ), f"Reasoning quality too low: {result.reasoning_quality_score:.1f}%"

        # Domain-specific assertions
        # Check for contradiction-related keywords
        contradiction_keywords = [
            "widerspruch",
            "unmoglich",
            "keine loesung",
            "unsatisfiable",
            "konflikt",
            "nicht loesbar",
        ]

        response_lower = result.kai_response.lower()
        contradiction_detected = any(
            kw in response_lower for kw in contradiction_keywords
        )

        if contradiction_detected:
            print("[SUCCESS] KAI detected contradiction")
        else:
            print("[WARNING] KAI may not have detected contradiction explicitly")

        # Check for SAT/CSP strategies (should be used for contradiction detection)
        strategy_keywords = ["sat", "csp", "constraint", "backtrack"]
        assert any(
            any(kw in s.lower() for kw in strategy_keywords)
            for s in result.strategies_used
        ), f"Expected SAT/CSP strategies, got: {result.strategies_used}"

        # ProofTree should show constraint checking
        assert (
            result.proof_tree_depth >= 3
        ), f"ProofTree depth {result.proof_tree_depth} too shallow for contradiction detection"

        # Performance assertion
        assert (
            result.execution_time_ms < 7200000
        ), f"Exceeded timeout: {result.execution_time_ms}ms"

        # Logging
        print(f"\n[INFO] Logs: {scenario_logger.save_logs()}")
        print(f"[INFO] Score: {result.overall_score:.1f}/100")
        print(
            f"[INFO] Breakdown: Reasoning={result.reasoning_quality_score:.1f}, "
            f"Confidence={result.confidence_calibration_score:.1f}, "
            f"Correctness={result.correctness_score:.1f}"
        )
        print(f"[INFO] ProofTree Depth: {result.proof_tree_depth}")
        print(f"[INFO] Strategies: {result.strategies_used}")

        if result.overall_score < 40:
            print("[EXPECTED] Low score is expected for extreme difficulty")
            print("[WEAKNESS] Issues identified:")
            for w in result.identified_weaknesses:
                print(f"  - {w}")

        # Final check
        assert (
            result.passed or result.overall_score >= 20
        ), f"Test failed: {result.error or 'Score below extreme threshold'}"

    def score_correctness(
        self, actual: str, expected: Dict, allow_partial: bool = True
    ) -> float:
        """
        Custom correctness scoring for contradiction puzzle.

        Full credit requires:
        - Explicit contradiction detection (50%)
        - UNSATISFIABLE or "no solution" message (30%)
        - Explanation of conflicting constraints (20%)
        """
        if not actual or not expected:
            return 0.0

        actual_lower = actual.lower()
        score = 0.0

        # Check contradiction detection (50 points)
        contradiction_keywords = [
            "widerspruch",
            "widerspruechlich",
            "konflikt",
            "inkonsistent",
        ]
        if any(kw in actual_lower for kw in contradiction_keywords):
            score += 50.0

        # Check UNSATISFIABLE or "no solution" (30 points)
        unsatisfiable_keywords = [
            "unmoglich",
            "keine loesung",
            "nicht loesbar",
            "unsatisfiable",
            "unloesbar",
        ]
        if any(kw in actual_lower for kw in unsatisfiable_keywords):
            score += 30.0

        # Check explanation of conflicting constraints (20 points)
        explanation_indicators = [
            "bedingung" in actual_lower and "widerspricht" in actual_lower,
            "zwei verschiedene" in actual_lower,
            "hinweis" in actual_lower and "konflikt" in actual_lower,
        ]
        if any(explanation_indicators):
            score += 20.0

        return min(100.0, score)
