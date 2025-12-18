"""
tests/integration_scenarios/logic_puzzles/extreme/test_open_ended_puzzle.py

Extreme Logic Puzzle: Open-Ended Underspecified Problem

Scenario: Logic puzzle with insufficient constraints, resulting in multiple
valid solutions. Tests KAI's ability to recognize underspecification, explore
the solution space, and present any valid solution while acknowledging
multiplicity.

Expected Reasoning:
- Constraint satisfaction
- Solution space exploration
- Underspecification detection
- Multiple solution awareness

Success Criteria:
- Finds at least one valid solution (weight: 40%)
- Recognizes multiple solutions exist (weight: 40%)
- Reasoning quality >= 20% (extreme threshold)
- Overall score >= 20%

NO Unicode characters - ASCII only (see CLAUDE.md encoding rules).
"""

import re
from typing import Dict

from tests.integration_scenarios.utils.scenario_base import ScenarioTestBase


class TestOpenEndedPuzzle(ScenarioTestBase):
    """Test: Open-ended puzzle with multiple valid solutions"""

    # REQUIRED: Class attributes
    DIFFICULTY = "extreme"
    DOMAIN = "logic_puzzles"
    TIMEOUT_SECONDS = 7200  # 2 hours

    # REQUIRED: Scoring weights
    REASONING_QUALITY_WEIGHT = 0.5
    CONFIDENCE_CALIBRATION_WEIGHT = 0.2
    CORRECTNESS_WEIGHT = 0.3

    def test_open_ended_puzzle(
        self,
        kai_worker_scenario_mode,
        progress_reporter,
        scenario_logger,
        confidence_tracker,
    ):
        """
        Test: Underspecified puzzle with multiple valid solutions.

        Valid solutions include:
        - Anna=Rot, Bob=Blau, Clara=Gruen
        - Anna=Rot, Bob=Gruen, Clara=Blau
        - Anna=Blau, Bob=Rot, Clara=Gruen
        - Anna=Gruen, Bob=Rot, Clara=Blau
        - Several more combinations

        This tests KAI's ability to:
        1. Find ANY valid solution
        2. Recognize underspecification
        3. Acknowledge multiple solutions exist
        4. Not over-commit to single answer
        """

        # Setup progress tracking
        progress_reporter.total_steps = 5
        progress_reporter.start()

        # Define test input (German language, ASCII only)
        input_text = """
        Es gibt drei Personen: Anna, Bob, Clara.
        Es gibt drei Farben: Rot, Blau, Gruen.
        Jede Person mag eine Farbe.

        Hinweise:
        1. Anna mag nicht Gruen.
        2. Bob mag keine Farbe, die mit R beginnt.
        3. Keine zwei Personen moegen dieselbe Farbe.

        Frage: Welche Farbe mag jede Person?

        Hinweis: Diese Aufgabe hat moeglicherweise mehrere Loesungen.
        """

        # Define expected outputs - ANY valid solution is acceptable
        expected_outputs = {
            "multi_solution": True,
            "any_valid_solution": True,
            "valid_solutions": [
                {"Anna": "Rot", "Bob": "Blau", "Clara": "Gruen"},
                {"Anna": "Rot", "Bob": "Gruen", "Clara": "Blau"},
                {"Anna": "Blau", "Bob": "Gruen", "Clara": "Rot"},
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
        # Check for constraint strategies
        strategy_keywords = ["constraint", "sat", "csp", "elimination"]
        assert any(
            any(kw in s.lower() for kw in strategy_keywords)
            for s in result.strategies_used
        ), f"Expected constraint strategies, got: {result.strategies_used}"

        # ProofTree should show solution exploration
        assert (
            result.proof_tree_depth >= 3
        ), f"ProofTree depth {result.proof_tree_depth} too shallow for solution search"

        # Performance assertion
        assert (
            result.execution_time_ms < 7200000
        ), f"Exceeded timeout: {result.execution_time_ms}ms"

        # Check for multiple solution awareness
        response_lower = result.kai_response.lower()
        multi_solution_indicators = [
            "mehrere loesungen" in response_lower,
            "verschiedene moeglichkeiten" in response_lower,
            "nicht eindeutig" in response_lower,
            "eine moegliche loesung" in response_lower,
        ]

        if any(multi_solution_indicators):
            print("[SUCCESS] KAI recognized multiple solution possibility")

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
        Custom correctness scoring for open-ended puzzle.

        Full credit requires:
        - Valid solution found (40%)
        - Recognition of multiple solutions (40%)
        - Constraint satisfaction verified (20%)
        """
        if not actual or not expected:
            return 0.0

        actual_lower = actual.lower()
        score = 0.0

        # Check for any valid solution (40 points)
        # Valid: Anna!=Gruen, Bob!=Rot, all different
        valid_solutions = expected.get("valid_solutions", [])

        solution_found = False
        for solution in valid_solutions:
            # Check if this solution is mentioned
            all_match = True
            for person, color in solution.items():
                pattern = rf"\b{person.lower()}\b.*\b{color.lower()}\b"
                if not re.search(pattern, actual_lower):
                    all_match = False
                    break

            if all_match:
                solution_found = True
                break

        if solution_found:
            score += 40.0
        else:
            # Partial credit for partial solution
            people = ["anna", "bob", "clara"]
            colors = ["rot", "blau", "gruen"]

            assignments = 0
            for person in people:
                for color in colors:
                    pattern = rf"\b{person}\b.*\b{color}\b"
                    if re.search(pattern, actual_lower):
                        assignments += 1
                        break

            if assignments >= 2:
                score += 20.0
            elif assignments >= 1:
                score += 10.0

        # Check recognition of multiple solutions (40 points)
        multi_keywords = [
            "mehrere",
            "verschiedene moeglichkeiten",
            "nicht eindeutig",
            "eine moegliche",
            "alternative",
        ]
        if any(kw in actual_lower for kw in multi_keywords):
            score += 40.0

        # Check constraint satisfaction mentioned (20 points)
        constraint_indicators = [
            "anna" in actual_lower and "nicht gruen" in actual_lower,
            "bob" in actual_lower and "nicht rot" in actual_lower,
            "unterschiedlich" in actual_lower,
        ]
        if any(constraint_indicators):
            score += 20.0

        return min(100.0, score)
