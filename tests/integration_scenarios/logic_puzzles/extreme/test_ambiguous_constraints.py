"""
tests/integration_scenarios/logic_puzzles/extreme/test_ambiguous_constraints.py

Extreme Logic Puzzle: Ambiguous Constraints Allowing Multiple Interpretations

Scenario: Logic puzzle with intentionally ambiguous phrasing that allows
multiple valid interpretations. Tests KAI's ability to recognize ambiguity,
explore multiple solution spaces, and present alternatives.

Expected Reasoning:
- Ambiguity detection in constraints
- Multiple interpretation exploration
- Solution space analysis
- Presentation of alternative solutions

Success Criteria:
- Recognizes ambiguity (weight: 40%)
- Provides multiple valid solutions (weight: 40%)
- Reasoning quality >= 20% (extreme threshold)
- Overall score >= 20%

NO Unicode characters - ASCII only (see CLAUDE.md encoding rules).
"""

import re
from typing import Dict

from tests.integration_scenarios.utils.scenario_base import ScenarioTestBase


class TestAmbiguousConstraints(ScenarioTestBase):
    """Test: Logic puzzle with ambiguous constraints and multiple interpretations"""

    # REQUIRED: Class attributes
    DIFFICULTY = "extreme"
    DOMAIN = "logic_puzzles"
    TIMEOUT_SECONDS = 7200  # 2 hours

    # REQUIRED: Scoring weights
    REASONING_QUALITY_WEIGHT = 0.5
    CONFIDENCE_CALIBRATION_WEIGHT = 0.2
    CORRECTNESS_WEIGHT = 0.3

    def test_ambiguous_constraints(
        self,
        kai_worker_scenario_mode,
        progress_reporter,
        scenario_logger,
        confidence_tracker,
    ):
        """
        Test: Puzzle with ambiguous phrasing allowing multiple interpretations.

        Ambiguity: "Anna und Bob moegen unterschiedliche Farben" could mean:
        - Anna's favorite != Bob's favorite (interpretation 1)
        - Anna likes colors Bob doesn't (interpretation 2)

        This tests KAI's ability to:
        1. Detect ambiguous phrasing
        2. Explore multiple interpretations
        3. Provide alternative solutions
        4. Explain the ambiguity
        """

        # Setup progress tracking
        progress_reporter.total_steps = 5
        progress_reporter.start()

        # Define test input (German language, ASCII only)
        input_text = """
        Es gibt drei Personen: Anna, Bob, Clara.
        Es gibt drei Farben: Rot, Blau, Gruen.

        Hinweise:
        1. Anna und Bob moegen unterschiedliche Farben.
        2. Clara mag keine blaue Farbe.
        3. Mindestens eine Person mag Rot.
        4. Nicht mehr als eine Person mag dieselbe Farbe.

        Frage: Welche Farbe mag jede Person?

        Hinweis: Der Ausdruck "moegen unterschiedliche Farben" kann
        mehrdeutig interpretiert werden.
        """

        # Define expected outputs
        expected_outputs = {
            "ambiguity_detected": True,
            "interpretation_count": 2,
            "multiple_solutions": True,
            "solution_sets": [
                # Interpretation 1: Anna != Bob favorite
                {"Anna": "Rot", "Bob": "Blau", "Clara": "Gruen"},
                {"Anna": "Rot", "Bob": "Gruen", "Clara": "Blau"},
                # More combinations possible
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
        # Check for ambiguity-related strategies
        ambiguity_keywords = [
            "ambig",
            "mehrdeutig",
            "interpretation",
            "alternativ",
            "mehrere",
        ]
        response_has_ambiguity = any(
            kw in result.kai_response.lower() for kw in ambiguity_keywords
        )

        if not response_has_ambiguity:
            print("[WARNING] Response may not explicitly mention ambiguity")

        # ProofTree depth should reflect exploration
        assert (
            result.proof_tree_depth >= 4
        ), f"ProofTree depth {result.proof_tree_depth} too shallow for ambiguity exploration"

        # Performance assertion
        assert (
            result.execution_time_ms < 7200000
        ), f"Exceeded timeout: {result.execution_time_ms}ms"

        # Check for multiple solutions mentioned
        response_lower = result.kai_response.lower()
        multi_solution_indicators = [
            "mehrere loesungen" in response_lower,
            "alternative" in response_lower,
            "beide interpretationen" in response_lower,
            "entweder" in response_lower and "oder" in response_lower,
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
        Custom correctness scoring for ambiguous puzzle.

        Full credit requires:
        - Recognition of ambiguity (40%)
        - Mention of multiple interpretations (30%)
        - At least one valid solution (30%)
        """
        if not actual or not expected:
            return 0.0

        actual_lower = actual.lower()
        score = 0.0

        # Check ambiguity recognition (40 points)
        ambiguity_keywords = [
            "ambig",
            "mehrdeutig",
            "unklar",
            "interpretation",
        ]
        if any(kw in actual_lower for kw in ambiguity_keywords):
            score += 40.0

        # Check multiple interpretations mentioned (30 points)
        multi_keywords = [
            "mehrere",
            "alternative",
            "verschiedene deutungen",
            "zwei interpretationen",
        ]
        if any(kw in actual_lower for kw in multi_keywords):
            score += 30.0

        # Check for at least one valid solution (30 points)
        # Valid assignment: each person gets a color
        people = ["anna", "bob", "clara"]
        colors = ["rot", "blau", "gruen"]

        assignments_found = 0
        for person in people:
            for color in colors:
                pattern = rf"\b{person}\b.*\b{color}\b"
                if re.search(pattern, actual_lower):
                    assignments_found += 1
                    break  # Count each person once

        if assignments_found >= 3:
            score += 30.0
        elif assignments_found >= 2:
            score += 20.0  # Partial credit
        elif assignments_found >= 1:
            score += 10.0  # Minimal credit

        return min(100.0, score)
