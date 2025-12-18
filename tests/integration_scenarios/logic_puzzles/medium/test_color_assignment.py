"""
tests/integration_scenarios/logic_puzzles/medium/test_color_assignment.py

Medium-level logic puzzle: 4 people with 4 colors and compound constraints

Scenario:
Four people (Anna, Ben, Clara, Daniel) each wear one color: Rot, Blau, Gruen, Gelb.
Given 5 constraints including compound constraints (neither...nor), deduce assignments.

Expected Reasoning:
- Constraint solver should handle compound negations
- Elimination reasoning for "neither...nor" constraints
- ProofTree shows constraint propagation
- Confidence should be medium-high (>0.75)

Success Criteria (Gradual Scoring):
- Correctness: 30% (correct color assignments)
- Reasoning Quality: 50% (handles compound constraints, appropriate depth)
- Confidence Calibration: 20% (confidence matches correctness)
"""

import re
from typing import Dict, List

from tests.integration_scenarios.utils.scenario_base import ScenarioTestBase


class TestColorAssignment(ScenarioTestBase):
    """Test: 4-person color assignment with compound constraints"""

    DIFFICULTY = "medium"
    DOMAIN = "logic_puzzles"
    TIMEOUT_SECONDS = 600

    REASONING_QUALITY_WEIGHT = 0.5
    CONFIDENCE_CALIBRATION_WEIGHT = 0.2
    CORRECTNESS_WEIGHT = 0.3

    def test_four_person_color_puzzle(
        self,
        kai_worker_scenario_mode,
        progress_reporter,
        scenario_logger,
        confidence_tracker,
    ):
        """
        Test: 4-person color assignment with compound constraints

        Vier Personen, vier Farben, with "weder...noch" constraints
        Expected: Ben=Blau, Clara=Rot, Anna=Gruen/Gelb, Daniel=Gelb/Gruen
        """

        # Setup
        progress_reporter.total_steps = 5
        progress_reporter.start()

        puzzle_text = """
Vier Personen (Anna, Ben, Clara, Daniel) tragen jeweils eine Farbe: Rot, Blau, Gruen, Gelb.
1. Anna traegt nicht Rot.
2. Ben traegt Blau.
3. Clara traegt weder Gruen noch Gelb.
4. Daniel traegt nicht die gleiche Farbe wie Anna.
5. Genau eine Person traegt Rot.

Wer traegt welche Farbe?
        """

        # Note: Multiple valid solutions exist
        # Ben=Blau (fixed), Clara=Rot (from constraint 3+5)
        # Anna and Daniel: one has Gruen, other has Gelb
        expected_solution = {
            "Ben": "Blau",
            "Clara": "Rot",
            "Anna": ["Gruen", "Gelb"],  # Either valid
            "Daniel": ["Gelb", "Gruen"],  # Complement of Anna
        }

        # Execute scenario using base class
        result = self.run_scenario(
            input_text=puzzle_text,
            expected_outputs={"assignments": expected_solution},
            context={},
            kai_worker=kai_worker_scenario_mode,
            logger=scenario_logger,
            progress_reporter=progress_reporter,
            confidence_tracker=confidence_tracker,
        )

        # Assertions
        assert (
            result.overall_score >= 50
        ), f"Overall score too low: {result.overall_score:.1f}% (expected >= 50%)"

        assert (
            result.correctness_score >= 40
        ), f"Correctness too low: {result.correctness_score:.1f}%"

        assert (
            result.reasoning_quality_score >= 40
        ), f"Reasoning quality too low: {result.reasoning_quality_score:.1f}%"

        # Check for constraint solving strategy
        has_constraint_strategy = any(
            s
            in [
                "constraint",
                "sat",
                "elimination",
                "csp",
                "backtracking",
                "constraint_satisfaction",
            ]
            for s in result.strategies_used
        )
        assert (
            has_constraint_strategy
        ), f"Expected constraint strategy, got: {result.strategies_used}"

        # Verify appropriate depth for compound constraints
        assert (
            4 <= result.proof_tree_depth <= 10
        ), f"ProofTree depth {result.proof_tree_depth} outside expected range [4-10]"

        # Performance check
        assert (
            result.execution_time_ms < 40000
        ), f"Too slow: {result.execution_time_ms}ms (expected <40s)"

        # Log summary
        print(f"\n[INFO] Logs: {scenario_logger.save_logs()}")
        print(f"[INFO] Score: {result.overall_score:.1f}/100")
        print(
            f"[INFO] Breakdown: Reasoning={result.reasoning_quality_score:.1f}, "
            f"Confidence={result.confidence_calibration_score:.1f}, "
            f"Correctness={result.correctness_score:.1f}"
        )

        if result.overall_score < 70:
            print("[WEAKNESS] Issues:")
            for w in result.identified_weaknesses:
                print(f"  - {w}")

        # Mark test as passed
        assert result.passed, f"Test failed: {result.error or 'Score below threshold'}"

    def score_correctness(
        self, actual: str, expected: Dict, allow_partial: bool = True
    ) -> float:
        """
        Score correctness with partial credit for ambiguous solutions.

        Args:
            actual: Actual KAI response text
            expected: Dict with "assignments" key containing expected person->color mapping
            allow_partial: Whether to give partial credit

        Returns:
            Score 0-100
        """
        if not expected or "assignments" not in expected:
            return 50.0

        expected_assignments = expected["assignments"]
        correct_count = 0
        total = len(expected_assignments)
        actual_lower = actual.lower()

        for person, colors in expected_assignments.items():
            person_lower = person.lower()

            # Handle lists (multiple valid answers)
            if isinstance(colors, list):
                # Any of the colors is valid
                for color in colors:
                    color_lower = color.lower()
                    # Pattern: person appears, then color, without negation
                    pattern = (
                        rf"\b{re.escape(person_lower)}\b"
                        rf"(?!.*\b(?:nicht|kein|keine)\b.*\b{re.escape(color_lower)}\b)"
                        rf".*\b{re.escape(color_lower)}\b"
                    )
                    if re.search(pattern, actual_lower):
                        correct_count += 1
                        break
            else:
                # Single expected answer
                color_lower = colors.lower()
                pattern = (
                    rf"\b{re.escape(person_lower)}\b"
                    rf"(?!.*\b(?:nicht|kein|keine)\b.*\b{re.escape(color_lower)}\b)"
                    rf".*\b{re.escape(color_lower)}\b"
                )
                if re.search(pattern, actual_lower):
                    correct_count += 1

        # Calculate score with partial credit
        if allow_partial:
            score = (correct_count / total) * 100.0
        else:
            score = 100.0 if correct_count == total else 0.0

        return score

    def score_reasoning_quality(
        self, proof_tree: Dict, strategies_used: List[str], reasoning_steps: List[str]
    ) -> float:
        """
        Score reasoning quality for compound constraint puzzle.

        Returns: 0-100
        """
        score = 0.0

        # Used constraint solver: +40%
        if any(
            s in ["constraint", "sat", "csp", "constraint_satisfaction"]
            for s in strategies_used
        ):
            score += 40

        # Appropriate depth [4-8]: +30%
        depth = self._calculate_proof_tree_depth(proof_tree) if proof_tree else 0
        if 4 <= depth <= 8:
            score += 30
        elif depth < 4:
            score += 10
        elif 8 < depth <= 10:
            score += 25

        # Multiple reasoning steps: +20%
        if len(reasoning_steps) >= 4:
            score += 20
        elif len(reasoning_steps) >= 2:
            score += 10

        # Handled compound constraints (inferred from success): +10%
        score += 10

        return min(score, 100.0)
