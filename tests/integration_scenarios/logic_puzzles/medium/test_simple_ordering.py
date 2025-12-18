"""
tests/integration_scenarios/logic_puzzles/medium/test_simple_ordering.py

Medium-level logic puzzle: 5 people in a line with ordering constraints

Scenario:
Five people stand in a line with ordering constraints (A left of B, etc.).
Includes negative constraints ("not at end", "not directly next to").
Tests transitive reasoning and position constraints.

Expected Reasoning:
- Transitive relation reasoning (A < B < C)
- Position constraint handling (not at end, not adjacent)
- ProofTree shows ordering deductions
- Confidence medium-high (>0.75)

Success Criteria (Gradual Scoring):
- Correctness: 30% (correct ordering)
- Reasoning Quality: 50% (transitive reasoning, constraint handling)
- Confidence Calibration: 20% (confidence matches correctness)
"""

from typing import Dict, List

from tests.integration_scenarios.utils.scenario_base import ScenarioTestBase


class TestSimpleOrdering(ScenarioTestBase):
    """Test: 5-person ordering puzzle with transitive constraints"""

    DIFFICULTY = "medium"
    DOMAIN = "logic_puzzles"
    TIMEOUT_SECONDS = 600

    REASONING_QUALITY_WEIGHT = 0.5
    CONFIDENCE_CALIBRATION_WEIGHT = 0.2
    CORRECTNESS_WEIGHT = 0.3

    def test_five_person_ordering(
        self,
        kai_worker_scenario_mode,
        progress_reporter,
        scenario_logger,
        confidence_tracker,
    ):
        """
        Test: 5-person line ordering with transitive constraints

        Five people in a line with left/right constraints and negations.
        Expected: A, B, E, C, D (or similar valid ordering)
        """

        # Setup
        progress_reporter.total_steps = 5
        progress_reporter.start()

        puzzle_text = """
Fuenf Personen stehen in einer Reihe: A, B, C, D, E.
1. A steht links von B.
2. B steht links von C.
3. D steht rechts von C.
4. E steht nicht am Ende.
5. B steht nicht direkt neben D.

In welcher Reihenfolge stehen sie?
        """

        # Valid solution: A, B, E, C, D
        expected_solution = ["A", "B", "E", "C", "D"]

        # Execute scenario using base class
        result = self.run_scenario(
            input_text=puzzle_text,
            expected_outputs={"ordering": expected_solution},
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

        # Check for ordering/transitive strategy
        has_ordering_strategy = any(
            s
            in [
                "constraint",
                "transitive",
                "ordering",
                "graph_traversal",
                "constraint_satisfaction",
            ]
            for s in result.strategies_used
        )
        assert (
            has_ordering_strategy
        ), f"Expected ordering/transitive strategy, got: {result.strategies_used}"

        # Verify appropriate depth
        assert (
            5 <= result.proof_tree_depth <= 12
        ), f"ProofTree depth {result.proof_tree_depth} outside expected range [5-12]"

        # Performance
        assert (
            result.execution_time_ms < 45000
        ), f"Too slow: {result.execution_time_ms}ms"

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
        Score correctness based on ordering accuracy.

        Args:
            actual: Actual KAI response text
            expected: Dict with "ordering" key containing expected person list
            allow_partial: Whether to give partial credit

        Returns:
            Score 0-100
        """
        if not expected or "ordering" not in expected:
            return 50.0

        expected_order = expected["ordering"]

        # Extract ordering from response
        found_order = []
        for person in expected_order:
            if person in actual:
                found_order.append((actual.find(person), person))

        found_order.sort()  # Sort by position in response
        actual_order = [p[1] for p in found_order]

        if len(actual_order) < len(expected_order):
            return 0.0  # Missing people

        # Score based on correct positions (60%)
        correct_positions = 0
        for i, person in enumerate(expected_order):
            if i < len(actual_order) and actual_order[i] == person:
                correct_positions += 1

        # Also give partial credit for correct relative ordering (40%)
        correct_pairs = 0
        total_pairs = len(expected_order) - 1

        for i in range(len(expected_order) - 1):
            person1 = expected_order[i]
            person2 = expected_order[i + 1]
            if person1 in actual_order and person2 in actual_order:
                idx1 = actual_order.index(person1)
                idx2 = actual_order.index(person2)
                if idx1 < idx2:  # Correct relative order
                    correct_pairs += 1

        # Combine position score (60%) and relative order score (40%)
        position_score = (correct_positions / len(expected_order)) * 60.0
        relative_score = (
            (correct_pairs / total_pairs) * 40.0 if total_pairs > 0 else 0.0
        )

        return position_score + relative_score

    def score_reasoning_quality(
        self, proof_tree: Dict, strategies_used: List[str], reasoning_steps: List[str]
    ) -> float:
        """
        Score reasoning quality for ordering puzzle.

        Returns: 0-100
        """
        score = 0.0

        # Used transitive/ordering strategy: +40%
        if any(
            s in ["transitive", "ordering", "constraint", "constraint_satisfaction"]
            for s in strategies_used
        ):
            score += 40

        # Appropriate depth [5-10]: +30%
        depth = self._calculate_proof_tree_depth(proof_tree) if proof_tree else 0
        if 5 <= depth <= 10:
            score += 30
        elif depth < 5:
            score += 10
        elif 10 < depth <= 12:
            score += 25

        # Evidence of transitive reasoning: +20%
        if "transitive" in strategies_used or "graph_traversal" in strategies_used:
            score += 20
        elif len(reasoning_steps) >= 5:
            score += 10

        # Multiple reasoning steps: +10%
        if len(reasoning_steps) >= 3:
            score += 10

        return min(score, 100.0)
