"""
tests/integration_scenarios/logic_puzzles/hard/test_five_person_zebra.py

Hard-level logic puzzle: 5-person zebra puzzle with 15+ constraints

Scenario:
Classic 5-person puzzle with 5 attributes each (person, house color, drink, pet, job).
Given 15+ constraints including transitive, negative, and positional constraints.
Find who owns the zebra and who drinks water.

Expected Reasoning:
- Input Orchestrator should segment constraints
- SAT/CSP/Constraint solver required
- Backtracking and exhaustive search likely needed
- ProofTree should show constraint satisfaction with multiple branches
- Confidence should be medium-high (>0.70) for unique solution

Success Criteria (Gradual Scoring):
- Correctness: 30% (got right answer)
- Reasoning Quality: 50% (used appropriate strategies, deep reasoning, handled complex constraints)
- Confidence Calibration: 20% (confidence matches correctness)
"""

import re
from typing import Dict, List

from tests.integration_scenarios.utils.scenario_base import ScenarioTestBase


class TestFivePersonZebraPuzzle(ScenarioTestBase):
    """Test: Classic 5-person zebra puzzle with complex constraints"""

    DIFFICULTY = "hard"
    DOMAIN = "logic_puzzles"
    TIMEOUT_SECONDS = 1800  # 30 minutes

    # Scoring weights for this scenario type
    REASONING_QUALITY_WEIGHT = 0.5
    CONFIDENCE_CALIBRATION_WEIGHT = 0.2
    CORRECTNESS_WEIGHT = 0.3

    def test_five_person_zebra_puzzle(
        self,
        kai_worker_scenario_mode,
        progress_reporter,
        scenario_logger,
        confidence_tracker,
    ):
        """
        Test: Classic 5-person zebra puzzle

        Five people live in five houses with different attributes.
        Find who owns the zebra and who drinks water.

        Expected: German owns zebra, Norwegian drinks water
        """

        # Setup progress reporter
        progress_reporter.total_steps = 5
        progress_reporter.start()

        # Puzzle input
        puzzle_text = """
Es gibt fuenf Haeuser in fuenf verschiedenen Farben.
In jedem Haus lebt eine Person mit einem anderen Beruf.
Jeder hat ein anderes Haustier, trinkt ein anderes Getraenk und hat eine andere Nationalitaet.

Hinweise:
1. Der Brite lebt im roten Haus.
2. Der Schwede hat einen Hund.
3. Der Daene trinkt Tee.
4. Das gruene Haus steht direkt links vom weissen Haus.
5. Der Besitzer des gruenen Hauses trinkt Kaffee.
6. Die Person, die Pall Mall raucht, hat einen Vogel.
7. Der Besitzer des gelben Hauses raucht Dunhill.
8. Die Person im mittleren Haus trinkt Milch.
9. Der Norweger lebt im ersten Haus.
10. Die Person, die Blend raucht, lebt neben der Person mit der Katze.
11. Die Person mit dem Pferd lebt neben der Person, die Dunhill raucht.
12. Die Person, die Blue Master raucht, trinkt Bier.
13. Der Deutsche raucht Prince.
14. Der Norweger lebt neben dem blauen Haus.
15. Die Person, die Blend raucht, hat einen Nachbarn, der Wasser trinkt.

Frage: Wer hat das Zebra und wer trinkt Wasser?
        """

        # Expected solution
        expected_solution = {
            "zebra_owner": "Deutscher",
            "water_drinker": "Norweger",
        }

        # Execute scenario using base class
        result = self.run_scenario(
            input_text=puzzle_text,
            expected_outputs={"solution": expected_solution},
            context={},
            kai_worker=kai_worker_scenario_mode,
            logger=scenario_logger,
            progress_reporter=progress_reporter,
            confidence_tracker=confidence_tracker,
        )

        # Assertions
        assert (
            result.overall_score >= 40
        ), f"Overall score too low: {result.overall_score:.1f}% (expected >= 40%)"

        assert (
            result.correctness_score >= 30
        ), f"Expected at least 30% correctness, got {result.correctness_score:.1f}%"

        assert (
            result.reasoning_quality_score >= 35
        ), f"Reasoning quality too low: {result.reasoning_quality_score:.1f}%"

        # Check that appropriate strategies were used
        has_constraint_strategy = any(
            s in ["constraint", "sat", "csp", "backtracking", "exhaustive_search"]
            for s in result.strategies_used
        )
        assert (
            has_constraint_strategy
        ), f"Expected constraint/SAT strategy, got: {result.strategies_used}"

        # Verify reasoning depth is appropriate (deeper for hard puzzle)
        assert (
            10 <= result.proof_tree_depth <= 20
        ), f"ProofTree depth {result.proof_tree_depth} outside expected range [10-20]"

        # Check performance (allow longer for hard puzzle)
        assert (
            result.execution_time_ms < 120000
        ), f"Too slow: {result.execution_time_ms}ms (expected <120s for hard puzzle)"

        # Log summary
        print(f"\n[INFO] Detailed logs saved to: {scenario_logger.save_logs()}")
        print(f"[INFO] Overall Score: {result.overall_score:.1f}/100")
        print(
            f"[INFO] Breakdown: "
            f"Reasoning={result.reasoning_quality_score:.1f}, "
            f"Confidence={result.confidence_calibration_score:.1f}, "
            f"Correctness={result.correctness_score:.1f}"
        )

        # Identify weaknesses if score is low
        if result.overall_score < 60:
            print("[WEAKNESS] Identified issues:")
            for weakness in result.identified_weaknesses:
                print(f"  - {weakness}")

        # Mark test as passed
        assert result.passed, f"Test failed: {result.error or 'Score below threshold'}"

    def score_correctness(
        self, actual: str, expected: Dict, allow_partial: bool = True
    ) -> float:
        """
        Score correctness based on whether zebra owner and water drinker are correct.

        Args:
            actual: Actual KAI response text
            expected: Dict with "solution" key containing zebra_owner and water_drinker
            allow_partial: Whether to give partial credit

        Returns:
            Score 0-100
        """
        if not expected or "solution" not in expected:
            return 50.0  # Neutral score if no expected data

        solution = expected["solution"]
        correct_count = 0
        total = 2  # zebra_owner and water_drinker

        actual_lower = actual.lower()

        # Check zebra owner
        if "zebra_owner" in solution:
            zebra_owner = solution["zebra_owner"].lower()
            # Pattern: zebra owner mentioned (e.g., "Deutscher hat das Zebra")
            pattern = rf"\b{re.escape(zebra_owner)}\b.*\bzebra\b|\bzebra\b.*\b{re.escape(zebra_owner)}\b"
            if re.search(pattern, actual_lower):
                correct_count += 1

        # Check water drinker
        if "water_drinker" in solution:
            water_drinker = solution["water_drinker"].lower()
            # Pattern: water drinker mentioned (e.g., "Norweger trinkt Wasser")
            pattern = rf"\b{re.escape(water_drinker)}\b.*\bwasser\b|\bwasser\b.*\b{re.escape(water_drinker)}\b"
            if re.search(pattern, actual_lower):
                correct_count += 1

        # Calculate score with partial credit
        if allow_partial:
            # 2/2 = 100%, 1/2 = 50%, 0/2 = 0%
            score = (correct_count / total) * 100.0
        else:
            # All-or-nothing
            score = 100.0 if correct_count == total else 0.0

        return score

    def score_reasoning_quality(
        self, proof_tree: Dict, strategies_used: List[str], reasoning_steps: List[str]
    ) -> float:
        """
        Score reasoning quality based on strategy appropriateness and depth.

        Returns: 0-100 score
        """
        score = 0.0

        # Used appropriate strategy: +40%
        has_constraint = any(
            s in ["constraint", "sat", "csp", "backtracking", "exhaustive_search"]
            for s in strategies_used
        )
        if has_constraint:
            score += 40

        # Appropriate ProofTree depth [10-20]: +30%
        depth = self._calculate_proof_tree_depth(proof_tree) if proof_tree else 0
        if 10 <= depth <= 20:
            score += 30
        elif 8 <= depth < 10:
            score += 20  # Slightly shallow
        elif 20 < depth <= 25:
            score += 25  # Slightly deep but acceptable

        # Multiple reasoning steps (complex puzzle): +20%
        if len(reasoning_steps) >= 15:
            score += 20
        elif len(reasoning_steps) >= 10:
            score += 15
        elif len(reasoning_steps) >= 5:
            score += 10

        # Multiple strategies used (multimodal reasoning): +10%
        if len(set(strategies_used)) >= 2:
            score += 10
        elif len(set(strategies_used)) >= 1:
            score += 5

        return min(score, 100.0)
