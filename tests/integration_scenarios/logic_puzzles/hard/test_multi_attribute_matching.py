"""
tests/integration_scenarios/logic_puzzles/hard/test_multi_attribute_matching.py

Hard-level logic puzzle: 5 entities with 5 attributes each (5x5 grid assignment)

Scenario:
Five people with five different attributes each: name, job, city, pet, hobby.
Given 12+ constraints, determine the complete 5x5 assignment grid.

Expected Reasoning:
- Input Orchestrator should segment constraints
- CSP solver with constraint propagation
- Grid-based reasoning (position constraints)
- ProofTree should show progressive constraint elimination
- Confidence should be high (>0.75) for complete solution

Success Criteria (Gradual Scoring):
- Correctness: 30% (got right assignments)
- Reasoning Quality: 50% (used appropriate strategies, systematic elimination)
- Confidence Calibration: 20% (confidence matches correctness)
"""

import re
from typing import Dict, List

from tests.integration_scenarios.utils.scenario_base import ScenarioTestBase


class TestMultiAttributeMatching(ScenarioTestBase):
    """Test: 5-person puzzle with 5 attributes each"""

    DIFFICULTY = "hard"
    DOMAIN = "logic_puzzles"
    TIMEOUT_SECONDS = 1800  # 30 minutes

    # Scoring weights for this scenario type
    REASONING_QUALITY_WEIGHT = 0.5
    CONFIDENCE_CALIBRATION_WEIGHT = 0.2
    CORRECTNESS_WEIGHT = 0.3

    def test_multi_attribute_matching(
        self,
        kai_worker_scenario_mode,
        progress_reporter,
        scenario_logger,
        confidence_tracker,
    ):
        """
        Test: 5 people with 5 attributes each

        Determine complete assignment: person -> job, city, pet, hobby

        Expected: Complete 5x5 grid with all assignments correct
        """

        # Setup progress reporter
        progress_reporter.total_steps = 5
        progress_reporter.start()

        # Puzzle input
        puzzle_text = """
Fuenf Personen haben jeweils einen Beruf, wohnen in einer Stadt, haben ein Haustier und ein Hobby.
Namen: Anna, Ben, Clara, David, Emma
Berufe: Lehrer, Arzt, Ingenieur, Koch, Anwalt
Staedte: Berlin, Hamburg, Muenchen, Koeln, Frankfurt
Haustiere: Hund, Katze, Vogel, Fisch, Hamster
Hobbys: Lesen, Sport, Musik, Malen, Reisen

Hinweise:
1. Anna ist Lehrerin.
2. Die Person aus Berlin hat einen Hund.
3. Der Arzt wohnt in Hamburg.
4. Ben mag Lesen als Hobby.
5. Clara ist nicht die Anwaeltin.
6. Der Ingenieur hat eine Katze.
7. Die Person aus Muenchen malt gerne.
8. David ist Koch.
9. Emma wohnt nicht in Frankfurt.
10. Die Person mit dem Vogel treibt Sport.
11. Der Anwalt wohnt in Koeln.
12. Die Person aus Frankfurt hat einen Fisch.

Frage: Wer hat welchen Beruf, wohnt wo, hat welches Haustier und welches Hobby?
        """

        # Expected solution (partial - key assignments to check)
        expected_solution = {
            "Anna": {"beruf": "Lehrerin"},
            "David": {"beruf": "Koch"},
            "Ben": {"hobby": "Lesen"},
            "Berlin": {"haustier": "Hund"},
            "Hamburg": {"beruf": "Arzt"},
            "Muenchen": {"hobby": "Malen"},
            "Frankfurt": {"haustier": "Fisch"},
            "Koeln": {"beruf": "Anwalt"},
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
            s in ["constraint", "csp", "elimination", "grid_reasoning", "propagation"]
            for s in result.strategies_used
        )
        assert (
            has_constraint_strategy
        ), f"Expected CSP/constraint strategy, got: {result.strategies_used}"

        # Verify reasoning depth is appropriate
        assert (
            10 <= result.proof_tree_depth <= 20
        ), f"ProofTree depth {result.proof_tree_depth} outside expected range [10-20]"

        # Check performance
        assert (
            result.execution_time_ms < 120000
        ), f"Too slow: {result.execution_time_ms}ms (expected <120s)"

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
        Score correctness based on how many correct assignments were made.

        Args:
            actual: Actual KAI response text
            expected: Dict with "assignments" key
            allow_partial: Whether to give partial credit

        Returns:
            Score 0-100
        """
        if not expected or "assignments" not in expected:
            return 50.0

        assignments = expected["assignments"]
        correct_count = 0
        total = len(assignments)

        actual_lower = actual.lower()

        # Check each expected assignment
        for entity, attributes in assignments.items():
            entity_lower = entity.lower()

            for attr_type, attr_value in attributes.items():
                attr_lower = attr_value.lower()

                # Pattern: entity associated with attribute value
                pattern = (
                    rf"\b{re.escape(entity_lower)}\b.*\b{re.escape(attr_lower)}\b|"
                    rf"\b{re.escape(attr_lower)}\b.*\b{re.escape(entity_lower)}\b"
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
        Score reasoning quality based on strategy appropriateness and depth.

        Returns: 0-100 score
        """
        score = 0.0

        # Used appropriate strategy: +40%
        has_constraint = any(
            s in ["constraint", "csp", "elimination", "grid_reasoning", "propagation"]
            for s in strategies_used
        )
        if has_constraint:
            score += 40

        # Appropriate ProofTree depth [10-20]: +30%
        depth = self._calculate_proof_tree_depth(proof_tree) if proof_tree else 0
        if 10 <= depth <= 20:
            score += 30
        elif 8 <= depth < 10:
            score += 20
        elif 20 < depth <= 25:
            score += 25

        # Multiple reasoning steps: +20%
        if len(reasoning_steps) >= 12:
            score += 20
        elif len(reasoning_steps) >= 8:
            score += 15
        elif len(reasoning_steps) >= 5:
            score += 10

        # Systematic elimination visible: +10%
        if len(set(strategies_used)) >= 2:
            score += 10

        return min(score, 100.0)
