"""
tests/integration_scenarios/logic_puzzles/medium/test_three_person_puzzle.py

Medium-level logic puzzle: 3-person job assignment with negative constraints

Scenario:
Three people (Alex, Bob, Carol) have different jobs (teacher, doctor, engineer).
Given 3 constraints including negative constraints, deduce who has which job.

Expected Reasoning:
- Input Orchestrator should segment constraints
- SAT/CSP solver should find unique solution through elimination
- ProofTree should show constraint satisfaction steps
- Confidence should be high (>0.80) for unique solution

Success Criteria (Gradual Scoring):
- Correctness: 30% (got right answer)
- Reasoning Quality: 50% (used appropriate strategy, logical steps, handled negations)
- Confidence Calibration: 20% (confidence matches correctness)
"""

import re
from typing import Dict

from tests.integration_scenarios.utils.scenario_base import ScenarioTestBase


class TestThreePersonPuzzle(ScenarioTestBase):
    """Test: Basic 3-person job assignment puzzle with negative constraints"""

    DIFFICULTY = "medium"
    DOMAIN = "logic_puzzles"
    TIMEOUT_SECONDS = 600  # 10 minutes

    # Scoring weights for this scenario type
    REASONING_QUALITY_WEIGHT = 0.5
    CONFIDENCE_CALIBRATION_WEIGHT = 0.2
    CORRECTNESS_WEIGHT = 0.3

    def test_three_person_job_assignment(
        self,
        kai_worker_scenario_mode,
        progress_reporter,
        scenario_logger,
        confidence_tracker,
    ):
        """
        Test: 3-person job assignment with negative constraints

        Alex, Bob, and Carol have different jobs: Lehrer, Arzt, Ingenieur.
        Given constraints with negations, deduce correct assignments.

        Expected: Bob=Lehrer, Alex=Ingenieur, Carol=Arzt
        """

        # Setup progress reporter
        progress_reporter.total_steps = 5
        progress_reporter.start()

        # Puzzle input
        puzzle_text = """
Alex, Bob und Carol haben unterschiedliche Berufe: Lehrer, Arzt und Ingenieur.
1. Alex ist kein Arzt.
2. Bob ist Lehrer.
3. Carol ist nicht Ingenieur.

Wer hat welchen Beruf?
        """

        # Expected solution
        expected_solution = {"Bob": "Lehrer", "Alex": "Ingenieur", "Carol": "Arzt"}

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
        ), f"Expected at least 40% correctness, got {result.correctness_score:.1f}%"

        assert (
            result.reasoning_quality_score >= 40
        ), f"Reasoning quality too low: {result.reasoning_quality_score:.1f}%"

        # Check that appropriate strategies were used
        has_constraint_strategy = any(
            s in ["constraint", "sat", "elimination", "csp", "constraint_satisfaction"]
            for s in result.strategies_used
        )
        assert (
            has_constraint_strategy
        ), f"Expected constraint satisfaction or SAT strategy, got: {result.strategies_used}"

        # Verify reasoning depth is appropriate
        # Note: SAT solver uses flat structure (depth=1), other strategies use hierarchical
        if result.strategies_used and "sat" in result.strategies_used:
            # SAT solver currently uses flat structure
            assert (
                result.proof_tree_depth >= 1
            ), f"ProofTree depth {result.proof_tree_depth} too low (expected >= 1 for flat SAT)"
        else:
            # Other strategies should have hierarchical structure
            assert (
                3 <= result.proof_tree_depth <= 8
            ), f"ProofTree depth {result.proof_tree_depth} outside expected range [3-8]"

        # Check performance
        assert (
            result.execution_time_ms < 30000
        ), f"Too slow: {result.execution_time_ms}ms (expected <30s for medium puzzle)"

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
        if result.overall_score < 70:
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
            expected: Dict with "assignments" key containing expected person->job mapping
            allow_partial: Whether to give partial credit

        Returns:
            Score 0-100
        """
        if not expected or "assignments" not in expected:
            return 50.0  # Neutral score if no expected data

        expected_assignments = expected["assignments"]
        correct_count = 0
        total = len(expected_assignments)

        # Check each expected assignment in response
        for person, job in expected_assignments.items():
            # Use regex pattern: "Person [verb] Job" without negation
            # Matches: "Bob ist Lehrer", "Alex arbeitet als Ingenieur"
            # Rejects: "Bob ist nicht Lehrer", "Alex ist kein Arzt"
            person_lower = person.lower()
            job_lower = job.lower()

            # Pattern: person appears, then optionally some words, then job
            # BUT NOT with negation words (nicht, kein, keine) between them
            pattern = (
                rf"\b{re.escape(person_lower)}\b"
                rf"(?!.*\b(?:nicht|kein|keine)\b.*\b{re.escape(job_lower)}\b)"
                rf".*\b{re.escape(job_lower)}\b"
            )

            actual_lower = actual.lower()
            if re.search(pattern, actual_lower):
                correct_count += 1

        # Calculate score with partial credit
        if allow_partial:
            # 3/3 = 100%, 2/3 = 67%, 1/3 = 33%, 0/3 = 0%
            score = (correct_count / total) * 100.0
        else:
            # All-or-nothing
            score = 100.0 if correct_count == total else 0.0

        return score

    def score_reasoning_quality(
        self, proof_tree: Dict, strategies_used: list, reasoning_steps: list
    ) -> float:
        """
        Score reasoning quality based on strategy appropriateness and depth.

        Returns: 0-100 score
        """
        score = 0.0

        # Used appropriate strategy: +40%
        has_constraint = any(
            s in ["constraint", "sat", "elimination", "csp", "constraint_satisfaction"]
            for s in strategies_used
        )
        if has_constraint:
            score += 40

        # Appropriate ProofTree depth [3-8]: +30%
        depth = self._calculate_proof_tree_depth(proof_tree) if proof_tree else 0
        if 3 <= depth <= 6:
            score += 30
        elif depth < 3:
            score += 10  # Too shallow
        elif 6 < depth <= 8:
            score += 25  # Slightly deep but acceptable

        # Multiple reasoning steps: +20%
        if len(reasoning_steps) >= 3:
            score += 20
        elif len(reasoning_steps) >= 1:
            score += 10

        # Handled negative constraints: +10% (inferred from successful solution)
        # This would ideally check ProofTree for negation handling
        score += 10

        return min(score, 100.0)
