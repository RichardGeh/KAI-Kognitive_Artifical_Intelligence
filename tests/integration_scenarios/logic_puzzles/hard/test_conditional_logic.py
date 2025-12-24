"""
tests/integration_scenarios/logic_puzzles/hard/test_conditional_logic.py

Hard-level logic puzzle: Nested if-then constraints with contrapositive reasoning

Scenario:
Logic puzzle requiring modus ponens, modus tollens, and contrapositive reasoning.
Nested conditional constraints (if-then-else) with multiple levels.

Expected Reasoning:
- Logic engine should handle conditional statements
- Modus ponens: If A then B, A is true => B is true
- Modus tollens: If A then B, B is false => A is false
- Contrapositive: If A then B <=> If not B then not A
- ProofTree should show logical inference chains
- Confidence should be high (>0.80) for deductive logic

Success Criteria (Gradual Scoring):
- Correctness: 30% (got right answer)
- Reasoning Quality: 50% (used logic engine, correct inference rules)
- Confidence Calibration: 20% (confidence matches correctness)
"""

import re
from typing import Dict, List

from tests.integration_scenarios.utils.scenario_base import ScenarioTestBase


class TestConditionalLogic(ScenarioTestBase):
    """Test: Nested conditional logic with modus ponens/tollens"""

    DIFFICULTY = "hard"
    DOMAIN = "logic_puzzles"
    TIMEOUT_SECONDS = 120  # 2 minutes

    # Scoring weights for this scenario type
    REASONING_QUALITY_WEIGHT = 0.5
    CONFIDENCE_CALIBRATION_WEIGHT = 0.2
    CORRECTNESS_WEIGHT = 0.3

    def test_conditional_logic_puzzle(
        self,
        kai_worker_scenario_mode,
        progress_reporter,
        scenario_logger,
        confidence_tracker,
    ):
        """
        Test: Nested conditional logic with contrapositive reasoning

        Apply modus ponens, modus tollens, and contrapositive inference
        to deduce correct conclusions.

        Expected: Korrekte logische Schlussfolgering
        """

        # Setup progress reporter
        progress_reporter.total_steps = 5
        progress_reporter.start()

        # Puzzle input
        puzzle_text = """
Logik-Raetsel mit bedingten Aussagen:

Gegeben sind folgende Regeln:
1. Wenn Anna ins Kino geht, dann geht Ben auch ins Kino.
2. Wenn Ben ins Kino geht, dann geht Clara nicht ins Kino.
3. Wenn Clara nicht ins Kino geht, dann geht David ins Kino.
4. Wenn David ins Kino geht, dann geht Emma auch ins Kino.
5. Wenn Emma ins Kino geht, dann geht Franz nicht ins Kino.

Zusaetzliche Fakten:
- Anna geht ins Kino.
- Franz geht nicht ins Kino.

Fragen:
a) Geht Ben ins Kino?
b) Geht Clara ins Kino?
c) Geht David ins Kino?
d) Geht Emma ins Kino?

Begruende deine Antworten mit den logischen Schlussregeln.
        """

        # Expected solution
        expected_solution = {
            "Ben": "ja",
            "Clara": "nein",
            "David": "ja",
            "Emma": "ja",
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
        # SAT solver is a valid strategy for conditional logic puzzles
        has_logic_strategy = any(
            s
            in [
                "logic",
                "modus_ponens",
                "modus_tollens",
                "contrapositive",
                "inference",
                "sat",  # SAT solver is valid for conditional constraints
                "logic_puzzle",
            ]
            for s in result.strategies_used
        )
        assert (
            has_logic_strategy
        ), f"Expected logic/inference/SAT strategy, got: {result.strategies_used}"

        # Verify reasoning depth is appropriate
        # SAT solver may produce shallower trees than explicit logic inference
        assert (
            3 <= result.proof_tree_depth <= 18
        ), f"ProofTree depth {result.proof_tree_depth} outside expected range [3-18]"

        # Check performance
        assert (
            result.execution_time_ms < 90000
        ), f"Too slow: {result.execution_time_ms}ms (expected <90s)"

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
        Score correctness based on how many correct answers were given.

        Args:
            actual: Actual KAI response text
            expected: Dict with "solution" key containing person->answer mapping
            allow_partial: Whether to give partial credit

        Returns:
            Score 0-100
        """
        if not expected or "solution" not in expected:
            return 50.0

        solution = expected["solution"]
        correct_count = 0
        total = len(solution)

        actual_lower = actual.lower()

        # Check each expected answer
        for person, answer in solution.items():
            person_lower = person.lower()
            answer_lower = answer.lower()

            # Pattern: person goes (ja) or doesn't go (nein)
            if answer_lower == "ja":
                # Look for affirmative: "Ben geht ins Kino", "Ben: ja"
                pattern = (
                    rf"\b{re.escape(person_lower)}\b.*\b(?:geht|ja)\b"
                    rf"(?!.*\b(?:nicht|nein)\b)"
                )
            else:  # "nein"
                # Look for negative: "Clara geht nicht", "Clara: nein"
                pattern = rf"\b{re.escape(person_lower)}\b.*\b(?:nicht|nein)\b"

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
        # SAT solver is a valid strategy for conditional logic puzzles
        has_logic = any(
            s
            in [
                "logic",
                "modus_ponens",
                "modus_tollens",
                "contrapositive",
                "inference",
                "sat",  # SAT solver handles conditional constraints
                "logic_puzzle",
            ]
            for s in strategies_used
        )
        if has_logic:
            score += 40

        # Appropriate ProofTree depth: +30%
        # SAT solver may produce shallower trees than explicit logic inference
        depth = self._calculate_proof_tree_depth(proof_tree) if proof_tree else 0
        if 3 <= depth <= 18:
            score += 30
        elif depth > 18:
            score += 25

        # Multiple reasoning steps (chained inference): +20%
        if len(reasoning_steps) >= 8:
            score += 20
        elif len(reasoning_steps) >= 5:
            score += 15
        elif len(reasoning_steps) >= 3:
            score += 10

        # Explicit inference rules mentioned: +10%
        inference_keywords = [
            "modus",
            "wenn.*dann",
            "weil",
            "folgt",
            "daher",
            "deshalb",
            "unit",  # SAT unit propagation
            "assignment",
        ]
        actual_text = " ".join(reasoning_steps).lower()
        if any(re.search(keyword, actual_text) for keyword in inference_keywords):
            score += 10

        return min(score, 100.0)
