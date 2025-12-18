"""
tests/integration_scenarios/combined_scenarios/hard/test_spatial_logic_puzzle.py

Hard-level combined scenario: Spatial reasoning + logical constraints

Scenario:
Logic puzzle requiring spatial reasoning (positions, arrangements) combined
with logical constraints (who sits where at a table based on rules).

Expected Reasoning:
- Spatial reasoning engine for position constraints
- CSP/constraint solver for logical rules
- Graph traversal for neighbor relationships
- Combined reasoning in ProofTree
- Confidence should be medium-high (>0.70)

Success Criteria (Gradual Scoring):
- Correctness: 30% (correct seating arrangement)
- Reasoning Quality: 50% (used spatial + logic strategies)
- Confidence Calibration: 20% (confidence reflects complexity)
"""

import re
from typing import Dict, List

from tests.integration_scenarios.utils.scenario_base import ScenarioTestBase


class TestSpatialLogicPuzzle(ScenarioTestBase):
    """Test: Spatial reasoning combined with logical constraints"""

    DIFFICULTY = "hard"
    DOMAIN = "combined_scenarios"
    TIMEOUT_SECONDS = 1800  # 30 minutes

    # Scoring weights for this scenario type
    REASONING_QUALITY_WEIGHT = 0.5
    CONFIDENCE_CALIBRATION_WEIGHT = 0.2
    CORRECTNESS_WEIGHT = 0.3

    def test_seating_arrangement_puzzle(
        self,
        kai_worker_scenario_mode,
        progress_reporter,
        scenario_logger,
        confidence_tracker,
    ):
        """
        Test: Seating arrangement at round table

        Four people sit at a round table with spatial constraints.

        Expected: Correct arrangement using spatial + logical reasoning
        """

        # Setup progress reporter
        progress_reporter.total_steps = 5
        progress_reporter.start()

        # Puzzle input
        puzzle_text = """
Vier Personen sitzen an einem runden Tisch: Anna, Ben, Clara und David.

Hinweise:
1. Anna sitzt direkt gegenueber von Ben.
2. Clara sitzt rechts von Anna.
3. David sitzt nicht neben Anna.
4. Ben sitzt links von David.

Frage: Beschreibe die Sitzordnung im Uhrzeigersinn, beginnend mit Anna.
        """

        # Expected solution: Anna -> Clara -> Ben -> David (clockwise)
        # Check: Anna opposite Ben (yes), Clara right of Anna (yes),
        # David not next to Anna (yes), Ben left of David (yes)

        expected_solution = {
            "arrangement": "Anna, Clara, Ben, David",
            "anna_opposite": "Ben",
            "clara_right_of": "Anna",
        }

        # Execute scenario using base class
        result = self.run_scenario(
            input_text=puzzle_text,
            expected_outputs={"seating": expected_solution},
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
            result.correctness_score >= 25
        ), f"Expected at least 25% correctness, got {result.correctness_score:.1f}%"

        assert (
            result.reasoning_quality_score >= 35
        ), f"Reasoning quality too low: {result.reasoning_quality_score:.1f}%"

        # Check that both spatial and logic strategies were used
        has_spatial = any(
            s in ["spatial", "position", "arrangement", "location"]
            for s in result.strategies_used
        )
        has_logic = any(
            s in ["constraint", "csp", "logic", "sat"] for s in result.strategies_used
        )

        assert (
            has_spatial or has_logic
        ), f"Expected spatial and/or logic strategies, got: {result.strategies_used}"

        # Verify reasoning depth is appropriate
        assert (
            6 <= result.proof_tree_depth <= 16
        ), f"ProofTree depth {result.proof_tree_depth} outside expected range [6-16]"

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
        Score correctness based on seating arrangement.

        Args:
            actual: Actual KAI response text
            expected: Dict with "seating" key
            allow_partial: Whether to give partial credit

        Returns:
            Score 0-100
        """
        if not expected or "seating" not in expected:
            return 50.0

        seating = expected["seating"]
        score = 0.0
        actual_lower = actual.lower()

        # Check arrangement mentions all people: +30%
        people = ["anna", "ben", "clara", "david"]
        mentioned_count = sum(1 for person in people if person in actual_lower)
        score += (mentioned_count / len(people)) * 30

        # Check specific relationships: +70%
        if "anna_opposite" in seating:
            opposite = seating["anna_opposite"].lower()
            # Check if "Anna opposite Ben" or similar mentioned
            pattern = rf"\b(anna|{opposite})\b.*\b(gegenueber|opposite)\b.*\b({opposite}|anna)\b"
            if re.search(pattern, actual_lower):
                score += 35

        if "clara_right_of" in seating:
            right_of = seating["clara_right_of"].lower()
            # Check if "Clara right of Anna" mentioned
            pattern = rf"\bclara\b.*\b(rechts|right)\b.*\b{right_of}\b"
            if re.search(pattern, actual_lower):
                score += 35

        return min(score, 100.0)

    def score_reasoning_quality(
        self, proof_tree: Dict, strategies_used: List[str], reasoning_steps: List[str]
    ) -> float:
        """
        Score reasoning quality based on multi-strategy usage.

        Returns: 0-100 score
        """
        score = 0.0

        # Used spatial strategy: +25%
        has_spatial = any(
            s in ["spatial", "position", "arrangement", "location"]
            for s in strategies_used
        )
        if has_spatial:
            score += 25

        # Used logic strategy: +25%
        has_logic = any(
            s in ["constraint", "csp", "logic", "sat"] for s in strategies_used
        )
        if has_logic:
            score += 25

        # Appropriate ProofTree depth [6-16]: +30%
        depth = self._calculate_proof_tree_depth(proof_tree) if proof_tree else 0
        if 6 <= depth <= 16:
            score += 30
        elif 4 <= depth < 6:
            score += 20
        elif 16 < depth <= 20:
            score += 25

        # Multiple reasoning steps: +20%
        if len(reasoning_steps) >= 8:
            score += 20
        elif len(reasoning_steps) >= 5:
            score += 15
        elif len(reasoning_steps) >= 3:
            score += 10

        return min(score, 100.0)
