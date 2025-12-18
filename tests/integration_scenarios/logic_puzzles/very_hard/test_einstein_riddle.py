"""
tests/integration_scenarios/logic_puzzles/very_hard/test_einstein_riddle.py

Einstein's Riddle (Zebra Puzzle) - Full version with 20+ constraints.
Classic logic puzzle requiring systematic constraint satisfaction with backtracking.

Expected Reasoning: constraint, sat, exhaustive_search, backtracking, pruning
Success Criteria:
- Reasoning Quality >= 30% (depth 15-30 expected)
- Confidence Calibration >= 30%
- Correctness >= 30% (partial credit for some assignments)
- Overall >= 30%
"""

import re
from typing import Dict

from tests.integration_scenarios.utils.scenario_base import ScenarioTestBase


class TestEinsteinRiddle(ScenarioTestBase):
    """Test: Einstein's Riddle (Zebra Puzzle) - Very Hard Logic Puzzle"""

    # REQUIRED: Class attributes
    DIFFICULTY = "very_hard"
    DOMAIN = "logic_puzzles"
    TIMEOUT_SECONDS = 3600  # 1 hour

    # REQUIRED: Scoring weights
    REASONING_QUALITY_WEIGHT = 0.5
    CONFIDENCE_CALIBRATION_WEIGHT = 0.2
    CORRECTNESS_WEIGHT = 0.3

    def test_einstein_riddle(
        self,
        kai_worker_scenario_mode,
        progress_reporter,
        scenario_logger,
        confidence_tracker,
    ):
        """
        Test: Einstein's Riddle (Zebra Puzzle) with 15 constraints.

        Classic logic puzzle with 5 houses, 5 colors, 5 nationalities,
        5 beverages, 5 pets, 5 cigarette brands.

        Question: Who owns the zebra? Who drinks water?
        """

        # Setup progress tracking
        progress_reporter.total_steps = 5
        progress_reporter.start()

        # Define test input (German language, ASCII only)
        input_text = """
        Es gibt fuenf Haeuser in fuenf verschiedenen Farben.
        In jedem Haus wohnt eine Person einer anderen Nationalitaet.
        Jeder Hausbewohner bevorzugt ein bestimmtes Getraenk, raucht eine
        bestimmte Zigarettenmarke und haelt ein bestimmtes Haustier.

        Hinweise:
        1. Der Brite lebt im roten Haus.
        2. Der Schwede haelt einen Hund.
        3. Der Daene trinkt gerne Tee.
        4. Das gruene Haus steht links vom weissen Haus.
        5. Der Besitzer des gruenen Hauses trinkt Kaffee.
        6. Die Person, die Pall Mall raucht, haelt einen Vogel.
        7. Der Besitzer des gelben Hauses raucht Dunhill.
        8. Die Person im mittleren Haus trinkt Milch.
        9. Der Norweger wohnt im ersten Haus.
        10. Die Person, die Blend raucht, wohnt neben der Person mit der Katze.
        11. Die Person mit dem Pferd wohnt neben der Person, die Dunhill raucht.
        12. Die Person, die Blue Master raucht, trinkt gerne Bier.
        13. Der Deutsche raucht Prince.
        14. Der Norweger wohnt neben dem blauen Haus.
        15. Die Person, die Blend raucht, hat einen Nachbarn, der Wasser trinkt.

        Frage: Wer besitzt das Zebra und wer trinkt Wasser?
        """

        # Define expected outputs
        # Known solution: German owns zebra, Norwegian drinks water
        expected_outputs = {
            "zebra_owner": "Deutsche",  # German
            "water_drinker": "Norweger",  # Norwegian
            "assignments": {
                "house_1": {"color": "gelb", "nationality": "Norweger"},
                "house_2": {"color": "blau"},
                "house_3": {"beverage": "Milch"},
            },
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
        assert (
            result.overall_score >= 30
        ), f"Overall score too low: {result.overall_score:.1f}% (expected >= 30%)"

        assert (
            result.correctness_score >= 20
        ), f"Correctness too low: {result.correctness_score:.1f}%"

        assert (
            result.reasoning_quality_score >= 30
        ), f"Reasoning quality too low: {result.reasoning_quality_score:.1f}%"

        # Domain-specific assertions - check for constraint/SAT strategies
        constraint_strategies = [
            "constraint",
            "sat",
            "csp",
            "backtracking",
            "pruning",
            "exhaustive",
            "elimination",
        ]
        found_strategy = any(
            any(cs in s.lower() for cs in constraint_strategies)
            for s in result.strategies_used
        )

        assert (
            found_strategy
        ), f"Expected constraint/SAT strategy, got: {result.strategies_used}"

        # Check proof tree depth (very hard puzzle should have deep tree)
        assert (
            8 <= result.proof_tree_depth <= 30
        ), f"ProofTree depth {result.proof_tree_depth} outside expected range 8-30"

        # Performance assertion (1 hour timeout)
        assert (
            result.execution_time_ms < 3600000
        ), f"Too slow: {result.execution_time_ms}ms (expected <1 hour)"

        # Logging
        log_file = scenario_logger.save_logs()
        print(f"\n[INFO] Detailed logs saved to: {log_file}")
        print(f"[INFO] Overall Score: {result.overall_score:.1f}/100")
        print(
            f"[INFO] Breakdown: Reasoning={result.reasoning_quality_score:.1f}, "
            f"Confidence={result.confidence_calibration_score:.1f}, "
            f"Correctness={result.correctness_score:.1f}"
        )

        if result.overall_score < 50:
            print("[WEAKNESS] Identified issues:")
            for weakness in result.identified_weaknesses:
                print(f"  - {weakness}")

        if result.improvement_suggestions:
            print("[SUGGESTION] Improvements:")
            for suggestion in result.improvement_suggestions:
                print(f"  - {suggestion}")

        # Final check
        assert result.passed, f"Test failed: {result.error or 'Score below threshold'}"

    def score_correctness(
        self, actual: str, expected: Dict, allow_partial: bool = True
    ) -> float:
        """
        Custom correctness scoring for Einstein's Riddle.

        Partial credit for:
        - Identifying zebra owner (50 points)
        - Identifying water drinker (50 points)
        """
        score = 0.0

        # Check for zebra owner
        zebra_owner = expected.get("zebra_owner", "")
        if zebra_owner:
            # Negation-aware pattern
            pattern = rf"\b{zebra_owner}\b.*\bZebra\b|Zebra\b.*\b{zebra_owner}\b"
            if re.search(pattern, actual, re.IGNORECASE):
                # Check no negation
                negation_pattern = (
                    rf"\b(?:nicht|kein|keine)\b.*\b{zebra_owner}\b.*\bZebra\b"
                )
                if not re.search(negation_pattern, actual, re.IGNORECASE):
                    score += 50.0

        # Check for water drinker
        water_drinker = expected.get("water_drinker", "")
        if water_drinker:
            # Negation-aware pattern
            pattern = rf"\b{water_drinker}\b.*\bWasser\b|Wasser\b.*\b{water_drinker}\b"
            if re.search(pattern, actual, re.IGNORECASE):
                # Check no negation
                negation_pattern = (
                    rf"\b(?:nicht|kein|keine)\b.*\b{water_drinker}\b.*\bWasser\b"
                )
                if not re.search(negation_pattern, actual, re.IGNORECASE):
                    score += 50.0

        return score
