"""
tests/integration_scenarios/combined_scenarios/medium/test_logic_with_facts.py

Medium-level combined: Logic reasoning with learned facts

Scenario:
Learn facts, then apply logical reasoning.
"Ein Vogel kann fliegen. Ein Pinguin ist ein Vogel. Kann ein Pinguin fliegen?"

Expected Reasoning:
- Fact learning (Vogel→fliegen, Pinguin IS_A Vogel)
- Transitive reasoning
- Contradiction handling (if taught Pinguin cannot fly)

Success Criteria (Gradual Scoring):
- Correctness: 30% (correct logical conclusion)
- Reasoning Quality: 50% (fact learning + logic)
- Confidence Calibration: 20% (confidence matches correctness)
"""

from typing import Dict, List

from tests.integration_scenarios.utils.scenario_base import ScenarioTestBase


class TestLogicWithFacts(ScenarioTestBase):
    """Test: Logic reasoning combined with fact learning"""

    DIFFICULTY = "medium"
    DOMAIN = "combined_scenarios"
    TIMEOUT_SECONDS = 600
    REASONING_QUALITY_WEIGHT = 0.5
    CONFIDENCE_CALIBRATION_WEIGHT = 0.2
    CORRECTNESS_WEIGHT = 0.3

    def test_logic_reasoning_with_facts(
        self,
        kai_worker_scenario_mode,
        progress_reporter,
        scenario_logger,
        confidence_tracker,
    ):
        """Test: Learn facts then reason logically"""
        progress_reporter.total_steps = 5
        progress_reporter.start()

        input_text = """
Ein Vogel kann fliegen.
Ein Pinguin ist ein Vogel.
Kann ein Pinguin fliegen?
        """

        result = self.run_scenario(
            input_text=input_text,
            expected_outputs={"question": "fliegen", "subject": "Pinguin"},
            kai_worker=kai_worker_scenario_mode,
            logger=scenario_logger,
            progress_reporter=progress_reporter,
            confidence_tracker=confidence_tracker,
        )

        assert result.overall_score >= 50, f"Score: {result.overall_score:.1f}%"
        print(f"\n[INFO] Score: {result.overall_score:.1f}/100")
        assert result.passed

    def score_correctness(
        self, actual: str, expected: Dict, allow_partial: bool = True
    ) -> float:
        if not expected:
            return 50.0
        actual_lower = actual.lower()
        has_answer = len(actual.strip()) > 10
        mentions_subject = expected.get("subject", "").lower() in actual_lower
        return (50.0 if has_answer else 0.0) + (50.0 if mentions_subject else 0.0)

    def score_reasoning_quality(
        self, proof_tree: Dict, strategies_used: List[str], reasoning_steps: List[str]
    ) -> float:
        score = 0.0
        if len(strategies_used) >= 2:
            score += 50
        elif len(strategies_used) >= 1:
            score += 30
        if len(reasoning_steps) >= 3:
            score += 30
        if proof_tree and self._calculate_proof_tree_depth(proof_tree) >= 2:
            score += 20
        return min(score, 100.0)
