"""
tests/integration_scenarios/combined_scenarios/medium/test_dialogue_with_learning.py

Medium-level combined: Dialogue with learning integration

Scenario:
Learn a fact in one turn, then answer questions about it in subsequent turns.
"Lerne: Ein Hund bellt. Was macht ein Hund?"

Expected Reasoning:
- Fact learning (Hund→bellt)
- Episodic memory tracking
- Answer retrieval from learned facts

Success Criteria (Gradual Scoring):
- Correctness: 30% (correct answer retrieved)
- Reasoning Quality: 50% (learning + memory + retrieval)
- Confidence Calibration: 20% (confidence matches correctness)
"""

from typing import Dict, List

from tests.integration_scenarios.utils.scenario_base import ScenarioTestBase


class TestDialogueWithLearning(ScenarioTestBase):
    """Test: Dialogue combined with active learning"""

    DIFFICULTY = "medium"
    DOMAIN = "combined_scenarios"
    TIMEOUT_SECONDS = 600
    REASONING_QUALITY_WEIGHT = 0.5
    CONFIDENCE_CALIBRATION_WEIGHT = 0.2
    CORRECTNESS_WEIGHT = 0.3

    def test_dialogue_and_learning(
        self,
        kai_worker_scenario_mode,
        progress_reporter,
        scenario_logger,
        confidence_tracker,
    ):
        """Test: Learn then answer about learned fact"""
        progress_reporter.total_steps = 5
        progress_reporter.start()

        input_text = """
Lerne: Ein Hund bellt.
Was macht ein Hund?
        """

        result = self.run_scenario(
            input_text=input_text,
            expected_outputs={"learned": "bellt", "entity": "Hund"},
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
        learned = expected.get("learned", "")
        actual_lower = actual.lower()
        contains_learned = learned.lower() in actual_lower
        return 100.0 if contains_learned else 30.0

    def score_reasoning_quality(
        self, proof_tree: Dict, strategies_used: List[str], reasoning_steps: List[str]
    ) -> float:
        score = 0.0
        if len(strategies_used) >= 2:
            score += 50
        elif len(strategies_used) >= 1:
            score += 30
        if len(reasoning_steps) >= 2:
            score += 30
        if proof_tree:
            score += 20
        return min(score, 100.0)
