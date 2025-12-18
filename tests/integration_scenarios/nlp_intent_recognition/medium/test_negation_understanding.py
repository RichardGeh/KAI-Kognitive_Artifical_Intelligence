"""
tests/integration_scenarios/nlp_intent_recognition/medium/test_negation_understanding.py

Medium-level NLP: Negation understanding

Scenario:
Test KAI's ability to understand and handle negations correctly.
"Ein Pinguin ist ein Vogel aber kann nicht fliegen."

Expected Reasoning:
- Negation detection (nicht, kein)
- Contradiction handling
- Correct fact storage (Pinguin IS_A Vogel, NOT capable of flying)

Success Criteria (Gradual Scoring):
- Correctness: 30% (negation correctly handled)
- Reasoning Quality: 50% (negation detection, fact storage)
- Confidence Calibration: 20% (confidence matches correctness)
"""

from typing import Dict, List

from tests.integration_scenarios.utils.scenario_base import ScenarioTestBase


class TestNegationUnderstanding(ScenarioTestBase):
    """Test: Negation understanding and handling"""

    DIFFICULTY = "medium"
    DOMAIN = "nlp_intent_recognition"
    TIMEOUT_SECONDS = 600
    REASONING_QUALITY_WEIGHT = 0.5
    CONFIDENCE_CALIBRATION_WEIGHT = 0.2
    CORRECTNESS_WEIGHT = 0.3

    def test_negation_handling(
        self,
        kai_worker_scenario_mode,
        progress_reporter,
        scenario_logger,
        confidence_tracker,
    ):
        """Test: Understand negations in statements"""
        progress_reporter.total_steps = 5
        progress_reporter.start()

        input_text = "Ein Pinguin ist ein Vogel aber kann nicht fliegen."

        result = self.run_scenario(
            input_text=input_text,
            expected_outputs={"entity": "Pinguin", "negation": "fliegen"},
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
        entity = expected.get("entity", "")
        negation = expected.get("negation", "")
        actual_lower = actual.lower()
        entity_present = entity.lower() in actual_lower
        negation_handled = negation.lower() in actual_lower or "nicht" in actual_lower
        return (50.0 if entity_present else 0.0) + (50.0 if negation_handled else 0.0)

    def score_reasoning_quality(
        self, proof_tree: Dict, strategies_used: List[str], reasoning_steps: List[str]
    ) -> float:
        score = 0.0
        if len(strategies_used) >= 1:
            score += 50
        if len(reasoning_steps) >= 2:
            score += 30
        if proof_tree:
            score += 20
        return min(score, 100.0)
