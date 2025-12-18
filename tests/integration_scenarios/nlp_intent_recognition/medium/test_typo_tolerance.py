"""
tests/integration_scenarios/nlp_intent_recognition/medium/test_typo_tolerance.py

Medium-level NLP: Typo tolerance and normalization

Scenario:
Test KAI's ability to handle typos and minor spelling errors.
"Was is ein Afel?" should be understood as "Was ist ein Apfel?"

Expected Reasoning:
- Typo detection (is→ist, Afel→Apfel)
- Normalization and correction
- Correct query processing despite typos

Success Criteria (Gradual Scoring):
- Correctness: 30% (query understood despite typos)
- Reasoning Quality: 50% (typo tolerance, normalization)
- Confidence Calibration: 20% (confidence matches correctness)
"""

from typing import Dict, List

from tests.integration_scenarios.utils.scenario_base import ScenarioTestBase


class TestTypoTolerance(ScenarioTestBase):
    """Test: Typo tolerance and error correction"""

    DIFFICULTY = "medium"
    DOMAIN = "nlp_intent_recognition"
    TIMEOUT_SECONDS = 600
    REASONING_QUALITY_WEIGHT = 0.5
    CONFIDENCE_CALIBRATION_WEIGHT = 0.2
    CORRECTNESS_WEIGHT = 0.3

    def test_typo_correction(
        self,
        kai_worker_scenario_mode,
        progress_reporter,
        scenario_logger,
        confidence_tracker,
    ):
        """Test: Understand queries with typos"""
        progress_reporter.total_steps = 5
        progress_reporter.start()

        input_text = "Was is ein Afel?"  # Typos: is→ist, Afel→Apfel

        result = self.run_scenario(
            input_text=input_text,
            expected_outputs={"concept": "Apfel", "query_type": "QUESTION"},
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
        concept = expected.get("concept", "")
        actual_lower = actual.lower()
        concept_understood = concept.lower() in actual_lower or "frucht" in actual_lower
        return 100.0 if concept_understood else 30.0

    def score_reasoning_quality(
        self, proof_tree: Dict, strategies_used: List[str], reasoning_steps: List[str]
    ) -> float:
        score = 0.0
        if len(strategies_used) >= 1:
            score += 50
        if len(reasoning_steps) >= 1:
            score += 30
        if proof_tree:
            score += 20
        return min(score, 100.0)
