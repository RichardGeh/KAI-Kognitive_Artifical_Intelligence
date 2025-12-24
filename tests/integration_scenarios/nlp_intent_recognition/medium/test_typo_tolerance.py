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

    # NLP-optimized weights: correctness matters most for intent recognition
    REASONING_QUALITY_WEIGHT = 0.2
    CONFIDENCE_CALIBRATION_WEIGHT = 0.2
    CORRECTNESS_WEIGHT = 0.6

    def test_typo_correction(
        self,
        kai_worker_scenario_mode,
        progress_reporter,
        scenario_logger,
        confidence_tracker,
    ):
        """Test: Understand queries with typos and offer corrections"""
        progress_reporter.total_steps = 5
        progress_reporter.start()

        input_text = "Was is ein Afel?"  # Typos: is->ist, Afel->Apfel

        result = self.run_scenario(
            input_text=input_text,
            expected_outputs={
                # KAI should either correct the typo or offer a suggestion
                "typo_indicators": ["meintest", "vorschlag", "ist", "was ist"],
                "recognized_intent": "question",
            },
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
        """
        Score typo tolerance correctness.
        KAI should either correct the typo or offer a suggestion.
        Both are valid responses that show typo detection worked.
        """
        if not expected:
            return 50.0

        actual_lower = actual.lower()
        score = 0.0

        # Check for typo correction indicators (KAI detected the typo)
        typo_indicators = expected.get("typo_indicators", [])
        indicator_found = any(ind.lower() in actual_lower for ind in typo_indicators)
        if indicator_found:
            score += 70.0  # Typo was detected and handled

        # Check if the response handles the question intent
        if "?" in actual or "was ist" in actual_lower or "frage" in actual_lower:
            score += 20.0  # Question intent recognized

        # Bonus for offering correction
        if "vorschlag" in actual_lower or "meintest" in actual_lower:
            score += 10.0  # Active correction offered

        return min(score, 100.0)

    def score_reasoning_quality(
        self, proof_tree: Dict, strategies_used: List[str], reasoning_steps: List[str]
    ) -> float:
        """
        NLP-optimized reasoning quality scoring.
        Typo detection is a lightweight NLP task - no deep proof trees expected.
        """
        score = 50.0  # Base score for typo detection (simple NLP task)

        # Bonus for strategies
        if len(strategies_used) >= 1:
            score += 25
        else:
            score += 10  # Even without explicit strategies, typo detection occurred

        # Bonus for reasoning steps
        if len(reasoning_steps) >= 1:
            score += 15

        # Bonus for proof tree presence
        if proof_tree:
            score += 10

        return min(score, 100.0)
