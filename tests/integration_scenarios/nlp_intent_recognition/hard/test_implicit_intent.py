"""
tests/integration_scenarios/nlp_intent_recognition/hard/test_implicit_intent.py

Hard-level NLP intent recognition: Infer unstated intent from context

Scenario:
Query doesn't explicitly state what user wants, but intent can be inferred.
Example: "Es regnet draussen." (implicit: I need an umbrella / I can't go out)
KAI should infer practical consequences or implied requests.

Expected Reasoning:
- Intent detection identifies implicit intent
- Inference engine deduces practical implications
- Production System generates contextually appropriate response
- Confidence should be medium (<0.75) due to inference

Success Criteria (Gradual Scoring):
- Correctness: 30% (inferred correct intent)
- Reasoning Quality: 50% (appropriate inference, logical connection)
- Confidence Calibration: 20% (confidence reflects inference uncertainty)
"""

from typing import Dict, List

from tests.integration_scenarios.utils.scenario_base import ScenarioTestBase


class TestImplicitIntent(ScenarioTestBase):
    """Test: Infer unstated intent from context"""

    DIFFICULTY = "hard"
    DOMAIN = "nlp_intent_recognition"
    TIMEOUT_SECONDS = 1800  # 30 minutes

    # Scoring weights for this scenario type
    REASONING_QUALITY_WEIGHT = 0.5
    CONFIDENCE_CALIBRATION_WEIGHT = 0.2
    CORRECTNESS_WEIGHT = 0.3

    def test_implicit_intent_inference(
        self,
        kai_worker_scenario_mode,
        progress_reporter,
        scenario_logger,
        confidence_tracker,
    ):
        """
        Test: Infer implicit intent from statement

        Statement: "Es regnet draussen."
        Implicit intent: User might need umbrella, can't go out, etc.

        Expected: Infer practical consequence or helpful response
        """

        # Setup progress reporter
        progress_reporter.total_steps = 5
        progress_reporter.start()

        # Query with implicit intent
        query = """
Es regnet draussen.
        """

        # Expected: Infer consequence (need umbrella, stay inside, etc.)
        expected_outputs = {
            "inferred_intent": True,
            "practical_consequence": ["regenschirm", "nass", "drinnen", "warten"],
        }

        # Execute scenario
        result = self.run_scenario(
            input_text=query,
            expected_outputs=expected_outputs,
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

        # Check that inference was used
        has_inference = any(
            s in ["inference", "abductive", "intent", "meaning"]
            for s in result.strategies_used
        )
        assert (
            has_inference
        ), f"Expected inference strategy, got: {result.strategies_used}"

        # Log summary
        print(f"\n[INFO] Detailed logs saved to: {scenario_logger.save_logs()}")
        print(f"[INFO] Overall Score: {result.overall_score:.1f}/100")
        print(
            f"[INFO] Breakdown: "
            f"Reasoning={result.reasoning_quality_score:.1f}, "
            f"Confidence={result.confidence_calibration_score:.1f}, "
            f"Correctness={result.correctness_score:.1f}"
        )
        print(f"[INFO] Final confidence: {result.final_confidence:.2f}")

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
        Score correctness based on intent inference quality.

        Args:
            actual: Actual KAI response text
            expected: Dict with inference criteria
            allow_partial: Whether to give partial credit

        Returns:
            Score 0-100
        """
        if not expected:
            return 50.0

        score = 0.0
        actual_lower = actual.lower()

        # Check if practical consequence was inferred: +60%
        if "practical_consequence" in expected:
            consequences = expected["practical_consequence"]
            mentioned_count = sum(
                1 for consequence in consequences if consequence in actual_lower
            )

            if mentioned_count >= 2:
                score += 60
            elif mentioned_count >= 1:
                score += 40

        # Check if response is helpful (not just acknowledging): +40%
        helpful_markers = [
            "solltest",
            "koenntest",
            "empfehle",
            "vielleicht",
            "am besten",
        ]
        if any(marker in actual_lower for marker in helpful_markers):
            score += 40

        return min(score, 100.0)

    def score_reasoning_quality(
        self, proof_tree: Dict, strategies_used: List[str], reasoning_steps: List[str]
    ) -> float:
        """
        Score reasoning quality based on inference usage.

        Returns: 0-100 score
        """
        score = 0.0

        # Used inference: +40%
        has_inference = any(
            s in ["inference", "abductive", "intent", "meaning"]
            for s in strategies_used
        )
        if has_inference:
            score += 40

        # ProofTree shows inference chain: +30%
        depth = self._calculate_proof_tree_depth(proof_tree) if proof_tree else 0
        if depth >= 4:
            score += 30
        elif depth >= 2:
            score += 20
        elif depth >= 1:
            score += 10

        # Multiple reasoning steps: +20%
        if len(reasoning_steps) >= 4:
            score += 20
        elif len(reasoning_steps) >= 2:
            score += 15
        elif len(reasoning_steps) >= 1:
            score += 10

        # Multiple strategies: +10%
        if len(set(strategies_used)) >= 2:
            score += 10

        return min(score, 100.0)
