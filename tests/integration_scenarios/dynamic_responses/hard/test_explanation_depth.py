"""
tests/integration_scenarios/dynamic_responses/hard/test_explanation_depth.py

Hard-level dynamic response: Adaptive explanation depth (brief vs. detailed)

Scenario:
KAI should adjust explanation depth based on explicit or implicit user requests.
"Kurz" / "knapp" -> brief response
"Ausfuehrlich" / "detailliert" -> detailed response
Default -> medium depth

Expected Reasoning:
- Production System should recognize depth cues
- Production rules for elaboration should fire appropriately
- Response length and detail should match request
- Confidence should be high (>0.75) for clear depth cues

Success Criteria (Gradual Scoring):
- Correctness: 30% (depth matches request)
- Reasoning Quality: 50% (Production System used, appropriate elaboration)
- Confidence Calibration: 20% (confidence matches appropriateness)
"""

import re
from typing import Dict, List

from tests.integration_scenarios.utils.scenario_base import ScenarioTestBase


class TestExplanationDepth(ScenarioTestBase):
    """Test: Adaptive explanation depth in responses"""

    DIFFICULTY = "hard"
    DOMAIN = "dynamic_responses"
    TIMEOUT_SECONDS = 1800  # 30 minutes

    # Scoring weights for this scenario type
    REASONING_QUALITY_WEIGHT = 0.5
    CONFIDENCE_CALIBRATION_WEIGHT = 0.2
    CORRECTNESS_WEIGHT = 0.3

    def test_explanation_depth_adjustment(
        self,
        kai_worker_scenario_mode,
        progress_reporter,
        scenario_logger,
        confidence_tracker,
    ):
        """
        Test: Adjust explanation depth based on user request

        Brief request should get concise response.
        Detailed request should get elaborate response.

        Expected: Depth-appropriate responses
        """

        # Setup progress reporter
        progress_reporter.total_steps = 5
        progress_reporter.start()

        # Test 1: Brief request
        brief_query = """
Was ist Photosynthese? (Bitte kurz und knapp.)
        """

        # Test 2: Detailed request
        detailed_query = """
Was ist Photosynthese? Erklaere es mir bitte ausfuehrlich und detailliert.
        """

        # Expected: Brief response < 50 words, detailed response > 100 words
        expected_brief = {
            "max_words": 50,
            "contains_key_concept": "Licht",
        }

        expected_detailed = {
            "min_words": 100,
            "contains_key_concept": "Licht",
            "mentions_process_steps": True,
        }

        # Execute brief query
        result_brief = self.run_scenario(
            input_text=brief_query,
            expected_outputs={"depth": expected_brief},
            context={},
            kai_worker=kai_worker_scenario_mode,
            logger=scenario_logger,
            progress_reporter=progress_reporter,
            confidence_tracker=confidence_tracker,
        )

        # Execute detailed query
        result_detailed = self.run_scenario(
            input_text=detailed_query,
            expected_outputs={"depth": expected_detailed},
            context={},
            kai_worker=kai_worker_scenario_mode,
            logger=scenario_logger,
            progress_reporter=progress_reporter,
            confidence_tracker=confidence_tracker,
        )

        # Assertions - use average of both results
        avg_score = (result_brief.overall_score + result_detailed.overall_score) / 2
        assert (
            avg_score >= 40
        ), f"Overall score too low: {avg_score:.1f}% (expected >= 40%)"

        # Check that Production System was used
        has_production_system = any(
            "production" in s.lower()
            for s in result_brief.strategies_used + result_detailed.strategies_used
        )
        assert (
            has_production_system
        ), f"Expected Production System, got: {result_brief.strategies_used}"

        # Log summary
        print(f"\n[INFO] Brief query score: {result_brief.overall_score:.1f}/100")
        print(f"[INFO] Detailed query score: {result_detailed.overall_score:.1f}/100")
        print(f"[INFO] Average score: {avg_score:.1f}/100")

        # Mark test as passed
        assert result_brief.passed or result_detailed.passed, "Both queries failed"

    def score_correctness(
        self, actual: str, expected: Dict, allow_partial: bool = True
    ) -> float:
        """
        Score correctness based on depth appropriateness.

        Args:
            actual: Actual KAI response text
            expected: Dict with "depth" key
            allow_partial: Whether to give partial credit

        Returns:
            Score 0-100
        """
        if not expected or "depth" not in expected:
            return 50.0

        score = 0.0
        depth = expected["depth"]

        # Count words in response
        word_count = len(actual.split())

        # Check length constraints: +40%
        if "max_words" in depth:
            max_words = depth["max_words"]
            if word_count <= max_words:
                score += 40
            elif word_count <= max_words * 1.5:
                score += 20  # Slightly over, partial credit

        if "min_words" in depth:
            min_words = depth["min_words"]
            if word_count >= min_words:
                score += 40
            elif word_count >= min_words * 0.7:
                score += 20  # Slightly under, partial credit

        # Check key concept mentioned: +30%
        if "contains_key_concept" in depth:
            key_concept = depth["contains_key_concept"].lower()
            if key_concept in actual.lower():
                score += 30

        # Check process steps (for detailed): +30%
        if "mentions_process_steps" in depth and depth["mentions_process_steps"]:
            # Look for numbered steps or sequential markers
            has_steps = bool(
                re.search(
                    r"\b(1\.|2\.|3\.|erstens|zweitens|zuerst|dann|danach)\b",
                    actual.lower(),
                )
            )
            if has_steps:
                score += 30

        return min(score, 100.0)

    def score_reasoning_quality(
        self, proof_tree: Dict, strategies_used: List[str], reasoning_steps: List[str]
    ) -> float:
        """
        Score reasoning quality based on Production System usage.

        Returns: 0-100 score
        """
        score = 0.0

        # Used Production System: +50%
        has_production = any("production" in s.lower() for s in strategies_used)
        if has_production:
            score += 50

        # ProofTree shows rule application: +30%
        depth = self._calculate_proof_tree_depth(proof_tree) if proof_tree else 0
        if depth >= 3:
            score += 30
        elif depth >= 1:
            score += 15

        # Multiple reasoning steps: +20%
        if len(reasoning_steps) >= 3:
            score += 20
        elif len(reasoning_steps) >= 1:
            score += 10

        return min(score, 100.0)
