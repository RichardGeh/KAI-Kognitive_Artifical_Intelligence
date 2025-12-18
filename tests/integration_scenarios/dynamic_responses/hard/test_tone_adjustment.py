"""
tests/integration_scenarios/dynamic_responses/hard/test_tone_adjustment.py

Hard-level dynamic response: Adjust tone (formal vs. casual) based on context

Scenario:
KAI should adapt response tone based on context cues in the query.
Formal context requires formal language (Sie, korrekte Grammatik).
Casual context allows casual language (du, Umgangssprache).

Expected Reasoning:
- Production System should recognize tone context
- Production rules for formal/casual language should fire
- Response should match the tone of the query
- Confidence should be high (>0.75) for clear tone context

Success Criteria (Gradual Scoring):
- Correctness: 30% (tone matches context)
- Reasoning Quality: 50% (Production System used, appropriate rules applied)
- Confidence Calibration: 20% (confidence matches appropriateness)
"""

import re
from typing import Dict, List

from tests.integration_scenarios.utils.scenario_base import ScenarioTestBase


class TestToneAdjustment(ScenarioTestBase):
    """Test: Tone adjustment in responses (formal vs. casual)"""

    DIFFICULTY = "hard"
    DOMAIN = "dynamic_responses"
    TIMEOUT_SECONDS = 1800  # 30 minutes

    # Scoring weights for this scenario type
    REASONING_QUALITY_WEIGHT = 0.5
    CONFIDENCE_CALIBRATION_WEIGHT = 0.2
    CORRECTNESS_WEIGHT = 0.3

    def test_tone_adjustment_formal_vs_casual(
        self,
        kai_worker_scenario_mode,
        progress_reporter,
        scenario_logger,
        confidence_tracker,
    ):
        """
        Test: Adjust response tone based on query tone

        Formal query should get formal response.
        Casual query should get casual response.

        Expected: Tone-appropriate responses
        """

        # Setup progress reporter
        progress_reporter.total_steps = 5
        progress_reporter.start()

        # Test 1: Formal query
        formal_query = """
Sehr geehrter KAI,
ich moechte Sie bitten, mir die Hauptstadt von Deutschland mitzuteilen.
Mit freundlichen Gruessen
        """

        # Test 2: Casual query
        casual_query = """
Hey KAI, was ist die Hauptstadt von Deutschland?
        """

        # Expected: Formal response uses "Sie", casual uses "du" or neutral
        expected_formal_tone = {
            "uses_sie": True,
            "formal_greeting": True,
            "no_slang": True,
        }

        expected_casual_tone = {
            "uses_sie": False,
            "casual_language": True,
        }

        # Execute formal query
        result_formal = self.run_scenario(
            input_text=formal_query,
            expected_outputs={"tone": expected_formal_tone, "answer": "Berlin"},
            context={},
            kai_worker=kai_worker_scenario_mode,
            logger=scenario_logger,
            progress_reporter=progress_reporter,
            confidence_tracker=confidence_tracker,
        )

        # Execute casual query
        result_casual = self.run_scenario(
            input_text=casual_query,
            expected_outputs={"tone": expected_casual_tone, "answer": "Berlin"},
            context={},
            kai_worker=kai_worker_scenario_mode,
            logger=scenario_logger,
            progress_reporter=progress_reporter,
            confidence_tracker=confidence_tracker,
        )

        # Assertions - use average of both results
        avg_score = (result_formal.overall_score + result_casual.overall_score) / 2
        assert (
            avg_score >= 40
        ), f"Overall score too low: {avg_score:.1f}% (expected >= 40%)"

        # Check that Production System was used
        has_production_system = any(
            "production" in s.lower()
            for s in result_formal.strategies_used + result_casual.strategies_used
        )
        assert (
            has_production_system
        ), f"Expected Production System, got: {result_formal.strategies_used}"

        # Log summary
        print(f"\n[INFO] Formal query score: {result_formal.overall_score:.1f}/100")
        print(f"[INFO] Casual query score: {result_casual.overall_score:.1f}/100")
        print(f"[INFO] Average score: {avg_score:.1f}/100")

        # Mark test as passed
        assert result_formal.passed or result_casual.passed, "Both queries failed"

    def score_correctness(
        self, actual: str, expected: Dict, allow_partial: bool = True
    ) -> float:
        """
        Score correctness based on tone appropriateness.

        Args:
            actual: Actual KAI response text
            expected: Dict with "tone" and "answer" keys
            allow_partial: Whether to give partial credit

        Returns:
            Score 0-100
        """
        if not expected:
            return 50.0

        score = 0.0
        actual_lower = actual.lower()

        # Check answer correctness: +50%
        if "answer" in expected:
            answer = expected["answer"].lower()
            if answer in actual_lower:
                score += 50

        # Check tone: +50%
        if "tone" in expected:
            tone = expected["tone"]
            tone_score = 0

            if "uses_sie" in tone:
                if tone["uses_sie"]:
                    # Should use "Sie"
                    if re.search(r"\bsie\b", actual_lower):
                        tone_score += 20
                else:
                    # Should NOT use "Sie" (or use "du")
                    if not re.search(r"\bsie\b", actual_lower) or re.search(
                        r"\bdu\b", actual_lower
                    ):
                        tone_score += 20

            if "formal_greeting" in tone and tone["formal_greeting"]:
                # Check for formal greeting
                formal_greetings = ["guten tag", "sehr geehrte", "mit freundlichen"]
                if any(greeting in actual_lower for greeting in formal_greetings):
                    tone_score += 15

            if "no_slang" in tone and tone["no_slang"]:
                # Check no slang words
                slang_words = ["cool", "krass", "geil", "ey"]
                if not any(slang in actual_lower for slang in slang_words):
                    tone_score += 15

            if "casual_language" in tone and tone["casual_language"]:
                # Check for casual markers
                casual_markers = ["hey", "hallo", "klar", "einfach"]
                if any(marker in actual_lower for marker in casual_markers):
                    tone_score += 15

            score += tone_score

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
