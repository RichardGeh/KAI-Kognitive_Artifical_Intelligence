"""
tests/integration_scenarios/nlp_intent_recognition/hard/test_context_dependent_meaning.py

Hard-level NLP intent recognition: Words with context-dependent meanings (polysemy)

Scenario:
Query uses words with multiple meanings that depend on context.
Example: "Bank" (financial institution vs. bench), "Schloss" (lock vs. castle)
KAI should infer correct meaning from context.

Expected Reasoning:
- Intent detection identifies polysemous words
- Context analysis determines correct meaning
- Graph query retrieves contextually appropriate information
- Confidence should be high (>0.75) with clear context

Success Criteria (Gradual Scoring):
- Correctness: 30% (correct meaning identified)
- Reasoning Quality: 50% (context analysis, appropriate disambiguation)
- Confidence Calibration: 20% (confidence reflects context clarity)
"""

from typing import Dict, List

from tests.integration_scenarios.utils.scenario_base import ScenarioTestBase


class TestContextDependentMeaning(ScenarioTestBase):
    """Test: Disambiguate polysemous words using context"""

    DIFFICULTY = "hard"
    DOMAIN = "nlp_intent_recognition"
    TIMEOUT_SECONDS = 1800  # 30 minutes

    # Scoring weights for this scenario type
    REASONING_QUALITY_WEIGHT = 0.5
    CONFIDENCE_CALIBRATION_WEIGHT = 0.2
    CORRECTNESS_WEIGHT = 0.3

    def test_context_dependent_word_meaning(
        self,
        kai_worker_scenario_mode,
        progress_reporter,
        scenario_logger,
        confidence_tracker,
    ):
        """
        Test: Disambiguate polysemous words

        Word: "Bank" in different contexts
        Context 1: Financial (Bank = financial institution)
        Context 2: Furniture (Bank = bench)

        Expected: Correct meaning based on context
        """

        # Setup progress reporter
        progress_reporter.total_steps = 5
        progress_reporter.start()

        # Query 1: Bank = financial institution
        query1 = """
Ich muss Geld von der Bank abheben. Was ist eine Bank?
        """

        # Query 2: Bank = bench
        query2 = """
Ich setze mich auf die Bank im Park. Was ist eine Bank?
        """

        # Expected meanings
        expected_financial = {
            "meaning": "finanzinstitut",
            "context_cues": ["geld", "abheben"],
        }

        expected_furniture = {
            "meaning": "sitzbank",
            "context_cues": ["sitzen", "park"],
        }

        # Execute query 1
        result1 = self.run_scenario(
            input_text=query1,
            expected_outputs={"meaning": expected_financial},
            context={},
            kai_worker=kai_worker_scenario_mode,
            logger=scenario_logger,
            progress_reporter=progress_reporter,
            confidence_tracker=confidence_tracker,
        )

        # Execute query 2
        result2 = self.run_scenario(
            input_text=query2,
            expected_outputs={"meaning": expected_furniture},
            context={},
            kai_worker=kai_worker_scenario_mode,
            logger=scenario_logger,
            progress_reporter=progress_reporter,
            confidence_tracker=confidence_tracker,
        )

        # Assertions - use average of both results
        avg_score = (result1.overall_score + result2.overall_score) / 2
        assert (
            avg_score >= 40
        ), f"Overall score too low: {avg_score:.1f}% (expected >= 40%)"

        # Check that context analysis was used
        has_context_analysis = any(
            s in ["context", "intent", "meaning", "disambiguation"]
            for s in result1.strategies_used + result2.strategies_used
        )
        assert (
            has_context_analysis
        ), f"Expected context analysis, got: {result1.strategies_used}"

        # Log summary
        print(f"\n[INFO] Query 1 (financial) score: {result1.overall_score:.1f}/100")
        print(f"[INFO] Query 2 (furniture) score: {result2.overall_score:.1f}/100")
        print(f"[INFO] Average score: {avg_score:.1f}/100")

        # Mark test as passed if at least one query passed
        assert result1.passed or result2.passed, "Both queries failed"

    def score_correctness(
        self, actual: str, expected: Dict, allow_partial: bool = True
    ) -> float:
        """
        Score correctness based on meaning disambiguation.

        Args:
            actual: Actual KAI response text
            expected: Dict with "meaning" key
            allow_partial: Whether to give partial credit

        Returns:
            Score 0-100
        """
        if not expected or "meaning" not in expected:
            return 50.0

        score = 0.0
        actual_lower = actual.lower()
        meaning_config = expected["meaning"]

        # Check if correct meaning is mentioned: +60%
        if meaning_config["meaning"] in actual_lower:
            score += 60

        # Check if context cues are referenced: +40%
        if "context_cues" in meaning_config:
            cues = meaning_config["context_cues"]
            mentioned_cues = sum(1 for cue in cues if cue in actual_lower)

            if mentioned_cues >= 2:
                score += 40
            elif mentioned_cues >= 1:
                score += 20

        return min(score, 100.0)

    def score_reasoning_quality(
        self, proof_tree: Dict, strategies_used: List[str], reasoning_steps: List[str]
    ) -> float:
        """
        Score reasoning quality based on disambiguation process.

        Returns: 0-100 score
        """
        score = 0.0

        # Used context analysis: +40%
        has_context = any(
            s in ["context", "intent", "meaning", "disambiguation"]
            for s in strategies_used
        )
        if has_context:
            score += 40

        # ProofTree shows disambiguation steps: +30%
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
