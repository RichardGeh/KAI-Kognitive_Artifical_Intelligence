"""
tests/integration_scenarios/nlp_intent_recognition/hard/test_ambiguous_phrasing.py

Hard-level NLP intent recognition: Ambiguous queries with multiple possible intents

Scenario:
Query that could be interpreted multiple ways.
KAI should recognize ambiguity and either:
1. Ask for clarification
2. Address all plausible interpretations
3. Choose most likely interpretation with lowered confidence

Expected Reasoning:
- Intent detection should identify multiple possibilities
- Confidence should reflect uncertainty (<0.70 for ambiguous)
- Production System may generate multiple responses
- ProofTree should show branching for different interpretations

Success Criteria (Gradual Scoring):
- Correctness: 30% (recognized ambiguity or addressed main interpretation)
- Reasoning Quality: 50% (appropriate handling of ambiguity)
- Confidence Calibration: 20% (confidence reflects uncertainty)
"""

from typing import Dict, List

from tests.integration_scenarios.utils.scenario_base import ScenarioTestBase


class TestAmbiguousPhrasing(ScenarioTestBase):
    """Test: Handle ambiguous queries with multiple interpretations"""

    DIFFICULTY = "hard"
    DOMAIN = "nlp_intent_recognition"
    TIMEOUT_SECONDS = 1800  # 30 minutes

    # Scoring weights for this scenario type
    REASONING_QUALITY_WEIGHT = 0.5
    CONFIDENCE_CALIBRATION_WEIGHT = 0.2
    CORRECTNESS_WEIGHT = 0.3

    def test_ambiguous_query_handling(
        self,
        kai_worker_scenario_mode,
        progress_reporter,
        scenario_logger,
        confidence_tracker,
    ):
        """
        Test: Handle ambiguous query

        Query: "Wie schwer ist das?" could mean:
        1. Wie schwierig ist das? (How difficult is this?)
        2. Wie viel wiegt das? (How much does this weigh?)

        Expected: Recognize ambiguity or address plausible interpretation
        """

        # Setup progress reporter
        progress_reporter.total_steps = 5
        progress_reporter.start()

        # Ambiguous query
        query = """
Wie schwer ist das?
        """

        # Expected: Recognition of ambiguity OR addressing one interpretation
        expected_outputs = {
            "recognized_ambiguity": True,
            "possible_interpretations": ["schwierig", "gewicht", "wiegt"],
            "confidence_reflects_ambiguity": True,
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

        # Check that intent recognition was used
        has_intent_recognition = any(
            s in ["intent", "linguistic", "nlp", "meaning"]
            for s in result.strategies_used
        )
        assert (
            has_intent_recognition
        ), f"Expected intent recognition, got: {result.strategies_used}"

        # Check that confidence reflects ambiguity (should be lower)
        assert (
            result.final_confidence < 0.85
        ), f"Confidence too high for ambiguous query: {result.final_confidence}"

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
        Score correctness based on ambiguity handling.

        Args:
            actual: Actual KAI response text
            expected: Dict with ambiguity recognition criteria
            allow_partial: Whether to give partial credit

        Returns:
            Score 0-100
        """
        if not expected:
            return 50.0

        score = 0.0
        actual_lower = actual.lower()

        # Check if ambiguity was recognized: +40%
        if "recognized_ambiguity" in expected and expected["recognized_ambiguity"]:
            ambiguity_markers = [
                "mehrdeutig",
                "unklar",
                "mehrere bedeutungen",
                "verschiedene",
            ]
            if any(marker in actual_lower for marker in ambiguity_markers):
                score += 40

        # Check if interpretations were addressed: +40%
        if "possible_interpretations" in expected:
            interpretations = expected["possible_interpretations"]
            mentioned_count = sum(
                1
                for interpretation in interpretations
                if interpretation in actual_lower
            )

            if mentioned_count >= 2:
                score += 40
            elif mentioned_count >= 1:
                score += 25

        # Base score for having a response: +20%
        if len(actual) > 20:
            score += 20

        return min(score, 100.0)

    def score_reasoning_quality(
        self, proof_tree: Dict, strategies_used: List[str], reasoning_steps: List[str]
    ) -> float:
        """
        Score reasoning quality based on intent recognition usage.

        Returns: 0-100 score
        """
        score = 0.0

        # Used intent recognition: +40%
        has_intent = any(
            s in ["intent", "linguistic", "nlp", "meaning"] for s in strategies_used
        )
        if has_intent:
            score += 40

        # ProofTree shows branching/alternatives: +30%
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

        # Multiple strategies (exploring alternatives): +10%
        if len(set(strategies_used)) >= 2:
            score += 10

        return min(score, 100.0)
