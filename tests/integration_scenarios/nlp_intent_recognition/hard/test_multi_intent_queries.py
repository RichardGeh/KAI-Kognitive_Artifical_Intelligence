"""
tests/integration_scenarios/nlp_intent_recognition/hard/test_multi_intent_queries.py

Hard-level NLP intent recognition: Queries with multiple intents

Scenario:
Query contains multiple distinct intents that should all be addressed.
Example: "Was ist ein Apfel und wie macht man Apfelkuchen?"
KAI should recognize and address both intents (definition + recipe).

Expected Reasoning:
- Intent detection identifies multiple intents
- Input Orchestrator may segment query
- Multiple sub-goals generated
- Production System generates comprehensive response
- Confidence should be medium-high (>0.70)

Success Criteria (Gradual Scoring):
- Correctness: 30% (all intents addressed)
- Reasoning Quality: 50% (multi-intent recognition, comprehensive response)
- Confidence Calibration: 20% (confidence reflects completeness)
"""

from typing import Dict, List

from tests.integration_scenarios.utils.scenario_base import ScenarioTestBase


class TestMultiIntentQueries(ScenarioTestBase):
    """Test: Handle queries with multiple distinct intents"""

    DIFFICULTY = "hard"
    DOMAIN = "nlp_intent_recognition"
    TIMEOUT_SECONDS = 1800  # 30 minutes

    # Scoring weights for this scenario type
    REASONING_QUALITY_WEIGHT = 0.5
    CONFIDENCE_CALIBRATION_WEIGHT = 0.2
    CORRECTNESS_WEIGHT = 0.3

    def test_multi_intent_query_handling(
        self,
        kai_worker_scenario_mode,
        progress_reporter,
        scenario_logger,
        confidence_tracker,
    ):
        """
        Test: Handle query with multiple intents

        Query: "Was ist ein Apfel und wie macht man Apfelkuchen?"
        Intent 1: Define Apfel (QUERY_DEFINITION)
        Intent 2: Explain Apfelkuchen recipe (QUERY_PROCESS)

        Expected: Both intents addressed in response
        """

        # Setup progress reporter
        progress_reporter.total_steps = 5
        progress_reporter.start()

        # Query with multiple intents
        query = """
Was ist ein Apfel und wie macht man Apfelkuchen?
        """

        # Expected: Both intents addressed
        expected_outputs = {
            "intent_count": 2,
            "intent_1": {"type": "definition", "topic": "apfel"},
            "intent_2": {"type": "process", "topic": "apfelkuchen"},
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
            result.correctness_score >= 30
        ), f"Expected at least 30% correctness, got {result.correctness_score:.1f}%"

        assert (
            result.reasoning_quality_score >= 35
        ), f"Reasoning quality too low: {result.reasoning_quality_score:.1f}%"

        # Check that intent recognition and orchestration were used
        has_multi_intent = any(
            s in ["intent", "orchestrator", "multi", "segment"]
            for s in result.strategies_used
        )
        assert (
            has_multi_intent
        ), f"Expected multi-intent handling, got: {result.strategies_used}"

        # Log summary
        print(f"\n[INFO] Detailed logs saved to: {scenario_logger.save_logs()}")
        print(f"[INFO] Overall Score: {result.overall_score:.1f}/100")
        print(
            f"[INFO] Breakdown: "
            f"Reasoning={result.reasoning_quality_score:.1f}, "
            f"Confidence={result.confidence_calibration_score:.1f}, "
            f"Correctness={result.correctness_score:.1f}"
        )

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
        Score correctness based on how many intents were addressed.

        Args:
            actual: Actual KAI response text
            expected: Dict with intent information
            allow_partial: Whether to give partial credit

        Returns:
            Score 0-100
        """
        if not expected:
            return 50.0

        score = 0.0
        actual_lower = actual.lower()

        intent_count = expected.get("intent_count", 2)
        addressed_count = 0

        # Check intent 1
        if "intent_1" in expected:
            intent1 = expected["intent_1"]
            topic1 = intent1.get("topic", "").lower()
            if topic1 in actual_lower:
                addressed_count += 1

        # Check intent 2
        if "intent_2" in expected:
            intent2 = expected["intent_2"]
            topic2 = intent2.get("topic", "").lower()
            if topic2 in actual_lower:
                addressed_count += 1

        # Score based on how many intents were addressed
        if allow_partial:
            score = (addressed_count / intent_count) * 100.0
        else:
            score = 100.0 if addressed_count == intent_count else 0.0

        return score

    def score_reasoning_quality(
        self, proof_tree: Dict, strategies_used: List[str], reasoning_steps: List[str]
    ) -> float:
        """
        Score reasoning quality based on multi-intent handling.

        Returns: 0-100 score
        """
        score = 0.0

        # Used multi-intent handling: +40%
        has_multi_intent = any(
            s in ["intent", "orchestrator", "multi", "segment"] for s in strategies_used
        )
        if has_multi_intent:
            score += 40

        # ProofTree shows multiple branches/goals: +30%
        depth = self._calculate_proof_tree_depth(proof_tree) if proof_tree else 0
        if depth >= 6:
            score += 30
        elif depth >= 4:
            score += 20
        elif depth >= 2:
            score += 10

        # Multiple reasoning steps: +20%
        if len(reasoning_steps) >= 6:
            score += 20
        elif len(reasoning_steps) >= 4:
            score += 15
        elif len(reasoning_steps) >= 2:
            score += 10

        # Multiple strategies: +10%
        if len(set(strategies_used)) >= 3:
            score += 10
        elif len(set(strategies_used)) >= 2:
            score += 5

        return min(score, 100.0)
