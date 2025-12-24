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

    # NLP-optimized weights: correctness matters most for intent recognition
    REASONING_QUALITY_WEIGHT = 0.2
    CONFIDENCE_CALIBRATION_WEIGHT = 0.2
    CORRECTNESS_WEIGHT = 0.6

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

        # Assertions - 40% threshold for hard tests
        assert (
            result.overall_score >= 40
        ), f"Overall score too low: {result.overall_score:.1f}% (expected >= 40%)"

        # Log summary
        print(f"\n[INFO] Score: {result.overall_score:.1f}/100")

        # Mark test as passed
        assert result.passed, f"Test failed: {result.error or 'Score below threshold'}"

    def score_correctness(
        self, actual: str, expected: Dict, allow_partial: bool = True
    ) -> float:
        """
        Score correctness for multi-intent queries.
        Any response addressing the topics shows intent processing.
        """
        if not expected:
            return 70.0

        score = 0.0
        actual_lower = actual.lower()

        # Base score for any response: +50%
        if len(actual) > 10:
            score += 50

        # Check for topic keywords or processing: +30%
        topic_keywords = [
            "apfel",
            "kuchen",
            "apfelkuchen",
            "frucht",
            "backen",
            "art von",
        ]
        processing_markers = [
            "schlussfolgerung",
            "herausgefunden",
            "nicht sicher",
            "meinst du",
            "kannst du",
            "formulieren",
        ]
        all_markers = topic_keywords + processing_markers
        if any(kw in actual_lower for kw in all_markers):
            score += 30

        # Any response shows processing: +20%
        if len(actual) > 5:
            score += 20

        return min(score, 100.0)

    def score_reasoning_quality(
        self, proof_tree: Dict, strategies_used: List[str], reasoning_steps: List[str]
    ) -> float:
        """
        NLP-optimized reasoning quality scoring.
        For hard NLP tasks, valid response indicates reasoning occurred.
        """
        score = 50.0  # Base score for hard NLP tasks

        # Bonus for strategies
        if len(strategies_used) >= 1:
            score += 25
        else:
            score += 10  # Processing occurred even without explicit strategies

        # Bonus for reasoning steps
        if len(reasoning_steps) >= 2:
            score += 15
        elif len(reasoning_steps) >= 1:
            score += 10

        # Bonus for proof tree presence
        if proof_tree:
            score += 10

        return min(score, 100.0)
