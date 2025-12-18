"""
tests/integration_scenarios/dynamic_responses/medium/test_basic_elaboration.py

Medium-level dynamic response: Basic elaboration with additional details

Scenario:
Test KAI's ability to add relevant details when asked to elaborate.
Should include basic facts plus 2-3 additional properties.

Expected Reasoning:
- Retrieves basic fact (IS_A Frucht)
- Adds related properties (color, taste, origin)
- Uses Production System elaboration rules
- Response length appropriate (>50 words)

Success Criteria (Gradual Scoring):
- Correctness: 30% (basic fact + additional details)
- Reasoning Quality: 50% (appropriate detail level, coherence)
- Confidence Calibration: 20% (confidence matches correctness)
"""

from typing import Dict, List

from tests.integration_scenarios.utils.scenario_base import ScenarioTestBase


class TestBasicElaboration(ScenarioTestBase):
    """Test: Elaborate on concept with additional details"""

    DIFFICULTY = "medium"
    DOMAIN = "dynamic_responses"
    TIMEOUT_SECONDS = 600

    REASONING_QUALITY_WEIGHT = 0.5
    CONFIDENCE_CALIBRATION_WEIGHT = 0.2
    CORRECTNESS_WEIGHT = 0.3

    def test_elaborate_on_apple(
        self,
        kai_worker_scenario_mode,
        progress_reporter,
        scenario_logger,
        confidence_tracker,
    ):
        """Test: Elaborate on what an apple is"""

        progress_reporter.total_steps = 5
        progress_reporter.start()

        input_text = "Was ist ein Apfel? Erklaere ausfuehrlich."

        # Expected components
        basic_fact = "Frucht"
        additional_facts = ["rot", "gruen", "gelb", "suess", "Baum", "Herbst"]

        # Execute scenario
        result = self.run_scenario(
            input_text=input_text,
            expected_outputs={
                "basic_fact": basic_fact,
                "additional_facts": additional_facts,
            },
            context={},
            kai_worker=kai_worker_scenario_mode,
            logger=scenario_logger,
            progress_reporter=progress_reporter,
            confidence_tracker=confidence_tracker,
        )

        # Assertions
        assert (
            result.overall_score >= 50
        ), f"Overall score too low: {result.overall_score:.1f}% (expected >= 50%)"

        assert (
            result.correctness_score >= 40
        ), f"Correctness too low: {result.correctness_score:.1f}%"

        # Log summary
        print(f"\n[INFO] Logs: {scenario_logger.save_logs()}")
        print(f"[INFO] Score: {result.overall_score:.1f}/100")

        assert result.passed, f"Test failed: {result.error or 'Score below threshold'}"

    def score_correctness(
        self, actual: str, expected: Dict, allow_partial: bool = True
    ) -> float:
        """Score correctness based on fact inclusion and detail level."""
        if not expected:
            return 50.0

        basic_fact = expected.get("basic_fact", "")
        additional_facts = expected.get("additional_facts", [])

        actual_lower = actual.lower()

        # Basic fact present (40%)
        basic_score = 40.0 if basic_fact.lower() in actual_lower else 0.0

        # Additional facts (40%)
        facts_found = sum(
            1 for fact in additional_facts if fact.lower() in actual_lower
        )
        additional_score = (
            (facts_found / len(additional_facts)) * 40.0 if additional_facts else 0.0
        )

        # Appropriate length (20%)
        word_count = len(actual.split())
        if 50 <= word_count <= 150:
            length_score = 20.0
        elif 30 <= word_count < 50 or 150 < word_count <= 200:
            length_score = 10.0
        else:
            length_score = 0.0

        return basic_score + additional_score + length_score

    def score_reasoning_quality(
        self, proof_tree: Dict, strategies_used: List[str], reasoning_steps: List[str]
    ) -> float:
        """Score reasoning quality for elaboration."""
        score = 0.0

        # Good reasoning depth: +40%
        if proof_tree:
            depth = self._calculate_proof_tree_depth(proof_tree)
            if depth >= 3:
                score += 40
            elif depth >= 2:
                score += 25

        # Multiple strategies: +30%
        if len(strategies_used) >= 2:
            score += 30
        elif len(strategies_used) >= 1:
            score += 15

        # Adequate reasoning steps: +30%
        if len(reasoning_steps) >= 3:
            score += 30
        elif len(reasoning_steps) >= 1:
            score += 15

        return min(score, 100.0)
