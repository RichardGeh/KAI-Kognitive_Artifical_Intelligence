"""
tests/integration_scenarios/dynamic_responses/medium/test_simple_reformulation.py

Medium-level dynamic response: Simple reformulation/rephrasing

Scenario:
Test KAI's ability to rephrase a statement while preserving meaning.
Uses Production System for response variability.

Expected Reasoning:
- Production System should generate varied response
- Key concepts must be preserved (Apfel, Frucht, Baum)
- Semantic similarity should be high (>= 0.75)
- Response should not be verbatim copy

Success Criteria (Gradual Scoring):
- Correctness: 30% (semantic similarity + key concepts)
- Reasoning Quality: 50% (production system usage, natural phrasing)
- Confidence Calibration: 20% (confidence matches correctness)
"""

from typing import Dict, List

from tests.integration_scenarios.utils.scenario_base import ScenarioTestBase


class TestSimpleReformulation(ScenarioTestBase):
    """Test: Simple statement reformulation with meaning preservation"""

    DIFFICULTY = "medium"
    DOMAIN = "dynamic_responses"
    TIMEOUT_SECONDS = 600

    REASONING_QUALITY_WEIGHT = 0.5
    CONFIDENCE_CALIBRATION_WEIGHT = 0.2
    CORRECTNESS_WEIGHT = 0.3

    def test_reformulate_apple_description(
        self,
        kai_worker_scenario_mode,
        progress_reporter,
        scenario_logger,
        confidence_tracker,
    ):
        """
        Test: Reformulate statement about apples

        Input: Explain differently what an apple is
        Expected: Preserve key concepts (Apfel, Frucht, Baum) with different phrasing
        """

        # Setup
        progress_reporter.total_steps = 5
        progress_reporter.start()

        input_text = """
Erklaere anders: Ein Apfel ist eine Frucht, die am Baum waechst und oft rot ist.
        """

        key_concepts = ["Apfel", "Frucht", "Baum"]
        original_statement = (
            "Ein Apfel ist eine Frucht, die am Baum waechst und oft rot ist."
        )

        # Execute scenario using base class
        result = self.run_scenario(
            input_text=input_text,
            expected_outputs={
                "key_concepts": key_concepts,
                "original": original_statement,
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
        print(
            f"[INFO] Breakdown: Reasoning={result.reasoning_quality_score:.1f}, "
            f"Confidence={result.confidence_calibration_score:.1f}, "
            f"Correctness={result.correctness_score:.1f}"
        )

        # Mark test as passed
        assert result.passed, f"Test failed: {result.error or 'Score below threshold'}"

    def score_correctness(
        self, actual: str, expected: Dict, allow_partial: bool = True
    ) -> float:
        """
        Score correctness based on semantic similarity and key concept preservation.

        Returns: 0-100
        """
        if not expected:
            return 50.0

        key_concepts = expected.get("key_concepts", [])
        original = expected.get("original", "")

        # Score key concept preservation (60%)
        concepts_found = sum(
            1 for concept in key_concepts if concept.lower() in actual.lower()
        )
        concept_score = (
            (concepts_found / len(key_concepts)) * 60.0 if key_concepts else 0.0
        )

        # Score semantic similarity via word overlap (30%)
        actual_words = set(actual.lower().split())
        original_words = set(original.lower().split())
        # Remove common stop words
        stop_words = {"ein", "eine", "der", "die", "das", "ist", "und", "am", "oft"}
        actual_words -= stop_words
        original_words -= stop_words

        overlap = len(actual_words & original_words)
        union = len(actual_words | original_words)
        similarity_score = (overlap / union) * 30.0 if union > 0 else 0.0

        # Score for not being verbatim (10%)
        actual_normalized = " ".join(sorted(actual.lower().split()))
        original_normalized = " ".join(sorted(original.lower().split()))
        is_different = actual_normalized != original_normalized
        verbatim_score = 10.0 if is_different else 0.0

        return concept_score + similarity_score + verbatim_score

    def score_reasoning_quality(
        self, proof_tree: Dict, strategies_used: List[str], reasoning_steps: List[str]
    ) -> float:
        """
        Score reasoning quality for reformulation.

        Returns: 0-100
        """
        score = 0.0

        # Production system usage: +50%
        response_length = len(reasoning_steps)
        if response_length >= 3:
            score += 50
        elif response_length >= 1:
            score += 30

        # Multiple strategies: +30%
        if len(strategies_used) >= 2:
            score += 30
        elif len(strategies_used) >= 1:
            score += 20

        # Basic reasoning present: +20%
        if proof_tree and self._calculate_proof_tree_depth(proof_tree) >= 2:
            score += 20
        elif reasoning_steps:
            score += 10

        return min(score, 100.0)
