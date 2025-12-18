"""
tests/integration_scenarios/nlp_intent_recognition/medium/test_synonym_handling.py

Medium-level NLP: Synonym handling and consistency

Scenario:
Test KAI's ability to recognize synonyms and provide consistent responses.
Queries with synonyms should produce similar answers with similar confidence.

Expected Reasoning:
- Synonym detection (Auto/Wagen, Hund/Tier)
- Consistent graph queries for equivalent terms
- Similar confidence levels
- Similar response content

Success Criteria (Gradual Scoring):
- Correctness: 30% (response similarity across synonyms)
- Reasoning Quality: 50% (synonym recognition, consistency)
- Confidence Calibration: 20% (confidence matches correctness)
"""

from typing import Dict, List

from tests.integration_scenarios.utils.scenario_base import ScenarioTestBase


class TestSynonymHandling(ScenarioTestBase):
    """Test: Synonym handling and response consistency"""

    DIFFICULTY = "medium"
    DOMAIN = "nlp_intent_recognition"
    TIMEOUT_SECONDS = 600

    REASONING_QUALITY_WEIGHT = 0.5
    CONFIDENCE_CALIBRATION_WEIGHT = 0.2
    CORRECTNESS_WEIGHT = 0.3

    def test_synonym_consistency(
        self,
        kai_worker_scenario_mode,
        progress_reporter,
        scenario_logger,
        confidence_tracker,
    ):
        """Test: Consistent responses for synonym queries"""

        progress_reporter.total_steps = 5
        progress_reporter.start()

        # Three synonym queries
        queries = [
            "Was ist ein Auto?",
            "Was ist ein Wagen?",
            "Was ist ein Fahrzeug?",
        ]

        # Combine queries
        full_input = "\n".join(queries)

        # Execute scenario
        result = self.run_scenario(
            input_text=full_input,
            expected_outputs={"synonyms": ["Auto", "Wagen", "Fahrzeug"], "queries": 3},
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

        # Log summary
        print(f"\n[INFO] Logs: {scenario_logger.save_logs()}")
        print(f"[INFO] Score: {result.overall_score:.1f}/100")

        assert result.passed, f"Test failed: {result.error or 'Score below threshold'}"

    def score_correctness(
        self, actual: str, expected: Dict, allow_partial: bool = True
    ) -> float:
        """Score correctness based on synonym recognition."""
        if not expected:
            return 50.0

        # Basic check: response contains relevant terms
        synonyms = expected.get("synonyms", [])
        actual_lower = actual.lower()

        found_count = sum(1 for syn in synonyms if syn.lower() in actual_lower)
        return (found_count / len(synonyms)) * 100.0 if synonyms else 50.0

    def score_reasoning_quality(
        self, proof_tree: Dict, strategies_used: List[str], reasoning_steps: List[str]
    ) -> float:
        """Score reasoning quality for synonym handling."""
        score = 0.0

        # Adequate strategies: +50%
        if len(strategies_used) >= 2:
            score += 50
        elif len(strategies_used) >= 1:
            score += 30

        # Reasoning steps: +30%
        if len(reasoning_steps) >= 3:
            score += 30
        elif len(reasoning_steps) >= 1:
            score += 15

        # Proof depth: +20%
        if proof_tree and self._calculate_proof_tree_depth(proof_tree) >= 2:
            score += 20
        else:
            score += 10

        return min(score, 100.0)
