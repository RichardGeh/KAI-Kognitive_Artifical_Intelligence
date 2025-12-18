"""
tests/integration_scenarios/dynamic_responses/hard/test_creative_examples.py

Hard-level dynamic response: Generate relevant creative examples when asked

Scenario:
KAI should generate 3-5 diverse, relevant examples when explicitly requested.
Examples should be creative (not just memorized), contextually appropriate,
and demonstrate understanding of the concept.

Expected Reasoning:
- Production System should recognize "Beispiel" request
- Example generation rules should fire
- Multiple examples should be generated
- Examples should be diverse and relevant
- Confidence should be medium-high (>0.70)

Success Criteria (Gradual Scoring):
- Correctness: 30% (examples are relevant and diverse)
- Reasoning Quality: 50% (Production System used, creative generation)
- Confidence Calibration: 20% (confidence matches quality)
"""

import re
from typing import Dict, List

from tests.integration_scenarios.utils.scenario_base import ScenarioTestBase


class TestCreativeExamples(ScenarioTestBase):
    """Test: Generate creative, relevant examples on request"""

    DIFFICULTY = "hard"
    DOMAIN = "dynamic_responses"
    TIMEOUT_SECONDS = 1800  # 30 minutes

    # Scoring weights for this scenario type
    REASONING_QUALITY_WEIGHT = 0.5
    CONFIDENCE_CALIBRATION_WEIGHT = 0.2
    CORRECTNESS_WEIGHT = 0.3

    def test_creative_example_generation(
        self,
        kai_worker_scenario_mode,
        progress_reporter,
        scenario_logger,
        confidence_tracker,
    ):
        """
        Test: Generate diverse, relevant examples

        Request for examples should produce 3-5 creative examples.

        Expected: Multiple diverse examples generated
        """

        # Setup progress reporter
        progress_reporter.total_steps = 5
        progress_reporter.start()

        # Query requesting examples
        query = """
Was sind Metaphern? Gib mir bitte 5 verschiedene Beispiele fuer Metaphern.
        """

        # Expected: 3-5 examples, each different, all metaphors
        expected_outputs = {
            "min_examples": 3,
            "max_examples": 5,
            "concept": "Metapher",
            "diversity": True,
        }

        # Execute scenario
        result = self.run_scenario(
            input_text=query,
            expected_outputs={"examples": expected_outputs},
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

        # Check that Production System was used
        has_production_system = any(
            "production" in s.lower() for s in result.strategies_used
        )
        assert (
            has_production_system
        ), f"Expected Production System, got: {result.strategies_used}"

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
        Score correctness based on example quality and quantity.

        Args:
            actual: Actual KAI response text
            expected: Dict with "examples" key
            allow_partial: Whether to give partial credit

        Returns:
            Score 0-100
        """
        if not expected or "examples" not in expected:
            return 50.0

        score = 0.0
        examples_config = expected["examples"]
        actual_lower = actual.lower()

        # Count examples (look for numbered lists or bullet points)
        example_patterns = [
            r"\d+\.\s+",  # 1. 2. 3.
            r"-\s+",  # - - -
            r"\*\s+",  # * * *
        ]

        example_count = 0
        for pattern in example_patterns:
            matches = re.findall(pattern, actual)
            if matches:
                example_count = max(example_count, len(matches))

        # Check example count: +40%
        min_examples = examples_config.get("min_examples", 3)
        max_examples = examples_config.get("max_examples", 5)

        if min_examples <= example_count <= max_examples:
            score += 40
        elif example_count >= min_examples:
            score += 30  # Over max but still good
        elif example_count >= min_examples - 1:
            score += 20  # Close to min

        # Check concept mentioned: +30%
        if "concept" in examples_config:
            concept = examples_config["concept"].lower()
            if concept in actual_lower:
                score += 30

        # Check diversity (examples are different): +30%
        if "diversity" in examples_config and examples_config["diversity"]:
            # Simple check: look for distinct content in examples
            # Split by numbers and check if they're different
            lines = actual.split("\n")
            example_lines = [line for line in lines if re.match(r"\d+\.", line.strip())]

            if len(example_lines) >= min_examples:
                # Check if examples are sufficiently different
                unique_words = set()
                for line in example_lines:
                    words = line.lower().split()
                    unique_words.update(words)

                # If many unique words, examples are likely diverse
                if len(unique_words) >= len(example_lines) * 3:
                    score += 30
                else:
                    score += 15  # Some diversity

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
        if depth >= 4:
            score += 30
        elif depth >= 2:
            score += 20
        elif depth >= 1:
            score += 10

        # Multiple reasoning steps: +20%
        if len(reasoning_steps) >= 5:
            score += 20
        elif len(reasoning_steps) >= 3:
            score += 15
        elif len(reasoning_steps) >= 1:
            score += 10

        return min(score, 100.0)
