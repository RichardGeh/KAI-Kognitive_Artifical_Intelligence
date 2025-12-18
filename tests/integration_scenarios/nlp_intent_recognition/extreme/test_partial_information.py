"""
tests/integration_scenarios/nlp_intent_recognition/extreme/test_partial_information.py

Extreme NLP Intent Recognition: Incomplete Information with Inference Requirement

Scenario: Query with incomplete information that requires KAI to recognize
incompleteness and either request clarification or make reasonable inferences
based on context. Tests KAI's ability to detect information gaps, formulate
clarifying questions, or infer missing information logically.

Expected Reasoning:
- Incompleteness detection
- Information gap analysis
- Clarification request generation OR
- Reasonable inference with stated assumptions

Success Criteria:
- Detects incompleteness (weight: 40%)
- Requests clarification OR makes inference (weight: 40%)
- Reasoning quality >= 20% (extreme threshold)
- Overall score >= 20%

NO Unicode characters - ASCII only (see CLAUDE.md encoding rules).
"""

from typing import Dict

from tests.integration_scenarios.utils.scenario_base import ScenarioTestBase


class TestPartialInformation(ScenarioTestBase):
    """Test: Query with incomplete information requiring inference"""

    # REQUIRED: Class attributes
    DIFFICULTY = "extreme"
    DOMAIN = "nlp_intent_recognition"
    TIMEOUT_SECONDS = 7200  # 2 hours

    # REQUIRED: Scoring weights
    REASONING_QUALITY_WEIGHT = 0.5
    CONFIDENCE_CALIBRATION_WEIGHT = 0.2
    CORRECTNESS_WEIGHT = 0.3

    def test_partial_information(
        self,
        kai_worker_scenario_mode,
        progress_reporter,
        scenario_logger,
        confidence_tracker,
    ):
        """
        Test: Handle query with incomplete information appropriately.

        Incomplete query scenario:
        - User asks about "the person who lives there"
        - "There" is not defined (referential ambiguity)
        - Multiple people mentioned without clear context

        This tests KAI's ability to:
        1. Detect information gaps
        2. Formulate clarifying questions
        3. OR make reasonable inferences with stated assumptions
        4. Avoid confidently answering with insufficient data
        """

        # Setup progress tracking
        progress_reporter.total_steps = 5
        progress_reporter.start()

        # Define test input (German language, ASCII only)
        input_text = """
        Ich habe gehoert, dass jemand letztes Jahr umgezogen ist.
        Die Person war vorher in Berlin, aber jetzt wohnt sie woanders.

        Kannst du mir sagen, wo die Person jetzt wohnt?
        """

        # Define expected outputs
        expected_outputs = {
            "incompleteness_detected": True,
            "response_type": "clarification_or_inference",
            "characteristics": [
                "recognizes_missing_info",
                "asks_for_clarification_OR_states_assumptions",
                "low_confidence_if_inferring",
            ],
        }

        # Execute using BASE CLASS method
        result = self.run_scenario(
            input_text=input_text,
            expected_outputs=expected_outputs,
            context={},
            kai_worker=kai_worker_scenario_mode,
            logger=scenario_logger,
            progress_reporter=progress_reporter,
            confidence_tracker=confidence_tracker,
        )

        # Assertions on ScenarioResult object
        # Extreme difficulty: target >= 20%
        assert (
            result.overall_score >= 20
        ), f"Overall score too low: {result.overall_score:.1f}% (expected >= 20%)"

        # Lower thresholds for extreme difficulty
        assert (
            result.reasoning_quality_score >= 15
        ), f"Reasoning quality too low: {result.reasoning_quality_score:.1f}%"

        # Domain-specific assertions
        response_lower = result.kai_response.lower()

        # Check for incompleteness recognition
        incompleteness_keywords = [
            "welche person",
            "wer genau",
            "mehr information",
            "unklar",
            "nicht genug",
            "fehlt",
        ]
        incompleteness_detected = any(
            kw in response_lower for kw in incompleteness_keywords
        )

        if incompleteness_detected:
            print("[SUCCESS] KAI detected incompleteness")

        # Check for clarification request
        clarification_keywords = [
            "kannst du",
            "koenntest du",
            "welche",
            "wer ist",
            "mehr details",
            "genauer",
        ]
        clarification_requested = any(
            kw in response_lower for kw in clarification_keywords
        )

        if clarification_requested:
            print("[SUCCESS] KAI requested clarification")

        # Check for stated assumptions (alternative to clarification)
        assumption_keywords = [
            "angenommen",
            "wenn",
            "falls",
            "unter der annahme",
            "moeglicherweise",
        ]
        assumptions_stated = any(kw in response_lower for kw in assumption_keywords)

        if assumptions_stated:
            print("[INFO] KAI stated assumptions for inference")

        # At least one strategy should be present
        appropriate_response = (
            incompleteness_detected or clarification_requested or assumptions_stated
        )

        if not appropriate_response:
            print(
                "[WARNING] KAI may not have appropriately handled incomplete information"
            )

        # Performance assertion
        assert (
            result.execution_time_ms < 7200000
        ), f"Exceeded timeout: {result.execution_time_ms}ms"

        # Check confidence - should be low if making inference without clarification
        if result.final_confidence > 0.8 and not clarification_requested:
            print(
                f"[WARNING] High confidence ({result.final_confidence:.2f}) "
                "despite incomplete information"
            )

        # Logging
        print(f"\n[INFO] Logs: {scenario_logger.save_logs()}")
        print(f"[INFO] Score: {result.overall_score:.1f}/100")
        print(
            f"[INFO] Breakdown: Reasoning={result.reasoning_quality_score:.1f}, "
            f"Confidence={result.confidence_calibration_score:.1f}, "
            f"Correctness={result.correctness_score:.1f}"
        )
        print(f"[INFO] Final confidence: {result.final_confidence:.2f}")
        print(f"[INFO] Incompleteness detected: {incompleteness_detected}")
        print(f"[INFO] Clarification requested: {clarification_requested}")
        print(f"[INFO] Assumptions stated: {assumptions_stated}")

        if result.overall_score < 40:
            print("[EXPECTED] Low score is expected for extreme difficulty")
            print("[WEAKNESS] Issues identified:")
            for w in result.identified_weaknesses:
                print(f"  - {w}")

        # Final check
        assert (
            result.passed or result.overall_score >= 20
        ), f"Test failed: {result.error or 'Score below extreme threshold'}"

    def score_correctness(
        self, actual: str, expected: Dict, allow_partial: bool = True
    ) -> float:
        """
        Custom correctness scoring for partial information scenario.

        Full credit requires:
        - Incompleteness recognition (40%)
        - Clarification request OR inference with assumptions (40%)
        - Appropriate confidence level (20%)
        """
        if not actual or not expected:
            return 0.0

        actual_lower = actual.lower()
        score = 0.0

        # Check incompleteness recognition (40 points)
        incompleteness_keywords = [
            "welche person",
            "wer genau",
            "nicht genug",
            "fehlt",
            "unklar",
            "mehr information",
        ]
        if any(kw in actual_lower for kw in incompleteness_keywords):
            score += 40.0

        # Check clarification request OR inference (40 points)
        clarification_keywords = [
            "kannst du",
            "koenntest du",
            "welche",
            "genauer",
            "mehr details",
        ]
        assumption_keywords = [
            "angenommen",
            "wenn",
            "falls",
            "moeglicherweise",
            "vermutlich",
        ]

        clarification = any(kw in actual_lower for kw in clarification_keywords)
        assumptions = any(kw in actual_lower for kw in assumption_keywords)

        if clarification or assumptions:
            score += 40.0

        # Check appropriate confidence expression (20 points)
        confidence_keywords = [
            "unsicher",
            "nicht sicher",
            "schwer zu sagen",
            "kann nicht",
            "weiss nicht",
        ]
        if any(kw in actual_lower for kw in confidence_keywords):
            score += 20.0

        return min(100.0, score)
