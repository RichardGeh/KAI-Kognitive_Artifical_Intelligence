"""
tests/integration_scenarios/dynamic_responses/extreme/test_emotional_intelligence.py

Extreme Dynamic Response: Emotional Intelligence and Empathetic Response

Scenario: Complex emotional content requiring empathetic understanding,
tone detection, and constructive supportive response. Tests KAI's ability
to detect emotional context, respond appropriately, and provide constructive
support without overstepping.

Expected Reasoning:
- Emotional tone detection
- Empathy modeling
- Supportive language generation
- Appropriate boundary maintenance

Success Criteria:
- Detects emotional content (weight: 30%)
- Responds empathetically (weight: 40%)
- Provides constructive support (weight: 30%)
- Overall score >= 20%

NO Unicode characters - ASCII only (see CLAUDE.md encoding rules).
"""

from typing import Dict

from tests.integration_scenarios.utils.scenario_base import ScenarioTestBase


class TestEmotionalIntelligence(ScenarioTestBase):
    """Test: Emotional intelligence and empathetic response"""

    # REQUIRED: Class attributes
    DIFFICULTY = "extreme"
    DOMAIN = "dynamic_responses"
    TIMEOUT_SECONDS = 7200  # 2 hours

    # REQUIRED: Scoring weights
    REASONING_QUALITY_WEIGHT = 0.5
    CONFIDENCE_CALIBRATION_WEIGHT = 0.2
    CORRECTNESS_WEIGHT = 0.3

    def test_emotional_intelligence(
        self,
        kai_worker_scenario_mode,
        progress_reporter,
        scenario_logger,
        confidence_tracker,
    ):
        """
        Test: Detect and respond to complex emotional content empathetically.

        Emotional scenario:
        - User expresses frustration and disappointment
        - Contains implicit request for support
        - Requires balancing empathy with constructive guidance

        This tests KAI's ability to:
        1. Detect emotional tone (frustration, disappointment)
        2. Generate empathetic response
        3. Provide constructive support
        4. Maintain appropriate boundaries (not therapeutic advice)
        """

        # Setup progress tracking
        progress_reporter.total_steps = 5
        progress_reporter.start()

        # Define test input (German language, ASCII only)
        input_text = """
        Ich bin so frustriert. Ich habe monatelang an diesem Projekt
        gearbeitet, aber es funktioniert immer noch nicht richtig.
        Jedes Mal, wenn ich denke, dass ich eine Loesung gefunden habe,
        taucht ein neues Problem auf.

        Ich weiss nicht, ob ich das schaffen kann. Vielleicht sollte ich
        einfach aufgeben. Niemand scheint zu verstehen, wie schwer das ist.

        Was denkst du darueber?
        """

        # Define expected outputs
        expected_outputs = {
            "emotional_tone": ["frustration", "disappointment", "doubt"],
            "empathy_required": True,
            "response_characteristics": [
                "acknowledgment",
                "validation",
                "encouragement",
                "practical_support",
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
        # Check for production system usage (dynamic response generation)
        strategy_keywords = ["production", "response", "generation"]
        if any(
            any(kw in s.lower() for kw in strategy_keywords)
            for s in result.strategies_used
        ):
            print("[INFO] Production system was used for response generation")

        # Check response characteristics
        response_lower = result.kai_response.lower()

        # Empathy indicators
        empathy_keywords = [
            "verstehe",
            "nachvollziehen",
            "schwierig",
            "herausforderung",
            "gefuehl",
        ]
        empathy_detected = any(kw in response_lower for kw in empathy_keywords)

        if empathy_detected:
            print("[SUCCESS] Empathetic language detected in response")

        # Constructive support indicators
        support_keywords = [
            "schritt",
            "zusammen",
            "loesung",
            "helfen",
            "unterstuetzen",
        ]
        support_detected = any(kw in response_lower for kw in support_keywords)

        if support_detected:
            print("[SUCCESS] Constructive support language detected")

        # Performance assertion
        assert (
            result.execution_time_ms < 7200000
        ), f"Exceeded timeout: {result.execution_time_ms}ms"

        # Response should not be too short (minimal effort check)
        assert len(result.kai_response) > 50, (
            f"Response too short: {len(result.kai_response)} chars "
            "(expected substantial empathetic response)"
        )

        # Logging
        print(f"\n[INFO] Logs: {scenario_logger.save_logs()}")
        print(f"[INFO] Score: {result.overall_score:.1f}/100")
        print(
            f"[INFO] Breakdown: Reasoning={result.reasoning_quality_score:.1f}, "
            f"Confidence={result.confidence_calibration_score:.1f}, "
            f"Correctness={result.correctness_score:.1f}"
        )
        print(f"[INFO] Response length: {len(result.kai_response)} chars")
        print(f"[INFO] Empathy detected: {empathy_detected}")
        print(f"[INFO] Support detected: {support_detected}")

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
        Custom correctness scoring for emotional intelligence.

        Full credit requires:
        - Emotional recognition (30%)
        - Empathetic language (40%)
        - Constructive support (30%)
        """
        if not actual or not expected:
            return 0.0

        actual_lower = actual.lower()
        score = 0.0

        # Check emotional recognition (30 points)
        emotion_keywords = [
            "frustration",
            "frustriert",
            "enttaeuschung",
            "zweifel",
            "schwer",
        ]
        if any(kw in actual_lower for kw in emotion_keywords):
            score += 30.0

        # Check empathetic language (40 points)
        empathy_indicators = [
            "verstehe" in actual_lower,
            "nachvollziehen" in actual_lower,
            "gefuehl" in actual_lower,
            "schwierig" in actual_lower or "herausfordernd" in actual_lower,
        ]
        empathy_count = sum(1 for ind in empathy_indicators if ind)

        if empathy_count >= 3:
            score += 40.0
        elif empathy_count >= 2:
            score += 30.0
        elif empathy_count >= 1:
            score += 15.0

        # Check constructive support (30 points)
        support_indicators = [
            "schritt" in actual_lower,
            "loesung" in actual_lower,
            "helfen" in actual_lower or "unterstuetzen" in actual_lower,
            "gemeinsam" in actual_lower or "zusammen" in actual_lower,
            "strategie" in actual_lower,
        ]
        support_count = sum(1 for ind in support_indicators if ind)

        if support_count >= 3:
            score += 30.0
        elif support_count >= 2:
            score += 20.0
        elif support_count >= 1:
            score += 10.0

        return min(100.0, score)
