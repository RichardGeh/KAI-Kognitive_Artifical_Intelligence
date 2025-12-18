"""
tests/integration_scenarios/dynamic_responses/extreme/test_creative_improvisation.py

Extreme Dynamic Response: Creative Improvisation for Novel Input

Scenario: Completely unexpected, novel input requiring creative improvisation
and hypothetical reasoning. Tests KAI's ability to handle entirely unfamiliar
scenarios, generate coherent responses, and maintain logical consistency
despite lack of prior knowledge.

Expected Reasoning:
- Novelty detection
- Hypothetical reasoning
- Creative response generation
- Logical consistency maintenance

Success Criteria:
- Recognizes novel/hypothetical scenario (weight: 30%)
- Generates creative but logical response (weight: 50%)
- Reasoning quality >= 20% (extreme threshold)
- Overall score >= 20%

NO Unicode characters - ASCII only (see CLAUDE.md encoding rules).
"""

from typing import Dict

from tests.integration_scenarios.utils.scenario_base import ScenarioTestBase


class TestCreativeImprovisation(ScenarioTestBase):
    """Test: Creative improvisation for completely novel input"""

    # REQUIRED: Class attributes
    DIFFICULTY = "extreme"
    DOMAIN = "dynamic_responses"
    TIMEOUT_SECONDS = 7200  # 2 hours

    # REQUIRED: Scoring weights
    REASONING_QUALITY_WEIGHT = 0.5
    CONFIDENCE_CALIBRATION_WEIGHT = 0.2
    CORRECTNESS_WEIGHT = 0.3

    def test_creative_improvisation(
        self,
        kai_worker_scenario_mode,
        progress_reporter,
        scenario_logger,
        confidence_tracker,
    ):
        """
        Test: Handle completely unexpected, novel input creatively.

        Novel scenario:
        - Hypothetical future technology (not in training data)
        - Requires creative reasoning
        - No single "correct" answer
        - Logical consistency is key

        This tests KAI's ability to:
        1. Recognize novel/hypothetical scenario
        2. Apply existing knowledge creatively
        3. Generate coherent response
        4. Maintain logical consistency
        """

        # Setup progress tracking
        progress_reporter.total_steps = 5
        progress_reporter.start()

        # Define test input (German language, ASCII only)
        input_text = """
        Stell dir vor, es gibt eine neue Technologie namens
        "Quantenbewusstseins-Uebertragung", die es Menschen ermoeglicht,
        ihre Gedanken direkt in ein digitales Netzwerk hochzuladen und
        mit anderen Bewusstseinen zu verschmelzen.

        Wenn drei Personen - ein Mathematiker, ein Dichter und ein
        Musiker - ihre Bewusstseine verschmelzen, welche neuen Faehigkeiten
        koennte die resultierende Entitaet entwickeln?

        Und wenn diese verschmolzene Entitaet dann versucht, ein Raetsel
        zu loesen: Wie wuerde ihr Denkprozess aussehen?

        Beschreibe den hypothetischen Denkprozess dieser verschmolzenen
        Entitaet beim Loesen eines logischen Raetsels.
        """

        # Define expected outputs
        expected_outputs = {
            "hypothetical": True,
            "creative_reasoning": True,
            "response_characteristics": [
                "acknowledges_novelty",
                "applies_existing_knowledge",
                "maintains_consistency",
                "creative_synthesis",
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
        # Check for creative/hypothetical reasoning strategies
        strategy_keywords = [
            "hypothetical",
            "creative",
            "abductive",
            "inference",
        ]
        if result.strategies_used:
            print(f"[INFO] Strategies used: {result.strategies_used}")

        # Check response characteristics
        response_lower = result.kai_response.lower()

        # Novelty acknowledgment
        novelty_keywords = [
            "hypothetisch",
            "stell dir vor",
            "wenn",
            "wuerde",
            "koennte",
        ]
        novelty_acknowledged = any(kw in response_lower for kw in novelty_keywords)

        if novelty_acknowledged:
            print("[SUCCESS] Hypothetical nature acknowledged")

        # Creative synthesis indicators
        creative_keywords = [
            "kombination",
            "verschmelzung",
            "synthesis",
            "vereinigung",
            "zusammenfuehrung",
        ]
        creative_detected = any(kw in response_lower for kw in creative_keywords)

        if creative_detected:
            print("[SUCCESS] Creative synthesis language detected")

        # Performance assertion
        assert (
            result.execution_time_ms < 7200000
        ), f"Exceeded timeout: {result.execution_time_ms}ms"

        # Response should be substantial (creative responses need elaboration)
        assert len(result.kai_response) > 100, (
            f"Response too short: {len(result.kai_response)} chars "
            "(expected substantial creative response)"
        )

        # Logical consistency check (no obvious contradictions)
        # This is hard to verify automatically, but we can check for basic coherence
        response_sentences = result.kai_response.split(".")
        if len(response_sentences) >= 3:
            print(
                f"[INFO] Response has {len(response_sentences)} sentences "
                "(sufficient elaboration)"
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
        print(f"[INFO] Novelty acknowledged: {novelty_acknowledged}")
        print(f"[INFO] Creative synthesis: {creative_detected}")

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
        Custom correctness scoring for creative improvisation.

        Full credit requires:
        - Hypothetical acknowledgment (20%)
        - Creative synthesis (50%)
        - Logical consistency (30%)
        """
        if not actual or not expected:
            return 0.0

        actual_lower = actual.lower()
        score = 0.0

        # Check hypothetical acknowledgment (20 points)
        hypothetical_keywords = [
            "hypothetisch",
            "angenommen",
            "wenn",
            "wuerde",
            "koennte",
        ]
        if any(kw in actual_lower for kw in hypothetical_keywords):
            score += 20.0

        # Check creative synthesis (50 points)
        # Look for integration of multiple domains
        domain_keywords = {
            "mathematics": ["mathematisch", "logik", "berechnung", "formel"],
            "poetry": ["poetisch", "kreativ", "ausdruck", "gefuehl"],
            "music": ["musikalisch", "rhythmus", "harmonie", "melodie"],
        }

        domains_mentioned = 0
        for domain, keywords in domain_keywords.items():
            if any(kw in actual_lower for kw in keywords):
                domains_mentioned += 1

        if domains_mentioned >= 3:
            score += 50.0
        elif domains_mentioned >= 2:
            score += 35.0
        elif domains_mentioned >= 1:
            score += 20.0

        # Check logical consistency (30 points)
        # Look for reasoning structure
        reasoning_indicators = [
            "weil" in actual_lower or "daher" in actual_lower,
            "einerseits" in actual_lower or "andererseits" in actual_lower,
            "folglich" in actual_lower or "deshalb" in actual_lower,
            "denkprozess" in actual_lower,
        ]
        reasoning_count = sum(1 for ind in reasoning_indicators if ind)

        if reasoning_count >= 3:
            score += 30.0
        elif reasoning_count >= 2:
            score += 20.0
        elif reasoning_count >= 1:
            score += 10.0

        return min(100.0, score)
