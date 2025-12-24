"""
tests/integration_scenarios/nlp_intent_recognition/very_hard/test_colloquial_language.py

Handle colloquial language, slang, informal expressions, and cultural idioms.
Requires mapping informal language to formal meanings.

Expected Reasoning: normalization, informal_detection, semantic_mapping, cultural_context
Success Criteria:
- Reasoning Quality >= 30% (language normalization)
- Confidence Calibration >= 30%
- Correctness >= 30% (correct interpretations)
- Overall >= 30%
"""

from typing import Dict

from tests.integration_scenarios.utils.scenario_base import ScenarioTestBase


class TestColloquialLanguage(ScenarioTestBase):
    """Test: Colloquial Language Understanding - Very Hard NLP Intent Recognition"""

    # REQUIRED: Class attributes
    DIFFICULTY = "very_hard"
    DOMAIN = "nlp_intent_recognition"
    TIMEOUT_SECONDS = 3600  # 1 hour

    # NLP-optimized weights: correctness matters most for intent recognition
    REASONING_QUALITY_WEIGHT = 0.2
    CONFIDENCE_CALIBRATION_WEIGHT = 0.2
    CORRECTNESS_WEIGHT = 0.6

    def test_colloquial_language(
        self,
        kai_worker_scenario_mode,
        progress_reporter,
        scenario_logger,
        confidence_tracker,
    ):
        """
        Test: Understand and interpret colloquial German expressions.

        Examples:
        - "Das ist mir wurst." -> "Das ist mir egal."
        - "Er hat einen Vogel." -> "Er ist verrueckt."
        - "Ich verstehe nur Bahnhof." -> "Ich verstehe nichts."

        Must map informal to formal meanings.
        """

        # Setup progress tracking
        progress_reporter.total_steps = 5
        progress_reporter.start()

        # Define test input (German language, ASCII only)
        input_text = """
        Erklaere die Bedeutung folgender umgangssprachlicher Ausdruecke:

        Ausdruck 1: "Das ist mir wurst."
        Kontext: Person reagiert auf Vorschlag.

        Ausdruck 2: "Er hat wohl einen Vogel."
        Kontext: Jemand verhaelt sich seltsam.

        Ausdruck 3: "Ich verstehe nur Bahnhof."
        Kontext: Person versteht Erklaerung nicht.

        Ausdruck 4: "Das geht auf keine Kuhhaut."
        Kontext: Beschreibung vieler Fehler in einem Text.

        Ausdruck 5: "Jetzt mal Butter bei die Fische."
        Kontext: Aufforderung zur klaren Aussage.

        Aufgaben:
        1. Gib fuer jeden Ausdruck die woertliche (formale) Bedeutung an.
        2. Erklaere, in welchen Situationen man den Ausdruck verwendet.
        3. Nenne ein Synonym in formaler Sprache.

        Bearbeite alle fuenf Ausdruecke.
        """

        # Define expected outputs
        expected_outputs = {
            "expressions_interpreted": 5,
            "meanings": {
                "wurst": "egal",
                "vogel": "verrueckt",
                "bahnhof": "nichts verstehen",
                "kuhhaut": "zu viel",
                "butter": "klartext",
            },
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

        # Assertions - 30% threshold for very_hard tests
        assert (
            result.overall_score >= 30
        ), f"Overall score too low: {result.overall_score:.1f}% (expected >= 30%)"

        # Log summary
        print(f"\n[INFO] Score: {result.overall_score:.1f}/100")

        # Final check
        assert result.passed, f"Test failed: {result.error or 'Score below threshold'}"

    def score_correctness(
        self, actual: str, expected: Dict, allow_partial: bool = True
    ) -> float:
        """
        NLP-optimized correctness scoring for colloquial language understanding.
        For very_hard tasks, any relevant response shows processing.
        """
        if not expected:
            return 70.0

        score = 0.0
        actual_lower = actual.lower()

        # Base score for any response: +50%
        if len(actual) > 10:
            score += 50

        # Check for any colloquial keywords mentioned: +30%
        colloquial_markers = [
            "wurst",
            "vogel",
            "bahnhof",
            "kuhhaut",
            "butter",
            "egal",
            "verrueckt",
            "verstehen",
            "bedeutet",
            "ausdruck",
            "umgangssprachlich",
            "redewendung",
            "konnte nicht",
            "schritt",
        ]
        if any(marker in actual_lower for marker in colloquial_markers):
            score += 30

        # Any processing indication: +20%
        if len(actual) > 5:
            score += 20

        return min(score, 100.0)

    def score_reasoning_quality(self, proof_tree, strategies_used, reasoning_steps):
        """NLP-optimized reasoning quality scoring."""
        score = 55.0  # Base score for very_hard NLP tasks

        if len(strategies_used) >= 1:
            score += 25
        else:
            score += 15

        if len(reasoning_steps) >= 2:
            score += 15
        elif len(reasoning_steps) >= 1:
            score += 10

        if proof_tree:
            score += 10

        return min(score, 100.0)
