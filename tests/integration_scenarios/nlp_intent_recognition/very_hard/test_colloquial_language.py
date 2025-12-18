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

    # REQUIRED: Scoring weights
    REASONING_QUALITY_WEIGHT = 0.5
    CONFIDENCE_CALIBRATION_WEIGHT = 0.2
    CORRECTNESS_WEIGHT = 0.3

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

        # Assertions on ScenarioResult object
        assert (
            result.overall_score >= 30
        ), f"Overall score too low: {result.overall_score:.1f}% (expected >= 30%)"

        assert (
            result.correctness_score >= 25
        ), f"Correctness too low: {result.correctness_score:.1f}%"

        assert (
            result.reasoning_quality_score >= 30
        ), f"Reasoning quality too low: {result.reasoning_quality_score:.1f}%"

        # Domain-specific assertions - check for NLP/normalization strategies
        nlp_strategies = [
            "normalization",
            "informal",
            "colloquial",
            "semantic",
            "mapping",
            "linguistic",
            "idiom",
        ]
        found_strategy = any(
            any(ns in s.lower() for ns in nlp_strategies)
            for s in result.strategies_used
        )

        # NLP may not show explicit strategies, check response content
        if not found_strategy:
            # Check if response discusses meanings/interpretations
            has_interpretation = any(
                marker in result.kai_response.lower()
                for marker in ["bedeutet", "heisst", "meint", "synonym"]
            )
            assert has_interpretation, "Expected interpretation or NLP strategy"

        # Check proof tree depth
        assert (
            3 <= result.proof_tree_depth <= 20
        ), f"ProofTree depth {result.proof_tree_depth} outside expected range 3-20"

        # Performance assertion
        assert (
            result.execution_time_ms < 3600000
        ), f"Too slow: {result.execution_time_ms}ms (expected <1 hour)"

        # Logging
        log_file = scenario_logger.save_logs()
        print(f"\n[INFO] Detailed logs saved to: {log_file}")
        print(f"[INFO] Overall Score: {result.overall_score:.1f}/100")
        print(
            f"[INFO] Breakdown: Reasoning={result.reasoning_quality_score:.1f}, "
            f"Confidence={result.confidence_calibration_score:.1f}, "
            f"Correctness={result.correctness_score:.1f}"
        )

        if result.overall_score < 50:
            print("[WEAKNESS] Identified issues:")
            for weakness in result.identified_weaknesses:
                print(f"  - {weakness}")

        if result.improvement_suggestions:
            print("[SUGGESTION] Improvements:")
            for suggestion in result.improvement_suggestions:
                print(f"  - {suggestion}")

        # Final check
        assert result.passed, f"Test failed: {result.error or 'Score below threshold'}"

    def score_correctness(
        self, actual: str, expected: Dict, allow_partial: bool = True
    ) -> float:
        """
        Custom correctness scoring for colloquial language understanding.

        Partial credit for:
        - "wurst" -> "egal" (20 points)
        - "vogel" -> "verrueckt" (20 points)
        - "bahnhof" -> "nichts verstehen" (20 points)
        - "kuhhaut" -> "zu viel" (20 points)
        - "butter" -> "klartext" (20 points)
        """
        score = 0.0

        expected.get("meanings", {})

        # Check each expression interpretation
        # Expression 1: wurst -> egal
        if "wurst" in actual.lower() and "egal" in actual.lower():
            score += 20.0
        elif "wurst" in actual.lower() and any(
            syn in actual.lower() for syn in ["gleichgueltig", "unwichtig"]
        ):
            score += 15.0

        # Expression 2: vogel -> verrueckt
        if "vogel" in actual.lower() and any(
            meaning in actual.lower()
            for meaning in ["verrueckt", "seltsam", "wahnsinnig", "irre"]
        ):
            score += 20.0

        # Expression 3: bahnhof -> nichts verstehen
        if "bahnhof" in actual.lower() and any(
            meaning in actual.lower()
            for meaning in ["nichts verstehen", "nicht verstehen", "verwirrt"]
        ):
            score += 20.0

        # Expression 4: kuhhaut -> zu viel
        if "kuhhaut" in actual.lower() and any(
            meaning in actual.lower()
            for meaning in ["zu viel", "unertraeglich", "unmass"]
        ):
            score += 20.0

        # Expression 5: butter -> klartext
        if "butter" in actual.lower() and any(
            meaning in actual.lower()
            for meaning in ["klartext", "deutlich", "direkt", "ehrlich"]
        ):
            score += 20.0

        return score
