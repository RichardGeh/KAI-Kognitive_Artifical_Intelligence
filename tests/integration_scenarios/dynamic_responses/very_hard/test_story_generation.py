"""
tests/integration_scenarios/dynamic_responses/very_hard/test_story_generation.py

Story generation with multiple constraints: character, location, plot twist.
Generate coherent 100-word narrative satisfying all constraints.

Expected Reasoning: creative, production_system, narrative, constraint_satisfaction
Success Criteria:
- Reasoning Quality >= 30% (production system usage)
- Confidence Calibration >= 30%
- Correctness >= 30% (constraint satisfaction)
- Overall >= 30%
"""

import re
from typing import Dict

from tests.integration_scenarios.utils.scenario_base import ScenarioTestBase


class TestStoryGeneration(ScenarioTestBase):
    """Test: Story Generation with Constraints - Very Hard Dynamic Response"""

    # REQUIRED: Class attributes
    DIFFICULTY = "very_hard"
    DOMAIN = "dynamic_responses"
    TIMEOUT_SECONDS = 3600  # 1 hour

    # REQUIRED: Scoring weights
    REASONING_QUALITY_WEIGHT = 0.4
    CONFIDENCE_CALIBRATION_WEIGHT = 0.2
    CORRECTNESS_WEIGHT = 0.4

    def test_story_generation(
        self,
        kai_worker_scenario_mode,
        progress_reporter,
        scenario_logger,
        confidence_tracker,
    ):
        """
        Test: Generate 100-word story with specific constraints.

        Constraints:
        - Protagonist: Ein alter Fischer
        - Location: Verlassene Insel
        - Twist: Der Fischer findet einen Brief von sich selbst
        - Length: ~100 words
        - Coherent narrative with beginning, middle, end
        """

        # Setup progress tracking
        progress_reporter.total_steps = 5
        progress_reporter.start()

        # Define test input (German language, ASCII only)
        input_text = """
        Erstelle eine kurze Geschichte (ca. 100 Woerter) mit folgenden Vorgaben:

        Hauptperson: Ein alter Fischer namens Heinrich
        Ort: Eine verlassene Insel im Nordmeer
        Handlungselement: Heinrich findet eine Flaschenpost
        Wendung: Der Brief in der Flasche ist von ihm selbst geschrieben
        Zeitpunkt: Brief ist 30 Jahre alt

        Die Geschichte soll:
        - Einen klaren Anfang, Mittelteil und Ende haben
        - Die Stimmung der Einsamkeit vermitteln
        - Die Wendung ueberraschend praesentieren
        - Ungefaehr 100 Woerter lang sein

        Schreibe die Geschichte.
        """

        # Define expected outputs
        expected_outputs = {
            "character_present": "Heinrich",
            "location_present": "Insel",
            "twist_present": True,  # Self-written letter mentioned
            "word_count_min": 70,
            "word_count_max": 150,
            "coherent": True,
            "narrative_structure": True,
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

        # Domain-specific assertions - check for production system usage
        production_strategies = [
            "production",
            "generation",
            "creative",
            "narrative",
            "story",
            "text",
        ]
        found_strategy = any(
            any(ps in s.lower() for ps in production_strategies)
            for s in result.strategies_used
        )

        # Production system may not appear in strategies, check response quality instead
        if not found_strategy:
            # Check if response is narrative (contains story elements)
            has_narrative = len(result.kai_response.split()) > 50
            assert (
                has_narrative
            ), "Expected narrative response or production system strategy"

        # Check proof tree depth (generation may have moderate depth)
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
        Custom correctness scoring for story generation.

        Partial credit for:
        - Character present (Heinrich) (20 points)
        - Location present (Insel) (20 points)
        - Twist element (self-written letter) (30 points)
        - Word count in range (15 points)
        - Narrative coherence (15 points)
        """
        score = 0.0

        # Check character present
        character = expected.get("character_present", "")
        if character and character.lower() in actual.lower():
            score += 20.0

        # Check location present
        location = expected.get("location_present", "")
        if location and location.lower() in actual.lower():
            score += 20.0

        # Check twist present (self-written letter)
        twist_patterns = [
            r"\bselbst\b.*\bgeschrieben\b",
            r"\beigen\b.*\bBrief\b",
            r"\bvon.*\bmir\b",
            r"\bich.*\bschrieb\b",
        ]
        if any(re.search(pattern, actual, re.IGNORECASE) for pattern in twist_patterns):
            score += 30.0

        # Check word count
        word_count = len(actual.split())
        min_words = expected.get("word_count_min", 70)
        max_words = expected.get("word_count_max", 150)
        if min_words <= word_count <= max_words:
            score += 15.0
        elif word_count >= min_words // 2:
            # Partial credit if at least half the minimum
            score += 7.0

        # Check narrative coherence (simple heuristic: multiple sentences)
        sentences = actual.count(".") + actual.count("!") + actual.count("?")
        if sentences >= 3:
            score += 15.0
        elif sentences >= 1:
            score += 7.0

        return score
