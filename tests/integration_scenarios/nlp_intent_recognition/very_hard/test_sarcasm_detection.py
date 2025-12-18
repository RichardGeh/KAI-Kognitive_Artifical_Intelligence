"""
tests/integration_scenarios/nlp_intent_recognition/very_hard/test_sarcasm_detection.py

Detect sarcasm and irony in statements where literal meaning differs from intent.
Requires understanding context, tone markers, and common sarcastic patterns.

Expected Reasoning: sentiment, non_literal, context, pragmatics
Success Criteria:
- Reasoning Quality >= 30% (contextual analysis)
- Confidence Calibration >= 30%
- Correctness >= 30% (sarcasm detected)
- Overall >= 30%
"""

import re
from typing import Dict

from tests.integration_scenarios.utils.scenario_base import ScenarioTestBase


class TestSarcasmDetection(ScenarioTestBase):
    """Test: Sarcasm Detection - Very Hard NLP Intent Recognition"""

    # REQUIRED: Class attributes
    DIFFICULTY = "very_hard"
    DOMAIN = "nlp_intent_recognition"
    TIMEOUT_SECONDS = 3600  # 1 hour

    # REQUIRED: Scoring weights
    REASONING_QUALITY_WEIGHT = 0.5
    CONFIDENCE_CALIBRATION_WEIGHT = 0.2
    CORRECTNESS_WEIGHT = 0.3

    def test_sarcasm_detection(
        self,
        kai_worker_scenario_mode,
        progress_reporter,
        scenario_logger,
        confidence_tracker,
    ):
        """
        Test: Detect sarcasm in multiple statements.

        Examples:
        1. "Toll, schon wieder Regen. Genau was ich wollte."
        2. "Das hat ja super funktioniert!" (nach Fehler)
        3. "Wie nett von dir, mich zu warten - nur drei Stunden."

        Must identify sarcastic intent vs. literal meaning.
        """

        # Setup progress tracking
        progress_reporter.total_steps = 5
        progress_reporter.start()

        # Define test input (German language, ASCII only)
        input_text = """
        Analysiere die folgenden Aussagen auf Sarkasmus oder Ironie:

        Aussage 1: "Toll, schon wieder Montag. Ich liebe Montage so sehr!"
        Kontext: Person sagt dies muede am Montagmorgen.

        Aussage 2: "Das Wetter ist heute wirklich herrlich."
        Kontext: Es regnet stark und ist kalt.

        Aussage 3: "Danke, dass du mich informiert hast. Nur eine Woche zu spaet."
        Kontext: Person erfaehrt wichtige Nachricht sehr verspaetet.

        Aussage 4: "Mein Auto ist kaputt. Das ist genau das, was ich heute brauchte."
        Kontext: Person hat ohnehin einen stressigen Tag.

        Fragen:
        1. Welche Aussagen sind sarkastisch oder ironisch?
        2. Was ist die tatsaechliche Bedeutung (nicht woertlich)?
        3. Welche Hinweise deuten auf Sarkasmus hin?

        Analysiere jede Aussage einzeln.
        """

        # Define expected outputs
        expected_outputs = {
            "sarcastic_statements": [1, 2, 3, 4],  # All are sarcastic
            "literal_different_from_intent": True,
            "context_considered": True,
            "tone_markers_identified": True,
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

        # Domain-specific assertions - check for sentiment/context strategies
        nlp_strategies = [
            "sentiment",
            "context",
            "pragmatic",
            "non_literal",
            "irony",
            "sarcasm",
            "intent",
        ]
        found_strategy = any(
            any(ns in s.lower() for ns in nlp_strategies)
            for s in result.strategies_used
        )

        # NLP may not show explicit strategies, check response content
        if not found_strategy:
            # Check if response discusses sarcasm/irony
            has_sarcasm_analysis = any(
                marker in result.kai_response.lower()
                for marker in ["sarkasm", "ironisch", "tatsaech", "gemeint"]
            )
            assert has_sarcasm_analysis, "Expected sarcasm analysis or NLP strategy"

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
        Custom correctness scoring for sarcasm detection.

        Partial credit for:
        - Identifying statement 1 as sarcastic (15 points)
        - Identifying statement 2 as sarcastic (15 points)
        - Identifying statement 3 as sarcastic (15 points)
        - Identifying statement 4 as sarcastic (15 points)
        - Explaining actual meaning vs. literal (20 points)
        - Identifying tone markers/context (20 points)
        """
        score = 0.0

        # Check if sarcasm identified for each statement
        # Statement 1: Montag
        if re.search(
            r"\b1\b.*\bsarkastisch|ironisch\b.*\b1\b|Montag.*\bsarkastisch\b",
            actual,
            re.IGNORECASE,
        ):
            score += 15.0

        # Statement 2: Wetter
        if re.search(
            r"\b2\b.*\bsarkastisch|ironisch\b.*\b2\b|Wetter.*\bsarkastisch|ironisch\b",
            actual,
            re.IGNORECASE,
        ):
            score += 15.0

        # Statement 3: Informiert
        if re.search(
            r"\b3\b.*\bsarkastisch|ironisch\b.*\b3\b|informiert.*\bsarkastisch|ironisch\b",
            actual,
            re.IGNORECASE,
        ):
            score += 15.0

        # Statement 4: Auto
        if re.search(
            r"\b4\b.*\bsarkastisch|ironisch\b.*\b4\b|Auto.*\bsarkastisch|ironisch\b",
            actual,
            re.IGNORECASE,
        ):
            score += 15.0

        # Check for explanation of actual meaning
        meaning_patterns = [
            r"\btatsaechlich\b",
            r"\bwahrheit\b",
            r"\bmeint\b",
            r"\bgemeint\b",
            r"\bgegenteil\b",
            r"\bnicht.*\bwoertlich\b",
        ]
        if any(
            re.search(pattern, actual, re.IGNORECASE) for pattern in meaning_patterns
        ):
            score += 20.0

        # Check for tone marker identification
        marker_patterns = [
            r"\bKontext\b",
            r"\bHinweis\b",
            r"\bUebertreibung\b",
            r"\bWiderspruch\b",
            r"\bkontrast\b",
        ]
        markers_found = sum(
            1
            for pattern in marker_patterns
            if re.search(pattern, actual, re.IGNORECASE)
        )
        if markers_found >= 2:
            score += 20.0
        elif markers_found >= 1:
            score += 10.0

        return score
