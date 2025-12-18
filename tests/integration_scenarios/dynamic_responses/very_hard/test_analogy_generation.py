"""
tests/integration_scenarios/dynamic_responses/very_hard/test_analogy_generation.py

Generate relevant analogy for complex concept with explicit mapping.
Create creative but accurate analogy with clear source-target correspondence.

Expected Reasoning: analogy, mapping, creative, semantic_similarity
Success Criteria:
- Reasoning Quality >= 30% (analogy reasoning)
- Confidence Calibration >= 30%
- Correctness >= 30% (relevant analogy with mapping)
- Overall >= 30%
"""

import re
from typing import Dict

from tests.integration_scenarios.utils.scenario_base import ScenarioTestBase


class TestAnalogyGeneration(ScenarioTestBase):
    """Test: Analogy Generation - Very Hard Dynamic Response"""

    # REQUIRED: Class attributes
    DIFFICULTY = "very_hard"
    DOMAIN = "dynamic_responses"
    TIMEOUT_SECONDS = 3600  # 1 hour

    # REQUIRED: Scoring weights
    REASONING_QUALITY_WEIGHT = 0.4
    CONFIDENCE_CALIBRATION_WEIGHT = 0.2
    CORRECTNESS_WEIGHT = 0.4

    def test_analogy_generation(
        self,
        kai_worker_scenario_mode,
        progress_reporter,
        scenario_logger,
        confidence_tracker,
    ):
        """
        Test: Generate analogy for complex concept with mapping.

        Requirements:
        - Source domain (familiar concept)
        - Target domain (complex concept to explain)
        - Explicit mapping of key features
        - Explanation of how analogy helps understanding
        """

        # Setup progress tracking
        progress_reporter.total_steps = 5
        progress_reporter.start()

        # Define test input (German language, ASCII only)
        input_text = """
        Erstelle eine Analogie, um folgenden komplexen Begriff zu erklaeren:

        Zielbegriff: Das menschliche Immunsystem

        Anforderungen:
        1. Waehle einen vertrauten Bereich als Quelle der Analogie
           (z.B. Militaer, Gebaeude, Technologie)
        2. Beschreibe die Analogie klar und verstaendlich
        3. Zeige explizite Zuordnungen zwischen Quelle und Ziel:
           - Was entspricht den weissen Blutkoerperchen?
           - Was entspricht Krankheitserregern?
           - Was entspricht Antikoerpern?
        4. Erklaere, welche Aspekte die Analogie gut erfasst
        5. Nenne auch Grenzen der Analogie (was sie nicht erklaert)

        Erstelle die vollstaendige Analogie mit allen Zuordnungen.
        """

        # Define expected outputs
        expected_outputs = {
            "source_domain_present": True,
            "target_domain": "Immunsystem",
            "mappings_count": 3,  # At least 3 explicit mappings
            "explanation_present": True,
            "limitations_present": True,
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

        # Domain-specific assertions - check for analogy/mapping strategies
        analogy_strategies = [
            "analogy",
            "mapping",
            "creative",
            "semantic",
            "similarity",
            "production",
        ]
        found_strategy = any(
            any(ast in s.lower() for ast in analogy_strategies)
            for s in result.strategies_used
        )

        # Analogy may not show in strategies, check response quality
        if not found_strategy:
            # Check if response contains analogy markers
            has_analogy = any(
                marker in result.kai_response.lower()
                for marker in ["wie", "entspricht", "aehnlich", "vergleich"]
            )
            assert has_analogy, "Expected analogy response or analogy strategy"

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
        Custom correctness scoring for analogy generation.

        Partial credit for:
        - Target domain mentioned (Immunsystem) (15 points)
        - Source domain identified (20 points)
        - Explicit mappings (30 points, 10 per mapping)
        - Explanation of analogy (20 points)
        - Limitations mentioned (15 points)
        """
        score = 0.0

        # Check target domain mentioned
        target = expected.get("target_domain", "")
        if target and target.lower() in actual.lower():
            score += 15.0

        # Check source domain (look for common analogy sources)
        source_domains = [
            "militaer",
            "armee",
            "festung",
            "polizei",
            "sicherheit",
            "fabrik",
            "computer",
            "netzwerk",
        ]
        if any(domain in actual.lower() for domain in source_domains):
            score += 20.0

        # Check for explicit mappings (look for "entspricht", "ist wie", "=")
        mapping_patterns = [
            r"\bentspricht\b",
            r"\bist wie\b",
            r"\bvergleichbar mit\b",
            r"\b=\b",
            r"\baehnelt\b",
        ]
        mappings_found = sum(
            1
            for pattern in mapping_patterns
            if re.search(pattern, actual, re.IGNORECASE)
        )

        # Give 10 points per mapping, up to 30 points
        score += min(mappings_found * 10.0, 30.0)

        # Check for explanation
        explanation_patterns = [
            r"\bhilft zu verstehen\b",
            r"\berklaert\b",
            r"\bzeigt\b",
            r"\bveranschaulicht\b",
            r"\bdeshalb\b",
        ]
        if any(
            re.search(pattern, actual, re.IGNORECASE)
            for pattern in explanation_patterns
        ):
            score += 20.0

        # Check for limitations
        limitation_patterns = [
            r"\baber\b",
            r"\bjedoch\b",
            r"\bGrenze\b",
            r"\bbegrenzt\b",
            r"\bnicht.*\berklaert\b",
            r"\bunvollstaendig\b",
        ]
        if any(
            re.search(pattern, actual, re.IGNORECASE) for pattern in limitation_patterns
        ):
            score += 15.0

        return score
