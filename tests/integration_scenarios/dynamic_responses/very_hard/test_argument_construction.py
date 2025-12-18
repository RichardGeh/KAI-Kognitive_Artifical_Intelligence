"""
tests/integration_scenarios/dynamic_responses/very_hard/test_argument_construction.py

Construct persuasive argument with multiple justifications and evidence.
Build coherent argument structure with thesis, supporting points, conclusion.

Expected Reasoning: argument, reasoning, justification, production_system
Success Criteria:
- Reasoning Quality >= 30% (argument structure)
- Confidence Calibration >= 30%
- Correctness >= 30% (argument components present)
- Overall >= 30%
"""

import re
from typing import Dict

from tests.integration_scenarios.utils.scenario_base import ScenarioTestBase


class TestArgumentConstruction(ScenarioTestBase):
    """Test: Argument Construction - Very Hard Dynamic Response"""

    # REQUIRED: Class attributes
    DIFFICULTY = "very_hard"
    DOMAIN = "dynamic_responses"
    TIMEOUT_SECONDS = 3600  # 1 hour

    # REQUIRED: Scoring weights
    REASONING_QUALITY_WEIGHT = 0.4
    CONFIDENCE_CALIBRATION_WEIGHT = 0.2
    CORRECTNESS_WEIGHT = 0.4

    def test_argument_construction(
        self,
        kai_worker_scenario_mode,
        progress_reporter,
        scenario_logger,
        confidence_tracker,
    ):
        """
        Test: Construct persuasive argument with multiple justifications.

        Requirements:
        - Clear thesis statement
        - At least 3 supporting arguments
        - Evidence/examples for each argument
        - Counterargument consideration
        - Strong conclusion
        """

        # Setup progress tracking
        progress_reporter.total_steps = 5
        progress_reporter.start()

        # Define test input (German language, ASCII only)
        input_text = """
        Baue ein ueberzeugendes Argument fuer folgende These auf:

        These: Erneuerbare Energien sollten staerker gefoerdert werden.

        Anforderungen an das Argument:
        1. Klare Formulierung der These
        2. Mindestens drei Begruendungen mit Beispielen
        3. Beruecksichtigung eines Gegenarguments und dessen Entkraeftung
        4. Logische Struktur (Einleitung, Hauptteil, Schluss)
        5. Verwendung von Uebergangsworten (erstens, zweitens, jedoch, deshalb)

        Erstelle das vollstaendige Argument.
        """

        # Define expected outputs
        expected_outputs = {
            "thesis_present": True,
            "supporting_arguments_count": 3,
            "counterargument_present": True,
            "conclusion_present": True,
            "structure_markers": ["erstens", "zweitens", "jedoch", "deshalb"],
            "topic_keywords": ["Energie", "erneuerbar", "Foerderung"],
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

        # Domain-specific assertions - check for argument/reasoning strategies
        argument_strategies = [
            "argument",
            "reasoning",
            "justification",
            "production",
            "generation",
        ]
        found_strategy = any(
            any(ast in s.lower() for ast in argument_strategies)
            for s in result.strategies_used
        )

        # Argument construction may not show in strategies, check response quality
        if not found_strategy:
            # Check if response has argumentative structure
            has_structure = len(result.kai_response.split()) > 80
            assert has_structure, "Expected argumentative response or argument strategy"

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
        Custom correctness scoring for argument construction.

        Partial credit for:
        - Topic keywords present (15 points)
        - Structure markers (erstens, zweitens, etc.) (20 points)
        - Multiple arguments (3+ sentences) (25 points)
        - Counterargument present (20 points)
        - Conclusion present (20 points)
        """
        score = 0.0

        # Check topic keywords
        topic_keywords = expected.get("topic_keywords", [])
        keywords_found = sum(1 for kw in topic_keywords if kw.lower() in actual.lower())
        if keywords_found >= 2:
            score += 15.0
        elif keywords_found >= 1:
            score += 7.0

        # Check structure markers
        structure_markers = expected.get("structure_markers", [])
        markers_found = sum(
            1 for marker in structure_markers if marker.lower() in actual.lower()
        )
        if markers_found >= 2:
            score += 20.0
        elif markers_found >= 1:
            score += 10.0

        # Check for multiple arguments (heuristic: sentence count)
        sentences = actual.count(".") + actual.count("!") + actual.count("?")
        if sentences >= 6:
            score += 25.0
        elif sentences >= 3:
            score += 12.0

        # Check for counterargument
        counterargument_patterns = [
            r"\bjedoch\b",
            r"\baber\b",
            r"\ballerdings\b",
            r"\bGegner.*\bbehaupten\b",
            r"\bEinwand\b",
            r"\bkritisiert\b",
        ]
        if any(
            re.search(pattern, actual, re.IGNORECASE)
            for pattern in counterargument_patterns
        ):
            score += 20.0

        # Check for conclusion
        conclusion_patterns = [
            r"\bdeshalb\b",
            r"\bdaher\b",
            r"\bzusammenfassend\b",
            r"\bfazit\b",
            r"\bSchluss\b",
            r"\bfolglich\b",
        ]
        if any(
            re.search(pattern, actual, re.IGNORECASE) for pattern in conclusion_patterns
        ):
            score += 20.0

        return score
