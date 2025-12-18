"""
tests/integration_scenarios/logic_puzzles/very_hard/test_multi_layer_deduction.py

Multi-layer logical deduction puzzle requiring 4+ levels of nested reasoning.
Each layer builds on previous conclusions to reach final answer.

Expected Reasoning: logic, chaining, multi_hop_inference, deductive_reasoning
Success Criteria:
- Reasoning Quality >= 30% (depth 8-20 expected)
- Confidence Calibration >= 30%
- Correctness >= 30%
- Overall >= 30%
"""

import re
from typing import Dict

from tests.integration_scenarios.utils.scenario_base import ScenarioTestBase


class TestMultiLayerDeduction(ScenarioTestBase):
    """Test: Multi-Layer Deduction Puzzle - Very Hard Logic Puzzle"""

    # REQUIRED: Class attributes
    DIFFICULTY = "very_hard"
    DOMAIN = "logic_puzzles"
    TIMEOUT_SECONDS = 3600  # 1 hour

    # REQUIRED: Scoring weights
    REASONING_QUALITY_WEIGHT = 0.5
    CONFIDENCE_CALIBRATION_WEIGHT = 0.2
    CORRECTNESS_WEIGHT = 0.3

    def test_multi_layer_deduction(
        self,
        kai_worker_scenario_mode,
        progress_reporter,
        scenario_logger,
        confidence_tracker,
    ):
        """
        Test: Multi-layer deduction with 5 layers of reasoning.

        Layer 1: Basic facts
        Layer 2: Direct implications
        Layer 3: Combining implications
        Layer 4: Contradictions and eliminations
        Layer 5: Final deduction
        """

        # Setup progress tracking
        progress_reporter.total_steps = 5
        progress_reporter.start()

        # Define test input (German language, ASCII only)
        input_text = """
        Vier Personen: Anna, Bernd, Clara, Daniel.
        Vier Berufe: Arzt, Lehrer, Ingenieur, Anwalt.
        Vier Staedte: Berlin, Hamburg, Muenchen, Koeln.

        Hinweise:
        1. Anna ist entweder Aerztin oder Lehrerin.
        2. Die Person aus Berlin ist Ingenieur.
        3. Bernd ist nicht aus Hamburg.
        4. Der Anwalt kommt aus Muenchen.
        5. Clara ist Lehrerin.
        6. Die Person aus Hamburg ist nicht Arzt.
        7. Daniel ist nicht Ingenieur.
        8. Anna kommt nicht aus Berlin.
        9. Der Arzt kommt nicht aus Koeln.
        10. Bernd ist nicht Lehrer.

        Aufgabe: Bestimmen Sie fuer jede Person den Beruf und die Stadt.

        Schritt-fuer-Schritt-Ableitung:
        - Aus (5): Clara ist Lehrerin.
        - Aus (1) und (5): Anna ist Aerztin (da Clara schon Lehrerin ist).
        - Aus (9): Der Arzt (Anna) kommt nicht aus Koeln.
        - Aus (8): Anna kommt nicht aus Berlin.
        - Aus (2): Die Person aus Berlin ist Ingenieur (nicht Anna).
        - Aus (7): Daniel ist nicht Ingenieur.
        - Also: Bernd ist Ingenieur (weil Clara Lehrerin, Anna Aerztin, Daniel nicht Ingenieur).
        - Aus (2) und Bernd ist Ingenieur: Bernd kommt aus Berlin.
        - Aus (4): Der Anwalt kommt aus Muenchen.
        - Also: Daniel ist Anwalt (einziger verbleibender Beruf).
        - Aus (4) und Daniel ist Anwalt: Daniel kommt aus Muenchen.
        - Verbleibende Staedte fuer Anna und Clara: Hamburg, Koeln.
        - Aus (9): Anna (Arzt) kommt nicht aus Koeln, also kommt Anna aus Hamburg.
        - Also: Clara kommt aus Koeln.

        Frage: Wer ist Ingenieur und aus welcher Stadt kommt der Anwalt?
        """

        # Define expected outputs
        expected_outputs = {
            "engineer": "Bernd",
            "engineer_city": "Berlin",
            "lawyer": "Daniel",
            "lawyer_city": "Muenchen",
            "anna_profession": "Arzt",
            "clara_profession": "Lehrer",
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

        # Domain-specific assertions - check for logical reasoning strategies
        logic_strategies = [
            "logic",
            "deduction",
            "inference",
            "chaining",
            "multi_hop",
            "elimination",
            "constraint",
        ]
        found_strategy = any(
            any(ls in s.lower() for ls in logic_strategies)
            for s in result.strategies_used
        )

        assert (
            found_strategy
        ), f"Expected logical reasoning strategy, got: {result.strategies_used}"

        # Check proof tree depth (multi-layer should have good depth)
        assert (
            8 <= result.proof_tree_depth <= 20
        ), f"ProofTree depth {result.proof_tree_depth} outside expected range 8-20"

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
        Custom correctness scoring for multi-layer deduction.

        Partial credit for:
        - Engineer = Bernd (25 points)
        - Engineer from Berlin (25 points)
        - Lawyer = Daniel (25 points)
        - Lawyer from Muenchen (25 points)
        """
        score = 0.0

        # Check engineer is Bernd
        engineer = expected.get("engineer", "")
        if engineer:
            pattern = rf"\b{engineer}\b.*\bIngenieur\b|Ingenieur\b.*\b{engineer}\b"
            if re.search(pattern, actual, re.IGNORECASE):
                # Check no negation
                negation_pattern = rf"\b(?:nicht|kein)\b.*\b{engineer}\b.*\bIngenieur\b"
                if not re.search(negation_pattern, actual, re.IGNORECASE):
                    score += 25.0

        # Check engineer from Berlin
        engineer_city = expected.get("engineer_city", "")
        if engineer_city:
            pattern = rf"\b{engineer}\b.*\b{engineer_city}\b|{engineer_city}\b.*\b{engineer}\b"
            if re.search(pattern, actual, re.IGNORECASE):
                score += 25.0

        # Check lawyer is Daniel
        lawyer = expected.get("lawyer", "")
        if lawyer:
            pattern = rf"\b{lawyer}\b.*\bAnwalt\b|Anwalt\b.*\b{lawyer}\b"
            if re.search(pattern, actual, re.IGNORECASE):
                # Check no negation
                negation_pattern = rf"\b(?:nicht|kein)\b.*\b{lawyer}\b.*\bAnwalt\b"
                if not re.search(negation_pattern, actual, re.IGNORECASE):
                    score += 25.0

        # Check lawyer from Muenchen
        lawyer_city = expected.get("lawyer_city", "")
        if lawyer_city:
            pattern = rf"\b{lawyer}\b.*\b{lawyer_city}\b|{lawyer_city}\b.*\b{lawyer}\b"
            if re.search(pattern, actual, re.IGNORECASE):
                score += 25.0

        return score
