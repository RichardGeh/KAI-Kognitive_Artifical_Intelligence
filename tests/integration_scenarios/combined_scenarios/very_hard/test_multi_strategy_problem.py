"""
tests/integration_scenarios/combined_scenarios/very_hard/test_multi_strategy_problem.py

Crime investigation requiring abductive + probabilistic + constraint reasoning.
Multiple suspects, evidence, alibis - find most likely culprit.

Expected Reasoning: abductive, probabilistic, constraint, evidence_evaluation, hypothesis
Success Criteria:
- Reasoning Quality >= 30% (multi-strategy coordination)
- Confidence Calibration >= 30%
- Correctness >= 30%
- Overall >= 30%
"""

import re
from typing import Dict

from tests.integration_scenarios.utils.scenario_base import ScenarioTestBase


class TestMultiStrategyProblem(ScenarioTestBase):
    """Test: Multi-Strategy Crime Investigation - Very Hard Combined Scenario"""

    # REQUIRED: Class attributes
    DIFFICULTY = "very_hard"
    DOMAIN = "combined_scenarios"
    TIMEOUT_SECONDS = 3600  # 1 hour

    # REQUIRED: Scoring weights
    REASONING_QUALITY_WEIGHT = 0.5
    CONFIDENCE_CALIBRATION_WEIGHT = 0.2
    CORRECTNESS_WEIGHT = 0.3

    def test_multi_strategy_problem(
        self,
        kai_worker_scenario_mode,
        progress_reporter,
        scenario_logger,
        confidence_tracker,
    ):
        """
        Test: Crime investigation with multiple reasoning strategies.

        Requires:
        - Abductive reasoning (hypothesis generation)
        - Probabilistic reasoning (likelihood evaluation)
        - Constraint satisfaction (alibi checking)
        - Evidence evaluation
        """

        # Setup progress tracking
        progress_reporter.total_steps = 5
        progress_reporter.start()

        # Define test input (German language, ASCII only)
        input_text = """
        Kriminalfall - Finde den Taeter:

        Tatbestand:
        - Ein wertvolles Gemaelde wurde am Dienstag zwischen 20:00 und 22:00 Uhr gestohlen.
        - Der Diebstahl geschah im 3. Stock eines Gebaeudes.
        - Der Alarm wurde um 21:30 Uhr ausgeloest.
        - Es gibt keine Anzeichen von Gewaltanwendung.

        Verdaechtige:

        1. Anna (Museumsangestellte):
           - Hat Zugang zum Gebaeude.
           - Kennt das Alarmsystem.
           - Alibi: War von 19:00 bis 23:00 Uhr im Theater (2 Zeugen).
           - Motiv: Geldprobleme bekannt.

        2. Bernd (ehemaliger Mitarbeiter):
           - Wurde vor 6 Monaten entlassen.
           - Kennt das Gebaeude gut.
           - Alibi: War zu Hause (keine Zeugen).
           - Motiv: Rache wegen Entlassung.

        3. Clara (Kunsthaendlerin):
           - Kein direkter Zugang zum Gebaeude.
           - Hat grosses Interesse an diesem Gemaelde.
           - Alibi: War im Restaurant (Kellner als Zeuge, aber nur bis 20:30 Uhr).
           - Motiv: Hoher Wiederverkaufswert.

        4. Daniel (Sicherheitsdienst):
           - Hatte Schicht von 18:00 bis 22:00 Uhr.
           - Kontrollierte das Gebaeude um 20:00 Uhr.
           - Alibi: War die ganze Zeit im Dienst (aber allein im 3. Stock um 21:00 Uhr).
           - Motiv: Unklar, hat Schulden.

        Beweise:
        - Fingerabdruecke von Anna und Daniel am Tatort (normal, da beide dort arbeiten).
        - Ueberwachungsvideo zeigt Person in dunkler Kleidung um 21:15 Uhr.
        - Alarmcode wurde korrekt eingegeben (Insider-Wissen).

        Fragen:
        1. Wer ist der wahrscheinlichste Taeter?
        2. Welche Beweise stuetzen diese Hypothese?
        3. Welche Verdaechtigen koennen mit hoher Sicherheit ausgeschlossen werden?
        4. Welche zusaetzlichen Informationen waeren hilfreich?

        Analysiere den Fall mit systematischer Schlussfolgerung.
        """

        # Define expected outputs
        # Analysis:
        # - Anna: Strong alibi (2 witnesses, full time coverage) -> excluded
        # - Bernd: Weak alibi, motive, knowledge -> possible
        # - Clara: Partial alibi (only until 20:30), motive, no access -> less likely
        # - Daniel: Opportunity (alone at crime scene), insider knowledge, motive -> most likely
        expected_outputs = {
            "most_likely_culprit": "Daniel",
            "excluded_suspects": ["Anna"],
            "key_evidence": ["Insider-Wissen", "allein", "Gelegenheit"],
            "reasoning_types": ["abductive", "probabilistic", "constraint"],
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

        # Domain-specific assertions - check for multiple strategies
        investigation_strategies = [
            "abductive",
            "probabilistic",
            "constraint",
            "evidence",
            "hypothesis",
            "inference",
            "deduction",
        ]
        found_strategies = sum(
            1
            for strategy in result.strategies_used
            if any(inv in strategy.lower() for inv in investigation_strategies)
        )

        assert (
            found_strategies >= 2
        ), f"Expected multiple investigation strategies (>=2), got: {result.strategies_used}"

        # Check proof tree depth (complex multi-strategy problem)
        assert (
            8 <= result.proof_tree_depth <= 25
        ), f"ProofTree depth {result.proof_tree_depth} outside expected range 8-25"

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
        Custom correctness scoring for multi-strategy crime investigation.

        Partial credit for:
        - Identifies Daniel as most likely (40 points)
        - Excludes Anna (strong alibi) (20 points)
        - Mentions key evidence (opportunity, insider knowledge) (20 points)
        - Uses multiple reasoning types (20 points)
        """
        score = 0.0

        # Check most likely culprit
        culprit = expected.get("most_likely_culprit", "")
        if culprit:
            pattern = rf"\b{culprit}\b.*\bwahrscheinlich|Taeter|schuldig\b"
            if re.search(pattern, actual, re.IGNORECASE):
                score += 40.0

        # Check excluded suspects (Anna with strong alibi)
        excluded = expected.get("excluded_suspects", [])
        for suspect in excluded:
            if re.search(
                rf"\b{suspect}\b.*\bausgeschlossen|unwahrscheinlich\b",
                actual,
                re.IGNORECASE,
            ):
                score += 20.0
                break

        # Check key evidence mentioned
        key_evidence = expected.get("key_evidence", [])
        evidence_found = sum(
            1 for evidence in key_evidence if evidence.lower() in actual.lower()
        )
        if evidence_found >= 2:
            score += 20.0
        elif evidence_found >= 1:
            score += 10.0

        # Check for multiple reasoning types
        reasoning_types = expected.get("reasoning_types", [])
        reasoning_found = sum(
            1 for rtype in reasoning_types if rtype.lower() in actual.lower()
        )
        if reasoning_found >= 2:
            score += 20.0
        elif reasoning_found >= 1:
            score += 10.0

        return score
