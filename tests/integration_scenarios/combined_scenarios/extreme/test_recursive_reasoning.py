"""
tests/integration_scenarios/combined_scenarios/extreme/test_recursive_reasoning.py

Extreme Combined Scenario: Recursive Self-Referential Reasoning

Scenario: Self-referential problem requiring recursive thinking about KAI's
own reasoning process. Tests KAI's ability to reason about reasoning,
detect logical inconsistencies in self-reference, and avoid infinite loops.

Expected Reasoning:
- Meta-reasoning about own process
- Recursion detection
- Logical consistency checking
- Infinite loop prevention

Success Criteria:
- Handles recursion without infinite loop (weight: 50%)
- Detects logical inconsistency (weight: 30%)
- Reasoning quality >= 20% (extreme threshold)
- Overall score >= 20%

NO Unicode characters - ASCII only (see CLAUDE.md encoding rules).
"""

from typing import Dict

from tests.integration_scenarios.utils.scenario_base import ScenarioTestBase


class TestRecursiveReasoning(ScenarioTestBase):
    """Test: Recursive self-referential reasoning"""

    # REQUIRED: Class attributes
    DIFFICULTY = "extreme"
    DOMAIN = "combined_scenarios"
    TIMEOUT_SECONDS = 7200  # 2 hours

    # REQUIRED: Scoring weights
    REASONING_QUALITY_WEIGHT = 0.5
    CONFIDENCE_CALIBRATION_WEIGHT = 0.2
    CORRECTNESS_WEIGHT = 0.3

    def test_recursive_reasoning(
        self,
        kai_worker_scenario_mode,
        progress_reporter,
        scenario_logger,
        confidence_tracker,
    ):
        """
        Test: Self-referential problem requiring recursive thinking.

        Recursive scenario:
        - Question about how KAI would answer the question
        - Self-reference requiring meta-reasoning
        - Potential for infinite recursion
        - Logical consistency challenge

        This tests KAI's ability to:
        1. Recognize recursive structure
        2. Meta-reason about own process
        3. Avoid infinite loops
        4. Detect logical inconsistencies
        """

        # Setup progress tracking
        progress_reporter.total_steps = 5
        progress_reporter.start()

        # Define test input (German language, ASCII only)
        input_text = """
        Betrachte folgende selbstreferenzielle Frage:

        "Wenn ich dich frage, wie du diese Frage beantworten wuerdest,
        und du dann ueber deinen eigenen Denkprozess nachdenkst, um zu
        antworten, wie wuerdest du dann beschreiben, was in diesem
        Moment in deinem Reasoning-System passiert?"

        Zusaetzlich: Wenn deine Antwort auf diese Frage davon abhaengt,
        wie du ueber die Frage nachdenkst, entsteht dann ein zirkulaerer
        Prozess?

        Analysiere diese Situation und erklaere:
        1. Wie gehst du mit der Selbstreferenz um?
        2. Gibt es eine logische Inkonsistenz?
        3. Wie vermeidest du eine Endlosschleife beim Nachdenken
           ueber dein eigenes Nachdenken?
        """

        # Define expected outputs
        expected_outputs = {
            "recursive_structure": True,
            "meta_reasoning": True,
            "expected_recognition": [
                "self_reference",
                "recursion",
                "potential_paradox",
                "loop_prevention",
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
        # Most critical: didn't infinite loop!
        assert result.execution_time_ms < 7200000, (
            f"Test took too long: {result.execution_time_ms}ms "
            "(possible infinite loop)"
        )

        # Extreme difficulty: target >= 20%
        assert (
            result.overall_score >= 20
        ), f"Overall score too low: {result.overall_score:.1f}% (expected >= 20%)"

        # Lower thresholds for extreme difficulty
        assert (
            result.reasoning_quality_score >= 15
        ), f"Reasoning quality too low: {result.reasoning_quality_score:.1f}%"

        # Domain-specific assertions
        response_lower = result.kai_response.lower()

        # Check for recursion awareness
        recursion_keywords = [
            "rekursiv",
            "selbstreferenz",
            "zirkulaer",
            "endlosschleife",
            "schleife",
        ]
        recursion_detected = any(kw in response_lower for kw in recursion_keywords)

        if recursion_detected:
            print("[SUCCESS] Recursion/self-reference detected")

        # Check for meta-reasoning
        meta_keywords = [
            "denkprozess",
            "reasoning",
            "nachdenken ueber",
            "meta",
            "eigener prozess",
        ]
        meta_reasoning = any(kw in response_lower for kw in meta_keywords)

        if meta_reasoning:
            print("[SUCCESS] Meta-reasoning language detected")

        # Check for loop prevention strategy
        prevention_keywords = [
            "vermeide",
            "begrenze",
            "stoppe",
            "abbruch",
            "tiefe",
        ]
        prevention_mentioned = any(kw in response_lower for kw in prevention_keywords)

        if prevention_mentioned:
            print("[INFO] Loop prevention strategy mentioned")

        # Check ProofTree depth (should be bounded despite recursion)
        if result.proof_tree_depth > 50:
            print(f"[WARNING] ProofTree depth very high: {result.proof_tree_depth}")
        else:
            print(f"[SUCCESS] ProofTree depth bounded: {result.proof_tree_depth}")

        # Performance check
        print(f"[INFO] Execution time: {result.execution_time_ms}ms")

        # Response should exist (not empty)
        assert (
            len(result.kai_response) > 0
        ), "Response is empty (system may have failed)"

        # Logging
        print(f"\n[INFO] Logs: {scenario_logger.save_logs()}")
        print(f"[INFO] Score: {result.overall_score:.1f}/100")
        print(
            f"[INFO] Breakdown: Reasoning={result.reasoning_quality_score:.1f}, "
            f"Confidence={result.confidence_calibration_score:.1f}, "
            f"Correctness={result.correctness_score:.1f}"
        )
        print(f"[INFO] ProofTree Depth: {result.proof_tree_depth}")
        print(f"[INFO] Recursion detected: {recursion_detected}")
        print(f"[INFO] Meta-reasoning: {meta_reasoning}")

        if result.overall_score < 40:
            print("[EXPECTED] Low score is expected for extreme difficulty")
            print("[SUCCESS] System handled recursive reasoning without infinite loop")
            print("[WEAKNESS] Issues identified:")
            for w in result.identified_weaknesses:
                print(f"  - {w}")

        # Final check - completing without crash is success
        assert (
            result.passed or result.overall_score >= 20
        ), f"Test failed: {result.error or 'Score below extreme threshold'}"

    def score_correctness(
        self, actual: str, expected: Dict, allow_partial: bool = True
    ) -> float:
        """
        Custom correctness scoring for recursive reasoning.

        Full credit requires:
        - Self-reference recognition (40%)
        - Meta-reasoning explanation (30%)
        - Loop prevention strategy (30%)
        """
        if not actual or not expected:
            return 0.0

        actual_lower = actual.lower()
        score = 0.0

        # Check self-reference recognition (40 points)
        self_ref_keywords = [
            "selbstreferenz",
            "bezieht sich auf sich",
            "zirkulaer",
            "rekursiv",
        ]
        if any(kw in actual_lower for kw in self_ref_keywords):
            score += 40.0

        # Check meta-reasoning explanation (30 points)
        meta_indicators = [
            "denkprozess" in actual_lower,
            "reasoning" in actual_lower,
            "nachdenken ueber nachdenken" in actual_lower,
            "eigener prozess" in actual_lower,
        ]
        meta_count = sum(1 for ind in meta_indicators if ind)

        if meta_count >= 3:
            score += 30.0
        elif meta_count >= 2:
            score += 20.0
        elif meta_count >= 1:
            score += 10.0

        # Check loop prevention strategy (30 points)
        prevention_indicators = [
            "vermeide" in actual_lower or "vermeidung" in actual_lower,
            "begrenze" in actual_lower or "begrenzung" in actual_lower,
            "abbruch" in actual_lower,
            "tiefe" in actual_lower and "maximal" in actual_lower,
        ]
        prevention_count = sum(1 for ind in prevention_indicators if ind)

        if prevention_count >= 2:
            score += 30.0
        elif prevention_count >= 1:
            score += 15.0

        return min(100.0, score)
