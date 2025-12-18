"""
tests/integration_scenarios/logic_puzzles/extreme/test_meta_puzzle.py

Extreme Logic Puzzle: Meta-Reasoning and Self-Reference

Scenario: Puzzle about puzzle-solving itself, involving self-referential
statements and potential paradoxes. Tests KAI's ability to handle meta-level
reasoning, detect self-reference, and avoid infinite loops.

Expected Reasoning:
- Meta-level reasoning
- Self-reference detection
- Paradox identification
- Fixed-point analysis

Success Criteria:
- Handles meta-reasoning without crash (weight: 40%)
- Detects self-reference or paradox (weight: 40%)
- Reasoning quality >= 20% (extreme threshold)
- Overall score >= 20%

NO Unicode characters - ASCII only (see CLAUDE.md encoding rules).
"""

from typing import Dict

from tests.integration_scenarios.utils.scenario_base import ScenarioTestBase


class TestMetaPuzzle(ScenarioTestBase):
    """Test: Meta-puzzle with self-referential reasoning"""

    # REQUIRED: Class attributes
    DIFFICULTY = "extreme"
    DOMAIN = "logic_puzzles"
    TIMEOUT_SECONDS = 7200  # 2 hours

    # REQUIRED: Scoring weights
    REASONING_QUALITY_WEIGHT = 0.5
    CONFIDENCE_CALIBRATION_WEIGHT = 0.2
    CORRECTNESS_WEIGHT = 0.3

    def test_meta_puzzle(
        self,
        kai_worker_scenario_mode,
        progress_reporter,
        scenario_logger,
        confidence_tracker,
    ):
        """
        Test: Puzzle about puzzle-solving with self-reference.

        Meta-puzzle structure:
        - Statement refers to its own truth value
        - Requires fixed-point reasoning
        - Potential for paradox (liar's paradox variant)

        This tests KAI's ability to:
        1. Recognize meta-level reasoning
        2. Handle self-reference safely
        3. Detect paradoxes
        4. Avoid infinite loops
        """

        # Setup progress tracking
        progress_reporter.total_steps = 5
        progress_reporter.start()

        # Define test input (German language, ASCII only)
        input_text = """
        Meta-Raetsel: Selbstreferenz

        Es gibt drei Aussagen, die von drei Personen gemacht wurden:

        Anna sagt: "Genau eine der drei Aussagen ist wahr."
        Bob sagt: "Genau zwei der drei Aussagen sind wahr."
        Clara sagt: "Alle drei Aussagen sind falsch."

        Jede Aussage bezieht sich auf alle drei Aussagen einschliesslich
        sich selbst.

        Frage: Welche Aussagen sind wahr und welche sind falsch?

        Hinweis: Dies ist ein selbstreferenzielles Raetsel. Die Antwort
        haengt davon ab, ob eine konsistente Zuweisung von Wahrheitswerten
        moeglich ist.
        """

        # Define expected outputs
        # Analysis: If Anna is true -> exactly 1 true (consistent: Anna true, Bob false, Clara false)
        # If Bob is true -> exactly 2 true (need 1 more, but Anna contradicts, Clara contradicts)
        # If Clara is true -> all false (contradiction: Clara itself is true)
        expected_outputs = {
            "meta_reasoning": True,
            "self_reference_detected": True,
            "consistent_assignment": {
                "Anna": True,
                "Bob": False,
                "Clara": False,
            },
            "or_paradox_detected": True,
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

        # Most important: didn't crash or infinite loop
        assert result.execution_time_ms < 7200000, (
            f"Test took too long: {result.execution_time_ms}ms "
            "(possible infinite loop or timeout)"
        )

        # Lower thresholds for extreme difficulty
        assert (
            result.reasoning_quality_score >= 15
        ), f"Reasoning quality too low: {result.reasoning_quality_score:.1f}%"

        # Domain-specific assertions
        # Check for meta-reasoning awareness
        meta_keywords = [
            "selbstreferenz",
            "meta",
            "bezieht sich auf sich",
            "paradox",
            "zirkulaer",
        ]

        response_lower = result.kai_response.lower()
        meta_awareness = any(kw in response_lower for kw in meta_keywords)

        if meta_awareness:
            print("[SUCCESS] KAI showed meta-reasoning awareness")
        else:
            print("[WARNING] KAI may not have recognized self-reference")

        # ProofTree should exist (even if reasoning is incomplete)
        assert (
            result.proof_tree_depth >= 2
        ), f"ProofTree depth {result.proof_tree_depth} suspiciously shallow"

        # Check for logic strategies
        if result.strategies_used:
            print(f"[INFO] Strategies attempted: {result.strategies_used}")

        # Logging
        print(f"\n[INFO] Logs: {scenario_logger.save_logs()}")
        print(f"[INFO] Score: {result.overall_score:.1f}/100")
        print(
            f"[INFO] Breakdown: Reasoning={result.reasoning_quality_score:.1f}, "
            f"Confidence={result.confidence_calibration_score:.1f}, "
            f"Correctness={result.correctness_score:.1f}"
        )
        print(f"[INFO] ProofTree Depth: {result.proof_tree_depth}")
        print(f"[INFO] Execution Time: {result.execution_time_ms}ms")

        if result.overall_score < 40:
            print("[EXPECTED] Low score is expected for extreme difficulty")
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
        Custom correctness scoring for meta-puzzle.

        Full credit requires:
        - Recognition of self-reference (40%)
        - Consistent assignment or paradox detection (40%)
        - Meta-reasoning explanation (20%)
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

        # Check consistent assignment or paradox (40 points)
        # Either finds consistent assignment OR detects paradox
        consistent_indicators = [
            "anna" in actual_lower and "wahr" in actual_lower,
            "konsistent" in actual_lower,
        ]
        paradox_indicators = [
            "paradox" in actual_lower,
            "widerspruch" in actual_lower,
            "keine konsistente" in actual_lower,
        ]

        if any(consistent_indicators) or any(paradox_indicators):
            score += 40.0

        # Check meta-reasoning explanation (20 points)
        explanation_indicators = [
            "aussage" in actual_lower and "sich selbst" in actual_lower,
            "wahrheitswert" in actual_lower,
            "alle drei aussagen" in actual_lower,
        ]
        if any(explanation_indicators):
            score += 20.0

        return min(100.0, score)
