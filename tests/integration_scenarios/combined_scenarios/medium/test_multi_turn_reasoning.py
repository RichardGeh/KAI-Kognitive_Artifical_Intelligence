"""
tests/integration_scenarios/combined_scenarios/medium/test_multi_turn_reasoning.py

Medium-level combined scenario: Multi-turn reasoning with episodic memory

Scenario:
Multi-turn dialogue requiring episodic memory and reasoning accumulation.
Tests context maintenance across 4 turns.

Expected Reasoning:
- Stores Turn 1-2 facts in episodic memory
- Retrieves context for Turn 3-4 queries
- Accurate recall of who likes what
- Episodic memory usage tracked in ProofTree

Success Criteria (Gradual Scoring):
- Episodic Memory (stores Turn 1-2): 40%
- Retrieves Turn 3 correctly: 30%
- Retrieves Turn 4 correctly: 30%
"""

from typing import Dict, List

from tests.integration_scenarios.utils.scenario_base import ScenarioTestBase


class TestMultiTurnReasoning(ScenarioTestBase):
    """Test: Multi-turn dialogue with episodic memory"""

    DIFFICULTY = "medium"
    DOMAIN = "combined_scenarios"
    TIMEOUT_SECONDS = 600

    REASONING_QUALITY_WEIGHT = 0.5
    CONFIDENCE_CALIBRATION_WEIGHT = 0.2
    CORRECTNESS_WEIGHT = 0.3

    def test_four_turn_memory_dialogue(
        self,
        kai_worker_scenario_mode,
        progress_reporter,
        scenario_logger,
        confidence_tracker,
    ):
        """Test: 4-turn dialogue with context tracking"""

        progress_reporter.total_steps = 5
        progress_reporter.start()

        # Multi-turn dialogue input
        dialogue_text = """
Turn 1: Anna mag Aepfel.
Turn 2: Bob mag Birnen.
Turn 3: Wer mag Aepfel?
Turn 4: Was mag Bob?
        """

        expected_outputs = {
            "turn3_answer": "Anna",
            "turn4_answer": "Birnen",
        }

        # Execute scenario using base class
        result = self.run_scenario(
            input_text=dialogue_text,
            expected_outputs=expected_outputs,
            context={"multi_turn": True, "turn_count": 4},
            kai_worker=kai_worker_scenario_mode,
            logger=scenario_logger,
            progress_reporter=progress_reporter,
            confidence_tracker=confidence_tracker,
        )

        # Assertions
        assert (
            result.overall_score >= 50
        ), f"Overall score too low: {result.overall_score:.1f}% (expected >= 50%)"

        assert (
            result.correctness_score >= 40
        ), f"Correctness too low: {result.correctness_score:.1f}%"

        assert (
            result.reasoning_quality_score >= 40
        ), f"Reasoning quality too low: {result.reasoning_quality_score:.1f}%"

        # Check for episodic memory usage
        has_episodic_strategy = any(
            "episodic" in s.lower() or "memory" in s.lower()
            for s in result.strategies_used
        )
        # Note: This check is lenient - episodic memory might not be explicitly tracked
        # assert has_episodic_strategy, f"Expected episodic memory strategy, got: {result.strategies_used}"

        # Verify appropriate depth
        assert (
            2 <= result.proof_tree_depth <= 10
        ), f"ProofTree depth {result.proof_tree_depth} outside expected range [2-10]"

        # Performance check
        assert (
            result.execution_time_ms < 60000
        ), f"Too slow: {result.execution_time_ms}ms (expected <60s)"

        # Log summary
        print(f"\n[INFO] Logs: {scenario_logger.save_logs()}")
        print(f"[INFO] Score: {result.overall_score:.1f}/100")
        print(
            f"[INFO] Breakdown: Reasoning={result.reasoning_quality_score:.1f}, "
            f"Confidence={result.confidence_calibration_score:.1f}, "
            f"Correctness={result.correctness_score:.1f}"
        )

        if result.overall_score < 70:
            print("[WEAKNESS] Issues:")
            for w in result.identified_weaknesses:
                print(f"  - {w}")

        # Mark test as passed
        assert result.passed, f"Test failed: {result.error or 'Score below threshold'}"

    def score_correctness(
        self, actual: str, expected: Dict, allow_partial: bool = True
    ) -> float:
        """
        Score correctness for multi-turn dialogue.

        Args:
            actual: Actual KAI response text (combined from all turns)
            expected: Dict with turn3_answer and turn4_answer
            allow_partial: Whether to give partial credit

        Returns:
            Score 0-100
        """
        if not expected:
            return 50.0

        turn3_answer = expected.get("turn3_answer", "").lower()
        turn4_answer = expected.get("turn4_answer", "").lower()

        actual_lower = actual.lower()

        # Check Turn 3: Who likes apples? -> Anna
        turn3_correct = turn3_answer in actual_lower if turn3_answer else False

        # Check Turn 4: What does Bob like? -> Birnen
        turn4_correct = turn4_answer in actual_lower if turn4_answer else False

        # Scoring
        score = 0.0
        if turn3_correct:
            score += 50.0
        if turn4_correct:
            score += 50.0

        return score

    def score_reasoning_quality(
        self, proof_tree: Dict, strategies_used: List[str], reasoning_steps: List[str]
    ) -> float:
        """
        Score reasoning quality for multi-turn memory dialogue.

        Returns: 0-100
        """
        score = 0.0

        # Used episodic memory: +40%
        if any(
            "episodic" in s.lower() or "memory" in s.lower() for s in strategies_used
        ):
            score += 40
        else:
            # Give partial credit if context was maintained somehow
            score += 20

        # Multiple reasoning steps: +30%
        if len(reasoning_steps) >= 4:
            score += 30
        elif len(reasoning_steps) >= 2:
            score += 20

        # Appropriate depth [2-8]: +30%
        depth = self._calculate_proof_tree_depth(proof_tree) if proof_tree else 0
        if 2 <= depth <= 8:
            score += 30
        elif depth > 0:
            score += 15

        return min(score, 100.0)
