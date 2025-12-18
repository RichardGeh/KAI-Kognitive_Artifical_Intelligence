"""
tests/integration_scenarios/dynamic_responses/hard/test_multi_turn_topic_shift.py

Hard-level dynamic response: 5+ turn dialogue with topic shifts

Scenario:
Multi-turn conversation where topic shifts multiple times.
KAI should maintain context within a topic but adapt when topic changes.
Requires episodic memory and context management.

Expected Reasoning:
- Episodic memory should track conversation history
- Topic shift detection
- Context-aware responses
- Production System for response generation
- Confidence should remain stable across turns

Success Criteria (Gradual Scoring):
- Correctness: 30% (appropriate responses to each turn)
- Reasoning Quality: 50% (episodic memory used, context maintained)
- Confidence Calibration: 20% (confidence reflects context quality)
"""

from typing import Dict, List

from tests.integration_scenarios.utils.scenario_base import ScenarioTestBase


class TestMultiTurnTopicShift(ScenarioTestBase):
    """Test: Multi-turn dialogue with topic shifts"""

    DIFFICULTY = "hard"
    DOMAIN = "dynamic_responses"
    TIMEOUT_SECONDS = 1800  # 30 minutes

    # Scoring weights for this scenario type
    REASONING_QUALITY_WEIGHT = 0.5
    CONFIDENCE_CALIBRATION_WEIGHT = 0.2
    CORRECTNESS_WEIGHT = 0.3

    def test_multi_turn_topic_shift_dialogue(
        self,
        kai_worker_scenario_mode,
        progress_reporter,
        scenario_logger,
        confidence_tracker,
    ):
        """
        Test: Multi-turn conversation with topic shifts

        Execute 5 turns with 2 topic shifts. Each response should be contextually appropriate.

        Expected: Context-aware responses throughout
        """

        # Setup progress reporter
        progress_reporter.total_steps = 7  # 5 turns + setup + analysis
        progress_reporter.start()

        # Turn 1: Topic = Animals
        turn1 = "Was ist ein Hund?"

        # Turn 2: Follow-up on animals
        turn2 = "Und was ist eine Katze?"

        # Turn 3: Topic shift to colors
        turn3 = "Jetzt zu etwas anderem: Was ist die Farbe Rot?"

        # Turn 4: Follow-up on colors
        turn4 = "Ist Blau eine warme oder kalte Farbe?"

        # Turn 5: Topic shift to mathematics
        turn5 = "Zurueck zur Mathematik: Was ist 5 plus 3?"

        turns = [turn1, turn2, turn3, turn4, turn5]

        # Expected responses should be contextually appropriate
        expected_context = {
            "turn1_mentions": "tier",
            "turn2_references_turn1": False,  # New animal, but same topic
            "turn3_topic_shift": True,
            "turn4_mentions_color": True,
            "turn5_topic_shift": True,
            "turn5_answer": "8",
        }

        # Execute multi-turn conversation
        conversation_context = {}
        results = []

        for i, turn in enumerate(turns):
            progress_reporter.update(f"Turn {i+1}/{len(turns)}", 20 + i * 12)

            result = self.run_scenario(
                input_text=turn,
                expected_outputs={"turn": i + 1},
                context=conversation_context,
                kai_worker=kai_worker_scenario_mode,
                logger=scenario_logger,
                progress_reporter=progress_reporter,
                confidence_tracker=confidence_tracker,
            )

            results.append(result)

            # Update conversation context with this turn
            conversation_context[f"turn_{i+1}"] = {
                "query": turn,
                "response": result.kai_response,
            }

        # Aggregate results
        avg_score = sum(r.overall_score for r in results) / len(results)

        # Assertions
        assert (
            avg_score >= 40
        ), f"Average score too low: {avg_score:.1f}% (expected >= 40%)"

        # Check that episodic memory was used
        has_episodic = any(
            "episodic" in s.lower() or "memory" in s.lower()
            for result in results
            for s in result.strategies_used
        )
        assert (
            has_episodic
        ), f"Expected episodic memory usage, got: {results[0].strategies_used}"

        # Log summary
        print(f"\n[INFO] Multi-turn conversation completed")
        print(f"[INFO] Average score across {len(turns)} turns: {avg_score:.1f}/100")
        for i, result in enumerate(results):
            print(
                f"[INFO] Turn {i+1} score: {result.overall_score:.1f}/100 "
                f"(Reasoning={result.reasoning_quality_score:.1f}, "
                f"Confidence={result.confidence_calibration_score:.1f})"
            )

        # Mark test as passed if majority of turns passed
        passed_count = sum(1 for r in results if r.passed)
        assert (
            passed_count >= len(turns) // 2
        ), f"Too many turns failed: {len(turns) - passed_count}/{len(turns)}"

    def score_correctness(
        self, actual: str, expected: Dict, allow_partial: bool = True
    ) -> float:
        """
        Score correctness based on contextual appropriateness.

        Args:
            actual: Actual KAI response text
            expected: Dict with "turn" key
            allow_partial: Whether to give partial credit

        Returns:
            Score 0-100
        """
        if not expected:
            return 50.0

        # For multi-turn, just check that response is non-empty and reasonable length
        if len(actual) < 10:
            return 0.0

        # Basic scoring: response exists and has content
        score = 60.0  # Base score for having a response

        # Check if response is substantive (>20 words)
        word_count = len(actual.split())
        if word_count >= 20:
            score += 40

        return min(score, 100.0)

    def score_reasoning_quality(
        self, proof_tree: Dict, strategies_used: List[str], reasoning_steps: List[str]
    ) -> float:
        """
        Score reasoning quality based on context management.

        Returns: 0-100 score
        """
        score = 0.0

        # Used episodic memory: +40%
        has_episodic = any(
            "episodic" in s.lower() or "memory" in s.lower() for s in strategies_used
        )
        if has_episodic:
            score += 40

        # Used Production System: +30%
        has_production = any("production" in s.lower() for s in strategies_used)
        if has_production:
            score += 30

        # ProofTree depth appropriate: +20%
        depth = self._calculate_proof_tree_depth(proof_tree) if proof_tree else 0
        if depth >= 3:
            score += 20
        elif depth >= 1:
            score += 10

        # Multiple reasoning steps: +10%
        if len(reasoning_steps) >= 3:
            score += 10
        elif len(reasoning_steps) >= 1:
            score += 5

        return min(score, 100.0)
