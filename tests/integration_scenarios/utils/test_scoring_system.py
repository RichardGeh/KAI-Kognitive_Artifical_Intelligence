"""
tests/integration_scenarios/utils/test_scoring_system.py

Unit tests for scoring_system.py module.
Tests all scoring functions with known inputs/outputs.

NO Unicode characters - ASCII only (see CLAUDE.md encoding rules).
"""

from tests.integration_scenarios.utils.scoring_system import (
    calculate_calibration_error,
    score_partial_correctness,
    score_proof_tree_quality,
    score_reasoning_coherence,
)


class TestScoreProofTreeQuality:
    """Test score_proof_tree_quality function"""

    def test_empty_proof_tree(self):
        """Test scoring with empty proof tree"""
        score, obs = score_proof_tree_quality({}, (2, 10), [], "logic_puzzles")
        assert score == 10.0  # Minimal score for empty tree
        assert any("No proof tree" in o for o in obs)

    def test_good_depth_tree(self):
        """Test scoring with appropriate depth"""
        tree = {
            "step": "root",
            "children": [
                {
                    "step": "child1",
                    "children": [{"step": "leaf1"}, {"step": "leaf2"}],
                }
            ],
        }
        score, obs = score_proof_tree_quality(tree, (2, 5), [], "logic_puzzles")
        assert score > 40.0  # Should get full depth score (depth=3)
        assert any("depth: 3" in o.lower() for o in obs)

    def test_shallow_tree_penalty(self):
        """Test that shallow trees are penalized"""
        tree = {"step": "root"}  # Depth 1
        score, obs = score_proof_tree_quality(tree, (3, 10), [], "logic_puzzles")
        assert score < 40.0  # Should be penalized for shallow depth
        assert any("too shallow" in o.lower() for o in obs)

    def test_deep_tree_penalty(self):
        """Test that excessively deep trees are penalized"""
        # Create a very deep tree (depth 15)
        tree = {"step": "root"}
        current = tree
        for i in range(14):
            current["children"] = [{"step": f"level{i}"}]
            current = current["children"][0]

        score, obs = score_proof_tree_quality(tree, (2, 10), [], "logic_puzzles")
        assert score < 60.0  # Should be penalized for excessive depth
        assert any("deeper than expected" in o.lower() for o in obs)

    def test_strategy_extraction(self):
        """Test that strategies are extracted from tree"""
        tree = {
            "step": "root",
            "strategy": "logic_engine",
            "children": [{"step": "child1", "strategy": "graph_traversal"}],
        }
        score, obs = score_proof_tree_quality(tree, (2, 10), [], "logic_puzzles")
        strategies_obs = [o for o in obs if "Strategies used" in o][0]
        assert "logic_engine" in strategies_obs
        assert "graph_traversal" in strategies_obs

    def test_missing_required_strategies(self):
        """Test penalty for missing required strategies"""
        tree = {"step": "root", "strategy": "logic_engine"}
        score_with_req, obs = score_proof_tree_quality(
            tree, (1, 10), ["constraint_solver"], "logic_puzzles"
        )
        score_without_req, _ = score_proof_tree_quality(
            tree, (1, 10), [], "logic_puzzles"
        )

        assert score_with_req < score_without_req  # Penalty applied
        assert any("Missing expected" in o for o in obs)


class TestScoreReasoningCoherence:
    """Test score_reasoning_coherence function"""

    def test_empty_steps(self):
        """Test with no reasoning steps"""
        score, issues = score_reasoning_coherence([], {})
        assert score == 20.0  # Minimal score
        assert any("No reasoning steps" in i for i in issues)

    def test_good_coherence(self):
        """Test with diverse, reasonable steps"""
        steps = [
            "Step 1: Identify constraints",
            "Step 2: Analyze first constraint",
            "Step 3: Deduce value for entity A",
            "Step 4: Propagate to entity B",
            "Step 5: Verify solution",
        ]
        score, issues = score_reasoning_coherence(steps, {})
        assert score > 80.0  # Should score high
        assert any("coherent" in i.lower() for i in issues)

    def test_circular_reasoning_penalty(self):
        """Test detection of repeated steps"""
        steps = [
            "Check constraint A",
            "Check constraint A",
            "Check constraint A",  # Repeated 3 times
        ]
        score, issues = score_reasoning_coherence(steps, {})
        assert score < 80.0  # Should be penalized
        assert any("circular" in i.lower() or "repeated" in i.lower() for i in issues)

    def test_few_steps_penalty(self):
        """Test penalty for too few steps"""
        steps = ["Single step"]
        score, issues = score_reasoning_coherence(steps, {})
        assert score < 80.0
        assert any("few" in i.lower() for i in issues)

    def test_many_steps_penalty(self):
        """Test penalty for excessive steps"""
        steps = [f"Step {i}" for i in range(150)]  # 150 steps
        score, issues = score_reasoning_coherence(steps, {})
        assert score < 90.0
        assert any("Excessive" in i for i in issues)

    def test_trivial_steps_penalty(self):
        """Test penalty for empty/trivial steps"""
        steps = ["Real step", "", "   ", "x", "Another real step"]
        score, issues = score_reasoning_coherence(steps, {})
        assert any("trivial" in i.lower() or "empty" in i.lower() for i in issues)


class TestCalculateCalibrationError:
    """Test calculate_calibration_error function"""

    def test_perfect_calibration(self):
        """Test with perfect calibration"""
        # 80% confidence, 80% correct
        confidences = [0.8] * 10
        correctness = [True] * 8 + [False] * 2
        metrics = calculate_calibration_error(confidences, correctness)

        assert metrics["ece"] < 0.1  # Very low calibration error
        assert metrics["brier_score"] < 0.2

    def test_overconfident(self):
        """Test with overconfident predictions"""
        # High confidence but low accuracy
        confidences = [0.9] * 10
        correctness = [True] * 5 + [False] * 5  # Only 50% correct
        metrics = calculate_calibration_error(confidences, correctness)

        assert metrics["ece"] > 0.3  # High calibration error
        assert metrics["mce"] > 0.3

    def test_underconfident(self):
        """Test with underconfident predictions"""
        # Low confidence but high accuracy
        confidences = [0.5] * 10
        correctness = [True] * 9 + [False] * 1  # 90% correct
        metrics = calculate_calibration_error(confidences, correctness)

        assert metrics["ece"] > 0.3  # High calibration error (underconfident)

    def test_empty_input(self):
        """Test with empty inputs"""
        metrics = calculate_calibration_error([], [])
        assert metrics["ece"] == 0.5  # Default neutral value
        assert metrics["mce"] == 0.5

    def test_mismatched_lengths(self):
        """Test with mismatched input lengths"""
        confidences = [0.8, 0.9, 0.7]
        correctness = [True, False]  # Shorter
        metrics = calculate_calibration_error(confidences, correctness)

        # Should handle gracefully by truncating
        assert "ece" in metrics
        assert "brier_score" in metrics

    def test_brier_score_calculation(self):
        """Test Brier score calculation"""
        # All correct with high confidence
        confidences = [0.9, 0.8, 0.95]
        correctness = [True, True, True]
        metrics = calculate_calibration_error(confidences, correctness)

        assert metrics["brier_score"] < 0.1  # Low Brier = good

        # All wrong with high confidence
        confidences2 = [0.9, 0.8, 0.95]
        correctness2 = [False, False, False]
        metrics2 = calculate_calibration_error(confidences2, correctness2)

        assert metrics2["brier_score"] > 0.7  # High Brier = bad


class TestScorePartialCorrectness:
    """Test score_partial_correctness function"""

    def test_logic_puzzle_full_correct(self):
        """Test logic puzzle with all entities correct"""
        actual = "Alex: Teacher, Bob: Doctor, Carol: Engineer"
        expected = {"Alex": "Teacher", "Bob": "Doctor", "Carol": "Engineer"}
        score, explanation = score_partial_correctness(
            actual, expected, "logic_puzzles"
        )

        assert score == 100.0
        assert "3/3" in explanation

    def test_logic_puzzle_partial_correct(self):
        """Test logic puzzle with some entities correct"""
        actual = "Alex: Teacher, Bob: Engineer, Carol: Engineer"
        expected = {"Alex": "Teacher", "Bob": "Doctor", "Carol": "Engineer"}
        score, explanation = score_partial_correctness(
            actual, expected, "logic_puzzles"
        )

        assert 60.0 < score < 70.0  # 2/3 correct
        assert "2/3" in explanation

    def test_logic_puzzle_none_correct(self):
        """Test logic puzzle with no entities correct"""
        actual = "Random text"
        expected = {"Alex": "Teacher", "Bob": "Doctor"}
        score, explanation = score_partial_correctness(
            actual, expected, "logic_puzzles"
        )

        assert score == 0.0
        assert "0/2" in explanation

    def test_dynamic_response_key_facts(self):
        """Test dynamic response with key facts"""
        actual = "A dog is an animal. Dogs can bark. They are loyal."
        expected = {"key_facts": ["animal", "bark", "loyal"]}
        score, explanation = score_partial_correctness(
            actual, expected, "dynamic_responses"
        )

        assert score == 100.0  # All facts included
        assert "3/3" in explanation

    def test_dynamic_response_answer(self):
        """Test dynamic response with expected answer"""
        actual = "The answer is Paris."
        expected = {"answer": "Paris"}
        score, explanation = score_partial_correctness(
            actual, expected, "dynamic_responses"
        )

        assert score == 100.0
        assert "answer found" in explanation.lower()

    def test_nlp_intent_correct(self):
        """Test NLP intent recognition - correct"""
        actual = "Detected intent: QUESTION"
        expected = {"intent": "QUESTION"}
        score, explanation = score_partial_correctness(
            actual, expected, "nlp_intent_recognition"
        )

        assert score == 100.0
        assert "intent" in explanation.lower()

    def test_nlp_intent_incorrect(self):
        """Test NLP intent recognition - incorrect"""
        actual = "Detected intent: COMMAND"
        expected = {"intent": "QUESTION"}
        score, explanation = score_partial_correctness(
            actual, expected, "nlp_intent_recognition"
        )

        assert score == 0.0
        assert "not found" in explanation.lower()

    def test_combined_scenario(self):
        """Test combined scenario with multiple criteria"""
        actual = "Alex: Teacher. Dogs are animals."
        expected = {"entities": {"Alex": "Teacher"}, "key_facts": ["animal"]}
        score, explanation = score_partial_correctness(
            actual, expected, "combined_scenarios"
        )

        assert score == 100.0  # Both criteria met
        assert "Combined" in explanation

    def test_generic_string_match(self):
        """Test generic string matching"""
        actual = "The sky is blue"
        expected = "blue"
        score, explanation = score_partial_correctness(actual, expected, "generic")

        assert score == 100.0
        assert "found" in explanation.lower()

    def test_generic_partial_match(self):
        """Test generic partial match"""
        actual = "The sky is blue"
        expected = "blue beautiful sky"
        score, explanation = score_partial_correctness(actual, expected, "generic")

        assert score == 50.0  # Partial match
        assert "Partial" in explanation

    def test_no_expected_output(self):
        """Test with no expected output"""
        actual = "Some output"
        expected = {}
        score, explanation = score_partial_correctness(
            actual, expected, "logic_puzzles"
        )

        assert score == 50.0  # Neutral
        assert "No expected" in explanation
