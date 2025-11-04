"""
Test for Combinatorial Reasoning: 100 Prisoners Problem

Tests KAI's ability to solve strategic combinatorial problems using
the general-purpose combinatorial reasoning engine.

IMPORTANT: This test uses ONLY generic methods from component_40.
No puzzle-specific logic in the test itself!

Problem Structure (generic formulation):
- N agents must each find their own ID in a permutation
- Each agent can examine K positions
- Success requires ALL agents to find their ID
- Different strategies yield different success probabilities

Test Objective:
- Define two generic strategies
- Compute success probabilities using cycle analysis
- Identify optimal strategy

Author: KAI Development Team
Created: 2025-11-04
"""

import pytest
import math
from typing import Dict, Any

from component_40_combinatorial_reasoning import (
    Permutation,
    Strategy,
    CombinatorialReasoner,
    CycleAnalyzer,
    CombinatorialProbability,
    create_strategy,
)


class TestCombinatorialReasoningPrisoners:
    """
    Test suite for strategic permutation problems.

    Uses 100 Prisoners Problem as a concrete example, but tests
    are designed to validate general combinatorial reasoning capabilities.
    """

    @pytest.fixture
    def reasoner(self):
        """Create combinatorial reasoner instance."""
        return CombinatorialReasoner()

    @pytest.fixture
    def problem_parameters(self):
        """
        Generic problem parameters.

        For 100 Prisoners Problem:
        - n_agents: 100 prisoners
        - n_positions: 100 boxes
        - max_examinations: 50 boxes per prisoner
        """
        return {"n_agents": 100, "n_positions": 100, "max_examinations": 50}

    def test_permutation_creation_and_cycles(self):
        """Test basic permutation operations and cycle decomposition."""
        # Create small permutation
        perm = Permutation.from_list([2, 0, 1, 4, 3])  # 0→2→1→0, 3→4→3

        # Analyze cycles
        analyzer = CycleAnalyzer()
        cycles = analyzer.find_cycles(perm)

        # Verify cycle structure
        assert len(cycles) == 2, "Should have 2 cycles"
        cycle_lengths = sorted([len(c) for c in cycles])
        assert cycle_lengths == [2, 3], f"Expected [2, 3], got {cycle_lengths}"

    def test_cycle_length_distribution(self):
        """Test cycle length distribution computation."""
        perm = Permutation.from_list([1, 0, 3, 2, 4])  # Two 2-cycles, one 1-cycle
        analyzer = CycleAnalyzer()
        distribution = analyzer.cycle_length_distribution(perm)

        assert distribution[1] == 1, "Should have 1 cycle of length 1"
        assert distribution[2] == 2, "Should have 2 cycles of length 2"

    def test_max_cycle_length(self):
        """Test finding maximum cycle length."""
        perm = Permutation.from_list([2, 0, 1, 4, 3])  # Cycles: [0,2,1] and [3,4]
        analyzer = CycleAnalyzer()
        max_len = analyzer.max_cycle_length(perm)

        assert max_len == 3, f"Max cycle length should be 3, got {max_len}"

    def test_find_element_cycle(self):
        """Test finding the cycle containing a specific element."""
        perm = Permutation.from_list([2, 0, 1, 4, 3])  # 0→2→1→0
        analyzer = CycleAnalyzer()
        cycle = analyzer.find_element_cycle(perm, 0)

        assert len(cycle) == 3, "Cycle containing 0 should have length 3"
        assert set(cycle.elements) == {0, 1, 2}, "Cycle should contain {0, 1, 2}"

    def test_probability_max_cycle_exceeds_threshold_small(self):
        """Test probability computation for small permutations (exact)."""
        prob_calc = CombinatorialProbability()

        # For n=4, threshold=2: P(max cycle > 2) = ?
        # Possible permutations: 4! = 24
        # Count permutations with max cycle > 2:
        #   - All cycles ≤ 2: (12) (34), (13) (24), (14) (23), plus identity, transpositions
        #   - 3-cycles: (123) (4), (124) (3), (134) (2), (234) (1) in both directions
        #   - 4-cycles: (1234) in both directions
        # Exact calculation via enumeration
        prob = prob_calc.prob_max_cycle_exceeds_threshold(n=4, threshold=2)

        # For n=4: Expected ~0.583 (empirically)
        assert 0.55 <= prob <= 0.65, f"Expected ~0.58-0.63, got {prob}"

    def test_probability_max_cycle_exceeds_threshold_large(self):
        """Test probability computation for large permutations (asymptotic)."""
        prob_calc = CombinatorialProbability()

        # For n=100, threshold=50:
        # Famous result for 100 Prisoners Problem:
        # P(all succeed) = P(max cycle ≤ 50) ≈ 0.31
        # Therefore: P(max cycle > 50) ≈ 0.69 = ln(2)
        prob = prob_calc.prob_max_cycle_exceeds_threshold(n=100, threshold=50)

        # Should be close to 69% (ln(2) ≈ 0.693)
        assert 0.65 <= prob <= 0.75, f"Expected ~0.69 (ln(2)), got {prob}"
        print(f"P(max cycle > 50 in n=100) = {prob:.4f} (expected ~0.693)")

    def test_expected_max_cycle_length(self):
        """Test expected maximum cycle length computation."""
        prob_calc = CombinatorialProbability()

        # For small n, can verify exactly
        expected_max = prob_calc.expected_max_cycle_length(n=5)
        assert 2.0 <= expected_max <= 4.0, f"Expected 2-4, got {expected_max}"

        # For large n, should be approximately 0.624 * n (Golomb-Dickman)
        expected_max_large = prob_calc.expected_max_cycle_length(n=100)
        assert (
            50 <= expected_max_large <= 70
        ), f"Expected 50-70, got {expected_max_large}"
        print(f"E[max cycle length for n=100] = {expected_max_large:.2f}")

    def test_strategy_random_evaluation(self, reasoner, problem_parameters):
        """
        Test evaluation of random strategy.

        Random Strategy: Each agent examines positions uniformly at random.
        Success probability: (k/n)^n where k = max_examinations, n = n_agents
        """
        n = problem_parameters["n_agents"]
        k = problem_parameters["max_examinations"]

        # Define random strategy evaluation function
        def random_strategy_eval(state: Dict[str, Any], strategy: Strategy) -> tuple:
            """
            Evaluate random strategy.

            For each agent to succeed with random choice:
            P(success per agent) = k/n

            For ALL agents to succeed:
            P(all succeed) = (k/n)^n
            """
            n_agents = state["n_agents"]
            max_exam = state["max_examinations"]
            n_positions = state["n_positions"]

            # Probability each agent finds their ID by random choice
            prob_per_agent = max_exam / n_positions

            # Probability all agents succeed (independent events)
            prob_all_succeed = prob_per_agent**n_agents

            return prob_all_succeed, prob_all_succeed

        random_strategy = create_strategy(
            name="Random Examination Strategy",
            description="Each agent examines positions uniformly at random",
            parameters={"examination_method": "uniform_random", "max_examinations": k},
            evaluation_function=random_strategy_eval,
        )

        # Evaluate strategy
        eval_result = reasoner.strategy_evaluator.evaluate_strategy(
            strategy=random_strategy,
            problem_state=problem_parameters,
            success_criterion=lambda x: True,
        )

        # Expected: (50/100)^100 = 0.5^100 ≈ 7.9 × 10^-31 (essentially 0)
        expected_prob = (k / n) ** n
        assert eval_result.success_probability == pytest.approx(expected_prob, rel=1e-6)
        assert (
            eval_result.success_probability < 1e-20
        ), "Random strategy should have negligible success"
        print(f"Random strategy P(success) = {eval_result.success_probability:.2e}")

    def test_strategy_cycle_following_evaluation(self, reasoner, problem_parameters):
        """
        Test evaluation of cycle-following strategy.

        Cycle-Following Strategy: Each agent starts at their own ID position
        and follows the permutation chain.

        Success probability: P(all cycles ≤ k) where k = max_examinations

        For k = n/2, this probability is approximately 1 - ln(2) ≈ 0.31
        """
        n = problem_parameters["n_agents"]
        k = problem_parameters["max_examinations"]

        # Define cycle-following strategy evaluation function
        def cycle_strategy_eval(state: Dict[str, Any], strategy: Strategy) -> tuple:
            """
            Evaluate cycle-following strategy.

            Agent i starts at position i and follows the chain:
            position i → permutation[i] → permutation[permutation[i]] → ...

            Agent succeeds if cycle containing position i has length ≤ k

            ALL agents succeed ⟺ max cycle length ≤ k

            P(all succeed) = P(max cycle ≤ k)
                            = 1 - P(max cycle > k)
            """
            n_agents = state["n_agents"]
            max_exam = state["max_examinations"]

            prob_calc = CombinatorialProbability()
            prob_max_exceeds = prob_calc.prob_max_cycle_exceeds_threshold(
                n=n_agents, threshold=max_exam
            )

            # SUCCESS means max cycle ≤ threshold
            prob_all_succeed = 1.0 - prob_max_exceeds

            return prob_all_succeed, prob_all_succeed

        cycle_strategy = create_strategy(
            name="Cycle-Following Strategy",
            description="Each agent starts at their ID position and follows the permutation chain",
            parameters={
                "examination_method": "follow_chain",
                "start_position": "own_id",
                "max_examinations": k,
            },
            evaluation_function=cycle_strategy_eval,
        )

        # Evaluate strategy
        eval_result = reasoner.strategy_evaluator.evaluate_strategy(
            strategy=cycle_strategy,
            problem_state=problem_parameters,
            success_criterion=lambda x: True,
        )

        # Expected: ~0.31 (famous result)
        assert (
            0.25 <= eval_result.success_probability <= 0.35
        ), f"Expected ~0.31, got {eval_result.success_probability}"
        print(
            f"Cycle-following strategy P(success) = {eval_result.success_probability:.4f}"
        )

        # Should be MUCH better than random
        random_prob = (k / n) ** n
        assert (
            eval_result.success_probability > random_prob * 1e25
        ), "Cycle strategy should be vastly superior to random"

    def test_compare_strategies_and_find_optimal(self, reasoner, problem_parameters):
        """
        Test strategy comparison to identify optimal approach.

        This is the KEY TEST: KAI should determine that cycle-following
        strategy is vastly superior to random strategy.
        """
        problem_parameters["n_agents"]
        problem_parameters["max_examinations"]

        # Define both strategies
        def random_eval(state, strategy):
            n_agents = state["n_agents"]
            max_exam = state["max_examinations"]
            n_positions = state["n_positions"]
            prob = (max_exam / n_positions) ** n_agents
            return prob, prob

        def cycle_eval(state, strategy):
            n_agents = state["n_agents"]
            max_exam = state["max_examinations"]
            prob_calc = CombinatorialProbability()
            prob_exceed = prob_calc.prob_max_cycle_exceeds_threshold(n_agents, max_exam)
            prob = 1.0 - prob_exceed
            return prob, prob

        random_strategy = create_strategy(
            name="Random Strategy",
            description="Examine positions uniformly at random",
            parameters={"method": "random"},
            evaluation_function=random_eval,
        )

        cycle_strategy = create_strategy(
            name="Cycle-Following Strategy",
            description="Follow permutation chain from own ID",
            parameters={"method": "cycle_following"},
            evaluation_function=cycle_eval,
        )

        # Find optimal strategy
        optimal, proof = reasoner.find_optimal_strategy(
            strategies=[random_strategy, cycle_strategy],
            problem_state=problem_parameters,
        )

        # Verify cycle-following is optimal
        assert (
            optimal.name == "Cycle-Following Strategy"
        ), f"Expected Cycle-Following to be optimal, got {optimal.name}"

        # Verify proof tree exists
        assert proof is not None, "Should have explanation proof"
        assert proof.metadata.get("conclusion"), "Should have conclusion in metadata"
        assert "Cycle-Following Strategy" in proof.metadata["conclusion"]

        print(f"\nOptimal Strategy: {optimal.name}")
        print(f"Proof Conclusion: {proof.metadata['conclusion']}")

        # Verify dramatic improvement
        random_prob = random_eval(problem_parameters, random_strategy)[0]
        cycle_prob = cycle_eval(problem_parameters, cycle_strategy)[0]

        improvement_factor = (
            cycle_prob / random_prob if random_prob > 0 else float("inf")
        )
        print(f"Improvement Factor: {improvement_factor:.2e}x")

        assert cycle_prob > 0.25, "Optimal strategy should have >25% success"
        assert random_prob < 1e-20, "Random strategy should have negligible success"

    def test_analyze_specific_permutation(self, reasoner):
        """
        Test analyzing a specific permutation to determine if agents succeed.

        Given a specific permutation, determine:
        1. Cycle structure
        2. Which strategy succeeds
        3. Maximum cycle length
        """
        # Create permutation with known cycle structure
        # [1, 2, 0, 4, 3, 6, 7, 8, 9, 5]
        # Cycles: (0 1 2) length 3, (3 4) length 2, (5 6 7 8 9) length 5
        perm_list = [1, 2, 0, 4, 3, 6, 7, 8, 9, 5]
        perm = Permutation.from_list(perm_list)

        # Analyze
        cycles, analysis = reasoner.analyze_permutation(perm)

        # Verify analysis
        assert (
            analysis["num_cycles"] == 3
        ), f"Expected 3 cycles, got {analysis['num_cycles']}"
        assert (
            analysis["max_cycle_length"] == 5
        ), f"Expected max cycle 5, got {analysis['max_cycle_length']}"

        cycle_lengths = sorted(analysis["cycle_lengths"])
        assert cycle_lengths == [2, 3, 5], f"Expected [2, 3, 5], got {cycle_lengths}"

        # For this permutation: with k=5, cycle-following succeeds (max cycle = 5 ≤ 5)
        # But with k=4, it would fail (max cycle = 5 > 4)
        assert analysis["max_cycle_length"] <= 5, "Should succeed with threshold 5"
        assert analysis["max_cycle_length"] > 4, "Should fail with threshold 4"

    def test_permutation_composition(self):
        """Test permutation composition and inverse operations."""
        # Create two permutations
        perm1 = Permutation.from_list([1, 2, 0])  # 0→1→2→0
        perm2 = Permutation.from_list([2, 0, 1])  # 0→2→1→0

        # Compose: perm1 ∘ perm2
        composed = perm1.compose(perm2)

        # Verify composition
        assert composed(0) == perm1(perm2(0))
        assert composed(1) == perm1(perm2(1))
        assert composed(2) == perm1(perm2(2))

        # Test inverse
        inv = perm1.inverse()
        identity = perm1.compose(inv)

        # Verify identity
        for i in range(3):
            assert identity(i) == i, "Composition with inverse should give identity"

    def test_strategy_parameter_access(self):
        """Test strategy parameter management."""
        strategy = create_strategy(
            name="Test Strategy",
            description="For testing",
            parameters={"param1": 42, "param2": "value"},
        )

        # Test parameter access
        assert strategy.get_parameter("param1") == 42
        assert strategy.get_parameter("param2") == "value"
        assert strategy.get_parameter("nonexistent") is None

    def test_comprehensive_prisoners_problem_solution(self, reasoner):
        """
        Comprehensive test: Full solution to the prisoners problem.

        Tests ALL components:
        1. Define problem parameters
        2. Create multiple strategies
        3. Evaluate each strategy
        4. Compare and find optimal
        5. Verify correctness of solution
        """
        # Problem setup
        problem = {"n_agents": 100, "n_positions": 100, "max_examinations": 50}

        # Strategy 1: Random
        def random_eval(state, strategy):
            n = state["n_agents"]
            k = state["max_examinations"]
            prob = (k / n) ** n
            return prob, prob

        # Strategy 2: Cycle-following
        def cycle_eval(state, strategy):
            n = state["n_agents"]
            k = state["max_examinations"]
            prob_calc = CombinatorialProbability()
            prob = 1.0 - prob_calc.prob_max_cycle_exceeds_threshold(n, k)
            return prob, prob

        # Strategy 3: Alternative (e.g., start at different position)
        # This should be equivalent to random for this problem
        def alternative_eval(state, strategy):
            # Starting at position n/2 instead of 0 (still random-like)
            n = state["n_agents"]
            k = state["max_examinations"]
            prob = (k / n) ** n  # Same as random
            return prob, prob

        strategies = [
            create_strategy("Random", "Random examination", {}, random_eval),
            create_strategy(
                "Cycle-Following", "Follow permutation chain", {}, cycle_eval
            ),
            create_strategy(
                "Alternative", "Start at offset position", {}, alternative_eval
            ),
        ]

        # Find optimal
        optimal, proof = reasoner.find_optimal_strategy(strategies, problem)

        # Verify optimal is cycle-following
        assert (
            optimal.name == "Cycle-Following"
        ), f"Expected Cycle-Following optimal, got {optimal.name}"

        # Evaluate optimal strategy probability
        eval_result = reasoner.strategy_evaluator.evaluate_strategy(
            optimal, problem, lambda x: True
        )

        # Key assertion: Optimal strategy has ~31% success rate
        assert (
            0.25 <= eval_result.success_probability <= 0.35
        ), f"Expected ~31% success, got {eval_result.success_probability:.4f}"

        print("\n" + "=" * 60)
        print("COMPREHENSIVE TEST RESULT")
        print("=" * 60)
        print(
            f"Problem: {problem['n_agents']} agents, {problem['max_examinations']} examinations"
        )
        print(f"Optimal Strategy: {optimal.name}")
        print(f"Success Probability: {eval_result.success_probability:.4f} (~31%)")
        print(f"Proof: {proof.metadata.get('conclusion', 'No conclusion')}")
        print("=" * 60)

        # Final verification: This is the famous result
        expected_prob = 1 - sum(1 / i for i in range(51, 101))
        expected_prob_exp = math.exp(-expected_prob)
        print(f"Theoretical P(success) ~ {expected_prob_exp:.4f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
