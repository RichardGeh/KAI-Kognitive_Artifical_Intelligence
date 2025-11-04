"""
tests/test_epistemic_common_knowledge.py

Test suite for Common Knowledge functionality in component_35_epistemic_engine.py

Tests:
- C() operator (Full Fixed-Point Iteration)
- propagate_common_knowledge() (Public Announcement)
- Fixed-Point convergence
- Positive and negative test cases

Autor: KAI Development Team
Erstellt: 2025-11-01
"""

import pytest
from component_35_epistemic_engine import EpistemicEngine
from component_1_netzwerk import KonzeptNetzwerk


class TestCOperatorFixedPoint:
    """Test C operator with Fixed-Point Iteration"""

    @pytest.fixture
    def netzwerk(self):
        """Create real KonzeptNetzwerk for integration tests"""
        return KonzeptNetzwerk()

    @pytest.fixture
    def engine(self, netzwerk):
        """Create EpistemicEngine with real netzwerk"""
        return EpistemicEngine(netzwerk)

    def test_C_operator_with_propagated_knowledge(self, engine):
        """Test C operator returns True after propagate_common_knowledge"""
        # Setup: Create agents
        group = ["alice", "bob", "carol"]
        for agent_id in group:
            engine.create_agent(agent_id, agent_id.capitalize())

        # Propagate common knowledge
        count = engine.propagate_common_knowledge(group, "lottery_winner_announced")
        assert count > 0

        # Test: C operator should return True
        result = engine.C(group, "lottery_winner_announced", max_iterations=5)
        assert result is True

    def test_C_operator_negative_not_common(self, engine):
        """Test C operator returns False for non-common knowledge"""
        # Setup
        group = ["alice", "bob", "carol"]
        for agent_id in group:
            engine.create_agent(agent_id, agent_id.capitalize())

        # Only Alice knows the secret
        engine.add_knowledge("alice", "alices_secret")

        # Test: Should NOT be common knowledge
        result = engine.C(group, "alices_secret", max_iterations=5)
        assert result is False

    def test_C_operator_partial_knowledge(self, engine):
        """Test C operator when only some agents know"""
        # Setup
        group = ["alice", "bob", "carol"]
        for agent_id in group:
            engine.create_agent(agent_id, agent_id.capitalize())

        # Only Alice and Bob know
        engine.add_knowledge("alice", "partial_fact")
        engine.add_knowledge("bob", "partial_fact")
        # Carol doesn't know

        # Test: Should NOT be common knowledge
        result = engine.C(group, "partial_fact", max_iterations=5)
        assert result is False

    def test_C_operator_everyone_knows_but_no_meta(self, engine):
        """Test C operator when everyone knows but no meta-knowledge"""
        # Setup
        group = ["alice", "bob", "carol"]
        for agent_id in group:
            engine.create_agent(agent_id, agent_id.capitalize())

        # Everyone knows, but no meta-knowledge
        engine.add_group_knowledge(group, "basic_fact")

        # Test: Should NOT be common knowledge (needs meta-knowledge)
        result = engine.C(group, "basic_fact", max_iterations=5)
        assert result is False

    def test_C_operator_fixed_point_convergence(self, engine):
        """Test that C operator converges to fixed point"""
        # Setup
        group = ["alice", "bob"]
        for agent_id in group:
            engine.create_agent(agent_id, agent_id.capitalize())

        # Propagate common knowledge (small group for faster test)
        engine.propagate_common_knowledge(group, "public_announcement", max_depth=2)

        # Test with different max_iterations
        result1 = engine.C(group, "public_announcement", max_iterations=3)
        result2 = engine.C(group, "public_announcement", max_iterations=10)

        # Should converge to same result
        assert result1 == result2

    def test_C_operator_empty_group(self, engine):
        """Test C operator with empty group"""
        # Test: Empty group should return False
        result = engine.C([], "some_fact", max_iterations=5)
        assert result is False

    def test_C_operator_single_agent(self, engine):
        """Test C operator with single agent"""
        # Setup
        engine.create_agent("alice", "Alice")
        engine.add_knowledge("alice", "alice_knows")

        # Test: Single agent who knows = common knowledge in group of 1
        result = engine.C(["alice"], "alice_knows", max_iterations=5)
        assert result is True

    def test_C_operator_two_agents_with_meta(self, engine):
        """Test C operator with two agents and meta-knowledge"""
        # Setup
        group = ["alice", "bob"]
        for agent_id in group:
            engine.create_agent(agent_id, agent_id.capitalize())

        # Both know proposition
        engine.add_knowledge("alice", "shared_secret")
        engine.add_knowledge("bob", "shared_secret")

        # Add meta-knowledge: Alice knows that Bob knows
        engine.add_nested_knowledge("alice", ["bob"], "shared_secret")
        # Bob knows that Alice knows
        engine.add_nested_knowledge("bob", ["alice"], "shared_secret")

        # Test: Should be common knowledge
        result = engine.C(group, "shared_secret", max_iterations=5)
        assert result is True

    def test_C_operator_max_iterations_reached(self, engine):
        """Test C operator behavior when max_iterations reached"""
        # Setup: Complex scenario that might not converge quickly
        group = ["alice", "bob", "carol", "dave"]
        for agent_id in group:
            engine.create_agent(agent_id, agent_id.capitalize())

        # Partial knowledge setup
        engine.add_knowledge("alice", "complex_fact")
        engine.add_knowledge("bob", "complex_fact")

        # Test with very low max_iterations
        result = engine.C(group, "complex_fact", max_iterations=1)
        # Should return False (didn't converge or incomplete knowledge)
        assert result is False


class TestPropagateCommonKnowledge:
    """Test propagate_common_knowledge method"""

    @pytest.fixture
    def netzwerk(self):
        """Create real KonzeptNetzwerk"""
        return KonzeptNetzwerk()

    @pytest.fixture
    def engine(self, netzwerk):
        """Create EpistemicEngine"""
        return EpistemicEngine(netzwerk)

    def test_propagate_basic(self, engine):
        """Test basic propagate_common_knowledge functionality"""
        # Setup
        group = ["alice", "bob", "carol"]
        for agent_id in group:
            engine.create_agent(agent_id, agent_id.capitalize())

        # Propagate
        count = engine.propagate_common_knowledge(group, "test_prop", max_depth=2)

        # Verify count is positive
        assert count > 0

        # Verify all agents know the proposition
        for agent_id in group:
            assert engine.K(agent_id, "test_prop") is True

    def test_propagate_count_calculation(self, engine):
        """Test that propagate_common_knowledge returns correct count"""
        # Setup
        group = ["alice", "bob"]
        for agent_id in group:
            engine.create_agent(agent_id, agent_id.capitalize())

        # Propagate with max_depth=1
        count = engine.propagate_common_knowledge(group, "count_test", max_depth=1)

        # Expected count:
        # Level 0: 2 agents learn proposition = 2
        # Level 1: 2 observers * 1 subject each (excluding self) = 2
        # Total = 4
        assert count == 4

    def test_propagate_with_depth_2(self, engine):
        """Test propagate_common_knowledge with max_depth=2"""
        # Setup
        group = ["alice", "bob", "carol"]
        for agent_id in group:
            engine.create_agent(agent_id, agent_id.capitalize())

        # Propagate with max_depth=2
        count = engine.propagate_common_knowledge(group, "depth2_test", max_depth=2)

        # Count should include Level 0, Level 1, and Level 2
        # Level 0: 3 agents = 3
        # Level 1: 3 * 2 = 6 (each agent learns about 2 others)
        # Level 2: 3 * 2 * 2 = 12 (nested meta-knowledge)
        # Total = 21
        assert count == 21

    def test_propagate_creates_meta_knowledge(self, engine):
        """Test that propagate creates meta-knowledge structures"""
        # Setup
        group = ["alice", "bob"]
        for agent_id in group:
            engine.create_agent(agent_id, agent_id.capitalize())

        # Propagate
        engine.propagate_common_knowledge(group, "meta_test", max_depth=2)

        # Verify meta-knowledge: Alice knows that Bob knows
        assert engine.K_n("alice", ["bob"], "meta_test") is True

        # Verify reverse: Bob knows that Alice knows
        assert engine.K_n("bob", ["alice"], "meta_test") is True

    def test_propagate_empty_group(self, engine):
        """Test propagate_common_knowledge with empty group"""
        # Propagate with empty group
        count = engine.propagate_common_knowledge([], "empty_test", max_depth=2)

        # Should return 0 (no agents to propagate to)
        assert count == 0

    def test_propagate_single_agent(self, engine):
        """Test propagate_common_knowledge with single agent"""
        # Setup
        engine.create_agent("alice", "Alice")

        # Propagate
        count = engine.propagate_common_knowledge(["alice"], "single_test", max_depth=2)

        # Only Level 0 knowledge (1 agent)
        # No Level 1 (no other agents to know about)
        assert count == 1

        # Verify agent knows proposition
        assert engine.K("alice", "single_test") is True

    def test_propagate_large_group(self, engine):
        """Test propagate_common_knowledge with larger group"""
        # Setup
        group = ["a1", "a2", "a3", "a4", "a5"]
        for agent_id in group:
            engine.create_agent(agent_id, agent_id.upper())

        # Propagate
        count = engine.propagate_common_knowledge(
            group, "large_group_test", max_depth=1
        )

        # Level 0: 5 agents = 5
        # Level 1: 5 * 4 = 20 (each knows about 4 others)
        # Total = 25
        assert count == 25

        # Verify all know the proposition
        for agent_id in group:
            assert engine.K(agent_id, "large_group_test") is True

    def test_propagate_multiple_propositions(self, engine):
        """Test propagating multiple different propositions"""
        # Setup
        group = ["alice", "bob"]
        for agent_id in group:
            engine.create_agent(agent_id, agent_id.capitalize())

        # Propagate multiple propositions
        count1 = engine.propagate_common_knowledge(group, "prop1", max_depth=1)
        count2 = engine.propagate_common_knowledge(group, "prop2", max_depth=1)

        assert count1 > 0
        assert count2 > 0

        # Verify both propositions are known
        assert engine.K("alice", "prop1") is True
        assert engine.K("alice", "prop2") is True
        assert engine.K("bob", "prop1") is True
        assert engine.K("bob", "prop2") is True


class TestIntegration:
    """Integration tests combining C operator and propagate_common_knowledge"""

    @pytest.fixture
    def netzwerk(self):
        """Create real KonzeptNetzwerk"""
        return KonzeptNetzwerk()

    @pytest.fixture
    def engine(self, netzwerk):
        """Create EpistemicEngine"""
        return EpistemicEngine(netzwerk)

    def test_full_workflow_public_announcement(self, engine):
        """Test complete workflow: propagate -> verify with C operator"""
        # Setup
        group = ["alice", "bob", "carol"]
        for agent_id in group:
            engine.create_agent(agent_id, agent_id.capitalize())

        # Simulate public announcement
        count = engine.propagate_common_knowledge(group, "meeting_at_3pm", max_depth=2)
        assert count > 0

        # Verify it's common knowledge
        assert engine.C(group, "meeting_at_3pm", max_iterations=5) is True

        # Verify each agent knows
        assert engine.E(group, "meeting_at_3pm") is True

    def test_lottery_puzzle_scenario(self, engine):
        """Test Blue Eyes Puzzle / Muddy Children style scenario"""
        # Setup: 3 agents
        group = ["alice", "bob", "carol"]
        for agent_id in group:
            engine.create_agent(agent_id, agent_id.capitalize())

        # Public announcement: "At least one person won the lottery"
        engine.propagate_common_knowledge(group, "at_least_one_winner", max_depth=2)

        # Verify common knowledge
        assert engine.C(group, "at_least_one_winner", max_iterations=5) is True

        # All agents know
        assert engine.E(group, "at_least_one_winner") is True

        # Each agent knows the others know
        assert engine.K_n("alice", ["bob"], "at_least_one_winner") is True
        assert engine.K_n("bob", ["carol"], "at_least_one_winner") is True
        assert engine.K_n("carol", ["alice"], "at_least_one_winner") is True

    def test_compare_C_simple_vs_C_full(self, engine):
        """Compare C_simple (approximation) vs C (full fixed-point)"""
        # Setup with unique agent IDs
        group = ["alice_compare", "bob_compare"]
        for agent_id in group:
            engine.create_agent(agent_id, agent_id.capitalize())

        # Propagate common knowledge
        engine.propagate_common_knowledge(group, "compare_test_unique", max_depth=2)

        # Both should return True for properly established common knowledge
        result_simple = engine.C_simple(group, "compare_test_unique", max_depth=2)
        result_full = engine.C(group, "compare_test_unique", max_iterations=5)

        # Both should agree on True
        assert result_simple is True
        assert result_full is True

    def test_gradual_knowledge_buildup(self, engine):
        """Test building up common knowledge gradually"""
        # Setup with unique agent IDs
        group = ["alice_gradual", "bob_gradual"]
        for agent_id in group:
            engine.create_agent(agent_id, agent_id.capitalize())

        # Step 1: Only Alice knows
        engine.add_knowledge("alice_gradual", "gradual_fact_unique")
        assert engine.C(group, "gradual_fact_unique", max_iterations=3) is False

        # Step 2: Both know, but no meta-knowledge
        engine.add_knowledge("bob_gradual", "gradual_fact_unique")
        assert engine.C(group, "gradual_fact_unique", max_iterations=3) is False

        # Step 3: Add meta-knowledge
        engine.add_nested_knowledge(
            "alice_gradual", ["bob_gradual"], "gradual_fact_unique"
        )
        engine.add_nested_knowledge(
            "bob_gradual", ["alice_gradual"], "gradual_fact_unique"
        )

        # Now it's common knowledge
        assert engine.C(group, "gradual_fact_unique", max_iterations=3) is True

    def test_common_knowledge_subset_groups(self, engine):
        """Test common knowledge in different subsets of agents"""
        # Setup
        all_agents = ["alice", "bob", "carol", "dave"]
        for agent_id in all_agents:
            engine.create_agent(agent_id, agent_id.capitalize())

        # Group 1: Alice and Bob
        group1 = ["alice", "bob"]
        engine.propagate_common_knowledge(group1, "group1_secret", max_depth=2)

        # Group 2: Carol and Dave
        group2 = ["carol", "dave"]
        engine.propagate_common_knowledge(group2, "group2_secret", max_depth=2)

        # Verify group1_secret is common in group1 but not in group2
        assert engine.C(group1, "group1_secret", max_iterations=3) is True
        assert engine.C(group2, "group1_secret", max_iterations=3) is False

        # Verify group2_secret is common in group2 but not in group1
        assert engine.C(group2, "group2_secret", max_iterations=3) is True
        assert engine.C(group1, "group2_secret", max_iterations=3) is False

    def test_knowledge_base_consistency(self, engine):
        """Test that knowledge base remains consistent after operations"""
        # Setup
        group = ["alice", "bob"]
        for agent_id in group:
            engine.create_agent(agent_id, agent_id.capitalize())

        # Propagate
        engine.propagate_common_knowledge(group, "consistency_test", max_depth=2)

        # Verify consistency: E implies K for each agent
        assert engine.E(group, "consistency_test") is True
        assert engine.K("alice", "consistency_test") is True
        assert engine.K("bob", "consistency_test") is True

        # Verify C implies E
        assert engine.C(group, "consistency_test", max_iterations=3) is True
        assert engine.E(group, "consistency_test") is True


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    @pytest.fixture
    def netzwerk(self):
        """Create real KonzeptNetzwerk"""
        return KonzeptNetzwerk()

    @pytest.fixture
    def engine(self, netzwerk):
        """Create EpistemicEngine"""
        return EpistemicEngine(netzwerk)

    def test_C_operator_with_unknown_agents(self, engine):
        """Test C operator with agents that don't exist"""
        # Test with non-existent agents (should return False)
        result = engine.C(
            ["nonexistent1", "nonexistent2"], "some_fact", max_iterations=3
        )
        assert result is False

    def test_propagate_with_zero_depth(self, engine):
        """Test propagate_common_knowledge with max_depth=0"""
        # Setup
        group = ["alice", "bob"]
        for agent_id in group:
            engine.create_agent(agent_id, agent_id.capitalize())

        # Propagate with depth 0 (only Level 0 knowledge)
        count = engine.propagate_common_knowledge(group, "depth0_test", max_depth=0)

        # Should only add basic knowledge (2 agents)
        assert count == 2

        # Verify basic knowledge exists
        assert engine.K("alice", "depth0_test") is True
        assert engine.K("bob", "depth0_test") is True

    def test_C_operator_very_high_iterations(self, engine):
        """Test C operator with very high max_iterations"""
        # Setup
        group = ["alice", "bob"]
        for agent_id in group:
            engine.create_agent(agent_id, agent_id.capitalize())

        # Propagate
        engine.propagate_common_knowledge(group, "high_iter_test", max_depth=2)

        # Test with very high iterations (should converge early)
        result = engine.C(group, "high_iter_test", max_iterations=100)
        assert result is True

    def test_empty_proposition_id(self, engine):
        """Test with empty proposition ID"""
        # Setup
        group = ["alice"]
        engine.create_agent("alice", "Alice")

        # Propagate empty proposition
        count = engine.propagate_common_knowledge(group, "", max_depth=1)
        assert count >= 1

        # Should work (validation is application's responsibility)
        result = engine.C(group, "", max_iterations=3)
        # Result depends on whether empty string is treated as valid


if __name__ == "__main__":
    print("Starting Common Knowledge Tests...")
    pytest.main([__file__, "-v"])
