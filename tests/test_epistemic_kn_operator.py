"""
test_epistemic_kn_operator.py

Tests für den K^n Operator (Nested Knowledge) im EpistemicEngine

Phase 3, Step 3.1: Meta-Level Reasoning
"""

import pytest
from component_1_netzwerk import KonzeptNetzwerk
from component_35_epistemic_engine import EpistemicEngine


@pytest.fixture
def netzwerk():
    """Fixture: KonzeptNetzwerk für Tests"""
    return KonzeptNetzwerk()


@pytest.fixture
def engine(netzwerk):
    """Fixture: EpistemicEngine für Tests"""
    return EpistemicEngine(netzwerk)


@pytest.fixture
def agents(engine):
    """Fixture: Erstelle Test-Agenten"""
    engine.create_agent("alice", "Alice")
    engine.create_agent("bob", "Bob")
    engine.create_agent("carol", "Carol")
    return ["alice", "bob", "carol"]


class TestNestedSignature:
    """Tests für _create_nested_signature() helper"""

    def test_empty_chain(self, engine):
        """Test: Leere Agent Chain gibt Proposition zurück"""
        result = engine._create_nested_signature([], "p")
        assert result == "p"

    def test_single_agent(self, engine):
        """Test: Ein Agent in Chain"""
        result = engine._create_nested_signature(["bob"], "p")
        assert result == "K(bob, p)"

    def test_two_agents(self, engine):
        """Test: Zwei Agenten in Chain"""
        result = engine._create_nested_signature(["bob", "carol"], "p")
        assert result == "K(bob, K(carol, p))"

    def test_three_agents(self, engine):
        """Test: Drei Agenten in Chain"""
        result = engine._create_nested_signature(["alice", "bob", "carol"], "p")
        assert result == "K(alice, K(bob, K(carol, p)))"

    def test_complex_proposition(self, engine):
        """Test: Komplexe Proposition ID"""
        result = engine._create_nested_signature(["bob"], "secret_password_123")
        assert result == "K(bob, secret_password_123)"


class TestKnOperator1Level:
    """Tests für K_n Operator mit 1-Level Nesting"""

    def test_1level_basic(self, engine, agents):
        """Test: K_n(alice, [bob], p) = Alice knows that Bob knows p"""
        # Setup: Bob knows secret_password
        engine.add_knowledge("bob", "secret_password")

        # Alice knows that Bob knows secret_password
        engine.add_nested_knowledge("alice", ["bob"], "secret_password")

        # Verify
        assert engine.K_n("alice", ["bob"], "secret_password") is True

    def test_1level_false_observer(self, engine, agents):
        """Test: Alice doesn't know about Carol's knowledge"""
        # Setup: Carol knows secret_password
        engine.add_knowledge("carol", "secret_password")

        # Alice doesn't know about Carol
        assert engine.K_n("alice", ["carol"], "secret_password") is False

    def test_1level_false_proposition(self, engine, agents):
        """Test: Alice knows Bob knows X, but query for Y"""
        # Setup: Bob knows fact_x
        engine.add_knowledge("bob", "fact_x")
        engine.add_nested_knowledge("alice", ["bob"], "fact_x")

        # Query for different proposition
        assert engine.K_n("alice", ["bob"], "fact_y") is False

    def test_1level_multiple_nested_beliefs(self, engine, agents):
        """Test: Alice knows about multiple of Bob's beliefs"""
        # Setup: Bob knows multiple facts
        engine.add_knowledge("bob", "fact_1")
        engine.add_knowledge("bob", "fact_2")

        # Alice knows Bob knows both
        engine.add_nested_knowledge("alice", ["bob"], "fact_1")
        engine.add_nested_knowledge("alice", ["bob"], "fact_2")

        # Verify both
        assert engine.K_n("alice", ["bob"], "fact_1") is True
        assert engine.K_n("alice", ["bob"], "fact_2") is True

    def test_1level_different_observers(self, engine, agents):
        """Test: Multiple agents know about Bob's knowledge"""
        # Setup: Bob knows secret
        engine.add_knowledge("bob", "secret")

        # Both Alice and Carol know that Bob knows
        engine.add_nested_knowledge("alice", ["bob"], "secret")
        engine.add_nested_knowledge("carol", ["bob"], "secret")

        # Verify both
        assert engine.K_n("alice", ["bob"], "secret") is True
        assert engine.K_n("carol", ["bob"], "secret") is True


class TestKnOperator2Level:
    """Tests für K_n Operator mit 2-Level Nesting"""

    def test_2level_basic(self, engine, agents):
        """Test: K_n(carol, [bob, alice], p) = Carol knows that Bob knows that Alice knows p"""
        # Setup: Alice knows secret_password
        engine.add_knowledge("alice", "secret_password")

        # Bob knows that Alice knows
        engine.add_nested_knowledge("bob", ["alice"], "secret_password")

        # Carol knows that Bob knows that Alice knows
        engine.add_nested_knowledge("carol", ["bob", "alice"], "secret_password")

        # Verify
        assert engine.K_n("carol", ["bob", "alice"], "secret_password") is True

    def test_2level_false(self, engine, agents):
        """Test: Carol doesn't know about 2-level nesting"""
        # Setup: Only 1-level nesting
        engine.add_knowledge("alice", "fact_x")
        engine.add_nested_knowledge("bob", ["alice"], "fact_x")

        # Carol doesn't know
        assert engine.K_n("carol", ["bob", "alice"], "fact_x") is False

    def test_2level_partial_knowledge(self, engine, agents):
        """Test: Carol knows about Bob, but not about deeper nesting"""
        # Use unique propositions to avoid contamination
        fact_id = "fact_partial_test"
        other_fact_id = "other_fact_partial_test"

        # Setup: Alice knows fact
        engine.add_knowledge("alice", fact_id)

        # Bob knows that Alice knows
        engine.add_nested_knowledge("bob", ["alice"], fact_id)

        # Carol only knows that Bob knows something (not specifically about Alice)
        engine.add_nested_knowledge("carol", ["bob"], other_fact_id)

        # Carol doesn't know the 2-level chain about 'fact'
        assert engine.K_n("carol", ["bob", "alice"], fact_id) is False

    def test_2level_independent_chains(self, engine, agents):
        """Test: Verschiedene 2-level chains sind unabhängig"""
        # Use unique propositions to avoid contamination
        fact_1_id = "fact_chain_1_unique"
        fact_2_id = "fact_chain_2_unique"

        # Chain 1: Carol -> Bob -> Alice knows fact_1
        engine.add_knowledge("alice", fact_1_id)
        engine.add_nested_knowledge("bob", ["alice"], fact_1_id)
        engine.add_nested_knowledge("carol", ["bob", "alice"], fact_1_id)

        # Chain 2: Alice -> Carol -> Bob knows fact_2
        engine.add_knowledge("bob", fact_2_id)
        engine.add_nested_knowledge("carol", ["bob"], fact_2_id)
        engine.add_nested_knowledge("alice", ["carol", "bob"], fact_2_id)

        # Verify both chains independently
        assert engine.K_n("carol", ["bob", "alice"], fact_1_id) is True
        assert engine.K_n("alice", ["carol", "bob"], fact_2_id) is True

        # Cross-chain queries should fail
        assert engine.K_n("carol", ["bob", "alice"], fact_2_id) is False
        assert engine.K_n("alice", ["carol", "bob"], fact_1_id) is False


class TestKnOperator3Level:
    """Tests für K_n Operator mit 3-Level Nesting (optional, stress test)"""

    def test_3level_basic(self, engine):
        """Test: 3-level nesting with 4 agents"""
        # Create 4 agents
        engine.create_agent("alice", "Alice")
        engine.create_agent("bob", "Bob")
        engine.create_agent("carol", "Carol")
        engine.create_agent("dave", "Dave")

        # Dave knows secret
        engine.add_knowledge("dave", "secret")

        # Carol knows that Dave knows
        engine.add_nested_knowledge("carol", ["dave"], "secret")

        # Bob knows that Carol knows that Dave knows
        engine.add_nested_knowledge("bob", ["carol", "dave"], "secret")

        # Alice knows that Bob knows that Carol knows that Dave knows
        engine.add_nested_knowledge("alice", ["bob", "carol", "dave"], "secret")

        # Verify
        assert engine.K_n("alice", ["bob", "carol", "dave"], "secret") is True

    def test_3level_missing_intermediate(self, engine):
        """Test: 3-level chain mit fehlendem Zwischenglied schlägt fehl"""
        # Use unique agent IDs to avoid test contamination
        alice_id = "alice_missing_test"
        bob_id = "bob_missing_test"
        carol_id = "carol_missing_test"
        dave_id = "dave_missing_test"

        engine.create_agent(alice_id, "Alice Missing")
        engine.create_agent(bob_id, "Bob Missing")
        engine.create_agent(carol_id, "Carol Missing")
        engine.create_agent(dave_id, "Dave Missing")

        # Dave knows secret
        engine.add_knowledge(dave_id, "secret_missing")

        # Carol knows that Dave knows
        engine.add_nested_knowledge(carol_id, [dave_id], "secret_missing")

        # SKIP: Bob doesn't know about Carol-Dave chain

        # Alice knows that Bob knows (but Bob doesn't actually know)
        engine.add_nested_knowledge(
            alice_id, [bob_id, carol_id, dave_id], "secret_missing"
        )

        # This should return True for Alice's belief (Alice believes the chain exists)
        # But Bob doesn't actually have the knowledge
        assert (
            engine.K_n(alice_id, [bob_id, carol_id, dave_id], "secret_missing") is True
        )
        assert engine.K_n(bob_id, [carol_id, dave_id], "secret_missing") is False


class TestKnOperatorBaseCase:
    """Tests für K_n Basis-Fall (empty nested_knowledge)"""

    def test_base_case_empty_chain(self, engine, agents):
        """Test: K_n(alice, [], p) = K(alice, p)"""
        # Setup: Alice knows fact
        engine.add_knowledge("alice", "fact")

        # Query with empty chain should delegate to K()
        assert engine.K_n("alice", [], "fact") is True
        assert engine.K_n("alice", [], "unknown_fact") is False

    def test_base_case_vs_regular_k(self, engine, agents):
        """Test: K_n(alice, [], p) verhält sich identisch zu K(alice, p)"""
        engine.add_knowledge("bob", "known_fact")

        # Both should return same result
        assert engine.K_n("bob", [], "known_fact") == engine.K("bob", "known_fact")
        assert engine.K_n("bob", [], "unknown_fact") == engine.K("bob", "unknown_fact")


class TestAddNestedKnowledge:
    """Tests für add_nested_knowledge() method"""

    def test_add_1level(self, engine, agents):
        """Test: Füge 1-level nested knowledge hinzu"""
        # Add: Alice knows that Bob knows secret
        success = engine.add_nested_knowledge("alice", ["bob"], "secret")
        assert success is True

        # Verify it was added
        assert engine.K_n("alice", ["bob"], "secret") is True

    def test_add_2level(self, engine, agents):
        """Test: Füge 2-level nested knowledge hinzu"""
        # Add: Carol knows that Bob knows that Alice knows fact
        success = engine.add_nested_knowledge("carol", ["bob", "alice"], "fact")
        assert success is True

        # Verify it was added
        assert engine.K_n("carol", ["bob", "alice"], "fact") is True

    def test_add_base_case(self, engine, agents):
        """Test: add_nested_knowledge mit empty chain delegiert zu add_knowledge"""
        # Add with empty chain
        success = engine.add_nested_knowledge("alice", [], "simple_fact")
        assert success is True

        # Should be added as regular knowledge (not nested)
        assert engine.K("alice", "simple_fact") is True
        assert engine.K_n("alice", [], "simple_fact") is True

    def test_add_multiple_nested_beliefs(self, engine, agents):
        """Test: Füge mehrere nested beliefs hinzu"""
        # Add multiple
        engine.add_nested_knowledge("alice", ["bob"], "fact_1")
        engine.add_nested_knowledge("alice", ["bob"], "fact_2")
        engine.add_nested_knowledge("alice", ["carol"], "fact_3")

        # Verify all
        assert engine.K_n("alice", ["bob"], "fact_1") is True
        assert engine.K_n("alice", ["bob"], "fact_2") is True
        assert engine.K_n("alice", ["carol"], "fact_3") is True

    def test_add_idempotent(self, engine, agents):
        """Test: Mehrfaches Hinzufügen desselben nested knowledge"""
        # Add same knowledge twice
        engine.add_nested_knowledge("alice", ["bob"], "repeated_fact")
        engine.add_nested_knowledge("alice", ["bob"], "repeated_fact")

        # Should still work (idempotent)
        assert engine.K_n("alice", ["bob"], "repeated_fact") is True


class TestMetaLevelProperty:
    """Tests für korrekte meta_level Werte in MetaBelief Nodes"""

    def test_meta_level_1(self, engine, netzwerk, agents):
        """Test: 1-level nesting hat meta_level=1"""
        engine.add_nested_knowledge("alice", ["bob"], "fact")

        # Query Neo4j direkt
        with netzwerk.driver.session(database="neo4j") as session:
            result = session.run(
                """
                MATCH (alice:Agent {id: 'alice'})-[:KNOWS_THAT]->(mb:MetaBelief)
                WHERE mb.proposition = 'K(bob, fact)'
                RETURN mb.meta_level AS level
                """
            )
            record = result.single()
            assert record is not None
            assert record["level"] == 1

    def test_meta_level_2(self, engine, netzwerk, agents):
        """Test: 2-level nesting hat meta_level=2"""
        engine.add_nested_knowledge("carol", ["bob", "alice"], "fact")

        # Query Neo4j direkt
        # Note: The stored signature is "K(alice, fact)" (what Bob supposedly knows)
        # NOT "K(bob, K(alice, fact))" (the full chain)
        with netzwerk.driver.session(database="neo4j") as session:
            result = session.run(
                """
                MATCH (carol:Agent {id: 'carol'})-[:KNOWS_THAT]->(mb:MetaBelief)
                WHERE mb.proposition = 'K(alice, fact)'
                  AND mb.meta_level = 2
                MATCH (mb)-[:ABOUT_AGENT]->(bob:Agent {id: 'bob'})
                RETURN mb.meta_level AS level
                """
            )
            record = result.single()
            assert record is not None
            assert record["level"] == 2

    def test_different_meta_levels(self, engine, netzwerk, agents):
        """Test: Verschiedene meta_levels für verschiedene nested chains"""
        # Add 1-level and 2-level
        engine.add_nested_knowledge("alice", ["bob"], "fact_1")
        engine.add_nested_knowledge("carol", ["bob", "alice"], "fact_2")

        # Query both
        with netzwerk.driver.session(database="neo4j") as session:
            # 1-level
            result1 = session.run(
                """
                MATCH (alice:Agent {id: 'alice'})-[:KNOWS_THAT]->(mb:MetaBelief)
                WHERE mb.proposition = 'K(bob, fact_1)'
                  AND mb.meta_level = 1
                RETURN mb.meta_level AS level
                """
            )
            record1 = result1.single()
            assert record1 is not None
            assert record1["level"] == 1

            # 2-level
            # Note: The stored signature is "K(alice, fact_2)" (what Bob supposedly knows)
            result2 = session.run(
                """
                MATCH (carol:Agent {id: 'carol'})-[:KNOWS_THAT]->(mb:MetaBelief)
                WHERE mb.proposition = 'K(alice, fact_2)'
                  AND mb.meta_level = 2
                MATCH (mb)-[:ABOUT_AGENT]->(bob:Agent {id: 'bob'})
                RETURN mb.meta_level AS level
                """
            )
            record2 = result2.single()
            assert record2 is not None
            assert record2["level"] == 2


class TestEdgeCases:
    """Tests für Edge Cases und Fehlerbehandlung"""

    def test_nonexistent_agent(self, engine):
        """Test: Query für nicht-existierenden Agent"""
        # Agent "unknown" doesn't exist
        result = engine.K_n("unknown", ["bob"], "fact")
        assert result is False

    def test_empty_proposition(self, engine, agents):
        """Test: Leere Proposition ID"""
        engine.add_nested_knowledge("alice", ["bob"], "")
        assert engine.K_n("alice", ["bob"], "") is True

    def test_special_characters_in_proposition(self, engine, agents):
        """Test: Sonderzeichen in Proposition"""
        prop = "fact_with_special_chars_!@#$%"
        engine.add_nested_knowledge("alice", ["bob"], prop)
        assert engine.K_n("alice", ["bob"], prop) is True

    def test_very_long_chain(self, engine):
        """Test: Sehr lange nested chain (Stress Test)"""
        # Create 5 agents
        agent_ids = []
        for i in range(5):
            agent_id = f"agent_{i}"
            engine.create_agent(agent_id, f"Agent {i}")
            agent_ids.append(agent_id)

        # Build chain: agent_0 knows that agent_1 knows that agent_2... knows fact
        # For simplicity, just test that the system doesn't crash
        chain = agent_ids[1:]  # [agent_1, agent_2, agent_3, agent_4]

        try:
            engine.add_nested_knowledge(agent_ids[0], chain, "deep_fact")
            result = engine.K_n(agent_ids[0], chain, "deep_fact")
            # Should succeed
            assert result is True
        except Exception as e:
            pytest.fail(f"Very long chain caused exception: {e}")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
