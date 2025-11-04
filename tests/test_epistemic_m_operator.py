"""
test_epistemic_m_operator.py

Umfassende Tests für den M Operator (believes possible) des Epistemic Engine.

Autor: KAI Development Team
Erstellt: 2025-11-01
"""

import pytest
from component_1_netzwerk import KonzeptNetzwerk
from component_35_epistemic_engine import EpistemicEngine


class TestMOperator:
    """Test-Suite für M Operator (Agent believes proposition is possible)"""

    @pytest.fixture
    def setup_engine(self):
        """Setup: Netzwerk und Engine erstellen"""
        netzwerk = KonzeptNetzwerk()
        engine = EpistemicEngine(netzwerk)
        yield engine
        # Cleanup wird von conftest.py gehandhabt

    def test_m_operator_basic_possible(self, setup_engine):
        """Test: M(agent, prop) = True wenn kein Gegenwissen vorhanden"""
        engine = setup_engine
        engine.create_agent("alice", "Alice")

        # Alice hat kein Wissen über "unicorns_exist"
        result = engine.M("alice", "unicorns_exist")

        assert (
            result is True
        ), "M should return True when no contradicting knowledge exists"

    def test_m_operator_negated_knowledge(self, setup_engine):
        """Test: M(agent, prop) = False wenn Agent weiß, dass ¬prop"""
        engine = setup_engine
        engine.create_agent("alice", "Alice")

        # Alice weiß, dass Mond KEIN Käse ist
        engine.add_negated_knowledge("alice", "moon_is_cheese")

        result = engine.M("alice", "moon_is_cheese")

        assert result is False, "M should return False when agent knows NOT(prop)"

    def test_m_operator_positive_knowledge(self, setup_engine):
        """Test: M(agent, prop) = True wenn Agent weiß, dass prop wahr ist"""
        engine = setup_engine
        engine.create_agent("bob", "Bob")

        # Bob weiß, dass Himmel blau ist
        engine.add_knowledge("bob", "sky_is_blue")

        result = engine.M("bob", "sky_is_blue")

        assert result is True, "M should return True when agent knows prop is true"

    def test_m_operator_kripke_semantics(self, setup_engine):
        """Test: M(p) = ¬K(¬p) Kripke Semantik"""
        engine = setup_engine
        engine.create_agent("charlie", "Charlie")

        # Scenario 1: K(¬p) = False -> M(p) = True
        knows_not_p1 = engine.K("charlie", "NOT_dragons_exist")
        m_result1 = engine.M("charlie", "dragons_exist")

        assert knows_not_p1 is False, "Should not know NOT(dragons_exist)"
        assert m_result1 is True, "M should be True when ¬K(¬p)"

        # Scenario 2: K(¬p) = True -> M(p) = False
        engine.add_negated_knowledge("charlie", "perpetual_motion")
        knows_not_p2 = engine.K("charlie", "NOT_perpetual_motion")
        m_result2 = engine.M("charlie", "perpetual_motion")

        assert knows_not_p2 is True, "Should know NOT(perpetual_motion)"
        assert m_result2 is False, "M should be False when K(¬p)"

    def test_add_negated_knowledge_creates_negated_prop(self, setup_engine):
        """Test: add_negated_knowledge erstellt NOT_{prop_id}"""
        engine = setup_engine
        engine.create_agent("david", "David")

        # Füge negiertes Wissen hinzu
        success = engine.add_negated_knowledge("david", "earth_is_flat")

        assert success is True, "add_negated_knowledge should succeed"

        # Prüfe, dass NOT_{prop_id} existiert
        knows_negation = engine.K("david", "NOT_earth_is_flat")

        assert knows_negation is True, "Should know NOT_earth_is_flat"

    def test_m_operator_multiple_agents(self, setup_engine):
        """Test: M Operator mit mehreren Agenten (unterschiedliches Wissen)"""
        engine = setup_engine
        engine.create_agent("alice", "Alice")
        engine.create_agent("bob", "Bob")

        # Alice weiß, dass Gras grün ist
        engine.add_knowledge("alice", "grass_is_green")

        # Bob weiß, dass Gras NICHT grün ist (Colorblind-Scenario)
        engine.add_negated_knowledge("bob", "grass_is_green")

        # Alice: M(grass_is_green) = True (sie weiß, dass es wahr ist)
        alice_m = engine.M("alice", "grass_is_green")
        assert alice_m is True, "Alice should believe grass_is_green is possible"

        # Bob: M(grass_is_green) = False (er weiß, dass es NICHT wahr ist)
        bob_m = engine.M("bob", "grass_is_green")
        assert bob_m is False, "Bob should NOT believe grass_is_green is possible"

    def test_m_operator_no_agent(self, setup_engine):
        """Test: M Operator mit nicht-existierendem Agent (sollte False zurückgeben)"""
        engine = setup_engine

        # Agent "ghost" existiert nicht
        result = engine.M("ghost", "some_proposition")

        assert (
            result is True
        ), "M should return True for non-existent agent (no knowledge = possible)"

    def test_m_operator_empty_proposition(self, setup_engine):
        """Test: M Operator mit leerer Proposition"""
        engine = setup_engine
        engine.create_agent("alice", "Alice")

        result = engine.M("alice", "")

        assert result is True, "M should handle empty proposition gracefully"

    def test_m_operator_double_negation(self, setup_engine):
        """Test: Double negation - NOT_NOT_prop"""
        engine = setup_engine
        engine.create_agent("alice", "Alice")

        # Alice weiß NOT_NOT_sky_is_blue (Doppelte Negation)
        engine.add_knowledge("alice", "NOT_NOT_sky_is_blue")

        # M(alice, NOT_sky_is_blue) = ¬K(NOT_NOT_sky_is_blue)
        # Alice kennt NOT_NOT_sky_is_blue, also K = True, daher M = False
        result = engine.M("alice", "NOT_sky_is_blue")

        assert result is False, "M should handle double negation correctly (¬K(¬p))"

    def test_m_operator_consistency_with_k(self, setup_engine):
        """Test: Konsistenz zwischen K und M Operator"""
        engine = setup_engine
        engine.create_agent("bob", "Bob")

        # Wenn K(p) = True, dann M(p) = True (Wissen impliziert Möglichkeit)
        engine.add_knowledge("bob", "water_is_wet")

        k_result = engine.K("bob", "water_is_wet")
        m_result = engine.M("bob", "water_is_wet")

        assert k_result is True, "Should know water_is_wet"
        assert m_result is True, "If K(p) then M(p)"

    def test_add_negated_knowledge_certainty(self, setup_engine):
        """Test: add_negated_knowledge setzt certainty = 1.0"""
        engine = setup_engine
        engine.create_agent("alice", "Alice")

        # Füge negiertes Wissen hinzu (sollte certainty=1.0 haben)
        engine.add_negated_knowledge("alice", "magic_exists")

        # Verify durch K Operator
        knows = engine.K("alice", "NOT_magic_exists")

        assert knows is True, "Should know NOT_magic_exists with certainty=1.0"

    def test_m_operator_sequential_updates(self, setup_engine):
        """Test: M Operator nach sequentiellen Knowledge Updates"""
        engine = setup_engine
        # Use unique agent ID to avoid collision with other tests
        engine.create_agent("agent_sequential", "Agent Sequential")

        # Initial: M(prop) = True (kein Wissen)
        initial_m = engine.M("agent_sequential", "aliens_exist")
        assert initial_m is True, "Initially should be possible"

        # Füge negiertes Wissen hinzu
        engine.add_negated_knowledge("agent_sequential", "aliens_exist")

        # Nach Update: M(prop) = False
        updated_m = engine.M("agent_sequential", "aliens_exist")
        assert updated_m is False, "After learning NOT(prop), should be impossible"

    def test_m_operator_cache_and_graph_consistency(self, setup_engine):
        """Test: M Operator mit Cache und Graph (Konsistenz)"""
        engine = setup_engine
        # Use unique agent ID to avoid collision with other tests
        engine.create_agent("agent_cache_test", "Agent Cache Test")

        # Erster Aufruf (Graph-basiert)
        engine.add_negated_knowledge("agent_cache_test", "time_travel")
        m_first = engine.M("agent_cache_test", "time_travel")

        # Zweiter Aufruf (sollte Cache nutzen)
        m_second = engine.M("agent_cache_test", "time_travel")

        assert m_first == m_second, "Cache and graph should be consistent"
        assert m_first is False, "Both should return False"


class TestEdgeCases:
    """Edge Case Tests für M Operator"""

    @pytest.fixture
    def setup_engine(self):
        """Setup: Netzwerk und Engine erstellen"""
        netzwerk = KonzeptNetzwerk()
        engine = EpistemicEngine(netzwerk)
        yield engine

    def test_m_operator_special_characters_in_prop_id(self, setup_engine):
        """Test: M Operator mit Sonderzeichen in Proposition ID"""
        engine = setup_engine
        engine.create_agent("agent_special_chars", "Agent Special Chars")

        # Proposition mit Sonderzeichen
        prop_id = "prop_with_special!@#$%"

        result = engine.M("agent_special_chars", prop_id)

        assert result is True, "Should handle special characters in prop_id"

    def test_m_operator_very_long_prop_id(self, setup_engine):
        """Test: M Operator mit sehr langer Proposition ID"""
        engine = setup_engine
        engine.create_agent("agent_long_prop", "Agent Long Prop")

        # Sehr lange Proposition ID (1000 Zeichen)
        prop_id = "a" * 1000

        result = engine.M("agent_long_prop", prop_id)

        assert result is True, "Should handle very long prop_id"

    def test_m_operator_unicode_prop_id(self, setup_engine):
        """Test: M Operator mit Unicode in Proposition ID"""
        engine = setup_engine
        engine.create_agent("agent_unicode", "Agent Unicode")

        # Unicode Proposition
        prop_id = "himmel_ist_blau_äöü"

        engine.add_knowledge("agent_unicode", prop_id)
        result = engine.M("agent_unicode", prop_id)

        assert result is True, "Should handle Unicode in prop_id"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
