"""
Tests für component_38_elimination_reasoning.py

Testet generische Elimination & Deduktion ohne puzzle-spezifische Logik.
"""

import pytest
from component_38_elimination_reasoning import (
    StatementType,
    AgentStatement,
    EliminationRule,
    DeductiveChain,
    EliminationContext,
    EliminationReasoner,
)
from component_37_partial_observation import (
    WorldObject,
    PartialObserver,
    PartialObservationReasoner,
)
from component_35_epistemic_engine import EpistemicEngine
from component_1_netzwerk import KonzeptNetzwerk


class TestAgentStatement:
    """Test agent statement data structure"""

    def test_create_statement(self):
        """Test creating agent statement"""
        stmt = AgentStatement(
            speaker="alice", statement_type=StatementType.I_KNOW, turn=1
        )

        assert stmt.speaker == "alice"
        assert stmt.statement_type == StatementType.I_KNOW
        assert stmt.turn == 1

    def test_meta_statement(self):
        """Test statement about another agent"""
        stmt = AgentStatement(
            speaker="alice",
            statement_type=StatementType.I_KNOW_OTHER_DOESNT_KNOW,
            about_agent="bob",
            turn=1,
        )

        assert stmt.speaker == "alice"
        assert stmt.about_agent == "bob"
        assert stmt.statement_type == StatementType.I_KNOW_OTHER_DOESNT_KNOW


class TestEliminationRule:
    """Test elimination rules"""

    def test_simple_elimination_rule(self):
        """Test applying simple elimination rule"""
        objects = [
            WorldObject("obj1", {"value": 1}),
            WorldObject("obj2", {"value": 2}),
            WorldObject("obj3", {"value": 3}),
        ]

        # Rule: Keep only objects with value > 1
        def filter_fn(obj, stmt, ctx):
            return obj.get_property("value") > 1

        rule = EliminationRule(
            name="eliminate_value_le_1",
            statement_type=StatementType.I_KNOW,
            filter_predicate=filter_fn,
            explanation_template="Eliminated {eliminated_count} objects",
        )

        # Create mock context
        netzwerk = KonzeptNetzwerk()
        engine = EpistemicEngine(netzwerk)
        reasoner = PartialObservationReasoner(engine)
        reasoner.add_objects(objects)

        context = EliminationContext(
            reasoner=reasoner, observers={}, current_candidates=objects
        )

        stmt = AgentStatement("alice", StatementType.I_KNOW, turn=1)

        # Apply rule
        filtered, explanation = rule.apply(objects, stmt, context)

        assert len(filtered) == 2
        assert filtered[0].object_id == "obj2"
        assert filtered[1].object_id == "obj3"
        assert "1" in explanation  # Eliminated 1 object (obj1)


class TestDeductiveChain:
    """Test deductive chain"""

    def test_create_chain(self):
        """Test creating deductive chain"""
        objects = [WorldObject("obj1", {"x": 1}), WorldObject("obj2", {"x": 2})]

        chain = DeductiveChain(objects)

        assert len(chain.steps) == 0
        assert len(chain.current_candidates) == 2
        assert chain.initial_count == 2

    def test_add_step(self):
        """Test adding deductive step"""
        objects = [
            WorldObject("obj1", {"x": 1}),
            WorldObject("obj2", {"x": 2}),
            WorldObject("obj3", {"x": 3}),
        ]

        chain = DeductiveChain(objects)

        stmt = AgentStatement("alice", StatementType.I_KNOW, turn=1)
        eliminated = [objects[0]]  # Eliminate obj1

        chain.add_step(stmt, eliminated, "Eliminated obj1")

        assert len(chain.steps) == 1
        assert len(chain.current_candidates) == 2
        assert chain.steps[0].turn == 1

    def test_is_solved(self):
        """Test checking if unique solution found"""
        objects = [WorldObject("obj1", {"x": 1}), WorldObject("obj2", {"x": 2})]

        chain = DeductiveChain(objects)
        assert chain.is_solved() is False

        # Eliminate one
        stmt = AgentStatement("alice", StatementType.I_KNOW, turn=1)
        chain.add_step(stmt, [objects[0]], "Eliminated obj1")

        assert chain.is_solved() is True
        assert chain.get_solution().object_id == "obj2"

    def test_is_contradictory(self):
        """Test detecting contradiction"""
        objects = [WorldObject("obj1", {"x": 1})]

        chain = DeductiveChain(objects)
        assert chain.is_contradictory() is False

        # Eliminate all
        stmt = AgentStatement("alice", StatementType.I_KNOW, turn=1)
        chain.add_step(stmt, objects, "Eliminated all")

        assert chain.is_contradictory() is True

    def test_generate_proof_tree(self):
        """Test generating proof tree from chain"""
        objects = [WorldObject("obj1", {"x": 1}), WorldObject("obj2", {"x": 2})]

        chain = DeductiveChain(objects)

        stmt = AgentStatement("alice", StatementType.I_KNOW, turn=1)
        chain.add_step(stmt, [objects[0]], "Eliminated obj1")

        proof_tree = chain.generate_proof_tree("Test Query")

        assert proof_tree.query == "Test Query"
        assert len(proof_tree.root_steps) >= 2  # Initial + elimination + conclusion


class TestEliminationContext:
    """Test elimination context"""

    def test_get_observer_observation(self):
        """Test getting observer's observation of object"""
        netzwerk = KonzeptNetzwerk()
        engine = EpistemicEngine(netzwerk)
        reasoner = PartialObservationReasoner(engine)

        obj = WorldObject("obj1", {"x": 1, "y": 2})
        objects = [obj]
        reasoner.add_objects(objects)

        observer = PartialObserver("obs1", observable_properties=["x"])
        reasoner.add_observer(observer)

        context = EliminationContext(
            reasoner=reasoner, observers={"obs1": observer}, current_candidates=objects
        )

        observation = context.get_observer_observation("obs1", obj)

        assert observation == {"x": 1}
        assert "y" not in observation

    def test_can_observer_identify(self):
        """Test checking if observer can identify object"""
        netzwerk = KonzeptNetzwerk()
        engine = EpistemicEngine(netzwerk)
        reasoner = PartialObservationReasoner(engine)

        objects = [
            WorldObject("obj1", {"x": 1, "y": 2}),
            WorldObject("obj2", {"x": 1, "y": 3}),
        ]
        reasoner.add_objects(objects)

        observer = PartialObserver("obs1", observable_properties=["x"])
        reasoner.add_observer(observer)

        context = EliminationContext(
            reasoner=reasoner, observers={"obs1": observer}, current_candidates=objects
        )

        # obs1 sees x=1 for both → cannot identify
        assert context.can_observer_identify("obs1", objects[0]) is False

        # After eliminating obj2
        context.current_candidates = [objects[0]]

        # Now obs1 can identify (only one candidate)
        assert context.can_observer_identify("obs1", objects[0]) is True


class TestEliminationReasoner:
    """Test elimination reasoner"""

    def test_create_reasoner(self):
        """Test creating elimination reasoner"""
        netzwerk = KonzeptNetzwerk()
        engine = EpistemicEngine(netzwerk)
        reasoner = PartialObservationReasoner(engine)

        elim_reasoner = EliminationReasoner(reasoner)

        assert len(elim_reasoner.rules) == 0

    def test_add_rule(self):
        """Test adding rule to reasoner"""
        netzwerk = KonzeptNetzwerk()
        engine = EpistemicEngine(netzwerk)
        reasoner = PartialObservationReasoner(engine)
        elim_reasoner = EliminationReasoner(reasoner)

        rule = EliminationRule(
            name="test_rule",
            statement_type=StatementType.I_KNOW,
            filter_predicate=lambda obj, stmt, ctx: True,
            explanation_template="Test",
        )

        elim_reasoner.add_rule(rule)

        assert len(elim_reasoner.rules) == 1

    def test_create_standard_rules(self):
        """Test creating standard elimination rules"""
        netzwerk = KonzeptNetzwerk()
        engine = EpistemicEngine(netzwerk)
        reasoner = PartialObservationReasoner(engine)
        elim_reasoner = EliminationReasoner(reasoner)

        elim_reasoner.create_standard_rules()

        # Should create at least 3 standard rules
        assert len(elim_reasoner.rules) >= 3

    def test_process_simple_statements(self):
        """Test processing simple statement sequence"""
        netzwerk = KonzeptNetzwerk()
        engine = EpistemicEngine(netzwerk)
        reasoner = PartialObservationReasoner(engine)

        # Setup objects
        objects = [
            WorldObject("obj1", {"value": 1}),
            WorldObject("obj2", {"value": 2}),
            WorldObject("obj3", {"value": 3}),
        ]
        reasoner.add_objects(objects)

        # Observer sees only value
        observer = PartialObserver("alice", observable_properties=["value"])
        reasoner.add_observer(observer)
        engine.create_agent("alice", "Alice")

        # Elimination reasoner
        elim_reasoner = EliminationReasoner(reasoner)

        # Custom rule: Keep only value >= 2
        def filter_fn(obj, stmt, ctx):
            return obj.get_property("value") >= 2

        rule = EliminationRule(
            name="eliminate_lt_2",
            statement_type=StatementType.I_KNOW,
            filter_predicate=filter_fn,
            explanation_template="Eliminated {eliminated_count} objects with value < 2",
        )
        elim_reasoner.add_rule(rule)

        # Process statement
        statements = [AgentStatement("alice", StatementType.I_KNOW, turn=1)]

        solution, proof_tree = elim_reasoner.process_statements(objects, statements)

        # Should not have unique solution (2 and 3 remain)
        assert solution is None
        assert len(elim_reasoner.deductive_chain.current_candidates) == 2

    def test_process_to_unique_solution(self):
        """Test processing statements to unique solution"""
        netzwerk = KonzeptNetzwerk()
        engine = EpistemicEngine(netzwerk)
        reasoner = PartialObservationReasoner(engine)

        # Setup objects with unique identifiers
        objects = [
            WorldObject("obj1", {"type": "A", "number": 1}),
            WorldObject("obj2", {"type": "B", "number": 2}),
            WorldObject("obj3", {"type": "C", "number": 3}),
        ]
        reasoner.add_objects(objects)

        # Observer sees only number
        observer = PartialObserver("alice", observable_properties=["number"])
        reasoner.add_observer(observer)
        engine.create_agent("alice", "Alice")

        # Elimination reasoner with standard rules
        elim_reasoner = EliminationReasoner(reasoner)
        elim_reasoner.create_standard_rules()

        # Since all numbers are unique, "I know" should identify unique object
        statements = [AgentStatement("alice", StatementType.I_KNOW, turn=1)]

        # Each object is uniquely identifiable by number → all satisfy "I know"
        # So no elimination happens with standard rules alone

        # Let's add custom filter
        def filter_only_obj2(obj, stmt, ctx):
            return obj.object_id == "obj2"

        rule = EliminationRule(
            name="keep_only_obj2",
            statement_type=StatementType.I_KNOW,
            filter_predicate=filter_only_obj2,
            explanation_template="Kept only obj2",
            priority=20,  # Higher priority than standard rules
        )
        elim_reasoner.add_rule(rule)

        solution, proof_tree = elim_reasoner.process_statements(objects, statements)

        assert solution is not None
        assert solution.object_id == "obj2"


class TestIntegrationScenarios:
    """Integration tests with realistic scenarios"""

    def test_two_agent_unique_identification(self):
        """Test scenario where one agent can uniquely identify"""
        netzwerk = KonzeptNetzwerk()
        engine = EpistemicEngine(netzwerk)
        reasoner = PartialObservationReasoner(engine)

        # Setup: Objects with partial uniqueness
        objects = [
            WorldObject("obj1", {"x": 1, "y": 10}),  # y=10 unique
            WorldObject("obj2", {"x": 1, "y": 20}),  # y=20 unique
            WorldObject("obj3", {"x": 2, "y": 30}),  # x=2 unique, y=30 unique
        ]
        reasoner.add_objects(objects)

        # Observer X sees only x
        obs_x = PartialObserver("alice", observable_properties=["x"])
        # Observer Y sees only y
        obs_y = PartialObserver("bob", observable_properties=["y"])

        reasoner.add_observer(obs_x)
        reasoner.add_observer(obs_y)

        engine.create_agent("alice", "Alice")
        engine.create_agent("bob", "Bob")

        # Bob can always identify (all y values unique)
        # Alice cannot identify x=1 (two objects)

        elim_reasoner = EliminationReasoner(reasoner)
        elim_reasoner.create_standard_rules()

        # Bob says "I know"
        statements = [AgentStatement("bob", StatementType.I_KNOW, turn=1)]

        # All objects have unique y → all should remain
        solution, proof_tree = elim_reasoner.process_statements(objects, statements)

        # Should not reduce candidates (all satisfy "bob can identify")
        assert len(elim_reasoner.deductive_chain.current_candidates) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
