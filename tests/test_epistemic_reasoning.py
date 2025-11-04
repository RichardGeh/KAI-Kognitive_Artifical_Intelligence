"""
tests/test_epistemic_reasoning.py

Umfassende Tests für Generic Epistemic Reasoning System

Test Coverage:
1. LogicExpression Evaluation (K, M, PropertyEq, And/Or/Not)
2. Rule Triggering & Action Execution
3. Blue Eyes Puzzle (Klassischer Test: Tag N = N departures für N blue-eyed)
4. Reasoning Trace Generation
5. Fixed-Point Iteration
6. Event Propagation

Autor: KAI Development Team
Erstellt: 2025-11-01
"""

import pytest
from component_36_epistemic_reasoning import (
    EpistemicReasoner,
    EpistemicRule,
    K_Expr,
    M_Expr,
    PropertyEq,
    HasProperty,
    And,
    Or,
    Not,
    ForAll,
    Exists,
    SelfRef,
    create_blue_eyes_puzzle,
)
from component_35_epistemic_engine import EpistemicEngine
from component_1_netzwerk import KonzeptNetzwerk


@pytest.fixture
def netzwerk():
    """Provide clean KonzeptNetzwerk for each test"""
    return KonzeptNetzwerk()


@pytest.fixture
def engine(netzwerk):
    """Provide EpistemicEngine for each test"""
    return EpistemicEngine(netzwerk)


@pytest.fixture
def reasoner(engine):
    """Provide EpistemicReasoner for each test"""
    return EpistemicReasoner(engine)


class TestLogicExpressions:
    """Test Logic Expression DSL"""

    def test_property_eq_evaluation(self, reasoner):
        """Test PropertyEq expression"""
        reasoner.create_agent("alice", eye_color="blue")

        context = {"agent": "alice", "turn": 1}
        expr = PropertyEq("alice", "eye_color", "blue")

        assert expr.evaluate(reasoner, context) is True

        expr_false = PropertyEq("alice", "eye_color", "green")
        assert expr_false.evaluate(reasoner, context) is False

    def test_property_eq_with_selfref(self, reasoner):
        """Test PropertyEq with SelfRef"""
        reasoner.create_agent("bob", eye_color="brown")

        context = {"agent": "bob", "turn": 1}
        expr = PropertyEq(SelfRef(), "eye_color", "brown")

        assert expr.evaluate(reasoner, context) is True

    def test_has_property(self, reasoner):
        """Test HasProperty expression"""
        reasoner.create_agent("carol", age=25)

        context = {"agent": "carol", "turn": 1}
        expr = HasProperty("carol", "age")

        assert expr.evaluate(reasoner, context) is True

        expr_false = HasProperty("carol", "height")
        assert expr_false.evaluate(reasoner, context) is False

    def test_k_expr_evaluation(self, reasoner):
        """Test K_Expr (Knowledge operator)"""
        reasoner.create_agent("alice")
        reasoner.engine.add_knowledge("alice", "sky_is_blue")

        context = {"agent": "alice", "turn": 1}
        expr = K_Expr("alice", "sky_is_blue")

        assert expr.evaluate(reasoner, context) is True

        expr_false = K_Expr("alice", "grass_is_red")
        assert expr_false.evaluate(reasoner, context) is False

    def test_m_expr_evaluation(self, reasoner):
        """Test M_Expr (Belief operator)"""
        reasoner.create_agent("alice")

        # Alice doesn't know moon is NOT cheese -> considers it possible
        context = {"agent": "alice", "turn": 1}
        expr = M_Expr("alice", "moon_is_cheese")

        assert expr.evaluate(reasoner, context) is True

        # Alice knows moon is NOT cheese
        reasoner.engine.add_negated_knowledge("alice", "moon_is_cheese")
        assert expr.evaluate(reasoner, context) is False

    def test_and_expression(self, reasoner):
        """Test And combinator"""
        reasoner.create_agent("alice", age=30, city="berlin")

        context = {"agent": "alice", "turn": 1}
        expr = And(
            PropertyEq("alice", "age", 30), PropertyEq("alice", "city", "berlin")
        )

        assert expr.evaluate(reasoner, context) is True

        expr_false = And(
            PropertyEq("alice", "age", 30), PropertyEq("alice", "city", "paris")
        )
        assert expr_false.evaluate(reasoner, context) is False

    def test_or_expression(self, reasoner):
        """Test Or combinator"""
        reasoner.create_agent("bob", eye_color="blue")

        context = {"agent": "bob", "turn": 1}
        expr = Or(
            PropertyEq("bob", "eye_color", "blue"),
            PropertyEq("bob", "eye_color", "green"),
        )

        assert expr.evaluate(reasoner, context) is True

    def test_not_expression(self, reasoner):
        """Test Not combinator"""
        reasoner.create_agent("carol", eye_color="brown")

        context = {"agent": "carol", "turn": 1}
        expr = Not(PropertyEq("carol", "eye_color", "blue"))

        assert expr.evaluate(reasoner, context) is True

    def test_operator_overloading(self, reasoner):
        """Test operator overloading (&, |, ~)"""
        reasoner.create_agent("alice", age=25, city="berlin")

        context = {"agent": "alice", "turn": 1}

        # Test & (and)
        expr1 = PropertyEq("alice", "age", 25) & PropertyEq("alice", "city", "berlin")
        assert expr1.evaluate(reasoner, context) is True

        # Test | (or)
        expr2 = PropertyEq("alice", "age", 30) | PropertyEq("alice", "city", "berlin")
        assert expr2.evaluate(reasoner, context) is True

        # Test ~ (not)
        expr3 = ~PropertyEq("alice", "age", 30)
        assert expr3.evaluate(reasoner, context) is True

    def test_forall_quantifier(self, reasoner):
        """Test ForAll quantifier"""
        agents = ["alice", "bob", "carol"]
        for agent in agents:
            reasoner.create_agent(agent, status="active")

        context = {"turn": 1}
        expr = ForAll(agents, lambda agent: PropertyEq(agent, "status", "active"))

        assert expr.evaluate(reasoner, context) is True

    def test_exists_quantifier(self, reasoner):
        """Test Exists quantifier"""
        reasoner.create_agent("alice", eye_color="brown")
        reasoner.create_agent("bob", eye_color="blue")
        reasoner.create_agent("carol", eye_color="brown")

        agents = ["alice", "bob", "carol"]
        context = {"turn": 1}
        expr = Exists(agents, lambda agent: PropertyEq(agent, "eye_color", "blue"))

        assert expr.evaluate(reasoner, context) is True


class TestEpistemicRules:
    """Test Epistemic Rule System"""

    def test_rule_creation(self):
        """Test rule creation"""
        rule = EpistemicRule(
            name="test_rule",
            condition=PropertyEq("alice", "ready", True),
            action=lambda agent, r, ctx: ctx["actions"].append((agent, "execute")),
            priority=5,
        )

        assert rule.name == "test_rule"
        assert rule.priority == 5
        assert rule.timing == "immediate"

    def test_rule_condition_check(self, reasoner):
        """Test rule condition checking"""
        reasoner.create_agent("alice", ready=True)

        rule = EpistemicRule(
            name="check_ready",
            condition=PropertyEq(SelfRef(), "ready", True),
            action=lambda agent, r, ctx: None,
        )

        context = {"turn": 1}
        assert rule.check_condition("alice", reasoner, context) is True

    def test_rule_action_trigger(self, reasoner):
        """Test rule action triggering"""
        reasoner.create_agent("alice", ready=True)

        actions_log = []

        rule = EpistemicRule(
            name="execute_when_ready",
            condition=PropertyEq(SelfRef(), "ready", True),
            action=lambda agent, r, ctx: actions_log.append(agent),
        )

        context = {"turn": 1}
        rule.trigger_action("alice", reasoner, context)

        assert "alice" in actions_log

    def test_rule_priority_ordering(self, reasoner):
        """Test that rules are sorted by priority"""
        rule1 = EpistemicRule(
            name="low_priority",
            condition=PropertyEq("alice", "x", 1),
            action=lambda a, r, c: None,
            priority=1,
        )

        rule2 = EpistemicRule(
            name="high_priority",
            condition=PropertyEq("alice", "x", 1),
            action=lambda a, r, c: None,
            priority=10,
        )

        reasoner.add_rule(rule1)
        reasoner.add_rule(rule2)

        # Rules should be sorted by priority (highest first)
        assert reasoner.rules[0].name == "high_priority"
        assert reasoner.rules[1].name == "low_priority"


class TestEpistemicReasoner:
    """Test EpistemicReasoner core functionality"""

    def test_create_agent(self, reasoner):
        """Test agent creation"""
        reasoner.create_agent("alice", age=30, city="berlin")

        assert "alice" in reasoner.state.active_agents
        assert reasoner.state.agent_properties["alice"]["age"] == 30
        assert reasoner.state.agent_properties["alice"]["city"] == "berlin"

    def test_add_rule(self, reasoner):
        """Test rule addition"""
        rule = EpistemicRule(
            name="test_rule",
            condition=PropertyEq("alice", "x", 1),
            action=lambda a, r, c: None,
        )

        reasoner.add_rule(rule)
        assert len(reasoner.rules) == 1
        assert reasoner.rules[0].name == "test_rule"

    def test_observation_rule(self, reasoner):
        """Test observation rule application"""
        reasoner.create_agent("alice", eye_color="blue")
        reasoner.create_agent("bob", eye_color="brown")

        # Everyone observes others' eye color
        reasoner.add_observation_rule("eye_color")

        # Alice should know Bob's eye color
        prop_id = "bob_has_eye_color_brown"
        assert reasoner.engine.K_n("alice", ["bob"], prop_id) is True

        # Bob should know Alice's eye color
        prop_id = "alice_has_eye_color_blue"
        assert reasoner.engine.K_n("bob", ["alice"], prop_id) is True

    def test_simulate_step(self, reasoner):
        """Test single reasoning step"""
        reasoner.create_agent("alice", ready=True)

        rule = EpistemicRule(
            name="depart_when_ready",
            condition=PropertyEq(SelfRef(), "ready", True),
            action=lambda agent, r, ctx: ctx["actions"].append((agent, "depart")),
        )
        reasoner.add_rule(rule)

        actions = reasoner.simulate_step()
        assert "alice" in actions

    def test_event_propagation(self, reasoner):
        """Test event propagation after action"""
        reasoner.create_agent("alice", x=1)
        reasoner.create_agent("bob", x=1)

        # Alice departs
        reasoner._propagate_event("alice", "departed")

        # Alice should be removed from active agents
        assert "alice" not in reasoner.state.active_agents
        assert "alice" in reasoner.state.departed_agents

    def test_reasoning_trace(self, reasoner):
        """Test reasoning trace generation"""
        reasoner.create_agent("alice", ready=True)

        rule = EpistemicRule(
            name="trace_test",
            condition=PropertyEq(SelfRef(), "ready", True),
            action=lambda agent, r, ctx: ctx["actions"].append((agent, "act")),
        )
        reasoner.add_rule(rule)

        reasoner.simulate_step()

        trace = reasoner.get_reasoning_trace()
        assert len(trace) > 0
        assert trace[0].rule_name == "trace_test"

    def test_export_state(self, reasoner):
        """Test state export"""
        reasoner.create_agent("alice", x=1)
        reasoner.create_agent("bob", x=2)

        state = reasoner.export_state()

        assert state["turn"] == 0
        assert len(state["active_agents"]) == 2
        assert "alice" in state["active_agents"]


class TestBlueEyesPuzzle:
    """Test Blue Eyes Puzzle (Classic Epistemic Logic Problem)"""

    def test_blue_eyes_3_people(self, reasoner):
        """Test Blue Eyes with 3 people, 3 blue-eyed"""
        solution = create_blue_eyes_puzzle(reasoner, num_people=3, num_blue_eyes=3)

        # Expected: All 3 depart on day 3
        assert 3 in solution
        assert len(solution[3]) == 3

    def test_blue_eyes_5_people_2_blue(self, reasoner):
        """Test Blue Eyes with 5 people, 2 blue-eyed"""
        solution = create_blue_eyes_puzzle(reasoner, num_people=5, num_blue_eyes=2)

        # Expected: 2 depart on day 2
        assert 2 in solution
        assert len(solution[2]) == 2

    def test_blue_eyes_10_people_1_blue(self, reasoner):
        """Test Blue Eyes with 10 people, 1 blue-eyed"""
        solution = create_blue_eyes_puzzle(reasoner, num_people=10, num_blue_eyes=1)

        # Expected: 1 departs on day 1
        assert 1 in solution
        assert len(solution[1]) == 1

    @pytest.mark.slow
    def test_blue_eyes_100_people_10_blue(self, reasoner):
        """Test Blue Eyes with 100 people, 10 blue-eyed (SLOW)"""
        solution = create_blue_eyes_puzzle(reasoner, num_people=100, num_blue_eyes=10)

        # Expected: All 10 depart on day 10
        assert 10 in solution
        assert len(solution[10]) == 10

    def test_blue_eyes_no_blue_eyes(self, reasoner):
        """Test Blue Eyes with no blue-eyed people"""
        # Edge case: If nobody has blue eyes, nobody departs
        # (But common knowledge "someone has blue eyes" would be violated)
        solution = create_blue_eyes_puzzle(reasoner, num_people=5, num_blue_eyes=0)

        # Expected: No departures
        assert len(solution) == 0

    def test_blue_eyes_all_blue_eyes(self, reasoner):
        """Test Blue Eyes where everyone has blue eyes"""
        solution = create_blue_eyes_puzzle(reasoner, num_people=5, num_blue_eyes=5)

        # Expected: All 5 depart on day 5
        assert 5 in solution
        assert len(solution[5]) == 5


class TestFixedPointReasoning:
    """Test Fixed-Point Reasoning Loop"""

    def test_fixed_point_convergence(self, reasoner):
        """Test that reasoning converges to fixed point"""
        reasoner.create_agent("alice", counter=0)

        # Rule that never triggers (always false condition)
        rule = EpistemicRule(
            name="never_trigger",
            condition=PropertyEq("alice", "counter", 999),
            action=lambda agent, r, ctx: ctx["actions"].append((agent, "act")),
        )
        reasoner.add_rule(rule)

        solution = reasoner.solve(max_iterations=10)

        # Expected: No actions, converges immediately
        assert len(solution) == 0

    def test_max_iterations_limit(self, reasoner):
        """Test that solve respects max_iterations"""
        reasoner.create_agent("alice", x=1)

        solution = reasoner.solve(max_iterations=5)

        # Should stop at turn 5 even if not converged
        assert reasoner.state.turn <= 5


class TestComplexScenarios:
    """Test complex multi-agent scenarios"""

    def test_multiple_agents_different_properties(self, reasoner):
        """Test reasoning with heterogeneous agents"""
        reasoner.create_agent("alice", role="leader", ready=True)
        reasoner.create_agent("bob", role="follower", ready=False)
        reasoner.create_agent("carol", role="follower", ready=True)

        actions_log = []

        rule = EpistemicRule(
            name="leaders_act_when_ready",
            condition=And(
                PropertyEq(SelfRef(), "role", "leader"),
                PropertyEq(SelfRef(), "ready", True),
            ),
            action=lambda agent, r, ctx: actions_log.append(agent),
        )
        reasoner.add_rule(rule)

        reasoner.simulate_step()

        # Only alice should act
        assert "alice" in actions_log
        assert "bob" not in actions_log
        assert "carol" not in actions_log

    def test_sequential_rule_execution(self, reasoner):
        """Test that rules execute in priority order"""
        reasoner.create_agent("alice", state="start")

        execution_order = []

        rule1 = EpistemicRule(
            name="low_priority",
            condition=PropertyEq(SelfRef(), "state", "start"),
            action=lambda a, r, c: execution_order.append("low"),
            priority=1,
        )

        rule2 = EpistemicRule(
            name="high_priority",
            condition=PropertyEq(SelfRef(), "state", "start"),
            action=lambda a, r, c: execution_order.append("high"),
            priority=10,
        )

        reasoner.add_rule(rule1)
        reasoner.add_rule(rule2)

        reasoner.simulate_step()

        # High priority should execute first
        assert execution_order[0] == "high"
        assert execution_order[1] == "low"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
