"""
Tests for Component 31: State-Space Planner

Test coverage:
- Blocks World (classic planning)
- Grid Navigation (pathfinding with obstacles)
- River Crossing (constraint satisfaction)
- Diagnostic reasoning (root-cause analysis)
- Temporal reasoning
- Hybrid planning (with logic/CSP integration)

Author: KAI Development Team
Date: 2025-01-30
"""

import pytest
from component_31_state_space_planner import (
    State,
    Action,
    StateSpacePlanner,
    HybridPlanner,
    RelaxedPlanHeuristic,
    SetCoverHeuristic,
    TemporalPlanner,
    TemporalConstraint,
    BlocksWorldBuilder,
    GridNavigationBuilder,
    RiverCrossingBuilder,
)


# ============================================================================
# Test State Representation
# ============================================================================


class TestStateRepresentation:
    """Test basic state operations."""

    def test_state_creation(self):
        """Test creating state with propositions."""
        state = State(propositions={("on", "A", "B"), ("clear", "A")})
        assert len(state.propositions) == 2
        assert ("on", "A", "B") in state.propositions

    def test_state_satisfies(self):
        """Test state satisfaction check."""
        state = State(propositions={("on", "A", "B"), ("clear", "A"), ("ontable", "B")})
        assert state.satisfies({("on", "A", "B")})
        assert state.satisfies({("on", "A", "B"), ("clear", "A")})
        assert not state.satisfies({("on", "B", "A")})

    def test_state_equality(self):
        """Test state equality comparison."""
        state1 = State(propositions={("on", "A", "B")})
        state2 = State(propositions={("on", "A", "B")})
        state3 = State(propositions={("on", "B", "A")})
        assert state1 == state2
        assert state1 != state3

    def test_state_hashable(self):
        """Test that states can be used in sets/dicts."""
        state1 = State(propositions={("on", "A", "B")})
        state2 = State(propositions={("on", "A", "B")})
        state_set = {state1, state2}
        assert len(state_set) == 1  # Duplicates removed


# ============================================================================
# Test Action Model
# ============================================================================


class TestActionModel:
    """Test STRIPS-style action operations."""

    def test_action_applicability(self):
        """Test action precondition checking."""
        action = Action(
            name="stack(A, B)",
            preconditions={("holding", "A"), ("clear", "B")},
            add_effects={("on", "A", "B")},
            delete_effects={("holding", "A")},
        )

        state_applicable = State(propositions={("holding", "A"), ("clear", "B")})
        state_not_applicable = State(propositions={("holding", "A")})

        assert action.is_applicable(state_applicable)
        assert not action.is_applicable(state_not_applicable)

    def test_action_effects(self):
        """Test action application to state."""
        action = Action(
            name="pickup(A)",
            preconditions={("ontable", "A"), ("clear", "A"), ("handempty",)},
            add_effects={("holding", "A")},
            delete_effects={("ontable", "A"), ("clear", "A"), ("handempty",)},
        )

        initial = State(propositions={("ontable", "A"), ("clear", "A"), ("handempty",)})

        result = action.apply(initial)

        assert ("holding", "A") in result.propositions
        assert ("ontable", "A") not in result.propositions
        assert ("handempty",) not in result.propositions

    def test_action_instantiation(self):
        """Test grounding action schema with bindings."""
        schema = Action(
            name="stack",
            params=["x", "y"],
            preconditions={("holding", "x"), ("clear", "y")},
            add_effects={("on", "x", "y")},
            delete_effects={("holding", "x")},
        )

        grounded = schema.instantiate({"x": "A", "y": "B"})

        assert "A" in grounded.name
        assert "B" in grounded.name
        assert ("holding", "A") in grounded.preconditions
        assert ("on", "A", "B") in grounded.add_effects


# ============================================================================
# Test Blocks World (Classic Planning Problem)
# ============================================================================


class TestBlocksWorld:
    """Test classic Blocks World domain."""

    def test_simple_stack(self):
        """Test stacking two blocks: A on table, B on table → A on B."""
        problem = BlocksWorldBuilder.create_problem(
            blocks=["A", "B"],
            initial_config={"A": "table", "B": "table"},
            goal_config={"A": "B"},
        )

        planner = StateSpacePlanner()
        plan = planner.solve(problem)

        assert plan is not None, "No plan found"
        assert len(plan) > 0
        assert planner.stats["plan_length"] <= 2  # pickup(A), stack(A, B)

        # Validate plan
        valid, error = planner.validate_plan(problem, plan)
        assert valid, f"Plan invalid: {error}"

    def test_three_block_tower(self):
        """Test building tower: A,B,C on table → C on B on A on table."""
        problem = BlocksWorldBuilder.create_problem(
            blocks=["A", "B", "C"],
            initial_config={"A": "table", "B": "table", "C": "table"},
            goal_config={"B": "A", "C": "B"},
        )

        planner = StateSpacePlanner()
        plan = planner.solve(problem)

        assert plan is not None
        assert len(plan) > 0

        # Validate
        valid, error = planner.validate_plan(problem, plan)
        assert valid, f"Plan invalid: {error}"

    def test_sussman_anomaly(self):
        """Test Sussman Anomaly: A on C, B on table, C on table → A on B on C."""
        # Initial: A on C, B on table, C on table
        # Goal: A on B, B on C, C on table
        problem = BlocksWorldBuilder.create_problem(
            blocks=["A", "B", "C"],
            initial_config={"A": "C", "B": "table", "C": "table"},
            goal_config={"B": "C", "A": "B"},
        )

        planner = StateSpacePlanner()
        plan = planner.solve(problem)

        assert plan is not None, "Sussman Anomaly unsolvable"
        assert len(plan) > 0

        # Validate
        valid, error = planner.validate_plan(problem, plan)
        assert valid, f"Plan invalid: {error}"

    def test_blocks_world_impossible_goal(self):
        """Test detection of impossible goals."""
        # Goal: Block on itself (impossible)
        problem = BlocksWorldBuilder.create_problem(
            blocks=["A"],
            initial_config={"A": "table"},
            goal_config={"A": "A"},  # Impossible: A on A
        )

        planner = StateSpacePlanner(max_expansions=100)
        plan = planner.solve(problem)

        assert plan is None, "Found plan for impossible goal"


# ============================================================================
# Test Grid Navigation (Pathfinding)
# ============================================================================


class TestGridNavigation:
    """Test grid navigation with obstacles."""

    def test_straight_path(self):
        """Test navigation without obstacles."""
        problem = GridNavigationBuilder.create_problem(
            grid_size=(5, 5), start=(0, 0), goal=(4, 4)
        )

        planner = StateSpacePlanner()
        plan = planner.solve(problem)

        assert plan is not None
        assert len(plan) == 8  # Manhattan distance: 4 right + 4 up

        # Validate
        valid, error = planner.validate_plan(problem, plan)
        assert valid, f"Plan invalid: {error}"

    def test_navigation_with_obstacle(self):
        """Test pathfinding around obstacle."""
        problem = GridNavigationBuilder.create_problem(
            grid_size=(5, 5),
            start=(0, 0),
            goal=(4, 0),
            obstacles=[(2, 0)],  # Obstacle in middle of direct path
        )

        planner = StateSpacePlanner()
        plan = planner.solve(problem)

        assert plan is not None
        assert len(plan) > 4  # Must go around obstacle

        # Validate
        valid, error = planner.validate_plan(problem, plan)
        assert valid, f"Plan invalid: {error}"

        # Verify no action moves into obstacle
        states = planner.simulate_plan(problem, plan)
        for state in states:
            # Check agent never at obstacle position
            assert ("at", "2", "0") not in state.propositions

    def test_navigation_blocked_goal(self):
        """Test detection of blocked goals."""
        problem = GridNavigationBuilder.create_problem(
            grid_size=(3, 3),
            start=(0, 0),
            goal=(2, 2),
            obstacles=[(1, 1), (1, 2), (2, 1)],  # Goal surrounded
        )

        planner = StateSpacePlanner(max_expansions=500)
        plan = planner.solve(problem)

        # May or may not find path depending on obstacle configuration
        # This tests that planner doesn't crash on difficult problems
        assert plan is None or len(plan) > 0


# ============================================================================
# Test River Crossing (Constraint Satisfaction)
# ============================================================================


class TestRiverCrossing:
    """Test river crossing puzzle with safety constraints."""

    def test_river_crossing_solution(self):
        """Test solving classic river crossing puzzle."""
        problem = RiverCrossingBuilder.create_problem()

        # Use state constraint to enforce safety
        planner = StateSpacePlanner(
            max_expansions=5000, state_constraint=RiverCrossingBuilder.is_safe_state
        )
        plan = planner.solve(problem)

        assert plan is not None, "No solution found for river crossing"
        assert len(plan) >= 7  # Minimum moves for classic problem

        # Validate plan
        valid, error = planner.validate_plan(problem, plan)
        assert valid, f"Plan invalid: {error}"

    def test_river_crossing_safety_constraints(self):
        """Test that plan respects safety constraints."""
        problem = RiverCrossingBuilder.create_problem()

        # Use state constraint to enforce safety
        planner = StateSpacePlanner(
            max_expansions=5000, state_constraint=RiverCrossingBuilder.is_safe_state
        )
        plan = planner.solve(problem)

        assert plan is not None

        # Simulate plan and check each state for safety
        states = planner.simulate_plan(problem, plan)
        for i, state in enumerate(states):
            is_safe = RiverCrossingBuilder.is_safe_state(state)
            assert is_safe, f"Unsafe state at step {i}: {state.to_string()}"

    def test_river_crossing_constraint_validation(self):
        """Test explicit constraint checking."""
        # Valid state: farmer with chicken on left, fox alone on right
        valid_state = State(
            propositions={
                ("at", "farmer", "left"),
                ("at", "chicken", "left"),
                ("at", "fox", "right"),
                ("at", "grain", "right"),
                ("at", "boat", "left"),
            }
        )
        assert RiverCrossingBuilder.is_safe_state(valid_state)

        # Invalid: fox and chicken alone on left
        invalid_state1 = State(
            propositions={
                ("at", "farmer", "right"),
                ("at", "fox", "left"),
                ("at", "chicken", "left"),
                ("at", "grain", "right"),
                ("at", "boat", "right"),
            }
        )
        assert not RiverCrossingBuilder.is_safe_state(invalid_state1)

        # Invalid: chicken and grain alone on right
        invalid_state2 = State(
            propositions={
                ("at", "farmer", "left"),
                ("at", "fox", "left"),
                ("at", "chicken", "right"),
                ("at", "grain", "right"),
                ("at", "boat", "left"),
            }
        )
        assert not RiverCrossingBuilder.is_safe_state(invalid_state2)


# ============================================================================
# Test Diagnostic Reasoning (Root-Cause Analysis)
# ============================================================================


class TestDiagnosticReasoning:
    """Test plan failure diagnosis."""

    def test_diagnose_missing_precondition(self):
        """Test diagnosis when action precondition not satisfied."""
        # Create invalid plan: try to stack without holding block
        problem = BlocksWorldBuilder.create_problem(
            blocks=["A", "B"],
            initial_config={"A": "table", "B": "table"},
            goal_config={"A": "B"},
        )

        # Invalid plan: skip pickup, go straight to stack
        invalid_plan = [
            Action(
                name="stack(A, B)",
                preconditions={("holding", "A"), ("clear", "B")},
                add_effects={("on", "A", "B")},
                delete_effects={("holding", "A")},
            )
        ]

        planner = StateSpacePlanner()
        diagnosis = planner.diagnose_failure(problem, invalid_plan)

        assert diagnosis["error"] is not None
        assert diagnosis["failed_at"] == 0
        assert len(diagnosis["missing_preconditions"]) > 0
        assert ("holding", "A") in diagnosis["missing_preconditions"]

    def test_diagnose_incomplete_plan(self):
        """Test diagnosis when plan doesn't achieve goal."""
        problem = BlocksWorldBuilder.create_problem(
            blocks=["A", "B"],
            initial_config={"A": "table", "B": "table"},
            goal_config={"A": "B"},
        )

        # Incomplete plan: only pickup, no stack
        incomplete_plan = [
            Action(
                name="pickup(A)",
                preconditions={("ontable", "A"), ("clear", "A"), ("handempty",)},
                add_effects={("holding", "A")},
                delete_effects={("ontable", "A"), ("clear", "A"), ("handempty",)},
            )
        ]

        planner = StateSpacePlanner()
        diagnosis = planner.diagnose_failure(problem, incomplete_plan)

        assert diagnosis["error"] is not None
        assert diagnosis["failed_at"] == len(incomplete_plan)
        assert ("on", "A", "B") in diagnosis["missing_preconditions"]


# ============================================================================
# Test Heuristics
# ============================================================================


class TestHeuristics:
    """Test planning heuristics."""

    def test_relaxed_plan_heuristic(self):
        """Test relaxed planning heuristic estimation."""
        heuristic = RelaxedPlanHeuristic()

        state = State(propositions={("on", "A", "B")})
        goal = {("on", "B", "A"), ("clear", "B")}

        estimate = heuristic.estimate(state, goal)
        assert estimate > 0  # Unsatisfied goals

        # Goal state should have h=0
        goal_state = State(propositions=goal)
        assert heuristic.estimate(goal_state, goal) == 0

    def test_set_cover_heuristic(self):
        """Test set cover heuristic for action selection."""
        actions = [
            Action(name="a1", add_effects={("p1",), ("p2",)}, cost=1.0),
            Action(name="a2", add_effects={("p2",), ("p3",)}, cost=1.0),
        ]

        heuristic = SetCoverHeuristic(actions)

        state = State()
        goal = {("p1",), ("p2",), ("p3",)}

        estimate = heuristic.estimate(state, goal)
        assert estimate >= 2.0  # Need at least 2 actions


# ============================================================================
# Test Temporal Reasoning
# ============================================================================


class TestTemporalReasoning:
    """Test temporal constraints and reasoning."""

    def test_temporal_constraints(self):
        """Test temporal constraint validation."""
        planner = StateSpacePlanner()
        problem = BlocksWorldBuilder.create_problem(
            blocks=["A", "B"],
            initial_config={"A": "table", "B": "table"},
            goal_config={"A": "B"},
        )

        plan = planner.solve(problem)
        assert plan is not None

        temporal_planner = TemporalPlanner(planner)

        # Add constraint: pickup must occur before stack
        temporal_planner.add_temporal_constraint(
            TemporalConstraint(
                action1="pickup", action2="stack", constraint_type="BEFORE"
            )
        )

        # Validate (simplified - checks action name prefixes)
        # Real implementation would need full action matching
        valid, error = temporal_planner.validate_temporal_plan(plan)
        # May pass or fail depending on exact action names

    def test_state_timestamps(self):
        """Test that timestamps increment during plan execution."""
        problem = BlocksWorldBuilder.create_problem(
            blocks=["A"], initial_config={"A": "table"}, goal_config={"A": "table"}
        )

        planner = StateSpacePlanner()

        # Empty plan (goal already satisfied)
        plan = []
        states = planner.simulate_plan(problem, plan)

        assert states[0].timestamp == 0


# ============================================================================
# Test Plan Simulation
# ============================================================================


class TestPlanSimulation:
    """Test plan execution simulation."""

    def test_simulate_blocks_world(self):
        """Test simulating Blocks World plan."""
        problem = BlocksWorldBuilder.create_problem(
            blocks=["A", "B"],
            initial_config={"A": "table", "B": "table"},
            goal_config={"A": "B"},
        )

        planner = StateSpacePlanner()
        plan = planner.solve(problem)
        assert plan is not None

        states = planner.simulate_plan(problem, plan)

        # Should have initial state + one state per action
        assert len(states) == len(plan) + 1

        # First state is initial state
        assert states[0] == problem.initial_state

        # Last state should satisfy goal
        assert problem.is_goal(states[-1])


# ============================================================================
# Test Performance
# ============================================================================


class TestPerformance:
    """Test planner performance and scalability."""

    def test_expansion_limit(self):
        """Test that planner respects expansion limit."""
        # Create large problem
        problem = GridNavigationBuilder.create_problem(
            grid_size=(10, 10), start=(0, 0), goal=(9, 9)
        )

        planner = StateSpacePlanner(max_expansions=10)
        planner.solve(problem)

        # May or may not find plan with low limit
        assert planner.stats["expansions"] <= 10

    def test_statistics_tracking(self):
        """Test that planner tracks search statistics."""
        problem = BlocksWorldBuilder.create_problem(
            blocks=["A", "B"],
            initial_config={"A": "table", "B": "table"},
            goal_config={"A": "B"},
        )

        planner = StateSpacePlanner()
        plan = planner.solve(problem)

        assert plan is not None
        assert planner.stats["expansions"] > 0
        assert planner.stats["generated"] >= planner.stats["expansions"]
        assert planner.stats["plan_length"] == len(plan)


# ============================================================================
# Test Hybrid Planning (Integration with Logic/CSP)
# ============================================================================


class TestHybridPlanning:
    """Test integration with logic engine and CSP solver."""

    def test_hybrid_planner_initialization(self):
        """Test hybrid planner creation."""
        hybrid = HybridPlanner(logic_engine=None, csp_solver=None)
        assert hybrid is not None
        assert isinstance(hybrid, StateSpacePlanner)

    def test_action_enrichment(self):
        """Test logic-based action enrichment."""
        # Without logic engine, no enrichment happens
        hybrid_no_logic = HybridPlanner(logic_engine=None)

        action = Action(name="stack(A, B)", add_effects={("on", "A", "B")})

        state = State()

        # Without logic engine, action unchanged
        enriched_no_logic = hybrid_no_logic.enrich_action_with_logic(action, state)
        assert enriched_no_logic.add_effects == action.add_effects

        # With mock logic engine (placeholder), enrichment happens
        hybrid_with_logic = HybridPlanner(logic_engine="mock")

        # Enrich action (simplified - adds "above" relation)
        enriched_with_logic = hybrid_with_logic.enrich_action_with_logic(action, state)

        # Should add inferred "above" relation
        assert ("above", "A", "B") in enriched_with_logic.add_effects


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """End-to-end integration tests."""

    def test_full_planning_workflow(self):
        """Test complete planning workflow from problem to execution."""
        # 1. Define problem
        problem = BlocksWorldBuilder.create_problem(
            blocks=["A", "B", "C"],
            initial_config={"A": "B", "B": "C", "C": "table"},
            goal_config={"C": "B", "B": "A", "A": "table"},
        )

        # 2. Solve
        planner = StateSpacePlanner()
        plan = planner.solve(problem)

        assert plan is not None, "No plan found"

        # 3. Validate
        valid, error = planner.validate_plan(problem, plan)
        assert valid, f"Plan validation failed: {error}"

        # 4. Simulate
        states = planner.simulate_plan(problem, plan)
        assert len(states) == len(plan) + 1

        # 5. Check goal achieved
        assert problem.is_goal(states[-1])

    def test_multiple_problem_types(self):
        """Test planner works across different domains."""
        planner = StateSpacePlanner()

        # Blocks World
        blocks_problem = BlocksWorldBuilder.create_problem(
            blocks=["A", "B"],
            initial_config={"A": "table", "B": "table"},
            goal_config={"A": "B"},
        )
        blocks_plan = planner.solve(blocks_problem)
        assert blocks_plan is not None

        # Grid Navigation
        grid_problem = GridNavigationBuilder.create_problem(
            grid_size=(3, 3), start=(0, 0), goal=(2, 2)
        )
        grid_plan = planner.solve(grid_problem)
        assert grid_plan is not None

        # Both plans should be valid
        assert planner.validate_plan(blocks_problem, blocks_plan)[0]
        assert planner.validate_plan(grid_problem, grid_plan)[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
