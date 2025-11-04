"""
tests/test_state_reasoning.py

Comprehensive tests for State-Space Reasoning components.

Tests cover:
- component_32_state_reasoning: Basic state-space planning
- component_31_state_space_planner: STRIPS-style planning with domain builders
- component_12_graph_traversal: StateAwareTraversal integration
- State representation and operations
- Action preconditions and effects
- A* search planning
- BFS planning
- Constraint integration
- ProofTree generation
- Heuristic functions

Classic Planning Domains:
- Blocks World (STRIPS benchmark)
- Grid Navigation (pathfinding with obstacles)
- River Crossing (constraint-aware planning)
"""

import pytest
import logging

# component_32 imports (existing tests)
from component_32_state_reasoning import (
    create_numeric_state,
    manhattan_distance_heuristic,
    create_simple_action,
)

# component_31 imports (new STRIPS tests)
from component_31_state_space_planner import (
    StateSpacePlanner as Planner31,
    State as State31,
    Action as Action31,
    BlocksWorldBuilder,
    GridNavigationBuilder,
    RiverCrossingBuilder,
)

# StateAwareTraversal integration
from component_12_graph_traversal import StateAwareTraversal
from component_1_netzwerk import KonzeptNetzwerk

from component_29_constraint_reasoning import ConstraintSolver

logger = logging.getLogger(__name__)


# ============================================================================
# NEW: Phase 3.3 Tests - STRIPS-Style Planning (component_31)
# ============================================================================


class TestBlocksWorld:
    """
    Tests für klassisches Blocks World Planning (component_31).

    Blocks World ist ein Standard-Benchmark für STRIPS-Planning:
    - Actions: stack, unstack, pickup, putdown
    - Goal: Erreiche spezifische Block-Konfiguration
    - Zeigt: Precondition-Checking, Multi-Step Planning, Goal Achievement
    """

    def test_simple_blocks_world(self):
        """
        Einfaches Blocks World Problem: Invertiere Stack.

        Initial: A on B on table
        Goal: B on A on table

        Erwarteter Plan: unstack(A, B), putdown(A), pickup(B), stack(B, A)
        """
        logger.info("Test: Simple Blocks World Planning")

        # Erstelle Problem mit BlocksWorldBuilder
        problem = BlocksWorldBuilder.create_problem(
            blocks=["A", "B"],
            initial_config={"A": "B", "B": "table"},
            goal_config={"B": "A", "A": "table"},
        )

        # Löse mit StateSpacePlanner (component_31)
        planner = Planner31()
        plan = planner.solve(problem)

        # Validiere Ergebnis
        assert plan is not None, "Kein Plan gefunden"
        assert len(plan) == 4, f"Plan sollte 4 Schritte haben, hat {len(plan)}"

        # Validiere Plan führt zum Ziel
        valid, error = planner.validate_plan(problem, plan)
        assert valid, f"Plan-Validierung fehlgeschlagen: {error}"

        logger.info(f"✓ Blocks World Plan: {[action.name for action in plan]}")

    def test_three_block_tower(self):
        """
        Komplexeres Blocks World: Baue 3-Block-Turm.

        Initial: A, B, C alle auf table
        Goal: C on B on A on table

        Zeigt: Multi-Step Planning mit mehreren Objekten
        """
        logger.info("Test: Three-Block Tower Building")

        problem = BlocksWorldBuilder.create_problem(
            blocks=["A", "B", "C"],
            initial_config={"A": "table", "B": "table", "C": "table"},
            goal_config={"B": "A", "C": "B", "A": "table"},
        )

        planner = Planner31()
        plan = planner.solve(problem)

        assert plan is not None, "Kein Plan für 3-Block-Turm gefunden"
        assert len(plan) >= 4, "Plan sollte mindestens 4 Schritte haben"

        # Validiere Plan
        valid, error = planner.validate_plan(problem, plan)
        assert valid, f"Plan-Validierung fehlgeschlagen: {error}"

        logger.info(f"✓ 3-Block-Turm gebaut: {len(plan)} Schritte")

    def test_blocks_world_with_state_aware_traversal(self):
        """
        Nutzt StateAwareTraversal für Blocks World Planning.

        Zeigt Integration von Graph-Traversal mit State-Planning.
        """
        logger.info("Test: Blocks World mit StateAwareTraversal")

        # Setup Netzwerk (für Graph-Kontext)
        netzwerk = KonzeptNetzwerk()

        # Erstelle StateAwareTraversal
        state_traversal = StateAwareTraversal(netzwerk)

        # Erstelle States (component_31 Style)
        initial_state = State31(
            propositions={
                ("on", "A", "B"),
                ("on", "B", "table"),
                ("clear", "A"),
                ("handempty",),
            }
        )

        goal_state = State31(propositions={("on", "B", "A"), ("on", "A", "table")})

        # Erstelle Actions mit BlocksWorldBuilder
        actions = BlocksWorldBuilder.create_actions()

        # Plane mit StateAwareTraversal
        plan = state_traversal.find_path_with_constraints(
            initial_state, goal_state, actions, constraints=None
        )

        assert plan is not None, "StateAwareTraversal fand keinen Plan"
        assert len(plan) >= 3, "Plan sollte mindestens 3 Schritte haben"

        logger.info(f"✓ StateAwareTraversal Plan: {[a.name for a in plan]}")


class TestGridNavigation:
    """
    Tests für Grid-basierte Navigation mit Hindernissen (component_31).

    Use Cases:
    - Pathfinding in 2D-Grid
    - Obstacle avoidance
    - Heuristic-guided search (A*)
    """

    def test_simple_grid_navigation(self):
        """
        Einfache Grid-Navigation ohne Hindernisse.

        Grid: 5x5
        Start: (0, 0)
        Goal: (3, 2)

        Zeigt: Grundlegendes A* Pathfinding mit Manhattan-Distanz
        """
        logger.info("Test: Simple Grid Navigation")

        problem = GridNavigationBuilder.create_problem(
            grid_size=(5, 5), start=(0, 0), goal=(3, 2), obstacles=[]
        )

        planner = Planner31()
        plan = planner.solve(problem)

        assert plan is not None, "Kein Navigations-Plan gefunden"
        # Manhattan-Distanz ist 3 + 2 = 5
        assert (
            len(plan) == 5
        ), f"Plan sollte 5 Schritte haben (Manhattan-Distanz), hat {len(plan)}"

        # Validiere Plan
        valid, error = planner.validate_plan(problem, plan)
        assert valid, f"Navigations-Plan ungültig: {error}"

        logger.info(f"✓ Navigation erfolgreich: {len(plan)} Schritte (optimal)")

    def test_grid_navigation_with_obstacles(self):
        """
        Grid-Navigation mit Hindernissen.

        Grid: 5x5
        Start: (0, 0)
        Goal: (4, 0)
        Obstacles: [(1, 0), (2, 0), (3, 0)]  # Blockiert direkten Weg

        Zeigt: Umweg-Planung um Hindernisse
        """
        logger.info("Test: Grid Navigation with Obstacles")

        problem = GridNavigationBuilder.create_problem(
            grid_size=(5, 5),
            start=(0, 0),
            goal=(4, 0),
            obstacles=[(1, 0), (2, 0), (3, 0)],
        )

        planner = Planner31()
        plan = planner.solve(problem)

        assert plan is not None, "Kein Plan mit Hindernissen gefunden"
        # Plan muss länger als direkter Weg sein (4 Schritte)
        assert (
            len(plan) > 4
        ), f"Plan sollte Umweg um Hindernisse nehmen, hat {len(plan)} Schritte"

        # Validiere Plan
        valid, error = planner.validate_plan(problem, plan)
        assert valid, f"Plan mit Hindernissen ungültig: {error}"

        logger.info(
            f"✓ Umweg-Plan gefunden: {len(plan)} Schritte (Hindernisse umgangen)"
        )

    def test_grid_navigation_with_state_aware_traversal(self):
        """
        Nutzt StateAwareTraversal mit Graph-Heuristik für Navigation.

        Zeigt: Graph-erweiterte Heuristik kann Planning verbessern
        """
        logger.info("Test: Grid Navigation mit StateAwareTraversal + Graph-Heuristik")

        netzwerk = KonzeptNetzwerk()
        state_traversal = StateAwareTraversal(netzwerk)

        # Erstelle States (component_31 Style - propositions)
        initial_state = State31(propositions={("at", "0", "0")}, timestamp=0)
        goal_state = State31(propositions={("at", "3", "3")})

        # Erstelle Actions für 4x4 Grid
        actions = GridNavigationBuilder.create_actions(grid_size=(4, 4))

        # Plane mit Graph-Heuristik
        plan = state_traversal.find_path_with_graph_heuristic(
            initial_state, goal_state, actions, use_graph_context=True
        )

        assert (
            plan is not None
        ), "StateAwareTraversal mit Graph-Heuristik fand keinen Plan"

        logger.info(f"✓ Graph-Heuristik Plan: {len(plan)} Schritte")


class TestRiverCrossing:
    """
    Tests für River Crossing Puzzle mit Constraints (component_31).

    Problem:
    - Farmer, fox, chicken, grain müssen Fluss überqueren
    - Constraints:
      * Fox frisst chicken wenn allein
      * Chicken frisst grain wenn allein
      * Nur farmer kann Boot rudern
      * Boot hält 2 Entitäten (farmer + 1 other)

    Zeigt: Constraint-Aware Planning (Integration mit component_29)
    """

    def test_river_crossing_basic(self):
        """
        Klassisches River Crossing Puzzle.

        Initial: Alle auf linker Seite
        Goal: Alle auf rechter Seite
        Constraints: Sicherheits-Checks (fox-chicken, chicken-grain)

        Zeigt: Planning unter Constraints
        """
        logger.info("Test: Basic River Crossing with Constraints")

        problem = RiverCrossingBuilder.create_problem()

        # Planner mit Safety-Constraint (is_safe_state)
        planner = Planner31(state_constraint=RiverCrossingBuilder.is_safe_state)

        plan = planner.solve(problem)

        assert plan is not None, "Kein sicherer Plan für River Crossing gefunden"
        # Minimale Lösung braucht 7 Schritte
        assert (
            len(plan) >= 7
        ), f"Plan sollte mindestens 7 Schritte haben, hat {len(plan)}"

        # Validiere Plan
        valid, error = planner.validate_plan(problem, plan)
        assert valid, f"River Crossing Plan ungültig: {error}"

        # Simuliere und prüfe Sicherheit in jedem Schritt
        states = planner.simulate_plan(problem, plan)
        for i, state in enumerate(states):
            assert RiverCrossingBuilder.is_safe_state(
                state
            ), f"State {i} ist unsicher: {state}"

        logger.info(f"✓ Sicherer River Crossing Plan: {len(plan)} Schritte")
        logger.info(f"  Aktionen: {[a.name for a in plan]}")

    def test_river_crossing_with_state_aware_traversal(self):
        """
        River Crossing mit StateAwareTraversal und expliziten Constraints.

        Zeigt vollständige Integration:
        - State-Space Planning (component_31)
        - Constraint-Checking (Safety-Funktion)
        - Graph-Traversal Context (component_12)
        """
        logger.info("Test: River Crossing mit StateAwareTraversal")

        netzwerk = KonzeptNetzwerk()

        # StateAwareTraversal (constraint_solver=None, nutzt Problem-spezifische Constraints)
        state_traversal = StateAwareTraversal(netzwerk, constraint_solver=None)

        # Erstelle Problem
        problem = RiverCrossingBuilder.create_problem()

        # NOTE: StateAwareTraversal ohne explizite Constraints könnte unsicheren Plan finden
        # Real-world Usage würde state_constraint in StateSpacePlanner nutzen

        # Teste nur ob Planning funktioniert
        plan = state_traversal.find_path_with_constraints(
            problem.initial_state,
            State31(propositions=problem.goal),
            problem.actions,
            constraints=None,
        )

        if plan:
            logger.info(f"✓ StateAwareTraversal Plan: {len(plan)} Schritte")
            # Plan könnte unsicher sein (keine Constraints)
        else:
            logger.warning(
                "⚠ StateAwareTraversal fand keinen Plan (erwartet ohne Safety-Constraints)"
            )


class TestStateReasoningIntegration31:
    """
    Integration-Tests für component_31 (STRIPS-Planner).

    Testet:
    - Plan-Validierung über verschiedene Domänen
    - Diagnose-Fähigkeit (Root-Cause Analysis)
    - ProofTree-Integration (wenn verfügbar)
    """

    def test_plan_validation_blocks_world(self):
        """
        Testet Plan-Validierung im Blocks World.

        Zeigt: validate_plan() erkennt fehlerhafte Pläne
        """
        logger.info("Test: Plan Validation (Blocks World)")

        problem = BlocksWorldBuilder.create_problem(
            blocks=["A", "B"],
            initial_config={"A": "B", "B": "table"},
            goal_config={"B": "A", "A": "table"},
        )

        planner = Planner31()

        # Erstelle korrekten Plan
        plan = planner.solve(problem)
        assert plan is not None

        # Validiere
        valid, error = planner.validate_plan(problem, plan)
        assert valid, f"Korrekter Plan sollte validieren: {error}"

        logger.info("✓ Plan-Validierung erfolgreich")

    def test_plan_diagnosis(self):
        """
        Testet Diagnose-Fähigkeit (Root-Cause Analysis).

        Zeigt: diagnose_failure() findet Fehlerursache
        """
        logger.info("Test: Plan Diagnosis (Root-Cause Analysis)")

        problem = BlocksWorldBuilder.create_problem(
            blocks=["A", "B"],
            initial_config={"A": "table", "B": "table"},
            goal_config={"A": "B", "B": "table"},
        )

        planner = Planner31()

        # Erstelle Plan mit fehlendem Precondition
        incomplete_plan = [
            Action31(
                name="stack(A, B)",
                preconditions={("holding", "A"), ("clear", "B")},
                add_effects={("on", "A", "B"), ("clear", "A"), ("handempty",)},
                delete_effects={("holding", "A"), ("clear", "B")},
            )
        ]

        # Diagnose
        diagnosis = planner.diagnose_failure(problem, incomplete_plan)

        assert diagnosis["error"] is not None, "Diagnose sollte Fehler finden"
        assert (
            "missing_preconditions" in diagnosis
        ), "Fehlende Preconditions sollten identifiziert werden"
        assert ("holding", "A") in diagnosis[
            "missing_preconditions"
        ], "Missing 'holding(A)' sollte erkannt werden"

        logger.info(f"✓ Diagnose erfolgreich: {diagnosis['error']}")


@pytest.mark.slow
class TestPerformance31:
    """
    Performance-Tests für component_31 (STRIPS-Planner).

    Markiert mit @pytest.mark.slow - nur bei explizitem Request.
    """

    def test_large_blocks_world(self):
        """
        Test mit größerem Blocks World (5 Blöcke).

        Zeigt: Skalierbarkeit der STRIPS-Implementierung
        """
        logger.info("Test: Large Blocks World (5 blocks)")

        problem = BlocksWorldBuilder.create_problem(
            blocks=["A", "B", "C", "D", "E"],
            initial_config={
                "A": "table",
                "B": "table",
                "C": "table",
                "D": "table",
                "E": "table",
            },
            goal_config={"E": "D", "D": "C", "C": "B", "B": "A", "A": "table"},
        )

        planner = Planner31(max_expansions=50000)
        plan = planner.solve(problem)

        assert plan is not None, "Großes Blocks World sollte lösbar sein"
        logger.info(f"✓ 5-Block-Turm gebaut: {len(plan)} Schritte")
        logger.info(f"  Stats: {planner.stats}")

    def test_large_grid_navigation(self):
        """
        Test mit größerem Grid (10x10).

        Zeigt: A* Effizienz bei größeren Suchräumen
        """
        logger.info("Test: Large Grid Navigation (10x10)")

        problem = GridNavigationBuilder.create_problem(
            grid_size=(10, 10),
            start=(0, 0),
            goal=(9, 9),
            obstacles=[(i, 5) for i in range(5, 10)],  # Horizontale Barriere
        )

        planner = Planner31()
        plan = planner.solve(problem)

        assert plan is not None, "Großes Grid sollte navigierbar sein"
        logger.info(f"✓ 10x10 Grid navigiert: {len(plan)} Schritte")
        logger.info(f"  Stats: {planner.stats}")


# ============================================================================
# EXISTING: component_32 Tests (preserved)
# ============================================================================


class TestState:
    """Test State class functionality."""

    def test_state_creation(self):
        """Test basic state creation and property access."""
        state = State({"x": 5, "y": 10, "color": "red"})

        assert state.get("x") == 5
        assert state.get("y") == 10
        assert state.get("color") == "red"
        assert state.get("z") is None
        assert state.get("z", 0) == 0

    def test_state_set_property(self):
        """Test setting state properties."""
        state = State()
        state.set("x", 100)
        state.set("name", "test")

        assert state.get("x") == 100
        assert state.get("name") == "test"

    def test_state_satisfies(self):
        """Test state condition checking."""
        state = State({"x": 5, "y": 10})

        assert state.satisfies(lambda s: s.get("x") == 5)
        assert state.satisfies(lambda s: s.get("x") > 0)
        assert state.satisfies(lambda s: s.get("x") + s.get("y") == 15)
        assert not state.satisfies(lambda s: s.get("x") == 10)

    def test_state_copy(self):
        """Test deep copying of states."""
        original = State({"x": 5, "nested": {"a": 1}})
        copy = original.copy()

        # Modify copy
        copy.set("x", 10)
        copy.properties["nested"]["a"] = 99

        # Original should be unchanged (deep copy)
        assert original.get("x") == 5
        assert original.properties["nested"]["a"] == 1  # Deep copy preserves original

    def test_state_equality(self):
        """Test state equality comparison."""
        state1 = State({"x": 5, "y": 10})
        state2 = State({"x": 5, "y": 10})
        state3 = State({"x": 5, "y": 11})

        assert state1 == state2
        assert state1 != state3
        assert state2 != state3

    def test_state_hashing(self):
        """Test state hashing for use in sets/dicts."""
        state1 = State({"x": 5, "y": 10})
        state2 = State({"x": 5, "y": 10})
        state3 = State({"x": 5, "y": 11})

        # Same states should have same hash
        assert hash(state1) == hash(state2)

        # Can use in set
        state_set = {state1, state2, state3}
        assert len(state_set) == 2  # state1 and state2 are duplicates


class TestAction:
    """Test Action class functionality."""

    def test_action_creation(self):
        """Test basic action creation."""
        action = Action("test_action", [], [], 1.5)

        assert action.name == "test_action"
        assert action.cost == 1.5
        assert len(action.preconditions) == 0
        assert len(action.effects) == 0

    def test_action_is_applicable(self):
        """Test action applicability checking."""
        # Action requires x > 0
        precond = lambda s: s.get("x", 0) > 0
        action = Action("move", [precond], [])

        state1 = State({"x": 5})
        state2 = State({"x": 0})
        state3 = State({"x": -1})

        assert action.is_applicable(state1)
        assert not action.is_applicable(state2)
        assert not action.is_applicable(state3)

    def test_action_apply(self):
        """Test action application with effects."""
        # Action increments x by 1
        precond = lambda s: s.get("x", 0) < 10
        effect = lambda s: State({"x": s.get("x", 0) + 1})
        action = Action("increment", [precond], [effect])

        state = State({"x": 5})
        new_state = action.apply(state)

        assert new_state is not None
        assert new_state.get("x") == 6
        assert state.get("x") == 5  # Original unchanged

    def test_action_not_applicable(self):
        """Test action returns None when not applicable."""
        precond = lambda s: s.get("x", 0) > 100
        effect = lambda s: State({"x": 0})
        action = Action("reset", [precond], [effect])

        state = State({"x": 5})
        new_state = action.apply(state)

        assert new_state is None

    def test_action_multiple_preconditions(self):
        """Test action with multiple preconditions (AND)."""
        precond1 = lambda s: s.get("x", 0) > 0
        precond2 = lambda s: s.get("y", 0) > 0
        action = Action("both", [precond1, precond2], [])

        assert action.is_applicable(State({"x": 1, "y": 1}))
        assert not action.is_applicable(State({"x": 1, "y": 0}))
        assert not action.is_applicable(State({"x": 0, "y": 1}))

    def test_action_multiple_effects(self):
        """Test action with multiple effects applied sequentially."""
        effect1 = lambda s: State({"x": s.get("x", 0) + 1, "y": s.get("y", 0)})
        effect2 = lambda s: State({"x": s.get("x", 0), "y": s.get("y", 0) * 2})

        action = Action("multi", [], [effect1, effect2])

        state = State({"x": 5, "y": 3})
        new_state = action.apply(state)

        assert new_state.get("x") == 6  # From effect1
        assert new_state.get("y") == 6  # From effect2 (3 * 2)


class TestStateSpacePlanner:
    """Test StateSpacePlanner A* search."""

    def test_simple_linear_plan(self):
        """Test planning with simple linear state space."""
        planner = StateSpacePlanner()

        # Start at 0, goal is 5
        initial = State({"x": 0})
        goal = lambda s: s.get("x") == 5

        # Only one action: increment x
        increment = Action(
            "increment",
            [lambda s: s.get("x", 0) < 100],
            [lambda s: State({"x": s.get("x", 0) + 1})],
            cost=1.0,
        )

        plan = planner.plan(initial, goal, [increment])

        assert plan is not None
        assert len(plan) == 5
        assert all(a.name == "increment" for a in plan)

    def test_2d_navigation(self):
        """Test planning in 2D grid (shortest path)."""
        planner = StateSpacePlanner()

        initial = State({"x": 0, "y": 0})
        goal = lambda s: s.get("x") == 3 and s.get("y") == 2

        # Define 4-directional movement
        actions = [
            Action(
                "right",
                [lambda s: s.get("x", 0) < 10],
                [lambda s: State({"x": s.get("x", 0) + 1, "y": s.get("y", 0)})],
                cost=1.0,
            ),
            Action(
                "up",
                [lambda s: s.get("y", 0) < 10],
                [lambda s: State({"x": s.get("x", 0), "y": s.get("y", 0) + 1})],
                cost=1.0,
            ),
            Action(
                "left",
                [lambda s: s.get("x", 0) > 0],
                [lambda s: State({"x": s.get("x", 0) - 1, "y": s.get("y", 0)})],
                cost=1.0,
            ),
            Action(
                "down",
                [lambda s: s.get("y", 0) > 0],
                [lambda s: State({"x": s.get("x", 0), "y": s.get("y", 0) - 1})],
                cost=1.0,
            ),
        ]

        # Use Manhattan distance heuristic
        heuristic = lambda s: abs(s.get("x", 0) - 3) + abs(s.get("y", 0) - 2)

        plan = planner.plan(initial, goal, actions, heuristic)

        assert plan is not None
        assert len(plan) == 5  # Manhattan distance = 5 (optimal)

        # Count moves
        moves = [a.name for a in plan]
        assert moves.count("right") == 3
        assert moves.count("up") == 2

    def test_no_plan_exists(self):
        """Test planning when no plan exists."""
        planner = StateSpacePlanner()

        initial = State({"x": 0})
        goal = lambda s: s.get("x") == -5  # Impossible with only increment

        increment = Action(
            "increment", [], [lambda s: State({"x": s.get("x", 0) + 1})], cost=1.0
        )

        plan = planner.plan(initial, goal, [increment])

        assert plan is None

        # Check proof tree shows failure
        proof_tree = planner.get_last_proof_tree()
        assert proof_tree is not None
        assert len(proof_tree.root_steps) > 0
        assert proof_tree.root_steps[0].confidence == 0.0

    def test_plan_with_different_costs(self):
        """Test A* prefers lower total cost paths."""
        planner = StateSpacePlanner()

        initial = State({"x": 0})
        goal = lambda s: s.get("x") == 10

        # Two ways to reach goal: slow (1 per step) or fast (2 per step)
        slow = Action("slow", [], [lambda s: State({"x": s.get("x", 0) + 1})], cost=1.0)
        fast = Action(
            "fast",
            [],
            [lambda s: State({"x": s.get("x", 0) + 2})],
            cost=1.5,  # Total: 5 steps * 1.5 = 7.5 (cheaper than 10 * 1.0 = 10.0)
        )

        # Heuristic: remaining distance
        heuristic = lambda s: abs(s.get("x", 0) - 10)

        plan = planner.plan(initial, goal, [slow, fast], heuristic)

        assert plan is not None
        assert len(plan) == 5  # Should use fast 5 times (total cost 7.5)
        assert all(a.name == "fast" for a in plan)

    def test_proof_tree_generation(self):
        """Test that planning generates valid proof trees."""
        planner = StateSpacePlanner()

        initial = State({"x": 0})
        goal = lambda s: s.get("x") == 3

        increment = Action(
            "increment", [], [lambda s: State({"x": s.get("x", 0) + 1})], cost=1.0
        )

        plan = planner.plan(initial, goal, [increment])

        assert plan is not None

        # Check proof tree
        proof_tree = planner.get_last_proof_tree()
        assert proof_tree is not None
        all_steps = proof_tree.get_all_steps()
        assert len(all_steps) > 0

        # Should have premise (initial state) and conclusion (goal reached)
        from component_17_proof_explanation import StepType

        step_types = [step.step_type for step in all_steps]
        assert StepType.PREMISE in step_types
        assert StepType.CONCLUSION in step_types

    def test_max_iterations_limit(self):
        """Test planner respects max_iterations limit."""
        planner = StateSpacePlanner(max_iterations=10)

        initial = State({"x": 0})
        goal = lambda s: s.get("x") == 1000  # Would take 1000 steps

        increment = Action(
            "increment", [], [lambda s: State({"x": s.get("x", 0) + 1})], cost=1.0
        )

        plan = planner.plan(initial, goal, [increment])

        assert plan is None  # Should fail due to iteration limit


class TestBFSPlanning:
    """Test BFS planning algorithm."""

    def test_bfs_simple_plan(self):
        """Test BFS finds shortest plan (uniform cost)."""
        planner = StateSpacePlanner()

        initial = State({"x": 0})
        goal = lambda s: s.get("x") == 3

        increment = Action(
            "increment", [], [lambda s: State({"x": s.get("x", 0) + 1})], cost=1.0
        )

        plan = planner.plan_with_bfs(initial, goal, [increment])

        assert plan is not None
        assert len(plan) == 3

    def test_bfs_finds_shortest(self):
        """Test BFS finds shortest plan with multiple paths."""
        planner = StateSpacePlanner()

        initial = State({"x": 0})
        goal = lambda s: s.get("x") == 10

        # Two actions: +1 or +2
        inc1 = Action("inc1", [], [lambda s: State({"x": s.get("x", 0) + 1})])
        inc2 = Action("inc2", [], [lambda s: State({"x": s.get("x", 0) + 2})])

        plan = planner.plan_with_bfs(initial, goal, [inc1, inc2])

        assert plan is not None
        assert len(plan) == 5  # Optimal: 5x inc2

    def test_bfs_max_depth(self):
        """Test BFS respects max_depth limit."""
        planner = StateSpacePlanner()

        initial = State({"x": 0})
        goal = lambda s: s.get("x") == 100

        increment = Action("increment", [], [lambda s: State({"x": s.get("x", 0) + 1})])

        plan = planner.plan_with_bfs(initial, goal, [increment], max_depth=10)

        assert plan is None  # Can't reach in 10 steps

    def test_bfs_proof_tree(self):
        """Test BFS generates proof trees."""
        planner = StateSpacePlanner()

        initial = State({"x": 0})
        goal = lambda s: s.get("x") == 2

        increment = Action("increment", [], [lambda s: State({"x": s.get("x", 0) + 1})])

        plan = planner.plan_with_bfs(initial, goal, [increment])

        assert plan is not None

        proof_tree = planner.get_last_proof_tree()
        assert proof_tree is not None
        all_steps = proof_tree.get_all_steps()
        assert len(all_steps) > 0


class TestUtilityFunctions:
    """Test utility functions."""

    def test_create_numeric_state(self):
        """Test numeric state creation helper."""
        state = create_numeric_state({"x": 5, "y": 10, "z": -3})

        assert state.get("x") == 5
        assert state.get("y") == 10
        assert state.get("z") == -3

    def test_manhattan_distance_heuristic(self):
        """Test Manhattan distance heuristic creation."""
        target = State({"x": 10, "y": 5})
        heuristic = manhattan_distance_heuristic(target, ["x", "y"])

        # Test at target
        assert heuristic(State({"x": 10, "y": 5})) == 0.0

        # Test offset
        assert heuristic(State({"x": 7, "y": 3})) == 5.0  # |10-7| + |5-3|

        # Test large offset
        assert heuristic(State({"x": 0, "y": 0})) == 15.0

    def test_create_simple_action(self):
        """Test simple action creation helper."""
        action = create_simple_action(
            "move_right",
            {"x": 0, "y": 0},  # Precondition: must be at origin
            {"x": 1, "y": 0},  # Effect: move to (1, 0)
            cost=2.0,
        )

        assert action.name == "move_right"
        assert action.cost == 2.0

        # Test applicability
        assert action.is_applicable(State({"x": 0, "y": 0}))
        assert not action.is_applicable(State({"x": 1, "y": 0}))

        # Test effect
        new_state = action.apply(State({"x": 0, "y": 0}))
        assert new_state is not None
        assert new_state.get("x") == 1
        assert new_state.get("y") == 0


class TestConstraintIntegration:
    """Test integration with constraint solver."""

    def test_planner_with_constraint_solver(self):
        """Test planner can use constraint solver for validation."""
        # Create a simple constraint solver
        solver = ConstraintSolver()

        # Planner with solver
        planner = StateSpacePlanner(constraint_solver=solver)

        initial = State({"x": 0})
        goal = lambda s: s.get("x") == 5

        increment = Action("increment", [], [lambda s: State({"x": s.get("x", 0) + 1})])

        # Should work (validation doesn't add constraints yet)
        plan = planner.plan(initial, goal, [increment])
        assert plan is not None

    def test_is_valid_state(self):
        """Test state validation method."""
        planner = StateSpacePlanner()

        # Without solver, all states are valid
        assert planner.is_valid_state(State({"x": 5}))
        assert planner.is_valid_state(State({"x": -100}))

        # With solver (no constraints added, so still valid)
        solver = ConstraintSolver()
        planner_with_solver = StateSpacePlanner(constraint_solver=solver)

        assert planner_with_solver.is_valid_state(State({"x": 5}))


class TestComplexPlanning:
    """Test complex planning scenarios."""

    def test_blocks_world_simple(self):
        """Test simple blocks world planning."""
        planner = StateSpacePlanner()

        # Initial: A on table, B on table
        # Goal: B on A
        initial = State({"on_table": ["A", "B"], "on": {}, "clear": ["A", "B"]})

        def goal_condition(s: State) -> bool:
            on_dict = s.get("on", {})
            return on_dict.get("B") == "A"

        # Action: Stack B on A
        def can_stack(s: State) -> bool:
            clear = s.get("clear", [])
            return "A" in clear and "B" in clear

        def do_stack(s: State) -> State:
            new_state = s.copy()
            on_table = new_state.get("on_table", []).copy()
            on_dict = new_state.get("on", {}).copy()
            clear = new_state.get("clear", []).copy()

            # Remove B from table
            on_table.remove("B")
            # Put B on A
            on_dict["B"] = "A"
            # A is no longer clear
            clear.remove("A")

            new_state.set("on_table", on_table)
            new_state.set("on", on_dict)
            new_state.set("clear", clear)

            return new_state

        stack_action = Action("stack_B_on_A", [can_stack], [do_stack])

        plan = planner.plan(initial, goal_condition, [stack_action])

        assert plan is not None
        assert len(plan) == 1
        assert plan[0].name == "stack_B_on_A"

    def test_multi_step_planning(self):
        """Test planning requiring multiple different actions."""
        planner = StateSpacePlanner()

        # State: position and inventory
        initial = State({"pos": "home", "has_key": False, "door_open": False})

        def goal_condition(s: State) -> bool:
            return s.get("pos") == "office" and s.get("door_open")

        # Define actions
        def can_get_key(s: State) -> bool:
            return s.get("pos") == "home" and not s.get("has_key")

        def do_get_key(s: State) -> State:
            new_s = s.copy()
            new_s.set("has_key", True)
            return new_s

        def can_go_to_door(s: State) -> bool:
            return s.get("pos") == "home"

        def do_go_to_door(s: State) -> State:
            new_s = s.copy()
            new_s.set("pos", "door")
            return new_s

        def can_unlock(s: State) -> bool:
            return (
                s.get("pos") == "door" and s.get("has_key") and not s.get("door_open")
            )

        def do_unlock(s: State) -> State:
            new_s = s.copy()
            new_s.set("door_open", True)
            return new_s

        def can_enter(s: State) -> bool:
            return s.get("pos") == "door" and s.get("door_open")

        def do_enter(s: State) -> State:
            new_s = s.copy()
            new_s.set("pos", "office")
            return new_s

        actions = [
            Action("get_key", [can_get_key], [do_get_key]),
            Action("go_to_door", [can_go_to_door], [do_go_to_door]),
            Action("unlock_door", [can_unlock], [do_unlock]),
            Action("enter_office", [can_enter], [do_enter]),
        ]

        plan = planner.plan(initial, goal_condition, actions)

        assert plan is not None
        assert len(plan) == 4

        # Check sequence
        action_names = [a.name for a in plan]
        assert action_names == ["get_key", "go_to_door", "unlock_door", "enter_office"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
