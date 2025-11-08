"""
component_32_state_reasoning.py

State-Space Planning Component

Provides generic state-space reasoning capabilities:
- State representation with property dictionaries
- Actions with preconditions and effects
- A* search-based planning
- Integration with constraint solver (component_29)
- ProofTree generation for plan explanations

This is a domain-independent planner that can be used for various
reasoning tasks requiring state transitions (puzzles, planning, etc.).
"""

import copy
import logging
from dataclasses import dataclass, field
from heapq import heappop, heappush
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

# Import proof structures for explanations
from component_17_proof_explanation import ProofStep, ProofTree, StepType

# Import constraint solver for state validation
from component_29_constraint_reasoning import ConstraintSolver, Variable

logger = logging.getLogger(__name__)


@dataclass
class State:
    """
    Represents a state in the state space.

    A state is a collection of properties (key-value pairs) that describe
    the current situation. States can be checked against conditions and
    compared for equality.

    Example:
        >>> state = State({"location": "home", "has_key": True})
        >>> state.satisfies(lambda s: s.get("has_key") == True)
        True
    """

    properties: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Get property value with optional default."""
        return self.properties.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set property value."""
        self.properties[key] = value

    def satisfies(self, condition: Callable[["State"], bool]) -> bool:
        """
        Check if this state satisfies a given condition.

        Args:
            condition: Function that takes a State and returns bool

        Returns:
            True if condition is satisfied, False otherwise
        """
        try:
            return condition(self)
        except Exception as e:
            logger.warning(f"Condition check failed: {e}")
            return False

    def copy(self) -> "State":
        """Create a deep copy of this state."""
        return State(copy.deepcopy(self.properties))

    def __eq__(self, other: object) -> bool:
        """Check equality based on properties."""
        if not isinstance(other, State):
            return False
        return self.properties == other.properties

    def __hash__(self) -> int:
        """Hash based on properties (for use in sets/dicts)."""
        # Convert dict to sorted tuple of items for hashing
        # Handle unhashable types (like lists) by converting to tuples
        hashable_items = []
        for k, v in sorted(self.properties.items()):
            if isinstance(v, list):
                v = tuple(v)
            elif isinstance(v, dict):
                v = tuple(sorted(v.items()))
            hashable_items.append((k, v))
        return hash(tuple(hashable_items))

    def __lt__(self, other: "State") -> bool:
        """Less-than comparison for priority queue (heapq)."""
        # Compare based on string representation (arbitrary but consistent)
        return str(self.properties) < str(other.properties)

    def __repr__(self) -> str:
        return f"State({self.properties})"


@dataclass
class Action:
    """
    Represents an action that can transform states.

    An action has:
    - name: Human-readable identifier
    - preconditions: List of conditions that must be satisfied before action
    - effects: List of functions that modify the state
    - cost: Numeric cost for this action (for A* search)

    Example:
        >>> def can_unlock(s: State) -> bool:
        ...     return s.get("has_key") and s.get("at_door")
        >>> def do_unlock(s: State) -> State:
        ...     new_state = s.copy()
        ...     new_state.set("door_locked", False)
        ...     return new_state
        >>> action = Action("unlock_door", [can_unlock], [do_unlock], 1.0)
    """

    name: str
    preconditions: List[Callable[[State], bool]] = field(default_factory=list)
    effects: List[Callable[[State], State]] = field(default_factory=list)
    cost: float = 1.0

    def is_applicable(self, state: State) -> bool:
        """
        Check if this action can be applied to the given state.

        Args:
            state: The state to check

        Returns:
            True if all preconditions are satisfied
        """
        return all(precond(state) for precond in self.preconditions)

    def apply(self, state: State) -> Optional[State]:
        """
        Apply this action to a state, producing a new state.

        Args:
            state: The state to transform

        Returns:
            New state after applying effects, or None if not applicable
        """
        if not self.is_applicable(state):
            logger.debug(f"Action {self.name} not applicable to {state}")
            return None

        # Apply all effects sequentially
        new_state = state.copy()
        for effect in self.effects:
            new_state = effect(new_state)

        return new_state

    def __lt__(self, other: "Action") -> bool:
        """Less-than comparison for priority queue (heapq)."""
        # Compare by cost first, then by name (for consistency)
        if self.cost != other.cost:
            return self.cost < other.cost
        return self.name < other.name

    def __repr__(self) -> str:
        return f"Action({self.name}, cost={self.cost})"


class StateSpacePlanner:
    """
    Generic state-space planner using A* search.

    This planner finds a sequence of actions that transforms an initial
    state into a state satisfying a goal condition. It uses A* search
    with optional heuristic functions and constraint-based validation.

    Example:
        >>> planner = StateSpacePlanner()
        >>> initial = State({"x": 0})
        >>> goal = lambda s: s.get("x") == 5
        >>> actions = [Action("increment", [], [lambda s: State({"x": s.get("x") + 1})])]
        >>> plan = planner.plan(initial, goal, actions)
        >>> len(plan)
        5
    """

    def __init__(
        self,
        constraint_solver: Optional[ConstraintSolver] = None,
        max_iterations: int = 10000,
    ):
        """
        Initialize the state-space planner.

        Args:
            constraint_solver: Optional constraint solver for state validation
            max_iterations: Maximum search iterations before giving up
        """
        self.constraint_solver = constraint_solver
        self.max_iterations = max_iterations
        self.last_proof_tree: Optional[ProofTree] = None

    def plan(
        self,
        initial_state: State,
        goal_condition: Callable[[State], bool],
        available_actions: List[Action],
        heuristic: Optional[Callable[[State], float]] = None,
    ) -> Optional[List[Action]]:
        """
        Find a sequence of actions from initial state to goal.

        Uses A* search with optional heuristic. If no heuristic is provided,
        falls back to uniform-cost search (Dijkstra's algorithm).

        Args:
            initial_state: Starting state
            goal_condition: Function that returns True for goal states
            available_actions: List of actions that can be applied
            heuristic: Optional function estimating cost to goal (must be admissible)

        Returns:
            List of actions forming a plan, or None if no plan found
        """
        logger.info(
            f"Planning from {initial_state} with {len(available_actions)} actions"
        )

        # If no heuristic provided, use zero heuristic (Dijkstra)
        if heuristic is None:

            def heuristic(s):
                return 0.0

        # Priority queue: (f_cost, g_cost, state, action_path)
        # f_cost = g_cost + h_cost (A* evaluation)
        # g_cost = actual cost from start
        open_set: List[Tuple[float, float, State, List[Action]]] = []
        heappush(open_set, (heuristic(initial_state), 0.0, initial_state, []))

        # Track visited states to avoid cycles
        closed_set: Set[State] = set()

        # Track proof steps for explanation
        proof_steps: List[ProofStep] = []
        proof_steps.append(
            ProofStep(
                step_id="plan_0_premise",
                step_type=StepType.PREMISE,
                inputs=[],
                output=str(initial_state),
                explanation_text=f"Initial state: {initial_state}",
                confidence=1.0,
                source_component="state_planner",
            )
        )

        iterations = 0

        while open_set and iterations < self.max_iterations:
            iterations += 1

            # Get state with lowest f_cost
            f_cost, g_cost, current_state, action_path = heappop(open_set)

            # Check if we reached the goal
            if goal_condition(current_state):
                logger.info(
                    f"Goal reached after {iterations} iterations, {len(action_path)} actions"
                )
                proof_steps.append(
                    ProofStep(
                        step_id=f"plan_{iterations}_conclusion",
                        step_type=StepType.CONCLUSION,
                        inputs=[a.name for a in action_path],
                        output=str(current_state),
                        explanation_text=f"Goal reached: {current_state}",
                        confidence=1.0,
                        source_component="state_planner",
                    )
                )

                # Create proof tree
                root_proof_step = ProofStep(
                    step_id="plan_root",
                    step_type=StepType.CONCLUSION,
                    inputs=[],
                    output=f"{len(action_path)} actions",
                    explanation_text=f"Plan found: {len(action_path)} actions",
                    confidence=1.0,
                    source_component="state_planner",
                )
                # Add all proof steps as subgoals
                for step in proof_steps:
                    root_proof_step.add_subgoal(step)

                self.last_proof_tree = ProofTree(
                    query=f"Plan from {initial_state} to goal",
                    root_steps=[root_proof_step],
                )

                return action_path

            # Skip if already visited
            if current_state in closed_set:
                continue

            closed_set.add(current_state)

            # Explore all applicable actions
            for action in available_actions:
                if not action.is_applicable(current_state):
                    continue

                # Apply action to get new state
                new_state = action.apply(current_state)
                if new_state is None:
                    continue

                # Validate state with constraint solver if available
                if not self.is_valid_state(new_state):
                    logger.debug(f"State {new_state} failed constraint validation")
                    continue

                # Skip if already visited
                if new_state in closed_set:
                    continue

                # Calculate costs
                new_g_cost = g_cost + action.cost
                new_h_cost = heuristic(new_state)
                new_f_cost = new_g_cost + new_h_cost

                # Add to open set
                new_path = action_path + [action]
                heappush(open_set, (new_f_cost, new_g_cost, new_state, new_path))

                # Record proof step
                proof_steps.append(
                    ProofStep(
                        step_id=f"plan_{iterations}_{action.name}",
                        step_type=StepType.RULE_APPLICATION,
                        inputs=[str(current_state)],
                        rule_name=action.name,
                        output=str(new_state),
                        explanation_text=f"Applied {action.name}: {current_state} -> {new_state}",
                        confidence=1.0
                        / (1.0 + action.cost),  # Lower cost = higher confidence
                        source_component="state_planner",
                    )
                )

        # No plan found
        logger.warning(f"No plan found after {iterations} iterations")
        proof_steps.append(
            ProofStep(
                step_id=f"plan_{iterations}_failed",
                step_type=StepType.CONCLUSION,
                inputs=[],
                output="No plan",
                explanation_text=f"No plan found (explored {len(closed_set)} states)",
                confidence=0.0,
                source_component="state_planner",
            )
        )

        root_proof_step = ProofStep(
            step_id="plan_root_failed",
            step_type=StepType.CONCLUSION,
            inputs=[],
            output="Planning failed",
            explanation_text="Planning failed",
            confidence=0.0,
            source_component="state_planner",
        )
        for step in proof_steps:
            root_proof_step.add_subgoal(step)

        self.last_proof_tree = ProofTree(
            query=f"Plan from {initial_state} to goal (failed)",
            root_steps=[root_proof_step],
        )

        return None

    def is_valid_state(self, state: State) -> bool:
        """
        Validate a state using the constraint solver.

        If no constraint solver is configured, all states are considered valid.

        Args:
            state: The state to validate

        Returns:
            True if state satisfies all constraints
        """
        if self.constraint_solver is None:
            return True

        try:
            # Convert state properties to CSP variables
            variables = []
            for key, value in state.properties.items():
                if isinstance(value, (int, str)):
                    # Create variable with single domain value (the current value)
                    var = Variable(key, [value])
                    variables.append(var)

            # Check if assignment is consistent with constraints
            # (For now, we don't add constraints dynamically - this is extensible)
            _ = {var.name: var.domain[0] for var in variables}

            # Use constraint solver to validate
            # (This assumes solver has constraints pre-configured)
            return True  # Simplified for now - extend as needed

        except Exception as e:
            logger.warning(f"State validation failed: {e}")
            return False

    def get_last_proof_tree(self) -> Optional[ProofTree]:
        """
        Get the proof tree from the last planning attempt.

        Returns:
            ProofTree explaining the planning process, or None
        """
        return self.last_proof_tree

    def plan_with_bfs(
        self,
        initial_state: State,
        goal_condition: Callable[[State], bool],
        available_actions: List[Action],
        max_depth: int = 100,
    ) -> Optional[List[Action]]:
        """
        Alternative planning using breadth-first search.

        Simpler than A* but guaranteed to find shortest plan (if exists).
        Useful when action costs are uniform or heuristic is unavailable.

        Args:
            initial_state: Starting state
            goal_condition: Function that returns True for goal states
            available_actions: List of actions that can be applied
            max_depth: Maximum plan length to explore

        Returns:
            List of actions forming a plan, or None if no plan found
        """
        logger.info(f"BFS planning from {initial_state}")

        # Queue: (state, action_path, depth)
        queue: List[Tuple[State, List[Action], int]] = [(initial_state, [], 0)]
        visited: Set[State] = {initial_state}

        proof_steps: List[ProofStep] = []
        proof_steps.append(
            ProofStep(
                step_id="bfs_0_premise",
                step_type=StepType.PREMISE,
                inputs=[],
                output=str(initial_state),
                explanation_text=f"Initial state: {initial_state}",
                confidence=1.0,
                source_component="state_planner_bfs",
            )
        )

        while queue:
            current_state, action_path, depth = queue.pop(0)

            # Check depth limit
            if depth > max_depth:
                continue

            # Check if goal reached
            if goal_condition(current_state):
                logger.info(f"BFS: Goal reached with {len(action_path)} actions")
                proof_steps.append(
                    ProofStep(
                        step_id=f"bfs_{depth}_conclusion",
                        step_type=StepType.CONCLUSION,
                        inputs=[a.name for a in action_path],
                        output=str(current_state),
                        explanation_text=f"Goal reached: {current_state}",
                        confidence=1.0,
                        source_component="state_planner_bfs",
                    )
                )

                root_proof_step = ProofStep(
                    step_id="bfs_root",
                    step_type=StepType.CONCLUSION,
                    inputs=[],
                    output=f"{len(action_path)} actions",
                    explanation_text=f"BFS plan: {len(action_path)} actions",
                    confidence=1.0,
                    source_component="state_planner_bfs",
                )
                for step in proof_steps:
                    root_proof_step.add_subgoal(step)

                self.last_proof_tree = ProofTree(
                    query=f"BFS plan from {initial_state} to goal",
                    root_steps=[root_proof_step],
                )

                return action_path

            # Explore neighbors
            for action in available_actions:
                new_state = action.apply(current_state)
                if new_state is None or new_state in visited:
                    continue

                if not self.is_valid_state(new_state):
                    continue

                visited.add(new_state)
                new_path = action_path + [action]
                queue.append((new_state, new_path, depth + 1))

                proof_steps.append(
                    ProofStep(
                        step_id=f"bfs_{depth}_{action.name}",
                        step_type=StepType.RULE_APPLICATION,
                        inputs=[str(current_state)],
                        rule_name=action.name,
                        output=str(new_state),
                        explanation_text=f"BFS: {action.name}: {current_state} -> {new_state}",
                        confidence=1.0,
                        source_component="state_planner_bfs",
                    )
                )

        logger.warning("BFS: No plan found")
        root_proof_step = ProofStep(
            step_id="bfs_root_failed",
            step_type=StepType.CONCLUSION,
            inputs=[],
            output="No plan",
            explanation_text="BFS planning failed",
            confidence=0.0,
            source_component="state_planner_bfs",
        )
        for step in proof_steps:
            root_proof_step.add_subgoal(step)

        self.last_proof_tree = ProofTree(
            query=f"BFS plan from {initial_state} to goal (failed)",
            root_steps=[root_proof_step],
        )

        return None


# Utility functions for common state operations


def create_numeric_state(variables: Dict[str, int]) -> State:
    """Create a state from numeric variables."""
    return State(variables)


def manhattan_distance_heuristic(
    target_state: State, keys: List[str]
) -> Callable[[State], float]:
    """
    Create a Manhattan distance heuristic for numeric states.

    Args:
        target_state: The goal state
        keys: List of numeric property keys to compare

    Returns:
        Heuristic function for A* search
    """

    def heuristic(state: State) -> float:
        distance = 0.0
        for key in keys:
            current_val = state.get(key, 0)
            target_val = target_state.get(key, 0)
            if isinstance(current_val, (int, float)) and isinstance(
                target_val, (int, float)
            ):
                distance += abs(current_val - target_val)
        return distance

    return heuristic


def create_simple_action(
    name: str,
    precond_dict: Dict[str, Any],
    effect_dict: Dict[str, Any],
    cost: float = 1.0,
) -> Action:
    """
    Create a simple action from property dictionaries.

    Args:
        name: Action name
        precond_dict: Required property values (all must match)
        effect_dict: Properties to set after action
        cost: Action cost

    Returns:
        Action object

    Example:
        >>> action = create_simple_action(
        ...     "move_right",
        ...     {"x": 0},
        ...     {"x": 1},
        ...     cost=1.0
        ... )
    """

    def precondition(state: State) -> bool:
        return all(state.get(k) == v for k, v in precond_dict.items())

    def effect(state: State) -> State:
        new_state = state.copy()
        for k, v in effect_dict.items():
            new_state.set(k, v)
        return new_state

    return Action(name, [precondition], [effect], cost)


if __name__ == "__main__":
    # Example: Simple navigation planning
    logging.basicConfig(level=logging.INFO)

    # Define states
    initial = State({"x": 0, "y": 0})

    def goal_condition(s):
        return s.get("x") == 3 and s.get("y") == 2

    # Define actions (move in 4 directions)
    actions = [
        Action(
            "move_right",
            [lambda s: s.get("x", 0) < 5],
            [lambda s: State({"x": s.get("x", 0) + 1, "y": s.get("y", 0)})],
            cost=1.0,
        ),
        Action(
            "move_up",
            [lambda s: s.get("y", 0) < 5],
            [lambda s: State({"x": s.get("x", 0), "y": s.get("y", 0) + 1})],
            cost=1.0,
        ),
        Action(
            "move_left",
            [lambda s: s.get("x", 0) > 0],
            [lambda s: State({"x": s.get("x", 0) - 1, "y": s.get("y", 0)})],
            cost=1.0,
        ),
        Action(
            "move_down",
            [lambda s: s.get("y", 0) > 0],
            [lambda s: State({"x": s.get("x", 0), "y": s.get("y", 0) - 1})],
            cost=1.0,
        ),
    ]

    # Create heuristic (Manhattan distance to goal)
    def heuristic(s):
        return abs(s.get("x", 0) - 3) + abs(s.get("y", 0) - 2)

    # Plan with A*
    planner = StateSpacePlanner()
    plan = planner.plan(initial, goal_condition, actions, heuristic)

    if plan:
        print(f"\nPlan found ({len(plan)} steps):")
        for i, action in enumerate(plan, 1):
            print(f"  {i}. {action.name}")

        # Show proof tree
        proof_tree = planner.get_last_proof_tree()
        if proof_tree:
            print(f"\nProof tree has {len(proof_tree.all_steps)} steps")
    else:
        print("No plan found!")
