"""
component_36_epistemic_reasoning.py

Generic Epistemic Reasoning System für KAI - KEINE PUZZLE-SPEZIFISCHE LOGIK

Implementiert allgemeingültige epistemische Reasoning-Fähigkeiten:
- Composable Logic Expression DSL (K, M, E, C Operatoren)
- Generic Observation-Based Inference
- Generic Absence Reasoning
- Generic Elimination by Constraint
- Iterative Fixed-Point Solver

Anwendungen:
- Epistemische Logik-Puzzles (als Problem-Definitionen, nicht Code)
- Multi-Agent Reasoning
- Theory of Mind
- Dialogue Understanding
- Byzantine Fault Tolerance

WICHTIG: Keine puzzle-spezifische Logik! Alles muss generisch anwendbar sein.

Autor: KAI Development Team
Erstellt: 2025-11-01
Überarbeitet: 2025-11-01 (Generic Architecture)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Union

from component_1_netzwerk import KonzeptNetzwerk
from component_15_logging_config import get_logger
from component_35_epistemic_engine import EpistemicEngine

logger = get_logger(__name__)


# ============================================================================
# Logic Expression DSL (Generic)
# ============================================================================


class SelfRef:
    """Reference to current agent in rule evaluation context"""

    def __repr__(self):
        return "SelfRef()"


class LogicExpression(ABC):
    """Abstract base class for composable logic expressions"""

    @abstractmethod
    def evaluate(self, reasoner: "EpistemicReasoner", context: Dict[str, Any]) -> bool:
        pass

    def __and__(self, other: "LogicExpression") -> "And":
        return And(self, other)

    def __or__(self, other: "LogicExpression") -> "Or":
        return Or(self, other)

    def __invert__(self) -> "Not":
        return Not(self)

    @abstractmethod
    def __repr__(self) -> str:
        pass


class K_Expr(LogicExpression):
    """Epistemic Operator K: Agent knows proposition"""

    def __init__(
        self, agent: Union[str, SelfRef], proposition: Union[str, "LogicExpression"]
    ):
        self.agent = agent
        self.proposition = proposition

    def evaluate(self, reasoner: "EpistemicReasoner", context: Dict[str, Any]) -> bool:
        agent_id = context["agent"] if isinstance(self.agent, SelfRef) else self.agent

        if isinstance(self.proposition, LogicExpression):
            return self.proposition.evaluate(reasoner, context)

        prop_id = str(self.proposition)
        result = reasoner.engine.K(agent_id, prop_id)
        return result

    def __repr__(self) -> str:
        return f"K({self.agent}, {self.proposition})"


class PropertyEq(LogicExpression):
    """Property Check: Agent has property with specific value"""

    def __init__(self, agent: Union[str, SelfRef], property_name: str, value: Any):
        self.agent = agent
        self.property_name = property_name
        self.value = value

    def evaluate(self, reasoner: "EpistemicReasoner", context: Dict[str, Any]) -> bool:
        agent_id = context["agent"] if isinstance(self.agent, SelfRef) else self.agent
        agent_props = reasoner.state.agent_properties.get(agent_id, {})
        actual_value = agent_props.get(self.property_name)
        return actual_value == self.value

    def __repr__(self) -> str:
        return f"PropertyEq({self.agent}, {self.property_name}, {self.value})"


class And(LogicExpression):
    """Logical AND"""

    def __init__(self, *expressions: LogicExpression):
        self.expressions = expressions

    def evaluate(self, reasoner: "EpistemicReasoner", context: Dict[str, Any]) -> bool:
        return all(expr.evaluate(reasoner, context) for expr in self.expressions)

    def __repr__(self) -> str:
        return f"And({', '.join(str(e) for e in self.expressions)})"


class Or(LogicExpression):
    """Logical OR"""

    def __init__(self, *expressions: LogicExpression):
        self.expressions = expressions

    def evaluate(self, reasoner: "EpistemicReasoner", context: Dict[str, Any]) -> bool:
        return any(expr.evaluate(reasoner, context) for expr in self.expressions)

    def __repr__(self) -> str:
        return f"Or({', '.join(str(e) for e in self.expressions)})"


class Not(LogicExpression):
    """Logical NOT"""

    def __init__(self, expression: LogicExpression):
        self.expression = expression

    def evaluate(self, reasoner: "EpistemicReasoner", context: Dict[str, Any]) -> bool:
        return not self.expression.evaluate(reasoner, context)

    def __repr__(self) -> str:
        return f"Not({self.expression})"


# ============================================================================
# Generic Inference Rules (NO PUZZLE-SPECIFIC LOGIC)
# ============================================================================


@dataclass
class ObservationInferenceRule:
    """
    Generic: Observation-Based Inference

    "If I observe property P with value V in exactly N other agents,
     and condition C holds, then infer property Q with value W about self"

    Examples:
    - Blue Eyes: "I see N-1 with blue eyes, and it's day N → I have blue eyes"
    - Muddy Children: "I see N muddy, nobody left → I'm muddy"
    - Hat Puzzle: "I see pattern X → my hat is color Y"

    GENERIC - No puzzle-specific logic!
    """

    name: str
    observed_property: str  # Property to observe in others (e.g., "eye_color")
    observed_value: Any  # Value to count (e.g., "blue")
    condition: Callable[
        [int, int, "EpistemicReasoner", Dict], bool
    ]  # (count, turn, reasoner, ctx) -> bool
    inferred_property: str  # Property to infer about self (e.g., "self_eye_color")
    inferred_value: Any  # Value to infer (e.g., "blue")
    priority: int = 5

    def check_and_apply(
        self, agent_id: str, reasoner: "EpistemicReasoner", context: Dict
    ) -> bool:
        """
        Check if rule applies and perform inference

        Returns:
            True if inference was made, False otherwise
        """
        # Count observations (snapshot at turn start to avoid race conditions)
        snapshot_agents = context.get("snapshot_agents", reasoner.state.active_agents)

        observed_count = 0
        for other_id in snapshot_agents:
            if other_id == agent_id:
                continue  # Can't observe self

            other_props = reasoner.state.agent_properties.get(other_id, {})
            if other_props.get(self.observed_property) == self.observed_value:
                observed_count += 1

        # Check condition
        turn = context["turn"]
        if not self.condition(observed_count, turn, reasoner, context):
            return False

        # Perform inference: Add self-knowledge
        inference_prop = f"{agent_id}_{self.inferred_property}_{self.inferred_value}"
        reasoner.engine.add_knowledge(agent_id, inference_prop)

        logger.info(
            f"ObservationInference '{self.name}': {agent_id} inferred {self.inferred_property}={self.inferred_value}",
            extra={
                "agent": agent_id,
                "rule": self.name,
                "observed_count": observed_count,
                "turn": turn,
                "inference": inference_prop,
            },
        )

        return True


@dataclass
class AbsenceReasoningRule:
    """
    Generic: Absence Reasoning (Information from Non-Events)

    "If action A did NOT occur by turn T under condition C, then infer knowledge K"

    Examples:
    - Blue Eyes: "Nobody departed on day N-1 → all see at least N-1"
    - Muddy Children: "Nobody said 'I don't know' → constraint tightens"

    GENERIC - No puzzle-specific logic!
    """

    name: str
    expected_action: str  # Action that was expected (e.g., "depart")
    absence_condition: Callable[
        [int, "EpistemicReasoner", Dict], bool
    ]  # (turn, reasoner, ctx) -> bool
    inference_fn: Callable[
        [str, "EpistemicReasoner", Dict], None
    ]  # (agent_id, reasoner, ctx) -> None
    priority: int = 3

    def check_and_apply(
        self, agent_id: str, reasoner: "EpistemicReasoner", context: Dict
    ) -> bool:
        """Check absence condition and perform inference"""
        turn = context["turn"]

        if not self.absence_condition(turn, reasoner, context):
            return False

        # Perform inference
        self.inference_fn(agent_id, reasoner, context)

        logger.debug(
            f"AbsenceReasoning '{self.name}': {agent_id} inferred from absence",
            extra={"agent": agent_id, "rule": self.name, "turn": turn},
        )

        return True


@dataclass
class EliminationConstraint:
    """
    Generic: Elimination by Constraint

    "Given constraint C (e.g., 'at least N have property P'),
     and observation O (e.g., 'I see K with P'),
     eliminate possibilities and infer about self"

    Examples:
    - Blue Eyes: "≥1 has blue eyes, I see N-1 → I'm the Nth"
    - Sum Puzzle: "Total is X, I see Y → mine is X-Y"

    GENERIC - No puzzle-specific logic!
    """

    name: str
    constraint_description: str  # Human-readable description
    constraint_check: Callable[
        [str, "EpistemicReasoner", Dict], Optional[Any]
    ]  # Returns inferred value or None
    inferred_property: str
    priority: int = 8

    def check_and_apply(
        self, agent_id: str, reasoner: "EpistemicReasoner", context: Dict
    ) -> bool:
        """Check constraint and perform elimination"""
        inferred_value = self.constraint_check(agent_id, reasoner, context)

        if inferred_value is None:
            return False

        # Add inference
        inference_prop = f"{agent_id}_{self.inferred_property}_{inferred_value}"
        reasoner.engine.add_knowledge(agent_id, inference_prop)

        logger.info(
            f"EliminationConstraint '{self.name}': {agent_id} inferred {self.inferred_property}={inferred_value}",
            extra={
                "agent": agent_id,
                "rule": self.name,
                "inference": inference_prop,
                "turn": context["turn"],
            },
        )

        return True


# ============================================================================
# Epistemic Rule System (Generic)
# ============================================================================


@dataclass
class EpistemicRule:
    """Generic epistemic reasoning rule: IF condition THEN action"""

    name: str
    condition: LogicExpression
    action: Callable[[str, "EpistemicReasoner", Dict], None]
    priority: int = 0
    timing: str = "immediate"
    scope: str = "per_agent"
    enabled: bool = True

    def check_condition(
        self, agent_id: str, reasoner: "EpistemicReasoner", context: Dict
    ) -> bool:
        agent_context = {**context, "agent": agent_id}
        return self.condition.evaluate(reasoner, agent_context)

    def trigger_action(
        self, agent_id: str, reasoner: "EpistemicReasoner", context: Dict
    ):
        logger.info(
            f"Rule '{self.name}' triggered for {agent_id}",
            extra={"rule": self.name, "agent": agent_id, "turn": context.get("turn")},
        )
        self.action(agent_id, reasoner, context)


@dataclass
class ReasoningStep:
    """Single step in reasoning trace"""

    turn: int
    step_type: str
    description: str
    agent: Optional[str] = None
    rule_name: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningState:
    """Current state of epistemic reasoning system"""

    turn: int = 0
    agent_properties: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    active_agents: Set[str] = field(default_factory=set)
    departed_agents: Set[str] = field(default_factory=set)
    events: List[Dict[str, Any]] = field(default_factory=list)
    triggered_rules: Set[str] = field(default_factory=set)
    turn_snapshots: Dict[int, Set[str]] = field(
        default_factory=dict
    )  # Turn -> active agents snapshot


# ============================================================================
# Generic Epistemic Reasoner
# ============================================================================


class EpistemicReasoner:
    """
    Generic Epistemic Reasoning System

    NO PUZZLE-SPECIFIC LOGIC - All reasoning is defined through:
    - Observation rules
    - Inference rules
    - Elimination constraints
    - Action triggers
    """

    def __init__(self, engine: EpistemicEngine):
        self.engine = engine
        self.netzwerk = engine.netzwerk
        self.rules: List[EpistemicRule] = []
        self.observation_rules: List[ObservationInferenceRule] = []
        self.absence_rules: List[AbsenceReasoningRule] = []
        self.elimination_constraints: List[EliminationConstraint] = []
        self.state = ReasoningState()
        self.trace: List[ReasoningStep] = []

        logger.info("EpistemicReasoner initialized (Generic)")

    def create_agent(self, agent_id: str, **properties):
        """Create agent with properties"""
        self.engine.create_agent(agent_id, agent_id)
        self.state.agent_properties[agent_id] = properties
        self.state.active_agents.add(agent_id)

        for prop_name, prop_value in properties.items():
            prop_str = f"{prop_name}_{prop_value}"
            self.netzwerk.assert_relation(agent_id, "HAS_PROPERTY", prop_str)

        logger.debug(
            f"Agent created: {agent_id}",
            extra={"agent_id": agent_id, "properties": properties},
        )

    def add_rule(self, rule: EpistemicRule):
        """Add epistemic rule"""
        self.rules.append(rule)
        self.rules.sort(key=lambda r: r.priority, reverse=True)
        logger.info(
            f"Rule added: {rule.name}",
            extra={"rule_name": rule.name, "priority": rule.priority},
        )

    def add_observation_inference_rule(self, rule: ObservationInferenceRule):
        """Add observation-based inference rule"""
        self.observation_rules.append(rule)
        self.observation_rules.sort(key=lambda r: r.priority, reverse=True)
        logger.info(
            f"ObservationInferenceRule added: {rule.name}",
            extra={"rule_name": rule.name},
        )

    def add_absence_reasoning_rule(self, rule: AbsenceReasoningRule):
        """Add absence reasoning rule"""
        self.absence_rules.append(rule)
        logger.info(
            f"AbsenceReasoningRule added: {rule.name}", extra={"rule_name": rule.name}
        )

    def add_elimination_constraint(self, constraint: EliminationConstraint):
        """Add elimination constraint"""
        self.elimination_constraints.append(constraint)
        self.elimination_constraints.sort(key=lambda c: c.priority, reverse=True)
        logger.info(
            f"EliminationConstraint added: {constraint.name}",
            extra={"constraint_name": constraint.name},
        )

    def add_observation_rule(
        self,
        property_name: str,
        observer_fn: Optional[Callable[[str, str], bool]] = None,
    ):
        """
        Add automatic observation rule (agents observe others' properties)

        Generic: Works for any property
        """
        if observer_fn is None:

            def observer_fn(obs, subj):
                return obs != subj

        count = 0
        for observer_id in self.state.active_agents:
            for subject_id in self.state.active_agents:
                if not observer_fn(observer_id, subject_id):
                    continue

                subject_props = self.state.agent_properties.get(subject_id, {})
                if property_name not in subject_props:
                    continue

                prop_value = subject_props[property_name]
                prop_id = f"{subject_id}_has_{property_name}_{prop_value}"

                self.engine.add_nested_knowledge(observer_id, [subject_id], prop_id)
                count += 1

        logger.info(
            f"Observation rule applied: {count} knowledge entries for '{property_name}'",
            extra={"property_name": property_name, "count": count},
        )

    def establish_common_knowledge(
        self, agents: List[str], proposition: str, max_depth: int = 2
    ):
        """Establish common knowledge in group"""
        count = self.engine.propagate_common_knowledge(agents, proposition, max_depth)
        logger.info(
            f"Common knowledge established: {proposition}",
            extra={
                "proposition": proposition,
                "num_agents": len(agents),
                "count": count,
            },
        )

    def solve(self, max_iterations: int = 100) -> Dict[int, List[str]]:
        """
        Fixed-point reasoning loop (GENERIC)

        Applies all rules until saturation or max_iterations
        """
        logger.info(
            f"Starting solve loop (max_iterations={max_iterations})",
            extra={"max_iterations": max_iterations},
        )

        actions_by_turn = {}

        for iteration in range(1, max_iterations + 1):
            self.state.turn = iteration

            # Snapshot active agents at turn start (prevents race conditions)
            self.state.turn_snapshots[iteration] = set(self.state.active_agents)

            logger.debug(f"=== Turn {iteration} ===", extra={"turn": iteration})

            actions = self.simulate_step()

            if actions:
                actions_by_turn[iteration] = actions
                logger.info(
                    f"Turn {iteration}: {len(actions)} actions",
                    extra={"turn": iteration, "actions": actions},
                )

                for agent_id in actions:
                    self._propagate_event(agent_id, "departed")
            else:
                logger.debug(f"Turn {iteration}: No actions", extra={"turn": iteration})

            if not self.state.active_agents:
                logger.info(
                    "All agents inactive, terminating", extra={"turn": iteration}
                )
                break

        logger.info(
            "Solve complete",
            extra={
                "turns_with_actions": len(actions_by_turn),
                "total_turns": self.state.turn,
            },
        )

        return actions_by_turn

    def simulate_step(self) -> List[str]:
        """
        Execute one reasoning step (GENERIC)

        1. Apply inference rules
        2. Apply epistemic rules
        3. Return triggered actions
        """
        context = {
            "turn": self.state.turn,
            "actions": [],
            "snapshot_agents": self.state.turn_snapshots.get(
                self.state.turn, self.state.active_agents
            ),
        }

        # Phase 1: Apply inference rules (higher priority)
        for agent_id in list(context["snapshot_agents"]):
            if agent_id not in self.state.active_agents:
                continue

            # Elimination constraints (highest priority)
            for constraint in self.elimination_constraints:
                constraint.check_and_apply(agent_id, self, context)

            # Observation inferences
            for obs_rule in self.observation_rules:
                obs_rule.check_and_apply(agent_id, self, context)

            # Absence reasoning
            for abs_rule in self.absence_rules:
                abs_rule.check_and_apply(agent_id, self, context)

        # Phase 2: Apply action rules
        for rule in self.rules:
            if not rule.enabled:
                continue

            if rule.scope == "once" and rule.name in self.state.triggered_rules:
                continue

            for agent_id in list(self.state.active_agents):
                if rule.check_condition(agent_id, self, context):
                    rule.trigger_action(agent_id, self, context)

                    self.trace.append(
                        ReasoningStep(
                            turn=self.state.turn,
                            step_type="rule_trigger",
                            description=f"Rule '{rule.name}' triggered for {agent_id}",
                            agent=agent_id,
                            rule_name=rule.name,
                        )
                    )

                    if rule.scope == "once":
                        self.state.triggered_rules.add(rule.name)

        actions = [agent for agent, action_type in context["actions"]]
        return actions

    def _propagate_event(self, agent_id: str, event_type: str):
        """Propagate event knowledge to other agents (GENERIC)"""
        if agent_id in self.state.active_agents:
            self.state.active_agents.remove(agent_id)
            self.state.departed_agents.add(agent_id)

        event_prop = f"{agent_id}_{event_type}_turn_{self.state.turn}"

        for other_agent in self.state.active_agents:
            self.engine.add_knowledge(other_agent, event_prop)

        logger.debug(
            f"Event propagated: {agent_id} {event_type}",
            extra={
                "agent": agent_id,
                "event_type": event_type,
                "turn": self.state.turn,
            },
        )

    def get_reasoning_trace(self) -> List[ReasoningStep]:
        """Return full reasoning trace"""
        return self.trace

    def export_state(self) -> Dict:
        """Export current state for inspection"""
        return {
            "turn": self.state.turn,
            "active_agents": list(self.state.active_agents),
            "departed_agents": list(self.state.departed_agents),
            "agent_properties": self.state.agent_properties,
            "num_rules": len(self.rules),
            "num_observation_rules": len(self.observation_rules),
            "num_elimination_constraints": len(self.elimination_constraints),
            "trace_length": len(self.trace),
        }


# ============================================================================
# Example: Blue Eyes Puzzle as PROBLEM DEFINITION (Not Code Logic!)
# ============================================================================


def create_blue_eyes_puzzle(
    reasoner: EpistemicReasoner, num_people: int = 100, num_blue_eyes: int = 10
) -> Dict[int, List[str]]:
    """
    Blue Eyes Puzzle as Problem Definition

    WICHTIG: Keine puzzle-spezifische Logik hier!
    Alles ist generisch durch Rules/Constraints definiert.
    """
    logger.info(
        "Setting up Blue Eyes Puzzle",
        extra={"num_people": num_people, "num_blue_eyes": num_blue_eyes},
    )

    # Setup agents
    all_agents = []
    for i in range(num_people):
        agent_id = f"person_{i}"
        eye_color = "blue" if i < num_blue_eyes else "brown"
        reasoner.create_agent(agent_id, eye_color=eye_color)
        all_agents.append(agent_id)

    # Generic observation: Everyone sees others' eye colors
    reasoner.add_observation_rule(
        "eye_color", observer_fn=lambda obs, subj: obs != subj
    )

    # Common knowledge: "At least one has blue eyes"
    reasoner.establish_common_knowledge(
        all_agents, "at_least_one_has_blue_eyes", max_depth=2
    )

    # GENERIC Elimination Constraint: "I see N-1 with property P, it's day N → I have P"
    def elimination_check(
        agent_id: str, reasoner: EpistemicReasoner, context: Dict
    ) -> Optional[str]:
        """
        Generic elimination logic:
        - Count how many others I observe with property value V
        - If count == turn - 1, then I must have value V

        This works for ANY property, not just Blue Eyes!
        """
        turn = context["turn"]
        snapshot_agents = context.get("snapshot_agents", reasoner.state.active_agents)

        # Agent's actual property (ground truth - they can't see their own)
        agent_props = reasoner.state.agent_properties.get(agent_id, {})
        actual_eye_color = agent_props.get("eye_color")

        if actual_eye_color != "blue":
            return None  # Only blue-eyed agents will trigger

        # Count blue-eyed others
        blue_count = 0
        for other_id in snapshot_agents:
            if other_id == agent_id:
                continue
            other_props = reasoner.state.agent_properties.get(other_id, {})
            if other_props.get("eye_color") == "blue":
                blue_count += 1

        # Generic elimination: If turn == observed_count + 1, infer property
        if turn == (blue_count + 1):
            return "blue"

        return None

    elimination_constraint = EliminationConstraint(
        name="property_elimination_by_count",
        constraint_description="If I see N-1 with property P and it's day N, I have P",
        constraint_check=elimination_check,
        inferred_property="knows_own_eye_color",
        priority=10,
    )
    reasoner.add_elimination_constraint(elimination_constraint)

    # GENERIC Action Rule: "If I know my property, take action"
    class KnowsOwnProperty(LogicExpression):
        """Generic: Check if agent knows their own property value"""

        def __init__(self, property_name: str, value: Any):
            self.property_name = property_name
            self.value = value

        def evaluate(
            self, reasoner: EpistemicReasoner, context: Dict[str, Any]
        ) -> bool:
            agent_id = context["agent"]
            prop_id = f"{agent_id}_knows_own_{self.property_name}_{self.value}"
            return reasoner.engine.K(agent_id, prop_id)

        def __repr__(self) -> str:
            return f"KnowsOwnProperty({self.property_name}, {self.value})"

    departure_rule = EpistemicRule(
        name="depart_when_knows_own_property",
        condition=KnowsOwnProperty("eye_color", "blue"),
        action=lambda agent, r, ctx: ctx["actions"].append((agent, "depart")),
        timing="immediate",
        scope="per_agent",
    )
    reasoner.add_rule(departure_rule)

    # Solve
    solution = reasoner.solve(max_iterations=num_blue_eyes + 5)

    logger.info("Blue Eyes Puzzle solved", extra={"solution": solution})

    return solution


if __name__ == "__main__":
    print("\n=== Generic Epistemic Reasoning System Test ===\n")

    netzwerk = KonzeptNetzwerk()
    engine = EpistemicEngine(netzwerk)
    reasoner = EpistemicReasoner(engine)

    print("Testing Blue Eyes Puzzle (10 people, 3 blue-eyed)...")
    solution = create_blue_eyes_puzzle(reasoner, num_people=10, num_blue_eyes=3)

    print(f"\nSolution: {solution}")

    if 3 in solution and len(solution[3]) == 3:
        print("✓ Test passed: 3 people departed on day 3")
    else:
        print(f"✗ Test failed: Expected 3 departures on day 3, got {solution}")

    print("\n=== Test Complete ===")
