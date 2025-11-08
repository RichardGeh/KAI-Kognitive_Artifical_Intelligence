"""
component_38_elimination_reasoning.py

Generic Elimination & Deductive Reasoning Engine

Implementiert allgemeingültige Eliminationslogik für Multi-Agent-Puzzles:
- Statement-Based Elimination (Agent sagt X → eliminiere inkonsistente Möglichkeiten)
- Deductive Chains (mehrere Elimination-Schritte kombinieren)
- Meta-Knowledge Elimination (Higher-Order Reasoning)
- Iterative Refinement (Fixed-Point Elimination)

WICHTIG: Keine puzzle-spezifische Logik! Komplett generisch.

Anwendungen:
- Logic Puzzles (Cheryl's Birthday, Sum and Product, etc.)
- Knights and Knaves (Truth-teller vs Liar reasoning)
- Zebra Puzzles (Einstein's Riddle)
- Sudoku-ähnliche Deduktion
- Multi-Agent Information Asymmetry

Autor: KAI Development Team
Erstellt: 2025-11-01
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from component_15_logging_config import get_logger
from component_17_proof_explanation import ProofStep, ProofTree, StepType
from component_35_epistemic_engine import EpistemicEngine
from component_37_partial_observation import (
    PartialObservationReasoner,
    PartialObserver,
    WorldObject,
)

logger = get_logger(__name__)


# ============================================================================
# Statement Types (Generic)
# ============================================================================


class StatementType(Enum):
    """Types of statements agents can make"""

    I_KNOW = "i_know"  # "I know the answer"
    I_DONT_KNOW = "i_dont_know"  # "I don't know the answer"
    I_KNOW_OTHER_DOESNT_KNOW = (
        "i_know_other_doesnt_know"  # "I know the other doesn't know"
    )
    I_KNOW_OTHER_KNOWS = "i_know_other_knows"  # "I know the other knows"
    NOW_I_KNOW = "now_i_know"  # "Now I know" (after update)
    STILL_DONT_KNOW = "still_dont_know"  # "I still don't know"


@dataclass
class AgentStatement:
    """
    Generic statement made by an agent during reasoning.

    Examples:
    - "Albert: I know Bernard doesn't know" → (albert, I_KNOW_OTHER_DOESNT_KNOW, bernard)
    - "Bernard: Now I know" → (bernard, NOW_I_KNOW, None)
    """

    speaker: str
    statement_type: StatementType
    about_agent: Optional[str] = None  # For meta-statements about other agents
    turn: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Elimination Rules (Generic)
# ============================================================================


@dataclass
class EliminationRule:
    """
    Generic elimination rule: Given statement S, eliminate objects.

    Rule structure:
    - Precondition: When can this rule apply?
    - Filter: Which objects should be eliminated?
    - Explanation: Why were objects eliminated?
    """

    name: str
    statement_type: StatementType
    filter_predicate: Callable[
        [WorldObject, AgentStatement, "EliminationContext"], bool
    ]
    explanation_template: str
    priority: int = 5

    def apply(
        self,
        candidates: List[WorldObject],
        statement: AgentStatement,
        context: "EliminationContext",
    ) -> Tuple[List[WorldObject], str]:
        """
        Apply elimination rule to candidates.

        Args:
            candidates: Current candidate objects
            statement: Agent statement triggering elimination
            context: Reasoning context

        Returns:
            Tuple of (filtered_objects, explanation)
        """
        filtered = []
        eliminated = []

        for obj in candidates:
            if self.filter_predicate(obj, statement, context):
                filtered.append(obj)
            else:
                eliminated.append(obj)

        explanation = self.explanation_template.format(
            speaker=statement.speaker,
            about_agent=statement.about_agent if statement.about_agent else "other",
            eliminated_count=len(eliminated),
            remaining_count=len(filtered),
            eliminated_ids=[o.object_id for o in eliminated],
        )

        logger.info(
            f"Elimination rule '{self.name}' applied",
            extra={
                "rule": self.name,
                "before": len(candidates),
                "after": len(filtered),
                "eliminated": len(eliminated),
            },
        )

        return filtered, explanation


# ============================================================================
# Deductive Chains (Generic)
# ============================================================================


@dataclass
class DeductiveStep:
    """
    Single step in deductive reasoning chain.

    Tracks: statement → elimination → new candidates → new knowledge
    """

    turn: int
    statement: AgentStatement
    candidates_before: List[WorldObject]
    candidates_after: List[WorldObject]
    eliminated_objects: List[WorldObject]
    explanation: str
    confidence: float = 1.0


class DeductiveChain:
    """
    Sequence of deductive steps building to a conclusion.

    Generic: Works for ANY puzzle with sequential statements.
    """

    def __init__(self, initial_candidates: List[WorldObject]):
        self.steps: List[DeductiveStep] = []
        self.current_candidates = initial_candidates.copy()
        self.initial_count = len(initial_candidates)

    def add_step(
        self, statement: AgentStatement, eliminated: List[WorldObject], explanation: str
    ):
        """Add deductive step to chain"""
        candidates_before = self.current_candidates.copy()

        # Update current candidates
        self.current_candidates = [
            obj for obj in self.current_candidates if obj not in eliminated
        ]

        step = DeductiveStep(
            turn=statement.turn,
            statement=statement,
            candidates_before=candidates_before,
            candidates_after=self.current_candidates.copy(),
            eliminated_objects=eliminated,
            explanation=explanation,
        )

        self.steps.append(step)

        logger.debug(
            f"Deductive step added: {len(candidates_before)} → {len(self.current_candidates)}",
            extra={
                "turn": statement.turn,
                "speaker": statement.speaker,
                "eliminated": len(eliminated),
                "remaining": len(self.current_candidates),
            },
        )

    def get_solution(self) -> Optional[WorldObject]:
        """Get unique solution if exactly one candidate remains"""
        if len(self.current_candidates) == 1:
            return self.current_candidates[0]
        return None

    def is_solved(self) -> bool:
        """Check if unique solution found"""
        return len(self.current_candidates) == 1

    def is_contradictory(self) -> bool:
        """Check if no candidates remain (contradiction)"""
        return len(self.current_candidates) == 0

    def generate_proof_tree(self, query: str) -> ProofTree:
        """
        Generate ProofTree from deductive chain.

        Returns:
            ProofTree visualizing entire reasoning process
        """
        proof_tree = ProofTree(query=query)

        # Initial premise
        initial_step = ProofStep(
            step_id="step_0",
            step_type=StepType.PREMISE,
            explanation_text=f"Startmenge: {self.initial_count} Kandidaten",
            confidence=1.0,
            output=f"{self.initial_count} Objekte möglich",
            source_component="component_38_elimination_reasoning",
        )
        proof_tree.add_root_step(initial_step)

        # Each elimination step
        for i, step in enumerate(self.steps, start=1):
            step_type = (
                StepType.INFERENCE if step.candidates_after else StepType.CONTRADICTION
            )

            proof_step = ProofStep(
                step_id=f"step_{i}",
                step_type=step_type,
                explanation_text=step.explanation,
                confidence=step.confidence,
                output=f"{len(step.candidates_after)} Kandidaten verbleibend",
                source_component="component_38_elimination_reasoning",
            )
            proof_tree.add_root_step(proof_step)

        # Conclusion
        if self.is_solved():
            solution = self.get_solution()
            conclusion_step = ProofStep(
                step_id="step_conclusion",
                step_type=StepType.CONCLUSION,
                explanation_text=f"Eindeutige Lösung gefunden: {solution.object_id}",
                confidence=1.0,
                output=str(solution.properties),
                source_component="component_38_elimination_reasoning",
            )
            proof_tree.add_root_step(conclusion_step)
        elif self.is_contradictory():
            contradiction_step = ProofStep(
                step_id="step_contradiction",
                step_type=StepType.CONTRADICTION,
                explanation_text="Widerspruch: Keine Kandidaten verbleibend",
                confidence=0.0,
                output="Keine Lösung möglich",
                source_component="component_38_elimination_reasoning",
            )
            proof_tree.add_root_step(contradiction_step)

        return proof_tree


# ============================================================================
# Elimination Context (Generic)
# ============================================================================


@dataclass
class EliminationContext:
    """
    Context for elimination reasoning.

    Tracks:
    - Current candidates
    - Observers and their observations
    - Statement history
    - Meta-reasoning capabilities
    """

    reasoner: PartialObservationReasoner
    observers: Dict[str, PartialObserver]
    current_candidates: List[WorldObject]
    statement_history: List[AgentStatement] = field(default_factory=list)
    turn: int = 0

    def get_observer_observation(
        self, observer_id: str, obj: WorldObject
    ) -> Dict[str, Any]:
        """Get what observer sees for given object"""
        observer = self.observers.get(observer_id)
        if not observer:
            return {}
        return observer.observe(obj)

    def can_observer_identify(self, observer_id: str, obj: WorldObject) -> bool:
        """Check if observer can uniquely identify object among current candidates"""
        observation = self.get_observer_observation(observer_id, obj)

        matching = [
            candidate
            for candidate in self.current_candidates
            if self.get_observer_observation(observer_id, candidate) == observation
        ]

        return len(matching) == 1


# ============================================================================
# Elimination Reasoner (Generic)
# ============================================================================


class EliminationReasoner:
    """
    Generic elimination-based reasoning system.

    Combines:
    - Statement interpretation
    - Elimination rules
    - Deductive chains
    - Proof tree generation
    """

    def __init__(self, reasoner: PartialObservationReasoner):
        self.reasoner = reasoner
        self.rules: List[EliminationRule] = []
        self.deductive_chain: Optional[DeductiveChain] = None

        logger.info("EliminationReasoner initialized")

    def add_rule(self, rule: EliminationRule):
        """Add elimination rule"""
        self.rules.append(rule)
        self.rules.sort(key=lambda r: r.priority, reverse=True)

        logger.info(
            f"Elimination rule added: {rule.name}",
            extra={"rule": rule.name, "priority": rule.priority},
        )

    def create_standard_rules(self):
        """
        Create standard elimination rules (GENERIC).

        These rules work for ANY partial observation puzzle.
        """

        # Rule 1: "I know" → eliminate non-unique observations
        def filter_i_know(
            obj: WorldObject, stmt: AgentStatement, ctx: EliminationContext
        ) -> bool:
            """Keep only objects where speaker can uniquely identify"""
            return ctx.can_observer_identify(stmt.speaker, obj)

        self.add_rule(
            EliminationRule(
                name="eliminate_non_unique_for_i_know",
                statement_type=StatementType.I_KNOW,
                filter_predicate=filter_i_know,
                explanation_template="{speaker} sagt 'Ich weiß es' → eliminiere {eliminated_count} nicht-eindeutige Objekte",
                priority=10,
            )
        )

        # Rule 2: "Now I know" → eliminate non-unique after update
        def filter_now_i_know(
            obj: WorldObject, stmt: AgentStatement, ctx: EliminationContext
        ) -> bool:
            """Keep only objects where speaker can NOW uniquely identify"""
            return ctx.can_observer_identify(stmt.speaker, obj)

        self.add_rule(
            EliminationRule(
                name="eliminate_non_unique_for_now_i_know",
                statement_type=StatementType.NOW_I_KNOW,
                filter_predicate=filter_now_i_know,
                explanation_template="{speaker} sagt 'Jetzt weiß ich es' → eliminiere {eliminated_count} Objekte",
                priority=10,
            )
        )

        # Rule 3: "I know other doesn't know" → eliminate partitions with unique identifiers
        def filter_i_know_other_doesnt(
            obj: WorldObject, stmt: AgentStatement, ctx: EliminationContext
        ) -> bool:
            """
            Keep only objects where:
            - Speaker sees observation O
            - ALL objects with observation O are such that other agent CANNOT uniquely identify

            Generic version of: Albert sees month M, and NO date in month M has unique day
            """
            if not stmt.about_agent:
                return True  # No other agent specified, can't apply rule

            speaker = ctx.observers.get(stmt.speaker)
            other = ctx.observers.get(stmt.about_agent)

            if not speaker or not other:
                return True

            # What does speaker see for this object?
            speaker_obs = speaker.observe(obj)

            # Get all current candidates matching speaker's observation
            same_observation_objs = [
                candidate
                for candidate in ctx.current_candidates
                if speaker.observe(candidate) == speaker_obs
            ]

            # Check if ANY of these objects would allow other to uniquely identify
            for same_obj in same_observation_objs:
                other_obs = other.observe(same_obj)

                # Among current candidates, is other's observation unique?
                matching_for_other = [
                    c for c in ctx.current_candidates if other.observe(c) == other_obs
                ]

                if len(matching_for_other) == 1:
                    # Other COULD identify this object → speaker CANNOT know other doesn't know
                    return False

            # For ALL objects with speaker's observation, other cannot uniquely identify
            return True

        self.add_rule(
            EliminationRule(
                name="eliminate_partitions_with_unique_identifier",
                statement_type=StatementType.I_KNOW_OTHER_DOESNT_KNOW,
                filter_predicate=filter_i_know_other_doesnt,
                explanation_template="{speaker} sagt 'Ich weiß, dass {about_agent} es nicht weiß' → eliminiere {eliminated_count} Partitionen",
                priority=9,
            )
        )

        logger.info(
            "Standard elimination rules created", extra={"rule_count": len(self.rules)}
        )

    def process_statements(
        self, initial_candidates: List[WorldObject], statements: List[AgentStatement]
    ) -> Tuple[Optional[WorldObject], ProofTree]:
        """
        Process sequence of statements and find solution.

        Args:
            initial_candidates: Initial set of possible objects
            statements: Sequence of agent statements

        Returns:
            Tuple of (solution, proof_tree)
        """
        logger.info(
            f"Processing {len(statements)} statements",
            extra={
                "initial_candidates": len(initial_candidates),
                "statements": len(statements),
            },
        )

        # Initialize deductive chain
        self.deductive_chain = DeductiveChain(initial_candidates)

        # Create elimination context
        context = EliminationContext(
            reasoner=self.reasoner,
            observers=self.reasoner.observers,
            current_candidates=initial_candidates.copy(),
        )

        # Process each statement
        for stmt in statements:
            context.turn = stmt.turn
            context.statement_history.append(stmt)

            logger.debug(
                f"Processing statement from {stmt.speaker}: {stmt.statement_type.value}",
                extra={
                    "turn": stmt.turn,
                    "speaker": stmt.speaker,
                    "type": stmt.statement_type.value,
                },
            )

            # Find applicable rules
            applicable_rules = [
                rule
                for rule in self.rules
                if rule.statement_type == stmt.statement_type
            ]

            # Apply rules
            for rule in applicable_rules:
                new_candidates, explanation = rule.apply(
                    context.current_candidates, stmt, context
                )

                eliminated = [
                    obj
                    for obj in context.current_candidates
                    if obj not in new_candidates
                ]

                if eliminated:
                    # Update context
                    context.current_candidates = new_candidates

                    # Add to deductive chain
                    self.deductive_chain.add_step(stmt, eliminated, explanation)

                    logger.info(
                        f"Rule '{rule.name}' eliminated {len(eliminated)} objects",
                        extra={
                            "rule": rule.name,
                            "eliminated": len(eliminated),
                            "remaining": len(new_candidates),
                        },
                    )

            # Check if solved or contradictory
            if self.deductive_chain.is_solved():
                logger.info(
                    "Unique solution found!",
                    extra={"solution": self.deductive_chain.get_solution().object_id},
                )
                break

            if self.deductive_chain.is_contradictory():
                logger.warning("Contradiction detected - no candidates remain")
                break

        # Generate proof tree
        query = "Löse Puzzle mit {} Aussagen".format(len(statements))
        proof_tree = self.deductive_chain.generate_proof_tree(query)

        solution = self.deductive_chain.get_solution()

        logger.info(
            "Statement processing complete",
            extra={
                "solution_found": solution is not None,
                "final_candidates": len(self.deductive_chain.current_candidates),
                "steps": len(self.deductive_chain.steps),
            },
        )

        return solution, proof_tree


if __name__ == "__main__":
    print("\n=== Elimination Reasoning Test ===\n")

    from component_1_netzwerk import KonzeptNetzwerk

    # Setup
    netzwerk = KonzeptNetzwerk()
    engine = EpistemicEngine(netzwerk)
    reasoner = PartialObservationReasoner(engine)

    # Create simple test scenario: number pairs
    objects = [
        WorldObject("pair1", {"sum": 5, "product": 6}),  # 2+3
        WorldObject("pair2", {"sum": 5, "product": 4}),  # 1+4
        WorldObject("pair3", {"sum": 7, "product": 12}),  # 3+4
    ]

    reasoner.add_objects(objects)

    # Observers
    sum_obs = PartialObserver("sum_person", observable_properties=["sum"])
    prod_obs = PartialObserver("product_person", observable_properties=["product"])

    reasoner.add_observer(sum_obs)
    reasoner.add_observer(prod_obs)

    # Elimination reasoner
    elim_reasoner = EliminationReasoner(reasoner)
    elim_reasoner.create_standard_rules()

    # Test statements
    statements = [
        AgentStatement(
            speaker="product_person",
            statement_type=StatementType.I_KNOW_OTHER_DOESNT_KNOW,
            about_agent="sum_person",
            turn=1,
        )
    ]

    # Process
    solution, proof_tree = elim_reasoner.process_statements(objects, statements)

    print(f"Solution: {solution.object_id if solution else 'None'}")
    print(f"Proof tree steps: {len(proof_tree.root_steps)}")

    print("\n✓ Test complete!")
