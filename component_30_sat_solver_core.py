"""
component_30_sat_solver_core.py

Core SAT Solver Implementation - DPLL Algorithm and Data Structures

This module provides the foundational SAT solving infrastructure:
- Core data structures (Literal, Clause, CNFFormula)
- DPLL algorithm with unit propagation and pure literal elimination
- Basic satisfiability checking
- Integration with ProofTree for explanations

SPLIT FROM: component_30_sat_solver.py (Task 12 - Phase 4, 2025-11-28)

Author: KAI Development Team
Date: 2025-10-30 | Refactored: 2025-11-28
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from component_15_logging_config import get_logger

# Integration mit Proof System
try:
    from component_17_proof_explanation import ProofStep, ProofTree, StepType

    PROOF_AVAILABLE = True
except ImportError:
    PROOF_AVAILABLE = False
    logging.warning(
        "component_17 not available, using stub classes for proof generation"
    )

    # Provide complete stub classes to prevent runtime errors
    class StepType:
        """Stub for StepType enum when component_17 not available."""

        PREMISE = "PREMISE"
        INFERENCE = "INFERENCE"
        CONCLUSION = "CONCLUSION"
        ASSUMPTION = "ASSUMPTION"
        CONTRADICTION = "CONTRADICTION"
        RULE_APPLICATION = "RULE_APPLICATION"

    class ProofStep:
        """Stub for ProofStep when component_17 not available."""

        def __init__(self, *args, **kwargs):
            pass

    class ProofTree:
        """Stub for ProofTree when component_17 not available."""

        def __init__(self, *args, **kwargs):
            pass

        def add_root_step(self, step):
            pass


logger = get_logger(__name__)


# ============================================================================
# Data Structures
# ============================================================================


class SATResult(Enum):
    """Result of SAT solving."""

    SATISFIABLE = "satisfiable"
    UNSATISFIABLE = "unsatisfiable"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Literal:
    """
    A propositional literal (variable or its negation).

    Immutable for use in sets/dicts.
    """

    variable: str
    negated: bool = False

    def __neg__(self):
        """Return the negation of this literal."""
        return Literal(self.variable, not self.negated)

    def __str__(self):
        return f"{'NOT ' if self.negated else ''}{self.variable}"

    def __repr__(self):
        return str(self)

    def to_dimacs(self, var_mapping: Dict[str, int]) -> int:
        """Convert to DIMACS format (positive/negative integer)."""
        var_id = var_mapping[self.variable]
        return -var_id if self.negated else var_id


@dataclass
class Clause:
    """
    A disjunction of literals (OR-connected).

    Empty clause represents FALSE (unsatisfiable).
    """

    literals: FrozenSet[Literal]

    def __init__(self, literals: Set[Literal] | FrozenSet[Literal]):
        if isinstance(literals, frozenset):
            object.__setattr__(self, "literals", literals)
        else:
            object.__setattr__(self, "literals", frozenset(literals))

    def __hash__(self):
        return hash(self.literals)

    def __eq__(self, other):
        return isinstance(other, Clause) and self.literals == other.literals

    def is_empty(self) -> bool:
        """Check if clause is empty (represents FALSE)."""
        return len(self.literals) == 0

    def is_unit(self) -> bool:
        """Check if clause has exactly one literal (unit clause)."""
        return len(self.literals) == 1

    def get_unit_literal(self) -> Optional[Literal]:
        """Get the single literal if this is a unit clause."""
        if self.is_unit():
            return next(iter(self.literals))
        return None

    def simplify(self, assignment: Dict[str, bool]) -> Optional["Clause"]:
        """
        Simplify clause given partial assignment.

        Returns:
            None if clause is satisfied
            New simplified clause otherwise
        """
        new_literals = set()

        for lit in self.literals:
            if lit.variable in assignment:
                # Check if literal is satisfied
                value = assignment[lit.variable]
                if (value and not lit.negated) or (not value and lit.negated):
                    # Clause is satisfied
                    return None
                # Otherwise, literal is false, skip it
            else:
                # Variable not assigned, keep literal
                new_literals.add(lit)

        return Clause(new_literals)

    def __str__(self):
        if self.is_empty():
            return "[FALSE]"  # Empty clause (FALSE)
        return " OR ".join(str(lit) for lit in sorted(self.literals, key=str))

    def __repr__(self):
        return f"Clause({self.literals})"


@dataclass
class CNFFormula:
    """
    A formula in Conjunctive Normal Form (AND of ORs).

    Represents: C1 AND C2 AND ... AND Cn where each Ci is a clause.
    """

    clauses: List[Clause]
    variables: Set[str] = field(default_factory=set)

    def __post_init__(self):
        """Extract all variables from clauses."""
        if not self.variables:
            for clause in self.clauses:
                for lit in clause.literals:
                    self.variables.add(lit.variable)

    def add_clause(self, clause: Clause):
        """Add a clause to the formula."""
        self.clauses.append(clause)
        for lit in clause.literals:
            self.variables.add(lit.variable)

    def is_empty(self) -> bool:
        """Check if formula has no clauses (represents TRUE)."""
        return len(self.clauses) == 0

    def has_empty_clause(self) -> bool:
        """Check if formula contains an empty clause (unsatisfiable)."""
        return any(c.is_empty() for c in self.clauses)

    def get_unit_clauses(self) -> List[Literal]:
        """Get all unit clause literals."""
        return [
            lit
            for c in self.clauses
            if c.is_unit() and (lit := c.get_unit_literal()) is not None
        ]

    def get_pure_literals(self, assignment: Dict[str, bool]) -> List[Literal]:
        """
        Find pure literals (variables that appear only positively or only negatively).

        Args:
            assignment: Current partial assignment (to skip assigned variables)
        """
        literal_polarities: Dict[str, Set[bool]] = (
            {}
        )  # variable -> Set[bool] (negated values seen)

        for clause in self.clauses:
            for lit in clause.literals:
                if lit.variable not in assignment:
                    if lit.variable not in literal_polarities:
                        literal_polarities[lit.variable] = set()
                    literal_polarities[lit.variable].add(lit.negated)

        # Pure literal has only one polarity
        pure_literals = []
        for var, polarities in literal_polarities.items():
            if len(polarities) == 1:
                negated = next(iter(polarities))
                pure_literals.append(Literal(var, negated))

        return pure_literals

    def simplify(self, assignment: Dict[str, bool]) -> "CNFFormula":
        """
        Simplify formula given partial assignment.

        Removes satisfied clauses and false literals.
        """
        new_clauses = []
        for clause in self.clauses:
            simplified = clause.simplify(assignment)
            if simplified is not None:  # Clause not satisfied
                new_clauses.append(simplified)

        return CNFFormula(new_clauses, self.variables.copy())

    def __str__(self):
        if self.is_empty():
            return "[TRUE]"  # Empty formula (TRUE)
        return " AND ".join(f"({c})" for c in self.clauses)


# ============================================================================
# DPLL Solver
# ============================================================================


class DPLLSolver:
    """
    DPLL-based SAT solver with unit propagation and pure literal elimination.

    Implements:
    - Unit propagation
    - Pure literal elimination
    - Backtracking search
    - ProofTree generation for explanations

    Thread Safety:
        This class is NOT thread-safe. Create separate instances for concurrent use.
    """

    def __init__(self, enable_proof: bool = False):
        """
        Initialize DPLL solver.

        Args:
            enable_proof: Whether to generate proof trees (requires component_17)
        """
        self.decision_level = 0
        self.propagation_count = 0
        self.conflict_count = 0
        self.enable_proof = enable_proof and PROOF_AVAILABLE
        self.proof_steps: List = []
        self.decision_stack: List[Tuple[str, bool]] = []
        # Track current parent for hierarchical proof construction
        self._current_parent: Optional[ProofStep] = None
        self._root_step: Optional[ProofStep] = None

    def solve(
        self, formula: CNFFormula, initial_assignment: Optional[Dict[str, bool]] = None
    ) -> Tuple[SATResult, Optional[Dict[str, bool]]]:
        """
        Solve SAT problem using DPLL algorithm.

        Args:
            formula: CNF formula to solve
            initial_assignment: Optional partial assignment to start with

        Returns:
            (result, model) where model is None if UNSATISFIABLE
        """
        logger.info(
            "Starting DPLL solver",
            extra={
                "num_clauses": len(formula.clauses),
                "num_variables": len(formula.variables),
            },
        )

        self.propagation_count = 0
        self.conflict_count = 0
        self.decision_level = 0
        self.proof_steps = []  # Reset proof steps
        self.decision_stack = []
        self._current_parent = None
        self._root_step = None

        assignment = initial_assignment.copy() if initial_assignment else {}

        if self.enable_proof and PROOF_AVAILABLE:
            # Create root step for entire SAT solving process
            self._root_step = self._create_proof_step(
                StepType.PREMISE,
                "SAT Solving",
                f"Find satisfying assignment for {len(formula.clauses)} clauses, "
                f"{len(formula.variables)} variables",
                confidence=1.0,
            )
            # Set root as current parent for all subsequent steps
            self._current_parent = self._root_step

        result, model = self._dpll(formula, assignment)

        if self.enable_proof and PROOF_AVAILABLE and self._root_step:
            # Add conclusion as child of root step
            conclusion_step = self._create_proof_step(
                StepType.CONCLUSION,
                "SAT" if result == SATResult.SATISFIABLE else "UNSAT",
                (
                    f"Found satisfying assignment: {model}"
                    if result == SATResult.SATISFIABLE
                    else "No satisfying assignment exists"
                ),
                confidence=1.0,
            )
            self._root_step.add_subgoal(conclusion_step)

        logger.info(
            "DPLL solver finished",
            extra={
                "result": result.value,
                "propagations": self.propagation_count,
                "conflicts": self.conflict_count,
            },
        )

        return result, model

    def _dpll(
        self, formula: CNFFormula, assignment: Dict[str, bool]
    ) -> Tuple[SATResult, Optional[Dict[str, bool]]]:
        """
        Recursive DPLL algorithm.

        Returns:
            (result, model)
        """
        # Simplify formula with current assignment
        simplified = formula.simplify(assignment)

        # Base cases
        if simplified.is_empty():
            # All clauses satisfied
            return SATResult.SATISFIABLE, assignment.copy()

        if simplified.has_empty_clause():
            # Conflict: empty clause means unsatisfiable
            self.conflict_count += 1
            return SATResult.UNSATISFIABLE, None

        # Unit propagation
        unit_literals = simplified.get_unit_clauses()
        if unit_literals:
            # Create parent step for unit propagation phase
            unit_prop_parent = None
            saved_parent = self._current_parent

            if self.enable_proof and PROOF_AVAILABLE:
                unit_prop_parent = self._add_proof_step(
                    StepType.INFERENCE,
                    "Unit Propagation Phase",
                    f"Apply unit propagation to {len(unit_literals)} unit clauses",
                    confidence=1.0,
                )
                if unit_prop_parent:
                    self._current_parent = unit_prop_parent

            new_assignment = assignment.copy()
            for lit in unit_literals:
                # Assign value to satisfy unit literal
                value = not lit.negated
                if lit.variable in new_assignment:
                    # Check consistency
                    if new_assignment[lit.variable] != value:
                        self.conflict_count += 1
                        if self.enable_proof and PROOF_AVAILABLE:
                            self._add_proof_step(
                                StepType.CONTRADICTION,
                                "Conflict in Unit Propagation",
                                f"Variable {lit.variable} already assigned to {new_assignment[lit.variable]}, "
                                f"but unit clause requires {value}",
                                confidence=1.0,
                            )
                        # Restore parent
                        self._current_parent = saved_parent
                        return SATResult.UNSATISFIABLE, None
                else:
                    new_assignment[lit.variable] = value
                    self.propagation_count += 1
                    if self.enable_proof and PROOF_AVAILABLE:
                        self._add_proof_step(
                            StepType.INFERENCE,
                            "Unit Clause Assignment",
                            f"Forced assignment: {lit.variable} = {value} (from unit clause)",
                            confidence=1.0,
                        )

            # Restore parent before recursion
            if self.enable_proof and PROOF_AVAILABLE:
                self._current_parent = saved_parent

            # Recurse with propagated assignments
            return self._dpll(simplified, new_assignment)

        # Pure literal elimination
        pure_literals = simplified.get_pure_literals(assignment)
        if pure_literals:
            # Create parent step for pure literal elimination phase
            pure_lit_parent = None
            saved_parent = self._current_parent

            if self.enable_proof and PROOF_AVAILABLE:
                pure_lit_parent = self._add_proof_step(
                    StepType.INFERENCE,
                    "Pure Literal Elimination Phase",
                    f"Apply pure literal elimination to {len(pure_literals)} pure literals",
                    confidence=1.0,
                )
                if pure_lit_parent:
                    self._current_parent = pure_lit_parent

            new_assignment = assignment.copy()
            for lit in pure_literals:
                # Assign value to satisfy pure literal
                value = not lit.negated
                new_assignment[lit.variable] = value
                self.propagation_count += 1
                if self.enable_proof and PROOF_AVAILABLE:
                    self._add_proof_step(
                        StepType.INFERENCE,
                        "Pure Literal Assignment",
                        f"Pure literal: {lit.variable} = {value} (appears only with one polarity)",
                        confidence=1.0,
                    )

            # Restore parent before recursion
            if self.enable_proof and PROOF_AVAILABLE:
                self._current_parent = saved_parent

            # Recurse with pure literal assignments
            return self._dpll(simplified, new_assignment)

        # Choose a variable to branch on (decision heuristic)
        var = self._choose_variable(simplified, assignment)
        if var is None:
            # All variables assigned (shouldn't happen due to earlier checks)
            return SATResult.SATISFIABLE, assignment.copy()

        self.decision_level += 1

        # Create parent step for branching
        branch_parent = None
        saved_parent = self._current_parent

        if self.enable_proof and PROOF_AVAILABLE:
            branch_parent = self._add_proof_step(
                StepType.ASSUMPTION,
                "Branching Decision",
                f"Branch on {var} at decision level {self.decision_level}",
                confidence=0.5,
            )
            if branch_parent:
                self._current_parent = branch_parent

        # Try assigning True first
        true_branch_step = None
        if self.enable_proof and PROOF_AVAILABLE:
            true_branch_step = self._add_proof_step(
                StepType.ASSUMPTION,
                "Try True Branch",
                f"Assume {var} = True",
                confidence=0.5,
            )
            if true_branch_step:
                self._current_parent = true_branch_step

        self.decision_stack.append((var, True))
        new_assignment_true = assignment.copy()
        new_assignment_true[var] = True
        result, model = self._dpll(simplified, new_assignment_true)

        # Restore parent after True branch
        if self.enable_proof and PROOF_AVAILABLE:
            self._current_parent = branch_parent if branch_parent else saved_parent

        if result == SATResult.SATISFIABLE:
            self.decision_stack.pop()
            self.decision_level -= 1
            # Restore original parent
            if self.enable_proof and PROOF_AVAILABLE:
                self._current_parent = saved_parent
            return result, model

        # Backtrack: try assigning False
        false_branch_step = None
        if self.enable_proof and PROOF_AVAILABLE:
            false_branch_step = self._add_proof_step(
                StepType.ASSUMPTION,
                "Try False Branch (Backtrack)",
                f"Assume {var} = False (after True branch failed)",
                confidence=0.5,
            )
            if false_branch_step:
                self._current_parent = false_branch_step

        self.decision_stack[-1] = (var, False)
        new_assignment_false = assignment.copy()
        new_assignment_false[var] = False
        result, model = self._dpll(simplified, new_assignment_false)

        self.decision_stack.pop()
        self.decision_level -= 1

        # Restore original parent
        if self.enable_proof and PROOF_AVAILABLE:
            self._current_parent = saved_parent

        return result, model

    def _choose_variable(
        self, formula: CNFFormula, assignment: Dict[str, bool]
    ) -> Optional[str]:
        """
        Choose next variable to branch on.

        Heuristic: Choose variable with most occurrences in smallest clauses
        (similar to VSIDS but simpler).
        """
        unassigned = formula.variables - set(assignment.keys())
        if not unassigned:
            return None

        # Count occurrences in small clauses (weighted by 1/clause_size)
        scores = {var: 0.0 for var in unassigned}

        for clause in formula.clauses:
            clause_size = len(clause.literals)
            if clause_size == 0:
                continue

            weight = 1.0 / clause_size
            for lit in clause.literals:
                if lit.variable in unassigned:
                    scores[lit.variable] += weight

        # Return variable with highest score
        return max(unassigned, key=lambda v: scores[v])

    def _create_proof_step(
        self, step_type: "StepType", description: str, details: str, confidence: float
    ) -> Optional[ProofStep]:
        """
        Create a ProofStep without adding to parent yet.

        Returns:
            ProofStep or None if proof disabled
        """
        if not self.enable_proof or not PROOF_AVAILABLE:
            return None

        try:
            step_id = f"sat_step_{len(self.proof_steps)}"
            step = ProofStep(
                step_id=step_id,
                step_type=step_type,
                rule_name=description,
                explanation_text=details,
                confidence=confidence,
                source_component="component_30_sat_solver",
            )
            self.proof_steps.append(step)
            return step
        except Exception as e:
            logger.warning(f"Failed to create proof step: {e}")
            return None

    def _add_proof_step(
        self, step_type: "StepType", description: str, details: str, confidence: float
    ) -> Optional[ProofStep]:
        """
        Add proof step as child of current parent (hierarchical).

        Returns:
            Created ProofStep or None if proof disabled
        """
        if not self.enable_proof or not PROOF_AVAILABLE:
            return None

        step = self._create_proof_step(step_type, description, details, confidence)
        if step and self._current_parent:
            self._current_parent.add_subgoal(step)

        return step

    def get_proof_tree(self, query: str = "SAT Solution") -> Optional["ProofTree"]:
        """
        Create proof tree from collected steps with hierarchical structure.

        Args:
            query: Query string for the proof tree

        Returns:
            ProofTree: Hierarchical proof tree with root_step containing all nested steps
            None: If proof disabled or component_17 not available
        """
        if not self.enable_proof or not PROOF_AVAILABLE:
            return None

        try:
            # Use root_step if available (hierarchical), otherwise fall back to flat list
            if self._root_step:
                root_steps = [self._root_step]
                logger.info(
                    f"Creating hierarchical ProofTree: 1 root, "
                    f"{len(self.proof_steps)} total steps"
                )
            else:
                # Fallback: flat structure (shouldn't happen with new code)
                root_steps = self.proof_steps
                if len(root_steps) > 10:
                    logger.warning(
                        f"Creating flat ProofTree with {len(root_steps)} roots "
                        f"(expected hierarchical structure)"
                    )

            return ProofTree(query=query, root_steps=root_steps)
        except Exception as e:
            logger.warning(f"Failed to create proof tree: {e}")
            return None


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    "SATResult",
    "Literal",
    "Clause",
    "CNFFormula",
    "DPLLSolver",
    "PROOF_AVAILABLE",
    "StepType",
    "ProofStep",
    "ProofTree",
]
