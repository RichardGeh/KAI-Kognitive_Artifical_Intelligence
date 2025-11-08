"""
Component 29: Constraint Reasoning Engine

Implements Constraint Satisfaction Problem (CSP) solving with:
- Generic Variable/Constraint/Problem data structures
- Backtracking search with constraint propagation
- Arc Consistency (AC-3) algorithm
- Heuristics: MRV (Minimum Remaining Values), LCV (Least Constraining Value)
- Integration with ProofTree for solution visualization

Supports:
- N-Queens, Graph Coloring, Logic Grid Puzzles
- Binary and N-ary constraints
- Domain filtering and constraint propagation
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from component_15_logging_config import get_logger
from component_17_proof_explanation import ProofStep, ProofTree, StepType
from kai_exceptions import ConstraintReasoningError

logger = get_logger(__name__)


class ConstraintType(Enum):
    """Types of constraints"""

    UNARY = "unary"  # Single variable: var1 != 'red'
    BINARY = "binary"  # Two variables: var1 != var2
    N_ARY = "n_ary"  # N variables: all_different([var1, var2, var3])
    ALL_DIFFERENT = "all_different"  # All variables must have different values


@dataclass
class Variable:
    """
    Variable in a CSP with a domain of possible values.

    Attributes:
        name: Unique identifier for the variable
        domain: Set of possible values the variable can take
        value: Current assignment (None if unassigned)
        original_domain: Initial domain (for backtracking)
    """

    name: str
    domain: Set[Any]
    value: Optional[Any] = None
    original_domain: Set[Any] = field(default_factory=set)

    def __post_init__(self):
        """Store original domain for reset operations and ensure domain is a set"""
        # Convert domain to set if it's a list
        if isinstance(self.domain, list):
            self.domain = set(self.domain)

        if not self.original_domain:
            self.original_domain = self.domain.copy()

    def is_assigned(self) -> bool:
        """Check if variable has been assigned a value"""
        return self.value is not None

    def reset_domain(self):
        """Reset domain to original values"""
        self.domain = self.original_domain.copy()
        self.value = None

    def assign(self, value: Any) -> bool:
        """
        Assign a value if it's in the domain.

        Returns:
            True if assignment successful, False otherwise
        """
        if value not in self.domain:
            logger.warning(
                "Attempted to assign invalid value",
                extra={"variable": self.name, "value": value, "domain": self.domain},
            )
            return False
        self.value = value
        return True

    def unassign(self):
        """Remove current assignment"""
        self.value = None


class Constraint:
    """
    Constraint over one or more variables.

    Attributes:
        name: Human-readable constraint description
        scope: List of variable names involved in constraint
        predicate: Function that checks if assignment satisfies constraint
        constraint_type: UNARY, BINARY, or N_ARY
    """

    def __init__(
        self,
        name: Optional[str] = None,
        scope: Optional[List[str]] = None,
        predicate: Optional[Callable[[Dict[str, Any]], bool]] = None,
        # Backwards compatibility parameters
        variables: Optional[List[str]] = None,
        constraint_function: Optional[Callable[[Dict[str, Any]], bool]] = None,
        constraint_type: Optional[ConstraintType] = None,
    ):
        """
        Initialize constraint with flexible API.

        Can be called with either:
        - name, scope, predicate (preferred)
        - variables, constraint_function, constraint_type (legacy)
        """
        # Support legacy API
        if variables is not None:
            self.scope = variables
            self.name = name or f"constraint_{'_'.join(variables)}"
        else:
            self.scope = scope
            self.name = name

        if constraint_function is not None:
            self.predicate = constraint_function
        else:
            self.predicate = predicate

        # Determine constraint type from scope
        if constraint_type is not None:
            self.constraint_type = constraint_type
        elif len(self.scope) == 1:
            self.constraint_type = ConstraintType.UNARY
        elif len(self.scope) == 2:
            self.constraint_type = ConstraintType.BINARY
        else:
            self.constraint_type = ConstraintType.N_ARY

    def is_satisfied(self, assignment: Dict[str, Any]) -> bool:
        """
        Check if constraint is satisfied by given assignment.

        Returns:
            True if satisfied or not all variables assigned yet
        """
        # Check if all variables in scope are assigned
        if not all(var in assignment for var in self.scope):
            return True  # Can't check yet, assume satisfied

        # Build partial assignment for constraint
        relevant_assignment = {var: assignment[var] for var in self.scope}

        try:
            return self.predicate(relevant_assignment)
        except Exception as e:
            logger.error(
                "Constraint predicate failed",
                extra={
                    "constraint": self.name,
                    "scope": self.scope,
                    "assignment": relevant_assignment,
                    "error": str(e),
                },
            )
            return False

    def get_related_variables(self, var_name: str) -> List[str]:
        """
        Get other variables in constraint scope.

        Args:
            var_name: Variable to exclude

        Returns:
            List of other variable names in scope
        """
        return [v for v in self.scope if v != var_name]


class ConstraintProblem:
    """
    Constraint Satisfaction Problem definition.

    Attributes:
        name: Problem description
        variables: Dict mapping variable names to Variable objects
        constraints: List of constraints
    """

    def __init__(
        self,
        variables: Any,  # Can be Dict[str, Variable] or List[Variable]
        constraints: List[Constraint] = None,
        name: str = "",
    ):
        """
        Initialize constraint problem.

        Args:
            variables: Either Dict[str, Variable] or List[Variable]
            constraints: List of constraints
            name: Problem name
        """
        self.name = name or "CSP"
        self.constraints = constraints or []

        # Convert list of variables to dict if needed
        if isinstance(variables, list):
            self.variables = {var.name: var for var in variables}
        else:
            self.variables = variables

    def get_variable(self, name: str) -> Optional[Variable]:
        """Get variable by name"""
        return self.variables.get(name)

    def get_constraints_for_variable(self, var_name: str) -> List[Constraint]:
        """Get all constraints involving a variable"""
        return [c for c in self.constraints if var_name in c.scope]

    def get_binary_constraints(self, var1: str, var2: str) -> List[Constraint]:
        """Get all binary constraints between two variables"""
        return [
            c
            for c in self.constraints
            if c.constraint_type == ConstraintType.BINARY
            and var1 in c.scope
            and var2 in c.scope
        ]

    def get_neighbors(self, var_name: str) -> Set[str]:
        """Get all variables that share constraints with given variable"""
        neighbors = set()
        for constraint in self.get_constraints_for_variable(var_name):
            neighbors.update(constraint.get_related_variables(var_name))
        return neighbors

    def is_complete(self, assignment: Dict[str, Any]) -> bool:
        """Check if all variables are assigned"""
        return len(assignment) == len(self.variables)

    def is_consistent(self, assignment: Dict[str, Any]) -> bool:
        """Check if assignment satisfies all applicable constraints"""
        for constraint in self.constraints:
            if not constraint.is_satisfied(assignment):
                return False
        return True

    def reset(self):
        """Reset all variables to unassigned state"""
        for var in self.variables.values():
            var.reset_domain()


class ConstraintSolver:
    """
    Generic CSP solver using backtracking with constraint propagation.

    Features:
    - Backtracking search
    - Arc Consistency (AC-3) for constraint propagation
    - MRV (Minimum Remaining Values) variable selection
    - LCV (Least Constraining Value) value ordering
    - ProofTree generation for solution tracking
    """

    def __init__(
        self,
        problem: Optional[ConstraintProblem] = None,
        use_ac3: bool = True,
        use_mrv: bool = True,
        use_lcv: bool = True,
    ):
        """
        Initialize solver with optional problem and heuristics.

        Args:
            problem: ConstraintProblem to solve (optional, can be passed to solve() instead)
            use_ac3: Enable Arc Consistency preprocessing
            use_mrv: Enable Minimum Remaining Values heuristic
            use_lcv: Enable Least Constraining Value heuristic
        """
        self.problem = problem
        self.use_ac3 = use_ac3
        self.use_mrv = use_mrv
        self.use_lcv = use_lcv
        self.steps: List[ProofStep] = []  # For ProofTree generation
        self.backtrack_count = 0
        self.constraint_checks = 0
        self.step_counter = 0  # For unique step IDs

    def _create_proof_step(
        self,
        step_type: StepType,
        description: str,
        confidence: float = 1.0,
        output: str = "",
    ) -> ProofStep:
        """Helper to create ProofStep with correct fields"""
        self.step_counter += 1
        return ProofStep(
            step_id=f"csp_step_{self.step_counter}",
            step_type=step_type,
            explanation_text=description,
            confidence=confidence,
            output=output,
            source_component="component_29_constraint_reasoning",
        )

    def solve(
        self, problem: Optional[ConstraintProblem] = None, track_proof: bool = True
    ) -> Tuple[Optional[Dict[str, Any]], Optional[ProofTree]]:
        """
        Solve CSP using backtracking with constraint propagation.

        Args:
            problem: ConstraintProblem to solve (uses stored problem if None)
            track_proof: Whether to generate ProofTree

        Returns:
            Tuple of (solution_assignment, proof_tree)
            solution_assignment is None if no solution exists
        """
        # Use stored problem if not provided
        if problem is None:
            if self.problem is None:
                raise ValueError("No problem provided and no problem stored in solver")
            problem = self.problem

        logger.info(
            "Starting CSP solve",
            extra={
                "problem": problem.name,
                "variables": len(problem.variables),
                "constraints": len(problem.constraints),
                "use_ac3": self.use_ac3,
                "use_mrv": self.use_mrv,
                "use_lcv": self.use_lcv,
            },
        )

        self.steps = [] if track_proof else None
        self.backtrack_count = 0
        self.constraint_checks = 0

        # Initial constraint propagation
        if self.use_ac3:
            if track_proof:
                self.steps.append(
                    self._create_proof_step(
                        step_type=StepType.PREMISE,
                        description=f"Anwende AC-3 Constraint Propagation auf Problem '{problem.name}'",
                    )
                )

            if not self.ac3(problem):
                logger.warning("AC-3 detected inconsistency - no solution exists")
                if track_proof:
                    self.steps.append(
                        self._create_proof_step(
                            step_type=StepType.CONTRADICTION,
                            description="AC-3 hat Inkonsistenz erkannt - keine Lösung möglich",
                            confidence=0.0,
                        )
                    )
                    proof_tree = ProofTree(query=f"Löse CSP: {problem.name}")
                    for step in self.steps:
                        proof_tree.add_root_step(step)
                    return None, proof_tree
                return None, None

            if track_proof:
                self.steps.append(
                    self._create_proof_step(
                        step_type=StepType.INFERENCE,
                        description="AC-3 erfolgreich - Domains reduziert",
                    )
                )

        # Backtracking search
        assignment: Dict[str, Any] = {}
        solution = self._backtrack(assignment, problem)

        # Generate proof tree
        proof_tree = None
        if track_proof:
            if solution:
                self.steps.append(
                    self._create_proof_step(
                        step_type=StepType.CONCLUSION,
                        description=f"Lösung gefunden: {solution}",
                        output=str(solution),
                    )
                )
            else:
                self.steps.append(
                    self._create_proof_step(
                        step_type=StepType.CONTRADICTION,
                        description="Keine Lösung gefunden (Backtracking erschöpft)",
                        confidence=0.0,
                    )
                )

            proof_tree = ProofTree(query=f"Löse CSP: {problem.name}")
            for step in self.steps:
                proof_tree.add_root_step(step)

        logger.info(
            "CSP solve completed",
            extra={
                "problem": problem.name,
                "solution_found": solution is not None,
                "backtracks": self.backtrack_count,
                "constraint_checks": self.constraint_checks,
            },
        )

        return solution, proof_tree

    def _backtrack(
        self, assignment: Dict[str, Any], problem: ConstraintProblem
    ) -> Optional[Dict[str, Any]]:
        """
        Recursive backtracking search.

        Args:
            assignment: Current partial assignment
            problem: ConstraintProblem being solved

        Returns:
            Complete assignment if solution found, None otherwise
        """
        # Base case: complete and consistent assignment
        if problem.is_complete(assignment):
            if problem.is_consistent(assignment):
                return assignment
            return None

        # Select next variable
        var = self.select_variable(assignment, problem)

        if self.steps is not None:
            self.steps.append(
                self._create_proof_step(
                    step_type=StepType.ASSUMPTION,
                    description=f"Wähle Variable '{var.name}' (Domain: {var.domain})",
                )
            )

        # Try each value in variable's domain
        for value in self.order_values(var, assignment, problem):
            if self.steps is not None:
                self.steps.append(
                    self._create_proof_step(
                        step_type=StepType.ASSUMPTION,
                        description=f"Versuche '{var.name}' = {value}",
                    )
                )

            # Assign value
            var.assign(value)
            assignment[var.name] = value

            # Check consistency
            self.constraint_checks += 1
            if problem.is_consistent(assignment):
                if self.steps is not None:
                    self.steps.append(
                        self._create_proof_step(
                            step_type=StepType.INFERENCE,
                            description=f"'{var.name}' = {value} ist konsistent",
                        )
                    )

                # Recurse
                result = self._backtrack(assignment, problem)
                if result is not None:
                    return result

            # Backtrack
            self.backtrack_count += 1
            if self.steps is not None:
                self.steps.append(
                    self._create_proof_step(
                        step_type=StepType.CONTRADICTION,
                        description=f"'{var.name}' = {value} führt zu Inkonsistenz - Backtrack",
                        confidence=0.0,
                    )
                )

            var.unassign()
            del assignment[var.name]

        return None

    def ac3(self, problem: Optional[ConstraintProblem] = None) -> bool:
        """
        Arc Consistency Algorithm 3 (AC-3).

        Enforces arc consistency by removing values from variable domains
        that cannot participate in any solution.

        Args:
            problem: ConstraintProblem to make arc consistent (uses self.problem if not provided)

        Returns:
            True if problem is arc consistent, False if inconsistent
        """
        # Use self.problem if no problem is provided
        if problem is None:
            problem = self.problem

        if problem is None:
            raise ValueError("No problem provided and solver has no problem set")
        # Initialize queue with all arcs (var pairs in constraints)
        queue: List[Tuple[str, str]] = []

        for constraint in problem.constraints:
            if constraint.constraint_type == ConstraintType.BINARY:
                var1, var2 = constraint.scope
                queue.append((var1, var2))
                queue.append((var2, var1))

        logger.debug(
            "AC-3 starting", extra={"problem": problem.name, "arcs": len(queue)}
        )

        # Process arcs until queue is empty
        while queue:
            var1_name, var2_name = queue.pop(0)

            # Try to revise domain of var1 based on var2
            if self._revise(problem, var1_name, var2_name):
                var1 = problem.get_variable(var1_name)

                # If domain becomes empty, problem is inconsistent
                if not var1.domain:
                    logger.debug(
                        "AC-3 detected empty domain", extra={"variable": var1_name}
                    )
                    return False

                # Add all arcs (neighbor, var1) back to queue
                for neighbor in problem.get_neighbors(var1_name):
                    if neighbor != var2_name:
                        queue.append((neighbor, var1_name))

        logger.debug("AC-3 completed successfully")
        return True

    def _revise(
        self, problem: ConstraintProblem, var1_name: str, var2_name: str
    ) -> bool:
        """
        Revise domain of var1 to be arc consistent with var2.

        Args:
            problem: ConstraintProblem
            var1_name: Variable to revise
            var2_name: Variable to check against

        Returns:
            True if domain was revised (values removed), False otherwise
        """
        var1 = problem.get_variable(var1_name)
        var2 = problem.get_variable(var2_name)
        revised = False

        # Get binary constraints between var1 and var2
        constraints = problem.get_binary_constraints(var1_name, var2_name)

        # For each value in var1's domain
        values_to_remove = []
        for value1 in var1.domain:
            # Check if there exists a value in var2's domain that satisfies all constraints
            satisfiable = False
            for value2 in var2.domain:
                assignment = {var1_name: value1, var2_name: value2}

                # Check all binary constraints
                if all(c.is_satisfied(assignment) for c in constraints):
                    satisfiable = True
                    break

            # If no value in var2's domain satisfies constraints, remove value1
            if not satisfiable:
                values_to_remove.append(value1)
                revised = True

        # Remove unsatisfiable values
        for value in values_to_remove:
            var1.domain.discard(value)
            logger.debug(
                "AC-3 removed value",
                extra={
                    "variable": var1_name,
                    "value": value,
                    "reason": f"No compatible value in {var2_name}",
                },
            )

        return revised

    def select_variable(
        self, assignment: Dict[str, Any], problem: ConstraintProblem
    ) -> Variable:
        """
        Select next unassigned variable to assign.

        Uses MRV (Minimum Remaining Values) heuristic if enabled:
        Select variable with smallest domain (most constrained).

        Args:
            assignment: Current assignment
            problem: ConstraintProblem

        Returns:
            Next Variable to assign
        """
        unassigned = [
            var for var in problem.variables.values() if not var.is_assigned()
        ]

        if not unassigned:
            raise ConstraintReasoningError("No unassigned variables found")

        if self.use_mrv:
            # MRV: Select variable with smallest domain
            return min(unassigned, key=lambda v: len(v.domain))
        else:
            # Default: Select first unassigned variable
            return unassigned[0]

    def order_values(
        self, var: Variable, assignment: Dict[str, Any], problem: ConstraintProblem
    ) -> List[Any]:
        """
        Order values in variable's domain.

        Uses LCV (Least Constraining Value) heuristic if enabled:
        Order values by how many options they eliminate for neighbors.

        Args:
            var: Variable to order values for
            assignment: Current assignment
            problem: ConstraintProblem

        Returns:
            Ordered list of values to try
        """
        if self.use_lcv:
            # LCV: Order values by least constraining first
            def count_conflicts(value: Any) -> int:
                """Count how many neighbor values this assignment would eliminate"""
                conflicts = 0
                test_assignment = assignment.copy()
                test_assignment[var.name] = value

                # Check each neighbor
                for neighbor_name in problem.get_neighbors(var.name):
                    neighbor = problem.get_variable(neighbor_name)
                    if neighbor.is_assigned():
                        continue

                    # Count how many values in neighbor's domain would be eliminated
                    for neighbor_value in neighbor.domain:
                        test_assignment[neighbor_name] = neighbor_value
                        if not problem.is_consistent(test_assignment):
                            conflicts += 1
                        del test_assignment[neighbor_name]

                return conflicts

            # Sort by fewest conflicts (least constraining first)
            return sorted(var.domain, key=count_conflicts)
        else:
            # Default: Return domain as-is
            return list(var.domain)


# Helper functions for common constraint types


def all_different_constraint(variables: List[str]) -> Constraint:
    """
    Create constraint that all variables must have different values.

    Args:
        variables: List of variable names

    Returns:
        N-ary all-different constraint
    """

    def predicate(assignment: Dict[str, Any]) -> bool:
        values = [assignment[var] for var in variables if var in assignment]
        return len(values) == len(set(values))  # All unique

    return Constraint(
        name=f"AllDifferent({', '.join(variables)})",
        scope=variables,
        predicate=predicate,
    )


def not_equal_constraint(var1: str, var2: str) -> Constraint:
    """
    Create binary inequality constraint.

    Args:
        var1: First variable name
        var2: Second variable name

    Returns:
        Binary not-equal constraint
    """

    def predicate(assignment: Dict[str, Any]) -> bool:
        return assignment.get(var1) != assignment.get(var2)

    return Constraint(name=f"{var1} != {var2}", scope=[var1, var2], predicate=predicate)


def unary_constraint(var: str, allowed_values: Set[Any]) -> Constraint:
    """
    Create unary constraint that variable must be in allowed set.

    Args:
        var: Variable name
        allowed_values: Set of allowed values

    Returns:
        Unary constraint
    """

    def predicate(assignment: Dict[str, Any]) -> bool:
        return assignment.get(var) in allowed_values

    return Constraint(
        name=f"{var} in {allowed_values}", scope=[var], predicate=predicate
    )


def custom_constraint(
    name: str, scope: List[str], predicate: Callable[[Dict[str, Any]], bool]
) -> Constraint:
    """
    Create custom constraint with arbitrary predicate.

    Args:
        name: Constraint description
        scope: Variable names in scope
        predicate: Function that checks constraint satisfaction

    Returns:
        Custom constraint
    """
    return Constraint(name=name, scope=scope, predicate=predicate)
