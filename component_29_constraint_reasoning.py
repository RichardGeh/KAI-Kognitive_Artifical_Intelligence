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
from infrastructure.interfaces import BaseReasoningEngine, ReasoningResult
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
            # Re-raise with context instead of silently returning False
            # This prevents masking predicate errors as constraint violations
            raise ConstraintReasoningError(
                f"Constraint '{self.name}' predicate raised exception: {e}",
                context={
                    "constraint": self.name,
                    "scope": self.scope,
                    "assignment": relevant_assignment,
                    "original_error": str(e),
                },
            ) from e

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


class ConstraintSolver(BaseReasoningEngine):
    """
    Generic CSP solver using backtracking with constraint propagation.

    Features:
    - Backtracking search
    - Arc Consistency (AC-3) for constraint propagation
    - MRV (Minimum Remaining Values) variable selection
    - LCV (Least Constraining Value) value ordering
    - ProofTree generation for solution tracking

    Implements BaseReasoningEngine for integration with reasoning orchestrator.
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

    # ==================== BASE REASONING ENGINE INTERFACE ====================

    def reason(self, query: str, context: Dict[str, Any]) -> ReasoningResult:
        """
        Solve a constraint satisfaction problem.

        Args:
            query: Problem description
            context: Context with 'problem' (ConstraintProblem) or problem parameters

        Returns:
            ReasoningResult with solution
        """
        problem = context.get("problem")
        if not problem and self.problem:
            problem = self.problem

        if not problem:
            return ReasoningResult(
                success=False,
                answer="No constraint problem provided",
                confidence=0.0,
                strategy_used="constraint_satisfaction",
            )

        # Solve the problem
        solution, proof_tree = self.solve(problem, track_proof=True)

        if solution:
            answer = f"Solution found: {solution}"
            return ReasoningResult(
                success=True,
                answer=answer,
                confidence=1.0,  # CSP solutions are definitive
                proof_tree=proof_tree,
                strategy_used="constraint_satisfaction_backtracking",
                metadata={
                    "solution": solution,
                    "backtracks": self.backtrack_count,
                    "constraint_checks": self.constraint_checks,
                },
            )
        else:
            return ReasoningResult(
                success=False,
                answer="No solution exists for this constraint problem",
                confidence=0.0,
                proof_tree=proof_tree,
                strategy_used="constraint_satisfaction",
            )

    def get_capabilities(self) -> List[str]:
        """Return list of reasoning capabilities."""
        capabilities = ["constraint_satisfaction", "csp_solving", "backtracking"]
        if self.use_ac3:
            capabilities.append("arc_consistency")
        if self.use_mrv:
            capabilities.append("mrv_heuristic")
        if self.use_lcv:
            capabilities.append("lcv_heuristic")
        return capabilities

    def estimate_cost(self, query: str) -> float:
        """
        Estimate computational cost for CSP solving.

        CSP can be very expensive depending on problem size.

        Returns:
            Cost estimate in [0.0, 1.0] range
        """
        # CSP solving is expensive
        base_cost = 0.8

        # Query complexity
        query_complexity = min(len(query) / 200.0, 0.15)

        total_cost = base_cost + query_complexity

        return min(total_cost, 1.0)


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


# =============================================================================
# Constraint Translation (LogicalConstraint -> CSP)
# =============================================================================


def translate_logical_constraints_to_csp(constraint_problem_obj):
    """
    Uebersetzt ein LogicalConstraint-Problem (component_60) in ein CSP-Problem.

    Diese Funktion dient als Bruecke zwischen dem generischen Constraint-Detector
    (component_60_constraint_detector) und dem CSP-Solver (component_29).

    Args:
        constraint_problem_obj: ConstraintProblem aus component_60 mit:
            - variables: Dict[str, LogicalVariable]
            - constraints: List[LogicalConstraint]
            - context: str (original text)

    Returns:
        ConstraintProblem (CSP) mit Variables und Constraints fuer den Solver

    Beispiel:
        >>> from component_60_constraint_detector import ConstraintDetector
        >>> detector = ConstraintDetector()
        >>> text = "Wenn Leo Brandy bestellt, bestellt Mark auch einen. Mark oder Nick, aber nie beide."
        >>> problem = detector.detect_constraint_problem(text)
        >>> csp_problem = translate_logical_constraints_to_csp(problem)
        >>> solver = CSPSolver()
        >>> solution = solver.solve(csp_problem)
    """
    # Erstelle CSP Variables aus LogicalVariables
    csp_variables = {}
    for var_name, logical_var in constraint_problem_obj.variables.items():
        # Konvertiere domain zu set (falls noch nicht)
        domain = (
            logical_var.domain
            if isinstance(logical_var.domain, set)
            else set(logical_var.domain)
        )

        # Falls domain leer ist, nutze default boolean domain
        if not domain:
            domain = {"true", "false", "1", "0"}

        csp_variables[var_name] = Variable(
            name=var_name, domain=domain, value=logical_var.value
        )

    # Uebersetze LogicalConstraints zu CSP Constraints
    csp_constraints = []

    for logical_constraint in constraint_problem_obj.constraints:
        constraint_type = logical_constraint.constraint_type
        logical_constraint.variables
        conditions = logical_constraint.conditions

        if constraint_type == "IMPLIES":
            # A IMPLIES B: Wenn A dann B
            # Logik: NOT A OR B
            premise = conditions.get("premise", "")
            conclusion = conditions.get("conclusion", "")

            # Extrahiere Variablen-Namen aus premise/conclusion
            # (vereinfacht: nutze erste gefundene Variable)
            var_a = _extract_variable_from_text(
                premise, constraint_problem_obj.variables
            )
            var_b = _extract_variable_from_text(
                conclusion, constraint_problem_obj.variables
            )

            if var_a and var_b:

                def implies_predicate(assignment, a=var_a, b=var_b):
                    val_a = assignment.get(a)
                    val_b = assignment.get(b)
                    # A -> B entspricht: NOT A OR B
                    # Wenn A aktiv ist, muss B auch aktiv sein
                    return _is_false(val_a) or _is_true(val_b)

                csp_constraints.append(
                    Constraint(
                        name=f"{var_a} IMPLIES {var_b}",
                        scope=[var_a, var_b],
                        predicate=implies_predicate,
                    )
                )

        elif constraint_type == "XOR":
            # A XOR B: Entweder A oder B, aber nicht beide
            a = conditions.get("a", "")
            b = conditions.get("b", "")

            var_a = _extract_variable_from_text(a, constraint_problem_obj.variables)
            var_b = _extract_variable_from_text(b, constraint_problem_obj.variables)

            if var_a and var_b:

                def xor_predicate(assignment, a=var_a, b=var_b):
                    val_a = assignment.get(a)
                    val_b = assignment.get(b)
                    # XOR: (A AND NOT B) OR (NOT A AND B)
                    a_true = _is_true(val_a)
                    b_true = _is_true(val_b)
                    return (a_true and not b_true) or (not a_true and b_true)

                csp_constraints.append(
                    Constraint(
                        name=f"{var_a} XOR {var_b}",
                        scope=[var_a, var_b],
                        predicate=xor_predicate,
                    )
                )

        elif constraint_type == "AND":
            # A AND B: Beide muessen wahr sein
            a = conditions.get("a", "")
            b = conditions.get("b", "")

            var_a = _extract_variable_from_text(a, constraint_problem_obj.variables)
            var_b = _extract_variable_from_text(b, constraint_problem_obj.variables)

            if var_a and var_b:

                def and_predicate(assignment, a=var_a, b=var_b):
                    # A AND B: Beide muessen wahr sein
                    return _is_true(assignment.get(a)) and _is_true(assignment.get(b))

                csp_constraints.append(
                    Constraint(
                        name=f"{var_a} AND {var_b}",
                        scope=[var_a, var_b],
                        predicate=and_predicate,
                    )
                )

        elif constraint_type == "OR":
            # A OR B: Mindestens einer muss wahr sein
            a = conditions.get("a", "")
            b = conditions.get("b", "")

            var_a = _extract_variable_from_text(a, constraint_problem_obj.variables)
            var_b = _extract_variable_from_text(b, constraint_problem_obj.variables)

            if var_a and var_b:

                def or_predicate(assignment, a=var_a, b=var_b):
                    # A OR B: Mindestens einer muss wahr sein
                    return _is_true(assignment.get(a)) or _is_true(assignment.get(b))

                csp_constraints.append(
                    Constraint(
                        name=f"{var_a} OR {var_b}",
                        scope=[var_a, var_b],
                        predicate=or_predicate,
                    )
                )

        elif constraint_type == "NOT":
            # NOT A: A muss falsch sein
            negated = conditions.get("negated", "")
            var = _extract_variable_from_text(negated, constraint_problem_obj.variables)

            if var:

                def not_predicate(assignment, v=var):
                    return _is_false(assignment.get(v))

                csp_constraints.append(
                    Constraint(name=f"NOT {var}", scope=[var], predicate=not_predicate)
                )

    # Erstelle CSP-Problem
    csp_problem = ConstraintProblem(
        variables=csp_variables,
        constraints=csp_constraints,
        name=f"LogicPuzzle (confidence={constraint_problem_obj.confidence:.2f})",
    )

    logger.info(
        f"[CSP-Translation] | "
        f"variables={len(csp_variables)}, constraints={len(csp_constraints)}"
    )

    return csp_problem


def _extract_variable_from_text(text: str, variables: Dict) -> Optional[str]:
    """
    Extrahiert Variablen-Namen aus Text-Fragment.

    Args:
        text: Text-Fragment (z.B. "Leo bestellt Brandy")
        variables: Dict von LogicalVariable objekten

    Returns:
        Variablen-Name wenn gefunden, sonst None
    """
    text_lower = text.lower()
    for var_name in variables.keys():
        if var_name.lower() in text_lower:
            return var_name
    return None


def _is_true(value) -> bool:
    """Prueft ob Wert als 'true' interpretiert werden soll."""
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"true", "1", "aktiv", "an", "oben", "ja", "yes"}
    return bool(value)


def _is_false(value) -> bool:
    """Prueft ob Wert als 'false' interpretiert werden soll."""
    if value is None:
        return True  # Unassigned gilt als false
    return not _is_true(value)


# =============================================================================
# Position-based Constraint Helpers for Zebra Puzzles
# =============================================================================


def directly_left_of_constraint(pos_var_a: str, pos_var_b: str) -> Constraint:
    """
    Create constraint: pos(A) + 1 == pos(B)

    Entity A is directly to the left of entity B (adjacent, A before B).

    Args:
        pos_var_a: Position variable name for entity A
        pos_var_b: Position variable name for entity B

    Returns:
        Binary constraint enforcing A is directly left of B
    """

    def predicate(assignment: Dict[str, Any]) -> bool:
        pos_a = assignment.get(pos_var_a)
        pos_b = assignment.get(pos_var_b)
        if pos_a is None or pos_b is None:
            return True  # Not yet assigned
        return pos_a + 1 == pos_b

    return Constraint(
        name=f"{pos_var_a} directly_left_of {pos_var_b}",
        scope=[pos_var_a, pos_var_b],
        predicate=predicate,
    )


def directly_right_of_constraint(pos_var_a: str, pos_var_b: str) -> Constraint:
    """
    Create constraint: pos(A) - 1 == pos(B) (i.e., pos(A) == pos(B) + 1)

    Entity A is directly to the right of entity B (adjacent, A after B).

    Args:
        pos_var_a: Position variable name for entity A
        pos_var_b: Position variable name for entity B

    Returns:
        Binary constraint enforcing A is directly right of B
    """

    def predicate(assignment: Dict[str, Any]) -> bool:
        pos_a = assignment.get(pos_var_a)
        pos_b = assignment.get(pos_var_b)
        if pos_a is None or pos_b is None:
            return True  # Not yet assigned
        return pos_a == pos_b + 1

    return Constraint(
        name=f"{pos_var_a} directly_right_of {pos_var_b}",
        scope=[pos_var_a, pos_var_b],
        predicate=predicate,
    )


def adjacent_to_constraint(pos_var_a: str, pos_var_b: str) -> Constraint:
    """
    Create constraint: |pos(A) - pos(B)| == 1

    Entity A is adjacent to entity B (either left or right neighbor).

    Args:
        pos_var_a: Position variable name for entity A
        pos_var_b: Position variable name for entity B

    Returns:
        Binary constraint enforcing A and B are neighbors
    """

    def predicate(assignment: Dict[str, Any]) -> bool:
        pos_a = assignment.get(pos_var_a)
        pos_b = assignment.get(pos_var_b)
        if pos_a is None or pos_b is None:
            return True  # Not yet assigned
        return abs(pos_a - pos_b) == 1

    return Constraint(
        name=f"{pos_var_a} adjacent_to {pos_var_b}",
        scope=[pos_var_a, pos_var_b],
        predicate=predicate,
    )


def at_position_constraint(pos_var: str, position: int) -> Constraint:
    """
    Create constraint: pos(A) == N

    Entity A is at a specific position.

    Args:
        pos_var: Position variable name
        position: The required position (1-indexed typically)

    Returns:
        Unary constraint enforcing specific position
    """

    def predicate(assignment: Dict[str, Any]) -> bool:
        pos = assignment.get(pos_var)
        if pos is None:
            return True  # Not yet assigned
        return pos == position

    return Constraint(
        name=f"{pos_var} at_position {position}",
        scope=[pos_var],
        predicate=predicate,
    )


def same_position_constraint(pos_var_a: str, pos_var_b: str) -> Constraint:
    """
    Create constraint: pos(A) == pos(B)

    Two attributes share the same position (belong to same entity in Zebra puzzle).

    Args:
        pos_var_a: Position variable for attribute A
        pos_var_b: Position variable for attribute B

    Returns:
        Binary constraint enforcing same position
    """

    def predicate(assignment: Dict[str, Any]) -> bool:
        pos_a = assignment.get(pos_var_a)
        pos_b = assignment.get(pos_var_b)
        if pos_a is None or pos_b is None:
            return True  # Not yet assigned
        return pos_a == pos_b

    return Constraint(
        name=f"{pos_var_a} same_position_as {pos_var_b}",
        scope=[pos_var_a, pos_var_b],
        predicate=predicate,
    )


def left_of_constraint(pos_var_a: str, pos_var_b: str) -> Constraint:
    """
    Create constraint: pos(A) < pos(B)

    Entity A is somewhere to the left of entity B (not necessarily adjacent).

    Args:
        pos_var_a: Position variable name for entity A
        pos_var_b: Position variable name for entity B

    Returns:
        Binary constraint enforcing A is left of B
    """

    def predicate(assignment: Dict[str, Any]) -> bool:
        pos_a = assignment.get(pos_var_a)
        pos_b = assignment.get(pos_var_b)
        if pos_a is None or pos_b is None:
            return True  # Not yet assigned
        return pos_a < pos_b

    return Constraint(
        name=f"{pos_var_a} left_of {pos_var_b}",
        scope=[pos_var_a, pos_var_b],
        predicate=predicate,
    )


def different_position_constraint(pos_var_a: str, pos_var_b: str) -> Constraint:
    """
    Create constraint: pos(A) != pos(B)

    Two attributes must NOT share the same position (belong to different entities).
    Used for negation constraints like "Clara ist nicht die Anwaeltin".

    Args:
        pos_var_a: Position variable for attribute A
        pos_var_b: Position variable for attribute B

    Returns:
        Binary constraint enforcing different positions
    """

    def predicate(assignment: Dict[str, Any]) -> bool:
        pos_a = assignment.get(pos_var_a)
        pos_b = assignment.get(pos_var_b)
        if pos_a is None or pos_b is None:
            return True  # Not yet assigned
        return pos_a != pos_b

    return Constraint(
        name=f"{pos_var_a} different_position_from {pos_var_b}",
        scope=[pos_var_a, pos_var_b],
        predicate=predicate,
    )


def greater_than_constraint(pos_var_a: str, pos_var_b: str) -> Constraint:
    """
    Create constraint for ordering puzzles: A > B means pos(A) < pos(B)

    In ordering puzzles where position 1 = greatest and position N = smallest:
    - "A ist groesser als B" means A has a lower position number (closer to 1)
    - Example: Anna (pos=1) > Ben (pos=2) because 1 < 2

    Args:
        pos_var_a: Position variable name for entity A (the greater one)
        pos_var_b: Position variable name for entity B (the lesser one)

    Returns:
        Binary constraint enforcing pos(A) < pos(B)
    """

    def predicate(assignment: Dict[str, Any]) -> bool:
        pos_a = assignment.get(pos_var_a)
        pos_b = assignment.get(pos_var_b)
        if pos_a is None or pos_b is None:
            return True  # Not yet assigned
        return pos_a < pos_b

    return Constraint(
        name=f"{pos_var_a} greater_than {pos_var_b}",
        scope=[pos_var_a, pos_var_b],
        predicate=predicate,
    )


def less_than_constraint(pos_var_a: str, pos_var_b: str) -> Constraint:
    """
    Create constraint for ordering puzzles: A < B means pos(A) > pos(B)

    In ordering puzzles where position 1 = greatest and position N = smallest:
    - "A ist kleiner als B" means A has a higher position number (farther from 1)
    - Example: Ben (pos=2) < Anna (pos=1) because 2 > 1

    Args:
        pos_var_a: Position variable name for entity A (the lesser one)
        pos_var_b: Position variable name for entity B (the greater one)

    Returns:
        Binary constraint enforcing pos(A) > pos(B)
    """

    def predicate(assignment: Dict[str, Any]) -> bool:
        pos_a = assignment.get(pos_var_a)
        pos_b = assignment.get(pos_var_b)
        if pos_a is None or pos_b is None:
            return True  # Not yet assigned
        return pos_a > pos_b

    return Constraint(
        name=f"{pos_var_a} less_than {pos_var_b}",
        scope=[pos_var_a, pos_var_b],
        predicate=predicate,
    )
