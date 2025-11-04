"""
Tests for Component 29: Constraint Reasoning Engine

Test Coverage:
- Variable, Constraint, ConstraintProblem data structures
- ConstraintSolver with backtracking and AC-3
- MRV and LCV heuristics
- N-Queens Problem
- Graph Coloring
- Logic Grid Puzzles
- ProofTree integration
"""

import pytest
from component_29_constraint_reasoning import (
    Variable,
    ConstraintProblem,
    ConstraintSolver,
    ConstraintType,
    all_different_constraint,
    not_equal_constraint,
    unary_constraint,
    custom_constraint,
)
from component_17_proof_explanation import StepType


class TestVariable:
    """Test Variable data structure"""

    def test_variable_creation(self):
        """Test creating a variable with domain"""
        var = Variable(name="X", domain={1, 2, 3})

        assert var.name == "X"
        assert var.domain == {1, 2, 3}
        assert var.value is None
        assert not var.is_assigned()
        assert var.original_domain == {1, 2, 3}

    def test_variable_assignment(self):
        """Test assigning valid value"""
        var = Variable(name="X", domain={1, 2, 3})

        assert var.assign(2)
        assert var.value == 2
        assert var.is_assigned()

    def test_variable_invalid_assignment(self):
        """Test assigning value outside domain"""
        var = Variable(name="X", domain={1, 2, 3})

        assert not var.assign(5)  # Not in domain
        assert var.value is None

    def test_variable_unassign(self):
        """Test removing assignment"""
        var = Variable(name="X", domain={1, 2, 3})
        var.assign(2)

        var.unassign()
        assert var.value is None
        assert not var.is_assigned()

    def test_variable_reset_domain(self):
        """Test resetting domain to original"""
        var = Variable(name="X", domain={1, 2, 3})
        var.domain = {1}  # Simulate domain reduction
        var.assign(1)

        var.reset_domain()
        assert var.domain == {1, 2, 3}
        assert var.value is None


class TestConstraint:
    """Test Constraint data structure"""

    def test_unary_constraint(self):
        """Test unary constraint detection"""
        constraint = unary_constraint("X", {1, 2})

        assert constraint.constraint_type == ConstraintType.UNARY
        assert len(constraint.scope) == 1
        assert constraint.scope[0] == "X"

    def test_binary_constraint(self):
        """Test binary constraint detection"""
        constraint = not_equal_constraint("X", "Y")

        assert constraint.constraint_type == ConstraintType.BINARY
        assert len(constraint.scope) == 2
        assert "X" in constraint.scope
        assert "Y" in constraint.scope

    def test_nary_constraint(self):
        """Test n-ary constraint detection"""
        constraint = all_different_constraint(["X", "Y", "Z"])

        assert constraint.constraint_type == ConstraintType.N_ARY
        assert len(constraint.scope) == 3

    def test_constraint_satisfaction(self):
        """Test constraint checking"""
        constraint = not_equal_constraint("X", "Y")

        # Satisfied
        assert constraint.is_satisfied({"X": 1, "Y": 2})

        # Violated
        assert not constraint.is_satisfied({"X": 1, "Y": 1})

        # Incomplete assignment (assume satisfied)
        assert constraint.is_satisfied({"X": 1})

    def test_all_different_constraint(self):
        """Test all-different constraint"""
        constraint = all_different_constraint(["X", "Y", "Z"])

        # All different
        assert constraint.is_satisfied({"X": 1, "Y": 2, "Z": 3})

        # Two same
        assert not constraint.is_satisfied({"X": 1, "Y": 1, "Z": 2})

    def test_custom_constraint(self):
        """Test custom predicate constraint"""
        # X + Y < 10
        constraint = custom_constraint(
            name="X + Y < 10",
            scope=["X", "Y"],
            predicate=lambda a: a["X"] + a["Y"] < 10,
        )

        assert constraint.is_satisfied({"X": 3, "Y": 4})  # 3 + 4 = 7 < 10
        assert not constraint.is_satisfied({"X": 6, "Y": 7})  # 6 + 7 = 13 >= 10


class TestConstraintProblem:
    """Test ConstraintProblem data structure"""

    def test_problem_creation(self):
        """Test creating a CSP"""
        problem = ConstraintProblem(
            name="Test Problem",
            variables={"X": Variable("X", {1, 2}), "Y": Variable("Y", {1, 2})},
            constraints=[not_equal_constraint("X", "Y")],
        )

        assert problem.name == "Test Problem"
        assert len(problem.variables) == 2
        assert len(problem.constraints) == 1

    def test_get_variable(self):
        """Test retrieving variable"""
        var_x = Variable("X", {1, 2})
        problem = ConstraintProblem(name="Test", variables={"X": var_x}, constraints=[])

        assert problem.get_variable("X") == var_x
        assert problem.get_variable("Y") is None

    def test_get_constraints_for_variable(self):
        """Test retrieving constraints for variable"""
        c1 = not_equal_constraint("X", "Y")
        c2 = not_equal_constraint("X", "Z")
        c3 = not_equal_constraint("Y", "Z")

        problem = ConstraintProblem(
            name="Test",
            variables={
                "X": Variable("X", {1, 2}),
                "Y": Variable("Y", {1, 2}),
                "Z": Variable("Z", {1, 2}),
            },
            constraints=[c1, c2, c3],
        )

        x_constraints = problem.get_constraints_for_variable("X")
        assert len(x_constraints) == 2
        assert c1 in x_constraints
        assert c2 in x_constraints

    def test_get_neighbors(self):
        """Test finding neighbor variables"""
        problem = ConstraintProblem(
            name="Test",
            variables={
                "X": Variable("X", {1, 2}),
                "Y": Variable("Y", {1, 2}),
                "Z": Variable("Z", {1, 2}),
            },
            constraints=[
                not_equal_constraint("X", "Y"),
                not_equal_constraint("X", "Z"),
            ],
        )

        neighbors = problem.get_neighbors("X")
        assert neighbors == {"Y", "Z"}

    def test_is_complete(self):
        """Test checking if assignment is complete"""
        problem = ConstraintProblem(
            name="Test",
            variables={"X": Variable("X", {1, 2}), "Y": Variable("Y", {1, 2})},
            constraints=[],
        )

        assert not problem.is_complete({"X": 1})
        assert problem.is_complete({"X": 1, "Y": 2})

    def test_is_consistent(self):
        """Test checking assignment consistency"""
        problem = ConstraintProblem(
            name="Test",
            variables={"X": Variable("X", {1, 2}), "Y": Variable("Y", {1, 2})},
            constraints=[not_equal_constraint("X", "Y")],
        )

        assert problem.is_consistent({"X": 1, "Y": 2})  # X != Y
        assert not problem.is_consistent({"X": 1, "Y": 1})  # X == Y


class TestConstraintSolver:
    """Test ConstraintSolver with various problems"""

    def test_simple_binary_csp(self):
        """Test solving simple binary CSP"""
        # X != Y, both in {1, 2}
        problem = ConstraintProblem(
            name="Simple Binary",
            variables={"X": Variable("X", {1, 2}), "Y": Variable("Y", {1, 2})},
            constraints=[not_equal_constraint("X", "Y")],
        )

        solver = ConstraintSolver()
        solution, proof_tree = solver.solve(problem)

        assert solution is not None
        assert len(solution) == 2
        assert solution["X"] != solution["Y"]
        assert proof_tree is not None

    def test_no_solution_csp(self):
        """Test CSP with no solution"""
        # X != Y and X == Y (contradiction)
        problem = ConstraintProblem(
            name="Impossible",
            variables={"X": Variable("X", {1}), "Y": Variable("Y", {1})},
            constraints=[not_equal_constraint("X", "Y")],
        )

        solver = ConstraintSolver()
        solution, proof_tree = solver.solve(problem)

        assert solution is None
        assert proof_tree is not None
        # Check that there's a CONTRADICTION step
        all_steps = proof_tree.get_all_steps()
        assert any(step.step_type == StepType.CONTRADICTION for step in all_steps)

    def test_ac3_domain_reduction(self):
        """Test AC-3 reduces domains correctly"""
        # X != Y, X in {1}, Y in {1, 2}
        # AC-3 should remove 1 from Y's domain
        problem = ConstraintProblem(
            name="AC-3 Test",
            variables={"X": Variable("X", {1}), "Y": Variable("Y", {1, 2})},
            constraints=[not_equal_constraint("X", "Y")],
        )

        solver = ConstraintSolver(use_ac3=True)
        solver.ac3(problem)

        # Y's domain should be reduced to {2}
        assert problem.get_variable("Y").domain == {2}

    def test_solver_without_heuristics(self):
        """Test solver with all heuristics disabled"""
        problem = ConstraintProblem(
            name="No Heuristics",
            variables={"X": Variable("X", {1, 2}), "Y": Variable("Y", {1, 2})},
            constraints=[not_equal_constraint("X", "Y")],
        )

        solver = ConstraintSolver(use_ac3=False, use_mrv=False, use_lcv=False)
        solution, _ = solver.solve(problem)

        assert solution is not None


class TestNQueensProblem:
    """Test N-Queens Problem (classical CSP)"""

    def create_nqueens_problem(self, n: int) -> ConstraintProblem:
        """
        Create N-Queens CSP.

        Variables: Q0, Q1, ..., Q(n-1) representing queen positions
        Domain: {0, 1, ..., n-1} (column positions)
        Constraints: No two queens attack each other
        """
        # Create variables (one per row)
        variables = {}
        for i in range(n):
            variables[f"Q{i}"] = Variable(f"Q{i}", set(range(n)))

        # Create constraints
        constraints = []
        for i in range(n):
            for j in range(i + 1, n):
                qi = f"Q{i}"
                qj = f"Q{j}"

                # No two queens in same column
                constraints.append(not_equal_constraint(qi, qj))

                # No two queens on same diagonal
                def no_diagonal_attack(assignment, row1=i, row2=j):
                    q1 = assignment.get(f"Q{row1}")
                    q2 = assignment.get(f"Q{row2}")
                    if q1 is None or q2 is None:
                        return True
                    # Check diagonal: |row1 - row2| != |col1 - col2|
                    return abs(row1 - row2) != abs(q1 - q2)

                constraints.append(
                    custom_constraint(
                        name=f"Q{i} and Q{j} not on diagonal",
                        scope=[qi, qj],
                        predicate=no_diagonal_attack,
                    )
                )

        return ConstraintProblem(
            name=f"{n}-Queens Problem", variables=variables, constraints=constraints
        )

    def test_4queens(self):
        """Test 4-Queens problem"""
        problem = self.create_nqueens_problem(4)
        solver = ConstraintSolver(use_ac3=True, use_mrv=True, use_lcv=True)
        solution, proof_tree = solver.solve(problem)

        assert solution is not None, "4-Queens should have a solution"
        assert len(solution) == 4

        # Verify solution is valid
        positions = [solution[f"Q{i}"] for i in range(4)]

        # No two in same column
        assert len(set(positions)) == 4

        # No two on same diagonal
        for i in range(4):
            for j in range(i + 1, 4):
                assert abs(i - j) != abs(positions[i] - positions[j])

        # Verify proof tree
        assert proof_tree is not None
        # Check that there's a CONCLUSION step
        all_steps = proof_tree.get_all_steps()
        assert any(step.step_type == StepType.CONCLUSION for step in all_steps)

    def test_8queens(self):
        """Test 8-Queens problem (more challenging)"""
        problem = self.create_nqueens_problem(8)
        solver = ConstraintSolver(use_ac3=True, use_mrv=True, use_lcv=True)
        solution, proof_tree = solver.solve(problem)

        assert solution is not None, "8-Queens should have a solution"
        assert len(solution) == 8

        # Verify solution
        positions = [solution[f"Q{i}"] for i in range(8)]
        assert len(set(positions)) == 8  # All different columns

        # Check diagonals
        for i in range(8):
            for j in range(i + 1, 8):
                assert abs(i - j) != abs(positions[i] - positions[j])

    @pytest.mark.slow
    def test_12queens(self):
        """Test 12-Queens problem (performance test)"""
        problem = self.create_nqueens_problem(12)
        solver = ConstraintSolver(use_ac3=True, use_mrv=True, use_lcv=True)
        solution, _ = solver.solve(
            problem, track_proof=False
        )  # Disable proof for speed

        assert solution is not None, "12-Queens should have a solution"
        assert len(solution) == 12


class TestGraphColoringProblem:
    """Test Graph Coloring Problem (Constraint Propagation)"""

    def create_graph_coloring_problem(
        self, nodes: list, edges: list, colors: list
    ) -> ConstraintProblem:
        """
        Create graph coloring CSP.

        Args:
            nodes: List of node names
            edges: List of (node1, node2) tuples
            colors: List of available colors

        Returns:
            ConstraintProblem for graph coloring
        """
        # Create variables (one per node)
        variables = {}
        color_set = set(colors)
        for node in nodes:
            variables[node] = Variable(node, color_set.copy())

        # Create constraints (adjacent nodes must have different colors)
        constraints = []
        for node1, node2 in edges:
            constraints.append(not_equal_constraint(node1, node2))

        return ConstraintProblem(
            name="Graph Coloring", variables=variables, constraints=constraints
        )

    def test_simple_triangle(self):
        """Test 3-coloring a triangle (needs 3 colors)"""
        problem = self.create_graph_coloring_problem(
            nodes=["A", "B", "C"],
            edges=[("A", "B"), ("B", "C"), ("C", "A")],
            colors=["red", "green", "blue"],
        )

        solver = ConstraintSolver()
        solution, proof_tree = solver.solve(problem)

        assert solution is not None
        assert len(solution) == 3

        # Verify all adjacent nodes have different colors
        assert solution["A"] != solution["B"]
        assert solution["B"] != solution["C"]
        assert solution["C"] != solution["A"]

    def test_australia_map_coloring(self):
        """Test coloring map of Australia with 3 colors"""
        # States: WA, NT, SA, Q, NSW, V, T
        # Classic graph coloring problem
        problem = self.create_graph_coloring_problem(
            nodes=["WA", "NT", "SA", "Q", "NSW", "V", "T"],
            edges=[
                ("WA", "NT"),
                ("WA", "SA"),
                ("NT", "SA"),
                ("NT", "Q"),
                ("SA", "Q"),
                ("SA", "NSW"),
                ("SA", "V"),
                ("Q", "NSW"),
                ("NSW", "V"),
            ],
            colors=["red", "green", "blue"],
        )

        solver = ConstraintSolver(use_ac3=True, use_mrv=True, use_lcv=True)
        solution, proof_tree = solver.solve(problem)

        assert solution is not None, "Australia map should be 3-colorable"
        assert len(solution) == 7

        # Verify all edges respect coloring
        edges = [
            ("WA", "NT"),
            ("WA", "SA"),
            ("NT", "SA"),
            ("NT", "Q"),
            ("SA", "Q"),
            ("SA", "NSW"),
            ("SA", "V"),
            ("Q", "NSW"),
            ("NSW", "V"),
        ]
        for node1, node2 in edges:
            assert solution[node1] != solution[node2]

        assert proof_tree is not None
        # Check that there's a CONCLUSION step
        all_steps = proof_tree.get_all_steps()
        assert any(step.step_type == StepType.CONCLUSION for step in all_steps)

    def test_impossible_coloring(self):
        """Test graph that cannot be colored with given colors"""
        # Triangle with only 2 colors (impossible)
        problem = self.create_graph_coloring_problem(
            nodes=["A", "B", "C"],
            edges=[("A", "B"), ("B", "C"), ("C", "A")],
            colors=["red", "green"],  # Not enough colors
        )

        solver = ConstraintSolver(use_ac3=True)
        solution, proof_tree = solver.solve(problem)

        assert solution is None, "Should be impossible with only 2 colors"
        assert proof_tree is not None
        # Check that there's a CONTRADICTION step
        all_steps = proof_tree.get_all_steps()
        assert any(step.step_type == StepType.CONTRADICTION for step in all_steps)


class TestLogicGridPuzzle:
    """Test Logic Grid Puzzle (Multi-Domain CSP)"""

    def test_simple_logic_puzzle(self):
        """
        Test simple logic grid puzzle:

        Three people (Alice, Bob, Carol) each have a pet (cat, dog, bird)
        and a favorite color (red, blue, green).

        Clues:
        1. Alice doesn't have the cat
        2. Bob's favorite color is red
        3. The person with the dog likes blue
        4. Carol doesn't like green
        """
        # Variables: person-pet and person-color mappings
        variables = {
            "Alice_pet": Variable("Alice_pet", {"cat", "dog", "bird"}),
            "Bob_pet": Variable("Bob_pet", {"cat", "dog", "bird"}),
            "Carol_pet": Variable("Carol_pet", {"cat", "dog", "bird"}),
            "Alice_color": Variable("Alice_color", {"red", "blue", "green"}),
            "Bob_color": Variable("Bob_color", {"red", "blue", "green"}),
            "Carol_color": Variable("Carol_color", {"red", "blue", "green"}),
        }

        constraints = []

        # All different pets
        constraints.append(
            all_different_constraint(["Alice_pet", "Bob_pet", "Carol_pet"])
        )

        # All different colors
        constraints.append(
            all_different_constraint(["Alice_color", "Bob_color", "Carol_color"])
        )

        # Clue 1: Alice doesn't have the cat
        constraints.append(
            custom_constraint(
                name="Alice not cat",
                scope=["Alice_pet"],
                predicate=lambda a: a["Alice_pet"] != "cat",
            )
        )

        # Clue 2: Bob's favorite color is red
        constraints.append(
            custom_constraint(
                name="Bob likes red",
                scope=["Bob_color"],
                predicate=lambda a: a["Bob_color"] == "red",
            )
        )

        # Clue 3: Person with dog likes blue
        constraints.append(
            custom_constraint(
                name="Dog owner likes blue",
                scope=[
                    "Alice_pet",
                    "Bob_pet",
                    "Carol_pet",
                    "Alice_color",
                    "Bob_color",
                    "Carol_color",
                ],
                predicate=lambda a: (
                    (a["Alice_pet"] == "dog" and a["Alice_color"] == "blue")
                    or (a["Bob_pet"] == "dog" and a["Bob_color"] == "blue")
                    or (a["Carol_pet"] == "dog" and a["Carol_color"] == "blue")
                ),
            )
        )

        # Clue 4: Carol doesn't like green
        constraints.append(
            custom_constraint(
                name="Carol not green",
                scope=["Carol_color"],
                predicate=lambda a: a["Carol_color"] != "green",
            )
        )

        problem = ConstraintProblem(
            name="Logic Grid Puzzle", variables=variables, constraints=constraints
        )

        solver = ConstraintSolver(use_ac3=True, use_mrv=True, use_lcv=True)
        solution, proof_tree = solver.solve(problem)

        assert solution is not None, "Puzzle should have a solution"

        # Verify clues
        assert solution["Alice_pet"] != "cat"  # Clue 1
        assert solution["Bob_color"] == "red"  # Clue 2

        # Clue 3: Dog owner likes blue
        if solution["Alice_pet"] == "dog":
            assert solution["Alice_color"] == "blue"
        elif solution["Bob_pet"] == "dog":
            assert solution["Bob_color"] == "blue"
        else:
            assert solution["Carol_pet"] == "dog"
            assert solution["Carol_color"] == "blue"

        assert solution["Carol_color"] != "green"  # Clue 4

        # Verify proof tree
        assert proof_tree is not None
        # Check that there's a CONCLUSION step
        all_steps = proof_tree.get_all_steps()
        assert any(step.step_type == StepType.CONCLUSION for step in all_steps)


class TestProofTreeIntegration:
    """Test ProofTree generation during solving"""

    def test_proof_tree_structure(self):
        """Test that proof tree contains expected steps"""
        problem = ConstraintProblem(
            name="Simple",
            variables={"X": Variable("X", {1, 2}), "Y": Variable("Y", {1, 2})},
            constraints=[not_equal_constraint("X", "Y")],
        )

        solver = ConstraintSolver(use_ac3=True)
        solution, proof_tree = solver.solve(problem, track_proof=True)

        assert proof_tree is not None
        all_steps = proof_tree.get_all_steps()
        assert len(all_steps) > 0

        # Check for expected step types
        step_types = {step.step_type for step in all_steps}
        assert StepType.PREMISE in step_types  # AC-3 premise
        assert StepType.CONCLUSION in step_types  # Final solution

    def test_proof_tree_disabled(self):
        """Test solving without proof tree tracking"""
        problem = ConstraintProblem(
            name="Simple",
            variables={"X": Variable("X", {1, 2}), "Y": Variable("Y", {1, 2})},
            constraints=[not_equal_constraint("X", "Y")],
        )

        solver = ConstraintSolver()
        solution, proof_tree = solver.solve(problem, track_proof=False)

        assert solution is not None
        assert proof_tree is None  # No proof tree generated


class TestPerformanceMetrics:
    """Test solver performance tracking"""

    def test_backtrack_counting(self):
        """Test that backtrack count is tracked"""
        problem = ConstraintProblem(
            name="Test",
            variables={
                "X": Variable("X", {1, 2, 3}),
                "Y": Variable("Y", {1, 2, 3}),
                "Z": Variable("Z", {1, 2, 3}),
            },
            constraints=[all_different_constraint(["X", "Y", "Z"])],
        )

        solver = ConstraintSolver(use_ac3=False)  # Disable AC-3 to force backtracks
        solver.solve(problem, track_proof=False)

        assert solver.backtrack_count >= 0

    def test_constraint_check_counting(self):
        """Test that constraint checks are counted"""
        problem = ConstraintProblem(
            name="Test",
            variables={"X": Variable("X", {1, 2}), "Y": Variable("Y", {1, 2})},
            constraints=[not_equal_constraint("X", "Y")],
        )

        solver = ConstraintSolver()
        solver.solve(problem, track_proof=False)

        assert solver.constraint_checks > 0
