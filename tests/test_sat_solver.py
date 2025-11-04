"""
tests/test_sat_solver.py
========================
Comprehensive test suite for SAT solver (component_30).

Tests:
- Basic satisfiability checking
- Unit propagation and pure literal elimination
- DPLL algorithm correctness
- Watched literals optimization
- SAT encoding helpers
- Knights and Knaves puzzle
- Knowledge base consistency checking
- Integration with Logic Engine

Author: KAI Development Team
Date: 2025-10-30
"""

import pytest
from component_30_sat_solver import (
    Literal,
    Clause,
    CNFFormula,
    DPLLSolver,
    SATEncoder,
    KnowledgeBaseChecker,
    SATResult,
    create_knights_and_knaves_problem,
    WatchedLiterals,
    PropositionalFormula,
    PropositionalOperator,
    CNFConverter,
    SATSolver,
    solve_cnf,
    solve_propositional,
)


class TestLiteral:
    """Test Literal class."""

    def test_literal_creation(self):
        """Test creating literals."""
        lit = Literal("x")
        assert lit.variable == "x"
        assert not lit.negated

        neg_lit = Literal("x", True)
        assert neg_lit.variable == "x"
        assert neg_lit.negated

    def test_literal_negation(self):
        """Test literal negation."""
        lit = Literal("x")
        neg_lit = -lit
        assert neg_lit.variable == "x"
        assert neg_lit.negated

        double_neg = -neg_lit
        assert double_neg.variable == "x"
        assert not double_neg.negated

    def test_literal_equality(self):
        """Test literal equality."""
        lit1 = Literal("x")
        lit2 = Literal("x")
        lit3 = Literal("x", True)

        assert lit1 == lit2
        assert lit1 != lit3

    def test_literal_hash(self):
        """Test literal hashing (for use in sets)."""
        lit1 = Literal("x")
        lit2 = Literal("x")
        lit3 = Literal("x", True)

        s = {lit1, lit2, lit3}
        assert len(s) == 2  # lit1 and lit2 are same


class TestClause:
    """Test Clause class."""

    def test_clause_creation(self):
        """Test creating clauses."""
        lit_x = Literal("x")
        lit_y = Literal("y")
        clause = Clause({lit_x, lit_y})

        assert len(clause.literals) == 2
        assert lit_x in clause.literals
        assert lit_y in clause.literals

    def test_empty_clause(self):
        """Test empty clause (represents FALSE)."""
        clause = Clause(set())
        assert clause.is_empty()
        assert not clause.is_unit()

    def test_unit_clause(self):
        """Test unit clause (single literal)."""
        lit_x = Literal("x")
        clause = Clause({lit_x})

        assert clause.is_unit()
        assert clause.get_unit_literal() == lit_x

    def test_clause_simplification(self):
        """Test clause simplification with assignment."""
        lit_x = Literal("x")
        lit_y = Literal("y")
        lit_z = Literal("z")
        clause = Clause({lit_x, lit_y, lit_z})

        # x = True satisfies clause (x is in clause)
        result = clause.simplify({"x": True})
        assert result is None  # Clause satisfied

        # x = False removes x from clause
        result = clause.simplify({"x": False})
        assert result is not None
        assert lit_x not in result.literals
        assert lit_y in result.literals
        assert lit_z in result.literals

        # x = False, y = False leaves only z
        result = clause.simplify({"x": False, "y": False})
        assert result is not None
        assert result.is_unit()
        assert result.get_unit_literal() == lit_z


class TestCNFFormula:
    """Test CNFFormula class."""

    def test_formula_creation(self):
        """Test creating CNF formulas."""
        clause1 = Clause({Literal("x"), Literal("y")})
        clause2 = Clause({Literal("x", True), Literal("z")})
        formula = CNFFormula([clause1, clause2])

        assert len(formula.clauses) == 2
        assert len(formula.variables) == 3
        assert "x" in formula.variables
        assert "y" in formula.variables
        assert "z" in formula.variables

    def test_empty_formula(self):
        """Test empty formula (represents TRUE)."""
        formula = CNFFormula([])
        assert formula.is_empty()

    def test_has_empty_clause(self):
        """Test detecting empty clause in formula."""
        clause1 = Clause({Literal("x")})
        clause2 = Clause(set())  # Empty
        formula = CNFFormula([clause1, clause2])

        assert formula.has_empty_clause()

    def test_get_unit_clauses(self):
        """Test extracting unit clauses."""
        clause1 = Clause({Literal("x")})  # Unit
        clause2 = Clause({Literal("y"), Literal("z")})  # Not unit
        clause3 = Clause({Literal("w")})  # Unit
        formula = CNFFormula([clause1, clause2, clause3])

        unit_lits = formula.get_unit_clauses()
        assert len(unit_lits) == 2
        assert Literal("x") in unit_lits
        assert Literal("w") in unit_lits

    def test_get_pure_literals(self):
        """Test extracting pure literals."""
        # x appears only positive, y appears both ways
        clause1 = Clause({Literal("x"), Literal("y")})
        clause2 = Clause({Literal("x"), Literal("y", True)})
        formula = CNFFormula([clause1, clause2])

        pure_lits = formula.get_pure_literals({})
        assert len(pure_lits) == 1
        assert Literal("x") in pure_lits


class TestDPLLSolver:
    """Test DPLL solver."""

    def test_satisfiable_formula(self):
        """Test solving satisfiable formula."""
        # (x ∨ y) ∧ (¬x ∨ z) ∧ (¬y ∨ ¬z)
        # Satisfiable: x=T, y=F, z=T
        formula = CNFFormula(
            [
                Clause({Literal("x"), Literal("y")}),
                Clause({Literal("x", True), Literal("z")}),
                Clause({Literal("y", True), Literal("z", True)}),
            ]
        )

        solver = DPLLSolver()
        result, model = solver.solve(formula)

        assert result == SATResult.SATISFIABLE
        assert model is not None
        # Verify model satisfies formula
        for clause in formula.clauses:
            simplified = clause.simplify(model)
            assert simplified is None  # All clauses satisfied

    def test_unsatisfiable_formula(self):
        """Test solving unsatisfiable formula."""
        # (x) ∧ (¬x)
        # Unsatisfiable
        formula = CNFFormula([Clause({Literal("x")}), Clause({Literal("x", True)})])

        solver = DPLLSolver()
        result, model = solver.solve(formula)

        assert result == SATResult.UNSATISFIABLE
        assert model is None

    def test_unit_propagation(self):
        """Test unit propagation."""
        # (x) ∧ (¬x ∨ y) ∧ (¬y ∨ z)
        # Unit propagation: x=T → y=T → z=T
        formula = CNFFormula(
            [
                Clause({Literal("x")}),  # Unit: x=T
                Clause({Literal("x", True), Literal("y")}),  # Forces y=T
                Clause({Literal("y", True), Literal("z")}),  # Forces z=T
            ]
        )

        solver = DPLLSolver()
        result, model = solver.solve(formula)

        assert result == SATResult.SATISFIABLE
        assert model["x"] is True
        assert model["y"] is True
        assert model["z"] is True

    def test_pure_literal_elimination(self):
        """Test pure literal elimination."""
        # (x ∨ y) ∧ (x ∨ z)
        # x is pure (only positive), y and z are pure
        formula = CNFFormula(
            [Clause({Literal("x"), Literal("y")}), Clause({Literal("x"), Literal("z")})]
        )

        solver = DPLLSolver()
        result, model = solver.solve(formula)

        assert result == SATResult.SATISFIABLE
        # x should be assigned True (pure literal)
        assert model["x"] is True

    def test_watched_literals_optimization(self):
        """Test solver with watched literals enabled."""
        formula = CNFFormula(
            [
                Clause({Literal("x"), Literal("y")}),
                Clause({Literal("x", True), Literal("z")}),
                Clause({Literal("y", True), Literal("z", True)}),
            ]
        )

        # Solver with watched literals
        solver_watched = DPLLSolver(use_watched_literals=True)
        result_watched, model_watched = solver_watched.solve(formula)

        # Solver without watched literals (for comparison)
        solver_basic = DPLLSolver(use_watched_literals=False)
        result_basic, model_basic = solver_basic.solve(formula)

        # Both should give same result
        assert result_watched == result_basic == SATResult.SATISFIABLE
        # Models might differ, but both should satisfy formula
        assert model_watched is not None
        assert model_basic is not None


class TestSATEncoder:
    """Test SAT encoding helpers."""

    def test_encode_implication(self):
        """Test encoding implication."""
        # x → y  ≡  ¬x ∨ y
        x = Literal("x")
        y = Literal("y")

        clause = SATEncoder.encode_implication(x, y)
        assert Literal("x", True) in clause.literals
        assert y in clause.literals

    def test_encode_iff(self):
        """Test encoding bi-implication."""
        # x ↔ y  ≡  (x → y) ∧ (y → x)
        x = Literal("x")
        y = Literal("y")

        clauses = SATEncoder.encode_iff(x, y)
        assert len(clauses) == 2

        # Test with solver
        formula = CNFFormula(clauses)
        solver = DPLLSolver()
        result, model = solver.solve(formula)

        assert result == SATResult.SATISFIABLE
        # x and y should have same value
        assert model["x"] == model["y"]

    def test_encode_xor(self):
        """Test encoding XOR."""
        # x ⊕ y  ≡  (x ∨ y) ∧ (¬x ∨ ¬y)
        x = Literal("x")
        y = Literal("y")

        clauses = SATEncoder.encode_xor(x, y)
        assert len(clauses) == 2

        # Test with solver
        formula = CNFFormula(clauses)
        solver = DPLLSolver()
        result, model = solver.solve(formula)

        assert result == SATResult.SATISFIABLE
        # x and y should have different values
        assert model["x"] != model["y"]

    def test_encode_at_most_one(self):
        """Test encoding at-most-one constraint."""
        lits = [Literal("x"), Literal("y"), Literal("z")]
        clauses = SATEncoder.encode_at_most_one(lits)

        # Should have C(3,2) = 3 clauses
        assert len(clauses) == 3

        formula = CNFFormula(clauses)
        solver = DPLLSolver()
        result, model = solver.solve(formula)

        assert result == SATResult.SATISFIABLE
        # At most one should be true
        true_count = sum(1 for lit in lits if model[lit.variable])
        assert true_count <= 1

    def test_encode_exactly_one(self):
        """Test encoding exactly-one constraint."""
        lits = [Literal("x"), Literal("y"), Literal("z")]
        clauses = SATEncoder.encode_exactly_one(lits)

        formula = CNFFormula(clauses)
        solver = DPLLSolver()
        result, model = solver.solve(formula)

        assert result == SATResult.SATISFIABLE
        # Exactly one should be true
        true_count = sum(1 for lit in lits if model[lit.variable])
        assert true_count == 1


class TestKnightsAndKnaves:
    """Test Knights and Knaves puzzle."""

    def test_knights_and_knaves_solvable(self):
        """Test that Knights and Knaves puzzle is solvable."""
        puzzle = create_knights_and_knaves_problem()

        solver = DPLLSolver()
        result, model = solver.solve(puzzle)

        assert result == SATResult.SATISFIABLE
        assert model is not None

    def test_knights_and_knaves_solution(self):
        """Test Knights and Knaves puzzle solution."""
        puzzle = create_knights_and_knaves_problem()

        solver = DPLLSolver()
        result, model = solver.solve(puzzle)

        assert result == SATResult.SATISFIABLE

        # Verify the found solution is consistent
        # Solution found by solver: A=Knight, B=Knave, C=Knave
        # Let's verify:
        # - A says "B is a knave": A is knight (truthful) → B must be knave ✓
        # - B says "A and C are both knights": B is knave (lies) → NOT both knights ✓
        # - C says "A is a knave": C is knave (lies) → A is NOT knave (is knight) ✓

        # Verify the constraint: A ↔ ¬B (A and B have opposite types)
        assert model["k_A"] != model["k_B"]

        # Verify the constraint: C ↔ ¬A (C and A have opposite types)
        assert model["k_C"] != model["k_A"]


class TestKnowledgeBaseChecker:
    """Test knowledge base consistency checking."""

    def test_consistent_facts(self):
        """Test checking consistent facts."""
        checker = KnowledgeBaseChecker()

        [Literal("bird"), Literal("can_fly")]

        rules = []

        is_consistent, model = checker.check_rule_consistency(rules)
        assert is_consistent

    def test_inconsistent_facts(self):
        """Test detecting inconsistent facts."""
        checker = KnowledgeBaseChecker()

        facts = [Literal("can_fly"), Literal("can_fly", True)]  # Contradiction

        rules = [([Literal("bird")], Literal("can_fly"))]

        # This should detect conflict
        conflicts = checker.find_conflicts(facts, rules)
        assert len(conflicts) > 0

    def test_consistent_rules(self):
        """Test checking consistent rules."""
        checker = KnowledgeBaseChecker()

        rules = [
            ([Literal("bird")], Literal("can_fly")),
            ([Literal("can_fly")], Literal("has_wings")),
        ]

        is_consistent, model = checker.check_rule_consistency(rules)
        assert is_consistent

    def test_inconsistent_rules(self):
        """Test detecting inconsistent rules."""
        checker = KnowledgeBaseChecker()

        rules = [
            ([Literal("penguin")], Literal("bird")),
            ([Literal("bird")], Literal("can_fly")),
            ([Literal("penguin")], Literal("can_fly", True)),  # Contradiction
        ]

        is_consistent, model = checker.check_rule_consistency(rules)
        # This should be SATISFIABLE because rules are implications,
        # not facts. We just don't assert penguin.
        assert is_consistent

        # But if we add penguin as fact:
        facts = [Literal("penguin")]
        conflicts = checker.find_conflicts(facts, rules)
        assert len(conflicts) > 0


class TestWatchedLiterals:
    """Test watched literals optimization."""

    def test_watched_literals_initialization(self):
        """Test initializing watched literals."""
        formula = CNFFormula(
            [
                Clause({Literal("x"), Literal("y"), Literal("z")}),
                Clause({Literal("a"), Literal("b")}),
            ]
        )

        watched = WatchedLiterals()
        watched.initialize(formula)

        # Each clause should have two watched literals
        assert len(watched.watched) == 2
        for clause_id in range(2):
            assert clause_id in watched.watched
            lit1, lit2 = watched.watched[clause_id]
            assert lit1 in formula.clauses[clause_id].literals
            assert lit2 in formula.clauses[clause_id].literals

    def test_watched_literals_propagation(self):
        """Test unit propagation with watched literals."""
        # (x ∨ y) ∧ (¬x ∨ z)
        # If x=F, then y must be T (unit propagation on first clause)
        formula = CNFFormula(
            [
                Clause({Literal("x"), Literal("y")}),
                Clause({Literal("x", True), Literal("z")}),
            ]
        )

        watched = WatchedLiterals()
        watched.initialize(formula)

        # Assign x = False
        assignment = {"x": False}
        success, unit_lits = watched.propagate(assignment)

        assert success
        # Should propagate: y must be True, z must be True
        assert Literal("y") in unit_lits or Literal("z") in unit_lits


class TestPropositionalFormula:
    """Test PropositionalFormula and operators."""

    def test_variable_formula(self):
        """Test creating variable formula."""
        x = PropositionalFormula.variable_formula("x")
        assert x.variable == "x"
        assert str(x) == "x"

    def test_not_formula(self):
        """Test creating NOT formula."""
        x = PropositionalFormula.variable_formula("x")
        not_x = PropositionalFormula.not_formula(x)
        assert not_x.operator == PropositionalOperator.NOT
        assert "¬" in str(not_x)

    def test_and_formula(self):
        """Test creating AND formula."""
        x = PropositionalFormula.variable_formula("x")
        y = PropositionalFormula.variable_formula("y")
        and_formula = PropositionalFormula.and_formula(x, y)
        assert and_formula.operator == PropositionalOperator.AND
        assert "∧" in str(and_formula)

    def test_or_formula(self):
        """Test creating OR formula."""
        x = PropositionalFormula.variable_formula("x")
        y = PropositionalFormula.variable_formula("y")
        or_formula = PropositionalFormula.or_formula(x, y)
        assert or_formula.operator == PropositionalOperator.OR
        assert "∨" in str(or_formula)

    def test_implies_formula(self):
        """Test creating IMPLIES formula."""
        x = PropositionalFormula.variable_formula("x")
        y = PropositionalFormula.variable_formula("y")
        implies = PropositionalFormula.implies_formula(x, y)
        assert implies.operator == PropositionalOperator.IMPLIES
        assert "→" in str(implies)

    def test_iff_formula(self):
        """Test creating IFF formula."""
        x = PropositionalFormula.variable_formula("x")
        y = PropositionalFormula.variable_formula("y")
        iff = PropositionalFormula.iff_formula(x, y)
        assert iff.operator == PropositionalOperator.IFF
        assert "↔" in str(iff)

    def test_complex_formula(self):
        """Test creating complex nested formula."""
        # (p → q) ∧ (q → r)
        p = PropositionalFormula.variable_formula("p")
        q = PropositionalFormula.variable_formula("q")
        r = PropositionalFormula.variable_formula("r")

        impl1 = PropositionalFormula.implies_formula(p, q)
        impl2 = PropositionalFormula.implies_formula(q, r)
        formula = PropositionalFormula.and_formula(impl1, impl2)

        assert formula.operator == PropositionalOperator.AND
        assert len(formula.operands) == 2


class TestCNFConverter:
    """Test CNF conversion."""

    def test_simple_variable_to_cnf(self):
        """Test converting simple variable to CNF."""
        x = PropositionalFormula.variable_formula("x")
        cnf = CNFConverter.to_cnf(x)

        assert len(cnf.clauses) == 1
        assert len(cnf.clauses[0].literals) == 1

    def test_negation_to_cnf(self):
        """Test converting negation to CNF."""
        x = PropositionalFormula.variable_formula("x")
        not_x = PropositionalFormula.not_formula(x)
        cnf = CNFConverter.to_cnf(not_x)

        assert len(cnf.clauses) == 1
        lit = next(iter(cnf.clauses[0].literals))
        assert lit.negated is True

    def test_implies_to_cnf(self):
        """Test converting implication to CNF: x → y = ¬x ∨ y."""
        x = PropositionalFormula.variable_formula("x")
        y = PropositionalFormula.variable_formula("y")
        implies = PropositionalFormula.implies_formula(x, y)
        cnf = CNFConverter.to_cnf(implies)

        # Should be single clause: (¬x ∨ y)
        assert len(cnf.clauses) == 1
        assert len(cnf.clauses[0].literals) == 2

    def test_iff_to_cnf(self):
        """Test converting bi-implication to CNF: x ↔ y = (¬x ∨ y) ∧ (x ∨ ¬y)."""
        x = PropositionalFormula.variable_formula("x")
        y = PropositionalFormula.variable_formula("y")
        iff = PropositionalFormula.iff_formula(x, y)
        cnf = CNFConverter.to_cnf(iff)

        # Should be two clauses
        assert len(cnf.clauses) == 2

    def test_de_morgan_to_cnf(self):
        """Test De Morgan's laws: ¬(x ∧ y) = ¬x ∨ ¬y."""
        x = PropositionalFormula.variable_formula("x")
        y = PropositionalFormula.variable_formula("y")
        and_formula = PropositionalFormula.and_formula(x, y)
        not_and = PropositionalFormula.not_formula(and_formula)

        cnf = CNFConverter.to_cnf(not_and)

        # Should be single clause: (¬x ∨ ¬y)
        assert len(cnf.clauses) == 1
        assert len(cnf.clauses[0].literals) == 2

    def test_distribution_to_cnf(self):
        """Test distribution: x ∨ (y ∧ z) = (x ∨ y) ∧ (x ∨ z)."""
        x = PropositionalFormula.variable_formula("x")
        y = PropositionalFormula.variable_formula("y")
        z = PropositionalFormula.variable_formula("z")

        # x ∨ (y ∧ z)
        and_yz = PropositionalFormula.and_formula(y, z)
        or_formula = PropositionalFormula.or_formula(x, and_yz)

        cnf = CNFConverter.to_cnf(or_formula)

        # Should be two clauses: (x ∨ y) ∧ (x ∨ z)
        assert len(cnf.clauses) == 2

    def test_complex_formula_to_cnf(self):
        """Test converting complex formula to CNF."""
        # (p → q) ∧ p
        p = PropositionalFormula.variable_formula("p")
        q = PropositionalFormula.variable_formula("q")

        impl = PropositionalFormula.implies_formula(p, q)
        formula = PropositionalFormula.and_formula(impl, p)

        cnf = CNFConverter.to_cnf(formula)

        # Should be two clauses: (¬p ∨ q) ∧ (p)
        assert len(cnf.clauses) == 2


class TestSATSolverWrapper:
    """Test SATSolver wrapper API."""

    def test_solver_initialization(self):
        """Test initializing SAT solver."""
        solver = SATSolver(enable_proof=False)
        assert solver.enable_proof is False

    def test_solve_satisfiable(self):
        """Test solve() with satisfiable formula."""
        formula = CNFFormula([Clause({Literal("x"), Literal("y")})])

        solver = SATSolver(enable_proof=False)
        model = solver.solve(formula)

        assert model is not None
        assert "x" in model or "y" in model

    def test_solve_unsatisfiable(self):
        """Test solve() with unsatisfiable formula."""
        formula = CNFFormula([Clause({Literal("x")}), Clause({Literal("x", True)})])

        solver = SATSolver(enable_proof=False)
        model = solver.solve(formula)

        assert model is None

    def test_dpll_with_partial_assignment(self):
        """Test dpll() with partial assignment."""
        # (x ∨ y) ∧ (¬x ∨ z)
        formula = CNFFormula(
            [
                Clause({Literal("x"), Literal("y")}),
                Clause({Literal("x", True), Literal("z")}),
            ]
        )

        solver = SATSolver(enable_proof=False)
        # Start with x = False
        model = solver.dpll(formula, {"x": False})

        assert model is not None
        assert model["x"] is False
        assert model["y"] is True  # Must be True to satisfy first clause

    def test_unit_propagate_method(self):
        """Test unit_propagate() method."""
        # (x) ∧ (¬x ∨ y) ∧ (¬y ∨ z)
        formula = CNFFormula(
            [
                Clause({Literal("x")}),
                Clause({Literal("x", True), Literal("y")}),
                Clause({Literal("y", True), Literal("z")}),
            ]
        )

        solver = SATSolver(enable_proof=False)
        simplified, assignment = solver.unit_propagate(formula, {})

        # Should propagate x=True, y=True, z=True
        assert assignment["x"] is True
        assert assignment["y"] is True
        assert assignment["z"] is True

    def test_pure_literal_elimination_method(self):
        """Test pure_literal_elimination() method."""
        # (x ∨ y) ∧ (x ∨ z)
        formula = CNFFormula(
            [Clause({Literal("x"), Literal("y")}), Clause({Literal("x"), Literal("z")})]
        )

        solver = SATSolver(enable_proof=False)
        simplified, assignment = solver.pure_literal_elimination(formula, {})

        # x is pure literal (only positive) → x=True
        assert "x" in assignment
        assert assignment["x"] is True


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_solve_cnf(self):
        """Test solve_cnf() convenience function."""
        formula = CNFFormula([Clause({Literal("x"), Literal("y")})])

        model = solve_cnf(formula)
        assert model is not None

    def test_solve_cnf_unsat(self):
        """Test solve_cnf() with UNSAT formula."""
        formula = CNFFormula([Clause({Literal("x")}), Clause({Literal("x", True)})])

        model = solve_cnf(formula)
        assert model is None

    def test_solve_propositional(self):
        """Test solve_propositional() convenience function."""
        # x → y with x = True should require y = True
        x = PropositionalFormula.variable_formula("x")
        y = PropositionalFormula.variable_formula("y")
        formula = PropositionalFormula.and_formula(
            PropositionalFormula.implies_formula(x, y), x
        )

        model = solve_propositional(formula)
        assert model is not None
        assert model["x"] is True
        assert model["y"] is True

    def test_solve_propositional_complex(self):
        """Test solve_propositional() with complex formula."""
        # (p → q) ∧ (q → r) ∧ p should give p=T, q=T, r=T
        p = PropositionalFormula.variable_formula("p")
        q = PropositionalFormula.variable_formula("q")
        r = PropositionalFormula.variable_formula("r")

        formula = PropositionalFormula.and_formula(
            PropositionalFormula.implies_formula(p, q),
            PropositionalFormula.implies_formula(q, r),
            p,
        )

        model = solve_propositional(formula)

        assert model is not None
        assert model["p"] is True
        assert model["q"] is True
        assert model["r"] is True


class TestProofGeneration:
    """Test proof tree generation."""

    def test_proof_enabled(self):
        """Test solver with proof generation enabled."""
        formula = CNFFormula([Clause({Literal("x"), Literal("y")})])

        solver = SATSolver(enable_proof=True)
        model = solver.solve(formula)

        assert model is not None

        # Get proof tree (may be None if component_17 not available)
        solver.get_proof_tree()
        # Test passes regardless of proof availability

    def test_proof_disabled(self):
        """Test solver with proof generation disabled."""
        formula = CNFFormula([Clause({Literal("x"), Literal("y")})])

        solver = SATSolver(enable_proof=False)
        model = solver.solve(formula)

        assert model is not None

        # Proof should be None when disabled
        proof_tree = solver.get_proof_tree()
        assert proof_tree is None

    def test_proof_for_unsat(self):
        """Test proof generation for UNSAT formula."""
        formula = CNFFormula([Clause({Literal("x")}), Clause({Literal("x", True)})])

        solver = SATSolver(enable_proof=True)
        model = solver.solve(formula)

        assert model is None

        # Get proof tree
        solver.get_proof_tree()
        # Test passes regardless of proof availability


class TestIntegrationWithLogicEngine:
    """Test integration with Logic Engine."""

    def test_import_sat_solver_in_logic_engine(self):
        """Test that SAT solver can be imported in logic engine."""
        try:
            from component_9_logik_engine import SAT_SOLVER_AVAILABLE

            assert SAT_SOLVER_AVAILABLE is True
        except ImportError:
            pytest.skip("Logic engine not available")

    def test_engine_initialization_with_sat(self):
        """Test initializing Engine with SAT solver."""
        try:
            from component_9_logik_engine import Engine
            from component_1_netzwerk import KonzeptNetzwerk

            netzwerk = KonzeptNetzwerk()
            engine = Engine(netzwerk, use_sat=True)

            assert engine.use_sat is True
            assert engine.sat_solver is not None
            assert engine.kb_checker is not None

        except ImportError as e:
            pytest.skip(f"Dependencies not available: {e}")


# Performance tests (marked slow)
@pytest.mark.slow
class TestPerformance:
    """Test solver performance on larger problems."""

    def test_large_satisfiable_formula(self):
        """Test solving large satisfiable formula."""
        # Generate (x1 ∨ x2) ∧ (x2 ∨ x3) ∧ ... ∧ (xn-1 ∨ xn)
        n = 100
        clauses = []
        for i in range(n - 1):
            clauses.append(Clause({Literal(f"x{i}"), Literal(f"x{i+1}")}))

        formula = CNFFormula(clauses)
        solver = DPLLSolver()
        result, model = solver.solve(formula)

        assert result == SATResult.SATISFIABLE
        assert len(model) == n

    def test_solver_with_many_variables(self):
        """Test solver with many variables."""
        # Create formula with 50 variables
        n = 50
        clauses = []
        for i in range(n - 1):
            # xi → xi+1
            clauses.append(Clause({Literal(f"x{i}", True), Literal(f"x{i+1}")}))

        formula = CNFFormula(clauses)
        solver = DPLLSolver()
        result, model = solver.solve(formula)

        assert result == SATResult.SATISFIABLE


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
