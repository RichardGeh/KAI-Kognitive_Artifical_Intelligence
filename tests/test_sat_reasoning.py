"""
Tests for Component 30: SAT-based Reasoning

Test Coverage:
- Consistency checking of rule sets
- Contradiction detection in knowledge bases
- Knights and Knaves puzzle as SAT application
- Generic SAT capabilities (not puzzle-specific parsing)

Focus: Demonstrates SAT-Solver's ability to verify logical consistency,
detect contradictions, and solve constraint problems.
"""

import pytest
from component_30_sat_solver import (
    Literal,
    Clause,
    CNFFormula,
    DPLLSolver,
    SATSolver,
    SATResult,
    SATEncoder,
    KnowledgeBaseChecker,
    PropositionalFormula,
    CNFConverter,
    create_knights_and_knaves_problem,
)


class TestConsistencyChecking:
    """Test consistency checking of rule sets"""

    def test_consistent_rules(self):
        """Test that consistent rules are detected as satisfiable"""
        # Rules: A → B, B → C (consistent)
        # CNF: (¬A ∨ B) ∧ (¬B ∨ C)
        formula = CNFFormula(
            [
                Clause({Literal("A", True), Literal("B")}),  # ¬A ∨ B
                Clause({Literal("B", True), Literal("C")}),  # ¬B ∨ C
            ]
        )

        solver = DPLLSolver()
        result, model = solver.solve(formula)

        assert result == SATResult.SATISFIABLE
        assert model is not None

        # Verify model satisfies rules
        # If A is true, B must be true
        if model.get("A", False):
            assert model.get("B", False)
        # If B is true, C must be true
        if model.get("B", False):
            assert model.get("C", False)

    def test_circular_implication_consistency(self):
        """Test consistency of circular implications: A → B, B → C, C → A"""
        # Rules: A → B, B → C, C → A (consistent: all true or all false)
        # CNF: (¬A ∨ B) ∧ (¬B ∨ C) ∧ (¬C ∨ A)
        formula = CNFFormula(
            [
                Clause({Literal("A", True), Literal("B")}),  # ¬A ∨ B
                Clause({Literal("B", True), Literal("C")}),  # ¬B ∨ C
                Clause({Literal("C", True), Literal("A")}),  # ¬C ∨ A
            ]
        )

        solver = DPLLSolver()
        result, model = solver.solve(formula)

        assert result == SATResult.SATISFIABLE
        assert model is not None

        # Verify all three have same value (all true or all false)
        a_val = model.get("A", False)
        b_val = model.get("B", False)
        c_val = model.get("C", False)
        assert a_val == b_val == c_val

    def test_inconsistent_circular_rules(self):
        """Test inconsistency: A → B, B → C, C → ¬A, A (forced contradiction)"""
        # Rules: A → B, B → C, C → ¬A, and fact: A is true
        # CNF: (¬A ∨ B) ∧ (¬B ∨ C) ∧ (¬C ∨ ¬A) ∧ (A)
        # This forces: A → B → C → ¬A (contradiction with A)
        formula = CNFFormula(
            [
                Clause({Literal("A", True), Literal("B")}),  # ¬A ∨ B
                Clause({Literal("B", True), Literal("C")}),  # ¬B ∨ C
                Clause({Literal("C", True), Literal("A", True)}),  # ¬C ∨ ¬A
                Clause({Literal("A")}),  # A must be true
            ]
        )

        solver = DPLLSolver()
        result, model = solver.solve(formula)

        assert result == SATResult.UNSATISFIABLE
        assert model is None

    def test_check_rule_consistency_with_checker(self):
        """Test KnowledgeBaseChecker.check_rule_consistency()"""
        checker = KnowledgeBaseChecker()

        # Consistent rules: bird → can_fly, sparrow → bird
        rules = [
            ([Literal("bird")], Literal("can_fly")),
            ([Literal("sparrow")], Literal("bird")),
        ]

        is_consistent, model = checker.check_rule_consistency(rules)

        assert is_consistent
        assert model is not None

    def test_multiple_formula_consistency(self):
        """Test DPLLSolver.check_consistency() with multiple formulas"""
        solver = DPLLSolver()

        # Formula 1: A ∨ B
        formula1 = CNFFormula([Clause({Literal("A"), Literal("B")})])

        # Formula 2: ¬A ∨ C
        formula2 = CNFFormula([Clause({Literal("A", True), Literal("C")})])

        # Formula 3: ¬B ∨ C
        formula3 = CNFFormula([Clause({Literal("B", True), Literal("C")})])

        is_consistent, conflicting = solver.check_consistency(
            [formula1, formula2, formula3]
        )

        assert is_consistent
        assert conflicting is None


class TestContradictionDetection:
    """Test contradiction detection in knowledge bases"""

    def test_direct_contradiction(self):
        """Test detection of direct contradiction: X ∧ ¬X"""
        # CNF: (X) ∧ (¬X)
        formula = CNFFormula(
            [
                Clause({Literal("X")}),  # X must be true
                Clause({Literal("X", True)}),  # X must be false
            ]
        )

        solver = DPLLSolver()
        result, model = solver.solve(formula)

        assert result == SATResult.UNSATISFIABLE
        assert model is None

    def test_knowledge_base_contradiction_dog_animal_plant(self):
        """
        Test: 'Hund ist Tier', 'Hund ist Pflanze', 'Tier und Pflanze schließen sich aus'

        Encoding:
        - dog_is_animal: Hund ist Tier
        - dog_is_plant: Hund ist Pflanze
        - Constraint: ¬dog_is_animal ∨ ¬dog_is_plant (can't be both)
        """
        # Facts + Constraint
        formula = CNFFormula(
            [
                Clause({Literal("dog_is_animal")}),  # Hund ist Tier
                Clause({Literal("dog_is_plant")}),  # Hund ist Pflanze
                # Tier und Pflanze schließen sich aus
                Clause({Literal("dog_is_animal", True), Literal("dog_is_plant", True)}),
            ]
        )

        solver = DPLLSolver()
        result, model = solver.solve(formula)

        assert result == SATResult.UNSATISFIABLE
        assert model is None

    def test_find_conflicts_with_checker(self):
        """Test KnowledgeBaseChecker.find_conflicts()"""
        checker = KnowledgeBaseChecker()

        # Facts: bird, penguin
        facts = [Literal("bird"), Literal("penguin")]

        # Rules:
        # - bird → can_fly
        # - penguin → bird
        # - penguin → ¬can_fly (contradiction with bird → can_fly)
        rules = [
            ([Literal("bird")], Literal("can_fly")),
            ([Literal("penguin")], Literal("bird")),
            ([Literal("penguin")], Literal("can_fly", True)),  # ¬can_fly
        ]

        conflicts = checker.find_conflicts(facts, rules)

        assert len(conflicts) > 0
        assert "inconsistent" in conflicts[0].lower()

    def test_transitive_contradiction(self):
        """Test transitive contradiction: A → B, B → C, A, ¬C"""
        # Rules: A → B, B → C
        # Facts: A is true, C is false
        # This should be inconsistent
        formula = CNFFormula(
            [
                Clause({Literal("A", True), Literal("B")}),  # ¬A ∨ B
                Clause({Literal("B", True), Literal("C")}),  # ¬B ∨ C
                Clause({Literal("A")}),  # A is true
                Clause({Literal("C", True)}),  # C is false
            ]
        )

        solver = DPLLSolver()
        result, model = solver.solve(formula)

        assert result == SATResult.UNSATISFIABLE
        assert model is None

    def test_conflicting_formulas_detection(self):
        """Test detection of conflicting formula subset"""
        solver = DPLLSolver()

        # Formula 1: A (consistent)
        formula1 = CNFFormula([Clause({Literal("A")})])

        # Formula 2: ¬A (conflicts with formula1)
        formula2 = CNFFormula([Clause({Literal("A", True)})])

        # Formula 3: B ∨ C (consistent, no conflict)
        formula3 = CNFFormula([Clause({Literal("B"), Literal("C")})])

        is_consistent, conflicting = solver.check_consistency(
            [formula1, formula2, formula3]
        )

        assert not is_consistent
        assert conflicting is not None
        # Should identify formulas 0 and 1 as conflicting
        assert 0 in conflicting or 1 in conflicting


class TestKnightsAndKnavesSAT:
    """
    Test Knights and Knaves puzzle as SAT application.

    Purpose: Demonstrates SAT-Solver's ability to encode and solve
    logical puzzles using propositional logic, NOT as a primary
    feature but as a test of SAT capabilities.

    Encoding:
    - Variables: k_A, k_B, k_C (True = knight, False = knave)
    - Constraints: Encode statements using logical operators
    """

    def test_knights_and_knaves_puzzle(self):
        """
        Test classic Knights and Knaves puzzle:

        Three people (A, B, C). Knights always tell truth, Knaves always lie.
        - A says: "B is a knave"
        - B says: "A and C are both knights"
        - C says: "A is a knave"

        Expected solution: A=Knave, B=Knave, C=Knight
        """
        # Use pre-built puzzle from component_30
        formula = create_knights_and_knaves_problem()

        solver = DPLLSolver()
        result, model = solver.solve(formula)

        assert result == SATResult.SATISFIABLE
        assert model is not None

        # Verify expected solution
        assert model["k_A"] is True  # A is knight
        assert model["k_B"] is False  # B is knave
        assert model["k_C"] is False  # C is knave

    def test_simple_knights_knaves_two_people(self):
        """
        Test simpler Knights and Knaves with two people:

        - A says: "We are both of the same type" (both knights or both knaves)
        - B says: "We are of different types" (one knight, one knave)

        Solution: A=Knight, B=Knave
        """
        formula = CNFFormula([])

        k_A = Literal("k_A")
        k_B = Literal("k_B")

        # A says: "We are both of the same type" (k_A ↔ k_B)
        # If A is knight: statement is true → k_A ↔ k_B
        # If A is knave: statement is false → ¬(k_A ↔ k_B) = (k_A XOR k_B)
        # Encoding: k_A → (k_A ↔ k_B) AND ¬k_A → (k_A XOR k_B)

        # k_A → (k_A ↔ k_B) = k_A → ((¬k_A ∨ k_B) ∧ (k_A ∨ ¬k_B))
        # = (¬k_A ∨ (¬k_A ∨ k_B)) ∧ (¬k_A ∨ k_A ∨ ¬k_B)
        # = (¬k_A ∨ k_B) ∧ ⊤
        formula.add_clause(Clause({-k_A, k_B}))  # Clause 1
        formula.add_clause(Clause({-k_A, -k_B}))  # Clause 2: from full biconditional

        # ¬k_A → (k_A XOR k_B) = ¬k_A → ((k_A ∨ k_B) ∧ (¬k_A ∨ ¬k_B))
        # = (k_A ∨ k_A ∨ k_B) ∧ (k_A ∨ ¬k_A ∨ ¬k_B)
        # = (k_A ∨ k_B) ∧ ⊤
        formula.add_clause(Clause({k_A, k_B}))  # Clause 3

        # B says: "We are of different types" (k_A XOR k_B)
        # If B is knight: statement is true → k_A XOR k_B
        # If B is knave: statement is false → ¬(k_A XOR k_B) = (k_A ↔ k_B)

        # k_B → (k_A XOR k_B) = k_B → ((k_A ∨ k_B) ∧ (¬k_A ∨ ¬k_B))
        # = (¬k_B ∨ k_A ∨ k_B) ∧ (¬k_B ∨ ¬k_A ∨ ¬k_B)
        # = ⊤ ∧ (¬k_B ∨ ¬k_A)
        formula.add_clause(Clause({-k_B, -k_A}))  # Clause 4

        # ¬k_B → (k_A ↔ k_B) = ¬k_B → ((¬k_A ∨ k_B) ∧ (k_A ∨ ¬k_B))
        # = (k_B ∨ ¬k_A ∨ k_B) ∧ (k_B ∨ k_A ∨ ¬k_B)
        # = (k_B ∨ ¬k_A) ∧ ⊤
        formula.add_clause(Clause({k_B, -k_A}))  # Clause 5

        solver = DPLLSolver()
        result, model = solver.solve(formula)

        assert result == SATResult.SATISFIABLE
        assert model is not None

        # Expected solution (determined by solving the constraints)
        # The actual solution is A=Knave, B=Knight
        assert model.get("k_A", True) is False
        assert model.get("k_B", False) is True

    def test_knights_knaves_encoding_with_sat_encoder(self):
        """
        Test manual encoding of Knights and Knaves constraints using SATEncoder.

        Demonstrates how to use SATEncoder for building logical constraints.
        """
        formula = CNFFormula([])

        k_A = Literal("k_A")
        k_B = Literal("k_B")

        # A says: "We are both knights" (k_A ∧ k_B)
        # If A is knight: statement is true (k_A → (k_A ∧ k_B))
        #   Simplifies to: k_A → k_B
        # If A is knave: statement is false (¬k_A → ¬(k_A ∧ k_B))
        #   Which is always true (since ¬k_A makes k_A ∧ k_B false)

        # k_A → k_B: (¬k_A ∨ k_B)
        formula.add_clause(Clause({-k_A, k_B}))

        # B says: "A is a knave" (¬k_A)
        # If B is knight: A is knave (k_B → ¬k_A)
        # If B is knave: A is knight (¬k_B → k_A)
        # Equivalent: k_B ↔ ¬k_A
        formula.clauses.extend(SATEncoder.encode_iff(k_B, -k_A))

        solver = DPLLSolver()
        result, model = solver.solve(formula)

        assert result == SATResult.SATISFIABLE
        assert model is not None

        # From k_A → k_B and k_B ↔ ¬k_A:
        # If k_A is true: k_B must be true (from k_A → k_B)
        #   But k_B ↔ ¬k_A means k_B is false (contradiction)
        # So k_A must be false
        assert model["k_A"] is False


class TestGenericSATCapabilities:
    """Test generic SAT solver capabilities"""

    def test_sat_encoder_implication(self):
        """Test SATEncoder.encode_implication()"""
        # A → B
        clause = SATEncoder.encode_implication(Literal("A"), Literal("B"))

        # Should be: ¬A ∨ B
        assert Literal("A", True) in clause.literals or Literal("B") in clause.literals

    def test_sat_encoder_iff(self):
        """Test SATEncoder.encode_iff() for biconditional"""
        # A ↔ B
        clauses = SATEncoder.encode_iff(Literal("A"), Literal("B"))

        # Should be: (¬A ∨ B) ∧ (¬B ∨ A)
        assert len(clauses) == 2

        formula = CNFFormula(clauses)
        solver = DPLLSolver()
        result, model = solver.solve(formula)

        assert result == SATResult.SATISFIABLE
        # A and B must have same value
        if model:
            assert model.get("A") == model.get("B")

    def test_sat_encoder_exactly_one(self):
        """Test SATEncoder.encode_exactly_one()"""
        # Exactly one of {A, B, C} must be true
        literals = [Literal("A"), Literal("B"), Literal("C")]
        clauses = SATEncoder.encode_exactly_one(literals)

        formula = CNFFormula(clauses)
        solver = DPLLSolver()
        result, model = solver.solve(formula)

        assert result == SATResult.SATISFIABLE
        assert model is not None

        # Verify exactly one is true
        true_count = sum(
            [model.get("A", False), model.get("B", False), model.get("C", False)]
        )
        assert true_count == 1

    def test_sat_encoder_at_most_one(self):
        """Test SATEncoder.encode_at_most_one()"""
        # At most one of {A, B, C} can be true
        literals = [Literal("A"), Literal("B"), Literal("C")]
        clauses = SATEncoder.encode_at_most_one(literals)

        formula = CNFFormula(clauses)
        solver = DPLLSolver()
        result, model = solver.solve(formula)

        assert result == SATResult.SATISFIABLE
        assert model is not None

        # Verify at most one is true
        true_count = sum(
            [model.get("A", False), model.get("B", False), model.get("C", False)]
        )
        assert true_count <= 1

    def test_cnf_converter_implication(self):
        """Test CNFConverter with implication"""
        # A → B
        formula = PropositionalFormula.implies_formula(
            PropositionalFormula.variable_formula("A"),
            PropositionalFormula.variable_formula("B"),
        )

        cnf = CNFConverter.to_cnf(formula)

        # Should convert to: ¬A ∨ B
        assert len(cnf.clauses) >= 1

        solver = DPLLSolver()
        result, model = solver.solve(cnf)

        assert result == SATResult.SATISFIABLE

    def test_cnf_converter_iff(self):
        """Test CNFConverter with biconditional"""
        # A ↔ B
        formula = PropositionalFormula.iff_formula(
            PropositionalFormula.variable_formula("A"),
            PropositionalFormula.variable_formula("B"),
        )

        cnf = CNFConverter.to_cnf(formula)

        # Should convert to: (¬A ∨ B) ∧ (¬B ∨ A)
        assert len(cnf.clauses) == 2

        solver = DPLLSolver()
        result, model = solver.solve(cnf)

        assert result == SATResult.SATISFIABLE
        # A and B should have same value
        if model:
            assert model.get("A") == model.get("B")

    def test_simplified_sat_solver_api(self):
        """Test simplified SATSolver API"""
        # Use high-level SATSolver API
        solver = SATSolver(enable_proof=False)

        formula = CNFFormula(
            [
                Clause({Literal("X"), Literal("Y")}),
                Clause({Literal("X", True), Literal("Z")}),
            ]
        )

        model = solver.solve(formula)

        assert model is not None
        assert "X" in model or "Y" in model

    def test_unit_propagation_api(self):
        """Test SATSolver.unit_propagate()"""
        solver = SATSolver(enable_proof=False)

        # Formula: (X) ∧ (¬X ∨ Y)
        # Unit clause: X must be true
        # Unit propagation should infer Y = true
        formula = CNFFormula(
            [
                Clause({Literal("X")}),  # Unit clause
                Clause({Literal("X", True), Literal("Y")}),  # ¬X ∨ Y
            ]
        )

        assignment = {}
        simplified, extended = solver.unit_propagate(formula, assignment)

        assert "X" in extended
        assert extended["X"] is True

    def test_pure_literal_elimination_api(self):
        """Test SATSolver.pure_literal_elimination()"""
        solver = SATSolver(enable_proof=False)

        # Formula: (X ∨ Y) ∧ (X ∨ Z)
        # X appears only positively (pure literal)
        formula = CNFFormula(
            [Clause({Literal("X"), Literal("Y")}), Clause({Literal("X"), Literal("Z")})]
        )

        assignment = {}
        simplified, extended = solver.pure_literal_elimination(formula, assignment)

        # X should be assigned true (pure literal)
        assert "X" in extended
        assert extended["X"] is True


class TestProofTreeIntegration:
    """Test ProofTree integration with SAT solver"""

    def test_proof_tree_generation_sat(self):
        """Test that proof tree is generated for SAT problem"""
        solver = SATSolver(enable_proof=True)

        formula = CNFFormula(
            [
                Clause({Literal("A"), Literal("B")}),
                Clause({Literal("A", True), Literal("C")}),
            ]
        )

        model = solver.solve(formula)

        assert model is not None

        proof_tree = solver.get_proof_tree("SAT Solution")
        assert proof_tree is not None
        # Should have some proof steps
        all_steps = proof_tree.get_all_steps()
        assert len(all_steps) > 0

    def test_proof_tree_generation_unsat(self):
        """Test that proof tree is generated for UNSAT problem"""
        solver = SATSolver(enable_proof=True)

        # Contradiction: X ∧ ¬X
        formula = CNFFormula([Clause({Literal("X")}), Clause({Literal("X", True)})])

        model = solver.solve(formula)

        assert model is None  # UNSAT

        proof_tree = solver.get_proof_tree("SAT Solution")
        assert proof_tree is not None
        # Should contain contradiction step
        all_steps = proof_tree.get_all_steps()
        assert any(
            "Conflict" in step.rule_name
            or "UNSAT" in step.rule_name
            or "Conflict" in step.explanation_text
            or "UNSAT" in step.explanation_text
            for step in all_steps
        )

    def test_proof_disabled(self):
        """Test that proof tree is None when disabled"""
        solver = SATSolver(enable_proof=False)

        formula = CNFFormula([Clause({Literal("A"), Literal("B")})])

        model = solver.solve(formula)
        assert model is not None

        proof_tree = solver.get_proof_tree()
        assert proof_tree is None  # Proof disabled


class TestPerformanceAndEdgeCases:
    """Test performance and edge cases"""

    def test_empty_formula(self):
        """Test empty formula (always satisfiable)"""
        formula = CNFFormula([])

        solver = DPLLSolver()
        result, model = solver.solve(formula)

        assert result == SATResult.SATISFIABLE
        assert model is not None

    def test_single_variable(self):
        """Test single variable formula"""
        formula = CNFFormula([Clause({Literal("X")})])

        solver = DPLLSolver()
        result, model = solver.solve(formula)

        assert result == SATResult.SATISFIABLE
        assert model is not None
        assert model["X"] is True

    def test_large_clause(self):
        """Test clause with many literals"""
        # X1 ∨ X2 ∨ ... ∨ X10
        literals = {Literal(f"X{i}") for i in range(10)}
        formula = CNFFormula([Clause(literals)])

        solver = DPLLSolver()
        result, model = solver.solve(formula)

        assert result == SATResult.SATISFIABLE
        assert model is not None
        # At least one Xi should be true
        assert any(model.get(f"X{i}", False) for i in range(10))

    @pytest.mark.slow
    def test_moderately_complex_formula(self):
        """Test moderately complex formula (20 variables, 50 clauses)"""
        import random

        random.seed(42)

        clauses = []
        for _ in range(50):
            # Random 3-SAT clause
            vars_in_clause = random.sample([f"X{i}" for i in range(20)], 3)
            literals = {
                Literal(var, random.choice([True, False])) for var in vars_in_clause
            }
            clauses.append(Clause(literals))

        formula = CNFFormula(clauses)

        solver = DPLLSolver()
        result, model = solver.solve(formula)

        # Should find solution or determine unsatisfiable
        assert result in [SATResult.SATISFIABLE, SATResult.UNSATISFIABLE]
        # Verify solution if found
        if result == SATResult.SATISFIABLE:
            assert model is not None
            # Verify all clauses are satisfied
            for clause in formula.clauses:
                simplified = clause.simplify(model)
                assert simplified is None  # Clause is satisfied
