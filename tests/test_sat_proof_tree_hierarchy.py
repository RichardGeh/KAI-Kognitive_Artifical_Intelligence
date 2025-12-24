"""
Test hierarchical ProofTree structure in SAT solver (Issue #6 fix verification)

This test verifies that the SAT solver creates proper hierarchical ProofTree structures
with depth 4-10 instead of flat structures with depth 1.

Before fix: All proof steps added as root_steps (flat array, depth=1)
After fix: Hierarchical nesting with proper parent-child relationships (depth 3-6)
"""

from component_30_sat_solver_core import Clause, CNFFormula, DPLLSolver, Literal


class TestSATProofTreeHierarchy:
    """Test suite for hierarchical ProofTree structure"""

    def test_proof_tree_has_single_root(self):
        """Verify ProofTree has 1 root step, not 19+ flat steps"""
        clauses = [
            Clause({Literal("A", False), Literal("B", False)}),
            Clause({Literal("A", True), Literal("C", False)}),
            Clause({Literal("B", True), Literal("C", False)}),
        ]
        formula = CNFFormula(clauses)

        solver = DPLLSolver(enable_proof=True)
        result, _ = solver.solve(formula)

        assert result.value == "satisfiable"

        proof_tree = solver.get_proof_tree()
        assert proof_tree is not None, "ProofTree should be generated"

        # CRITICAL: Must have 1 root (not 19+ flat roots)
        num_roots = len(proof_tree.root_steps)
        assert (
            num_roots == 1
        ), f"Expected 1 root step, got {num_roots} (flat structure bug)"

    def test_proof_tree_depth_hierarchical(self):
        """Verify ProofTree has depth >= 3 (hierarchical, not flat)"""
        clauses = [
            Clause({Literal("A", False), Literal("B", False)}),
            Clause({Literal("A", True), Literal("C", False)}),
            Clause({Literal("B", True), Literal("C", False)}),
        ]
        formula = CNFFormula(clauses)

        solver = DPLLSolver(enable_proof=True)
        solver.solve(formula)

        proof_tree = solver.get_proof_tree()
        tree_dict = proof_tree.to_dict()

        # Calculate depth
        def calculate_depth(step_dict, current=1):
            subgoals = step_dict.get("subgoals", [])
            if not subgoals:
                return current
            return max(calculate_depth(sg, current + 1) for sg in subgoals)

        depth = max(calculate_depth(root) for root in tree_dict["root_steps"])

        # CRITICAL: Depth must be >= 3 (hierarchical structure)
        # Before fix: depth was 1 (all steps at root level)
        # After fix: depth should be 3-6 depending on problem complexity
        assert (
            depth >= 3
        ), f"ProofTree depth {depth} too shallow (expected >= 3 for hierarchical structure)"

    def test_proof_tree_nesting_with_branching(self):
        """Verify branching creates deeper nesting (depth >= 4)"""
        # Problem that requires branching
        clauses = [
            Clause({Literal("A", False), Literal("B", False)}),
            Clause({Literal("A", True), Literal("C", False)}),
            Clause({Literal("B", True), Literal("C", True)}),
            Clause({Literal("A", False), Literal("C", True)}),
        ]
        formula = CNFFormula(clauses)

        solver = DPLLSolver(enable_proof=True)
        solver.solve(formula)

        proof_tree = solver.get_proof_tree()
        tree_dict = proof_tree.to_dict()

        def calculate_depth(step_dict, current=1):
            subgoals = step_dict.get("subgoals", [])
            if not subgoals:
                return current
            return max(calculate_depth(sg, current + 1) for sg in subgoals)

        depth = max(calculate_depth(root) for root in tree_dict["root_steps"])

        # With branching, expect depth >= 4
        # Structure: Root → Branch → Try Branch → Unit Propagation → Assignments
        assert (
            depth >= 4
        ), f"ProofTree depth {depth} too shallow for branching (expected >= 4)"

    def test_proof_tree_has_phase_organization(self):
        """Verify proof steps are organized by phase (unit prop, pure literal, branching)"""
        clauses = [
            Clause({Literal("A", False), Literal("B", False)}),
            Clause({Literal("A", True), Literal("C", False)}),
        ]
        formula = CNFFormula(clauses)

        solver = DPLLSolver(enable_proof=True)
        solver.solve(formula)

        proof_tree = solver.get_proof_tree()
        tree_dict = proof_tree.to_dict()

        # Root step should be "SAT Solving"
        root = tree_dict["root_steps"][0]
        assert (
            "SAT Solving" in root["rule_name"]
            or "SAT Solving" in root["explanation_text"]
        )

        # Should have phase steps as children (e.g., "Unit Propagation Phase", "Pure Literal Elimination Phase")
        def collect_all_steps(step_dict):
            steps = [step_dict]
            for subgoal in step_dict.get("subgoals", []):
                steps.extend(collect_all_steps(subgoal))
            return steps

        all_steps = collect_all_steps(root)
        step_descriptions = [
            s.get("rule_name", "") + s.get("explanation_text", "") for s in all_steps
        ]

        # Check that phase steps exist
        has_phase_organization = any(
            "Phase" in desc
            or "Propagation" in desc
            or "Elimination" in desc
            or "Branch" in desc
            for desc in step_descriptions
        )

        assert (
            has_phase_organization
        ), "ProofTree should have phase-organized steps (e.g., 'Unit Propagation Phase')"

    def test_proof_tree_no_flat_structure_warning(self, caplog):
        """Verify no flat structure warnings are logged"""
        import logging

        caplog.set_level(logging.WARNING)

        clauses = [
            Clause({Literal("A", False), Literal("B", False)}),
            Clause({Literal("A", True), Literal("C", False)}),
        ]
        formula = CNFFormula(clauses)

        solver = DPLLSolver(enable_proof=True)
        solver.solve(formula)
        solver.get_proof_tree()

        # Should NOT have warnings about flat structure
        flat_warnings = [
            record
            for record in caplog.records
            if "flat ProofTree" in record.message
            or "expected hierarchical structure" in record.message.lower()
        ]

        assert (
            len(flat_warnings) == 0
        ), f"Should not have flat structure warnings: {[r.message for r in flat_warnings]}"

    def test_proof_tree_total_steps_vs_roots(self):
        """Verify total steps >> root steps (indicates nesting)"""
        clauses = [
            Clause({Literal("A", False), Literal("B", False)}),
            Clause({Literal("A", True), Literal("C", False)}),
            Clause({Literal("B", True), Literal("C", False)}),
        ]
        formula = CNFFormula(clauses)

        solver = DPLLSolver(enable_proof=True)
        solver.solve(formula)

        proof_tree = solver.get_proof_tree()

        num_roots = len(proof_tree.root_steps)
        total_steps = len(solver.proof_steps)

        # Total steps should be much larger than root steps (indicates nesting)
        # Before fix: num_roots == total_steps (all steps at root level)
        # After fix: num_roots == 1, total_steps > 5
        assert (
            total_steps > num_roots
        ), f"Total steps ({total_steps}) should be > root steps ({num_roots})"
        assert num_roots == 1, f"Should have exactly 1 root step, got {num_roots}"
        assert (
            total_steps >= 5
        ), f"Should have at least 5 total steps (showing actual work), got {total_steps}"
