# tests/test_logic_puzzle_uniqueness_constraints.py
"""
Regression tests for uniqueness constraint generation in logic puzzle solver.

Ensures that assignment puzzles (entity-to-object bijections) generate
proper uniqueness constraints to prevent partial solutions.
"""

import pytest

from component_45_logic_puzzle_parser import LogicConditionParser


class TestUniquenessConstraints:
    """Test uniqueness constraint generation for assignment puzzles"""

    @pytest.fixture
    def parser(self):
        """Create LogicConditionParser instance"""
        return LogicConditionParser()

    def test_basic_assignment_puzzle_detection(self, parser):
        """Test detection of assignment puzzle pattern"""
        text = "Alex, Bob und Carol haben unterschiedliche Berufe: Lehrer, Arzt und Ingenieur."
        entities = ["Alex", "Bob", "Carol"]

        # Parse conditions (should detect assignment puzzle)
        conditions = parser.parse_conditions(text, entities)

        # Should generate uniqueness constraints
        # For 3 entities and 3 objects:
        # - 3 at-least-one (each entity has at least one object)
        # - 3*C(3,2) = 9 at-most-one (each entity has at most one object)
        # - 3 at-least-one (each object assigned to at least one entity)
        # - 3*C(3,2) = 9 at-most-one (each object assigned to at most one entity)
        # Total: 24 uniqueness constraints
        uniqueness_count = sum(
            1
            for c in conditions
            if c.condition_type in ["DISJUNCTION", "NEVER_BOTH"]
            and "hat_" in str(c.operands)
        )

        assert (
            uniqueness_count == 24
        ), f"Expected 24 uniqueness constraints, got {uniqueness_count}"

    def test_uniqueness_constraints_entity_filtering(self, parser):
        """Test that objects are not treated as entities in uniqueness constraints"""
        # This tests the fix for the bug where "Lehrer", "Arzt", "Ingenieur"
        # were incorrectly included as entities, causing over-constrained systems

        text = "Alex, Bob und Carol haben unterschiedliche Berufe: Lehrer, Arzt und Ingenieur."
        # Simulate entity extractor including objects as entities
        entities = ["Alex", "Bob", "Carol", "Lehrer", "Arzt", "Ingenieur"]

        conditions = parser.parse_conditions(text, entities)

        # Parser should filter out objects (Lehrer, Arzt, Ingenieur) from entities
        # So uniqueness constraints should only involve actual entities (Alex, Bob, Carol)
        uniqueness_count = sum(
            1
            for c in conditions
            if c.condition_type in ["DISJUNCTION", "NEVER_BOTH"]
            and "hat_" in str(c.operands)
        )

        # Should still be 24 (not 72 which would happen if 6 entities were used)
        assert (
            uniqueness_count == 24
        ), f"Expected 24 uniqueness constraints after filtering, got {uniqueness_count}"

    def test_variable_normalization_to_lowercase(self, parser):
        """Test that variable names are normalized to lowercase"""
        # This tests the fix for duplicate variables with different casings
        # (e.g., "Alex_hat_arzt" and "alex_hat_arzt")

        text = "Alex ist kein Arzt. Bob ist Lehrer."
        entities = ["Alex", "Bob"]

        parser.parse_conditions(text, entities)

        # All variables should be lowercase
        for var_name in parser.variables.keys():
            entity_part = var_name.split("_")[0]
            assert (
                entity_part.islower()
            ), f"Variable {var_name} should have lowercase entity part"

    def test_multi_operand_disjunction(self, parser):
        """Test that DISJUNCTION supports multiple operands (not just 2)"""
        text = "Alex, Bob und Carol haben unterschiedliche Berufe: Lehrer, Arzt und Ingenieur."
        entities = ["Alex", "Bob", "Carol"]

        conditions = parser.parse_conditions(text, entities)

        # Find at-least-one constraints (e.g., alex_hat_lehrer OR alex_hat_arzt OR alex_hat_ingenieur)
        disjunctions = [c for c in conditions if c.condition_type == "DISJUNCTION"]

        # Should have 6 disjunctions (3 for entities, 3 for objects)
        assert (
            len(disjunctions) == 6
        ), f"Expected 6 DISJUNCTION conditions, got {len(disjunctions)}"

        # Each should have 3 operands
        for disj in disjunctions:
            assert (
                len(disj.operands) == 3
            ), f"Expected 3 operands in DISJUNCTION, got {len(disj.operands)}"

    def test_never_both_constraints(self, parser):
        """Test generation of NEVER_BOTH constraints for mutual exclusivity"""
        text = "Alex, Bob und Carol haben unterschiedliche Berufe: Lehrer, Arzt und Ingenieur."
        entities = ["Alex", "Bob", "Carol"]

        conditions = parser.parse_conditions(text, entities)

        # Count NEVER_BOTH constraints
        never_both_count = sum(
            1 for c in conditions if c.condition_type == "NEVER_BOTH"
        )

        # Should have 18 NEVER_BOTH constraints:
        # - 3 entities * C(3,2) = 9 (each entity cannot have two objects)
        # - 3 objects * C(3,2) = 9 (each object cannot be assigned to two entities)
        assert (
            never_both_count == 18
        ), f"Expected 18 NEVER_BOTH constraints, got {never_both_count}"

    def test_assignment_puzzle_indicators(self, parser):
        """Test detection of various assignment puzzle indicators"""
        test_cases = [
            "Alex und Bob haben unterschiedliche Jobs: X und Y.",  # "unterschiedliche"
            "Alex und Bob haben verschiedene Jobs: X und Y.",  # "verschiedene"
            "Wer hat welchen Job?",  # "wer hat"
            "Jobs: X, Y und Z",  # colon enumeration
        ]

        for text in test_cases:
            is_assignment = parser._is_assignment_puzzle(text, ["Alex", "Bob"])
            assert is_assignment, f"Failed to detect assignment puzzle in: '{text}'"

    def test_non_assignment_puzzle(self, parser):
        """Test that non-assignment puzzles don't trigger uniqueness constraints"""
        text = "Alex ist größer als Bob. Bob ist größer als Carol."
        entities = ["Alex", "Bob", "Carol"]

        conditions = parser.parse_conditions(text, entities)

        # Should NOT generate uniqueness constraints (no assignment pattern)
        uniqueness_count = sum(
            1
            for c in conditions
            if c.condition_type in ["DISJUNCTION", "NEVER_BOTH"]
            and "hat_" in str(c.operands)
        )

        assert (
            uniqueness_count == 0
        ), f"Non-assignment puzzle should not generate uniqueness constraints, got {uniqueness_count}"
