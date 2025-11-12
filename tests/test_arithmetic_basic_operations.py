"""
Tests für die Grundrechenarten im Arithmetik-Modul
Testet Addition, Subtraktion, Multiplikation und Division
"""

from decimal import Decimal
from fractions import Fraction

import pytest

from component_1_netzwerk_core import KonzeptNetzwerkCore
from component_17_proof_explanation import StepType
from component_52_arithmetic_reasoning import (
    Addition,
    ArithmeticEngine,
    Division,
)


@pytest.fixture
def netzwerk():
    """Erstellt ein Test-Netzwerk"""
    netzwerk = KonzeptNetzwerkCore(
        uri="bolt://127.0.0.1:7687",
        user="neo4j",
        password="password",
    )
    yield netzwerk
    netzwerk.close()


@pytest.fixture
def engine(netzwerk):
    """Erstellt eine ArithmeticEngine"""
    return ArithmeticEngine(netzwerk)


class TestAddition:
    """Tests für Addition"""

    def test_addition_basic(self, engine):
        """Test: Einfache Addition"""
        result = engine.calculate("+", 3, 5)

        assert result.value == 8
        assert result.confidence == 1.0
        assert result.metadata["operation"] == "addition"

    def test_addition_proof_tree(self, engine):
        """Test: Proof Tree für Addition"""
        result = engine.calculate("+", 3, 5)

        # Überprüfe Proof Tree Struktur
        assert result.proof_tree is not None
        assert "3 + 5" in result.proof_tree.query

        # Überprüfe Root Step (PREMISE)
        root_steps = result.proof_tree.root_steps
        assert len(root_steps) == 1
        assert root_steps[0].step_type == StepType.PREMISE
        assert "Gegeben" in root_steps[0].explanation_text

        # Überprüfe Child Steps (subgoals)
        subgoals = root_steps[0].subgoals
        assert len(subgoals) == 1
        assert subgoals[0].step_type == StepType.RULE_APPLICATION
        assert "Addition" in subgoals[0].explanation_text

        # Überprüfe Conclusion
        conclusion = subgoals[0].subgoals[0]
        assert conclusion.step_type == StepType.CONCLUSION
        assert "8" in conclusion.output

    def test_addition_negative_numbers(self, engine):
        """Test: Addition mit negativen Zahlen"""
        result = engine.calculate("+", -5, 3)
        assert result.value == -2

    def test_addition_floats(self, engine):
        """Test: Addition mit Floats"""
        result = engine.calculate("+", 3.5, 2.5)
        assert result.value == 6.0

    def test_addition_fractions(self, engine):
        """Test: Addition mit Fractions"""
        result = engine.calculate("+", Fraction(1, 2), Fraction(1, 3))
        assert result.value == Fraction(5, 6)

    def test_addition_invalid_operand(self, engine):
        """Test: Addition mit ungültigem Operanden"""
        with pytest.raises(ValueError, match="Validierung fehlgeschlagen"):
            engine.calculate("+", "invalid", 5)


class TestSubtraction:
    """Tests für Subtraktion"""

    def test_subtraction_basic(self, engine):
        """Test: Einfache Subtraktion"""
        result = engine.calculate("-", 10, 3)

        assert result.value == 7
        assert result.confidence == 1.0
        assert result.metadata["operation"] == "subtraction"

    def test_subtraction_proof_tree(self, engine):
        """Test: Proof Tree für Subtraktion"""
        result = engine.calculate("-", 10, 3)

        # Überprüfe Root Step (PREMISE)
        root_steps = result.proof_tree.root_steps
        assert len(root_steps) == 1
        assert root_steps[0].step_type == StepType.PREMISE
        assert "Minuend" in root_steps[0].explanation_text
        assert "Subtrahend" in root_steps[0].explanation_text

        # Überprüfe RULE_APPLICATION
        rule_step = root_steps[0].subgoals[0]
        assert rule_step.step_type == StepType.RULE_APPLICATION
        assert "Subtraktion" in rule_step.explanation_text

        # Überprüfe CONCLUSION
        conclusion = rule_step.subgoals[0]
        assert conclusion.step_type == StepType.CONCLUSION
        assert "7" in conclusion.output

    def test_subtraction_negative_result(self, engine):
        """Test: Subtraktion mit negativem Ergebnis"""
        result = engine.calculate("-", 3, 10)
        assert result.value == -7

    def test_subtraction_floats(self, engine):
        """Test: Subtraktion mit Floats"""
        result = engine.calculate("-", 7.5, 2.5)
        assert result.value == 5.0


class TestMultiplication:
    """Tests für Multiplikation"""

    def test_multiplication_basic(self, engine):
        """Test: Einfache Multiplikation"""
        result = engine.calculate("*", 4, 5)

        assert result.value == 20
        assert result.confidence == 1.0
        assert result.metadata["operation"] == "multiplication"

    def test_multiplication_proof_tree(self, engine):
        """Test: Proof Tree für Multiplikation"""
        result = engine.calculate("*", 4, 5)

        # Überprüfe Root Step (PREMISE)
        root_steps = result.proof_tree.root_steps
        assert len(root_steps) == 1
        assert root_steps[0].step_type == StepType.PREMISE
        assert "Faktoren" in root_steps[0].explanation_text

        # Überprüfe RULE_APPLICATION
        rule_step = root_steps[0].subgoals[0]
        assert rule_step.step_type == StepType.RULE_APPLICATION
        assert "Multiplikation" in rule_step.explanation_text

        # Überprüfe CONCLUSION
        conclusion = rule_step.subgoals[0]
        assert conclusion.step_type == StepType.CONCLUSION
        assert "20" in conclusion.output

    def test_multiplication_by_zero(self, engine):
        """Test: Multiplikation mit 0"""
        result = engine.calculate("*", 5, 0)
        assert result.value == 0

    def test_multiplication_negative(self, engine):
        """Test: Multiplikation mit negativen Zahlen"""
        result = engine.calculate("*", -3, 4)
        assert result.value == -12

    def test_multiplication_floats(self, engine):
        """Test: Multiplikation mit Floats"""
        result = engine.calculate("*", 2.5, 4.0)
        assert result.value == 10.0


class TestDivision:
    """Tests für Division"""

    def test_division_basic_integers(self, engine):
        """Test: Einfache Division mit ganzen Zahlen (Fraction)"""
        result = engine.calculate("/", 10, 2)

        # Bei Integer-Division sollte Fraction verwendet werden
        assert isinstance(result.value, Fraction)
        assert result.value == Fraction(5, 1)  # 10/2 = 5
        assert result.confidence == 1.0
        assert result.metadata["operation"] == "division"
        assert result.metadata["result_type"] == "Fraction"

    def test_division_proof_tree(self, engine):
        """Test: Proof Tree für Division mit CONSTRAINT_CHECK"""
        result = engine.calculate("/", 10, 2)

        # Überprüfe Root Step (PREMISE)
        root_steps = result.proof_tree.root_steps
        assert len(root_steps) == 1
        assert root_steps[0].step_type == StepType.PREMISE
        assert "Dividend" in root_steps[0].explanation_text
        assert "Divisor" in root_steps[0].explanation_text

        # Überprüfe Constraint Check (wichtig für Division!)
        constraint_step = root_steps[0].subgoals[0]
        assert constraint_step.step_type == StepType.PREMISE
        assert "≠ 0" in constraint_step.explanation_text
        assert constraint_step.metadata["constraint"] == "division_by_zero"
        assert constraint_step.metadata["check_passed"] is True

        # Überprüfe RULE_APPLICATION
        rule_step = constraint_step.subgoals[0]
        assert rule_step.step_type == StepType.RULE_APPLICATION
        assert "Division" in rule_step.explanation_text

        # Überprüfe CONCLUSION
        conclusion = rule_step.subgoals[0]
        assert conclusion.step_type == StepType.CONCLUSION
        assert "exakter Bruch" in conclusion.explanation_text

    def test_division_exact_fraction(self, engine):
        """Test: Division mit exaktem Bruch"""
        result = engine.calculate("/", 1, 3)

        # 1/3 sollte als Fraction gespeichert werden
        assert isinstance(result.value, Fraction)
        assert result.value == Fraction(1, 3)
        assert str(result.value) == "1/3"

    def test_division_simplification(self, engine):
        """Test: Division mit Vereinfachung"""
        result = engine.calculate("/", 6, 9)

        # 6/9 = 2/3 (Fraction vereinfacht automatisch)
        assert isinstance(result.value, Fraction)
        assert result.value == Fraction(2, 3)

    def test_division_by_zero(self, engine):
        """Test: Division durch 0 wirft ValueError"""
        with pytest.raises(ValueError, match="Division durch Null"):
            engine.calculate("/", 10, 0)

    def test_division_floats(self, engine):
        """Test: Division mit Floats (keine Fraction)"""
        result = engine.calculate("/", 7.5, 2.5)

        # Bei Float-Division kein Fraction
        assert isinstance(result.value, float)
        assert result.value == 3.0

    def test_division_negative(self, engine):
        """Test: Division mit negativen Zahlen"""
        result = engine.calculate("/", -10, 2)
        assert result.value == Fraction(-5, 1)

    def test_division_decimal(self, engine):
        """Test: Division mit Decimal"""
        result = engine.calculate("/", Decimal("10"), Decimal("3"))
        assert isinstance(result.value, Decimal)


class TestOperationRegistry:
    """Tests für die Operation Registry"""

    def test_registry_lookup_by_symbol(self, engine):
        """Test: Operation lookup via Symbol"""
        op = engine.registry.get("+")
        assert op is not None
        assert isinstance(op, Addition)

    def test_registry_lookup_by_name(self, engine):
        """Test: Operation lookup via Name"""
        op = engine.registry.get("division")
        assert op is not None
        assert isinstance(op, Division)

    def test_registry_list_operations(self, engine):
        """Test: Liste aller Operationen"""
        ops = engine.registry.list_operations()
        assert "+" in ops
        assert "-" in ops
        assert "*" in ops
        assert "/" in ops
        assert "addition" in ops
        assert "subtraction" in ops
        assert "multiplication" in ops
        assert "division" in ops


class TestEdgeCases:
    """Tests für Edge Cases"""

    def test_large_numbers(self, engine):
        """Test: Sehr große Zahlen"""
        result = engine.calculate("+", 10**100, 10**100)
        assert result.value == 2 * 10**100

    def test_very_small_fractions(self, engine):
        """Test: Sehr kleine Brüche"""
        result = engine.calculate("/", 1, 1000000)
        assert result.value == Fraction(1, 1000000)

    def test_mixed_types(self, engine):
        """Test: Gemischte Typen (int + float)"""
        result = engine.calculate("+", 5, 3.5)
        assert result.value == 8.5

    def test_zero_operations(self, engine):
        """Test: Operationen mit 0"""
        assert engine.calculate("+", 0, 0).value == 0
        assert engine.calculate("-", 0, 0).value == 0
        assert engine.calculate("*", 0, 5).value == 0

    def test_unknown_operation(self, engine):
        """Test: Unbekannte Operation"""
        with pytest.raises(ValueError, match="Unbekannte Operation"):
            engine.calculate("^", 2, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
