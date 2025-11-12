"""
Tests für ComparisonEngine und transitive Inferenz
"""

import pytest

from component_1_netzwerk_core import KonzeptNetzwerkCore
from component_52_arithmetic_reasoning import (
    ArithmeticEngine,
    ArithmeticResult,
    ComparisonEngine,
)


@pytest.fixture
def netzwerk():
    """Test-Netzwerk"""
    return KonzeptNetzwerkCore(
        uri="bolt://127.0.0.1:7687", user="neo4j", password="password"
    )


@pytest.fixture
def comparison_engine(netzwerk):
    """ComparisonEngine Fixture"""
    return ComparisonEngine(netzwerk)


@pytest.fixture
def arithmetic_engine(netzwerk):
    """ArithmeticEngine Fixture"""
    return ArithmeticEngine(netzwerk)


class TestComparisonBasics:
    """Tests für Basis-Vergleichsoperationen"""

    def test_less_than_true(self, comparison_engine):
        """Test: 3 < 5 = True"""
        result = comparison_engine.compare(3, 5, "<")
        assert result.value is True
        assert result.confidence == 1.0
        assert result.metadata["operator"] == "<"
        assert result.proof_tree is not None

    def test_less_than_false(self, comparison_engine):
        """Test: 5 < 3 = False"""
        result = comparison_engine.compare(5, 3, "<")
        assert result.value is False
        assert result.confidence == 1.0

    def test_greater_than_true(self, comparison_engine):
        """Test: 7 > 2 = True"""
        result = comparison_engine.compare(7, 2, ">")
        assert result.value is True

    def test_greater_than_false(self, comparison_engine):
        """Test: 2 > 7 = False"""
        result = comparison_engine.compare(2, 7, ">")
        assert result.value is False

    def test_equal_true(self, comparison_engine):
        """Test: 5 = 5 = True"""
        result = comparison_engine.compare(5, 5, "=")
        assert result.value is True

    def test_equal_false(self, comparison_engine):
        """Test: 5 = 3 = False"""
        result = comparison_engine.compare(5, 3, "=")
        assert result.value is False

    def test_less_equal_true_less(self, comparison_engine):
        """Test: 3 <= 5 = True"""
        result = comparison_engine.compare(3, 5, "<=")
        assert result.value is True

    def test_less_equal_true_equal(self, comparison_engine):
        """Test: 5 <= 5 = True"""
        result = comparison_engine.compare(5, 5, "<=")
        assert result.value is True

    def test_less_equal_false(self, comparison_engine):
        """Test: 7 <= 5 = False"""
        result = comparison_engine.compare(7, 5, "<=")
        assert result.value is False

    def test_greater_equal_true_greater(self, comparison_engine):
        """Test: 7 >= 3 = True"""
        result = comparison_engine.compare(7, 3, ">=")
        assert result.value is True

    def test_greater_equal_true_equal(self, comparison_engine):
        """Test: 5 >= 5 = True"""
        result = comparison_engine.compare(5, 5, ">=")
        assert result.value is True

    def test_greater_equal_false(self, comparison_engine):
        """Test: 3 >= 7 = False"""
        result = comparison_engine.compare(3, 7, ">=")
        assert result.value is False

    def test_invalid_operator(self, comparison_engine):
        """Test: Ungültiger Operator wirft ValueError"""
        with pytest.raises(ValueError, match="Unbekannter Vergleichsoperator"):
            comparison_engine.compare(3, 5, "!=")

    def test_float_comparison(self, comparison_engine):
        """Test: Vergleich mit Floats"""
        result = comparison_engine.compare(3.5, 5.2, "<")
        assert result.value is True

    def test_negative_numbers(self, comparison_engine):
        """Test: Vergleich mit negativen Zahlen"""
        result = comparison_engine.compare(-5, -3, "<")
        assert result.value is True

    def test_zero_comparison(self, comparison_engine):
        """Test: Vergleich mit Null"""
        result = comparison_engine.compare(0, 5, "<")
        assert result.value is True


class TestTransitiveInference:
    """Tests für transitive Inferenz"""

    def test_simple_transitive_less_than(self, comparison_engine):
        """Test: 3 < 5 ∧ 5 < 7 → 3 < 7"""
        relations = [(3, "<", 5), (5, "<", 7)]
        inferred = comparison_engine.transitive_inference(relations)
        assert (3, "<", 7) in inferred
        assert len(inferred) == 1

    def test_triple_transitive_chain(self, comparison_engine):
        """Test: 3 < 5 ∧ 5 < 7 ∧ 7 < 9 → 3<7, 3<9, 5<9"""
        relations = [(3, "<", 5), (5, "<", 7), (7, "<", 9)]
        inferred = comparison_engine.transitive_inference(relations)

        # Erwartete Ableitungen
        assert (3, "<", 7) in inferred
        assert (5, "<", 9) in inferred
        assert (3, "<", 9) in inferred
        assert len(inferred) == 3

    def test_simple_transitive_greater_than(self, comparison_engine):
        """Test: 7 > 5 ∧ 5 > 3 → 7 > 3"""
        relations = [(7, ">", 5), (5, ">", 3)]
        inferred = comparison_engine.transitive_inference(relations)
        assert (7, ">", 3) in inferred

    def test_less_equal_transitive(self, comparison_engine):
        """Test: 3 <= 5 ∧ 5 <= 7 → 3 <= 7"""
        relations = [(3, "<=", 5), (5, "<=", 7)]
        inferred = comparison_engine.transitive_inference(relations)
        assert (3, "<=", 7) in inferred

    def test_greater_equal_transitive(self, comparison_engine):
        """Test: 7 >= 5 ∧ 5 >= 3 → 7 >= 3"""
        relations = [(7, ">=", 5), (5, ">=", 3)]
        inferred = comparison_engine.transitive_inference(relations)
        assert (7, ">=", 3) in inferred

    def test_no_transitive_for_equal(self, comparison_engine):
        """Test: Gleichheit ist nicht transitiv in diesem Kontext"""
        relations = [(3, "=", 3), (3, "=", 3)]
        inferred = comparison_engine.transitive_inference(relations)
        assert len(inferred) == 0

    def test_mixed_operators_no_inference(self, comparison_engine):
        """Test: Gemischte Operatoren → keine Inferenz"""
        relations = [(3, "<", 5), (5, ">", 7)]
        inferred = comparison_engine.transitive_inference(relations)
        assert len(inferred) == 0

    def test_no_chain_no_inference(self, comparison_engine):
        """Test: Keine Kette → keine Inferenz"""
        relations = [(3, "<", 5), (7, "<", 9)]
        inferred = comparison_engine.transitive_inference(relations)
        assert len(inferred) == 0

    def test_duplicate_prevention(self, comparison_engine):
        """Test: Duplikate werden verhindert"""
        relations = [(3, "<", 5), (5, "<", 7), (3, "<", 7)]
        inferred = comparison_engine.transitive_inference(relations)
        # (3, "<", 7) ist bereits in relations, sollte nicht in inferred sein
        assert (3, "<", 7) not in inferred


class TestTransitiveProof:
    """Tests für Proof-Tree-Generierung bei transitiver Inferenz"""

    def test_transitive_proof_structure(self, comparison_engine):
        """Test: Proof Tree für transitive Inferenz hat korrekte Struktur"""
        relations = [(3, "<", 5), (5, "<", 7)]
        result = comparison_engine.build_transitive_proof(relations)

        assert isinstance(result, ArithmeticResult)
        assert result.value == [(3, "<", 7)]
        assert result.proof_tree is not None
        assert len(result.proof_tree.root_steps) == 1

    def test_transitive_proof_with_no_inference(self, comparison_engine):
        """Test: Proof Tree wenn keine Inferenz möglich"""
        relations = [(3, "<", 5), (7, "<", 9)]
        result = comparison_engine.build_transitive_proof(relations)

        assert result.value == []
        assert result.proof_tree is not None
        assert (
            "Keine neuen Relationen ableitbar"
            in result.proof_tree.root_steps[0].subgoals[0].subgoals[0].output
        )


class TestArithmeticEngineIntegration:
    """Tests für Integration in ArithmeticEngine"""

    def test_arithmetic_engine_compare(self, arithmetic_engine):
        """Test: compare() durch ArithmeticEngine"""
        result = arithmetic_engine.compare(3, 5, "<")
        assert result.value is True

    def test_arithmetic_engine_transitive_inference(self, arithmetic_engine):
        """Test: transitive_inference() durch ArithmeticEngine"""
        relations = [(3, "<", 5), (5, "<", 7)]
        result = arithmetic_engine.transitive_inference(relations)
        assert result.value == [(3, "<", 7)]


class TestEdgeCases:
    """Tests für Edge Cases"""

    def test_comparison_with_same_number(self, comparison_engine):
        """Test: Vergleich derselben Zahl"""
        assert comparison_engine.compare(5, 5, "=").value is True
        assert comparison_engine.compare(5, 5, "<").value is False
        assert comparison_engine.compare(5, 5, ">").value is False
        assert comparison_engine.compare(5, 5, "<=").value is True
        assert comparison_engine.compare(5, 5, ">=").value is True

    def test_transitive_with_empty_relations(self, comparison_engine):
        """Test: Leere Relationen-Liste"""
        inferred = comparison_engine.transitive_inference([])
        assert inferred == []

    def test_transitive_with_single_relation(self, comparison_engine):
        """Test: Einzelne Relation"""
        inferred = comparison_engine.transitive_inference([(3, "<", 5)])
        assert inferred == []

    def test_large_number_comparison(self, comparison_engine):
        """Test: Vergleich großer Zahlen"""
        result = comparison_engine.compare(1000000, 1000001, "<")
        assert result.value is True

    def test_comparison_proof_has_three_steps(self, comparison_engine):
        """Test: Proof Tree hat 3 Schritte (PREMISE, RULE_APPLICATION, CONCLUSION)"""
        result = comparison_engine.compare(3, 5, "<")

        # Root step (PREMISE)
        assert len(result.proof_tree.root_steps) == 1
        root = result.proof_tree.root_steps[0]

        # Zweiter Schritt (RULE_APPLICATION)
        assert len(root.subgoals) == 1
        rule_step = root.subgoals[0]

        # Dritter Schritt (CONCLUSION)
        assert len(rule_step.subgoals) == 1
        conclusion = rule_step.subgoals[0]

        assert "Ergebnis" in conclusion.explanation_text
