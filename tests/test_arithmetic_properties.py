"""
Tests für PropertyChecker (gerade, ungerade, prim, Teiler)
"""

import pytest

from component_1_netzwerk_core import KonzeptNetzwerkCore
from component_52_arithmetic_reasoning import (
    ArithmeticEngine,
    PropertyChecker,
)


@pytest.fixture
def netzwerk():
    """Test-Netzwerk"""
    return KonzeptNetzwerkCore(
        uri="bolt://127.0.0.1:7687", user="neo4j", password="password"
    )


@pytest.fixture
def property_checker(netzwerk):
    """PropertyChecker Fixture"""
    return PropertyChecker(netzwerk)


@pytest.fixture
def arithmetic_engine(netzwerk):
    """ArithmeticEngine Fixture"""
    return ArithmeticEngine(netzwerk)


class TestEvenOdd:
    """Tests für gerade/ungerade Prüfung"""

    def test_is_even_true(self, property_checker):
        """Test: 4 ist gerade"""
        result = property_checker.is_even(4)
        assert result.value is True
        assert result.confidence == 1.0
        assert result.metadata["property"] == "even"
        assert result.proof_tree is not None

    def test_is_even_false(self, property_checker):
        """Test: 5 ist nicht gerade"""
        result = property_checker.is_even(5)
        assert result.value is False
        assert result.metadata["property"] == "odd"

    def test_is_even_zero(self, property_checker):
        """Test: 0 ist gerade"""
        result = property_checker.is_even(0)
        assert result.value is True

    def test_is_even_negative(self, property_checker):
        """Test: -4 ist gerade"""
        result = property_checker.is_even(-4)
        assert result.value is True

    def test_is_even_large_number(self, property_checker):
        """Test: 1000000 ist gerade"""
        result = property_checker.is_even(1000000)
        assert result.value is True

    def test_is_odd_true(self, property_checker):
        """Test: 5 ist ungerade"""
        result = property_checker.is_odd(5)
        assert result.value is True
        assert result.metadata["property"] == "odd"

    def test_is_odd_false(self, property_checker):
        """Test: 4 ist nicht ungerade"""
        result = property_checker.is_odd(4)
        assert result.value is False
        assert result.metadata["property"] == "even"

    def test_is_odd_negative(self, property_checker):
        """Test: -5 ist ungerade"""
        result = property_checker.is_odd(-5)
        assert result.value is True

    def test_is_even_with_float_raises_error(self, property_checker):
        """Test: Float wirft ValueError"""
        with pytest.raises(ValueError, match="is_even benötigt Integer"):
            property_checker.is_even(4.5)

    def test_is_odd_with_float_raises_error(self, property_checker):
        """Test: Float wirft ValueError bei is_odd"""
        with pytest.raises(ValueError, match="is_even benötigt Integer"):
            property_checker.is_odd(5.5)


class TestPrimeNumbers:
    """Tests für Primzahl-Prüfung"""

    def test_is_prime_2(self, property_checker):
        """Test: 2 ist Primzahl (kleinste Primzahl)"""
        result = property_checker.is_prime(2)
        assert result.value is True
        assert result.metadata["property"] == "prime"

    def test_is_prime_3(self, property_checker):
        """Test: 3 ist Primzahl"""
        result = property_checker.is_prime(3)
        assert result.value is True

    def test_is_prime_5(self, property_checker):
        """Test: 5 ist Primzahl"""
        result = property_checker.is_prime(5)
        assert result.value is True

    def test_is_prime_7(self, property_checker):
        """Test: 7 ist Primzahl"""
        result = property_checker.is_prime(7)
        assert result.value is True

    def test_is_prime_11(self, property_checker):
        """Test: 11 ist Primzahl"""
        result = property_checker.is_prime(11)
        assert result.value is True

    def test_is_prime_13(self, property_checker):
        """Test: 13 ist Primzahl"""
        result = property_checker.is_prime(13)
        assert result.value is True

    def test_is_prime_17(self, property_checker):
        """Test: 17 ist Primzahl"""
        result = property_checker.is_prime(17)
        assert result.value is True

    def test_is_prime_19(self, property_checker):
        """Test: 19 ist Primzahl"""
        result = property_checker.is_prime(19)
        assert result.value is True

    def test_is_prime_23(self, property_checker):
        """Test: 23 ist Primzahl"""
        result = property_checker.is_prime(23)
        assert result.value is True

    def test_is_not_prime_1(self, property_checker):
        """Test: 1 ist keine Primzahl"""
        result = property_checker.is_prime(1)
        assert result.value is False
        assert result.metadata["property"] == "composite"

    def test_is_not_prime_0(self, property_checker):
        """Test: 0 ist keine Primzahl"""
        result = property_checker.is_prime(0)
        assert result.value is False

    def test_is_not_prime_negative(self, property_checker):
        """Test: -5 ist keine Primzahl"""
        result = property_checker.is_prime(-5)
        assert result.value is False

    def test_is_not_prime_4(self, property_checker):
        """Test: 4 ist keine Primzahl (2*2)"""
        result = property_checker.is_prime(4)
        assert result.value is False

    def test_is_not_prime_6(self, property_checker):
        """Test: 6 ist keine Primzahl (2*3)"""
        result = property_checker.is_prime(6)
        assert result.value is False

    def test_is_not_prime_8(self, property_checker):
        """Test: 8 ist keine Primzahl (2*4)"""
        result = property_checker.is_prime(8)
        assert result.value is False

    def test_is_not_prime_9(self, property_checker):
        """Test: 9 ist keine Primzahl (3*3)"""
        result = property_checker.is_prime(9)
        assert result.value is False

    def test_is_not_prime_10(self, property_checker):
        """Test: 10 ist keine Primzahl (2*5)"""
        result = property_checker.is_prime(10)
        assert result.value is False

    def test_is_not_prime_15(self, property_checker):
        """Test: 15 ist keine Primzahl (3*5)"""
        result = property_checker.is_prime(15)
        assert result.value is False

    def test_is_not_prime_21(self, property_checker):
        """Test: 21 ist keine Primzahl (3*7)"""
        result = property_checker.is_prime(21)
        assert result.value is False

    def test_is_prime_large(self, property_checker):
        """Test: 97 ist Primzahl (größere Primzahl)"""
        result = property_checker.is_prime(97)
        assert result.value is True

    def test_is_not_prime_large(self, property_checker):
        """Test: 100 ist keine Primzahl"""
        result = property_checker.is_prime(100)
        assert result.value is False

    def test_is_prime_with_float_raises_error(self, property_checker):
        """Test: Float wirft ValueError"""
        with pytest.raises(ValueError, match="is_prime benötigt Integer"):
            property_checker.is_prime(5.5)

    def test_prime_proof_structure(self, property_checker):
        """Test: Proof Tree für Primzahl hat korrekte Struktur"""
        result = property_checker.is_prime(7)
        assert result.proof_tree is not None
        assert len(result.proof_tree.root_steps) == 1

        # Prüfe Schritte (PREMISE → RULE_APPLICATION → CONCLUSION)
        root = result.proof_tree.root_steps[0]
        assert len(root.subgoals) == 1
        assert len(root.subgoals[0].subgoals) == 1


class TestDivisors:
    """Tests für Teiler-Berechnung"""

    def test_divisors_of_1(self, property_checker):
        """Test: Teiler von 1 = [1]"""
        result = property_checker.find_divisors(1)
        assert result.value == [1]
        assert result.metadata["count"] == 1

    def test_divisors_of_2(self, property_checker):
        """Test: Teiler von 2 = [1, 2]"""
        result = property_checker.find_divisors(2)
        assert result.value == [1, 2]
        assert result.metadata["count"] == 2

    def test_divisors_of_6(self, property_checker):
        """Test: Teiler von 6 = [1, 2, 3, 6]"""
        result = property_checker.find_divisors(6)
        assert result.value == [1, 2, 3, 6]
        assert result.metadata["count"] == 4

    def test_divisors_of_12(self, property_checker):
        """Test: Teiler von 12 = [1, 2, 3, 4, 6, 12]"""
        result = property_checker.find_divisors(12)
        assert result.value == [1, 2, 3, 4, 6, 12]
        assert result.metadata["count"] == 6

    def test_divisors_of_prime(self, property_checker):
        """Test: Teiler von Primzahl = [1, p]"""
        result = property_checker.find_divisors(7)
        assert result.value == [1, 7]
        assert result.metadata["count"] == 2

    def test_divisors_of_24(self, property_checker):
        """Test: Teiler von 24 = [1, 2, 3, 4, 6, 8, 12, 24]"""
        result = property_checker.find_divisors(24)
        assert result.value == [1, 2, 3, 4, 6, 8, 12, 24]
        assert result.metadata["count"] == 8

    def test_divisors_of_negative(self, property_checker):
        """Test: Teiler von -6 = [1, 2, 3, 6] (Betrag)"""
        result = property_checker.find_divisors(-6)
        assert result.value == [1, 2, 3, 6]

    def test_divisors_of_zero_raises_error(self, property_checker):
        """Test: Teiler von 0 wirft ValueError"""
        with pytest.raises(ValueError, match="0 hat unendlich viele Teiler"):
            property_checker.find_divisors(0)

    def test_divisors_with_float_raises_error(self, property_checker):
        """Test: Float wirft ValueError"""
        with pytest.raises(ValueError, match="find_divisors benötigt Integer"):
            property_checker.find_divisors(6.5)

    def test_divisors_proof_structure(self, property_checker):
        """Test: Proof Tree für Teiler hat korrekte Struktur"""
        result = property_checker.find_divisors(12)
        assert result.proof_tree is not None
        assert len(result.proof_tree.root_steps) == 1

    def test_divisors_of_perfect_square(self, property_checker):
        """Test: Teiler von Quadratzahl 16 = [1, 2, 4, 8, 16]"""
        result = property_checker.find_divisors(16)
        assert result.value == [1, 2, 4, 8, 16]
        assert result.metadata["count"] == 5

    def test_divisors_of_100(self, property_checker):
        """Test: Teiler von 100"""
        result = property_checker.find_divisors(100)
        expected = [1, 2, 4, 5, 10, 20, 25, 50, 100]
        assert result.value == expected
        assert result.metadata["count"] == 9


class TestArithmeticEngineIntegration:
    """Tests für Integration in ArithmeticEngine"""

    def test_check_property_even(self, arithmetic_engine):
        """Test: check_property() für gerade"""
        result = arithmetic_engine.check_property(4, "even")
        assert result.value is True

    def test_check_property_odd(self, arithmetic_engine):
        """Test: check_property() für ungerade"""
        result = arithmetic_engine.check_property(5, "odd")
        assert result.value is True

    def test_check_property_prime(self, arithmetic_engine):
        """Test: check_property() für Primzahl"""
        result = arithmetic_engine.check_property(7, "prime")
        assert result.value is True

    def test_check_property_invalid(self, arithmetic_engine):
        """Test: Ungültige Eigenschaft wirft ValueError"""
        with pytest.raises(ValueError, match="Unbekannte Eigenschaft"):
            arithmetic_engine.check_property(5, "invalid")

    def test_find_divisors_integration(self, arithmetic_engine):
        """Test: find_divisors() durch ArithmeticEngine"""
        result = arithmetic_engine.find_divisors(12)
        assert result.value == [1, 2, 3, 4, 6, 12]


class TestCombinedProperties:
    """Tests für kombinierte Eigenschaften"""

    def test_even_and_prime(self, property_checker):
        """Test: 2 ist gerade UND prim (einzige gerade Primzahl)"""
        even_result = property_checker.is_even(2)
        prime_result = property_checker.is_prime(2)
        assert even_result.value is True
        assert prime_result.value is True

    def test_odd_primes(self, property_checker):
        """Test: Alle Primzahlen > 2 sind ungerade"""
        primes = [3, 5, 7, 11, 13, 17, 19, 23]
        for p in primes:
            odd_result = property_checker.is_odd(p)
            prime_result = property_checker.is_prime(p)
            assert odd_result.value is True, f"{p} sollte ungerade sein"
            assert prime_result.value is True, f"{p} sollte prim sein"

    def test_prime_has_two_divisors(self, property_checker):
        """Test: Primzahlen haben genau 2 Teiler"""
        primes = [2, 3, 5, 7, 11, 13]
        for p in primes:
            divisor_result = property_checker.find_divisors(p)
            assert len(divisor_result.value) == 2, f"{p} sollte 2 Teiler haben"
            assert divisor_result.value == [1, p]

    def test_composite_has_more_than_two_divisors(self, property_checker):
        """Test: Zusammengesetzte Zahlen haben > 2 Teiler"""
        composites = [4, 6, 8, 9, 10, 12, 15]
        for c in composites:
            divisor_result = property_checker.find_divisors(c)
            assert len(divisor_result.value) > 2, f"{c} sollte mehr als 2 Teiler haben"


class TestEdgeCases:
    """Tests für Edge Cases"""

    def test_even_one(self, property_checker):
        """Test: 1 ist ungerade"""
        result = property_checker.is_even(1)
        assert result.value is False

    def test_prime_hundred(self, property_checker):
        """Test: 100 ist keine Primzahl"""
        result = property_checker.is_prime(100)
        assert result.value is False

    def test_divisors_order(self, property_checker):
        """Test: Teiler sind aufsteigend sortiert"""
        result = property_checker.find_divisors(24)
        assert result.value == sorted(result.value)

    def test_divisors_completeness(self, property_checker):
        """Test: Alle Teiler werden gefunden"""
        result = property_checker.find_divisors(20)
        # 20 = 2^2 * 5, Teiler: 1, 2, 4, 5, 10, 20
        for d in result.value:
            assert 20 % d == 0, f"{d} sollte 20 teilen"


class TestPerformance:
    """Tests für Performance (größere Zahlen)"""

    def test_prime_check_large_prime(self, property_checker):
        """Test: Große Primzahl (997)"""
        result = property_checker.is_prime(997)
        assert result.value is True

    def test_prime_check_large_composite(self, property_checker):
        """Test: Große zusammengesetzte Zahl (1000)"""
        result = property_checker.is_prime(1000)
        assert result.value is False

    def test_divisors_of_large_number(self, property_checker):
        """Test: Teiler einer großen Zahl (nicht zu groß wegen Performance)"""
        result = property_checker.find_divisors(120)
        # 120 = 2^3 * 3 * 5 hat viele Teiler
        assert len(result.value) == 16
