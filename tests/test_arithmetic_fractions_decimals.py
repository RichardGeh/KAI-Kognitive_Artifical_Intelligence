"""
Tests für Bruchrechnung und Dezimalzahlen (Phase 3)
Testet RationalArithmetic und DecimalArithmetic
"""

from decimal import Decimal, getcontext
from fractions import Fraction

import pytest

from component_52_arithmetic_reasoning import (
    DecimalArithmetic,
    RationalArithmetic,
)


class TestRationalArithmetic:
    """Tests für Bruchrechnung"""

    def setup_method(self):
        """Setup für jeden Test"""
        self.rational = RationalArithmetic()

    def test_add_fractions(self):
        """Test: Brüche addieren"""
        # 1/2 + 1/3 = 5/6
        result = self.rational.add(Fraction(1, 2), Fraction(1, 3))
        assert result == Fraction(5, 6)

    def test_add_fractions_same_denominator(self):
        """Test: Brüche mit gleichem Nenner addieren"""
        # 1/4 + 2/4 = 3/4
        result = self.rational.add(Fraction(1, 4), Fraction(2, 4))
        assert result == Fraction(3, 4)

    def test_subtract_fractions(self):
        """Test: Brüche subtrahieren"""
        # 3/4 - 1/2 = 1/4
        result = self.rational.subtract(Fraction(3, 4), Fraction(1, 2))
        assert result == Fraction(1, 4)

    def test_multiply_fractions(self):
        """Test: Brüche multiplizieren"""
        # 2/3 * 3/4 = 1/2
        result = self.rational.multiply(Fraction(2, 3), Fraction(3, 4))
        assert result == Fraction(1, 2)

    def test_divide_fractions(self):
        """Test: Brüche dividieren"""
        # 1/2 : 1/4 = 2
        result = self.rational.divide(Fraction(1, 2), Fraction(1, 4))
        assert result == Fraction(2, 1)

    def test_divide_by_zero(self):
        """Test: Division durch Null schlägt fehl"""
        with pytest.raises(ValueError, match="Division durch Null"):
            self.rational.divide(Fraction(1, 2), Fraction(0, 1))

    def test_simplify(self):
        """Test: Bruch kürzen (automatisch durch Fraction)"""
        # 6/8 wird automatisch zu 3/4 gekürzt
        fraction = Fraction(6, 8)
        result = self.rational.simplify(fraction)
        assert result == Fraction(3, 4)

    def test_to_mixed_number_improper(self):
        """Test: Unechter Bruch zu gemischter Zahl"""
        # 7/3 → 2 und 1/3
        whole, fraction = self.rational.to_mixed_number(Fraction(7, 3))
        assert whole == 2
        assert fraction == Fraction(1, 3)

    def test_to_mixed_number_proper(self):
        """Test: Echter Bruch zu gemischter Zahl"""
        # 2/3 → 0 und 2/3
        whole, fraction = self.rational.to_mixed_number(Fraction(2, 3))
        assert whole == 0
        assert fraction == Fraction(2, 3)

    def test_to_mixed_number_exact(self):
        """Test: Ganzzahl zu gemischter Zahl"""
        # 6/3 = 2 → 2 und 0/3
        whole, fraction = self.rational.to_mixed_number(Fraction(6, 3))
        assert whole == 2
        assert fraction == Fraction(0, 3)

    def test_from_mixed_number(self):
        """Test: Gemischte Zahl zu Bruch"""
        # 2 und 1/3 → 7/3
        result = self.rational.from_mixed_number(2, Fraction(1, 3))
        assert result == Fraction(7, 3)

    def test_gcd(self):
        """Test: Größter gemeinsamer Teiler"""
        assert self.rational.gcd(12, 8) == 4
        assert self.rational.gcd(15, 25) == 5
        assert self.rational.gcd(7, 13) == 1  # Teilerfremd

    def test_lcm(self):
        """Test: Kleinstes gemeinsames Vielfaches"""
        assert self.rational.lcm(4, 6) == 12
        assert self.rational.lcm(3, 7) == 21
        assert self.rational.lcm(12, 18) == 36

    def test_lcm_with_zero(self):
        """Test: LCM mit Null"""
        assert self.rational.lcm(0, 5) == 0
        assert self.rational.lcm(5, 0) == 0

    def test_compare_equal(self):
        """Test: Brüche vergleichen (gleich)"""
        assert self.rational.compare(Fraction(1, 2), Fraction(2, 4), "=")
        assert self.rational.compare(Fraction(1, 2), Fraction(2, 4), "==")

    def test_compare_less_than(self):
        """Test: Brüche vergleichen (kleiner)"""
        assert self.rational.compare(Fraction(1, 3), Fraction(1, 2), "<")
        assert not self.rational.compare(Fraction(1, 2), Fraction(1, 3), "<")

    def test_compare_greater_than(self):
        """Test: Brüche vergleichen (größer)"""
        assert self.rational.compare(Fraction(2, 3), Fraction(1, 2), ">")
        assert not self.rational.compare(Fraction(1, 2), Fraction(2, 3), ">")

    def test_compare_less_equal(self):
        """Test: Brüche vergleichen (kleiner gleich)"""
        assert self.rational.compare(Fraction(1, 2), Fraction(1, 2), "<=")
        assert self.rational.compare(Fraction(1, 3), Fraction(1, 2), "<=")

    def test_compare_greater_equal(self):
        """Test: Brüche vergleichen (größer gleich)"""
        assert self.rational.compare(Fraction(1, 2), Fraction(1, 2), ">=")
        assert self.rational.compare(Fraction(2, 3), Fraction(1, 2), ">=")

    def test_compare_invalid_operator(self):
        """Test: Ungültiger Vergleichsoperator"""
        with pytest.raises(ValueError, match="Unbekannter Operator"):
            self.rational.compare(Fraction(1, 2), Fraction(1, 3), "??")

    def test_negative_fractions(self):
        """Test: Negative Brüche"""
        # -1/2 + 1/3 = -1/6
        result = self.rational.add(Fraction(-1, 2), Fraction(1, 3))
        assert result == Fraction(-1, 6)

    def test_fraction_with_integers(self):
        """Test: Bruch mit Ganzzahlen"""
        # 3/1 (=3) + 1/2 = 7/2
        result = self.rational.add(Fraction(3, 1), Fraction(1, 2))
        assert result == Fraction(7, 2)


class TestDecimalArithmetic:
    """Tests für Dezimalzahl-Arithmetik"""

    def setup_method(self):
        """Setup für jeden Test"""
        self.decimal = DecimalArithmetic(precision=10)

    def test_set_precision(self):
        """Test: Präzision setzen"""
        self.decimal.set_precision(20)
        assert self.decimal.precision == 20
        assert getcontext().prec == 20

    def test_add_decimals(self):
        """Test: Dezimalzahlen addieren"""
        result = self.decimal.calculate("+", 1.5, 2.3)
        assert result == Decimal("3.8")

    def test_add_multiple_decimals(self):
        """Test: Mehrere Dezimalzahlen addieren"""
        result = self.decimal.calculate("+", 1.1, 2.2, 3.3)
        assert result == Decimal("6.6")

    def test_subtract_decimals(self):
        """Test: Dezimalzahlen subtrahieren"""
        result = self.decimal.calculate("-", 5.5, 2.2)
        assert result == Decimal("3.3")

    def test_multiply_decimals(self):
        """Test: Dezimalzahlen multiplizieren"""
        result = self.decimal.calculate("*", 2.5, 4.0)
        assert result == Decimal("10.0")

    def test_divide_decimals(self):
        """Test: Dezimalzahlen dividieren"""
        result = self.decimal.calculate("/", 10.0, 2.5)
        assert result == Decimal("4.0")

    def test_divide_by_zero(self):
        """Test: Division durch Null schlägt fehl"""
        with pytest.raises(ValueError, match="Division durch Null"):
            self.decimal.calculate("/", 10.0, 0.0)

    def test_invalid_operation(self):
        """Test: Ungültige Operation"""
        with pytest.raises(ValueError, match="Unbekannte Operation"):
            self.decimal.calculate("%", 10.0, 3.0)

    def test_round_float(self):
        """Test: Float runden"""
        result = self.decimal.round(3.14159, 2)
        assert result == 3.14

    def test_round_float_no_decimals(self):
        """Test: Float auf Ganzzahl runden"""
        result = self.decimal.round(3.7, 0)
        assert result == 4.0

    def test_round_decimal(self):
        """Test: Decimal runden"""
        value = Decimal("3.14159")
        result = self.decimal.round_decimal(value, 2)
        assert result == Decimal("3.14")

    def test_round_decimal_no_decimals(self):
        """Test: Decimal auf Ganzzahl runden"""
        value = Decimal("3.7")
        result = self.decimal.round_decimal(value, 0)
        assert result == Decimal("4")

    def test_to_fraction(self):
        """Test: Decimal zu Fraction konvertieren"""
        value = Decimal("0.5")
        result = self.decimal.to_fraction(value)
        assert result == Fraction(1, 2)

    def test_to_fraction_complex(self):
        """Test: Komplexes Decimal zu Fraction"""
        value = Decimal("0.75")
        result = self.decimal.to_fraction(value)
        assert result == Fraction(3, 4)

    def test_from_fraction(self):
        """Test: Fraction zu Decimal konvertieren"""
        fraction = Fraction(1, 2)
        result = self.decimal.from_fraction(fraction)
        assert result == Decimal("0.5")

    def test_from_fraction_repeating(self):
        """Test: Fraction mit periodischer Dezimaldarstellung"""
        fraction = Fraction(1, 3)
        result = self.decimal.from_fraction(fraction)
        # Mit Präzision 10: 0.3333333333
        assert str(result).startswith("0.333")

    def test_high_precision(self):
        """Test: Hohe Präzision"""
        self.decimal.set_precision(50)
        result = self.decimal.calculate("/", 1, 3)
        # 50 signifikante Stellen
        result_str = str(result)
        assert len(result_str.replace(".", "")) >= 45

    def test_precision_preservation(self):
        """Test: Präzision bleibt erhalten"""
        # Mit Präzision 10
        result1 = self.decimal.calculate("+", 0.1, 0.2)

        # Setze Präzision 20
        self.decimal.set_precision(20)
        result2 = self.decimal.calculate("+", 0.1, 0.2)

        # Beide sollten 0.3 sein, aber mit unterschiedlicher interner Präzision
        assert result1 == Decimal("0.3")
        assert result2 == Decimal("0.3")

    def test_negative_decimals(self):
        """Test: Negative Dezimalzahlen"""
        result = self.decimal.calculate("+", -1.5, 2.5)
        assert result == Decimal("1.0")

    def test_large_numbers(self):
        """Test: Große Zahlen"""
        result = self.decimal.calculate("*", 1000000, 1000000)
        assert result == Decimal("1000000000000")

    def test_small_numbers(self):
        """Test: Sehr kleine Zahlen"""
        result = self.decimal.calculate("*", 0.000001, 0.000001)
        assert result == Decimal("0.000000000001")


class TestRationalDecimalConversion:
    """Tests für Konvertierung zwischen Rational und Decimal"""

    def setup_method(self):
        """Setup für jeden Test"""
        self.rational = RationalArithmetic()
        self.decimal = DecimalArithmetic(precision=10)

    def test_fraction_to_decimal_to_fraction(self):
        """Test: Fraction → Decimal → Fraction (Roundtrip)"""
        original = Fraction(3, 4)

        # Fraction → Decimal
        dec = self.decimal.from_fraction(original)
        assert dec == Decimal("0.75")

        # Decimal → Fraction
        result = self.decimal.to_fraction(dec)
        assert result == original

    def test_exact_fractions(self):
        """Test: Exakte Brüche (keine Rundungsfehler)"""
        fractions = [
            Fraction(1, 2),  # 0.5
            Fraction(1, 4),  # 0.25
            Fraction(3, 8),  # 0.375
            Fraction(7, 16),  # 0.4375
        ]

        for frac in fractions:
            # Fraction → Decimal → Fraction
            dec = self.decimal.from_fraction(frac)
            result = self.decimal.to_fraction(dec)
            assert result == frac, f"Roundtrip failed for {frac}"

    def test_repeating_decimal(self):
        """Test: Periodische Dezimalzahl (Präzisionsverlust)"""
        original = Fraction(1, 3)  # 0.333...

        # Fraction → Decimal (Rundung)
        dec = self.decimal.from_fraction(original)

        # Decimal → Fraction (nicht mehr exakt 1/3)
        result = self.decimal.to_fraction(dec)

        # Sollte nah bei 1/3 sein, aber nicht exakt
        # (wegen Präzision)
        assert abs(float(result) - 1 / 3) < 0.0001


class TestIntegrationWithArithmeticEngine:
    """Integrationstests mit ArithmeticEngine"""

    def test_division_returns_fraction(self):
        """Test: Division in Division-Klasse gibt Fraction zurück"""
        from component_52_arithmetic_reasoning import Division

        div = Division()
        result = div.execute(7, 3)

        # Ergebnis sollte Fraction sein
        assert isinstance(result.value, Fraction)
        assert result.value == Fraction(7, 3)

    def test_division_with_exact_result(self):
        """Test: Division mit ganzzahligem Ergebnis"""
        from component_52_arithmetic_reasoning import Division

        div = Division()
        result = div.execute(6, 3)

        # Ergebnis sollte Fraction(2, 1) sein
        assert isinstance(result.value, Fraction)
        assert result.value == Fraction(2, 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
