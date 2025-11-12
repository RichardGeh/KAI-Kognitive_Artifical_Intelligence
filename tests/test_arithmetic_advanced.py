"""
Tests für erweiterte Arithmetik (Phase 3.3-3.5)
Testet PowerArithmetic, ModuloArithmetic und MathematicalConstants
"""

import math
from decimal import Decimal

import pytest

from component_52_arithmetic_reasoning import (
    MathematicalConstants,
    ModuloArithmetic,
    PowerArithmetic,
)


class TestPowerArithmetic:
    """Tests für Potenzen und Wurzeln"""

    def setup_method(self):
        """Setup für jeden Test"""
        self.power = PowerArithmetic()

    # Potenzen
    def test_power_positive_integer(self):
        """Test: Ganzzahlige Potenz"""
        assert self.power.power(2, 3) == 8
        assert self.power.power(5, 2) == 25

    def test_power_zero_exponent(self):
        """Test: Potenz mit Exponent 0"""
        assert self.power.power(5, 0) == 1
        assert self.power.power(100, 0) == 1

    def test_power_negative_exponent(self):
        """Test: Negative Potenz"""
        assert self.power.power(2, -1) == 0.5
        assert self.power.power(4, -2) == 0.0625

    def test_power_fractional_exponent(self):
        """Test: Gebrochener Exponent (Wurzel)"""
        assert self.power.power(4, 0.5) == 2.0
        assert self.power.power(27, 1 / 3) == pytest.approx(3.0, abs=1e-10)

    def test_power_zero_base_positive_exponent(self):
        """Test: 0^n (n > 0)"""
        assert self.power.power(0, 5) == 0

    def test_power_zero_base_negative_exponent(self):
        """Test: 0^n (n < 0) sollte Fehler werfen"""
        with pytest.raises(ValueError, match="0 kann nicht zu einer negativen Potenz"):
            self.power.power(0, -1)

    def test_power_negative_base_integer_exponent(self):
        """Test: Negative Basis mit ganzzahligem Exponenten"""
        assert self.power.power(-2, 3) == -8
        assert self.power.power(-2, 2) == 4

    def test_power_negative_base_fractional_exponent(self):
        """Test: Negative Basis mit gebrochenem Exponenten (komplexe Zahl)"""
        with pytest.raises(ValueError, match="komplexe Zahl"):
            self.power.power(-4, 0.5)

    def test_square(self):
        """Test: Quadrat"""
        assert self.power.square(5) == 25
        assert self.power.square(-3) == 9

    def test_cube(self):
        """Test: Kubik"""
        assert self.power.cube(3) == 27
        assert self.power.cube(-2) == -8

    # Wurzeln
    def test_sqrt_positive(self):
        """Test: Quadratwurzel (positiv)"""
        assert self.power.sqrt(4) == 2.0
        assert self.power.sqrt(9) == 3.0
        assert self.power.sqrt(2) == pytest.approx(1.414213, rel=1e-5)

    def test_sqrt_zero(self):
        """Test: Quadratwurzel von 0"""
        assert self.power.sqrt(0) == 0.0

    def test_sqrt_negative(self):
        """Test: Quadratwurzel von negativen Zahlen"""
        with pytest.raises(ValueError, match="komplexe Zahl"):
            self.power.sqrt(-4)

    def test_cbrt_positive(self):
        """Test: Kubikwurzel (positiv)"""
        assert self.power.cbrt(8) == pytest.approx(2.0, abs=1e-10)
        assert self.power.cbrt(27) == pytest.approx(3.0, abs=1e-10)

    def test_cbrt_negative(self):
        """Test: Kubikwurzel (negativ)"""
        assert self.power.cbrt(-8) == pytest.approx(-2.0, abs=1e-10)
        assert self.power.cbrt(-27) == pytest.approx(-3.0, abs=1e-10)

    def test_nth_root_basic(self):
        """Test: n-te Wurzel (Basis-Fälle)"""
        assert self.power.nth_root(8, 3) == pytest.approx(2.0, abs=1e-10)
        assert self.power.nth_root(16, 4) == pytest.approx(2.0, abs=1e-10)

    def test_nth_root_first(self):
        """Test: 1-te Wurzel (Identität)"""
        assert self.power.nth_root(42, 1) == 42

    def test_nth_root_even_negative(self):
        """Test: Gerade Wurzel von negativer Zahl"""
        with pytest.raises(ValueError, match="komplexe Zahl"):
            self.power.nth_root(-16, 4)

    def test_nth_root_odd_negative(self):
        """Test: Ungerade Wurzel von negativer Zahl"""
        assert self.power.nth_root(-8, 3) == pytest.approx(-2.0, abs=1e-10)

    def test_nth_root_invalid_degree(self):
        """Test: Ungültiger Wurzelgrad"""
        with pytest.raises(ValueError, match="Wurzelgrad muss > 0"):
            self.power.nth_root(8, 0)
        with pytest.raises(ValueError, match="Wurzelgrad muss > 0"):
            self.power.nth_root(8, -2)

    # Exponentialfunktion und Logarithmus
    def test_exp(self):
        """Test: Exponentialfunktion e^x"""
        assert self.power.exp(0) == 1.0
        assert self.power.exp(1) == pytest.approx(math.e, rel=1e-10)
        assert self.power.exp(2) == pytest.approx(math.e**2, rel=1e-10)

    def test_log_natural(self):
        """Test: Natürlicher Logarithmus ln(x)"""
        assert self.power.log(1) == 0.0
        assert self.power.log(math.e) == pytest.approx(1.0, rel=1e-10)
        assert self.power.log(10) == pytest.approx(2.302585, rel=1e-5)

    def test_log_base_10(self):
        """Test: Logarithmus zur Basis 10"""
        assert self.power.log(100, 10) == pytest.approx(2.0, rel=1e-10)
        assert self.power.log(1000, 10) == pytest.approx(3.0, rel=1e-10)
        assert self.power.log10(100) == pytest.approx(2.0, rel=1e-10)

    def test_log_base_2(self):
        """Test: Logarithmus zur Basis 2"""
        assert self.power.log(8, 2) == 3.0
        assert self.power.log(1024, 2) == 10.0
        assert self.power.log2(8) == 3.0

    def test_log_negative(self):
        """Test: Logarithmus von negativen Zahlen"""
        with pytest.raises(ValueError, match="nur für positive Zahlen"):
            self.power.log(-5)
        with pytest.raises(ValueError, match="nur für positive Zahlen"):
            self.power.log10(-5)

    def test_log_zero(self):
        """Test: Logarithmus von 0"""
        with pytest.raises(ValueError, match="nur für positive Zahlen"):
            self.power.log(0)

    def test_log_invalid_base(self):
        """Test: Ungültige Logarithmus-Basis"""
        with pytest.raises(ValueError, match="Basis muss > 0"):
            self.power.log(10, -2)
        with pytest.raises(ValueError, match="Basis muss.*≠ 1"):
            self.power.log(10, 1)


class TestModuloArithmetic:
    """Tests für Modulo-Arithmetik"""

    def setup_method(self):
        """Setup für jeden Test"""
        self.mod = ModuloArithmetic()

    def test_modulo_basic(self):
        """Test: Basis-Modulo"""
        assert self.mod.modulo(7, 3) == 1
        assert self.mod.modulo(10, 4) == 2

    def test_modulo_exact_division(self):
        """Test: Modulo bei exakter Teilung"""
        assert self.mod.modulo(9, 3) == 0
        assert self.mod.modulo(20, 5) == 0

    def test_modulo_negative_dividend(self):
        """Test: Modulo mit negativem Dividenden"""
        # Python-Konvention: Ergebnis hat Vorzeichen des Divisors
        assert self.mod.modulo(-7, 3) == 2
        assert self.mod.modulo(-10, 4) == 2

    def test_modulo_negative_divisor(self):
        """Test: Modulo mit negativem Divisor"""
        assert self.mod.modulo(7, -3) == -2

    def test_modulo_zero_divisor(self):
        """Test: Modulo durch Null"""
        with pytest.raises(ValueError, match="Modulo durch Null"):
            self.mod.modulo(7, 0)

    def test_remainder(self):
        """Test: Rest-Operation (gleich wie modulo)"""
        assert self.mod.remainder(7, 3) == 1
        assert self.mod.remainder(10, 4) == 2

    def test_divmod_op(self):
        """Test: divmod (Quotient und Rest)"""
        quot, rem = self.mod.divmod_op(7, 3)
        assert quot == 2
        assert rem == 1

        quot, rem = self.mod.divmod_op(20, 5)
        assert quot == 4
        assert rem == 0

    def test_divmod_zero_divisor(self):
        """Test: divmod durch Null"""
        with pytest.raises(ValueError, match="Division durch Null"):
            self.mod.divmod_op(7, 0)

    def test_mod_add(self):
        """Test: Modulare Addition"""
        assert self.mod.mod_add(5, 7, 10) == 2  # (5+7) mod 10 = 2
        assert self.mod.mod_add(8, 9, 12) == 5  # (8+9) mod 12 = 5

    def test_mod_subtract(self):
        """Test: Modulare Subtraktion"""
        assert (
            self.mod.mod_subtract(5, 7, 10) == 8
        )  # (5-7) mod 10 = -2 mod 10 = 8 (Python-Konvention)
        assert self.mod.mod_subtract(10, 3, 7) == 0  # (10-3) mod 7 = 0

    def test_mod_multiply(self):
        """Test: Modulare Multiplikation"""
        assert self.mod.mod_multiply(5, 7, 10) == 5  # (5*7) mod 10 = 35 mod 10 = 5
        assert self.mod.mod_multiply(6, 8, 12) == 0  # (6*8) mod 12 = 48 mod 12 = 0

    def test_mod_power(self):
        """Test: Modulare Potenzierung"""
        assert (
            self.mod.mod_power(2, 10, 1000) == 24
        )  # 2^10 mod 1000 = 1024 mod 1000 = 24
        assert self.mod.mod_power(3, 100, 7) == 4  # 3^100 mod 7 = 4

    def test_mod_power_large_exponent(self):
        """Test: Modulare Potenzierung mit großem Exponenten"""
        # Effizient durch Square-and-Multiply
        result = self.mod.mod_power(2, 1000, 1000000)
        assert isinstance(result, int)

    def test_mod_power_negative_exponent(self):
        """Test: Negative Exponenten nicht unterstützt"""
        with pytest.raises(ValueError, match="Negative Exponenten"):
            self.mod.mod_power(2, -5, 10)

    def test_mod_power_zero_modulo(self):
        """Test: Modulo durch Null"""
        with pytest.raises(ValueError, match="Modulo durch Null"):
            self.mod.mod_power(2, 5, 0)

    def test_is_congruent_true(self):
        """Test: Kongruenz (wahr)"""
        assert self.mod.is_congruent(7, 1, 3)  # 7 ≡ 1 (mod 3)
        assert self.mod.is_congruent(10, 2, 4)  # 10 ≡ 2 (mod 4)
        assert self.mod.is_congruent(15, 0, 5)  # 15 ≡ 0 (mod 5)

    def test_is_congruent_false(self):
        """Test: Kongruenz (falsch)"""
        assert not self.mod.is_congruent(7, 2, 3)  # 7 ≢ 2 (mod 3)
        assert not self.mod.is_congruent(10, 3, 4)  # 10 ≢ 3 (mod 4)

    def test_mod_inverse_exists(self):
        """Test: Modulares Inverse existiert"""
        # 3*5 = 15 ≡ 1 (mod 7)
        inv = self.mod.mod_inverse(3, 7)
        assert inv == 5

        # Verifikation
        assert (3 * inv) % 7 == 1

    def test_mod_inverse_more_cases(self):
        """Test: Weitere Fälle für modulares Inverse"""
        inv = self.mod.mod_inverse(5, 11)
        assert inv == 9  # 5*9 = 45 ≡ 1 (mod 11)

        inv = self.mod.mod_inverse(7, 26)
        assert inv == 15  # 7*15 = 105 ≡ 1 (mod 26)

    def test_mod_inverse_not_exists(self):
        """Test: Modulares Inverse existiert nicht"""
        # 4 und 8 sind nicht teilerfremd → kein Inverse
        inv = self.mod.mod_inverse(4, 8)
        assert inv is None

        # 6 und 9 sind nicht teilerfremd
        inv = self.mod.mod_inverse(6, 9)
        assert inv is None


class TestMathematicalConstants:
    """Tests für mathematische Konstanten"""

    def setup_method(self):
        """Setup für jeden Test"""
        self.const_float = MathematicalConstants(use_decimal=False)
        self.const_decimal = MathematicalConstants(use_decimal=True)

    # Konstanten
    def test_pi_float(self):
        """Test: π (Float)"""
        pi = self.const_float.pi()
        assert isinstance(pi, float)
        assert pi == pytest.approx(3.14159265, rel=1e-7)

    def test_pi_decimal(self):
        """Test: π (Decimal)"""
        pi = self.const_decimal.pi()
        assert isinstance(pi, Decimal)
        # Prüfe erste 10 Dezimalstellen
        assert str(pi)[:12] == "3.1415926535"

    def test_e_float(self):
        """Test: e (Float)"""
        e = self.const_float.e()
        assert isinstance(e, float)
        assert e == pytest.approx(2.71828182, rel=1e-7)

    def test_e_decimal(self):
        """Test: e (Decimal)"""
        e = self.const_decimal.e()
        assert isinstance(e, Decimal)
        assert str(e)[:12] == "2.7182818284"

    def test_tau(self):
        """Test: τ = 2π"""
        tau = self.const_float.tau()
        assert tau == pytest.approx(2 * math.pi, rel=1e-10)

    def test_golden_ratio(self):
        """Test: Goldener Schnitt φ"""
        phi = self.const_float.golden_ratio()
        assert phi == pytest.approx(1.618033988, rel=1e-7)

    def test_sqrt_constants(self):
        """Test: Wurzel-Konstanten"""
        assert self.const_float.sqrt_2() == pytest.approx(math.sqrt(2), rel=1e-10)
        assert self.const_float.sqrt_3() == pytest.approx(math.sqrt(3), rel=1e-10)
        assert self.const_float.sqrt_5() == pytest.approx(math.sqrt(5), rel=1e-10)

    # Geometrie
    def test_circle_area(self):
        """Test: Kreisfläche A = πr²"""
        area = self.const_float.circle_area(5)
        assert area == pytest.approx(math.pi * 25, rel=1e-10)

    def test_circle_circumference(self):
        """Test: Kreisumfang C = 2πr"""
        circumference = self.const_float.circle_circumference(5)
        assert circumference == pytest.approx(2 * math.pi * 5, rel=1e-10)

    def test_sphere_volume(self):
        """Test: Kugelvolumen V = 4/3 πr³"""
        volume = self.const_float.sphere_volume(3)
        assert volume == pytest.approx((4 / 3) * math.pi * 27, rel=1e-10)

    def test_sphere_surface(self):
        """Test: Kugeloberfläche A = 4πr²"""
        surface = self.const_float.sphere_surface(3)
        assert surface == pytest.approx(4 * math.pi * 9, rel=1e-10)

    def test_cylinder_volume(self):
        """Test: Zylindervolumen V = πr²h"""
        volume = self.const_float.cylinder_volume(2, 5)
        assert volume == pytest.approx(math.pi * 4 * 5, rel=1e-10)

    # Winkelkonvertierung
    def test_degrees_to_radians(self):
        """Test: Grad → Radiant"""
        assert self.const_float.degrees_to_radians(180) == pytest.approx(
            math.pi, rel=1e-10
        )
        assert self.const_float.degrees_to_radians(90) == pytest.approx(
            math.pi / 2, rel=1e-10
        )
        assert self.const_float.degrees_to_radians(360) == pytest.approx(
            2 * math.pi, rel=1e-10
        )

    def test_radians_to_degrees(self):
        """Test: Radiant → Grad"""
        assert self.const_float.radians_to_degrees(math.pi) == pytest.approx(
            180, rel=1e-10
        )
        assert self.const_float.radians_to_degrees(math.pi / 2) == pytest.approx(
            90, rel=1e-10
        )

    # Trigonometrie
    def test_sin_radians(self):
        """Test: Sinus (Radiant)"""
        assert self.const_float.sin(0) == 0
        assert self.const_float.sin(math.pi / 2) == pytest.approx(1.0, abs=1e-10)
        assert self.const_float.sin(math.pi) == pytest.approx(0.0, abs=1e-10)

    def test_sin_degrees(self):
        """Test: Sinus (Grad)"""
        assert self.const_float.sin(0, use_degrees=True) == 0
        assert self.const_float.sin(90, use_degrees=True) == pytest.approx(
            1.0, abs=1e-10
        )
        assert self.const_float.sin(180, use_degrees=True) == pytest.approx(
            0.0, abs=1e-10
        )

    def test_cos_radians(self):
        """Test: Kosinus (Radiant)"""
        assert self.const_float.cos(0) == pytest.approx(1.0, abs=1e-10)
        assert self.const_float.cos(math.pi / 2) == pytest.approx(0.0, abs=1e-10)
        assert self.const_float.cos(math.pi) == pytest.approx(-1.0, abs=1e-10)

    def test_cos_degrees(self):
        """Test: Kosinus (Grad)"""
        assert self.const_float.cos(0, use_degrees=True) == pytest.approx(
            1.0, abs=1e-10
        )
        assert self.const_float.cos(90, use_degrees=True) == pytest.approx(
            0.0, abs=1e-10
        )
        assert self.const_float.cos(180, use_degrees=True) == pytest.approx(
            -1.0, abs=1e-10
        )

    def test_tan_radians(self):
        """Test: Tangens (Radiant)"""
        assert self.const_float.tan(0) == 0
        assert self.const_float.tan(math.pi / 4) == pytest.approx(1.0, rel=1e-10)

    def test_tan_degrees(self):
        """Test: Tangens (Grad)"""
        assert self.const_float.tan(0, use_degrees=True) == 0
        assert self.const_float.tan(45, use_degrees=True) == pytest.approx(
            1.0, rel=1e-10
        )


class TestIntegrationWithArithmeticEngine:
    """Integrationstests mit ArithmeticEngine"""

    def test_all_modules_accessible(self):
        """Test: Alle neuen Module sind in ArithmeticEngine verfügbar"""
        from component_52_arithmetic_reasoning import ArithmeticEngine

        # Mock-Netzwerk
        class MockNetzwerk:
            pass

        engine = ArithmeticEngine(MockNetzwerk())

        # Prüfe dass alle Module existieren
        assert hasattr(engine, "power_arithmetic")
        assert hasattr(engine, "modulo_arithmetic")
        assert hasattr(engine, "math_constants")

        # Prüfe dass sie die richtigen Typen haben
        assert isinstance(engine.power_arithmetic, PowerArithmetic)
        assert isinstance(engine.modulo_arithmetic, ModuloArithmetic)
        assert isinstance(engine.math_constants, MathematicalConstants)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
