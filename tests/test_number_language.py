"""
Tests für das Zahl-Wort-System (component_53)
Testet bidirektionale Konvertierung: Zahl ↔ Wort
"""

import pytest

from component_1_netzwerk_core import KonzeptNetzwerkCore
from component_53_number_language import (
    NumberFormatter,
    NumberLanguageConnector,
    NumberParser,
)


@pytest.fixture
def netzwerk():
    """Erstellt ein Test-Netzwerk"""
    netzwerk = KonzeptNetzwerkCore(
        uri="bolt://127.0.0.1:7687", user="neo4j", password="password"
    )
    yield netzwerk
    netzwerk.close()


@pytest.fixture
def parser():
    """Erstellt einen NumberParser (ohne Neo4j)"""
    return NumberParser(netzwerk=None)


@pytest.fixture
def formatter():
    """Erstellt einen NumberFormatter"""
    return NumberFormatter()


@pytest.fixture
def connector(netzwerk):
    """Erstellt einen NumberLanguageConnector mit Neo4j"""
    return NumberLanguageConnector(netzwerk)


class TestNumberParser:
    """Tests für NumberParser (Wort → Zahl)"""

    def test_parse_basic_numbers_0_to_10(self, parser):
        """Test: Basis-Zahlen 0-10"""
        assert parser.parse("null") == 0
        assert parser.parse("eins") == 1
        assert parser.parse("zwei") == 2
        assert parser.parse("drei") == 3
        assert parser.parse("vier") == 4
        assert parser.parse("fünf") == 5
        assert parser.parse("sechs") == 6
        assert parser.parse("sieben") == 7
        assert parser.parse("acht") == 8
        assert parser.parse("neun") == 9
        assert parser.parse("zehn") == 10

    def test_parse_basic_numbers_11_to_20(self, parser):
        """Test: Basis-Zahlen 11-20"""
        assert parser.parse("elf") == 11
        assert parser.parse("zwölf") == 12
        assert parser.parse("dreizehn") == 13
        assert parser.parse("vierzehn") == 14
        assert parser.parse("fünfzehn") == 15
        assert parser.parse("sechzehn") == 16
        assert parser.parse("siebzehn") == 17
        assert parser.parse("achtzehn") == 18
        assert parser.parse("neunzehn") == 19
        assert parser.parse("zwanzig") == 20

    def test_parse_tens_21_to_99(self, parser):
        """Test: Zehner 21-99"""
        assert parser.parse("einundzwanzig") == 21
        assert parser.parse("zweiundzwanzig") == 22
        assert parser.parse("dreiunddreißig") == 33
        assert parser.parse("vierundvierzig") == 44
        assert parser.parse("fünfundfünfzig") == 55
        assert parser.parse("sechsundsechzig") == 66
        assert parser.parse("siebenundsiebzig") == 77
        assert parser.parse("achtundachtzig") == 88
        assert parser.parse("neunundneunzig") == 99

    def test_parse_pure_tens(self, parser):
        """Test: Reine Zehner (30, 40, ..., 90)"""
        assert parser.parse("dreißig") == 30
        assert parser.parse("vierzig") == 40
        assert parser.parse("fünfzig") == 50
        assert parser.parse("sechzig") == 60
        assert parser.parse("siebzig") == 70
        assert parser.parse("achtzig") == 80
        assert parser.parse("neunzig") == 90

    def test_parse_hundreds_100_to_900(self, parser):
        """Test: Hunderter (100, 200, ..., 900)"""
        assert parser.parse("einhundert") == 100
        assert parser.parse("zweihundert") == 200
        assert parser.parse("dreihundert") == 300
        assert parser.parse("vierhundert") == 400
        assert parser.parse("fünfhundert") == 500
        assert parser.parse("sechshundert") == 600
        assert parser.parse("siebenhundert") == 700
        assert parser.parse("achthundert") == 800
        assert parser.parse("neunhundert") == 900

    def test_parse_complex_hundreds(self, parser):
        """Test: Komplexe Hunderter (123, 456, 789)"""
        assert parser.parse("einhundertdreiundzwanzig") == 123
        assert parser.parse("zweihundertfünfundvierzig") == 245
        assert parser.parse("vierhundertsechsundfünfzig") == 456
        assert parser.parse("siebenhundertneunundachtzig") == 789
        assert parser.parse("neunhundertneunundneunzig") == 999

    def test_parse_edge_cases(self, parser):
        """Test: Edge Cases"""
        # Ein vs. Eins
        assert parser.parse("ein") == 1
        assert parser.parse("eins") == 1

        # Leerzeichen sollten ignoriert werden
        assert parser.parse("ein hundert") == 100
        assert parser.parse("zwei hundert fünf") == 205

        # Groß-/Kleinschreibung
        assert parser.parse("DREI") == 3
        assert parser.parse("Zehn") == 10

    def test_parse_invalid_input(self, parser):
        """Test: Ungültige Eingaben"""
        assert parser.parse("ungültig") is None
        assert parser.parse("") is None
        assert parser.parse("zweiundund") is None


class TestNumberFormatter:
    """Tests für NumberFormatter (Zahl → Wort)"""

    def test_format_basic_numbers_0_to_10(self, formatter):
        """Test: Basis-Zahlen 0-10"""
        assert formatter.format(0) == "null"
        assert formatter.format(1) == "eins"
        assert formatter.format(2) == "zwei"
        assert formatter.format(3) == "drei"
        assert formatter.format(4) == "vier"
        assert formatter.format(5) == "fünf"
        assert formatter.format(6) == "sechs"
        assert formatter.format(7) == "sieben"
        assert formatter.format(8) == "acht"
        assert formatter.format(9) == "neun"
        assert formatter.format(10) == "zehn"

    def test_format_basic_numbers_11_to_20(self, formatter):
        """Test: Basis-Zahlen 11-20"""
        assert formatter.format(11) == "elf"
        assert formatter.format(12) == "zwölf"
        assert formatter.format(13) == "dreizehn"
        assert formatter.format(14) == "vierzehn"
        assert formatter.format(15) == "fünfzehn"
        assert formatter.format(16) == "sechzehn"
        assert formatter.format(17) == "siebzehn"
        assert formatter.format(18) == "achtzehn"
        assert formatter.format(19) == "neunzehn"
        assert formatter.format(20) == "zwanzig"

    def test_format_tens_21_to_99(self, formatter):
        """Test: Zehner 21-99"""
        assert formatter.format(21) == "einundzwanzig"
        assert formatter.format(22) == "zweiundzwanzig"
        assert formatter.format(33) == "dreiunddreißig"
        assert formatter.format(44) == "vierundvierzig"
        assert formatter.format(55) == "fünfundfünfzig"
        assert formatter.format(66) == "sechsundsechzig"
        assert formatter.format(77) == "siebenundsiebzig"
        assert formatter.format(88) == "achtundachtzig"
        assert formatter.format(99) == "neunundneunzig"

    def test_format_pure_tens(self, formatter):
        """Test: Reine Zehner (30, 40, ..., 90)"""
        assert formatter.format(30) == "dreißig"
        assert formatter.format(40) == "vierzig"
        assert formatter.format(50) == "fünfzig"
        assert formatter.format(60) == "sechzig"
        assert formatter.format(70) == "siebzig"
        assert formatter.format(80) == "achtzig"
        assert formatter.format(90) == "neunzig"

    def test_format_hundreds_100_to_900(self, formatter):
        """Test: Hunderter (100, 200, ..., 900)"""
        assert formatter.format(100) == "einhundert"
        assert formatter.format(200) == "zweihundert"
        assert formatter.format(300) == "dreihundert"
        assert formatter.format(400) == "vierhundert"
        assert formatter.format(500) == "fünfhundert"
        assert formatter.format(600) == "sechshundert"
        assert formatter.format(700) == "siebenhundert"
        assert formatter.format(800) == "achthundert"
        assert formatter.format(900) == "neunhundert"

    def test_format_complex_hundreds(self, formatter):
        """Test: Komplexe Hunderter (123, 456, 789)"""
        assert formatter.format(123) == "einhundertdreiundzwanzig"
        assert formatter.format(245) == "zweihundertfünfundvierzig"
        assert formatter.format(456) == "vierhundertsechsundfünfzig"
        assert formatter.format(789) == "siebenhundertneunundachtzig"
        assert formatter.format(999) == "neunhundertneunundneunzig"

    def test_format_negative_numbers(self, formatter):
        """Test: Negative Zahlen"""
        assert formatter.format(-1) == "minuseins"
        assert formatter.format(-10) == "minuszehn"
        assert formatter.format(-99) == "minusneunundneunzig"

    def test_format_edge_cases(self, formatter):
        """Test: Edge Cases"""
        # Eins wird zu "ein" in Komposita
        assert formatter.format(21) == "einundzwanzig"
        assert formatter.format(100) == "einhundert"
        assert formatter.format(101) == "einhunderteins"


class TestRoundtrip:
    """Tests für Roundtrip: parse(format(n)) == n"""

    def test_roundtrip_0_to_20(self, parser, formatter):
        """Test: Roundtrip für 0-20"""
        for n in range(21):
            word = formatter.format(n)
            parsed = parser.parse(word)
            assert parsed == n, f"Roundtrip fehlgeschlagen für {n}: {word} → {parsed}"

    def test_roundtrip_21_to_99(self, parser, formatter):
        """Test: Roundtrip für 21-99"""
        for n in range(21, 100):
            word = formatter.format(n)
            parsed = parser.parse(word)
            assert parsed == n, f"Roundtrip fehlgeschlagen für {n}: {word} → {parsed}"

    def test_roundtrip_100_to_999(self, parser, formatter):
        """Test: Roundtrip für 100-999"""
        # Teste alle Hunderter + Stichproben
        test_numbers = list(range(100, 1000, 100))  # 100, 200, ..., 900
        test_numbers.extend([123, 245, 456, 678, 789, 999])  # Stichproben

        for n in test_numbers:
            word = formatter.format(n)
            parsed = parser.parse(word)
            assert parsed == n, f"Roundtrip fehlgeschlagen für {n}: {word} → {parsed}"

    def test_roundtrip_full_range_0_to_999(self, parser, formatter):
        """Test: Vollständiger Roundtrip für 0-999"""
        failed = []

        for n in range(1000):
            word = formatter.format(n)
            parsed = parser.parse(word)

            if parsed != n:
                failed.append((n, word, parsed))

        assert (
            len(failed) == 0
        ), f"Roundtrip fehlgeschlagen für {len(failed)} Zahlen: {failed[:10]}"


class TestNeo4jIntegration:
    """Tests für Neo4j Integration"""

    def test_learn_number(self, connector):
        """Test: Zahl-Wort-Zuordnung lernen"""
        success = connector.learn_number("test_drei", 3)
        assert success is True

        # Verifikation
        value = connector.query_number("test_drei")
        assert value == 3

    def test_query_number(self, connector):
        """Test: Zahlenwert abfragen"""
        # Lerne zuerst
        connector.learn_number("test_fünf", 5)

        # Query
        value = connector.query_number("test_fünf")
        assert value == 5

    def test_query_word(self, connector):
        """Test: Wort für Zahlenwert abfragen"""
        # Lerne zuerst
        connector.learn_number("test_sieben", 7)

        # Query
        word = connector.query_word(7)
        # Kann "test_sieben" oder ein anderes Wort für 7 sein
        assert word is not None

    def test_query_nonexistent_number(self, connector):
        """Test: Query für nicht existierende Zahl"""
        value = connector.query_number("nicht_existent_xyz")
        assert value is None

    def test_initialize_basic_numbers(self, connector):
        """Test: Basis-Zahlen initialisieren"""
        # Initialisiere 0-20
        count = connector.initialize_basic_numbers(20)
        assert count == 21  # 0 bis 20 = 21 Zahlen

        # Verifikation
        assert connector.query_number("null") == 0
        assert connector.query_number("zehn") == 10
        assert connector.query_number("zwanzig") == 20


class TestSpecialCases:
    """Tests für Spezialfälle"""

    def test_ein_vs_eins(self, parser, formatter):
        """Test: 'ein' vs 'eins' Behandlung"""
        # Parser sollte beide akzeptieren
        assert parser.parse("ein") == 1
        assert parser.parse("eins") == 1

        # Formatter nutzt "eins" für standalone 1
        assert formatter.format(1) == "eins"

        # Aber "ein" in Komposita
        assert formatter.format(21) == "einundzwanzig"
        assert formatter.format(100) == "einhundert"

    def test_whitespace_handling(self, parser):
        """Test: Leerzeichen-Behandlung"""
        assert parser.parse("einhundert") == 100
        assert parser.parse("ein hundert") == 100
        assert parser.parse("  drei  ") == 3

    def test_case_insensitivity(self, parser):
        """Test: Groß-/Kleinschreibung"""
        assert parser.parse("drei") == 3
        assert parser.parse("DREI") == 3
        assert parser.parse("Drei") == 3

    def test_combined_numbers(self, parser):
        """Test: Komplexe kombinierte Zahlen"""
        # 245 = zweihundertfünfundvierzig
        assert parser.parse("zweihundertfünfundvierzig") == 245

        # 999 = neunhundertneunundneunzig
        assert parser.parse("neunhundertneunundneunzig") == 999


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
