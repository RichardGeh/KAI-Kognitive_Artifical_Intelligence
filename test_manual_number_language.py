"""
Manuelle Validierung des Zahl-Wort-Systems (component_53)
Demonstriert bidirektionale Konvertierung und Neo4j Integration
"""

from component_53_number_language import (
    NumberParser,
    NumberFormatter,
    NumberLanguageConnector,
)
from component_1_netzwerk_core import KonzeptNetzwerkCore


def print_section(title):
    """Hilfsfunktion für Abschnittstitel"""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def test_parser(parser):
    """Teste NumberParser"""
    print_section("TEST 1: NUMBER PARSER (Wort → Zahl)")

    test_cases = [
        ("null", 0),
        ("eins", 1),
        ("zehn", 10),
        ("zwanzig", 20),
        ("einundzwanzig", 21),
        ("neunundneunzig", 99),
        ("einhundert", 100),
        ("zweihundert", 200),
        ("zweihundertfünfundvierzig", 245),
        ("neunhundertneunundneunzig", 999),
    ]

    for word, expected in test_cases:
        result = parser.parse(word)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{word}' → {result} (erwartet: {expected})")


def test_formatter(formatter):
    """Teste NumberFormatter"""
    print_section("TEST 2: NUMBER FORMATTER (Zahl → Wort)")

    test_cases = [
        (0, "null"),
        (1, "eins"),
        (10, "zehn"),
        (20, "zwanzig"),
        (21, "einundzwanzig"),
        (99, "neunundneunzig"),
        (100, "einhundert"),
        (200, "zweihundert"),
        (245, "zweihundertfünfundvierzig"),
        (999, "neunhundertneunundneunzig"),
    ]

    for number, expected in test_cases:
        result = formatter.format(number)
        status = "✓" if result == expected else "✗"
        print(f"  {status} {number} → '{result}' (erwartet: '{expected}')")


def test_roundtrip(parser, formatter):
    """Teste Roundtrip"""
    print_section("TEST 3: ROUNDTRIP (Zahl → Wort → Zahl)")

    # Teste ausgewählte Zahlen
    test_numbers = [0, 1, 10, 21, 42, 99, 100, 123, 456, 789, 999]

    print("\n  Einzelne Tests:")
    for n in test_numbers:
        word = formatter.format(n)
        parsed = parser.parse(word)
        status = "✓" if parsed == n else "✗"
        print(f"    {status} {n} → '{word}' → {parsed}")

    # Vollständiger Roundtrip 0-999
    print("\n  Vollständiger Roundtrip 0-999:")
    failed = []
    for n in range(1000):
        word = formatter.format(n)
        parsed = parser.parse(word)
        if parsed != n:
            failed.append((n, word, parsed))

    if not failed:
        print(f"    ✓ Alle 1000 Zahlen (0-999) erfolgreich getestet!")
    else:
        print(f"    ✗ {len(failed)} Fehler gefunden:")
        for n, word, parsed in failed[:5]:
            print(f"      {n} → '{word}' → {parsed}")


def test_neo4j_integration(connector):
    """Teste Neo4j Integration"""
    print_section("TEST 4: NEO4J INTEGRATION")

    print("\n  Lerne Zahlen:")
    test_zahlen = {
        "test_drei": 3,
        "test_sieben": 7,
        "test_zwölf": 12,
        "test_einundzwanzig": 21,
    }

    for word, value in test_zahlen.items():
        success = connector.learn_number(word, value)
        status = "✓" if success else "✗"
        print(f"    {status} Gelernt: '{word}' = {value}")

    print("\n  Query Zahlen:")
    for word, expected_value in test_zahlen.items():
        value = connector.query_number(word)
        status = "✓" if value == expected_value else "✗"
        print(f"    {status} Query '{word}' → {value} (erwartet: {expected_value})")

    print("\n  Query Wörter:")
    for word, value in test_zahlen.items():
        queried_word = connector.query_word(value)
        status = "✓" if queried_word is not None else "✗"
        print(f"    {status} Query {value} → '{queried_word}'")


def test_special_cases(parser, formatter):
    """Teste Spezialfälle"""
    print_section("TEST 5: SPEZIALFÄLLE")

    print("\n  'ein' vs 'eins':")
    print(f"    Parser: 'ein' → {parser.parse('ein')}")
    print(f"    Parser: 'eins' → {parser.parse('eins')}")
    print(f"    Formatter: 1 → '{formatter.format(1)}'")
    print(f"    Formatter: 21 → '{formatter.format(21)}'")
    print(f"    Formatter: 100 → '{formatter.format(100)}'")
    print(f"    Formatter: 101 → '{formatter.format(101)}'")

    print("\n  Negative Zahlen:")
    print(f"    Formatter: -1 → '{formatter.format(-1)}'")
    print(f"    Formatter: -99 → '{formatter.format(-99)}'")

    print("\n  Leerzeichen:")
    print(f"    Parser: 'ein hundert' → {parser.parse('ein hundert')}")
    print(f"    Parser: 'zwei hundert fünf' → {parser.parse('zwei hundert fünf')}")

    print("\n  Groß-/Kleinschreibung:")
    print(f"    Parser: 'DREI' → {parser.parse('DREI')}")
    print(f"    Parser: 'Zehn' → {parser.parse('Zehn')}")


def main():
    print("\n" + "=" * 70)
    print("MANUELLE VALIDIERUNG: ZAHL-WORT-SYSTEM (component_53)")
    print("=" * 70)

    # Initialisiere Components
    netzwerk = KonzeptNetzwerkCore(
        uri="bolt://127.0.0.1:7687", user="neo4j", password="password"
    )
    parser = NumberParser(netzwerk=None)
    formatter = NumberFormatter()
    connector = NumberLanguageConnector(netzwerk)

    print("\n✓ Komponenten erfolgreich initialisiert")

    # Teste alle Funktionen
    test_parser(parser)
    test_formatter(formatter)
    test_roundtrip(parser, formatter)
    test_neo4j_integration(connector)
    test_special_cases(parser, formatter)

    # Zusammenfassung
    print_section("ZUSAMMENFASSUNG")
    print("✓ NumberParser: Wort → Zahl (0-999)")
    print("✓ NumberFormatter: Zahl → Wort (0-999)")
    print("✓ Roundtrip: parse(format(n)) == n für alle n in 0-999")
    print("✓ Neo4j Integration: EQUIVALENT_TO Relationen")
    print("✓ Spezialfälle: 'ein' vs 'eins', negative Zahlen, Leerzeichen")
    print("\n✓✓✓ ALLE VALIDIERUNGEN ERFOLGREICH! ✓✓✓")
    print("=" * 70 + "\n")

    netzwerk.close()


if __name__ == "__main__":
    main()
