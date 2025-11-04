"""
Test für den neuen 'Lerne:' Befehl
"""

import re


def test_lerne_pattern():
    """Testet ob das Lerne:-Muster korrekt erkannt wird"""

    test_cases = [
        ("Lerne: Apfel", "Apfel"),
        ("lerne: Ein Hund ist ein Tier", "Ein Hund ist ein Tier"),
        ("LERNE: Katzen können miauen", "Katzen können miauen"),
        ("lerne:    Test mit Leerzeichen   ", "Test mit Leerzeichen"),
    ]

    pattern = r"^\s*lerne:\s*(.+)\s*$"

    for text, expected in test_cases:
        match = re.match(pattern, text, re.IGNORECASE)
        if match:
            captured = match.group(1).strip()
            print(f"[OK] '{text}' -> '{captured}'")
            assert captured == expected, f"Expected '{expected}', got '{captured}'"
        else:
            print(f"[FAIL] '{text}' nicht erkannt!")
            assert False, f"Pattern sollte '{text}' erkennen"

    print("\n[OK] Alle Tests bestanden!")


def test_declarative_detection():
    """Testet die Erkennung von deklarativen Aussagen vs. einzelnen Wörtern"""

    declarative_patterns = [
        "ist ein",
        "ist eine",
        "sind",
        "kann",
        "können",
        "hat",
        "haben",
        "liegt in",
        "befindet sich",
    ]

    declarative_tests = [
        ("Ein Apfel ist eine Frucht", True),
        ("Katzen sind Tiere", True),
        ("Vögel können fliegen", True),
        ("Berlin liegt in Deutschland", True),
        ("Apfel", False),
        ("Banane", False),
        ("Ein Konzept", False),
    ]

    for text, expected_declarative in declarative_tests:
        is_declarative = any(
            pattern in text.lower() for pattern in declarative_patterns
        )
        status = "[OK]" if is_declarative == expected_declarative else "[FAIL]"
        print(
            f"{status} '{text}' -> declarative={is_declarative} (expected={expected_declarative})"
        )
        assert is_declarative == expected_declarative

    print("\n[OK] Alle Deklarativ-Tests bestanden!")


if __name__ == "__main__":
    print("=== Test: Lerne-Pattern ===")
    test_lerne_pattern()

    print("\n=== Test: Deklarative Erkennung ===")
    test_declarative_detection()
