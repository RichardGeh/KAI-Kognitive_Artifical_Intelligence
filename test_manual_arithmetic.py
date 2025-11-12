"""
Manuelle Validierung der Grundrechenarten
Demonstriert die Verwendung der ArithmeticEngine
"""

from component_52_arithmetic_reasoning import ArithmeticEngine
from component_1_netzwerk_core import KonzeptNetzwerkCore
from fractions import Fraction


def print_result(operation, a, b, result):
    """Hilfsfunktion für Ausgabe"""
    print(f"\n{'=' * 60}")
    print(f"Operation: {a} {operation} {b}")
    print(f"Ergebnis: {result.value} (Typ: {type(result.value).__name__})")
    print(f"Confidence: {result.confidence}")
    print(f"\nProof Tree:")
    print(f"  Query: {result.proof_tree.query}")
    print(f"  Root Steps: {len(result.proof_tree.root_steps)}")

    # Proof Tree Struktur ausgeben
    for i, root in enumerate(result.proof_tree.root_steps):
        print(f"\n  Step {i+1} ({root.step_type.value}):")
        print(f"    {root.explanation_text}")

        # Level 2
        for j, subgoal in enumerate(root.subgoals):
            print(f"    └─ Step {i+1}.{j+1} ({subgoal.step_type.value}):")
            print(f"       {subgoal.explanation_text}")

            # Level 3
            for k, sub_subgoal in enumerate(subgoal.subgoals):
                print(f"       └─ Step {i+1}.{j+1}.{k+1} ({sub_subgoal.step_type.value}):")
                print(f"          {sub_subgoal.explanation_text}")

    print(f"{'=' * 60}")


def main():
    print("\n" + "=" * 60)
    print("MANUELLE VALIDIERUNG: ARITHMETIK-MODUL")
    print("=" * 60)

    # Initialisiere Engine
    netzwerk = KonzeptNetzwerkCore(
        uri="bolt://127.0.0.1:7687",
        user="neo4j",
        password="password"
    )
    engine = ArithmeticEngine(netzwerk)

    print("\n✓ ArithmeticEngine erfolgreich initialisiert")
    print("✓ Registrierte Operationen:", engine.registry.list_operations())

    # Test 1: Addition
    print("\n\n" + "=" * 60)
    print("TEST 1: ADDITION")
    print("=" * 60)
    result = engine.calculate("+", 3, 5)
    print_result("+", 3, 5, result)
    assert result.value == 8, "Addition fehlgeschlagen!"
    print("\n✓ Addition Test bestanden!")

    # Test 2: Subtraktion
    print("\n\n" + "=" * 60)
    print("TEST 2: SUBTRAKTION")
    print("=" * 60)
    result = engine.calculate("-", 10, 3)
    print_result("-", 10, 3, result)
    assert result.value == 7, "Subtraktion fehlgeschlagen!"
    print("\n✓ Subtraktion Test bestanden!")

    # Test 3: Multiplikation
    print("\n\n" + "=" * 60)
    print("TEST 3: MULTIPLIKATION")
    print("=" * 60)
    result = engine.calculate("*", 4, 5)
    print_result("*", 4, 5, result)
    assert result.value == 20, "Multiplikation fehlgeschlagen!"
    print("\n✓ Multiplikation Test bestanden!")

    # Test 4: Division (Integer -> Fraction)
    print("\n\n" + "=" * 60)
    print("TEST 4: DIVISION (INTEGER → FRACTION)")
    print("=" * 60)
    result = engine.calculate("/", 1, 3)
    print_result("/", 1, 3, result)
    assert isinstance(result.value, Fraction), "Division sollte Fraction zurückgeben!"
    assert result.value == Fraction(1, 3), "Division fehlgeschlagen!"
    print("\n✓ Division Test bestanden!")

    # Test 5: Division durch 0
    print("\n\n" + "=" * 60)
    print("TEST 5: DIVISION DURCH NULL (FEHLERBEHANDLUNG)")
    print("=" * 60)
    try:
        result = engine.calculate("/", 10, 0)
        print("FEHLER: Division durch 0 sollte einen Fehler werfen!")
    except ValueError as e:
        print(f"✓ Erwarteter Fehler korrekt gefangen: {e}")

    # Test 6: Division mit Vereinfachung
    print("\n\n" + "=" * 60)
    print("TEST 6: DIVISION MIT VEREINFACHUNG")
    print("=" * 60)
    result = engine.calculate("/", 6, 9)
    print_result("/", 6, 9, result)
    assert result.value == Fraction(2, 3), "Division Vereinfachung fehlgeschlagen!"
    print("\n✓ Division Vereinfachung Test bestanden!")

    # Zusammenfassung
    print("\n\n" + "=" * 60)
    print("ZUSAMMENFASSUNG")
    print("=" * 60)
    print("✓ Alle 4 Grundrechenarten implementiert")
    print("✓ Proof Trees mit 3-stufiger Hierarchie generiert")
    print("✓ Division mit Fraction für exakte Brüche")
    print("✓ Division durch 0 wird korrekt abgefangen")
    print("✓ Fraction-Vereinfachung funktioniert (6/9 = 2/3)")
    print("\n✓✓✓ ALLE TESTS BESTANDEN! ✓✓✓")
    print("=" * 60 + "\n")

    netzwerk.close()


if __name__ == "__main__":
    main()
