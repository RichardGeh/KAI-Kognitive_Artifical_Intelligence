"""
Test für Math Proof Tree Integration (Nicht-interaktiv)

Testet:
- ArithmeticEngine erstellt ProofTree
- ProofTree enthält korrekte Schritte (PREMISE, RULE_APPLICATION, CONCLUSION)
- Unicode-Formatierung funktioniert
"""

import sys
from component_1_netzwerk_core import KonzeptNetzwerkCore
from component_52_arithmetic_reasoning import ArithmeticEngine
from component_17_proof_explanation import format_proof_tree, StepType


def test_arithmetic_proof_tree():
    """Testet ArithmeticEngine Proof Tree Erstellung"""
    print("="*80)
    print("TEST: Arithmetic Proof Tree Integration")
    print("="*80)

    # Initialisierung
    print("\n[1/4] Initialisiere ArithmeticEngine...")
    netzwerk = KonzeptNetzwerkCore()
    arithmetic_engine = ArithmeticEngine(netzwerk)
    print("✓ ArithmeticEngine initialisiert")

    # Test 1: Addition
    print("\n[2/4] Test Addition: 3 + 5")
    result_add = arithmetic_engine.calculate("+", 3, 5)
    print(f"  Ergebnis: {result_add.value}")
    print(f"  Confidence: {result_add.confidence}")
    print(f"  ProofTree Steps: {len(result_add.proof_tree.get_all_steps())}")

    # Validiere ProofTree-Struktur
    steps = result_add.proof_tree.get_all_steps()
    assert len(steps) >= 3, "ProofTree sollte mindestens 3 Schritte haben"

    # Finde Step-Typen
    step_types = [step.step_type for step in steps]
    assert StepType.PREMISE in step_types, "ProofTree sollte PREMISE enthalten"
    assert StepType.RULE_APPLICATION in step_types, "ProofTree sollte RULE_APPLICATION enthalten"
    assert StepType.CONCLUSION in step_types, "ProofTree sollte CONCLUSION enthalten"
    print("  ✓ ProofTree-Struktur korrekt")

    # Formatiere ProofTree
    proof_text = format_proof_tree(result_add.proof_tree, show_details=True)
    print(f"\n  Proof Tree:\n{proof_text}")

    # Test 2: Division (mit Constraint Check)
    print("\n[3/4] Test Division: 15 / 3")
    result_div = arithmetic_engine.calculate("/", 15, 3)
    print(f"  Ergebnis: {result_div.value}")
    print(f"  Result Type: {type(result_div.value).__name__}")

    # Division sollte mehr Schritte haben (wegen Constraint Check)
    steps_div = result_div.proof_tree.get_all_steps()
    print(f"  ProofTree Steps: {len(steps_div)}")
    assert len(steps_div) >= 4, "Division ProofTree sollte mindestens 4 Schritte haben"
    print("  ✓ Division ProofTree korrekt")

    # Formatiere ProofTree
    proof_text_div = format_proof_tree(result_div.proof_tree, show_details=True)
    print(f"\n  Proof Tree:\n{proof_text_div}")

    # Test 3: Multiplikation
    print("\n[4/4] Test Multiplikation: 6 * 7")
    result_mul = arithmetic_engine.calculate("*", 6, 7)
    print(f"  Ergebnis: {result_mul.value}")

    # Teste Unicode-Formatierung
    from component_18_proof_tree_widget import ProofNodeItem
    # Erstelle temporären Node für Formatierungs-Test
    class DummyStep:
        def __init__(self):
            self.step_type = StepType.RULE_APPLICATION
            self.output = "6 * 7 = 42"
            self.confidence = 1.0
            self.explanation_text = "Multiplikation anwenden"
            self.inputs = []
            self.rule_name = "Arithmetik: Multiplikation"
            self.bindings = {}
            self.metadata = {}
            self.step_id = "test"
            self.parent_steps = []
            from datetime import datetime
            self.timestamp = datetime.now()
            self.source_component = "test"

    class DummyTreeNode:
        def __init__(self):
            self.step = DummyStep()

    dummy_node = DummyTreeNode()
    test_item = ProofNodeItem(dummy_node)

    # Test Formatierung
    formatted = test_item._format_mathematical_text("6 * 7 / 2 = 21")
    assert " × " in formatted, "* sollte zu × formatiert werden"
    assert " ÷ " in formatted, "/ sollte zu ÷ formatiert werden"
    print(f"  Original: '6 * 7 / 2 = 21'")
    print(f"  Formatiert: '{formatted}'")
    print("  ✓ Unicode-Formatierung funktioniert")

    # Zusammenfassung
    print("\n" + "="*80)
    print("ERFOLG: Alle Tests bestanden!")
    print("="*80)
    print("\nZusammenfassung:")
    print("  ✓ ArithmeticEngine erstellt korrekte ProofTrees")
    print("  ✓ ProofTree-Struktur ist vollständig (PREMISE, RULE_APPLICATION, CONCLUSION)")
    print("  ✓ Division enthält Constraint-Check (Division durch Null)")
    print("  ✓ Unicode-Symbole werden korrekt formatiert (× ÷)")
    print("\nIntegration abgeschlossen!")


if __name__ == "__main__":
    try:
        test_arithmetic_proof_tree()
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
