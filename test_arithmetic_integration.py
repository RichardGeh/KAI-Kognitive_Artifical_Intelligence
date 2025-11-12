#!/usr/bin/env python3
"""
End-to-End Integrations-Test für arithmetische Berechnungen (Schritt 1.5)

Testet den vollständigen Flow:
1. Intent Detection (MeaningExtractor) → ARITHMETIC_QUESTION
2. Goal Planning (GoalPlanner) → PERFORM_CALCULATION mit SubGoals
3. SubGoal Execution (ArithmeticStrategy) → "acht"
"""

from component_4_goal_planner import GoalPlanner
from component_5_linguistik_strukturen import GoalType, MeaningPointCategory
from component_6_linguistik_engine import LinguisticPreprocessor
from component_7_meaning_extractor import MeaningPointExtractor
from component_11_embedding_service import EmbeddingService


def test_arithmetic_integration():
    """Testet den vollständigen arithmetischen Flow End-to-End"""

    print("🧪 END-TO-END TEST: Arithmetische Integration (Schritt 1.5)")
    print("=" * 70)

    # ========================================================================
    # PHASE 1: Intent Detection
    # ========================================================================

    print("\n📝 PHASE 1: Intent Detection")
    print("-" * 70)

    preprocessor = LinguisticPreprocessor()
    embedding_service = EmbeddingService()
    extractor = MeaningPointExtractor(
        embedding_service=embedding_service,
        preprocessor=preprocessor,
        prototyping_engine=None,
    )

    test_input = "Was ist drei plus fünf?"
    print(f"Input: '{test_input}'")

    # Preprocessing
    doc = preprocessor.process(test_input)

    # Extraktion
    meaning_points = extractor.extract(doc)

    if not meaning_points:
        print("   ❌ FEHLER: Keine MeaningPoints extrahiert")
        return False

    mp = meaning_points[0]
    print(f"   → Kategorie: {mp.category.name}")
    print(f"   → Confidence: {mp.confidence:.2f}")

    if mp.category != MeaningPointCategory.ARITHMETIC_QUESTION:
        print(f"   ❌ FEHLER: Falsche Kategorie (erwartet: ARITHMETIC_QUESTION)")
        return False

    print("   ✓ Intent Detection erfolgreich!")

    # ========================================================================
    # PHASE 2: Goal Planning
    # ========================================================================

    print("\n📝 PHASE 2: Goal Planning")
    print("-" * 70)

    planner = GoalPlanner()
    plan = planner.create_plan(mp)

    if not plan:
        print("   ❌ FEHLER: Kein Plan erstellt")
        return False

    print(f"   → Goal Type: {plan.type.name}")
    print(f"   → Description: {plan.description}")
    print(f"   → Sub-Goals: {len(plan.sub_goals)}")

    if plan.type != GoalType.PERFORM_CALCULATION:
        print(f"   ❌ FEHLER: Falscher Goal Type (erwartet: PERFORM_CALCULATION)")
        return False

    if len(plan.sub_goals) != 4:
        print(f"   ❌ FEHLER: Falsche Anzahl Sub-Goals (erwartet: 4, bekommen: {len(plan.sub_goals)})")
        return False

    for i, sg in enumerate(plan.sub_goals, 1):
        print(f"      {i}. {sg.description}")

    print("   ✓ Goal Planning erfolgreich!")

    # ========================================================================
    # PHASE 3: SubGoal Execution (simuliert)
    # ========================================================================

    print("\n📝 PHASE 3: SubGoal Execution (simuliert)")
    print("-" * 70)

    # Simuliere ArithmeticStrategy ohne KaiWorker
    from kai_sub_goal_executor import ArithmeticStrategy

    class MockWorker:
        """Mock-Worker für isoliertes Testen"""

        pass

    strategy = ArithmeticStrategy(MockWorker())
    context = {}

    # SubGoal 1: Parse arithmetischen Ausdruck
    sg1 = plan.sub_goals[0]
    print(f"\n   SubGoal 1: {sg1.description}")
    success, result = strategy.execute(sg1, context)

    if not success:
        print(f"   ❌ FEHLER: {result.get('error')}")
        return False

    context.update(result)
    print(f"      → Operator: {context.get('operator')}")
    print(f"      → Operand 1 (Wort): {context.get('operand1_word')}")
    print(f"      → Operand 2 (Wort): {context.get('operand2_word')}")
    print("      ✓ Parsing erfolgreich!")

    # SubGoal 2: Konvertiere Zahlwörter zu Zahlen
    sg2 = plan.sub_goals[1]
    print(f"\n   SubGoal 2: {sg2.description}")
    success, result = strategy.execute(sg2, context)

    if not success:
        print(f"   ❌ FEHLER: {result.get('error')}")
        return False

    context.update(result)
    print(f"      → Operand 1 (Zahl): {context.get('operand1')}")
    print(f"      → Operand 2 (Zahl): {context.get('operand2')}")
    print("      ✓ Konvertierung erfolgreich!")

    # SubGoal 3: Führe arithmetische Operation aus
    sg3 = plan.sub_goals[2]
    print(f"\n   SubGoal 3: {sg3.description}")
    success, result = strategy.execute(sg3, context)

    if not success:
        print(f"   ❌ FEHLER: {result.get('error')}")
        return False

    context.update(result)
    result_value = context.get("result_value")
    print(f"      → Ergebnis (Zahl): {result_value}")
    print(f"      → Confidence: {context.get('confidence'):.2f}")
    print("      ✓ Berechnung erfolgreich!")

    # SubGoal 4: Formatiere Ergebnis als Zahlwort
    sg4 = plan.sub_goals[3]
    print(f"\n   SubGoal 4: {sg4.description}")
    success, result = strategy.execute(sg4, context)

    if not success:
        print(f"   ❌ FEHLER: {result.get('error')}")
        return False

    context.update(result)
    result_word = context.get("result_word")
    final_answer = context.get("final_answer")
    print(f"      → Ergebnis (Wort): {result_word}")
    print(f"      → Finale Antwort: {final_answer}")
    print("      ✓ Formatierung erfolgreich!")

    # ========================================================================
    # VALIDATION
    # ========================================================================

    print("\n📝 VALIDATION")
    print("-" * 70)

    if result_value != 8:
        print(f"   ❌ FEHLER: Falsches Ergebnis (erwartet: 8, bekommen: {result_value})")
        return False

    if result_word != "acht":
        print(f"   ❌ FEHLER: Falsches Zahlwort (erwartet: 'acht', bekommen: '{result_word}')")
        return False

    # ========================================================================
    # SUCCESS
    # ========================================================================

    print("\n" + "=" * 70)
    print("✓ END-TO-END TEST ERFOLGREICH!")
    print(f"✓ Input: '{test_input}'")
    print(f"✓ Output: '{final_answer}'")
    print("=" * 70)

    return True


if __name__ == "__main__":
    try:
        success = test_arithmetic_integration()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ EXCEPTION: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
