# test_auto_detect_e2e.py
"""
End-to-End-Test für autonomes Definition-Lernen (Phase 1-3 komplett).

Testet den vollständigen Workflow:
1. MeaningExtractor erkennt Definition automatisch
2. GoalPlanner erstellt Plan basierend auf Confidence
3. SubGoalExecutor führt Plan aus
4. Bei hoher Confidence (>=0.85): Direktes Speichern
5. Bei mittlerer Confidence (0.70-0.84): Bestätigung erforderlich
6. Bei niedriger Confidence (<0.70): Clarification
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from component_6_linguistik_engine import LinguisticPreprocessor
from component_7_meaning_extractor import MeaningPointExtractor
from component_11_embedding_service import EmbeddingService
from component_4_goal_planner import GoalPlanner
from component_5_linguistik_strukturen import MeaningPointCategory, GoalType

# Logging konfigurieren
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)


def test_e2e_high_confidence_auto_save():
    """
    Test 1: Hohe Confidence (>=0.85) -> Direktes Speichern ohne Confirmation

    Beispiele:
    - IS_A: 0.92
    - CAPABLE_OF: 0.91
    - LOCATED_IN: 0.93
    """
    logger.info("\n" + "=" * 70)
    logger.info("TEST 1: HOHE CONFIDENCE -> AUTO-SAVE")
    logger.info("=" * 70)

    # Initialisiere Services
    embedding_service = EmbeddingService()
    preprocessor = LinguisticPreprocessor()
    extractor = MeaningPointExtractor(
        embedding_service, preprocessor, prototyping_engine=None
    )
    planner = GoalPlanner()

    test_cases = [
        ("Ein Hund ist ein Tier", "IS_A", 0.92),
        ("Vögel können fliegen", "CAPABLE_OF", 0.91),
        ("Berlin liegt in Deutschland", "LOCATED_IN", 0.93),
    ]

    for text, expected_relation, expected_confidence in test_cases:
        logger.info(f"\n-> Teste: '{text}'")

        # SCHRITT 1: MeaningExtractor erkennt Definition
        doc = preprocessor.process(text)
        meaning_points = extractor.extract(doc)
        mp = meaning_points[0]

        # Validiere MeaningPoint
        assert (
            mp.category == MeaningPointCategory.DEFINITION
        ), f"Falsche Kategorie: {mp.category}"
        assert (
            mp.arguments.get("relation_type") == expected_relation
        ), f"Falsche Relation: {mp.arguments.get('relation_type')}"
        assert (
            mp.confidence == expected_confidence
        ), f"Falsche Confidence: {mp.confidence}"

        logger.info(
            f"  [OK] MeaningPoint erkannt: {mp.category.name} (Confidence: {mp.confidence:.2f})"
        )

        # SCHRITT 2: GoalPlanner erstellt Plan basierend auf Confidence
        plan = planner.create_plan(mp)

        # Validiere Plan
        assert plan is not None, "Plan ist None"
        assert plan.type == GoalType.LEARN_KNOWLEDGE, f"Falscher GoalType: {plan.type}"

        # WICHTIG: Bei hoher Confidence sollte KEIN Confirmation-SubGoal existieren
        confirmation_exists = any(
            "Bestätige die erkannte Absicht" in sg.description for sg in plan.sub_goals
        )
        assert (
            not confirmation_exists
        ), f"Confirmation-SubGoal sollte bei Confidence {mp.confidence:.2f} nicht existieren!"

        # Erwartete SubGoals für auto-erkannte Definition:
        # 1. "Extrahiere Subjekt, Relation und Objekt."
        # 2. "Speichere die Relation im Wissensgraphen."
        # 3. "Formuliere eine Lernbestätigung."
        assert (
            len(plan.sub_goals) == 3
        ), f"Erwartete 3 SubGoals, erhielt {len(plan.sub_goals)}"

        logger.info(f"  [OK] Plan erstellt: {plan.type.value} (ohne Confirmation)")
        logger.info(f"    SubGoals: {len(plan.sub_goals)}")
        for i, sg in enumerate(plan.sub_goals, 1):
            logger.info(f"      {i}. {sg.description}")

        # ERFOLG
        logger.info(f"  [SUCCESS] TEST BESTANDEN für '{text}'")

    logger.info("\n" + "=" * 70)
    logger.info(
        "[SUCCESS] TEST 1 ERFOLGREICH: Alle hohen Confidence-Fälle ohne Confirmation"
    )
    logger.info("=" * 70)


def test_e2e_medium_confidence_confirmation():
    """
    Test 2: Mittlere Confidence (0.70-0.84) -> Confirmation erforderlich

    Beispiele:
    - HAS_PROPERTY: 0.78
    """
    logger.info("\n" + "=" * 70)
    logger.info("TEST 2: MITTLERE CONFIDENCE -> CONFIRMATION ERFORDERLICH")
    logger.info("=" * 70)

    # Initialisiere Services
    embedding_service = EmbeddingService()
    preprocessor = LinguisticPreprocessor()
    extractor = MeaningPointExtractor(
        embedding_service, preprocessor, prototyping_engine=None
    )
    planner = GoalPlanner()

    test_cases = [
        ("Der Apfel ist rot", "HAS_PROPERTY", 0.78),
        ("Schnee ist weiß", "HAS_PROPERTY", 0.78),
    ]

    for text, expected_relation, expected_confidence in test_cases:
        logger.info(f"\n-> Teste: '{text}'")

        # SCHRITT 1: MeaningExtractor erkennt Definition
        doc = preprocessor.process(text)
        meaning_points = extractor.extract(doc)
        mp = meaning_points[0]

        # Validiere MeaningPoint
        assert (
            mp.category == MeaningPointCategory.DEFINITION
        ), f"Falsche Kategorie: {mp.category}"
        assert (
            mp.arguments.get("relation_type") == expected_relation
        ), f"Falsche Relation: {mp.arguments.get('relation_type')}"
        assert (
            mp.confidence == expected_confidence
        ), f"Falsche Confidence: {mp.confidence}"

        logger.info(
            f"  [OK] MeaningPoint erkannt: {mp.category.name} (Confidence: {mp.confidence:.2f})"
        )

        # SCHRITT 2: GoalPlanner erstellt Plan mit Confirmation
        plan = planner.create_plan(mp)

        # Validiere Plan
        assert plan is not None, "Plan ist None"
        assert plan.type == GoalType.LEARN_KNOWLEDGE, f"Falscher GoalType: {plan.type}"

        # WICHTIG: Bei mittlerer Confidence sollte Confirmation-SubGoal existieren
        confirmation_exists = any(
            "Bestätige die erkannte Absicht" in sg.description for sg in plan.sub_goals
        )
        assert (
            confirmation_exists
        ), f"Confirmation-SubGoal sollte bei Confidence {mp.confidence:.2f} existieren!"

        # Erwartete SubGoals für auto-erkannte Definition mit Confirmation:
        # 1. "Bestätige die erkannte Absicht." (hinzugefügt vom Planner)
        # 2. "Extrahiere Subjekt, Relation und Objekt."
        # 3. "Speichere die Relation im Wissensgraphen."
        # 4. "Formuliere eine Lernbestätigung."
        assert (
            len(plan.sub_goals) == 4
        ), f"Erwartete 4 SubGoals (mit Confirmation), erhielt {len(plan.sub_goals)}"

        # Prüfe dass Confirmation das erste SubGoal ist
        first_subgoal = plan.sub_goals[0].description
        assert (
            "Bestätige die erkannte Absicht" in first_subgoal
        ), f"Erstes SubGoal sollte Confirmation sein, ist aber: {first_subgoal}"

        logger.info(f"  [OK] Plan erstellt: {plan.type.value} (mit Confirmation)")
        logger.info(f"    SubGoals: {len(plan.sub_goals)}")
        for i, sg in enumerate(plan.sub_goals, 1):
            logger.info(f"      {i}. {sg.description}")

        # ERFOLG
        logger.info(f"  [SUCCESS] TEST BESTANDEN für '{text}'")

    logger.info("\n" + "=" * 70)
    logger.info(
        "[SUCCESS] TEST 2 ERFOLGREICH: Alle mittleren Confidence-Fälle mit Confirmation"
    )
    logger.info("=" * 70)


def test_confidence_thresholds():
    """
    Test 3: Validiere Confidence-Schwellwerte

    - >= 0.85: Auto-Save (kein Confirmation-SubGoal)
    - 0.70-0.84: Confirmation (Confirmation-SubGoal am Anfang)
    - < 0.70: Clarification (würde zu CLARIFY_INTENT führen)
    """
    logger.info("\n" + "=" * 70)
    logger.info("TEST 3: CONFIDENCE-SCHWELLWERTE VALIDIERUNG")
    logger.info("=" * 70)

    from component_5_linguistik_strukturen import MeaningPoint, Modality, Polarity

    planner = GoalPlanner()

    # Test 3a: Grenzfall exakt 0.85 (sollte Auto-Save sein)
    logger.info("\n-> Teste Grenzfall: Confidence = 0.85 (exakt)")
    mp_085 = MeaningPoint(
        id="mp-test-085",
        category=MeaningPointCategory.DEFINITION,
        cue="test",
        text_span="Test ist etwas",
        modality=Modality.DECLARATIVE,
        polarity=Polarity.POSITIVE,
        confidence=0.85,
        arguments={
            "subject": "test",
            "relation_type": "IS_A",
            "object": "etwas",
            "auto_detected": True,
        },
    )
    plan_085 = planner.create_plan(mp_085)
    assert plan_085 is not None
    confirmation_exists_085 = any(
        "Bestätige die erkannte Absicht" in sg.description for sg in plan_085.sub_goals
    )
    # Bei exakt 0.85 sollte es KEINE Confirmation geben (>= 0.85)
    assert (
        not confirmation_exists_085
    ), "Bei Confidence 0.85 sollte kein Confirmation-SubGoal existieren!"
    logger.info("  [OK] Confidence 0.85 -> Auto-Save (korrekt)")

    # Test 3b: Grenzfall 0.84 (sollte Confirmation sein)
    logger.info("\n-> Teste Grenzfall: Confidence = 0.84 (knapp unter Schwelle)")
    mp_084 = MeaningPoint(
        id="mp-test-084",
        category=MeaningPointCategory.DEFINITION,
        cue="test",
        text_span="Test ist etwas",
        modality=Modality.DECLARATIVE,
        polarity=Polarity.POSITIVE,
        confidence=0.84,
        arguments={
            "subject": "test",
            "relation_type": "IS_A",
            "object": "etwas",
            "auto_detected": True,
        },
    )
    plan_084 = planner.create_plan(mp_084)
    assert plan_084 is not None
    confirmation_exists_084 = any(
        "Bestätige die erkannte Absicht" in sg.description for sg in plan_084.sub_goals
    )
    # Bei 0.84 sollte es Confirmation geben (< 0.85)
    assert (
        confirmation_exists_084
    ), "Bei Confidence 0.84 sollte Confirmation-SubGoal existieren!"
    logger.info("  [OK] Confidence 0.84 -> Confirmation (korrekt)")

    # Test 3c: Grenzfall 0.70 (sollte noch Confirmation sein, nicht Clarification)
    logger.info(
        "\n-> Teste Grenzfall: Confidence = 0.70 (untere Grenze für Confirmation)"
    )
    mp_070 = MeaningPoint(
        id="mp-test-070",
        category=MeaningPointCategory.DEFINITION,
        cue="test",
        text_span="Test ist etwas",
        modality=Modality.DECLARATIVE,
        polarity=Polarity.POSITIVE,
        confidence=0.70,
        arguments={
            "subject": "test",
            "relation_type": "IS_A",
            "object": "etwas",
            "auto_detected": True,
        },
    )
    plan_070 = planner.create_plan(mp_070)
    assert plan_070 is not None
    # Bei 0.70 sollte es noch Confirmation geben (>= 0.4, < 0.85)
    assert (
        plan_070.type != GoalType.CLARIFY_INTENT
    ), "Bei Confidence 0.70 sollte es noch LEARN_KNOWLEDGE sein, nicht CLARIFY_INTENT!"
    confirmation_exists_070 = any(
        "Bestätige die erkannte Absicht" in sg.description for sg in plan_070.sub_goals
    )
    assert (
        confirmation_exists_070
    ), "Bei Confidence 0.70 sollte Confirmation-SubGoal existieren!"
    logger.info("  [OK] Confidence 0.70 -> Confirmation (korrekt)")

    # Test 3d: Grenzfall 0.39 (sollte Clarification sein)
    logger.info("\n-> Teste Grenzfall: Confidence = 0.39 (Clarification-Bereich)")
    mp_039 = MeaningPoint(
        id="mp-test-039",
        category=MeaningPointCategory.DEFINITION,
        cue="test",
        text_span="Test ist etwas",
        modality=Modality.DECLARATIVE,
        polarity=Polarity.POSITIVE,
        confidence=0.39,
        arguments={
            "subject": "test",
            "relation_type": "IS_A",
            "object": "etwas",
            "auto_detected": True,
        },
    )
    plan_039 = planner.create_plan(mp_039)
    assert plan_039 is not None
    # Bei 0.39 sollte es Clarification sein (< 0.4)
    assert (
        plan_039.type == GoalType.CLARIFY_INTENT
    ), f"Bei Confidence 0.39 sollte GoalType CLARIFY_INTENT sein, ist aber {plan_039.type}!"
    logger.info("  [OK] Confidence 0.39 -> Clarification (korrekt)")

    logger.info("\n" + "=" * 70)
    logger.info("[SUCCESS] TEST 3 ERFOLGREICH: Alle Confidence-Schwellwerte korrekt")
    logger.info("=" * 70)


def run_all_tests():
    """Führt alle End-to-End-Tests aus."""
    logger.info("\n" + "=" * 70)
    logger.info("STARTE END-TO-END-TESTS FÜR AUTONOMES DEFINITION-LERNEN")
    logger.info("=" * 70)

    try:
        # Test 1: Hohe Confidence -> Auto-Save
        test_e2e_high_confidence_auto_save()

        # Test 2: Mittlere Confidence -> Confirmation
        test_e2e_medium_confidence_confirmation()

        # Test 3: Confidence-Schwellwerte
        test_confidence_thresholds()

        # Zusammenfassung
        logger.info("\n" + "=" * 70)
        logger.info("🎉 ALLE END-TO-END-TESTS ERFOLGREICH! 🎉")
        logger.info("=" * 70)
        logger.info("\nZusammenfassung:")
        logger.info("  [SUCCESS] Hohe Confidence (≥0.85): Auto-Save funktioniert")
        logger.info(
            "  [SUCCESS] Mittlere Confidence (0.70-0.84): Confirmation funktioniert"
        )
        logger.info(
            "  [SUCCESS] Niedrige Confidence (<0.40): Clarification funktioniert"
        )
        logger.info("  [SUCCESS] Confidence-Schwellwerte korrekt implementiert")
        logger.info(
            "\n  -> Autonomes Definition-Lernen ist vollständig funktionsfähig!"
        )
        logger.info("=" * 70)

        return True

    except AssertionError as e:
        logger.error(f"\n[ERROR] TEST FEHLGESCHLAGEN: {e}")
        return False
    except Exception as e:
        logger.error(f"\n[ERROR] KRITISCHER FEHLER: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
