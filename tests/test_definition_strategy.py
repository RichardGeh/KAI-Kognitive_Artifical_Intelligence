# test_definition_strategy.py
"""
Testet die DefinitionStrategy für auto-erkannte Definitionen.

Validiert:
1. Extraktion von Subject, Relation, Object aus MeaningPoint
2. Korrektes Handling von auto-erkannten Definitionen vs. manuellen
3. Formulierung der Bestätigungsnachricht
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from component_5_linguistik_strukturen import (
    MeaningPoint,
    MeaningPointCategory,
    Modality,
    Polarity,
)
from kai_sub_goal_executor import DefinitionStrategy

# Logging konfigurieren
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MockWorker:
    """Mock KaiWorker für isolierte Strategy-Tests."""

    def __init__(self):
        self.netzwerk = MockNetzwerk()
        self.context = None

    def _emit_context_update(self):
        pass


class MockNetzwerk:
    """Mock KonzeptNetzwerk für Speicher-Tests."""

    def __init__(self):
        self.stored_relations = []

    def assert_relation(self, subject, relation_type, obj, source_text):
        """Simuliert Relation-Speicherung."""
        self.stored_relations.append(
            {
                "subject": subject,
                "relation_type": relation_type,
                "object": obj,
                "source": source_text,
                "created": True,  # Simuliere neuen Eintrag
            }
        )
        logger.info(f"  -> Mock-Speicherung: ({subject})-[{relation_type}]->({obj})")
        return True  # Simuliere erfolgreiche Erstellung


def test_extract_relation_triple():
    """
    Test 1: Prüft ob die Strategy subject, relation, object korrekt extrahiert.
    """
    logger.info("\n" + "=" * 70)
    logger.info("TEST 1: EXTRAKTION VON RELATION-TRIPEL")
    logger.info("=" * 70)

    worker = MockWorker()
    strategy = DefinitionStrategy(worker)

    # Erstelle MeaningPoint für auto-erkannte IS_A Definition
    mp = MeaningPoint(
        id="mp-test-1",
        category=MeaningPointCategory.DEFINITION,
        cue="auto_detect_is_a",
        text_span="Ein Hund ist ein Tier",
        modality=Modality.DECLARATIVE,
        polarity=Polarity.POSITIVE,
        confidence=0.92,
        arguments={
            "subject": "hund",
            "relation_type": "IS_A",
            "object": "tier",
            "auto_detected": True,
        },
    )

    # Teste Extraktion
    context = {"intent": mp}
    success, result = strategy._extract_relation_triple(mp)

    # Validierung
    assert success, "Extraktion sollte erfolgreich sein"
    assert (
        result["auto_subject"] == "hund"
    ), f"Falsches Subject: {result.get('auto_subject')}"
    assert (
        result["auto_relation"] == "IS_A"
    ), f"Falsche Relation: {result.get('auto_relation')}"
    assert (
        result["auto_object"] == "tier"
    ), f"Falsches Object: {result.get('auto_object')}"

    logger.info(f"  [OK] Subject extrahiert: {result['auto_subject']}")
    logger.info(f"  [OK] Relation extrahiert: {result['auto_relation']}")
    logger.info(f"  [OK] Object extrahiert: {result['auto_object']}")
    logger.info("  [SUCCESS] TEST BESTANDEN")


def test_store_relation():
    """
    Test 2: Prüft ob die Strategy die Relation korrekt speichert.
    """
    logger.info("\n" + "=" * 70)
    logger.info("TEST 2: SPEICHERUNG VON RELATIONEN")
    logger.info("=" * 70)

    worker = MockWorker()
    strategy = DefinitionStrategy(worker)

    # Erstelle MeaningPoint
    mp = MeaningPoint(
        id="mp-test-2",
        category=MeaningPointCategory.DEFINITION,
        cue="auto_detect_capable_of",
        text_span="Vögel können fliegen",
        modality=Modality.DECLARATIVE,
        polarity=Polarity.POSITIVE,
        confidence=0.91,
        arguments={
            "subject": "vögel",
            "relation_type": "CAPABLE_OF",
            "object": "fliegen",
            "auto_detected": True,
        },
    )

    # Simuliere vorherige Extraktion
    context = {
        "intent": mp,
        "auto_subject": "vögel",
        "auto_relation": "CAPABLE_OF",
        "auto_object": "fliegen",
    }

    # Teste Speicherung
    success, result = strategy._store_relation(mp, context)

    # Validierung
    assert success, "Speicherung sollte erfolgreich sein"
    assert (
        result["relation_created"] == True
    ), "Relation sollte als 'created' markiert sein"

    # Prüfe ob Mock-Netzwerk aufgerufen wurde
    assert (
        len(worker.netzwerk.stored_relations) == 1
    ), "Genau eine Relation sollte gespeichert worden sein"

    stored = worker.netzwerk.stored_relations[0]
    assert stored["subject"] == "vögel"
    assert stored["relation_type"] == "CAPABLE_OF"
    assert stored["object"] == "fliegen"

    logger.info(
        f"  [OK] Relation gespeichert: ({stored['subject']})-[{stored['relation_type']}]->({stored['object']})"
    )
    logger.info("  [SUCCESS] TEST BESTANDEN")


def test_formulate_confirmation():
    """
    Test 3: Prüft ob die Bestätigungsnachricht korrekt formuliert wird.
    """
    logger.info("\n" + "=" * 70)
    logger.info("TEST 3: FORMULIERUNG DER BESTÄTIGUNG")
    logger.info("=" * 70)

    worker = MockWorker()
    strategy = DefinitionStrategy(worker)

    test_cases = [
        {
            "name": "IS_A Relation (neu erstellt)",
            "context": {
                "relation_created": True,
                "auto_subject": "hund",
                "auto_object": "tier",
            },
            "expected_contains": ["hund", "tier", "gemerkt"],
        },
        {
            "name": "CAPABLE_OF Relation (bereits bekannt)",
            "context": {
                "relation_created": False,
                "auto_subject": "vogel",
                "auto_object": "fliegen",
            },
            "expected_contains": ["vogel", "bereits", "kannte"],
        },
    ]

    for test_case in test_cases:
        logger.info(f"\n-> Teste: {test_case['name']}")

        success, result = strategy._formulate_confirmation(test_case["context"])

        # Validierung
        assert success, f"Formulierung sollte erfolgreich sein für {test_case['name']}"
        assert "final_response" in result, "Result sollte 'final_response' enthalten"

        response = result["final_response"]
        logger.info(f"  -> Antwort: '{response}'")

        # Prüfe ob erwartete Begriffe in der Antwort vorkommen
        for expected_word in test_case["expected_contains"]:
            assert (
                expected_word.lower() in response.lower()
            ), f"Antwort sollte '{expected_word}' enthalten: {response}"
            logger.info(f"    [OK] Enthält '{expected_word}'")

        logger.info(f"  [SUCCESS] TEST BESTANDEN für {test_case['name']}")

    logger.info("\n" + "=" * 70)
    logger.info("[SUCCESS] TEST 3 ERFOLGREICH: Bestätigungsnachrichten korrekt")
    logger.info("=" * 70)


def test_can_handle_definition_subgoals():
    """
    Test 4: Prüft ob die Strategy die richtigen SubGoals erkennt.
    """
    logger.info("\n" + "=" * 70)
    logger.info("TEST 4: SUBGOAL-ERKENNUNG")
    logger.info("=" * 70)

    worker = MockWorker()
    strategy = DefinitionStrategy(worker)

    expected_subgoals = [
        "Extrahiere Subjekt, Relation und Objekt.",
        "Speichere die Relation im Wissensgraphen.",
        "Formuliere eine Lernbestätigung.",
        "Extrahiere Thema, Informationstyp und Inhalt.",  # Manuelles Definiere
        "Speichere die Information direkt im Wissensgraphen.",
        "Formuliere eine Bestätigungsantwort.",
    ]

    unexpected_subgoals = [
        "Identifiziere das Thema der Frage.",  # Question-Strategy
        "Verarbeite Beispielsatz zu Vektor.",  # Pattern-Learning-Strategy
        "Extrahiere den zu ingestierenden Text.",  # Ingestion-Strategy
    ]

    # Teste erwartete SubGoals
    logger.info("\n-> Teste erwartete SubGoals:")
    for subgoal_desc in expected_subgoals:
        can_handle = strategy.can_handle(subgoal_desc)
        assert can_handle, f"Strategy sollte '{subgoal_desc}' handhaben können"
        logger.info(f"  [OK] Kann handhaben: '{subgoal_desc[:50]}...'")

    # Teste unerwartete SubGoals
    logger.info("\n-> Teste unerwartete SubGoals:")
    for subgoal_desc in unexpected_subgoals:
        can_handle = strategy.can_handle(subgoal_desc)
        assert (
            not can_handle
        ), f"Strategy sollte '{subgoal_desc}' NICHT handhaben können"
        logger.info(f"  [OK] Kann nicht handhaben: '{subgoal_desc[:50]}...'")

    logger.info("\n  [SUCCESS] TEST BESTANDEN")


def run_all_tests():
    """Führt alle DefinitionStrategy-Tests aus."""
    logger.info("\n" + "=" * 70)
    logger.info("STARTE DEFINITION-STRATEGY TESTS")
    logger.info("=" * 70)

    try:
        # Test 1: Extraktion
        test_extract_relation_triple()

        # Test 2: Speicherung
        test_store_relation()

        # Test 3: Bestätigung
        test_formulate_confirmation()

        # Test 4: SubGoal-Erkennung
        test_can_handle_definition_subgoals()

        # Zusammenfassung
        logger.info("\n" + "=" * 70)
        logger.info("🎉 ALLE DEFINITION-STRATEGY TESTS ERFOLGREICH! 🎉")
        logger.info("=" * 70)
        logger.info("\nZusammenfassung:")
        logger.info("  [SUCCESS] Extraktion von Relation-Tripeln funktioniert")
        logger.info("  [SUCCESS] Speicherung im Mock-Netzwerk funktioniert")
        logger.info("  [SUCCESS] Bestätigungsnachrichten korrekt formuliert")
        logger.info("  [SUCCESS] SubGoal-Erkennung korrekt")
        logger.info("\n  -> DefinitionStrategy ist vollständig funktionsfähig!")
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
