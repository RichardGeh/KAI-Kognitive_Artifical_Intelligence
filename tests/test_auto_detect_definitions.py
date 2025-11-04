# test_auto_detect_definitions.py
"""
Testskript für die automatische Definition-Erkennung (Schritt 2).
Testet die Erkennung deklarativer Aussagen ohne "Ingestiere Text:"-Befehl.
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from component_6_linguistik_engine import LinguisticPreprocessor
from component_7_meaning_extractor import MeaningPointExtractor
from component_11_embedding_service import EmbeddingService
from component_5_linguistik_strukturen import MeaningPointCategory

# Logging konfigurieren
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def test_auto_detection():
    """Teste die automatische Erkennung verschiedener deklarativer Aussagen."""

    # Initialisiere Services
    logger.info("Initialisiere Services...")
    embedding_service = EmbeddingService()
    preprocessor = LinguisticPreprocessor()
    extractor = MeaningPointExtractor(
        embedding_service, preprocessor, prototyping_engine=None
    )

    # Test-Fälle: (Eingabe, erwartete Kategorie, erwartete Relation)
    test_cases = [
        # IS_A Muster
        ("Ein Hund ist ein Tier", MeaningPointCategory.DEFINITION, "IS_A"),
        ("Der Apfel ist eine Frucht", MeaningPointCategory.DEFINITION, "IS_A"),
        ("Katzen sind Tiere", MeaningPointCategory.DEFINITION, "IS_A"),
        # HAS_PROPERTY Muster
        ("Der Apfel ist rot", MeaningPointCategory.DEFINITION, "HAS_PROPERTY"),
        ("Schnee ist weiß", MeaningPointCategory.DEFINITION, "HAS_PROPERTY"),
        # CAPABLE_OF Muster
        ("Vögel können fliegen", MeaningPointCategory.DEFINITION, "CAPABLE_OF"),
        ("Ein Hund kann bellen", MeaningPointCategory.DEFINITION, "CAPABLE_OF"),
        # PART_OF Muster
        ("Ein Auto hat Räder", MeaningPointCategory.DEFINITION, "PART_OF"),
        (
            "Der Hund gehört zu den Säugetieren",
            MeaningPointCategory.DEFINITION,
            "PART_OF",
        ),
        # LOCATED_IN Muster
        ("Berlin liegt in Deutschland", MeaningPointCategory.DEFINITION, "LOCATED_IN"),
        ("Der Turm ist in Paris", MeaningPointCategory.DEFINITION, "LOCATED_IN"),
        # Negative Tests: Fragen sollten NICHT als Definitionen erkannt werden
        ("Was ist ein Hund?", MeaningPointCategory.QUESTION, None),
        ("Wo ist Berlin?", MeaningPointCategory.QUESTION, None),
    ]

    logger.info("\n" + "=" * 60)
    logger.info("STARTE AUTO-DETECTION TESTS")
    logger.info("=" * 60 + "\n")

    passed = 0
    failed = 0

    for text, expected_category, expected_relation in test_cases:
        logger.info(f"\nTeste: '{text}'")
        logger.info(f"  Erwartete Kategorie: {expected_category.name}")
        if expected_relation:
            logger.info(f"  Erwartete Relation: {expected_relation}")

        # Verarbeite Text
        doc = preprocessor.process(text)
        meaning_points = extractor.extract(doc)

        if not meaning_points:
            logger.error(f"  [ERROR] FEHLER: Keine MeaningPoints extrahiert!")
            failed += 1
            continue

        mp = meaning_points[0]

        # Prüfe Kategorie
        if mp.category != expected_category:
            logger.error(
                f"  [ERROR] FEHLER: Falsche Kategorie! "
                f"Erwartet: {expected_category.name}, Erhalten: {mp.category.name}"
            )
            failed += 1
            continue

        # Prüfe Relation (nur bei DEFINITION)
        if expected_relation:
            actual_relation = mp.arguments.get("relation_type")
            if actual_relation != expected_relation:
                logger.error(
                    f"  [ERROR] FEHLER: Falsche Relation! "
                    f"Erwartet: {expected_relation}, Erhalten: {actual_relation}"
                )
                failed += 1
                continue

            # Zeige extrahierte Informationen
            subject = mp.arguments.get("subject", "?")
            obj = mp.arguments.get("object", "?")
            logger.info(
                f"  [OK] Extrahiert: ({subject}) -[{actual_relation}]-> ({obj})"
            )

        logger.info(f"  [OK] TEST BESTANDEN (Confidence: {mp.confidence:.2f})")
        passed += 1

    # Zusammenfassung
    logger.info("\n" + "=" * 60)
    logger.info("TEST-ZUSAMMENFASSUNG")
    logger.info("=" * 60)
    logger.info(f"Gesamt: {len(test_cases)} Tests")
    logger.info(f"[OK] Bestanden: {passed}")
    logger.info(f"[ERROR] Fehlgeschlagen: {failed}")
    logger.info(f"Erfolgsquote: {passed / len(test_cases) * 100:.1f}%")
    logger.info("=" * 60 + "\n")

    return failed == 0


if __name__ == "__main__":
    try:
        success = test_auto_detection()
        exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Kritischer Fehler: {e}", exc_info=True)
        exit(1)
