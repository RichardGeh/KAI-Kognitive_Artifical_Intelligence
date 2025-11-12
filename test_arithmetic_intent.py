#!/usr/bin/env python3
"""
Testskript für die arithmetische Intent-Erkennung (Schritt 1.4)
"""

from component_6_linguistik_engine import LinguisticPreprocessor
from component_7_meaning_extractor import MeaningPointExtractor
from component_11_embedding_service import EmbeddingService
from component_5_linguistik_strukturen import MeaningPointCategory


def test_arithmetic_detection():
    """Testet die Erkennung arithmetischer Fragen"""

    # Initialisiere Services
    preprocessor = LinguisticPreprocessor()
    embedding_service = EmbeddingService()
    extractor = MeaningPointExtractor(
        embedding_service=embedding_service,
        preprocessor=preprocessor,
        prototyping_engine=None
    )

    # Test-Fälle
    test_cases = [
        ("Was ist drei plus fünf?", True, 0.95),
        ("Wie viel ist 7 mal 8?", True, 0.95),
        ("Wieviel sind 10 durch 2?", True, 0.95),
        ("Berechne 15 minus 6", True, 0.95),  # "berechne" ist auch question_trigger
        ("Was ist 3 + 5?", True, 0.95),
        ("Rechne 100 dividiert durch 4", True, 0.95),  # "rechne" ist auch question_trigger
        ("Was ist ein Apfel?", False, None),  # Keine arithmetische Frage
        ("Ein Apfel ist eine Frucht.", False, None),  # Definition
    ]

    print("🧪 TEST: Arithmetische Intent-Erkennung (Schritt 1.4)")
    print("=" * 60)

    passed = 0
    failed = 0

    for i, (text, should_detect, expected_confidence) in enumerate(test_cases, 1):
        print(f"\n📝 Test {i}: '{text}'")

        # Preprocessing
        doc = preprocessor.process(text)

        # Extraktion
        meaning_points = extractor.extract(doc)

        # Validierung
        if not meaning_points:
            print(f"   ❌ FEHLER: Keine MeaningPoints zurückgegeben")
            failed += 1
            continue

        mp = meaning_points[0]
        is_arithmetic = mp.category == MeaningPointCategory.ARITHMETIC_QUESTION

        if should_detect:
            if is_arithmetic:
                confidence_ok = abs(mp.confidence - expected_confidence) < 0.01
                if confidence_ok:
                    print(f"   ✓ PASSED: Erkannt als ARITHMETIC_QUESTION")
                    print(f"     Confidence: {mp.confidence:.2f} (erwartet: {expected_confidence})")
                    passed += 1
                else:
                    print(f"   ❌ FAILED: Confidence stimmt nicht")
                    print(f"     Erhalten: {mp.confidence:.2f}, Erwartet: {expected_confidence}")
                    failed += 1
            else:
                print(f"   ❌ FAILED: Nicht als arithmetisch erkannt")
                print(f"     Kategorie: {mp.category.name}")
                failed += 1
        else:
            if not is_arithmetic:
                print(f"   ✓ PASSED: Korrekt NICHT als arithmetisch erkannt")
                print(f"     Kategorie: {mp.category.name}")
                passed += 1
            else:
                print(f"   ❌ FAILED: Fälschlicherweise als arithmetisch erkannt")
                failed += 1

    # Zusammenfassung
    print("\n" + "=" * 60)
    print(f"📊 ERGEBNIS: {passed}/{len(test_cases)} Tests bestanden")

    if failed == 0:
        print("✓ Alle Tests erfolgreich!")
        return True
    else:
        print(f"❌ {failed} Tests fehlgeschlagen")
        return False


if __name__ == "__main__":
    success = test_arithmetic_detection()
    exit(0 if success else 1)
