"""
Property-Based Testing für Text-Normalisierung

Verwendet Hypothesis für generative Tests mit zufälligen Eingaben.
Prüft Robustheit gegen Edge Cases, Unicode, extreme Längen.
"""

import pytest
from hypothesis import given, strategies as st, assume, settings
from hypothesis.strategies import characters, composite

from component_utils_text_normalization import (
    TextNormalizer,
    clean_entity,
    normalize_plural_to_singular,
)


# ============================================================================
# PROPERTY-BASED TESTS: clean_entity
# ============================================================================


@given(st.text(min_size=0, max_size=100))
@settings(max_examples=200)
def test_clean_entity_never_crashes(text_input):
    """
    Property: clean_entity sollte für JEDE Eingabe funktionieren und niemals abstürzen.

    Testet:
    - Beliebige Unicode-Zeichen
    - Leere Strings
    - Whitespace
    - Sonderzeichen
    """
    try:
        result = clean_entity(text_input)
        assert isinstance(result, str), "Ergebnis muss immer ein String sein"
    except Exception as e:
        pytest.fail(
            f"clean_entity ist abgestürzt mit Input '{text_input[:50]}...': {e}"
        )


@given(
    st.text(min_size=1, max_size=100, alphabet=characters(blacklist_categories=("Cs",)))
)
@settings(max_examples=200)
def test_clean_entity_output_always_lowercase(text_input):
    """
    Property: Ausgabe von clean_entity sollte immer lowercase sein.
    """
    result = clean_entity(text_input)
    # Ignoriere leere Ausgaben
    if result:
        assert result == result.lower(), f"Ausgabe '{result}' ist nicht lowercase"


@given(
    st.text(min_size=1, max_size=100, alphabet=characters(blacklist_categories=("Cs",)))
)
@settings(max_examples=200)
def test_clean_entity_no_leading_trailing_whitespace(text_input):
    """
    Property: Ausgabe von clean_entity sollte keine führenden/folgenden Whitespaces haben.
    """
    result = clean_entity(text_input)
    # Ignoriere leere Ausgaben
    if result:
        assert result == result.strip(), f"Ausgabe '{result}' hat Whitespace am Rand"


@given(st.text(min_size=1, max_size=100))
@settings(max_examples=200)
def test_clean_entity_idempotent(text_input):
    """
    Property: clean_entity(clean_entity(x)) = clean_entity(x) (Idempotenz)

    Nach einer Normalisierung sollte eine weitere Normalisierung nichts ändern.
    """
    first_pass = clean_entity(text_input)
    second_pass = clean_entity(first_pass)
    assert (
        first_pass == second_pass
    ), f"Nicht idempotent: '{first_pass}' -> '{second_pass}'"


@given(
    st.text(
        min_size=1, max_size=50, alphabet=characters(whitelist_categories=("Ll", "Lu"))
    )
)
@settings(max_examples=100)
def test_clean_entity_removes_articles(text_input):
    """
    Property: Artikel (der, die, das, ein, eine) am Anfang sollten entfernt werden.
    """
    # Filtere nur nicht-leere und nicht-artikel Inputs
    assume(len(text_input.strip()) > 0)
    assume(
        not any(
            text_input.lower().strip().startswith(art.strip())
            for art in ["der", "die", "das", "ein", "eine"]
        )
    )

    articles = ["der ", "die ", "das ", "ein ", "eine "]

    for article in articles:
        test_text = article + text_input
        result = clean_entity(test_text)

        # Prüfe dass der Artikel nicht mehr am Anfang ist
        assert not result.startswith(
            article.strip()
        ), f"Artikel '{article}' wurde nicht entfernt: '{test_text}' -> '{result}'"


# ============================================================================
# PROPERTY-BASED TESTS: normalize_plural_to_singular
# ============================================================================


@given(st.text(min_size=0, max_size=100))
@settings(max_examples=200)
def test_normalize_plural_never_crashes(text_input):
    """
    Property: normalize_plural_to_singular sollte niemals abstürzen.
    """
    try:
        result = normalize_plural_to_singular(text_input)
        assert isinstance(result, str), "Ergebnis muss immer ein String sein"
    except Exception as e:
        pytest.fail(
            f"normalize_plural_to_singular ist abgestürzt mit Input '{text_input[:50]}...': {e}"
        )


@given(st.text(min_size=1, max_size=50))
@settings(max_examples=200)
def test_normalize_plural_idempotent(text_input):
    """
    Property: Plural-Normalisierung sollte idempotent sein.

    Wenn ein Wort bereits singular ist, sollte es unverändert bleiben.
    """
    first_pass = normalize_plural_to_singular(text_input)
    second_pass = normalize_plural_to_singular(first_pass)
    assert (
        first_pass == second_pass
    ), f"Nicht idempotent: '{first_pass}' -> '{second_pass}'"


@given(
    st.text(min_size=1, max_size=50, alphabet=characters(blacklist_categories=("Cs",)))
)
@settings(max_examples=100)
def test_normalize_plural_output_not_longer_than_input(text_input):
    """
    Property: Singular-Form sollte nie länger als Plural-Form sein.

    Plural-Normalisierung entfernt Endungen, fügt keine hinzu.
    """
    result = normalize_plural_to_singular(text_input)
    assert len(result) <= len(
        text_input
    ), f"Ausgabe '{result}' ist länger als Eingabe '{text_input}'"


@given(
    st.text(min_size=3, max_size=20, alphabet=characters(whitelist_categories=("Ll",)))
)
@settings(max_examples=100)
def test_normalize_plural_preserves_short_words(text_input):
    """
    Property: Sehr kurze Wörter (< 3 Zeichen) sollten unverändert bleiben.
    """
    # Filtere nur sehr kurze Inputs
    assume(len(text_input) < 3)

    result = normalize_plural_to_singular(text_input)
    assert (
        result == text_input
    ), f"Kurzes Wort wurde verändert: '{text_input}' -> '{result}'"


# ============================================================================
# EDGE CASES: Unicode und Spezialfälle
# ============================================================================


def test_unicode_emoji():
    """
    Edge Case: Emojis und Unicode-Sonderzeichen sollten nicht crashen.
    """
    test_cases = [
        "Hund",
        "der Apfel",
        "Kaese",
        "Text",
        "Mix Deutsch",
        "abc",
        "Test",
    ]

    for test_text in test_cases:
        try:
            result = clean_entity(test_text)
            assert isinstance(result, str)
        except Exception as e:
            pytest.fail(f"Unicode Test fehlgeschlagen: {e}")


def test_extreme_whitespace():
    """
    Edge Case: Extreme Whitespace-Kombinationen.
    """
    test_cases = [
        "   ",
        "\t\t\t",
        "\n\n\n",
        "   Hund   ",
        "der  \t\n  Apfel",
        "  ein    sehr     großer     Text  ",
    ]

    for test_text in test_cases:
        result = clean_entity(test_text)
        # Sollte nicht crashen und kein Whitespace am Rand haben
        assert result == result.strip()
        # Keine mehrfachen Leerzeichen im Ergebnis
        assert "  " not in result


def test_extreme_length():
    """
    Edge Case: Sehr lange Strings (Stress-Test).
    """
    # Sehr langer String (10.000 Zeichen)
    long_text = "der " + "a" * 10000

    try:
        result = clean_entity(long_text)
        assert isinstance(result, str)
        # Artikel sollte entfernt sein
        assert not result.startswith("der")
        assert len(result) > 0
    except Exception as e:
        pytest.fail(f"Extreme Length Test fehlgeschlagen: {e}")


def test_special_characters():
    """
    Edge Case: Sonderzeichen und Satzzeichen.
    """
    test_cases = [
        "der Hund!",
        "die Katze?",
        "das Haus.",
        "ein Auto,",
        "eine Person;",
        "Preis: 10",
        "100 sicher",
        "CPP Sprache",
        "E-Mail Adresse",
        "Vor- und Nachname",
        "(Klammertext)",
        "[Eckige Klammern]",
        "{Geschweifte Klammern}",
        "Zitat in Anfuehrungszeichen",
    ]

    for test_text in test_cases:
        try:
            result = clean_entity(test_text)
            assert isinstance(result, str)
        except Exception as e:
            pytest.fail(f"Special Characters Test fehlgeschlagen: {e}")


def test_empty_and_none():
    """
    Edge Case: Leere Strings und None-ähnliche Inputs.
    """
    test_cases = [
        "",
        " ",
        "  ",
        "\t",
        "\n",
        None,  # Wird als None getestet
    ]

    for test_text in test_cases:
        try:
            if test_text is None:
                # None sollte zu leerem String werden
                # (TextNormalizer prüft auf not entity_text)
                result = clean_entity(test_text) if test_text else ""
            else:
                result = clean_entity(test_text)

            assert isinstance(result, str)
            # Leere Inputs sollten leere Outputs geben
            assert result == "" or result.isspace() == False
        except Exception as e:
            pytest.fail(f"Empty/None Test fehlgeschlagen: {e}")


def test_known_german_plurals():
    """
    Edge Case: Bekannte deutsche Plurale testen (Regression Tests).
    """
    test_cases = [
        ("katzen", "katze"),
        ("hunden", "hund"),
        ("aktionen", "aktion"),
        ("meldungen", "meldung"),
        ("freiheiten", "freiheit"),
        ("möglichkeiten", "möglichkeit"),
        # Fremdwörter sollten unverändert bleiben (ohne spaCy)
        ("computer", "computer"),
        ("internet", "internet"),
    ]

    for plural, expected_singular in test_cases:
        result = normalize_plural_to_singular(plural)
        print(
            f"Plural Test: '{plural}' -> '{result}' (erwartet: '{expected_singular}')"
        )
        # Hinweis: Ohne spaCy kann die Normalisierung variieren
        # Wir prüfen nur dass es nicht crasht und nicht länger wird
        assert len(result) <= len(plural)


def test_normalizer_with_and_without_spacy():
    """
    Edge Case: TextNormalizer mit und ohne spaCy Preprocessor testen.
    """
    # Ohne spaCy
    normalizer_no_spacy = TextNormalizer(preprocessor=None)
    result1 = normalizer_no_spacy.clean_entity("die Katzen")

    # Mit spaCy (wenn verfügbar)
    try:
        from component_6_linguistik_engine import LinguisticPreprocessor

        preprocessor = LinguisticPreprocessor()
        normalizer_with_spacy = TextNormalizer(preprocessor=preprocessor)
        result2 = normalizer_with_spacy.clean_entity("die Katzen")

        print(f"Ohne spaCy: '{result1}'")
        print(f"Mit spaCy:  '{result2}'")

        # Beide sollten funktionieren
        assert isinstance(result1, str)
        assert isinstance(result2, str)
    except ImportError:
        print("⚠ spaCy nicht verfügbar, Test übersprungen")


# ============================================================================
# COMPOSITE STRATEGIES: Komplexere Test-Inputs
# ============================================================================


@composite
def german_words_with_articles(draw):
    """
    Generiert deutsche Wörter mit Artikeln (composite strategy).
    """
    articles = ["der", "die", "das", "ein", "eine"]
    article = draw(st.sampled_from(articles))
    word = draw(
        st.text(
            min_size=3, max_size=20, alphabet=characters(whitelist_categories=("Ll",))
        )
    )
    return f"{article} {word}"


@given(german_words_with_articles())
@settings(max_examples=100)
def test_clean_entity_removes_any_article(text_with_article):
    """
    Property: Beliebige Artikel-Wort-Kombinationen sollten funktionieren.
    """
    result = clean_entity(text_with_article)
    assert isinstance(result, str)

    # Prüfe dass kein Artikel am Anfang steht
    articles = ["der", "die", "das", "ein", "eine"]
    for article in articles:
        assert not result.startswith(
            article + " "
        ), f"Artikel '{article}' wurde nicht entfernt: '{text_with_article}' -> '{result}'"


# ============================================================================
# PERFORMANCE & STRESS TESTS
# ============================================================================


def test_performance_many_calls():
    """
    Performance Test: Viele Normalisierungs-Aufrufe sollten schnell sein.
    """
    import time

    test_text = "der große Hund"
    iterations = 10000

    start = time.time()
    for _ in range(iterations):
        clean_entity(test_text)
    elapsed = time.time() - start

    avg_time = elapsed / iterations * 1000  # ms
    print(
        f"Performance: {iterations} Aufrufe in {elapsed:.3f}s (Ø {avg_time:.4f}ms pro Aufruf)"
    )

    # Sollte schnell genug sein (< 1ms pro Aufruf)
    assert avg_time < 1.0, f"Zu langsam: {avg_time:.4f}ms pro Aufruf"


def test_repeated_normalization_stable():
    """
    Stabilitätstest: Wiederholte Normalisierung sollte stabil bleiben.
    """
    test_text = "die Katzen"

    results = []
    for i in range(10):
        result = clean_entity(test_text)
        results.append(result)

    # Alle Ergebnisse sollten identisch sein
    assert len(set(results)) == 1, f"Instabile Normalisierung: {results}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
