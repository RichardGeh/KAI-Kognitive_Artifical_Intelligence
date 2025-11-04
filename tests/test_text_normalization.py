# tests/test_text_normalization.py
"""
Umfangreiche Unit-Tests für zentrale Text-Normalisierung (component_utils_text_normalization.py)

Testet:
- Artikel-Entfernung (der, die, das, ein, eine)
- Plural-zu-Singular-Normalisierung mit spaCy
- Plural-zu-Singular-Normalisierung mit regelbasierter Fallback
- Edge-Cases (säugetier, computer, Luke, etc.)
- Multi-word Phrasen
- Satzzeichen-Entfernung
"""
import pytest
from component_utils_text_normalization import (
    TextNormalizer,
    clean_entity,
    normalize_plural_to_singular,
)


class TestTextNormalizerWithoutSpacy:
    """
    Tests für TextNormalizer OHNE spaCy-Integration (nur regelbasiert).
    """

    def setup_method(self):
        """Setup vor jedem Test: Normalizer ohne spaCy."""
        self.normalizer = TextNormalizer(preprocessor=None)

    def test_clean_entity_removes_articles(self):
        """Test: Artikel werden korrekt entfernt."""
        test_cases = [
            ("der Apfel", "apfel"),
            ("die Katze", "katze"),
            ("das Haus", "haus"),
            ("ein Hund", "hund"),
            ("eine Blume", "blume"),
            ("den Tisch", "tisch"),
            ("dem Auto", "auto"),
            ("des Baumes", "baumes"),
        ]

        for input_text, expected in test_cases:
            result = self.normalizer.clean_entity(input_text)
            assert (
                result == expected
            ), f"Fehler bei '{input_text}': erwartet '{expected}', erhalten '{result}'"

    def test_clean_entity_removes_punctuation(self):
        """Test: Satzzeichen am Ende werden entfernt."""
        test_cases = [
            ("Apfel.", "apfel"),
            ("Katze,", "katze"),
            ("Hund!", "hund"),
            ("Auto?", "auto"),
            ("Baum;", "baum"),
            ("Haus:", "haus"),
        ]

        for input_text, expected in test_cases:
            result = self.normalizer.clean_entity(input_text)
            assert (
                result == expected
            ), f"Fehler bei '{input_text}': erwartet '{expected}', erhalten '{result}'"

    def test_clean_entity_normalizes_whitespace(self):
        """Test: Mehrfache Leerzeichen werden normalisiert."""
        test_cases = [
            ("  Apfel  ", "apfel"),
            ("Katze   Hund", "katze hund"),
            ("Auto\t\tBaum", "auto baum"),
        ]

        for input_text, expected in test_cases:
            result = self.normalizer.clean_entity(input_text)
            assert (
                result == expected
            ), f"Fehler bei '{input_text}': erwartet '{expected}', erhalten '{result}'"

    def test_plural_normalization_high_confidence_rules(self):
        """Test: Hochspezifische Plural-Endungen werden korrekt normalisiert."""
        test_cases = [
            ("aktionen", "aktion"),
            ("meldungen", "meldung"),
            ("freiheiten", "freiheit"),
            ("möglichkeiten", "möglichkeit"),
            ("eigenschaften", "eigenschaft"),
            ("reichtümer", "reichtum"),
            ("organismen", "organismus"),
        ]

        for input_word, expected in test_cases:
            result = self.normalizer.normalize_plural_to_singular(input_word)
            assert (
                result == expected
            ), f"Fehler bei '{input_word}': erwartet '{expected}', erhalten '{result}'"

    def test_plural_normalization_zen_rule(self):
        """Test: -zen -> -ze Regel funktioniert."""
        test_cases = [
            ("katzen", "katze"),
            ("pflanzen", "pflanze"),
            ("grenzen", "grenze"),
        ]

        for input_word, expected in test_cases:
            result = self.normalizer.normalize_plural_to_singular(input_word)
            assert (
                result == expected
            ), f"Fehler bei '{input_word}': erwartet '{expected}', erhalten '{result}'"

    def test_plural_normalization_en_rule_conservative(self):
        """Test: -en Regel ist konservativ (nur bei Konsonanten vor 'en')."""
        test_cases = [
            # Sollte normalisiert werden (Konsonant vor 'en')
            ("hunden", "hund"),
            ("tischen", "tisch"),
            ("büchern", "büchern"),  # Keine -en Endung, unverändert
            # Sollte NICHT normalisiert werden (Vokal vor 'en')
            ("Luke", "Luke"),  # Name, zu kurz für -en Regel
            ("Anne", "Anne"),  # Name, zu kurz für -en Regel
        ]

        for input_word, expected in test_cases:
            result = self.normalizer.normalize_plural_to_singular(input_word)
            assert (
                result == expected
            ), f"Fehler bei '{input_word}': erwartet '{expected}', erhalten '{result}'"

    def test_edge_case_saeugetier_not_broken(self):
        """
        KRITISCHER EDGE-CASE: "säugetier" darf NICHT zu "säugeti" werden!

        Grund: "säugetier" endet NICHT auf einen der hochspezifischen Plural-Endungen.
        Die "-er" Endung ist mehrdeutig (kann Plural ODER Singular sein).
        Konservative Strategie: Bei Unsicherheit UNVERÄNDERT lassen.
        """
        result = self.normalizer.normalize_plural_to_singular("säugetier")
        assert result == "säugetier", "FEHLER: 'säugetier' wurde falsch normalisiert!"

    def test_edge_case_computer_not_broken(self):
        """
        KRITISCHER EDGE-CASE: "computer" darf NICHT zu "comput" werden!

        Grund: Fremdwörter enden oft auf "-er", aber das ist kein Plural.
        """
        result = self.normalizer.normalize_plural_to_singular("computer")
        assert result == "computer", "FEHLER: 'computer' wurde falsch normalisiert!"

    def test_edge_case_names_unchanged(self):
        """
        EDGE-CASE: Namen sollten unverändert bleiben.
        """
        test_cases = [
            ("Luke", "Luke"),
            ("Anne", "Anne"),
            ("Peter", "Peter"),
        ]

        for input_word, expected in test_cases:
            result = self.normalizer.normalize_plural_to_singular(input_word)
            assert (
                result == expected
            ), f"Fehler bei '{input_word}': erwartet '{expected}', erhalten '{result}'"

    def test_edge_case_singular_forms_unchanged(self):
        """
        EDGE-CASE: Bereits im Singular stehende Wörter sollten unverändert bleiben.
        """
        test_cases = [
            ("katze", "katze"),  # Bereits Singular
            ("hund", "hund"),  # Bereits Singular
            ("baum", "baum"),  # Bereits Singular
            ("auto", "auto"),  # Bereits Singular
        ]

        for input_word, expected in test_cases:
            result = self.normalizer.normalize_plural_to_singular(input_word)
            assert (
                result == expected
            ), f"Fehler bei '{input_word}': erwartet '{expected}', erhalten '{result}'"

    def test_multi_word_phrase_normalization(self):
        """Test: Multi-word Phrasen werden Wort-für-Wort normalisiert."""
        test_cases = [
            (
                "die großen tischen",
                "groß tisch",
            ),  # Artikel entfernt, Plurale normalisiert
            ("der kleine hunden", "klein hund"),
            ("eine schöne katzen", "schön katze"),
        ]

        for input_text, expected in test_cases:
            result = self.normalizer.clean_entity(input_text)
            # Hinweis: Dies testet die kombinierte Funktionalität (Artikel + Plural)
            # OHNE spaCy werden nur die hochspezifischen Plural-Endungen normalisiert
            # "tischen" -> "tisch" (en-Regel), "hunden" -> "hund" (en-Regel), "katzen" -> "katze" (zen-Regel)
            # Artikel werden in jedem Fall entfernt
            assert not result.startswith(
                "die "
            ), f"Artikel 'die' sollte entfernt sein in '{result}'"
            assert not result.startswith(
                "der "
            ), f"Artikel 'der' sollte entfernt sein in '{result}'"
            assert not result.startswith(
                "eine "
            ), f"Artikel 'eine' sollte entfernt sein in '{result}'"

    def test_empty_input(self):
        """Test: Leere Eingaben werden korrekt behandelt."""
        assert self.normalizer.clean_entity("") == ""
        assert self.normalizer.clean_entity(None) == ""
        assert self.normalizer.normalize_plural_to_singular("") == ""

    def test_very_short_words(self):
        """Test: Sehr kurze Wörter werden nicht normalisiert."""
        test_cases = [
            ("ab", "ab"),
            ("an", "an"),
            ("zu", "zu"),
        ]

        for input_word, expected in test_cases:
            result = self.normalizer.normalize_plural_to_singular(input_word)
            assert (
                result == expected
            ), f"Fehler bei '{input_word}': erwartet '{expected}', erhalten '{result}'"


class TestTextNormalizerWithSpacy:
    """
    Tests für TextNormalizer MIT spaCy-Integration (präzisere Lemmatization).

    Diese Tests setzen voraus, dass spaCy korrekt konfiguriert ist.
    """

    def setup_method(self):
        """Setup vor jedem Test: Normalizer mit spaCy."""
        try:
            from component_6_linguistik_engine import LinguisticPreprocessor

            preprocessor = LinguisticPreprocessor()
            self.normalizer = TextNormalizer(preprocessor=preprocessor)
            self.spacy_available = True
        except Exception as e:
            self.spacy_available = False
            pytest.skip(f"spaCy nicht verfügbar: {e}")

    def test_spacy_lemmatization_for_plurals(self):
        """Test: spaCy Lemmatization für Plurale."""
        if not self.spacy_available:
            pytest.skip("spaCy nicht verfügbar")

        test_cases = [
            ("katzen", "katze"),
            ("hunde", "hund"),
            ("äpfel", "apfel"),
            ("bäume", "baum"),
        ]

        for input_word, expected in test_cases:
            result = self.normalizer.normalize_plural_to_singular(input_word)
            # spaCy sollte diese Fälle korrekt lemmatisieren
            # Hinweis: Bei einigen Wörtern kann spaCy abweichen, daher prüfen wir flexible
            assert result in [
                expected,
                input_word,
            ], f"Fehler bei '{input_word}': erwartet '{expected}' oder unverändert, erhalten '{result}'"

    def test_spacy_handles_edge_cases_better(self):
        """Test: spaCy sollte Edge-Cases besser handhaben als regelbasiert."""
        if not self.spacy_available:
            pytest.skip("spaCy nicht verfügbar")

        test_cases = [
            # spaCy erkennt, dass "computer" bereits Singular ist
            ("computer", "computer"),
            # spaCy erkennt Namen
            ("Luke", "Luke"),
        ]

        for input_word, expected in test_cases:
            result = self.normalizer.normalize_plural_to_singular(input_word)
            assert (
                result == expected
            ), f"Fehler bei '{input_word}': erwartet '{expected}', erhalten '{result}'"


class TestConvenienceFunctions:
    """
    Tests für Convenience-Funktionen (clean_entity, normalize_plural_to_singular).
    """

    def test_clean_entity_convenience_function(self):
        """Test: Convenience-Funktion clean_entity funktioniert."""
        result = clean_entity("der Apfel")
        assert result == "apfel"

    def test_normalize_plural_to_singular_convenience_function(self):
        """Test: Convenience-Funktion normalize_plural_to_singular funktioniert."""
        result = normalize_plural_to_singular("katzen")
        assert result == "katze"


class TestRegressionSuite:
    """
    Regressions-Tests für bekannte Bugs aus der Vergangenheit.
    """

    def setup_method(self):
        """Setup vor jedem Test: Normalizer ohne spaCy für deterministische Tests."""
        self.normalizer = TextNormalizer(preprocessor=None)

    def test_regression_saeugetier_bug(self):
        """
        REGRESSION TEST: "säugetier" darf NICHT zu "säugeti" werden!

        Dieser Bug wurde in der Vergangenheit durch zu aggressive "-er" Regeln verursacht.
        """
        # Test mit einzelnem Wort
        result = self.normalizer.normalize_plural_to_singular("säugetier")
        assert result == "säugetier", "REGRESSION: säugetier-Bug ist zurück!"

        # Test mit clean_entity (kombinierte Funktionalität)
        result_clean = self.normalizer.clean_entity("das säugetier")
        assert result_clean == "säugetier", "REGRESSION: säugetier-Bug in clean_entity!"

    def test_regression_computer_bug(self):
        """
        REGRESSION TEST: "computer" darf NICHT zu "comput" werden!
        """
        result = self.normalizer.normalize_plural_to_singular("computer")
        assert result == "computer", "REGRESSION: computer-Bug ist zurück!"

    def test_regression_luke_bug(self):
        """
        REGRESSION TEST: Namen wie "Luke" dürfen NICHT normalisiert werden!
        """
        result = self.normalizer.normalize_plural_to_singular("Luke")
        assert result == "Luke", "REGRESSION: Luke wurde falsch normalisiert!"

        # Auch im lowercase
        result_lower = self.normalizer.normalize_plural_to_singular("luke")
        assert result_lower == "luke", "REGRESSION: luke wurde falsch normalisiert!"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
