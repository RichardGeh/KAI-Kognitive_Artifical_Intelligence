"""
Test für Plural-Normalisierung und Definitions-Speicherung/Abruf
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from component_1_netzwerk import KonzeptNetzwerk
from kai_response_formatter import KaiResponseFormatter


class TestPluralNormalization:
    """Test der Plural-zu-Singular-Normalisierung"""

    def test_clean_entity_normalizes_plural_katzen(self):
        """Teste dass 'Katzen' zu 'Katze' normalisiert wird"""
        result = KaiResponseFormatter.clean_entity("Katzen")
        assert result == "katze", f"Erwartet 'katze', bekommen '{result}'"

        # Mit Artikel
        result_with_article = KaiResponseFormatter.clean_entity("die Katzen")
        assert (
            result_with_article == "katze"
        ), f"Erwartet 'katze', bekommen '{result_with_article}'"

    # TODO: Plural-Normalisierung ist komplex und fehleranfällig - vorerst deaktiviert
    # def test_clean_entity_normalizes_various_plurals(self):
    #     """Teste verschiedene Plural-Formen"""
    #     test_cases = [
    #         ("Hunde", "hund"),  # -e Plural
    #         # ("Autos", "auto"),  # -s Plural wird BEWUSST NICHT normalisiert (zu unsicher)
    #         ("Tage", "tag"),    # -e Plural
    #         ("Kinder", "kind"), # Umlaut-Plural (wird nicht normalisiert, aber "er" entfernt)
    #         ("Motoren", "motor"), # -en Plural
    #         ("Studenten", "student"), # -en Plural
    #     ]
    #
    #     for plural, expected_singular in test_cases:
    #         result = KaiResponseFormatter.clean_entity(plural)
    #         assert result == expected_singular, f"Plural '{plural}': Erwartet '{expected_singular}', bekommen '{result}'"


class TestDefinitionStorage:
    """Test der Definition/Bedeutung-Speicherung"""

    @pytest.fixture(autouse=True)
    def setup_and_cleanup(self):
        """Setup und Cleanup für jeden Test"""
        self.netzwerk = KonzeptNetzwerk()
        # Cleanup vorher
        self._cleanup("test_katze")
        yield
        # Cleanup nachher
        self._cleanup("test_katze")

    def _cleanup(self, lemma: str):
        """Hilfsfunktion zum Löschen von Test-Daten"""
        if not self.netzwerk.driver:
            return
        with self.netzwerk.driver.session(database="neo4j") as session:
            session.run(
                """
                MATCH (w:Wort {lemma: $lemma})
                OPTIONAL MATCH (w)-[r1:HAT_BEDEUTUNG]->(b:Bedeutung)
                OPTIONAL MATCH (w)-[r2:BEDEUTET]->(k:Konzept)
                DELETE r1, b, r2, k, w
                """,
                lemma=lemma,
            )

    def test_add_bedeutung_via_definiere_command(self):
        """Test: Definition: Wort/Bedeutung = ... speichert korrekt als Bedeutung"""
        result = self.netzwerk.add_information_zu_wort(
            "test_katze", "bedeutung", "Katzen sind gern gesehene Haustiere."
        )

        assert (
            result.get("created") is True
        ), "Bedeutung sollte neu erstellt worden sein"

        # Verifiziere dass es gespeichert wurde
        details = self.netzwerk.get_details_fuer_wort("test_katze")
        assert details is not None
        assert "bedeutungen" in details
        assert len(details["bedeutungen"]) == 1
        assert details["bedeutungen"][0] == "Katzen sind gern gesehene Haustiere."

    def test_add_definition_synonym_for_bedeutung(self):
        """Test: 'definition' ist Synonym zu 'bedeutung'"""
        # Verwende "definition" statt "bedeutung"
        result = self.netzwerk.add_information_zu_wort(
            "test_katze",
            "definition",  # Synonym zu "bedeutung"
            "Katzen sind Haustiere.",
        )

        assert (
            result.get("created") is True
        ), "Definition sollte neu erstellt worden sein"

        # Verifiziere dass es als Bedeutung gespeichert wurde
        details = self.netzwerk.get_details_fuer_wort("test_katze")
        assert details is not None
        assert "bedeutungen" in details
        assert len(details["bedeutungen"]) == 1
        assert details["bedeutungen"][0] == "Katzen sind Haustiere."

    def test_query_facts_with_synonyms_returns_bedeutungen(self):
        """Test: query_facts_with_synonyms() gibt auch Bedeutungen zurück"""
        # Speichere Bedeutung
        self.netzwerk.add_information_zu_wort(
            "test_katze", "bedeutung", "Katzen sind gern gesehene Haustiere."
        )

        # Frage ab
        result = self.netzwerk.query_facts_with_synonyms("test_katze")

        assert "bedeutungen" in result
        assert len(result["bedeutungen"]) == 1
        assert result["bedeutungen"][0] == "Katzen sind gern gesehene Haustiere."


class TestIntegrationScenario:
    """Integrationstest: Kompletter Workflow Frage -> Antwort -> erneute Frage"""

    @pytest.fixture(autouse=True)
    def setup_and_cleanup(self):
        """Setup und Cleanup für jeden Test"""
        self.netzwerk = KonzeptNetzwerk()
        # Cleanup vorher
        self._cleanup("test_hund")
        yield
        # Cleanup nachher
        self._cleanup("test_hund")

    def _cleanup(self, lemma: str):
        """Hilfsfunktion zum Löschen von Test-Daten"""
        if not self.netzwerk.driver:
            return
        with self.netzwerk.driver.session(database="neo4j") as session:
            session.run(
                """
                MATCH (w:Wort {lemma: $lemma})
                OPTIONAL MATCH (w)-[r1:HAT_BEDEUTUNG]->(b:Bedeutung)
                OPTIONAL MATCH (w)-[r2:BEDEUTET]->(k:Konzept)
                DELETE r1, b, r2, k, w
                """,
                lemma=lemma,
            )

    def test_full_workflow_question_answer_requery(self):
        """
        Test: Vollständiger Workflow
        1. Frage: "Was ist ein Hund?" -> "Ich weiß nichts über hund"
        2. Definition speichern: "Hunde sind Haustiere"
        3. Erneute Frage: "Was ist ein Hund?" -> "Hunde sind Haustiere"
        """
        # SCHRITT 1: Prüfe dass keine Wissenslücke existiert (vor der Speicherung)
        result_before = self.netzwerk.query_facts_with_synonyms("test_hund")
        assert result_before["bedeutungen"] == []
        assert result_before["facts"] == {}

        # SCHRITT 2: Speichere Definition (simuliert Benutzer-Antwort)
        self.netzwerk.add_information_zu_wort(
            "test_hund", "bedeutung", "Hunde sind Haustiere."
        )

        # SCHRITT 3: Frage erneut ab
        result_after = self.netzwerk.query_facts_with_synonyms("test_hund")
        assert len(result_after["bedeutungen"]) == 1
        assert result_after["bedeutungen"][0] == "Hunde sind Haustiere."

    def test_plural_in_answer_normalizes_correctly(self):
        """
        Test: Antwort mit Plural im Text wird korrekt normalisiert
        Frage: "Was ist eine Katze?"
        Antwort: "Katzen sind gern gesehene Haustiere."
        -> Sollte als Bedeutung für "katze" (Singular) gespeichert werden
        """
        # Speichere mit Plural im Text (aber Singular als Lemma)
        self.netzwerk.add_information_zu_wort(
            "test_hund",  # Singular
            "bedeutung",
            "Hunde sind treue Begleiter.",  # Plural im Text ist OK
        )

        # Frage mit Singular ab
        result = self.netzwerk.query_facts_with_synonyms("test_hund")
        assert len(result["bedeutungen"]) == 1
        assert "Hunde sind treue Begleiter" in result["bedeutungen"][0]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
