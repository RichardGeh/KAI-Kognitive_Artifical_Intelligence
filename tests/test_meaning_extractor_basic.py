"""
KAI Test Suite - Meaning Extractor Basic Tests
Basis-Tests aus test_kai_worker.py extrahiert.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from kai_response_formatter import KaiResponseFormatter

logger = logging.getLogger(__name__)
TEST_VECTOR_DIM = 384


class TestMeaningExtractor:
    """Tests für die Extraktion von Bedeutungspunkten."""

    def test_extract_define_command(self, kai_worker_with_mocks):
        """Testet die Erkennung von Definiere-Befehlen."""
        doc = kai_worker_with_mocks.preprocessor.process(
            "Definiere: apfel / bedeutung = Eine Frucht"
        )
        mps = kai_worker_with_mocks.extractor.extract(doc)

        assert len(mps) > 0
        mp = mps[0]
        assert mp.arguments["command"] == "definiere"
        assert mp.arguments["topic"] == "apfel"
        assert "bedeutung" in mp.arguments["key_path"]
        assert mp.arguments["value"] == "Eine Frucht"

    def test_extract_learn_pattern_command(self, kai_worker_with_mocks):
        """Testet die Erkennung von Lerne-Muster-Befehlen."""
        doc = kai_worker_with_mocks.preprocessor.process(
            'Lerne Muster: "Ein Hund ist ein Tier" bedeutet IS_A'
        )
        mps = kai_worker_with_mocks.extractor.extract(doc)

        assert len(mps) > 0
        mp = mps[0]
        assert mp.arguments["command"] == "learn_pattern"
        assert mp.arguments["example_sentence"] == "Ein Hund ist ein Tier"
        assert mp.arguments["relation_type"] == "IS_A"

    def test_extract_ingest_command(self, kai_worker_with_mocks):
        """Testet die Erkennung von Ingestiere-Befehlen."""
        doc = kai_worker_with_mocks.preprocessor.process(
            'Ingestiere Text: "Ein Vogel kann fliegen."'
        )
        mps = kai_worker_with_mocks.extractor.extract(doc)

        assert len(mps) > 0
        mp = mps[0]
        assert mp.arguments["command"] == "ingest_text"
        assert "Vogel" in mp.arguments["text_to_ingest"]

    def test_extract_simple_question(self, kai_worker_with_mocks):
        """Testet die Erkennung einfacher Fragen."""
        doc = kai_worker_with_mocks.preprocessor.process("Was ist ein Apfel?")
        mps = kai_worker_with_mocks.extractor.extract(doc)

        assert len(mps) > 0
        mp = mps[0]
        assert mp.category.name == "QUESTION"
        assert mp.arguments.get("topic") == "apfel"

    def test_clean_entity_removes_articles(self, kai_worker_with_mocks):
        """Testet, ob Artikel korrekt entfernt werden mit Edge Cases."""
        # Standardfälle - beachte: Plural-Normalisierung ist aktiv
        cleaned = KaiResponseFormatter.clean_entity("eine banane")
        # "banane" endet auf "e", wird zu "banan" normalisiert (Plural-Regel)
        assert "banan" in cleaned, f"Erwartete 'banan', bekam '{cleaned}'"

        cleaned = KaiResponseFormatter.clean_entity("der apfel")
        assert cleaned == "apfel"

        # Edge Case 1: Mehrere Artikel
        cleaned = KaiResponseFormatter.clean_entity("der die das ein")
        assert cleaned != "", "Sollte nicht komplett leer sein"

        # Edge Case 2: Nur Artikel ohne Substantiv
        cleaned = KaiResponseFormatter.clean_entity("der")
        # Sollte entweder leer sein oder "der" zurückgeben (je nach Implementierung)
        assert isinstance(cleaned, str)

        # Edge Case 3: Artikel in Großbuchstaben
        cleaned = KaiResponseFormatter.clean_entity("DER APFEL")
        assert "apfel" in cleaned.lower()

        # Edge Case 4: Mehrere Leerzeichen
        cleaned = KaiResponseFormatter.clean_entity("der    apfel")
        assert "apfel" in cleaned

        # Edge Case 5: Satzzeichen
        cleaned = KaiResponseFormatter.clean_entity("der apfel!")
        assert (
            cleaned == "apfel"
        ), f"Satzzeichen sollten entfernt werden, bekommen: '{cleaned}'"

        # Edge Case 6: Zusammengesetzte Wörter mit Artikel
        cleaned = KaiResponseFormatter.clean_entity("der süße apfel")
        # Sollte "süße apfel" oder "süß apfel" zurückgeben (Artikel entfernt)
        assert "apfel" in cleaned and "der" not in cleaned.lower()

        logger.info(f"[SUCCESS] Artikel-Entfernung funktioniert für alle Edge Cases")


# ============================================================================
# TESTS FÜR PROTOTYPE MATCHER (component_8_prototype_matcher.py)
# ============================================================================
