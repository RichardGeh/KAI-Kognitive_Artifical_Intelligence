"""
KAI Test Suite - Enhanced NLP Tests
Extrahiert aus test_kai_worker.py fuer bessere Wartbarkeit.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

logger = logging.getLogger(__name__)
TEST_VECTOR_DIM = 384


class TestEnhancedNLP:
    """Tests für verbesserte NLP-Fähigkeiten: Fuzzy Matching, erweiterte Frageformate."""

    def test_find_similar_words_basic(
        self, netzwerk_session, embedding_service_session, clean_test_concepts
    ):
        """Testet grundlegendes Fuzzy-Matching für Tippfehler."""
        # Setup: Erstelle einige bekannte Wörter
        known_words = [
            f"{clean_test_concepts}katze",
            f"{clean_test_concepts}hund",
            f"{clean_test_concepts}vogel",
        ]

        for word in known_words:
            netzwerk_session.ensure_wort_und_konzept(word)

        # TEST: Suche nach ähnlichem Wort (simulierter Tippfehler)
        query_word = f"{clean_test_concepts}kaze"  # Tippfehler von "katze"

        similar = netzwerk_session.find_similar_words(
            query_word,
            embedding_service_session,
            similarity_threshold=0.65,
            max_results=3,
        )

        # Sollte mindestens einen Vorschlag finden
        assert len(similar) > 0, "Fuzzy-Matching sollte ähnliche Wörter finden"

        # Der beste Match sollte einer unserer Test-Wörter sein
        best_match = similar[0]
        found_words = [word for word in known_words if word in best_match["word"]]
        assert (
            len(found_words) > 0
        ), f"Bester Match sollte eines unserer Test-Wörter enthalten, ist aber: {best_match['word']}"
        assert (
            best_match["similarity"] >= 0.65
        ), f"Similarity sollte >= 0.65 sein, ist aber {best_match['similarity']}"

        logger.info(
            f"[SUCCESS] Fuzzy-Matching fand: {similar[0]['word']} ({similar[0]['similarity']:.2%})"
        )

    def test_fuzzy_matching_in_question_workflow(
        self, kai_worker_with_mocks, clean_test_concepts
    ):
        """Testet ob Fuzzy-Matching bei unbekannten Wörtern in Fragen funktioniert."""
        # Setup: Erstelle bekanntes Wort
        known_word = f"{clean_test_concepts}apfel_fuzzy"
        kai_worker_with_mocks.netzwerk.ensure_wort_und_konzept(known_word)
        kai_worker_with_mocks.netzwerk.assert_relation(
            known_word, "IS_A", f"{clean_test_concepts}frucht_fuzzy", "test"
        )

        # Frage mit Tippfehler
        typo_word = f"{clean_test_concepts}apfle_fuzzy"  # Vertauschte Buchstaben
        query = f"Was ist ein {typo_word}?"

        kai_worker_with_mocks.process_query(query)

        response = kai_worker_with_mocks.signals.finished.emit.call_args.args[0]
        response_lower = response.text.lower()

        # Sollte entweder:
        # 1. Vorschlag machen ("Meintest du vielleicht...?")
        # 2. Oder direkt Wissen über das ähnliche Wort zeigen (wenn Fuzzy-Match hoch genug)
        assert any(
            keyword in response_lower
            for keyword in ["meintest du", "ähnlich", known_word, "frucht_fuzzy"]
        ), f"Sollte Fuzzy-Match-Vorschlag oder Wissen zeigen. Response: {response.text}"

        logger.info("[SUCCESS] Fuzzy-Matching im Frage-Workflow funktioniert")

    def test_show_all_knowledge_query(self, kai_worker_with_mocks, clean_test_concepts):
        """Testet die neue 'Was weißt du über X alles?' Frageform."""
        # Setup: Erstelle umfassendes Wissen über ein Konzept
        test_word = f"{clean_test_concepts}comprehensive_test"

        kai_worker_with_mocks.netzwerk.assert_relation(
            test_word, "IS_A", f"{clean_test_concepts}tier", "test1"
        )
        kai_worker_with_mocks.netzwerk.assert_relation(
            test_word, "HAS_PROPERTY", f"{clean_test_concepts}groß", "test2"
        )
        kai_worker_with_mocks.netzwerk.assert_relation(
            test_word, "CAPABLE_OF", f"{clean_test_concepts}laufen", "test3"
        )
        kai_worker_with_mocks.netzwerk.add_information_zu_wort(
            test_word, "bedeutung", "Ein besonderes Tier"
        )

        # TEST: Frage mit "alles" Keyword
        query = f"Was weißt du über {test_word} alles?"
        kai_worker_with_mocks.process_query(query)

        response = kai_worker_with_mocks.signals.finished.emit.call_args.args[0]
        response_lower = response.text.lower()

        # Sollte ALLE Fakten enthalten
        assert "tier" in response_lower, "Sollte IS_A Relation enthalten"
        assert "groß" in response_lower, "Sollte HAS_PROPERTY Relation enthalten"
        assert "laufen" in response_lower, "Sollte CAPABLE_OF Relation enthalten"
        assert "besonderes tier" in response_lower, "Sollte Bedeutung enthalten"

        logger.info("[SUCCESS] 'Was weißt du alles über X?' zeigt umfassendes Wissen")

    def test_alternative_show_all_query_format(
        self, kai_worker_with_mocks, clean_test_concepts
    ):
        """Testet alternative Formulierung 'Zeige alles über X'."""
        test_word = f"{clean_test_concepts}show_test"

        kai_worker_with_mocks.netzwerk.assert_relation(
            test_word, "IS_A", f"{clean_test_concepts}objekt", "test"
        )

        # Alternative Formulierung
        query = f"Zeige alles über {test_word}"
        kai_worker_with_mocks.process_query(query)

        response = kai_worker_with_mocks.signals.finished.emit.call_args.args[0]

        # Sollte erkannt werden
        assert (
            "objekt" in response.text.lower()
        ), f"Sollte Wissen über {test_word} zeigen. Response: {response.text}"

        logger.info(
            "[SUCCESS] Alternative Formulierung 'Zeige alles über X' funktioniert"
        )

    def test_flexible_question_word_recognition(
        self, kai_worker_with_mocks, clean_test_concepts
    ):
        """Testet erweiterte Fragewort-Erkennung (wie, warum, wer, etc.)."""
        # WICHTIG: Stelle sicher dass kein Kontext aktiv ist
        kai_worker_with_mocks.context.clear()

        test_word = f"{clean_test_concepts}flexible_q"

        kai_worker_with_mocks.netzwerk.ensure_wort_und_konzept(test_word)
        kai_worker_with_mocks.netzwerk.assert_relation(
            test_word, "IS_A", f"{clean_test_concepts}konzept", "test"
        )

        # Verschiedene Frageworte
        questions = [
            f"Wie funktioniert {test_word}?",
            f"Warum ist {test_word} wichtig?",
            f"Wer kennt {test_word}?",
        ]

        for question in questions:
            # Kontext zurücksetzen vor jeder Frage
            kai_worker_with_mocks.context.clear()
            kai_worker_with_mocks.signals.reset_mock()
            kai_worker_with_mocks.process_query(question)

            # Sollte nicht crashen und eine Antwort geben
            assert (
                kai_worker_with_mocks.signals.finished.emit.called
            ), f"Sollte für Frage '{question}' eine Antwort geben"

            response = kai_worker_with_mocks.signals.finished.emit.call_args.args[0]

            # Sollte entweder Wissen zeigen oder Wissenslücke melden
            response_lower = response.text.lower()
            assert any(
                keyword in response_lower
                for keyword in [
                    test_word,
                    "weiß nichts",
                    "konzept",
                    "nicht sicher",
                    "anders formulieren",
                ]
            ), f"Sollte relevante Antwort auf '{question}' geben. Response: {response.text}"

        logger.info("[SUCCESS] Flexible Fragewort-Erkennung funktioniert")

    def test_fuzzy_matching_no_suggestions_when_exact_match(
        self, netzwerk_session, embedding_service_session, clean_test_concepts
    ):
        """Testet dass Fuzzy-Matching keine Vorschläge macht bei exakter Übereinstimmung."""
        exact_word = f"{clean_test_concepts}exakt_unique_word"
        netzwerk_session.ensure_wort_und_konzept(exact_word)

        # Suche nach exakt demselben Wort
        similar = netzwerk_session.find_similar_words(
            exact_word,
            embedding_service_session,
            similarity_threshold=0.70,
            max_results=3,
        )

        # Sollte KEINE Vorschläge geben die exakt dem Suchwort entsprechen
        # (aber es könnten andere ähnliche Wörter gefunden werden)
        exact_matches = [s for s in similar if s["word"].lower() == exact_word.lower()]
        assert (
            len(exact_matches) == 0
        ), f"Exakte Übereinstimmung sollte ausgefiltert sein, bekam aber: {exact_matches}"

        logger.info(
            f"[SUCCESS] Fuzzy-Matching filtert exakte Übereinstimmungen korrekt aus ({len(similar)} andere ähnliche Wörter gefunden)"
        )

    def test_fuzzy_matching_threshold_behavior(
        self, netzwerk_session, embedding_service_session, clean_test_concepts
    ):
        """Testet dass Fuzzy-Matching Threshold korrekt angewendet wird."""
        # Setup
        word1 = f"{clean_test_concepts}banane_threshold"
        word2 = f"{clean_test_concepts}apfel_threshold"

        netzwerk_session.ensure_wort_und_konzept(word1)
        netzwerk_session.ensure_wort_und_konzept(word2)

        # Suche mit sehr hohem Threshold (0.95)
        similar_high = netzwerk_session.find_similar_words(
            f"{clean_test_concepts}banan_threshold",  # Kleiner Tippfehler
            embedding_service_session,
            similarity_threshold=0.95,
            max_results=3,
        )

        # Suche mit niedrigem Threshold (0.60)
        similar_low = netzwerk_session.find_similar_words(
            f"{clean_test_concepts}banan_threshold",
            embedding_service_session,
            similarity_threshold=0.60,
            max_results=3,
        )

        # Niedrigerer Threshold sollte mehr oder gleich viele Ergebnisse liefern
        assert len(similar_low) >= len(
            similar_high
        ), f"Niedrigerer Threshold sollte mehr Ergebnisse liefern. Low: {len(similar_low)}, High: {len(similar_high)}"

        logger.info(
            f"[SUCCESS] Threshold-Verhalten korrekt: High={len(similar_high)}, Low={len(similar_low)}"
        )

    def test_get_all_known_words(self, netzwerk_session, clean_test_concepts):
        """Testet die get_all_known_words() Hilfsmethode."""
        # Setup: Erstelle mehrere Wörter
        test_words = [
            f"{clean_test_concepts}word1",
            f"{clean_test_concepts}word2",
            f"{clean_test_concepts}word3",
        ]

        for word in test_words:
            netzwerk_session.ensure_wort_und_konzept(word)

        # TEST
        all_words = netzwerk_session.get_all_known_words()

        # Sollte mindestens unsere Test-Wörter enthalten
        for test_word in test_words:
            assert (
                test_word in all_words
            ), f"'{test_word}' sollte in bekannten Wörtern sein"

        # Sollte sortiert sein
        assert all_words == sorted(
            all_words
        ), "Wörter sollten sortiert zurückgegeben werden"

        logger.info(
            f"[SUCCESS] get_all_known_words() gibt {len(all_words)} Wörter zurück (inkl. Test-Wörter)"
        )


# ============================================================================
# TESTS FÜR MULTI-TURN-DIALOG-SYSTEM (PHASE 2)
# ============================================================================
