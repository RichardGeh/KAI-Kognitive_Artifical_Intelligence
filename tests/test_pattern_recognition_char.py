# tests/test_pattern_recognition_char.py
"""Tests für Buchstaben-Ebene Pattern Recognition (Tippfehler-Korrektur)"""

import pytest
from component_1_netzwerk import KonzeptNetzwerk
from component_19_pattern_recognition_char import (
    keyboard_distance,
    weighted_levenshtein,
    TypoCandidateFinder,
)


@pytest.fixture
def netzwerk():
    """Neo4j Connection"""
    nw = KonzeptNetzwerk()
    # Füge Test-Wörter hinzu
    for word in ["katze", "kitze", "hund", "haus", "maus"]:
        nw.ensure_wort_und_konzept(f"test_{word}")
    yield nw
    nw.close()


class TestKeyboardDistance:
    """Tests für QWERTZ-Layout-basierte Distance"""

    def test_identical_chars(self):
        """Identische Zeichen = 0 Kosten"""
        assert keyboard_distance("a", "a") == 0.0

    def test_neighbor_keys(self):
        """Nachbar-Tasten = 0.3 Kosten"""
        assert keyboard_distance("k", "l") == 0.3
        assert keyboard_distance("e", "r") == 0.3

    def test_special_chars(self):
        """Sonderzeichen-Verwechslung = 0.5 Kosten"""
        assert keyboard_distance("ß", "s") == 0.5
        assert keyboard_distance("ä", "a") == 0.5

    def test_non_neighbors(self):
        """Nicht-Nachbarn = 1.0 Kosten"""
        assert keyboard_distance("a", "z") == 1.0


class TestWeightedLevenshtein:
    """Tests für gewichtete Levenshtein Distance"""

    def test_identical_words(self):
        """Identische Wörter = 0 Distance"""
        assert weighted_levenshtein("Katze", "Katze") == 0.0

    def test_neighbor_substitution(self):
        """Nachbar-Taste-Ersetzung = 0.3"""
        dist = weighted_levenshtein("Katze", "Katzr")  # e->r sind Nachbarn
        assert dist == 0.3

    def test_regular_substitution(self):
        """Reguläre Ersetzung = 1.0"""
        dist = weighted_levenshtein("Katze", "Katse")  # z->s nicht Nachbarn
        assert dist == 1.0


class TestTypoCandidateFinder:
    """Tests für Tippfehler-Kandidaten-Suche"""

    def test_find_candidates_for_typo(self, netzwerk):
        """Test: Findet Kandidaten für Tippfehler"""
        finder = TypoCandidateFinder(netzwerk)

        candidates = finder.find_candidates("test_ktzae", max_candidates=3)

        assert len(candidates) > 0
        # Sollte test_katze finden
        words = [c["word"] for c in candidates]
        assert "test_katze" in words

    def test_confidence_scoring(self, netzwerk):
        """Test: Confidence-Scores sind plausibel"""
        finder = TypoCandidateFinder(netzwerk)

        candidates = finder.find_candidates("test_ktzae")

        for candidate in candidates:
            # Confidence zwischen 0 und 1
            assert 0.0 <= candidate["confidence"] <= 1.0
            # Distance vorhanden
            assert "distance" in candidate


class TestFeedbackIntegration:
    """Tests für Feedback-Speicherung"""

    def test_store_positive_feedback(self, netzwerk):
        """Test: Positives Feedback speichern"""
        feedback_id = netzwerk.store_typo_feedback(
            original_input="test_ktzae",
            suggested_word="test_katze",
            actual_word="test_katze",
            user_accepted=True,
            confidence=0.92,
        )

        assert feedback_id is not None

    def test_store_negative_feedback(self, netzwerk):
        """Test: Negatives Feedback ('Nein, ich meine...')"""
        feedback_id = netzwerk.store_typo_feedback(
            original_input="test_ktzae",
            suggested_word="test_katze",
            actual_word="test_kitze",
            user_accepted=False,
            confidence=0.85,
            correction_reason="user_correction",
        )

        assert feedback_id is not None

        # Hole Feedback zurück
        feedback_list = netzwerk.get_typo_feedback_for_input("test_ktzae")
        assert len(feedback_list) > 0
        assert feedback_list[0]["actual_word"] == "test_kitze"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
