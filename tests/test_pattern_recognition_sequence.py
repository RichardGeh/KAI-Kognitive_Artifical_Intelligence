# tests/test_pattern_recognition_sequence.py
"""Tests für Wortfolgen-Mustererkennung"""

import pytest
from component_1_netzwerk import KonzeptNetzwerk
from component_20_pattern_recognition_sequence import SequencePredictor


@pytest.fixture
def netzwerk_with_sequences():
    """Neo4j mit Test-Sequenzen"""
    nw = KonzeptNetzwerk()

    # Erstelle Test-Wörter
    words = ["das", "haus", "ist", "groß", "rot", "alt"]
    for word in words:
        nw.ensure_wort_und_konzept(word)

    # Erstelle Sequences (simuliert mehrfaches Vorkommen)
    # "das Haus ist groß" (10x)
    for _ in range(10):
        nw.add_word_connection("das", "haus", distance=1, direction="before")
        nw.add_word_connection("haus", "ist", distance=1, direction="before")
        nw.add_word_connection("ist", "groß", distance=1, direction="before")

    # "das Haus ist rot" (5x)
    for _ in range(5):
        nw.add_word_connection("ist", "rot", distance=1, direction="before")

    yield nw
    nw.close()


class TestSequencePredictor:
    """Tests für Sequence Prediction"""

    def test_predict_next_word(self, netzwerk_with_sequences):
        """Test: Sagt nächstes Wort vorher"""
        predictor = SequencePredictor(netzwerk_with_sequences)

        predictions = predictor.predict_next_word(["ist"])

        assert len(predictions) > 0
        # "groß" sollte höchste Confidence haben (10x vs 5x)
        assert predictions[0]["word"] == "groß"
        assert predictions[0]["count"] >= 10

    def test_bootstrap_mechanism(self, netzwerk_with_sequences):
        """Test: Bootstrap - nur Sequences mit genug Vorkommen"""
        predictor = SequencePredictor(netzwerk_with_sequences)
        predictor.min_sequence_count = 8  # Erhöhe Schwellenwert

        predictions = predictor.predict_next_word(["ist"])

        # "groß" (10x) sollte drin sein, "rot" (5x) nicht
        words = [p["word"] for p in predictions]
        assert "groß" in words
        assert "rot" not in words

    def test_sentence_completion(self, netzwerk_with_sequences):
        """Test: Vervollständigt Satz"""
        predictor = SequencePredictor(netzwerk_with_sequences)

        predictions = predictor.predict_completion("das Haus ist")

        assert len(predictions) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
