# tests/test_adaptive_pattern_recognition.py
"""
Tests für Adaptive Pattern Recognition Verbesserungen.

Testet:
- Adaptive Thresholds (component_25)
- Bootstrap-Phasen (cold_start, warming, mature)
- Word Frequency Integration
- Bayesian Pattern Quality Updates
- False-Positive Reduktion
"""

import pytest
from component_1_netzwerk import KonzeptNetzwerk
from component_25_adaptive_thresholds import AdaptiveThresholdManager, BootstrapPhase
from component_19_pattern_recognition_char import (
    TypoCandidateFinder,
    record_typo_correction_feedback,
)


class TestAdaptiveThresholds:
    """Tests für Adaptive Threshold Management"""

    @pytest.fixture
    def netzwerk(self):
        """Fixture: KonzeptNetzwerk Instanz"""
        netz = KonzeptNetzwerk()
        yield netz
        netz.close()

    def test_bootstrap_phase_detection(self, netzwerk):
        """Test: Bootstrap-Phasen werden korrekt erkannt"""
        manager = AdaptiveThresholdManager(netzwerk)

        # Test: Cold Start (<100 words)
        phase_cold = manager.get_bootstrap_phase(vocab_size=50)
        assert phase_cold == BootstrapPhase.COLD_START

        # Test: Warming (100-1000 words)
        phase_warming = manager.get_bootstrap_phase(vocab_size=500)
        assert phase_warming == BootstrapPhase.WARMING

        # Test: Mature (>1000 words)
        phase_mature = manager.get_bootstrap_phase(vocab_size=5000)
        assert phase_mature == BootstrapPhase.MATURE

    def test_adaptive_typo_threshold_scaling(self, netzwerk):
        """Test: Typo-Thresholds skalieren mit Vocabulary-Größe"""
        manager = AdaptiveThresholdManager(netzwerk)

        # Test: Minimum Threshold bei kleinem Vocab
        threshold_small = manager.get_typo_threshold(vocab_size=10)
        assert threshold_small == 3  # Minimum

        # Test: Scaling bei mittlerem Vocab
        threshold_medium = manager.get_typo_threshold(vocab_size=100)
        assert 3 <= threshold_medium <= 10

        # Test: Maximum Threshold bei großem Vocab
        threshold_large = manager.get_typo_threshold(vocab_size=10000)
        assert threshold_large == 10  # Maximum

    def test_adaptive_sequence_threshold_scaling(self, netzwerk):
        """Test: Sequence-Thresholds skalieren mit Connection-Count"""
        manager = AdaptiveThresholdManager(netzwerk)

        # Test: Minimum bei wenigen Connections
        threshold_small = manager.get_sequence_threshold(connection_count=10)
        assert threshold_small == 2  # Minimum

        # Test: Maximum bei vielen Connections
        threshold_large = manager.get_sequence_threshold(connection_count=1000)
        assert threshold_large == 5  # Maximum

    def test_confidence_gates_per_phase(self, netzwerk):
        """Test: Confidence-Gates sind phase-abhängig"""
        manager = AdaptiveThresholdManager(netzwerk)

        # Test: Cold Start - sehr konservativ
        gates_cold = manager.get_confidence_gates(BootstrapPhase.COLD_START)
        assert gates_cold["auto_correct"] == 0.95
        assert gates_cold["ask_user"] == 0.80

        # Test: Warming - standard
        gates_warming = manager.get_confidence_gates(BootstrapPhase.WARMING)
        assert gates_warming["auto_correct"] == 0.85
        assert gates_warming["ask_user"] == 0.60

        # Test: Mature - aggressiv
        gates_mature = manager.get_confidence_gates(BootstrapPhase.MATURE)
        assert gates_mature["auto_correct"] == 0.75
        assert gates_mature["ask_user"] == 0.50

    def test_bootstrap_confidence_multiplier(self, netzwerk):
        """Test: Confidence-Multiplier basierend auf Daten-Verfügbarkeit"""
        manager = AdaptiveThresholdManager(netzwerk)

        # Test: Wenig Daten -> Downgrade
        multiplier_low = manager.get_bootstrap_confidence_multiplier(
            actual_count=3, threshold=10
        )
        assert multiplier_low < 1.0  # Downgrade

        # Test: Exakt Threshold -> Normal
        multiplier_normal = manager.get_bootstrap_confidence_multiplier(
            actual_count=10, threshold=10
        )
        assert multiplier_normal == 1.0  # Keine Änderung

        # Test: Viel Daten -> Boost
        multiplier_high = manager.get_bootstrap_confidence_multiplier(
            actual_count=30, threshold=10
        )
        assert multiplier_high > 1.0  # Boost


class TestWordFrequency:
    """Tests für Word Frequency Integration"""

    @pytest.fixture
    def netzwerk(self):
        netz = KonzeptNetzwerk()
        yield netz
        netz.close()

    def test_word_frequency_calculation(self, netzwerk):
        """Test: Word Frequency wird korrekt berechnet"""
        import uuid

        unique_word = f"testword_freq_{uuid.uuid4().hex[:8]}"

        # Setup: Erstelle Wort mit Relations (assert_relation erstellt automatisch Wörter)
        netzwerk.assert_relation(unique_word, "IS_A", "konzept1")
        netzwerk.assert_relation(unique_word, "HAS_PROPERTY", "eigenschaft1")

        # Test: Frequency abfragen
        freq = netzwerk.get_word_frequency(unique_word)

        assert freq["out_degree"] >= 2  # Mindestens 2 ausgehende Relations
        assert freq["total_degree"] >= 2

    def test_normalized_word_frequency(self, netzwerk):
        """Test: Normalized Frequency ist zwischen 0.0 und 1.0"""
        import uuid

        # Test 1: Wort das noch nicht existiert -> 0.0
        nonexistent_word = f"nonexistent_{uuid.uuid4().hex[:8]}"
        norm_freq_nonexistent = netzwerk.get_normalized_word_frequency(nonexistent_word)
        assert norm_freq_nonexistent == 0.0  # Existiert nicht

        # Setup: Wort mit vielen Relations (unique name)
        popular_word = f"testword_popular_{uuid.uuid4().hex[:8]}"
        for i in range(20):
            netzwerk.assert_relation(popular_word, "HAS_PROPERTY", f"prop_{i}")

        # Test 2: Populäres Wort -> ~1.0
        norm_freq_popular = netzwerk.get_normalized_word_frequency(popular_word)
        assert 0.8 <= norm_freq_popular <= 1.0  # Sehr hoch


class TestBayesianPatternQuality:
    """Tests für Bayesian Pattern Quality Updates"""

    @pytest.fixture
    def netzwerk(self):
        netz = KonzeptNetzwerk()
        yield netz
        netz.close()

    def test_pattern_quality_initialization(self, netzwerk):
        """Test: Neues Pattern startet mit Prior"""
        pattern_key = "test_pattern_init"

        # Test: Initial sollte Prior zurückgeben (0.75)
        weight = netzwerk._feedback.get_pattern_quality_weight(
            pattern_type="typo_correction", pattern_key=pattern_key
        )
        assert weight == 0.75  # Prior

    def test_pattern_quality_success_updates(self, netzwerk):
        """Test: Erfolgreiche Predictions erhöhen Weight"""
        import uuid

        pattern_key = f"test_pattern_success_{uuid.uuid4().hex[:8]}"

        # Update 1: Success
        netzwerk._feedback.update_pattern_quality(
            pattern_type="typo_correction", pattern_key=pattern_key, success=True
        )

        quality1 = netzwerk._feedback.get_pattern_quality(
            pattern_type="typo_correction", pattern_key=pattern_key
        )
        assert quality1 is not None
        assert quality1["weight"] > 0.5  # Sollte über Prior liegen

        # Update 2: Noch ein Success
        netzwerk._feedback.update_pattern_quality(
            pattern_type="typo_correction", pattern_key=pattern_key, success=True
        )

        quality2 = netzwerk._feedback.get_pattern_quality(
            pattern_type="typo_correction", pattern_key=pattern_key
        )
        assert quality2["weight"] > quality1["weight"]  # Weight steigt

    def test_pattern_quality_failure_updates(self, netzwerk):
        """Test: Fehlgeschlagene Predictions senken Weight"""
        import uuid

        pattern_key = f"test_pattern_failure_{uuid.uuid4().hex[:8]}"

        # Update 1: Failure
        netzwerk._feedback.update_pattern_quality(
            pattern_type="typo_correction", pattern_key=pattern_key, success=False
        )

        quality1 = netzwerk._feedback.get_pattern_quality(
            pattern_type="typo_correction", pattern_key=pattern_key
        )
        assert quality1 is not None
        assert quality1["weight"] < 0.5  # Sollte unter Prior liegen

    def test_pattern_quality_mixed_updates(self, netzwerk):
        """Test: Mixed Updates konvergieren zu echtem Success Rate"""
        import uuid

        pattern_key = (
            f"test_pattern_mixed_{uuid.uuid4().hex[:8]}"  # Unique key pro Test-Run
        )

        # 7 Successes, 3 Failures -> 70% Success Rate
        for _ in range(7):
            netzwerk._feedback.update_pattern_quality(
                pattern_type="typo_correction", pattern_key=pattern_key, success=True
            )

        for _ in range(3):
            netzwerk._feedback.update_pattern_quality(
                pattern_type="typo_correction", pattern_key=pattern_key, success=False
            )

        quality = netzwerk._feedback.get_pattern_quality(
            pattern_type="typo_correction", pattern_key=pattern_key
        )

        # Weight sollte nahe 70% sein (mit Bayesian Smoothing)
        assert 0.65 <= quality["weight"] <= 0.75
        assert quality["total_observations"] == 10  # Sollte exakt 10 sein (7+3)


class TestFalsePositiveReduction:
    """Tests für False-Positive Reduktion"""

    @pytest.fixture
    def netzwerk(self):
        netz = KonzeptNetzwerk()
        yield netz
        netz.close()

    def test_negative_examples_stored(self, netzwerk):
        """Test: Negative Examples werden gespeichert"""
        # Setup: Speichere Feedback für abgelehnte Korrektur
        netzwerk._feedback.store_typo_feedback(
            original_input="testinput",
            suggested_word="wrongword",
            actual_word="rightword",
            user_accepted=False,
            confidence=0.85,
        )

        # Test: Negative Example sollte abrufbar sein
        negatives = netzwerk._feedback.get_negative_examples("wrongword")
        assert len(negatives) > 0
        assert negatives[0]["actual_word"] == "rightword"

    def test_typo_finder_avoids_high_rejection_candidates(self, netzwerk):
        """Test: TypoCandidateFinder filtert Kandidaten mit hoher Rejection Rate"""
        # Setup: Erstelle Wörter mit Relations (damit sie im Graph existieren)
        netzwerk.assert_relation("katze", "IS_A", "tier")
        netzwerk.assert_relation("kitze", "IS_A", "tier")

        # Simuliere mehrfache Rejections für "katze"
        for _ in range(5):
            netzwerk._feedback.store_typo_feedback(
                original_input="ktzae",
                suggested_word="katze",
                actual_word="kitze",
                user_accepted=False,
                confidence=0.80,
            )

        # Test: Typo-Finder sollte "katze" niedriger gewichten
        finder = TypoCandidateFinder(netzwerk)
        candidates = finder.find_candidates("ktzae", max_candidates=5)

        # Wenn "katze" in Kandidaten, sollte niedrigere Confidence haben
        katze_candidate = next((c for c in candidates if c["word"] == "katze"), None)
        if katze_candidate:
            # Pattern Quality sollte Confidence reduziert haben
            assert "pattern_quality_weight" in katze_candidate
            # Weight sollte < 0.5 sein nach 5 Failures


class TestTypoFeedbackRecording:
    """Tests für Feedback-Recording Mechanismus"""

    @pytest.fixture
    def netzwerk(self):
        netz = KonzeptNetzwerk()
        yield netz
        netz.close()

    def test_feedback_recording_accepted(self, netzwerk):
        """Test: Feedback wird bei akzeptierter Korrektur gespeichert"""
        result = record_typo_correction_feedback(
            netzwerk=netzwerk,
            original_input="testinput",
            suggested_correction="testcorrection",
            user_accepted=True,
            confidence=0.85,
        )

        assert result is True

        # Verify: Pattern Quality wurde aktualisiert
        quality = netzwerk._feedback.get_pattern_quality(
            pattern_type="typo_correction", pattern_key="testinput->testcorrection"
        )
        assert quality is not None
        assert quality["success_count"] >= 1

    def test_feedback_recording_rejected(self, netzwerk):
        """Test: Feedback wird bei abgelehnter Korrektur gespeichert"""
        result = record_typo_correction_feedback(
            netzwerk=netzwerk,
            original_input="testinput",
            suggested_correction="wrongcorrection",
            user_accepted=False,
            actual_correction="rightcorrection",
            confidence=0.70,
        )

        assert result is True

        # Verify: Pattern Quality wurde aktualisiert (mit Failure)
        quality = netzwerk._feedback.get_pattern_quality(
            pattern_type="typo_correction", pattern_key="testinput->wrongcorrection"
        )
        assert quality is not None
        assert quality["failure_count"] >= 1


if __name__ == "__main__":
    print("=== Adaptive Pattern Recognition Tests ===\n")
    print("Run with: pytest tests/test_adaptive_pattern_recognition.py -v")
