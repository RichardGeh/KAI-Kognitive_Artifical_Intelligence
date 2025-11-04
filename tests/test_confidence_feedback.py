# tests/test_confidence_feedback.py
"""
Tests für das Confidence Feedback & Training System

Testet:
- Feedback-Erfassung
- Confidence-Adjustierung
- Historischer Konsensus
- Persistierung in Neo4j (Mock)
- Statistiken und Lernkurven
"""
import pytest

from component_confidence_feedback import (
    ConfidenceFeedbackManager,
    FeedbackType,
    get_feedback_manager,
)


class TestFeedbackCollection:
    """Tests für Feedback-Erfassung"""

    def test_submit_positive_feedback(self):
        """Teste positives Feedback (CORRECT)"""
        fm = ConfidenceFeedbackManager()

        fb = fm.submit_positive_feedback(
            relation_type="IS_A", subject="hund", object="tier", original_confidence=0.7
        )

        assert fb.feedback_type == FeedbackType.CORRECT
        assert fb.relation_type == "IS_A"
        assert fb.subject == "hund"
        assert fb.object == "tier"
        assert fb.original_confidence == 0.7
        # Positives Feedback sollte Confidence erhöhen
        assert fb.adjusted_confidence > 0.7

    def test_submit_negative_feedback(self):
        """Teste negatives Feedback (INCORRECT)"""
        fm = ConfidenceFeedbackManager()

        fb = fm.submit_negative_feedback(
            relation_type="IS_A",
            subject="wal",
            object="fisch",
            original_confidence=0.85,
            comment="Wal ist kein Fisch!",
        )

        assert fb.feedback_type == FeedbackType.INCORRECT
        assert fb.user_comment == "Wal ist kein Fisch!"
        # Negatives Feedback sollte Confidence stark reduzieren
        assert fb.adjusted_confidence < 0.85

    def test_submit_uncertain_feedback(self):
        """Teste unsicheres Feedback (UNCERTAIN)"""
        fm = ConfidenceFeedbackManager()

        fb = fm.submit_feedback(
            relation_type="HAS_PROPERTY",
            subject="apfel",
            object="rot",
            original_confidence=0.6,
            feedback_type=FeedbackType.UNCERTAIN,
        )

        assert fb.feedback_type == FeedbackType.UNCERTAIN
        # Unsicheres Feedback sollte Confidence leicht reduzieren
        assert fb.adjusted_confidence < 0.6
        assert fb.adjusted_confidence > 0.5  # Nicht zu stark


class TestConfidenceAdjustment:
    """Tests für Confidence-Adjustierung"""

    def test_positive_feedback_increases_confidence(self):
        """Teste dass positives Feedback Confidence erhöht"""
        fm = ConfidenceFeedbackManager()

        fb = fm.submit_positive_feedback("IS_A", "katze", "tier", 0.5)

        assert fb.adjusted_confidence > 0.5
        assert fb.adjusted_confidence <= 1.0  # Max 1.0

    def test_negative_feedback_decreases_confidence(self):
        """Teste dass negatives Feedback Confidence reduziert"""
        fm = ConfidenceFeedbackManager()

        fb = fm.submit_negative_feedback("IS_A", "sonne", "planet", 0.8)

        assert fb.adjusted_confidence < 0.8
        assert fb.adjusted_confidence >= 0.0  # Min 0.0

    def test_confidence_bounded(self):
        """Teste dass Confidence immer zwischen 0 und 1 bleibt"""
        fm = ConfidenceFeedbackManager()

        # Teste obere Grenze
        fb_high = fm.submit_positive_feedback("IS_A", "test", "test", 0.95)
        assert fb_high.adjusted_confidence <= 1.0

        # Teste untere Grenze
        fb_low = fm.submit_negative_feedback("IS_A", "test", "test", 0.1)
        assert fb_low.adjusted_confidence >= 0.0

    def test_adjustment_strength(self):
        """Teste dass Adjustierung-Stärke sinnvoll ist"""
        fm = ConfidenceFeedbackManager()
        fm.adjustment_rate = 0.1  # 10% Adjustierung

        fb = fm.submit_positive_feedback("IS_A", "vogel", "tier", 0.5)

        # Bei 50% Confidence, sollte Increase ca. 0.1 * (1.0 - 0.5) = 0.05 sein
        expected = 0.5 + 0.1 * (1.0 - 0.5)
        assert fb.adjusted_confidence == pytest.approx(expected, abs=0.01)


class TestHistoricalConsensus:
    """Tests für historischen Konsensus"""

    def test_consensus_from_multiple_feedback(self):
        """Teste Konsensus-Berechnung aus mehreren Feedbacks"""
        fm = ConfidenceFeedbackManager()
        fm.min_feedback_count = 3

        # Sende 3 positive Feedbacks
        for _ in range(3):
            fm.submit_positive_feedback("IS_A", "hund", "tier", 0.6)

        # Nächstes Feedback sollte Konsensus berücksichtigen
        fb = fm.submit_positive_feedback("IS_A", "hund", "tier", 0.6)

        # Mit 4 positiven Feedbacks sollte Confidence nahe 1.0 sein
        assert fb.adjusted_confidence > 0.7

    def test_mixed_feedback_consensus(self):
        """Teste Konsensus bei gemischtem Feedback"""
        fm = ConfidenceFeedbackManager()
        fm.min_feedback_count = 4

        # 3 positive, 1 negatives Feedback
        for _ in range(3):
            fm.submit_positive_feedback("HAS_PROPERTY", "rose", "rot", 0.7)
        fm.submit_negative_feedback("HAS_PROPERTY", "rose", "rot", 0.7)

        # Nächstes Feedback
        fb = fm.submit_positive_feedback("HAS_PROPERTY", "rose", "rot", 0.7)

        # Konsensus: 4/5 = 0.8 correct
        # Sollte Confidence etwas erhöhen, aber nicht zu viel wegen mixed feedback
        assert 0.7 < fb.adjusted_confidence < 0.9

    def test_consensus_requires_minimum_feedback(self):
        """Teste dass Konsensus erst nach min_feedback_count greift"""
        fm = ConfidenceFeedbackManager()
        fm.min_feedback_count = 5

        # Sende 2 Feedbacks (unter Minimum)
        fm.submit_positive_feedback("IS_A", "test", "test", 0.5)
        fb = fm.submit_positive_feedback("IS_A", "test", "test", 0.5)

        # Konsensus sollte noch NICHT greifen
        # Adjustierung sollte nur vom einzelnen Feedback kommen
        assert len(fm._feedback_history) == 2
        # Prüfe dass es keine dramatische Änderung gibt (Konsensus würde stärker ändern)
        assert 0.5 < fb.adjusted_confidence < 0.6


class TestFeedbackPersistence:
    """Tests für Persistierung (ohne echte Neo4j-Connection)"""

    def test_feedback_stored_in_history(self):
        """Teste dass Feedback in Historie gespeichert wird"""
        fm = ConfidenceFeedbackManager()

        initial_count = len(fm._feedback_history)

        fm.submit_positive_feedback("IS_A", "baum", "pflanze", 0.8)

        assert len(fm._feedback_history) == initial_count + 1

    def test_persistence_without_netzwerk(self):
        """Teste dass Persistierung ohne Netzwerk gracefully failstest"""
        fm = ConfidenceFeedbackManager(netzwerk=None)

        # Sollte keine Exception werfen
        fb = fm.submit_positive_feedback("IS_A", "test", "test", 0.7)
        assert fb is not None

    def test_get_feedback_for_relation(self):
        """Teste Abrufen von Feedback für spezifische Relation"""
        fm = ConfidenceFeedbackManager()

        # Sende Feedback für verschiedene Relationen
        fm.submit_positive_feedback("IS_A", "hund", "tier", 0.7)
        fm.submit_positive_feedback("IS_A", "hund", "tier", 0.7)
        fm.submit_positive_feedback("HAS_PROPERTY", "hund", "vierbeinig", 0.8)

        # Hole Feedback nur für "hund IS_A tier"
        feedback_list = fm.get_feedback_for_relation("IS_A", "hund", "tier")

        assert len(feedback_list) == 2
        assert all(fb.relation_type == "IS_A" for fb in feedback_list)
        assert all(fb.subject == "hund" for fb in feedback_list)


class TestStatisticsAndReporting:
    """Tests für Statistiken und Reporting"""

    def test_get_feedback_statistics(self):
        """Teste Statistik-Generierung"""
        fm = ConfidenceFeedbackManager()

        # Sende gemischtes Feedback
        fm.submit_positive_feedback("IS_A", "test1", "test1", 0.7)
        fm.submit_positive_feedback("IS_A", "test2", "test2", 0.6)
        fm.submit_negative_feedback("IS_A", "test3", "test3", 0.8)
        fm.submit_feedback("IS_A", "test4", "test4", 0.5, FeedbackType.UNCERTAIN)

        stats = fm.get_feedback_statistics()

        assert stats["total_feedback"] == 4
        assert stats["correct_count"] == 2
        assert stats["incorrect_count"] == 1
        assert stats["uncertain_count"] == 1
        assert stats["accuracy_rate"] == 2 / 4  # 50%
        assert stats["average_adjustment"] > 0.0

    def test_get_learning_curve(self):
        """Teste Lernkurven-Berechnung"""
        fm = ConfidenceFeedbackManager()

        # Sende 20 Feedbacks (überwiegend positiv am Anfang, gemischt später)
        for i in range(20):
            if i < 10:
                fm.submit_positive_feedback("IS_A", f"test{i}", f"test{i}", 0.7)
            else:
                # Gemischt
                if i % 2 == 0:
                    fm.submit_positive_feedback("IS_A", f"test{i}", f"test{i}", 0.7)
                else:
                    fm.submit_negative_feedback("IS_A", f"test{i}", f"test{i}", 0.7)

        curve = fm.get_learning_curve(window_size=10)

        assert len(curve) > 0
        # Erste Messung sollte 100% sein (10 positive)
        assert curve[0] == 1.0
        # Spätere Messungen sollten niedriger sein (gemischt)
        assert curve[-1] < 1.0

    def test_empty_statistics(self):
        """Teste Statistiken bei leerem Feedback"""
        fm = ConfidenceFeedbackManager()

        stats = fm.get_feedback_statistics()

        assert stats["total_feedback"] == 0
        assert stats["accuracy_rate"] == 0.0


class TestGlobalInstance:
    """Tests für globale Feedback-Manager-Instanz"""

    def test_get_feedback_manager_singleton(self):
        """Teste dass get_feedback_manager() Singleton ist"""
        fm1 = get_feedback_manager()
        fm2 = get_feedback_manager()

        assert fm1 is fm2  # Gleiche Objekt-Identität


class TestEdgeCases:
    """Tests für Edge-Cases"""

    def test_feedback_with_very_high_confidence(self):
        """Teste Feedback bei sehr hoher ursprünglicher Confidence"""
        fm = ConfidenceFeedbackManager()

        fb = fm.submit_positive_feedback("IS_A", "test", "test", 0.99)

        # Sollte trotzdem nicht über 1.0 gehen
        assert fb.adjusted_confidence <= 1.0

    def test_feedback_with_very_low_confidence(self):
        """Teste Feedback bei sehr niedriger ursprünglicher Confidence"""
        fm = ConfidenceFeedbackManager()

        fb = fm.submit_negative_feedback("IS_A", "test", "test", 0.01)

        # Sollte trotzdem nicht unter 0.0 gehen
        assert fb.adjusted_confidence >= 0.0

    def test_rapid_feedback_sequence(self):
        """Teste schnelle Feedback-Sequenz"""
        fm = ConfidenceFeedbackManager()

        # Sende 100 Feedbacks schnell hintereinander
        for i in range(100):
            fm.submit_positive_feedback("IS_A", "test", "test", 0.5)

        assert len(fm._feedback_history) == 100

        stats = fm.get_feedback_statistics()
        assert stats["total_feedback"] == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
