"""
tests/test_feedback_integration.py

End-to-End Integration Test für User Feedback Loop (Phase 3.4)

Testet:
- KaiWorker → ResponseFormatter → FeedbackHandler Integration
- Answer Tracking bei evaluate_and_enrich_response()
- Feedback Processing mit Confidence Updates
- Meta-Learning Integration (wenn verfügbar)
- UI-Flow Simulation

Author: KAI Development Team
Created: 2025-11-08
"""

import pytest

from component_1_netzwerk import KonzeptNetzwerk
from component_51_feedback_handler import FeedbackHandler, FeedbackType
from kai_response_formatter import KaiResponseFormatter

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def netzwerk():
    """Test KonzeptNetzwerk (echte Verbindung)"""
    nw = KonzeptNetzwerk()
    yield nw
    nw.close()


@pytest.fixture
def feedback_handler(netzwerk):
    """FeedbackHandler mit echtem Netzwerk"""
    handler = FeedbackHandler(netzwerk=netzwerk, meta_learning=None)
    return handler


@pytest.fixture
def response_formatter(feedback_handler):
    """ResponseFormatter mit FeedbackHandler"""
    formatter = KaiResponseFormatter(feedback_handler=feedback_handler)
    return formatter


# ============================================================================
# Integration Tests
# ============================================================================


class TestFeedbackIntegrationFlow:
    """End-to-End Tests für Feedback-Integration"""

    def test_answer_tracking_in_response_formatter(self, response_formatter):
        """Test: ResponseFormatter trackt Antworten automatisch"""
        result = response_formatter.evaluate_and_enrich_response(
            question="Was ist ein Hund?",
            answer_text="Ein Hund ist ein Säugetier.",
            confidence=0.9,
            strategy="resonance",
            used_relations=["IS_A"],
            used_concepts=["hund", "säugetier"],
            track_for_feedback=True,
        )

        # Answer ID sollte vorhanden sein
        assert result["answer_id"] is not None
        assert len(result["answer_id"]) > 0

        # Strategy sollte gesetzt sein
        assert result["strategy"] == "resonance"

        # Antwort sollte im FeedbackHandler gespeichert sein
        answer_record = response_formatter.feedback_handler.get_answer(
            result["answer_id"]
        )
        assert answer_record is not None
        assert answer_record.query == "Was ist ein Hund?"
        assert answer_record.strategy == "resonance"

    def test_answer_not_tracked_when_disabled(self, response_formatter):
        """Test: Answer Tracking kann deaktiviert werden"""
        result = response_formatter.evaluate_and_enrich_response(
            question="Test?",
            answer_text="Test answer",
            confidence=0.8,
            track_for_feedback=False,  # Explizit deaktiviert
        )

        # Answer ID sollte None sein
        assert result["answer_id"] is None

    def test_complete_feedback_flow(self, response_formatter, feedback_handler):
        """Test: Kompletter Feedback-Flow von Response bis Processing"""
        # Schritt 1: Erstelle Response mit Tracking
        result = response_formatter.evaluate_and_enrich_response(
            question="Kann ein Pinguin fliegen?",
            answer_text="Nein, ein Pinguin kann nicht fliegen.",
            confidence=0.95,
            strategy="logic",
            used_relations=["CAPABLE_OF"],
        )

        answer_id = result["answer_id"]
        assert answer_id is not None

        # Schritt 2: User gibt positives Feedback
        feedback_result = feedback_handler.process_user_feedback(
            answer_id=answer_id, feedback_type=FeedbackType.CORRECT
        )

        assert feedback_result["success"] is True
        assert len(feedback_result["actions_taken"]) > 0

        # Schritt 3: Prüfe, dass Confidence erhöht wurde
        assert len(feedback_result["confidence_changes"]) > 0
        # Bei korrektem Feedback: factor=1.1
        for rel_id, new_conf in feedback_result["confidence_changes"].items():
            assert new_conf >= 1.0  # factor ist 1.1

        # Schritt 4: Prüfe Statistiken
        stats = feedback_handler.get_feedback_stats()
        assert stats["total_feedbacks"] >= 1
        assert stats["correct_count"] >= 1

    def test_incorrect_feedback_creates_inhibition(
        self, response_formatter, feedback_handler, netzwerk
    ):
        """Test: Inkorrektes Feedback erzeugt Inhibition Pattern"""
        # Schritt 1: Erstelle falsche Antwort
        result = response_formatter.evaluate_and_enrich_response(
            question="Ist ein Wal ein Fisch?",
            answer_text="Ja, ein Wal ist ein Fisch.",  # FALSCH!
            confidence=0.8,
            strategy="graph_traversal",
            used_relations=["IS_A"],
        )

        answer_id = result["answer_id"]

        # Schritt 2: User gibt negatives Feedback
        feedback_result = feedback_handler.process_user_feedback(
            answer_id=answer_id,
            feedback_type=FeedbackType.INCORRECT,
            correction="Nein, ein Wal ist ein Säugetier, kein Fisch.",
        )

        assert feedback_result["success"] is True

        # Schritt 3: Prüfe, dass Inhibition Pattern erwähnt wird
        assert any(
            "Inhibition" in action for action in feedback_result["actions_taken"]
        )

        # Schritt 4: Confidence sollte reduziert sein
        # factor=0.85
        for rel_id, new_conf in feedback_result["confidence_changes"].items():
            assert new_conf < 1.0  # factor ist 0.85

    def test_multiple_feedbacks_update_statistics(
        self, response_formatter, feedback_handler
    ):
        """Test: Mehrere Feedbacks aktualisieren Statistiken korrekt"""
        # Erstelle 3 Antworten
        answers = []
        for i in range(3):
            result = response_formatter.evaluate_and_enrich_response(
                question=f"Frage {i}?",
                answer_text=f"Antwort {i}",
                confidence=0.8 + i * 0.05,
                strategy="resonance",
            )
            answers.append(result["answer_id"])

        # Gib unterschiedliche Feedbacks
        feedback_handler.process_user_feedback(answers[0], FeedbackType.CORRECT)
        feedback_handler.process_user_feedback(answers[1], FeedbackType.CORRECT)
        feedback_handler.process_user_feedback(answers[2], FeedbackType.INCORRECT)

        # Prüfe Statistiken
        stats = feedback_handler.get_feedback_stats()
        assert stats["total_feedbacks"] >= 3
        assert stats["correct_count"] >= 2
        assert stats["incorrect_count"] >= 1
        assert stats["accuracy"] >= 0.6  # 2/3 = 0.666...

        # Prüfe Strategy-Breakdown
        breakdown = feedback_handler.get_strategy_feedback_breakdown()
        assert "resonance" in breakdown
        assert breakdown["resonance"]["correct"] >= 2
        assert breakdown["resonance"]["incorrect"] >= 1

    def test_feedback_without_meta_learning(self, response_formatter, feedback_handler):
        """Test: Feedback funktioniert ohne Meta-Learning"""
        # FeedbackHandler wurde ohne meta_learning erstellt (fixture)
        assert feedback_handler.meta_learning is None

        # Sollte trotzdem funktionieren
        result = response_formatter.evaluate_and_enrich_response(
            question="Test?",
            answer_text="Test answer",
            confidence=0.9,
            strategy="logic",
        )

        feedback_result = feedback_handler.process_user_feedback(
            answer_id=result["answer_id"], feedback_type=FeedbackType.CORRECT
        )

        assert feedback_result["success"] is True

    def test_unsure_feedback_slight_reduction(
        self, response_formatter, feedback_handler
    ):
        """Test: Unsicheres Feedback reduziert Confidence leicht"""
        result = response_formatter.evaluate_and_enrich_response(
            question="Ist ein Delfin ein Säugetier?",
            answer_text="Ja, ein Delfin ist ein Säugetier.",
            confidence=0.9,
            strategy="graph_traversal",
            used_relations=["IS_A"],
        )

        feedback_result = feedback_handler.process_user_feedback(
            answer_id=result["answer_id"],
            feedback_type=FeedbackType.UNSURE,
            user_comment="Bin mir nicht sicher",
        )

        assert feedback_result["success"] is True

        # Confidence sollte leicht reduziert sein (factor=0.98)
        for rel_id, new_conf in feedback_result["confidence_changes"].items():
            assert new_conf < 1.0  # factor ist 0.98
            assert new_conf >= 0.95  # Nicht zu stark reduziert

    def test_partially_correct_feedback_small_boost(
        self, response_formatter, feedback_handler
    ):
        """Test: Teilweise korrektes Feedback gibt kleinen Boost"""
        result = response_formatter.evaluate_and_enrich_response(
            question="Was sind die Eigenschaften von Wasser?",
            answer_text="Wasser ist flüssig.",  # Nur teilweise richtig
            confidence=0.7,
            strategy="logic",
            used_relations=["HAS_PROPERTY"],
        )

        feedback_result = feedback_handler.process_user_feedback(
            answer_id=result["answer_id"],
            feedback_type=FeedbackType.PARTIALLY_CORRECT,
        )

        assert feedback_result["success"] is True

        # Confidence sollte leicht erhöht sein (factor=1.02)
        for rel_id, new_conf in feedback_result["confidence_changes"].items():
            assert new_conf > 1.0  # factor ist 1.02
            assert new_conf <= 1.05  # Nicht zu stark erhöht


# ============================================================================
# Performance Tests
# ============================================================================


class TestFeedbackPerformance:
    """Tests für Performance der Feedback-Integration"""

    def test_track_large_batch_of_answers(self, response_formatter, feedback_handler):
        """Test: Viele Antworten schnell tracken"""
        answer_ids = []

        # Tracke 50 Antworten
        for i in range(50):
            result = response_formatter.evaluate_and_enrich_response(
                question=f"Question {i}?",
                answer_text=f"Answer {i}",
                confidence=0.8,
                strategy="resonance",
            )
            answer_ids.append(result["answer_id"])

        # Alle sollten tracked sein
        assert len(answer_ids) == 50
        assert all(aid is not None for aid in answer_ids)

        # Statistiken prüfen
        stats = feedback_handler.get_feedback_stats()
        assert stats["tracked_answers"] >= 50

    def test_feedback_history_retrieval(self, response_formatter, feedback_handler):
        """Test: Feedback-History kann abgerufen werden"""
        # Erstelle einige Antworten mit Feedback
        for i in range(5):
            result = response_formatter.evaluate_and_enrich_response(
                question=f"Q{i}?",
                answer_text=f"A{i}",
                confidence=0.8,
                strategy="logic",
            )
            feedback_handler.process_user_feedback(
                result["answer_id"], FeedbackType.CORRECT
            )

        # Hole History
        history = feedback_handler.get_feedback_history(limit=10)

        assert len(history) >= 5
        # Neueste zuerst
        assert history[0]["feedback_type"] == "correct"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
