"""
tests/test_feedback_handler.py

Tests für FeedbackHandler (Phase 3.4: User Feedback Loop)

Tests:
- Answer Tracking
- Feedback Processing (correct/incorrect/unsure/partially_correct)
- Confidence Updates
- Meta-Learning Integration
- Statistics und History
- Edge Cases

Author: KAI Development Team
Created: 2025-11-08
"""

from unittest.mock import MagicMock

import pytest

from component_46_meta_learning import MetaLearningEngine
from component_51_feedback_handler import (
    FeedbackHandler,
    FeedbackType,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def netzwerk_mock():
    """Mock KonzeptNetzwerk"""
    mock = MagicMock()
    return mock


@pytest.fixture
def meta_learning_mock():
    """Mock MetaLearningEngine"""
    mock = MagicMock(spec=MetaLearningEngine)
    mock.record_strategy_usage_with_feedback = MagicMock()
    return mock


@pytest.fixture
def feedback_handler(netzwerk_mock, meta_learning_mock):
    """FeedbackHandler mit Mocks"""
    handler = FeedbackHandler(netzwerk=netzwerk_mock, meta_learning=meta_learning_mock)
    return handler


# ============================================================================
# Initialization Tests
# ============================================================================


class TestInitialization:
    """Tests für FeedbackHandler Initialization"""

    def test_handler_initialization(self, netzwerk_mock, meta_learning_mock):
        """Test: Handler wird korrekt initialisiert"""
        handler = FeedbackHandler(netzwerk_mock, meta_learning_mock)

        assert handler is not None
        assert handler.netzwerk == netzwerk_mock
        assert handler.meta_learning == meta_learning_mock
        assert handler.answer_records == {}
        assert handler.feedback_records == {}
        assert handler.feedback_stats["total_feedbacks"] == 0

    def test_initialization_without_meta_learning(self, netzwerk_mock):
        """Test: Handler funktioniert ohne MetaLearning"""
        handler = FeedbackHandler(netzwerk_mock, meta_learning=None)

        assert handler.meta_learning is None
        assert handler.confidence_manager is not None


# ============================================================================
# Answer Tracking Tests
# ============================================================================


class TestAnswerTracking:
    """Tests für Answer Tracking"""

    def test_track_answer_creates_record(self, feedback_handler):
        """Test: track_answer() erstellt AnswerRecord"""
        answer_id = feedback_handler.track_answer(
            query="Was ist ein Hund?",
            answer_text="Ein Hund ist ein Säugetier.",
            confidence=0.9,
            strategy="resonance",
        )

        assert answer_id is not None
        assert answer_id in feedback_handler.answer_records

        record = feedback_handler.get_answer(answer_id)
        assert record.query == "Was ist ein Hund?"
        assert record.answer_text == "Ein Hund ist ein Säugetier."
        assert record.confidence == 0.9
        assert record.strategy == "resonance"

    def test_track_answer_with_all_metadata(self, feedback_handler):
        """Test: track_answer() mit allen Metadaten"""
        answer_id = feedback_handler.track_answer(
            query="Test?",
            answer_text="Answer",
            confidence=0.8,
            strategy="logic",
            used_relations=["rel1", "rel2"],
            used_concepts=["concept1"],
            proof_tree=MagicMock(),
            reasoning_paths=[MagicMock()],
            evaluation_score=0.85,
            metadata={"custom": "data"},
        )

        record = feedback_handler.get_answer(answer_id)
        assert len(record.used_relations) == 2
        assert len(record.used_concepts) == 1
        assert record.proof_tree is not None
        assert record.evaluation_score == 0.85
        assert record.metadata["custom"] == "data"

    def test_get_answer_nonexistent_returns_none(self, feedback_handler):
        """Test: get_answer() für nicht-existente ID gibt None"""
        result = feedback_handler.get_answer("nonexistent-id")
        assert result is None

    def test_multiple_answers_tracked(self, feedback_handler):
        """Test: Mehrere Antworten können getrackt werden"""
        id1 = feedback_handler.track_answer("Q1?", "A1", 0.9, "s1")
        id2 = feedback_handler.track_answer("Q2?", "A2", 0.8, "s2")
        id3 = feedback_handler.track_answer("Q3?", "A3", 0.7, "s3")

        assert id1 != id2 != id3
        assert len(feedback_handler.answer_records) == 3


# ============================================================================
# Feedback Processing Tests
# ============================================================================


class TestFeedbackProcessing:
    """Tests für process_user_feedback()"""

    def test_process_correct_feedback(self, feedback_handler):
        """Test: Verarbeitung von korrektem Feedback"""
        # Track answer first
        answer_id = feedback_handler.track_answer(
            query="Test?",
            answer_text="Answer",
            confidence=0.8,
            strategy="resonance",
            used_relations=["rel1", "rel2"],
        )

        # Process feedback
        result = feedback_handler.process_user_feedback(
            answer_id=answer_id, feedback_type=FeedbackType.CORRECT
        )

        assert result["success"] is True
        assert len(result["actions_taken"]) > 0
        assert "Confidence" in result["actions_taken"][0]

        # Check statistics updated
        assert feedback_handler.feedback_stats["total_feedbacks"] == 1
        assert feedback_handler.feedback_stats["correct_count"] == 1

    def test_process_incorrect_feedback(self, feedback_handler):
        """Test: Verarbeitung von inkorrektem Feedback"""
        answer_id = feedback_handler.track_answer(
            "Test?", "Wrong answer", 0.9, "logic", used_relations=["rel1"]
        )

        result = feedback_handler.process_user_feedback(
            answer_id=answer_id,
            feedback_type=FeedbackType.INCORRECT,
            correction="Correct answer is X",
        )

        assert result["success"] is True
        # Inhibition pattern sollte erstellt werden
        assert any("Inhibition" in action for action in result["actions_taken"])

        # Statistics
        assert feedback_handler.feedback_stats["incorrect_count"] == 1

    def test_process_unsure_feedback(self, feedback_handler):
        """Test: Verarbeitung von unsicherem Feedback"""
        answer_id = feedback_handler.track_answer(
            "Test?", "Maybe?", 0.5, "probabilistic"
        )

        result = feedback_handler.process_user_feedback(
            answer_id=answer_id,
            feedback_type=FeedbackType.UNSURE,
            user_comment="Not sure about this",
        )

        assert result["success"] is True
        assert feedback_handler.feedback_stats["unsure_count"] == 1

    def test_process_partially_correct_feedback(self, feedback_handler):
        """Test: Verarbeitung von teilweise korrektem Feedback"""
        answer_id = feedback_handler.track_answer(
            "Test?", "Partial answer", 0.7, "graph_traversal"
        )

        result = feedback_handler.process_user_feedback(
            answer_id=answer_id, feedback_type=FeedbackType.PARTIALLY_CORRECT
        )

        assert result["success"] is True
        assert feedback_handler.feedback_stats["partially_correct_count"] == 1

    def test_process_feedback_for_nonexistent_answer(self, feedback_handler):
        """Test: Feedback für nicht-existente Answer-ID"""
        result = feedback_handler.process_user_feedback(
            answer_id="nonexistent", feedback_type=FeedbackType.CORRECT
        )

        assert result["success"] is False
        assert "nicht gefunden" in result["message"]

    def test_feedback_with_user_comment(self, feedback_handler):
        """Test: Feedback mit Benutzer-Kommentar"""
        answer_id = feedback_handler.track_answer("Test?", "Answer", 0.8, "resonance")

        result = feedback_handler.process_user_feedback(
            answer_id=answer_id,
            feedback_type=FeedbackType.CORRECT,
            user_comment="Great answer!",
        )

        assert result["success"] is True

        # Check feedback record
        feedback_id = list(feedback_handler.feedback_records.keys())[0]
        feedback = feedback_handler.feedback_records[feedback_id]
        assert feedback.user_comment == "Great answer!"


# ============================================================================
# Meta-Learning Integration Tests
# ============================================================================


class TestMetaLearningIntegration:
    """Tests für Meta-Learning Integration"""

    def test_meta_learning_called_on_feedback(
        self, feedback_handler, meta_learning_mock
    ):
        """Test: Meta-Learning wird bei Feedback informiert"""
        answer_id = feedback_handler.track_answer("Test?", "Answer", 0.8, "resonance")

        feedback_handler.process_user_feedback(
            answer_id=answer_id, feedback_type=FeedbackType.CORRECT
        )

        # Verify meta_learning wurde aufgerufen
        assert meta_learning_mock.record_strategy_usage_with_feedback.called

    def test_correct_feedback_recorded_as_success(
        self, feedback_handler, meta_learning_mock
    ):
        """Test: Korrektes Feedback wird als success=True recorded"""
        answer_id = feedback_handler.track_answer("Test?", "Answer", 0.8, "logic")

        feedback_handler.process_user_feedback(
            answer_id=answer_id, feedback_type=FeedbackType.CORRECT
        )

        # Check call arguments
        call_args = meta_learning_mock.record_strategy_usage_with_feedback.call_args
        assert call_args[1]["success"] is True

    def test_incorrect_feedback_recorded_as_failure(
        self, feedback_handler, meta_learning_mock
    ):
        """Test: Inkorrektes Feedback wird als success=False recorded"""
        answer_id = feedback_handler.track_answer("Test?", "Wrong", 0.9, "abductive")

        feedback_handler.process_user_feedback(
            answer_id=answer_id, feedback_type=FeedbackType.INCORRECT
        )

        call_args = meta_learning_mock.record_strategy_usage_with_feedback.call_args
        assert call_args[1]["success"] is False

    def test_no_crash_without_meta_learning(self, netzwerk_mock):
        """Test: Kein Crash wenn Meta-Learning fehlt"""
        handler = FeedbackHandler(netzwerk_mock, meta_learning=None)

        answer_id = handler.track_answer("Test?", "Answer", 0.8, "resonance")
        result = handler.process_user_feedback(answer_id, FeedbackType.CORRECT)

        # Sollte funktionieren, aber Meta-Learning Action fehlt
        assert result["success"] is True


# ============================================================================
# Statistics Tests
# ============================================================================


class TestStatistics:
    """Tests für Feedback-Statistiken"""

    def test_get_feedback_stats_initial(self, feedback_handler):
        """Test: Initiale Statistiken sind leer"""
        stats = feedback_handler.get_feedback_stats()

        assert stats["total_feedbacks"] == 0
        assert stats["correct_count"] == 0
        assert stats["incorrect_count"] == 0
        assert stats["accuracy"] == 0.0
        assert stats["tracked_answers"] == 0

    def test_get_feedback_stats_after_feedbacks(self, feedback_handler):
        """Test: Statistiken nach mehreren Feedbacks"""
        # Track 3 answers
        id1 = feedback_handler.track_answer("Q1?", "A1", 0.9, "s1")
        id2 = feedback_handler.track_answer("Q2?", "A2", 0.8, "s2")
        id3 = feedback_handler.track_answer("Q3?", "A3", 0.7, "s3")

        # Give feedback
        feedback_handler.process_user_feedback(id1, FeedbackType.CORRECT)
        feedback_handler.process_user_feedback(id2, FeedbackType.CORRECT)
        feedback_handler.process_user_feedback(id3, FeedbackType.INCORRECT)

        stats = feedback_handler.get_feedback_stats()

        assert stats["total_feedbacks"] == 3
        assert stats["correct_count"] == 2
        assert stats["incorrect_count"] == 1
        assert stats["accuracy"] == 2.0 / 3.0
        assert stats["tracked_answers"] == 3

    def test_accuracy_with_partially_correct(self, feedback_handler):
        """Test: Accuracy-Berechnung mit teilweise korrekten Antworten"""
        id1 = feedback_handler.track_answer("Q1?", "A1", 0.9, "s1")
        id2 = feedback_handler.track_answer("Q2?", "A2", 0.8, "s2")

        feedback_handler.process_user_feedback(id1, FeedbackType.CORRECT)
        feedback_handler.process_user_feedback(id2, FeedbackType.PARTIALLY_CORRECT)

        stats = feedback_handler.get_feedback_stats()

        # 1 correct + 0.5 * 1 partially = 1.5 / 2 = 0.75
        assert stats["accuracy"] == 0.75

    def test_get_strategy_feedback_breakdown(self, feedback_handler):
        """Test: Feedback-Breakdown pro Strategy"""
        id1 = feedback_handler.track_answer("Q1?", "A1", 0.9, "resonance")
        id2 = feedback_handler.track_answer("Q2?", "A2", 0.8, "resonance")
        id3 = feedback_handler.track_answer("Q3?", "A3", 0.7, "logic")

        feedback_handler.process_user_feedback(id1, FeedbackType.CORRECT)
        feedback_handler.process_user_feedback(id2, FeedbackType.INCORRECT)
        feedback_handler.process_user_feedback(id3, FeedbackType.CORRECT)

        breakdown = feedback_handler.get_strategy_feedback_breakdown()

        assert "resonance" in breakdown
        assert breakdown["resonance"]["correct"] == 1
        assert breakdown["resonance"]["incorrect"] == 1

        assert "logic" in breakdown
        assert breakdown["logic"]["correct"] == 1


# ============================================================================
# Feedback History Tests
# ============================================================================


class TestFeedbackHistory:
    """Tests für Feedback History"""

    def test_get_feedback_history_empty(self, feedback_handler):
        """Test: Leere History initial"""
        history = feedback_handler.get_feedback_history()
        assert history == []

    def test_get_feedback_history(self, feedback_handler):
        """Test: Feedback History wird gespeichert"""
        id1 = feedback_handler.track_answer("Q1?", "A1", 0.9, "s1")
        id2 = feedback_handler.track_answer("Q2?", "A2", 0.8, "s2")

        feedback_handler.process_user_feedback(id1, FeedbackType.CORRECT)
        feedback_handler.process_user_feedback(id2, FeedbackType.INCORRECT)

        history = feedback_handler.get_feedback_history(limit=10)

        assert len(history) == 2
        # Neueste zuerst
        assert history[0]["feedback_type"] == "incorrect"
        assert history[1]["feedback_type"] == "correct"

    def test_get_feedback_history_with_limit(self, feedback_handler):
        """Test: History mit Limit"""
        for i in range(5):
            aid = feedback_handler.track_answer(f"Q{i}?", f"A{i}", 0.8, "s1")
            feedback_handler.process_user_feedback(aid, FeedbackType.CORRECT)

        history = feedback_handler.get_feedback_history(limit=3)
        assert len(history) == 3


# ============================================================================
# Edge Cases Tests
# ============================================================================


class TestEdgeCases:
    """Tests für Edge Cases"""

    def test_multiple_feedbacks_same_answer(self, feedback_handler):
        """Test: Mehrere Feedbacks für gleiche Antwort"""
        answer_id = feedback_handler.track_answer("Test?", "Answer", 0.8, "s1")

        # Erstes Feedback
        result1 = feedback_handler.process_user_feedback(answer_id, FeedbackType.UNSURE)
        assert result1["success"] is True

        # Zweites Feedback (Update)
        result2 = feedback_handler.process_user_feedback(
            answer_id, FeedbackType.CORRECT
        )
        assert result2["success"] is True

        # Stats sollten beide Feedbacks zählen
        assert feedback_handler.feedback_stats["total_feedbacks"] == 2

    def test_feedback_without_used_relations(self, feedback_handler):
        """Test: Feedback ohne verwendete Relationen"""
        answer_id = feedback_handler.track_answer(
            "Test?", "Answer", 0.8, "s1", used_relations=None  # Keine Relationen
        )

        result = feedback_handler.process_user_feedback(answer_id, FeedbackType.CORRECT)

        # Sollte funktionieren, aber keine Confidence-Changes
        assert result["success"] is True
        assert len(result["confidence_changes"]) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
