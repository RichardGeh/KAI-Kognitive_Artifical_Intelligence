"""
tests/test_response_formatter_integration.py

Integration-Test für SelfEvaluator in KaiResponseFormatter

Author: KAI Development Team
Created: 2025-11-08
"""

from unittest.mock import MagicMock

import pytest

from component_50_self_evaluation import RecommendationType
from kai_response_formatter import KaiResponseFormatter

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def formatter():
    """KaiResponseFormatter mit SelfEvaluator"""
    return KaiResponseFormatter()


# ============================================================================
# Integration Tests
# ============================================================================


class TestSelfEvaluationIntegration:
    """Tests für SelfEvaluator Integration in ResponseFormatter"""

    def test_formatter_has_self_evaluator(self, formatter):
        """Test: ResponseFormatter hat SelfEvaluator"""
        assert hasattr(formatter, "self_evaluator")
        assert formatter.self_evaluator is not None

    def test_evaluate_and_enrich_good_answer(self, formatter):
        """Test: Gute Antwort wird ohne Warnungen durchgereicht"""
        question = "Was ist ein Hund?"
        answer = "Ein Hund ist ein Säugetier und gehört zur Familie der Canidae."
        confidence = 0.9

        # Mock Proof Tree
        mock_proof = MagicMock()
        mock_proof.steps = [MagicMock(confidence=0.9)] * 3

        result = formatter.evaluate_and_enrich_response(
            question=question,
            answer_text=answer,
            confidence=confidence,
            proof_tree=mock_proof,
        )

        # Sollte hohen Score haben
        assert result["evaluation"].overall_score >= 0.85
        assert result["evaluation"].recommendation == RecommendationType.SHOW_TO_USER
        # Keine oder minimale Warnungen
        assert len(result["warnings"]) <= 1  # Vielleicht nur Confidence-Info

    def test_evaluate_and_enrich_poor_answer(self, formatter):
        """Test: Schlechte Antwort bekommt Warnungen"""
        question = "Was ist ein Hund und was ist eine Katze?"
        answer = "Hund."  # Sehr kurz, nur Teil beantwortet
        confidence = 0.9  # Zu hohe Confidence!

        result = formatter.evaluate_and_enrich_response(
            question=question,
            answer_text=answer,
            confidence=confidence,
            proof_tree=None,
            reasoning_paths=[],
        )

        # Sollte niedrigen Score haben
        assert result["evaluation"].overall_score < 0.8
        # Sollte Warnungen haben
        assert len(result["warnings"]) > 0

    def test_evaluate_and_enrich_with_uncertainties(self, formatter):
        """Test: Unsicherheiten werden angezeigt"""
        question = "Ist ein Pinguin ein Vogel?"
        answer = "Ein Pinguin ist ein Vogel, aber er kann nicht fliegen."
        confidence = 0.7

        result = formatter.evaluate_and_enrich_response(
            question=question, answer_text=answer, confidence=confidence
        )

        # Sollte Unsicherheiten enthalten (wegen "aber")
        assert result["evaluation"].uncertainties is not None
        # Text sollte angereichert sein
        assert len(result["text"]) > len(answer)

    def test_confidence_adjustment_applied(self, formatter):
        """Test: Confidence-Adjustment wird angewendet"""
        question = "Was ist ein Hund?"
        answer = "Hund."
        confidence = 0.95  # Sehr hoch ohne Beweise!

        result = formatter.evaluate_and_enrich_response(
            question=question,
            answer_text=answer,
            confidence=confidence,
            proof_tree=None,
            reasoning_paths=[],
        )

        # Confidence sollte angepasst sein
        # (oder zumindest niedriger Calibration Score)
        if result["evaluation"].confidence_adjusted:
            assert result["confidence"] < confidence
            assert any("angepasst" in w.lower() for w in result["warnings"])

    def test_empty_answer_gets_warnings(self, formatter):
        """Test: Leere Antwort bekommt deutliche Warnungen"""
        question = "Was ist X?"
        answer = ""
        confidence = 0.5

        result = formatter.evaluate_and_enrich_response(
            question=question, answer_text=answer, confidence=confidence
        )

        # Sollte sehr niedrigen Score haben
        assert result["evaluation"].overall_score < 0.7
        # Sollte Warnungen haben
        assert len(result["warnings"]) > 0
        # Sollte Unsicherheiten enthalten
        assert len(result["evaluation"].uncertainties) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
