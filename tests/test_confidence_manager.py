# tests/test_confidence_manager.py
"""
Umfassende Tests für das Confidence-Based Learning System

Tests für:
- ConfidenceManager (Klassifizierung, Kombination, Decay)
- ConfidenceThresholds (Auto-Accept, Ask-User, Reject)
- ConfidenceMetrics (Datenhaltung, Erklärungen)
- Integration mit Response Formatter
"""
import pytest
from datetime import datetime, timedelta

from component_confidence_manager import (
    ConfidenceManager,
    ConfidenceLevel,
    ConfidenceMetrics,
    CombinationStrategy,
    get_confidence_manager,
)


class TestConfidenceClassification:
    """Tests für Confidence-Klassifizierung"""

    def test_classify_high_confidence(self):
        """Teste HIGH-Klassifizierung für confidence >= 0.8"""
        cm = ConfidenceManager()

        assert cm.classify_confidence(0.8) == ConfidenceLevel.HIGH
        assert cm.classify_confidence(0.9) == ConfidenceLevel.HIGH
        assert cm.classify_confidence(1.0) == ConfidenceLevel.HIGH

    def test_classify_medium_confidence(self):
        """Teste MEDIUM-Klassifizierung für 0.5 <= confidence < 0.8"""
        cm = ConfidenceManager()

        assert cm.classify_confidence(0.5) == ConfidenceLevel.MEDIUM
        assert cm.classify_confidence(0.6) == ConfidenceLevel.MEDIUM
        assert cm.classify_confidence(0.79) == ConfidenceLevel.MEDIUM

    def test_classify_low_confidence(self):
        """Teste LOW-Klassifizierung für 0.3 <= confidence < 0.5"""
        cm = ConfidenceManager()

        assert cm.classify_confidence(0.3) == ConfidenceLevel.LOW
        assert cm.classify_confidence(0.4) == ConfidenceLevel.LOW
        assert cm.classify_confidence(0.49) == ConfidenceLevel.LOW

    def test_classify_unknown_confidence(self):
        """Teste UNKNOWN-Klassifizierung für confidence < 0.3"""
        cm = ConfidenceManager()

        assert cm.classify_confidence(0.0) == ConfidenceLevel.UNKNOWN
        assert cm.classify_confidence(0.1) == ConfidenceLevel.UNKNOWN
        assert cm.classify_confidence(0.29) == ConfidenceLevel.UNKNOWN


class TestThresholdDecisions:
    """Tests für Threshold-basierte Entscheidungen"""

    def test_should_auto_accept_standard(self):
        """Teste Auto-Accept für Standard-Thresholds (>= 0.8)"""
        cm = ConfidenceManager()

        assert cm.should_auto_accept(0.8, auto_detected=False) is True
        assert cm.should_auto_accept(0.9, auto_detected=False) is True
        assert cm.should_auto_accept(0.79, auto_detected=False) is False

    def test_should_auto_accept_auto_detected(self):
        """Teste Auto-Accept für auto-detected mit höherem Threshold (>= 0.85)"""
        cm = ConfidenceManager()

        assert cm.should_auto_accept(0.85, auto_detected=True) is True
        assert cm.should_auto_accept(0.9, auto_detected=True) is True
        assert cm.should_auto_accept(0.84, auto_detected=True) is False

    def test_should_ask_user(self):
        """Teste Ask-User für 0.5 <= confidence < 0.8"""
        cm = ConfidenceManager()

        assert cm.should_ask_user(0.5) is True
        assert cm.should_ask_user(0.7) is True
        assert cm.should_ask_user(0.79) is True
        assert cm.should_ask_user(0.8) is False
        assert cm.should_ask_user(0.49) is False

    def test_should_reject(self):
        """Teste Reject für confidence < 0.3"""
        cm = ConfidenceManager()

        assert cm.should_reject(0.0) is True
        assert cm.should_reject(0.2) is True
        assert cm.should_reject(0.29) is True
        assert cm.should_reject(0.3) is False


class TestConfidenceCombination:
    """Tests für Kombinierung mehrerer Confidence-Werte"""

    def test_combine_minimum(self):
        """Teste MINIMUM-Strategie (weakest link)"""
        cm = ConfidenceManager()

        confidences = [0.9, 0.8, 0.85]
        metrics = cm.combine_confidences(
            confidences, strategy=CombinationStrategy.MINIMUM
        )

        assert metrics.value == 0.8  # Minimum
        assert metrics.level == ConfidenceLevel.HIGH
        assert metrics.combination_strategy == CombinationStrategy.MINIMUM

    def test_combine_maximum(self):
        """Teste MAXIMUM-Strategie (strongest link)"""
        cm = ConfidenceManager()

        confidences = [0.9, 0.8, 0.85]
        metrics = cm.combine_confidences(
            confidences, strategy=CombinationStrategy.MAXIMUM
        )

        assert metrics.value == 0.9  # Maximum
        assert metrics.level == ConfidenceLevel.HIGH

    def test_combine_average(self):
        """Teste AVERAGE-Strategie"""
        cm = ConfidenceManager()

        confidences = [0.6, 0.8, 1.0]
        metrics = cm.combine_confidences(
            confidences, strategy=CombinationStrategy.AVERAGE
        )

        # (0.6 + 0.8 + 1.0) / 3 = 0.8 (theoretisch), aber Floating-Point kann 0.7999... sein
        assert metrics.value == pytest.approx(0.8, abs=0.01)
        # Prüfe Level flexibel wegen Rundung: 0.7999... -> MEDIUM, 0.8 -> HIGH
        assert metrics.level in [ConfidenceLevel.MEDIUM, ConfidenceLevel.HIGH]

    def test_combine_weighted_average(self):
        """Teste WEIGHTED_AVERAGE-Strategie"""
        cm = ConfidenceManager()

        confidences = [0.6, 0.8, 1.0]
        weights = [0.2, 0.3, 0.5]  # Summe = 1.0
        metrics = cm.combine_confidences(
            confidences, strategy=CombinationStrategy.WEIGHTED_AVERAGE, weights=weights
        )

        # (0.6*0.2 + 0.8*0.3 + 1.0*0.5) = 0.12 + 0.24 + 0.5 = 0.86
        assert metrics.value == pytest.approx(0.86, abs=0.01)
        assert metrics.level == ConfidenceLevel.HIGH

    def test_combine_bayesian(self):
        """Teste BAYESIAN (Noisy-OR) Strategie"""
        cm = ConfidenceManager()

        confidences = [0.6, 0.7]
        metrics = cm.combine_confidences(
            confidences, strategy=CombinationStrategy.BAYESIAN
        )

        # Noisy-OR: 1 - (1-0.6)*(1-0.7) = 1 - 0.4*0.3 = 1 - 0.12 = 0.88
        assert metrics.value == pytest.approx(0.88, abs=0.01)
        assert metrics.level == ConfidenceLevel.HIGH

    def test_combine_empty_list(self):
        """Teste Behandlung leerer Confidence-Liste"""
        cm = ConfidenceManager()

        metrics = cm.combine_confidences([], strategy=CombinationStrategy.MINIMUM)

        assert metrics.value == 0.0
        assert metrics.level == ConfidenceLevel.UNKNOWN


class TestConfidenceDecay:
    """Tests für zeitbasierte Confidence-Reduktion"""

    def test_no_decay_for_recent_facts(self):
        """Teste dass Fakten < 30 Tage keinen Decay erhalten"""
        cm = ConfidenceManager()

        recent_timestamp = datetime.now() - timedelta(days=10)
        metrics = cm.apply_decay(0.9, recent_timestamp)

        assert metrics.value == 0.9
        assert metrics.decay_applied is False
        assert "zu neu" in metrics.explanation.lower()

    def test_decay_for_old_facts(self):
        """Teste Decay für Fakten > 30 Tage"""
        cm = ConfidenceManager()

        # 365 Tage alt (2 Half-Lives = ~180 Tage)
        old_timestamp = datetime.now() - timedelta(days=365)
        metrics = cm.apply_decay(0.9, old_timestamp)

        # Nach 2 Half-Lives: 0.9 * 0.5^2 = 0.9 * 0.25 = 0.225
        # Aber Minimum ist 0.3
        assert metrics.value >= 0.3  # Clipping auf Minimum
        assert metrics.decay_applied is True
        assert metrics.original_value == 0.9

    def test_decay_respects_minimum(self):
        """Teste dass Decay nie unter min_confidence fällt"""
        cm = ConfidenceManager()

        # 730 Tage alt (4 Half-Lives)
        very_old_timestamp = datetime.now() - timedelta(days=730)
        metrics = cm.apply_decay(1.0, very_old_timestamp)

        # Nach 4 Half-Lives: 1.0 * 0.5^4 = 0.0625
        # Aber Minimum ist 0.3
        assert metrics.value == 0.3
        assert metrics.decay_applied is True

    def test_decay_half_life(self):
        """Teste dass Confidence nach Half-Life auf 50% reduziert ist"""
        cm = ConfidenceManager()

        # Exakt 180 Tage alt (1 Half-Life)
        half_life_timestamp = datetime.now() - timedelta(days=180)
        metrics = cm.apply_decay(0.8, half_life_timestamp)

        # Nach 1 Half-Life: 0.8 * 0.5 = 0.4
        assert metrics.value == pytest.approx(0.4, abs=0.05)
        assert metrics.decay_applied is True


class TestSpecializedMethods:
    """Tests für spezialisierte Confidence-Berechnungen"""

    def test_graph_traversal_confidence(self):
        """Teste Confidence-Berechnung für Graph-Traversal (MINIMUM)"""
        cm = ConfidenceManager()

        edge_confidences = [0.9, 0.85, 0.95]
        metrics = cm.calculate_graph_traversal_confidence(edge_confidences)

        assert metrics.value == 0.85  # Minimum (weakest link)
        assert metrics.combination_strategy == CombinationStrategy.MINIMUM

    def test_rule_confidence(self):
        """Teste Confidence-Berechnung für Regel-basierte Inferenz"""
        cm = ConfidenceManager()

        premise_confidences = [0.8, 0.9]
        rule_strength = 0.95
        metrics = cm.calculate_rule_confidence(premise_confidences, rule_strength)

        # min(0.8, 0.9) * 0.95 = 0.8 * 0.95 = 0.76
        assert metrics.value == pytest.approx(0.76, abs=0.01)

    def test_hypothesis_confidence(self):
        """Teste Confidence-Berechnung für Hypothesen (Abductive)"""
        cm = ConfidenceManager()

        # Standard-Gewichte: coverage=0.3, simplicity=0.2, coherence=0.3, specificity=0.2
        metrics = cm.calculate_hypothesis_confidence(
            coverage=0.8, simplicity=0.9, coherence=0.7, specificity=0.85
        )

        # Weighted average: 0.8*0.3 + 0.9*0.2 + 0.7*0.3 + 0.85*0.2
        # = 0.24 + 0.18 + 0.21 + 0.17 = 0.8
        assert metrics.value == pytest.approx(0.8, abs=0.01)
        assert metrics.combination_strategy == CombinationStrategy.WEIGHTED_AVERAGE


class TestConfidenceMetrics:
    """Tests für ConfidenceMetrics Datenstruktur"""

    def test_metrics_creation(self):
        """Teste erfolgreiche Erstellung von ConfidenceMetrics"""
        metrics = ConfidenceMetrics(
            value=0.75,
            level=ConfidenceLevel.MEDIUM,
            source_confidences=[0.7, 0.8],
            combination_strategy=CombinationStrategy.AVERAGE,
            explanation="Test explanation",
        )

        assert metrics.value == 0.75
        assert metrics.level == ConfidenceLevel.MEDIUM
        assert len(metrics.source_confidences) == 2
        assert metrics.decay_applied is False

    def test_metrics_validation(self):
        """Teste Validierung von ungültigen Confidence-Werten"""
        with pytest.raises(ValueError):
            ConfidenceMetrics(value=1.5, level=ConfidenceLevel.HIGH)

        with pytest.raises(ValueError):
            ConfidenceMetrics(value=-0.1, level=ConfidenceLevel.UNKNOWN)


class TestUIFeedback:
    """Tests für UI-Feedback-Generierung"""

    def test_generate_ui_feedback_high(self):
        """Teste UI-Feedback für hohe Confidence"""
        cm = ConfidenceManager()

        feedback = cm.generate_ui_feedback(0.9, context="Antwort")

        assert "Antwort" in feedback
        assert "0.90" in feedback
        assert "Hoch" in feedback

    def test_generate_ui_feedback_medium(self):
        """Teste UI-Feedback für mittlere Confidence"""
        cm = ConfidenceManager()

        feedback = cm.generate_ui_feedback(0.6, context="Fakt")

        assert "Fakt" in feedback
        assert "0.60" in feedback
        assert "Mittel" in feedback
        assert "bestätigen" in feedback.lower()

    def test_generate_ui_feedback_low(self):
        """Teste UI-Feedback für niedrige Confidence"""
        cm = ConfidenceManager()

        feedback = cm.generate_ui_feedback(0.4, context="Hypothese")

        assert "Hypothese" in feedback
        assert "0.40" in feedback
        assert "Niedrig" in feedback

    def test_explain_confidence_verbose(self):
        """Teste ausführliche Confidence-Erklärung"""
        cm = ConfidenceManager()

        confidences = [0.7, 0.8, 0.9]
        metrics = cm.combine_confidences(
            confidences, strategy=CombinationStrategy.AVERAGE
        )

        explanation = cm.explain_confidence(metrics, verbose=True)

        assert "Confidence:" in explanation
        assert "MEDIUM" in explanation or "HIGH" in explanation
        assert "Quell-Confidences" in explanation


class TestGlobalInstance:
    """Tests für globale ConfidenceManager-Instanz"""

    def test_get_confidence_manager_singleton(self):
        """Teste dass get_confidence_manager() immer dieselbe Instanz zurückgibt"""
        cm1 = get_confidence_manager()
        cm2 = get_confidence_manager()

        assert cm1 is cm2  # Gleiche Objekt-Identität

    def test_confidence_manager_initialization(self):
        """Teste korrekte Initialisierung des ConfidenceManagers"""
        cm = get_confidence_manager()

        assert cm.thresholds.AUTO_ACCEPT == 0.8
        assert cm.thresholds.ASK_USER == 0.5
        assert cm.thresholds.REJECT == 0.3


class TestEdgeCases:
    """Tests für Edge-Cases und Fehlerbehandlung"""

    def test_combine_single_confidence(self):
        """Teste Kombinierung einer einzelnen Confidence"""
        cm = ConfidenceManager()

        metrics = cm.combine_confidences([0.7])

        assert metrics.value == 0.7

    def test_weighted_average_invalid_weights(self):
        """Teste Fehlerbehandlung bei ungültigen Gewichten"""
        cm = ConfidenceManager()

        with pytest.raises(ValueError):
            cm.combine_confidences(
                [0.7, 0.8],
                strategy=CombinationStrategy.WEIGHTED_AVERAGE,
                weights=[0.5],  # Falsche Anzahl
            )

    def test_rule_confidence_invalid_strength(self):
        """Teste Fehlerbehandlung bei ungültiger Regel-Stärke"""
        cm = ConfidenceManager()

        with pytest.raises(ValueError):
            cm.calculate_rule_confidence([0.8], rule_strength=1.5)  # > 1.0

    def test_get_threshold_invalid_action(self):
        """Teste Fehlerbehandlung bei ungültigem Action-Typ"""
        cm = ConfidenceManager()

        with pytest.raises(ValueError):
            cm.get_threshold_for_action("invalid_action")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
