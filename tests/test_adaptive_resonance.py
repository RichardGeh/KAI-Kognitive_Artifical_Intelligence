"""
tests/test_adaptive_resonance.py

Tests für AdaptiveResonanceEngine (Phase 3.2: Adaptive Hyperparameter Tuning)

Tests:
- Auto-tuning basierend auf Graph-Größe
- Performance-basiertes Tuning
- Accuracy-basiertes Tuning
- Integration mit MetaLearningEngine
- Parameter boundaries

Author: KAI Development Team
Last Updated: 2025-11-08
"""

from unittest.mock import MagicMock, patch

import pytest

from component_44_resonance_engine import AdaptiveResonanceEngine, ResonanceEngine
from component_46_meta_learning import StrategyPerformance

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def netzwerk_mock():
    """Mock KonzeptNetzwerk mit konfigurierbarer Node-Count"""
    mock = MagicMock()  # Kein spec, damit get_node_count() funktioniert
    mock.get_node_count.return_value = 5000  # Default: mittlerer Graph
    return mock


@pytest.fixture
def meta_learning_mock():
    """Mock MetaLearningEngine mit konfigurierbaren Stats"""
    mock = MagicMock()  # Kein spec für Flexibilität

    # Default stats for 'resonance' strategy
    default_stats = StrategyPerformance(
        strategy_name="resonance",
        queries_handled=10,
        success_count=7,
        failure_count=3,
        success_rate=0.7,
        avg_confidence=0.75,
        avg_response_time=1.5,
    )
    mock.get_strategy_stats.return_value = default_stats

    return mock


@pytest.fixture
def adaptive_engine(netzwerk_mock, meta_learning_mock):
    """AdaptiveResonanceEngine mit Mocks"""
    engine = AdaptiveResonanceEngine(
        netzwerk=netzwerk_mock, meta_learning=meta_learning_mock
    )
    return engine


# ============================================================================
# Initialization Tests
# ============================================================================


class TestInitialization:
    """Tests für AdaptiveResonanceEngine Initialization"""

    def test_engine_initialization(self, netzwerk_mock, meta_learning_mock):
        """Test: Engine wird korrekt initialisiert"""
        engine = AdaptiveResonanceEngine(
            netzwerk=netzwerk_mock, meta_learning=meta_learning_mock
        )

        assert engine is not None
        assert engine.meta_learning == meta_learning_mock
        assert engine.tuning_history == []
        assert "_initial_hyperparameters" in dir(engine)

    def test_inherits_from_resonance_engine(self, adaptive_engine):
        """Test: AdaptiveResonanceEngine erbt von ResonanceEngine"""
        assert isinstance(adaptive_engine, ResonanceEngine)

    def test_initial_hyperparameters_stored(self, adaptive_engine):
        """Test: Initiale Hyperparameter werden gespeichert"""
        initial = adaptive_engine._initial_hyperparameters

        assert "activation_threshold" in initial
        assert "decay_factor" in initial
        assert "resonance_boost" in initial
        assert "max_waves" in initial
        assert "max_concepts_per_wave" in initial


# ============================================================================
# Auto-Tuning Tests: Graph-Size
# ============================================================================


class TestGraphSizeTuning:
    """Tests für Graph-Size basiertes Tuning"""

    def test_small_graph_liberal_parameters(self, netzwerk_mock, meta_learning_mock):
        """Test: Kleine Graphen (<1000) erhalten liberale Parameter"""
        netzwerk_mock.get_node_count.return_value = 500

        engine = AdaptiveResonanceEngine(
            netzwerk_mock, meta_learning=meta_learning_mock
        )
        params = engine.auto_tune_hyperparameters()

        # Kleiner Graph: Mehr Exploration
        assert params["activation_threshold"] <= 0.25  # Liberal
        assert params["max_waves"] >= 5  # Viele Waves
        assert params["max_concepts_per_wave"] >= 100  # Wenig Pruning

    def test_medium_graph_balanced_parameters(self, netzwerk_mock, meta_learning_mock):
        """Test: Mittlere Graphen (1000-10000) erhalten balanced Parameter"""
        netzwerk_mock.get_node_count.return_value = 5000

        engine = AdaptiveResonanceEngine(
            netzwerk_mock, meta_learning=meta_learning_mock
        )
        params = engine.auto_tune_hyperparameters()

        # Mittlerer Graph: Balanced
        assert 0.25 <= params["activation_threshold"] <= 0.35
        assert 4 <= params["max_waves"] <= 6
        assert 80 <= params["max_concepts_per_wave"] <= 120

    def test_large_graph_conservative_parameters(
        self, netzwerk_mock, meta_learning_mock
    ):
        """Test: Große Graphen (10000-50000) erhalten conservative Parameter"""
        netzwerk_mock.get_node_count.return_value = 30000

        engine = AdaptiveResonanceEngine(
            netzwerk_mock, meta_learning=meta_learning_mock
        )
        params = engine.auto_tune_hyperparameters()

        # Großer Graph: Conservative
        assert params["activation_threshold"] >= 0.3
        assert params["max_waves"] <= 5
        assert params["max_concepts_per_wave"] <= 100

    def test_very_large_graph_aggressive_pruning(
        self, netzwerk_mock, meta_learning_mock
    ):
        """Test: Sehr große Graphen (>50000) erhalten aggressives Pruning"""
        netzwerk_mock.get_node_count.return_value = 100000

        engine = AdaptiveResonanceEngine(
            netzwerk_mock, meta_learning=meta_learning_mock
        )
        params = engine.auto_tune_hyperparameters()

        # Sehr großer Graph: Aggressiv
        assert params["activation_threshold"] >= 0.35
        assert params["max_waves"] <= 4
        assert params["max_concepts_per_wave"] <= 60


# ============================================================================
# Auto-Tuning Tests: Query-Time (Performance)
# ============================================================================


class TestQueryTimeTuning:
    """Tests für Query-Time basiertes Tuning"""

    def test_slow_queries_reduce_parameters(self, netzwerk_mock, meta_learning_mock):
        """Test: Langsame Queries (>5s) reduzieren Waves und Concepts"""
        # Setup: Langsame Queries
        slow_stats = StrategyPerformance(
            strategy_name="resonance",
            queries_handled=10,
            success_rate=0.7,
            avg_confidence=0.75,
            avg_response_time=6.0,  # Zu langsam!
        )
        meta_learning_mock.get_strategy_stats.return_value = slow_stats

        engine = AdaptiveResonanceEngine(
            netzwerk_mock, meta_learning=meta_learning_mock
        )
        initial_waves = engine.max_waves
        initial_concepts = engine.max_concepts_per_wave

        params = engine.auto_tune_hyperparameters()

        # Sollte Pruning erhöhen
        assert params["max_waves"] < initial_waves
        assert params["max_concepts_per_wave"] < initial_concepts

    def test_fast_queries_increase_exploration(self, netzwerk_mock, meta_learning_mock):
        """Test: Schnelle Queries (<0.5s) erlauben mehr Exploration"""
        # Setup: Schnelle Queries
        fast_stats = StrategyPerformance(
            strategy_name="resonance",
            queries_handled=10,
            success_rate=0.7,
            avg_confidence=0.75,
            avg_response_time=0.3,  # Sehr schnell!
        )
        meta_learning_mock.get_strategy_stats.return_value = fast_stats

        engine = AdaptiveResonanceEngine(
            netzwerk_mock, meta_learning=meta_learning_mock
        )
        initial_waves = engine.max_waves
        initial_concepts = engine.max_concepts_per_wave

        params = engine.auto_tune_hyperparameters()

        # Sollte mehr Exploration erlauben
        assert params["max_waves"] >= initial_waves
        assert params["max_concepts_per_wave"] >= initial_concepts


# ============================================================================
# Auto-Tuning Tests: Accuracy
# ============================================================================


class TestAccuracyTuning:
    """Tests für Accuracy basiertes Tuning"""

    def test_low_accuracy_increases_exploration(
        self, netzwerk_mock, meta_learning_mock
    ):
        """Test: Niedrige Accuracy (<0.6) erhöht Exploration"""
        # Setup: Niedrige Accuracy
        low_acc_stats = StrategyPerformance(
            strategy_name="resonance",
            queries_handled=20,
            success_rate=0.5,  # Niedrig!
            avg_confidence=0.6,
            avg_response_time=1.0,
        )
        meta_learning_mock.get_strategy_stats.return_value = low_acc_stats

        engine = AdaptiveResonanceEngine(
            netzwerk_mock, meta_learning=meta_learning_mock
        )
        initial_waves = engine.max_waves
        initial_decay = engine.decay_factor

        params = engine.auto_tune_hyperparameters()

        # Sollte mehr Exploration haben
        assert params["max_waves"] > initial_waves
        assert params["decay_factor"] >= initial_decay

    def test_high_accuracy_maintains_or_optimizes(
        self, netzwerk_mock, meta_learning_mock
    ):
        """Test: Hohe Accuracy (>0.8) behält Parameter oder optimiert Performance"""
        # Setup: Hohe Accuracy
        high_acc_stats = StrategyPerformance(
            strategy_name="resonance",
            queries_handled=20,
            success_rate=0.9,  # Sehr gut!
            avg_confidence=0.85,
            avg_response_time=1.5,
        )
        meta_learning_mock.get_strategy_stats.return_value = high_acc_stats

        engine = AdaptiveResonanceEngine(
            netzwerk_mock, meta_learning=meta_learning_mock
        )
        params = engine.auto_tune_hyperparameters()

        # Parameter sollten in vernünftigem Bereich bleiben
        assert 0.2 <= params["activation_threshold"] <= 0.5
        assert 3 <= params["max_waves"] <= 7


# ============================================================================
# Parameter Boundaries Tests
# ============================================================================


class TestParameterBoundaries:
    """Tests für Parameter Sicherheits-Boundaries"""

    def test_activation_threshold_boundaries(self, adaptive_engine):
        """Test: activation_threshold bleibt in [0.1, 0.6]"""
        params = adaptive_engine.auto_tune_hyperparameters()

        assert 0.1 <= params["activation_threshold"] <= 0.6

    def test_decay_factor_boundaries(self, adaptive_engine):
        """Test: decay_factor bleibt in [0.5, 0.9]"""
        params = adaptive_engine.auto_tune_hyperparameters()

        assert 0.5 <= params["decay_factor"] <= 0.9

    def test_resonance_boost_boundaries(self, adaptive_engine):
        """Test: resonance_boost bleibt in [0.1, 0.8]"""
        params = adaptive_engine.auto_tune_hyperparameters()

        assert 0.1 <= params["resonance_boost"] <= 0.8

    def test_max_waves_boundaries(self, adaptive_engine):
        """Test: max_waves bleibt in [2, 10]"""
        params = adaptive_engine.auto_tune_hyperparameters()

        assert 2 <= params["max_waves"] <= 10

    def test_max_concepts_per_wave_boundaries(self, adaptive_engine):
        """Test: max_concepts_per_wave bleibt in [20, 300]"""
        params = adaptive_engine.auto_tune_hyperparameters()

        assert 20 <= params["max_concepts_per_wave"] <= 300


# ============================================================================
# Tuning History Tests
# ============================================================================


class TestTuningHistory:
    """Tests für Tuning History Tracking"""

    def test_tuning_creates_history_entry(self, adaptive_engine):
        """Test: Auto-tuning erstellt History-Eintrag"""
        assert len(adaptive_engine.tuning_history) == 0

        adaptive_engine.auto_tune_hyperparameters()

        assert len(adaptive_engine.tuning_history) == 1

    def test_multiple_tunings_create_multiple_entries(self, adaptive_engine):
        """Test: Mehrere Tunings erstellen mehrere Einträge"""
        for _ in range(5):
            adaptive_engine.auto_tune_hyperparameters()

        assert len(adaptive_engine.tuning_history) == 5

    def test_history_entry_contains_required_fields(self, adaptive_engine):
        """Test: History-Eintrag enthält alle erforderlichen Felder"""
        adaptive_engine.auto_tune_hyperparameters()

        entry = adaptive_engine.tuning_history[0]

        assert "timestamp" in entry
        assert "graph_size" in entry
        assert "avg_query_time" in entry
        assert "avg_accuracy" in entry
        assert "parameters" in entry


# ============================================================================
# Integration Tests
# ============================================================================


class TestMetaLearningIntegration:
    """Tests für MetaLearningEngine Integration"""

    def test_uses_meta_learning_stats(self, netzwerk_mock, meta_learning_mock):
        """Test: Engine verwendet MetaLearningEngine Stats"""
        engine = AdaptiveResonanceEngine(
            netzwerk_mock, meta_learning=meta_learning_mock
        )

        engine.auto_tune_hyperparameters()

        # Verify meta_learning.get_strategy_stats wurde aufgerufen
        meta_learning_mock.get_strategy_stats.assert_called_once_with("resonance")

    def test_handles_missing_meta_learning(self, netzwerk_mock):
        """Test: Engine funktioniert ohne MetaLearningEngine"""
        engine = AdaptiveResonanceEngine(netzwerk_mock, meta_learning=None)

        # Sollte nicht crashen
        params = engine.auto_tune_hyperparameters()

        assert params is not None
        assert isinstance(params, dict)

    def test_handles_missing_strategy_stats(self, netzwerk_mock, meta_learning_mock):
        """Test: Engine handled fehlende Strategy-Stats gracefully"""
        meta_learning_mock.get_strategy_stats.return_value = None

        engine = AdaptiveResonanceEngine(
            netzwerk_mock, meta_learning=meta_learning_mock
        )
        params = engine.auto_tune_hyperparameters()

        # Sollte Defaults verwenden
        assert params is not None


# ============================================================================
# Tuning Stats Tests
# ============================================================================


class TestTuningStats:
    """Tests für get_tuning_stats()"""

    def test_get_tuning_stats_structure(self, adaptive_engine):
        """Test: get_tuning_stats() gibt korrektes Format zurück"""
        stats = adaptive_engine.get_tuning_stats()

        assert "current_parameters" in stats
        assert "initial_parameters" in stats
        assert "tuning_history" in stats
        assert "total_tunings" in stats

    def test_current_parameters_match_engine_state(self, adaptive_engine):
        """Test: current_parameters entsprechen Engine-State"""
        adaptive_engine.auto_tune_hyperparameters()

        stats = adaptive_engine.get_tuning_stats()
        current = stats["current_parameters"]

        assert current["activation_threshold"] == adaptive_engine.activation_threshold
        assert current["decay_factor"] == adaptive_engine.decay_factor
        assert current["max_waves"] == adaptive_engine.max_waves


# ============================================================================
# Reset Tests
# ============================================================================


class TestReset:
    """Tests für reset_to_defaults()"""

    def test_reset_to_defaults(self, adaptive_engine):
        """Test: reset_to_defaults() stellt initiale Werte wieder her"""
        initial = adaptive_engine._initial_hyperparameters.copy()

        # Tune parameters
        adaptive_engine.auto_tune_hyperparameters()

        # Parameters sollten geändert sein (wahrscheinlich)
        # Reset
        adaptive_engine.reset_to_defaults()

        # Sollten wieder initial sein
        assert adaptive_engine.activation_threshold == initial["activation_threshold"]
        assert adaptive_engine.decay_factor == initial["decay_factor"]
        assert adaptive_engine.max_waves == initial["max_waves"]


# ============================================================================
# Auto-Tune Parameter in activate_concept Tests
# ============================================================================


class TestAutoTuneInActivateConcept:
    """Tests für auto_tune Parameter in activate_concept()"""

    @patch.object(AdaptiveResonanceEngine, "auto_tune_hyperparameters")
    def test_activate_concept_without_auto_tune(
        self, mock_auto_tune, adaptive_engine, netzwerk_mock
    ):
        """Test: activate_concept() ohne auto_tune ruft Tuning NICHT auf"""
        # Mock activate_concept der parent class
        with patch.object(ResonanceEngine, "activate_concept") as mock_parent:
            adaptive_engine.activate_concept("test", auto_tune=False)

            # Auto-tuning sollte NICHT aufgerufen werden
            mock_auto_tune.assert_not_called()

    @patch.object(AdaptiveResonanceEngine, "auto_tune_hyperparameters")
    def test_activate_concept_with_auto_tune(self, mock_auto_tune, adaptive_engine):
        """Test: activate_concept() mit auto_tune=True ruft Tuning auf"""
        # Mock activate_concept der parent class
        with patch.object(ResonanceEngine, "activate_concept") as mock_parent:
            adaptive_engine.activate_concept("test", auto_tune=True)

            # Auto-tuning sollte aufgerufen werden
            mock_auto_tune.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
