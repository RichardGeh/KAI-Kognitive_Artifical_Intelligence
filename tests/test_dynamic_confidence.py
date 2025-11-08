# tests/test_dynamic_confidence.py
"""
Tests für Dynamic Confidence System (component_47_dynamic_confidence.py)

Test-Kategorien:
1. Temporal Decay Tests
2. Usage Reinforcement Tests
3. Combined Tests (Decay + Reinforcement)
4. Usage Tracking Tests
5. Backwards Compatibility Tests
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock, patch

import pytest

from component_47_dynamic_confidence import (
    DynamicConfidenceConfig,
    DynamicConfidenceManager,
    UsageStatistics,
)
from component_confidence_manager import CombinationStrategy, ConfidenceLevel

# ==================== FIXTURES ====================


@pytest.fixture
def mock_netzwerk():
    """Mock KonzeptNetzwerk für Tests."""
    netzwerk = Mock()
    netzwerk.driver = Mock()
    return netzwerk


@pytest.fixture
def default_config():
    """Standard-Konfiguration für Tests."""
    return DynamicConfidenceConfig(
        half_life_days=100,
        min_confidence=0.3,
        usage_boost_per_use=0.1,
        max_usage_boost=0.5,
        recency_threshold_days=7,
        recency_boost_factor=1.2,
        enable_temporal_decay=True,
        enable_usage_reinforcement=True,
    )


@pytest.fixture
def manager(mock_netzwerk, default_config):
    """DynamicConfidenceManager mit Standard-Konfiguration."""
    return DynamicConfidenceManager(mock_netzwerk, default_config)


# ==================== TEMPORAL DECAY TESTS ====================


def test_temporal_decay_reduces_confidence(manager):
    """Test: Temporal Decay reduziert Confidence mit Zeit."""
    # Fact ist 100 Tage alt (genau 1 Halbwertszeit)
    old_timestamp = datetime.now() - timedelta(days=100)

    # Mock usage statistics (keine Usage)
    with patch.object(manager, "_get_usage_statistics") as mock_stats:
        mock_stats.return_value = UsageStatistics()

        metrics = manager.calculate_dynamic_confidence(
            base_confidence=0.8,
            timestamp=old_timestamp,
            subject="test",
            relation="IS_A",
            object_="test",
        )

        # Nach 1 Halbwertszeit sollte Confidence auf 50% sein
        # 0.8 * 0.5 = 0.4
        assert 0.38 < metrics.value < 0.42  # Toleranz für Floating-Point
        assert "Decay" in metrics.explanation


def test_temporal_decay_respects_min_confidence(manager):
    """Test: Decay geht nie unter min_confidence."""
    # Sehr alter Fact (10 Halbwertszeiten = 1000 Tage)
    very_old = datetime.now() - timedelta(days=1000)

    with patch.object(manager, "_get_usage_statistics") as mock_stats:
        mock_stats.return_value = UsageStatistics()

        metrics = manager.calculate_dynamic_confidence(
            base_confidence=0.9,
            timestamp=very_old,
            subject="test",
            relation="IS_A",
            object_="test",
        )

        # Sollte min_confidence sein (0.3)
        assert metrics.value == manager.config.min_confidence


def test_temporal_decay_disabled(mock_netzwerk):
    """Test: Kein Decay wenn disabled."""
    config = DynamicConfidenceConfig(enable_temporal_decay=False)
    manager = DynamicConfidenceManager(mock_netzwerk, config)

    old_timestamp = datetime.now() - timedelta(days=365)

    with patch.object(manager, "_get_usage_statistics") as mock_stats:
        mock_stats.return_value = UsageStatistics()

        metrics = manager.calculate_dynamic_confidence(
            base_confidence=0.8,
            timestamp=old_timestamp,
            subject="test",
            relation="IS_A",
            object_="test",
        )

        # Confidence sollte unverändert sein
        assert metrics.value == 0.8
        assert "Decay: disabled" in metrics.explanation


def test_temporal_decay_with_different_half_lives(mock_netzwerk):
    """Test: Verschiedene Halbwertszeiten führen zu verschiedenen Decay-Raten."""
    # Kurze Halbwertszeit (30 Tage)
    config_short = DynamicConfidenceConfig(half_life_days=30)
    manager_short = DynamicConfidenceManager(mock_netzwerk, config_short)

    # Lange Halbwertszeit (500 Tage, Development Mode)
    config_long = DynamicConfidenceConfig(half_life_days=500)
    manager_long = DynamicConfidenceManager(mock_netzwerk, config_long)

    timestamp = datetime.now() - timedelta(days=90)

    with patch.object(manager_short, "_get_usage_statistics") as mock_stats:
        mock_stats.return_value = UsageStatistics()

        metrics_short = manager_short.calculate_dynamic_confidence(
            base_confidence=0.8,
            timestamp=timestamp,
            subject="test",
            relation="IS_A",
            object_="test",
        )

    with patch.object(manager_long, "_get_usage_statistics") as mock_stats:
        mock_stats.return_value = UsageStatistics()

        metrics_long = manager_long.calculate_dynamic_confidence(
            base_confidence=0.8,
            timestamp=timestamp,
            subject="test",
            relation="IS_A",
            object_="test",
        )

    # Kurze Halbwertszeit sollte mehr Decay haben
    assert metrics_short.value < metrics_long.value


# ==================== USAGE REINFORCEMENT TESTS ====================


def test_usage_reinforcement_increases_confidence(manager):
    """Test: Usage Reinforcement erhöht Confidence."""
    # Fact mit 5 Nutzungen
    usage_stats = UsageStatistics(
        usage_count=5,
        last_used=datetime.now() - timedelta(days=10),
        first_used=datetime.now() - timedelta(days=100),
    )

    with patch.object(manager, "_get_usage_statistics") as mock_stats:
        mock_stats.return_value = usage_stats

        # Neuer Fact (kein Decay)
        metrics = manager.calculate_dynamic_confidence(
            base_confidence=0.7,
            timestamp=datetime.now(),
            subject="test",
            relation="IS_A",
            object_="test",
        )

        # Usage Boost: 5 * 0.1 = 0.5 (50%)
        # New Confidence: 0.7 * (1 + 0.5) = 1.05 -> capped at 1.0
        assert metrics.value == 1.0
        assert "Usage: 5x" in metrics.explanation


def test_usage_reinforcement_caps_at_max(manager):
    """Test: Usage Boost wird bei max_usage_boost gekappt."""
    # Sehr viele Nutzungen (100x)
    usage_stats = UsageStatistics(
        usage_count=100,
        last_used=datetime.now() - timedelta(days=10),
    )

    with patch.object(manager, "_get_usage_statistics") as mock_stats:
        mock_stats.return_value = usage_stats

        metrics = manager.calculate_dynamic_confidence(
            base_confidence=0.6,
            timestamp=datetime.now(),
            subject="test",
            relation="IS_A",
            object_="test",
        )

        # Max Boost: 0.5 (50%)
        # New Confidence: 0.6 * (1 + 0.5) = 0.9
        assert 0.88 < metrics.value < 0.92


def test_recency_bonus_applied(manager):
    """Test: Recency Bonus wird angewendet für kürzlich genutzte Facts."""
    # Kürzlich genutzt (vor 3 Tagen)
    usage_stats_recent = UsageStatistics(
        usage_count=2,
        last_used=datetime.now() - timedelta(days=3),
    )

    # Lange nicht genutzt (vor 30 Tagen)
    usage_stats_old = UsageStatistics(
        usage_count=2,
        last_used=datetime.now() - timedelta(days=30),
    )

    timestamp = datetime.now() - timedelta(days=50)

    # Mit Recency Bonus
    with patch.object(manager, "_get_usage_statistics") as mock_stats:
        mock_stats.return_value = usage_stats_recent

        metrics_recent = manager.calculate_dynamic_confidence(
            base_confidence=0.7,
            timestamp=timestamp,
            subject="test",
            relation="IS_A",
            object_="test",
        )

    # Ohne Recency Bonus
    with patch.object(manager, "_get_usage_statistics") as mock_stats:
        mock_stats.return_value = usage_stats_old

        metrics_old = manager.calculate_dynamic_confidence(
            base_confidence=0.7,
            timestamp=timestamp,
            subject="test",
            relation="IS_A",
            object_="test",
        )

    # Recency Bonus sollte Confidence erhöhen
    assert metrics_recent.value > metrics_old.value
    assert "Recency:" in metrics_recent.explanation


def test_usage_reinforcement_disabled(mock_netzwerk):
    """Test: Kein Reinforcement wenn disabled."""
    config = DynamicConfidenceConfig(enable_usage_reinforcement=False)
    manager = DynamicConfidenceManager(mock_netzwerk, config)

    # Mock hohe Usage
    usage_stats = UsageStatistics(usage_count=10)

    with patch.object(manager, "_get_usage_statistics") as mock_stats:
        mock_stats.return_value = usage_stats

        metrics = manager.calculate_dynamic_confidence(
            base_confidence=0.7,
            timestamp=datetime.now(),
            subject="test",
            relation="IS_A",
            object_="test",
        )

        # Confidence sollte unverändert sein (mit Toleranz für Floating-Point)
        assert 0.69 < metrics.value < 0.71
        assert "Reinforcement: disabled" in metrics.explanation


# ==================== COMBINED TESTS ====================


def test_decay_with_reinforcement_cancels_out(manager):
    """Test: Decay und Reinforcement können sich ausgleichen."""
    # Fact ist 50 Tage alt (halbe Halbwertszeit)
    timestamp = datetime.now() - timedelta(days=50)

    # Moderate Usage (3x)
    usage_stats = UsageStatistics(
        usage_count=3,
        last_used=datetime.now() - timedelta(days=10),
    )

    with patch.object(manager, "_get_usage_statistics") as mock_stats:
        mock_stats.return_value = usage_stats

        metrics = manager.calculate_dynamic_confidence(
            base_confidence=0.8,
            timestamp=timestamp,
            subject="test",
            relation="IS_A",
            object_="test",
        )

        # Decay reduziert, Usage erhöht -> sollte ungefähr ausgeglichen sein
        # Aber nicht exakt, da es nicht linear ist
        assert 0.7 < metrics.value < 0.9


def test_old_fact_high_usage_still_confident(manager):
    """Test: Alter Fact mit hoher Usage bleibt confident."""
    # Sehr alter Fact (200 Tage = 2 Halbwertszeiten)
    timestamp = datetime.now() - timedelta(days=200)

    # Sehr hohe Usage (10x)
    usage_stats = UsageStatistics(
        usage_count=10,
        last_used=datetime.now() - timedelta(days=2),  # Kürzlich genutzt
    )

    with patch.object(manager, "_get_usage_statistics") as mock_stats:
        mock_stats.return_value = usage_stats

        metrics = manager.calculate_dynamic_confidence(
            base_confidence=0.9,
            timestamp=timestamp,
            subject="test",
            relation="IS_A",
            object_="test",
        )

        # Trotz Alter sollte durch hohe Usage + Recency Bonus noch HIGH sein
        assert metrics.level in [ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM]
        assert metrics.value > 0.5


def test_new_fact_no_usage_stable(manager):
    """Test: Neuer Fact ohne Usage bleibt stabil."""
    # Heute erstellt
    timestamp = datetime.now()

    # Keine Usage
    usage_stats = UsageStatistics()

    with patch.object(manager, "_get_usage_statistics") as mock_stats:
        mock_stats.return_value = usage_stats

        metrics = manager.calculate_dynamic_confidence(
            base_confidence=0.85,
            timestamp=timestamp,
            subject="test",
            relation="IS_A",
            object_="test",
        )

        # Sollte fast unverändert sein (minimal Decay, keine Usage)
        assert 0.84 < metrics.value < 0.86


# ==================== USAGE TRACKING TESTS ====================


def test_get_usage_statistics_empty(manager, mock_netzwerk):
    """Test: Leere Usage Statistics wenn kein Fact existiert."""
    # Mock DB Query ohne Ergebnisse
    mock_session = MagicMock()
    mock_session.__enter__.return_value = mock_session
    mock_session.run.return_value.single.return_value = None
    mock_netzwerk.driver.session.return_value = mock_session

    stats = manager._get_usage_statistics("test", "IS_A", "test")

    assert stats.usage_count == 0
    assert stats.last_used is None
    assert stats.first_used is None
    assert stats.episode_ids == []


def test_get_usage_statistics_with_data(manager, mock_netzwerk):
    """Test: Usage Statistics werden korrekt aus DB gelesen."""
    # Mock DB Query mit Ergebnissen
    mock_session = MagicMock()
    mock_session.__enter__.return_value = mock_session

    # Neo4j timestamp() gibt Millisekunden seit Epoch
    now_ms = int(datetime.now().timestamp() * 1000)
    past_ms = int((datetime.now() - timedelta(days=50)).timestamp() * 1000)

    mock_record = {
        "usage_count": 5,
        "last_used": now_ms,
        "first_used": past_ms,
        "episode_ids": ["ep1", "ep2", "ep3"],
    }
    mock_session.run.return_value.single.return_value = mock_record
    mock_netzwerk.driver.session.return_value = mock_session

    stats = manager._get_usage_statistics("hund", "IS_A", "tier")

    assert stats.usage_count == 5
    assert stats.last_used is not None
    assert stats.first_used is not None
    assert len(stats.episode_ids) == 3


def test_track_usage(manager, mock_netzwerk):
    """Test: Usage Tracking ruft link_fact_to_episode auf."""
    manager.netzwerk.link_fact_to_episode = Mock(return_value=True)

    success = manager.track_usage("hund", "IS_A", "tier", "episode123")

    assert success
    manager.netzwerk.link_fact_to_episode.assert_called_once_with(
        "hund", "IS_A", "tier", "episode123"
    )


def test_track_usage_no_driver(manager):
    """Test: Track Usage fail gracefully wenn kein Driver."""
    manager.netzwerk.driver = None

    success = manager.track_usage("test", "IS_A", "test", "ep1")

    assert success is False


# ==================== BACKWARDS COMPATIBILITY TESTS ====================


def test_classify_confidence_wrapper(manager):
    """Test: classify_confidence Wrapper funktioniert."""
    level = manager.classify_confidence(0.85)
    assert level == ConfidenceLevel.HIGH

    level = manager.classify_confidence(0.6)
    assert level == ConfidenceLevel.MEDIUM

    level = manager.classify_confidence(0.2)
    assert level == ConfidenceLevel.UNKNOWN


def test_combine_confidences_wrapper(manager):
    """Test: combine_confidences Wrapper funktioniert."""
    metrics = manager.combine_confidences(
        [0.8, 0.7, 0.9], strategy=CombinationStrategy.AVERAGE
    )

    assert 0.79 < metrics.value < 0.81  # (0.8 + 0.7 + 0.9) / 3 = 0.8


def test_should_auto_accept_wrapper(manager):
    """Test: should_auto_accept Wrapper funktioniert."""
    assert manager.should_auto_accept(0.85) is True
    assert manager.should_auto_accept(0.7) is False


def test_should_ask_user_wrapper(manager):
    """Test: should_ask_user Wrapper funktioniert."""
    assert manager.should_ask_user(0.65) is True
    assert manager.should_ask_user(0.85) is False
    assert manager.should_ask_user(0.25) is False


def test_should_reject_wrapper(manager):
    """Test: should_reject Wrapper funktioniert."""
    assert manager.should_reject(0.2) is True
    assert manager.should_reject(0.5) is False


def test_all_wrapper_methods_exist(manager):
    """Test: Alle wichtigen Wrapper-Methoden existieren."""
    methods = [
        "classify_confidence",
        "should_auto_accept",
        "should_ask_user",
        "should_reject",
        "combine_confidences",
        "calculate_graph_traversal_confidence",
        "calculate_rule_confidence",
        "calculate_hypothesis_confidence",
        "generate_ui_feedback",
        "explain_confidence",
    ]

    for method in methods:
        assert hasattr(manager, method)
        assert callable(getattr(manager, method))


# ==================== CONFIGURATION TESTS ====================


def test_config_validation():
    """Test: Konfiguration wird validiert."""
    # Ungültige Halbwertszeit
    with pytest.raises(ValueError, match="half_life_days"):
        DynamicConfidenceConfig(half_life_days=-10)

    # Ungültige min_confidence
    with pytest.raises(ValueError, match="min_confidence"):
        DynamicConfidenceConfig(min_confidence=1.5)

    # Ungültige usage_boost_per_use
    with pytest.raises(ValueError, match="usage_boost_per_use"):
        DynamicConfidenceConfig(usage_boost_per_use=-0.1)


def test_config_custom_values(mock_netzwerk):
    """Test: Custom Config-Werte werden verwendet."""
    config = DynamicConfidenceConfig(
        half_life_days=200,
        min_confidence=0.2,
        usage_boost_per_use=0.05,
    )

    manager = DynamicConfidenceManager(mock_netzwerk, config)

    assert manager.config.half_life_days == 200
    assert manager.config.min_confidence == 0.2
    assert manager.config.usage_boost_per_use == 0.05


# ==================== EDGE CASES ====================


def test_calculate_confidence_with_none_timestamp(manager):
    """Test: None Timestamp wird korrekt behandelt (kein Decay)."""
    with patch.object(manager, "_get_usage_statistics") as mock_stats:
        mock_stats.return_value = UsageStatistics()

        metrics = manager.calculate_dynamic_confidence(
            base_confidence=0.8,
            timestamp=None,
            subject="test",
            relation="IS_A",
            object_="test",
        )

        # Sollte 0.8 sein (kein Decay)
        assert metrics.value == 0.8


def test_calculate_confidence_invalid_base(manager):
    """Test: Ungültige Base Confidence wirft Fehler."""
    with pytest.raises(ValueError, match="base_confidence"):
        manager.calculate_dynamic_confidence(
            base_confidence=1.5,  # Ungültig
            timestamp=datetime.now(),
            subject="test",
            relation="IS_A",
            object_="test",
        )


def test_calculate_confidence_clamps_at_1_0(manager):
    """Test: Finale Confidence wird bei 1.0 gekappt."""
    # Sehr hohe Base + hohe Usage
    usage_stats = UsageStatistics(usage_count=100)

    with patch.object(manager, "_get_usage_statistics") as mock_stats:
        mock_stats.return_value = usage_stats

        metrics = manager.calculate_dynamic_confidence(
            base_confidence=0.95,
            timestamp=datetime.now(),
            subject="test",
            relation="IS_A",
            object_="test",
        )

        # Sollte bei 1.0 geklapt werden
        assert metrics.value <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
# Created
