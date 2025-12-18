# tests/test_shared_fact_cache.py
"""
Test Suite für Quick Win #4: Shared Fact Cache

Testet die Performance-Optimierung zur Vermeidung redundanter Queries
in der Inference-Pipeline.

QUICK WIN #4 Ziele:
- Facts werden EINMAL geladen und zwischen Handlers geteilt
- Thread-safe Caching mit RLock
- 5-10x Speedup bei Multi-Strategy Reasoning
- Keine redundanten Neo4j-Queries (3-5x Overhead vermieden)
"""

import time
from unittest.mock import MagicMock

import pytest

from kai_inference_handler import KaiInferenceHandler


class TestSharedFactCache:
    """Test Suite für Shared Fact Cache Optimierung."""

    @pytest.fixture(autouse=True)
    def clear_cache(self):
        """Clear Shared Fact Cache vor jedem Test (Test-Isolation)."""
        from infrastructure.cache_manager import CacheManager

        cache_mgr = CacheManager()
        # Invalidiere gesamten inference_shared_facts Cache
        if "inference_shared_facts" in cache_mgr.caches:
            cache_mgr.caches["inference_shared_facts"].clear()
        yield
        # Cleanup nach Test
        if "inference_shared_facts" in cache_mgr.caches:
            cache_mgr.caches["inference_shared_facts"].clear()

    @pytest.fixture
    def mock_netzwerk(self):
        """Mock KonzeptNetzwerk mit query_graph_for_facts()."""
        netzwerk = MagicMock()

        # Simuliere Facts für "hund"
        netzwerk.query_graph_for_facts.return_value = {
            "IS_A": ["saugetier", "tier"],
            "HAS_PROPERTY": ["vierbeinig", "haarig"],
            "CAPABLE_OF": ["bellen", "laufen"],
        }

        # Fallback für query_graph_for_facts_with_confidence
        netzwerk.query_graph_for_facts_with_confidence.side_effect = AttributeError(
            "Not implemented"
        )

        return netzwerk

    @pytest.fixture
    def mock_engine(self):
        """Mock Logic Engine."""
        return MagicMock()

    @pytest.fixture
    def mock_graph_traversal(self):
        """Mock Graph Traversal."""
        return MagicMock()

    @pytest.fixture
    def mock_working_memory(self):
        """Mock Working Memory."""
        return MagicMock()

    @pytest.fixture
    def mock_signals(self):
        """Mock Signals."""
        return MagicMock()

    @pytest.fixture
    def inference_handler(
        self,
        mock_netzwerk,
        mock_engine,
        mock_graph_traversal,
        mock_working_memory,
        mock_signals,
    ):
        """Erstelle KaiInferenceHandler mit Mocks."""
        handler = KaiInferenceHandler(
            netzwerk=mock_netzwerk,
            engine=mock_engine,
            graph_traversal=mock_graph_traversal,
            working_memory=mock_working_memory,
            signals=mock_signals,
            enable_hybrid_reasoning=False,  # Deaktiviere für isolierte Tests
        )
        return handler

    def test_load_shared_facts_basic(self, inference_handler, mock_netzwerk):
        """Test: _load_shared_facts() lädt Facts korrekt."""
        # Load facts
        shared_facts = inference_handler._load_shared_facts(topic="hund")

        # Verify structure
        assert "facts" in shared_facts
        assert "facts_with_confidence" in shared_facts
        assert "topic" in shared_facts
        assert "cached" in shared_facts
        assert "cache_key" in shared_facts

        # Verify content
        assert shared_facts["topic"] == "hund"
        assert shared_facts["facts"]["IS_A"] == ["saugetier", "tier"]
        assert shared_facts["facts"]["HAS_PROPERTY"] == ["vierbeinig", "haarig"]
        assert shared_facts["cached"] is False  # First load not cached

        # Verify Neo4j query called
        mock_netzwerk.query_graph_for_facts.assert_called_once()

    def test_load_shared_facts_caching(self, inference_handler, mock_netzwerk):
        """Test: Wiederholte Calls nutzen Cache (keine redundanten Queries)."""
        # First call: Load from DB
        shared_facts_1 = inference_handler._load_shared_facts(topic="hund")
        assert shared_facts_1["cached"] is False
        assert mock_netzwerk.query_graph_for_facts.call_count == 1

        # Second call: Load from cache
        shared_facts_2 = inference_handler._load_shared_facts(topic="hund")
        assert shared_facts_2["cached"] is True
        # NO additional DB query!
        assert mock_netzwerk.query_graph_for_facts.call_count == 1

        # Verify same facts returned
        assert shared_facts_1["facts"] == shared_facts_2["facts"]

    def test_load_shared_facts_different_topics(self, inference_handler, mock_netzwerk):
        """Test: Verschiedene Topics haben separate Cache-Einträge."""
        # Load facts for "hund"
        shared_facts_hund = inference_handler._load_shared_facts(topic="hund")
        assert mock_netzwerk.query_graph_for_facts.call_count == 1

        # Load facts for "katze" (different topic)
        mock_netzwerk.query_graph_for_facts.return_value = {
            "IS_A": ["saugetier"],
            "CAPABLE_OF": ["miauen"],
        }
        shared_facts_katze = inference_handler._load_shared_facts(topic="katze")
        assert mock_netzwerk.query_graph_for_facts.call_count == 2  # New query

        # Verify different cache keys
        assert shared_facts_hund["cache_key"] != shared_facts_katze["cache_key"]

        # Re-load "hund" -> should use cache
        shared_facts_hund_2 = inference_handler._load_shared_facts(topic="hund")
        assert shared_facts_hund_2["cached"] is True
        assert mock_netzwerk.query_graph_for_facts.call_count == 2  # No new query

    def test_load_shared_facts_min_confidence_filter(
        self, inference_handler, mock_netzwerk
    ):
        """Test: min_confidence Parameter wird korrekt weitergegeben."""
        # Load with min_confidence=0.7
        inference_handler._load_shared_facts(topic="hund", min_confidence=0.7)

        # Verify query_graph_for_facts called with min_confidence
        mock_netzwerk.query_graph_for_facts.assert_called_with(
            topic="hund", min_confidence=0.7, sort_by_confidence=True
        )

    def test_load_shared_facts_relation_type_filter(
        self, inference_handler, mock_netzwerk
    ):
        """Test: relation_types Filter funktioniert."""
        # Load only IS_A relations
        shared_facts = inference_handler._load_shared_facts(
            topic="hund", relation_types=["IS_A", "HAS_PROPERTY"]
        )

        # Verify only requested relations included
        assert "IS_A" in shared_facts["facts"]
        assert "HAS_PROPERTY" in shared_facts["facts"]
        assert "CAPABLE_OF" not in shared_facts["facts"]

    def test_load_shared_facts_cache_key_includes_filters(self, inference_handler):
        """Test: Cache-Key beinhaltet min_confidence und relation_types."""
        # Load with different filters
        facts_1 = inference_handler._load_shared_facts(topic="hund", min_confidence=0.5)
        facts_2 = inference_handler._load_shared_facts(topic="hund", min_confidence=0.7)
        facts_3 = inference_handler._load_shared_facts(
            topic="hund", relation_types=["IS_A"]
        )

        # Verify different cache keys
        assert facts_1["cache_key"] != facts_2["cache_key"]
        assert facts_1["cache_key"] != facts_3["cache_key"]
        assert facts_2["cache_key"] != facts_3["cache_key"]

    def test_load_shared_facts_thread_safety(self, inference_handler, mock_netzwerk):
        """Test: Thread-Safety mit RLock."""
        import threading

        results = []
        errors = []

        def load_facts():
            try:
                facts = inference_handler._load_shared_facts(topic="hund")
                results.append(facts)
            except Exception as e:
                errors.append(e)

        # Starte 10 Threads parallel
        threads = [threading.Thread(target=load_facts) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify no errors
        assert len(errors) == 0

        # Verify all threads got facts
        assert len(results) == 10

        # Verify facts are consistent (same cache_key)
        cache_keys = [r["cache_key"] for r in results]
        assert all(key == cache_keys[0] for key in cache_keys)

    def test_try_backward_chaining_uses_shared_facts(
        self, inference_handler, mock_netzwerk
    ):
        """Test: try_backward_chaining_inference() nutzt Shared Fact Cache."""
        # Mock handlers to return None (so we can test cache usage)
        inference_handler._graph_traversal_handler.try_graph_traversal = MagicMock(
            return_value=None
        )
        inference_handler._backward_chaining_handler.try_backward_chaining = MagicMock(
            return_value=None
        )
        inference_handler._abductive_handler.try_abductive_reasoning = MagicMock(
            return_value=None
        )

        # Call try_backward_chaining_inference
        inference_handler.try_backward_chaining_inference(topic="hund")

        # Verify query_graph_for_facts called (for shared facts)
        assert mock_netzwerk.query_graph_for_facts.called

    def test_performance_improvement(self, inference_handler, mock_netzwerk):
        """Test: Performance Improvement durch Cache (5x Speedup erwartet)."""

        # Simuliere langsame DB-Query (50ms)
        def slow_query(*args, **kwargs):
            time.sleep(0.05)  # 50ms Delay
            return {
                "IS_A": ["saugetier"],
                "HAS_PROPERTY": ["vierbeinig"],
            }

        mock_netzwerk.query_graph_for_facts.side_effect = slow_query

        # First load: ~50ms
        start = time.time()
        facts_1 = inference_handler._load_shared_facts(topic="hund")
        duration_first = time.time() - start

        assert duration_first >= 0.05  # At least 50ms
        assert facts_1["cached"] is False

        # Second load: <10ms (from cache)
        start = time.time()
        facts_2 = inference_handler._load_shared_facts(topic="hund")
        duration_cached = time.time() - start

        assert duration_cached < 0.01  # Less than 10ms
        assert facts_2["cached"] is True

        # Performance improvement: At least 5x faster
        speedup = duration_first / duration_cached
        assert speedup >= 5.0

        print(
            f"\nPerformance: First={duration_first*1000:.1f}ms, "
            f"Cached={duration_cached*1000:.1f}ms, Speedup={speedup:.1f}x"
        )

    def test_cache_ttl_expiration(self, inference_handler, mock_netzwerk):
        """Test: Cache TTL (5 Minuten) funktioniert."""
        # Load facts (cache for 300s)
        facts_1 = inference_handler._load_shared_facts(topic="hund")
        assert facts_1["cached"] is False
        assert mock_netzwerk.query_graph_for_facts.call_count == 1

        # Immediate re-load: from cache
        facts_2 = inference_handler._load_shared_facts(topic="hund")
        assert facts_2["cached"] is True
        assert mock_netzwerk.query_graph_for_facts.call_count == 1

        # Simulate TTL expiration (mock time)
        # In real scenario, TTLCache would expire after 300s
        # For test, manually invalidate cache
        inference_handler._cache_manager.invalidate(
            "inference_shared_facts", facts_1["cache_key"]
        )

        # Re-load after invalidation: NEW query
        facts_3 = inference_handler._load_shared_facts(topic="hund")
        assert facts_3["cached"] is False
        assert mock_netzwerk.query_graph_for_facts.call_count == 2
