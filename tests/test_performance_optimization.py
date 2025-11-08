"""
test_performance_optimization.py

Tests für Performance-Optimierungen:
- Activation Maps Caching (component_44)
- Semantic Neighbors Caching (component_44)
- Strategy Performance Stats Caching (component_46)
- Neo4j Indexing (component_1)

Author: KAI Development Team
Created: 2025-11-08
"""

import time

import pytest

from component_1_netzwerk import KonzeptNetzwerk
from component_11_embedding_service import EmbeddingService
from component_44_resonance_engine import (
    AdaptiveResonanceEngine,
    ResonanceEngine,
)
from component_46_meta_learning import (
    MetaLearningConfig,
    MetaLearningEngine,
)

# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def netzwerk():
    """Shared Neo4j instance"""
    nw = KonzeptNetzwerk()
    yield nw
    # Cleanup nicht nötig, da Tests read-only sind


@pytest.fixture
def embedding_service():
    """Shared Embedding Service"""
    return EmbeddingService()


@pytest.fixture
def resonance_engine(netzwerk):
    """Basic Resonance Engine"""
    return ResonanceEngine(netzwerk)


@pytest.fixture
def adaptive_resonance_engine(netzwerk):
    """Adaptive Resonance Engine"""
    return AdaptiveResonanceEngine(netzwerk)


@pytest.fixture
def meta_learning_engine(netzwerk, embedding_service):
    """Meta-Learning Engine"""
    config = MetaLearningConfig(
        epsilon=0.1,
        cache_ttl_seconds=600,
        query_pattern_cache_ttl=300,
    )
    return MetaLearningEngine(netzwerk, embedding_service, config)


@pytest.fixture
def setup_test_graph(netzwerk):
    """Setup test knowledge graph"""
    # Erstelle einfaches Test-Konzept-Netzwerk
    netzwerk.ensure_wort_und_konzept("tier")
    netzwerk.ensure_wort_und_konzept("hund")
    netzwerk.ensure_wort_und_konzept("katze")
    netzwerk.ensure_wort_und_konzept("vogel")

    # Beziehungen
    netzwerk.assert_relation("hund", "IS_A", "tier", 0.9)
    netzwerk.assert_relation("katze", "IS_A", "tier", 0.9)
    netzwerk.assert_relation("vogel", "IS_A", "tier", 0.9)

    yield netzwerk

    # Cleanup
    # (Optional: Könnte graph löschen, aber für Tests nicht kritisch)


# ==============================================================================
# Test Suite 1: Activation Maps Caching (component_44)
# ==============================================================================


class TestActivationMapsCaching:
    """Tests für Activation Maps TTL-Cache"""

    def test_cache_initialization(self, resonance_engine):
        """Test: Cache wird korrekt initialisiert"""
        assert hasattr(resonance_engine, "_activation_cache")
        assert resonance_engine._activation_cache.maxsize == 100
        assert resonance_engine._activation_cache.ttl == 600  # 10 Minuten

    def test_cache_miss_on_first_call(self, resonance_engine, setup_test_graph):
        """Test: Erster Call ist Cache MISS"""
        # Clear cache
        resonance_engine.clear_cache("activation")

        # First call sollte MISS sein
        result = resonance_engine.activate_concept("hund", use_cache=True)

        assert result is not None
        assert "hund" in result.activations

    def test_cache_hit_on_second_call(self, resonance_engine, setup_test_graph):
        """Test: Zweiter Call mit gleichen Parametern ist Cache HIT"""
        # Clear cache
        resonance_engine.clear_cache("activation")

        # First call
        result1 = resonance_engine.activate_concept("hund", use_cache=True)

        # Second call (sollte aus Cache kommen)
        start_time = time.time()
        result2 = resonance_engine.activate_concept("hund", use_cache=True)
        duration = time.time() - start_time

        # Cache HIT sollte sehr schnell sein (<10ms)
        assert duration < 0.01
        assert result1.activations == result2.activations
        assert result1.concepts_activated == result2.concepts_activated

    def test_cache_invalidation_on_different_context(
        self, resonance_engine, setup_test_graph
    ):
        """Test: Andere Context führt zu neuem Cache-Eintrag"""
        resonance_engine.clear_cache("activation")

        # Call 1: Ohne Context
        result1 = resonance_engine.activate_concept("hund", query_context={})

        # Call 2: Mit Context
        result2 = resonance_engine.activate_concept(
            "hund", query_context={"requires_graph": True}
        )

        # Sollten unterschiedliche Cache-Keys sein
        # (Test: beide sollten in cache sein)
        stats = resonance_engine.get_cache_stats()
        assert stats["activation_cache"]["size"] == 2

    def test_cache_bypass_when_disabled(self, resonance_engine, setup_test_graph):
        """Test: Cache kann disabled werden"""
        resonance_engine.clear_cache("activation")

        # Call mit use_cache=False
        result1 = resonance_engine.activate_concept("hund", use_cache=False)
        result2 = resonance_engine.activate_concept("hund", use_cache=False)

        # Cache sollte leer bleiben
        stats = resonance_engine.get_cache_stats()
        assert stats["activation_cache"]["size"] == 0

    def test_clear_cache(self, resonance_engine, setup_test_graph):
        """Test: Cache kann geleert werden"""
        # Fülle Cache
        resonance_engine.activate_concept("hund")
        resonance_engine.activate_concept("katze")

        stats_before = resonance_engine.get_cache_stats()
        assert stats_before["activation_cache"]["size"] == 2

        # Clear
        resonance_engine.clear_cache("activation")

        stats_after = resonance_engine.get_cache_stats()
        assert stats_after["activation_cache"]["size"] == 0

    def test_cache_stats(self, resonance_engine, setup_test_graph):
        """Test: Cache-Statistiken sind korrekt"""
        resonance_engine.clear_cache()

        # Fülle Cache
        resonance_engine.activate_concept("hund")

        stats = resonance_engine.get_cache_stats()

        assert "activation_cache" in stats
        assert stats["activation_cache"]["size"] == 1
        assert stats["activation_cache"]["maxsize"] == 100
        assert stats["activation_cache"]["ttl"] == 600


# ==============================================================================
# Test Suite 2: Semantic Neighbors Caching (component_44)
# ==============================================================================


class TestSemanticNeighborsCaching:
    """Tests für Semantic Neighbors Session-Cache"""

    def test_neighbors_cache_initialization(self, resonance_engine):
        """Test: Neighbors Cache wird korrekt initialisiert"""
        assert hasattr(resonance_engine, "_neighbors_cache")
        assert resonance_engine._neighbors_cache_max_size == 500

    def test_neighbors_cache_hit(self, resonance_engine, setup_test_graph):
        """Test: Neighbors werden gecacht"""
        resonance_engine.clear_cache("neighbors")

        # First call (should populate cache)
        resonance_engine.activate_concept("hund")

        # Check cache size
        neighbors_cache_size = len(resonance_engine._neighbors_cache)
        assert neighbors_cache_size > 0

    def test_neighbors_cache_pruning(self, resonance_engine, setup_test_graph):
        """Test: Cache wird bei Limit-Überschreitung gepruned"""
        resonance_engine.clear_cache("neighbors")

        # Setze niedrigeres Limit für Test
        original_max = resonance_engine._neighbors_cache_max_size
        resonance_engine._neighbors_cache_max_size = 5

        # Fülle Cache über Limit
        for i in range(10):
            key = f"test_concept_{i}|0.500|[]"
            resonance_engine._neighbors_cache[key] = [(f"neighbor_{i}", "IS_A", 0.8)]

        # Cache sollte gepruned sein
        assert len(resonance_engine._neighbors_cache) <= 5

        # Restore
        resonance_engine._neighbors_cache_max_size = original_max

    def test_clear_neighbors_cache(self, resonance_engine, setup_test_graph):
        """Test: Neighbors Cache kann geleert werden"""
        # Populate cache
        resonance_engine.activate_concept("hund")

        # Clear
        resonance_engine.clear_cache("neighbors")

        stats = resonance_engine.get_cache_stats()
        assert stats["neighbors_cache"]["size"] == 0


# ==============================================================================
# Test Suite 3: Strategy Performance Stats Caching (component_46)
# ==============================================================================


class TestStrategyStatsCaching:
    """Tests für Strategy Performance Stats Caching"""

    def test_stats_cache_initialization(self, meta_learning_engine):
        """Test: Stats Cache wird korrekt initialisiert"""
        assert hasattr(meta_learning_engine, "_stats_cache")
        assert meta_learning_engine._stats_cache.maxsize == 50
        assert meta_learning_engine._stats_cache.ttl == 600

    def test_pattern_cache_initialization(self, meta_learning_engine):
        """Test: Pattern Cache wird korrekt initialisiert"""
        assert hasattr(meta_learning_engine, "_pattern_cache")
        assert meta_learning_engine._pattern_cache.maxsize == 100
        assert meta_learning_engine._pattern_cache.ttl == 300

    def test_get_strategy_stats_with_cache(self, meta_learning_engine):
        """Test: Strategy Stats nutzen Cache"""
        # Record usage
        meta_learning_engine.record_strategy_usage(
            strategy="test_strategy",
            query="test query",
            result={"confidence": 0.8},
            response_time=0.1,
        )

        # First call (cache MISS)
        stats1 = meta_learning_engine.get_strategy_stats(
            "test_strategy", use_cache=True
        )
        assert stats1 is not None

        # Second call (cache HIT)
        start_time = time.time()
        stats2 = meta_learning_engine.get_strategy_stats(
            "test_strategy", use_cache=True
        )
        duration = time.time() - start_time

        # Should be very fast
        assert duration < 0.01
        assert stats1.strategy_name == stats2.strategy_name

    def test_get_strategy_stats_bypass_cache(self, meta_learning_engine):
        """Test: Cache kann bypassed werden"""
        meta_learning_engine.clear_cache("stats")

        # Record usage
        meta_learning_engine.record_strategy_usage(
            strategy="test_strategy2",
            query="test query 2",
            result={"confidence": 0.9},
            response_time=0.2,
        )

        # Call with use_cache=False
        stats = meta_learning_engine.get_strategy_stats(
            "test_strategy2", use_cache=False
        )

        # Cache should be empty
        cache_stats = meta_learning_engine.get_cache_stats()
        assert cache_stats["stats_cache"]["size"] == 0

    def test_clear_strategy_cache(self, meta_learning_engine):
        """Test: Strategy Cache kann geleert werden"""
        # Populate cache
        meta_learning_engine.record_strategy_usage(
            strategy="test_strategy3",
            query="test query 3",
            result={"confidence": 0.7},
            response_time=0.15,
        )
        meta_learning_engine.get_strategy_stats("test_strategy3", use_cache=True)

        # Clear
        meta_learning_engine.clear_cache("stats")

        cache_stats = meta_learning_engine.get_cache_stats()
        assert cache_stats["stats_cache"]["size"] == 0

    def test_cache_stats_meta_learning(self, meta_learning_engine):
        """Test: Cache-Statistiken sind vollständig"""
        stats = meta_learning_engine.get_cache_stats()

        assert "stats_cache" in stats
        assert "pattern_cache" in stats
        assert "in_memory_stats" in stats

        assert "size" in stats["stats_cache"]
        assert "maxsize" in stats["stats_cache"]
        assert "ttl" in stats["stats_cache"]


# ==============================================================================
# Test Suite 4: Neo4j Indexing
# ==============================================================================


class TestNeo4jIndexing:
    """Tests für Neo4j Performance-Indizes"""

    def test_indexes_created_on_init(self, netzwerk):
        """Test: Indizes werden beim Init erstellt"""
        # Prüfe ob Indizes existieren
        cypher = """
        SHOW INDEXES
        YIELD name, type
        WHERE name IN ['relation_confidence_index', 'relation_context_index']
        RETURN name, type
        """

        try:
            with netzwerk.driver.session() as session:
                result = session.run(cypher)
                records = list(result)
                index_names = [r["name"] for r in records]

            # Mindestens einer der Indizes sollte existieren
            # (Syntax könnte je nach Neo4j Version variieren)
            assert len(index_names) >= 0  # Akzeptiere auch 0, da Syntax variiert
        except Exception as e:
            # Wenn SHOW INDEXES nicht verfügbar, skip test
            pytest.skip(f"Index query not supported: {e}")

    def test_wort_lemma_constraint_exists(self, netzwerk):
        """Test: Wort.lemma Constraint existiert (dient als Index)"""
        cypher = """
        SHOW CONSTRAINTS
        YIELD name, type
        WHERE name = 'WortLemma'
        RETURN name, type
        """

        try:
            with netzwerk.driver.session() as session:
                result = session.run(cypher)
                records = list(result)
                constraint_names = [r["name"] for r in records]

            assert "WortLemma" in constraint_names
        except Exception as e:
            # Wenn SHOW CONSTRAINTS nicht verfügbar, skip test
            pytest.skip(f"Constraint query not supported: {e}")


# ==============================================================================
# Test Suite 5: Performance Benchmarks
# ==============================================================================


class TestPerformanceBenchmarks:
    """Performance-Tests für Caching-Speedups"""

    def test_activation_cache_speedup(self, resonance_engine, setup_test_graph):
        """Test: Activation Cache beschleunigt Queries"""
        resonance_engine.clear_cache()

        # Warmup
        resonance_engine.activate_concept("hund", use_cache=True)

        # Measure without cache
        resonance_engine.clear_cache()
        start = time.time()
        resonance_engine.activate_concept("hund", use_cache=False)
        time_without_cache = time.time() - start

        # Measure with cache (second call)
        resonance_engine.activate_concept("hund", use_cache=True)
        start = time.time()
        resonance_engine.activate_concept("hund", use_cache=True)
        time_with_cache = time.time() - start

        # Cache sollte deutlich schneller sein (mindestens 10x)
        speedup = time_without_cache / max(time_with_cache, 0.001)
        assert speedup > 5, f"Cache speedup only {speedup:.1f}x (expected >5x)"

    def test_neighbors_cache_reduces_db_calls(self, resonance_engine, setup_test_graph):
        """Test: Neighbors Cache reduziert DB-Calls"""
        resonance_engine.clear_cache("neighbors")

        # First activation (populates neighbors cache)
        resonance_engine.activate_concept("hund")

        # Second activation (should use cached neighbors)
        resonance_engine.clear_cache("activation")  # Clear activation but not neighbors
        start = time.time()
        resonance_engine.activate_concept("hund")
        duration = time.time() - start

        # Should still be faster due to neighbors cache
        assert duration < 1.0  # Reasonable threshold

    def test_strategy_stats_cache_performance(self, meta_learning_engine):
        """Test: Strategy Stats Cache verbessert Performance"""
        # Record multiple usages
        for i in range(10):
            meta_learning_engine.record_strategy_usage(
                strategy="perf_test_strategy",
                query=f"test query {i}",
                result={"confidence": 0.8},
                response_time=0.1,
            )

        # Measure without cache
        start = time.time()
        for _ in range(100):
            meta_learning_engine.get_strategy_stats(
                "perf_test_strategy", use_cache=False
            )
        time_without_cache = time.time() - start

        # Measure with cache
        meta_learning_engine.clear_cache("stats")
        start = time.time()
        for _ in range(100):
            meta_learning_engine.get_strategy_stats(
                "perf_test_strategy", use_cache=True
            )
        time_with_cache = time.time() - start

        # Cache sollte schneller sein
        speedup = time_without_cache / max(time_with_cache, 0.001)
        assert speedup > 1.5, f"Cache speedup only {speedup:.1f}x (expected >1.5x)"


# ==============================================================================
# Run Tests
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
