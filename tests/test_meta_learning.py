"""
tests/test_meta_learning.py

Umfassende Tests für Meta-Learning Layer (Component 46)

Tests:
- StrategyPerformance updates
- MetaLearningEngine strategy selection
- Epsilon-Greedy exploration
- Pattern matching
- Neo4j persistence

Author: KAI Development Team
Last Updated: 2025-11-08
"""

import pytest

from component_1_netzwerk import KonzeptNetzwerk
from component_11_embedding_service import EmbeddingService
from component_46_meta_learning import (
    MetaLearningConfig,
    MetaLearningEngine,
    StrategyPerformance,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def netzwerk():
    """Neo4j Netzwerk für Tests"""
    nw = KonzeptNetzwerk(uri="bolt://127.0.0.1:7687", user="neo4j", password="password")
    yield nw
    nw.close()


@pytest.fixture
def embedding_service():
    """Embedding Service für Tests"""
    service = EmbeddingService()
    return service


@pytest.fixture
def meta_config():
    """Test-Konfiguration für MetaLearningEngine"""
    return MetaLearningConfig(
        epsilon=0.2,  # Höhere Exploration für Tests
        min_queries_for_confidence=3,
        persist_every_n_queries=5,
    )


@pytest.fixture
def meta_engine(netzwerk, embedding_service, meta_config):
    """MetaLearningEngine Instanz mit Cleanup"""
    # Cleanup: Lösche alle Test-spezifischen StrategyPerformance nodes
    with netzwerk.driver.session() as session:
        session.run(
            """
            MATCH (sp:StrategyPerformance)
            WHERE sp.strategy_name STARTS WITH 'test_'
               OR sp.strategy_name STARTS WITH 'strategy_'
               OR sp.strategy_name IN ['new_strategy', 'excellent', 'good', 'mediocre', 'bad',
                                      'good_strategy', 'bad_strategy', 'other_strategy', 'another_strategy',
                                      'definition_strategy', 'persist_test_strategy', 'auto_persist_test']
            DELETE sp
        """
        )

    engine = MetaLearningEngine(netzwerk, embedding_service, meta_config)

    yield engine

    # Cleanup nach Test
    with netzwerk.driver.session() as session:
        session.run(
            """
            MATCH (sp:StrategyPerformance)
            WHERE sp.strategy_name STARTS WITH 'test_'
               OR sp.strategy_name STARTS WITH 'strategy_'
               OR sp.strategy_name IN ['new_strategy', 'excellent', 'good', 'mediocre', 'bad',
                                      'good_strategy', 'bad_strategy', 'other_strategy', 'another_strategy',
                                      'definition_strategy', 'persist_test_strategy', 'auto_persist_test']
            DELETE sp
        """
        )


# ============================================================================
# StrategyPerformance Tests
# ============================================================================


class TestStrategyPerformance:
    """Tests für StrategyPerformance Dataclass"""

    def test_strategy_performance_creation(self):
        """Test: StrategyPerformance Objekt erstellen"""
        stats = StrategyPerformance(strategy_name="test_strategy")

        assert stats.strategy_name == "test_strategy"
        assert stats.queries_handled == 0
        assert stats.success_rate == 0.5  # Initial neutral
        assert stats.avg_confidence == 0.0
        assert stats.avg_response_time == 0.0

    def test_update_from_usage_success(self):
        """Test: Update bei erfolgreicher Query"""
        stats = StrategyPerformance(strategy_name="test")

        # First usage - success
        stats.update_from_usage(confidence=0.9, response_time=0.5, success=True)

        assert stats.queries_handled == 1
        assert stats.success_count == 1
        assert stats.failure_count == 0
        assert stats.success_rate > 0.5  # Laplace smoothing
        assert stats.avg_confidence == pytest.approx(0.09, abs=0.01)
        assert stats.avg_response_time == 0.5

    def test_update_from_usage_failure(self):
        """Test: Update bei fehlgeschlagener Query"""
        stats = StrategyPerformance(strategy_name="test")

        # First usage - failure
        stats.update_from_usage(confidence=0.3, response_time=1.2, success=False)

        assert stats.queries_handled == 1
        assert stats.success_count == 0
        assert stats.failure_count == 1
        assert stats.success_rate < 0.5  # Laplace smoothing
        assert stats.avg_confidence == pytest.approx(0.03, abs=0.01)
        assert stats.avg_response_time == 1.2

    def test_update_from_usage_no_feedback(self):
        """Test: Update ohne User-Feedback (success=None)"""
        stats = StrategyPerformance(strategy_name="test")

        stats.update_from_usage(confidence=0.7, response_time=0.8, success=None)

        assert stats.queries_handled == 1
        assert stats.success_count == 0  # Kein Feedback
        assert stats.failure_count == 0
        assert stats.success_rate == 0.5  # Bleibt neutral
        assert stats.avg_confidence == pytest.approx(0.07, abs=0.01)

    def test_update_from_usage_multiple(self):
        """Test: Mehrere Updates - Exponential Moving Average"""
        stats = StrategyPerformance(strategy_name="test")

        # Multiple updates
        for i in range(5):
            stats.update_from_usage(
                confidence=0.8,
                response_time=0.5,
                success=(i % 2 == 0),  # Alternierend success/failure
            )

        assert stats.queries_handled == 5
        assert stats.success_count == 3  # 0, 2, 4
        assert stats.failure_count == 2  # 1, 3

        # Success rate mit Laplace: (3+1)/(5+2) = 4/7 ≈ 0.57
        assert stats.success_rate == pytest.approx(0.57, abs=0.01)

        # Avg confidence sollte gegen 0.8 konvergieren (mit alpha=0.1)
        assert 0.3 < stats.avg_confidence < 0.6


# ============================================================================
# MetaLearningEngine - Initialization
# ============================================================================


class TestMetaLearningEngineInit:
    """Tests für MetaLearningEngine Initialization"""

    def test_engine_initialization(self, meta_engine):
        """Test: Engine wird korrekt initialisiert"""
        assert meta_engine is not None
        assert meta_engine.total_queries == 0
        assert isinstance(meta_engine.strategy_stats, dict)
        assert isinstance(meta_engine.config, MetaLearningConfig)

    def test_load_persisted_stats(self, netzwerk, embedding_service):
        """Test: Lade persistierte Stats aus Neo4j"""
        # Setup: Erstelle Strategy in Neo4j
        with netzwerk.driver.session() as session:
            session.run(
                """
                MERGE (sp:StrategyPerformance {strategy_name: 'test_loaded'})
                SET sp.queries_handled = 10,
                    sp.success_count = 7,
                    sp.failure_count = 3,
                    sp.success_rate = 0.7,
                    sp.avg_confidence = 0.8,
                    sp.avg_response_time = 0.5
            """
            )

        # Create engine - sollte Stats laden
        engine = MetaLearningEngine(netzwerk, embedding_service)

        assert "test_loaded" in engine.strategy_stats
        stats = engine.strategy_stats["test_loaded"]
        assert stats.queries_handled == 10
        assert stats.success_rate == 0.7

        # Cleanup
        with netzwerk.driver.session() as session:
            session.run(
                "MATCH (sp:StrategyPerformance {strategy_name: 'test_loaded'}) DELETE sp"
            )


# ============================================================================
# MetaLearningEngine - Record Strategy Usage
# ============================================================================


class TestRecordStrategyUsage:
    """Tests für record_strategy_usage()"""

    def test_record_new_strategy(self, meta_engine):
        """Test: Record usage für neue Strategy"""
        result = {"confidence": 0.85, "answer": "Test"}

        meta_engine.record_strategy_usage(
            strategy="new_strategy",
            query="Was ist ein Test?",
            result=result,
            response_time=0.5,
            user_feedback="correct",
        )

        assert "new_strategy" in meta_engine.strategy_stats
        stats = meta_engine.strategy_stats["new_strategy"]
        assert stats.queries_handled == 1
        assert stats.success_count == 1

    def test_record_with_correct_feedback(self, meta_engine):
        """Test: Record mit 'correct' Feedback"""
        result = {"confidence": 0.9}

        meta_engine.record_strategy_usage(
            strategy="test_strategy",
            query="Test query",
            result=result,
            response_time=0.3,
            user_feedback="correct",
        )

        stats = meta_engine.strategy_stats["test_strategy"]
        assert stats.success_count == 1
        assert stats.failure_count == 0

    def test_record_with_incorrect_feedback(self, meta_engine):
        """Test: Record mit 'incorrect' Feedback"""
        result = {"confidence": 0.4}

        meta_engine.record_strategy_usage(
            strategy="test_strategy",
            query="Wrong answer query",
            result=result,
            response_time=1.0,
            user_feedback="incorrect",
        )

        stats = meta_engine.strategy_stats["test_strategy"]
        assert stats.success_count == 0
        assert stats.failure_count == 1
        assert len(stats.failure_modes) > 0  # Failure pattern extracted

    def test_record_no_feedback(self, meta_engine):
        """Test: Record ohne User-Feedback"""
        result = {"confidence": 0.7}

        meta_engine.record_strategy_usage(
            strategy="test_strategy",
            query="No feedback query",
            result=result,
            response_time=0.6,
            user_feedback=None,
        )

        stats = meta_engine.strategy_stats["test_strategy"]
        assert stats.queries_handled == 1
        assert stats.success_count == 0  # Kein Feedback
        assert stats.failure_count == 0

    def test_record_updates_total_queries(self, meta_engine):
        """Test: total_queries wird inkrementiert"""
        initial_count = meta_engine.total_queries

        meta_engine.record_strategy_usage(
            strategy="test", query="Test", result={"confidence": 0.5}, response_time=0.5
        )

        assert meta_engine.total_queries == initial_count + 1


# ============================================================================
# MetaLearningEngine - Strategy Selection
# ============================================================================


class TestSelectBestStrategy:
    """Tests für select_best_strategy()"""

    def test_select_from_empty_stats(self, meta_engine):
        """Test: Selection wenn keine Stats vorhanden (Fallback)"""
        meta_engine.strategy_stats = {}  # Clear stats

        strategy, score = meta_engine.select_best_strategy("Test query")

        assert strategy == "direct_answer"  # Fallback
        assert 0.0 <= score <= 1.0

    def test_select_best_performing_strategy(self, meta_engine):
        """Test: Wähle Strategy mit bester Performance"""
        # Setup: Erstelle zwei Strategien mit unterschiedlicher Performance
        meta_engine.config.epsilon = 0.0  # Disable exploration

        # Strategy 1: Good performance
        for i in range(10):
            meta_engine.record_strategy_usage(
                strategy="good_strategy",
                query="Test query",
                result={"confidence": 0.9},
                response_time=0.3,
                user_feedback="correct",
            )

        # Strategy 2: Bad performance
        for i in range(10):
            meta_engine.record_strategy_usage(
                strategy="bad_strategy",
                query="Test query",
                result={"confidence": 0.4},
                response_time=1.5,
                user_feedback="incorrect",
            )

        # Select - sollte good_strategy wählen
        available = ["good_strategy", "bad_strategy"]
        strategy, score = meta_engine.select_best_strategy(
            query="New test query", available_strategies=available
        )

        assert strategy == "good_strategy"
        assert score > 0.5

    def test_epsilon_greedy_exploration(self, meta_engine):
        """Test: Epsilon-Greedy führt zu Exploration"""
        meta_engine.config.epsilon = 1.0  # 100% exploration

        # Setup strategy
        meta_engine.record_strategy_usage(
            strategy="best_strategy",
            query="Test",
            result={"confidence": 0.95},
            response_time=0.2,
            user_feedback="correct",
        )

        # Select multiple times - sollte verschiedene Strategien wählen wegen Exploration
        available = ["best_strategy", "other_strategy", "another_strategy"]

        selections = []
        for _ in range(20):
            strategy, score = meta_engine.select_best_strategy(
                query="Test query", available_strategies=available
            )
            selections.append(strategy)

        # Mit epsilon=1.0 sollten verschiedene Strategien gewählt werden
        unique_selections = set(selections)
        assert len(unique_selections) > 1  # Mindestens 2 verschiedene

    def test_epsilon_decay(self, meta_engine):
        """Test: Epsilon nimmt über Zeit ab"""
        meta_engine.config.epsilon
        meta_engine.config.epsilon = 0.5

        # Multiple selections (Exploitation)
        for _ in range(10):
            meta_engine.select_best_strategy(
                query="Test", available_strategies=["strategy1", "strategy2"]
            )

        # Epsilon sollte gesunken sein
        assert meta_engine.config.epsilon < 0.5

    def test_select_with_context_matching(self, meta_engine):
        """Test: Context-basierte Strategy-Auswahl"""
        meta_engine.config.epsilon = 0.0  # Disable exploration

        # Setup strategies
        meta_engine.record_strategy_usage(
            strategy="temporal_reasoning",
            query="When did X happen?",
            result={"confidence": 0.8},
            response_time=0.5,
            user_feedback="correct",
        )

        meta_engine.record_strategy_usage(
            strategy="probabilistic_reasoning",
            query="How likely is X?",
            result={"confidence": 0.8},
            response_time=0.5,
            user_feedback="correct",
        )

        # Select mit temporal context
        strategy, score = meta_engine.select_best_strategy(
            query="Wann geschah das?",
            context={"temporal_required": True},
            available_strategies=["temporal_reasoning", "probabilistic_reasoning"],
        )

        # Sollte temporal_reasoning wählen wegen Context
        assert strategy == "temporal_reasoning"

        # Select mit uncertainty context
        strategy, score = meta_engine.select_best_strategy(
            query="Wie wahrscheinlich ist das?",
            context={"uncertainty": True},
            available_strategies=["temporal_reasoning", "probabilistic_reasoning"],
        )

        # Sollte probabilistic_reasoning wählen
        assert strategy == "probabilistic_reasoning"


# ============================================================================
# Pattern Matching Tests
# ============================================================================


class TestPatternMatching:
    """Tests für Query-Pattern Matching"""

    def test_pattern_creation(self, meta_engine):
        """Test: Query-Patterns werden erstellt"""
        result = {"confidence": 0.8}

        # Record mehrere ähnliche Queries
        queries = ["Was ist ein Hund?", "Was ist eine Katze?", "Was ist ein Vogel?"]

        for query in queries:
            meta_engine.record_strategy_usage(
                strategy="definition_strategy",
                query=query,
                result=result,
                response_time=0.5,
                user_feedback="correct",
            )

        # Patterns sollten erstellt worden sein
        assert "definition_strategy" in meta_engine.query_patterns
        patterns = meta_engine.query_patterns["definition_strategy"]
        assert len(patterns) > 0

    def test_pattern_matching_similarity(self, meta_engine):
        """Test: Ähnliche Queries matchen selbes Pattern"""
        meta_engine.config.epsilon = 0.0  # Disable exploration

        # Train mit Pattern "Was ist X?"
        for i in range(5):
            meta_engine.record_strategy_usage(
                strategy="definition_strategy",
                query=f"Was ist ein Test{i}?",
                result={"confidence": 0.9},
                response_time=0.3,
                user_feedback="correct",
            )

        # Query ähnlich zu Pattern
        strategy, score = meta_engine.select_best_strategy(
            query="Was ist eine Definition?",
            available_strategies=["definition_strategy", "other_strategy"],
        )

        # Sollte definition_strategy wählen wegen Pattern-Match
        assert strategy == "definition_strategy"


# ============================================================================
# Neo4j Persistence Tests
# ============================================================================


class TestPersistence:
    """Tests für Neo4j Persistence"""

    def test_persist_strategy_stats(self, meta_engine, netzwerk):
        """Test: Strategy-Stats werden in Neo4j persistiert"""
        # Record usage
        for i in range(10):
            meta_engine.record_strategy_usage(
                strategy="persist_test_strategy",
                query=f"Test query {i}",
                result={"confidence": 0.8},
                response_time=0.5,
                user_feedback="correct",
            )

        # Trigger persistence
        meta_engine._persist_all_stats()

        # Verify in Neo4j
        with netzwerk.driver.session() as session:
            result = session.run(
                """
                MATCH (sp:StrategyPerformance {strategy_name: 'persist_test_strategy'})
                RETURN sp.queries_handled AS queries,
                       sp.success_rate AS success_rate
            """
            )
            records = [dict(r) for r in result]

        assert len(records) > 0
        assert records[0]["queries"] == 10

        # Cleanup
        with netzwerk.driver.session() as session:
            session.run(
                """
                MATCH (sp:StrategyPerformance {strategy_name: 'persist_test_strategy'})
                DELETE sp
            """
            )

    def test_auto_persist_after_n_queries(self, netzwerk, embedding_service):
        """Test: Auto-Persistence nach N Queries"""
        config = MetaLearningConfig(persist_every_n_queries=3)
        engine = MetaLearningEngine(netzwerk, embedding_service, config)

        # Record 3 queries - sollte auto-persist triggern
        for i in range(3):
            engine.record_strategy_usage(
                strategy="auto_persist_test",
                query=f"Query {i}",
                result={"confidence": 0.7},
                response_time=0.5,
            )

        # Verify persistence wurde getriggert
        assert engine.queries_since_last_persist == 0  # Reset nach persist

        # Cleanup
        with netzwerk.driver.session() as session:
            session.run(
                """
                MATCH (sp:StrategyPerformance {strategy_name: 'auto_persist_test'})
                DELETE sp
            """
            )


# ============================================================================
# Utility Functions Tests
# ============================================================================


class TestUtilityFunctions:
    """Tests für Utility-Funktionen"""

    def test_cosine_similarity(self, meta_engine):
        """Test: Cosine similarity berechnung"""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]

        similarity = meta_engine._cosine_similarity(vec1, vec2)
        assert similarity == pytest.approx(1.0)

        vec3 = [0.0, 1.0, 0.0]
        similarity = meta_engine._cosine_similarity(vec1, vec3)
        assert similarity == pytest.approx(0.0)

    def test_get_strategy_stats(self, meta_engine):
        """Test: Hole Stats für Strategy"""
        meta_engine.record_strategy_usage(
            strategy="test_get",
            query="Test",
            result={"confidence": 0.8},
            response_time=0.5,
        )

        stats = meta_engine.get_strategy_stats("test_get")
        assert stats is not None
        assert stats.strategy_name == "test_get"

        # Non-existent strategy
        stats = meta_engine.get_strategy_stats("does_not_exist")
        assert stats is None

    def test_get_top_strategies(self, meta_engine):
        """Test: Hole Top N Strategies"""
        # Create strategies mit verschiedener Performance
        strategies = [
            ("excellent", 0.95, "correct", 0.3),  # Fast + high confidence
            ("good", 0.80, "correct", 0.5),
            ("mediocre", 0.60, "neutral", 1.0),
            ("bad", 0.30, "incorrect", 2.0),
        ]

        for name, conf, feedback, response_time in strategies:
            for _ in range(10):  # More samples for stable stats
                meta_engine.record_strategy_usage(
                    strategy=name,
                    query="Test",
                    result={"confidence": conf},
                    response_time=response_time,
                    user_feedback=feedback,
                )

        # Get top 3
        top = meta_engine.get_top_strategies(n=3)

        assert len(top) == 3

        # Top strategies sollten 'excellent' und 'good' enthalten
        top_names = [name for name, score in top]
        assert "excellent" in top_names
        assert "good" in top_names
        # 'bad' sollte NICHT in Top 3 sein
        assert "bad" not in top_names

    def test_reset_epsilon(self, meta_engine):
        """Test: Reset Epsilon"""
        meta_engine.config.epsilon = 0.05

        meta_engine.reset_epsilon(new_epsilon=0.3)

        assert meta_engine.config.epsilon == 0.3


# ============================================================================
# Integration Tests
# ============================================================================


class TestMetaLearningIntegration:
    """Integration Tests"""

    def test_full_learning_cycle(self, meta_engine):
        """Test: Kompletter Learning-Cycle mit mehreren Strategien"""
        # Phase 1: Training mit klarer Performance-Differenz
        strategies_data = [
            (
                "strategy_a",
                "Was ist X?",
                0.95,
                "correct",
                0.3,
            ),  # Best: high conf, fast, correct
            ("strategy_b", "Wie viele X?", 0.75, "correct", 0.7),  # Good but slower
            (
                "strategy_c",
                "Warum X?",
                0.4,
                "incorrect",
                1.5,
            ),  # Bad: low conf, slow, incorrect
        ]

        for strategy, query_pattern, conf, feedback, response_time in strategies_data:
            for i in range(15):  # More samples for stable stats
                meta_engine.record_strategy_usage(
                    strategy=strategy,
                    query=f"{query_pattern} {i}",
                    result={"confidence": conf},
                    response_time=response_time,
                    user_feedback=feedback,
                )

        # Phase 2: Selection
        meta_engine.config.epsilon = 0.0  # Pure exploitation

        # Query ähnlich zu strategy_a pattern
        strategy, score = meta_engine.select_best_strategy(
            query="Was ist ein Baum?",
            available_strategies=["strategy_a", "strategy_b", "strategy_c"],
        )

        # Sollte NICHT die schlechteste Strategy wählen
        assert strategy != "strategy_c"
        # Sollte eine der guten Strategien wählen
        assert strategy in ["strategy_a", "strategy_b"]

        # Phase 3: Verify persistence
        meta_engine._persist_all_stats()

        stats = meta_engine.get_strategy_stats("strategy_a")
        assert stats.queries_handled == 15
        assert stats.success_rate > 0.8

    def test_adaptive_learning_over_time(self, meta_engine):
        """Test: Meta-Learning adaptiert sich über Zeit"""
        # Initially: strategy_old ist gut
        for i in range(10):
            meta_engine.record_strategy_usage(
                strategy="strategy_old",
                query="Test query",
                result={"confidence": 0.9},
                response_time=0.4,
                user_feedback="correct",
            )

        # Later: strategy_new wird besser
        for i in range(15):
            meta_engine.record_strategy_usage(
                strategy="strategy_new",
                query="Test query",
                result={"confidence": 0.95},
                response_time=0.3,
                user_feedback="correct",
            )

        meta_engine.config.epsilon = 0.0

        # Select - sollte jetzt strategy_new bevorzugen
        strategy, score = meta_engine.select_best_strategy(
            query="Test query", available_strategies=["strategy_old", "strategy_new"]
        )

        # strategy_new hat bessere Stats
        assert strategy == "strategy_new"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
