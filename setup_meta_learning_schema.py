"""
setup_meta_learning_schema.py

Neo4j Schema Setup für Meta-Learning Layer

Erstellt:
- StrategyPerformance nodes
- QueryPattern nodes
- Constraints und Indizes

Author: KAI Development Team
Last Updated: 2025-11-08
"""

from component_1_netzwerk import KonzeptNetzwerk
from component_15_logging_config import get_logger

logger = get_logger(__name__)


def setup_meta_learning_schema(netzwerk: KonzeptNetzwerk) -> None:
    """
    Erstelle Neo4j Schema für Meta-Learning

    Nodes:
    - StrategyPerformance: Performance-Statistiken pro Strategy
    - QueryPattern: Query-Patterns mit Embeddings
    - StrategyUsageEpisode: Einzelne Usage-Events (optional, für detaillierte Analyse)

    Relationships:
    - (StrategyPerformance)-[:HAS_PATTERN]->(QueryPattern)
    - (StrategyUsageEpisode)-[:USED_STRATEGY]->(StrategyPerformance)
    """

    try:
        logger.info("Setting up Meta-Learning Neo4j schema...")

        # ====================================================================
        # Constraints (Unique Identifiers)
        # ====================================================================

        # StrategyPerformance: strategy_name ist unique
        constraint_queries = [
            """
            CREATE CONSTRAINT strategy_performance_name IF NOT EXISTS
            FOR (sp:StrategyPerformance)
            REQUIRE sp.strategy_name IS UNIQUE
            """,
            # QueryPattern: pattern_id ist unique
            """
            CREATE CONSTRAINT query_pattern_id IF NOT EXISTS
            FOR (qp:QueryPattern)
            REQUIRE qp.pattern_id IS UNIQUE
            """,
            # StrategyUsageEpisode: episode_id ist unique
            """
            CREATE CONSTRAINT strategy_usage_episode_id IF NOT EXISTS
            FOR (sue:StrategyUsageEpisode)
            REQUIRE sue.episode_id IS UNIQUE
            """,
        ]

        for query in constraint_queries:
            try:
                netzwerk.query(query)
                logger.debug("Created constraint: %s", query.split("\n")[1].strip())
            except Exception as e:
                # Constraint might already exist
                logger.debug("Constraint already exists or error: %s", e)

        # ====================================================================
        # Indizes (Performance)
        # ====================================================================

        index_queries = [
            # StrategyPerformance: Index für Queries und Success Rate
            """
            CREATE INDEX strategy_perf_queries IF NOT EXISTS
            FOR (sp:StrategyPerformance)
            ON (sp.queries_handled)
            """,
            """
            CREATE INDEX strategy_perf_success_rate IF NOT EXISTS
            FOR (sp:StrategyPerformance)
            ON (sp.success_rate)
            """,
            """
            CREATE INDEX strategy_perf_last_used IF NOT EXISTS
            FOR (sp:StrategyPerformance)
            ON (sp.last_used)
            """,
            # QueryPattern: Index für Strategy
            """
            CREATE INDEX query_pattern_strategy IF NOT EXISTS
            FOR (qp:QueryPattern)
            ON (qp.associated_strategy)
            """,
            """
            CREATE INDEX query_pattern_success IF NOT EXISTS
            FOR (qp:QueryPattern)
            ON (qp.success_rate)
            """,
            # StrategyUsageEpisode: Index für Timestamp
            """
            CREATE INDEX strategy_usage_timestamp IF NOT EXISTS
            FOR (sue:StrategyUsageEpisode)
            ON (sue.timestamp)
            """,
            """
            CREATE INDEX strategy_usage_strategy IF NOT EXISTS
            FOR (sue:StrategyUsageEpisode)
            ON (sue.strategy_name)
            """,
        ]

        for query in index_queries:
            try:
                netzwerk.query(query)
                logger.debug("Created index: %s", query.split("\n")[1].strip())
            except Exception as e:
                logger.debug("Index already exists or error: %s", e)

        # ====================================================================
        # Initial Strategy Nodes (Optional: Pre-populate known strategies)
        # ====================================================================

        known_strategies = [
            "direct_answer",
            "logic_engine",
            "graph_traversal",
            "abductive_reasoning",
            "probabilistic_reasoning",
            "combinatorial_reasoning",
            "constraint_reasoning",
            # Zukünftige Strategien:
            "temporal_reasoning",
            "causal_reasoning",
            "analogical_reasoning",
        ]

        for strategy in known_strategies:
            query = """
            MERGE (sp:StrategyPerformance {strategy_name: $strategy_name})
            ON CREATE SET
                sp.queries_handled = 0,
                sp.success_count = 0,
                sp.failure_count = 0,
                sp.success_rate = 0.5,
                sp.avg_confidence = 0.0,
                sp.avg_response_time = 0.0,
                sp.failure_modes = [],
                sp.created_at = datetime(),
                sp.updated_at = datetime()
            RETURN sp
            """

            netzwerk.query(query, {"strategy_name": strategy})

        logger.info(
            "Pre-populated %d strategy performance nodes", len(known_strategies)
        )

        # ====================================================================
        # Verification
        # ====================================================================

        # Count nodes
        count_query = """
        MATCH (sp:StrategyPerformance)
        RETURN count(sp) AS count
        """
        result = netzwerk.query(count_query)
        count = result[0]["count"] if result else 0

        logger.info("✓ Meta-Learning schema setup complete!")
        logger.info("  - StrategyPerformance nodes: %d", count)
        logger.info("  - Constraints: 3 created")
        logger.info("  - Indizes: 7 created")

    except Exception as e:
        logger.error("Error setting up Meta-Learning schema: %s", e, exc_info=True)
        raise


if __name__ == "__main__":
    # Standalone execution
    import logging

    from component_15_logging_config import setup_logging

    setup_logging(console_level=logging.DEBUG)

    netzwerk = KonzeptNetzwerk(
        uri="bolt://127.0.0.1:7687", user="neo4j", password="password"
    )

    try:
        setup_meta_learning_schema(netzwerk)
        print("\n✓ Meta-Learning Schema erfolgreich erstellt!")

    except Exception as e:
        print(f"\n✗ Fehler beim Setup: {e}")

    finally:
        netzwerk.close()
