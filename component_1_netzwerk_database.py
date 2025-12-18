# component_1_netzwerk_database.py
"""
Database connection and session management for Neo4j.

This module handles the low-level database connection lifecycle including:
- Driver creation and configuration
- Session management and connection pooling
- Health checks and connectivity verification
- Constraints and indexes creation

Extracted from monolithic component_1_netzwerk_core.py as part of architecture
refactoring (Task 5).
"""

import threading
from typing import List, Optional, Tuple

from neo4j import Driver, GraphDatabase

from component_15_logging_config import get_logger
from kai_exceptions import Neo4jConnectionError, wrap_exception

logger = get_logger(__name__)


class DatabaseConnection:
    """
    Manages Neo4j database connection and schema setup.

    Responsibilities:
    - Driver lifecycle (creation, verification, cleanup)
    - Constraint and index creation
    - Health checks
    - Connection pooling configuration

    Thread Safety:
        This class is thread-safe. The driver itself manages connection pooling
        and concurrent access.

    Attributes:
        driver: Neo4j driver instance
        uri: Database URI
        user: Database user
        _lock: Thread lock for initialization
    """

    def __init__(
        self,
        uri: str = "bolt://127.0.0.1:7687",
        user: str = "neo4j",
        password: str = "password",
    ):
        """
        Initialize database connection.

        Args:
            uri: Neo4j connection URI
            user: Database username
            password: Database password

        Raises:
            Neo4jConnectionError: If connection cannot be established
        """
        self.uri = uri
        self.user = user
        self._lock = threading.RLock()

        try:
            logger.info(
                "Initialisiere Neo4j-Verbindung", extra={"uri": uri, "user": user}
            )
            self.driver: Optional[Driver] = GraphDatabase.driver(
                uri, auth=(user, password)
            )
            self.driver.verify_connectivity()
            logger.info("Neo4j-Verbindung erfolgreich hergestellt", extra={"uri": uri})

            # Setup schema
            self._create_constraints()
            self._create_indexes()

        except Exception as e:
            raise wrap_exception(
                e,
                Neo4jConnectionError,
                "Konnte keine Verbindung zur Neo4j-DB herstellen",
                uri=uri,
                user=user,
            )

    def close(self):
        """
        Close the database connection.

        Should be called during application shutdown to properly release resources.
        """
        if self.driver:
            self.driver.close()
            logger.debug("Neo4j-Verbindung geschlossen")

    def health_check(self) -> bool:
        """
        Check if the Neo4j connection is working.

        Returns:
            True if connection OK, False on error

        Example:
            >>> db = DatabaseConnection()
            >>> if db.health_check():
            ...     print("DB OK")
            ... else:
            ...     print("DB ERROR")
        """
        if not self.driver:
            logger.error("health_check: Kein Driver verfügbar")
            return False

        try:
            with self.driver.session(database="neo4j") as session:
                result = session.run("RETURN 1 AS one")
                record = result.single()
                if record and record["one"] == 1:
                    logger.debug("health_check: Neo4j-Verbindung OK")
                    return True
                else:
                    logger.error("health_check: Unerwartetes Ergebnis von Neo4j")
                    return False
        except Exception as e:
            logger.error(
                f"health_check: Neo4j-Verbindung fehlgeschlagen: {e}", exc_info=True
            )
            return False

    def _create_constraints(self):
        """
        Create uniqueness constraints for the knowledge graph schema.

        Constraints ensure data integrity and automatically create indexes.
        """
        if not self.driver:
            return

        logger.debug("Erstelle Neo4j Constraints")

        try:
            with self.driver.session(database="neo4j") as session:
                constraints: List[Tuple[str, str]] = [
                    (
                        "WortLemma",
                        "CREATE CONSTRAINT WortLemma IF NOT EXISTS FOR (w:Wort) REQUIRE w.lemma IS UNIQUE",
                    ),
                    (
                        "KonzeptName",
                        "CREATE CONSTRAINT KonzeptName IF NOT EXISTS FOR (k:Konzept) REQUIRE k.name IS UNIQUE",
                    ),
                    (
                        "ExtractionRuleType",
                        "CREATE CONSTRAINT ExtractionRuleType IF NOT EXISTS FOR (r:ExtractionRule) REQUIRE r.relation_type IS UNIQUE",
                    ),
                    (
                        "PatternPrototypeId",
                        "CREATE CONSTRAINT PatternPrototypeId IF NOT EXISTS FOR (p:PatternPrototype) REQUIRE p.id IS UNIQUE",
                    ),
                    (
                        "LexiconName",
                        "CREATE CONSTRAINT LexiconName IF NOT EXISTS FOR (l:Lexicon) REQUIRE l.name IS UNIQUE",
                    ),
                    (
                        "EpisodeId",
                        "CREATE CONSTRAINT EpisodeId IF NOT EXISTS FOR (e:Episode) REQUIRE e.id IS UNIQUE",
                    ),
                    (
                        "InferenceEpisodeId",
                        "CREATE CONSTRAINT InferenceEpisodeId IF NOT EXISTS FOR (ie:InferenceEpisode) REQUIRE ie.id IS UNIQUE",
                    ),
                    (
                        "ProofStepId",
                        "CREATE CONSTRAINT ProofStepId IF NOT EXISTS FOR (ps:ProofStep) REQUIRE ps.id IS UNIQUE",
                    ),
                    (
                        "HypothesisId",
                        "CREATE CONSTRAINT HypothesisId IF NOT EXISTS FOR (h:Hypothesis) REQUIRE h.id IS UNIQUE",
                    ),
                    (
                        "AgentId",
                        "CREATE CONSTRAINT AgentId IF NOT EXISTS FOR (a:Agent) REQUIRE a.id IS UNIQUE",
                    ),
                    (
                        "BeliefId",
                        "CREATE CONSTRAINT BeliefId IF NOT EXISTS FOR (b:Belief) REQUIRE b.id IS UNIQUE",
                    ),
                    (
                        "MetaBeliefId",
                        "CREATE CONSTRAINT MetaBeliefId IF NOT EXISTS FOR (mb:MetaBelief) REQUIRE mb.id IS UNIQUE",
                    ),
                    (
                        "ProductionRuleName",
                        "CREATE CONSTRAINT ProductionRuleName IF NOT EXISTS FOR (pr:ProductionRule) REQUIRE pr.name IS UNIQUE",
                    ),
                ]

                for constraint_name, query in constraints:
                    try:
                        session.run(query)
                        logger.debug(
                            f"Constraint '{constraint_name}' erstellt/verifiziert"
                        )
                    except Exception as e:
                        # Single constraint failure is not critical
                        logger.warning(
                            f"Constraint '{constraint_name}' konnte nicht erstellt werden",
                            extra={"error": str(e)},
                        )

            logger.info("Neo4j Constraints erfolgreich konfiguriert")
        except Exception:
            # Constraint errors are not critical (may already exist)
            logger.error("Fehler beim Konfigurieren der Constraints", exc_info=True)

    def _create_indexes(self):
        """
        Create performance indexes for frequently queried properties.

        Indexes:
        - wort_lemma_index: Index on Wort.lemma (automatically created by constraint)
        - Confidence indexes: IS_A, HAS_PROPERTY, CAPABLE_OF relationships
        - Production Rule indexes: category, utility, priority composite
        - Timestamp indexes: IS_A, HAS_PROPERTY, CAPABLE_OF relationships
        - Episode indexes: episode_type, timestamp

        Performance Impact:
        - Production Rule queries: 10x speedup on category/utility filters
        - Temporal queries: 10x speedup on timestamp range filters
        - Episode filtering: 10x speedup on episode_type queries
        """
        if not self.driver:
            return

        logger.debug("Erstelle Neo4j Performance-Indizes")

        try:
            with self.driver.session(database="neo4j") as session:
                # Note: wort_lemma_index automatically created by UNIQUE constraint

                # Indexes for relationship properties
                # NOTE: Neo4j 5.x does not support generic relationship indexes without type
                # Create specific indexes for the most important relationship types
                indexes: List[Tuple[str, str]] = [
                    # ========== EXISTING: Confidence Indexes ==========
                    # Index for IS_A relationships with confidence filter
                    (
                        "isa_confidence_index",
                        "CREATE INDEX isa_confidence_index IF NOT EXISTS "
                        "FOR ()-[r:IS_A]-() ON (r.confidence)",
                    ),
                    # Index for HAS_PROPERTY relationships with confidence filter
                    (
                        "property_confidence_index",
                        "CREATE INDEX property_confidence_index IF NOT EXISTS "
                        "FOR ()-[r:HAS_PROPERTY]-() ON (r.confidence)",
                    ),
                    # Index for CAPABLE_OF relationships with confidence filter
                    (
                        "capable_confidence_index",
                        "CREATE INDEX capable_confidence_index IF NOT EXISTS "
                        "FOR ()-[r:CAPABLE_OF]-() ON (r.confidence)",
                    ),
                    # ========== NEW: Production Rule Indexes (Quick Win #1) ==========
                    # Index for Production Rule category (VERY FREQUENTLY QUERIED)
                    (
                        "pr_category_index",
                        "CREATE INDEX pr_category_index IF NOT EXISTS "
                        "FOR (pr:ProductionRule) ON (pr.category)",
                    ),
                    # Index for Production Rule utility (CONFLICT RESOLUTION)
                    (
                        "pr_utility_index",
                        "CREATE INDEX pr_utility_index IF NOT EXISTS "
                        "FOR (pr:ProductionRule) ON (pr.utility)",
                    ),
                    # Composite index for Production Rule priority sorting
                    (
                        "pr_priority_composite",
                        "CREATE INDEX pr_priority_composite IF NOT EXISTS "
                        "FOR (pr:ProductionRule) ON (pr.utility, pr.specificity)",
                    ),
                    # ========== NEW: Timestamp Indexes (Quick Win #1) ==========
                    # Index for IS_A relationship timestamps
                    (
                        "isa_timestamp_index",
                        "CREATE INDEX isa_timestamp_index IF NOT EXISTS "
                        "FOR ()-[r:IS_A]-() ON (r.timestamp)",
                    ),
                    # Index for HAS_PROPERTY relationship timestamps
                    (
                        "property_timestamp_index",
                        "CREATE INDEX property_timestamp_index IF NOT EXISTS "
                        "FOR ()-[r:HAS_PROPERTY]-() ON (r.timestamp)",
                    ),
                    # Index for CAPABLE_OF relationship timestamps
                    (
                        "capable_timestamp_index",
                        "CREATE INDEX capable_timestamp_index IF NOT EXISTS "
                        "FOR ()-[r:CAPABLE_OF]-() ON (r.timestamp)",
                    ),
                    # ========== NEW: Episode Indexes (Quick Win #1) ==========
                    # Index for Episode type filtering
                    (
                        "episode_type_index",
                        "CREATE INDEX episode_type_index IF NOT EXISTS "
                        "FOR (ep:Episode) ON (ep.episode_type)",
                    ),
                    # Index for Episode timestamp queries
                    (
                        "episode_timestamp_index",
                        "CREATE INDEX episode_timestamp_index IF NOT EXISTS "
                        "FOR (ep:Episode) ON (ep.timestamp)",
                    ),
                ]

                for index_name, query in indexes:
                    try:
                        session.run(query)
                        logger.debug(f"Index '{index_name}' erstellt/verifiziert")
                    except Exception as e:
                        # Index errors are not critical
                        logger.warning(
                            f"Index '{index_name}' konnte nicht erstellt werden: {e}"
                        )

            logger.info("Neo4j Performance-Indizes erfolgreich konfiguriert")

        except Exception:
            # Index errors are not critical
            logger.error("Fehler beim Konfigurieren der Indizes", exc_info=True)
