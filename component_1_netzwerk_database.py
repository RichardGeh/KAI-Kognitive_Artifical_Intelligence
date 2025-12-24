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
            self._create_pattern_system_constraints()
            self._create_pattern_system_indexes()

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

    def _create_pattern_system_constraints(self):
        """
        Create uniqueness constraints for pattern discovery system.

        Constraints:
        - Utterance.id: Unique identifier for each utterance
        - Token.id: Unique identifier for each token
        - Pattern.id: Unique identifier for each pattern
        - PatternItem.id: Unique identifier for each pattern item
        - Slot.id: Unique identifier for each slot

        All queries use IF NOT EXISTS for idempotency.
        """
        if not self.driver:
            return

        logger.debug("Erstelle Pattern System Constraints")

        try:
            with self.driver.session(database="neo4j") as session:
                constraints: List[Tuple[str, str]] = [
                    (
                        "UtteranceId",
                        "CREATE CONSTRAINT UtteranceId IF NOT EXISTS "
                        "FOR (u:Utterance) REQUIRE u.id IS UNIQUE",
                    ),
                    (
                        "TokenId",
                        "CREATE CONSTRAINT TokenId IF NOT EXISTS "
                        "FOR (t:Token) REQUIRE t.id IS UNIQUE",
                    ),
                    (
                        "PatternId",
                        "CREATE CONSTRAINT PatternId IF NOT EXISTS "
                        "FOR (p:Pattern) REQUIRE p.id IS UNIQUE",
                    ),
                    (
                        "PatternItemId",
                        "CREATE CONSTRAINT PatternItemId IF NOT EXISTS "
                        "FOR (pi:PatternItem) REQUIRE pi.id IS UNIQUE",
                    ),
                    (
                        "SlotId",
                        "CREATE CONSTRAINT SlotId IF NOT EXISTS "
                        "FOR (s:Slot) REQUIRE s.id IS UNIQUE",
                    ),
                ]

                for constraint_name, query in constraints:
                    try:
                        session.run(query)
                        logger.debug(
                            f"Pattern constraint '{constraint_name}' erstellt/verifiziert"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Pattern constraint '{constraint_name}' konnte nicht erstellt werden",
                            extra={"error": str(e)},
                        )

            logger.info("Pattern System Constraints erfolgreich konfiguriert")
        except Exception:
            logger.error(
                "Fehler beim Konfigurieren der Pattern System Constraints",
                exc_info=True,
            )

    def _create_pattern_system_indexes(self):
        """
        Create performance-critical indexes for pattern discovery system.

        Indexes:
        - utterance_temporal: Composite index (timestamp, archived) for temporal queries
        - token_lemma: Token lemma lookup (anchor-based candidate filtering)
        - token_pos: Token POS lookup (structural matching)
        - token_composite: Composite (lemma, pos) for efficient structural queries
        - pattern_support: Pattern support for ranking
        - pattern_precision: Pattern precision for ranking
        - pattern_type_precision: Composite (type, precision) for filtered Top-N
        - pattern_last_matched: Pattern temporal tracking for cache invalidation
        - slot_allows: Relationship index for slot updates
        - allowed_lemma_value: AllowedLemma node index for MERGE lookup

        Performance Impact:
        - Composite indexes: 20-100x speedup for filtered queries
        - Anchor-based filtering: 20x reduction in pattern matching candidates
        - Single query sequences: Eliminates N+1 problem
        """
        if not self.driver:
            return

        logger.debug("Erstelle Pattern System Performance-Indizes")

        try:
            with self.driver.session(database="neo4j") as session:
                indexes: List[Tuple[str, str]] = [
                    # Composite index for temporal queries (timestamp + archived together)
                    (
                        "utterance_temporal",
                        "CREATE INDEX utterance_temporal IF NOT EXISTS "
                        "FOR (u:Utterance) ON (u.timestamp, u.archived)",
                    ),
                    # Separate index for archived-only queries (addresses Concern 1 from review)
                    (
                        "utterance_archived",
                        "CREATE INDEX utterance_archived IF NOT EXISTS "
                        "FOR (u:Utterance) ON (u.archived)",
                    ),
                    # Token lookup indexes (CRITICAL for anchor-based candidate filtering)
                    (
                        "token_lemma",
                        "CREATE INDEX token_lemma IF NOT EXISTS "
                        "FOR (t:Token) ON (t.lemma)",
                    ),
                    (
                        "token_pos",
                        "CREATE INDEX token_pos IF NOT EXISTS "
                        "FOR (t:Token) ON (t.pos)",
                    ),
                    # Composite index for token sequences (lemma + pos for structural matching)
                    (
                        "token_composite",
                        "CREATE INDEX token_composite IF NOT EXISTS "
                        "FOR (t:Token) ON (t.lemma, t.pos)",
                    ),
                    # Pattern ranking indexes (support + precision for Top-N queries)
                    (
                        "pattern_support",
                        "CREATE INDEX pattern_support IF NOT EXISTS "
                        "FOR (p:Pattern) ON (p.support)",
                    ),
                    (
                        "pattern_precision",
                        "CREATE INDEX pattern_precision IF NOT EXISTS "
                        "FOR (p:Pattern) ON (p.precision)",
                    ),
                    # Composite index for pattern selection (type + precision)
                    (
                        "pattern_type_precision",
                        "CREATE INDEX pattern_type_precision IF NOT EXISTS "
                        "FOR (p:Pattern) ON (p.type, p.precision)",
                    ),
                    # Pattern temporal tracking (lastMatched for cache invalidation)
                    (
                        "pattern_last_matched",
                        "CREATE INDEX pattern_last_matched IF NOT EXISTS "
                        "FOR (p:Pattern) ON (p.lastMatched)",
                    ),
                    # CRITICAL: Relationship index for slot updates
                    (
                        "slot_allows",
                        "CREATE INDEX slot_allows IF NOT EXISTS "
                        "FOR ()-[r:ALLOWS]-() ON (r.count)",
                    ),
                    # AllowedLemma node index for MERGE lookup
                    (
                        "allowed_lemma_value",
                        "CREATE INDEX allowed_lemma_value IF NOT EXISTS "
                        "FOR (al:AllowedLemma) ON (al.value)",
                    ),
                    # UsageContext fragment index for add_usage_context performance
                    (
                        "usagecontext_fragment_index",
                        "CREATE INDEX usagecontext_fragment_index IF NOT EXISTS "
                        "FOR (uc:UsageContext) ON (uc.fragment)",
                    ),
                ]

                for index_name, query in indexes:
                    try:
                        session.run(query)
                        logger.debug(
                            f"Pattern index '{index_name}' erstellt/verifiziert"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Pattern index '{index_name}' konnte nicht erstellt werden: {e}"
                        )

            logger.info("Pattern System Performance-Indizes erfolgreich konfiguriert")

        except Exception:
            logger.error(
                "Fehler beim Konfigurieren der Pattern System Indizes", exc_info=True
            )
