# component_1_netzwerk_patterns.py
"""
Pattern and Slot management for pattern discovery system.

This module handles operations related to learned patterns and their components:
- Pattern creation and statistics updates
- PatternItem creation (LITERAL and SLOT types)
- Slot creation and allowed value management
- Pattern-utterance matching recording
- Batch operations for efficiency

Extracted as part of Pattern Discovery System implementation (Part 1).
"""

import re
import threading
import uuid
from typing import Any, Dict, List, Optional

from neo4j import Driver

from component_15_logging_config import get_logger
from kai_exceptions import DatabaseException, wrap_exception

logger = get_logger(__name__)

# Valid pattern types (security - prevent injection)
VALID_PATTERN_TYPES = {"seed", "learned", "template"}

# Valid PatternItem kinds
VALID_ITEM_KINDS = {"LITERAL", "SLOT"}

# Entity name validation regex (from CLAUDE.md)
ENTITY_NAME_REGEX = re.compile(r"^[a-z_][a-z0-9_]{0,63}$")


class PatternManager:
    """
    Manages pattern and slot CRUD operations in Neo4j.

    Responsibilities:
    - Create/update pattern nodes
    - Create pattern items (literals and slots)
    - Manage slot allowed values
    - Record pattern-utterance matches
    - Batch operations for efficiency

    Thread Safety:
        This class is thread-safe. All database operations use parameterized
        queries to prevent injection.

    Attributes:
        driver: Neo4j driver instance
        _lock: Thread lock for critical operations
        _pattern_matcher: Optional pattern matcher for cache invalidation
    """

    def __init__(self, driver: Driver, pattern_matcher=None):
        if not driver:
            raise ValueError("Driver cannot be None")

        self.driver = driver
        self._lock = threading.RLock()
        self._pattern_matcher = pattern_matcher

        logger.debug("PatternManager initialisiert")

    def _validate_lemma(self, lemma: str) -> str:
        """
        Validate and sanitize lemma for AllowedLemma node (SECURITY).

        Args:
            lemma: Raw lemma value from user input

        Returns:
            Sanitized lemma value

        Raises:
            ValueError: If lemma is invalid or too long

        Security:
            - Prevents DOS via extremely long values
            - Ensures lemma matches entity name regex
            - Normalizes to lowercase alphanumeric + underscore
        """
        if not lemma or not lemma.strip():
            raise ValueError("Lemma cannot be empty")

        if len(lemma) > 63:
            raise ValueError(f"Lemma must be <= 63 chars, got: {len(lemma)}")

        # Normalize to lowercase alphanumeric + underscore
        sanitized = re.sub(r"[^a-z0-9_]", "", lemma.lower())

        if not sanitized:
            raise ValueError(
                f"Invalid lemma (no valid chars after sanitization): {lemma}"
            )

        if not sanitized[0].isalpha() and sanitized[0] != "_":
            # Try fixing: add underscore prefix if starts with digit
            if sanitized[0].isdigit():
                sanitized = "_" + sanitized[:62]
            else:
                raise ValueError(
                    f"Invalid lemma (must start with letter or underscore): {lemma}"
                )

        if not ENTITY_NAME_REGEX.match(sanitized):
            raise ValueError(
                f"Invalid lemma (does not match entity name regex): {lemma}"
            )

        return sanitized

    def create_pattern(
        self, name: str, pattern_type: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create Pattern node atomically with parameterized query."""
        if not name or not name.strip():
            raise ValueError("Pattern name cannot be empty")
        if pattern_type not in VALID_PATTERN_TYPES:
            raise ValueError(f"Pattern type must be one of {VALID_PATTERN_TYPES}")

        pattern_id = str(uuid.uuid4())
        query = """
        CREATE (p:Pattern {
            id: $id, name: $name, type: $type,
            support: $support, precision: $precision,
            createdAt: datetime(), lastMatched: $lastMatched
        })
        RETURN p.id AS id
        """

        try:
            with self.driver.session(database="neo4j") as session:
                result = session.run(
                    query,
                    {
                        "id": pattern_id,
                        "name": name,
                        "type": pattern_type,
                        "support": 0,
                        "precision": 0.5,
                        "lastMatched": None,
                    },
                )
                created_id = result.single()["id"]

                verify_query = "MATCH (p:Pattern {id: $id}) RETURN p"
                verification = session.run(verify_query, {"id": pattern_id})
                if not verification.single():
                    raise DatabaseException(f"Failed to create pattern: {pattern_id}")

                logger.debug(f"Pattern created: {name} (type={pattern_type})")

                # CRITICAL: Invalidate matcher cache to ensure new pattern appears in candidates
                if self._pattern_matcher is not None:
                    self._pattern_matcher._invalidate_candidate_cache()

                return created_id
        except Exception as e:
            raise wrap_exception(
                e, DatabaseException, "Failed to create pattern", name=name
            )

    def update_pattern_stats(
        self, pattern_id: str, support_increment: int, new_precision: float
    ):
        """Update pattern statistics atomically."""
        if not 0.0 <= new_precision <= 1.0:
            raise ValueError(f"Precision must be 0.0-1.0, got: {new_precision}")
        if support_increment < 0:
            raise ValueError(f"Support increment must be non-negative")

        query = """
        MATCH (p:Pattern {id: $id})
        SET p.support = p.support + $increment,
            p.precision = $precision, p.lastMatched = datetime()
        RETURN p.support AS support, p.precision AS precision
        """

        try:
            with self.driver.session(database="neo4j") as session:
                result = session.run(
                    query,
                    {
                        "id": pattern_id,
                        "increment": support_increment,
                        "precision": new_precision,
                    },
                )
                record = result.single()
                if record:
                    logger.debug(
                        f"Pattern stats updated: support={record['support']}, precision={record['precision']:.2f}"
                    )

                    # Invalidate matcher cache after stats update (precision affects ordering)
                    if self._pattern_matcher is not None:
                        self._pattern_matcher._invalidate_candidate_cache()
                else:
                    logger.warning(f"Pattern not found for stats update: {pattern_id}")
        except Exception as e:
            raise wrap_exception(
                e,
                DatabaseException,
                "Failed to update pattern stats",
                pattern_id=pattern_id,
            )

    def get_all_patterns(
        self, type_filter: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Retrieve patterns with optional type filter using composite index."""
        if limit <= 0 or limit > 10000:
            raise ValueError(f"Limit must be 1-10000, got: {limit}")
        if type_filter and type_filter not in VALID_PATTERN_TYPES:
            raise ValueError(f"Invalid type filter: {type_filter}")

        if type_filter:
            query = """
            MATCH (p:Pattern) WHERE p.type = $type
            RETURN p ORDER BY p.precision DESC, p.support DESC LIMIT $limit
            """
            params = {"type": type_filter, "limit": limit}
        else:
            query = """
            MATCH (p:Pattern)
            RETURN p ORDER BY p.precision DESC, p.support DESC LIMIT $limit
            """
            params = {"limit": limit}

        try:
            with self.driver.session(database="neo4j") as session:
                result = session.run(query, params)
                patterns = [dict(record["p"]) for record in result]
                logger.debug(
                    f"Retrieved {len(patterns)} patterns",
                    extra={"type_filter": type_filter},
                )
                return patterns
        except Exception as e:
            raise wrap_exception(
                e,
                DatabaseException,
                "Failed to retrieve patterns",
                type_filter=type_filter,
            )

    def batch_create_pattern_items(
        self, pattern_id: str, items: List[Dict[str, Any]]
    ) -> int:
        """Create multiple PatternItems using UNWIND batch operation."""
        if not items:
            return 0

        for item in items:
            if "id" not in item or "idx" not in item or "kind" not in item:
                raise ValueError("Each item must have 'id', 'idx', and 'kind' fields")
            if item["kind"] not in VALID_ITEM_KINDS:
                raise ValueError(f"Invalid item kind: {item['kind']}")
            if item["kind"] == "LITERAL" and "literalValue" not in item:
                raise ValueError("LITERAL items must have 'literalValue' field")
            if item["kind"] == "SLOT" and "slotId" not in item:
                raise ValueError("SLOT items must have 'slotId' field")

        query = """
        MATCH (p:Pattern {id: $pattern_id})
        UNWIND $items AS item
        CREATE (pi:PatternItem {id: item.id, idx: item.idx, kind: item.kind})
        CREATE (p)-[:HAS_ITEM {idx: item.idx}]->(pi)
        WITH pi, item
        FOREACH (_ IN CASE WHEN item.kind = 'LITERAL' THEN [1] ELSE [] END |
            SET pi.literalValue = item.literalValue
        )
        WITH pi, item
        FOREACH (_ IN CASE WHEN item.kind = 'SLOT' THEN [1] ELSE [] END |
            MERGE (s:Slot {id: item.slotId})
        )
        WITH pi, item
        OPTIONAL MATCH (s:Slot {id: item.slotId})
        WHERE item.kind = 'SLOT'
        FOREACH (_ IN CASE WHEN s IS NOT NULL THEN [1] ELSE [] END |
            CREATE (pi)-[:USES_SLOT]->(s)
        )
        RETURN count(DISTINCT pi) AS created_count
        """

        try:
            with self.driver.session(database="neo4j") as session:
                with session.begin_transaction() as tx:
                    try:
                        result = tx.run(
                            query, {"pattern_id": pattern_id, "items": items}
                        )
                        count = result.single()["created_count"]
                        tx.commit()
                        logger.debug(f"Batch created {count} pattern items")
                        return count
                    except Exception as inner_e:
                        tx.rollback()
                        raise inner_e
        except Exception as e:
            raise wrap_exception(
                e,
                DatabaseException,
                "Failed to batch create pattern items",
                pattern_id=pattern_id,
                item_count=len(items),
            )

    def create_slot(
        self,
        slot_type: str,
        allowed_values: Optional[List[str]] = None,
        min_count: int = 1,
        max_count: int = 1,
    ) -> str:
        """Create Slot node with initial allowed values using UNWIND."""
        if not slot_type or not slot_type.strip():
            raise ValueError("Slot type cannot be empty")
        if min_count < 1:
            raise ValueError(f"Min count must be >= 1, got: {min_count}")
        if max_count < -1 or max_count == 0:
            raise ValueError(f"Max count must be -1 (unbounded) or >= 1")
        if max_count > 0 and min_count > max_count:
            raise ValueError(f"Min count cannot exceed max count")

        slot_id = str(uuid.uuid4())

        if not allowed_values:
            query = """
            CREATE (s:Slot {id: $id, slotType: $type, minCount: $min_count, maxCount: $max_count})
            RETURN s.id AS id
            """
            params = {
                "id": slot_id,
                "type": slot_type,
                "min_count": min_count,
                "max_count": max_count,
            }
        else:
            # SECURITY: Validate and sanitize all allowed values
            sanitized_allowed = [self._validate_lemma(v) for v in allowed_values]

            query = """
            CREATE (s:Slot {id: $id, slotType: $type, minCount: $min_count, maxCount: $max_count})
            WITH s
            UNWIND $allowed AS lemma
            MERGE (al:AllowedLemma {lemma: lemma})
            CREATE (s)-[:ALLOWS {count: 1}]->(al)
            RETURN s.id AS id
            """
            params = {
                "id": slot_id,
                "type": slot_type,
                "min_count": min_count,
                "max_count": max_count,
                "allowed": sanitized_allowed,
            }

        try:
            with self.driver.session(database="neo4j") as session:
                with session.begin_transaction() as tx:
                    try:
                        result = tx.run(query, params)
                        created_id = result.single()["id"]
                        tx.commit()
                        logger.debug(
                            f"Slot created: {slot_type} (allowed_values={len(allowed_values) if allowed_values else 0})"
                        )
                        return created_id
                    except Exception as inner_e:
                        tx.rollback()
                        raise inner_e
        except Exception as e:
            raise wrap_exception(
                e, DatabaseException, "Failed to create slot", slot_type=slot_type
            )

    def update_slot_allowed(self, slot_id: str, lemma: str, count_increment: int = 1):
        """Update or create ALLOWS relationship with count using MERGE."""
        if not lemma or not lemma.strip():
            raise ValueError("Lemma cannot be empty")
        if count_increment < 1:
            raise ValueError(f"Count increment must be >= 1")

        # SECURITY: Validate and sanitize lemma before database operation
        sanitized_lemma = self._validate_lemma(lemma)

        query = """
        MATCH (s:Slot {id: $slot_id})
        MERGE (al:AllowedLemma {lemma: $lemma})
        MERGE (s)-[r:ALLOWS]->(al)
        ON CREATE SET r.count = $increment
        ON MATCH SET r.count = r.count + $increment
        RETURN r.count AS new_count
        """

        try:
            with self.driver.session(database="neo4j") as session:
                result = session.run(
                    query,
                    {
                        "slot_id": slot_id,
                        "lemma": sanitized_lemma,
                        "increment": count_increment,
                    },
                )
                record = result.single()
                if record:
                    logger.debug(
                        f"Slot allowed updated: {lemma} (count={record['new_count']})"
                    )
                else:
                    logger.warning(f"Slot not found for allowed update: {slot_id}")
        except Exception as e:
            raise wrap_exception(
                e,
                DatabaseException,
                "Failed to update slot allowed",
                slot_id=slot_id,
                lemma=lemma,
            )

    def match_utterance_to_pattern(
        self, utterance_id: str, pattern_id: str, score: float
    ):
        """Record MATCHED relationship between Utterance and Pattern."""
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"Score must be 0.0-1.0, got: {score}")

        query = """
        MATCH (u:Utterance {id: $utterance_id})
        MATCH (p:Pattern {id: $pattern_id})
        CREATE (u)-[:MATCHED {score: $score, timestamp: datetime()}]->(p)
        """

        try:
            with self.driver.session(database="neo4j") as session:
                session.run(
                    query,
                    {
                        "utterance_id": utterance_id,
                        "pattern_id": pattern_id,
                        "score": score,
                    },
                )
                logger.debug(
                    f"Match recorded: utterance {utterance_id[:8]}... -> pattern {pattern_id[:8]}... (score={score:.2f})"
                )
        except Exception as e:
            raise wrap_exception(
                e,
                DatabaseException,
                "Failed to match utterance to pattern",
                utterance_id=utterance_id,
                pattern_id=pattern_id,
            )

    def update_pattern_centroid(self, pattern_id: str, centroid: List[float]):
        """
        Update pattern embedding centroid.

        The centroid is the normalized average embedding of all matched utterances.
        Used for semantic similarity scoring in hybrid pattern matching.

        Args:
            pattern_id: Pattern UUID
            centroid: 384-dimensional normalized embedding vector

        Raises:
            ValueError: If centroid is invalid
            DatabaseException: If database operation fails

        Example:
            >>> centroid = [0.1, 0.2, ...]  # 384D normalized vector
            >>> manager.update_pattern_centroid(pattern_id, centroid)
        """
        if not centroid or len(centroid) != 384:
            raise ValueError(
                f"Centroid must be 384-dimensional, got: {len(centroid) if centroid else 0}"
            )

        # Verify normalization (should be close to 1.0)
        import numpy as np

        norm = np.linalg.norm(centroid)
        if not 0.95 <= norm <= 1.05:
            logger.warning(f"Centroid norm {norm:.3f} not normalized, normalizing...")
            if norm > 0:
                centroid = (np.array(centroid) / norm).tolist()

        query = """
        MATCH (p:Pattern {id: $pattern_id})
        SET p.centroid = $centroid
        RETURN p.id AS id
        """

        try:
            with self.driver.session(database="neo4j") as session:
                result = session.run(
                    query, {"pattern_id": pattern_id, "centroid": centroid}
                )
                record = result.single()
                if record:
                    logger.debug(f"Pattern centroid updated: {pattern_id[:8]}")
                else:
                    logger.warning(
                        f"Pattern not found for centroid update: {pattern_id[:8]}"
                    )
        except Exception as e:
            raise wrap_exception(
                e,
                DatabaseException,
                "Failed to update pattern centroid",
                pattern_id=pattern_id,
            )

    def get_pattern(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve single pattern with all properties.

        Args:
            pattern_id: Pattern UUID

        Returns:
            Pattern dict with all properties or None if not found

        Raises:
            DatabaseException: If database operation fails

        Example:
            >>> pattern = manager.get_pattern(pattern_id)
            >>> if pattern:
            ...     print(f"Pattern: {pattern['name']}, Precision: {pattern['precision']}")
        """
        query = """
        MATCH (p:Pattern {id: $pattern_id})
        RETURN p
        """

        try:
            with self.driver.session(database="neo4j") as session:
                result = session.run(query, {"pattern_id": pattern_id})
                record = result.single()
                return dict(record["p"]) if record else None
        except Exception as e:
            raise wrap_exception(
                e,
                DatabaseException,
                "Failed to retrieve pattern",
                pattern_id=pattern_id,
            )

    def get_pattern_items(self, pattern_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve pattern items ordered by index.

        Returns items with their properties including:
        - id: PatternItem UUID
        - idx: Position in pattern
        - kind: "LITERAL" or "SLOT"
        - literalValue: For LITERAL items
        - slotId: For SLOT items

        Args:
            pattern_id: Pattern UUID

        Returns:
            List of PatternItem dicts in sequence order

        Raises:
            DatabaseException: If database operation fails

        Example:
            >>> items = manager.get_pattern_items(pattern_id)
            >>> for item in items:
            ...     if item['kind'] == 'LITERAL':
            ...         print(f"Literal: {item['literalValue']}")
            ...     else:
            ...         print(f"Slot: {item['slotId']}")
        """
        query = """
        MATCH (p:Pattern {id: $pattern_id})-[r:HAS_ITEM]->(pi:PatternItem)
        OPTIONAL MATCH (pi)-[:USES_SLOT]->(s:Slot)
        RETURN pi, s.id AS slotId
        ORDER BY r.idx
        """

        try:
            with self.driver.session(database="neo4j") as session:
                result = session.run(query, {"pattern_id": pattern_id})
                items = []
                for record in result:
                    item = dict(record["pi"])
                    if record["slotId"]:
                        item["slotId"] = record["slotId"]
                    items.append(item)
                return items
        except Exception as e:
            raise wrap_exception(
                e,
                DatabaseException,
                "Failed to retrieve pattern items",
                pattern_id=pattern_id,
            )

    def create_pattern_item(
        self,
        pattern_id: str,
        idx: int,
        kind: str,
        literal_value: Optional[str] = None,
        slot_id: Optional[str] = None,
    ) -> str:
        """
        Create single PatternItem with HAS_ITEM relationship.

        This is a convenience method for creating individual items.
        For bulk creation, use batch_create_pattern_items() instead.

        Args:
            pattern_id: Pattern UUID
            idx: Position in pattern sequence
            kind: "LITERAL" or "SLOT"
            literal_value: Required for LITERAL kind
            slot_id: Required for SLOT kind

        Returns:
            PatternItem ID (UUID)

        Raises:
            ValueError: If validation fails
            DatabaseException: If database operation fails

        Example:
            >>> # Create literal item
            >>> item_id = manager.create_pattern_item(
            ...     pattern_id, idx=0, kind="LITERAL", literal_value="was"
            ... )
            >>> # Create slot item
            >>> slot_id = manager.create_slot("WH_WORD", ["was", "wer"])
            >>> item_id = manager.create_pattern_item(
            ...     pattern_id, idx=1, kind="SLOT", slot_id=slot_id
            ... )
        """
        if kind not in VALID_ITEM_KINDS:
            raise ValueError(f"Kind must be one of {VALID_ITEM_KINDS}")
        if kind == "LITERAL" and not literal_value:
            raise ValueError("LITERAL items require literal_value")
        if kind == "SLOT" and not slot_id:
            raise ValueError("SLOT items require slot_id")
        if idx < 0:
            raise ValueError("Index must be non-negative")

        item_id = str(uuid.uuid4())

        if kind == "LITERAL":
            query = """
            MATCH (p:Pattern {id: $pattern_id})
            CREATE (pi:PatternItem {id: $item_id, idx: $idx, kind: $kind, literalValue: $literal_value})
            CREATE (p)-[:HAS_ITEM {idx: $idx}]->(pi)
            RETURN pi.id AS id
            """
            params = {
                "pattern_id": pattern_id,
                "item_id": item_id,
                "idx": idx,
                "kind": kind,
                "literal_value": literal_value,
            }
        else:  # SLOT
            query = """
            MATCH (p:Pattern {id: $pattern_id})
            MATCH (s:Slot {id: $slot_id})
            CREATE (pi:PatternItem {id: $item_id, idx: $idx, kind: $kind})
            CREATE (p)-[:HAS_ITEM {idx: $idx}]->(pi)
            CREATE (pi)-[:USES_SLOT]->(s)
            RETURN pi.id AS id
            """
            params = {
                "pattern_id": pattern_id,
                "item_id": item_id,
                "idx": idx,
                "kind": kind,
                "slot_id": slot_id,
            }

        try:
            with self.driver.session(database="neo4j") as session:
                result = session.run(query, params)
                created_id = result.single()["id"]
                logger.debug(f"PatternItem created: {kind} at idx={idx}")
                return created_id
        except Exception as e:
            raise wrap_exception(
                e,
                DatabaseException,
                "Failed to create pattern item",
                pattern_id=pattern_id,
                kind=kind,
                idx=idx,
            )
