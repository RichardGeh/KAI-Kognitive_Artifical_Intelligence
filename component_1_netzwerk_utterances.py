# component_1_netzwerk_utterances.py
"""
Utterance and Token management for pattern discovery system.

This module handles operations related to storing and retrieving user utterances
and their linguistic tokenization:
- Utterance creation with embeddings
- Token creation with spaCy linguistic features (lemma, POS)
- Token sequence management (NEXT chain)
- Utterance archival mechanism (scalability)
- Batch operations for efficiency

Extracted as part of Pattern Discovery System implementation (Part 1).
"""

import re
import threading
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional

from neo4j import Driver

from component_15_logging_config import get_logger
from component_utils_text_normalization import TextNormalizer
from kai_exceptions import DatabaseException, wrap_exception

logger = get_logger(__name__)

# Whitelist of valid POS tags (security - prevent injection)
VALID_POS_TAGS = {
    "NOUN",
    "VERB",
    "ADJ",
    "ADV",
    "PRON",
    "DET",
    "ADP",
    "CONJ",
    "CCONJ",  # Coordinating conjunction (and, but, or)
    "SCONJ",
    "PUNCT",
    "NUM",
    "X",
    "PROPN",
    "AUX",
    "PART",
    "INTJ",
    "SYM",
    "SPACE",
}

# Entity name validation regex (from CLAUDE.md)
ENTITY_NAME_REGEX = re.compile(r"^[a-z_][a-z0-9_]{0,63}$")


class UtteranceManager:
    """
    Manages utterance and token CRUD operations in Neo4j.

    Responsibilities:
    - Create/read utterance nodes with embeddings
    - Create token nodes with linguistic features
    - Manage token sequences (NEXT chain)
    - Archival mechanism for old utterances
    - Batch operations for efficiency

    Thread Safety:
        This class is thread-safe. All database operations use parameterized
        queries to prevent injection.

    Attributes:
        driver: Neo4j driver instance
        normalizer: TextNormalizer for consistent text normalization
        _lock: Thread lock for critical operations
    """

    def __init__(self, driver: Driver, preprocessor=None):
        """
        Initialize utterance manager.

        Args:
            driver: Neo4j driver instance
            preprocessor: Optional LinguisticPreprocessor for text normalization

        Raises:
            ValueError: If driver is None
        """
        if not driver:
            raise ValueError("Driver cannot be None")

        self.driver = driver
        self.normalizer = TextNormalizer(preprocessor)
        self._lock = threading.RLock()

        logger.debug("UtteranceManager initialisiert")

    def create_utterance(
        self, text: str, embedding: List[float], user_id: Optional[str] = None
    ) -> str:
        """
        Create Utterance node with security validation.

        Uses parameterized query for injection prevention.
        Normalizes text using TextNormalizer.

        Args:
            text: Original user input text
            embedding: 384-dimensional embedding vector
            user_id: Optional user identifier

        Returns:
            Utterance ID (UUID)

        Raises:
            ValueError: If validation fails
            DatabaseException: If database operation fails

        Example:
            >>> utterance_id = manager.create_utterance(
            ...     "Was ist ein Hund?",
            ...     embedding_vector,
            ...     user_id="user_123"
            ... )
        """
        # Input validation
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        if len(text) > 10000:
            raise ValueError("Text must be <= 10000 characters")

        # Embedding validation (allow None/invalid for edge cases, will be skipped in clustering)
        if embedding is not None and not isinstance(embedding, list):
            raise ValueError("Embedding must be a list or None")

        if embedding and len(embedding) != 384:
            logger.warning(
                f"Invalid embedding dimension: {len(embedding)} (expected 384), storing as-is"
            )

        # Normalize text
        normalized = self.normalizer.clean_entity(text)

        # Generate UUID
        utterance_id = str(uuid.uuid4())

        # Parameterized query (security requirement)
        query = """
        CREATE (u:Utterance {
            id: $id,
            text: $text,
            normalized: $normalized,
            timestamp: datetime(),
            userId: $user_id,
            embedding: $embedding,
            archived: false
        })
        RETURN u.id AS id
        """

        try:
            with self.driver.session(database="neo4j") as session:
                result = session.run(
                    query,
                    {
                        "id": utterance_id,
                        "text": text,
                        "normalized": normalized,
                        "user_id": user_id,
                        "embedding": embedding,
                    },
                )
                created_id = result.single()["id"]

                # Verify write (CLAUDE.md requirement)
                verify_query = "MATCH (u:Utterance {id: $id}) RETURN u"
                verification = session.run(verify_query, {"id": utterance_id})
                if not verification.single():
                    raise DatabaseException(
                        f"Failed to create utterance: {utterance_id}"
                    )

                logger.debug(
                    f"Utterance created: {utterance_id[:8]}...",
                    extra={"length": len(text)},
                )
                return created_id

        except Exception as e:
            raise wrap_exception(
                e,
                DatabaseException,
                "Failed to create utterance",
                text_length=len(text),
            )

    def get_recent_utterances(
        self, limit: int = 100, archived: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Retrieve recent utterances efficiently using temporal index.

        Uses composite index (timestamp, archived) for fast filtering.
        Parameterized query for security.

        Args:
            limit: Maximum number of utterances to retrieve (1-10000)
            archived: Whether to retrieve archived utterances

        Returns:
            List of utterance dicts with properties

        Raises:
            ValueError: If limit is invalid
            DatabaseException: If database operation fails

        Example:
            >>> utterances = manager.get_recent_utterances(limit=50)
            >>> print(f"Retrieved {len(utterances)} utterances")
        """
        # Validation
        if limit <= 0 or limit > 10000:
            raise ValueError("Limit must be 1-10000")

        # Parameterized query using temporal index
        query = """
        MATCH (u:Utterance)
        WHERE u.archived = $archived
        RETURN u
        ORDER BY u.timestamp DESC
        LIMIT $limit
        """

        try:
            with self.driver.session(database="neo4j") as session:
                result = session.run(query, {"archived": archived, "limit": limit})
                utterances = [dict(record["u"]) for record in result]

                logger.debug(
                    f"Retrieved {len(utterances)} utterances",
                    extra={"archived": archived, "limit": limit},
                )
                return utterances

        except Exception as e:
            raise wrap_exception(
                e, DatabaseException, "Failed to retrieve utterances", limit=limit
            )

    def archive_old_utterances(self, days_threshold: int = 30) -> int:
        """
        Archive old utterances using batch UPDATE.

        Uses datetime arithmetic and indexed timestamp filter for efficiency.
        Idempotent - safe to call multiple times.

        Args:
            days_threshold: Archive utterances older than N days (minimum 1)

        Returns:
            Number of utterances archived

        Raises:
            ValueError: If days_threshold is invalid
            DatabaseException: If database operation fails

        Example:
            >>> archived_count = manager.archive_old_utterances(days_threshold=30)
            >>> print(f"Archived {archived_count} old utterances")
        """
        # Validation
        if days_threshold < 1:
            raise ValueError("Days threshold must be >= 1")

        # Parameterized query with datetime arithmetic
        query = """
        MATCH (u:Utterance)
        WHERE u.archived = false
          AND u.timestamp < datetime() - duration({days: $days})
        SET u.archived = true
        RETURN count(u) AS archived_count
        """

        try:
            with self.driver.session(database="neo4j") as session:
                result = session.run(query, {"days": days_threshold})
                count = result.single()["archived_count"]

                logger.info(
                    f"Archived {count} utterances older than {days_threshold} days"
                )
                return count

        except Exception as e:
            raise wrap_exception(
                e,
                DatabaseException,
                "Failed to archive utterances",
                days_threshold=days_threshold,
            )

    def create_token(
        self,
        surface: str,
        lemma: str,
        pos: str,
        utterance_id: str,
        idx: int,
    ) -> str:
        """
        Create Token node linked to Utterance with NEXT chain.

        Uses transaction for atomicity (Token + HAS_TOKEN + NEXT relations).
        Validates and normalizes lemma for security.
        Uses OPTIONAL MATCH + FOREACH pattern for conditional NEXT creation.

        Args:
            surface: Original word form (e.g., "Apfels")
            lemma: Base form (e.g., "apfel")
            pos: Part-of-speech tag (e.g., "NOUN")
            utterance_id: UUID of parent utterance
            idx: Token position in sequence (0-based)

        Returns:
            Token ID (UUID)

        Raises:
            ValueError: If validation fails
            DatabaseException: If database operation fails

        Example:
            >>> token_id = manager.create_token(
            ...     surface="Hund",
            ...     lemma="hund",
            ...     pos="NOUN",
            ...     utterance_id="123e4567-e89b-12d3-a456-426614174000",
            ...     idx=0
            ... )
        """
        # Validation
        if not surface or not pos:
            raise ValueError("Surface and POS are required")

        if idx < 0:
            raise ValueError("Index must be non-negative")

        # POS validation (whitelist)
        if pos not in VALID_POS_TAGS:
            raise ValueError(f"Invalid POS tag: {pos}")

        # Normalize and validate lemma (security requirement)
        # Allow empty lemma (will be normalized to "unknown")
        lemma_normalized = re.sub(r"[^a-z0-9_]", "", lemma.lower()) if lemma else ""
        if not lemma_normalized:
            lemma_normalized = "unknown"  # Fallback for empty/special chars

        if len(lemma_normalized) > 63:
            lemma_normalized = lemma_normalized[:63]

        if not ENTITY_NAME_REGEX.match(lemma_normalized):
            # Try fixing: add underscore prefix if starts with digit
            if lemma_normalized and lemma_normalized[0].isdigit():
                lemma_normalized = "_" + lemma_normalized[:62]
            else:
                lemma_normalized = "unknown"

        # Generate UUID
        token_id = str(uuid.uuid4())

        # Atomic transaction: Token + HAS_TOKEN + NEXT
        query = """
        MATCH (u:Utterance {id: $utterance_id})
        CREATE (t:Token {
            id: $token_id,
            surface: $surface,
            lemma: $lemma,
            pos: $pos,
            idx: $idx
        })
        CREATE (u)-[:HAS_TOKEN {idx: $idx}]->(t)
        WITH t, u
        OPTIONAL MATCH (prev:Token)<-[:HAS_TOKEN {idx: $prev_idx}]-(u)
        WHERE $idx > 0
        FOREACH (_ IN CASE WHEN prev IS NOT NULL THEN [1] ELSE [] END |
            CREATE (prev)-[:NEXT]->(t)
        )
        RETURN t.id AS id
        """

        try:
            with self.driver.session(database="neo4j") as session:
                with session.begin_transaction() as tx:
                    try:
                        result = tx.run(
                            query,
                            {
                                "utterance_id": utterance_id,
                                "token_id": token_id,
                                "surface": surface,
                                "lemma": lemma_normalized,
                                "pos": pos,
                                "idx": idx,
                                "prev_idx": idx - 1,
                            },
                        )
                        created_id = result.single()["id"]
                        tx.commit()

                        logger.debug(
                            f"Token created: {surface} ({lemma_normalized}/{pos}) at idx={idx}"
                        )
                        return created_id

                    except Exception as inner_e:
                        tx.rollback()
                        logger.error(
                            f"Token creation transaction failed: {inner_e}",
                            exc_info=True,
                        )
                        raise  # Re-raise same exception without wrapping

        except Exception as e:
            raise wrap_exception(
                e,
                DatabaseException,
                "Failed to create token",
                surface=surface,
                idx=idx,
            )

    def get_tokens_for_utterance(self, utterance_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve tokens in sequence order using idx property.

        Single query with ORDER BY (no N+1 problem).
        Parameterized query for security.

        WARNING: For multiple utterances, use get_tokens_for_utterances_batch()
        to avoid N+1 query problem. This method is for single utterance only.
        Calling this method in a loop for 100+ utterances will result in poor
        performance (100 queries instead of 1).

        Args:
            utterance_id: UUID of utterance

        Returns:
            List of token dicts in sequence order

        Raises:
            DatabaseException: If database operation fails

        Example:
            >>> # GOOD: Single utterance
            >>> tokens = manager.get_tokens_for_utterance(utterance_id)
            >>> for token in tokens:
            ...     print(f"{token['surface']} ({token['pos']})")
            >>>
            >>> # BAD: Multiple utterances (N+1 problem!)
            >>> # for uid in utterance_ids:
            >>> #     tokens = manager.get_tokens_for_utterance(uid)  # Don't do this!
            >>>
            >>> # GOOD: Multiple utterances
            >>> tokens_by_utterance = manager.get_tokens_for_utterances_batch(utterance_ids)
        """
        # Parameterized query with ORDER BY
        query = """
        MATCH (u:Utterance {id: $utterance_id})-[r:HAS_TOKEN]->(t:Token)
        RETURN t
        ORDER BY r.idx
        """

        try:
            with self.driver.session(database="neo4j") as session:
                result = session.run(query, {"utterance_id": utterance_id})
                tokens = [dict(record["t"]) for record in result]

                logger.debug(
                    f"Retrieved {len(tokens)} tokens for utterance {utterance_id[:8]}..."
                )
                return tokens

        except Exception as e:
            raise wrap_exception(
                e,
                DatabaseException,
                "Failed to retrieve tokens",
                utterance_id=utterance_id,
            )

    def get_tokens_for_utterances_batch(
        self, utterance_ids: List[str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Fetch tokens for multiple utterances in single query (batch optimization).

        Uses UNWIND for batch efficiency (100x speedup for 100 utterances).
        CRITICAL: Prevents N+1 query problem in LGG clustering.

        Args:
            utterance_ids: List of utterance UUIDs

        Returns:
            Dict mapping utterance_id to list of token dicts

        Raises:
            ValueError: If utterance_ids is empty
            DatabaseException: If database operation fails

        Example:
            >>> tokens_by_utterance = manager.get_tokens_for_utterances_batch(
            ...     ["uuid1", "uuid2", "uuid3"]
            ... )
            >>> for uid, tokens in tokens_by_utterance.items():
            ...     print(f"Utterance {uid[:8]}: {len(tokens)} tokens")
        """
        # Validation
        if not utterance_ids:
            raise ValueError("Utterance IDs list cannot be empty")

        # Batch query using UNWIND
        query = """
        UNWIND $utterance_ids AS uid
        MATCH (u:Utterance {id: uid})-[r:HAS_TOKEN]->(t:Token)
        RETURN uid, t
        ORDER BY uid, r.idx
        """

        try:
            with self.driver.session(database="neo4j") as session:
                result = session.run(query, {"utterance_ids": utterance_ids})

                # Group tokens by utterance
                tokens_by_utterance: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
                for record in result:
                    uid = record["uid"]
                    token = dict(record["t"])
                    tokens_by_utterance[uid].append(token)

                logger.debug(
                    f"Batch retrieved tokens for {len(tokens_by_utterance)} utterances"
                )
                return dict(tokens_by_utterance)

        except Exception as e:
            raise wrap_exception(
                e,
                DatabaseException,
                "Failed to batch retrieve tokens",
                count=len(utterance_ids),
            )

    def count_utterances(self, archived: bool = False) -> int:
        """
        Count utterances with optional archival filter.

        Used to trigger periodic pattern discovery every N new utterances.

        Args:
            archived: Whether to count archived utterances (default: False)

        Returns:
            Number of utterances matching filter

        Raises:
            DatabaseException: If database operation fails

        Example:
            >>> count = manager.count_utterances(archived=False)
            >>> print(f"Active utterances: {count}")
        """
        query = """
        MATCH (u:Utterance)
        WHERE u.archived = $archived
        RETURN count(u) AS count
        """

        try:
            with self.driver.session(database="neo4j") as session:
                result = session.run(query, {"archived": archived})
                count = result.single()["count"]

                logger.debug(f"Counted {count} utterances (archived={archived})")
                return count

        except Exception as e:
            raise wrap_exception(
                e, DatabaseException, "Failed to count utterances", archived=archived
            )

    def get_utterance(self, utterance_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve single utterance with all properties.

        Args:
            utterance_id: Utterance UUID

        Returns:
            Utterance dict with all properties or None if not found

        Raises:
            DatabaseException: If database operation fails

        Example:
            >>> utterance = manager.get_utterance(utterance_id)
            >>> if utterance:
            ...     print(f"Text: {utterance['text']}")
            ...     print(f"Embedding dim: {len(utterance['embedding'])}")
        """
        query = """
        MATCH (u:Utterance {id: $utterance_id})
        RETURN u
        """

        try:
            with self.driver.session(database="neo4j") as session:
                result = session.run(query, {"utterance_id": utterance_id})
                record = result.single()
                return dict(record["u"]) if record else None
        except Exception as e:
            raise wrap_exception(
                e,
                DatabaseException,
                "Failed to retrieve utterance",
                utterance_id=utterance_id,
            )
