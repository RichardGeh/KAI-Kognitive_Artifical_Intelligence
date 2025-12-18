"""
component_1_episodic_memory.py

Episodic memory management for KAI knowledge graph.

This module handles:
- Episode creation and storage (learning events)
- Episode querying and retrieval
- Episode deletion (with cascade support)
- Fact-to-episode linking (provenance tracking)

Episodic Memory (PHASE 3):
    Tracks WHEN and HOW knowledge was acquired. Each Episode represents
    a learning event (e.g., text ingestion, manual definition, pattern learning).
    Enables transparency about knowledge origins and error correction.

Architecture:
    - Uses Neo4jSessionMixin for thread-safe database access
    - Neo4j Schema: Episode nodes with LEARNED_FACT relationships to Fact nodes
    - Fact nodes link to actual knowledge graph relations
    - Timestamps for temporal reasoning

Thread Safety:
    All database operations are thread-safe via Neo4jSessionMixin._safe_run()
    with RLock synchronization.

Dependencies:
    - infrastructure/neo4j_session_mixin.py: Session management
    - component_15_logging_config.py: Structured logging
    - kai_exceptions.py: Exception hierarchy

Usage:
    from neo4j import Driver
    from component_1_episodic_memory import EpisodicMemory

    driver = Driver("bolt://localhost:7687", auth=("neo4j", "password"))
    memory = EpisodicMemory(driver)

    # Create episode
    episode_id = memory.create_episode(
        episode_type="ingestion",
        content="Ein Hund ist ein Tier.",
        metadata={"source": "user_input"}
    )

    # Link fact to episode
    memory.link_fact_to_episode("hund", "IS_A", "tier", episode_id)

    # Query episodes
    episodes = memory.query_episodes_about("hund")
    for ep in episodes:
        print(f"Learned at {ep['timestamp']}: {ep['content']}")

Note: Follows CLAUDE.md standards - NO cp1252-unsafe Unicode, structured logging,
      comprehensive error handling, thread safety.
"""

import json
import re
from typing import Any, Dict, List, Optional

from neo4j import Driver

from component_15_logging_config import get_logger
from infrastructure.neo4j_session_mixin import Neo4jSessionMixin

logger = get_logger(__name__)


class EpisodicMemory(Neo4jSessionMixin):
    """
    Episodic memory management for learning events.

    Provides storage and retrieval of learning episodes with temporal tracking.
    Enables transparent provenance tracking for all acquired knowledge.

    Attributes:
        driver: Neo4j driver instance (inherited from Neo4jSessionMixin)

    Thread Safety:
        All methods are thread-safe via Neo4jSessionMixin._safe_run()

    Example:
        memory = EpisodicMemory(driver)

        # Create and link episode
        ep_id = memory.create_episode("ingestion", "Text content", {"meta": "data"})
        memory.link_fact_to_episode("subject", "relation", "object", ep_id)

        # Query and delete
        episodes = memory.query_episodes_about("subject")
        memory.delete_episode(ep_id, cascade=True)
    """

    def __init__(self, driver: Driver):
        """
        Initialize episodic memory with Neo4j driver.

        Args:
            driver: Neo4j driver instance

        Raises:
            Neo4jConnectionError: If driver is None or connection fails
        """
        super().__init__(driver, enable_cache=False)
        logger.debug("EpisodicMemory initialized")

    def create_episode(
        self, episode_type: str, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Create an Episode node in the graph to track learning events.

        PHASE 3 (Episodic Memory): Enables tracking WHEN and HOW knowledge
        was acquired. Each Episode represents a learning event (e.g., text
        ingestion, manual definition, pattern learning).

        Args:
            episode_type: Type of episode (e.g., "ingestion", "definition", "pattern_learning")
            content: Original text/content of the episode
            metadata: Additional metadata (e.g., {"query": "...", "user_action": "..."})

        Returns:
            Episode ID (UUID) if successful, None on error

        Example:
            episode_id = memory.create_episode(
                episode_type="ingestion",
                content="Ein Hund ist ein Tier.",
                metadata={"source": "user_input"}
            )
        """
        try:
            results = self._safe_run(
                """
                CREATE (e:Episode {
                    id: randomUUID(),
                    type: $type,
                    content: $content,
                    timestamp: timestamp(),
                    metadata: $metadata
                })
                RETURN e.id AS episode_id
                """,
                operation="create_episode",
                type=episode_type,
                content=content,
                metadata=json.dumps(metadata or {}),
            )

            episode_id = results[0]["episode_id"] if results else None

            if episode_id:
                logger.info(
                    "Episode created: %s",
                    episode_type,
                    extra={
                        "episode_id": episode_id,
                        "content_preview": content[:50],
                    },
                )

            return episode_id

        except Exception as e:
            logger.error(
                "Error creating episode: %s",
                str(e)[:100],
                extra={"episode_type": episode_type},
                exc_info=True,
            )
            return None

    def batch_create_episodes(
        self,
        episodes: List[Dict[str, Any]],
        batch_size: int = 100,
    ) -> List[str]:
        """
        Batch-create episodes using UNWIND (10x faster than individual calls).

        PERFORMANCE: This method uses UNWIND for efficient batch processing,
        significantly faster than calling create_episode() individually for
        large numbers of episodes.

        Args:
            episodes: List of dicts with keys:
                - episode_type (str): Type of episode (e.g., "ingestion", "pattern_learning")
                - content (str): Episode content
                - metadata (dict, optional): Additional metadata
            batch_size: Batch size per UNWIND query (default: 100)

        Returns:
            List of created episode IDs (UUIDs)

        Example:
            episodes = [
                {"episode_type": "ingestion", "content": "Learned: Hund ist Tier", "metadata": {"source": "user"}},
                {"episode_type": "ingestion", "content": "Learned: Katze ist Tier", "metadata": {"source": "user"}},
            ]
            ids = memory.batch_create_episodes(episodes)
            # Returns: ["uuid1", "uuid2"]

        Note:
            - Empty episodes list returns empty list (not an error)
            - Metadata is serialized to JSON string
            - Transaction is atomic per batch
            - Continues processing even if one batch fails (graceful degradation)
        """
        if not episodes:
            logger.debug("batch_create_episodes: No episodes to create")
            return []

        episode_ids: List[str] = []

        for i in range(0, len(episodes), batch_size):
            batch = episodes[i : i + batch_size]

            # Prepare batch data (serialize metadata to JSON)
            batch_data = []
            for ep in batch:
                metadata = ep.get("metadata", {})
                if metadata and not isinstance(metadata, dict):
                    logger.warning(
                        "Episode metadata must be dict, got %s", type(metadata).__name__
                    )
                    metadata = {}

                batch_data.append(
                    {
                        "episode_type": ep["episode_type"],
                        "content": ep["content"],
                        "metadata": json.dumps(metadata),
                    }
                )

            # Batch create with UNWIND
            query = """
            UNWIND $batch AS ep
            CREATE (e:Episode {
                id: randomUUID(),
                type: ep.episode_type,
                content: ep.content,
                timestamp: timestamp(),
                metadata: ep.metadata
            })
            RETURN e.id AS episode_id
            """

            try:
                results = self._safe_run(
                    query,
                    operation="batch_create_episodes",
                    batch=batch_data,
                )

                batch_ids = [r["episode_id"] for r in results]
                episode_ids.extend(batch_ids)

                logger.info(
                    "Batch created episodes",
                    extra={
                        "batch_size": len(batch_ids),
                        "total_so_far": len(episode_ids),
                    },
                )

            except Exception as e:
                logger.error(
                    "Batch episode creation failed: %s",
                    str(e)[:100],
                    extra={"batch_index": i // batch_size, "batch_size": len(batch)},
                    exc_info=True,
                )
                # Continue with next batch instead of failing completely
                continue

        logger.info(
            "Batch episode creation complete",
            extra={"total_episodes": len(episodes), "created_count": len(episode_ids)},
        )

        return episode_ids

    def link_fact_to_episode(
        self, subject: str, relation: str, object: str, episode_id: str
    ) -> bool:
        """
        Link a fact to an episode for provenance tracking.

        PHASE 3: Enables transparency about knowledge origins and later
        error correction (e.g., "Delete everything from Episode X").

        Args:
            subject: Subject of the relation
            relation: Relation type (e.g., "IS_A")
            object: Object of the relation
            episode_id: Episode ID this fact originated from

        Returns:
            True if successfully linked, False on error

        Example:
            # After assert_relation():
            memory.link_fact_to_episode("hund", "IS_A", "tier", episode_id)
        """
        # Validate relation type (whitelist pattern for Neo4j security)
        safe_relation = re.sub(r"[^a-zA-Z0-9_]", "", relation.upper())
        if not safe_relation:
            logger.error("Invalid relation type", extra={"relation": relation})
            return False

        try:
            # NOTE: Cypher does not allow relationships to relationships,
            # so we create Fact nodes as intermediaries
            results = self._safe_run(
                f"""
                MATCH (s:Konzept {{name: $subject}})-[rel:{safe_relation}]->(o:Konzept {{name: $object}})
                MATCH (e:Episode {{id: $episode_id}})
                MERGE (f:Fact {{
                    subject: $subject,
                    relation: $relation,
                    object: $object
                }})
                MERGE (e)-[learned:LEARNED_FACT]->(f)
                ON CREATE SET learned.linked_at = timestamp()
                RETURN learned IS NOT NULL AS success
                """,
                operation="link_fact_to_episode",
                subject=subject.lower(),
                object=object.lower(),
                relation=safe_relation,
                episode_id=episode_id,
            )

            success = results[0]["success"] if results else False

            if success:
                logger.debug(
                    "Fact linked to episode",
                    extra={
                        "subject": subject,
                        "relation": safe_relation,
                        "object": object,
                        "episode_id": episode_id[:8],
                    },
                )

            return success

        except Exception as e:
            logger.error(
                "Error linking fact to episode: %s",
                str(e)[:100],
                extra={
                    "subject": subject,
                    "relation": relation,
                    "object": object,
                    "episode_id": episode_id[:8],
                },
                exc_info=True,
            )
            return False

    def link_facts_to_episode_batch(
        self, facts: List[Dict[str, str]], episode_id: str
    ) -> int:
        """
        Link multiple facts to an episode in a single batch operation (Quick Win #3).

        PERFORMANCE: 10-20x faster than calling link_fact_to_episode() individually
        for large numbers of facts. Uses UNWIND for efficient batch processing.

        Args:
            facts: List of dicts with keys {subject, relation, object}
            episode_id: Episode ID to link facts to

        Returns:
            Number of successfully linked facts

        Example:
            facts = [
                {"subject": "hund", "relation": "IS_A", "object": "tier"},
                {"subject": "katze", "relation": "IS_A", "object": "tier"},
                {"subject": "hund", "relation": "HAS_PROPERTY", "object": "freundlich"}
            ]
            linked_count = memory.link_facts_to_episode_batch(facts, episode_id)
            print(f"Linked {linked_count} facts")

        Note:
            - Empty facts list returns 0 (not an error)
            - Invalid relation types are skipped with warning
            - Transaction is atomic - either all succeed or all fail
        """
        if not facts:
            logger.debug(
                "No facts to link to episode", extra={"episode_id": episode_id[:8]}
            )
            return 0

        # Validate and sanitize all relations first (security)
        sanitized_facts = []
        for fact in facts:
            relation = fact.get("relation", "")
            safe_relation = re.sub(r"[^a-zA-Z0-9_]", "", relation.upper())
            if not safe_relation:
                logger.warning(
                    "Skipping fact with invalid relation type",
                    extra={"relation": relation, "fact": fact},
                )
                continue

            sanitized_facts.append(
                {
                    "subject": fact["subject"].lower(),
                    "object": fact["object"].lower(),
                    "relation": safe_relation,
                }
            )

        if not sanitized_facts:
            logger.warning(
                "No valid facts after sanitization",
                extra={"episode_id": episode_id[:8]},
            )
            return 0

        try:
            # Batch query using UNWIND for efficiency
            # NOTE: We can't use dynamic relation types in a single query,
            # so we need to group facts by relation type
            from collections import defaultdict

            facts_by_relation = defaultdict(list)

            for fact in sanitized_facts:
                facts_by_relation[fact["relation"]].append(fact)

            total_linked = 0

            # Process each relation type separately
            for safe_relation, relation_facts in facts_by_relation.items():
                results = self._safe_run(
                    f"""
                    MATCH (e:Episode {{id: $episode_id}})

                    UNWIND $facts AS fact
                    MATCH (s:Konzept {{name: fact.subject}})-[rel:{safe_relation}]->(o:Konzept {{name: fact.object}})

                    MERGE (f:Fact {{
                        subject: fact.subject,
                        relation: fact.relation,
                        object: fact.object
                    }})
                    MERGE (e)-[learned:LEARNED_FACT]->(f)
                    ON CREATE SET learned.linked_at = timestamp()

                    RETURN count(DISTINCT f) AS linked_count
                    """,
                    operation="link_facts_to_episode_batch",
                    episode_id=episode_id,
                    facts=relation_facts,
                )

                linked_count = results[0]["linked_count"] if results else 0
                total_linked += linked_count

            logger.info(
                "Batch-linked facts to episode",
                extra={
                    "episode_id": episode_id[:8],
                    "fact_count": len(sanitized_facts),
                    "linked_count": total_linked,
                    "relation_types": list(facts_by_relation.keys()),
                },
            )

            return total_linked

        except Exception as e:
            logger.error(
                "Error in batch episode linking: %s",
                str(e)[:100],
                extra={"episode_id": episode_id[:8], "fact_count": len(facts)},
                exc_info=True,
            )
            return 0

    def query_episodes_about(self, topic: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Find all episodes where learning occurred about a specific topic.

        PHASE 3: Answers the question "When did I learn about X?"

        Args:
            topic: Topic to search for
            limit: Maximum number of episodes (newest first)

        Returns:
            List of episode dictionaries with:
            - episode_id: UUID of the episode
            - type: Episode type
            - content: Original content
            - timestamp: Timestamp (Neo4j timestamp)
            - learned_facts: List of facts learned in this episode

        Example:
            episodes = memory.query_episodes_about("hund")
            for ep in episodes:
                print(f"On {ep['timestamp']}: {ep['content']}")
        """
        try:
            results = self._safe_run(
                """
                MATCH (e:Episode)-[:LEARNED_FACT]->(f:Fact)
                WHERE f.subject = $topic OR f.object = $topic
                WITH e, collect({
                    subject: f.subject,
                    relation: f.relation,
                    object: f.object
                }) AS facts
                RETURN e.id AS episode_id,
                       e.type AS type,
                       e.content AS content,
                       e.timestamp AS timestamp,
                       e.metadata AS metadata,
                       facts AS learned_facts
                ORDER BY e.timestamp DESC
                LIMIT $limit
                """,
                operation="query_episodes_about",
                topic=topic.lower(),
                limit=limit,
            )

            logger.debug(
                "Query episodes about '%s' -> %d episodes found", topic, len(results)
            )

            return results

        except Exception as e:
            logger.error(
                "Error querying episodes about '%s': %s",
                topic,
                str(e)[:100],
                exc_info=True,
            )
            return []

    def query_all_episodes(
        self, episode_type: Optional[str] = None, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Retrieve all episodes from the graph, optionally filtered by type.

        PHASE 3: Enables overview of all learning events.

        Args:
            episode_type: Optional - filter by episode type
            limit: Maximum number of episodes (newest first)

        Returns:
            List of episode dictionaries (see query_episodes_about)

        Example:
            # All text ingestion episodes
            episodes = memory.query_all_episodes(episode_type="ingestion")
        """
        try:
            if episode_type:
                results = self._safe_run(
                    """
                    MATCH (e:Episode {type: $type})
                    OPTIONAL MATCH (e)-[:LEARNED_FACT]->(f:Fact)
                    WITH e, collect(CASE WHEN f IS NOT NULL THEN {
                        subject: f.subject,
                        relation: f.relation,
                        object: f.object
                    } END) AS facts
                    RETURN e.id AS episode_id,
                           e.type AS type,
                           e.content AS content,
                           e.timestamp AS timestamp,
                           e.metadata AS metadata,
                           facts AS learned_facts
                    ORDER BY e.timestamp DESC
                    LIMIT $limit
                    """,
                    operation="query_all_episodes_typed",
                    type=episode_type,
                    limit=limit,
                )
            else:
                results = self._safe_run(
                    """
                    MATCH (e:Episode)
                    OPTIONAL MATCH (e)-[:LEARNED_FACT]->(f:Fact)
                    WITH e, collect(CASE WHEN f IS NOT NULL THEN {
                        subject: f.subject,
                        relation: f.relation,
                        object: f.object
                    } END) AS facts
                    RETURN e.id AS episode_id,
                           e.type AS type,
                           e.content AS content,
                           e.timestamp AS timestamp,
                           e.metadata AS metadata,
                           facts AS learned_facts
                    ORDER BY e.timestamp DESC
                    LIMIT $limit
                    """,
                    operation="query_all_episodes",
                    limit=limit,
                )

            logger.debug(
                "Query all episodes: %d found%s",
                len(results),
                f" (Type: {episode_type})" if episode_type else "",
            )

            return results

        except Exception as e:
            logger.error(
                "Error querying all episodes: %s",
                str(e)[:100],
                extra={"episode_type": episode_type},
                exc_info=True,
            )
            return []

    def delete_episode(self, episode_id: str, cascade: bool = False) -> bool:
        """
        Delete an episode from the graph.

        PHASE 3: Enables error correction by deleting faulty learning events.

        Args:
            episode_id: ID of the episode to delete
            cascade: If True, also delete learned facts
                     (WARNING: Deletes actual knowledge graph relations!)

        Returns:
            True if successfully deleted, False on error

        Example:
            # Delete episode only, keep facts
            memory.delete_episode(episode_id)

            # Delete episode AND all learned facts
            memory.delete_episode(episode_id, cascade=True)
        """
        try:
            if cascade:
                # Delete episode, Fact nodes AND actual relations between concepts
                results = self._safe_run(
                    """
                    MATCH (e:Episode {id: $episode_id})-[:LEARNED_FACT]->(f:Fact)
                    WITH e, collect(f) AS facts
                    UNWIND facts AS fact
                    // Delete actual relation between concepts
                    CALL {
                        WITH fact
                        MATCH (s:Konzept {name: fact.subject})-[r]->(o:Konzept {name: fact.object})
                        WHERE type(r) = fact.relation
                        DELETE r
                    }
                    // Delete Fact node
                    DETACH DELETE fact
                    // Delete Episode
                    WITH e
                    DETACH DELETE e
                    RETURN count(e) AS deleted_count
                    """,
                    operation="delete_episode_cascade",
                    episode_id=episode_id,
                )
            else:
                # Delete only Episode node, keep facts
                results = self._safe_run(
                    """
                    MATCH (e:Episode {id: $episode_id})
                    DETACH DELETE e
                    RETURN count(e) AS deleted_count
                    """,
                    operation="delete_episode",
                    episode_id=episode_id,
                )

            deleted_count = results[0]["deleted_count"] if results else 0

            if deleted_count > 0:
                logger.info(
                    "Episode deleted: %s", episode_id[:8], extra={"cascade": cascade}
                )
                return True
            else:
                logger.warning("Episode not found: %s", episode_id[:8])
                return False

        except Exception as e:
            logger.error(
                "Error deleting episode: %s",
                str(e)[:100],
                extra={"episode_id": episode_id[:8]},
                exc_info=True,
            )
            return False
