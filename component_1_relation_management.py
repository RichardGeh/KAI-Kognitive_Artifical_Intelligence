# component_1_relation_management.py
"""
Relation creation and assertion for Neo4j knowledge graph.

This module handles all operations related to creating and managing relations
between concepts, including:
- Relation assertions (assert_relation)
- Relation validation (safe relation type enforcement)
- Confidence tracking
- Source sentence tracking
- Theory of Mind relations (Agent, Belief, MetaBelief)

Extracted from monolithic component_1_netzwerk_core.py as part of architecture
refactoring (Task 5).
"""

import re
import threading
from collections import defaultdict
from typing import Any, Dict, List, Optional

from neo4j import Driver

from component_15_logging_config import PerformanceLogger, get_logger
from infrastructure import get_cache_manager

logger = get_logger(__name__)

# Whitelist of allowed relation types (security - prevent Cypher injection)
ALLOWED_RELATIONS = {
    "IS_A",
    "HAS_PROPERTY",
    "CAPABLE_OF",
    "PART_OF",
    "LOCATED_IN",
    "CANNOT_DO",
    "IS_NOT_A",
    "NOT_IS_A",
    "HAS_NOT_PROPERTY",
    "NOT_PART_OF",
    "NOT_LOCATED_IN",
    "TRIGGERS",
    "BEDEUTET",
    "CONNECTION",
    "LEARNED_FACT",
    "KNOWS",
    "KNOWS_ABOUT",
    "KNOWS_THAT",
    "ABOUT_AGENT",
    # Spatial relations
    "NORTH_OF",
    "SOUTH_OF",
    "EAST_OF",
    "WEST_OF",
    "ADJACENT_TO",
    "NEIGHBOR_ORTHOGONAL",
    "NEIGHBOR_DIAGONAL",
    "INSIDE",
    "CONTAINS",
    "ABOVE",
    "BELOW",
    "LOCATED_AT",
    "PATH_TO",
    "DISTANCE_TO",
    # Number language
    "EQUIVALENT_TO",
    "REPRESENTS_NUMBER",
}


class RelationManager:
    """
    Manages relation creation and assertion in Neo4j.

    Responsibilities:
    - Create relations between concepts
    - Validate relation types (safe naming)
    - Track confidence and source sentences
    - Theory of Mind support (Agents, Beliefs, MetaBeliefs)

    Thread Safety:
        This class is thread-safe. All critical sections are protected by locks.

    Attributes:
        driver: Neo4j driver instance
        word_manager: WordManager for word creation
        _lock: Thread lock for critical operations
    """

    def __init__(self, driver: Driver, word_manager):
        """
        Initialize relation manager.

        Args:
            driver: Neo4j driver instance
            word_manager: WordManager instance for word operations

        Raises:
            ValueError: If driver or word_manager is None
        """
        if not driver:
            raise ValueError("Driver cannot be None")
        if not word_manager:
            raise ValueError("WordManager cannot be None")

        self.driver = driver
        self.word_manager = word_manager
        self._lock = threading.RLock()

        # Cache manager for invalidation
        self.cache_mgr = get_cache_manager()

        logger.debug("RelationManager initialisiert")

    def assert_relation(
        self,
        subject: str,
        relation: str,
        object: str,
        source_sentence: Optional[str] = None,
        confidence: float = 0.85,
    ) -> bool:
        """
        Create an asserted relationship between two concepts.

        Args:
            subject: Subject concept
            relation: Relation type (will be sanitized and uppercased)
            object: Object concept
            source_sentence: Optional source sentence for provenance
            confidence: Confidence value (0.0-1.0, default: 0.85)

        Returns:
            True if operation succeeded (relation exists after this call), False on error

        Raises:
            ValueError: If relation type is not in whitelist or confidence out of range (security)
        """
        # Validate confidence range (security - prevent invalid data)
        if not (0.0 <= confidence <= 1.0):
            raise ValueError(f"Confidence must be in [0.0, 1.0], got {confidence}")

        if not self.driver:
            logger.error(
                "assert_relation: Kein DB-Driver verfügbar",
                extra={"subject": subject, "relation": relation, "object": object},
            )
            return False

        # Sanitize relation type (only alphanumeric and underscore)
        safe_relation: str = re.sub(r"[^a-zA-Z0-9_]", "", relation.upper())
        if not safe_relation:
            logger.error(
                "Ungültiger Relationstyp",
                extra={"relation": relation, "subject": subject, "object": object},
            )
            return False

        # Validate against whitelist (security - prevent Cypher injection)
        if safe_relation not in ALLOWED_RELATIONS:
            error_msg = (
                f"Relation type '{safe_relation}' not in whitelist. "
                f"Allowed: {', '.join(sorted(ALLOWED_RELATIONS))}"
            )
            logger.error(
                "Relation type not in whitelist",
                extra={
                    "relation": safe_relation,
                    "subject": subject,
                    "object": object,
                },
            )
            raise ValueError(error_msg)

        # Ensure both subject and object concepts exist
        if not self.word_manager.ensure_wort_und_konzept(
            subject
        ) or not self.word_manager.ensure_wort_und_konzept(object):
            logger.warning(
                "Konnte Subject oder Object nicht sicherstellen",
                extra={"subject": subject, "object": object},
            )
            return False

        try:
            with PerformanceLogger(
                logger.logger,
                "assert_relation",
                subject=subject,
                relation=safe_relation,
                object=object,
            ):
                with self.driver.session(database="neo4j") as session:
                    result = session.run(
                        f"""
                        MATCH (s:Konzept {{name: $subject}})
                        MATCH (o:Konzept {{name: $object}})
                        MERGE (s)-[rel:{safe_relation}]->(o)
                        ON CREATE SET
                            rel.source_text = $source,
                            rel.asserted_at = timestamp(),
                            rel.timestamp = datetime({{timezone: 'UTC'}}),
                            rel.confidence = $confidence,
                            rel.created = true
                        ON MATCH SET
                            rel.confidence = $confidence,
                            rel.created = false
                        RETURN rel.created AS was_created
                        """,
                        subject=subject.lower(),
                        object=object.lower(),
                        source=source_sentence,
                        confidence=confidence,
                    )
                    record = result.single()
                    was_created: bool = record["was_created"] if record else False

                    if was_created:
                        logger.info(
                            "Neue Relation erstellt",
                            extra={
                                "subject": subject,
                                "relation": safe_relation,
                                "object": object,
                                "source": source_sentence,
                            },
                        )
                    else:
                        logger.debug(
                            "Relation bereits vorhanden",
                            extra={
                                "subject": subject,
                                "relation": safe_relation,
                                "object": object,
                            },
                        )

                    # Return True if operation succeeded (relation exists after this call)
                    # False only if an error occurred
                    return True

        except Exception as e:
            # Specific exception for write errors
            logger.log_exception(
                e,
                "Fehler in assert_relation",
                subject=subject,
                relation=relation,
                object=object,
            )
            # Graceful degradation - return False
            return False

    def batch_assert_relations(
        self,
        relations: List[Dict[str, Any]],
        batch_size: int = 100,
    ) -> Dict[str, int]:
        """
        Batch-create relations using UNWIND (5-10x faster than individual calls).

        PERFORMANCE: This method uses UNWIND for efficient batch processing,
        significantly faster than calling assert_relation() individually for
        large numbers of relations.

        Args:
            relations: List of dicts with keys:
                - subject (str): Subject entity name
                - relation (str): Relation type
                - object (str): Object entity name
                - confidence (float, optional): Default 0.85
                - source_sentence (str, optional): Source text
            batch_size: Batch size per UNWIND query (default: 100)

        Returns:
            Dict with {relation_type: created_count}

        Raises:
            ValueError: If any confidence value is out of range [0.0, 1.0]
                       or if relation type is not in whitelist

        Example:
            relations = [
                {"subject": "hund", "relation": "IS_A", "object": "tier", "confidence": 0.9},
                {"subject": "katze", "relation": "IS_A", "object": "tier", "confidence": 0.9},
            ]
            counts = relation_mgr.batch_assert_relations(relations)
            # Returns: {"IS_A": 2}

        Note:
            - Empty relations list returns empty dict (not an error)
            - Invalid relation types raise ValueError
            - All words/concepts are created if they don't exist
            - Cache is invalidated for all affected subjects
            - Transaction is atomic per relation type batch
        """
        if not relations:
            logger.debug("batch_assert_relations: No relations to create")
            return {}

        # Validate all confidences upfront (fail fast)
        for rel in relations:
            conf = rel.get("confidence", 0.85)
            if not (0.0 <= conf <= 1.0):
                raise ValueError(
                    f"Confidence must be in [0.0, 1.0], got {conf} for relation {rel}"
                )

        # Group relations by type (to avoid dynamic f-string per relation)
        by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for rel in relations:
            # Sanitize relation type
            safe_rel = re.sub(r"[^a-zA-Z0-9_]", "", rel["relation"].upper())
            if not safe_rel:
                logger.warning(
                    "Skipping relation with invalid type", extra={"relation": rel}
                )
                continue

            # Validate against whitelist (security)
            if safe_rel not in ALLOWED_RELATIONS:
                raise ValueError(
                    f"Relation type '{safe_rel}' not in whitelist. "
                    f"Allowed: {', '.join(sorted(ALLOWED_RELATIONS))}"
                )

            by_type[safe_rel].append(rel)

        created_counts: Dict[str, int] = {}

        # Process each relation type separately
        for rel_type, batch_list in by_type.items():
            # First ensure all words/concepts exist
            for item in batch_list:
                self.word_manager.ensure_wort_und_konzept(item["subject"])
                self.word_manager.ensure_wort_und_konzept(item["object"])

            # Process in batches
            total_created = 0

            for i in range(0, len(batch_list), batch_size):
                chunk = batch_list[i : i + batch_size]

                # Prepare batch data (normalize to lowercase)
                batch_data = [
                    {
                        "subject": item["subject"].lower(),
                        "object": item["object"].lower(),
                        "confidence": item.get("confidence", 0.85),
                        "source_sentence": item.get("source_sentence", ""),
                    }
                    for item in chunk
                ]

                # Batch create with UNWIND
                query = f"""
                UNWIND $batch AS rel
                MATCH (s:Konzept {{name: rel.subject}})
                MATCH (o:Konzept {{name: rel.object}})
                MERGE (s)-[r:{rel_type}]->(o)
                ON CREATE SET
                    r.confidence = rel.confidence,
                    r.timestamp = datetime({{timezone: 'UTC'}}),
                    r.source_text = rel.source_sentence,
                    r.asserted_at = timestamp()
                ON MATCH SET
                    r.confidence = rel.confidence,
                    r.timestamp = datetime({{timezone: 'UTC'}})
                RETURN count(r) AS created
                """

                try:
                    with self.driver.session(database="neo4j") as session:
                        result = session.run(query, {"batch": batch_data})
                        count = result.single()["created"]
                        total_created += count
                        logger.debug(
                            f"Batch created {count} {rel_type} relations",
                            extra={"relation_type": rel_type, "batch_size": len(chunk)},
                        )
                except Exception as e:
                    logger.error(
                        f"Batch creation failed for {rel_type}",
                        extra={
                            "relation_type": rel_type,
                            "batch_size": len(chunk),
                            "error": str(e),
                        },
                        exc_info=True,
                    )
                    # Continue with next batch instead of failing completely
                    continue

            created_counts[rel_type] = total_created

            # Invalidate caches for affected subjects
            for item in batch_list:
                cache_key = (item["subject"].lower(), 0.0, None)
                self.cache_mgr.invalidate("netzwerk_facts", cache_key)

        logger.info(
            "Batch relations created",
            extra={
                "total_relations": len(relations),
                "relation_types": list(created_counts.keys()),
                "created_counts": created_counts,
            },
        )

        return created_counts

    def create_agent(
        self, agent_id: str, name: str, reasoning_capacity: int = 5
    ) -> bool:
        """
        Create an Agent node in Neo4j for Theory of Mind.

        Args:
            agent_id: Unique ID for the agent
            name: Name of the agent (e.g., "Alice", "Bob")
            reasoning_capacity: Max meta-level for reasoning (default: 5)

        Returns:
            True on success, False on error
        """
        if not self.driver:
            logger.error(
                "create_agent: Kein DB-Driver verfügbar", extra={"agent_id": agent_id}
            )
            return False

        try:
            with PerformanceLogger(
                logger.logger, "create_agent", agent_id=agent_id, name=name
            ):
                with self.driver.session(database="neo4j") as session:
                    with session.begin_transaction() as tx:
                        result = tx.run(
                            """
                            MERGE (a:Agent {id: $agent_id})
                            ON CREATE SET
                                a.name = $name,
                                a.reasoning_capacity = $reasoning_capacity,
                                a.created_at = timestamp()
                            RETURN a.id AS id
                            """,
                            agent_id=agent_id,
                            name=name,
                            reasoning_capacity=reasoning_capacity,
                        )
                        record = result.single()
                        tx.commit()

                        if not record or record["id"] != agent_id:
                            logger.error(
                                "create_agent: Verifikation fehlgeschlagen",
                                extra={"agent_id": agent_id},
                            )
                            return False

                        logger.info(
                            "Agent erfolgreich erstellt",
                            extra={"agent_id": agent_id, "name": name},
                        )
                        return True

        except Exception as e:
            logger.log_exception(
                e, "create_agent: Fehler", agent_id=agent_id, name=name
            )
            return False

    def add_belief(
        self, agent_id: str, proposition: str, certainty: float = 1.0
    ) -> bool:
        """
        Create a Belief node and connect it to an Agent via KNOWS relation.

        Args:
            agent_id: ID of the agent who has the belief
            proposition: The proposition/fact (e.g., "hund IS_A tier")
            certainty: Certainty/confidence (0.0 - 1.0, default: 1.0)

        Returns:
            True on success, False on error
        """
        if not self.driver:
            logger.error(
                "add_belief: Kein DB-Driver verfügbar", extra={"agent_id": agent_id}
            )
            return False

        try:
            with PerformanceLogger(
                logger.logger, "add_belief", agent_id=agent_id, proposition=proposition
            ):
                with self.driver.session(database="neo4j") as session:
                    with session.begin_transaction() as tx:
                        result = tx.run(
                            """
                            MATCH (a:Agent {id: $agent_id})
                            CREATE (b:Belief {
                                id: randomUUID(),
                                proposition: $proposition,
                                certainty: $certainty,
                                created_at: timestamp()
                            })
                            CREATE (a)-[:KNOWS]->(b)
                            RETURN b.id AS belief_id
                            """,
                            agent_id=agent_id,
                            proposition=proposition,
                            certainty=certainty,
                        )
                        record = result.single()
                        tx.commit()

                        if not record:
                            logger.error(
                                "add_belief: Belief konnte nicht erstellt werden",
                                extra={"agent_id": agent_id},
                            )
                            return False

                        logger.info(
                            "Belief erfolgreich erstellt",
                            extra={
                                "agent_id": agent_id,
                                "proposition": proposition,
                                "belief_id": record["belief_id"],
                            },
                        )
                        return True

        except Exception as e:
            logger.log_exception(
                e, "add_belief: Fehler", agent_id=agent_id, proposition=proposition
            )
            return False

    def add_meta_belief(
        self, observer_id: str, subject_id: str, proposition: str, meta_level: int
    ) -> bool:
        """
        Create a MetaBelief node for nested beliefs ("A knows that B knows P").

        Args:
            observer_id: ID of the agent who has the meta-belief (e.g., "Alice")
            subject_id: ID of the agent whose belief is referenced (e.g., "Bob")
            proposition: The proposition (e.g., "hund IS_A tier")
            meta_level: Nesting level (1 = "A knows B knows", 2 = "A knows B knows C knows", etc.)

        Returns:
            True on success, False on error
        """
        if not self.driver:
            logger.error(
                "add_meta_belief: Kein DB-Driver verfügbar",
                extra={"observer_id": observer_id, "subject_id": subject_id},
            )
            return False

        try:
            with PerformanceLogger(
                logger.logger,
                "add_meta_belief",
                observer_id=observer_id,
                subject_id=subject_id,
                proposition=proposition,
            ):
                with self.driver.session(database="neo4j") as session:
                    with session.begin_transaction() as tx:
                        result = tx.run(
                            """
                            MATCH (observer:Agent {id: $observer_id})
                            MATCH (subject:Agent {id: $subject_id})
                            CREATE (mb:MetaBelief {
                                id: randomUUID(),
                                proposition: $proposition,
                                meta_level: $meta_level,
                                created_at: timestamp()
                            })
                            CREATE (observer)-[:KNOWS_THAT]->(mb)
                            CREATE (mb)-[:ABOUT_AGENT]->(subject)
                            RETURN mb.id AS meta_belief_id
                            """,
                            observer_id=observer_id,
                            subject_id=subject_id,
                            proposition=proposition,
                            meta_level=meta_level,
                        )
                        record = result.single()
                        tx.commit()

                        if not record:
                            logger.error(
                                "add_meta_belief: MetaBelief konnte nicht erstellt werden",
                                extra={
                                    "observer_id": observer_id,
                                    "subject_id": subject_id,
                                },
                            )
                            return False

                        logger.info(
                            "MetaBelief erfolgreich erstellt",
                            extra={
                                "observer_id": observer_id,
                                "subject_id": subject_id,
                                "proposition": proposition,
                                "meta_level": meta_level,
                                "meta_belief_id": record["meta_belief_id"],
                            },
                        )
                        return True

        except Exception as e:
            logger.log_exception(
                e,
                "add_meta_belief: Fehler",
                observer_id=observer_id,
                subject_id=subject_id,
                proposition=proposition,
            )
            return False

    def get_node_count(self) -> int:
        """
        Get the total number of nodes in the graph.

        Counts all Wort and Konzept nodes for adaptive hyperparameter tuning.

        Returns:
            Number of nodes in the graph
        """
        if not self.driver:
            logger.error("get_node_count: Kein DB-Driver verfügbar")
            return 0

        try:
            with self.driver.session(database="neo4j") as session:
                result = session.run(
                    """
                    MATCH (n)
                    WHERE n:Wort OR n:Konzept
                    RETURN count(n) AS node_count
                    """
                )
                record = result.single()

                if not record:
                    logger.warning("get_node_count: Keine Nodes gefunden")
                    return 0

                count = record["node_count"]
                logger.debug(f"Graph enthält {count} Nodes")
                return count

        except Exception as e:
            logger.log_exception(e, "get_node_count: Fehler")
            return 0

    def assert_negation(
        self,
        subject: str,
        base_relation: str,
        object: str,
        source_sentence: Optional[str] = None,
    ) -> bool:
        """
        Create a negation relation (e.g., CANNOT_DO instead of CAPABLE_OF).

        This method handles negative assertions, which are critical for logic puzzles
        and exception-based reasoning (e.g., "penguins cannot fly").

        Mapping:
        - CAPABLE_OF -> CANNOT_DO
        - IS_A -> NOT_IS_A
        - HAS_PROPERTY -> HAS_NOT_PROPERTY
        - PART_OF -> NOT_PART_OF
        - LOCATED_IN -> NOT_LOCATED_IN

        Args:
            subject: Subject concept
            base_relation: Base relation type (without negation)
            object: Object concept
            source_sentence: Source sentence for provenance

        Returns:
            True if successfully created, False otherwise

        Example:
            >>> relation_mgr.assert_negation("pinguin", "CAPABLE_OF", "fliegen",
            ...                              "Ein Pinguin kann nicht fliegen")
            True
            # Creates: (pinguin)-[CANNOT_DO]->(fliegen)
        """
        # Mapping: Positive -> Negative Relation
        negation_map = {
            "CAPABLE_OF": "CANNOT_DO",
            "IS_A": "NOT_IS_A",
            "HAS_PROPERTY": "HAS_NOT_PROPERTY",
            "PART_OF": "NOT_PART_OF",
            "LOCATED_IN": "NOT_LOCATED_IN",
        }

        base_relation_upper = base_relation.upper()
        negated_relation = negation_map.get(base_relation_upper)

        if not negated_relation:
            logger.warning(
                f"assert_negation: Keine Negations-Regel für '{base_relation}', "
                f"verwende NOT_{base_relation_upper}",
                extra={
                    "subject": subject,
                    "base_relation": base_relation,
                    "object": object,
                },
            )
            negated_relation = f"NOT_{base_relation_upper}"

        # Create negation relation using existing assert_relation
        success = self.assert_relation(
            subject, negated_relation, object, source_sentence
        )

        if success:
            logger.info(
                "Negations-Relation erstellt",
                extra={
                    "subject": subject,
                    "negated_relation": negated_relation,
                    "object": object,
                    "base_relation": base_relation,
                },
            )
            # Note: Contradiction marking with positive relations could be added here
            # but is deferred to avoid complexity in Quick Win implementation

        return success

    def _is_safe_relation_type(self, relation_type: str) -> bool:
        """
        Validate relation type format to prevent Cypher injection.

        Relation types must be alphanumeric + underscore, starting with a letter.
        This prevents malicious queries via user-controlled relation types.

        Args:
            relation_type: Relation type to validate

        Returns:
            True if valid and safe, False otherwise

        Examples:
            >>> _is_safe_relation_type("IS_A")
            True
            >>> _is_safe_relation_type("HAS_PROPERTY")
            True
            >>> _is_safe_relation_type("'; DROP TABLE;")
            False
        """
        if not relation_type:
            return False

        # Must be alphanumeric + underscore, start with uppercase letter
        pattern = r"^[A-Z_][A-Z0-9_]*$"
        return bool(re.match(pattern, relation_type))

    def create_specialized_node(
        self,
        label: str,
        properties: Dict[str, Any],
        link_to_word: Optional[str] = None,
        relation_type: str = "EQUIVALENT_TO",
    ) -> bool:
        """
        Create a specialized node with custom label (e.g., NumberNode, Operation).

        Used by number language and spatial reasoning modules to create typed nodes
        beyond the standard Wort/Konzept schema.

        Args:
            label: Node label (e.g., "NumberNode", "Operation", "SpatialObject")
            properties: Dict of properties to set on the node (must include unique key)
            link_to_word: Optional lemma to link via relation (creates Wort if needed)
            relation_type: Relation type for word link (default: "EQUIVALENT_TO")

        Returns:
            True if successful, False otherwise

        Example:
            # Create NumberNode linked to word "fünf"
            success = relation_mgr.create_specialized_node(
                label="NumberNode",
                properties={"value": 5, "word": "fünf"},
                link_to_word="fünf",
                relation_type="EQUIVALENT_TO"
            )

        Note:
            - Label must be a valid Neo4j identifier (alphanumeric + underscore)
            - Properties dict should include a unique key for MERGE operation
            - If link_to_word is provided, a Wort node will be created if needed
        """
        if not self.driver:
            logger.error("create_specialized_node: Kein DB-Driver verfügbar")
            return False

        # Validate label format (alphanumeric + underscore only)
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", label):
            logger.error(
                f"create_specialized_node: Ungültiges Label '{label}' "
                f"(nur alphanumerisch und Underscore erlaubt)"
            )
            return False

        # Validate relation type if linking to word
        if link_to_word and not self._is_safe_relation_type(relation_type):
            logger.error(
                f"create_specialized_node: Ungültiger Relation-Type '{relation_type}'"
            )
            return False

        if not properties:
            logger.error(
                "create_specialized_node: Properties dict darf nicht leer sein"
            )
            return False

        try:
            with self._lock:
                with self.driver.session(database="neo4j") as session:
                    # Build MERGE clause from first property (assumed to be unique key)
                    # For NumberNode: {"value": 5} -> MERGE (n:NumberNode {value: $value})
                    # For Operation: {"name": "add"} -> MERGE (n:Operation {name: $name})
                    primary_key = list(properties.keys())[0]
                    merge_clause = (
                        f"MERGE (n:{label} {{{primary_key}: ${primary_key}}})"
                    )

                    # Build SET clause for additional properties
                    set_clause = "ON CREATE SET " + ", ".join(
                        f"n.{key} = ${key}" for key in properties.keys()
                    )

                    # Build full query
                    if link_to_word:
                        # Create Wort node and link
                        cypher = f"""
                        MERGE (w:Wort {{lemma: $lemma}})
                        {merge_clause}
                        {set_clause}
                        MERGE (w)-[r:{relation_type}]->(n)
                        ON CREATE SET r.confidence = 1.0, r.source = 'specialized_node'
                        RETURN n
                        """
                        params = {"lemma": link_to_word, **properties}
                    else:
                        # Just create specialized node
                        cypher = f"""
                        {merge_clause}
                        {set_clause}
                        RETURN n
                        """
                        params = properties

                    result = session.run(cypher, params)
                    record = result.single()

                    if record:
                        logger.info(
                            f"Specialized node created: {label} "
                            f"(properties: {properties}, linked_to: {link_to_word})"
                        )
                        return True
                    else:
                        logger.warning(
                            f"create_specialized_node: Kein Record zurückgegeben für {label}"
                        )
                        return False

        except Exception as e:
            logger.log_exception(
                e,
                "create_specialized_node: Fehler",
                label=label,
                properties=properties,
            )
            return False
