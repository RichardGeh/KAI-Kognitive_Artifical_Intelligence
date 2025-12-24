# component_1_netzwerk_core.py
"""
Core database operations and basic word/concept management (FACADE).

This module provides a unified interface to the Neo4j knowledge graph,
delegating to specialized modules for different responsibilities:
- component_1_netzwerk_database: Connection and schema management
- component_1_word_management: Word/concept CRUD operations
- component_1_relation_management: Relation assertions
- component_1_query_engine: Fact queries and semantic search

This facade maintains 100% backward compatibility with the original monolithic
component_1_netzwerk_core.py while providing a cleaner, more modular architecture.

Refactored as part of architecture improvement (Task 5, 2025-11-27).
"""

from typing import Any, Dict, List, Optional

from component_1_netzwerk_database import DatabaseConnection
from component_1_netzwerk_patterns import PatternManager
from component_1_netzwerk_utterances import UtteranceManager
from component_1_query_engine import QueryEngine
from component_1_relation_management import RelationManager
from component_1_word_management import INFO_TYPE_ALIASES, WordManager
from component_61_pattern_matcher import TemplatePatternMatcher

# Re-export for backward compatibility
__all__ = [
    "KonzeptNetzwerkCore",
    "INFO_TYPE_ALIASES",
    "sanitize_for_logging",
]


def sanitize_for_logging(text: str) -> str:
    """
    Remove cp1252-incompatible Unicode characters for Windows logging.

    Args:
        text: Input text with potentially problematic Unicode characters

    Returns:
        Sanitized text safe for cp1252 logging
    """
    if not text:
        return text

    try:
        # Try encoding/decoding with 'replace' for incompatible characters
        return text.encode("cp1252", errors="replace").decode("cp1252")
    except Exception:
        # Fallback: Keep only ASCII
        return text.encode("ascii", errors="replace").decode("ascii")


class KonzeptNetzwerkCore:
    """
    Core functionality for Neo4j knowledge graph management (FACADE).

    This class provides a unified interface to the knowledge graph by delegating
    to specialized modules:
    - DatabaseConnection: Connection and schema management
    - WordManager: Word/concept CRUD operations
    - RelationManager: Relation assertions
    - QueryEngine: Fact queries and semantic search

    Performance Optimization:
    - TTL cache for fact queries (5 minutes TTL, maxsize=500)
    - Cache for known words (10 minutes TTL, maxsize=100)
    - All caching managed via CacheManager

    Thread Safety:
        All operations are thread-safe through the underlying modules.

    Attributes:
        db: DatabaseConnection for connection management
        words: WordManager for word operations
        relations: RelationManager for relation operations
        queries: QueryEngine for query operations
        utterances: UtteranceManager for pattern discovery utterances
        patterns: PatternManager for pattern discovery patterns
    """

    def __init__(
        self,
        uri: str = "bolt://127.0.0.1:7687",
        user: str = "neo4j",
        password: str = "password",
        preprocessor=None,
    ):
        """
        Initialize the knowledge graph facade.

        Args:
            uri: Neo4j connection URI
            user: Database username
            password: Database password
            preprocessor: Optional LinguisticPreprocessor for text normalization

        Raises:
            Neo4jConnectionError: If connection cannot be established
        """
        # Initialize database connection
        self.db = DatabaseConnection(uri, user, password)

        # Initialize specialized modules
        self.words = WordManager(self.db.driver)
        self.relations = RelationManager(self.db.driver, self.words)
        self.queries = QueryEngine(self.db.driver, self.words)

        # Initialize pattern discovery modules
        self.utterances = UtteranceManager(self.db.driver, preprocessor)

        # Initialize pattern matcher first (Part 2: Pattern Matching)
        self.pattern_matcher = TemplatePatternMatcher(self)

        # Initialize patterns with pattern_matcher reference for cache invalidation
        self.patterns = PatternManager(
            self.db.driver, pattern_matcher=self.pattern_matcher
        )

        # Expose driver for backward compatibility
        self.driver = self.db.driver

    def close(self):
        """
        Close the Neo4j connection and clean up caches.
        """
        self.db.close()
        # Caches are managed by CacheManager and will be cleaned up automatically

    def health_check(self) -> bool:
        """
        Check if the Neo4j connection is working.

        Returns:
            True if connection OK, False on error
        """
        return self.db.health_check()

    # ============================================================================
    # Word Management (delegated to WordManager)
    # ============================================================================

    def ensure_wort_und_konzept(self, lemma: str) -> bool:
        """Ensure a word and its concept exist in the graph."""
        return self.words.ensure_wort_und_konzept(lemma)

    def set_wort_attribut(
        self, lemma: str, attribut_name: str, attribut_wert: Any
    ) -> bool:
        """Set an attribute directly on the :Wort node."""
        return self.words.set_wort_attribut(lemma, attribut_name, attribut_wert)

    def add_information_zu_wort(
        self, lemma: str, info_typ: str, info_inhalt: str
    ) -> Dict[str, Any]:
        """Add information to a word (meaning, synonym, etc.)."""
        return self.words.add_information_zu_wort(lemma, info_typ, info_inhalt)

    def get_details_fuer_wort(self, lemma: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a word."""
        return self.words.get_details_fuer_wort(lemma)

    def get_all_known_words(self) -> List[str]:
        """Get all known words (lemmas) from the graph."""
        return self.words.get_all_known_words()

    def word_exists(self, word: str) -> bool:
        """Check if a word exists in the knowledge graph."""
        return self.words.word_exists(word)

    def get_word_frequency(self, word: str) -> Dict[str, int]:
        """Calculate frequency metrics for a word."""
        return self.words.get_word_frequency(word)

    def get_normalized_word_frequency(self, word: str) -> float:
        """Get normalized word frequency (0.0 - 1.0)."""
        return self.words.get_normalized_word_frequency(word)

    # ============================================================================
    # Relation Management (delegated to RelationManager)
    # ============================================================================

    def assert_relation(
        self,
        subject: str,
        relation: str,
        object: str,
        source_sentence: Optional[str] = None,
        confidence: float = 0.85,
    ) -> bool:
        """Create an asserted relationship between two concepts."""
        return self.relations.assert_relation(
            subject, relation, object, source_sentence, confidence
        )

    def assert_negation(
        self,
        subject: str,
        base_relation: str,
        object: str,
        source_sentence: Optional[str] = None,
    ) -> bool:
        """
        Create a negation relation (e.g., CANNOT_DO instead of CAPABLE_OF).

        This facade method delegates to RelationManager.assert_negation().
        Critical for logic puzzles and exception-based reasoning.

        Args:
            subject: Subject concept
            base_relation: Base relation type (e.g., "CAPABLE_OF")
            object: Object concept
            source_sentence: Source sentence for provenance

        Returns:
            True if successfully created, False otherwise

        Example:
            >>> netzwerk.assert_negation("pinguin", "CAPABLE_OF", "fliegen",
            ...                          "Ein Pinguin kann nicht fliegen")
            True
        """
        return self.relations.assert_negation(
            subject, base_relation, object, source_sentence
        )

    def create_agent(
        self, agent_id: str, name: str, reasoning_capacity: int = 5
    ) -> bool:
        """Create an Agent node in Neo4j for Theory of Mind."""
        return self.relations.create_agent(agent_id, name, reasoning_capacity)

    def add_belief(
        self, agent_id: str, proposition: str, certainty: float = 1.0
    ) -> bool:
        """Create a Belief node and connect it to an Agent via KNOWS relation."""
        return self.relations.add_belief(agent_id, proposition, certainty)

    def add_meta_belief(
        self, observer_id: str, subject_id: str, proposition: str, meta_level: int
    ) -> bool:
        """Create a MetaBelief node for nested beliefs."""
        return self.relations.add_meta_belief(
            observer_id, subject_id, proposition, meta_level
        )

    def get_node_count(self) -> int:
        """Get the total number of nodes in the graph."""
        return self.relations.get_node_count()

    # ============================================================================
    # Query Operations (delegated to QueryEngine)
    # ============================================================================

    def query_graph_for_facts(
        self, topic: str, min_confidence: float = 0.0, sort_by_confidence: bool = False
    ) -> Dict[str, List[str]]:
        """
        Query the graph for all known outgoing facts (relationships) for a topic.

        Args:
            topic: The concept to query
            min_confidence: Minimum confidence threshold (0.0-1.0). Default 0.0 = no filter
            sort_by_confidence: Sort results by confidence DESC. Default False

        Returns:
            Dict with {relation_type: [target_concepts]}
        """
        return self.queries.query_graph_for_facts(
            topic, min_confidence, sort_by_confidence
        )

    def query_inverse_relations(
        self, topic: str, relation_type: Optional[str] = None
    ) -> Dict[str, List[str]]:
        """Query the graph for all incoming relations for a concept."""
        return self.queries.query_inverse_relations(topic, relation_type)

    def query_graph_for_facts_with_confidence(
        self, topic: str, min_confidence: float = 0.0
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Query the graph for all outgoing facts AND their confidence values and timestamps.

        Args:
            topic: The concept to query
            min_confidence: Minimum confidence threshold (0.0-1.0). Default 0.0 = no filter

        Returns:
            Dict with {relation_type: [{"target": str, "confidence": float, "timestamp": str}]}
        """
        return self.queries.query_graph_for_facts_with_confidence(topic, min_confidence)

    def query_inverse_relations_with_confidence(
        self, topic: str, relation_type: Optional[str] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Query the graph for incoming relations AND their confidence values."""
        return self.queries.query_inverse_relations_with_confidence(
            topic, relation_type
        )

    def query_facts_with_synonyms(self, topic: str) -> Dict[str, Any]:
        """Robust synonym-aware fact search."""
        return self.queries.query_facts_with_synonyms(topic)

    def find_similar_words(
        self,
        query_word: str,
        embedding_service=None,
        similarity_threshold: float = 0.75,
        max_results: int = 3,
    ) -> List[Dict[str, Any]]:
        """Find similar words for typo tolerance using semantic embeddings."""
        return self.queries.find_similar_words(
            query_word, embedding_service, similarity_threshold, max_results
        )

    def query_semantic_neighbors(
        self,
        lemma: str,
        allowed_relations: Optional[List[str]] = None,
        min_confidence: float = 0.0,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Find semantic neighbors with bidirectional relationship search.

        Used by resonance engine for spreading activation algorithm.

        Args:
            lemma: Word to find neighbors for
            allowed_relations: Optional list of relation types to filter
            min_confidence: Minimum confidence threshold (0.0-1.0)
            limit: Maximum number of neighbors to return

        Returns:
            List of dicts with neighbor info (neighbor, relation_type, confidence)
        """
        return self.queries.query_semantic_neighbors(
            lemma, allowed_relations, min_confidence, limit
        )

    def query_transitive_path(
        self,
        subject: Optional[str],
        predicate: str,
        object: Optional[str],
        max_hops: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Find transitive paths between concepts (multi-hop reasoning).

        Used by logic engine for backward chaining.

        Args:
            subject: Starting concept (None = match any)
            predicate: Relation type to follow
            object: Ending concept (None = match any)
            max_hops: Maximum path length (1-5)

        Returns:
            List of dicts with path info (subject, object, hops, path)
        """
        return self.queries.query_transitive_path(subject, predicate, object, max_hops)

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
            properties: Dict of properties to set on the node
            link_to_word: Optional lemma to link via relation
            relation_type: Relation type for word link (default: "EQUIVALENT_TO")

        Returns:
            True if successful, False otherwise
        """
        return self.relations.create_specialized_node(
            label, properties, link_to_word, relation_type
        )

    # ============================================================================
    # Production Rule Management (delegated to QueryEngine)
    # ============================================================================

    def create_production_rule(
        self,
        name: str,
        category: str,
        utility: float = 1.0,
        specificity: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Persist a production rule in Neo4j."""
        return self.queries.create_production_rule(
            name, category, utility, specificity, metadata
        )

    def get_production_rules(
        self,
        category: Optional[str] = None,
        min_utility: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Load production rules from Neo4j."""
        return self.queries.get_production_rules(category, min_utility, limit)

    def update_rule_stats(
        self,
        rule_name: str,
        applied: bool = True,
        success: Optional[bool] = None,
    ) -> bool:
        """Update the statistics of a production rule."""
        return self.queries.update_rule_stats(rule_name, applied, success)

    def get_production_rule_statistics(self) -> Dict[str, Any]:
        """Get statistics about all production rules."""
        return self.queries.get_production_rule_statistics()

    # ============================================================================
    # Batch Operations (delegated to specialized modules)
    # ============================================================================

    def batch_assert_relations(
        self, relations: List[Dict[str, Any]], batch_size: int = 100
    ) -> Dict[str, int]:
        """
        Batch-create relations using UNWIND (5-10x faster than individual calls).

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
        """
        return self.relations.batch_assert_relations(relations, batch_size)

    # ============================================================================
    # Cache Management (delegated to specialized modules)
    # ============================================================================

    def clear_cache(self, cache_type: Optional[str] = None):
        """
        Invalidate caches.

        Args:
            cache_type: Optional - 'facts', 'words', or None for all caches
        """
        if cache_type == "facts" or cache_type is None:
            self.queries.clear_cache("facts")

        if cache_type == "words" or cache_type is None:
            # WordManager handles its own cache invalidation
            pass

    def get_cache_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics about the caches.

        Returns:
            Dict with cache statistics for different cache types
        """
        stats = {}

        # Get stats from QueryEngine (fact cache)
        query_stats = self.queries.get_cache_stats()
        stats.update(query_stats)

        # Note: WordManager cache stats are not exposed in original API
        # but are managed via CacheManager

        return stats

    # ============================================================================
    # Pattern Discovery System (delegated to UtteranceManager and PatternManager)
    # ============================================================================

    def create_utterance(
        self, text: str, embedding: List[float], user_id: Optional[str] = None
    ) -> str:
        """
        Create Utterance node for pattern discovery.

        Args:
            text: Original user input text
            embedding: 384-dimensional embedding vector
            user_id: Optional user identifier

        Returns:
            Utterance ID (UUID)
        """
        return self.utterances.create_utterance(text, embedding, user_id)

    def get_recent_utterances(
        self, limit: int = 100, archived: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Retrieve recent utterances efficiently using temporal index.

        Args:
            limit: Maximum number of utterances (1-10000)
            archived: Whether to retrieve archived utterances

        Returns:
            List of utterance dicts
        """
        return self.utterances.get_recent_utterances(limit, archived)

    def archive_old_utterances(self, days_threshold: int = 30) -> int:
        """
        Archive old utterances for scalability.

        Args:
            days_threshold: Archive utterances older than N days

        Returns:
            Number of utterances archived
        """
        return self.utterances.archive_old_utterances(days_threshold)

    def create_token(
        self, surface: str, lemma: str, pos: str, utterance_id: str, idx: int
    ) -> str:
        """
        Create Token node linked to Utterance.

        Args:
            surface: Original word form
            lemma: Base form
            pos: Part-of-speech tag
            utterance_id: UUID of parent utterance
            idx: Token position (0-based)

        Returns:
            Token ID (UUID)
        """
        return self.utterances.create_token(surface, lemma, pos, utterance_id, idx)

    def get_tokens_for_utterance(self, utterance_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve tokens in sequence order.

        Args:
            utterance_id: UUID of utterance

        Returns:
            List of token dicts in sequence order
        """
        return self.utterances.get_tokens_for_utterance(utterance_id)

    def get_tokens_for_utterances_batch(
        self, utterance_ids: List[str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Fetch tokens for multiple utterances in single query (batch optimization).

        Args:
            utterance_ids: List of utterance UUIDs

        Returns:
            Dict mapping utterance_id to list of token dicts
        """
        return self.utterances.get_tokens_for_utterances_batch(utterance_ids)

    def create_pattern(
        self, name: str, pattern_type: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create Pattern node for pattern discovery.

        Args:
            name: Pattern name
            pattern_type: Type - "seed", "learned", or "template"
            metadata: Optional additional properties

        Returns:
            Pattern ID (UUID)
        """
        return self.patterns.create_pattern(name, pattern_type, metadata)

    def update_pattern_stats(
        self, pattern_id: str, support_increment: int, new_precision: float
    ):
        """
        Update pattern statistics atomically.

        Args:
            pattern_id: UUID of pattern
            support_increment: How much to increase support
            new_precision: New precision value (0.0-1.0)
        """
        return self.patterns.update_pattern_stats(
            pattern_id, support_increment, new_precision
        )

    def get_all_patterns(
        self, type_filter: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Retrieve patterns with optional type filter.

        Args:
            type_filter: Optional - filter by pattern type
            limit: Maximum number of patterns (1-10000)

        Returns:
            List of pattern dicts ordered by quality
        """
        return self.patterns.get_all_patterns(type_filter, limit)

    def batch_create_pattern_items(
        self, pattern_id: str, items: List[Dict[str, Any]]
    ) -> int:
        """
        Create multiple PatternItems using UNWIND batch operation.

        Args:
            pattern_id: UUID of parent pattern
            items: List of item dicts

        Returns:
            Number of items created
        """
        return self.patterns.batch_create_pattern_items(pattern_id, items)

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

        Args:
            pattern_id: Pattern UUID
            idx: Position in pattern sequence
            kind: "LITERAL" or "SLOT"
            literal_value: Required for LITERAL kind
            slot_id: Required for SLOT kind

        Returns:
            PatternItem ID (UUID)
        """
        return self.patterns.create_pattern_item(
            pattern_id, idx, kind, literal_value, slot_id
        )

    def create_slot(
        self,
        slot_type: str,
        allowed_values: Optional[List[str]] = None,
        min_count: int = 1,
        max_count: int = 1,
    ) -> str:
        """
        Create Slot node for pattern discovery.

        Args:
            slot_type: Slot type identifier
            allowed_values: Optional list of initially allowed lemmas
            min_count: Minimum token count
            max_count: Maximum token count

        Returns:
            Slot ID (UUID)
        """
        return self.patterns.create_slot(
            slot_type, allowed_values, min_count, max_count
        )

    def update_slot_allowed(self, slot_id: str, lemma: str, count_increment: int = 1):
        """
        Update or create ALLOWS relationship with count.

        Args:
            slot_id: UUID of slot
            lemma: Lemma value to allow
            count_increment: How much to increment count
        """
        return self.patterns.update_slot_allowed(slot_id, lemma, count_increment)

    def match_utterance_to_pattern(
        self, utterance_id: str, pattern_id: str, score: float
    ):
        """
        Record MATCHED relationship between Utterance and Pattern.

        Args:
            utterance_id: UUID of utterance
            pattern_id: UUID of pattern
            score: Match score (0.0-1.0)
        """
        return self.patterns.match_utterance_to_pattern(utterance_id, pattern_id, score)

    # ============================================================================
    # Pattern Matching (delegated to TemplatePatternMatcher)
    # ============================================================================

    def match_utterance_patterns(self, utterance_id: str):
        """
        Match utterance against all patterns using hybrid scoring.

        Combines template-based structural matching with embedding-based
        semantic similarity for robust pattern recognition.

        Args:
            utterance_id: UUID of utterance to match

        Returns:
            List of (pattern_id, score) tuples sorted by score (highest first)

        Example:
            >>> matches = netzwerk.match_utterance_patterns(utterance_id)
            >>> for pattern_id, score in matches[:5]:  # Top 5
            ...     print(f"Pattern {pattern_id[:8]}: {score:.2f}")
        """
        return self.pattern_matcher.match_utterance(utterance_id)

    def invalidate_pattern_candidate_cache(self):
        """
        Invalidate pattern candidate cache.

        CRITICAL: Call after pattern creation/update to ensure fresh results.
        This should be called when:
        - New patterns are created
        - Pattern statistics are updated
        - Slot allowed values are modified
        """
        self.pattern_matcher._invalidate_candidate_cache()

    # ============================================================================
    # Pattern Discovery System - Part 3 Extensions (delegated to managers)
    # ============================================================================

    def count_utterances(self, archived: bool = False) -> int:
        """
        Count utterances with optional archival filter.

        Args:
            archived: Whether to count archived utterances (default: False)

        Returns:
            Number of utterances matching filter
        """
        return self.utterances.count_utterances(archived)

    def update_pattern_centroid(self, pattern_id: str, centroid: List[float]):
        """
        Update pattern embedding centroid.

        Args:
            pattern_id: Pattern UUID
            centroid: 384-dimensional normalized embedding vector
        """
        self.patterns.update_pattern_centroid(pattern_id, centroid)

    def get_pattern(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve single pattern with all properties.

        Args:
            pattern_id: Pattern UUID

        Returns:
            Pattern dict or None if not found
        """
        return self.patterns.get_pattern(pattern_id)

    def get_pattern_items(self, pattern_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve pattern items ordered by index.

        Args:
            pattern_id: Pattern UUID

        Returns:
            List of PatternItem dicts in sequence order
        """
        return self.patterns.get_pattern_items(pattern_id)

    def get_utterance(self, utterance_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve single utterance with all properties.

        Args:
            utterance_id: Utterance UUID

        Returns:
            Utterance dict or None if not found
        """
        return self.utterances.get_utterance(utterance_id)
