# component_1_netzwerk.py
"""
Facade module for KonzeptNetzwerk - combines all sub-modules.

This is the main entry point for the knowledge graph functionality.
All existing code importing from this module will continue to work.

Modular structure:
- component_1_netzwerk_core: Database connection, word/concept management, relations
- component_1_netzwerk_patterns: Pattern learning and extraction rules
- component_1_netzwerk_memory: Episodic memory and hypothesis storage
- component_1_netzwerk_word_usage: Word usage tracking and context management
"""

from typing import Any, Dict, List, Optional

from component_1_netzwerk_common_words import CommonWordsManager
from component_1_netzwerk_core import INFO_TYPE_ALIASES, KonzeptNetzwerkCore
from component_1_netzwerk_feedback import KonzeptNetzwerkFeedback
from component_1_netzwerk_memory import KonzeptNetzwerkMemory
from component_1_netzwerk_patterns import KonzeptNetzwerkPatterns
from component_1_netzwerk_production_rules import KonzeptNetzwerkProductionRules
from component_1_netzwerk_word_usage import KonzeptNetzwerkWordUsage


class KonzeptNetzwerk:
    """
    Unified interface to the knowledge graph.

    Combines functionality from:
    - KonzeptNetzwerkCore: Database operations, word/concept management
    - KonzeptNetzwerkPatterns: Pattern learning and extraction rules
    - KonzeptNetzwerkMemory: Episodic memory and hypothesis storage
    - KonzeptNetzwerkWordUsage: Word usage tracking and context management
    """

    def __init__(
        self,
        uri: str = "bolt://127.0.0.1:7687",
        user: str = "neo4j",
        password: str = "password",
    ):
        # Initialize core (creates driver and connection)
        self._core = KonzeptNetzwerkCore(uri, user, password)

        # Initialize patterns, memory, word usage, feedback, production rules, and common words with the driver
        self._patterns = KonzeptNetzwerkPatterns(self._core.driver)
        self._memory = KonzeptNetzwerkMemory(self._core.driver)
        self._word_usage = KonzeptNetzwerkWordUsage(self._core.driver)
        self._feedback = KonzeptNetzwerkFeedback(self._core.driver)
        self._production_rules = KonzeptNetzwerkProductionRules(
            self._core.driver
        )  # PHASE 9
        self._common_words = CommonWordsManager(self._core.driver)

        # Create constraints for word usage, feedback, and common words
        self._word_usage._create_constraints()
        self._feedback._create_constraints()

    @property
    def driver(self):
        """Get the database driver."""
        return self._core.driver

    @driver.setter
    def driver(self, value):
        """Set the database driver (updates all sub-modules)."""
        self._core.driver = value
        self._patterns.driver = value
        self._memory.driver = value
        self._word_usage.driver = value
        self._feedback.driver = value
        self._production_rules.driver = value
        self._common_words.driver = value

    def close(self):
        """Close the database connection."""
        self._core.close()

    # ========== CORE METHODS ==========
    # Delegate to KonzeptNetzwerkCore

    def ensure_wort_und_konzept(self, lemma: str) -> bool:
        """Create or retrieve word and concept nodes."""
        return self._core.ensure_wort_und_konzept(lemma)

    def set_wort_attribut(
        self, lemma: str, attribut_name: str, attribut_wert: Any
    ) -> None:
        """Set an attribute on a word node."""
        return self._core.set_wort_attribut(lemma, attribut_name, attribut_wert)

    def add_information_zu_wort(
        self, lemma: str, info_typ: str, info_inhalt: str
    ) -> Dict[str, Any]:
        """Add information (definition/synonym) to a word."""
        return self._core.add_information_zu_wort(lemma, info_typ, info_inhalt)

    def get_details_fuer_wort(self, lemma: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a word."""
        return self._core.get_details_fuer_wort(lemma)

    def assert_relation(
        self,
        subject: str,
        relation: str,
        object: str,
        source_sentence: Optional[str] = None,
        confidence: float = 0.85,
    ) -> bool:
        """Create a relation between two concepts."""
        return self._core.assert_relation(
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
        Create a negation relation (Quick Win #5).

        Delegates to KonzeptNetzwerkCore.assert_negation().
        Critical for logic puzzles and exception-based reasoning.
        """
        return self._core.assert_negation(
            subject, base_relation, object, source_sentence
        )

    def query_graph_for_facts(
        self, topic: str, min_confidence: float = 0.0, sort_by_confidence: bool = False
    ) -> Dict[str, List[str]]:
        """
        Query all facts about a topic.

        Args:
            topic: The concept to query
            min_confidence: Minimum confidence threshold (0.0-1.0). Default 0.0 = no filter
            sort_by_confidence: Sort results by confidence DESC. Default False

        Returns:
            Dict with {relation_type: [target_concepts]}
        """
        return self._core.query_graph_for_facts(
            topic, min_confidence, sort_by_confidence
        )

    def query_inverse_relations(
        self, topic: str, relation_type: Optional[str] = None
    ) -> Dict[str, List[str]]:
        """Query all incoming relations for a topic (inverse/backward relations)."""
        return self._core.query_inverse_relations(topic, relation_type)

    def query_graph_for_facts_with_confidence(
        self, topic: str, min_confidence: float = 0.0
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Query all facts about a topic WITH confidence values.

        Args:
            topic: The concept to query
            min_confidence: Minimum confidence threshold (0.0-1.0). Default 0.0 = no filter

        Returns:
            Dict with {relation_type: [{"target": str, "confidence": float, "timestamp": str}]}
        """
        return self._core.query_graph_for_facts_with_confidence(topic, min_confidence)

    def query_inverse_relations_with_confidence(
        self, topic: str, relation_type: Optional[str] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Query all incoming relations for a topic WITH confidence values."""
        return self._core.query_inverse_relations_with_confidence(topic, relation_type)

    def query_facts_with_synonyms(self, topic: str) -> Dict[str, Any]:
        """Query facts including synonyms."""
        return self._core.query_facts_with_synonyms(topic)

    def get_all_known_words(self) -> List[str]:
        """Get all known words from the graph."""
        return self._core.get_all_known_words()

    def word_exists(self, lemma: str) -> bool:
        """Check if a word exists in the graph (fast check)."""
        return self._core.word_exists(lemma)

    def find_similar_words(
        self,
        query_word: str,
        embedding_service=None,
        similarity_threshold: float = 0.75,
        max_results: int = 3,
    ) -> List[Dict[str, Any]]:
        """Find similar words for typo tolerance."""
        return self._core.find_similar_words(
            query_word, embedding_service, similarity_threshold, max_results
        )

    def get_word_frequency(self, word: str) -> Dict[str, int]:
        """Get frequency metrics for a word (out/in/total degree)."""
        return self._core.get_word_frequency(word)

    def get_normalized_word_frequency(self, word: str) -> float:
        """Get normalized word frequency (0.0 - 1.0) using sigmoid."""
        return self._core.get_normalized_word_frequency(word)

    def query_semantic_neighbors(
        self,
        lemma: str,
        allowed_relations: Optional[List[str]] = None,
        min_confidence: float = 0.0,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Bidirectional neighbor search."""
        return self._core.query_semantic_neighbors(
            lemma, allowed_relations, min_confidence, limit
        )

    def query_transitive_path(
        self, subject: str, predicate: str, object_node: str, max_hops: int = 3
    ) -> Optional[List[Dict[str, Any]]]:
        """Multi-hop path finding."""
        return self._core.query_transitive_path(
            subject, predicate, object_node, max_hops
        )

    # ========== PATTERN METHODS ==========
    # Delegate to KonzeptNetzwerkPatterns

    def create_extraction_rule(self, relation_type: str, regex_pattern: str) -> bool:
        """Create or update an extraction rule."""
        return self._patterns.create_extraction_rule(relation_type, regex_pattern)

    def get_all_pattern_prototypes(self, category: str = None) -> List[Dict[str, Any]]:
        """Get all pattern prototypes, optionally filtered by category."""
        return self._patterns.get_all_pattern_prototypes(category)

    def create_pattern_prototype(
        self, initial_vector: List[float], category: str
    ) -> Optional[str]:
        """Create a new pattern prototype."""
        return self._patterns.create_pattern_prototype(initial_vector, category)

    def update_pattern_prototype(
        self,
        prototype_id: str,
        new_centroid: List[float],
        new_variance: List[float],
        new_count: int,
    ) -> None:
        """Update an existing pattern prototype."""
        return self._patterns.update_pattern_prototype(
            prototype_id, new_centroid, new_variance, new_count
        )

    def link_prototype_to_rule(self, prototype_id: str, relation_type: str) -> bool:
        """Link a pattern prototype to an extraction rule."""
        return self._patterns.link_prototype_to_rule(prototype_id, relation_type)

    def add_lexical_trigger(self, lemma: str) -> bool:
        """Add a lexical trigger word."""
        # Pass the ensure_wort_und_konzept method as callback
        return self._patterns.add_lexical_trigger(lemma, self.ensure_wort_und_konzept)

    def get_lexical_triggers(self) -> List[str]:
        """Get all lexical trigger words."""
        return self._patterns.get_lexical_triggers()

    def get_rule_for_prototype(self, prototype_id: str) -> Optional[Dict[str, str]]:
        """Get the extraction rule for a prototype."""
        return self._patterns.get_rule_for_prototype(prototype_id)

    def get_all_extraction_rules(self) -> List[Dict[str, str]]:
        """Get all extraction rules."""
        return self._patterns.get_all_extraction_rules()

    # ========== MEMORY METHODS ==========
    # Delegate to KonzeptNetzwerkMemory

    def create_episode(
        self, episode_type: str, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Create a learning episode."""
        return self._memory.create_episode(episode_type, content, metadata)

    def batch_create_episodes(
        self, episodes: List[Dict[str, Any]], batch_size: int = 100
    ) -> List[str]:
        """
        Batch-create episodes using UNWIND (10x faster than individual calls).

        Args:
            episodes: List of dicts with keys:
                - episode_type (str): Type of episode
                - content (str): Episode content
                - metadata (dict, optional): Additional metadata
            batch_size: Batch size per UNWIND query (default: 100)

        Returns:
            List of created episode IDs (UUIDs)
        """
        return self._memory.batch_create_episodes(episodes, batch_size)

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
        return self._core.batch_assert_relations(relations, batch_size)

    def link_fact_to_episode(
        self, subject: str, relation: str, object: str, episode_id: str
    ) -> bool:
        """Link a fact to an episode."""
        return self._memory.link_fact_to_episode(subject, relation, object, episode_id)

    def link_facts_to_episode_batch(
        self, facts: List[Dict[str, str]], episode_id: str
    ) -> int:
        """Link multiple facts to episode in batch (10-20x faster)."""
        return self._memory.link_facts_to_episode_batch(facts, episode_id)

    def query_episodes_about(self, topic: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Find episodes about a topic."""
        return self._memory.query_episodes_about(topic, limit)

    def query_all_episodes(
        self, episode_type: Optional[str] = None, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get all episodes, optionally filtered by type."""
        return self._memory.query_all_episodes(episode_type, limit)

    def delete_episode(self, episode_id: str, cascade: bool = False) -> bool:
        """Delete an episode."""
        return self._memory.delete_episode(episode_id, cascade)

    def create_inference_episode(
        self, inference_type: str, query: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Create an inference episode for reasoning tracking."""
        return self._memory.create_inference_episode(inference_type, query, metadata)

    def create_proof_step(
        self,
        goal: str,
        method: str,
        confidence: float,
        depth: int,
        bindings: Optional[Dict[str, Any]] = None,
        parent_step_id: Optional[str] = None,
    ) -> Optional[str]:
        """Create a proof step."""
        return self._memory.create_proof_step(
            goal, method, confidence, depth, bindings, parent_step_id
        )

    def link_inference_to_proof(
        self, inference_episode_id: str, proof_step_id: str
    ) -> bool:
        """Link an inference episode to a proof step."""
        return self._memory.link_inference_to_proof(inference_episode_id, proof_step_id)

    def link_inference_to_facts(
        self, inference_episode_id: str, fact_ids: List[str]
    ) -> bool:
        """Link an inference episode to facts."""
        return self._memory.link_inference_to_facts(inference_episode_id, fact_ids)

    def link_inference_to_rules(
        self, inference_episode_id: str, rule_ids: List[str]
    ) -> bool:
        """Link an inference episode to rules."""
        return self._memory.link_inference_to_rules(inference_episode_id, rule_ids)

    def query_inference_history(
        self,
        topic: Optional[str] = None,
        inference_type: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Query inference history."""
        return self._memory.query_inference_history(topic, inference_type, limit)

    def get_proof_tree(self, root_step_id: str) -> Optional[Dict[str, Any]]:
        """Get the proof tree for a reasoning step."""
        return self._memory.get_proof_tree(root_step_id)

    def explain_inference(self, episode_id: str) -> str:
        """Generate a natural language explanation of an inference."""
        return self._memory.explain_inference(episode_id)

    def store_hypothesis(
        self,
        hypothesis_id: str,
        explanation: str,
        observations: List[str],
        strategy: str,
        confidence: float,
        scores: Dict[str, float],
        abduced_facts: List[Dict[str, Any]],
        sources: List[str] = None,
        reasoning_trace: str = "",
    ) -> bool:
        """Store a hypothesis from abductive reasoning."""
        return self._memory.store_hypothesis(
            hypothesis_id,
            explanation,
            observations,
            strategy,
            confidence,
            scores,
            abduced_facts,
            sources,
            reasoning_trace,
        )

    def link_hypothesis_to_observations(
        self, hypothesis_id: str, observations: List[str]
    ) -> bool:
        """Link a hypothesis to observations."""
        return self._memory.link_hypothesis_to_observations(hypothesis_id, observations)

    def link_hypothesis_to_concepts(
        self, hypothesis_id: str, concepts: List[str]
    ) -> bool:
        """Link a hypothesis to concepts."""
        # Pass the ensure_wort_und_konzept method as callback
        return self._memory.link_hypothesis_to_concepts(
            hypothesis_id, concepts, self.ensure_wort_und_konzept
        )

    def query_hypotheses_about(
        self,
        topic: str,
        strategy: Optional[str] = None,
        min_confidence: float = 0.0,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Find hypotheses about a topic."""
        return self._memory.query_hypotheses_about(
            topic, strategy, min_confidence, limit
        )

    def get_best_hypothesis_for(
        self, topic: str, strategy: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get the best hypothesis for a topic."""
        return self._memory.get_best_hypothesis_for(topic, strategy)

    def explain_hypothesis(self, hypothesis_id: str) -> str:
        """Generate a natural language explanation of a hypothesis."""
        return self._memory.explain_hypothesis(hypothesis_id)

    # ========== WORD USAGE METHODS ==========
    # Delegate to KonzeptNetzwerkWordUsage

    def add_word_connection(
        self,
        word1_lemma: str,
        word2_lemma: str,
        distance: int = 1,
        direction: str = "before",
    ) -> bool:
        """Create or update CONNECTION edge between two words."""
        return self._word_usage.add_word_connection(
            word1_lemma, word2_lemma, distance, direction
        )

    def get_word_connections(
        self, word_lemma: str, direction: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get all CONNECTION edges for a word."""
        return self._word_usage.get_word_connections(word_lemma, direction)

    def add_usage_context(
        self,
        word_lemma: str,
        fragment: str,
        word_position: int,
        fragment_type: str = "window",
    ) -> bool:
        """Create or update UsageContext node for a word."""
        return self._word_usage.add_usage_context(
            word_lemma, fragment, word_position, fragment_type
        )

    def get_usage_contexts(
        self, word_lemma: str, min_count: int = 1, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get all UsageContext nodes for a word."""
        return self._word_usage.get_usage_contexts(word_lemma, min_count, limit)

    def calculate_fragment_similarity(self, fragment1: str, fragment2: str) -> float:
        """Calculate similarity between two text fragments."""
        return self._word_usage.calculate_fragment_similarity(fragment1, fragment2)

    # ========== FEEDBACK METHODS ==========
    # Delegate to KonzeptNetzwerkFeedback

    def store_typo_feedback(
        self,
        original_input: str,
        suggested_word: str,
        actual_word: str,
        user_accepted: bool,
        confidence: float,
        correction_reason: str = "user_correction",
    ) -> Optional[str]:
        """Store feedback for typo correction."""
        return self._feedback.store_typo_feedback(
            original_input,
            suggested_word,
            actual_word,
            user_accepted,
            confidence,
            correction_reason,
        )

    def get_typo_feedback_for_input(self, original_input: str) -> List[Dict[str, Any]]:
        """Get all feedback entries for a specific input."""
        return self._feedback.get_typo_feedback_for_input(original_input)

    def get_negative_examples(self, suggested_word: str) -> List[Dict[str, Any]]:
        """Get all negative examples for a suggested word."""
        return self._feedback.get_negative_examples(suggested_word)

    def update_pattern_quality(
        self, pattern_type: str, pattern_key: str, success: bool
    ) -> bool:
        """Update quality score of a pattern based on feedback."""
        return self._feedback.update_pattern_quality(pattern_type, pattern_key, success)

    def get_pattern_quality(
        self, pattern_type: str, pattern_key: str
    ) -> Optional[float]:
        """Get quality weight for a pattern."""
        return self._feedback.get_pattern_quality(pattern_type, pattern_key)

    # ========== THEORY OF MIND METHODS ==========
    # Delegate to KonzeptNetzwerkCore

    def create_agent(
        self, agent_id: str, name: str, reasoning_capacity: int = 5
    ) -> bool:
        """Create an Agent node for Theory of Mind."""
        return self._core.create_agent(agent_id, name, reasoning_capacity)

    def add_belief(
        self, agent_id: str, proposition: str, certainty: float = 1.0
    ) -> bool:
        """Add a Belief node with KNOWS relation to an agent."""
        return self._core.add_belief(agent_id, proposition, certainty)

    def add_meta_belief(
        self, observer_id: str, subject_id: str, proposition: str, meta_level: int
    ) -> bool:
        """Add a MetaBelief node for nested beliefs (A knows B knows P)."""
        return self._core.add_meta_belief(
            observer_id, subject_id, proposition, meta_level
        )

    # ========== PRODUCTION RULE METHODS (PHASE 9) ==========
    # Delegate to KonzeptNetzwerkProductionRules

    def create_production_rule(
        self,
        name: str,
        category: str,
        condition_code: str,
        action_code: str,
        utility: float = 1.0,
        specificity: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Create or update a production rule in Neo4j."""
        return self._production_rules.create_production_rule(
            name, category, condition_code, action_code, utility, specificity, metadata
        )

    def get_production_rule(self, name: str) -> Optional[Dict[str, Any]]:
        """Load a production rule from Neo4j."""
        return self._production_rules.get_production_rule(name)

    def get_all_production_rules(self) -> List[Dict[str, Any]]:
        """Load all production rules from Neo4j."""
        return self._production_rules.get_all_production_rules()

    def update_production_rule_stats(
        self,
        name: str,
        application_count: Optional[int] = None,
        success_count: Optional[int] = None,
        last_applied: Optional[Any] = None,
        force_sync: bool = False,
    ) -> bool:
        """Update statistics for a production rule."""
        return self._production_rules.update_production_rule_stats(
            name, application_count, success_count, last_applied, force_sync
        )

    def query_production_rules(
        self,
        category: Optional[str] = None,
        min_utility: Optional[float] = None,
        max_utility: Optional[float] = None,
        min_application_count: Optional[int] = None,
        order_by: str = "priority",
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Query production rules with filtering and sorting."""
        return self._production_rules.query_production_rules(
            category, min_utility, max_utility, min_application_count, order_by, limit
        )

    def get_production_rule_statistics(self) -> Dict[str, Any]:
        """Get aggregated statistics about all production rules."""
        return self._production_rules.get_rule_statistics()

    # ========== COMMON WORDS METHODS ==========
    # Delegate to CommonWordsManager

    def add_common_word(
        self, word: str, category: str, confidence: float = 1.0
    ) -> bool:
        """Add a common word (stop word) to the database."""
        return self._common_words.add_common_word(word, category, confidence)

    def add_common_words_batch(self, words_dict: dict) -> int:
        """Add multiple common words in a batch."""
        return self._common_words.add_common_words_batch(words_dict)

    def get_common_words(self, category: Optional[str] = None):
        """Get all common words (optionally filtered by category)."""
        return self._common_words.get_common_words(category)

    def is_common_word(self, word: str) -> bool:
        """Check if a word is a common word."""
        return self._common_words.is_common_word(word)

    def remove_common_word(self, word: str) -> bool:
        """Remove a common word from the database."""
        return self._common_words.remove_common_word(word)

    def get_common_words_statistics(self) -> dict:
        """Get statistics about common words."""
        return self._common_words.get_statistics()


# Export INFO_TYPE_ALIASES for backward compatibility
__all__ = ["KonzeptNetzwerk", "INFO_TYPE_ALIASES"]
