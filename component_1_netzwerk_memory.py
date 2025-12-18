"""
component_1_netzwerk_memory.py

FACADE for memory management subsystem.

This module provides a unified interface to the memory subsystem while delegating
to specialized modules:
    - component_1_episodic_memory.py: Episode storage and retrieval
    - component_1_inference_memory.py: Inference tracking and proof steps
    - component_1_hypothesis_memory.py: Hypothesis storage (abductive reasoning)

Backward Compatibility:
    This facade maintains the original KonzeptNetzwerkMemory API to ensure
    all existing code continues to work without modification. All methods
    delegate to the appropriate specialized module.

Architecture Refactoring (PHASE 3 - Task 8):
    Original file was 1,340 lines (168% over 800-line limit).
    Split into:
    - component_1_episodic_memory.py (458 lines)
    - component_1_inference_memory.py (594 lines)
    - component_1_hypothesis_memory.py (412 lines)
    - component_1_netzwerk_memory.py (this facade, ~100 lines)

Usage:
    # Original API still works:
    from neo4j import Driver
    from component_1_netzwerk_memory import KonzeptNetzwerkMemory

    driver = Driver("bolt://localhost:7687", auth=("neo4j", "password"))
    memory = KonzeptNetzwerkMemory(driver)

    # All original methods work as before:
    episode_id = memory.create_episode("ingestion", "Text", {"meta": "data"})
    memory.link_fact_to_episode("subject", "relation", "object", episode_id)
    episodes = memory.query_episodes_about("topic")

    inference_id = memory.create_inference_episode("backward_chaining", "Query?")
    step_id = memory.create_proof_step("goal", "fact", 1.0, 0)
    memory.link_inference_to_proof(inference_id, step_id)

    memory.store_hypothesis(hyp_id, "Explanation", ["obs"], "template", 0.8, {}, [])
    hypotheses = memory.query_hypotheses_about("topic")

Migration Path:
    While existing code can continue using KonzeptNetzwerkMemory, new code
    should import and use the specialized modules directly for better clarity:

    from component_1_episodic_memory import EpisodicMemory
    from component_1_inference_memory import InferenceMemory
    from component_1_hypothesis_memory import HypothesisMemory

    episodic = EpisodicMemory(driver)
    inference = InferenceMemory(driver)
    hypothesis = HypothesisMemory(driver)

Note: Follows CLAUDE.md standards - NO cp1252-unsafe Unicode, structured logging,
      comprehensive error handling, thread safety.
"""

from typing import Any, Callable, Dict, List, Optional

from neo4j import Driver

from component_1_episodic_memory import EpisodicMemory
from component_1_hypothesis_memory import HypothesisMemory
from component_1_inference_memory import InferenceMemory
from component_15_logging_config import get_logger

logger = get_logger(__name__)


class KonzeptNetzwerkMemory:
    """
    Facade for memory subsystem providing backward-compatible API.

    Delegates to specialized modules:
    - episodes: EpisodicMemory (learning event tracking)
    - inferences: InferenceMemory (reasoning process tracking)
    - hypotheses: HypothesisMemory (abductive reasoning outputs)

    Attributes:
        driver: Neo4j driver instance
        episodes: EpisodicMemory instance
        inferences: InferenceMemory instance
        hypotheses: HypothesisMemory instance

    Thread Safety:
        All delegated methods are thread-safe via Neo4jSessionMixin

    Example:
        memory = KonzeptNetzwerkMemory(driver)

        # Episode operations
        ep_id = memory.create_episode("ingestion", "Text content")
        memory.link_fact_to_episode("subj", "rel", "obj", ep_id)
        episodes = memory.query_episodes_about("topic")

        # Inference operations
        inf_id = memory.create_inference_episode("backward_chaining", "Query?")
        step_id = memory.create_proof_step("goal", "fact", 1.0, 0)
        memory.link_inference_to_proof(inf_id, step_id)

        # Hypothesis operations
        memory.store_hypothesis(hyp_id, "Explanation", ["obs"], "template", 0.8, {}, [])
        hypotheses = memory.query_hypotheses_about("topic")
    """

    def __init__(self, driver: Driver):
        """
        Initialize memory facade with Neo4j driver.

        Creates instances of all specialized memory modules.

        Args:
            driver: Neo4j driver instance

        Raises:
            Neo4jConnectionError: If driver is None or connection fails
        """
        self.driver = driver

        # Initialize specialized modules
        self.episodes = EpisodicMemory(driver)
        self.inferences = InferenceMemory(driver)
        self.hypotheses = HypothesisMemory(driver)

        logger.info("KonzeptNetzwerkMemory facade initialized")

    # ========================================================================
    # EPISODIC MEMORY (delegate to component_1_episodic_memory.py)
    # ========================================================================

    def create_episode(
        self, episode_type: str, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Create learning episode. Delegates to EpisodicMemory."""
        return self.episodes.create_episode(episode_type, content, metadata)

    def link_fact_to_episode(
        self, subject: str, relation: str, object: str, episode_id: str
    ) -> bool:
        """Link fact to episode. Delegates to EpisodicMemory."""
        return self.episodes.link_fact_to_episode(subject, relation, object, episode_id)

    def link_facts_to_episode_batch(
        self, facts: List[Dict[str, str]], episode_id: str
    ) -> int:
        """
        Link multiple facts to episode in batch (Quick Win #3).

        10-20x faster than individual link_fact_to_episode() calls.
        Delegates to EpisodicMemory.

        Args:
            facts: List of dicts with {subject, relation, object}
            episode_id: Episode ID

        Returns:
            Number of successfully linked facts
        """
        return self.episodes.link_facts_to_episode_batch(facts, episode_id)

    def query_episodes_about(self, topic: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Query episodes about topic. Delegates to EpisodicMemory."""
        return self.episodes.query_episodes_about(topic, limit)

    def query_all_episodes(
        self, episode_type: Optional[str] = None, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Query all episodes. Delegates to EpisodicMemory."""
        return self.episodes.query_all_episodes(episode_type, limit)

    def delete_episode(self, episode_id: str, cascade: bool = False) -> bool:
        """Delete episode. Delegates to EpisodicMemory."""
        return self.episodes.delete_episode(episode_id, cascade)

    # ========================================================================
    # INFERENCE MEMORY (delegate to component_1_inference_memory.py)
    # ========================================================================

    def create_inference_episode(
        self, inference_type: str, query: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Create inference episode. Delegates to InferenceMemory."""
        return self.inferences.create_inference_episode(inference_type, query, metadata)

    def create_proof_step(
        self,
        goal: str,
        method: str,
        confidence: float,
        depth: int,
        bindings: Optional[Dict[str, Any]] = None,
        parent_step_id: Optional[str] = None,
    ) -> Optional[str]:
        """Create proof step. Delegates to InferenceMemory."""
        return self.inferences.create_proof_step(
            goal, method, confidence, depth, bindings, parent_step_id
        )

    def link_inference_to_proof(
        self, inference_episode_id: str, proof_step_id: str
    ) -> bool:
        """Link inference to proof. Delegates to InferenceMemory."""
        return self.inferences.link_inference_to_proof(
            inference_episode_id, proof_step_id
        )

    def link_inference_to_facts(
        self, inference_episode_id: str, fact_ids: List[str]
    ) -> bool:
        """Link inference to facts. Delegates to InferenceMemory."""
        return self.inferences.link_inference_to_facts(inference_episode_id, fact_ids)

    def link_inference_to_rules(
        self, inference_episode_id: str, rule_ids: List[str]
    ) -> bool:
        """Link inference to rules. Delegates to InferenceMemory."""
        return self.inferences.link_inference_to_rules(inference_episode_id, rule_ids)

    def query_inference_history(
        self,
        topic: Optional[str] = None,
        inference_type: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Query inference history. Delegates to InferenceMemory."""
        return self.inferences.query_inference_history(topic, inference_type, limit)

    def get_proof_tree(self, root_step_id: str) -> Optional[Dict[str, Any]]:
        """Get proof tree. Delegates to InferenceMemory."""
        return self.inferences.get_proof_tree(root_step_id)

    def explain_inference(self, episode_id: str) -> str:
        """Explain inference. Delegates to InferenceMemory."""
        return self.inferences.explain_inference(episode_id)

    # ========================================================================
    # HYPOTHESIS MEMORY (delegate to component_1_hypothesis_memory.py)
    # ========================================================================

    def store_hypothesis(
        self,
        hypothesis_id: str,
        explanation: str,
        observations: List[str],
        strategy: str,
        confidence: float,
        scores: Dict[str, float],
        abduced_facts: List[Dict[str, Any]],
        sources: Optional[List[str]] = None,
        reasoning_trace: str = "",
    ) -> bool:
        """Store hypothesis. Delegates to HypothesisMemory."""
        return self.hypotheses.store_hypothesis(
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
        """Link hypothesis to observations. Delegates to HypothesisMemory."""
        return self.hypotheses.link_hypothesis_to_observations(
            hypothesis_id, observations
        )

    def link_hypothesis_to_concepts(
        self, hypothesis_id: str, concepts: List[str], ensure_wort_callback: Callable
    ) -> bool:
        """Link hypothesis to concepts. Delegates to HypothesisMemory."""
        return self.hypotheses.link_hypothesis_to_concepts(
            hypothesis_id, concepts, ensure_wort_callback
        )

    def query_hypotheses_about(
        self,
        topic: str,
        strategy: Optional[str] = None,
        min_confidence: float = 0.0,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Query hypotheses about topic. Delegates to HypothesisMemory."""
        return self.hypotheses.query_hypotheses_about(
            topic, strategy, min_confidence, limit
        )

    def get_best_hypothesis_for(
        self, topic: str, strategy: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get best hypothesis for topic. Delegates to HypothesisMemory."""
        return self.hypotheses.get_best_hypothesis_for(topic, strategy)

    def explain_hypothesis(self, hypothesis_id: str) -> str:
        """Explain hypothesis. Delegates to HypothesisMemory."""
        return self.hypotheses.explain_hypothesis(hypothesis_id)
