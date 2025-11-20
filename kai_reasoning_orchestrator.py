# kai_reasoning_orchestrator.py
"""
Reasoning Orchestrator für KAI - Hybrid Reasoning System

Koordiniert mehrere Reasoning Engines und kombiniert deren Ergebnisse
für robustere, uncertainty-aware Antworten.

Features:
- Hybrid Reasoning (Logic + Probabilistic + Graph + Abductive)
- Weighted Confidence Fusion
- Unified Proof Tree Generation
- Fallback-Strategien
- Result Aggregation

Architecture:
    1. Fast Path: Direct fact lookup
    2. Deterministic Reasoning: Logic Engine + Graph Traversal
    3. Probabilistic Enhancement: Uncertainty quantification
    4. Abductive Fallback: Hypothesis generation
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from component_1_netzwerk import KonzeptNetzwerk
from component_9_logik_engine import Engine, Goal
from component_9_logik_engine import ProofStep as LogicProofStep

# Import Unified Proof Explanation System
try:
    from component_17_proof_explanation import ProofStep as UnifiedProofStep
    from component_17_proof_explanation import (
        ProofTree,
        StepType,
        create_proof_tree_from_logic_engine,
        merge_proof_trees,
    )

    PROOF_SYSTEM_AVAILABLE = True
except ImportError:
    PROOF_SYSTEM_AVAILABLE = False

# Import Resonance Engine
try:
    from component_44_resonance_engine import AdaptiveResonanceEngine

    RESONANCE_ENGINE_AVAILABLE = True
except ImportError:
    RESONANCE_ENGINE_AVAILABLE = False

# Import Meta-Learning Engine
try:
    pass

    META_LEARNING_AVAILABLE = True
except ImportError:
    META_LEARNING_AVAILABLE = False

# Import Self-Evaluator
try:
    pass

    SELF_EVALUATION_AVAILABLE = True
except ImportError:
    SELF_EVALUATION_AVAILABLE = False

logger = logging.getLogger(__name__)


class ReasoningStrategy(Enum):
    """Available reasoning strategies"""

    DIRECT_FACT = "direct_fact"
    LOGIC_ENGINE = "logic_engine"
    GRAPH_TRAVERSAL = "graph_traversal"
    PROBABILISTIC = "probabilistic"
    ABDUCTIVE = "abductive"
    COMBINATORIAL = "combinatorial"  # Strategic/combinatorial reasoning
    SPATIAL = "spatial"  # Spatial reasoning (grids, shapes, positions)
    RESONANCE = "resonance"  # Spreading activation with resonance amplification
    CONSTRAINT = "constraint"  # Constraint satisfaction (logic puzzles, CSP)


@dataclass
class ReasoningResult:
    """
    Result from a single reasoning strategy.

    Attributes:
        strategy: Which strategy was used
        success: Whether reasoning succeeded
        confidence: Confidence score (0.0-1.0)
        inferred_facts: Dictionary of inferred facts
        proof_tree: Unified ProofTree (optional)
        proof_trace: Text explanation
        metadata: Additional strategy-specific data
        is_hypothesis: Whether result is abductive hypothesis
    """

    strategy: ReasoningStrategy
    success: bool
    confidence: float
    inferred_facts: Dict[str, List[str]] = field(default_factory=dict)
    proof_tree: Optional[ProofTree] = None
    proof_trace: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_hypothesis: bool = False
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AggregatedResult:
    """
    Aggregated result from multiple reasoning strategies.

    Combines evidence from multiple sources using weighted fusion.
    """

    combined_confidence: float
    inferred_facts: Dict[str, List[str]]
    merged_proof_tree: Optional[ProofTree]
    strategies_used: List[ReasoningStrategy]
    individual_results: List[ReasoningResult]
    explanation: str
    is_hypothesis: bool = False


class ReasoningOrchestrator:
    """
    Main orchestrator for hybrid reasoning.

    Coordinates multiple reasoning engines and aggregates results
    using weighted confidence fusion.
    """

    def __init__(
        self,
        netzwerk: KonzeptNetzwerk,
        logic_engine: Engine,
        graph_traversal,
        working_memory,
        signals,
        probabilistic_engine=None,
        abductive_engine=None,
        combinatorial_reasoner=None,
        spatial_reasoner=None,
        resonance_engine=None,
        meta_learning_engine=None,
        self_evaluator=None,
        config_path: Optional[str] = None,
    ):
        """
        Initialize Reasoning Orchestrator.

        Args:
            netzwerk: KonzeptNetzwerk instance
            logic_engine: Logic Engine instance
            graph_traversal: GraphTraversal instance
            working_memory: WorkingMemory instance
            signals: KaiSignals for UI updates
            probabilistic_engine: ProbabilisticEngine (optional)
            abductive_engine: AbductiveEngine (optional)
            combinatorial_reasoner: CombinatorialReasoner (optional)
            spatial_reasoner: SpatialReasoner (optional)
            resonance_engine: ResonanceEngine (optional)
            meta_learning_engine: MetaLearningEngine (optional)
            self_evaluator: SelfEvaluator (optional)
            config_path: Path to YAML configuration file (optional)
        """
        self.netzwerk = netzwerk
        self.logic_engine = logic_engine
        self.graph_traversal = graph_traversal
        self.working_memory = working_memory
        self.signals = signals
        self.probabilistic_engine = probabilistic_engine
        self.abductive_engine = abductive_engine
        self.combinatorial_reasoner = combinatorial_reasoner
        self.spatial_reasoner = spatial_reasoner
        self.resonance_engine = resonance_engine
        self.meta_learning_engine = meta_learning_engine
        self.self_evaluator = self_evaluator

        # Default Configuration
        self.enable_hybrid = True  # Combine Logic + Probabilistic
        self.min_confidence_threshold = 0.4  # Below this: try next strategy
        self.probabilistic_enhancement = (
            True  # Enhance deterministic with probabilistic
        )
        self.aggregation_method = (
            "noisy_or"  # noisy_or | weighted_avg | max | dempster_shafer
        )
        self.enable_parallel_execution = False  # Parallel strategy execution
        self.enable_result_caching = True  # Cache results for repeated queries

        # Strategy weights for weighted_avg aggregation
        self.strategy_weights = {
            ReasoningStrategy.DIRECT_FACT: 0.35,
            ReasoningStrategy.LOGIC_ENGINE: 0.18,
            ReasoningStrategy.GRAPH_TRAVERSAL: 0.14,
            ReasoningStrategy.RESONANCE: 0.12,  # Spreading activation
            ReasoningStrategy.CONSTRAINT: 0.11,  # CSP solving for logic puzzles
            ReasoningStrategy.SPATIAL: 0.10,
            ReasoningStrategy.COMBINATORIAL: 0.07,
            ReasoningStrategy.PROBABILISTIC: 0.03,
            ReasoningStrategy.ABDUCTIVE: 0.01,
        }

        # Load configuration from YAML if provided
        if config_path:
            self._load_config(config_path)

        # Result cache (LRU)
        from cachetools import LRUCache

        self._result_cache = (
            LRUCache(maxsize=100) if self.enable_result_caching else None
        )

        logger.info(
            f"ReasoningOrchestrator initialisiert: "
            f"aggregation={self.aggregation_method}, "
            f"parallel={self.enable_parallel_execution}, "
            f"caching={self.enable_result_caching}"
        )

    def query_with_hybrid_reasoning(
        self,
        topic: str,
        relation_type: str = "IS_A",
        strategies: Optional[List[ReasoningStrategy]] = None,
    ) -> Optional[AggregatedResult]:
        """
        Main entry point for hybrid reasoning.

        Tries multiple strategies and aggregates results:
        1. Fast Path (direct facts)
        2. Deterministic (Logic + Graph)
        3. Probabilistic Enhancement
        4. Abductive Fallback

        Args:
            topic: The topic to reason about
            relation_type: Type of relation to find
            strategies: Which strategies to use (None = all)

        Returns:
            AggregatedResult with combined evidence or None
        """
        logger.info(f"[Hybrid Reasoning] Query: {topic} ({relation_type})")

        # Check cache first
        cache_key = f"{topic}:{relation_type}:{str(strategies)}"
        if self._result_cache is not None and cache_key in self._result_cache:
            logger.debug(f"[Cache Hit] Returning cached result for {topic}")
            return self._result_cache[cache_key]

        if strategies is None:
            strategies = [
                ReasoningStrategy.DIRECT_FACT,
                ReasoningStrategy.GRAPH_TRAVERSAL,
                ReasoningStrategy.LOGIC_ENGINE,
                ReasoningStrategy.RESONANCE,
                ReasoningStrategy.CONSTRAINT,
                ReasoningStrategy.SPATIAL,
                ReasoningStrategy.PROBABILISTIC,
                ReasoningStrategy.ABDUCTIVE,
            ]

        results = []

        # Stage 1: Direct Fact Lookup (Fast Path)
        if ReasoningStrategy.DIRECT_FACT in strategies:
            direct_result = self._try_direct_fact_lookup(topic, relation_type)
            if direct_result and direct_result.success:
                if direct_result.confidence >= 0.95:
                    # High confidence direct fact - return immediately
                    logger.info(
                        "[Hybrid Reasoning] [OK] High-confidence direct fact found"
                    )
                    aggregated = self._create_aggregated_result([direct_result])
                    # Cache early exit result
                    if self._result_cache is not None:
                        self._result_cache[cache_key] = aggregated
                    return aggregated
                results.append(direct_result)

        # Stage 2: Deterministic Reasoning (Graph + Logic)
        deterministic_results = []

        if (
            self.enable_parallel_execution
            and len(
                [
                    s
                    for s in strategies
                    if s
                    in [
                        ReasoningStrategy.GRAPH_TRAVERSAL,
                        ReasoningStrategy.LOGIC_ENGINE,
                    ]
                ]
            )
            > 1
        ):
            # Parallel execution of independent strategies
            logger.debug("[Parallel Execution] Running Graph + Logic in parallel")

            from concurrent.futures import ThreadPoolExecutor, as_completed

            futures = {}
            with ThreadPoolExecutor(max_workers=2) as executor:
                if ReasoningStrategy.GRAPH_TRAVERSAL in strategies:
                    futures[
                        executor.submit(self._try_graph_traversal, topic, relation_type)
                    ] = "graph"
                if ReasoningStrategy.LOGIC_ENGINE in strategies:
                    futures[
                        executor.submit(self._try_logic_engine, topic, relation_type)
                    ] = "logic"

                for future in as_completed(futures):
                    strategy_name = futures[future]
                    try:
                        result = future.result()
                        if result and result.success:
                            deterministic_results.append(result)
                            logger.debug(
                                f"[Parallel Execution] {strategy_name} completed successfully"
                            )
                    except Exception as e:
                        logger.warning(
                            f"[Parallel Execution] {strategy_name} failed: {e}"
                        )
        else:
            # Sequential execution (default)
            if ReasoningStrategy.GRAPH_TRAVERSAL in strategies:
                graph_result = self._try_graph_traversal(topic, relation_type)
                if graph_result and graph_result.success:
                    deterministic_results.append(graph_result)

            if ReasoningStrategy.LOGIC_ENGINE in strategies:
                logic_result = self._try_logic_engine(topic, relation_type)
                if logic_result and logic_result.success:
                    deterministic_results.append(logic_result)

            if ReasoningStrategy.SPATIAL in strategies:
                spatial_result = self._try_spatial_reasoning(topic, relation_type)
                if spatial_result and spatial_result.success:
                    deterministic_results.append(spatial_result)

            if ReasoningStrategy.CONSTRAINT in strategies:
                constraint_result = self._try_constraint_solving(topic, relation_type)
                if constraint_result and constraint_result.success:
                    deterministic_results.append(constraint_result)

            if ReasoningStrategy.RESONANCE in strategies:
                resonance_result = self._try_resonance(topic, relation_type)
                if resonance_result and resonance_result.success:
                    deterministic_results.append(resonance_result)

        results.extend(deterministic_results)

        # Stage 3: Probabilistic Enhancement
        if (
            ReasoningStrategy.PROBABILISTIC in strategies
            and self.probabilistic_enhancement
        ):
            if deterministic_results:
                # Enhance deterministic results with probabilistic reasoning
                prob_result = self._enhance_with_probabilistic(
                    topic, relation_type, deterministic_results
                )
                if prob_result and prob_result.success:
                    results.append(prob_result)
            else:
                # Try standalone probabilistic reasoning
                prob_result = self._try_probabilistic(topic, relation_type)
                if prob_result and prob_result.success:
                    results.append(prob_result)

        # Check if we have sufficient results
        if results:
            aggregated = self._create_aggregated_result(results)
            if aggregated.combined_confidence >= self.min_confidence_threshold:
                logger.info(
                    f"[Hybrid Reasoning] [OK] Success with {len(results)} strategies "
                    f"(confidence: {aggregated.combined_confidence:.2f})"
                )
                # Cache result
                if self._result_cache is not None:
                    self._result_cache[cache_key] = aggregated
                return aggregated

        # Stage 4: Abductive Fallback (Hypothesis Generation)
        if ReasoningStrategy.ABDUCTIVE in strategies and self.abductive_engine:
            logger.info("[Hybrid Reasoning] Falling back to Abductive Reasoning")
            abd_result = self._try_abductive(topic, relation_type)
            if abd_result and abd_result.success:
                results.append(abd_result)
                aggregated = self._create_aggregated_result(results)
                # Cache result
                if self._result_cache is not None:
                    self._result_cache[cache_key] = aggregated
                return aggregated

        # No successful reasoning
        if results:
            # Return partial results
            aggregated = self._create_aggregated_result(results)
            # Cache partial results too
            if self._result_cache is not None:
                self._result_cache[cache_key] = aggregated
            return aggregated

        logger.info("[Hybrid Reasoning] [X] No strategy succeeded")
        return None

    def query_with_meta_learning(
        self,
        topic: str,
        relation_type: str = "IS_A",
        query_text: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
    ) -> Optional[AggregatedResult]:
        """
        Meta-Learning-basierte Strategy-Auswahl und Reasoning.

        Phase 4.1 Integration:
        1. MetaLearningEngine wählt beste Strategy basierend auf Query-Pattern
        2. Falls Resonance gewählt:
           - AdaptiveResonanceEngine mit auto-tuning
           - Self-Evaluation des Resultats
           - Fallback bei low confidence
        3. Alle Strategies werden für Meta-Learning getrackt

        Args:
            topic: Das Reasoning-Topic
            relation_type: Typ der Relation (IS_A, HAS_PROPERTY, etc.)
            query_text: Optional Query-Text für besseres Pattern-Matching
            context: Optional Context-Dict
            max_retries: Maximum Fallback-Versuche bei low confidence

        Returns:
            AggregatedResult oder None
        """
        if not self.meta_learning_engine:
            logger.warning(
                "[Meta-Learning] MetaLearningEngine not available, falling back to hybrid reasoning"
            )
            return self.query_with_hybrid_reasoning(topic, relation_type)

        logger.info(f"[Meta-Learning] Query: {topic} ({relation_type})")

        # Prepare context
        if context is None:
            context = {}
        context["topic"] = topic
        context["relation_type"] = relation_type

        # Use query_text or construct from topic
        query_for_ml = query_text or f"Was ist ein {topic}?"

        # Track attempted strategies to avoid infinite loops
        attempted_strategies = []
        retry_count = 0

        while retry_count < max_retries:
            # 1. Meta-Learning: Select best strategy
            available_strategies = self._get_available_strategy_names(
                exclude=attempted_strategies
            )

            if not available_strategies:
                logger.warning("[Meta-Learning] No more strategies available for retry")
                break

            selected_strategy, ml_confidence = (
                self.meta_learning_engine.select_best_strategy(
                    query=query_for_ml,
                    context=context,
                    available_strategies=available_strategies,
                )
            )

            logger.info(
                f"[Meta-Learning] Selected strategy: '{selected_strategy}' "
                f"(ML confidence: {ml_confidence:.2f}, attempt {retry_count + 1}/{max_retries})"
            )

            attempted_strategies.append(selected_strategy)

            # Track start time for performance monitoring
            import time

            start_time = time.time()

            # 2. Execute selected strategy
            result = None

            if selected_strategy == "resonance":
                result = self._execute_resonance_strategy(
                    topic, relation_type, context, query_for_ml
                )
            else:
                # Execute via existing hybrid reasoning for other strategies
                strategy_enum = self._map_strategy_name_to_enum(selected_strategy)
                if strategy_enum:
                    result = self._execute_single_strategy(
                        topic, relation_type, strategy_enum
                    )

            response_time = time.time() - start_time

            # 3. Self-Evaluation (if available and result exists)
            if result and self.self_evaluator:
                eval_result = self._evaluate_result_quality(
                    result, query_for_ml, topic, context
                )

                # FIX 2024-11: recommendation kann String oder Enum sein
                recommendation_value = (
                    eval_result.recommendation.value
                    if hasattr(eval_result.recommendation, "value")
                    else str(eval_result.recommendation)
                )

                logger.info(
                    f"[Self-Evaluation] Score: {eval_result.overall_score:.2f}, "
                    f"Recommendation: {recommendation_value}"
                )

                # Check if we should retry with different strategy
                if recommendation_value == "retry_different_strategy":
                    logger.info(
                        f"[Self-Evaluation] Recommends retry, attempting different strategy "
                        f"(attempt {retry_count + 1})"
                    )

                    # Record failed attempt
                    self._record_strategy_usage(
                        selected_strategy,
                        query_for_ml,
                        result,
                        response_time,
                        success=False,
                        context=context,
                    )

                    retry_count += 1
                    continue  # Try next strategy

                # Adjust confidence if suggested
                if eval_result.confidence_adjusted and eval_result.suggested_confidence:
                    logger.info(
                        f"[Self-Evaluation] Adjusting confidence: "
                        f"{result.combined_confidence:.2f} → {eval_result.suggested_confidence:.2f}"
                    )
                    result.combined_confidence = eval_result.suggested_confidence

            # 4. Record successful strategy usage
            if result:
                self._record_strategy_usage(
                    selected_strategy,
                    query_for_ml,
                    result,
                    response_time,
                    success=True,
                    context=context,
                )

                logger.info(
                    f"[Meta-Learning] Success with '{selected_strategy}' "
                    f"(confidence: {result.combined_confidence:.2f})"
                )
                return result
            else:
                # Strategy failed to produce result
                self._record_strategy_usage(
                    selected_strategy,
                    query_for_ml,
                    None,
                    response_time,
                    success=False,
                    context=context,
                )

                retry_count += 1

        # All retries exhausted
        logger.warning(
            f"[Meta-Learning] All retries exhausted ({retry_count} attempts)"
        )
        return None

    def _execute_resonance_strategy(
        self,
        topic: str,
        relation_type: str,
        context: Dict[str, Any],
        query_text: str,
    ) -> Optional[AggregatedResult]:
        """
        Executes Resonance Strategy with Adaptive Hyperparameters.

        Special handling for Resonance:
        1. Use AdaptiveResonanceEngine with auto-tuning
        2. Self-evaluation of activation map
        3. Enhanced proof tree generation

        Args:
            topic: Topic to reason about
            relation_type: Relation type
            context: Context dict
            query_text: Query text

        Returns:
            AggregatedResult or None
        """
        if not self.resonance_engine:
            logger.warning("[Resonance] ResonanceEngine not available")
            return None

        logger.info("[Resonance] Executing with adaptive hyperparameters")

        # 1. Auto-tune hyperparameters (if AdaptiveResonanceEngine)
        if isinstance(self.resonance_engine, AdaptiveResonanceEngine):
            try:
                self.resonance_engine.auto_tune_hyperparameters()
                logger.debug("[Resonance] Auto-tuning completed")
            except Exception as e:
                logger.warning(f"[Resonance] Auto-tuning failed: {e}")

        # 2. Execute resonance activation
        try:
            allowed_relations = context.get("allowed_relations", [relation_type])

            activation_map = self.resonance_engine.activate_concept(
                start_word=topic,
                query_context=context,
                allowed_relations=allowed_relations,
            )

            if not activation_map or activation_map.concepts_activated <= 1:
                logger.info("[Resonance] No significant activation")
                return None

            # 3. Extract inferred facts from activation map
            inferred_facts = {}
            top_concepts = activation_map.get_top_concepts(n=20)

            for concept, activation in top_concepts:
                if concept == topic:
                    continue  # Skip start concept

                # Find paths to determine relation type
                paths = activation_map.get_paths_to(concept)
                if paths:
                    best_path = max(paths, key=lambda p: p.confidence_product)
                    if best_path.relations:
                        rel_type = best_path.relations[0]
                        if rel_type not in inferred_facts:
                            inferred_facts[rel_type] = []
                        if concept not in inferred_facts[rel_type]:
                            inferred_facts[rel_type].append(concept)

            # Calculate confidence
            confidence = min(activation_map.max_activation, 1.0)

            # Boost for resonance points
            if activation_map.resonance_points:
                resonance_boost = len(activation_map.resonance_points) * 0.05
                confidence = min(confidence + resonance_boost, 1.0)

            # 4. Create ReasoningResult
            result = ReasoningResult(
                strategy=ReasoningStrategy.RESONANCE,
                success=True,
                confidence=confidence,
                inferred_facts=inferred_facts,
                proof_tree=None,  # Will be created below
                proof_trace=self.resonance_engine.get_activation_summary(
                    activation_map
                ),
                metadata={
                    "concepts_activated": activation_map.concepts_activated,
                    "waves_executed": activation_map.waves_executed,
                    "resonance_points": len(activation_map.resonance_points),
                    "max_activation": activation_map.max_activation,
                },
            )

            # 5. Create Proof Tree
            if PROOF_SYSTEM_AVAILABLE:
                proof_tree = ProofTree(query=query_text)

                # Root step: Activation summary
                summary = self.resonance_engine.get_activation_summary(activation_map)
                root_step = UnifiedProofStep(
                    step_id="resonance_activation",
                    step_type=StepType.INFERENCE,
                    inputs=[topic],
                    output=f"{activation_map.concepts_activated} Konzepte aktiviert",
                    confidence=confidence,
                    explanation_text=summary,
                    source_component="adaptive_resonance_engine",
                    metadata={
                        "waves": activation_map.waves_executed,
                        "resonance_points": len(activation_map.resonance_points),
                    },
                )
                proof_tree.add_root_step(root_step)

                result.proof_tree = proof_tree

            # 6. Create AggregatedResult
            aggregated = AggregatedResult(
                combined_confidence=confidence,
                inferred_facts=inferred_facts,
                merged_proof_tree=result.proof_tree,
                strategies_used=[ReasoningStrategy.RESONANCE],
                individual_results=[result],
                explanation=f"Resonance Strategy: {activation_map.concepts_activated} Konzepte aktiviert, "
                f"{len(activation_map.resonance_points)} Resonanz-Punkte",
                is_hypothesis=False,
            )

            return aggregated

        except Exception as e:
            logger.error(f"[Resonance] Execution failed: {e}", exc_info=True)
            return None

    def _try_direct_fact_lookup(
        self, topic: str, relation_type: str
    ) -> Optional[ReasoningResult]:
        """
        Fast path: Direct fact lookup in knowledge graph.
        """
        logger.debug(f"[Direct Fact] Querying: {topic}")

        try:
            facts = self.netzwerk.query_graph_for_facts(topic)

            if relation_type in facts and facts[relation_type]:
                # Found direct facts
                inferred_facts = {relation_type: facts[relation_type]}

                # Create simple proof tree
                proof_tree = None
                if PROOF_SYSTEM_AVAILABLE:
                    proof_tree = ProofTree(query=f"Was ist ein {topic}?")

                    for obj in facts[relation_type][:3]:  # Limit to 3
                        step = UnifiedProofStep(
                            step_id=f"direct_{topic}_{obj}",
                            step_type=StepType.FACT_MATCH,
                            inputs=[topic],
                            output=f"{topic} {relation_type} {obj}",
                            confidence=1.0,
                            explanation_text=f"Direkter Fakt in Wissensbasis: {topic} -> {obj}",
                            source_component="direct_fact_lookup",
                        )
                        proof_tree.add_root_step(step)

                return ReasoningResult(
                    strategy=ReasoningStrategy.DIRECT_FACT,
                    success=True,
                    confidence=1.0,
                    inferred_facts=inferred_facts,
                    proof_tree=proof_tree,
                    proof_trace=f"Direkte Fakten gefunden: {len(facts[relation_type])} Einträge",
                    metadata={"num_facts": len(facts[relation_type])},
                )

            return None

        except Exception as e:
            logger.warning(f"[Direct Fact] Fehler: {e}")
            return None

    def _try_graph_traversal(
        self, topic: str, relation_type: str
    ) -> Optional[ReasoningResult]:
        """
        Graph-based multi-hop reasoning.
        """
        logger.debug(f"[Graph Traversal] Topic: {topic}")

        try:
            paths = self.graph_traversal.find_transitive_relations(
                topic, relation_type, max_depth=5
            )

            if paths:
                # Extract inferred facts
                inferred_facts = {relation_type: []}
                for path in paths:
                    target = path.nodes[-1]
                    if target not in inferred_facts[relation_type]:
                        inferred_facts[relation_type].append(target)

                # Create proof tree
                proof_tree = None
                if PROOF_SYSTEM_AVAILABLE:
                    proof_tree = ProofTree(query=f"Was ist ein {topic}?")
                    for path in paths[:5]:
                        proof_step = self.graph_traversal.create_proof_step_from_path(
                            path, query=f"{topic} {relation_type}"
                        )
                        if proof_step:
                            proof_tree.add_root_step(proof_step)

                # Best path confidence
                best_confidence = paths[0].confidence if paths else 0.0

                # Track in working memory
                self.working_memory.add_reasoning_state(
                    step_type="graph_traversal_orchestrator",
                    description=f"Graph-Traversal für '{topic}' via Orchestrator",
                    data={
                        "num_paths": len(paths),
                        "inferred_facts": inferred_facts,
                        "relation_type": relation_type,
                    },
                    confidence=best_confidence,
                )

                return ReasoningResult(
                    strategy=ReasoningStrategy.GRAPH_TRAVERSAL,
                    success=True,
                    confidence=best_confidence,
                    inferred_facts=inferred_facts,
                    proof_tree=proof_tree,
                    proof_trace=f"Graph-Traversal: {len(paths)} Pfade gefunden",
                    metadata={
                        "num_paths": len(paths),
                        "avg_hops": sum(len(p.relations) for p in paths) / len(paths),
                    },
                )

            return None

        except Exception as e:
            logger.warning(f"[Graph Traversal] Fehler: {e}")
            return None

    def _try_logic_engine(
        self, topic: str, relation_type: str
    ) -> Optional[ReasoningResult]:
        """
        Rule-based backward chaining.
        """
        logger.debug(f"[Logic Engine] Topic: {topic}")

        try:
            # Create goal
            goal = Goal(
                pred=relation_type, args={"subject": topic.lower(), "object": None}
            )

            # Load facts
            all_facts = self._load_facts_from_graph(topic)
            for fact in all_facts:
                self.logic_engine.add_fact(fact)

            # Run with tracking
            query_text = f"Was ist ein {topic}?"
            proof = self.logic_engine.run_with_tracking(
                goal=goal,
                inference_type="backward_chaining",
                query=query_text,
                max_depth=5,
            )

            if proof:
                # Extract facts
                inferred_facts = self._extract_facts_from_proof(proof)

                # Create proof tree
                proof_tree = None
                if PROOF_SYSTEM_AVAILABLE:
                    proof_tree = create_proof_tree_from_logic_engine(
                        proof, query=query_text
                    )

                return ReasoningResult(
                    strategy=ReasoningStrategy.LOGIC_ENGINE,
                    success=True,
                    confidence=proof.confidence,
                    inferred_facts=inferred_facts,
                    proof_tree=proof_tree,
                    proof_trace=self.logic_engine.format_proof_trace(proof),
                    metadata={"method": proof.method, "depth": proof.goal.depth},
                )

            return None

        except Exception as e:
            logger.warning(f"[Logic Engine] Fehler: {e}")
            return None

    def _try_probabilistic(
        self, topic: str, relation_type: str
    ) -> Optional[ReasoningResult]:
        """
        Standalone probabilistic reasoning.
        """
        if not self.probabilistic_engine:
            return None

        logger.debug(f"[Probabilistic] Topic: {topic}")

        try:
            # Query probabilistic engine
            goal_sig = f"{relation_type}(subject={topic.lower()},object=?)"
            prob, conf = self.probabilistic_engine.query(goal_sig)

            if prob > 0.3:  # Minimum threshold
                # Create result
                explanation = self.probabilistic_engine.generate_response(
                    goal_sig, threshold_high=0.8, threshold_low=0.2
                )

                return ReasoningResult(
                    strategy=ReasoningStrategy.PROBABILISTIC,
                    success=True,
                    confidence=prob,
                    inferred_facts={},  # Probabilistic gibt keine konkreten Fakten
                    proof_tree=None,
                    proof_trace=explanation,
                    metadata={"probability": prob, "confidence": conf},
                )

            return None

        except Exception as e:
            logger.warning(f"[Probabilistic] Fehler: {e}")
            return None

    def _enhance_with_probabilistic(
        self,
        topic: str,
        relation_type: str,
        deterministic_results: List[ReasoningResult],
    ) -> Optional[ReasoningResult]:
        """
        Enhance deterministic results with probabilistic uncertainty quantification.
        """
        if not self.probabilistic_engine:
            return None

        logger.debug(
            f"[Probabilistic Enhancement] Enhancing {len(deterministic_results)} results"
        )

        try:
            # Add deterministic facts to probabilistic engine
            from component_16_probabilistic_engine import ProbabilisticFact

            for result in deterministic_results:
                for rel_type, objects in result.inferred_facts.items():
                    for obj in objects:
                        # FIX: strategy kann String oder Enum sein
                        strategy_value = (
                            result.strategy.value
                            if hasattr(result.strategy, "value")
                            else str(result.strategy)
                        )
                        fact = ProbabilisticFact(
                            pred=rel_type,
                            args={"subject": topic.lower(), "object": obj.lower()},
                            probability=result.confidence,
                            source=f"deterministic_{strategy_value}",
                        )
                        self.probabilistic_engine.add_fact(fact)

            # Run probabilistic inference
            derived_facts = self.probabilistic_engine.infer(max_iterations=3)

            if derived_facts:
                # Calculate enhanced confidence
                goal_sig = f"{relation_type}(subject={topic.lower()},object=?)"
                enhanced_prob, enhanced_conf = self.probabilistic_engine.query(goal_sig)

                return ReasoningResult(
                    strategy=ReasoningStrategy.PROBABILISTIC,
                    success=True,
                    confidence=enhanced_conf,
                    inferred_facts={},
                    proof_tree=None,
                    proof_trace=f"Probabilistische Verbesserung: P={enhanced_prob:.2f}, Conf={enhanced_conf:.2f}",
                    metadata={
                        "enhanced": True,
                        "base_results": len(deterministic_results),
                        "derived_facts": len(derived_facts),
                    },
                )

            return None

        except Exception as e:
            logger.warning(f"[Probabilistic Enhancement] Fehler: {e}")
            return None

    def _try_abductive(
        self, topic: str, relation_type: str
    ) -> Optional[ReasoningResult]:
        """
        Abductive hypothesis generation.
        """
        if not self.abductive_engine:
            return None

        logger.debug(f"[Abductive] Topic: {topic}")

        try:
            # Load context facts
            all_facts = self._load_facts_from_graph(topic)

            # Generate hypotheses
            observation = f"Es wurde nach '{topic}' gefragt"
            hypotheses = self.abductive_engine.generate_hypotheses(
                observation=observation,
                context_facts=all_facts,
                strategies=["template", "analogy", "causal_chain"],
                max_hypotheses=5,
            )

            if hypotheses:
                best_hypothesis = hypotheses[0]

                # Extract inferred facts
                inferred_facts = {}
                for fact in best_hypothesis.abduced_facts:
                    rel = fact.pred
                    obj = fact.args.get("object", "")
                    if rel not in inferred_facts:
                        inferred_facts[rel] = []
                    if obj and obj not in inferred_facts[rel]:
                        inferred_facts[rel].append(obj)

                # Create proof tree
                proof_tree = None
                if PROOF_SYSTEM_AVAILABLE:
                    proof_tree = ProofTree(query=observation)
                    hypothesis_steps = (
                        self.abductive_engine.create_multi_hypothesis_proof_chain(
                            hypotheses[:3], query=observation
                        )
                    )
                    for step in hypothesis_steps:
                        proof_tree.add_root_step(step)

                return ReasoningResult(
                    strategy=ReasoningStrategy.ABDUCTIVE,
                    success=True,
                    confidence=best_hypothesis.confidence,
                    inferred_facts=inferred_facts,
                    proof_tree=proof_tree,
                    proof_trace=best_hypothesis.explanation,
                    metadata={
                        "strategy": best_hypothesis.strategy,
                        "num_hypotheses": len(hypotheses),
                        "scores": best_hypothesis.scores,
                    },
                    is_hypothesis=True,
                )

            return None

        except Exception as e:
            logger.warning(f"[Abductive] Fehler: {e}")
            return None

    def _try_spatial_reasoning(
        self, topic: str, relation_type: str = None
    ) -> Optional[ReasoningResult]:
        """
        Spatial reasoning using SpatialReasoner.

        Handles queries about spatial relations, positions, grids, shapes,
        movement patterns, and constraints.
        """
        if not self.spatial_reasoner:
            return None

        logger.debug(f"[Spatial Reasoning] Topic: {topic}")

        try:
            # Use spatial reasoner to infer spatial relations
            result = self.spatial_reasoner.infer_spatial_relations(
                subject=topic, relation_type=relation_type
            )

            if result and result.success:
                # Track in working memory
                self.working_memory.add_reasoning_state(
                    step_type="spatial_reasoning",
                    description=f"Räumliches Reasoning für '{topic}'",
                    data={
                        "relations": result.relations,
                        "confidence": result.confidence,
                    },
                    confidence=result.confidence,
                )

                # Create proof tree if available
                proof_tree = None
                if PROOF_SYSTEM_AVAILABLE:
                    proof_tree = ProofTree(query=f"Räumliche Relationen für {topic}")

                    # Add spatial reasoning steps
                    for rel_type, targets in result.relations.items():
                        for target in targets:
                            step = UnifiedProofStep(
                                step_type=StepType.QUERY,
                                description=f"{topic} {rel_type} {target}",
                                confidence=result.confidence,
                                metadata={
                                    "source": "spatial_reasoning",
                                    "relation": rel_type,
                                },
                            )
                            proof_tree.add_root_step(step)

                return ReasoningResult(
                    strategy=ReasoningStrategy.SPATIAL,
                    success=True,
                    confidence=result.confidence,
                    inferred_facts=result.relations,
                    proof_tree=proof_tree,
                    proof_trace=f"Spatial Reasoning: {len(result.relations)} Relationen gefunden",
                    metadata={
                        "relation_types": list(result.relations.keys()),
                        "total_relations": sum(
                            len(v) for v in result.relations.values()
                        ),
                    },
                )

            return None

        except Exception as e:
            logger.warning(f"[Spatial Reasoning] Fehler: {e}")
            return None

    def _try_constraint_solving(
        self, topic: str, relation_type: str = None
    ) -> Optional[ReasoningResult]:
        """
        Constraint Satisfaction Problem (CSP) solving for logic puzzles.

        Checks if there is a constraint_problem in the working memory
        (detected by ConstraintDetector in kai_worker.py). If yes, attempts
        to solve it using the CSP solver.

        Note: Most of the constraint solving logic is handled in
        ConstraintSolvingStrategy (kai_sub_goal_executor.py) which has
        direct access to the intent context. This method provides a
        fallback for reasoning orchestrator calls.

        Args:
            topic: The query topic (e.g., "Leo", "Schalter 3")
            relation_type: Optional relation type (not used for CSP)

        Returns:
            ReasoningResult with solution or None
        """
        logger.debug(f"[Constraint Solving] Topic: {topic}")

        try:
            # Check if there is a constraint problem in working memory
            states = self.working_memory.get_reasoning_trace()
            constraint_problem = None

            for state in states:
                if state.step_type == "constraint_problem_detected":
                    constraint_problem = state.data.get("problem")
                    break

            if not constraint_problem:
                logger.debug("[Constraint Solving] Kein Constraint-Problem im Kontext")
                return None

            # Import CSP solver and translator
            from component_29_constraint_reasoning import (
                ConstraintSolver,
                translate_logical_constraints_to_csp,
            )

            # Translate logical constraints to CSP
            csp_problem = translate_logical_constraints_to_csp(constraint_problem)

            # Solve CSP
            solver = ConstraintSolver(use_ac3=True, use_mrv=True, use_lcv=True)
            solution = solver.solve(csp_problem)

            if solution:
                # Format solution as inferred facts
                inferred_facts = {}
                for var_name, value in solution.items():
                    # Create a "HAS_VALUE" relation for each variable
                    if "HAS_VALUE" not in inferred_facts:
                        inferred_facts["HAS_VALUE"] = []
                    inferred_facts["HAS_VALUE"].append(f"{var_name}={value}")

                # Track in working memory
                self.working_memory.add_reasoning_state(
                    step_type="constraint_solving",
                    description=f"CSP-Lösung für {len(solution)} Variablen gefunden",
                    data={
                        "solution": solution,
                        "confidence": constraint_problem.confidence,
                    },
                    confidence=constraint_problem.confidence,
                )

                # Create proof tree if available
                proof_tree = None
                if PROOF_SYSTEM_AVAILABLE:
                    proof_tree = ProofTree(query=f"Constraint-Lösung für {topic}")

                    # Add solution steps
                    for var_name, value in solution.items():
                        step = UnifiedProofStep(
                            step_type=StepType.QUERY,
                            description=f"{var_name} = {value}",
                            confidence=constraint_problem.confidence,
                            metadata={
                                "source": "constraint_solving",
                                "csp_variables": len(solution),
                            },
                        )
                        proof_tree.add_root_step(step)

                return ReasoningResult(
                    strategy=ReasoningStrategy.CONSTRAINT,
                    success=True,
                    confidence=constraint_problem.confidence,
                    inferred_facts=inferred_facts,
                    proof_tree=proof_tree,
                    proof_trace=f"CSP-Lösung: {len(solution)} Variablen gelöst",
                    metadata={
                        "solution": solution,
                        "csp_variables": len(solution),
                        "csp_constraints": len(csp_problem.constraints),
                    },
                )

            logger.debug("[Constraint Solving] Keine Lösung gefunden")
            return None

        except Exception as e:
            logger.warning(f"[Constraint Solving] Fehler: {e}")
            return None

    def _try_resonance(
        self, topic: str, relation_type: str = None
    ) -> Optional[ReasoningResult]:
        """
        Spreading activation with resonance amplification.

        Uses ResonanceEngine to perform wave-based activation propagation
        across the knowledge graph, with resonance boost when multiple paths
        converge on the same concept.

        Args:
            topic: The start concept for activation
            relation_type: Optional relation type filter

        Returns:
            ReasoningResult with activated concepts or None
        """
        if not self.resonance_engine:
            return None

        logger.debug(f"[Resonance] Topic: {topic}")

        try:
            # Prepare query context
            query_context = {}
            allowed_relations = []
            if relation_type:
                allowed_relations = [relation_type]

            # Activate concept with spreading activation
            activation_map = self.resonance_engine.activate_concept(
                start_word=topic,
                query_context=query_context,
                allowed_relations=allowed_relations,
            )

            if activation_map.concepts_activated > 1:  # More than just start concept
                # Extract inferred facts from activated concepts
                inferred_facts = {}

                # Get top activated concepts
                top_concepts = activation_map.get_top_concepts(n=20)

                for concept, activation in top_concepts:
                    if concept == topic:
                        continue  # Skip start concept

                    # Find paths to this concept to determine relation type
                    paths = activation_map.get_paths_to(concept)
                    if paths:
                        # Use relation from strongest path
                        best_path = max(paths, key=lambda p: p.confidence_product)
                        if best_path.relations:
                            rel_type = best_path.relations[0]
                            if rel_type not in inferred_facts:
                                inferred_facts[rel_type] = []
                            if concept not in inferred_facts[rel_type]:
                                inferred_facts[rel_type].append(concept)

                # Calculate confidence based on activation map
                # Use max activation as confidence (since it reflects strongest evidence)
                confidence = min(activation_map.max_activation, 1.0)

                # Boost confidence if resonance points found
                if activation_map.resonance_points:
                    resonance_boost = len(activation_map.resonance_points) * 0.05
                    confidence = min(confidence + resonance_boost, 1.0)

                # Create proof tree
                proof_tree = None
                if PROOF_SYSTEM_AVAILABLE:
                    proof_tree = ProofTree(query=f"Resonanz-Aktivierung für {topic}")

                    # Add activation explanation as root step
                    summary = self.resonance_engine.get_activation_summary(
                        activation_map
                    )

                    root_step = UnifiedProofStep(
                        step_id="resonance_activation",
                        step_type=StepType.INFERENCE,
                        inputs=[topic],
                        output=f"{activation_map.concepts_activated} Konzepte aktiviert",
                        confidence=confidence,
                        explanation_text=summary,
                        source_component="resonance_engine",
                        metadata={
                            "waves": activation_map.waves_executed,
                            "resonance_points": len(activation_map.resonance_points),
                            "max_activation": activation_map.max_activation,
                        },
                    )
                    proof_tree.add_root_step(root_step)

                    # Add top resonance points as substeps
                    for rp in sorted(
                        activation_map.resonance_points,
                        key=lambda x: x.resonance_boost,
                        reverse=True,
                    )[:5]:
                        explanation = self.resonance_engine.explain_activation(
                            rp.concept, activation_map, max_paths=3
                        )
                        resonance_step = UnifiedProofStep(
                            step_id=f"resonance_{rp.concept}",
                            step_type=StepType.FACT_MATCH,
                            inputs=[topic],
                            output=f"{rp.concept} (Resonanz: {rp.num_paths} Pfade)",
                            confidence=min(rp.resonance_boost, 1.0),
                            explanation_text=explanation,
                            source_component="resonance_engine",
                            metadata={"resonance_boost": rp.resonance_boost},
                        )
                        proof_tree.add_child_step(
                            "resonance_activation", resonance_step
                        )

                # Track in working memory
                self.working_memory.add_reasoning_state(
                    step_type="resonance_activation",
                    description=f"Spreading Activation von '{topic}'",
                    data={
                        "concepts_activated": activation_map.concepts_activated,
                        "waves": activation_map.waves_executed,
                        "resonance_points": len(activation_map.resonance_points),
                        "inferred_facts": inferred_facts,
                    },
                    confidence=confidence,
                )

                # Generate proof trace
                proof_trace = self.resonance_engine.get_activation_summary(
                    activation_map
                )

                return ReasoningResult(
                    strategy=ReasoningStrategy.RESONANCE,
                    success=True,
                    confidence=confidence,
                    inferred_facts=inferred_facts,
                    proof_tree=proof_tree,
                    proof_trace=proof_trace,
                    metadata={
                        "concepts_activated": activation_map.concepts_activated,
                        "waves_executed": activation_map.waves_executed,
                        "resonance_points": len(activation_map.resonance_points),
                        "max_activation": activation_map.max_activation,
                        "total_paths": len(activation_map.reasoning_paths),
                    },
                )

            return None

        except Exception as e:
            logger.warning(f"[Resonance] Fehler: {e}")
            return None

    def _create_aggregated_result(
        self, results: List[ReasoningResult]
    ) -> AggregatedResult:
        """
        Aggregate multiple reasoning results using configured aggregation method.

        Supports: noisy_or, weighted_avg, max, dempster_shafer
        """
        if not results:
            raise ValueError("Cannot aggregate empty results")

        logger.debug(
            f"[Aggregation] Combining {len(results)} results using {self.aggregation_method}"
        )

        # Combine confidences using selected method
        if self.aggregation_method == "noisy_or":
            confidences = [r.confidence for r in results]
            combined_confidence = self._noisy_or(confidences)
        elif self.aggregation_method == "weighted_avg":
            combined_confidence = self._weighted_average(results)
        elif self.aggregation_method == "max":
            confidences = [r.confidence for r in results]
            combined_confidence = self._maximum(confidences)
        elif self.aggregation_method == "dempster_shafer":
            combined_confidence = self._dempster_shafer(results)
        else:
            logger.warning(
                f"Unknown aggregation method: {self.aggregation_method}, falling back to noisy_or"
            )
            confidences = [r.confidence for r in results]
            combined_confidence = self._noisy_or(confidences)

        # Merge inferred facts (union)
        merged_facts = {}
        for result in results:
            for rel_type, objects in result.inferred_facts.items():
                if rel_type not in merged_facts:
                    merged_facts[rel_type] = []
                for obj in objects:
                    if obj not in merged_facts[rel_type]:
                        merged_facts[rel_type].append(obj)

        # Merge proof trees
        merged_proof_tree = None
        if PROOF_SYSTEM_AVAILABLE:
            proof_trees = [r.proof_tree for r in results if r.proof_tree]
            if proof_trees:
                query = proof_trees[0].query
                merged_proof_tree = merge_proof_trees(proof_trees, query)

        # Emit merged proof tree
        if merged_proof_tree and self.signals:
            self.signals.proof_tree_update.emit(merged_proof_tree)

        # Generate explanation
        strategies_used = [r.strategy for r in results]
        # FIX: strategy kann String oder Enum sein
        strategy_names = ", ".join(
            s.value if hasattr(s, "value") else str(s) for s in strategies_used
        )

        explanation = (
            f"Kombiniertes Ergebnis aus {len(results)} Strategien ({strategy_names}). "
            f"Kombinierte Konfidenz: {combined_confidence:.2f}"
        )

        # Check if any result is hypothesis
        is_hypothesis = any(r.is_hypothesis for r in results)

        return AggregatedResult(
            combined_confidence=combined_confidence,
            inferred_facts=merged_facts,
            merged_proof_tree=merged_proof_tree,
            strategies_used=strategies_used,
            individual_results=results,
            explanation=explanation,
            is_hypothesis=is_hypothesis,
        )

    def _load_config(self, config_path: str):
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to YAML config file
        """
        try:
            from pathlib import Path

            import yaml

            config_file = Path(config_path)
            if not config_file.exists():
                logger.warning(f"Config file not found: {config_path}, using defaults")
                return

            with open(config_file, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            # Load orchestrator settings
            if "orchestrator" in config:
                orch_config = config["orchestrator"]
                self.enable_hybrid = orch_config.get(
                    "enable_hybrid", self.enable_hybrid
                )
                self.min_confidence_threshold = orch_config.get(
                    "min_confidence_threshold", self.min_confidence_threshold
                )
                self.probabilistic_enhancement = orch_config.get(
                    "probabilistic_enhancement", self.probabilistic_enhancement
                )
                self.aggregation_method = orch_config.get(
                    "aggregation_method", self.aggregation_method
                )
                self.enable_parallel_execution = orch_config.get(
                    "enable_parallel_execution", self.enable_parallel_execution
                )
                self.enable_result_caching = orch_config.get(
                    "enable_result_caching", self.enable_result_caching
                )

            # Load strategy weights
            if "strategy_weights" in config:
                for strategy_name, weight in config["strategy_weights"].items():
                    try:
                        strategy = ReasoningStrategy(strategy_name)
                        self.strategy_weights[strategy] = weight
                    except ValueError:
                        logger.warning(f"Unknown strategy in config: {strategy_name}")

            logger.info(f"[OK] Configuration loaded from {config_path}")

        except ImportError:
            logger.warning(
                "PyYAML not installed, cannot load config. Install: pip install pyyaml"
            )
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")

    def _noisy_or(self, probabilities: List[float]) -> float:
        """
        Noisy-OR combination for redundant evidence.

        P(E | C1, C2, ..., Cn) = 1 - ∏(1 - P(E | Ci))

        At least one source is sufficient.
        """
        if not probabilities:
            return 0.0

        product = 1.0
        for p in probabilities:
            product *= 1.0 - p

        return 1.0 - product

    def _weighted_average(self, results: List[ReasoningResult]) -> float:
        """
        Weighted average combination.

        Combined = Σ(wi * Pi) / Σ(wi)

        Uses strategy_weights for weighting.
        """
        if not results:
            return 0.0

        weighted_sum = 0.0
        total_weight = 0.0

        for result in results:
            weight = self.strategy_weights.get(result.strategy, 0.1)
            weighted_sum += weight * result.confidence
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _maximum(self, probabilities: List[float]) -> float:
        """
        Maximum confidence (best-case scenario).

        Combined = max(P1, P2, ..., Pn)

        Takes the most confident source.
        """
        return max(probabilities) if probabilities else 0.0

    def _dempster_shafer(self, results: List[ReasoningResult]) -> float:
        """
        Dempster-Shafer combination for uncertain evidence.

        Combines belief masses from multiple sources accounting for conflict.

        Simplified implementation:
        m1 ⊕ m2 = (m1 * m2) / (1 - K)
        where K = conflict mass
        """
        if not results:
            return 0.0

        if len(results) == 1:
            return results[0].confidence

        # Initialize with first result
        combined_belief = results[0].confidence
        combined_disbelief = 1.0 - results[0].confidence

        # Combine with subsequent results
        for result in results[1:]:
            belief = result.confidence
            disbelief = 1.0 - result.confidence

            # Calculate conflict
            conflict = combined_belief * disbelief + combined_disbelief * belief

            if conflict >= 1.0:
                # Total conflict - fall back to noisy-or
                logger.warning(
                    "Dempster-Shafer: Total conflict detected, falling back to Noisy-OR"
                )
                return self._noisy_or([r.confidence for r in results])

            # Combine beliefs
            new_belief = (combined_belief * belief) / (1.0 - conflict)
            new_disbelief = (combined_disbelief * disbelief) / (1.0 - conflict)

            combined_belief = new_belief
            combined_disbelief = new_disbelief

        return combined_belief

    # ========================================================================
    # Meta-Learning Helper Methods
    # ========================================================================

    def _get_available_strategy_names(
        self, exclude: Optional[List[str]] = None
    ) -> List[str]:
        """
        Returns list of available strategy names (string format for Meta-Learning).

        Args:
            exclude: Optional list of strategy names to exclude

        Returns:
            List of available strategy names
        """
        exclude = exclude or []
        strategies = []

        # Map available engines to strategy names
        strategy_availability = {
            "direct_fact": True,  # Always available
            "logic_engine": self.logic_engine is not None,
            "graph_traversal": self.graph_traversal is not None,
            "probabilistic": self.probabilistic_engine is not None,
            "abductive": self.abductive_engine is not None,
            "combinatorial": self.combinatorial_reasoner is not None,
            "spatial": self.spatial_reasoner is not None,
            "resonance": self.resonance_engine is not None,
        }

        for name, available in strategy_availability.items():
            if available and name not in exclude:
                strategies.append(name)

        return strategies

    def _map_strategy_name_to_enum(
        self, strategy_name: str
    ) -> Optional[ReasoningStrategy]:
        """
        Maps strategy name (string) to ReasoningStrategy enum.

        Args:
            strategy_name: Strategy name as string

        Returns:
            ReasoningStrategy enum or None
        """
        mapping = {
            "direct_fact": ReasoningStrategy.DIRECT_FACT,
            "logic_engine": ReasoningStrategy.LOGIC_ENGINE,
            "graph_traversal": ReasoningStrategy.GRAPH_TRAVERSAL,
            "probabilistic": ReasoningStrategy.PROBABILISTIC,
            "abductive": ReasoningStrategy.ABDUCTIVE,
            "combinatorial": ReasoningStrategy.COMBINATORIAL,
            "spatial": ReasoningStrategy.SPATIAL,
            "resonance": ReasoningStrategy.RESONANCE,
        }

        return mapping.get(strategy_name)

    def _execute_single_strategy(
        self,
        topic: str,
        relation_type: str,
        strategy: ReasoningStrategy,
    ) -> Optional[AggregatedResult]:
        """
        Executes a single strategy and returns result.

        Args:
            topic: Topic to reason about
            relation_type: Relation type
            strategy: Strategy to execute

        Returns:
            AggregatedResult or None
        """
        try:
            result = None

            if strategy == ReasoningStrategy.DIRECT_FACT:
                result = self._try_direct_fact_lookup(topic, relation_type)
            elif strategy == ReasoningStrategy.GRAPH_TRAVERSAL:
                result = self._try_graph_traversal(topic, relation_type)
            elif strategy == ReasoningStrategy.LOGIC_ENGINE:
                result = self._try_logic_engine(topic, relation_type)
            elif strategy == ReasoningStrategy.PROBABILISTIC:
                result = self._try_probabilistic(topic, relation_type)
            elif strategy == ReasoningStrategy.ABDUCTIVE:
                result = self._try_abductive(topic, relation_type)
            elif strategy == ReasoningStrategy.SPATIAL:
                result = self._try_spatial_reasoning(topic, relation_type)
            elif strategy == ReasoningStrategy.RESONANCE:
                result = self._try_resonance(topic, relation_type)

            if result:
                # Wrap in AggregatedResult
                return self._create_aggregated_result([result])

            return None

        except Exception as e:
            logger.error(f"Error executing strategy {strategy}: {e}", exc_info=True)
            return None

    def _evaluate_result_quality(
        self,
        result: AggregatedResult,
        query: str,
        topic: str,
        context: Dict[str, Any],
    ):
        """
        Evaluates result quality using SelfEvaluator.

        Args:
            result: The AggregatedResult to evaluate
            query: Original query text
            topic: Topic
            context: Context dict

        Returns:
            EvaluationResult from SelfEvaluator
        """
        if not self.self_evaluator:
            # Return dummy result if evaluator not available
            from component_50_self_evaluation import (
                EvaluationResult,
                RecommendationType,
            )

            return EvaluationResult(
                overall_score=0.7,
                checks={},
                uncertainties=[],
                recommendation=RecommendationType.SHOW_TO_USER,
            )

        # Prepare answer dict for evaluator
        answer = {
            "text": result.explanation,
            "confidence": result.combined_confidence,
            "proof_tree": result.merged_proof_tree,
            "reasoning_paths": [],  # TODO: Extract from result if available
        }

        # Evaluate
        eval_result = self.self_evaluator.evaluate_answer(
            question=query, answer=answer, context=context
        )

        return eval_result

    def _record_strategy_usage(
        self,
        strategy_name: str,
        query: str,
        result: Optional[AggregatedResult],
        response_time: float,
        success: bool,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Records strategy usage in MetaLearningEngine.

        Args:
            strategy_name: Name of the strategy
            query: Query text
            result: Result (can be None if failed)
            response_time: Time taken in seconds
            success: Whether strategy succeeded
            context: Optional context dict
        """
        if not self.meta_learning_engine:
            return

        # Prepare result dict for MetaLearningEngine
        result_dict = {
            "confidence": result.combined_confidence if result else 0.0,
            "success": success,
        }

        if not success:
            result_dict["error"] = "Strategy failed to produce result"

        # Determine user feedback (for now, automatic based on confidence)
        user_feedback = None
        if result:
            if result.combined_confidence >= 0.8:
                user_feedback = "correct"  # High confidence assumed correct
            elif result.combined_confidence < 0.4:
                user_feedback = "incorrect"  # Low confidence assumed incorrect
            else:
                user_feedback = "neutral"

        # Record usage
        try:
            self.meta_learning_engine.record_strategy_usage(
                strategy=strategy_name,
                query=query,
                result=result_dict,
                response_time=response_time,
                context=context,
                user_feedback=user_feedback,
            )

            logger.debug(
                f"[Meta-Learning] Recorded usage for '{strategy_name}': "
                f"success={success}, confidence={result_dict['confidence']:.2f}, "
                f"time={response_time:.3f}s"
            )

        except Exception as e:
            logger.error(f"Failed to record strategy usage: {e}")

    def _load_facts_from_graph(self, topic: str):
        """Load facts from knowledge graph (helper method)."""
        from component_9_logik_engine import Fact

        facts = []
        fact_data = self.netzwerk.query_graph_for_facts(topic)

        for relation_type, objects in fact_data.items():
            for obj in objects:
                fact = Fact(
                    pred=relation_type,
                    args={"subject": topic.lower(), "object": obj.lower()},
                    confidence=1.0,
                    source="graph",
                )
                facts.append(fact)

        return facts

    def _extract_facts_from_proof(self, proof: LogicProofStep) -> Dict[str, List[str]]:
        """Extract facts from Logic Engine proof (helper method)."""
        facts = {}

        if proof.supporting_facts:
            for fact in proof.supporting_facts:
                relation = fact.pred
                obj = fact.args.get("object", "")

                if relation not in facts:
                    facts[relation] = []

                if obj and obj not in facts[relation]:
                    facts[relation].append(obj)

        # Recursive through subgoals
        for subproof in proof.subgoals:
            subfacts = self._extract_facts_from_proof(subproof)
            for relation, objects in subfacts.items():
                if relation not in facts:
                    facts[relation] = []
                facts[relation].extend([o for o in objects if o not in facts[relation]])

        return facts
