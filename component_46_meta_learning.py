"""
component_46_meta_learning.py

Meta-Learning Layer: Strategy Performance Tracking & Selection

Kernkonzept:
- Trackt Performance jeder Reasoning-Strategy
- Lernt, welche Strategy für welche Fragen optimal ist
- Epsilon-Greedy Exploration/Exploitation
- Persistiert Statistiken in Neo4j

Author: KAI Development Team
Last Updated: 2025-11-08
"""

import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from cachetools import TTLCache

from component_1_netzwerk import KonzeptNetzwerk
from component_11_embedding_service import EmbeddingService
from component_15_logging_config import get_logger
from kai_exceptions import KAIException

logger = get_logger(__name__)


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class StrategyPerformance:
    """Performance-Statistiken für eine Reasoning-Strategy"""

    strategy_name: str
    queries_handled: int = 0
    success_count: int = 0
    failure_count: int = 0
    success_rate: float = 0.5  # Initial neutral
    avg_confidence: float = 0.0
    avg_response_time: float = 0.0
    total_response_time: float = 0.0
    typical_query_patterns: List[str] = field(default_factory=list)
    failure_modes: List[str] = field(default_factory=list)
    last_used: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)

    def update_from_usage(
        self, confidence: float, response_time: float, success: Optional[bool] = None
    ) -> None:
        """Update statistiken basierend auf neuem usage"""
        self.queries_handled += 1

        # Update Confidence (exponential moving average)
        alpha = 0.1  # Learning rate
        self.avg_confidence = (1 - alpha) * self.avg_confidence + alpha * confidence

        # Update Response Time
        self.total_response_time += response_time
        self.avg_response_time = self.total_response_time / self.queries_handled

        # Update Success Rate (wenn Feedback vorhanden)
        if success is not None:
            if success:
                self.success_count += 1
            else:
                self.failure_count += 1

            # Recalculate success rate mit Laplace smoothing
            self.success_rate = (self.success_count + 1) / (self.queries_handled + 2)

        self.last_used = datetime.now()


@dataclass
class QueryPattern:
    """Erkanntes Query-Pattern für Strategy-Matching"""

    pattern_text: str
    embedding: Optional[List[float]] = None
    associated_strategy: str = ""
    success_count: int = 0
    total_count: int = 0

    @property
    def success_rate(self) -> float:
        if self.total_count == 0:
            return 0.5
        return self.success_count / self.total_count


@dataclass
class StrategyUsageEpisode:
    """Einzelne Strategy-Verwendung für Tracking"""

    timestamp: datetime
    strategy_name: str
    query: str
    query_embedding: List[float]
    context: Dict[str, Any]
    result_confidence: float
    response_time: float
    user_feedback: Optional[str] = None  # 'correct', 'incorrect', 'neutral'
    failure_reason: Optional[str] = None


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class MetaLearningConfig:
    """Konfiguration für Meta-Learning Engine"""

    # Exploration/Exploitation
    epsilon: float = 0.1  # 10% exploration
    epsilon_decay: float = 0.995  # Decay über Zeit
    min_epsilon: float = 0.05

    # Learning rates
    confidence_alpha: float = 0.1
    success_rate_alpha: float = 0.1

    # Pattern matching
    pattern_similarity_threshold: float = 0.85
    max_patterns_per_strategy: int = 50

    # Performance thresholds
    min_queries_for_confidence: int = 5  # Mindest-Queries bevor Strategy bevorzugt wird

    # Scoring weights
    pattern_weight: float = 0.4
    performance_weight: float = 0.4
    context_weight: float = 0.2

    # Neo4j persistence
    persist_every_n_queries: int = 10

    # Cache
    cache_ttl_seconds: int = 600  # 10 Minuten (für Strategy Stats)
    query_pattern_cache_ttl: int = 300  # 5 Minuten (für Query Patterns)


# ============================================================================
# Meta-Learning Engine
# ============================================================================


class MetaLearningEngine:
    """
    Meta-Reasoning Engine für Strategy-Auswahl und Performance-Tracking

    Funktionen:
    1. Trackt Performance jeder Reasoning-Strategy
    2. Lernt Query-Patterns für jede Strategy
    3. Wählt optimale Strategy für neue Queries
    4. Epsilon-Greedy Exploration
    5. Persistiert Statistiken in Neo4j
    """

    def __init__(
        self,
        netzwerk: KonzeptNetzwerk,
        embedding_service: EmbeddingService,
        config: Optional[MetaLearningConfig] = None,
    ):
        self.netzwerk = netzwerk
        self.embedding_service = embedding_service
        self.config = config or MetaLearningConfig()

        # In-Memory state
        self.strategy_stats: Dict[str, StrategyPerformance] = {}
        self.query_patterns: Dict[str, List[QueryPattern]] = defaultdict(list)
        self.usage_history: List[StrategyUsageEpisode] = []

        # Counters
        self.total_queries: int = 0
        self.queries_since_last_persist: int = 0

        # Performance Optimization: Caching
        # Strategy Stats Cache (TTL: 10 Minuten) für schnellen Zugriff
        self._stats_cache: TTLCache = TTLCache(
            maxsize=50, ttl=self.config.cache_ttl_seconds
        )
        # Query Pattern Cache (TTL: 5 Minuten)
        self._pattern_cache: TTLCache = TTLCache(
            maxsize=100, ttl=self.config.query_pattern_cache_ttl
        )

        # Load from Neo4j
        self._load_persisted_stats()

        logger.info(
            "MetaLearningEngine initialized with %d strategies",
            len(self.strategy_stats),
        )

    # ========================================================================
    # Helper: Generic Neo4j Query
    # ========================================================================

    def _execute_query(
        self, cypher: str, params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute generic Neo4j query"""
        try:
            with self.netzwerk.driver.session() as session:
                result = session.run(cypher, params or {})
                return [dict(record) for record in result]
        except Exception as e:
            logger.error("Neo4j query failed: %s", e)
            return []

    # ========================================================================
    # Core Functions: Strategy Usage Recording
    # ========================================================================

    def record_strategy_usage(
        self,
        strategy: str,
        query: str,
        result: Dict[str, Any],
        response_time: float,
        context: Optional[Dict[str, Any]] = None,
        user_feedback: Optional[str] = None,
    ) -> None:
        """
        Tracked jede Strategy-Verwendung

        Args:
            strategy: Name der verwendeten Strategy
            query: User-Query
            result: Ergebnis der Strategy (muss 'confidence' enthalten)
            response_time: Zeit in Sekunden
            context: Optional context dict
            user_feedback: 'correct', 'incorrect', 'neutral'
        """
        try:
            # Initialisiere Strategy stats falls neu
            if strategy not in self.strategy_stats:
                self.strategy_stats[strategy] = StrategyPerformance(
                    strategy_name=strategy
                )

            stats = self.strategy_stats[strategy]

            # Extrahiere Confidence
            confidence = result.get("confidence", 0.5)

            # Determine Success basierend auf Feedback
            success = None
            if user_feedback == "correct":
                success = True
            elif user_feedback == "incorrect":
                success = False
                # Extract failure mode
                failure_mode = self._extract_failure_pattern(query, result)
                if failure_mode and failure_mode not in stats.failure_modes:
                    stats.failure_modes.append(failure_mode)
                    if len(stats.failure_modes) > 20:  # Limit
                        stats.failure_modes.pop(0)

            # Update Stats
            stats.update_from_usage(confidence, response_time, success)

            # Query Embedding für Pattern-Learning
            query_embedding = self.embedding_service.get_embedding(query)

            # Record Episode
            episode = StrategyUsageEpisode(
                timestamp=datetime.now(),
                strategy_name=strategy,
                query=query,
                query_embedding=query_embedding,
                context=context or {},
                result_confidence=confidence,
                response_time=response_time,
                user_feedback=user_feedback,
                failure_reason=result.get("error") if not success else None,
            )
            self.usage_history.append(episode)

            # Update Query Patterns
            self._update_query_patterns(strategy, query, query_embedding, success)

            # Increment counters
            self.total_queries += 1
            self.queries_since_last_persist += 1

            # Persist to Neo4j periodisch
            if self.queries_since_last_persist >= self.config.persist_every_n_queries:
                self._persist_all_stats()
                self.queries_since_last_persist = 0

            logger.debug(
                "Recorded usage for strategy '%s': confidence=%.2f, "
                "response_time=%.3fs, feedback=%s",
                strategy,
                confidence,
                response_time,
                user_feedback,
            )

        except Exception as e:
            logger.error("Error recording strategy usage: %s", e, exc_info=True)

    def record_strategy_usage_with_feedback(
        self,
        strategy_name: str,
        query: str,
        success: bool,
        confidence: float,
        response_time: float = 0.0,
        user_feedback: str = "neutral",
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Vereinfachte Methode für FeedbackHandler

        Speziell für User-Feedback-Loop optimiert.

        Args:
            strategy_name: Name der Strategy
            query: User-Query
            success: True wenn korrekt, False wenn inkorrekt
            confidence: Confidence-Wert
            response_time: Optional Response-Zeit
            user_feedback: 'correct', 'incorrect', 'unsure', 'partially_correct'
            context: Optional Context
        """
        # Erstelle vereinfachtes result dict
        result = {"confidence": confidence, "success": success}

        # Map user_feedback zu internem Format
        feedback_map = {
            "correct": "correct",
            "incorrect": "incorrect",
            "unsure": "neutral",
            "partially_correct": "neutral",
        }
        internal_feedback = feedback_map.get(user_feedback, "neutral")

        # Rufe Hauptmethode auf
        self.record_strategy_usage(
            strategy=strategy_name,
            query=query,
            result=result,
            response_time=response_time,
            context=context,
            user_feedback=internal_feedback,
        )

        logger.info(
            f"Recorded user feedback | strategy={strategy_name}, "
            f"feedback={user_feedback}, success={success}"
        )

    # ========================================================================
    # Core Functions: Strategy Selection (Meta-Reasoning)
    # ========================================================================

    def select_best_strategy(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        available_strategies: Optional[List[str]] = None,
    ) -> Tuple[str, float]:
        """
        Meta-Reasoning: Welche Strategy ist am besten für diese Query?

        Args:
            query: User-Query
            context: Optional context dict
            available_strategies: Liste verfügbarer Strategien (None = alle)

        Returns:
            (strategy_name, confidence_score)
        """
        try:
            context = context or {}

            # Epsilon-Greedy: Exploration vs. Exploitation
            if random.random() < self.config.epsilon:
                # EXPLORATION: Random strategy
                strategies = available_strategies or list(self.strategy_stats.keys())
                if not strategies:
                    return ("direct_answer", 0.5)  # Fallback

                selected = random.choice(strategies)
                logger.debug("Epsilon-greedy EXPLORATION: selected '%s'", selected)
                return (selected, 0.3)  # Low confidence für exploration

            # EXPLOITATION: Best strategy basierend auf Scoring
            query_embedding = self.embedding_service.get_embedding(query)

            strategy_scores: Dict[str, float] = {}
            strategies_to_evaluate = available_strategies or list(
                self.strategy_stats.keys()
            )

            if not strategies_to_evaluate:
                return ("direct_answer", 0.5)

            for strategy_name in strategies_to_evaluate:
                if strategy_name not in self.strategy_stats:
                    # Neue Strategy: neutral score
                    strategy_scores[strategy_name] = 0.5
                    continue

                stats = self.strategy_stats[strategy_name]

                # 1. Pattern-basierte Scoring
                pattern_score = self._match_query_patterns(
                    query_embedding, strategy_name
                )

                # 2. Performance-basierte Scoring
                perf_score = self._calculate_performance_score(stats)

                # 3. Context-basierte Scoring
                context_score = self._match_context_requirements(context, strategy_name)

                # Aggregierte Score (weighted sum)
                aggregated_score = (
                    pattern_score * self.config.pattern_weight
                    + perf_score * self.config.performance_weight
                    + context_score * self.config.context_weight
                )

                strategy_scores[strategy_name] = aggregated_score

                logger.debug(
                    "Strategy '%s' scores: pattern=%.2f, perf=%.2f, "
                    "context=%.2f → total=%.2f",
                    strategy_name,
                    pattern_score,
                    perf_score,
                    context_score,
                    aggregated_score,
                )

            # Select best strategy
            best_strategy = max(strategy_scores, key=strategy_scores.get)
            best_score = strategy_scores[best_strategy]

            # Decay epsilon over time
            self.config.epsilon = max(
                self.config.min_epsilon, self.config.epsilon * self.config.epsilon_decay
            )

            logger.info(
                "Selected strategy '%s' with score %.2f (epsilon=%.3f)",
                best_strategy,
                best_score,
                self.config.epsilon,
            )

            return (best_strategy, best_score)

        except Exception as e:
            logger.error("Error selecting strategy: %s", e, exc_info=True)
            return ("direct_answer", 0.3)  # Safe fallback

    # ========================================================================
    # Scoring Components
    # ========================================================================

    def _match_query_patterns(
        self, query_embedding: List[float], strategy_name: str
    ) -> float:
        """
        Pattern-basierte Scoring: Ähnlichkeit zu erfolgreichen früheren Queries

        Returns:
            Score 0.0-1.0
        """
        if strategy_name not in self.query_patterns:
            return 0.5  # Neutral

        patterns = self.query_patterns[strategy_name]
        if not patterns:
            return 0.5

        # Finde ähnlichste Patterns (cosine similarity via embeddings)
        max_similarity = 0.0
        matching_pattern = None

        for pattern in patterns:
            if pattern.embedding is None:
                continue

            similarity = self._cosine_similarity(query_embedding, pattern.embedding)
            if similarity > max_similarity:
                max_similarity = similarity
                matching_pattern = pattern

        if (
            matching_pattern
            and max_similarity >= self.config.pattern_similarity_threshold
        ):
            # Gewichte similarity mit Pattern success rate
            pattern_success = matching_pattern.success_rate
            return max_similarity * 0.6 + pattern_success * 0.4

        return 0.5  # Kein passendes Pattern gefunden

    def _calculate_performance_score(self, stats: StrategyPerformance) -> float:
        """
        Performance-basierte Scoring

        Returns:
            Score 0.0-1.0
        """
        # Wenn zu wenig Queries, neutral score
        if stats.queries_handled < self.config.min_queries_for_confidence:
            return 0.5

        # Weighted combination von Success Rate und Avg Confidence
        success_component = stats.success_rate
        confidence_component = stats.avg_confidence

        # Zusätzlich: Bonus für schnelle Response Time
        # Normalisiere response time (angenommen: <1s = gut, >5s = schlecht)
        speed_bonus = max(0, 1.0 - stats.avg_response_time / 5.0) * 0.1

        score = success_component * 0.6 + confidence_component * 0.4 + speed_bonus

        return min(1.0, score)

    def _match_context_requirements(
        self, context: Dict[str, Any], strategy_name: str
    ) -> float:
        """
        Context-basierte Scoring: Passt Strategy zu Context?

        Heuristiken:
        - temporal_reasoning → benötigt temporale Keywords
        - graph_traversal → benötigt Relationen
        - probabilistic → benötigt Unsicherheit
        - constraint_reasoning → benötigt Constraints

        Returns:
            Score 0.0-1.0
        """
        # Default: neutral
        score = 0.5

        # Temporal keywords
        if "temporal_required" in context and context["temporal_required"]:
            if strategy_name in ["temporal_reasoning", "causal_reasoning"]:
                score = 0.9
            else:
                score = 0.3

        # Graph-based
        if "requires_graph" in context and context["requires_graph"]:
            if strategy_name in ["graph_traversal", "abductive_reasoning"]:
                score = 0.9
            else:
                score = 0.4

        # Probabilistic
        if "uncertainty" in context and context["uncertainty"]:
            if strategy_name == "probabilistic_reasoning":
                score = 0.95
            else:
                score = 0.5

        # Constraint satisfaction
        if "constraints" in context and len(context.get("constraints", [])) > 0:
            if strategy_name == "constraint_reasoning":
                score = 0.95
            else:
                score = 0.4

        # Kombinatorische Probleme
        if "combinatorial" in context and context["combinatorial"]:
            if strategy_name == "combinatorial_reasoning":
                score = 0.95
            else:
                score = 0.3

        return score

    # ========================================================================
    # Pattern Learning
    # ========================================================================

    def _update_query_patterns(
        self,
        strategy: str,
        query: str,
        query_embedding: List[float],
        success: Optional[bool],
    ) -> None:
        """Update Query-Patterns für Strategy"""
        try:
            # Finde ähnliche existierende Patterns
            existing_pattern = None
            max_similarity = 0.0

            for pattern in self.query_patterns[strategy]:
                if pattern.embedding is None:
                    continue

                similarity = self._cosine_similarity(query_embedding, pattern.embedding)
                if similarity > max_similarity:
                    max_similarity = similarity
                    existing_pattern = pattern

            # Wenn ähnliches Pattern existiert, update
            if (
                existing_pattern
                and max_similarity >= self.config.pattern_similarity_threshold
            ):
                existing_pattern.total_count += 1
                if success:
                    existing_pattern.success_count += 1
            else:
                # Neues Pattern erstellen
                new_pattern = QueryPattern(
                    pattern_text=query[:100],  # Truncate
                    embedding=query_embedding,
                    associated_strategy=strategy,
                    success_count=1 if success else 0,
                    total_count=1,
                )
                self.query_patterns[strategy].append(new_pattern)

                # Limit Patterns pro Strategy
                if (
                    len(self.query_patterns[strategy])
                    > self.config.max_patterns_per_strategy
                ):
                    # Remove Pattern mit niedrigster Success Rate
                    self.query_patterns[strategy].sort(key=lambda p: p.success_rate)
                    self.query_patterns[strategy].pop(0)

        except Exception as e:
            logger.error("Error updating query patterns: %s", e)

    def _extract_failure_pattern(
        self, query: str, result: Dict[str, Any]
    ) -> Optional[str]:
        """Extrahiere Failure Pattern aus fehlgeschlagener Query"""
        try:
            # Einfache Heuristik: Extract Keywords aus Query
            tokens = query.lower().split()

            # Filter stopwords (simple)
            stopwords = {
                "der",
                "die",
                "das",
                "ein",
                "eine",
                "ist",
                "sind",
                "hat",
                "haben",
            }
            keywords = [t for t in tokens if t not in stopwords and len(t) > 3]

            if keywords:
                return " ".join(keywords[:3])  # Top 3 keywords

            return query[:50]  # Fallback

        except Exception:
            return None

    # ========================================================================
    # Neo4j Persistence
    # ========================================================================

    def _persist_all_stats(self) -> None:
        """Persistiere alle Strategy-Statistiken in Neo4j"""
        try:
            for strategy_name, stats in self.strategy_stats.items():
                self._persist_strategy_stats(strategy_name, stats)

            logger.info(
                "Persisted stats for %d strategies to Neo4j", len(self.strategy_stats)
            )

        except Exception as e:
            logger.error("Error persisting stats: %s", e)

    def _persist_strategy_stats(
        self, strategy_name: str, stats: StrategyPerformance
    ) -> None:
        """Persistiere einzelne Strategy-Stats in Neo4j"""
        try:
            # Erstelle/Update StrategyPerformance Node
            query = """
            MERGE (sp:StrategyPerformance {strategy_name: $strategy_name})
            SET sp.queries_handled = $queries_handled,
                sp.success_count = $success_count,
                sp.failure_count = $failure_count,
                sp.success_rate = $success_rate,
                sp.avg_confidence = $avg_confidence,
                sp.avg_response_time = $avg_response_time,
                sp.failure_modes = $failure_modes,
                sp.last_used = datetime($last_used),
                sp.updated_at = datetime()
            RETURN sp
            """

            result = self._execute_query(
                query,
                {
                    "strategy_name": strategy_name,
                    "queries_handled": stats.queries_handled,
                    "success_count": stats.success_count,
                    "failure_count": stats.failure_count,
                    "success_rate": stats.success_rate,
                    "avg_confidence": stats.avg_confidence,
                    "avg_response_time": stats.avg_response_time,
                    "failure_modes": stats.failure_modes,
                    "last_used": (
                        stats.last_used.isoformat() if stats.last_used else None
                    ),
                },
            )

            if result:
                logger.debug("Persisted stats for strategy '%s'", strategy_name)

        except Exception as e:
            logger.error(
                "Error persisting strategy stats for '%s': %s", strategy_name, e
            )

    def _load_persisted_stats(self) -> None:
        """Lade Strategy-Statistiken aus Neo4j"""
        try:
            query = """
            MATCH (sp:StrategyPerformance)
            RETURN sp.strategy_name AS strategy_name,
                   sp.queries_handled AS queries_handled,
                   sp.success_count AS success_count,
                   sp.failure_count AS failure_count,
                   sp.success_rate AS success_rate,
                   sp.avg_confidence AS avg_confidence,
                   sp.avg_response_time AS avg_response_time,
                   sp.failure_modes AS failure_modes,
                   sp.last_used AS last_used
            """

            results = self._execute_query(query)

            for record in results:
                strategy_name = record["strategy_name"]

                stats = StrategyPerformance(
                    strategy_name=strategy_name,
                    queries_handled=record["queries_handled"] or 0,
                    success_count=record["success_count"] or 0,
                    failure_count=record["failure_count"] or 0,
                    success_rate=record["success_rate"] or 0.5,
                    avg_confidence=record["avg_confidence"] or 0.0,
                    avg_response_time=record["avg_response_time"] or 0.0,
                    total_response_time=0.0,  # Wird neu berechnet
                    typical_query_patterns=[],  # TODO: Load from separate nodes
                    failure_modes=record["failure_modes"] or [],
                    last_used=None,  # TODO: Parse datetime
                )

                # Recalculate total_response_time
                stats.total_response_time = (
                    stats.avg_response_time * stats.queries_handled
                )

                self.strategy_stats[strategy_name] = stats

            logger.info(
                "Loaded %d strategy statistics from Neo4j", len(self.strategy_stats)
            )

        except Exception as e:
            logger.error("Error loading persisted stats: %s", e)

    # ========================================================================
    # Utility Functions
    # ========================================================================

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Berechne Cosine Similarity zwischen zwei Vektoren"""
        try:
            import math

            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            magnitude1 = math.sqrt(sum(a * a for a in vec1))
            magnitude2 = math.sqrt(sum(b * b for b in vec2))

            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0

            return dot_product / (magnitude1 * magnitude2)

        except Exception:
            return 0.0

    def get_strategy_stats(
        self, strategy_name: str, use_cache: bool = True
    ) -> Optional[StrategyPerformance]:
        """
        Hole Performance-Stats für Strategy

        Args:
            strategy_name: Name der Strategy
            use_cache: Nutze TTL-Cache (default: True)

        Returns:
            StrategyPerformance oder None
        """
        # Cache Lookup
        if use_cache and strategy_name in self._stats_cache:
            logger.debug(f"Stats cache HIT for strategy '{strategy_name}'")
            return self._stats_cache[strategy_name]

        # Get from in-memory stats
        stats = self.strategy_stats.get(strategy_name)

        # Cache Write
        if use_cache and stats is not None:
            self._stats_cache[strategy_name] = stats
            logger.debug(f"Cached stats for strategy '{strategy_name}'")

        return stats

    def get_all_stats(self) -> Dict[str, StrategyPerformance]:
        """Hole alle Strategy-Stats"""
        return self.strategy_stats.copy()

    def get_top_strategies(self, n: int = 5) -> List[Tuple[str, float]]:
        """
        Hole Top N Strategies basierend auf Performance

        Returns:
            List of (strategy_name, score) tuples
        """
        strategy_scores = []

        for name, stats in self.strategy_stats.items():
            score = self._calculate_performance_score(stats)
            strategy_scores.append((name, score))

        strategy_scores.sort(key=lambda x: x[1], reverse=True)
        return strategy_scores[:n]

    def reset_epsilon(self, new_epsilon: float = 0.1) -> None:
        """Reset Epsilon für neue Exploration-Phase"""
        self.config.epsilon = new_epsilon
        logger.info("Reset epsilon to %.3f", new_epsilon)

    def clear_cache(self, cache_type: Optional[str] = None) -> None:
        """
        Leert Caches

        Args:
            cache_type: 'stats', 'patterns', oder None für beide
        """
        if cache_type == "stats" or cache_type is None:
            self._stats_cache.clear()
            logger.info("Strategy stats cache cleared")

        if cache_type == "patterns" or cache_type is None:
            self._pattern_cache.clear()
            logger.info("Query pattern cache cleared")

    def get_cache_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Gibt Cache-Statistiken zurück

        Returns:
            Dict mit Cache-Größen und TTLs
        """
        return {
            "stats_cache": {
                "size": len(self._stats_cache),
                "maxsize": self._stats_cache.maxsize,
                "ttl": self._stats_cache.ttl,
            },
            "pattern_cache": {
                "size": len(self._pattern_cache),
                "maxsize": self._pattern_cache.maxsize,
                "ttl": self._pattern_cache.ttl,
            },
            "in_memory_stats": {
                "strategies": len(self.strategy_stats),
                "total_queries": self.total_queries,
                "usage_history_size": len(self.usage_history),
            },
        }


# ============================================================================
# Exception Handling
# ============================================================================


class MetaLearningException(KAIException):
    """Base exception für Meta-Learning Fehler"""


class StrategySelectionException(MetaLearningException):
    """Exception bei Strategy-Auswahl"""


class PersistenceException(MetaLearningException):
    """Exception bei Neo4j Persistence"""
