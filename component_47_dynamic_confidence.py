# component_47_dynamic_confidence.py
"""
Dynamic Confidence System für KAI - Phase 1: Cognitive Resonance

Implementiert dynamische Confidence-Berechnung mit:
- Temporal Decay: Confidence nimmt mit Zeit ab (konfigurierbare Halbwertszeit)
- Usage Reinforcement: Häufige Nutzung erhöht Confidence (tracked via Episodic Memory)
- On-the-fly Berechnung: Keine vorberechneten Werte, dynamisch bei jedem Query
- Backwards Compatible: Drop-in replacement für ConfidenceManager

Design-Prinzipien:
- Confidence wird NICHT mehr statisch gespeichert
- Jeder Query berechnet Confidence dynamisch basierend auf:
  * Base Confidence (aus Relation Property)
  * Alter des Facts (Temporal Decay)
  * Nutzungshäufigkeit (Usage Reinforcement)
  * Recency (kürzliche Nutzung reduziert Decay)
- Episodic Memory tracked automatisch Fact-Nutzung
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from component_15_logging_config import get_logger
from component_confidence_manager import (
    CombinationStrategy,
    ConfidenceLevel,
    ConfidenceManager,
    ConfidenceMetrics,
)

logger = get_logger(__name__)


# ==================== CONFIGURATION ====================


@dataclass
class DynamicConfidenceConfig:
    """
    Konfiguration für Dynamic Confidence System.

    Attributes:
        half_life_days: Halbwertszeit für Temporal Decay (Default: 500 Tage während Entwicklung)
        min_confidence: Minimale Confidence nach Decay (Default: 0.3)
        usage_boost_per_use: Boost pro Nutzung (Default: 0.05 = 5%)
        max_usage_boost: Maximaler Usage Boost (Default: 0.5 = 50%)
        recency_threshold_days: Schwelle für Recency Bonus (Default: 7 Tage)
        recency_boost_factor: Recency Boost Multiplikator (Default: 1.2 = 20%)
        enable_temporal_decay: Temporalen Decay aktivieren (Default: True)
        enable_usage_reinforcement: Usage Reinforcement aktivieren (Default: True)
    """

    half_life_days: float = 500.0  # Sehr lang während Entwicklung
    min_confidence: float = 0.3
    usage_boost_per_use: float = 0.05
    max_usage_boost: float = 0.5
    recency_threshold_days: int = 7
    recency_boost_factor: float = 1.2
    enable_temporal_decay: bool = True
    enable_usage_reinforcement: bool = True

    def __post_init__(self):
        """Validierung der Konfiguration."""
        if self.half_life_days <= 0:
            raise ValueError(f"half_life_days muss > 0 sein: {self.half_life_days}")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError(
                f"min_confidence muss in [0, 1] liegen: {self.min_confidence}"
            )
        if self.usage_boost_per_use < 0:
            raise ValueError(
                f"usage_boost_per_use muss >= 0 sein: {self.usage_boost_per_use}"
            )
        if not 0.0 <= self.max_usage_boost <= 1.0:
            raise ValueError(
                f"max_usage_boost muss in [0, 1] liegen: {self.max_usage_boost}"
            )


@dataclass
class UsageStatistics:
    """
    Nutzungsstatistiken für einen Fact.

    Attributes:
        usage_count: Anzahl der Nutzungen (über Episodes getrackt)
        last_used: Zeitpunkt der letzten Nutzung (None wenn nie genutzt)
        first_used: Zeitpunkt der ersten Nutzung (None wenn nie genutzt)
        episode_ids: Liste der Episode-IDs, in denen dieser Fact genutzt wurde
    """

    usage_count: int = 0
    last_used: Optional[datetime] = None
    first_used: Optional[datetime] = None
    episode_ids: List[str] = field(default_factory=list)


# ==================== DYNAMIC CONFIDENCE MANAGER ====================


class DynamicConfidenceManager:
    """
    Manager für dynamische Confidence-Berechnung mit Temporal Decay und Usage Reinforcement.

    Diese Klasse ersetzt/erweitert den statischen ConfidenceManager mit dynamischer
    Berechnung basierend auf Zeit und Nutzung.

    Verantwortlichkeiten:
    - On-the-fly Berechnung von Confidence (nicht gespeichert)
    - Temporal Decay (exponentieller Zerfall über Zeit)
    - Usage Reinforcement (Nutzung erhöht Confidence)
    - Tracking von Fact-Nutzung in Episodic Memory
    - Backwards Compatibility mit ConfidenceManager API
    """

    def __init__(
        self,
        netzwerk,
        config: Optional[DynamicConfidenceConfig] = None,
    ):
        """
        Initialisiert den DynamicConfidenceManager.

        Args:
            netzwerk: KonzeptNetzwerk-Instanz für DB-Zugriff
            config: Optional - Custom-Konfiguration (Default: DynamicConfidenceConfig())
        """
        self.netzwerk = netzwerk
        self.config = config or DynamicConfidenceConfig()

        # Basis-Manager für Backwards Compatibility
        self.base_manager = ConfidenceManager()

        logger.info(
            "DynamicConfidenceManager initialisiert",
            extra={
                "half_life_days": self.config.half_life_days,
                "temporal_decay": self.config.enable_temporal_decay,
                "usage_reinforcement": self.config.enable_usage_reinforcement,
            },
        )

    # ==================== CORE DYNAMIC CONFIDENCE CALCULATION ====================

    def calculate_dynamic_confidence(
        self,
        base_confidence: float,
        timestamp: Optional[datetime],
        subject: str,
        relation: str,
        object_: str,
        current_time: Optional[datetime] = None,
    ) -> ConfidenceMetrics:
        """
        Berechnet dynamische Confidence on-the-fly.

        Dies ist die Hauptmethode des Systems. Sie kombiniert:
        1. Temporal Decay (Confidence nimmt mit Zeit ab)
        2. Usage Reinforcement (häufige Nutzung erhöht Confidence)
        3. Recency Bonus (kürzliche Nutzung reduziert Decay)

        Args:
            base_confidence: Basis-Confidence aus Relation Property (0.0-1.0)
            timestamp: Zeitpunkt der Fact-Erstellung (None -> kein Decay)
            subject: Subject des Facts
            relation: Relationstyp
            object_: Object des Facts
            current_time: Aktueller Zeitpunkt (Default: datetime.now())

        Returns:
            ConfidenceMetrics mit dynamisch berechneter Confidence und Erklärung

        Example:
            >>> manager.calculate_dynamic_confidence(
            ...     base_confidence=0.9,
            ...     timestamp=datetime(2024, 1, 1),
            ...     subject="hund",
            ...     relation="IS_A",
            ...     object_="säugetier"
            ... )
            ConfidenceMetrics(value=0.87, level=HIGH, explanation="...")
        """
        if current_time is None:
            current_time = datetime.now()

        # Validierung
        if not 0.0 <= base_confidence <= 1.0:
            raise ValueError(
                f"base_confidence muss in [0, 1] liegen: {base_confidence}"
            )

        explanation_parts = []

        # Start mit Base Confidence
        current_conf = base_confidence
        explanation_parts.append(f"Base: {base_confidence:.3f}")

        # 1. Temporal Decay
        if self.config.enable_temporal_decay and timestamp is not None:
            decay_metrics = self._apply_temporal_decay(
                current_conf, timestamp, current_time
            )
            current_conf = decay_metrics.value
            explanation_parts.append(f"Decay: {current_conf:.3f}")
        else:
            explanation_parts.append("Decay: disabled")

        # 2. Usage Reinforcement
        if self.config.enable_usage_reinforcement:
            usage_stats = self._get_usage_statistics(subject, relation, object_)

            reinforcement_result = self._apply_usage_reinforcement(
                current_conf, usage_stats, current_time
            )
            current_conf = reinforcement_result["confidence"]

            if usage_stats.usage_count > 0:
                explanation_parts.append(
                    f"Usage: {usage_stats.usage_count}x "
                    f"(+{reinforcement_result['usage_boost']:.2%})"
                )

                if reinforcement_result["recency_applied"]:
                    explanation_parts.append(
                        f"Recency: {self.config.recency_boost_factor}x"
                    )
            else:
                explanation_parts.append("Usage: 0x")
        else:
            explanation_parts.append("Reinforcement: disabled")

        # Final Confidence (clamped)
        final_confidence = max(self.config.min_confidence, min(1.0, current_conf))

        # Klassifiziere
        level = self.base_manager.classify_confidence(final_confidence)

        explanation = (
            " | ".join(explanation_parts) + f" -> Final: {final_confidence:.3f}"
        )

        return ConfidenceMetrics(
            value=final_confidence,
            level=level,
            source_confidences=[base_confidence],
            explanation=explanation,
        )

    # ==================== TEMPORAL DECAY ====================

    def _apply_temporal_decay(
        self,
        confidence: float,
        timestamp: datetime,
        current_time: datetime,
    ) -> ConfidenceMetrics:
        """
        Wendet exponentiellen Temporal Decay an.

        Formel: confidence_new = confidence_original * 2^(-age / half_life)

        Args:
            confidence: Aktuelle Confidence
            timestamp: Zeitpunkt der Fact-Erstellung
            current_time: Aktueller Zeitpunkt

        Returns:
            ConfidenceMetrics mit gedecaytem Wert
        """
        # Berechne Alter in Tagen
        age_delta = current_time - timestamp
        age_days = age_delta.total_seconds() / (24 * 3600)

        # Exponentieller Decay
        decay_factor = math.pow(2, -age_days / self.config.half_life_days)
        decayed_confidence = confidence * decay_factor

        # Clipping
        decayed_confidence = max(self.config.min_confidence, decayed_confidence)

        level = self.base_manager.classify_confidence(decayed_confidence)

        explanation = (
            f"Temporal Decay: {age_days:.0f} Tage alt "
            f"(Halbwertszeit: {self.config.half_life_days:.0f} Tage) "
            f"-> {confidence:.3f} * {decay_factor:.3f} = {decayed_confidence:.3f}"
        )

        return ConfidenceMetrics(
            value=decayed_confidence,
            level=level,
            original_value=confidence,
            decay_applied=True,
            explanation=explanation,
        )

    # ==================== USAGE REINFORCEMENT ====================

    def _apply_usage_reinforcement(
        self,
        confidence: float,
        usage_stats: UsageStatistics,
        current_time: datetime,
    ) -> Dict[str, Any]:
        """
        Wendet Usage Reinforcement basierend auf Nutzungsstatistiken an.

        Args:
            confidence: Aktuelle Confidence (nach Decay)
            usage_stats: Nutzungsstatistiken
            current_time: Aktueller Zeitpunkt

        Returns:
            Dict mit:
            - confidence: Neue Confidence nach Reinforcement
            - usage_boost: Angewendeter Usage Boost (0.0-max_usage_boost)
            - recency_applied: Wurde Recency Bonus angewendet?
        """
        # Usage Boost: Linear mit Anzahl Nutzungen (mit Max)
        usage_boost = min(
            self.config.max_usage_boost,
            usage_stats.usage_count * self.config.usage_boost_per_use,
        )

        # Recency Bonus: Wenn kürzlich genutzt, reduziere Decay-Effekt
        recency_applied = False
        recency_factor = 1.0

        if usage_stats.last_used is not None:
            days_since_use = (current_time - usage_stats.last_used).days

            if days_since_use < self.config.recency_threshold_days:
                recency_factor = self.config.recency_boost_factor
                recency_applied = True

        # Anwenden: conf_new = conf * (1 + usage_boost) * recency_factor
        new_confidence = confidence * (1 + usage_boost) * recency_factor

        # Clipping
        new_confidence = min(1.0, new_confidence)

        return {
            "confidence": new_confidence,
            "usage_boost": usage_boost,
            "recency_applied": recency_applied,
        }

    # ==================== USAGE TRACKING ====================

    def _get_usage_statistics(
        self, subject: str, relation: str, object_: str
    ) -> UsageStatistics:
        """
        Fragt Episodic Memory nach Nutzungsstatistiken für einen Fact.

        Query:
        MATCH (f:Fact {subject: $s, relation: $r, object: $o})<-[learned:LEARNED_FACT]-(e:Episode)
        RETURN count(e) AS usage_count,
               max(e.timestamp) AS last_used,
               min(e.timestamp) AS first_used,
               collect(e.id) AS episode_ids

        Args:
            subject: Subject des Facts
            relation: Relationstyp
            object_: Object des Facts

        Returns:
            UsageStatistics mit allen Nutzungsinformationen
        """
        if not self.netzwerk or not self.netzwerk.driver:
            logger.warning("Kein Netzwerk-Driver verfügbar, Usage Statistics = 0")
            return UsageStatistics()

        try:
            with self.netzwerk.driver.session(database="neo4j") as session:
                result = session.run(
                    """
                    MATCH (f:Fact {subject: $subject, relation: $relation, object: $object})
                    OPTIONAL MATCH (e:Episode)-[learned:LEARNED_FACT]->(f)
                    WITH f, count(DISTINCT e) AS usage_count,
                         max(e.timestamp) AS last_used,
                         min(e.timestamp) AS first_used,
                         collect(DISTINCT e.id) AS episode_ids
                    RETURN usage_count,
                           last_used,
                           first_used,
                           episode_ids
                    """,
                    subject=subject.lower(),
                    relation=relation.upper(),
                    object=object_.lower(),
                )

                record = result.single()

                if record:
                    usage_count = record["usage_count"] or 0
                    last_used_ts = record["last_used"]
                    first_used_ts = record["first_used"]
                    episode_ids = record["episode_ids"] or []

                    # Convert Neo4j timestamps to datetime
                    last_used = None
                    first_used = None

                    if last_used_ts is not None:
                        # Neo4j timestamp() gibt Millisekunden seit Epoch zurück
                        last_used = datetime.fromtimestamp(last_used_ts / 1000.0)

                    if first_used_ts is not None:
                        first_used = datetime.fromtimestamp(first_used_ts / 1000.0)

                    return UsageStatistics(
                        usage_count=usage_count,
                        last_used=last_used,
                        first_used=first_used,
                        episode_ids=episode_ids,
                    )

                return UsageStatistics()

        except Exception as e:
            logger.error(
                f"Fehler beim Abrufen von Usage Statistics: {e}",
                exc_info=True,
                extra={"subject": subject, "relation": relation, "object": object_},
            )
            return UsageStatistics()

    def track_usage(
        self,
        subject: str,
        relation: str,
        object_: str,
        episode_id: str,
    ) -> bool:
        """
        Tracked die Nutzung eines Facts in einer Episode.

        Verknüpft den Fact-Node mit der Episode über LEARNED_FACT Relationship.
        Falls der Fact-Node noch nicht existiert, wird er erstellt.

        Args:
            subject: Subject des Facts
            relation: Relationstyp
            object_: Object des Facts
            episode_id: ID der Episode, in der der Fact genutzt wurde

        Returns:
            True wenn erfolgreich getrackt, False bei Fehler

        Example:
            >>> manager.track_usage("hund", "IS_A", "säugetier", episode_id)
            True
        """
        if not self.netzwerk or not self.netzwerk.driver:
            logger.warning("Kein Netzwerk-Driver verfügbar, Usage nicht getrackt")
            return False

        # Nutze die bestehende Methode aus netzwerk_memory
        success = self.netzwerk.link_fact_to_episode(
            subject, relation, object_, episode_id
        )

        if success:
            logger.debug(
                "Usage getrackt",
                extra={
                    "subject": subject,
                    "relation": relation,
                    "object": object_,
                    "episode_id": episode_id,
                },
            )

        return success

    # ==================== BACKWARDS COMPATIBILITY ====================

    def classify_confidence(self, confidence: float) -> ConfidenceLevel:
        """Wrapper für Backwards Compatibility."""
        return self.base_manager.classify_confidence(confidence)

    def should_auto_accept(
        self, confidence: float, auto_detected: bool = False
    ) -> bool:
        """Wrapper für Backwards Compatibility."""
        return self.base_manager.should_auto_accept(confidence, auto_detected)

    def should_ask_user(self, confidence: float) -> bool:
        """Wrapper für Backwards Compatibility."""
        return self.base_manager.should_ask_user(confidence)

    def should_reject(self, confidence: float) -> bool:
        """Wrapper für Backwards Compatibility."""
        return self.base_manager.should_reject(confidence)

    def combine_confidences(
        self,
        confidences: List[float],
        strategy: CombinationStrategy = CombinationStrategy.MINIMUM,
        weights: Optional[List[float]] = None,
    ) -> ConfidenceMetrics:
        """Wrapper für Backwards Compatibility."""
        return self.base_manager.combine_confidences(confidences, strategy, weights)

    def calculate_graph_traversal_confidence(
        self, edge_confidences: List[float]
    ) -> ConfidenceMetrics:
        """Wrapper für Backwards Compatibility."""
        return self.base_manager.calculate_graph_traversal_confidence(edge_confidences)

    def calculate_rule_confidence(
        self, premise_confidences: List[float], rule_strength: float = 1.0
    ) -> ConfidenceMetrics:
        """Wrapper für Backwards Compatibility."""
        return self.base_manager.calculate_rule_confidence(
            premise_confidences, rule_strength
        )

    def calculate_hypothesis_confidence(
        self,
        coverage: float,
        simplicity: float,
        coherence: float,
        specificity: float,
        weights: Optional[Dict[str, float]] = None,
    ) -> ConfidenceMetrics:
        """Wrapper für Backwards Compatibility."""
        return self.base_manager.calculate_hypothesis_confidence(
            coverage, simplicity, coherence, specificity, weights
        )

    def generate_ui_feedback(self, confidence: float, context: str = "") -> str:
        """Wrapper für Backwards Compatibility."""
        return self.base_manager.generate_ui_feedback(confidence, context)

    def explain_confidence(
        self, metrics: ConfidenceMetrics, verbose: bool = False
    ) -> str:
        """Wrapper für Backwards Compatibility."""
        return self.base_manager.explain_confidence(metrics, verbose)


# ==================== GLOBAL INSTANCE ====================

_global_dynamic_manager: Optional[DynamicConfidenceManager] = None


def get_dynamic_confidence_manager(
    netzwerk=None, config: Optional[DynamicConfidenceConfig] = None
) -> DynamicConfidenceManager:
    """
    Gibt die globale DynamicConfidenceManager-Instanz zurück.

    Erstellt eine neue Instanz beim ersten Aufruf (Lazy Initialization).

    Args:
        netzwerk: KonzeptNetzwerk-Instanz (nur beim ersten Aufruf erforderlich)
        config: Optional - Custom-Konfiguration

    Returns:
        Globale DynamicConfidenceManager-Instanz

    Raises:
        ValueError: Wenn beim ersten Aufruf kein netzwerk übergeben wurde
    """
    global _global_dynamic_manager

    if _global_dynamic_manager is None:
        if netzwerk is None:
            raise ValueError(
                "Beim ersten Aufruf muss netzwerk-Parameter übergeben werden"
            )

        _global_dynamic_manager = DynamicConfidenceManager(netzwerk, config)
        logger.info("Globale DynamicConfidenceManager-Instanz erstellt")

    return _global_dynamic_manager


# ==================== BEISPIEL-USAGE ====================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Mock Netzwerk für Demo
    class MockNetzwerk:
        driver = None

    netzwerk = MockNetzwerk()

    # Erstelle Manager mit kurzer Halbwertszeit für Demo
    config = DynamicConfidenceConfig(
        half_life_days=30,  # Kurz für Demo
        usage_boost_per_use=0.1,
        max_usage_boost=0.5,
    )
    manager = DynamicConfidenceManager(netzwerk, config)

    print("=== Dynamic Confidence System Demo ===\n")

    # Beispiel 1: Neuer Fact (keine Usage, kein Decay)
    print("Beispiel 1: Neuer Fact (heute erstellt)")
    metrics1 = manager.calculate_dynamic_confidence(
        base_confidence=0.8,
        timestamp=datetime.now(),
        subject="hund",
        relation="IS_A",
        object_="säugetier",
    )
    print(f"Confidence: {metrics1.value:.3f} ({metrics1.level.value})")
    print(f"Erklärung: {metrics1.explanation}\n")

    # Beispiel 2: Alter Fact (mit Decay)
    print("Beispiel 2: Alter Fact (90 Tage alt, keine Usage)")
    old_timestamp = datetime.now() - timedelta(days=90)
    metrics2 = manager.calculate_dynamic_confidence(
        base_confidence=0.8,
        timestamp=old_timestamp,
        subject="katze",
        relation="IS_A",
        object_="säugetier",
    )
    print(f"Confidence: {metrics2.value:.3f} ({metrics2.level.value})")
    print(f"Erklärung: {metrics2.explanation}\n")

    # Beispiel 3: Mit Mock Usage Statistics
    print("Beispiel 3: Alter Fact mit hoher Usage (Reinforcement)")

    # Override _get_usage_statistics für Demo
    original_get_stats = manager._get_usage_statistics

    def mock_usage_stats(subject, relation, object_):
        return UsageStatistics(
            usage_count=10,
            last_used=datetime.now() - timedelta(days=2),
            first_used=datetime.now() - timedelta(days=100),
        )

    manager._get_usage_statistics = mock_usage_stats

    metrics3 = manager.calculate_dynamic_confidence(
        base_confidence=0.7,
        timestamp=datetime.now() - timedelta(days=90),
        subject="vogel",
        relation="IS_A",
        object_="tier",
    )
    print(f"Confidence: {metrics3.value:.3f} ({metrics3.level.value})")
    print(f"Erklärung: {metrics3.explanation}\n")

    # Restore
    manager._get_usage_statistics = original_get_stats

    # Beispiel 4: Backwards Compatibility
    print("Beispiel 4: Backwards Compatibility (Kombinierung)")
    combined = manager.combine_confidences(
        [0.9, 0.8, 0.85], strategy=CombinationStrategy.AVERAGE
    )
    print(f"Kombinierte Confidence: {combined.value:.3f}")
    print(f"Erklärung: {combined.explanation}\n")
