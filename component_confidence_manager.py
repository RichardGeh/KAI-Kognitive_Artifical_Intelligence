# component_confidence_manager.py
"""
Confidence-Based Learning Management für KAI

Zentrales System für:
- Einheitliches Confidence-Scoring über alle Reasoning Engines
- Threshold-basierte Entscheidungen (Auto-Accept, Ask-User, Reject)
- Confidence-Decay für zeitbasierte Reduktion veralteter Fakten
- Confidence-Level-Klassifizierung

Design-Prinzipien:
- Single source of truth für alle Confidence-Berechnungen
- Konsistente Threshold-Werte über alle Komponenten
- Transparente Erklärungen für Confidence-Werte
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional

from component_15_logging_config import get_logger

logger = get_logger(__name__)


# ==================== ENUMS & CONSTANTS ====================


class ConfidenceLevel(Enum):
    """
    Klassifizierung von Confidence-Werten in intuitive Kategorien.

    Mapping:
    - HIGH: >= 0.8 (Sichere Ausführung ohne Rückfrage)
    - MEDIUM: 0.5-0.8 (Bestätigung erforderlich)
    - LOW: 0.3-0.5 (Warnung, unsichere Inferenz)
    - UNKNOWN: < 0.3 (Rückfrage erforderlich, keine Aktion)
    """

    HIGH = "high"  # >= 0.8: Direct execution
    MEDIUM = "medium"  # 0.5-0.8: Ask user for confirmation
    LOW = "low"  # 0.3-0.5: Warning, uncertain inference
    UNKNOWN = "unknown"  # < 0.3: Clarification required


class CombinationStrategy(Enum):
    """
    Strategien zum Kombinieren mehrerer Confidence-Werte.

    - MINIMUM: Weakest link (konservativ, für konjunktive Inferenzen)
    - MAXIMUM: Strongest link (optimistisch, für disjunktive Inferenzen)
    - AVERAGE: Arithmetisches Mittel (neutral)
    - WEIGHTED_AVERAGE: Gewichtetes Mittel (wenn manche Fakten wichtiger sind)
    - BAYESIAN: Noisy-OR für redundante Evidenz
    """

    MINIMUM = "minimum"
    MAXIMUM = "maximum"
    AVERAGE = "average"
    WEIGHTED_AVERAGE = "weighted_average"
    BAYESIAN = "bayesian"


# Globale Threshold-Werte (Single Source of Truth)
class ConfidenceThresholds:
    """
    Zentrale Definition aller Confidence-Thresholds im System.

    Diese Werte werden von allen Komponenten verwendet:
    - GoalPlanner: Für Confidence Gates (Clarification/Confirmation/Execution)
    - InferenceHandler: Für Reasoning-Entscheidungen
    - ResponseFormatter: Für UI-Feedback-Generierung
    """

    # Primäre Thresholds für Aktionen
    AUTO_ACCEPT = 0.8  # >= 0.8: Sichere Ausführung ohne Rückfrage
    ASK_USER = 0.5  # 0.5-0.8: Bestätigung erforderlich
    REJECT = 0.3  # < 0.3: Zu unsicher, Clarification erforderlich

    # Spezielle Thresholds für auto-detected definitions
    AUTO_DETECT_HIGH = 0.85  # Höherer Threshold für autonomes Lernen

    # Thresholds für Confidence-Level-Klassifizierung
    LEVEL_HIGH = 0.8
    LEVEL_MEDIUM = 0.5
    LEVEL_LOW = 0.3

    # Decay-Parameter
    DECAY_HALF_LIFE_DAYS = 180  # Nach 180 Tagen ist Confidence auf 50%
    DECAY_MIN_CONFIDENCE = 0.3  # Confidence wird nie unter diesen Wert reduziert


# ==================== DATA STRUCTURES ====================


@dataclass
class ConfidenceMetrics:
    """
    Erweiterte Confidence-Metriken für transparente Erklärungen.

    Attributes:
        value: Finale Confidence (0.0-1.0)
        level: Klassifizierung (HIGH, MEDIUM, LOW, UNKNOWN)
        source_confidences: Liste von Input-Confidence-Werten
        combination_strategy: Verwendete Kombinierungsstrategie
        decay_applied: Wurde zeitbasierter Decay angewendet?
        original_value: Ursprünglicher Wert vor Decay (falls anwendbar)
        explanation: Menschenlesbare Erklärung
    """

    value: float
    level: ConfidenceLevel
    source_confidences: List[float] = field(default_factory=list)
    combination_strategy: Optional[CombinationStrategy] = None
    decay_applied: bool = False
    original_value: Optional[float] = None
    explanation: str = ""

    def __post_init__(self):
        """Validierung der Confidence-Werte."""
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"Confidence muss in [0, 1] liegen: {self.value}")


# ==================== CONFIDENCE MANAGER ====================


class ConfidenceManager:
    """
    Zentraler Manager für alle Confidence-Berechnungen im System.

    Verantwortlichkeiten:
    - Einheitliches Confidence-Scoring über alle Reasoning Engines
    - Threshold-basierte Entscheidungen
    - Confidence-Decay für veraltete Fakten
    - Confidence-Level-Klassifizierung
    - Transparente Erklärungen
    """

    def __init__(self, thresholds: Optional[ConfidenceThresholds] = None):
        """
        Initialisiert den ConfidenceManager.

        Args:
            thresholds: Optionale Custom-Thresholds (Standard: ConfidenceThresholds)
        """
        self.thresholds = thresholds or ConfidenceThresholds()
        logger.info("ConfidenceManager initialisiert mit Standard-Thresholds")

    # ==================== CLASSIFICATION ====================

    def classify_confidence(self, confidence: float) -> ConfidenceLevel:
        """
        Klassifiziert einen Confidence-Wert in HIGH/MEDIUM/LOW/UNKNOWN.

        Args:
            confidence: Der zu klassifizierende Confidence-Wert (0.0-1.0)

        Returns:
            ConfidenceLevel enum
        """
        if confidence >= self.thresholds.LEVEL_HIGH:
            return ConfidenceLevel.HIGH
        elif confidence >= self.thresholds.LEVEL_MEDIUM:
            return ConfidenceLevel.MEDIUM
        elif confidence >= self.thresholds.LEVEL_LOW:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.UNKNOWN

    def should_auto_accept(
        self, confidence: float, auto_detected: bool = False
    ) -> bool:
        """
        Entscheidet ob eine Aktion automatisch ausgeführt werden soll.

        Args:
            confidence: Confidence der Aktion
            auto_detected: True für auto-erkannte Definitionen (höherer Threshold)

        Returns:
            True wenn Confidence >= Threshold (0.8 oder 0.85)
        """
        threshold = (
            self.thresholds.AUTO_DETECT_HIGH
            if auto_detected
            else self.thresholds.AUTO_ACCEPT
        )
        return confidence >= threshold

    def should_ask_user(self, confidence: float) -> bool:
        """
        Entscheidet ob Benutzerbestätigung erforderlich ist.

        Args:
            confidence: Confidence der Aktion

        Returns:
            True wenn 0.5 <= confidence < 0.8
        """
        return self.thresholds.ASK_USER <= confidence < self.thresholds.AUTO_ACCEPT

    def should_reject(self, confidence: float) -> bool:
        """
        Entscheidet ob eine Aktion abgelehnt werden soll (zu unsicher).

        Args:
            confidence: Confidence der Aktion

        Returns:
            True wenn confidence < 0.3
        """
        return confidence < self.thresholds.REJECT

    # ==================== COMBINATION ====================

    def combine_confidences(
        self,
        confidences: List[float],
        strategy: CombinationStrategy = CombinationStrategy.MINIMUM,
        weights: Optional[List[float]] = None,
    ) -> ConfidenceMetrics:
        """
        Kombiniert mehrere Confidence-Werte zu einem Gesamt-Confidence.

        Args:
            confidences: Liste von Confidence-Werten (0.0-1.0)
            strategy: Kombinierungsstrategie
            weights: Optionale Gewichte (nur für WEIGHTED_AVERAGE)

        Returns:
            ConfidenceMetrics mit kombiniertem Wert und Erklärung
        """
        if not confidences:
            logger.warning("Leere Confidence-Liste übergeben")
            return ConfidenceMetrics(
                value=0.0,
                level=ConfidenceLevel.UNKNOWN,
                explanation="Keine Input-Confidences vorhanden",
            )

        # Validierung
        for conf in confidences:
            if not 0.0 <= conf <= 1.0:
                raise ValueError(f"Ungültiger Confidence-Wert: {conf}")

        # Berechne kombinierten Wert
        if strategy == CombinationStrategy.MINIMUM:
            combined = min(confidences)
            explanation = f"Minimum von {len(confidences)} Werten (weakest link)"

        elif strategy == CombinationStrategy.MAXIMUM:
            combined = max(confidences)
            explanation = f"Maximum von {len(confidences)} Werten (strongest link)"

        elif strategy == CombinationStrategy.AVERAGE:
            combined = sum(confidences) / len(confidences)
            explanation = f"Durchschnitt von {len(confidences)} Werten"

        elif strategy == CombinationStrategy.WEIGHTED_AVERAGE:
            if weights is None or len(weights) != len(confidences):
                raise ValueError(
                    "Gewichte müssen für WEIGHTED_AVERAGE angegeben werden"
                )

            # Normalisiere Gewichte
            total_weight = sum(weights)
            if total_weight == 0:
                raise ValueError("Summe der Gewichte darf nicht 0 sein")

            combined = sum(c * w for c, w in zip(confidences, weights)) / total_weight
            explanation = f"Gewichteter Durchschnitt von {len(confidences)} Werten"

        elif strategy == CombinationStrategy.BAYESIAN:
            # Noisy-OR: P(E | C1, C2, ...) = 1 - ∏(1 - P(Ci))
            product = 1.0
            for conf in confidences:
                product *= 1.0 - conf
            combined = 1.0 - product
            explanation = f"Noisy-OR Kombination von {len(confidences)} Evidenzen"

        else:
            raise ValueError(f"Unbekannte Kombinierungsstrategie: {strategy}")

        # Klassifiziere
        level = self.classify_confidence(combined)

        return ConfidenceMetrics(
            value=combined,
            level=level,
            source_confidences=confidences,
            combination_strategy=strategy,
            explanation=explanation,
        )

    # ==================== DECAY ====================

    def apply_decay(
        self,
        confidence: float,
        timestamp: datetime,
        current_time: Optional[datetime] = None,
    ) -> ConfidenceMetrics:
        """
        Wendet zeitbasierten Confidence-Decay auf veraltete Fakten an.

        Decay-Formel: confidence_new = max(min_conf, conf_original * 2^(-age / half_life))

        Args:
            confidence: Ursprünglicher Confidence-Wert
            timestamp: Zeitstempel der Fact-Erstellung
            current_time: Aktueller Zeitpunkt (Standard: datetime.now())

        Returns:
            ConfidenceMetrics mit gedecaytem Wert und Erklärung
        """
        if current_time is None:
            current_time = datetime.now()

        # Berechne Alter in Tagen
        age_delta = current_time - timestamp
        age_days = age_delta.total_seconds() / (24 * 3600)

        # Decay nur für Fakten > 30 Tage alt
        if age_days < 30:
            return ConfidenceMetrics(
                value=confidence,
                level=self.classify_confidence(confidence),
                original_value=confidence,
                decay_applied=False,
                explanation="Fakt zu neu für Decay (< 30 Tage)",
            )

        # Exponentieller Decay: conf_new = conf_original * 2^(-age / half_life)
        half_life_days = self.thresholds.DECAY_HALF_LIFE_DAYS
        decay_factor = math.pow(2, -age_days / half_life_days)

        decayed_confidence = confidence * decay_factor

        # Clipping: Nie unter min_confidence
        decayed_confidence = max(
            self.thresholds.DECAY_MIN_CONFIDENCE, decayed_confidence
        )

        # Klassifiziere
        level = self.classify_confidence(decayed_confidence)

        explanation = (
            f"Decay angewendet: {age_days:.0f} Tage alt -> "
            f"{confidence:.3f} -> {decayed_confidence:.3f} "
            f"(Halbwertszeit: {half_life_days} Tage)"
        )

        logger.debug(explanation)

        return ConfidenceMetrics(
            value=decayed_confidence,
            level=level,
            original_value=confidence,
            decay_applied=True,
            explanation=explanation,
        )

    # ==================== UTILITY ====================

    def get_threshold_for_action(self, action_type: str) -> float:
        """
        Gibt den passenden Threshold für einen Aktionstyp zurück.

        Args:
            action_type: "auto_accept", "ask_user", "reject", "auto_detect"

        Returns:
            Threshold-Wert
        """
        mapping = {
            "auto_accept": self.thresholds.AUTO_ACCEPT,
            "ask_user": self.thresholds.ASK_USER,
            "reject": self.thresholds.REJECT,
            "auto_detect": self.thresholds.AUTO_DETECT_HIGH,
        }

        if action_type not in mapping:
            raise ValueError(f"Unbekannter Action-Typ: {action_type}")

        return mapping[action_type]

    def generate_ui_feedback(self, confidence: float, context: str = "") -> str:
        """
        Generiert UI-Feedback basierend auf Confidence-Level.

        Args:
            confidence: Confidence-Wert
            context: Optionaler Kontext (z.B. "Antwort", "Fakt", "Hypothese")

        Returns:
            Menschenlesbare Feedback-Nachricht
        """
        level = self.classify_confidence(confidence)

        if level == ConfidenceLevel.HIGH:
            return f"{context} (Konfidenz: {confidence:.2f} - Hoch)"

        elif level == ConfidenceLevel.MEDIUM:
            return f"{context} (Konfidenz: {confidence:.2f} - Mittel, bitte bestätigen)"

        elif level == ConfidenceLevel.LOW:
            return (
                f"{context} (Konfidenz: {confidence:.2f} - Niedrig, unsichere Inferenz)"
            )

        else:  # UNKNOWN
            return f"{context} (Konfidenz: {confidence:.2f} - Unbekannt, weitere Evidenz benötigt)"

    def explain_confidence(
        self, metrics: ConfidenceMetrics, verbose: bool = False
    ) -> str:
        """
        Erstellt eine detaillierte Erklärung für einen Confidence-Wert.

        Args:
            metrics: ConfidenceMetrics-Objekt
            verbose: True für ausführliche Erklärung

        Returns:
            Menschenlesbare Erklärung
        """
        parts = []

        # Basis-Info
        parts.append(f"Confidence: {metrics.value:.3f} ({metrics.level.value.upper()})")

        # Erklärung
        if metrics.explanation:
            parts.append(metrics.explanation)

        # Decay-Info
        if metrics.decay_applied and metrics.original_value is not None:
            parts.append(
                f"Ursprünglicher Wert: {metrics.original_value:.3f} "
                f"(reduziert durch zeitbasierten Decay)"
            )

        # Kombinierungs-Info
        if verbose and metrics.source_confidences:
            parts.append(
                f"Quell-Confidences: {[f'{c:.2f}' for c in metrics.source_confidences]}"
            )
            if metrics.combination_strategy:
                parts.append(f"Strategie: {metrics.combination_strategy.value}")

        return " | ".join(parts)

    # ==================== SPECIALIZED METHODS ====================

    def calculate_graph_traversal_confidence(
        self, edge_confidences: List[float]
    ) -> ConfidenceMetrics:
        """
        Berechnet Confidence für Graph-Traversal (Multi-Hop Reasoning).

        Verwendet MINIMUM-Strategie (weakest link principle).

        Args:
            edge_confidences: Confidence-Werte aller Kanten im Pfad

        Returns:
            ConfidenceMetrics mit Gesamt-Confidence
        """
        return self.combine_confidences(
            edge_confidences, strategy=CombinationStrategy.MINIMUM
        )

    def calculate_rule_confidence(
        self, premise_confidences: List[float], rule_strength: float = 1.0
    ) -> ConfidenceMetrics:
        """
        Berechnet Confidence für Regel-basierte Inferenz.

        Verwendet MINIMUM für Prämissen, dann gewichtet mit Regel-Stärke.

        Args:
            premise_confidences: Confidence-Werte aller Prämissen
            rule_strength: Stärke der Regel (0.0-1.0, Standard: 1.0)

        Returns:
            ConfidenceMetrics mit Gesamt-Confidence
        """
        if not 0.0 <= rule_strength <= 1.0:
            raise ValueError(f"Regel-Stärke muss in [0, 1] liegen: {rule_strength}")

        # Min der Prämissen
        premise_confidence = min(premise_confidences) if premise_confidences else 1.0

        # Gewichtet mit Regel-Stärke
        combined = premise_confidence * rule_strength

        level = self.classify_confidence(combined)

        return ConfidenceMetrics(
            value=combined,
            level=level,
            source_confidences=premise_confidences + [rule_strength],
            combination_strategy=CombinationStrategy.MINIMUM,
            explanation=(
                f"Regel-Inferenz: min(Prämissen)={premise_confidence:.3f} "
                f"* Regel-Stärke={rule_strength:.3f} = {combined:.3f}"
            ),
        )

    def calculate_hypothesis_confidence(
        self,
        coverage: float,
        simplicity: float,
        coherence: float,
        specificity: float,
        weights: Optional[Dict[str, float]] = None,
    ) -> ConfidenceMetrics:
        """
        Berechnet Confidence für Abductive Hypothesen.

        Verwendet WEIGHTED_AVERAGE mit anpassbaren Gewichten.

        Args:
            coverage: Coverage-Score (0.0-1.0)
            simplicity: Simplicity-Score (0.0-1.0)
            coherence: Coherence-Score (0.0-1.0)
            specificity: Specificity-Score (0.0-1.0)
            weights: Optionale Custom-Gewichte (Standard: 0.3, 0.2, 0.3, 0.2)

        Returns:
            ConfidenceMetrics mit Gesamt-Confidence
        """
        if weights is None:
            weights = {
                "coverage": 0.3,
                "simplicity": 0.2,
                "coherence": 0.3,
                "specificity": 0.2,
            }

        confidences = [coverage, simplicity, coherence, specificity]
        weight_list = [
            weights["coverage"],
            weights["simplicity"],
            weights["coherence"],
            weights["specificity"],
        ]

        return self.combine_confidences(
            confidences,
            strategy=CombinationStrategy.WEIGHTED_AVERAGE,
            weights=weight_list,
        )


# ==================== GLOBAL INSTANCE ====================

# Singleton-ähnliche globale Instanz für einfachen Zugriff
_global_confidence_manager: Optional[ConfidenceManager] = None


def get_confidence_manager() -> ConfidenceManager:
    """
    Gibt die globale ConfidenceManager-Instanz zurück.

    Erstellt eine neue Instanz beim ersten Aufruf (Lazy Initialization).

    Returns:
        Globale ConfidenceManager-Instanz
    """
    global _global_confidence_manager
    if _global_confidence_manager is None:
        _global_confidence_manager = ConfidenceManager()
        logger.info("Globale ConfidenceManager-Instanz erstellt")
    return _global_confidence_manager


# ==================== BEISPIEL-USAGE ====================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Initialisiere Manager
    cm = ConfidenceManager()

    # Beispiel 1: Klassifizierung
    print("=== Beispiel 1: Klassifizierung ===")
    for conf in [0.95, 0.7, 0.4, 0.2]:
        level = cm.classify_confidence(conf)
        print(f"Confidence {conf:.2f} -> {level.value.upper()}")

    # Beispiel 2: Kombinierung
    print("\n=== Beispiel 2: Kombinierung ===")
    confidences = [0.9, 0.8, 0.85]

    for strategy in [
        CombinationStrategy.MINIMUM,
        CombinationStrategy.AVERAGE,
        CombinationStrategy.BAYESIAN,
    ]:
        metrics = cm.combine_confidences(confidences, strategy=strategy)
        print(f"{strategy.value}: {metrics.value:.3f}")

    # Beispiel 3: Decay
    print("\n=== Beispiel 3: Confidence-Decay ===")
    old_timestamp = datetime.now() - timedelta(days=365)  # 1 Jahr alt
    metrics = cm.apply_decay(0.9, old_timestamp)
    print(cm.explain_confidence(metrics, verbose=True))

    # Beispiel 4: Threshold-Entscheidungen
    print("\n=== Beispiel 4: Threshold-Entscheidungen ===")
    for conf in [0.9, 0.7, 0.2]:
        action = (
            "Auto-Accept"
            if cm.should_auto_accept(conf)
            else "Ask User" if cm.should_ask_user(conf) else "Reject"
        )
        print(f"Confidence {conf:.2f} -> {action}")
