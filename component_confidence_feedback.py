# component_confidence_feedback.py
"""
Confidence Feedback & Training System für KAI

Ermöglicht Benutzern, Feedback zu Confidence-Scores zu geben und das System
basierend auf diesem Feedback adaptiv zu verbessern.

Features:
- Benutzer-Feedback sammeln (👍 Correct, 👎 Incorrect, [WARNING] Uncertain)
- Feedback in Neo4j persistieren
- Confidence-Adjustierung basierend auf historischem Feedback
- Lernkurven-Analyse und Reporting
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from component_15_logging_config import get_logger

logger = get_logger(__name__)


# ==================== ENUMS & DATA STRUCTURES ====================


class FeedbackType(Enum):
    """
    Typ des Benutzer-Feedbacks.

    - CORRECT: Benutzer bestätigt die Antwort als korrekt
    - INCORRECT: Benutzer markiert die Antwort als inkorrekt
    - UNCERTAIN: Benutzer ist unsicher über die Korrektheit
    """

    CORRECT = "correct"
    INCORRECT = "incorrect"
    UNCERTAIN = "uncertain"


@dataclass
class ConfidenceFeedback:
    """
    Einzelnes Benutzer-Feedback zu einer Confidence-Bewertung.

    Attributes:
        feedback_id: Eindeutige ID
        relation_type: Typ der Relation (z.B. "IS_A", "HAS_PROPERTY")
        subject: Subjekt der Relation
        object: Objekt der Relation
        original_confidence: Ursprüngliche Confidence vor Feedback
        feedback_type: Art des Feedbacks (CORRECT/INCORRECT/UNCERTAIN)
        timestamp: Zeitpunkt des Feedbacks
        user_comment: Optionaler Kommentar des Benutzers
        adjusted_confidence: Neue Confidence nach Feedback (wird berechnet)
    """

    feedback_id: str
    relation_type: str
    subject: str
    object: str
    original_confidence: float
    feedback_type: FeedbackType
    timestamp: datetime = field(default_factory=datetime.now)
    user_comment: Optional[str] = None
    adjusted_confidence: Optional[float] = None


# ==================== CONFIDENCE FEEDBACK MANAGER ====================


class ConfidenceFeedbackManager:
    """
    Manager für Benutzer-Feedback und adaptives Confidence-Training.

    Verantwortlichkeiten:
    - Sammeln von Benutzer-Feedback
    - Persistieren in Neo4j
    - Berechnung von adjustierten Confidence-Werten
    - Lernkurven-Analyse
    """

    def __init__(self, netzwerk=None):
        """
        Initialisiert den FeedbackManager.

        Args:
            netzwerk: Optional - KonzeptNetzwerk-Instanz für Persistierung
        """
        self.netzwerk = netzwerk
        self._feedback_history: List[ConfidenceFeedback] = []

        # Lernparameter
        self.adjustment_rate = 0.1  # Wie stark Feedback die Confidence beeinflusst
        self.min_feedback_count = 3  # Mindestanzahl Feedback vor Adjustierung

        logger.info("ConfidenceFeedbackManager initialisiert")

    # ==================== FEEDBACK COLLECTION ====================

    def submit_feedback(
        self,
        relation_type: str,
        subject: str,
        object: str,
        original_confidence: float,
        feedback_type: FeedbackType,
        user_comment: Optional[str] = None,
    ) -> ConfidenceFeedback:
        """
        Erfasst Benutzer-Feedback für eine Relation.

        Args:
            relation_type: Typ der Relation
            subject: Subjekt
            object: Objekt
            original_confidence: Ursprüngliche Confidence
            feedback_type: Art des Feedbacks
            user_comment: Optionaler Kommentar

        Returns:
            ConfidenceFeedback-Objekt mit berechneter adjusted_confidence
        """
        import uuid

        feedback = ConfidenceFeedback(
            feedback_id=f"feedback-{uuid.uuid4().hex[:8]}",
            relation_type=relation_type,
            subject=subject,
            object=object,
            original_confidence=original_confidence,
            feedback_type=feedback_type,
            user_comment=user_comment,
        )

        # Berechne adjustierte Confidence
        feedback.adjusted_confidence = self._calculate_adjusted_confidence(feedback)

        # Füge zur Historie hinzu
        self._feedback_history.append(feedback)

        # Persistiere in Neo4j (falls verfügbar)
        if self.netzwerk:
            self._persist_feedback(feedback)

        logger.info(
            f"Feedback erfasst: {feedback_type.value} für {subject} {relation_type} {object} "
            f"(Original: {original_confidence:.2f} -> Adjusted: {feedback.adjusted_confidence:.2f})"
        )

        return feedback

    def submit_positive_feedback(
        self, relation_type: str, subject: str, object: str, original_confidence: float
    ) -> ConfidenceFeedback:
        """Convenience-Methode für positives Feedback (👍)."""
        return self.submit_feedback(
            relation_type, subject, object, original_confidence, FeedbackType.CORRECT
        )

    def submit_negative_feedback(
        self,
        relation_type: str,
        subject: str,
        object: str,
        original_confidence: float,
        comment: Optional[str] = None,
    ) -> ConfidenceFeedback:
        """Convenience-Methode für negatives Feedback (👎)."""
        return self.submit_feedback(
            relation_type,
            subject,
            object,
            original_confidence,
            FeedbackType.INCORRECT,
            user_comment=comment,
        )

    # ==================== CONFIDENCE ADJUSTMENT ====================

    def _calculate_adjusted_confidence(self, feedback: ConfidenceFeedback) -> float:
        """
        Berechnet die adjustierte Confidence basierend auf Feedback.

        Algorithmus:
        - CORRECT: Erhöhe Confidence (aber max 1.0)
        - INCORRECT: Reduziere Confidence (aber min 0.0)
        - UNCERTAIN: Leichte Reduktion

        Die Adjustierung wird stärker, wenn mehr Feedback vorhanden ist.

        Args:
            feedback: Das neue Feedback

        Returns:
            Adjustierte Confidence (0.0-1.0)
        """
        original = feedback.original_confidence

        # Basale Adjustierung
        if feedback.feedback_type == FeedbackType.CORRECT:
            # Positives Feedback erhöht Confidence
            # Stärke der Erhöhung hängt davon ab, wie niedrig die ursprüngliche Confidence war
            increase = self.adjustment_rate * (1.0 - original)
            adjusted = min(1.0, original + increase)

        elif feedback.feedback_type == FeedbackType.INCORRECT:
            # Negatives Feedback reduziert Confidence dramatisch
            decrease = self.adjustment_rate * original * 2  # Doppelte Reduktion
            adjusted = max(0.0, original - decrease)

        else:  # UNCERTAIN
            # Unsicherheit reduziert Confidence leicht
            decrease = self.adjustment_rate * 0.5
            adjusted = max(0.0, original - decrease)

        # Historisches Feedback berücksichtigen
        historical_feedback = self._get_historical_feedback(
            feedback.relation_type, feedback.subject, feedback.object
        )

        if len(historical_feedback) >= self.min_feedback_count:
            # Berechne Konsensus aus historischem Feedback
            consensus_adjustment = self._calculate_consensus_adjustment(
                historical_feedback
            )
            adjusted = self._blend_with_consensus(adjusted, consensus_adjustment)

        return adjusted

    def _get_historical_feedback(
        self, relation_type: str, subject: str, object: str
    ) -> List[ConfidenceFeedback]:
        """
        Findet historisches Feedback für dieselbe Relation.

        Args:
            relation_type: Typ der Relation
            subject: Subjekt
            object: Objekt

        Returns:
            Liste von historischen ConfidenceFeedback-Objekten
        """
        return [
            fb
            for fb in self._feedback_history
            if (
                fb.relation_type == relation_type
                and fb.subject == subject
                and fb.object == object
            )
        ]

    def _calculate_consensus_adjustment(
        self, historical_feedback: List[ConfidenceFeedback]
    ) -> float:
        """
        Berechnet Konsensus aus historischem Feedback.

        Args:
            historical_feedback: Liste von Feedback-Objekten

        Returns:
            Konsensus-Confidence (0.0-1.0)
        """
        if not historical_feedback:
            return 0.5  # Neutral

        # Zähle Feedback-Typen
        correct_count = sum(
            1 for fb in historical_feedback if fb.feedback_type == FeedbackType.CORRECT
        )
        _ = sum(
            1
            for fb in historical_feedback
            if fb.feedback_type == FeedbackType.INCORRECT
        )
        total = len(historical_feedback)

        # Konsensus basiert auf Verhältnis correct/incorrect
        consensus = correct_count / total

        logger.debug(
            f"Konsensus berechnet: {correct_count}/{total} correct "
            f"-> {consensus:.2f}"
        )

        return consensus

    def _blend_with_consensus(
        self, current_confidence: float, consensus: float, blend_weight: float = 0.3
    ) -> float:
        """
        Kombiniert aktuelle Confidence mit historischem Konsensus.

        Args:
            current_confidence: Aktuelle Confidence
            consensus: Konsensus aus historischem Feedback
            blend_weight: Gewicht des Konsensus (0.0-1.0)

        Returns:
            Gemischte Confidence
        """
        blended = (1 - blend_weight) * current_confidence + blend_weight * consensus
        return max(0.0, min(1.0, blended))

    # ==================== PERSISTENCE ====================

    def _persist_feedback(self, feedback: ConfidenceFeedback):
        """
        Persistiert Feedback in Neo4j.

        Erstellt eine FeedbackNode und verknüpft sie mit der betroffenen Relation.

        Args:
            feedback: Das zu persistierende Feedback
        """
        if not self.netzwerk or not self.netzwerk.driver:
            logger.warning("Kein Netzwerk-Driver verfügbar, Feedback nicht persistiert")
            return

        try:
            with self.netzwerk.driver.session(database="neo4j") as session:
                session.run(
                    """
                    // Erstelle Feedback-Node
                    CREATE (fb:ConfidenceFeedback {
                        id: $feedback_id,
                        relation_type: $relation_type,
                        subject: $subject,
                        object: $object,
                        original_confidence: $original_confidence,
                        feedback_type: $feedback_type,
                        timestamp: datetime($timestamp),
                        user_comment: $user_comment,
                        adjusted_confidence: $adjusted_confidence
                    })

                    // Verknüpfe mit betroffenen Konzepten
                    WITH fb
                    MATCH (s:Konzept {name: $subject})
                    MATCH (o:Konzept {name: $object})
                    CREATE (s)-[:HAS_FEEDBACK]->(fb)
                    CREATE (fb)-[:AFFECTS_RELATION]->(o)

                    // Update die Relation selbst mit neuer Confidence
                    WITH fb, s, o
                    MATCH (s)-[r]->(o)
                    WHERE type(r) = $relation_type
                    SET r.confidence = $adjusted_confidence,
                        r.feedback_count = COALESCE(r.feedback_count, 0) + 1,
                        r.last_feedback_timestamp = datetime($timestamp)

                    RETURN fb.id AS feedback_id
                    """,
                    feedback_id=feedback.feedback_id,
                    relation_type=feedback.relation_type,
                    subject=feedback.subject.lower(),
                    object=feedback.object.lower(),
                    original_confidence=feedback.original_confidence,
                    feedback_type=feedback.feedback_type.value,
                    timestamp=feedback.timestamp.isoformat(),
                    user_comment=feedback.user_comment,
                    adjusted_confidence=feedback.adjusted_confidence,
                )

            logger.info(f"Feedback persistiert: {feedback.feedback_id}")

        except Exception as e:
            logger.error(f"Fehler beim Persistieren von Feedback: {e}", exc_info=True)

    # ==================== ANALYSIS & REPORTING ====================

    def get_feedback_statistics(self) -> Dict[str, Any]:
        """
        Gibt Statistiken über gesammeltes Feedback zurück.

        Returns:
            Dict mit Statistiken:
            - total_feedback: Gesamtanzahl
            - correct_count: Anzahl positiver Feedbacks
            - incorrect_count: Anzahl negativer Feedbacks
            - uncertain_count: Anzahl unsicherer Feedbacks
            - average_adjustment: Durchschnittliche Confidence-Adjustierung
        """
        total = len(self._feedback_history)
        correct = sum(
            1
            for fb in self._feedback_history
            if fb.feedback_type == FeedbackType.CORRECT
        )
        incorrect = sum(
            1
            for fb in self._feedback_history
            if fb.feedback_type == FeedbackType.INCORRECT
        )
        uncertain = sum(
            1
            for fb in self._feedback_history
            if fb.feedback_type == FeedbackType.UNCERTAIN
        )

        # Berechne durchschnittliche Adjustierung
        adjustments = [
            abs(fb.adjusted_confidence - fb.original_confidence)
            for fb in self._feedback_history
            if fb.adjusted_confidence is not None
        ]
        avg_adjustment = sum(adjustments) / len(adjustments) if adjustments else 0.0

        return {
            "total_feedback": total,
            "correct_count": correct,
            "incorrect_count": incorrect,
            "uncertain_count": uncertain,
            "average_adjustment": avg_adjustment,
            "accuracy_rate": correct / total if total > 0 else 0.0,
        }

    def get_learning_curve(self, window_size: int = 10) -> List[float]:
        """
        Berechnet die Lernkurve basierend auf Feedback-Genauigkeit über Zeit.

        Args:
            window_size: Größe des gleitenden Fensters

        Returns:
            Liste von Genauigkeitsraten über Zeit
        """
        if len(self._feedback_history) < window_size:
            return []

        curve = []
        for i in range(window_size, len(self._feedback_history) + 1):
            window = self._feedback_history[i - window_size : i]
            correct_in_window = sum(
                1 for fb in window if fb.feedback_type == FeedbackType.CORRECT
            )
            accuracy = correct_in_window / window_size
            curve.append(accuracy)

        return curve

    def get_feedback_for_relation(
        self, relation_type: str, subject: str, object: str
    ) -> List[ConfidenceFeedback]:
        """
        Gibt alle Feedbacks für eine spezifische Relation zurück.

        Args:
            relation_type: Typ der Relation
            subject: Subjekt
            object: Objekt

        Returns:
            Liste von ConfidenceFeedback-Objekten
        """
        return self._get_historical_feedback(relation_type, subject, object)


# ==================== GLOBAL INSTANCE ====================

_global_feedback_manager: Optional[ConfidenceFeedbackManager] = None


def get_feedback_manager(netzwerk=None) -> ConfidenceFeedbackManager:
    """
    Gibt die globale ConfidenceFeedbackManager-Instanz zurück.

    Args:
        netzwerk: Optional - KonzeptNetzwerk-Instanz (nur beim ersten Aufruf)

    Returns:
        Globale ConfidenceFeedbackManager-Instanz
    """
    global _global_feedback_manager
    if _global_feedback_manager is None:
        _global_feedback_manager = ConfidenceFeedbackManager(netzwerk=netzwerk)
        logger.info("Globale ConfidenceFeedbackManager-Instanz erstellt")
    return _global_feedback_manager


# ==================== BEISPIEL-USAGE ====================

if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    # Initialisiere Manager
    fm = ConfidenceFeedbackManager()

    # Beispiel 1: Positives Feedback
    print("=== Beispiel 1: Positives Feedback ===")
    fb1 = fm.submit_positive_feedback(
        relation_type="IS_A",
        subject="hund",
        object="säugetier",
        original_confidence=0.75,
    )
    print(
        f"Original: {fb1.original_confidence:.2f} -> Adjusted: {fb1.adjusted_confidence:.2f}"
    )

    # Beispiel 2: Negatives Feedback
    print("\n=== Beispiel 2: Negatives Feedback ===")
    fb2 = fm.submit_negative_feedback(
        relation_type="IS_A",
        subject="wal",
        object="fisch",
        original_confidence=0.85,
        comment="Wal ist ein Säugetier, kein Fisch!",
    )
    print(
        f"Original: {fb2.original_confidence:.2f} -> Adjusted: {fb2.adjusted_confidence:.2f}"
    )

    # Beispiel 3: Mehrfaches Feedback für dieselbe Relation
    print("\n=== Beispiel 3: Historischer Konsensus ===")
    for i in range(5):
        fm.submit_positive_feedback("HAS_PROPERTY", "apfel", "rot", 0.7)
    fb3 = fm.submit_positive_feedback("HAS_PROPERTY", "apfel", "rot", 0.7)
    print(f"Nach 6x positivem Feedback: {fb3.adjusted_confidence:.2f}")

    # Statistiken
    print("\n=== Statistiken ===")
    stats = fm.get_feedback_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
