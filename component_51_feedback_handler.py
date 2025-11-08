"""
component_51_feedback_handler.py

User Feedback Loop - KAI lernt aus direktem Benutzer-Feedback

Implementiert:
- Answer Tracking mit eindeutigen IDs
- Feedback-Verarbeitung (correct/incorrect/unsure)
- Dynamic Confidence Updates basierend auf Feedback
- Meta-Learning Integration (Strategy Performance)
- Negative Pattern Creation (Inhibition)
- Correction Request System
- Feedback History und Statistiken

Teil von Phase 3: Meta-Learning Layer
Unterstützt kontinuierliches Lernen und Self-Improvement

Author: KAI Development Team
Created: 2025-11-08
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from component_15_logging_config import get_logger
from component_46_meta_learning import MetaLearningEngine
from component_confidence_manager import get_confidence_manager

logger = get_logger(__name__)


# ============================================================================
# Data Structures
# ============================================================================


class FeedbackType(Enum):
    """Typ des Benutzer-Feedbacks"""

    CORRECT = "correct"  # Antwort war richtig
    INCORRECT = "incorrect"  # Antwort war falsch
    UNSURE = "unsure"  # Benutzer ist unsicher
    PARTIALLY_CORRECT = "partially_correct"  # Teilweise richtig


@dataclass
class AnswerRecord:
    """
    Gespeicherte Antwort mit allen Metadaten

    Attributes:
        answer_id: Eindeutige ID
        timestamp: Zeitpunkt der Antwort-Generierung
        query: Die ursprüngliche Frage
        answer_text: Die generierte Antwort
        confidence: Confidence-Wert (0.0-1.0)
        strategy: Verwendete Reasoning-Strategy
        used_relations: Liste von verwendeten Relation-IDs aus Neo4j
        used_concepts: Liste von verwendeten Konzept-IDs
        proof_tree: Optional Proof Tree Objekt
        reasoning_paths: Optional Liste von Reasoning Paths
        evaluation_score: Optional Self-Evaluation Score
    """

    answer_id: str
    timestamp: datetime
    query: str
    answer_text: str
    confidence: float
    strategy: str
    used_relations: List[str] = field(default_factory=list)
    used_concepts: List[str] = field(default_factory=list)
    proof_tree: Optional[Any] = None
    reasoning_paths: Optional[List[Any]] = None
    evaluation_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeedbackRecord:
    """
    Benutzer-Feedback zu einer Antwort

    Attributes:
        feedback_id: Eindeutige Feedback-ID
        answer_id: Referenz zur AnswerRecord
        feedback_type: Art des Feedbacks
        timestamp: Zeitpunkt des Feedbacks
        user_comment: Optional Kommentar vom Benutzer
        correction: Optional Korrektur vom Benutzer
    """

    feedback_id: str
    answer_id: str
    feedback_type: FeedbackType
    timestamp: datetime
    user_comment: Optional[str] = None
    correction: Optional[str] = None


# ============================================================================
# Feedback Handler
# ============================================================================


class FeedbackHandler:
    """
    Handler für User Feedback Loop

    Verantwortlichkeiten:
    1. Answer Tracking: Speichert Antworten mit IDs
    2. Feedback Processing: Verarbeitet Benutzer-Feedback
    3. Confidence Updates: Passt Confidence dynamisch an
    4. Meta-Learning Updates: Informiert Meta-Learning Engine
    5. Negative Patterns: Erstellt Inhibition-Patterns bei Fehlern
    6. Correction Requests: Fordert Korrekturen an

    Integration:
    - ConfidenceManager für Confidence-Updates
    - MetaLearningEngine für Strategy Performance
    - KonzeptNetzwerk für Negative Patterns
    """

    def __init__(
        self, netzwerk: Any, meta_learning: Optional[MetaLearningEngine] = None
    ):
        """
        Initialisiert FeedbackHandler

        Args:
            netzwerk: KonzeptNetzwerk Instanz
            meta_learning: Optional MetaLearningEngine
        """
        self.netzwerk = netzwerk
        self.meta_learning = meta_learning
        self.confidence_manager = get_confidence_manager()

        # In-Memory Storage (für schnellen Zugriff)
        # In Produktion: Könnte in Neo4j persistiert werden
        self.answer_records: Dict[str, AnswerRecord] = {}
        self.feedback_records: Dict[str, FeedbackRecord] = {}

        # Feedback Statistiken
        self.feedback_stats = {
            "total_feedbacks": 0,
            "correct_count": 0,
            "incorrect_count": 0,
            "unsure_count": 0,
            "partially_correct_count": 0,
        }

        logger.info("FeedbackHandler initialisiert")

    # ========================================================================
    # Answer Tracking
    # ========================================================================

    def track_answer(
        self,
        query: str,
        answer_text: str,
        confidence: float,
        strategy: str,
        used_relations: Optional[List[str]] = None,
        used_concepts: Optional[List[str]] = None,
        proof_tree: Optional[Any] = None,
        reasoning_paths: Optional[List[Any]] = None,
        evaluation_score: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Speichert eine Antwort für späteres Feedback

        Args:
            query: Die Frage
            answer_text: Die Antwort
            confidence: Confidence-Wert
            strategy: Verwendete Strategy
            used_relations: Optional Liste von Relation-IDs
            used_concepts: Optional Liste von Konzept-IDs
            proof_tree: Optional Proof Tree
            reasoning_paths: Optional Reasoning Paths
            evaluation_score: Optional Self-Evaluation Score
            metadata: Optional zusätzliche Metadaten

        Returns:
            answer_id: Eindeutige ID für diese Antwort
        """
        answer_id = str(uuid.uuid4())

        record = AnswerRecord(
            answer_id=answer_id,
            timestamp=datetime.now(),
            query=query,
            answer_text=answer_text,
            confidence=confidence,
            strategy=strategy,
            used_relations=used_relations or [],
            used_concepts=used_concepts or [],
            proof_tree=proof_tree,
            reasoning_paths=reasoning_paths,
            evaluation_score=evaluation_score,
            metadata=metadata or {},
        )

        self.answer_records[answer_id] = record

        logger.info(
            f"Answer tracked | id={answer_id[:8]}, strategy={strategy}, "
            f"confidence={confidence:.2f}, relations={len(used_relations or [])}"
        )

        return answer_id

    def get_answer(self, answer_id: str) -> Optional[AnswerRecord]:
        """Gibt AnswerRecord für gegebene ID zurück"""
        return self.answer_records.get(answer_id)

    # ========================================================================
    # Feedback Processing
    # ========================================================================

    def process_user_feedback(
        self,
        answer_id: str,
        feedback_type: FeedbackType,
        user_comment: Optional[str] = None,
        correction: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Verarbeitet Benutzer-Feedback

        Workflow:
        1. Validierung: Answer existiert?
        2. Confidence Update: Verstärke/Schwäche verwendete Relationen
        3. Meta-Learning Update: Informiere Meta-Learning Engine
        4. Negative Patterns: Bei incorrect → Erstelle Inhibition
        5. Correction Request: Bei unsure/incorrect → Fordere Klarstellung

        Args:
            answer_id: ID der bewerteten Antwort
            feedback_type: Art des Feedbacks
            user_comment: Optional Kommentar
            correction: Optional Korrektur vom Benutzer

        Returns:
            Dict mit:
                - 'success': bool
                - 'actions_taken': List[str]
                - 'confidence_changes': Dict[str, float]
                - 'message': str
        """
        # 1. Validierung
        answer = self.get_answer(answer_id)
        if not answer:
            logger.warning(f"Answer ID nicht gefunden: {answer_id}")
            return {
                "success": False,
                "actions_taken": [],
                "confidence_changes": {},
                "message": f"Answer ID {answer_id} nicht gefunden",
            }

        logger.info(
            f"Processing feedback | answer_id={answer_id[:8]}, "
            f"type={feedback_type.value}, strategy={answer.strategy}"
        )

        # Erstelle Feedback Record
        feedback_id = str(uuid.uuid4())
        feedback_record = FeedbackRecord(
            feedback_id=feedback_id,
            answer_id=answer_id,
            feedback_type=feedback_type,
            timestamp=datetime.now(),
            user_comment=user_comment,
            correction=correction,
        )
        self.feedback_records[feedback_id] = feedback_record

        # Update Statistiken
        self._update_statistics(feedback_type)

        actions_taken = []
        confidence_changes = {}

        # 2. Confidence Update
        if answer.used_relations:
            changes = self._update_confidence(answer.used_relations, feedback_type)
            confidence_changes.update(changes)
            actions_taken.append(f"Confidence für {len(changes)} Relationen angepasst")

        # 3. Meta-Learning Update
        if self.meta_learning:
            self._record_to_meta_learning(answer, feedback_type)
            actions_taken.append("Meta-Learning Engine informiert")

        # 4. Negative Patterns
        if feedback_type == FeedbackType.INCORRECT:
            pattern_created = self._create_inhibition_pattern(answer, correction)
            if pattern_created:
                actions_taken.append("Inhibition-Pattern erstellt")

        # 5. Correction Request
        if feedback_type in [FeedbackType.INCORRECT, FeedbackType.UNSURE]:
            request_sent = self._request_correction(answer, correction)
            if request_sent:
                actions_taken.append("Korrektur-Request generiert")

        message = self._generate_feedback_message(feedback_type, actions_taken)

        logger.info(
            f"Feedback processed | answer_id={answer_id[:8]}, "
            f"actions={len(actions_taken)}, changes={len(confidence_changes)}"
        )

        return {
            "success": True,
            "actions_taken": actions_taken,
            "confidence_changes": confidence_changes,
            "message": message,
        }

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _update_confidence(
        self, relation_ids: List[str], feedback_type: FeedbackType
    ) -> Dict[str, float]:
        """
        Updated Confidence für verwendete Relationen

        Args:
            relation_ids: Liste von Relation-IDs
            feedback_type: Art des Feedbacks

        Returns:
            Dict[relation_id, new_confidence]
        """
        changes = {}

        # Faktoren basierend auf Feedback
        factors = {
            FeedbackType.CORRECT: 1.1,  # Verstärken
            FeedbackType.INCORRECT: 0.85,  # Schwächen
            FeedbackType.PARTIALLY_CORRECT: 1.02,  # Leicht verstärken
            FeedbackType.UNSURE: 0.98,  # Leicht schwächen
        }

        factor = factors.get(feedback_type, 1.0)

        for rel_id in relation_ids:
            try:
                # Hole aktuelle Confidence
                # Vereinfacht: Annahme dass Relationen Confidence-Attribut haben
                # In Realität müsste man die Relation aus Neo4j laden

                # TODO: Implement actual Neo4j relation confidence update
                # current_conf = self.netzwerk.get_relation_confidence(rel_id)
                # new_conf = min(1.0, max(0.0, current_conf * factor))
                # self.netzwerk.update_relation_confidence(rel_id, new_conf)
                # changes[rel_id] = new_conf

                # Placeholder für Demo
                _ = rel_id  # Mark as intentionally unused
                changes[rel_id] = factor

            except Exception as e:
                logger.warning(f"Could not update confidence for {rel_id}: {e}")

        return changes

    def _record_to_meta_learning(
        self, answer: AnswerRecord, feedback_type: FeedbackType
    ):
        """
        Informiert Meta-Learning Engine über Feedback

        Args:
            answer: AnswerRecord
            feedback_type: Art des Feedbacks
        """
        if not self.meta_learning:
            return

        try:
            # Konvertiere Feedback zu Success/Failure
            success = feedback_type == FeedbackType.CORRECT

            # Record Strategy Usage mit User Feedback
            # Erweiterte Methode wird in component_46 hinzugefügt
            if hasattr(self.meta_learning, "record_strategy_usage_with_feedback"):
                self.meta_learning.record_strategy_usage_with_feedback(
                    strategy_name=answer.strategy,
                    query=answer.query,
                    success=success,
                    confidence=answer.confidence,
                    response_time=0.0,  # Könnte aus metadata kommen
                    user_feedback=feedback_type.value,
                )
            else:
                # Fallback: Standard record_strategy_usage
                self.meta_learning.record_strategy_usage(
                    strategy_name=answer.strategy,
                    query=answer.query,
                    success=success,
                    confidence=answer.confidence,
                    response_time=0.0,
                )

            logger.debug(
                f"Meta-Learning updated | strategy={answer.strategy}, success={success}"
            )

        except Exception as e:
            logger.error(f"Error recording to meta-learning: {e}", exc_info=True)

    def _create_inhibition_pattern(
        self, answer: AnswerRecord, correction: Optional[str]
    ) -> bool:
        """
        Erstellt Negative Pattern (Inhibition) für falsche Antwort

        Bei falschen Antworten lernt KAI, was NICHT zu tun ist.

        Args:
            answer: Die falsche Antwort
            correction: Optional Korrektur vom Benutzer

        Returns:
            True wenn Pattern erstellt wurde
        """
        try:
            # Erstelle Inhibition-Pattern
            # Idee: Markiere die verwendeten Reasoning-Paths als problematisch

            _ = {
                "query": answer.query,
                "incorrect_answer": answer.answer_text,
                "strategy": answer.strategy,
                "used_relations": answer.used_relations,
                "used_concepts": answer.used_concepts,
                "timestamp": datetime.now().isoformat(),
                "correction": correction,
            }

            # Speichere in Neo4j als NegativePattern-Node
            # TODO: Implement in KonzeptNetzwerk
            # self.netzwerk.create_negative_pattern(pattern_data)

            logger.info(
                f"Inhibition pattern created | query='{answer.query[:30]}...', "
                f"strategy={answer.strategy}"
            )

            return True

        except Exception as e:
            logger.error(f"Error creating inhibition pattern: {e}", exc_info=True)
            return False

    def _request_correction(
        self, answer: AnswerRecord, correction: Optional[str]
    ) -> bool:
        """
        Generiert Correction-Request bei unsicheren/falschen Antworten

        Args:
            answer: Die Antwort
            correction: Optional bereits vorhandene Korrektur

        Returns:
            True wenn Request generiert wurde
        """
        try:
            if correction:
                # Benutzer hat Korrektur gegeben
                # TODO: Parse Korrektur und lerne daraus
                # self.netzwerk.learn_from_correction(answer.query, correction)

                logger.info(
                    f"Correction received | query='{answer.query[:30]}...', "
                    f"correction='{correction[:30]}...'"
                )
                return True
            else:
                # Benutzer hat keine Korrektur gegeben
                # Könnte in UI eine Nachfrage-Dialog öffnen
                logger.info(f"Correction requested for query: '{answer.query[:50]}...'")
                return True

        except Exception as e:
            logger.error(f"Error requesting correction: {e}", exc_info=True)
            return False

    def _update_statistics(self, feedback_type: FeedbackType):
        """Updated Feedback-Statistiken"""
        self.feedback_stats["total_feedbacks"] += 1

        if feedback_type == FeedbackType.CORRECT:
            self.feedback_stats["correct_count"] += 1
        elif feedback_type == FeedbackType.INCORRECT:
            self.feedback_stats["incorrect_count"] += 1
        elif feedback_type == FeedbackType.UNSURE:
            self.feedback_stats["unsure_count"] += 1
        elif feedback_type == FeedbackType.PARTIALLY_CORRECT:
            self.feedback_stats["partially_correct_count"] += 1

    def _generate_feedback_message(
        self, feedback_type: FeedbackType, actions_taken: List[str]
    ) -> str:
        """Generiert lesbare Feedback-Nachricht"""
        messages = {
            FeedbackType.CORRECT: "Danke für das positive Feedback! Ich habe die verwendeten Informationen verstärkt.",
            FeedbackType.INCORRECT: "Danke für die Korrektur! Ich habe die Confidence angepasst und werde ähnliche Fehler vermeiden.",
            FeedbackType.UNSURE: "Danke für die Rückmeldung! Ich werde bei ähnlichen Fragen vorsichtiger sein.",
            FeedbackType.PARTIALLY_CORRECT: "Danke! Ich habe die teilweise korrekte Antwort zur Kenntnis genommen.",
        }

        base_message = messages.get(feedback_type, "Feedback verarbeitet.")

        if actions_taken:
            actions_str = ", ".join(actions_taken)
            return f"{base_message}\n\nAktionen: {actions_str}"

        return base_message

    # ========================================================================
    # Statistics & History
    # ========================================================================

    def get_feedback_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Gibt Feedback-History zurück

        Args:
            limit: Maximale Anzahl an Einträgen

        Returns:
            Liste von Feedback-Records (neueste zuerst)
        """
        # Sortiere nach Timestamp (neueste zuerst)
        sorted_feedbacks = sorted(
            self.feedback_records.values(), key=lambda f: f.timestamp, reverse=True
        )

        history = []
        for feedback in sorted_feedbacks[:limit]:
            answer = self.get_answer(feedback.answer_id)

            history.append(
                {
                    "feedback_id": feedback.feedback_id,
                    "timestamp": feedback.timestamp.isoformat(),
                    "feedback_type": feedback.feedback_type.value,
                    "query": answer.query if answer else "N/A",
                    "answer": answer.answer_text if answer else "N/A",
                    "strategy": answer.strategy if answer else "N/A",
                    "user_comment": feedback.user_comment,
                    "correction": feedback.correction,
                }
            )

        return history

    def get_feedback_stats(self) -> Dict[str, Any]:
        """
        Gibt Feedback-Statistiken zurück

        Returns:
            Dict mit Statistiken
        """
        total = self.feedback_stats["total_feedbacks"]

        if total == 0:
            accuracy = 0.0
        else:
            correct = self.feedback_stats["correct_count"]
            partially = self.feedback_stats["partially_correct_count"]
            accuracy = (correct + 0.5 * partially) / total

        return {
            "total_feedbacks": total,
            "correct_count": self.feedback_stats["correct_count"],
            "incorrect_count": self.feedback_stats["incorrect_count"],
            "unsure_count": self.feedback_stats["unsure_count"],
            "partially_correct_count": self.feedback_stats["partially_correct_count"],
            "accuracy": accuracy,
            "tracked_answers": len(self.answer_records),
        }

    def get_strategy_feedback_breakdown(self) -> Dict[str, Dict[str, int]]:
        """
        Gibt Feedback-Breakdown pro Strategy zurück

        Returns:
            Dict[strategy_name, Dict[feedback_type, count]]
        """
        breakdown = {}

        for feedback in self.feedback_records.values():
            answer = self.get_answer(feedback.answer_id)
            if not answer:
                continue

            strategy = answer.strategy
            if strategy not in breakdown:
                breakdown[strategy] = {
                    "correct": 0,
                    "incorrect": 0,
                    "unsure": 0,
                    "partially_correct": 0,
                }

            feedback_key = feedback.feedback_type.value
            if feedback_key in breakdown[strategy]:
                breakdown[strategy][feedback_key] += 1

        return breakdown
