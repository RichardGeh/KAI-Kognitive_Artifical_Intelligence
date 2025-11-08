"""
component_50_self_evaluation.py

Self-Evaluation Layer - KAI evaluiert eigene Antworten kritisch

Implementiert:
- Interne Konsistenz-Checks (Widersprüche?)
- Confidence-Kalibrierung (passt Confidence zur Beweislage?)
- Completeness-Checks (alle Fragen-Aspekte beantwortet?)
- Proof-Quality-Checks (Beweis-Ketten robust?)
- Uncertainty Detection (Was ist unsicher?)

Teil von Phase 3: Meta-Learning Layer
Unterstützt Transparent AI und Self-Awareness

Author: KAI Development Team
Created: 2025-11-08
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from component_15_logging_config import get_logger

logger = get_logger(__name__)


# ============================================================================
# Data Structures
# ============================================================================


class RecommendationType(Enum):
    """Empfehlung basierend auf Evaluation"""

    SHOW_TO_USER = "show_to_user"  # Antwort ist gut genug
    SHOW_WITH_WARNING = "show_with_warning"  # Zeige mit Unsicherheiten
    REQUEST_CLARIFICATION = "request_clarification"  # Frage zu vage
    RETRY_WITH_DIFFERENT_STRATEGY = "retry_different_strategy"  # Andere Strategy
    INSUFFICIENT_KNOWLEDGE = "insufficient_knowledge"  # Wissenslücke


@dataclass
class CheckResult:
    """Ergebnis eines einzelnen Checks"""

    score: float  # 0.0-1.0
    passed: bool
    issues: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        status = "✓" if self.passed else "✗"
        return f"{status} {self.score:.2f} ({len(self.issues)} issues)"


@dataclass
class EvaluationResult:
    """
    Gesamtergebnis der Self-Evaluation

    Attributes:
        overall_score: Gesamt-Score 0.0-1.0
        checks: Ergebnisse einzelner Checks
        uncertainties: Liste identifizierter Unsicherheiten
        recommendation: Was soll mit der Antwort passieren?
        confidence_adjusted: Falls True, wurde Confidence nach unten korrigiert
        suggested_confidence: Empfohlener Confidence-Wert
    """

    overall_score: float
    checks: Dict[str, CheckResult]
    uncertainties: List[str]
    recommendation: RecommendationType
    confidence_adjusted: bool = False
    suggested_confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"EvaluationResult(score={self.overall_score:.2f}, "
            f"recommendation={self.recommendation.value}, "
            f"uncertainties={len(self.uncertainties)})"
        )

    def get_summary(self) -> str:
        """Erstelle lesbare Zusammenfassung"""
        lines = []
        lines.append(f"Self-Evaluation Score: {self.overall_score:.1%}")
        lines.append(f"Empfehlung: {self.recommendation.value}")

        if self.uncertainties:
            lines.append("\nUnsicherheiten:")
            for u in self.uncertainties:
                lines.append(f"  ⚠ {u}")

        lines.append("\nChecks:")
        for check_name, result in self.checks.items():
            lines.append(f"  {result} {check_name}")

        return "\n".join(lines)


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class SelfEvaluationConfig:
    """Konfiguration für Self-Evaluator"""

    # Thresholds für Checks
    consistency_threshold: float = 0.7
    confidence_calibration_threshold: float = 0.6
    completeness_threshold: float = 0.8
    proof_quality_threshold: float = 0.7

    # Overall score thresholds
    excellent_threshold: float = 0.85
    good_threshold: float = 0.7
    acceptable_threshold: float = 0.6

    # Contradiction detection
    contradiction_keywords: List[str] = field(
        default_factory=lambda: [
            "aber",
            "jedoch",
            "allerdings",
            "trotzdem",
            "dennoch",
            "im Gegensatz",
            "widersprüchlich",
            "nicht",
            "kein",
        ]
    )

    # Question word detection for completeness
    question_words: List[str] = field(
        default_factory=lambda: ["was", "wer", "wie", "warum", "wo", "wann", "welche"]
    )


# ============================================================================
# Self-Evaluator
# ============================================================================


class SelfEvaluator:
    """
    Self-Evaluation Engine

    Evaluiert KAI's eigene Antworten auf:
    1. Consistency: Keine internen Widersprüche
    2. Confidence Calibration: Confidence passt zur Beweislage
    3. Completeness: Alle Fragen-Aspekte beantwortet
    4. Proof Quality: Beweis-Ketten sind robust

    Integration:
    - Wird von kai_response_formatter.py aufgerufen
    - Nutzt ProofTree aus component_17 für Proof-Quality
    - Nutzt MetaLearningEngine für Confidence-Kalibrierung
    """

    def __init__(self, config: Optional[SelfEvaluationConfig] = None):
        """
        Initialize Self-Evaluator

        Args:
            config: Optional SelfEvaluationConfig
        """
        self.config = config or SelfEvaluationConfig()

        logger.info(
            "SelfEvaluator initialized | "
            f"consistency_threshold={self.config.consistency_threshold}, "
            f"completeness_threshold={self.config.completeness_threshold}"
        )

    def evaluate_answer(
        self,
        question: str,
        answer: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> EvaluationResult:
        """
        Hauptmethode: Evaluiere Antwort

        Args:
            question: Die ursprüngliche Frage
            answer: Dict mit Antwort-Daten (muss 'text' und 'confidence' enthalten)
            context: Optional zusätzlicher Kontext

        Returns:
            EvaluationResult mit allen Checks
        """
        try:
            logger.info(f"Evaluating answer for question: '{question[:50]}...'")

            # Extract answer components
            answer_text = answer.get("text", "")
            confidence = answer.get("confidence", 0.5)
            proof_tree = answer.get("proof_tree")
            reasoning_paths = answer.get("reasoning_paths", [])

            # Run all checks
            checks = {}

            # 1. Consistency Check
            checks["consistency"] = self._check_consistency(
                answer_text, reasoning_paths
            )

            # 2. Confidence Calibration
            checks["confidence_calibration"] = self._check_confidence_calibration(
                confidence, proof_tree, reasoning_paths
            )

            # 3. Completeness
            checks["completeness"] = self._check_completeness(question, answer_text)

            # 4. Proof Quality
            checks["proof_quality"] = self._check_proof_quality(
                proof_tree, reasoning_paths
            )

            # Calculate overall score (weighted average)
            weights = {
                "consistency": 0.3,
                "confidence_calibration": 0.2,
                "completeness": 0.3,
                "proof_quality": 0.2,
            }

            overall_score = sum(checks[key].score * weights[key] for key in weights)

            # Collect uncertainties
            uncertainties = self._collect_uncertainties(checks)

            # Determine recommendation
            recommendation = self._determine_recommendation(
                overall_score, checks, confidence
            )

            # Check if confidence adjustment needed
            confidence_adjusted = False
            suggested_confidence = None

            if (
                checks["confidence_calibration"].score
                < self.config.confidence_calibration_threshold
            ):
                confidence_adjusted = True
                suggested_confidence = self._suggest_confidence_adjustment(
                    confidence, checks["confidence_calibration"]
                )

            result = EvaluationResult(
                overall_score=overall_score,
                checks=checks,
                uncertainties=uncertainties,
                recommendation=recommendation,
                confidence_adjusted=confidence_adjusted,
                suggested_confidence=suggested_confidence,
                metadata={
                    "original_confidence": confidence,
                    "question_length": len(question),
                    "answer_length": len(answer_text),
                },
            )

            logger.info(
                f"Evaluation complete | score={overall_score:.2f}, "
                f"recommendation={recommendation.value}"
            )

            return result

        except Exception as e:
            logger.error(f"Error in evaluate_answer: {e}", exc_info=True)
            # Return safe fallback
            return EvaluationResult(
                overall_score=0.5,
                checks={},
                uncertainties=["Evaluation konnte nicht durchgeführt werden"],
                recommendation=RecommendationType.SHOW_WITH_WARNING,
            )

    # ========================================================================
    # Check Methods
    # ========================================================================

    def _check_consistency(
        self, answer_text: str, reasoning_paths: List[Any]
    ) -> CheckResult:
        """
        Prüft interne Konsistenz der Antwort

        Sucht nach:
        - Widersprüchlichen Aussagen
        - Negationen in unmittelbarer Nähe
        - Logische Inkonsistenzen in Reasoning-Paths

        Args:
            answer_text: Antwort-Text
            reasoning_paths: Liste von ReasoningPaths

        Returns:
            CheckResult
        """
        issues = []
        score = 1.0

        # 1. Text-basierte Widerspruchs-Erkennung
        text_lower = answer_text.lower()

        # Suche nach Widerspruchs-Indikatoren
        contradiction_count = 0
        for keyword in self.config.contradiction_keywords:
            if keyword in text_lower:
                contradiction_count += 1

        # Mehr als 2 Widerspruchs-Keywords → verdächtig
        if contradiction_count > 2:
            issues.append(
                f"Viele Widerspruchs-Indikatoren gefunden ({contradiction_count})"
            )
            score -= 0.2

        # 2. Negationen-Check: "ist X" gefolgt von "ist nicht X"
        # Einfacher Regex-basierter Check
        sentences = re.split(r"[.!?]+", answer_text)
        for i, sent in enumerate(sentences[:-1]):
            # Suche nach Mustern wie "X ist Y" und danach "X ist nicht Y"
            if self._contains_negation_pattern(sent, sentences[i + 1]):
                issues.append(f"Möglicher Widerspruch zwischen Sätzen {i} und {i+1}")
                score -= 0.3

        # 3. Reasoning-Paths Konsistenz
        if reasoning_paths:
            # Check ob Paths zu unterschiedlichen Conclusionen führen
            path_inconsistencies = self._check_path_consistency(reasoning_paths)
            if path_inconsistencies:
                issues.extend(path_inconsistencies)
                score -= 0.2 * len(path_inconsistencies)

        # Clamp score
        score = max(0.0, min(1.0, score))

        passed = score >= self.config.consistency_threshold

        return CheckResult(
            score=score,
            passed=passed,
            issues=issues,
            details={"contradiction_keywords_found": contradiction_count},
        )

    def _check_confidence_calibration(
        self, confidence: float, proof_tree: Optional[Any], reasoning_paths: List[Any]
    ) -> CheckResult:
        """
        Prüft ob Confidence zur Beweislage passt

        Kriterien:
        - Hohe Confidence (>0.8) braucht starke Beweise
        - Niedrige Confidence (<0.5) sollte explizit Unsicherheit zeigen
        - Proof Tree Depth und Breadth sollten zu Confidence passen

        Args:
            confidence: Claimed confidence
            proof_tree: Optional ProofTree
            reasoning_paths: Liste von ReasoningPaths

        Returns:
            CheckResult
        """
        issues = []
        score = 1.0

        # 1. Hohe Confidence ohne Beweise?
        if confidence > 0.8:
            if not proof_tree and not reasoning_paths:
                issues.append("Hohe Confidence (>0.8) ohne Beweise")
                score -= 0.4
            elif proof_tree:
                # Check Proof Tree Quality
                # Simplified: Count steps
                try:
                    num_steps = len(getattr(proof_tree, "steps", []))
                    if num_steps < 2:
                        issues.append("Hohe Confidence aber sehr kurzer Beweis")
                        score -= 0.2
                except Exception:
                    pass

        # 2. Mittlere Confidence (0.5-0.8): Sollte moderate Beweise haben
        elif 0.5 <= confidence <= 0.8:
            if not reasoning_paths and not proof_tree:
                issues.append("Mittlere Confidence ohne Reasoning-Paths")
                score -= 0.2

        # 3. Niedrige Confidence (<0.5): OK, aber sollte explizit sein
        elif confidence < 0.5:
            # Das ist eigentlich gut - niedrige Confidence bei Unsicherheit
            # Kein Penalty
            pass

        # 4. Confidence vs. Reasoning-Paths Count
        if reasoning_paths:
            num_paths = len(reasoning_paths)

            # Viele Paths sollten höhere Confidence rechtfertigen
            if num_paths < 2 and confidence > 0.7:
                issues.append(
                    f"Nur {num_paths} Reasoning-Path(s) für Confidence {confidence:.2f}"
                )
                score -= 0.15

        # Clamp
        score = max(0.0, min(1.0, score))
        passed = score >= self.config.confidence_calibration_threshold

        return CheckResult(
            score=score,
            passed=passed,
            issues=issues,
            details={
                "confidence": confidence,
                "num_reasoning_paths": len(reasoning_paths) if reasoning_paths else 0,
                "has_proof_tree": proof_tree is not None,
            },
        )

    def _check_completeness(self, question: str, answer_text: str) -> CheckResult:
        """
        Prüft ob Antwort alle Aspekte der Frage beantwortet

        Heuristiken:
        - Multi-Part Questions (und/oder) → alle Teile beantwortet?
        - Question Words (was/wer/wie) → entsprechende Antwort?
        - Antwort-Länge proportional zu Fragen-Komplexität

        Args:
            question: Die Frage
            answer_text: Die Antwort

        Returns:
            CheckResult
        """
        issues = []
        score = 1.0

        question_lower = question.lower()
        answer_lower = answer_text.lower()

        # 0. Leere Antwort? → Sehr niedrig bewerten
        if len(answer_text.strip()) == 0:
            issues.append("Antwort ist leer")
            score = 0.1  # Sehr niedrig
            return CheckResult(
                score=score, passed=False, issues=issues, details={"answer_length": 0}
            )

        # 1. Multi-Part Question Detection
        # Suche nach "und" in der Frage
        parts = re.split(r"\s+und\s+", question_lower)
        if len(parts) > 1:
            # Multi-part question
            # Check ob alle Teile in Antwort erwähnt sind (grobe Heuristik)
            for i, part in enumerate(parts):
                # Extract keywords from part
                keywords = self._extract_keywords(part)
                if not any(kw in answer_lower for kw in keywords):
                    issues.append(
                        f"Teil {i+1} der Frage möglicherweise nicht beantwortet"
                    )
                    score -= 0.3 / len(parts)

        # 2. Question Word Detection
        question_words_found = []
        for qw in self.config.question_words:
            if qw in question_lower:
                question_words_found.append(qw)

        # Check ob Antwort entsprechende Information liefert
        # Sehr vereinfacht: Mindestlänge-Check
        if question_words_found:
            min_expected_length = 20 * len(question_words_found)
            if len(answer_text) < min_expected_length:
                issues.append(
                    f"Antwort erscheint zu kurz für {len(question_words_found)} Frage-Wörter"
                )
                score -= 0.2

        # 3. Antwort enthält "weiß nicht" oder ähnliches?
        uncertainty_phrases = [
            "weiß nicht",
            "unklar",
            "keine information",
            "kann nicht sagen",
            "unsicher",
        ]

        for phrase in uncertainty_phrases:
            if phrase in answer_lower:
                # Das ist OK - ehrliche Antwort
                # Aber score leicht reduzieren für Completeness
                score -= 0.1
                issues.append(f"Antwort enthält Unsicherheits-Phrase: '{phrase}'")
                break

        # Clamp
        score = max(0.0, min(1.0, score))
        passed = score >= self.config.completeness_threshold

        return CheckResult(
            score=score,
            passed=passed,
            issues=issues,
            details={
                "question_parts": len(parts),
                "question_words_found": question_words_found,
                "answer_length": len(answer_text),
            },
        )

    def _check_proof_quality(
        self, proof_tree: Optional[Any], reasoning_paths: List[Any]
    ) -> CheckResult:
        """
        Prüft Qualität des Beweises / der Reasoning-Kette

        Kriterien:
        - Proof Tree vorhanden und strukturiert?
        - Reasoning Paths mit Confidence-Werten?
        - Keine Sprünge in der Logik?

        Args:
            proof_tree: Optional ProofTree
            reasoning_paths: Liste von ReasoningPaths

        Returns:
            CheckResult
        """
        issues = []
        score = 1.0

        # 1. Proof Tree Check
        if proof_tree:
            try:
                # Check if ProofTree has steps
                steps = getattr(proof_tree, "steps", [])
                if not steps:
                    issues.append("Proof Tree ist leer")
                    score -= 0.3
                else:
                    # Check step confidence
                    low_confidence_steps = [
                        s for s in steps if getattr(s, "confidence", 1.0) < 0.5
                    ]
                    if low_confidence_steps:
                        issues.append(
                            f"{len(low_confidence_steps)} Beweis-Schritte mit niedriger Confidence"
                        )
                        score -= 0.1 * min(3, len(low_confidence_steps))

            except Exception as e:
                logger.warning(f"Could not analyze proof_tree: {e}")
                score -= 0.2
        else:
            # Kein Proof Tree - nicht unbedingt schlecht
            # Aber Reasoning Paths sollten vorhanden sein
            if not reasoning_paths:
                issues.append("Kein Proof Tree und keine Reasoning Paths")
                score -= 0.4

        # 2. Reasoning Paths Check
        if reasoning_paths:
            # Check Confidence-Product
            for i, path in enumerate(reasoning_paths):
                try:
                    conf_product = getattr(path, "confidence_product", None)
                    if conf_product and conf_product < 0.3:
                        issues.append(
                            f"Reasoning Path {i} hat sehr niedrige Confidence ({conf_product:.2f})"
                        )
                        score -= 0.1
                except Exception:
                    pass

        # Clamp
        score = max(0.0, min(1.0, score))
        passed = score >= self.config.proof_quality_threshold

        return CheckResult(
            score=score,
            passed=passed,
            issues=issues,
            details={
                "has_proof_tree": proof_tree is not None,
                "num_reasoning_paths": len(reasoning_paths) if reasoning_paths else 0,
            },
        )

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _contains_negation_pattern(self, sent1: str, sent2: str) -> bool:
        """
        Check ob zwei Sätze widersprüchliche Negationen enthalten

        Sehr vereinfacht: Sucht nach "ist" im ersten und "ist nicht" im zweiten
        """
        sent1_lower = sent1.lower()
        sent2_lower = sent2.lower()

        # Pattern: "ist" in sent1 and "ist nicht" in sent2
        if "ist" in sent1_lower and (
            "ist nicht" in sent2_lower or "nicht ist" in sent2_lower
        ):
            return True

        # Pattern: "kann" in sent1 and "kann nicht" in sent2
        if "kann" in sent1_lower and "kann nicht" in sent2_lower:
            return True

        return False

    def _check_path_consistency(self, reasoning_paths: List[Any]) -> List[str]:
        """
        Prüft ob Reasoning-Paths konsistent sind

        Returns:
            Liste von Inkonsistenzen
        """
        inconsistencies = []

        # Check: Paths zu gleichem Target sollten ähnliche Confidence haben
        target_groups = {}
        for path in reasoning_paths:
            try:
                target = getattr(path, "target", None)
                conf = getattr(path, "confidence_product", 0.5)

                if target:
                    if target not in target_groups:
                        target_groups[target] = []
                    target_groups[target].append(conf)
            except Exception:
                pass

        # Check variance in confidence for same target
        for target, confs in target_groups.items():
            if len(confs) > 1:
                variance = max(confs) - min(confs)
                if variance > 0.5:
                    inconsistencies.append(
                        f"Paths zu '{target}' haben stark unterschiedliche Confidence ({min(confs):.2f} - {max(confs):.2f})"
                    )

        return inconsistencies

    def _extract_keywords(self, text: str) -> List[str]:
        """Extrahiert Keywords aus Text (sehr einfach)"""
        # Remove stopwords and short words
        stopwords = {"der", "die", "das", "und", "oder", "ein", "eine", "ist", "sind"}
        words = text.lower().split()
        keywords = [w for w in words if len(w) > 3 and w not in stopwords]
        return keywords

    def _collect_uncertainties(self, checks: Dict[str, CheckResult]) -> List[str]:
        """Sammelt Unsicherheiten aus allen Checks"""
        uncertainties = []

        # Consistency issues
        if checks["consistency"].score < self.config.consistency_threshold:
            uncertainties.append("⚠ Mögliche Widersprüche in der Begründung")

        # Confidence calibration issues
        if (
            checks["confidence_calibration"].score
            < self.config.confidence_calibration_threshold
        ):
            uncertainties.append("⚠ Confidence könnte zu hoch oder zu niedrig sein")

        # Completeness issues
        if checks["completeness"].score < self.config.completeness_threshold:
            uncertainties.append("⚠ Frage möglicherweise nicht vollständig beantwortet")

        # Proof quality issues
        if checks["proof_quality"].score < self.config.proof_quality_threshold:
            uncertainties.append("⚠ Beweis-Qualität könnte verbessert werden")

        # Add specific issues from checks
        for check_name, result in checks.items():
            for issue in result.issues[:2]:  # Max 2 issues per check
                uncertainties.append(f"  - {issue}")

        return uncertainties

    def _determine_recommendation(
        self, overall_score: float, checks: Dict[str, CheckResult], confidence: float
    ) -> RecommendationType:
        """
        Bestimmt Empfehlung basierend auf Scores

        Args:
            overall_score: Gesamt-Score
            checks: Check-Ergebnisse
            confidence: Original confidence

        Returns:
            RecommendationType
        """
        # Excellent: Zeige ohne Warnung
        if overall_score >= self.config.excellent_threshold:
            return RecommendationType.SHOW_TO_USER

        # Good: Zeige mit leichten Warnungen
        elif overall_score >= self.config.good_threshold:
            return RecommendationType.SHOW_WITH_WARNING

        # Acceptable: Zeige mit deutlichen Warnungen
        elif overall_score >= self.config.acceptable_threshold:
            return RecommendationType.SHOW_WITH_WARNING

        # Poor: Check warum
        else:
            # Wenn Completeness sehr niedrig: Frage zu vage
            if checks["completeness"].score < 0.5:
                return RecommendationType.REQUEST_CLARIFICATION

            # Wenn Consistency sehr niedrig: Andere Strategy versuchen
            elif checks["consistency"].score < 0.4:
                return RecommendationType.RETRY_WITH_DIFFERENT_STRATEGY

            # Default: Insufficient Knowledge
            else:
                return RecommendationType.INSUFFICIENT_KNOWLEDGE

    def _suggest_confidence_adjustment(
        self, original_confidence: float, calibration_result: CheckResult
    ) -> float:
        """
        Schlägt angepassten Confidence-Wert vor

        Args:
            original_confidence: Original Confidence
            calibration_result: Ergebnis des Calibration-Checks

        Returns:
            Angepasster Confidence-Wert
        """
        # Wenn Calibration-Score niedrig: Confidence nach unten
        adjustment_factor = calibration_result.score

        # Mehr aggressive Anpassung bei sehr niedriger Calibration
        if calibration_result.score < 0.4:
            adjustment_factor *= 0.7  # Zusätzliche Reduktion

        suggested = original_confidence * adjustment_factor

        # Clamp
        suggested = max(0.1, min(0.9, suggested))

        logger.info(
            f"Confidence adjustment: {original_confidence:.2f} → {suggested:.2f} "
            f"(calibration_score={calibration_result.score:.2f})"
        )

        return suggested
