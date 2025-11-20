# kai_response_formatter.py
"""
Response Formatting Module für KAI

Verantwortlichkeiten:
- Antwort-Formatierung basierend auf Fragetyp
- Confidence-aware Response Generation (integriert mit ConfidenceManager)
- Pure functions ohne State
- Delegiert Text-Normalisierung an zentrale component_utils_text_normalization
"""
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from component_50_self_evaluation import (
    RecommendationType,
    SelfEvaluator,
)
from component_confidence_manager import ConfidenceLevel, get_confidence_manager
from component_utils_text_normalization import clean_entity as normalize_entity

# Optional: FeedbackHandler für Answer Tracking
try:
    from component_51_feedback_handler import FeedbackHandler

    FEEDBACK_HANDLER_AVAILABLE = True
except ImportError:
    FEEDBACK_HANDLER_AVAILABLE = False
    FeedbackHandler = None

logger = logging.getLogger(__name__)


@dataclass
class KaiResponse:
    """Datenstruktur für KAI-Antworten"""

    text: str
    trace: List[str] = field(default_factory=list)
    answer_id: Optional[str] = None  # Für Feedback-Tracking
    confidence: Optional[float] = None
    strategy: Optional[str] = None
    evaluation: Optional[Any] = None  # EvaluationResult
    proof_tree: Optional[Any] = None  # PHASE 6: ProofTree für Transparenz


class KaiResponseFormatter:
    """
    Formatter für KAI-Antworten basierend auf Fragetyp und Wissensstand.

    Diese Klasse ist zustandslos und enthält nur Formatierungs-Logik.
    Text-Normalisierung wurde in component_utils_text_normalization zentralisiert.

    PHASE: Confidence-Based Learning Integration
    - Verwendet ConfidenceManager für einheitliches Confidence-Feedback
    - Generiert confidence-aware Antworten für alle Reasoning-Typen
    """

    def __init__(self, feedback_handler: Optional[Any] = None):
        """
        Initialisiert den Formatter mit globalem ConfidenceManager und SelfEvaluator.

        Args:
            feedback_handler: Optional FeedbackHandler für Answer-Tracking
        """
        self.confidence_manager = get_confidence_manager()
        self.self_evaluator = SelfEvaluator()
        self.feedback_handler = feedback_handler

        if self.feedback_handler:
            logger.info(
                "KaiResponseFormatter initialisiert mit ConfidenceManager, SelfEvaluator und FeedbackHandler"
            )
        else:
            logger.info(
                "KaiResponseFormatter initialisiert mit ConfidenceManager und SelfEvaluator"
            )

    @staticmethod
    def clean_entity(entity_text: str) -> str:
        """
        Entfernt führende Artikel, bereinigt den Text und normalisiert Plurale zu Singularen.

        REFACTORED: Delegiert an zentrale component_utils_text_normalization.

        Args:
            entity_text: Der zu bereinigende Text

        Returns:
            Bereinigter und normalisierter Text
        """
        return normalize_entity(entity_text)

    def format_confidence_prefix(
        self, confidence: float, reasoning_type: str = "standard"
    ) -> str:
        """
        Generiert einen Confidence-aware Präfix für Antworten.

        NEUE METHODE: Confidence-Based Learning Integration

        Args:
            confidence: Confidence-Wert (0.0-1.0)
            reasoning_type: Art des Reasoning ("standard", "backward_chaining",
                           "hypothesis", "probabilistic", "graph_traversal")

        Returns:
            Formatierter Präfix-String
        """
        level = self.confidence_manager.classify_confidence(confidence)

        # Reasoning-spezifische Präfixe
        reasoning_prefixes = {
            "backward_chaining": "durch komplexe schlussfolgerung",
            "hypothesis": "basierend auf abduktiver schlussfolgerung",
            "probabilistic": "durch probabilistische inferenz",
            "graph_traversal": "über mehrere beziehungen hinweg",
            "standard": "",
        }

        reasoning_prefix = reasoning_prefixes.get(reasoning_type, "")

        # Confidence-Level-basierte Formulierungen
        if level == ConfidenceLevel.HIGH:
            if confidence >= 0.95:
                qualifier = "mit sehr hoher sicherheit"
            else:
                qualifier = "mit hoher sicherheit"

            if reasoning_prefix:
                return f"{reasoning_prefix} habe ich {qualifier} (konfidenz: {confidence:.0%}) herausgefunden:"

            return f"{qualifier} (konfidenz: {confidence:.0%}) weiß ich:"

        elif level == ConfidenceLevel.MEDIUM:
            qualifier = "mit mittlerer sicherheit"

            if reasoning_prefix:
                return f"{reasoning_prefix} vermute ich {qualifier} (konfidenz: {confidence:.0%}):"

            return f"{qualifier} (konfidenz: {confidence:.0%}) vermute ich:"

        elif level == ConfidenceLevel.LOW:
            qualifier = "mit geringer sicherheit"

            if reasoning_prefix:
                return f"{reasoning_prefix} vermute ich vorsichtig {qualifier} (konfidenz: {confidence:.0%}):"

            return f"{qualifier} (konfidenz: {confidence:.0%}) vermute ich vorsichtig:"

        else:  # UNKNOWN
            qualifier = "sehr unsicher"

            if reasoning_prefix:
                return f"{reasoning_prefix} bin ich {qualifier} (konfidenz: {confidence:.0%}), aber möglich:"

            return f"ich bin {qualifier} (konfidenz: {confidence:.0%}), aber möglicherweise:"

    def format_low_confidence_warning(self, confidence: float) -> str:
        """
        Generiert eine Warnung für niedrige Confidence-Werte.

        NEUE METHODE: Confidence-Based Learning Integration

        Args:
            confidence: Confidence-Wert (0.0-1.0)

        Returns:
            Warnungs-String oder leerer String (bei hoher Confidence)
        """
        level = self.confidence_manager.classify_confidence(confidence)

        if level == ConfidenceLevel.UNKNOWN:
            return "\n[WARNING] WARNUNG: Diese Antwort basiert auf sehr unsicheren Informationen. Weitere Evidenz wird dringend benötigt."

        elif level == ConfidenceLevel.LOW:
            return "\n[WARNING] HINWEIS: Diese Antwort ist unsicher. Bitte mit Vorsicht interpretieren."

        elif level == ConfidenceLevel.MEDIUM:
            return "\n[INFO] HINWEIS: Diese Antwort hat mittlere Sicherheit. Bestätigung empfohlen."

        # HIGH: Keine Warnung
        return ""

    def evaluate_and_enrich_response(
        self,
        question: str,
        answer_text: str,
        confidence: float,
        strategy: str = "unknown",
        used_relations: Optional[List[str]] = None,
        used_concepts: Optional[List[str]] = None,
        proof_tree: Optional[Any] = None,
        reasoning_paths: Optional[List[Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        track_for_feedback: bool = True,
    ) -> Dict[str, Any]:
        """
        Führt Self-Evaluation durch und reichert Antwort mit Warnungen an.

        ERWEITERT in Phase 3.4: Answer Tracking für User Feedback Loop

        Args:
            question: Die ursprüngliche Frage
            answer_text: Die generierte Antwort
            confidence: Claimed confidence
            strategy: Verwendete Reasoning-Strategy
            used_relations: Optional Liste von Relation-IDs
            used_concepts: Optional Liste von Konzept-IDs
            proof_tree: Optional ProofTree Objekt
            reasoning_paths: Optional Liste von ReasoningPaths
            context: Optional zusätzlicher Kontext
            track_for_feedback: Ob Antwort für Feedback getrackt werden soll

        Returns:
            Dict mit:
                - 'text': Angereicherte Antwort
                - 'confidence': Ggf. angepasste Confidence
                - 'evaluation': EvaluationResult Objekt
                - 'warnings': Liste von Warnungen
                - 'answer_id': Optional Answer-ID für Feedback (wenn tracked)
        """
        # Erstelle Answer-Dict für Evaluator
        answer_dict = {
            "text": answer_text,
            "confidence": confidence,
            "proof_tree": proof_tree,
            "reasoning_paths": reasoning_paths or [],
        }

        # Führe Evaluation durch
        evaluation = self.self_evaluator.evaluate_answer(question, answer_dict, context)

        # Angereicherte Antwort erstellen
        enriched_text = answer_text
        warnings = []

        # 1. Füge Unsicherheiten als Warnungen hinzu
        if evaluation.uncertainties:
            warnings.append("\n⚠ UNSICHERHEITEN:")
            for uncertainty in evaluation.uncertainties:
                warnings.append(f"  • {uncertainty}")

        # 2. Prüfe Recommendation
        if evaluation.recommendation == RecommendationType.SHOW_WITH_WARNING:
            warnings.append(
                f"\n[INFO] Diese Antwort hat eine Evaluation-Score von {evaluation.overall_score:.0%}. "
                "Bitte mit Vorsicht interpretieren."
            )
        elif evaluation.recommendation == RecommendationType.REQUEST_CLARIFICATION:
            warnings.append(
                "\n[WARNING] Die Antwort erscheint unvollständig. "
                "Bitte stelle die Frage präziser oder mit mehr Kontext."
            )
        elif (
            evaluation.recommendation
            == RecommendationType.RETRY_WITH_DIFFERENT_STRATEGY
        ):
            warnings.append(
                "\n[WARNING] Diese Reasoning-Strategy liefert möglicherweise keine optimale Antwort. "
                "Versuche es mit einer alternativen Formulierung."
            )
        elif evaluation.recommendation == RecommendationType.INSUFFICIENT_KNOWLEDGE:
            warnings.append(
                "\n[WARNING] Mein Wissen zu diesem Thema ist unzureichend. "
                "Bitte lehre mich mehr über dieses Thema."
            )

        # 3. Confidence-Adjustment
        adjusted_confidence = confidence
        if (
            evaluation.confidence_adjusted
            and evaluation.suggested_confidence is not None
        ):
            adjusted_confidence = evaluation.suggested_confidence
            warnings.append(
                f"\n[INFO] Confidence wurde angepasst: {confidence:.0%} → {adjusted_confidence:.0%} "
                f"(Grund: Beweislage unzureichend)"
            )

        # 4. Füge Standard-Confidence-Warning hinzu
        confidence_warning = self.format_low_confidence_warning(adjusted_confidence)
        if confidence_warning:
            warnings.append(confidence_warning)

        # Kombiniere Antwort mit Warnungen
        if warnings:
            enriched_text = answer_text + "\n" + "\n".join(warnings)

        # 5. Answer Tracking für Feedback (optional)
        answer_id = None
        if track_for_feedback and self.feedback_handler:
            try:
                answer_id = self.feedback_handler.track_answer(
                    query=question,
                    answer_text=enriched_text,
                    confidence=adjusted_confidence,
                    strategy=strategy,
                    used_relations=used_relations,
                    used_concepts=used_concepts,
                    proof_tree=proof_tree,
                    reasoning_paths=reasoning_paths,
                    evaluation_score=evaluation.overall_score,
                    metadata={
                        "original_confidence": confidence,
                        "confidence_adjusted": evaluation.confidence_adjusted,
                        "recommendation": evaluation.recommendation.value,
                    },
                )
                logger.debug(f"Answer tracked for feedback | id={answer_id[:8]}")
            except Exception as e:
                logger.warning(f"Could not track answer for feedback: {e}")

        return {
            "text": enriched_text,
            "confidence": adjusted_confidence,
            "evaluation": evaluation,
            "warnings": warnings,
            "answer_id": answer_id,
            "strategy": strategy,
        }

    @staticmethod
    def format_person_answer(
        topic: str,
        facts: Dict[str, List[str]],
        bedeutungen: List[str],
        synonyms: List[str],
    ) -> str:
        """
        Formatiert eine Antwort auf eine Wer-Frage (nach Personen/Akteuren).

        Priorisiert: IS_A (Personen-Typen), Synonyme, Bedeutungen

        Args:
            topic: Das Thema der Frage
            facts: Dictionary mit Relationstypen und Objekten
            bedeutungen: Liste von Bedeutungen/Definitionen
            synonyms: Liste von Synonymen

        Returns:
            Formatierte Antwort als String
        """
        # Build response parts
        response_parts: List[str] = []

        # Wenn es Bedeutungen gibt, zeige sie
        if bedeutungen:
            response_parts.append(f"'{topic}': {bedeutungen[0]}")
        else:
            response_parts.append(f"über '{topic}' weiß ich:")

        # Synonyme/Alternative Namen
        if synonyms:
            response_parts.append(f"auch bekannt als: {', '.join(synonyms)}")

        # IS_A für Personen-Kategorien
        if "IS_A" in facts:
            is_a_str = ", ".join(facts["IS_A"])
            response_parts.append(f"ist ein/eine {is_a_str}")

        # CAPABLE_OF für Fähigkeiten/Rollen
        if "CAPABLE_OF" in facts:
            capable_str = ", ".join(facts["CAPABLE_OF"])
            response_parts.append(f"kann {capable_str}")

        # ASSOCIATED_WITH für Assoziationen (wichtig für Reverse Lookup bei WER-Fragen!)
        # Beispiel: "Wer trinkt Brandy?" -> facts = {"ASSOCIATED_WITH": ["nick"]}
        if "ASSOCIATED_WITH" in facts:
            # Für WER-Fragen ist das Topic das Objekt (z.B. "brandy")
            # und die facts sind die Personen (z.B. ["nick"])
            persons = facts["ASSOCIATED_WITH"]
            if len(persons) == 1:
                return f"{persons[0].capitalize()}"
            elif len(persons) > 1:
                persons_str = ", ".join([p.capitalize() for p in persons[:-1]])
                return f"{persons_str} und {persons[-1].capitalize()}"

        # Wenn keine relevanten Fakten gefunden wurden
        if len(response_parts) == 1 and not bedeutungen:
            return f"Ich habe keine spezifischen Informationen darüber, wer oder was '{topic}' ist."

        return ". ".join(response_parts) + "."

    @staticmethod
    def format_time_answer(
        topic: str, facts: Dict[str, List[str]], bedeutungen: List[str]
    ) -> str:
        """
        Formatiert eine Antwort auf eine Wann-Frage (nach Zeit/Zeitpunkten).

        Args:
            topic: Das Thema der Frage
            facts: Dictionary mit Relationstypen und Objekten
            bedeutungen: Liste von Bedeutungen/Definitionen

        Returns:
            Formatierte Antwort als String
        """
        # Suche nach zeitbezogenen Relationen (falls vorhanden)
        time_relations = ["OCCURRED_AT", "HAPPENS_AT", "TIME", "DATE"]

        for relation in time_relations:
            if relation in facts:
                time_str = ", ".join(facts[relation])
                return f"'{topic}' findet statt: {time_str}."

        # Fallback: Zeige Bedeutung oder generelle Info
        if bedeutungen:
            return f"Ich weiß über '{topic}': {bedeutungen[0]}. Aber ich habe keine spezifischen zeitlichen Informationen."

        return f"Ich habe leider keine zeitlichen Informationen über '{topic}'. Ich kenne '{topic}' noch nicht im zeitlichen Kontext."

    @staticmethod
    def format_process_answer(
        topic: str, facts: Dict[str, List[str]], bedeutungen: List[str]
    ) -> str:
        """
        Formatiert eine Antwort auf eine Wie-Frage (nach Prozessen/Methoden).

        Priorisiert: Bedeutungen (enthalten oft Beschreibungen), CAPABLE_OF, HAS_PROPERTY

        Args:
            topic: Das Thema der Frage
            facts: Dictionary mit Relationstypen und Objekten
            bedeutungen: Liste von Bedeutungen/Definitionen

        Returns:
            Formatierte Antwort als String
        """
        parts: List[str] = []

        # Bedeutungen enthalten oft Prozess-Beschreibungen
        if bedeutungen:
            parts.append(f"'{topic}': {bedeutungen[0]}")

        # CAPABLE_OF für Funktionen/Fähigkeiten
        if "CAPABLE_OF" in facts:
            capable_str = ", ".join(facts["CAPABLE_OF"])
            parts.append(f"es kann {capable_str}")

        # HAS_PROPERTY für Eigenschaften (wie etwas funktioniert)
        if "HAS_PROPERTY" in facts:
            properties = ", ".join(facts["HAS_PROPERTY"])
            parts.append(f"eigenschaften: {properties}")

        # Fallback
        if not parts:
            return f"Ich habe keine spezifischen Informationen darüber, wie '{topic}' funktioniert oder abläuft."

        return ". ".join(parts) + "."

    @staticmethod
    def format_reason_answer(
        topic: str, facts: Dict[str, List[str]], bedeutungen: List[str]
    ) -> str:
        """
        Formatiert eine Antwort auf eine Warum-Frage (nach Gründen/Ursachen).

        Args:
            topic: Das Thema der Frage
            facts: Dictionary mit Relationstypen und Objekten
            bedeutungen: Liste von Bedeutungen/Definitionen

        Returns:
            Formatierte Antwort als String
        """
        # Suche nach kausal-relevanten Relationen
        causal_relations = ["CAUSED_BY", "REASON", "PURPOSE", "BECAUSE_OF"]

        for relation in causal_relations:
            if relation in facts:
                reason_str = ", ".join(facts[relation])
                return f"'{topic}' weil: {reason_str}."

        # Fallback: Zeige Bedeutung
        if bedeutungen:
            return f"Ich weiß über '{topic}': {bedeutungen[0]}. Aber ich habe keine spezifischen Informationen über Gründe oder Ursachen."

        return f"Ich habe leider keine Informationen über die Gründe oder Ursachen von '{topic}'. Ich kenne '{topic}' nicht im kausalen Zusammenhang."

    def generate_with_production_system(
        self,
        topic: str,
        facts: Dict[str, List[str]],
        bedeutungen: List[str],
        synonyms: List[str],
        query_type: str = "normal",
        confidence: Optional[float] = None,
        production_engine: Optional[Any] = None,
        signals: Optional[Any] = None,
    ) -> KaiResponse:
        """
        Generiert Antwort mit Production System (statt Pipeline).

        PHASE 5: Integration mit ResponseFormatter

        Args:
            topic: Das Thema der Frage
            facts: Dictionary mit Relationstypen und Objekten
            bedeutungen: Liste von Bedeutungen/Definitionen
            synonyms: Liste von Synonymen
            query_type: Typ der Query
            confidence: Optionale Konfidenz
            production_engine: Optional ProductionSystemEngine Instanz
            signals: Optional KaiSignals für UI-Updates (PHASE 5)

        Returns:
            KaiResponse mit Production-System generiertem Text
        """
        import time

        from component_54_production_system import (
            GenerationGoal,
            GenerationGoalType,
            ProductionSystemEngine,
            ResponseGenerationState,
            create_all_content_selection_rules,
        )

        start_time = time.time()

        try:
            # 1. Erstelle Production Engine (falls nicht gegeben)
            if production_engine is None:
                production_engine = ProductionSystemEngine(signals=signals)
                # Füge Standard-Regeln hinzu
                production_engine.add_rules(create_all_content_selection_rules())
                # TODO: Füge Lexicalization-Regeln hinzu in späteren Phasen

            # 2. Konvertiere facts/bedeutungen in available_facts Format
            available_facts = []

            # Bedeutungen als Fakten
            for bed in bedeutungen:
                available_facts.append(
                    {
                        "relation_type": "DEFINITION",  # PHASE 6 FIX: relation -> relation_type
                        "subject": topic,
                        "object": bed,
                        "confidence": confidence or 0.9,
                        "source": "direct",
                    }
                )

            # Synonyme als Fakten
            for syn in synonyms:
                available_facts.append(
                    {
                        "relation_type": "SYNONYM",  # PHASE 6 FIX: relation -> relation_type
                        "subject": topic,
                        "object": syn,
                        "confidence": confidence or 0.85,
                        "source": "direct",
                    }
                )

            # Andere Fakten
            for relation_type, objects in facts.items():
                for obj in objects:
                    available_facts.append(
                        {
                            "relation_type": relation_type,  # PHASE 6 FIX: relation -> relation_type
                            "subject": topic,
                            "object": obj,
                            "confidence": confidence or 0.8,
                            "source": "graph",
                        }
                    )

            # 3. Erstelle initialen State
            primary_goal = GenerationGoal(
                goal_type=GenerationGoalType.ANSWER_QUESTION,
                target_entity=topic,
                constraints={"query_type": query_type},
            )

            state = ResponseGenerationState(
                primary_goal=primary_goal,
                available_facts=available_facts,
                constraints={"max_sentences": 5 if query_type == "normal" else 10},
                current_query=f"Generiere Antwort für: {topic}",  # PHASE 6: Query für ProofTree
            )

            # 4. Generiere Antwort
            final_state = production_engine.generate(state)

            # 5. Extrahiere Ergebnis
            generated_text = final_state.get_full_text()

            # PHASE 6: Extrahiere ProofTree
            proof_tree = final_state.proof_tree

            # 6. Erstelle Trace
            trace = [
                "Production System Generierung gestartet",
                f"Ziel: {primary_goal.goal_type.value}",
                f"Verfügbare Fakten: {len(available_facts)}",
                f"Cycles: {final_state.cycle_count}",
                f"Generierte Sätze: {len(final_state.text.completed_sentences)}",
            ]

            # 7. Berechne durchschnittliche Confidence aus verwendeten Fakten
            if available_facts:
                avg_confidence = sum(
                    f.get("confidence", 0.5) for f in available_facts
                ) / len(available_facts)
            else:
                avg_confidence = confidence or 0.5

            # 8. Response Time
            response_time = time.time() - start_time
            trace.append(f"Generierungszeit: {response_time:.3f}s")

            # PHASE 6: Füge ProofTree-Info zum Trace hinzu
            if proof_tree:
                num_steps = len(proof_tree.get_all_steps())
                trace.append(f"ProofTree: {num_steps} Regelanwendungen")

            # PHASE 6: Optional - Emit ProofTree Signal für UI
            if (
                signals is not None
                and hasattr(signals, "proof_tree_update")
                and proof_tree
            ):
                try:
                    signals.proof_tree_update.emit(proof_tree)
                    logger.debug("ProofTree signal emitted for UI update")
                except Exception as e:
                    logger.debug(f"Could not emit proof_tree_update signal: {e}")

            # 9. Erstelle KaiResponse
            response = KaiResponse(
                text=(
                    generated_text
                    if generated_text
                    else f"Keine Antwort generiert für '{topic}'."
                ),
                trace=trace,
                confidence=avg_confidence,
                strategy="production_system",
                proof_tree=proof_tree,  # PHASE 6: ProofTree inkludieren
            )

            logger.info(
                f"Production System generated response | cycles={final_state.cycle_count}, "
                f"sentences={len(final_state.text.completed_sentences)}, time={response_time:.3f}s"
            )

            return response

        except Exception as e:
            logger.error(f"Error in production system generation: {e}", exc_info=True)

            # Fallback auf Pipeline
            fallback_text = self.format_standard_answer(
                topic, facts, bedeutungen, synonyms, query_type, confidence=confidence
            )

            return KaiResponse(
                text=fallback_text,
                trace=[f"Production System failed: {e}", "Fallback to pipeline"],
                confidence=confidence or 0.5,
                strategy="pipeline_fallback",
            )

    def format_standard_answer(
        self,
        topic: str,
        facts: Dict[str, List[str]],
        bedeutungen: List[str],
        synonyms: List[str],
        query_type: str = "normal",
        backward_chaining_used: bool = False,
        is_hypothesis: bool = False,
        confidence: Optional[float] = None,
    ) -> str:
        """
        Formatiert eine Standard-Antwort (für Was-Fragen und generische Fragen).

        UPDATED: Integriert mit ConfidenceManager für einheitliches Feedback

        Args:
            topic: Das Thema der Frage
            facts: Dictionary mit Relationstypen und Objekten
            bedeutungen: Liste von Bedeutungen/Definitionen
            synonyms: Liste von Synonymen
            query_type: Typ der Query ("normal" oder "show_all_knowledge")
            backward_chaining_used: Ob Backward-Chaining verwendet wurde
            is_hypothesis: Ob es sich um eine Hypothese handelt
            confidence: Optionale Konfidenz (0.0-1.0)

        Returns:
            Formatierte Antwort als String
        """
        # PHASE: Confidence-Based Learning - Wähle Einleitung basierend auf Methode und Confidence
        if confidence is not None:
            # Bestimme Reasoning-Typ für Context-aware Präfix
            if is_hypothesis:
                reasoning_type = "hypothesis"
            elif backward_chaining_used:
                reasoning_type = "backward_chaining"
            else:
                reasoning_type = "standard"

            # Generiere Confidence-aware Präfix
            prefix = self.format_confidence_prefix(confidence, reasoning_type)
            parts = [prefix]
        else:
            # Fallback auf alte Logik (für Backwards-Kompatibilität)
            if is_hypothesis:
                parts = ["basierend auf abduktiver schlussfolgerung vermute ich:"]
            elif backward_chaining_used:
                parts = ["durch komplexe schlussfolgerung habe ich herausgefunden:"]
            else:
                parts = [f"das weiß ich über {topic}:"]

        # PRIORITÄT 1: Zeige Bedeutungen/Definitionen zuerst (falls vorhanden)
        if bedeutungen:
            # Wenn es nur eine Bedeutung gibt, zeige sie direkt
            if len(bedeutungen) == 1:
                parts.append(bedeutungen[0])
            else:
                # Wenn es mehrere Bedeutungen gibt, liste sie auf
                for i, bed in enumerate(bedeutungen, 1):
                    parts.append(f"({i}) {bed}")

        # PRIORITÄT 2: Behandle Synonyme separat und zuerst
        if synonyms:
            syn_str = ", ".join(synonyms)
            parts.append(f"es ist auch bekannt als {syn_str}.")

        # TEIL_VON kann auch andere Beziehungen enthalten (nicht nur Synonyme)
        # Diese wurden bereits durch synonyms abgedeckt, also entfernen wir sie
        if "TEIL_VON" in facts:
            # Entferne Synonyme, die bereits erwähnt wurden
            other_parts = [
                p for p in facts["TEIL_VON"] if p not in synonyms and p != topic
            ]
            if other_parts:
                # Das sind echte "Teil von"-Beziehungen
                parts_str = ", ".join(other_parts)
                parts.append(f"es ist teil von {parts_str}.")
            # Entferne TEIL_VON aus facts, da wir es behandelt haben
            facts = {k: v for k, v in facts.items() if k != "TEIL_VON"}

        # Behandle IS_A
        if "IS_A" in facts:
            is_a_str = ", ".join(facts["IS_A"])
            parts.append(f"es ist eine art von {is_a_str}.")
            facts = {k: v for k, v in facts.items() if k != "IS_A"}

        # SPEZIAL: Bei "show_all_knowledge" Query, zeige ALLE Relationen detailliert
        if query_type == "show_all_knowledge":
            # Zeige alle verbleibenden Relationen ausführlich
            for relation, objects in facts.items():
                obj_str = ", ".join(objects)
                relation_str = relation.replace("_", " ").lower()
                parts.append(f"{relation_str}: {obj_str}.")
        else:
            # Normal: Nur zusammengefasst
            for relation, objects in facts.items():
                obj_str = ", ".join(objects)
                relation_str = relation.replace("_", " ").lower()
                parts.append(f"zudem: {relation_str} {obj_str}.")

        response = " ".join(parts)

        # PHASE: Confidence-Based Learning - Füge Warnung hinzu bei niedriger Confidence
        if confidence is not None:
            warning = self.format_low_confidence_warning(confidence)
            response += warning

        return response

    def format_episodic_answer(
        self, episodes: List[Dict], query_type: str, topic: Optional[str] = None
    ) -> str:
        """
        Formatiert eine Antwort für episodische Gedächtnis-Abfragen.

        Args:
            episodes: Liste von Episode-Dictionaries
            query_type: Typ der episodischen Query (when_learned, show_episodes, etc.)
            topic: Optionales Thema

        Returns:
            Formatierte Antwort als String
        """
        if not episodes:
            if topic:
                return f"Ich habe noch nichts über '{topic}' gelernt oder gefolgert."
            else:
                return "Ich habe noch keine Episoden gespeichert."

        # Header basierend auf Query-Typ
        if query_type == "when_learned":
            if topic:
                header = f"Ich habe {len(episodes)} Mal über '{topic}' gelernt:"
            else:
                header = f"Ich habe {len(episodes)} Lern-Episoden gespeichert:"
        elif query_type == "what_learned":
            header = f"Hier ist was ich über '{topic}' gelernt habe ({len(episodes)} Episoden):"
        elif query_type == "learning_history":
            if topic:
                header = f"Lernverlauf für '{topic}' ({len(episodes)} Episoden):"
            else:
                header = f"Gesamter Lernverlauf ({len(episodes)} Episoden):"
        else:  # show_episodes
            if topic:
                header = f"Episoden über '{topic}' ({len(episodes)} gesamt):"
            else:
                header = f"Alle gespeicherten Episoden ({len(episodes)} gesamt):"

        response_parts = [header, ""]

        # Zeige bis zu 5 Episoden in der Text-Antwort
        for i, episode in enumerate(episodes[:5], 1):
            # Zeitstempel formatieren
            timestamp = episode.get("timestamp")
            if timestamp:
                from datetime import datetime

                try:
                    dt = datetime.fromtimestamp(timestamp / 1000.0)
                    time_str = dt.strftime("%Y-%m-%d %H:%M")
                except Exception:
                    time_str = "?"
            else:
                time_str = "?"

            # Episode-Typ
            ep_type = episode.get("type", "?")

            # Inhalt/Query
            if "content" in episode:
                content = episode["content"][:80]
            elif "query" in episode:
                content = episode["query"][:80]
            else:
                content = "?"

            # Gelernte Fakten
            learned_facts = episode.get("learned_facts", [])
            learned_facts = [f for f in learned_facts if f is not None]
            facts_summary = ""
            if learned_facts:
                if len(learned_facts) == 1:
                    fact = learned_facts[0]
                    if isinstance(fact, dict):
                        facts_summary = f" -> [{fact.get('subject')} {fact.get('relation')} {fact.get('object')}]"
                else:
                    facts_summary = f" -> {len(learned_facts)} Fakten"

            response_parts.append(
                f"{i}. [{time_str}] {ep_type}: {content}{facts_summary}"
            )

        # Hinweis auf weitere Episoden
        if len(episodes) > 5:
            response_parts.append("")
            response_parts.append(f"... und {len(episodes) - 5} weitere Episoden.")

        # Hinweis auf UI-Tab
        response_parts.append("")
        response_parts.append(
            "💡 Tipp: Nutze den Tab 'Episodisches Gedächtnis' für Details und Filter!"
        )

        return "\n".join(response_parts)

    def format_spatial_answer(
        self,
        model_type: str,
        spatial_query_type: str,
        entities: List[Dict] = None,
        positions: Dict = None,
        relations: List[Dict] = None,
        plan: List = None,
        plan_length: int = 0,
        reachable: bool = True,
    ) -> str:
        """
        Formatiert eine Antwort für räumliche Reasoning-Abfragen.

        Args:
            model_type: Typ des räumlichen Modells (grid, positions, relations, path_finding)
            spatial_query_type: Typ der räumlichen Query
            entities: Liste von extrahierten Entitäten
            positions: Dictionary mit Positionen
            relations: Liste von räumlichen Relationen
            plan: Optionaler Plan (für Path-Finding)
            plan_length: Länge des Plans
            reachable: Ob das Ziel erreichbar ist

        Returns:
            Formatierte Antwort als String
        """
        entities = entities or []
        positions = positions or {}
        relations = relations or []

        response_parts = []

        # FALL 1: Grid-basierte Queries
        if spatial_query_type == "grid_query":
            if entities:
                grid_entity = entities[0]
                grid_config = grid_entity.get("config", {})
                rows = grid_config.get("rows", 8)
                cols = grid_config.get("cols", 8)

                response_parts.append(
                    f"Ich habe ein {rows}×{cols} Grid-Modell erstellt."
                )
                response_parts.append(f"Das Grid hat {rows * cols} Felder insgesamt.")

        # FALL 2: Positions-Queries
        elif spatial_query_type == "position_query":
            if positions:
                obj_count = len(positions)
                response_parts.append(
                    f"Ich habe {obj_count} {'Objekt' if obj_count == 1 else 'Objekte'} im räumlichen Modell:"
                )
                response_parts.append("")

                for obj_name, pos_data in positions.items():
                    x, y = pos_data.get("x", "?"), pos_data.get("y", "?")
                    response_parts.append(f"  - {obj_name}: Position ({x}, {y})")
            else:
                response_parts.append("Ich konnte keine Positionsinformationen finden.")

        # FALL 3: Relations-Queries
        elif spatial_query_type == "relation_query":
            if relations:
                rel = relations[0]
                subject = rel.get("subject", "unbekannt")
                relation_type = rel.get("relation", "ADJACENT_TO")
                target = rel.get("target", "unbekannt")

                # Übersetze Relation ins Deutsche
                relation_translations = {
                    "NORTH_OF": "nördlich von",
                    "SOUTH_OF": "südlich von",
                    "EAST_OF": "östlich von",
                    "WEST_OF": "westlich von",
                    "ADJACENT_TO": "neben",
                    "BETWEEN": "zwischen",
                    "ABOVE": "über",
                    "BELOW": "unter",
                }
                relation_german = relation_translations.get(
                    relation_type, relation_type.lower()
                )

                # Hier würde man normalerweise das räumliche Modell abfragen
                # Vereinfachte Antwort:
                response_parts.append(
                    f"Ich prüfe ob '{subject}' {relation_german} '{target}' liegt."
                )
                response_parts.append(
                    "Diese Query erfordert ein vollständig initialisiertes räumliches Modell."
                )
            else:
                response_parts.append("Keine räumlichen Relationen zum Prüfen.")

        # FALL 4: Path-Finding-Queries
        elif spatial_query_type == "path_finding":
            if not reachable:
                response_parts.append(
                    "Es gibt keinen Pfad zum Ziel (nicht erreichbar)."
                )
            elif plan and plan_length > 0:
                response_parts.append(
                    f"Ich habe einen Pfad gefunden! Länge: {plan_length} Schritte."
                )
                response_parts.append("")
                response_parts.append("Pfad:")
                for i, action in enumerate(plan[:10], 1):  # Zeige max. 10 Schritte
                    action_name = (
                        action.name if hasattr(action, "name") else str(action)
                    )
                    response_parts.append(f"  {i}. {action_name}")

                if plan_length > 10:
                    response_parts.append(
                        f"  ... und {plan_length - 10} weitere Schritte"
                    )
            else:
                response_parts.append(
                    "Path-Finding wurde durchgeführt, aber kein Plan generiert."
                )

        # FALLBACK: Generische Antwort
        else:
            response_parts.append(
                f"Räumliche Abfrage vom Typ '{spatial_query_type}' wurde verarbeitet."
            )
            response_parts.append(
                f"Modell-Typ: {model_type}, {len(entities)} Entitäten"
            )

        return (
            "\n".join(response_parts)
            if response_parts
            else "Keine räumliche Antwort verfügbar."
        )


# ============================================================================
# Response Generation Router (A/B Testing)
# ============================================================================


class ResponseGenerationRouter:
    """
    Router für A/B Testing zwischen Pipeline und Production System.

    PHASE 5: A/B Testing Infrastructure

    Funktionen:
    - Entscheidet welches System für eine Query verwendet wird
    - Initial: 50/50 Random Split
    - Später: Meta-Learning basierte Auswahl
    - Trackt System-Verwendung für Performance-Analyse
    """

    def __init__(
        self,
        formatter: KaiResponseFormatter,
        production_system_weight: float = 0.5,
        enable_meta_learning: bool = False,
        meta_engine: Optional[Any] = None,
    ):
        """
        Args:
            formatter: KaiResponseFormatter Instanz
            production_system_weight: Wahrscheinlichkeit für Production System (0.0-1.0)
            enable_meta_learning: Nutze Meta-Learning für Routing-Entscheidung
            meta_engine: Optional MetaLearningEngine Instanz
        """
        self.formatter = formatter
        self.production_weight = production_system_weight
        self.enable_meta_learning = enable_meta_learning
        self.meta_engine = meta_engine

        # Tracking
        from collections import defaultdict

        self.system_usage_counts: Dict[str, int] = defaultdict(int)
        self.total_queries: int = 0

        logger.info(
            f"ResponseGenerationRouter initialized | "
            f"production_weight={production_system_weight:.0%}, "
            f"meta_learning={'enabled' if enable_meta_learning else 'disabled'}"
        )

    def route_to_system(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Entscheidet welches System verwendet werden soll.

        Args:
            query: User Query
            context: Optional Context Dict

        Returns:
            "pipeline" oder "production"
        """
        import random

        self.total_queries += 1

        # MODE 1: Meta-Learning basierte Auswahl (wenn aktiviert)
        if self.enable_meta_learning and self.meta_engine:
            # Nutze Meta-Engine um beste "System-Strategy" zu wählen
            system = self._select_via_meta_learning(query, context)

        # MODE 2: Random A/B Split (default)
        else:
            rand = random.random()
            system = "production" if rand < self.production_weight else "pipeline"

        # Track usage
        self.system_usage_counts[system] += 1

        logger.debug(
            f"Routed to {system} system | "
            f"total={self.total_queries}, "
            f"production={self.system_usage_counts['production']}, "
            f"pipeline={self.system_usage_counts['pipeline']}"
        )

        return system

    def _select_via_meta_learning(
        self, query: str, context: Optional[Dict[str, Any]]
    ) -> str:
        """
        Nutze Meta-Learning Engine für System-Auswahl.

        Behandelt "pipeline" und "production" wie zwei Reasoning-Strategien.

        Returns:
            "pipeline" oder "production"
        """
        try:
            # Nutze MetaLearningEngine.select_best_strategy
            # mit "pipeline" und "production" als verfügbare Strategien
            available_systems = ["pipeline", "production"]

            best_system, confidence = self.meta_engine.select_best_strategy(
                query, context, available_strategies=available_systems
            )

            logger.debug(
                f"Meta-learning selected {best_system} with confidence {confidence:.2f}"
            )

            return best_system

        except Exception as e:
            logger.warning(f"Meta-learning routing failed: {e}, falling back to random")
            import random

            return "production" if random.random() < 0.5 else "pipeline"

    def generate_response(
        self,
        topic: str,
        facts: Dict[str, List[str]],
        bedeutungen: List[str],
        synonyms: List[str],
        query: str,
        query_type: str = "normal",
        confidence: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None,
        production_engine: Optional[Any] = None,
    ) -> KaiResponse:
        """
        Generiert Response mit automatischem Routing.

        Args:
            topic, facts, bedeutungen, synonyms: Standard Response-Formatter Args
            query: Original-Query (für Routing)
            query_type: Typ der Query
            confidence: Optional Confidence
            context: Optional Context
            production_engine: Optional ProductionSystemEngine

        Returns:
            KaiResponse (mit strategy="pipeline" oder "production_system")
        """
        import time

        start_time = time.time()

        # 1. Routing-Entscheidung
        system = self.route_to_system(query, context)

        # 2. Generiere mit gewähltem System
        if system == "production":
            response = self.formatter.generate_with_production_system(
                topic=topic,
                facts=facts,
                bedeutungen=bedeutungen,
                synonyms=synonyms,
                query_type=query_type,
                confidence=confidence,
                production_engine=production_engine,
            )
        else:  # pipeline
            response_text = self.formatter.format_standard_answer(
                topic=topic,
                facts=facts,
                bedeutungen=bedeutungen,
                synonyms=synonyms,
                query_type=query_type,
                confidence=confidence,
            )

            response = KaiResponse(
                text=response_text,
                trace=[f"Pipeline System Generierung", f"Query Type: {query_type}"],
                confidence=confidence or 0.8,
                strategy="pipeline",
            )

        # 3. Füge Routing-Info zum Trace hinzu
        response_time = time.time() - start_time
        response.trace.insert(
            0,
            f"Routing: {system} system gewählt (A/B split={self.production_weight:.0%})",
        )
        response.trace.append(f"Total Zeit: {response_time:.3f}s")

        # 4. Record für Meta-Learning (wenn aktiviert)
        if self.enable_meta_learning and self.meta_engine:
            result = {
                "confidence": response.confidence,
                "system": system,
                "response_time": response_time,
            }

            self.meta_engine.record_strategy_usage(
                strategy=system,  # Behandle system wie strategy
                query=query,
                result=result,
                response_time=response_time,
                context=context,
            )

        return response

    def get_statistics(self) -> Dict[str, Any]:
        """
        Gibt Routing-Statistiken zurück.

        Returns:
            Dict mit System-Usage-Counts und Percentages
        """
        stats = {
            "total_queries": self.total_queries,
            "system_usage": dict(self.system_usage_counts),
            "production_weight": self.production_weight,
            "meta_learning_enabled": self.enable_meta_learning,
        }

        if self.total_queries > 0:
            stats["production_percentage"] = (
                self.system_usage_counts["production"] / self.total_queries
            ) * 100
            stats["pipeline_percentage"] = (
                self.system_usage_counts["pipeline"] / self.total_queries
            ) * 100

        return stats

    def set_production_weight(self, weight: float) -> None:
        """
        Ändert Production System Weight.

        Args:
            weight: Neue Wahrscheinlichkeit (0.0-1.0)
        """
        if not 0.0 <= weight <= 1.0:
            raise ValueError("Weight must be between 0.0 and 1.0")

        old_weight = self.production_weight
        self.production_weight = weight

        logger.info(
            f"Production system weight changed: {old_weight:.0%} -> {weight:.0%}"
        )
