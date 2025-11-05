# kai_worker.py
"""
KAI Worker - Schlanker Orchestrator für Query-Verarbeitung

Verantwortlichkeiten (REFACTORED):
- Initialisierung aller Subsysteme via Dependency Injection
- Orchestrierung des Query-Processing-Flows
- Plan-Ausführung und Signal-Emission
- KEINE Business-Logik mehr (delegiert an Handler)
"""
import logging
from typing import Dict

from PySide6.QtCore import QObject, Signal

from component_1_netzwerk import KonzeptNetzwerk
from component_4_goal_planner import GoalPlanner
from component_5_linguistik_strukturen import (
    ContextAction,
    GoalStatus,
    KaiContext,
    MainGoal,
    MeaningPoint,
    MeaningPointCategory,
)
from component_6_linguistik_engine import LinguisticPreprocessor
from component_7_meaning_extractor import MeaningPointExtractor
from component_8_prototype_matcher import PrototypingEngine
from component_9_logik_engine import Engine
from component_11_embedding_service import EmbeddingService
from component_12_graph_traversal import GraphTraversal
from component_13_working_memory import (
    WorkingMemory,
    ContextType,
    format_reasoning_trace_for_ui,
)

# Import new modular handlers
from kai_response_formatter import KaiResponse, KaiResponseFormatter
from kai_context_manager import KaiContextManager
from kai_inference_handler import KaiInferenceHandler
from kai_ingestion_handler import KaiIngestionHandler
from kai_sub_goal_executor import SubGoalExecutor

# Import Pattern Recognition
from component_24_pattern_orchestrator import PatternOrchestrator
from kai_config import get_config

# Import Command Suggestions
from component_26_command_suggestions import get_command_suggester

# Import exception utilities for user-friendly error messages
from kai_exceptions import KAIException, get_user_friendly_message

# Import Input Orchestrator for multi-segment processing
from component_41_input_orchestrator import InputOrchestrator

# Import Unified Proof Explanation System
try:
    PROOF_SYSTEM_AVAILABLE = True
except ImportError:
    PROOF_SYSTEM_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Unified Proof Explanation System nicht verfügbar")

logger = logging.getLogger(__name__)


class KaiSignals(QObject):
    """
    PySide6 Signals für asynchrone UI-Kommunikation.

    Diese bleiben zentral im Worker, da sie UI-Koordination betreffen.
    """

    clear_goals = Signal()
    set_main_goal = Signal(str)
    add_sub_goal = Signal(str, str)
    update_sub_goal_status = Signal(str, str)
    finished = Signal(KaiResponse)
    needs_saving = Signal()
    inner_picture_update = Signal(str)  # Reasoning-Trace
    context_update = Signal(str)  # Context-Anzeige in UI
    proof_tree_update = Signal(object)  # ProofTree-Visualisierung
    episodic_data_update = Signal(
        list
    )  # Episodische-Gedächtnis-Daten (Liste von Episoden)
    file_progress_update = Signal(
        int, int, str
    )  # (current, total, message) für Datei-Ingestion
    preview_confirmation_needed = Signal(
        str, str, int
    )  # (preview, file_name, char_count) - blockiert Worker
    preview_confirmation_response = Signal(bool)  # (confirmed) - User-Antwort


class KaiWorker(QObject):
    """
    Schlanker Orchestrator für KAI Query-Processing.

    REFACTORED: Alle Business-Logik wurde in spezialisierte Handler extrahiert.
    Diese Klasse koordiniert nur noch den Gesamt-Flow.
    """

    def __init__(self, netzwerk: KonzeptNetzwerk, embedding_service: EmbeddingService):
        """
        Initialisiert den Worker und alle Subsysteme via Dependency Injection.

        Args:
            netzwerk: KonzeptNetzwerk für Datenspeicherung
            embedding_service: EmbeddingService für Vektor-Embeddings
        """
        super().__init__()
        self.netzwerk = netzwerk
        self.embedding_service = embedding_service
        self.signals = KaiSignals()
        self.context = KaiContext()
        self.working_memory = WorkingMemory(max_stack_depth=10)
        self.initialization_error_message = (
            None  # Für nutzerfreundliche Fehlermeldungen
        )

        # PHASE 8: Preview Confirmation Mechanism
        import threading

        self.preview_confirmation_event = threading.Event()
        self.preview_confirmation_result = False

        # Connect preview confirmation response signal to handler
        self.signals.preview_confirmation_response.connect(
            self._on_preview_confirmation_response
        )

        try:
            # Initialisiere Core-Komponenten
            self.preprocessor = LinguisticPreprocessor()
            self.planner = GoalPlanner()
            self.engine = Engine(self.netzwerk)
            self.engine.load_rules_from_graph(self.netzwerk)
            self.prototyping_engine = PrototypingEngine(
                self.netzwerk, self.embedding_service
            )
            self.graph_traversal = GraphTraversal(self.netzwerk)

            # MeaningPointExtractor mit Prototyping für vektor-basierte Erkennung
            self.extractor = MeaningPointExtractor(
                self.embedding_service, self.preprocessor, self.prototyping_engine
            )

            # REFACTORED: Initialisiere neue Handler via Dependency Injection
            self.response_formatter = KaiResponseFormatter()

            # Pattern Recognition Orchestrator (WICHTIG: VOR ContextManager initialisieren!)
            config = get_config()
            if config.get("pattern_recognition_enabled", True):
                self.pattern_orchestrator = PatternOrchestrator(self.netzwerk)
                logger.info("Pattern Recognition Orchestrator aktiviert")
            else:
                self.pattern_orchestrator = None
                logger.info("Pattern Recognition deaktiviert (Config)")

            self.context_manager = KaiContextManager(
                context=self.context,
                signals=self.signals,
                preprocessor=self.preprocessor,
                extractor=self.extractor,
                planner=self.planner,
                netzwerk=self.netzwerk,
                ingest_text_callback=self._ingest_text_callback,
                execute_plan_callback=self.execute_plan,
                pattern_orchestrator=self.pattern_orchestrator,
            )

            self.inference_handler = KaiInferenceHandler(
                netzwerk=self.netzwerk,
                engine=self.engine,
                graph_traversal=self.graph_traversal,
                working_memory=self.working_memory,
                signals=self.signals,
            )

            self.ingestion_handler = KaiIngestionHandler(
                netzwerk=self.netzwerk,
                preprocessor=self.preprocessor,
                prototyping_engine=self.prototyping_engine,
                embedding_service=self.embedding_service,
            )

            self.sub_goal_executor = SubGoalExecutor(worker=self)

            # Command Suggester für Tippfehler-Erkennung
            self.command_suggester = get_command_suggester()
            logger.info("Command Suggester aktiviert")

            # Input Orchestrator für Multi-Segment-Verarbeitung (Logik-Rätsel)
            self.input_orchestrator = InputOrchestrator(preprocessor=self.preprocessor)
            logger.info("Input Orchestrator aktiviert (für komplexe Eingaben)")

            self.is_initialized_successfully = True
            logger.info("KAI Worker & alle Subsysteme erfolgreich initialisiert.")

            if not self.netzwerk.get_all_extraction_rules():
                logger.critical("!!! KEINE EXTRAKTIONSREGELN GEFUNDEN !!!")

        except Exception as e:
            self.is_initialized_successfully = False

            # Generiere nutzerfreundliche Fehlermeldung
            if isinstance(e, KAIException):
                user_friendly_msg = get_user_friendly_message(e, include_details=False)
                logger.critical(user_friendly_msg)
                logger.debug(f"Technische Details: {e}", exc_info=True)
                self.initialization_error_message = user_friendly_msg
            else:
                # Generischer Fehler
                logger.critical(
                    f"FATALER FEHLER bei der Initialisierung: {e}", exc_info=True
                )
                self.initialization_error_message = "[ERROR] Ein unerwarteter Fehler ist bei der Initialisierung aufgetreten. Bitte überprüfe die Logs."

            # WICHTIG: Exception wird nicht weitergegeben
            # stattdessen: is_initialized_successfully = False
            # -> Graceful Degradation: Worker funktioniert nicht, UI kann aber error anzeigen

    def _ingest_text_callback(self, text: str) -> Dict[str, int]:
        """
        Callback für Context Manager: Delegiert an Ingestion Handler.

        Args:
            text: Der zu ingestierende Text

        Returns:
            Dictionary mit Statistiken (facts_created, learned_patterns, fallback_patterns)
        """
        return self.ingestion_handler.ingest_text(text)

    def process_query(self, query: str):
        """
        Haupteinstiegspunkt für Benutzereingaben.

        Orchestriert den gesamten Query-Processing-Flow:
        1. Prüfe auf aktiven Kontext (Multi-Turn-Dialog)
        2. Extrahiere Intent via MeaningPointExtractor
        3. Erstelle Plan via GoalPlanner
        4. Führe Plan aus und emittiere Ergebnis

        Args:
            query: Die Benutzereingabe
        """
        try:
            if not self.is_initialized_successfully:
                error_msg = (
                    self.initialization_error_message
                    or "[ERROR] KAI konnte nicht korrekt initialisiert werden."
                )
                self.signals.finished.emit(KaiResponse(text=error_msg))
                return

            # Prüfe auf aktiven Kontext (Multi-Turn-Dialog)
            if self.context.is_active():
                self.context_manager.handle_contextual_input(query)
                return

            # COMMAND SUGGESTIONS: Tippfehler-Erkennung für Befehle
            command_suggestion = self.command_suggester.suggest_command(query)
            if command_suggestion and command_suggestion["confidence"] >= 0.7:
                # Hohe Confidence: Zeige Vorschlag
                suggestion_text = (
                    f"💡 Meintest du '{command_suggestion['suggestion']}'?\n\n"
                    f"Deine Eingabe: {command_suggestion['original']}\n"
                    f"Vorschlag: {command_suggestion['full_suggestion']}\n\n"
                    f"Beschreibung: {command_suggestion['description']}\n"
                    f"Beispiel: {command_suggestion['example']}\n\n"
                    f"Antworte mit 'Ja' um den Vorschlag zu übernehmen, oder gib deinen Befehl korrigiert ein."
                )

                self.signals.finished.emit(KaiResponse(text=suggestion_text))

                # Setze Kontext für Antwort
                self.context.set_action(ContextAction.ERWARTE_BEFEHL_BESTAETIGUNG)
                self.context.set_data("command_suggestion", command_suggestion)
                self.context.set_data("original_query", query)
                return

            # PATTERN RECOGNITION: Tippfehler-Korrektur & Vorhersagen
            # WICHTIG: Muss auf ORIGINAL query laufen damit "Lerne:" erkannt wird!
            if self.pattern_orchestrator:
                pattern_result = self.pattern_orchestrator.process_input(query)

                # Bei Tippfehler-Rückfrage
                if pattern_result.get("needs_user_clarification"):
                    clarification_response = self._create_typo_clarification(
                        pattern_result
                    )
                    self.signals.finished.emit(clarification_response)
                    # Setze Kontext für Antwort
                    self.context.set_action(ContextAction.ERWARTE_TYPO_KLARSTELLUNG)
                    self.context.set_data("pattern_result", pattern_result)
                    self.context.set_data("original_query", query)
                    return

                # Nutze korrigierten Text
                query = pattern_result["corrected_text"]

                # Log Auto-Korrekturen
                if pattern_result.get("typo_corrections"):
                    for correction in pattern_result["typo_corrections"]:
                        if correction.get("decision") == "auto_corrected":
                            logger.info(
                                f"Auto-Korrektur: '{correction['original']}' -> '{correction['correction']}' (conf={correction['confidence']:.2f})"
                            )

            # WICHTIG: Entferne UI-Präfixe für weitere Verarbeitung
            # Pattern Recognition hat bereits auf Original-Text geprüft und ggf. übersprungen
            cleaned_query = self._remove_ui_prefixes(query)

            # ===================================================================
            # PHASE: INPUT ORCHESTRATION (Neu für Logik-Rätsel)
            # ===================================================================
            # Prüfe ob Eingabe orchestriert werden sollte (mehrere Segmente)
            orchestration_result = self.input_orchestrator.orchestrate_input(
                cleaned_query
            )

            if orchestration_result:
                # Komplexe Eingabe erkannt (Erklärungen + Fragen)
                logger.info("Orchestrierte Verarbeitung aktiviert")

                # Hole orchestrierten Plan
                orchestrated_plan = orchestration_result["plan"]
                segments = orchestration_result["segments"]

                # Bei jeder neuen Frage wird ein Snapshot gespeichert
                self.context.clear()

                # Sende Context-Update an UI
                self.context_manager.emit_context_update()

                # WICHTIG: Erstelle einen "dummy" Intent für execute_plan
                # Der Intent wird in den Sub-Goals nicht verwendet, da orchestrierte
                # Sub-Goals ihre eigenen Segment-Texte in metadata haben
                dummy_intent = MeaningPoint(
                    id=f"orchestrated-{len(segments)}",
                    category=MeaningPointCategory.DEFINITION,
                    cue="input_orchestrator",
                    text_span=cleaned_query,
                    confidence=0.9,
                    arguments={"orchestrated": True, "segment_count": len(segments)},
                )

                # Führe orchestrierten Plan aus
                # WICHTIG: Lernen passiert ZUERST, dann Fragen
                final_response = self.execute_plan(orchestrated_plan, dummy_intent)
                self.signals.finished.emit(final_response)
                return

            # ===================================================================
            # NORMALE VERARBEITUNG (wie bisher)
            # ===================================================================
            # Normale Verarbeitung ohne Kontext
            doc = self.preprocessor.process(cleaned_query)
            meaning_points = self.extractor.extract(doc)

            # Bei jeder neuen Frage wird ein Snapshot gespeichert
            self.context.clear()

            # Extrahiere Entitäten für Session-Tracking
            self.context_manager.extract_and_track_entities(doc)

            # Sende Context-Update an UI
            self.context_manager.emit_context_update()

            if not meaning_points:
                self.signals.finished.emit(
                    KaiResponse(text="Ich habe die Absicht nicht verstanden.")
                )
                return

            primary_intent = meaning_points[0]
            plan = self.planner.create_plan(primary_intent)

            if not plan:
                self.signals.finished.emit(
                    KaiResponse(
                        text="Ich weiß, was du meinst, habe aber noch keinen Plan dafür."
                    )
                )
                return

            final_response = self.execute_plan(plan, primary_intent)
            self.signals.finished.emit(final_response)

        except KAIException as e:
            # Nutzerfreundliche Fehlermeldung für KAI-spezifische Exceptions
            user_friendly_msg = get_user_friendly_message(e, include_details=False)
            logger.error(
                f"KAI Exception während Query-Verarbeitung: {user_friendly_msg}"
            )
            logger.debug(f"Technische Details: {e}", exc_info=True)
            self.signals.finished.emit(KaiResponse(text=user_friendly_msg))

        except Exception as e:
            # Generische Exception - Log ausführlich, zeige nutzerfreundliche Nachricht
            logger.error(f"Unerwartete Exception in process_query: {e}", exc_info=True)
            self.signals.finished.emit(
                KaiResponse(
                    text="[ERROR] Ein unerwarteter Fehler ist aufgetreten. Bitte versuche es erneut."
                )
            )

    def execute_plan(self, plan: MainGoal, intent: MeaningPoint) -> KaiResponse:
        """
        Führt einen Plan aus, indem alle Sub-Goals sequentiell abgearbeitet werden.

        Orchestriert:
        1. Plan-Setup (Signals emittieren, Working Memory initialisieren)
        2. Sub-Goal-Ausführung via SubGoalExecutor (Strategy Pattern)
        3. Reasoning-Trace-Updates an UI
        4. Erfolgs-/Fehlerbehandlung

        Args:
            plan: Der auszuführende MainGoal
            intent: Der ursprüngliche MeaningPoint

        Returns:
            KaiResponse mit finaler Antwort
        """
        try:
            logger.info(f"Führe Plan aus: {plan.description}")
            plan.status = GoalStatus.IN_PROGRESS
            self.signals.clear_goals.emit()
            self.signals.set_main_goal.emit(plan.description)
            knowledge_context = {"intent": intent}
            final_response_text = "Plan ausgeführt, aber keine Antwort formuliert."

            # Erstelle Working Memory Kontext für diesen Plan
            context_type_mapping = {
                MeaningPointCategory.QUESTION: ContextType.QUESTION,
                MeaningPointCategory.COMMAND: ContextType.PATTERN_LEARNING,
                MeaningPointCategory.DEFINITION: ContextType.DEFINITION,
            }
            context_type = context_type_mapping.get(
                intent.category, ContextType.QUESTION
            )

            # Extrahiere relevante Entitäten aus Intent
            entities = []
            if "topic" in intent.arguments:
                entities.append(intent.arguments["topic"])
            if "subject" in intent.arguments:
                entities.append(intent.arguments["subject"])
            if "object" in intent.arguments:
                entities.append(intent.arguments["object"])

            frame_id = self.working_memory.push_context(
                context_type=context_type,
                query=intent.text_span or plan.description,
                entities=entities,
                metadata={"plan_description": plan.description},
            )

            # Initialer Reasoning-State
            self.working_memory.add_reasoning_state(
                step_type="plan_start",
                description=f"Starte Plan: {plan.description}",
                data={"intent_category": intent.category.value},
            )

            # REFACTORED: Sub-Goal-Ausführung delegiert an SubGoalExecutor
            for sub_goal in plan.sub_goals:
                self.signals.add_sub_goal.emit(sub_goal.id, sub_goal.description)
                sub_goal.status = GoalStatus.IN_PROGRESS

                # Tracke Sub-Goal Start
                self.working_memory.add_reasoning_state(
                    step_type="sub_goal_start",
                    description=f"Beginne: {sub_goal.description}",
                    data={"sub_goal_id": sub_goal.id},
                )

                # STRATEGY PATTERN: SubGoalExecutor findet passende Strategy
                success, result = self.sub_goal_executor.execute_sub_goal(
                    sub_goal, knowledge_context
                )

                if success:
                    sub_goal.status = GoalStatus.SUCCESS
                    sub_goal.result = result
                    knowledge_context.update(result)
                    logger.info(f"  -> SUCCESS: {sub_goal.description}")

                    # Tracke Sub-Goal Success
                    self.working_memory.add_reasoning_state(
                        step_type="sub_goal_success",
                        description=f"Erfolgreich: {sub_goal.description}",
                        data={"result": result},
                    )

                    if "final_response" in result:
                        final_response_text = result["final_response"]
                else:
                    sub_goal.status = GoalStatus.FAILED
                    sub_goal.error_message = result.get("error", "Unbekannter Fehler")
                    plan.status = GoalStatus.FAILED
                    logger.error(
                        f"  -> FAILED: {sub_goal.description} | Grund: {sub_goal.error_message}"
                    )

                    # Tracke Sub-Goal Failure
                    self.working_memory.add_reasoning_state(
                        step_type="sub_goal_failed",
                        description=f"Fehlgeschlagen: {sub_goal.description}",
                        data={"error": sub_goal.error_message},
                        confidence=0.0,
                    )

                    self.signals.update_sub_goal_status.emit(
                        sub_goal.id, sub_goal.status.name
                    )

                    # Pop context bei Fehler
                    self.working_memory.pop_context()

                    return KaiResponse(
                        text=f"Ich konnte den Schritt '{sub_goal.description}' nicht ausführen."
                    )

                self.signals.update_sub_goal_status.emit(
                    sub_goal.id, sub_goal.status.name
                )

                # Sende Reasoning-Trace Update an UI
                trace_str = format_reasoning_trace_for_ui(self.working_memory)
                self.signals.inner_picture_update.emit(trace_str)

            plan.status = GoalStatus.SUCCESS
            logger.info(f"Plan '{plan.description}' erfolgreich abgeschlossen.")

            # Finaler Reasoning-State
            self.working_memory.add_reasoning_state(
                step_type="plan_complete",
                description=f"Plan erfolgreich abgeschlossen: {plan.description}",
                data={"response": final_response_text},
            )

            # Pop context nach erfolgreicher Ausführung
            self.working_memory.pop_context()

            return KaiResponse(text=final_response_text)

        except KAIException as e:
            # Nutzerfreundliche Fehlermeldung für KAI-spezifische Exceptions
            user_friendly_msg = get_user_friendly_message(e, include_details=False)
            logger.error(f"KAI Exception während Plan-Ausführung: {user_friendly_msg}")
            logger.debug(f"Technische Details: {e}", exc_info=True)

            # Bereinige Working Memory bei Fehler
            try:
                self.working_memory.pop_context()
            except:
                pass

            return KaiResponse(text=user_friendly_msg)

        except Exception as e:
            # Generische Exception - Log ausführlich, zeige nutzerfreundliche Nachricht
            logger.error(f"Unerwartete Exception in execute_plan: {e}", exc_info=True)

            # Bereinige Working Memory bei Fehler
            try:
                self.working_memory.pop_context()
            except:
                pass

            return KaiResponse(
                text="[ERROR] Ein unerwarteter Fehler ist während der Ausführung aufgetreten. Bitte versuche es erneut."
            )

    # ========================================================================
    # PATTERN RECOGNITION HELPERS
    # ========================================================================

    def _remove_ui_prefixes(self, query: str) -> str:
        """
        Entfernt UI-Präfixe wie "Frage:", "Lerne:" etc. aus der Query.

        Diese Präfixe werden vom Benutzer zur Übersicht eingegeben, sollten aber
        nicht als Teil der eigentlichen Eingabe verarbeitet werden.

        Args:
            query: Die ursprüngliche Query mit möglichem Präfix

        Returns:
            Query ohne Präfix
        """
        import re

        # Liste von UI-Präfixen die entfernt werden sollen
        # Diese werden als Anzeigepräfix verwendet, aber nicht für die Verarbeitung
        ui_prefixes = [
            r"^\s*frage:\s*",
            r"^\s*lerne:\s*",
            r"^\s*definiere:\s*",
            r"^\s*erkläre:\s*",
            r"^\s*zeige:\s*",
            r"^\s*liste:\s*",
        ]

        cleaned = query
        for prefix_pattern in ui_prefixes:
            match = re.match(prefix_pattern, cleaned, re.IGNORECASE)
            if match:
                # Entferne das Präfix
                cleaned = cleaned[match.end() :]
                logger.info(f"UI-Präfix entfernt: '{query}' -> '{cleaned}'")
                break

        return cleaned.strip()

    def _create_typo_clarification(self, pattern_result: Dict) -> KaiResponse:
        """
        Erstellt "Meintest du?" Rückfrage bei Tippfehler-Unsicherheit.

        Args:
            pattern_result: Ergebnis von PatternOrchestrator.process_input()

        Returns:
            KaiResponse mit Rückfrage
        """
        corrections = pattern_result.get("typo_corrections", [])

        # Finde alle Wörter die eine Rückfrage brauchen
        unclear_words = [c for c in corrections if c.get("decision") == "ask_user"]

        if not unclear_words:
            return KaiResponse(text="Ich bin unsicher - bitte formuliere neu.")

        # Baue Rückfrage
        clarification_text = "Ich bin unsicher bei folgenden Wörtern:\n\n"

        for i, word_correction in enumerate(unclear_words):
            original = word_correction["original"]
            candidates = word_correction.get("candidates", [])

            clarification_text += f"{i+1}. '{original}' - Meintest du:\n"

            for j, candidate in enumerate(candidates[:3]):  # Max 3 Kandidaten
                clarification_text += f"   {chr(97+j)}) {candidate['word']} ({candidate['confidence']:.0%})\n"

            clarification_text += f"   oder '{original}' war korrekt?\n\n"

        clarification_text += (
            "Optionen:\n"
            "- Bestätige mit '{original}' oder 'Ja' wenn das Original korrekt ist\n"
            "- Korrigiere mit 'Nein, ich meine X' wenn ich falsch liege\n"
            "- Überspringe mit 'weiter' oder 'ignorieren' um fortzufahren"
        )

        return KaiResponse(text=clarification_text)

    # ========================================================================
    # LEGACY-METHODEN FÜR BACKWARD-KOMPATIBILITÄT
    # (werden von SubGoalExecutor-Strategien benötigt)
    # ========================================================================

    def _emit_context_update(self):
        """Legacy-Wrapper für Context Manager."""
        self.context_manager.emit_context_update()

    def _on_preview_confirmation_response(self, confirmed: bool):
        """
        PHASE 8: Handler für Preview-Bestätigung vom User.

        Args:
            confirmed: True wenn User Ingestion bestätigt hat, False bei Abbruch
        """
        logger.info(f"Preview-Bestätigung erhalten: {'Ja' if confirmed else 'Nein'}")
        self.preview_confirmation_result = confirmed
        self.preview_confirmation_event.set()  # Wecke wartenden Worker-Thread auf

    def wait_for_preview_confirmation(
        self, preview: str, file_name: str, char_count: int
    ) -> bool:
        """
        PHASE 8: Emittiert Preview-Signal und wartet auf User-Bestätigung.

        Args:
            preview: Text-Preview (erste 500 Zeichen)
            file_name: Name der Datei
            char_count: Gesamtanzahl Zeichen

        Returns:
            True wenn User bestätigt hat, False bei Abbruch
        """
        logger.info(f"Warte auf Preview-Bestätigung für: {file_name}")

        # Reset Event und Result
        self.preview_confirmation_event.clear()
        self.preview_confirmation_result = False

        # Emittiere Signal an UI (blockiert nicht)
        self.signals.preview_confirmation_needed.emit(preview, file_name, char_count)

        # Warte auf Antwort (blockiert Worker-Thread, aber nicht UI-Thread)
        self.preview_confirmation_event.wait(timeout=60.0)  # 60 Sekunden Timeout

        if not self.preview_confirmation_event.is_set():
            logger.warning("Preview-Bestätigung Timeout - Abbruch")
            return False

        logger.info(f"Preview-Bestätigung: {self.preview_confirmation_result}")
        return self.preview_confirmation_result
