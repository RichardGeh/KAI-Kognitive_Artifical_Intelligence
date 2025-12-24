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

from PySide6.QtCore import QObject, Signal, Slot

from component_1_netzwerk import KonzeptNetzwerk
from component_4_goal_planner import GoalPlanner
from component_5_linguistik_strukturen import (
    Modality,  # FIX: Required for MeaningPoint creation
)
from component_5_linguistik_strukturen import (
    Polarity,  # FIX: Required for MeaningPoint creation
)
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
    ContextType,
    WorkingMemory,
    format_reasoning_trace_for_ui,
)

# Import Pattern Recognition
from component_24_pattern_orchestrator import PatternOrchestrator

# Import Command Suggestions
from component_26_command_suggestions import get_command_suggester

# Import Input Orchestrator for multi-segment processing
from component_41_input_orchestrator import InputOrchestrator

# Import Constraint Detector for logic puzzle recognition
from component_60_constraint_detector import ConstraintDetector
from kai_config import get_config
from kai_context_manager import KaiContextManager

# Import exception utilities for user-friendly error messages
from kai_exceptions import KAIException, get_user_friendly_message
from kai_inference_handler import KaiInferenceHandler
from kai_ingestion_handler import KaiIngestionHandler

# Import new modular handlers
from kai_response_formatter import KaiResponse, KaiResponseFormatter
from kai_sub_goal_executor import SubGoalExecutor

# Import Unified Proof Explanation System
try:
    PROOF_SYSTEM_AVAILABLE = True
except ImportError:
    PROOF_SYSTEM_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Unified Proof Explanation System nicht verfügbar")

logger = logging.getLogger(__name__)

# Entity extraction exclusion list for logic puzzle heuristic
# Filters out domain nouns, colors, articles, and common puzzle words
# that should not be treated as puzzle entities.
# Synchronized with kai_strategy_dispatcher.py:1181-1216
ENTITY_EXCLUSIONS = {
    # Articles
    "Der",
    "Die",
    "Das",
    "Ein",
    "Eine",
    "Einen",
    # Domain nouns (both capitalized and lowercase)
    "Farbe",
    "Farben",
    "farbe",
    "farben",
    "Beruf",
    "Berufe",
    "beruf",
    "berufe",
    "Person",
    "Personen",
    "person",
    "personen",
    # Colors (common in color assignment puzzles) - BOTH cases
    "Rot",
    "rot",
    "Blau",
    "blau",
    "Gruen",
    "gruen",
    "Gelb",
    "gelb",
    "Schwarz",
    "schwarz",
    "Weiss",
    "weiss",
    "Grau",
    "grau",
    "Orange",
    "orange",
    "Lila",
    "lila",
    "Rosa",
    "rosa",
    "Braun",
    "braun",
    # German color adjective declined forms (neuter/accusative) - FIX for test_negative_constraints
    "Gelbes",
    "gelbes",
    "Rotes",
    "rotes",
    "Blaues",
    "blaues",
    "Gruenes",
    "gruenes",
    "Weisses",
    "weisses",
    "Schwarzes",
    "schwarzes",
    # Question words and common puzzle vocabulary
    "Vier",
    "Fuenf",
    "Drei",
    "Genau",
    "Wer",
    "Was",
    "Wie",
    "Welche",
    # German indefinite pronouns - FIX for test_negative_constraints
    "Jede",
    "jede",
    "Jeder",
    "jeder",
    "Jedes",
    "jedes",
    "Alle",
    "alle",
    "Keine",
    "keine",
    "Keiner",
    "keiner",
    # German command verbs (imperatives) - FIX for test_negative_constraints
    "Finde",
    "finde",
    "Gib",
    "gib",
    "Nenne",
    "nenne",
    "Bestimme",
    "bestimme",
    "Zeige",
    "zeige",
    "Erklaere",
    "erklaere",
    "Berechne",
    "berechne",
    # Dialogue turn labels (meta-words, not entities)
    "Turn",
    "Step",
    "Schritt",
    "Runde",
    "Punkt",
    "Teil",
    # Arithmetic puzzle nouns (German nouns start capitalized but are not entities)
    "Alter",
    "Jahre",
    "Jahr",
    "Summe",
    "Differenz",
    "Produkt",
    "Quotient",
    "Hinweise",
    "Hinweis",
    "Frage",
    "Antwort",
    "Loesung",
    # Spatial puzzle nouns
    "Tisch",
    "Reihe",
    "Platz",
    "Position",
    "Sitzordnung",
    "Uhrzeigersinn",
    # Punctuation artifacts - FIX for regex parsing errors
    "):",
    "(:",
}


class KaiSignals(QObject):
    """
    PySide6 Signals für asynchrone UI-Kommunikation.

    Diese bleiben zentral im Worker, da sie UI-Koordination betreffen.
    """

    # Input signal (from UI to Worker)
    query_submitted = Signal(str)  # User query to process

    # Output signals (from Worker to UI)
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
    production_system_trace = Signal(
        str, str
    )  # (rule_name, description) - PHASE 5: Production System Regelanwendungen


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

            # Constraint Detector für Logik-Puzzle-Erkennung
            self.constraint_detector = ConstraintDetector(
                min_conditional_rules=2,  # Mindestens 2 CONDITIONAL-Pattern (optimiert)
                confidence_threshold=0.65,  # 65% Confidence erforderlich (optimiert)
            )
            logger.info("Constraint Detector aktiviert (min_rules=2, threshold=0.65)")

            # PHASE 9: Production System Engine mit Neo4j Repository
            from component_54_production_system import (
                ProductionSystemEngine,
                create_all_content_selection_rules,
            )

            self.production_system_engine = ProductionSystemEngine(
                signals=self.signals,
                neo4j_repository=self.netzwerk._production_rules,  # Access internal repository
            )

            # Füge Standard-Regeln hinzu (falls nicht in Neo4j vorhanden)
            standard_rules = create_all_content_selection_rules()
            self.production_system_engine.add_rules(standard_rules)

            # Versuche Regeln aus Neo4j zu laden (Stats-Sync)
            loaded_count = self.production_system_engine.load_rules_from_neo4j()
            if loaded_count > 0:
                logger.info(
                    f"Production System: {loaded_count} Regel-Stats aus Neo4j geladen"
                )
            else:
                # Initial Save: Speichere Standard-Regeln zu Neo4j
                saved_count = self.production_system_engine.save_rules_to_neo4j()
                logger.info(
                    f"Production System: {saved_count} Standard-Regeln initial zu Neo4j gespeichert"
                )

            logger.info("Production System Engine aktiviert (mit Neo4j Persistierung)")

            # PHASE 10: Pattern Discovery Engine (Part 3: Clustering & Template Induction)
            from component_61_pattern_discovery import PatternDiscoveryEngine

            self.pattern_discovery = PatternDiscoveryEngine(self.netzwerk)
            self._last_discovery_count = 0  # Track when to trigger discovery
            logger.info("Pattern Discovery Engine aktiviert (clustering + LGG)")

            self.is_initialized_successfully = True
            logger.info("KAI Worker & alle Subsysteme erfolgreich initialisiert.")

            # Connect query_submitted signal to process_query slot
            self.signals.query_submitted.connect(self.process_query)
            logger.info("Query processing signal connected")

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

    @Slot(str)
    def process_query(self, query: str):
        """
        Haupteinstiegspunkt für Benutzereingaben (läuft im Worker-Thread).

        Orchestriert den gesamten Query-Processing-Flow:
        1. Speichere Utterance für Pattern Discovery (GANZ AM ANFANG)
        2. Prüfe auf aktiven Kontext (Multi-Turn-Dialog)
        3. Extrahiere Intent via MeaningPointExtractor
        4. Erstelle Plan via GoalPlanner
        5. Führe Plan aus und emittiere Ergebnis

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

            # ===================================================================
            # PHASE: UTTERANCE STORAGE (Runs FIRST, before any processing)
            # ===================================================================
            # Store EVERY user input for pattern discovery system
            # CRITICAL: This runs BEFORE goal planning so we capture raw input
            try:
                utterance_id = self._store_utterance(query)
                logger.debug(f"Utterance stored: {utterance_id[:8]}...")

                # PHASE 10: Trigger periodic pattern discovery (Part 3)
                # Only call discovery function if threshold might be reached (optimization)
                utterance_count = self.netzwerk.count_utterances(archived=False)
                if utterance_count >= self._last_discovery_count + 50:
                    self._periodic_pattern_discovery()

            except Exception as e:
                # Non-critical: Continue processing even if storage fails
                logger.warning(f"Failed to store utterance: {e}")
                utterance_id = None

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

            # WICHTIG: Entferne UI-Präfixe für weitere Verarbeitung
            # (Muss vor allen anderen Processing-Schritten passieren)
            cleaned_query = self._remove_ui_prefixes(query)

            # ===================================================================
            # PHASE: INPUT ORCHESTRATION (Runs FIRST for complex inputs)
            # ===================================================================
            # Prüfe ob Eingabe orchestriert werden sollte (mehrere Segmente)
            # WICHTIG: Dies läuft VOR Pattern Recognition, damit komplexe
            # Multi-Line Logic Puzzles nicht durch Typo Detection blockiert werden
            should_orchestrate = self.input_orchestrator.should_orchestrate(
                cleaned_query
            )

            if should_orchestrate:
                logger.info(
                    "Komplexe Multi-Line-Eingabe erkannt, überspringe Pattern Recognition"
                )
                orchestration_result = self.input_orchestrator.orchestrate_input(
                    cleaned_query
                )
            else:
                orchestration_result = None

            # ===================================================================
            # PHASE: PATTERN RECOGNITION (Only for simple single-line inputs)
            # ===================================================================
            # Nur ausführen wenn KEINE Orchestration aktiviert wurde
            if not should_orchestrate and self.pattern_orchestrator:
                logger.debug(
                    "Einfache Eingabe erkannt, führe Pattern Recognition durch"
                )
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

                # Nutze korrigierten Text für weitere Verarbeitung
                cleaned_query = pattern_result["corrected_text"]

                # Log Auto-Korrekturen
                if pattern_result.get("typo_corrections"):
                    for correction in pattern_result["typo_corrections"]:
                        if correction.get("decision") == "auto_corrected":
                            logger.info(
                                f"Auto-Korrektur: '{correction['original']}' -> '{correction['correction']}' (conf={correction['confidence']:.2f})"
                            )

            # ===================================================================
            # PHASE: ORCHESTRATED PROCESSING (Complex inputs)
            # ===================================================================
            if orchestration_result:
                # Komplexe Eingabe erkannt (Erklärungen + Fragen)
                logger.info("Orchestrierte Verarbeitung aktiviert")

                # SPEZIALFALL: LOGIK-RÄTSEL
                # Wenn ein Logik-Rätsel erkannt wurde, nutze LogicPuzzleSolver
                if orchestration_result.get("is_logic_puzzle", False):
                    logger.info(
                        "[LOGIC-PUZZLE] Logik-Rätsel erkannt, nutze LogicPuzzleSolver"
                    )

                    try:
                        # ===== HYBRID ENTITY EXTRACTION =====
                        # 1. Primär: spaCy NER (Named Entity Recognition)
                        # 2. Sekundär: Kapitalisierungs-Heuristik + Neo4j CommonWords Filter
                        # 3. Tertiär: Fallback zu Standard-Orchestration wenn leer
                        import re
                        from collections import Counter

                        from component_45_logic_puzzle_solver import LogicPuzzleSolver

                        entities = []
                        ner_entities = set()

                        # PRIMÄR: Nutze spaCy NER für PERSON entities
                        # German model uses "PER" instead of "PERSON"
                        try:
                            if self.preprocessor:
                                doc = self.preprocessor.nlp(cleaned_query)
                                person_entities = [
                                    ent.text
                                    for ent in doc.ents
                                    if ent.label_ in ("PERSON", "PER")
                                    # FIX: Filter NER entities through ENTITY_EXCLUSIONS
                                    and ent.text not in ENTITY_EXCLUSIONS
                                    # FIX: Filter punctuation artifacts from NER
                                    and ent.text.isalpha()
                                ]
                                if person_entities:
                                    ner_entities = set(person_entities)
                                    logger.info(
                                        f"[LOGIC-PUZZLE] spaCy NER: {len(ner_entities)} PER entities gefunden: {list(ner_entities)}"
                                    )
                        except Exception as e:
                            logger.warning(
                                f"[LOGIC-PUZZLE] spaCy NER fehlgeschlagen: {e}"
                            )

                        # SEKUNDÄR: Kapitalisierungs-Heuristik (IMMER anwenden als Ergänzung)
                        # weil spaCy NER nicht alle Namen erkennt
                        logger.debug(
                            "[LOGIC-PUZZLE] Ergänze mit Kapitalisierungs-Heuristik"
                        )

                        # Extrahiere kapitalisierte Wörter
                        words = re.findall(r"\b([A-Z][a-z]+)\b", cleaned_query)
                        word_counts = Counter(words)

                        # Hole CommonWords aus Neo4j (dynamisch!)
                        common_words_set = self.netzwerk.get_common_words()
                        logger.debug(
                            f"[LOGIC-PUZZLE] {len(common_words_set)} CommonWords aus Neo4j geladen"
                        )

                        # IMPROVED: Erkenne kurze Namen (3-6 chars) auch mit count=1
                        # Längere Wörter brauchen count >= 2 um False Positives zu vermeiden
                        heuristic_entities = []
                        for w, count in word_counts.items():
                            if w in ENTITY_EXCLUSIONS:
                                continue
                            if w.lower() in common_words_set:
                                continue
                            # Short names (3-6 chars) - likely proper names
                            if 3 <= len(w) <= 6 and count >= 1:
                                heuristic_entities.append(w)
                            # Longer words need count >= 2
                            elif count >= 2:
                                heuristic_entities.append(w)

                        # Kombiniere NER + Heuristik
                        entities = list(ner_entities | set(heuristic_entities))

                        if not entities:
                            logger.warning(
                                "[LOGIC-PUZZLE] Keine Entitäten erkannt, fallback zu Standard-Orchestration"
                            )
                        else:
                            logger.info(
                                f"[LOGIC-PUZZLE] Erkannte Entitäten: {entities}"
                            )

                            # Finde die Frage (letztes Segment)
                            segments = orchestration_result["segments"]
                            question_segments = [s for s in segments if s.is_question()]
                            question = (
                                question_segments[-1].text if question_segments else ""
                            )

                            # Extrahiere Bedingungen (alle Erklärungen)
                            explanation_segments = [
                                s for s in segments if s.is_explanation()
                            ]
                            conditions_text = "\n".join(
                                s.text for s in explanation_segments
                            )

                            # DEBUG: Zeige Segmente und Bedingungen
                            logger.info(
                                f"[LOGIC-PUZZLE] Segmente: {len(segments)} gesamt"
                            )
                            for i, seg in enumerate(segments, 1):
                                logger.info(
                                    f"  [{i}] {seg.segment_type.value}: '{seg.text}'"
                                )
                            logger.info(
                                f"[LOGIC-PUZZLE] Bedingungen-Text ({len(explanation_segments)} Segmente):"
                            )
                            logger.info(f"{conditions_text}")
                            logger.info(f"[LOGIC-PUZZLE] Frage: '{question}'")

                            # Löse das Rätsel
                            solver = LogicPuzzleSolver()
                            result = solver.solve(conditions_text, entities, question)

                            # Formatiere Antwort als KaiResponse
                            if result["result"] == "SATISFIABLE":
                                response = KaiResponse(
                                    text=result["answer"],
                                    trace=["Logic Puzzle Solver"],
                                    strategy="logic_puzzle",
                                    confidence=1.0,
                                    proof_tree=result.get(
                                        "proof_tree"
                                    ),  # Include ProofTree
                                )
                                self.signals.finished.emit(response)
                                logger.info(
                                    f"[LOGIC-PUZZLE] Lösung gefunden: {result['answer']}"
                                )
                                return
                            else:
                                error_msg = "Das Rätsel hat keine Lösung (Widerspruch in den Bedingungen)."
                                response = KaiResponse(
                                    text=error_msg,
                                    trace=["Logic Puzzle Solver - UNSAT"],
                                    strategy="logic_puzzle",
                                    confidence=0.0,
                                )
                                self.signals.finished.emit(response)
                                logger.warning(f"[LOGIC-PUZZLE] {error_msg}")
                                return

                    except Exception as e:
                        logger.error(
                            f"[LOGIC-PUZZLE] Fehler beim Lösen: {e}", exc_info=True
                        )
                        # Fallback zu Standard-Orchestration

                # STANDARD-ORCHESTRATION (nicht Logik-Rätsel)
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
                    modality=Modality.DECLARATIVE,  # FIX: Required parameter
                    polarity=Polarity.POSITIVE,  # FIX: Required parameter
                    confidence=0.9,
                    arguments={"orchestrated": True, "segment_count": len(segments)},
                )

                # FIX: Store puzzle metadata in intent arguments if this is a logic puzzle
                # This allows question answering strategy to route to logic puzzle solver
                if orchestration_result.get("is_logic_puzzle"):
                    # Extract entities and conditions even if direct solving wasn't attempted
                    question_segments = [s for s in segments if s.is_question()]
                    explanation_segments = [s for s in segments if s.is_explanation()]

                    puzzle_metadata = {
                        "type": "logic_puzzle",
                        "puzzle_type": orchestration_result.get("puzzle_type"),
                        "segments": segments,
                        "question": (
                            question_segments[-1].text if question_segments else ""
                        ),
                        "conditions": "\n".join(s.text for s in explanation_segments),
                        "orchestration_result": orchestration_result,
                    }
                    dummy_intent.arguments["puzzle_metadata"] = puzzle_metadata
                    logger.info(
                        f"[ORCHESTRATION] Stored puzzle metadata for question answering strategy"
                    )

                # Führe orchestrierten Plan aus
                # WICHTIG: Lernen passiert ZUERST, dann Fragen
                final_response = self.execute_plan(orchestrated_plan, dummy_intent)
                self.signals.finished.emit(final_response)
                return

            # ===================================================================
            # PHASE: CONSTRAINT DETECTION (Neu für Logik-Puzzle)
            # ===================================================================
            # Prüfe ob Eingabe ein Constraint-Problem darstellt (z.B. Logik-Rätsel)
            constraint_problem = self.constraint_detector.detect_constraint_problem(
                text=cleaned_query
            )

            if constraint_problem:
                logger.info(
                    f"[Constraint-Problem erkannt] | "
                    f"variables={len(constraint_problem.variables)}, "
                    f"constraints={len(constraint_problem.constraints)}, "
                    f"confidence={constraint_problem.confidence:.2f}"
                )
                # Speichere Constraint-Problem in Working Memory für Reasoning-Orchestrator
                self.working_memory.add_reasoning_state(
                    step_type="constraint_problem_detected",
                    description=f"Logik-Puzzle mit {len(constraint_problem.variables)} Variablen erkannt",
                    data={"problem": constraint_problem},
                    confidence=constraint_problem.confidence,
                )

                # Signal fuer UI (Inner Picture Display)
                constraint_summary = (
                    f"[CONSTRAINT-PROBLEM ERKANNT]\n"
                    f"  Variablen: {', '.join(constraint_problem.variables.keys())}\n"
                    f"  Constraints: {len(constraint_problem.constraints)} "
                    f"({', '.join(c.constraint_type for c in constraint_problem.constraints)})\n"
                    f"  Confidence: {constraint_problem.confidence:.2%}"
                )
                self.signals.inner_picture_update.emit(constraint_summary)

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

            # Füge Constraint-Problem zum Intent hinzu (falls erkannt)
            if constraint_problem:
                primary_intent.arguments["constraint_problem"] = constraint_problem

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

            # FIX: Propagate puzzle metadata and orchestration_result from intent to knowledge_context
            # This allows sub-goal strategies to detect logic puzzles and route to solver
            if "puzzle_metadata" in intent.arguments:
                puzzle_metadata = intent.arguments["puzzle_metadata"]
                knowledge_context["puzzle_metadata"] = puzzle_metadata
                # Also propagate full orchestration_result if available
                if "orchestration_result" in puzzle_metadata:
                    knowledge_context["orchestration_result"] = puzzle_metadata[
                        "orchestration_result"
                    ]
                logger.debug(
                    f"[EXECUTE_PLAN] Propagated puzzle_metadata and orchestration_result to knowledge_context"
                )

            final_response_text = "Plan ausgeführt, aber keine Antwort formuliert."
            proof_tree = None
            final_confidence = 0.8

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

            _ = self.working_memory.push_context(
                context_type=context_type,
                query=intent.text_span or plan.description,
                entities=entities,
                metadata={"plan_description": plan.description},
            )  # frame_id unused

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
                        # Extract proof_tree and confidence from result
                        if "proof_tree" in result:
                            proof_tree = result["proof_tree"]
                        if "confidence" in result:
                            final_confidence = result["confidence"]
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

            return KaiResponse(
                text=final_response_text,
                proof_tree=proof_tree,
                confidence=final_confidence,
            )

        except KAIException as e:
            # Nutzerfreundliche Fehlermeldung für KAI-spezifische Exceptions
            user_friendly_msg = get_user_friendly_message(e, include_details=False)
            logger.error(f"KAI Exception während Plan-Ausführung: {user_friendly_msg}")
            logger.debug(f"Technische Details: {e}", exc_info=True)

            # Bereinige Working Memory bei Fehler
            try:
                self.working_memory.pop_context()
            except Exception:
                pass

            return KaiResponse(text=user_friendly_msg)

        except Exception as e:
            # Generische Exception - Log ausführlich, zeige nutzerfreundliche Nachricht
            logger.error(f"Unerwartete Exception in execute_plan: {e}", exc_info=True)

            # Bereinige Working Memory bei Fehler
            try:
                self.working_memory.pop_context()
            except Exception:
                pass

            return KaiResponse(
                text="[ERROR] Ein unerwarteter Fehler ist während der Ausführung aufgetreten. Bitte versuche es erneut."
            )

    # ========================================================================
    # UTTERANCE STORAGE (Pattern Discovery System - Part 2)
    # ========================================================================

    def _store_utterance(self, text: str) -> str:
        """
        Stores user input as Utterance with Tokens for pattern discovery.

        Uses batch token creation for 10+ tokens (5-10x speedup on long sentences).
        All operations use parameterized queries for security.

        Args:
            text: User input text

        Returns:
            Utterance ID (UUID)

        Raises:
            DatabaseException: If storage fails
        """
        # 1. Create embedding (uses component_11 with caching)
        embedding = self.embedding_service.get_embedding(text)

        # 2. Create Utterance node (parameterized query)
        utterance_id = self.netzwerk.create_utterance(
            text=text,
            embedding=(
                embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
            ),
        )

        # 3. Tokenization with spaCy (uses component_6)
        doc = self.preprocessor.process(text)

        # 4. BATCH TOKEN CREATION (critical for long sentences)
        # Instead of N queries, use UNWIND for single query
        if len(doc) >= 10:
            # Batch mode: 10+ tokens -> UNWIND
            self._batch_create_tokens(utterance_id, doc)
        else:
            # Sequential mode: <10 tokens -> individual queries
            for idx, token in enumerate(doc):
                self.netzwerk.create_token(
                    surface=token.text,
                    lemma=token.lemma_,
                    pos=token.pos_,
                    utterance_id=utterance_id,
                    idx=idx,
                )

        return utterance_id

    def _batch_create_tokens(self, utterance_id: str, doc):
        """
        Batch creates tokens using UNWIND (Neo4j best practice).

        10x faster than N individual CREATE queries.
        Uses transaction for atomicity (all tokens or none).
        Creates NEXT chain in single query.

        Args:
            utterance_id: UUID of parent utterance
            doc: spaCy Doc object with tokens

        Raises:
            DatabaseException: If batch creation fails
        """
        import uuid as uuid_module

        tokens_data = [
            {
                "id": str(uuid_module.uuid4()),
                "surface": token.text,
                "lemma": token.lemma_,
                "pos": token.pos_,
                "idx": idx,
            }
            for idx, token in enumerate(doc)
        ]

        query = """
        MATCH (u:Utterance {id: $utterance_id})
        UNWIND $tokens AS tok
        CREATE (t:Token {
            id: tok.id,
            surface: tok.surface,
            lemma: tok.lemma,
            pos: tok.pos,
            idx: tok.idx
        })
        CREATE (u)-[:HAS_TOKEN {idx: tok.idx}]->(t)
        WITH t, tok
        ORDER BY tok.idx
        WITH collect(t) AS tokens
        UNWIND range(0, size(tokens) - 2) AS i
        WITH tokens[i] AS curr, tokens[i+1] AS next
        CREATE (curr)-[:NEXT]->(next)
        RETURN count(*) AS created_count
        """

        try:
            with self.netzwerk.driver.session(database="neo4j") as session:
                with session.begin_transaction() as tx:
                    try:
                        result = tx.run(
                            query, {"utterance_id": utterance_id, "tokens": tokens_data}
                        )
                        record = result.single()
                        count = record["created_count"] if record else 0
                        tx.commit()
                        logger.debug(
                            f"Batch created {len(tokens_data)} tokens (NEXT chains: {count})",
                            extra={
                                "utterance_id": utterance_id[:8],
                                "token_count": len(tokens_data),
                            },
                        )
                    except Exception as inner_e:
                        tx.rollback()
                        raise inner_e
        except Exception as e:
            from kai_exceptions import DatabaseException, wrap_exception

            raise wrap_exception(
                e,
                DatabaseException,
                "Failed to batch create tokens",
                utterance_id=utterance_id,
                token_count=len(tokens_data),
            )

    def _periodic_pattern_discovery(self):
        """
        Trigger pattern discovery every 50 new utterances (Part 3: Clustering & Induction).

        Algorithm:
        1. Count non-archived utterances
        2. Check if count >= last_count + 50
        3. If yes:
           a. Cluster utterances by embedding similarity
           b. Induce template from each cluster via LGG
           c. Create Pattern/PatternItem/Slot nodes
           d. Update last_count tracker

        Called after each utterance storage to check if threshold reached.
        Non-blocking: Runs in worker thread, doesn't affect query processing.
        """
        try:
            # 1. Count non-archived utterances
            utterance_count = self.netzwerk.count_utterances(archived=False)

            # 2. Check if we've accumulated enough new utterances
            threshold = 50  # Configurable: Trigger every N utterances
            if utterance_count >= self._last_discovery_count + threshold:
                logger.info(
                    f"Pattern Discovery triggered: "
                    f"{utterance_count} utterances (threshold={threshold})"
                )

                # 3. Cluster utterances by embedding similarity
                clusters = self.pattern_discovery.cluster_utterances(
                    min_cluster_size=3,  # At least 3 examples per pattern
                    similarity_threshold=0.85,  # High similarity = specific patterns
                )

                if not clusters:
                    logger.info("Pattern Discovery: No clusters found")
                    self._last_discovery_count = utterance_count
                    return

                logger.info(f"Pattern Discovery: Found {len(clusters)} clusters")

                # 4. Induce template from each cluster
                patterns_created = 0
                for cluster in clusters:
                    try:
                        pattern_id = (
                            self.pattern_discovery.induce_template_from_cluster(cluster)
                        )
                        patterns_created += 1
                        logger.debug(
                            f"Pattern created from cluster: {pattern_id[:8]} "
                            f"(size={cluster['size']})"
                        )
                    except Exception as cluster_e:
                        # Log but don't fail entire discovery if one cluster fails
                        logger.warning(
                            f"Failed to create pattern from cluster "
                            f"{cluster.get('cluster_id')}: {cluster_e}"
                        )

                logger.info(
                    f"Pattern Discovery complete: "
                    f"{patterns_created} new patterns created from {len(clusters)} clusters"
                )

                # 5. Update tracker
                self._last_discovery_count = utterance_count

        except Exception as e:
            # Pattern discovery is non-critical - log error but don't crash
            logger.error(f"Pattern discovery failed: {e}", exc_info=True)

    def _cleanup_old_utterances(self):
        """
        Archiviert Utterances älter als 30 Tage.

        Called periodically (e.g., daily) to prevent unbounded graph growth.
        Archived utterances are marked but not deleted (statistics still available).
        """
        try:
            archived_count = self.netzwerk.archive_old_utterances(days_threshold=30)
            logger.info(
                f"Archived {archived_count} old utterances (older than 30 days)"
            )
        except Exception as e:
            logger.error(f"Failed to archive old utterances: {e}", exc_info=True)

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
