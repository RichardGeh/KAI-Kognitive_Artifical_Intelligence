# component_4_goal_planner.py
from typing import Optional

from component_5_linguistik_strukturen import (
    GoalType,
    MainGoal,
    MeaningPoint,
    MeaningPointCategory,
    SubGoal,
)
from component_15_logging_config import get_logger

logger = get_logger(__name__)


class GoalPlanner:
    """
    Erstellt einen strukturierten, ausführbaren Plan (MainGoal)
    basierend auf einer erkannten Nutzerabsicht (MeaningPoint).

    Phase 2: Implementiert Confidence Gates für abgestufte Reaktionen:
    - confidence < 0.4: Clarification (völlige Unsicherheit)
    - confidence < 0.8: Confirmation (mittlere Unsicherheit)
    - confidence >= 0.8: Direct execution (hohe Sicherheit)

    Phase 3 (Schritt 3): Erweitert für autonomes Lernen:
    - Auto-erkannte Definitionen mit confidence >= 0.85: Direct save
    - Auto-erkannte Definitionen mit 0.70 <= confidence < 0.85: Confirmation
    - Auto-erkannte Definitionen mit confidence < 0.70: Clarification
    """

    def create_plan(self, meaning_point: MeaningPoint) -> Optional[MainGoal]:
        """
        Erstellt einen strukturierten, ausführbaren Plan (MainGoal)
        basierend auf der erkannten Nutzerabsicht (MeaningPoint).

        Implementiert Confidence Gates (Phase 2):
        1. confidence < 0.4 -> Clarification Plan
        2. confidence < 0.8 -> Confirmation + Original Plan
        3. confidence >= 0.8 -> Direct Execution
        4. category == UNKNOWN -> Clarification Plan

        Returns:
            MainGoal oder None wenn keine Handlung möglich
        """
        try:
            confidence = meaning_point.confidence
            category = meaning_point.category
            command = meaning_point.arguments.get("command")

            logger.debug(
                f"create_plan: category={category.name}, confidence={confidence:.2f}"
            )

            # GATE 0: Handle UNKNOWN category explicitly
            if category == MeaningPointCategory.UNKNOWN:
                logger.info("UNKNOWN category detected -> Clarification Plan")
                return self._plan_for_clarification(meaning_point)

            # GATE 1: Low confidence (< 0.4) -> Ask for clarification
            if confidence < 0.4:
                logger.info(f"Low confidence ({confidence:.2f}) -> Clarification Plan")
                return self._plan_for_clarification(meaning_point)

            # GATE 2: Medium confidence (< 0.85 for auto-detected definitions, < 0.8 for others) -> Request confirmation
            # PHASE 3 (Schritt 3): Spezialbehandlung für auto-erkannte Definitionen
            confirmation_threshold = (
                0.85
                if (
                    category == MeaningPointCategory.DEFINITION
                    and meaning_point.arguments.get("auto_detected")
                )
                else 0.8
            )

            if confidence < confirmation_threshold:
                logger.info(
                    f"Medium confidence ({confidence:.2f}, threshold={confirmation_threshold}) -> Confirmation Plan"
                )
                # Get base plan for the detected category
                base_plan = self._get_base_plan(category, command, meaning_point)
                if base_plan:
                    return self._plan_for_confirmation(base_plan, meaning_point)
                else:
                    # Fallback to clarification if no base plan available
                    return self._plan_for_clarification(meaning_point)

            # GATE 3: High confidence (>= 0.85 for auto-detected definitions, >= 0.8 for others) -> Direct execution
            logger.info(
                f"High confidence ({confidence:.2f}, threshold={confirmation_threshold}) -> Direct Execution"
            )
            return self._get_base_plan(category, command, meaning_point)

        except Exception as e:
            logger.error(f"Error in create_plan: {e}", exc_info=True)
            return None

    def _get_base_plan(
        self, category: MeaningPointCategory, command: Optional[str], mp: MeaningPoint
    ) -> Optional[MainGoal]:
        """
        Ermittelt den Basisplan für eine gegebene Kategorie/Befehl.
        Wird von den Confidence Gates aufgerufen.

        Args:
            category: Die erkannte Kategorie
            command: Optionaler Befehl (bei COMMAND category)
            mp: Der MeaningPoint mit allen Details

        Returns:
            MainGoal oder None
        """
        # REFAKTORISIERT: Verwende Tupel-Pattern für sauberes Matching
        match (category, command):
            case (MeaningPointCategory.ARITHMETIC_QUESTION, _):
                # Arithmetische Fragen (Phase Mathematik)
                return self._plan_for_arithmetic(mp)

            case (MeaningPointCategory.QUESTION, _):
                # Check if it's an episodic memory query
                query_type = mp.arguments.get("query_type")
                if query_type == "episodic_memory":
                    return self._plan_for_episodic_memory(mp)
                # Check if it's a spatial reasoning query
                if query_type == "spatial_reasoning":
                    return self._plan_for_spatial_reasoning(mp)
                return self._plan_for_question(mp)

            case (MeaningPointCategory.DEFINITION, _):
                return self._plan_for_auto_detected_definition(mp)

            case (MeaningPointCategory.COMMAND, "definiere"):
                return self._plan_for_learning_command(mp)

            case (MeaningPointCategory.COMMAND, "learn_pattern"):
                return self._plan_for_teaching_command(mp)

            case (MeaningPointCategory.COMMAND, "ingest_text"):
                return self._plan_for_ingestion(mp)

            case (MeaningPointCategory.COMMAND, "learn_simple"):
                return self._plan_for_simple_learning(mp)

            case (MeaningPointCategory.COMMAND, "read_file"):
                return self._plan_for_file_reading(mp)

            case (MeaningPointCategory.COMMAND, "ingest_document"):
                return self._plan_for_file_reading(mp)

            case (MeaningPointCategory.COMMAND, "process_pdf"):
                return self._plan_for_file_reading(mp)

            case (MeaningPointCategory.COMMAND, "load_file"):
                return self._plan_for_file_reading(mp)

            case _:
                logger.warning(
                    f"No base plan available for category={category.name}, command={command}"
                )
                return None

    def _plan_for_question(self, mp: MeaningPoint) -> MainGoal:
        """Erstellt einen standardisierten Plan zur Beantwortung einer Frage."""
        plan = MainGoal(
            type=GoalType.ANSWER_QUESTION,
            description=f"Beantworte die Frage: '{mp.text_span}'",
        )
        plan.sub_goals = [
            SubGoal(description="Identifiziere das Thema der Frage."),
            SubGoal(description="Frage den Wissensgraphen nach gelernten Fakten ab."),
            SubGoal(description="Prüfe auf Wissenslücken."),
            SubGoal(description="Formuliere eine Antwort oder eine Rückfrage."),
        ]
        return plan

    def _plan_for_episodic_memory(self, mp: MeaningPoint) -> MainGoal:
        """
        Erstellt einen Plan für episodische Gedächtnis-Abfragen.

        Unterstützt:
        - "Wann habe ich X gelernt?"
        - "Zeige mir Episoden über X"
        - "Zeige Lernverlauf"

        Args:
            mp: MeaningPoint mit query_type="episodic_memory"

        Returns:
            MainGoal mit ANSWER_QUESTION type
        """
        mp.arguments.get("episodic_query_type", "show_episodes")
        topic = mp.arguments.get("topic")

        topic_desc = f" über '{topic}'" if topic else ""
        plan = MainGoal(
            type=GoalType.ANSWER_QUESTION,
            description=f"Beantworte episodische Abfrage: '{mp.text_span}'",
        )
        plan.sub_goals = [
            SubGoal(description=f"Frage episodisches Gedächtnis ab{topic_desc}."),
            SubGoal(
                description="Formuliere eine Antwort mit Episoden-Zusammenfassung."
            ),
        ]
        return plan

    def _plan_for_spatial_reasoning(self, mp: MeaningPoint) -> MainGoal:
        """
        Erstellt einen Plan für räumliche Reasoning-Abfragen.

        Unterstützt:
        - Grid-basierte Queries (Schachbrett, Sudoku, etc.)
        - Positions-Queries (Wo liegt X?)
        - Relations-Queries (Ist X nördlich von Y?)
        - Path-Finding (Wie komme ich von X nach Y?)

        Args:
            mp: MeaningPoint mit query_type="spatial_reasoning"

        Returns:
            MainGoal mit ANSWER_QUESTION type
        """
        spatial_query_type = mp.arguments.get("spatial_query_type", "position_query")

        plan = MainGoal(
            type=GoalType.ANSWER_QUESTION,
            description=f"Beantworte räumliche Abfrage: '{mp.text_span}'",
        )

        # Erstelle Sub-Goals abhängig vom Query-Typ
        if spatial_query_type == "grid_query":
            plan.sub_goals = [
                SubGoal(description="Extrahiere räumliche Entitäten und Positionen."),
                SubGoal(description="Erstelle räumliches Modell (Grid)."),
                SubGoal(description="Formuliere räumliche Antwort."),
            ]
        elif spatial_query_type == "position_query":
            plan.sub_goals = [
                SubGoal(description="Extrahiere räumliche Entitäten und Positionen."),
                SubGoal(description="Erstelle räumliches Modell (Positionen)."),
                SubGoal(description="Formuliere räumliche Antwort."),
            ]
        elif spatial_query_type == "relation_query":
            plan.sub_goals = [
                SubGoal(description="Extrahiere räumliche Entitäten und Positionen."),
                SubGoal(description="Erstelle räumliches Modell (Relationen)."),
                SubGoal(description="Löse räumliche Constraints."),
                SubGoal(description="Formuliere räumliche Antwort."),
            ]
        elif spatial_query_type == "path_finding":
            plan.sub_goals = [
                SubGoal(description="Extrahiere räumliche Entitäten und Positionen."),
                SubGoal(description="Erstelle räumliches Modell (Path-Finding)."),
                SubGoal(description="Plane räumliche Aktionen."),
                SubGoal(description="Formuliere räumliche Antwort."),
            ]
        else:
            # Fallback: Generischer Plan
            plan.sub_goals = [
                SubGoal(description="Extrahiere räumliche Entitäten und Positionen."),
                SubGoal(description="Erstelle räumliches Modell."),
                SubGoal(description="Formuliere räumliche Antwort."),
            ]

        logger.debug(
            f"Plan für räumliche Abfrage erstellt: {spatial_query_type}, "
            f"{len(plan.sub_goals)} Sub-Goals"
        )

        return plan

    def _plan_for_auto_detected_definition(self, mp: MeaningPoint) -> MainGoal:
        """
        Erstellt einen Plan für automatisch erkannte Definitionen (deklarative Aussagen).
        Diese Methode ermöglicht natürliches Lernen ohne "Ingestiere Text:"-Befehl.

        Args:
            mp: MeaningPoint mit DEFINITION category und auto_detected flag

        Returns:
            MainGoal mit LEARN_KNOWLEDGE type
        """
        relation_type = mp.arguments.get("relation_type", "IS_A")
        subject = mp.arguments.get("subject", "unbekannt")
        object_entity = mp.arguments.get("object", "unbekannt")

        plan = MainGoal(
            type=GoalType.LEARN_KNOWLEDGE,
            description=f"Lerne automatisch erkannte Relation: '{subject}' {relation_type} '{object_entity}'",
        )
        plan.sub_goals = [
            SubGoal(description="Extrahiere Subjekt, Relation und Objekt."),
            SubGoal(description="Speichere die Relation im Wissensgraphen."),
            SubGoal(description="Formuliere eine Lernbestätigung."),
        ]

        logger.debug(
            f"Plan für auto-erkannte Definition erstellt: "
            f"{subject} {relation_type} {object_entity}"
        )

        return plan

    def _plan_for_learning_command(self, mp: MeaningPoint) -> MainGoal:
        """Erstellt einen Plan zum Lernen von Wissen aus einem 'Definiere:'-Befehl."""
        plan = MainGoal(
            type=GoalType.LEARN_KNOWLEDGE,
            description=f"Lerne Wissen aus Befehl: '{mp.text_span}'",
        )
        plan.sub_goals = [
            # KORREKTUR: Die Schritte spiegeln jetzt den direkten Netzwerkaufruf wider.
            SubGoal(description="Extrahiere Thema, Informationstyp und Inhalt."),
            SubGoal(description="Speichere die Information direkt im Wissensgraphen."),
            SubGoal(description="Formuliere eine Bestätigungsantwort."),
        ]
        return plan

    def _plan_for_teaching_command(self, mp: MeaningPoint) -> MainGoal:
        # ... (diese Methode bleibt unverändert)
        """Erstellt einen Plan, um ein Satzmuster mit einer Regel zu verknüpfen."""
        plan = MainGoal(
            type=GoalType.LEARN_KNOWLEDGE,  # Es ist eine Form des Meta-Lernens
            description=f"Lehre KAI die Bedeutung des Musters: '{mp.arguments.get('example_sentence')}'",
        )
        plan.sub_goals = [
            SubGoal(description="Verarbeite Beispielsatz zu Vektor."),
            SubGoal(description="Finde oder erstelle zugehörigen Muster-Prototypen."),
            SubGoal(description="Verknüpfe Prototyp mit Extraktionsregel."),
            SubGoal(description="Formuliere eine Lernbestätigung."),
        ]
        return plan

    def _plan_for_ingestion(self, mp: MeaningPoint) -> MainGoal:
        # ... (diese Methode bleibt unverändert)
        """Erstellt einen Plan zur Verarbeitung von unstrukturiertem Text."""
        plan = MainGoal(
            type=GoalType.PERFORM_TASK,
            description=f"Ingestiere Text: '{mp.arguments.get('text_to_ingest')[:30]}...'",
        )
        plan.sub_goals = [
            SubGoal(description="Extrahiere den zu ingestierenden Text."),
            SubGoal(description="Verarbeite Sätze durch die Ingestion-Pipeline."),
            SubGoal(description="Formuliere einen Ingestion-Bericht."),
        ]
        return plan

    def _plan_for_simple_learning(self, mp: MeaningPoint) -> MainGoal:
        """Erstellt einen Plan für einfaches Lernen mit 'Lerne: <text>' Befehl."""
        text_to_learn = mp.arguments.get("text_to_learn", "")
        plan = MainGoal(
            type=GoalType.LEARN_KNOWLEDGE,
            description=f"Lerne: '{text_to_learn[:30]}...'",
        )
        plan.sub_goals = [
            SubGoal(description="Analysiere den zu lernenden Text."),
            SubGoal(description="Extrahiere oder speichere Wissen."),
            SubGoal(description="Formuliere Lernbestätigung."),
        ]
        return plan

    def _plan_for_file_reading(self, mp: MeaningPoint) -> MainGoal:
        """
        Erstellt einen Plan für die Ingestion von Textdateien.

        Phase 3: Datei-Ingestion Pipeline
        - VALIDATE_FILE: Prüft ob Datei existiert und lesbar ist
        - EXTRACT_TEXT: Extrahiert Text aus der Datei
        - LEARN_KNOWLEDGE: Verarbeitet extrahierten Text durch Ingestion-Pipeline

        Args:
            mp: MeaningPoint mit file_path in arguments

        Returns:
            MainGoal mit READ_DOCUMENT type
        """
        file_path = mp.arguments.get("file_path", "")
        plan = MainGoal(
            type=GoalType.READ_DOCUMENT,
            description=f"Lese und verarbeite Datei: '{file_path}'",
        )
        plan.sub_goals = [
            SubGoal(description="Validiere Dateipfad und Lesbarkeit."),
            SubGoal(description="Extrahiere Text aus der Datei."),
            SubGoal(
                description="Verarbeite extrahierten Text durch Ingestion-Pipeline."
            ),
            SubGoal(description="Formuliere Ingestion-Bericht."),
        ]

        logger.debug(f"Plan für Datei-Ingestion erstellt: {file_path}")
        return plan

    def _plan_for_arithmetic(self, mp: MeaningPoint) -> MainGoal:
        """
        Erstellt einen Plan für arithmetische Berechnungen.

        Phase Mathematik: Verarbeitet arithmetische Fragen wie "Was ist 3 + 5?".

        Args:
            mp: Der MeaningPoint mit ARITHMETIC_QUESTION Kategorie

        Returns:
            MainGoal mit PERFORM_CALCULATION Typ
        """
        query_text = mp.arguments.get("query_text", mp.text_span)

        plan = MainGoal(
            type=GoalType.PERFORM_CALCULATION,
            description=f"Berechne: '{query_text}'",
        )
        plan.sub_goals = [
            SubGoal(
                description="Parse arithmetischen Ausdruck aus Text.",
                metadata={"query_text": query_text},
            ),
            SubGoal(
                description="Konvertiere Zahlwörter zu Zahlen.",
                metadata={"query_text": query_text},
            ),
            SubGoal(
                description="Führe arithmetische Operation aus.",
                metadata={"query_text": query_text},
            ),
            SubGoal(
                description="Formatiere Ergebnis als Zahlwort.",
                metadata={"query_text": query_text},
            ),
        ]

        logger.debug(f"Plan für arithmetische Berechnung erstellt: '{query_text}'")
        return plan

    def _plan_for_clarification(self, mp: MeaningPoint) -> MainGoal:
        """
        Erstellt einen Plan für niedrige Confidence oder UNKNOWN category.
        Fragt den Nutzer nach Klarstellung.

        Phase 2: Neue Methode für Confidence Gate 1 (< 0.4)

        Args:
            mp: Der MeaningPoint mit niedriger Confidence oder UNKNOWN

        Returns:
            MainGoal mit CLARIFY_INTENT Typ
        """
        original_text = (
            mp.text_span if mp.text_span else mp.arguments.get("original_text", "")
        )

        plan = MainGoal(
            type=GoalType.CLARIFY_INTENT,
            description=f"Kläre die Absicht: '{original_text[:50]}...'",
        )
        plan.sub_goals = [
            SubGoal(description="Formuliere eine allgemeine Rückfrage zur Klärung."),
        ]

        logger.debug(f"Created clarification plan for input: '{original_text[:30]}...'")
        return plan

    def _plan_for_confirmation(self, base_plan: MainGoal, mp: MeaningPoint) -> MainGoal:
        """
        Modifiziert einen existierenden Plan, um eine Bestätigung anzufordern.
        Fügt ein Confirmation-SubGoal am Anfang hinzu.

        Phase 2: Neue Methode für Confidence Gate 2 (0.4 <= conf < 0.8)

        Args:
            base_plan: Der ursprüngliche Plan für die erkannte Absicht
            mp: Der MeaningPoint mit mittlerer Confidence

        Returns:
            Modifizierter MainGoal mit Confirmation-SubGoal als erstem Schritt
        """
        # Erstelle Bestätigungs-SubGoal
        confirmation_goal = SubGoal(description="Bestätige die erkannte Absicht.")

        # Füge Bestätigung am Anfang der SubGoals ein
        base_plan.sub_goals.insert(0, confirmation_goal)

        # Update description to indicate confirmation is needed
        base_plan.description = f"[Bestätigung erforderlich] {base_plan.description}"

        logger.debug(
            f"Added confirmation step to plan. Confidence: {mp.confidence:.2f}, "
            f"Category: {mp.category.name}"
        )

        return base_plan
