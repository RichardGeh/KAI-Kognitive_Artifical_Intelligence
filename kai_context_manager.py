# kai_context_manager.py
"""
Context Management Module für KAI

Verantwortlichkeiten:
- Verwaltung von Multi-Turn-Dialogen
- Entity-Tracking über Gesprächskontext
- Kontextuelle Eingabeverarbeitung (Follow-up, Bestätigung, Feedback)
"""
import logging
from typing import Any, Callable

from component_1_netzwerk import KonzeptNetzwerk
from component_4_goal_planner import GoalPlanner
from component_5_linguistik_strukturen import (
    ContextAction,
    KaiContext,
    MeaningPointCategory,
)
from component_6_linguistik_engine import LinguisticPreprocessor
from component_7_meaning_extractor import MeaningPointExtractor
from kai_response_formatter import KaiResponse, KaiResponseFormatter

logger = logging.getLogger(__name__)


class KaiContextManager:
    """
    Manager für Kontext und Multi-Turn-Dialoge.

    Diese Klasse verwaltet:
    - Entitäten-Tracking aus Benutzereingaben
    - Kontextuelle Verarbeitung (Follow-ups, Bestätigungen, Feedback)
    - Context-Updates an UI
    """

    def __init__(
        self,
        context: KaiContext,
        signals,  # KaiSignals
        preprocessor: LinguisticPreprocessor,
        extractor: MeaningPointExtractor,
        planner: GoalPlanner,
        netzwerk: KonzeptNetzwerk,
        ingest_text_callback: Callable[[str], dict],
        execute_plan_callback: Callable[[Any, Any], KaiResponse],
        pattern_orchestrator=None,
    ):
        """
        Initialisiert den Context Manager.

        Args:
            context: KaiContext-Instanz für Zustandsverwaltung
            signals: KaiSignals-Instanz für UI-Kommunikation
            preprocessor: Linguistischer Preprocessor für Text-Analyse
            extractor: MeaningPoint-Extraktor für Intent-Erkennung
            planner: Goal-Planner für Plan-Erstellung
            netzwerk: KonzeptNetzwerk für Datenspeicherung
            ingest_text_callback: Callback für Text-Ingestion (_ingest_text)
            execute_plan_callback: Callback für Plan-Ausführung (execute_plan)
            pattern_orchestrator: Optional PatternOrchestrator für Typo-Whitelist-Verwaltung
        """
        self.context = context
        self.signals = signals
        self.preprocessor = preprocessor
        self.extractor = extractor
        self.planner = planner
        self.netzwerk = netzwerk
        self.ingest_text_callback = ingest_text_callback
        self.execute_plan_callback = execute_plan_callback
        self.formatter = KaiResponseFormatter()
        self.pattern_orchestrator = pattern_orchestrator

    def extract_and_track_entities(self, doc):
        """
        PHASE 2 (Multi-Turn): Extrahiert Entitäten aus dem spaCy-Doc und fügt sie zum Kontext hinzu.

        Args:
            doc: spaCy Doc-Objekt
        """
        # Tracke Substantive und Eigennamen als Entitäten
        for token in doc:
            if token.pos_ in ["NOUN", "PROPN"] and len(token.text) > 2:
                entity = self.formatter.clean_entity(token.text)
                if entity:
                    self.context.add_entity(entity)

        # Tracke auch Named Entities
        for ent in doc.ents:
            entity = self.formatter.clean_entity(ent.text)
            if entity:
                self.context.add_entity(entity)

    def emit_context_update(self):
        """
        PHASE 2 (Multi-Turn): Sendet ein Context-Update an die UI.
        """
        context_summary = self.context.get_context_summary()
        self.signals.context_update.emit(context_summary)

    def handle_contextual_input(self, query: str):
        """
        PHASE 4: Verarbeitet Benutzereingaben im Kontext einer vorherigen Interaktion.

        Unterstützt:
        - ERWARTE_BEISPIELSATZ: Nach Wissenslücke, ingestiert Beispieltext
        - ERWARTE_BESTAETIGUNG: Nach Confirmation-Request, prüft Ja/Nein
        - ERWARTE_FEEDBACK_ZU_CLARIFICATION: Nach Clarification-Request

        Args:
            query: Die Benutzereingabe im Kontext
        """
        action = self.context.aktion
        logger.info(
            f"Verarbeite kontextuelle Eingabe: '{query}' (Aktion: {action.value})"
        )

        try:
            if action == ContextAction.ERWARTE_BEISPIELSATZ:
                self._handle_example_sentence(query)
            elif action == ContextAction.ERWARTE_BESTAETIGUNG:
                self._handle_confirmation(query)
            elif action == ContextAction.ERWARTE_FEEDBACK_ZU_CLARIFICATION:
                self._handle_clarification_feedback(query)
            elif action == ContextAction.ERWARTE_TYPO_KLARSTELLUNG:
                self._handle_typo_clarification(query)
            elif action == ContextAction.ERWARTE_BEFEHL_BESTAETIGUNG:
                self._handle_command_confirmation(query)
            else:
                # Unerwartete Aktion
                logger.error(f"Unbekannte Kontext-Aktion: {action}")
                self.context.clear()
                self.emit_context_update()
                self.signals.finished.emit(
                    KaiResponse(
                        text="Ich bin unsicher, wie ich diese Antwort verarbeiten soll."
                    )
                )

        except Exception as e:
            logger.error(f"Fehler bei kontextueller Verarbeitung: {e}", exc_info=True)
            self.context.clear()
            self.emit_context_update()
            self.signals.finished.emit(
                KaiResponse(text="Es gab einen Fehler bei der Verarbeitung.")
            )

    def _handle_example_sentence(self, query: str):
        """
        Verarbeitet einen Beispielsatz nach Wissenslücke.

        Args:
            query: Der Beispielsatz vom Benutzer
        """
        logger.info(f"Verarbeite Antwort als Beispielsatz: '{query}'")

        # Hole das Thema aus dem Kontext
        topic = self.context.thema

        # Versuche zunächst, die Antwort als direkte Definition zu speichern
        # Prüfe ob die Antwort das Thema enthält (z.B. "Katzen sind...")
        if topic and topic in query.lower():
            # Speichere die komplette Antwort als Bedeutung
            logger.info(f"Speichere als direkte Definition für '{topic}'")
            result = self.netzwerk.add_information_zu_wort(
                topic, "bedeutung", query.strip()
            )

            if result.get("created"):
                response_text = (
                    f"Danke, ich habe mir die Bedeutung von '{topic}' gemerkt."
                )
                self.context.clear()
                self.signals.finished.emit(KaiResponse(text=response_text))
                return

        # Fallback: Versuche Fakten zu extrahieren via Ingestion
        stats = self.ingest_text_callback(query)
        facts_learned = stats["facts_created"]

        if facts_learned > 0:
            response_text = "Danke, das habe ich verstanden und gelernt."
        else:
            # Wenn auch die Ingestion fehlschlägt, speichere als Bedeutung
            if topic:
                logger.info(
                    f"Ingestion fehlgeschlagen, speichere als Bedeutung für '{topic}'"
                )
                self.netzwerk.add_information_zu_wort(topic, "bedeutung", query.strip())
                response_text = f"Danke, ich habe mir das über '{topic}' gemerkt."
            else:
                response_text = "Danke. Ich konnte daraus zwar keinen Fakt extrahieren, aber ich habe es mir gemerkt."

        self.context.clear()
        # PHASE 2 (Multi-Turn): Sende Context-Update nach clear()
        self.emit_context_update()
        self.signals.finished.emit(KaiResponse(text=response_text))

    def _handle_confirmation(self, query: str):
        """
        Verarbeitet eine Ja/Nein-Bestätigung.

        Args:
            query: Die Bestätigungsantwort vom Benutzer
        """
        query_lower = query.lower().strip()

        # Erkenne Ja/Nein-Antworten
        if any(
            word in query_lower
            for word in ["ja", "yes", "korrekt", "richtig", "stimmt", "genau"]
        ):
            # Bestätigung erhalten -> Führe Plan aus
            logger.info("Bestätigung erhalten -> Führe ursprünglichen Plan aus")

            # Hole gespeicherten Intent und Kontext
            original_intent = self.context.original_intent
            self.context.metadata.get("sub_goal_context", {})

            # Erstelle neuen Plan für bestätigten Intent
            plan = self.planner.create_plan(original_intent)

            if plan:
                # Entferne das Bestätigungs-SubGoal (erstes SubGoal)
                if plan.sub_goals and "Bestätige" in plan.sub_goals[0].description:
                    plan.sub_goals.pop(0)

                # Führe Plan aus
                self.context.clear()
                self.emit_context_update()
                final_response = self.execute_plan_callback(plan, original_intent)
                self.signals.finished.emit(final_response)
            else:
                self.context.clear()
                self.emit_context_update()
                self.signals.finished.emit(
                    KaiResponse(text="Ich konnte den Plan nicht erstellen.")
                )

        elif any(word in query_lower for word in ["nein", "no", "falsch", "nicht"]):
            # PHASE 5.2: Ablehnung erhalten -> Konstruktiver Lernvorschlag
            logger.info("Ablehnung erhalten -> Konstruktiver Lernvorschlag")

            # Hole ursprünglichen Text für konstruktiven Vorschlag
            original_intent = self.context.original_intent
            original_intent.text_span if original_intent else ""

            self.context.clear()
            self.emit_context_update()
            self.signals.finished.emit(
                KaiResponse(
                    text=(
                        "Ok, ich habe mich wohl geirrt. Was wolltest du stattdessen tun? "
                        "Du kannst es anders formulieren, oder mir mit "
                        "'Lerne Muster: \"Beispielsatz\" bedeutet KATEGORIE' ein Beispiel geben."
                    )
                )
            )

        else:
            # Unklare Antwort
            logger.warning(f"Unklare Bestätigungsantwort: '{query}'")
            self.context.clear()
            self.emit_context_update()
            self.signals.finished.emit(
                KaiResponse(
                    text="Ich verstehe nicht ganz. Bitte antworte mit 'Ja' oder 'Nein'."
                )
            )

    def _handle_clarification_feedback(self, query: str):
        """
        Verarbeitet Feedback nach einem Clarification-Request.

        Args:
            query: Das Feedback vom Benutzer
        """
        logger.info(f"Verarbeite Feedback zu Clarification: '{query}'")

        # Prüfe ob es ein "Lerne Muster:"-Befehl ist
        doc = self.preprocessor.process(query)
        meaning_points = self.extractor.extract(doc)

        if (
            meaning_points
            and meaning_points[0].category == MeaningPointCategory.COMMAND
        ):
            command = meaning_points[0].arguments.get("command")
            if command == "learn_pattern":
                # Der Nutzer gibt ein Pattern-Learning-Beispiel
                logger.info("Pattern-Learning-Befehl nach Clarification erkannt")

                # Verarbeite den Lernbefehl
                plan = self.planner.create_plan(meaning_points[0])
                if plan:
                    # Führe das Lernen aus
                    learn_response = self.execute_plan_callback(plan, meaning_points[0])

                    # Hole ursprüngliche Query aus Kontext
                    original_query = self.context.metadata.get("original_query", "")

                    # Lösche Kontext für Retry
                    self.context.clear()

                    # Versuche die ursprüngliche Frage ERNEUT
                    if original_query:
                        logger.info(
                            f"Wiederhole ursprüngliche Query: '{original_query[:50]}...'"
                        )
                        # Emitiere Lern-Bestätigung nicht, gehe direkt zur Retry
                        self.signals.finished.emit(
                            KaiResponse(
                                text="Danke, das habe ich gelernt. Lass mich die ursprüngliche Frage nochmal versuchen..."
                            )
                        )
                        # Triggere Neuverarbeitung durch separaten Aufruf
                        # HINWEIS: In der Praxis sollte dies über ein Signal geschehen,
                        # hier simulieren wir es durch direkte Verarbeitung
                        return  # UI wird die Query neu senden müssen
                    else:
                        self.signals.finished.emit(learn_response)
                        return

        # Fallback: Verarbeite als normalen neuen Input (Reformulierung)
        logger.info("Verarbeite als Reformulierung der ursprünglichen Frage")
        self.context.clear()
        self.emit_context_update()
        # Triggere normale Verarbeitung durch erneuten Aufruf
        # HINWEIS: Dies muss vom Caller (KaiWorker) behandelt werden
        # Wir signalisieren dies durch eine spezielle Response
        self.signals.finished.emit(
            KaiResponse(
                text="__REPROCESS_QUERY__",  # Spezial-Marker für KaiWorker
                trace=[query],  # Originale Query in trace
            )
        )

    def _handle_typo_clarification(self, query: str):
        """
        Verarbeitet Feedback nach einer Tippfehler-Rückfrage.

        Unterstützt:
        - "Nein, ich meine X" -> Nutzer korrigiert KAI
        - "Ja" / "Korrekt" -> Nutzer akzeptiert Vorschlag
        - Direkte Korrektur (einfach das richtige Wort eingeben)

        Args:
            query: Das Feedback vom Benutzer
        """
        logger.info(f"Verarbeite Tippfehler-Feedback: '{query}'")

        # Hole Pattern-Result und Original-Query aus Context
        pattern_result = self.context.metadata.get("pattern_result", {})
        original_query = self.context.metadata.get("original_query", "")

        if not pattern_result or not original_query:
            logger.error("Kein Pattern-Result oder Original-Query im Kontext")
            self.context.clear()
            self.emit_context_update()
            self.signals.finished.emit(
                KaiResponse(text="Es gab einen Fehler bei der Verarbeitung.")
            )
            return

        query_lower = query.lower().strip()
        typo_corrections = pattern_result.get("typo_corrections", [])

        # === FALL 0: Nutzer möchte Typo-Erkennung überspringen ===
        if any(
            word in query_lower
            for word in ["weiter", "ignorieren", "überspringen", "skip", "egal"]
        ):
            logger.info("Nutzer überspringt Typo-Korrektur")

            # Füge Original-Wörter zur Whitelist hinzu (vermeidet Loop)
            for correction in typo_corrections:
                if correction.get("decision") == "ask_user":
                    original_word = correction["original"].strip(".,!?;:'\"")
                    if self.pattern_orchestrator:
                        self.pattern_orchestrator.add_to_typo_whitelist(original_word)
                    logger.info(f"'{original_word}' zur Typo-Whitelist hinzugefügt")

            # Verwende ORIGINAL Query (keine Korrektur)
            corrected_query = original_query
            self.context.clear()
            self.emit_context_update()

            # Signalisiere Neuverarbeitung mit Original-Query
            self.signals.finished.emit(
                KaiResponse(text="__REPROCESS_QUERY__", trace=[corrected_query])
            )
            return

        # === FALL 1: Nutzer akzeptiert Vorschlag ===
        if any(
            word in query_lower
            for word in ["ja", "yes", "korrekt", "richtig", "stimmt", "genau"]
        ):
            logger.info("Nutzer akzeptiert Tippfehler-Korrektur")

            # Speichere positive Feedback für alle Auto-Korrekturen
            for correction in typo_corrections:
                if correction.get("decision") == "ask_user" and correction.get(
                    "candidates"
                ):
                    best_candidate = correction["candidates"][0]
                    self.netzwerk.store_typo_feedback(
                        original_input=correction["original"],
                        suggested_word=best_candidate["word"],
                        actual_word=best_candidate["word"],
                        user_accepted=True,
                        confidence=best_candidate["confidence"],
                    )

            # Verarbeite korrigierte Query
            corrected_query = pattern_result.get("corrected_text", original_query)
            self.context.clear()
            self.emit_context_update()

            # Signalisiere Neuverarbeitung
            self.signals.finished.emit(
                KaiResponse(text="__REPROCESS_QUERY__", trace=[corrected_query])
            )
            return

        # === FALL 2: "Nein, ich meine X" ===
        if "nein" in query_lower or "nicht" in query_lower:
            # Extrahiere das richtige Wort nach "meine"
            actual_word = None

            if "meine" in query_lower:
                parts = query_lower.split("meine", 1)
                if len(parts) > 1:
                    actual_word = (
                        parts[1].strip().split()[0] if parts[1].strip() else None
                    )

            if actual_word:
                logger.info(f"Nutzer korrigiert zu: '{actual_word}'")

                # Speichere negative Feedback für falsche Vorschläge
                for correction in typo_corrections:
                    if correction.get("decision") == "ask_user" and correction.get(
                        "candidates"
                    ):
                        best_candidate = correction["candidates"][0]
                        self.netzwerk.store_typo_feedback(
                            original_input=correction["original"],
                            suggested_word=best_candidate["word"],
                            actual_word=actual_word,
                            user_accepted=False,
                            confidence=best_candidate["confidence"],
                        )

                # Ersetze Tippfehler im Original-Query
                corrected_query = original_query
                for correction in typo_corrections:
                    corrected_query = corrected_query.replace(
                        correction["original"], actual_word
                    )

                self.context.clear()
                self.emit_context_update()

                # Signalisiere Neuverarbeitung mit korrigierter Query
                self.signals.finished.emit(
                    KaiResponse(text="__REPROCESS_QUERY__", trace=[corrected_query])
                )
                return
            else:
                # Kein Wort gefunden -> Bitte um Klarstellung
                logger.warning("'Nein, ich meine' ohne erkennbares Wort")
                self.context.clear()
                self.emit_context_update()
                self.signals.finished.emit(
                    KaiResponse(
                        text="Ich verstehe nicht ganz. Bitte formuliere: 'Nein, ich meine <Wort>'"
                    )
                )
                return

        # === FALL 3: Direkte Eingabe (einfach das richtige Wort) ===
        # Nutzer gibt einfach das richtige Wort ein

        # WICHTIG: Prüfe ob Benutzer mit dem ORIGINAL-Wort antwortet
        # Das bedeutet: "Mein Wort war korrekt, es ist KEIN Tippfehler"
        user_input_clean = query.strip().lower().rstrip("?!.,;:")

        # Finde das ursprüngliche fehlerhafte Wort
        original_word = None
        for correction in typo_corrections:
            if correction.get("decision") == "ask_user":
                original_word = correction["original"].lower().rstrip("?!.,;:")
                break

        # Prüfe ob Benutzer mit Original-Wort antwortet (Bestätigung)
        is_confirmation = original_word and user_input_clean == original_word

        if is_confirmation:
            logger.info(f"Benutzer bestätigt: '{query}' war korrekt (kein Tippfehler)")

            # Speichere Bestätigung: Original war korrekt, Suggestions waren falsch
            for correction in typo_corrections:
                if correction.get("decision") == "ask_user" and correction.get(
                    "candidates"
                ):
                    best_candidate = correction["candidates"][0]
                    self.netzwerk.store_typo_feedback(
                        original_input=correction["original"],
                        suggested_word=best_candidate["word"],
                        actual_word=correction["original"],  # Original war korrekt
                        user_accepted=False,  # Suggestion wurde NICHT akzeptiert
                        confidence=best_candidate["confidence"],
                        correction_reason="user_confirmed_original",
                    )

            # WICHTIG: Füge zur Session-Whitelist hinzu (vermeidet Loop!)
            for correction in typo_corrections:
                if correction.get("decision") == "ask_user":
                    original_word = correction["original"].strip(".,!?;:'\"")
                    if self.pattern_orchestrator:
                        self.pattern_orchestrator.add_to_typo_whitelist(original_word)
                    logger.info(f"'{original_word}' zur Typo-Whitelist hinzugefügt")

            # Verwende ORIGINAL Query (keine Korrektur nötig)
            corrected_query = original_query
        else:
            logger.info(f"Interpretiere als Korrektur: Benutzer meint '{query}'")

            # Speichere Korrektur: Benutzer gibt echtes Wort an
            for correction in typo_corrections:
                if correction.get("decision") == "ask_user" and correction.get(
                    "candidates"
                ):
                    best_candidate = correction["candidates"][0]
                    self.netzwerk.store_typo_feedback(
                        original_input=correction["original"],
                        suggested_word=best_candidate["word"],
                        actual_word=query.strip(),
                        user_accepted=False,  # Suggestion war falsch
                        confidence=best_candidate["confidence"],
                    )

            # Ersetze Tippfehler im Original-Query
            corrected_query = original_query
            for correction in typo_corrections:
                corrected_query = corrected_query.replace(
                    correction["original"], query.strip()
                )

        self.context.clear()
        self.emit_context_update()

        # Signalisiere Neuverarbeitung
        self.signals.finished.emit(
            KaiResponse(text="__REPROCESS_QUERY__", trace=[corrected_query])
        )

    def _handle_command_confirmation(self, query: str):
        """
        Verarbeitet die Bestätigung eines Befehlsvorschlags.

        Args:
            query: Die Benutzereingabe (Ja/Nein oder korrigierter Befehl)
        """
        logger.info(f"Verarbeite Befehlsbestätigung: '{query}'")

        command_suggestion = self.context.get_data("command_suggestion")
        self.context.get_data("original_query")

        if not command_suggestion:
            logger.error("Kein Befehlsvorschlag im Kontext gefunden")
            self.context.clear()
            self.emit_context_update()
            self.signals.finished.emit(
                KaiResponse(
                    text="Der Kontext ist verloren gegangen. Bitte wiederhole deine Eingabe."
                )
            )
            return

        query_lower = query.strip().lower()

        # === FALL 1: Nutzer bestätigt mit "Ja" ===
        if query_lower in ["ja", "j", "yes", "y", "ok", "genau"]:
            logger.info("Nutzer bestätigt Befehlsvorschlag")

            corrected_query = command_suggestion["full_suggestion"]

            self.context.clear()
            self.emit_context_update()

            # Signalisiere Neuverarbeitung mit korrigiertem Befehl
            self.signals.finished.emit(
                KaiResponse(text="__REPROCESS_QUERY__", trace=[corrected_query])
            )
            return

        # === FALL 2: Nutzer lehnt ab mit "Nein" oder gibt korrigierten Befehl direkt ein ===
        if query_lower in ["nein", "n", "no"]:
            logger.info("Nutzer lehnt Befehlsvorschlag ab")
            self.context.clear()
            self.emit_context_update()
            self.signals.finished.emit(
                KaiResponse(text="Ok, verstanden. Bitte gib deinen Befehl korrekt ein.")
            )
            return

        # === FALL 3: Nutzer gibt korrigierten Befehl direkt ein ===
        logger.info(f"Interpretiere als direkten Befehl: '{query}'")

        self.context.clear()
        self.emit_context_update()

        # Signalisiere Neuverarbeitung mit der Nutzer-Eingabe
        self.signals.finished.emit(
            KaiResponse(text="__REPROCESS_QUERY__", trace=[query])
        )
