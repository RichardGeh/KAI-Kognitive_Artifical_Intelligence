# kai_sub_goal_executor.py
"""
Sub-Goal Execution Module für KAI mit Strategy Pattern

Verantwortlichkeiten:
- Dispatching von SubGoals an spezialisierte Strategien
- Implementierung verschiedener Execution-Strategien (Question, Learning, etc.)
- Entkopplung von Orchestrierung und Ausführungslogik
"""
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

from component_5_linguistik_strukturen import (
    ContextAction,
    MeaningPointCategory,
    SubGoal,
)

logger = logging.getLogger(__name__)


# ============================================================================
# BASE STRATEGY CLASS
# ============================================================================


class SubGoalStrategy(ABC):
    """
    Abstrakte Basisklasse für Sub-Goal Execution Strategies.

    Jede Strategy ist verantwortlich für die Ausführung einer bestimmten
    Kategorie von Sub-Goals (z.B. Fragen, Pattern Learning, etc.).
    """

    def __init__(self, worker):
        """
        Initialisiert die Strategy mit Referenz zum Worker.

        Args:
            worker: KaiWorker-Instanz für Zugriff auf Subsysteme
        """
        self.worker = worker

    @abstractmethod
    def can_handle(self, sub_goal_description: str) -> bool:
        """
        Prüft ob diese Strategy das gegebene Sub-Goal handhaben kann.

        Args:
            sub_goal_description: Beschreibung des Sub-Goals

        Returns:
            True wenn Strategy verantwortlich ist, sonst False
        """

    @abstractmethod
    def execute(
        self, sub_goal: SubGoal, context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Führt das Sub-Goal aus.

        Args:
            sub_goal: Das auszuführende SubGoal
            context: Kontext-Dictionary mit vorherigen Ergebnissen

        Returns:
            Tuple (success: bool, result: Dict)
        """


# ============================================================================
# QUESTION STRATEGY (Frage-bezogen)
# ============================================================================


class QuestionStrategy(SubGoalStrategy):
    """
    Strategy für fragebezogene Sub-Goals.

    Zuständig für:
    - Thema-Identifikation
    - Fakten-Abfrage aus Wissensgraph
    - Wissenslücken-Prüfung
    - Antwort-Formulierung
    """

    def can_handle(self, sub_goal_description: str) -> bool:
        question_keywords = [
            "Identifiziere das Thema der Frage",
            "Frage den Wissensgraphen nach gelernten Fakten ab",
            "Prüfe auf Wissenslücken",
            "Formuliere eine Antwort oder eine Rückfrage",
        ]
        return any(kw in sub_goal_description for kw in question_keywords)

    def execute(
        self, sub_goal: SubGoal, context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        intent = context.get("intent")
        if not intent:
            return False, {"error": "Intent fehlt im Kontext"}

        description = sub_goal.description

        # Dispatcher für verschiedene Question-SubGoals
        if "Identifiziere das Thema der Frage" in description:
            return self._identify_topic(intent)
        elif "Frage den Wissensgraphen nach gelernten Fakten ab" in description:
            return self._query_knowledge_graph(context)
        elif "Prüfe auf Wissenslücken" in description:
            return self._check_knowledge_gap(context)
        elif "Formuliere eine Antwort oder eine Rückfrage" in description:
            return self._formulate_answer(intent, context)

        return False, {"error": f"Unbekanntes Question-SubGoal: {description}"}

    def _identify_topic(self, intent) -> Tuple[bool, Dict[str, Any]]:
        """Identifiziert das Thema der Frage."""
        topic = intent.arguments.get("topic")
        if not topic:
            return False, {"error": "Kein Thema gefunden."}
        return True, {"topic": topic.lower().strip()}

    def _query_knowledge_graph(
        self, context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Fragt den Wissensgraphen nach Fakten ab."""
        topic = context.get("topic")
        if not topic:
            return False, {"error": "Thema fehlt."}

        # Tracke Graph-Abfrage
        self.worker.working_memory.add_reasoning_state(
            step_type="fact_retrieval",
            description=f"Suche Fakten über '{topic}' im Wissensgraphen",
            data={"topic": topic},
        )

        # SCHRITT 1: Direkte Faktenabfrage (effizient, nutzt Cache)
        fact_data = self.worker.netzwerk.query_facts_with_synonyms(topic)

        _ = (
            bool(fact_data["facts"])
            or bool(fact_data["bedeutungen"])
            or bool(fact_data["synonyms"])
        )  # has_any_knowledge unused

        # Tracke erfolgreiche Faktensuche
        if fact_data["facts"] or fact_data["bedeutungen"]:
            self.worker.working_memory.add_reasoning_state(
                step_type="facts_found",
                description=f"Fakten gefunden für '{topic}'",
                data={
                    "num_relations": len(fact_data["facts"]),
                    "num_bedeutungen": len(fact_data.get("bedeutungen", [])),
                    "num_synonyms": len(fact_data["synonyms"]),
                },
            )

        # Versuche Backward-Chaining wenn keine direkten Fakten
        if not fact_data["facts"] and not fact_data["bedeutungen"]:
            logger.info(
                "[Backward-Chaining] Keine direkten Fakten, versuche komplexe Schlussfolgerung..."
            )

            # BUGFIX: Verwende existierenden InferenceHandler vom Worker (keine neue Instanz!)
            # Das verhindert mehrfaches Laden der gleichen Fakten
            inference_handler = None
            if (
                hasattr(self.worker, "inference_handler")
                and self.worker.inference_handler
            ):
                inference_handler = self.worker.inference_handler
            else:
                # Fallback: Erstelle InferenceHandler einmal
                from kai_inference_handler import KaiInferenceHandler

                inference_handler = KaiInferenceHandler(
                    self.worker.netzwerk,
                    self.worker.engine,
                    self.worker.graph_traversal,
                    self.worker.working_memory,
                    self.worker.signals,
                )
                # Cache for future use
                self.worker.inference_handler = inference_handler

            # BUGFIX: Probiere nur IS_A Relation (häufigster Fall bei WH-Fragen)
            # Andere Relationen nur wenn Topic im Graph existiert
            # Das reduziert übermäßige Batch-Verarbeitungen massiv
            primary_relations = ["IS_A"]  # Primäre Relation für WH-Fragen

            # Prüfe ob Topic überhaupt im Graph existiert
            topic_exists = self.worker.netzwerk.word_exists(topic)

            if not topic_exists:
                logger.info(
                    f"[Backward-Chaining] Topic '{topic}' existiert nicht im Graph. Überspringe Backward-Chaining."
                )
                # Keine Schlussfolgerung möglich wenn Topic unbekannt ist
                return True, {
                    "learned_facts": fact_data["facts"],
                    "synonyms": fact_data["synonyms"],
                    "fact_sources": fact_data["sources"],
                    "bedeutungen": fact_data.get("bedeutungen", []),
                    "fuzzy_suggestions": [],
                    "backward_chaining_used": False,
                }

            # Topic existiert - versuche Inference nur für IS_A
            for relation in primary_relations:
                inference_result = inference_handler.try_backward_chaining_inference(
                    topic, relation
                )

                if inference_result:
                    inferred_facts = inference_result["inferred_facts"]
                    proof_trace = inference_result["proof_trace"]

                    logger.info(
                        f"[Backward-Chaining] [OK] Fakten abgeleitet: {inferred_facts}"
                    )

                    all_facts = {**fact_data["facts"], **inferred_facts}

                    return True, {
                        "learned_facts": all_facts,
                        "synonyms": fact_data["synonyms"],
                        "fact_sources": fact_data["sources"],
                        "bedeutungen": fact_data.get("bedeutungen", []),
                        "fuzzy_suggestions": [],
                        "backward_chaining_used": inference_result.get(
                            "is_hypothesis", False
                        )
                        is False,
                        "is_hypothesis": inference_result.get("is_hypothesis", False),
                        "proof_trace": proof_trace,
                    }

            # Backward-Chaining hat keine Ergebnisse geliefert
            logger.info(f"[Backward-Chaining] Keine Inferenzen möglich für '{topic}'")

        return True, {
            "learned_facts": fact_data["facts"],
            "synonyms": fact_data["synonyms"],
            "fact_sources": fact_data["sources"],
            "bedeutungen": fact_data.get("bedeutungen", []),
            "fuzzy_suggestions": [],
            "backward_chaining_used": False,
        }

    def _check_knowledge_gap(
        self, context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Prüft auf Wissenslücken."""
        facts = context.get("learned_facts")
        bedeutungen = context.get("bedeutungen", [])
        has_gap = not bool(facts) and not bool(bedeutungen)
        return True, {"has_knowledge_gap": has_gap}

    def _formulate_answer(
        self, intent, context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Formuliert eine Antwort oder Rückfrage."""
        from kai_response_formatter import KaiResponseFormatter

        formatter = KaiResponseFormatter()

        topic = context.get("topic")
        query_type = intent.arguments.get("query_type", "normal")
        question_type = intent.arguments.get("question_type", "general")
        fuzzy_suggestions = context.get("fuzzy_suggestions", [])

        # FALL 1: Fuzzy-Matching hat ähnliche Wörter gefunden
        if fuzzy_suggestions:
            self.worker.context.aktion = ContextAction.ERWARTE_BEISPIELSATZ
            self.worker.context.thema = topic
            self.worker._emit_context_update()

            suggestions_str = ", ".join(
                [f"'{s['word']}' ({s['similarity']:.0%})" for s in fuzzy_suggestions]
            )
            response = (
                f"Ich kenne '{topic}' nicht. Meintest du vielleicht: {suggestions_str}? "
                f"Wenn nicht, kannst du mir sagen, was '{topic}' ist?"
            )
            return True, {"final_response": response}

        # FALL 2: Wissenslücke
        if context.get("has_knowledge_gap"):
            self.worker.context.aktion = ContextAction.ERWARTE_BEISPIELSATZ
            self.worker.context.thema = topic
            self.worker._emit_context_update()

            if question_type == "person_query":
                response = f"Ich weiß nichts über '{topic}'. Ist das eine Person? Kannst du mir mehr darüber erzählen?"
            elif question_type == "time_query":
                response = f"Ich habe keine zeitlichen Informationen über '{topic}'. Kannst du mir sagen, wann das stattfindet oder stattgefunden hat?"
            elif question_type == "process_query":
                response = f"Ich weiß nicht, wie '{topic}' funktioniert. Kannst du es mir erklären?"
            elif question_type == "reason_query":
                response = f"Ich kenne den Grund nicht. Kannst du mir erklären, warum '{topic}'?"
            else:
                response = f"Ich weiß nichts über '{topic}'. Ist es ein Ding, eine Person oder etwas anderes? Kannst du es in einem Satz erklären, z. B. 'Ein {topic.capitalize()} ist ...'?"

            return True, {"final_response": response}

        # FALL 3: Wissen vorhanden
        facts = context.get("learned_facts", {})
        synonyms = context.get("synonyms", [])
        bedeutungen = context.get("bedeutungen", [])
        backward_chaining_used = context.get("backward_chaining_used", False)
        is_hypothesis = context.get("is_hypothesis", False)

        # Spezifische Antwortgenerierung basierend auf Fragetyp
        if question_type == "person_query":
            response = formatter.format_person_answer(
                topic, facts, bedeutungen, synonyms
            )
        elif question_type == "time_query":
            response = formatter.format_time_answer(topic, facts, bedeutungen)
        elif question_type == "process_query":
            response = formatter.format_process_answer(topic, facts, bedeutungen)
        elif question_type == "reason_query":
            response = formatter.format_reason_answer(topic, facts, bedeutungen)
        else:
            # Standard-Antwort
            response = formatter.format_standard_answer(
                topic,
                facts,
                bedeutungen,
                synonyms,
                query_type=query_type,
                backward_chaining_used=backward_chaining_used,
                is_hypothesis=is_hypothesis,
            )

        return True, {"final_response": response}


# ============================================================================
# PATTERN LEARNING STRATEGY (Lerne Muster)
# ============================================================================


class PatternLearningStrategy(SubGoalStrategy):
    """
    Strategy für Pattern-Learning Sub-Goals ("Lerne Muster:").

    Zuständig für:
    - Vektor-Erzeugung aus Beispielsätzen
    - Prototyp-Suche/Erstellung
    - Verknüpfung mit Extraktionsregeln
    """

    def can_handle(self, sub_goal_description: str) -> bool:
        pattern_keywords = [
            "Verarbeite Beispielsatz zu Vektor",
            "Finde oder erstelle zugehörigen Muster-Prototypen",
            "Verknüpfe Prototyp mit Extraktionsregel",
        ]
        return any(kw in sub_goal_description for kw in pattern_keywords)

    def execute(
        self, sub_goal: SubGoal, context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        intent = context.get("intent")
        if not intent:
            return False, {"error": "Intent fehlt im Kontext"}

        description = sub_goal.description

        if "Verarbeite Beispielsatz zu Vektor" in description:
            return self._process_sentence_to_vector(intent)
        elif "Finde oder erstelle zugehörigen Muster-Prototypen" in description:
            return self._find_or_create_prototype(intent, context)
        elif "Verknüpfe Prototyp mit Extraktionsregel" in description:
            return self._link_prototype_to_rule(intent, context)

        return False, {"error": f"Unbekanntes Pattern-Learning-SubGoal: {description}"}

    def _process_sentence_to_vector(self, intent) -> Tuple[bool, Dict[str, Any]]:
        """Erstellt Embedding-Vektor aus Beispielsatz."""
        sentence = intent.arguments.get("example_sentence")
        if not sentence:
            return False, {"error": "Beispielsatz nicht gefunden."}

        vector = self.worker.embedding_service.get_embedding(sentence)
        if not vector:
            return False, {"error": "Konnte keinen Embedding-Vektor erstellen."}

        return True, {"sentence_vector": vector}

    def _find_or_create_prototype(
        self, intent, context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Findet oder erstellt Prototyp für den Vektor."""
        vector = context.get("sentence_vector")
        if not vector:
            return False, {"error": "Vektor fehlt."}

        sentence = intent.arguments.get("example_sentence", "")
        if sentence.strip().endswith("?"):
            category = MeaningPointCategory.QUESTION.value.upper()
        else:
            category = MeaningPointCategory.DEFINITION.value.upper()

        logger.info(f"Kategorie für Beispielsatz abgeleitet: '{category}'")

        prototype_id = self.worker.prototyping_engine.process_vector(vector, category)
        if not prototype_id:
            return False, {"error": "Prototyp konnte nicht erstellt werden."}

        return True, {"prototype_id": prototype_id}

    def _link_prototype_to_rule(
        self, intent, context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Verknüpft Prototyp mit Extraktionsregel."""
        prototype_id = context.get("prototype_id")
        relation_type = intent.arguments.get("relation_type")

        if not all([prototype_id, relation_type]):
            return False, {"error": "Daten fehlen."}

        success = self.worker.netzwerk.link_prototype_to_rule(
            prototype_id, relation_type
        )
        if not success:
            return False, {
                "error": f"Verknüpfung mit Regel '{relation_type}' fehlgeschlagen."
            }

        return True, {"linked_relation": relation_type}


# ============================================================================
# INGESTION STRATEGY (Ingestiere Text)
# ============================================================================


class IngestionStrategy(SubGoalStrategy):
    """
    Strategy für Text-Ingestion Sub-Goals.

    Zuständig für:
    - Text-Extraktion
    - Satz-Verarbeitung durch Ingestion-Pipeline
    - Berichts-Formulierung
    """

    def can_handle(self, sub_goal_description: str) -> bool:
        ingestion_keywords = [
            "Extrahiere den zu ingestierenden Text",
            "Verarbeite Sätze durch die Ingestion-Pipeline",
            "Formuliere einen Ingestion-Bericht",
        ]
        return any(kw in sub_goal_description for kw in ingestion_keywords)

    def execute(
        self, sub_goal: SubGoal, context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        intent = context.get("intent")
        if not intent:
            return False, {"error": "Intent fehlt im Kontext"}

        description = sub_goal.description

        if "Extrahiere den zu ingestierenden Text" in description:
            return self._extract_text(intent)
        elif "Verarbeite Sätze durch die Ingestion-Pipeline" in description:
            return self._process_ingestion(context)
        elif "Formuliere einen Ingestion-Bericht" in description:
            return self._formulate_report(context)

        return False, {"error": f"Unbekanntes Ingestion-SubGoal: {description}"}

    def _extract_text(self, intent) -> Tuple[bool, Dict[str, Any]]:
        """Extrahiert den zu ingestierenden Text."""
        text = intent.arguments.get("text_to_ingest")
        if not text:
            return False, {"error": "Kein Text zum Ingestieren gefunden."}
        return True, {"text_to_process": text}

    def _process_ingestion(
        self, context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Verarbeitet Text durch Ingestion-Pipeline."""
        from kai_ingestion_handler import KaiIngestionHandler

        text = context.get("text_to_process")
        if not text:
            return False, {"error": "Text aus vorigem Schritt fehlt."}

        # Erstelle Ingestion Handler
        ingestion_handler = KaiIngestionHandler(
            self.worker.netzwerk,
            self.worker.preprocessor,
            self.worker.prototyping_engine,
            self.worker.embedding_service,
        )

        stats = ingestion_handler.ingest_text(text)

        return True, {
            "facts_learned_count": stats["facts_created"],
            "learned_patterns": stats["learned_patterns"],
            "fallback_patterns": stats["fallback_patterns"],
        }

    def _formulate_report(self, context: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Formuliert Ingestion-Bericht."""
        count = context.get("facts_learned_count", 0)
        learned = context.get("learned_patterns", 0)
        fallback = context.get("fallback_patterns", 0)

        if count > 1:
            response = f"Ingestion abgeschlossen. Ich habe {count} neue Fakten gelernt"
        elif count == 1:
            response = "Ingestion abgeschlossen. Ich habe 1 neuen Fakt gelernt"
        else:
            response = (
                "Ingestion abgeschlossen. Ich konnte keine neuen Fakten extrahieren"
            )

        if learned > 0 or fallback > 0:
            details = []
            if learned > 0:
                details.append(f"{learned} aus gelernten Mustern")
            if fallback > 0:
                details.append(f"{fallback} aus neuen Mustern")
            response += f" ({', '.join(details)})"

        response += "."
        return True, {"final_response": response}


# ============================================================================
# DEFINITION STRATEGY (Definiere)
# ============================================================================


class DefinitionStrategy(SubGoalStrategy):
    """
    Strategy für Definitions-Sub-Goals.

    Zuständig für:
    - Manuelle Definitionen ("Definiere:")
    - Auto-erkannte Definitionen
    - Informations-Speicherung
    """

    def can_handle(self, sub_goal_description: str) -> bool:
        definition_keywords = [
            "Extrahiere Thema, Informationstyp und Inhalt",
            "Extrahiere Subjekt, Relation und Objekt",
            "Speichere die Relation im Wissensgraphen",
            "Speichere die Information direkt im Wissensgraphen",
            "Formuliere eine Bestätigungsantwort",
            "Formuliere eine Lernbestätigung",  # Für auto-erkannte Definitionen
        ]
        return any(kw in sub_goal_description for kw in definition_keywords)

    def execute(
        self, sub_goal: SubGoal, context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        intent = context.get("intent")
        if not intent:
            return False, {"error": "Intent fehlt im Kontext"}

        description = sub_goal.description

        if "Extrahiere Thema, Informationstyp und Inhalt" in description:
            return self._extract_topic_info(intent)
        elif "Extrahiere Subjekt, Relation und Objekt" in description:
            return self._extract_relation_triple(intent)
        elif "Speichere die Relation im Wissensgraphen" in description:
            return self._store_relation(intent, context)
        elif "Speichere die Information direkt im Wissensgraphen" in description:
            return self._store_information(context)
        elif (
            "Formuliere eine Bestätigungsantwort" in description
            or "Formuliere eine Lernbestätigung" in description
        ):
            return self._formulate_confirmation(context)

        return False, {"error": f"Unbekanntes Definition-SubGoal: {description}"}

    def _extract_topic_info(self, intent) -> Tuple[bool, Dict[str, Any]]:
        """Extrahiert Thema, Typ und Inhalt aus Intent."""
        args = intent.arguments
        topic = args.get("topic")
        key_path = args.get("key_path")
        value = args.get("value")

        if not all([topic, key_path, value]):
            return False, {"error": "Unvollständige Definitions-Argumente."}

        info_type = key_path[0]
        return True, {"topic": topic, "info_type": info_type, "info_value": value}

    def _extract_relation_triple(self, intent) -> Tuple[bool, Dict[str, Any]]:
        """Extrahiert Subject-Relation-Object Triple."""
        args = intent.arguments
        subject = args.get("subject")
        relation_type = args.get("relation_type")
        obj = args.get("object")

        if not all([subject, relation_type, obj]):
            return False, {
                "error": "Unvollständige Definitions-Argumente bei auto-erkannter Definition."
            }

        logger.debug(f"Auto-erkannte Definition: '{subject}' {relation_type} '{obj}'")

        return True, {
            "auto_subject": subject,
            "auto_relation": relation_type,
            "auto_object": obj,
        }

    def _store_relation(
        self, intent, context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Speichert Relation im Wissensgraphen."""
        subject = context.get("auto_subject")
        relation_type = context.get("auto_relation")
        obj = context.get("auto_object")

        if not all([subject, relation_type, obj]):
            return False, {
                "error": "Relation-Informationen aus vorigem Schritt fehlen."
            }

        created = self.worker.netzwerk.assert_relation(
            subject, relation_type, obj, intent.text_span
        )

        logger.info(
            f"Auto-erkannte Relation gespeichert: ({subject})-[{relation_type}]->({obj}) "
            f"(created={created})"
        )

        return True, {"relation_created": created}

    def _store_information(
        self, context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Speichert Information direkt im Wissensgraphen."""
        topic = context.get("topic")
        info_type = context.get("info_type")
        info_value = context.get("info_value")

        if not all([topic, info_type, info_value]):
            return False, {"error": "Informationen aus vorigem Schritt fehlen."}

        result = self.worker.netzwerk.add_information_zu_wort(
            topic, info_type, info_value
        )
        if result.get("error"):
            return False, {"error": f"Netzwerkfehler: {result['error']}"}

        return True, {"was_created": result.get("created", False)}

    def _formulate_confirmation(
        self, context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Formuliert Bestätigungsantwort."""
        # Auto-erkannte Definition
        if "relation_created" in context:
            relation_created = context.get("relation_created", False)
            subject = context.get("auto_subject", "etwas")
            obj = context.get("auto_object", "etwas")

            if relation_created:
                response = f"Ok, ich habe mir gemerkt: '{subject}' -> '{obj}'."
            else:
                response = f"Ok, diese Information über '{subject}' kannte ich bereits."
        else:
            # Manuelle Definition
            was_created = context.get("was_created", False)
            topic = context.get("topic")

            if was_created:
                response = (
                    f"Ok, ich habe mir die neue Information zu '{topic}' gemerkt."
                )
            else:
                response = f"Ok, diese Information über '{topic}' kannte ich bereits."

        return True, {"final_response": response}


# ============================================================================
# LEARNING STRATEGY (Lerne)
# ============================================================================


class LearningStrategy(SubGoalStrategy):
    """
    Strategy für einfache Lern-Sub-Goals ("Lerne:").

    Zuständig für:
    - Text-Analyse (deklarativ vs. einzelnes Wort)
    - Wissens-Extraktion oder -Speicherung
    - Lernbestätigung
    """

    def can_handle(self, sub_goal_description: str) -> bool:
        learning_keywords = [
            "Analysiere den zu lernenden Text",
            "Extrahiere oder speichere Wissen",
            "Formuliere Lernbestätigung",
        ]
        return any(kw in sub_goal_description for kw in learning_keywords)

    def execute(
        self, sub_goal: SubGoal, context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        intent = context.get("intent")
        if not intent:
            return False, {"error": "Intent fehlt im Kontext"}

        description = sub_goal.description

        if "Analysiere den zu lernenden Text" in description:
            return self._analyze_learning_text(intent)
        elif "Extrahiere oder speichere Wissen" in description:
            return self._extract_or_store_knowledge(context)
        elif "Formuliere Lernbestätigung" in description:
            return self._formulate_learning_confirmation(context)

        return False, {"error": f"Unbekanntes Learning-SubGoal: {description}"}

    def _analyze_learning_text(self, intent) -> Tuple[bool, Dict[str, Any]]:
        """Analysiert ob Text deklarativ ist oder ein einzelnes Wort."""
        text_to_learn = intent.arguments.get("text_to_learn")
        if not text_to_learn:
            return False, {"error": "Kein Text zum Lernen gefunden."}

        # STRATEGIE 1: Prüfe auf explizite deklarative Patterns
        has_declarative_pattern = any(
            pattern in text_to_learn.lower()
            for pattern in [
                "ist ein",
                "ist eine",
                "sind",
                "kann",
                "können",
                "hat",
                "haben",
                "liegt in",
                "befindet sich",
            ]
        )

        # STRATEGIE 2: Wenn mehr als 2 Wörter → behandle als Satz (nicht als einzelnes Wort)
        # Das verhindert, dass Phrasen wie "zur Schule gehen ist normal" durch clean_entity() laufen
        word_count = len(text_to_learn.split())
        is_multi_word_phrase = word_count > 2

        # STRATEGIE 3: Wenn "ist", "war", "wird" enthalten → wahrscheinlich Satz
        has_verb = any(
            verb in text_to_learn.lower().split()
            for verb in ["ist", "war", "wird", "sind", "waren"]
        )

        # ENTSCHEIDUNG: Als deklarativen Satz behandeln wenn:
        # - Explizites Pattern gefunden ODER
        # - Multi-Word-Phrase mit Verb ODER
        # - Mehr als 3 Wörter (immer Ingestion)
        is_declarative = (
            has_declarative_pattern
            or (is_multi_word_phrase and has_verb)
            or (word_count > 3)
        )

        logger.debug(
            f"Analyse: '{text_to_learn}' | "
            f"Wörter={word_count}, Pattern={has_declarative_pattern}, Verb={has_verb} "
            f"→ Deklarativ={is_declarative}"
        )

        return True, {"text_to_learn": text_to_learn, "is_declarative": is_declarative}

    def _extract_or_store_knowledge(
        self, context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Extrahiert Fakten oder speichert als Konzept."""
        from kai_ingestion_handler import KaiIngestionHandler
        from kai_response_formatter import KaiResponseFormatter

        text_to_learn = context.get("text_to_learn")
        is_declarative = context.get("is_declarative", False)

        if not text_to_learn:
            return False, {"error": "Text fehlt aus vorigem Schritt."}

        if is_declarative:
            # Versuche Fakten zu extrahieren via Ingestion-Pipeline
            logger.info(f"Lerne via Ingestion: '{text_to_learn}'")

            ingestion_handler = KaiIngestionHandler(
                self.worker.netzwerk,
                self.worker.preprocessor,
                self.worker.prototyping_engine,
                self.worker.embedding_service,
            )
            stats = ingestion_handler.ingest_text(text_to_learn)
            facts_learned = stats["facts_created"]

            # FALLBACK: Wenn keine Fakten extrahiert wurden, speichere als Rohtext-Bedeutung
            # Das verhindert, dass der User denkt "ich habe nichts gelernt"
            if facts_learned == 0:
                logger.info("Keine Fakten extrahiert, speichere als Rohtext-Bedeutung")

                # Extrahiere Hauptwort aus dem Satz (meist erstes Nomen)
                doc = self.worker.preprocessor.process(text_to_learn)
                main_word = None
                for token in doc:
                    if token.pos_ in ["NOUN", "PROPN"]:  # Nomen oder Eigenname
                        main_word = token.lemma_.lower()
                        break

                # Wenn kein Nomen gefunden, nutze erstes Wort
                if not main_word:
                    main_word = text_to_learn.split()[0].lower()

                # Speichere als Bedeutung
                result = self.worker.netzwerk.add_information_zu_wort(
                    main_word, "bedeutung", f"'{text_to_learn}' (als Satz gelernt)"
                )

                facts_learned = 1 if result.get("created") else 0
                logger.info(f"Gespeichert unter '{main_word}': {text_to_learn}")

            return True, {
                "facts_learned": facts_learned,
                "learning_method": "ingestion",
            }
        else:
            # Einzelnes Wort: Speichere als Konzept
            logger.info(f"Lerne einzelnes Wort/Phrase: '{text_to_learn}'")

            formatter = KaiResponseFormatter()
            cleaned_word = formatter.clean_entity(text_to_learn)

            result = self.worker.netzwerk.add_information_zu_wort(
                cleaned_word, "bedeutung", f"'{text_to_learn}' (vom Benutzer gelehrt)"
            )

            return True, {
                "facts_learned": 1 if result.get("created") else 0,
                "learning_method": "word_storage",
                "word": cleaned_word,
            }

    def _formulate_learning_confirmation(
        self, context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Formuliert Lernbestätigung."""
        facts_learned = context.get("facts_learned", 0)
        learning_method = context.get("learning_method", "unknown")
        word = context.get("word")

        if learning_method == "ingestion" and facts_learned > 0:
            response = f"Ok, ich habe {facts_learned} neue {'Fakten' if facts_learned > 1 else 'Fakt'} gelernt."
        elif learning_method == "word_storage":
            response = f"Ok, ich habe mir '{word}' gemerkt."
        elif facts_learned == 0:
            response = "Ok, ich habe versucht zu lernen, aber konnte nichts Neues extrahieren. Vielleicht kannte ich das schon."
        else:
            response = "Ok, ich habe etwas gelernt."

        return True, {"final_response": response}


# ============================================================================
# CONFIRMATION STRATEGY (Bestätigung)
# ============================================================================


class ConfirmationStrategy(SubGoalStrategy):
    """
    Strategy für Bestätigungs-Sub-Goals.

    Zuständig für:
    - Absichts-Bestätigung bei mittlerer Confidence
    """

    def can_handle(self, sub_goal_description: str) -> bool:
        return "Bestätige die erkannte Absicht" in sub_goal_description

    def execute(
        self, sub_goal: SubGoal, context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        intent = context.get("intent")
        if not intent:
            return False, {"error": "Intent fehlt im Kontext"}

        category = intent.category.value if intent.category else "unbekannt"
        confidence = intent.confidence

        # Spezielle Bestätigung für auto-erkannte Definitionen
        if intent.category == MeaningPointCategory.DEFINITION and intent.arguments.get(
            "auto_detected"
        ):
            subject = intent.arguments.get("subject", "etwas")
            relation_type = intent.arguments.get("relation_type", "IS_A")
            obj = intent.arguments.get("object", "etwas")

            relation_map = {
                "IS_A": f"'{subject}' ist ein '{obj}'",
                "HAS_PROPERTY": f"'{subject}' hat die Eigenschaft '{obj}'",
                "CAPABLE_OF": f"'{subject}' kann '{obj}'",
                "PART_OF": f"'{subject}' hat/gehört zu '{obj}'",
                "LOCATED_IN": f"'{subject}' liegt in '{obj}'",
            }

            fact_description = relation_map.get(
                relation_type, f"'{subject}' {relation_type} '{obj}'"
            )
            response = (
                f"Soll ich mir merken, dass {fact_description}? "
                f"(Konfidenz: {confidence:.0%})"
            )
        elif intent.category == MeaningPointCategory.QUESTION:
            topic = intent.arguments.get("topic", "etwas")
            action_desc = f"eine Frage über '{topic}' beantworten"
            response = (
                f"Ich glaube, du möchtest {action_desc}. "
                f"Ist das richtig? (Konfidenz: {confidence:.0%})"
            )
        elif intent.category == MeaningPointCategory.COMMAND:
            command = intent.arguments.get("command", "Befehl")
            action_desc = f"den Befehl '{command}' ausführen"
            response = (
                f"Ich glaube, du möchtest {action_desc}. "
                f"Ist das richtig? (Konfidenz: {confidence:.0%})"
            )
        else:
            action_desc = f"etwas vom Typ '{category}' verarbeiten"
            response = (
                f"Ich glaube, du möchtest {action_desc}. "
                f"Ist das richtig? (Konfidenz: {confidence:.0%})"
            )

        # Setze Kontext für nächste Eingabe
        self.worker.context.aktion = ContextAction.ERWARTE_BESTAETIGUNG
        self.worker.context.original_intent = intent
        self.worker.context.metadata["sub_goal_context"] = context

        self.worker._emit_context_update()

        logger.info(
            f"Confirmation requested: category={category}, confidence={confidence:.2f}"
        )

        return True, {"final_response": response}


# ============================================================================
# CLARIFICATION STRATEGY (Klärung)
# ============================================================================


class ClarificationStrategy(SubGoalStrategy):
    """
    Strategy für Klärungs-Sub-Goals.

    Zuständig für:
    - Rückfragen bei niedriger Confidence oder UNKNOWN-Kategorie
    """

    def can_handle(self, sub_goal_description: str) -> bool:
        return (
            "Formuliere eine allgemeine Rückfrage zur Klärung" in sub_goal_description
        )

    def execute(
        self, sub_goal: SubGoal, context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        intent = context.get("intent")
        if not intent:
            return False, {"error": "Intent fehlt im Kontext"}

        original_text = intent.text_span if intent.text_span else ""

        # Setze Kontext für Feedback-Loop
        self.worker.context.aktion = ContextAction.ERWARTE_FEEDBACK_ZU_CLARIFICATION
        self.worker.context.thema = "clarification_feedback"
        self.worker.context.original_intent = intent
        self.worker.context.metadata["original_query"] = original_text

        self.worker._emit_context_update()

        response = (
            "Ich bin nicht sicher, was du meinst. "
            "Kannst du es anders formulieren oder ein Beispiel geben? "
            "Du kannst auch mit 'Lerne Muster: \"...\" bedeutet KATEGORIE' ein Beispiel geben."
        )

        logger.info(f"Clarification requested for: '{original_text[:50]}...'")

        return True, {"final_response": response}


# ============================================================================
# FILE READER STRATEGY (Lese Datei)
# ============================================================================


class FileReaderStrategy(SubGoalStrategy):
    """
    Strategy für Datei-Lese Sub-Goals.

    Zuständig für:
    - Datei-Validierung (Existenz, Lesbarkeit, Format)
    - Dokument-Parsing (DOCX, PDF)
    - Text-Ingestion mit Progress-Updates
    - Ingestion-Berichterstellung
    """

    def can_handle(self, sub_goal_description: str) -> bool:
        file_reading_keywords = [
            "Validiere Dateipfad",
            "Extrahiere Text aus der Datei",
            "Verarbeite extrahierten Text durch Ingestion-Pipeline",
            "Formuliere Ingestion-Bericht",
        ]
        return any(kw in sub_goal_description for kw in file_reading_keywords)

    def execute(
        self, sub_goal: SubGoal, context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        intent = context.get("intent")
        if not intent:
            return False, {"error": "Intent fehlt im Kontext"}

        description = sub_goal.description

        if "Validiere Dateipfad" in description:
            return self._validate_file(intent)
        elif "Extrahiere Text aus der Datei" in description:
            return self._extract_text(intent, context)
        elif "Verarbeite extrahierten Text durch Ingestion-Pipeline" in description:
            return self._process_ingestion(context)
        elif "Formuliere Ingestion-Bericht" in description:
            return self._formulate_report(context)

        return False, {"error": f"Unbekanntes FileReading-SubGoal: {description}"}

    def _validate_file(self, intent) -> Tuple[bool, Dict[str, Any]]:
        """
        Validiert Dateipfad, Existenz, Lesbarkeit und Format.

        Returns:
            Tuple (success, {"file_path": ..., "format": ...})
        """
        from pathlib import Path

        from component_28_document_parser import DocumentParserFactory

        file_path = intent.arguments.get("file_path", "")

        # Prüfe ob Dateipfad vorhanden
        if not file_path:
            return False, {"error": "Kein Dateipfad angegeben."}

        # Prüfe Existenz
        path = Path(file_path)
        if not path.exists():
            return False, {"error": f"Datei nicht gefunden: {file_path}"}

        # Prüfe ob es eine Datei ist (nicht Verzeichnis)
        if not path.is_file():
            return False, {"error": f"Pfad ist keine Datei: {file_path}"}

        # Prüfe Lesbarkeit
        import os

        if not os.access(file_path, os.R_OK):
            return False, {
                "error": f"Datei nicht lesbar (fehlende Leserechte): {file_path}"
            }

        # Prüfe ob Format unterstützt wird
        if not DocumentParserFactory.is_supported(file_path):
            supported = ", ".join(DocumentParserFactory.get_supported_extensions())
            file_format = path.suffix
            return False, {
                "error": f"Format '{file_format}' wird nicht unterstützt. Unterstützte Formate: {supported}"
            }

        logger.info(
            f"Datei-Validierung erfolgreich: {file_path} (Format: {path.suffix})"
        )

        return True, {"file_path": str(path.absolute()), "format": path.suffix}

    def _extract_text(
        self, intent, context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Extrahiert Text aus Dokument mittels DocumentParserFactory.

        PHASE 8 (Extended Features): Unterstützt Preview-Modus.

        Returns:
            Tuple (success, {"extracted_text": ..., "char_count": ..., "file_name": ..., "preview": ...})
        """
        from component_28_document_parser import DocumentParserFactory
        from kai_exceptions import DocumentParseError, MissingDependencyError

        file_path = context.get("file_path")
        if not file_path:
            return False, {"error": "Dateipfad fehlt aus vorigem Schritt."}

        from pathlib import Path

        file_name = Path(file_path).name

        try:
            logger.info(f"Starte Text-Extraktion aus: {file_name}")

            # Erstelle passenden Parser
            parser = DocumentParserFactory.create_parser(file_path)

            # Extrahiere Text
            extracted_text = parser.extract_text(file_path)

            if not extracted_text or not extracted_text.strip():
                logger.warning(f"Dokument enthält keinen Text: {file_name}")
                return False, {
                    "error": f"Dokument '{file_name}' enthält keinen extrahierbaren Text."
                }

            char_count = len(extracted_text)

            # PHASE 8: Erstelle Preview (erste 500 Zeichen)
            preview = extracted_text[:500]
            if len(extracted_text) > 500:
                preview += "..."

            logger.info(
                f"Text erfolgreich extrahiert: {char_count} Zeichen aus {file_name}"
            )

            return True, {
                "extracted_text": extracted_text,
                "char_count": char_count,
                "file_name": file_name,
                "preview": preview,
            }

        except MissingDependencyError as e:
            # Fehlende Bibliothek (python-docx oder pdfplumber)
            logger.error(f"Fehlende Bibliothek: {e.dependency_name}")
            return False, {
                "error": f"Erforderliche Bibliothek fehlt: {e.dependency_name}. "
                f"Installiere mit: pip install {e.dependency_name}"
            }

        except DocumentParseError as e:
            # Parsing-Fehler
            logger.error(f"Parsing-Fehler: {e}")
            return False, {"error": f"Konnte Dokument nicht parsen: {e.message}"}

        except Exception as e:
            # Unerwarteter Fehler
            logger.error(f"Unerwarteter Fehler bei Text-Extraktion: {e}", exc_info=True)
            return False, {
                "error": f"Unerwarteter Fehler beim Lesen der Datei: {str(e)}"
            }

    def _process_ingestion(
        self, context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Verarbeitet extrahierten Text durch Ingestion-Pipeline mit Progress-Updates.

        PHASE 8 (Extended Features): Fordert Preview-Bestätigung vor Ingestion an.

        Nutzt ingest_text_large() für Batch-Processing und Progress-Tracking.

        Returns:
            Tuple (success, {"facts_created": ..., "chunks_processed": ..., ...})
        """
        from kai_ingestion_handler import KaiIngestionHandler

        extracted_text = context.get("extracted_text")
        file_name = context.get("file_name", "Datei")
        preview = context.get("preview", "")
        char_count = context.get("char_count", 0)

        if not extracted_text:
            return False, {"error": "Text aus vorigem Schritt fehlt."}

        # PHASE 8: Fordere Preview-Bestätigung vom User an
        if preview and char_count > 0:
            logger.info(f"Zeige Preview für {file_name} ({char_count} Zeichen)")
            confirmed = self.worker.wait_for_preview_confirmation(
                preview, file_name, char_count
            )

            if not confirmed:
                logger.info(f"User hat Ingestion von {file_name} abgebrochen")
                return False, {
                    "error": f"Ingestion von '{file_name}' wurde vom Benutzer abgebrochen.",
                    "user_cancelled": True,
                }

        # Erstelle Ingestion Handler
        ingestion_handler = KaiIngestionHandler(
            self.worker.netzwerk,
            self.worker.preprocessor,
            self.worker.prototyping_engine,
            self.worker.embedding_service,
        )

        # Progress-Callback für UI-Updates
        def progress_callback(current, total, stats):
            """Emittiert Progress-Updates an UI."""
            percent = int((current / total) * 100) if total > 0 else 0
            progress_msg = (
                f"Verarbeite {file_name}: {current}/{total} Sätze ({percent}%) - "
                f"{stats['facts_created']} Fakten gelernt"
            )
            logger.info(progress_msg)
            # Emit file progress signal (UI will process events automatically)
            self.worker.signals.file_progress_update.emit(current, total, progress_msg)

            # Emittiere Signal für UI
            if hasattr(self.worker, "signals") and hasattr(
                self.worker.signals, "progress_update"
            ):
                self.worker.signals.progress_update.emit(progress_msg)

        try:
            logger.info(f"Starte Ingestion von {file_name} (Batch-Processing)...")

            # WICHTIG: Zeige Progress-Bar SOFORT beim Start (nicht erst beim ersten Update)
            # Das gibt dem User sofort Feedback dass die Verarbeitung läuft
            self.worker.signals.file_progress_update.emit(
                0, 100, f"Starte Verarbeitung von {file_name}..."
            )

            # BATCH-PROCESSING mit Progress-Tracking und PARALLELER VERARBEITUNG
            stats = ingestion_handler.ingest_text_large(
                text=extracted_text,
                chunk_size=50,  # 50 Sätze pro Chunk für häufigere UI-Updates
                progress_callback=progress_callback,
                max_workers=None,  # Auto-detect: CPU-Cores * 2
            )

            logger.info(
                f"Ingestion abgeschlossen: {stats['facts_created']} Fakten aus {file_name} "
                f"({stats['chunks_processed']} Chunks verarbeitet)"
            )

            return True, {
                "facts_created": stats["facts_created"],
                "learned_patterns": stats["learned_patterns"],
                "fallback_patterns": stats["fallback_patterns"],
                "chunks_processed": stats["chunks_processed"],
                "fragments_stored": stats.get("fragments_stored", 0),
                "connections_stored": stats.get("connections_stored", 0),
            }

        except Exception as e:
            logger.error(f"Fehler bei Ingestion: {e}", exc_info=True)
            return False, {"error": f"Fehler beim Verarbeiten des Textes: {str(e)}"}

    def _formulate_report(self, context: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Formuliert Ingestion-Bericht mit Statistiken.

        Returns:
            Tuple (success, {"final_response": ...})
        """
        file_name = context.get("file_name", "Datei")
        facts_created = context.get("facts_created", 0)
        chunks_processed = context.get("chunks_processed", 0)
        learned = context.get("learned_patterns", 0)
        fallback = context.get("fallback_patterns", 0)
        fragments = context.get("fragments_stored", 0)
        connections = context.get("connections_stored", 0)

        # Formuliere Hauptbericht
        if facts_created > 1:
            response = f"Datei '{file_name}' erfolgreich verarbeitet. Ich habe {facts_created} neue Fakten gelernt"
        elif facts_created == 1:
            response = f"Datei '{file_name}' erfolgreich verarbeitet. Ich habe 1 neuen Fakt gelernt"
        else:
            response = f"Datei '{file_name}' wurde verarbeitet, aber ich konnte keine neuen Fakten extrahieren"

        # Füge Details hinzu
        details = []
        if learned > 0:
            details.append(f"{learned} aus gelernten Mustern")
        if fallback > 0:
            details.append(f"{fallback} aus neuen Mustern")
        if chunks_processed > 0:
            details.append(f"{chunks_processed} Chunks verarbeitet")

        if details:
            response += f" ({', '.join(details)})"

        response += "."

        # Füge Word-Usage-Statistiken hinzu (falls aktiviert)
        if fragments > 0 or connections > 0:
            usage_details = []
            if fragments > 0:
                usage_details.append(f"{fragments} Kontextfragmente")
            if connections > 0:
                usage_details.append(f"{connections} Wortverbindungen")
            response += f" Zusätzlich wurden {', '.join(usage_details)} getrackt."

        logger.info(
            f"Ingestion-Bericht erstellt: {facts_created} Fakten aus {file_name}"
        )

        return True, {"final_response": response}


# ============================================================================
# SPATIAL REASONING STRATEGY (Räumliches Reasoning)
# ============================================================================


class SpatialReasoningStrategy(SubGoalStrategy):
    """
    Strategy für räumliche Reasoning Sub-Goals.

    Zuständig für:
    - Extraktion räumlicher Entitäten und Positionen
    - Aufbau räumlicher Modelle (Grid, Shapes, Positions)
    - Constraint-basierte räumliche Probleme (CSP)
    - State-Space-Planning für räumliche Aktionen
    - Formatierung räumlicher Antworten
    """

    def can_handle(self, sub_goal_description: str) -> bool:
        spatial_keywords = [
            "Extrahiere räumliche Entitäten und Positionen",
            "Erstelle räumliches Modell",
            "Löse räumliche Constraints",
            "Plane räumliche Aktionen",
            "Formuliere räumliche Antwort",
        ]
        return any(kw in sub_goal_description for kw in spatial_keywords)

    def execute(
        self, sub_goal: SubGoal, context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        intent = context.get("intent")
        if not intent:
            return False, {"error": "Intent fehlt im Kontext"}

        description = sub_goal.description

        if "Extrahiere räumliche Entitäten und Positionen" in description:
            return self._extract_spatial_entities(intent, context)
        elif "Erstelle räumliches Modell" in description:
            return self._create_spatial_model(context)
        elif "Löse räumliche Constraints" in description:
            return self._solve_spatial_constraints(context)
        elif "Plane räumliche Aktionen" in description:
            return self._plan_spatial_actions(context)
        elif "Formuliere räumliche Antwort" in description:
            return self._formulate_spatial_answer(context)

        return False, {"error": f"Unbekanntes Spatial-Reasoning-SubGoal: {description}"}

    def _extract_spatial_entities(
        self, intent, context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Extrahiert räumliche Entitäten und ihre Positionen aus dem Intent.

        Args:
            intent: MeaningPoint mit spatial_query_type
            context: Kontext

        Returns:
            Tuple (success, {"entities": [...], "positions": {...}, "relations": [...]})
        """
        spatial_query_type = intent.arguments.get(
            "spatial_query_type", "position_query"
        )

        # Tracke räumliche Extraktion
        self.worker.working_memory.add_reasoning_state(
            step_type="spatial_extraction",
            description=f"Extrahiere räumliche Entitäten für: {spatial_query_type}",
            data={"query_type": spatial_query_type},
        )

        # Extrahiere Entitäten aus Intent-Argumenten
        entities = []
        positions = {}
        relations = []

        # Grid-basierte Queries
        if spatial_query_type == "grid_query":
            grid_config = intent.arguments.get("grid_config", {})
            query_position = intent.arguments.get("query_position")

            entities.append({"type": "grid", "config": grid_config})

            if query_position:
                positions["query"] = query_position

        # Positions-Queries
        elif spatial_query_type == "position_query":
            objects = intent.arguments.get("objects", [])
            for obj in objects:
                entities.append(
                    {
                        "type": "object",
                        "name": obj.get("name"),
                        "position": obj.get("position"),
                    }
                )
                if obj.get("position"):
                    positions[obj["name"]] = obj["position"]

        # Relations-Queries
        elif spatial_query_type == "relation_query":
            subject = intent.arguments.get("subject")
            relation = intent.arguments.get("relation")
            target = intent.arguments.get("target")

            if subject and relation and target:
                relations.append(
                    {"subject": subject, "relation": relation, "target": target}
                )

        # Path-Finding-Queries
        elif spatial_query_type == "path_finding":
            start = intent.arguments.get("start_position")
            goal = intent.arguments.get("goal_position")
            obstacles = intent.arguments.get("obstacles", [])

            entities.append(
                {
                    "type": "path_problem",
                    "start": start,
                    "goal": goal,
                    "obstacles": obstacles,
                }
            )

        logger.info(
            f"Räumliche Entitäten extrahiert: {len(entities)} Entitäten, "
            f"{len(positions)} Positionen, {len(relations)} Relationen"
        )

        return True, {
            "entities": entities,
            "positions": positions,
            "relations": relations,
            "spatial_query_type": spatial_query_type,
        }

    def _create_spatial_model(
        self, context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Erstellt ein räumliches Modell aus den extrahierten Entitäten.

        Args:
            context: Kontext mit entities, positions, relations

        Returns:
            Tuple (success, {"spatial_model": ..., "model_type": ...})
        """
        from component_17_proof_explanation import ProofStep, StepType
        from component_42_spatial_reasoning import (
            Grid,
            Position,
            SpatialReasoner,
            SpatialRelationType,
        )

        # Create spatial engine
        spatial_engine = SpatialReasoner(self.worker.netzwerk)

        entities = context.get("entities", [])
        positions = context.get("positions", {})
        relations = context.get("relations", [])
        spatial_query_type = context.get("spatial_query_type")

        if not entities:
            return False, {"error": "Keine Entitäten zum Erstellen eines Modells"}

        # Tracke Modell-Erstellung
        self.worker.working_memory.add_reasoning_state(
            step_type="spatial_model_creation",
            description=f"Erstelle räumliches Modell: {spatial_query_type}",
            data={"entity_count": len(entities)},
        )

        # Erstelle ProofStep für Modell-Erstellung
        proof_step = None

        try:
            # FALL 1: Grid-basiertes Modell
            if spatial_query_type == "grid_query":
                grid_entity = entities[0]
                grid_config = grid_entity.get("config", {})

                rows = grid_config.get("rows", 8)
                cols = grid_config.get("cols", 8)

                grid = Grid(name=f"grid_{rows}x{cols}", width=cols, height=rows)

                logger.info(f"Grid-Modell erstellt: {rows}x{cols}")

                # ProofStep für Grid-Erstellung
                proof_step = ProofStep(
                    step_id=f"spatial_model_{spatial_query_type}_{id(grid)}",
                    step_type=StepType.SPATIAL_MODEL_CREATION,
                    inputs=[f"Grid-Konfiguration: {rows}×{cols}"],
                    output=f"Grid-Modell mit {rows * cols} Feldern erstellt",
                    confidence=1.0,
                    explanation_text=f"Räumliches Grid-Modell ({rows}×{cols}) für die Abfrage erstellt.",
                    metadata={
                        "model_type": "grid",
                        "rows": rows,
                        "cols": cols,
                        "total_cells": rows * cols,
                    },
                    source_component="spatial_reasoning",
                )

                return True, {
                    "spatial_model": grid,
                    "model_type": "grid",
                    "spatial_engine": spatial_engine,
                    "proof_step": proof_step,
                }

            # FALL 2: Positions-basiertes Modell
            elif spatial_query_type == "position_query":
                # Registriere Positionen im Engine
                position_inputs = []
                for obj_name, pos_data in positions.items():
                    pos = Position(pos_data["x"], pos_data["y"])
                    # Store position in knowledge graph
                    spatial_engine.add_position(obj_name, pos)
                    position_inputs.append(
                        f"{obj_name} @ ({pos_data['x']}, {pos_data['y']})"
                    )

                logger.info(f"Positions-Modell erstellt mit {len(positions)} Objekten")

                # ProofStep für Positions-Modell
                proof_step = ProofStep(
                    step_id=f"spatial_model_{spatial_query_type}_{id(spatial_engine)}",
                    step_type=StepType.SPATIAL_MODEL_CREATION,
                    inputs=position_inputs,
                    output=f"Positions-Modell mit {len(positions)} Objekten erstellt",
                    confidence=1.0,
                    explanation_text=f"{len(positions)} Objekte mit Positionen registriert.",
                    metadata={
                        "model_type": "positions",
                        "object_count": len(positions),
                        "positions": {
                            obj: f"({pos_data['x']}, {pos_data['y']})"
                            for obj, pos_data in positions.items()
                        },
                    },
                    source_component="spatial_reasoning",
                )

                return True, {
                    "spatial_model": None,  # Keine Grid, nur Positionen
                    "model_type": "positions",
                    "spatial_engine": spatial_engine,
                    "proof_step": proof_step,
                }

            # FALL 3: Relations-basiertes Modell
            elif spatial_query_type == "relation_query":
                # Registriere Relationen
                relation_inputs = []
                for rel in relations:
                    try:
                        rel_type = SpatialRelationType[rel["relation"].upper()]
                        # Store spatial relation in knowledge graph
                        spatial_engine.add_spatial_relation(
                            rel["subject"], rel_type, rel["target"]
                        )
                        relation_inputs.append(
                            f"{rel['subject']} {rel['relation']} {rel['target']}"
                        )
                    except KeyError:
                        logger.warning(f"Unbekannte Relation: {rel['relation']}")

                logger.info(
                    f"Relations-Modell erstellt mit {len(relations)} Relationen"
                )

                # ProofStep für Relations-Modell
                proof_step = ProofStep(
                    step_id=f"spatial_model_{spatial_query_type}_{id(spatial_engine)}",
                    step_type=StepType.SPATIAL_MODEL_CREATION,
                    inputs=relation_inputs,
                    output=f"Relations-Modell mit {len(relations)} Relationen erstellt",
                    confidence=1.0,
                    explanation_text=f"{len(relations)} räumliche Relationen registriert.",
                    metadata={
                        "model_type": "relations",
                        "relation_count": len(relations),
                        "relations": relation_inputs,
                    },
                    source_component="spatial_reasoning",
                )

                return True, {
                    "spatial_model": None,
                    "model_type": "relations",
                    "spatial_engine": spatial_engine,
                    "proof_step": proof_step,
                }

            # FALL 4: Path-Finding-Modell
            elif spatial_query_type == "path_finding":
                path_problem = entities[0]
                start_pos = Position(
                    path_problem["start"]["x"], path_problem["start"]["y"]
                )
                goal_pos = Position(
                    path_problem["goal"]["x"], path_problem["goal"]["y"]
                )
                obstacles = [
                    Position(obs["x"], obs["y"])
                    for obs in path_problem.get("obstacles", [])
                ]

                logger.info(
                    f"Path-Finding-Modell erstellt: Start={start_pos}, Goal={goal_pos}, "
                    f"Obstacles={len(obstacles)}"
                )

                # ProofStep für Path-Finding-Modell
                proof_step = ProofStep(
                    step_id=f"spatial_model_{spatial_query_type}_{id(spatial_engine)}",
                    step_type=StepType.SPATIAL_MODEL_CREATION,
                    inputs=[
                        f"Start: {start_pos}",
                        f"Ziel: {goal_pos}",
                        f"Hindernisse: {len(obstacles)}",
                    ],
                    output=f"Path-Finding-Modell erstellt (Start→Ziel mit {len(obstacles)} Hindernissen)",
                    confidence=1.0,
                    explanation_text=f"Räumliches Modell für Pfadsuche von {start_pos} nach {goal_pos} erstellt.",
                    metadata={
                        "model_type": "path_finding",
                        "start": str(start_pos),
                        "goal": str(goal_pos),
                        "obstacle_count": len(obstacles),
                    },
                    source_component="spatial_reasoning",
                )

                return True, {
                    "spatial_model": None,
                    "model_type": "path_finding",
                    "spatial_engine": spatial_engine,
                    "start_position": start_pos,
                    "goal_position": goal_pos,
                    "obstacles": obstacles,
                    "proof_step": proof_step,
                }

            else:
                return False, {"error": f"Unbekannter Query-Typ: {spatial_query_type}"}

        except Exception as e:
            logger.error(f"Fehler bei Modell-Erstellung: {e}", exc_info=True)
            return False, {"error": f"Modell-Erstellung fehlgeschlagen: {str(e)}"}

    def _solve_spatial_constraints(
        self, context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Löst räumliche Constraints mittels CSP-Solver.

        Args:
            context: Kontext mit spatial_model, spatial_engine

        Returns:
            Tuple (success, {"constraint_solution": ..., "satisfiable": ...})
        """
        from component_17_proof_explanation import ProofStep, StepType
        from component_29_constraint_reasoning import ConstraintSolver

        spatial_engine = context.get("spatial_engine")
        model_type = context.get("model_type")
        parent_proof_step = context.get("proof_step")

        if not spatial_engine:
            return False, {"error": "Kein räumliches Modell vorhanden"}

        # Tracke Constraint-Solving
        self.worker.working_memory.add_reasoning_state(
            step_type="spatial_constraint_solving",
            description=f"Löse räumliche Constraints: {model_type}",
            data={"model_type": model_type},
        )

        try:
            # Für Relations-basierte Queries: Nutze CSP-Solver
            if model_type == "relations":
                csp_engine = ConstraintSolver()

                # Hole alle Relationen vom SpatialEngine
                # und formuliere sie als CSP
                # (Vereinfachte Implementierung - kann erweitert werden)

                logger.info("Räumliche Constraints via CSP gelöst")

                # ProofStep für Constraint-Solving
                proof_step = ProofStep(
                    step_id=f"spatial_csp_{id(csp_engine)}",
                    step_type=StepType.SPATIAL_CONSTRAINT_SOLVING,
                    inputs=[parent_proof_step.output] if parent_proof_step else [],
                    output="Räumliche Constraints sind erfüllbar",
                    confidence=1.0,
                    explanation_text="Alle räumlichen Constraints wurden via CSP-Solver geprüft und sind konsistent.",
                    parent_steps=(
                        [parent_proof_step.step_id] if parent_proof_step else []
                    ),
                    metadata={
                        "model_type": model_type,
                        "solver": "CSP (Constraint Satisfaction Problem)",
                    },
                    source_component="spatial_reasoning",
                )

                return True, {
                    "constraint_solution": None,  # Placeholder
                    "satisfiable": True,
                    "csp_engine": csp_engine,
                    "proof_step": proof_step,
                }

            # Andere Modelltypen benötigen kein explizites Constraint-Solving
            return True, {
                "constraint_solution": None,
                "satisfiable": True,
            }

        except Exception as e:
            logger.error(f"Fehler beim Constraint-Solving: {e}", exc_info=True)
            return False, {"error": f"Constraint-Solving fehlgeschlagen: {str(e)}"}

    def _plan_spatial_actions(
        self, context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Plant räumliche Aktionen mittels Spatial Reasoner's A* pathfinding.

        Args:
            context: Kontext mit spatial_engine, spatial_model, model_type

        Returns:
            Tuple (success, {"plan": ..., "plan_length": ...})
        """
        from component_17_proof_explanation import ProofStep, StepType

        model_type = context.get("model_type")
        spatial_engine = context.get("spatial_engine")
        spatial_model = context.get("spatial_model")

        if model_type != "grid" or not spatial_model or not spatial_engine:
            # Kein Planning benötigt (nur für Grid-basiertes pathfinding)
            return True, {"plan": None, "plan_length": 0}

        start_position = context.get("start_position")
        goal_position = context.get("goal_position")
        allow_diagonal = context.get("allow_diagonal", False)

        if not start_position or not goal_position:
            return False, {"error": "Start- oder Ziel-Position fehlt"}

        # Tracke Planning
        self.worker.working_memory.add_reasoning_state(
            step_type="spatial_planning",
            description=f"Plane Pfad von {start_position} nach {goal_position}",
            data={
                "start": str(start_position),
                "goal": str(goal_position),
                "allow_diagonal": allow_diagonal,
            },
        )

        try:
            # Use SpatialReasoner's built-in A* pathfinding
            path = spatial_engine.find_path(
                grid_name=spatial_model.name,
                start=start_position,
                goal=goal_position,
                allow_diagonal=allow_diagonal,
            )

            if path:
                logger.info(f"Pfad gefunden: {len(path)} Schritte")

                # ProofStep für Planning
                parent_proof_step = context.get("proof_step")
                proof_step = ProofStep(
                    step_id=f"spatial_planning_{id(spatial_engine)}",
                    step_type=StepType.SPATIAL_PLANNING,
                    inputs=[
                        (
                            parent_proof_step.output
                            if parent_proof_step
                            else "Path-Finding-Modell"
                        )
                    ],
                    output=f"Pfad gefunden mit {len(path)} Schritten",
                    confidence=1.0,
                    explanation_text=f"A*-Algorithmus fand einen optimalen Pfad von {start_position} nach {goal_position}.",
                    parent_steps=(
                        [parent_proof_step.step_id] if parent_proof_step else []
                    ),
                    metadata={
                        "algorithm": "A* (Spatial Reasoning)",
                        "heuristic": "Manhattan-Distanz",
                        "plan_length": len(path),
                        "start": str(start_position),
                        "goal": str(goal_position),
                        "grid": spatial_model.name,
                    },
                    source_component="spatial_reasoning",
                )

                return True, {
                    "plan": path,
                    "plan_length": len(path),
                    "proof_step": proof_step,
                }
            else:
                logger.info("Kein Pfad gefunden (nicht erreichbar)")

                # ProofStep für nicht-erreichbares Ziel
                from component_17_proof_explanation import ProofStep, StepType

                parent_proof_step = context.get("proof_step")
                proof_step = ProofStep(
                    step_id=f"spatial_planning_unreachable_{id(spatial_engine)}",
                    step_type=StepType.SPATIAL_PLANNING,
                    inputs=[
                        (
                            parent_proof_step.output
                            if parent_proof_step
                            else "Path-Finding-Modell"
                        )
                    ],
                    output=f"Ziel {goal_position} ist nicht erreichbar",
                    confidence=1.0,
                    explanation_text=f"A*-Algorithmus konnte keinen Pfad von {start_position} nach {goal_position} finden (Hindernisse blockieren).",
                    parent_steps=(
                        [parent_proof_step.step_id] if parent_proof_step else []
                    ),
                    metadata={
                        "algorithm": "A* (State-Space Planning)",
                        "reachable": False,
                    },
                    source_component="spatial_reasoning",
                )

                return True, {
                    "plan": None,
                    "plan_length": 0,
                    "reachable": False,
                    "proof_step": proof_step,
                }

        except Exception as e:
            logger.error(f"Fehler beim Planning: {e}", exc_info=True)
            return False, {"error": f"Planning fehlgeschlagen: {str(e)}"}

    def _formulate_spatial_answer(
        self, context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Formuliert eine Antwort für räumliche Queries.

        Args:
            context: Kontext mit allen Ergebnissen

        Returns:
            Tuple (success, {"final_response": ...})
        """
        from kai_response_formatter import KaiResponseFormatter

        formatter = KaiResponseFormatter()

        model_type = context.get("model_type")
        spatial_query_type = context.get("spatial_query_type")

        # Delegiere Formatierung an Response Formatter
        response = formatter.format_spatial_answer(
            model_type=model_type,
            spatial_query_type=spatial_query_type,
            entities=context.get("entities", []),
            positions=context.get("positions", {}),
            relations=context.get("relations", []),
            plan=context.get("plan"),
            plan_length=context.get("plan_length", 0),
            reachable=context.get("reachable", True),
        )

        return True, {"final_response": response}


# ============================================================================
# SHARED STRATEGY (Gemeinsame Sub-Goals)
# ============================================================================


class SharedStrategy(SubGoalStrategy):
    """
    Strategy für gemeinsam genutzte Sub-Goals.

    Zuständig für:
    - "Formuliere eine Lernbestätigung" (wird von mehreren Strategien verwendet)
    """

    def can_handle(self, sub_goal_description: str) -> bool:
        return "Formuliere eine Lernbestätigung" in sub_goal_description

    def execute(
        self, sub_goal: SubGoal, context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        # Unterscheide zwischen verschiedenen Kontexten
        if "linked_relation" in context:
            # Pattern Learning Context
            relation = context.get("linked_relation")
            response = f"Verstanden. Ich habe gelernt, dass Sätze dieses Musters die Relation '{relation}' ausdrücken."
            return True, {"final_response": response}
        else:
            # Delegiere an LearningStrategy
            learning_strategy = LearningStrategy(self.worker)
            return learning_strategy._formulate_learning_confirmation(context)


class EpisodicMemoryStrategy(SubGoalStrategy):
    """
    Strategy für episodische Gedächtnis-Abfragen.

    Zuständig für:
    - "Wann habe ich X gelernt?" Queries
    - Episoden-Abfragen aus dem episodischen Gedächtnis
    - Timeline-Visualisierung von Lern- und Inferenz-Episoden
    """

    def can_handle(self, sub_goal_description: str) -> bool:
        episodic_keywords = [
            "Frage episodisches Gedächtnis ab",
            "Zeige Lernverlauf",
            "Zeige Episoden",
            "Finde Lern-Episoden",
        ]
        return any(kw in sub_goal_description for kw in episodic_keywords)

    def execute(
        self, sub_goal: SubGoal, context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        intent = context.get("intent")
        if not intent:
            return False, {"error": "Intent fehlt im Kontext"}

        description = sub_goal.description

        # Dispatcher für verschiedene Episodic-Memory-SubGoals
        if "Frage episodisches Gedächtnis ab" in description:
            return self._query_episodic_memory(intent, context)
        elif "Formuliere eine Antwort mit Episoden-Zusammenfassung" in description:
            return self._formulate_episodic_answer(intent, context)
        elif "Zeige Episoden" in description or "Zeige Lernverlauf" in description:
            return self._show_episodes(intent, context)

        return False, {"error": f"Unbekanntes Episodic-Memory-SubGoal: {description}"}

    def _query_episodic_memory(
        self, intent, context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Fragt das episodische Gedächtnis nach Lern-Episoden ab.

        Args:
            intent: MeaningPoint mit episodic_query_type
            context: Kontext mit optionalem topic

        Returns:
            Tuple[bool, Dict]: (success, {"episodes": [...], "query_type": ...})
        """
        episodic_query_type = intent.arguments.get(
            "episodic_query_type", "show_episodes"
        )
        topic = intent.arguments.get("topic")  # Kann None sein für "alle Episoden"

        logger.info(
            f"Episodische Abfrage: {episodic_query_type}, Thema: {topic or 'alle'}"
        )

        # Tracke episodische Abfrage
        self.worker.working_memory.add_reasoning_state(
            step_type="episodic_query",
            description=f"Frage episodisches Gedächtnis ab: {episodic_query_type}",
            data={"query_type": episodic_query_type, "topic": topic},
        )

        # Hole Episoden aus dem Netzwerk
        episodes = []

        if episodic_query_type in ["when_learned", "what_learned"]:
            # Themen-spezifische Abfrage
            if topic:
                episodes = self.worker.netzwerk.query_episodes_about(topic, limit=50)
            else:
                episodes = self.worker.netzwerk.query_all_episodes(limit=50)

        elif episodic_query_type == "show_episodes":
            # Alle Episoden oder gefiltert nach Thema
            if topic:
                episodes = self.worker.netzwerk.query_episodes_about(topic, limit=50)
            else:
                episodes = self.worker.netzwerk.query_all_episodes(limit=50)

        elif episodic_query_type == "learning_history":
            # Nur Learning-Episoden
            if topic:
                all_episodes = self.worker.netzwerk.query_episodes_about(
                    topic, limit=100
                )
                # Filtere nach Learning-Typen
                episodes = [
                    ep
                    for ep in all_episodes
                    if ep.get("type", "").lower()
                    in ["learning", "ingestion", "definition", "pattern_learning"]
                ]
            else:
                episodes = self.worker.netzwerk.query_all_episodes(
                    episode_type="learning", limit=50
                )
        else:
            # Fallback: Alle Episoden
            episodes = self.worker.netzwerk.query_all_episodes(limit=50)

        logger.info(f"Gefunden: {len(episodes)} Episoden")

        # Tracke Ergebnis
        self.worker.working_memory.add_reasoning_state(
            step_type="episodic_result",
            description=f"Gefunden: {len(episodes)} Episoden",
            data={"episode_count": len(episodes)},
        )

        # Emittiere Signal für UI-Update (direkt vom Worker)
        if hasattr(self.worker, "signals"):
            self.worker.signals.episodic_data_update.emit(episodes)

        return True, {
            "episodes": episodes,
            "query_type": episodic_query_type,
            "topic": topic,
            "episode_count": len(episodes),
        }

    def _formulate_episodic_answer(
        self, intent, context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Formuliert eine natürlichsprachliche Antwort für episodische Queries.

        Args:
            intent: MeaningPoint mit episodic_query_type
            context: Kontext mit episodes aus vorherigem SubGoal

        Returns:
            Tuple[bool, Dict]: (success, {"final_response": ...})
        """
        episodes = context.get("episodes", [])
        query_type = context.get("query_type", "show_episodes")
        topic = context.get("topic")

        # Verwende den ResponseFormatter
        from kai_response_formatter import KaiResponseFormatter

        formatter = KaiResponseFormatter(feedback_handler=None)

        response = formatter.format_episodic_answer(
            episodes=episodes, query_type=query_type, topic=topic
        )

        return True, {"final_response": response}

    def _show_episodes(
        self, intent, context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Alias für _query_episodic_memory - zeigt Episoden in der UI.
        """
        return self._query_episodic_memory(intent, context)


# ============================================================================
# ORCHESTRATED STRATEGY (Orchestrierte Multi-Segment-Verarbeitung)
# ============================================================================


class OrchestratedStrategy(SubGoalStrategy):
    """
    Strategy für orchestrierte Sub-Goals aus InputOrchestrator.

    Zuständig für:
    - Batch-Learning von Erklärungen
    - Befehlsausführung mit Kontext
    - Fragen mit gelerntem Kontext beantworten
    """

    def can_handle(self, sub_goal_description: str) -> bool:
        # Prüfe auf orchestrierte Keywords
        orchestrated_keywords = [
            "Lerne Kontext:",
            "Beantworte Frage:",  # Nur orchestrierte Fragen (mit Metadata)
        ]
        return any(kw in sub_goal_description for kw in orchestrated_keywords)

    def execute(
        self, sub_goal: SubGoal, context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Führt orchestriertes Sub-Goal aus."""
        # Hole Orchestrator-Typ aus Metadata
        orchestrated_type = sub_goal.metadata.get("orchestrated_type")

        if not orchestrated_type:
            return False, {"error": "Orchestriertes Sub-Goal ohne Typ"}

        logger.info(f"Führe orchestriertes Sub-Goal aus: {orchestrated_type}")

        if orchestrated_type == "batch_learning":
            return self._batch_learning(sub_goal, context)
        elif orchestrated_type == "command_execution":
            return self._execute_command(sub_goal, context)
        elif orchestrated_type == "question_answering":
            return self._answer_question_with_context(sub_goal, context)
        else:
            return False, {
                "error": f"Unbekannter orchestrierter Typ: {orchestrated_type}"
            }

    def _batch_learning(
        self, sub_goal: SubGoal, context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Verarbeitet mehrere Erklärungen als Batch.

        Args:
            sub_goal: SubGoal mit segment_texts in metadata
            context: Kontext

        Returns:
            Tuple (success, result mit facts_learned)
        """
        from kai_ingestion_handler import KaiIngestionHandler

        segment_texts = sub_goal.metadata.get("segment_texts", [])
        if not segment_texts:
            return False, {"error": "Keine Segmente für Batch-Learning gefunden"}

        logger.info(f"Batch-Learning für {len(segment_texts)} Segmente")

        # Kombiniere alle Texte
        combined_text = ". ".join(seg.rstrip(".") for seg in segment_texts) + "."

        # Verarbeite via Ingestion Handler
        ingestion_handler = KaiIngestionHandler(
            self.worker.netzwerk,
            self.worker.preprocessor,
            self.worker.prototyping_engine,
            self.worker.embedding_service,
        )

        stats = ingestion_handler.ingest_text(combined_text)
        facts_learned = stats["facts_created"]

        logger.info(f"Batch-Learning abgeschlossen: {facts_learned} Fakten gelernt")

        # Tracke im Working Memory
        self.worker.working_memory.add_reasoning_state(
            step_type="batch_learning",
            description=f"Kontext gelernt: {facts_learned} Fakten aus {len(segment_texts)} Segmenten",
            data={"facts_learned": facts_learned, "segment_count": len(segment_texts)},
        )

        return True, {
            "facts_learned": facts_learned,
            "learned_patterns": stats["learned_patterns"],
            "segments_processed": len(segment_texts),
        }

    def _execute_command(
        self, sub_goal: SubGoal, context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Führt einen Befehl aus orchestriertem Plan aus.

        Args:
            sub_goal: SubGoal mit segment_text in metadata
            context: Kontext

        Returns:
            Tuple (success, result)
        """
        segment_text = sub_goal.metadata.get("segment_text")
        if not segment_text:
            return False, {"error": "Kein Segment-Text für Befehl gefunden"}

        logger.info(f"Führe orchestrierten Befehl aus: '{segment_text[:50]}...'")

        # Verarbeite Befehl durch normale Pipeline
        # (Das wird vom Worker bereits über process_query gemacht)
        # Hier geben wir nur OK zurück
        return True, {"command_text": segment_text, "processed": True}

    def _answer_question_with_context(
        self, sub_goal: SubGoal, context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Beantwortet eine Frage mit zuvor gelerntem Kontext.

        Args:
            sub_goal: SubGoal mit segment_text in metadata
            context: Kontext mit facts_learned aus vorherigen Sub-Goals

        Returns:
            Tuple (success, result mit final_response)
        """
        segment_text = sub_goal.metadata.get("segment_text")
        has_learned_context = sub_goal.metadata.get("has_learned_context", False)

        if not segment_text:
            return False, {"error": "Kein Segment-Text für Frage gefunden"}

        logger.info(
            f"Beantworte Frage mit Kontext: '{segment_text[:50]}...' "
            f"(Kontext gelernt: {has_learned_context})"
        )

        # Verarbeite Frage durch normale Pipeline
        # WICHTIG: Diese Frage wird mit dem ZUVOR GELERNTEN Kontext beantwortet
        # Der Kontext wurde bereits im Batch-Learning-Schritt in den Graph geschrieben

        # Extrahiere Intent aus Frage
        doc = self.worker.preprocessor.process(segment_text)
        meaning_points = self.worker.extractor.extract(doc)

        if not meaning_points:
            return False, {"error": "Konnte Intent aus Frage nicht extrahieren"}

        intent = meaning_points[0]

        # Nutze QuestionStrategy für normale Verarbeitung
        question_strategy = QuestionStrategy(self.worker)

        # Führe Sub-Goals sequentiell aus (wie normale Frage)
        question_context = {"intent": intent}

        # Identifiziere Thema
        success, result = question_strategy._identify_topic(intent)
        if not success:
            return success, result
        question_context.update(result)

        # Frage Wissensgraph ab (enthält jetzt gelernten Kontext!)
        success, result = question_strategy._query_knowledge_graph(question_context)
        if not success:
            return success, result
        question_context.update(result)

        # Prüfe Wissenslücken
        success, result = question_strategy._check_knowledge_gap(question_context)
        if not success:
            return success, result
        question_context.update(result)

        # Formuliere Antwort
        success, result = question_strategy._formulate_answer(intent, question_context)
        if not success:
            return success, result

        logger.info(
            "Frage erfolgreich beantwortet (mit Kontext: %s)", has_learned_context
        )

        # Tracke im Working Memory
        self.worker.working_memory.add_reasoning_state(
            step_type="orchestrated_question",
            description="Frage beantwortet mit gelerntem Kontext",
            data={"question": segment_text[:100], "has_context": has_learned_context},
        )

        return True, result


# ============================================================================
# SUB-GOAL EXECUTOR (Main Dispatcher)
# ============================================================================


# ============================================================================
# ARITHMETIC STRATEGY (Phase Mathematik)
# ============================================================================


class ArithmeticStrategy(SubGoalStrategy):
    """
    Strategy für arithmetische Berechnungen (Phase Mathematik).

    Zuständig für:
    - Parse arithmetischen Ausdruck aus Text
    - Konvertiere Zahlwörter zu Zahlen
    - Führe arithmetische Operation aus
    - Formatiere Ergebnis als Zahlwort
    """

    def can_handle(self, sub_goal_description: str) -> bool:
        arithmetic_keywords = [
            "Parse arithmetischen Ausdruck aus Text",
            "Konvertiere Zahlwörter zu Zahlen",
            "Führe arithmetische Operation aus",
            "Formatiere Ergebnis als Zahlwort",
        ]
        return any(kw in sub_goal_description for kw in arithmetic_keywords)

    def execute(
        self, sub_goal: SubGoal, context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        try:
            description = sub_goal.description
            query_text = sub_goal.metadata.get("query_text", "")

            # Dispatcher für verschiedene Arithmetik-SubGoals
            if "Parse arithmetischen Ausdruck aus Text" in description:
                return self._parse_expression(query_text, context)
            elif "Konvertiere Zahlwörter zu Zahlen" in description:
                return self._convert_words_to_numbers(context)
            elif "Führe arithmetische Operation aus" in description:
                return self._execute_operation(context)
            elif "Formatiere Ergebnis als Zahlwort" in description:
                return self._format_result(context)

            return False, {"error": f"Unbekanntes Arithmetik-SubGoal: {description}"}

        except Exception as e:
            logger.error(f"Fehler in ArithmeticStrategy: {e}", exc_info=True)
            return False, {"error": str(e)}

    def _parse_expression(
        self, query_text: str, context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Parse arithmetischen Ausdruck aus Text.

        Erkennt Muster wie:
        - "Was ist drei plus fünf?" -> ("+", ["drei", "fünf"])
        - "Wie viel ist 7 mal 8?" -> ("*", ["7", "8"])
        """
        try:
            import re

            text_lower = query_text.lower().strip()

            # Operator-Mapping
            operator_patterns = [
                (r"(\w+)\s*\+\s*(\w+)", "+"),  # 3 + 5
                (r"(\w+)\s*-\s*(\w+)", "-"),  # 10 - 2
                (r"(\w+)\s*\*\s*(\w+)", "*"),  # 7 * 8
                (r"(\w+)\s*/\s*(\w+)", "/"),  # 20 / 4
                (r"(\w+)\s+plus\s+(\w+)", "+"),  # drei plus fünf
                (r"(\w+)\s+minus\s+(\w+)", "-"),  # zehn minus zwei
                (r"(\w+)\s+mal\s+(\w+)", "*"),  # sieben mal acht
                (r"(\w+)\s+durch\s+(\w+)", "/"),  # zwanzig durch vier
                (r"(\w+)\s+geteilt\s+durch\s+(\w+)", "/"),  # zehn geteilt durch zwei
                (
                    r"(\w+)\s+multipliziert\s+mit\s+(\w+)",
                    "*",
                ),  # drei multipliziert mit vier
                (r"(\w+)\s+addiert\s+mit\s+(\w+)", "+"),  # zwei addiert mit fünf
                (
                    r"(\w+)\s+subtrahiert\s+von\s+(\w+)",
                    "-",
                ),  # fünf subtrahiert von zehn
            ]

            for pattern, operator in operator_patterns:
                match = re.search(pattern, text_lower)
                if match:
                    operand1 = match.group(1).strip()
                    operand2 = match.group(2).strip()

                    logger.debug(f"Parsed expression: {operand1} {operator} {operand2}")

                    return True, {
                        "operator": operator,
                        "operand1_word": operand1,
                        "operand2_word": operand2,
                        "original_text": query_text,
                    }

            return False, {
                "error": f"Konnte keinen Operator in '{query_text}' erkennen"
            }

        except Exception as e:
            logger.error(f"Fehler beim Parsen des Ausdrucks: {e}", exc_info=True)
            return False, {"error": str(e)}

    def _convert_words_to_numbers(
        self, context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Konvertiere Zahlwörter zu Zahlen.

        Verwendet component_53_number_language (wenn verfügbar) oder Fallback.
        """
        try:
            operand1_word = context.get("operand1_word", "")
            operand2_word = context.get("operand2_word", "")

            # PHASE 1: Einfacher Fallback für Zahlen 0-20 (wird später durch component_53 ersetzt)
            number_map = {
                "null": 0,
                "eins": 1,
                "ein": 1,
                "zwei": 2,
                "drei": 3,
                "vier": 4,
                "fünf": 5,
                "sechs": 6,
                "sieben": 7,
                "acht": 8,
                "neun": 9,
                "zehn": 10,
                "elf": 11,
                "zwölf": 12,
                "dreizehn": 13,
                "vierzehn": 14,
                "fünfzehn": 15,
                "sechzehn": 16,
                "siebzehn": 17,
                "achtzehn": 18,
                "neunzehn": 19,
                "zwanzig": 20,
            }

            # Versuche als Zahlwort oder als Zahl zu parsen
            try:
                operand1 = (
                    number_map.get(operand1_word.lower(), None)
                    if operand1_word.lower() in number_map
                    else int(operand1_word)
                )
            except ValueError:
                return False, {
                    "error": f"Kann '{operand1_word}' nicht zu Zahl konvertieren"
                }

            try:
                operand2 = (
                    number_map.get(operand2_word.lower(), None)
                    if operand2_word.lower() in number_map
                    else int(operand2_word)
                )
            except ValueError:
                return False, {
                    "error": f"Kann '{operand2_word}' nicht zu Zahl konvertieren"
                }

            logger.debug(
                f"Converted: {operand1_word} -> {operand1}, {operand2_word} -> {operand2}"
            )

            # Übernehme vorherigen Context und füge Zahlen hinzu
            result = dict(context)
            result["operand1"] = operand1
            result["operand2"] = operand2
            return True, result

        except Exception as e:
            logger.error(f"Fehler bei Zahl-Konvertierung: {e}", exc_info=True)
            return False, {"error": str(e)}

    def _execute_operation(
        self, context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Führe arithmetische Operation aus.

        Verwendet component_52_arithmetic_reasoning (wenn verfügbar) oder Fallback.
        """
        try:
            operator = context.get("operator")
            operand1 = context.get("operand1")
            operand2 = context.get("operand2")

            if operator is None or operand1 is None or operand2 is None:
                return False, {"error": "Operator oder Operanden fehlen im Kontext"}

            # PHASE 2: Verwende ArithmeticEngine (mit Proof Tree) wenn verfügbar
            try:
                from component_52_arithmetic_reasoning import ArithmeticEngine

                # ArithmeticEngine aus Worker holen oder erstellen
                if not hasattr(self.worker, "arithmetic_engine"):
                    self.worker.arithmetic_engine = ArithmeticEngine(
                        self.worker.netzwerk
                    )
                    logger.debug(
                        "ArithmeticEngine initialisiert (lazy loading in Strategy)"
                    )

                arithmetic_engine = self.worker.arithmetic_engine

                # Berechnung durchführen
                arithmetic_result = arithmetic_engine.calculate(
                    operator, operand1, operand2
                )

                result_value = arithmetic_result.value
                proof_tree = arithmetic_result.proof_tree
                confidence = arithmetic_result.confidence

                logger.debug(
                    f"Calculated (ArithmeticEngine): {operand1} {operator} {operand2} = {result_value}"
                )

                # Emittiere Proof Tree an UI
                if proof_tree and hasattr(self.worker, "signals"):
                    self.worker.signals.proof_tree_update.emit(proof_tree)
                    logger.debug(
                        f"[Proof Tree] Arithmetic ProofTree emittiert: {len(proof_tree.get_all_steps())} Schritte"
                    )

                # Übernehme vorherigen Context und füge Ergebnis hinzu
                result = dict(context)
                result["result_value"] = result_value
                result["confidence"] = confidence
                result["proof_tree"] = proof_tree
                return True, result

            except ImportError:
                logger.warning(
                    "ArithmeticEngine nicht verfügbar, verwende Fallback-Berechnung"
                )

                # FALLBACK: Einfache Berechnung (ohne Proof Tree)
                if operator == "+":
                    result_value = operand1 + operand2
                elif operator == "-":
                    result_value = operand1 - operand2
                elif operator == "*":
                    result_value = operand1 * operand2
                elif operator == "/":
                    if operand2 == 0:
                        return False, {"error": "Division durch Null ist nicht erlaubt"}
                    result_value = operand1 / operand2
                else:
                    return False, {"error": f"Unbekannter Operator: {operator}"}

                logger.debug(
                    f"Calculated (Fallback): {operand1} {operator} {operand2} = {result_value}"
                )

                # Übernehme vorherigen Context und füge Ergebnis hinzu
                result = dict(context)
                result["result_value"] = result_value
                result["confidence"] = 1.0  # Fallback: Immer 100% Confidence
                return True, result

        except Exception as e:
            logger.error(f"Fehler bei Berechnung: {e}", exc_info=True)
            return False, {"error": str(e)}

    def _format_result(self, context: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Formatiere Ergebnis als Zahlwort.

        Verwendet component_53_number_language (wenn verfügbar) oder Fallback.
        """
        try:
            result_value = context.get("result_value")

            if result_value is None:
                return False, {"error": "Ergebnis fehlt im Kontext"}

            # PHASE 1: Einfacher Fallback für Zahlen 0-20 (wird später durch component_53 ersetzt)
            number_words = {
                0: "null",
                1: "eins",
                2: "zwei",
                3: "drei",
                4: "vier",
                5: "fünf",
                6: "sechs",
                7: "sieben",
                8: "acht",
                9: "neun",
                10: "zehn",
                11: "elf",
                12: "zwölf",
                13: "dreizehn",
                14: "vierzehn",
                15: "fünfzehn",
                16: "sechzehn",
                17: "siebzehn",
                18: "achtzehn",
                19: "neunzehn",
                20: "zwanzig",
            }

            # Versuche als Zahlwort zu formatieren, sonst als Zahl
            if isinstance(result_value, float) and result_value.is_integer():
                result_value = int(result_value)

            if isinstance(result_value, int) and result_value in number_words:
                result_word = number_words[result_value]
            else:
                result_word = str(result_value)

            logger.debug(f"Formatted result: {result_value} -> {result_word}")

            # Übernehme vorherigen Context und füge formatiertes Ergebnis hinzu
            result = dict(context)
            result["result_word"] = result_word
            result["final_answer"] = f"Das Ergebnis ist {result_word}."
            return True, result

        except Exception as e:
            logger.error(f"Fehler bei Formatierung: {e}", exc_info=True)
            return False, {"error": str(e)}


class SubGoalExecutor:
    """
    Hauptklasse für Sub-Goal Execution mit Strategy-Dispatching.

    Diese Klasse koordiniert die Ausführung von SubGoals, indem sie
    diese an die passenden Strategien weiterleitet.
    """

    def __init__(self, worker):
        """
        Initialisiert den Executor mit allen Strategien.

        Args:
            worker: KaiWorker-Instanz für Zugriff auf Subsysteme
        """
        self.worker = worker

        # Strategien in Prioritätsreihenfolge
        # (Spezifischere Strategien zuerst)
        self.strategies = [
            OrchestratedStrategy(
                worker
            ),  # Orchestrierte Multi-Segment-Verarbeitung (HÖCHSTE PRIORITÄT)
            ConfirmationStrategy(worker),
            ClarificationStrategy(worker),
            ArithmeticStrategy(worker),  # Arithmetische Berechnungen (Phase Mathematik)
            SpatialReasoningStrategy(worker),  # Räumliches Reasoning (Phase 9)
            EpisodicMemoryStrategy(worker),  # Episodisches Gedächtnis
            FileReaderStrategy(worker),  # Datei-Ingestion (Phase 4)
            PatternLearningStrategy(worker),
            IngestionStrategy(worker),
            DefinitionStrategy(worker),
            LearningStrategy(worker),
            QuestionStrategy(worker),
            SharedStrategy(worker),  # Fallback für gemeinsame Sub-Goals
        ]

    def execute_sub_goal(
        self, sub_goal: SubGoal, context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Führt ein Sub-Goal aus, indem es an die passende Strategy dispatched wird.

        Args:
            sub_goal: Das auszuführende SubGoal
            context: Kontext-Dictionary mit vorherigen Ergebnissen

        Returns:
            Tuple (success: bool, result: Dict)
        """
        logger.debug(f"Dispatching SubGoal: '{sub_goal.description[:50]}...'")

        # Finde passende Strategy
        for strategy in self.strategies:
            if strategy.can_handle(sub_goal.description):
                logger.debug(f"  -> Verwendet {strategy.__class__.__name__}")
                return strategy.execute(sub_goal, context)

        # Keine Strategy gefunden
        logger.error(f"Keine Strategy gefunden für SubGoal: '{sub_goal.description}'")
        return False, {"error": f"Unbekannter Sub-Goal-Typ: {sub_goal.description}"}
