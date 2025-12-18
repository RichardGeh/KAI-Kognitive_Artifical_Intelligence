# kai_sub_goal_strategies_core.py
"""
Core Sub-Goal Strategies

Question, Learning, and Definition strategies.

Author: KAI Development Team
Date: 2025-11-14
"""

import logging
from typing import Any, Dict, Tuple

from component_5_linguistik_strukturen import (
    ContextAction,
    SubGoal,
)
from kai_sub_goal_strategy_base import SubGoalStrategy

logger = logging.getLogger(__name__)


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

        # Hole Intent aus Kontext für question_type Check
        intent = context.get("intent")
        question_type = intent.arguments.get("question_type") if intent else None

        # Tracke Graph-Abfrage
        self.worker.working_memory.add_reasoning_state(
            step_type="fact_retrieval",
            description=f"Suche Fakten über '{topic}' im Wissensgraphen",
            data={"topic": topic, "question_type": question_type},
        )

        # SCHRITT 1: Spezial-Handling für WER-Fragen (Reverse Lookup!)
        # Bei "Wer trinkt Brandy?" suchen wir: (person)-[:ASSOCIATED_WITH]->(brandy)
        if question_type == "person_query":
            logger.info(
                f"[WER-Frage] Verwende Reverse Lookup für '{topic}' (suche wer MIT '{topic}' assoziiert ist)"
            )
            # Inverse Relations: Finde alle Entitäten, die AUF 'topic' zeigen
            inverse_facts_raw = (
                self.worker.netzwerk.query_inverse_relations_with_confidence(topic)
            )

            # FORMAT-KONVERTIERUNG: Inverse Relations haben Format:
            # {"ASSOCIATED_WITH": [{"source": "nick", "confidence": 0.9}]}
            # Aber format_person_answer() erwartet:
            # {"ASSOCIATED_WITH": ["nick"]}
            # Konvertiere: Extrahiere "source" aus jedem Dict
            inverse_facts = {}
            for rel_type, entries in inverse_facts_raw.items():
                # Extrahiere nur die "source" (Person/Entität) aus jedem Entry
                inverse_facts[rel_type] = [entry["source"] for entry in entries]

            # Konvertiere zu fact_data Format
            fact_data = {
                "primary_topic": topic,
                "synonyms": [],
                "facts": inverse_facts,  # Jetzt im richtigen Format!
                "sources": {topic: inverse_facts},
                "bedeutungen": [],
            }

            logger.info(
                f"[WER-Frage] Gefunden: {sum(len(v) for v in inverse_facts.values())} inverse Relationen"
            )
        else:
            # SCHRITT 1: Direkte Faktenabfrage (effizient, nutzt Cache)
            fact_data = self.worker.netzwerk.query_facts_with_synonyms(topic)

        _ = (
            bool(fact_data["facts"])
            or bool(fact_data["bedeutungen"])
            or bool(fact_data["synonyms"])
        )  # has_any_knowledge unused

        # Create ProofTree for direct facts
        proof_tree = None
        if fact_data["facts"]:
            try:
                from component_17_proof_explanation import (
                    ProofStep,
                    ProofTree,
                    StepType,
                )

                proof_tree = ProofTree(query=f"Was macht {topic}?")

                # Create proof steps for each fact
                for relation_type, objects in fact_data["facts"].items():
                    for obj in objects[:3]:  # Limit to 3
                        step = ProofStep(
                            step_id=f"direct_{topic}_{relation_type}_{obj}",
                            step_type=StepType.FACT_MATCH,
                            inputs=[topic],
                            output=f"{topic} {relation_type} {obj}",
                            confidence=1.0,
                            explanation_text=f"Direkter Fakt: {topic} -> {obj}",
                            source_component="direct_fact_lookup",
                        )
                        proof_tree.add_root_step(step)

                logger.debug(
                    f"[ProofTree] Created for direct facts: {len(proof_tree.root_steps)} steps"
                )
            except Exception as e:
                logger.warning(f"[ProofTree] Failed to create: {e}")
                proof_tree = None

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
                    "proof_tree": proof_tree,
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
                        "proof_tree": inference_result.get("proof_tree"),
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
            "proof_tree": proof_tree,
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
        proof_tree = context.get("proof_tree")
        confidence = context.get("confidence", 0.8)

        # Spezifische Antwortgenerierung basierend auf Fragetyp
        if question_type == "person_query":
            response_text = formatter.format_person_answer(
                topic, facts, bedeutungen, synonyms
            )
        elif question_type == "time_query":
            response_text = formatter.format_time_answer(topic, facts, bedeutungen)
        elif question_type == "process_query":
            response_text = formatter.format_process_answer(topic, facts, bedeutungen)
        elif question_type == "reason_query":
            response_text = formatter.format_reason_answer(topic, facts, bedeutungen)
        else:
            # Standard-Antwort
            response_text = formatter.format_standard_answer(
                topic,
                facts,
                bedeutungen,
                synonyms,
                query_type=query_type,
                backward_chaining_used=backward_chaining_used,
                is_hypothesis=is_hypothesis,
            )

        return True, {
            "final_response": response_text,
            "proof_tree": proof_tree,
            "confidence": confidence,
        }


# ============================================================================
# PATTERN LEARNING STRATEGY (Lerne Muster)
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
