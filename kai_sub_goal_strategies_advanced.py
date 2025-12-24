# kai_sub_goal_strategies_advanced.py
"""
Advanced Execution Strategies

Shared knowledge, episodic memory, orchestration, and introspection strategies.

Author: KAI Development Team
Date: 2025-11-14
"""

import logging
from typing import Any, Dict, Optional, Tuple

from component_5_linguistik_strukturen import SubGoal
from kai_sub_goal_strategy_base import SubGoalStrategy

logger = logging.getLogger(__name__)


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
            # Delegiere an LearningStrategy (lazy import to avoid circular dependency)
            from kai_sub_goal_strategies_core import LearningStrategy

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
        from kai_ingestion_handler import KaiIngestionHandler

        segment_text = sub_goal.metadata.get("segment_text")
        if not segment_text:
            return False, {"error": "Kein Segment-Text für Befehl gefunden"}

        logger.info(f"Führe orchestrierten Befehl aus: '{segment_text[:50]}...'")

        # Erkenne Befehlstyp und führe aus
        segment_lower = segment_text.lower().strip()

        # Lerne-Befehl: "Lerne: Ein Hund bellt."
        if segment_lower.startswith("lerne:"):
            # Extrahiere Text nach "Lerne:"
            learn_text = segment_text[segment_text.index(":") + 1 :].strip()

            if not learn_text:
                return False, {"error": "Kein Text zum Lernen gefunden"}

            logger.info(f"Lerne: '{learn_text}'")

            # Verarbeite via Ingestion Handler
            ingestion_handler = KaiIngestionHandler(
                self.worker.netzwerk,
                self.worker.preprocessor,
                self.worker.prototyping_engine,
                self.worker.embedding_service,
            )

            stats = ingestion_handler.ingest_text(learn_text)
            facts_learned = stats.get("facts_created", 0)

            logger.info(
                f"Befehl 'Lerne:' abgeschlossen: {facts_learned} Fakten gelernt"
            )

            # Tracke im Working Memory
            self.worker.working_memory.add_reasoning_state(
                step_type="command_execution",
                description=f"Gelernt: {facts_learned} Fakten aus '{learn_text[:30]}...'",
                data={"facts_learned": facts_learned, "command": "lerne"},
            )

            return True, {
                "command": "lerne",
                "text": learn_text,
                "facts_learned": facts_learned,
                "processed": True,
            }

        # Definiere-Befehl: "Definiere: ..."
        elif segment_lower.startswith("definiere:"):
            define_text = segment_text[segment_text.index(":") + 1 :].strip()

            if not define_text:
                return False, {"error": "Kein Text zum Definieren gefunden"}

            logger.info(f"Definiere: '{define_text}'")

            ingestion_handler = KaiIngestionHandler(
                self.worker.netzwerk,
                self.worker.preprocessor,
                self.worker.prototyping_engine,
                self.worker.embedding_service,
            )

            stats = ingestion_handler.ingest_text(define_text)
            facts_learned = stats.get("facts_created", 0)

            logger.info(
                f"Befehl 'Definiere:' abgeschlossen: {facts_learned} Fakten gelernt"
            )

            return True, {
                "command": "definiere",
                "text": define_text,
                "facts_learned": facts_learned,
                "processed": True,
            }

        # Unbekannter Befehl - logge Warnung, aber fahre fort
        else:
            logger.warning(f"Unbekannter Befehlstyp: '{segment_text[:30]}...'")
            return True, {
                "command_text": segment_text,
                "processed": False,
                "warning": "Unbekannter Befehl",
            }

    def _extract_topic_from_text(self, doc, text: str) -> Optional[str]:
        """
        Extrahiert das Thema aus einem spaCy doc oder Text.

        Sucht nach Substantiven (NOUN) oder Eigennamen (PROPN),
        wobei Fragewörter und Artikel ignoriert werden.

        Args:
            doc: spaCy Doc Objekt
            text: Ursprünglicher Text

        Returns:
            Extrahiertes Thema oder None
        """
        # Ignoriere diese Wörter
        ignore_words = {
            "was",
            "wer",
            "wie",
            "wo",
            "wann",
            "warum",
            "kann",
            "ist",
            "hat",
            "ein",
            "eine",
            "der",
            "die",
            "das",
            "einem",
            "einer",
            "eines",
        }

        # Suche nach Substantiven
        for token in doc:
            if (
                token.pos_ in ("NOUN", "PROPN")
                and token.lemma_.lower() not in ignore_words
            ):
                # Verwende das Lemma (Grundform)
                return token.lemma_.lower()

        # Fallback: Suche nach dem ersten nicht-ignorierten Wort
        for token in doc:
            if token.lemma_.lower() not in ignore_words and len(token.text) > 2:
                return token.lemma_.lower()

        return None

    def _answer_question_with_context(
        self, sub_goal: SubGoal, context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Beantwortet eine Frage mit zuvor gelerntem Kontext.

        PHASE 4: Enhanced to detect logic puzzles and route to logic puzzle solver.

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

        # PHASE 4: Check if this is a logic puzzle orchestration result
        # If so, route directly to logic puzzle solver via strategy dispatcher
        orchestration_result = context.get("orchestration_result")

        if orchestration_result and orchestration_result.get("is_logic_puzzle"):
            puzzle_type = orchestration_result.get("puzzle_type", "UNKNOWN")
            puzzle_metadata = orchestration_result.get("metadata", {})

            logger.info(
                f"[Logic Puzzle Orchestration] Detected: {puzzle_type.value if hasattr(puzzle_type, 'value') else puzzle_type}"
            )

            # Extract entities from segments
            # FIX: Collect ALL words from ALL segments FIRST, then count globally
            # Previously the Counter was created inside the loop, resetting counts per segment
            import re
            from collections import Counter

            segments = orchestration_result.get("segments", [])

            # PHASE 1: Collect all capitalized words from all explanation segments
            all_words = []
            for seg in segments:
                if seg.is_explanation():
                    # Pattern matches single capital letters OR capitalized words
                    words = re.findall(r"\b[A-Z][a-z]*\b", seg.text)
                    all_words.extend(words)

            # PHASE 2: Count occurrences globally across all segments
            global_word_counts = Counter(all_words)

            # Entity exclusion list - words that should not be treated as entities
            # Synchronized with kai_worker.py ENTITY_EXCLUSIONS
            ENTITY_EXCLUSIONS = {
                # German articles
                "Der",
                "Die",
                "Das",
                "Ein",
                "Eine",
                # Conjunctions
                "Wenn",
                "Falls",
                "Aber",
                "Und",
                "Oder",
                # Meta-words / Turn labels
                "Turn",
                "Step",
                "Schritt",
                "Runde",
                "Punkt",
                "Teil",
                # German indefinite pronouns - FIX for test_negative_constraints
                "Jede",
                "Jeder",
                "Jedes",
                "Alle",
                "Keine",
                "Keiner",
                # German command verbs (imperatives)
                "Finde",
                "Gib",
                "Nenne",
                "Bestimme",
                "Zeige",
                "Erklaere",
                "Berechne",
                # Colors and declined forms
                "Rot",
                "Blau",
                "Gruen",
                "Gelb",
                "Schwarz",
                "Weiss",
                "Gelbes",
                "Rotes",
                "Blaues",
                "Gruenes",
                "Weisses",
                "Schwarzes",
            }

            # PHASE 3: Filter entities based on global counts
            # Include:
            # 1. Single letters that appear at least once (puzzle entities like A, B, C)
            # 2. Multi-letter words that appear at least 2x globally and aren't exclusions
            entities = []
            for word, count in global_word_counts.items():
                if word in ENTITY_EXCLUSIONS:
                    continue
                if len(word) == 1:  # Single letter (A, B, C, etc.)
                    entities.append(word)
                elif count >= 2:  # Multi-letter word appearing 2+ times globally
                    entities.append(word)

            # Remove duplicates while preserving order
            entities = list(dict.fromkeys(entities))

            # Reconstruct full puzzle text (all segments)
            full_puzzle_text = " ".join(seg.text for seg in segments)

            # Store puzzle context in working memory for dispatcher
            self.worker.working_memory.add_reasoning_state(
                step_type="orchestrated_logic_puzzle",
                description=f"Logik-Raetsel orchestriert: {puzzle_metadata.get('puzzle_classification', 'UNKNOWN')}",
                data={
                    "full_text": full_puzzle_text,
                    "question": segment_text,
                    "entities": entities,
                    "puzzle_classification": puzzle_metadata.get(
                        "puzzle_classification", "UNKNOWN"
                    ),
                    "segment_count": puzzle_metadata.get("total_segments", 0),
                    "explanation_count": puzzle_metadata.get("explanation_count", 0),
                    "question_count": puzzle_metadata.get("question_count", 0),
                },
                confidence=0.85,
            )

            logger.debug(
                f"[Logic Puzzle Orchestration] Stored context: {len(entities)} entities, "
                f"{len(full_puzzle_text)} chars full text"
            )

            # Route directly to logic puzzle solver via strategy dispatcher
            logger.info(
                "[Logic Puzzle Orchestration] Routing to LOGIC_PUZZLE strategy dispatcher"
            )

            try:
                # Get reasoning orchestrator from inference handler
                reasoning_orchestrator = None
                if (
                    hasattr(self.worker, "inference_handler")
                    and self.worker.inference_handler
                ):
                    reasoning_orchestrator = getattr(
                        self.worker.inference_handler, "_reasoning_orchestrator", None
                    )

                if not reasoning_orchestrator:
                    logger.warning(
                        "[Logic Puzzle Orchestration] No reasoning orchestrator available, falling back to direct solver"
                    )
                    # Fallback: Use logic puzzle solver directly
                    from component_45_logic_puzzle_solver_core import LogicPuzzleSolver

                    solver = LogicPuzzleSolver()
                    solver_result = solver.solve(
                        full_puzzle_text, entities, segment_text
                    )

                    if solver_result and solver_result.get("result") == "SATISFIABLE":
                        answer_text = solver_result.get("answer", "")
                        return True, {
                            "final_response": answer_text,
                            "proof_tree": solver_result.get("proof_tree"),
                            "confidence": solver_result.get("confidence", 0.7),
                        }
                    else:
                        return False, {
                            "error": "Logic puzzle solver konnte keine Lösung finden.",
                            "confidence": 0.0,
                        }

                from kai_reasoning_orchestrator import ReasoningStrategy

                # Execute LOGIC_PUZZLE strategy via dispatcher
                result = (
                    reasoning_orchestrator.strategy_dispatcher.execute_single_strategy(
                        topic="logic_puzzle",  # Not used for logic puzzles
                        relation_type=None,
                        strategy=ReasoningStrategy.LOGIC_PUZZLE,
                        query_text=full_puzzle_text,
                        context=context,
                    )
                )

                if result and result.success:
                    # Extract answer from inferred_facts
                    answer_text = result.metadata.get("answer", "")
                    if not answer_text and result.inferred_facts:
                        # Fallback: get from PUZZLE_SOLUTION
                        puzzle_solutions = result.inferred_facts.get(
                            "PUZZLE_SOLUTION", []
                        )
                        answer_text = puzzle_solutions[0] if puzzle_solutions else ""

                    logger.info(
                        f"[Logic Puzzle Orchestration] Solver succeeded with confidence {result.confidence:.2f}"
                    )

                    return True, {
                        "final_response": answer_text,
                        "proof_tree": result.proof_tree,
                        "confidence": result.confidence,
                    }
                else:
                    logger.warning(
                        "[Logic Puzzle Orchestration] Solver returned no result"
                    )
                    return False, {
                        "error": "Logic puzzle solver konnte keine Lösung finden.",
                        "confidence": 0.0,
                    }

            except Exception as e:
                logger.error(
                    f"[Logic Puzzle Orchestration] Solver error: {e}", exc_info=True
                )
                # Propagate exception to make bugs visible in tests
                raise

        # Standard question processing for non-puzzle questions
        # Extrahiere Intent aus Frage
        doc = self.worker.preprocessor.process(segment_text)
        meaning_points = self.worker.extractor.extract(doc)

        # Nutze QuestionStrategy für normale Verarbeitung (lazy import to avoid circular dependency)
        from kai_sub_goal_strategies_core import QuestionStrategy

        question_strategy = QuestionStrategy(self.worker)

        if meaning_points:
            intent = meaning_points[0]
            question_context = {"intent": intent}

            # Identifiziere Thema
            success, result = question_strategy._identify_topic(intent)
            if success:
                question_context.update(result)
            else:
                # Fallback: Extrahiere Thema aus Text direkt
                topic = self._extract_topic_from_text(doc, segment_text)
                if topic:
                    question_context["topic"] = topic
                    question_context["topic_source"] = "fallback_extraction"
                    logger.info(f"Thema via Fallback extrahiert: '{topic}'")
                else:
                    return False, {"error": "Kein Thema gefunden."}
        else:
            # Kein MeaningPoint - Fallback: Extrahiere Thema direkt aus spaCy doc
            logger.warning("Keine MeaningPoints, verwende Fallback-Extraktion")
            topic = self._extract_topic_from_text(doc, segment_text)
            if not topic:
                return False, {"error": "Konnte Intent aus Frage nicht extrahieren"}

            # Erstelle minimalen Kontext
            from component_5_linguistik_strukturen import (
                MeaningPoint,
                MeaningPointCategory,
                Modality,
                Polarity,
            )

            intent = MeaningPoint(
                id=f"fallback-{hash(segment_text) % 10000}",
                category=MeaningPointCategory.QUESTION,
                cue="fallback",
                text_span=segment_text,
                modality=Modality.INTERROGATIVE,
                polarity=Polarity.POSITIVE,
                confidence=0.7,
                arguments={"topic": topic},
            )
            question_context = {
                "intent": intent,
                "topic": topic,
                "topic_source": "fallback_extraction",
            }
            logger.info(f"Fallback-Intent erstellt mit Thema: '{topic}'")

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

        # PHASE 4: Boost confidence for logic puzzle answers
        if orchestration_result and orchestration_result.get("is_logic_puzzle"):
            # Apply puzzle-specific confidence boost
            if "final_response" in result and result["final_response"]:
                # Check if answer is not a knowledge gap message
                if "Ich weiß nicht" not in result["final_response"]:
                    original_confidence = question_context.get("confidence", 0.5)
                    boosted_confidence = min(original_confidence + 0.20, 0.95)
                    logger.info(
                        f"[Logic Puzzle Orchestration] Confidence boost: {original_confidence:.2f} -> {boosted_confidence:.2f}"
                    )
                    result["confidence"] = boosted_confidence

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


class IntrospectionStrategy(SubGoalStrategy):
    """
    Strategy für Introspektions-Queries über Production Rules (PHASE 9).

    Zuständig für:
    - Abfrage aller Regeln einer Kategorie
    - Identifikation meistverwendeter Regeln
    - Suche nach Regeln mit niedriger Utility
    - Aggregierte Statistiken über das Regelsystem

    Beispiel-Queries:
    - "Zeige mir alle Content-Selection Regeln"
    - "Welche Regel wurde am häufigsten verwendet?"
    - "Welche Regeln haben niedrige Utility?"
    """

    def can_handle(self, sub_goal_description: str) -> bool:
        introspection_keywords = [
            "zeige.*regeln",
            "welche regel.*häufigst",
            "welche regel.*verwendet",
            "niedrige.*utility",
            "niedrige.*priorität",
            "regel.*statistik",
            "alle.*regeln",
            "content.*selection.*regel",
            "lexicalization.*regel",
            "discourse.*regel",
            "syntax.*regel",
        ]

        import re

        description_lower = sub_goal_description.lower()
        return any(
            re.search(pattern, description_lower) for pattern in introspection_keywords
        )

    def execute(
        self, sub_goal: SubGoal, context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        try:
            sub_goal.description.lower()
            query_text = sub_goal.metadata.get("query_text", "").lower()

            # Zugriff auf KonzeptNetzwerk
            netzwerk = self.worker.netzwerk

            # CASE 1: Alle Regeln einer bestimmten Kategorie
            if any(
                cat in query_text
                for cat in [
                    "content-selection",
                    "content selection",
                    "lexicalization",
                    "discourse",
                    "syntax",
                ]
            ):
                return self._query_by_category(query_text, netzwerk, context)

            # CASE 2: Meistverwendete Regeln
            if any(
                kw in query_text for kw in ["häufigst", "meistverwendet", "am meisten"]
            ):
                return self._query_most_used(netzwerk, context)

            # CASE 3: Regeln mit niedriger Utility
            if any(
                kw in query_text
                for kw in ["niedrige utility", "niedrige priorität", "geringe utility"]
            ):
                return self._query_low_utility(netzwerk, context)

            # CASE 4: Alle Regeln (Übersicht)
            if "alle regeln" in query_text or "zeige mir alle" in query_text:
                return self._query_all_rules(netzwerk, context)

            # CASE 5: Aggregierte Statistiken
            if "statistik" in query_text:
                return self._query_statistics(netzwerk, context)

            # Fallback: Keine spezifische Abfrage erkannt
            logger.warning(f"Introspection-Query nicht erkannt: {query_text}")
            return False, {
                "error": f"Introspektions-Query nicht verstanden: {query_text}"
            }

        except Exception as e:
            logger.error(f"Fehler bei Introspection: {e}", exc_info=True)
            return False, {"error": str(e)}

    def _query_by_category(
        self, query_text: str, netzwerk, context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Abfrage aller Regeln einer Kategorie."""
        # Identifiziere Kategorie
        category_map = {
            "content-selection": "content_selection",
            "content selection": "content_selection",
            "lexicalization": "lexicalization",
            "discourse": "discourse",
            "syntax": "syntax",
        }

        category = None
        for key, value in category_map.items():
            if key in query_text:
                category = value
                break

        if not category:
            return False, {"error": "Kategorie nicht erkannt"}

        # Abfrage aus Neo4j
        rules = netzwerk.query_production_rules(category=category, order_by="priority")

        # Formatiere Antwort
        if not rules:
            answer = f"Keine Regeln in der Kategorie '{category}' gefunden."
        else:
            rule_list = []
            for rule in rules:
                name = rule["name"]
                utility = rule["utility"]
                priority = rule.get("priority", utility * rule.get("specificity", 1.0))
                app_count = rule["application_count"]
                rule_list.append(
                    f"  - {name} (Utility: {utility:.2f}, Priorität: {priority:.2f}, Verwendet: {app_count}x)"
                )

            answer = (
                f"Gefundene Regeln in Kategorie '{category}' ({len(rules)} Regeln):\n"
                + "\n".join(rule_list)
            )

        logger.info(f"Introspection (by category): {len(rules)} Regeln gefunden")

        result = dict(context)
        result["rules"] = rules
        result["final_answer"] = answer
        return True, result

    def _query_most_used(
        self, netzwerk, context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Abfrage der meistverwendeten Regeln."""
        # Top 5 meistverwendete Regeln
        rules = netzwerk.query_production_rules(
            min_application_count=1, order_by="usage", limit=5
        )

        if not rules:
            answer = "Keine Regeln wurden bisher verwendet."
        else:
            rule_list = []
            for rule in rules:
                name = rule["name"]
                app_count = rule["application_count"]
                success_count = rule["success_count"]
                category = rule["category"]
                rule_list.append(
                    f"  - {name} [{category}] ({app_count}x verwendet, {success_count}x erfolgreich)"
                )

            answer = f"Top {len(rules)} meistverwendete Regeln:\n" + "\n".join(
                rule_list
            )

        logger.info(f"Introspection (most used): {len(rules)} Regeln gefunden")

        result = dict(context)
        result["rules"] = rules
        result["final_answer"] = answer
        return True, result

    def _query_low_utility(
        self, netzwerk, context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Abfrage der Regeln mit niedriger Utility (<0.5)."""
        rules = netzwerk.query_production_rules(max_utility=0.5, order_by="priority")

        if not rules:
            answer = "Keine Regeln mit niedriger Utility (<0.5) gefunden."
        else:
            rule_list = []
            for rule in rules:
                name = rule["name"]
                utility = rule["utility"]
                category = rule["category"]
                rule_list.append(f"  - {name} [{category}] (Utility: {utility:.2f})")

            answer = (
                f"Regeln mit niedriger Utility (<0.5): {len(rules)} Regeln:\n"
                + "\n".join(rule_list)
            )

        logger.info(f"Introspection (low utility): {len(rules)} Regeln gefunden")

        result = dict(context)
        result["rules"] = rules
        result["final_answer"] = answer
        return True, result

    def _query_all_rules(
        self, netzwerk, context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Abfrage aller Regeln (Übersicht)."""
        rules = netzwerk.get_all_production_rules()

        if not rules:
            answer = "Keine Regeln im System gefunden."
        else:
            # Gruppiere nach Kategorie
            by_category = {}
            for rule in rules:
                category = rule["category"]
                if category not in by_category:
                    by_category[category] = []
                by_category[category].append(rule)

            category_summaries = []
            for category, cat_rules in sorted(by_category.items()):
                category_summaries.append(f"  - {category}: {len(cat_rules)} Regeln")

            answer = f"Übersicht über alle Regeln ({len(rules)} gesamt):\n" + "\n".join(
                category_summaries
            )

        logger.info(f"Introspection (all rules): {len(rules)} Regeln gefunden")

        result = dict(context)
        result["rules"] = rules
        result["final_answer"] = answer
        return True, result

    def _query_statistics(
        self, netzwerk, context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Abfrage aggregierter Statistiken."""
        stats = netzwerk.get_production_rule_statistics()

        if not stats:
            answer = "Keine Statistiken verfügbar."
        else:
            total_rules = stats.get("total_rules", 0)
            by_category = stats.get("by_category", {})
            most_used = stats.get("most_used", [])
            low_utility = stats.get("low_utility", [])

            # Formatiere Kategorie-Verteilung
            category_lines = []
            for category, cat_stats in sorted(by_category.items()):
                count = cat_stats["count"]
                avg_util = cat_stats["avg_utility"]
                avg_apps = cat_stats["avg_applications"]
                category_lines.append(
                    f"  - {category}: {count} Regeln (Ø Utility: {avg_util:.2f}, Ø Verwendungen: {avg_apps:.1f})"
                )

            # Formatiere Top 5 meistverwendet
            most_used_lines = []
            for rule in most_used[:5]:
                name = rule["name"]
                count = rule["count"]
                most_used_lines.append(f"  - {name}: {count}x")

            # Formatiere Low Utility
            low_utility_lines = []
            for rule in low_utility[:5]:
                name = rule["name"]
                util = rule["utility"]
                low_utility_lines.append(f"  - {name}: {util:.2f}")

            answer = f"""Produktionsregel-Statistiken:

Gesamt: {total_rules} Regeln

Nach Kategorie:
{chr(10).join(category_lines)}

Meistverwendet:
{chr(10).join(most_used_lines) if most_used_lines else '  (keine)'}

Niedrige Utility:
{chr(10).join(low_utility_lines) if low_utility_lines else '  (keine)'}"""

        logger.info("Introspection (statistics): Statistiken erstellt")

        result = dict(context)
        result["statistics"] = stats
        result["final_answer"] = answer
        return True, result
