# kai_inference_handler.py
"""
Inference Handler Module für KAI

Verantwortlichkeiten:
- Backward-Chaining Inference mit Logic Engine
- Graph-Traversal für Multi-Hop-Reasoning
- Abductive Reasoning für Hypothesengenerierung
- Proof Tree Generierung für Erklärbarkeit

PHASE: Confidence-Based Learning Integration
- Verwendet ConfidenceManager für einheitliche Confidence-Berechnung
- Confidence-Decay für veraltete Fakten
- Threshold-basierte Reasoning-Entscheidungen
"""
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from component_1_netzwerk import KonzeptNetzwerk
from component_9_logik_engine import Engine, Fact, Goal
from component_confidence_manager import get_confidence_manager

# Import exception utilities for user-friendly error messages
from kai_exceptions import (
    AbductiveReasoningError,
    GraphTraversalError,
    InferenceError,
    get_user_friendly_message,
)

logger = logging.getLogger(__name__)

# Import Unified Proof Explanation System
try:
    from component_17_proof_explanation import (
        ProofTree,
        create_proof_tree_from_logic_engine,
    )

    PROOF_SYSTEM_AVAILABLE = True
except ImportError:
    PROOF_SYSTEM_AVAILABLE = False
    logger.warning("Unified Proof Explanation System nicht verfügbar")


class KaiInferenceHandler:
    """
    Handler für komplexe Schlussfolgerungen (Backward-Chaining, Graph-Traversal, Abductive Reasoning).

    Diese Klasse verwaltet:
    - Backward-Chaining mit Logic Engine
    - Graph-Traversal für transitive Relationen
    - Abductive Reasoning für Hypothesen
    - Proof Tree Generierung und Signale
    """

    def __init__(
        self,
        netzwerk: KonzeptNetzwerk,
        engine: Engine,
        graph_traversal,  # GraphTraversal
        working_memory,  # WorkingMemory
        signals,  # KaiSignals
        enable_hybrid_reasoning: bool = True,
    ):
        """
        Initialisiert den Inference Handler.

        Args:
            netzwerk: KonzeptNetzwerk für Datenspeicherung
            engine: Logic Engine für Backward-Chaining
            graph_traversal: GraphTraversal-Engine für Multi-Hop-Reasoning
            working_memory: WorkingMemory für Reasoning-Trace
            signals: KaiSignals für UI-Kommunikation (proof_tree_update)
            enable_hybrid_reasoning: Aktiviert Hybrid Reasoning Orchestrator (default: True)
        """
        self.netzwerk = netzwerk
        self.engine = engine
        self.graph_traversal = graph_traversal
        self.working_memory = working_memory
        self.signals = signals

        # Lazy-Loading für Abductive Engine (nur bei Bedarf)
        self._abductive_engine = None

        # PHASE: Confidence-Based Learning - Integriere ConfidenceManager
        self.confidence_manager = get_confidence_manager()
        logger.info("InferenceHandler initialisiert mit ConfidenceManager")

        # COMBINATORIAL REASONING: Integriere CombinatorialReasoner
        self._combinatorial_reasoner = None
        try:
            from component_40_combinatorial_reasoning import CombinatorialReasoner

            self._combinatorial_reasoner = CombinatorialReasoner()
            logger.info("CombinatorialReasoner successfully integrated")
        except ImportError:
            logger.warning("CombinatorialReasoner nicht verfügbar")

        # SPATIAL REASONING: Integriere SpatialReasoner
        self._spatial_reasoner = None
        try:
            from component_42_spatial_reasoning import SpatialReasoner

            self._spatial_reasoner = SpatialReasoner(netzwerk=self.netzwerk)
            logger.info("SpatialReasoner successfully integrated")
        except ImportError:
            logger.warning("SpatialReasoner nicht verfügbar")

        # RESONANCE ENGINE: Integriere ResonanceEngine
        self._resonance_engine = None
        try:
            from component_44_resonance_engine import ResonanceEngine

            self._resonance_engine = ResonanceEngine(
                netzwerk=self.netzwerk, confidence_mgr=self.confidence_manager
            )
            logger.info("ResonanceEngine successfully integrated")
        except ImportError:
            logger.warning("ResonanceEngine nicht verfügbar")

        # HYBRID REASONING: Integriere ReasoningOrchestrator
        self.enable_hybrid_reasoning = enable_hybrid_reasoning
        self._reasoning_orchestrator = None

        if self.enable_hybrid_reasoning:
            try:
                from kai_reasoning_orchestrator import ReasoningOrchestrator

                self._reasoning_orchestrator = ReasoningOrchestrator(
                    netzwerk=self.netzwerk,
                    logic_engine=self.engine,
                    graph_traversal=self.graph_traversal,
                    combinatorial_reasoner=self._combinatorial_reasoner,
                    spatial_reasoner=self._spatial_reasoner,
                    resonance_engine=self._resonance_engine,
                    working_memory=self.working_memory,
                    signals=self.signals,
                    probabilistic_engine=None,  # Lazy-loaded
                    abductive_engine=None,  # Lazy-loaded via property
                )
                logger.info("[OK] Hybrid Reasoning Orchestrator aktiviert")
            except Exception as e:
                logger.warning(f"Konnte Hybrid Reasoning nicht aktivieren: {e}")
                self.enable_hybrid_reasoning = False

    @property
    def abductive_engine(self):
        """Lazy-Loading für Abductive Engine."""
        if self._abductive_engine is None:
            try:
                from component_14_abductive_engine import AbductiveEngine

                self._abductive_engine = AbductiveEngine(self.netzwerk, self.engine)
                logger.debug("Abductive Engine erfolgreich initialisiert")
                # Update Orchestrator if exists
                if self._reasoning_orchestrator:
                    self._reasoning_orchestrator.abductive_engine = (
                        self._abductive_engine
                    )
            except Exception as e:
                logger.warning(f"Abductive Engine konnte nicht geladen werden: {e}")
        return self._abductive_engine

    @property
    def probabilistic_engine(self):
        """Lazy-Loading für Probabilistic Engine."""
        if (
            not hasattr(self, "_probabilistic_engine")
            or self._probabilistic_engine is None
        ):
            try:
                from component_16_probabilistic_engine import ProbabilisticEngine

                self._probabilistic_engine = ProbabilisticEngine()
                logger.debug("Probabilistic Engine erfolgreich initialisiert")
                # Update Orchestrator if exists
                if self._reasoning_orchestrator:
                    self._reasoning_orchestrator.probabilistic_engine = (
                        self._probabilistic_engine
                    )
            except Exception as e:
                logger.warning(f"Probabilistic Engine konnte nicht geladen werden: {e}")
                self._probabilistic_engine = None
        return self._probabilistic_engine

    def try_backward_chaining_inference(
        self, topic: str, relation_type: str = "IS_A"
    ) -> Optional[Dict[str, Any]]:
        """
        PHASE 3 & 7: Versucht eine Frage durch Backward-Chaining und Multi-Hop-Reasoning zu beantworten.

        Diese Methode wird aufgerufen, wenn die direkte Graph-Abfrage keine Ergebnisse liefert.

        HYBRID REASONING (NEW):
        - Wenn aktiviert, nutzt ReasoningOrchestrator für kombiniertes Reasoning
        - Kombiniert Logic + Graph + Probabilistic + Abductive
        - Weighted Confidence Fusion

        LEGACY FALLBACK:
        Sie versucht komplexere Schlussfolgerungen durch:
        1. Graph-Traversal für transitive Relationen (PHASE 7)
        2. Regelbasiertes Backward-Chaining
        3. Abductive Reasoning für Hypothesen (Fallback)

        Args:
            topic: Das Thema der Frage (z.B. "hund")
            relation_type: Der Typ der gesuchten Relation (default: "IS_A")

        Returns:
            Dictionary mit "inferred_facts", "proof_trace", "confidence" und optional "is_hypothesis"
            oder None wenn keine Schlussfolgerung möglich ist
        """
        logger.info(
            f"[Multi-Hop Reasoning] Versuche komplexe Schlussfolgerung für '{topic}'"
        )

        # HYBRID REASONING PATH (NEW)
        if self.enable_hybrid_reasoning and self._reasoning_orchestrator:
            logger.info("[Multi-Hop Reasoning] -> Nutze Hybrid Reasoning Orchestrator")
            result = self.try_hybrid_reasoning(topic, relation_type)
            if result:
                return result
            else:
                logger.info(
                    "[Multi-Hop Reasoning] Hybrid Reasoning lieferte keine Ergebnisse, falle zurück auf Legacy"
                )

        # RESONANCE-BASED QA PATH (NEW): Try resonance inference for question queries
        # This is especially useful for queries that benefit from spreading activation
        # and semantic connection detection (e.g., "Kann ein Pinguin fliegen?")
        if self._resonance_engine:
            # Construct a simple query string from topic and relation
            # This will be parsed by _handle_resonance_inference to extract concepts
            query_text = self._construct_query_from_topic(topic, relation_type)
            logger.info(
                f"[Multi-Hop Reasoning] -> Versuche Resonance-Based QA für '{query_text}'"
            )

            result = self._handle_resonance_inference(
                query=query_text,
                context={"topic": topic, "relation_type": relation_type},
            )

            if result and result.get("answer"):
                # Convert resonance result to standard format
                # Extract facts from the answer if possible
                inferred_facts = result.get("overlap", {})
                if not inferred_facts:
                    # Create simple fact structure from contradictions or answer
                    inferred_facts = {relation_type: []}

                return {
                    "inferred_facts": inferred_facts,
                    "proof_trace": result["answer"],
                    "confidence": result.get("confidence", 0.5),
                    "resonance_based": True,
                    "proof_tree": result.get("proof_tree"),
                }

        # LEGACY FALLBACK PATH (ORIGINAL)

        # PHASE 7: Versuche zuerst Graph-Traversal für transitive Relationen
        # Dies ist effizienter und direkter als regelbasiertes Backward-Chaining
        result = self._try_graph_traversal(topic, relation_type)
        if result:
            return result

        # PHASE 3: Fallback auf regelbasiertes Backward-Chaining
        result = self._try_backward_chaining(topic, relation_type)
        if result:
            return result

        # ABDUCTIVE REASONING: Wenn alles fehlschlägt, versuche Hypothesen zu generieren
        result = self._try_abductive_reasoning(topic, relation_type)
        if result:
            return result

        return None

    def try_hybrid_reasoning(
        self, topic: str, relation_type: str = "IS_A"
    ) -> Optional[Dict[str, Any]]:
        """
        HYBRID REASONING: Nutzt ReasoningOrchestrator für kombiniertes Reasoning.

        Kombiniert mehrere Reasoning-Strategien und aggregiert Ergebnisse:
        - Direct Facts (Fast Path)
        - Graph Traversal (Multi-Hop)
        - Logic Engine (Rule-based)
        - Probabilistic Enhancement
        - Abductive Fallback (Hypotheses)

        Args:
            topic: Das Thema der Frage
            relation_type: Der Typ der gesuchten Relation

        Returns:
            Dictionary mit aggregierten Ergebnissen oder None
        """
        if not self._reasoning_orchestrator:
            logger.warning("[Hybrid Reasoning] Orchestrator nicht verfügbar")
            return None

        try:
            # Lazy-load engines falls noch nicht geschehen
            if self.probabilistic_engine:
                pass  # Property triggers lazy loading
            if self.abductive_engine:
                pass  # Property triggers lazy loading

            # Query mit Hybrid Reasoning
            aggregated_result = (
                self._reasoning_orchestrator.query_with_hybrid_reasoning(
                    topic=topic,
                    relation_type=relation_type,
                    strategies=None,  # None = alle verfügbaren Strategien
                )
            )

            if aggregated_result:
                logger.info(
                    f"[Hybrid Reasoning] [OK] Erfolg mit {len(aggregated_result.strategies_used)} Strategien "
                    f"(Konfidenz: {aggregated_result.combined_confidence:.2f})"
                )

                # Konvertiere AggregatedResult zu Legacy-Format
                return {
                    "inferred_facts": aggregated_result.inferred_facts,
                    "proof_trace": aggregated_result.explanation,
                    "confidence": aggregated_result.combined_confidence,
                    "is_hypothesis": aggregated_result.is_hypothesis,
                    "hybrid": True,
                    "strategies_used": [
                        str(s.value) for s in aggregated_result.strategies_used
                    ],
                    "num_strategies": len(aggregated_result.strategies_used),
                }

            return None

        except Exception as e:
            logger.error(f"[Hybrid Reasoning] Fehler: {e}", exc_info=True)
            return None

    def _try_graph_traversal(
        self, topic: str, relation_type: str
    ) -> Optional[Dict[str, Any]]:
        """
        Versucht Graph-Traversal für transitive Relationen.

        Args:
            topic: Das Thema der Frage
            relation_type: Der Typ der gesuchten Relation

        Returns:
            Dictionary mit Ergebnissen oder None
        """
        logger.info(
            f"[Graph-Traversal] Versuche transitive Relationen für '{topic}' ({relation_type})"
        )

        try:
            # Finde alle transitiven Relationen des gesuchten Typs
            paths = self.graph_traversal.find_transitive_relations(
                topic, relation_type, max_depth=5
            )

            if paths:
                # Extrahiere alle Zielknoten aus den gefundenen Pfaden
                inferred_facts = {relation_type: []}
                for path in paths:
                    # Der letzte Knoten im Pfad ist das Ziel
                    target = path.nodes[-1]
                    if target not in inferred_facts[relation_type]:
                        inferred_facts[relation_type].append(target)

                # Generiere Erklärungstrace aus den Pfaden
                proof_trace_parts = []
                for i, path in enumerate(paths[:3], 1):  # Zeige nur erste 3 Pfade
                    proof_trace_parts.append(f"Pfad {i}: {path.explanation}")

                proof_trace = "\n".join(proof_trace_parts)

                # PHASE: Confidence-Based Learning - Berechne Confidence für Pfade
                # Verwende ConfidenceManager für Graph-Traversal-Confidence
                # Nutze die Confidence des besten (kürzesten) Pfads
                best_path = paths[0] if paths else None
                best_path.confidence if best_path else 1.0

                # Zusätzliche Validierung durch ConfidenceManager
                confidence_metrics = (
                    self.confidence_manager.calculate_graph_traversal_confidence(
                        [path.confidence for path in paths[:5]]  # Betrachte Top 5 Pfade
                    )
                )

                logger.info(
                    f"[Graph-Traversal] [OK] {len(inferred_facts[relation_type])} Fakten gefunden via Traversal "
                    f"(Confidence: {confidence_metrics.value:.2f})"
                )

                # Tracke in Working Memory
                self.working_memory.add_reasoning_state(
                    step_type="graph_traversal_success",
                    description=f"Graph-Traversal erfolgreich für '{topic}'",
                    data={
                        "topic": topic,
                        "method": "graph_traversal",
                        "relation_type": relation_type,
                        "num_paths": len(paths),
                        "inferred_facts": inferred_facts,
                        "confidence_explanation": confidence_metrics.explanation,
                    },
                    confidence=confidence_metrics.value,
                )

                # PHASE 2 (Proof Tree): Generiere ProofTree für Graph-Traversal
                if PROOF_SYSTEM_AVAILABLE:
                    try:
                        proof_tree = ProofTree(query=f"Was ist ein {topic}?")
                        # Konvertiere Pfade zu ProofSteps
                        for path in paths[:5]:  # Zeige nur erste 5 Pfade
                            proof_step = (
                                self.graph_traversal.create_proof_step_from_path(
                                    path, query=f"{topic} {relation_type}"
                                )
                            )
                            if proof_step:
                                proof_tree.add_root_step(proof_step)

                        # Emittiere ProofTree an UI
                        self.signals.proof_tree_update.emit(proof_tree)
                        logger.debug(
                            f"[Proof Tree] Graph-Traversal ProofTree emittiert ({len(proof_tree.root_steps)} Pfade)"
                        )
                    except InferenceError as e:
                        logger.warning(
                            f"[Proof Tree] InferenceError beim Generieren des ProofTree: {e}",
                            exc_info=True,
                        )
                        user_msg = get_user_friendly_message(e)
                        logger.info(f"[Proof Tree] User-Message: {user_msg}")
                    except Exception as e:
                        logger.warning(
                            f"[Proof Tree] Unerwarteter Fehler beim Generieren des ProofTree: {e}",
                            exc_info=True,
                        )

                return {
                    "inferred_facts": inferred_facts,
                    "proof_trace": proof_trace,
                    "confidence": confidence_metrics.value,
                }
            else:
                logger.info(
                    "[Graph-Traversal] [X] Keine transitiven Relationen gefunden"
                )

        except GraphTraversalError as e:
            # Spezifischer Fehler bei Graph-Traversierung
            logger.warning(f"[Graph-Traversal] GraphTraversalError: {e}", exc_info=True)
            # Benutzerfreundliche Nachricht loggen
            user_msg = get_user_friendly_message(e)
            logger.info(f"[Graph-Traversal] User-Message: {user_msg}")
            # Graceful Degradation: Fallback auf Backward-Chaining
        except Exception as e:
            # Unerwarteter Fehler - wrap in GraphTraversalError
            logger.warning(
                f"[Graph-Traversal] Unerwarteter Fehler: {e}, falle zurück auf Backward-Chaining",
                exc_info=True,
            )

        return None

    def _try_backward_chaining(
        self, topic: str, relation_type: str
    ) -> Optional[Dict[str, Any]]:
        """
        Versucht regelbasiertes Backward-Chaining.

        Args:
            topic: Das Thema der Frage
            relation_type: Der Typ der gesuchten Relation

        Returns:
            Dictionary mit Ergebnissen oder None
        """
        logger.info("[Backward-Chaining] Versuche regelbasiertes Backward-Chaining...")

        # Erstelle Goal für Backward-Chaining
        # Beispiel: "Was ist ein Hund?" -> Goal: IS_A(hund, ?x)
        goal = Goal(
            pred=relation_type,
            args={"subject": topic.lower(), "object": None},  # Object unbekannt
        )

        # Lade bekannte Fakten in die Engine
        # Hole alle Fakten aus dem Graph und wandle sie in Engine-Facts um
        all_facts = self.load_facts_from_graph(topic)

        for fact in all_facts:
            self.engine.add_fact(fact)

        # EPISODIC MEMORY FOR REASONING: Verwende tracked version
        # Dies erstellt eine InferenceEpisode und persistiert den Beweisbaum
        query_text = f"Was ist ein {topic}?"
        proof = self.engine.run_with_tracking(
            goal=goal, inference_type="backward_chaining", query=query_text, max_depth=5
        )

        if proof:
            logger.info(f"[Backward-Chaining] [OK] Beweis gefunden für '{topic}'")

            # Extrahiere abgeleitete Fakten aus dem Beweis
            inferred_facts = self.extract_facts_from_proof(proof)
            proof_trace = self.engine.format_proof_trace(proof)

            # Tracke in Working Memory
            self.working_memory.add_reasoning_state(
                step_type="backward_chaining_success",
                description=f"Multi-Hop-Schlussfolgerung erfolgreich für '{topic}'",
                data={
                    "topic": topic,
                    "method": proof.method,
                    "confidence": proof.confidence,
                    "inferred_facts": inferred_facts,
                },
                confidence=proof.confidence,
            )

            # PHASE 2 (Proof Tree): Generiere ProofTree für Backward-Chaining
            if PROOF_SYSTEM_AVAILABLE:
                try:
                    proof_tree = create_proof_tree_from_logic_engine(
                        proof, query=query_text
                    )
                    # Emittiere ProofTree an UI
                    self.signals.proof_tree_update.emit(proof_tree)
                    logger.debug(
                        f"[Proof Tree] Backward-Chaining ProofTree emittiert ({len(proof_tree.get_all_steps())} Schritte)"
                    )
                except InferenceError as e:
                    logger.warning(
                        f"[Proof Tree] InferenceError beim Generieren des ProofTree: {e}",
                        exc_info=True,
                    )
                    user_msg = get_user_friendly_message(e)
                    logger.info(f"[Proof Tree] User-Message: {user_msg}")
                except Exception as e:
                    logger.warning(
                        f"[Proof Tree] Unerwarteter Fehler beim Generieren des ProofTree: {e}",
                        exc_info=True,
                    )

            return {
                "inferred_facts": inferred_facts,
                "proof_trace": proof_trace,
                "confidence": proof.confidence,
            }

        logger.info(
            f"[Backward-Chaining] [X] Keine Schlussfolgerung möglich für '{topic}'"
        )
        return None

    def _try_abductive_reasoning(
        self, topic: str, relation_type: str
    ) -> Optional[Dict[str, Any]]:
        """
        Versucht Abductive Reasoning zur Hypothesengenerierung.

        Args:
            topic: Das Thema der Frage
            relation_type: Der Typ der gesuchten Relation

        Returns:
            Dictionary mit Ergebnissen (markiert als "is_hypothesis") oder None
        """
        logger.info(
            f"[Abductive Reasoning] Versuche Hypothesen-Generierung für '{topic}'"
        )

        if not self.abductive_engine:
            logger.warning("[Abductive Reasoning] Engine nicht verfügbar")
            return None

        try:
            # Lade Kontextfakten für Hypothesengenerierung
            all_facts = self.load_facts_from_graph(topic)

            # Generiere Hypothesen
            observation = f"Beobachtung: Es wurde nach '{topic}' gefragt"
            hypotheses = self.abductive_engine.generate_hypotheses(
                observation=observation,
                context_facts=all_facts,
                strategies=["template", "analogy", "causal_chain"],
                max_hypotheses=5,
            )

            if hypotheses:
                # Nehme die beste Hypothese
                best_hypothesis = hypotheses[0]

                logger.info(
                    f"[Abductive Reasoning] [OK] Hypothese generiert: {best_hypothesis.explanation} "
                    f"(Konfidenz: {best_hypothesis.confidence:.2f})"
                )

                # Speichere Hypothese in Neo4j
                self.netzwerk.store_hypothesis(
                    hypothesis_id=best_hypothesis.id,
                    explanation=best_hypothesis.explanation,
                    observations=best_hypothesis.observations,
                    strategy=best_hypothesis.strategy,
                    confidence=best_hypothesis.confidence,
                    scores=best_hypothesis.scores,
                    abduced_facts=[
                        {"pred": f.pred, "args": f.args, "confidence": f.confidence}
                        for f in best_hypothesis.abduced_facts
                    ],
                    sources=best_hypothesis.sources,
                    reasoning_trace=best_hypothesis.reasoning_trace,
                )

                # Verknüpfe mit Konzept
                self.netzwerk.link_hypothesis_to_concepts(best_hypothesis.id, [topic])

                # Verknüpfe mit Beobachtungen
                self.netzwerk.link_hypothesis_to_observations(
                    best_hypothesis.id, best_hypothesis.observations
                )

                # Extrahiere abgeleitete Fakten
                inferred_facts = {}
                for fact in best_hypothesis.abduced_facts:
                    rel = fact.pred
                    obj = fact.args.get("object", "")
                    if rel not in inferred_facts:
                        inferred_facts[rel] = []
                    if obj and obj not in inferred_facts[rel]:
                        inferred_facts[rel].append(obj)

                # Formatiere Erklärung
                proof_trace = (
                    f"Abductive Reasoning ({best_hypothesis.strategy}):\n"
                    f"{best_hypothesis.explanation}\n"
                    f"Konfidenz: {best_hypothesis.confidence:.2f}\n"
                    f"Bewertung: Coverage={best_hypothesis.scores.get('coverage', 0):.2f}, "
                    f"Simplicity={best_hypothesis.scores.get('simplicity', 0):.2f}, "
                    f"Coherence={best_hypothesis.scores.get('coherence', 0):.2f}, "
                    f"Specificity={best_hypothesis.scores.get('specificity', 0):.2f}"
                )

                # Tracke in Working Memory
                self.working_memory.add_reasoning_state(
                    step_type="abductive_reasoning_success",
                    description=f"Abductive Reasoning erfolgreich für '{topic}'",
                    data={
                        "topic": topic,
                        "method": "abductive",
                        "strategy": best_hypothesis.strategy,
                        "confidence": best_hypothesis.confidence,
                        "hypotheses_generated": len(hypotheses),
                        "inferred_facts": inferred_facts,
                    },
                    confidence=best_hypothesis.confidence,
                )

                # PHASE 2 (Proof Tree): Generiere ProofTree für Abductive Reasoning
                if PROOF_SYSTEM_AVAILABLE:
                    try:
                        proof_tree = ProofTree(query=observation)
                        # Konvertiere alle Hypothesen zu ProofSteps
                        hypothesis_steps = (
                            self.abductive_engine.create_multi_hypothesis_proof_chain(
                                hypotheses[:3],  # Zeige nur Top 3 Hypothesen
                                query=observation,
                            )
                        )
                        for step in hypothesis_steps:
                            proof_tree.add_root_step(step)

                        # Emittiere ProofTree an UI
                        self.signals.proof_tree_update.emit(proof_tree)
                        logger.debug(
                            f"[Proof Tree] Abductive Reasoning ProofTree emittiert ({len(hypothesis_steps)} Hypothesen)"
                        )
                    except InferenceError as e:
                        logger.warning(
                            f"[Proof Tree] InferenceError beim Generieren des ProofTree: {e}",
                            exc_info=True,
                        )
                        user_msg = get_user_friendly_message(e)
                        logger.info(f"[Proof Tree] User-Message: {user_msg}")
                    except Exception as e:
                        logger.warning(
                            f"[Proof Tree] Unerwarteter Fehler beim Generieren des ProofTree: {e}",
                            exc_info=True,
                        )

                return {
                    "inferred_facts": inferred_facts,
                    "proof_trace": proof_trace,
                    "confidence": best_hypothesis.confidence,
                    "is_hypothesis": True,  # Markierung für Antwortgenerierung
                }
            else:
                logger.info(
                    f"[Abductive Reasoning] [X] Keine Hypothesen generiert für '{topic}'"
                )

        except AbductiveReasoningError as e:
            # Spezifischer Fehler beim Abductive Reasoning
            logger.warning(
                f"[Abductive Reasoning] AbductiveReasoningError: {e}", exc_info=True
            )
            user_msg = get_user_friendly_message(e)
            logger.info(f"[Abductive Reasoning] User-Message: {user_msg}")
        except Exception as e:
            # Wrap unerwarteter Fehler in AbductiveReasoningError
            logger.warning(
                f"[Abductive Reasoning] Unerwarteter Fehler: {e}", exc_info=True
            )

        return None

    def load_facts_from_graph(self, topic: str) -> List[Fact]:
        """
        Lädt relevante Fakten aus dem Neo4j-Graphen und wandelt sie in Engine-Facts um.

        UPDATED: Wendet Confidence-Decay für zeitbasierte Reduktion veralteter Fakten an

        Args:
            topic: Das zentrale Thema, für das Fakten geladen werden sollen

        Returns:
            Liste von Fact-Objekten mit angewendetem Confidence-Decay
        """
        facts = []

        # Lade direkte Fakten über das Thema MIT Confidence und Timestamps
        # FALLBACK: query_graph_for_facts_with_confidence existiert noch nicht in KonzeptNetzwerk
        # TODO: Implementiere diese Methode in component_1_netzwerk_core.py
        # Für jetzt: verwende query_graph_for_facts mit Dummy-Confidence
        try:
            fact_data_with_confidence = (
                self.netzwerk.query_graph_for_facts_with_confidence(topic)
            )
        except AttributeError:
            # Fallback: Konvertiere normale Facts zu Format mit Confidence
            logger.debug(
                "[Confidence-Decay] query_graph_for_facts_with_confidence nicht verfügbar, verwende Fallback"
            )
            fact_data = self.netzwerk.query_graph_for_facts(topic)
            fact_data_with_confidence = {}
            for relation_type, objects in fact_data.items():
                fact_data_with_confidence[relation_type] = [
                    {"target": obj, "confidence": 1.0, "timestamp": None}
                    for obj in objects
                ]

        for relation_type, targets_with_conf in fact_data_with_confidence.items():
            for target_info in targets_with_conf:
                target = target_info.get("target", "")
                fact_confidence = target_info.get("confidence", 1.0)
                # Timestamp falls vorhanden (TODO: Neo4j muss timestamps speichern)
                timestamp_str = target_info.get("timestamp")

                # PHASE: Confidence-Based Learning - Wende Confidence-Decay an
                if timestamp_str:
                    try:
                        # Parse timestamp (ISO format erwartet)
                        fact_timestamp = datetime.fromisoformat(timestamp_str)

                        # Wende Decay an
                        decay_metrics = self.confidence_manager.apply_decay(
                            fact_confidence, fact_timestamp
                        )

                        final_confidence = decay_metrics.value

                        if decay_metrics.decay_applied:
                            logger.debug(
                                f"[Confidence-Decay] {relation_type}({topic}, {target}): "
                                f"{fact_confidence:.3f} -> {final_confidence:.3f}"
                            )
                    except (ValueError, AttributeError) as e:
                        # Fallback wenn Timestamp nicht parsebar
                        logger.warning(
                            f"Konnte Timestamp nicht parsen für {topic}-{target}: {e}"
                        )
                        final_confidence = fact_confidence
                else:
                    # Kein Timestamp vorhanden - verwende ursprüngliche Confidence
                    final_confidence = fact_confidence

                fact = Fact(
                    pred=relation_type,
                    args={"subject": topic.lower(), "object": target.lower()},
                    confidence=final_confidence,
                    source="graph",
                )
                facts.append(fact)

        # Lade auch verwandte Fakten (1-Hop Nachbarn)
        # Dies hilft bei Multi-Hop-Reasoning
        for relation_type, targets_with_conf in fact_data_with_confidence.items():
            for target_info in targets_with_conf:
                target = target_info.get("target", "")
                # Lade Fakten über das Ziel-Objekt MIT Confidence
                try:
                    obj_facts_with_conf = (
                        self.netzwerk.query_graph_for_facts_with_confidence(target)
                    )
                except AttributeError:
                    # Fallback wie oben
                    obj_fact_data = self.netzwerk.query_graph_for_facts(target)
                    obj_facts_with_conf = {}
                    for obj_relation, obj_objects in obj_fact_data.items():
                        obj_facts_with_conf[obj_relation] = [
                            {"target": obj, "confidence": 1.0, "timestamp": None}
                            for obj in obj_objects
                        ]

                for obj_relation, obj_targets_with_conf in obj_facts_with_conf.items():
                    for obj_target_info in obj_targets_with_conf:
                        obj_target = obj_target_info.get("target", "")
                        obj_confidence = obj_target_info.get("confidence", 1.0)
                        obj_timestamp_str = obj_target_info.get("timestamp")

                        # PHASE: Confidence-Based Learning - Wende Decay auch hier an
                        if obj_timestamp_str:
                            try:
                                obj_timestamp = datetime.fromisoformat(
                                    obj_timestamp_str
                                )
                                decay_metrics = self.confidence_manager.apply_decay(
                                    obj_confidence, obj_timestamp
                                )
                                final_obj_confidence = decay_metrics.value
                            except (ValueError, AttributeError):
                                final_obj_confidence = obj_confidence
                        else:
                            final_obj_confidence = obj_confidence

                        fact = Fact(
                            pred=obj_relation,
                            args={
                                "subject": target.lower(),
                                "object": obj_target.lower(),
                            },
                            confidence=final_obj_confidence,
                            source="graph",
                        )
                        facts.append(fact)

        logger.debug(f"[Backward-Chaining] Geladen: {len(facts)} Fakten für '{topic}'")
        return facts

    def extract_facts_from_proof(self, proof) -> Dict[str, List[str]]:
        """
        Extrahiert strukturierte Fakten aus einem ProofStep.

        Args:
            proof: ProofStep-Objekt mit Beweisbaum

        Returns:
            Dictionary mit Relation-Types und zugehörigen Objekten
        """
        facts = {}

        # Sammle Fakten aus dem Beweis
        if proof.supporting_facts:
            for fact in proof.supporting_facts:
                relation = fact.pred
                obj = fact.args.get("object", "")

                if relation not in facts:
                    facts[relation] = []

                if obj and obj not in facts[relation]:
                    facts[relation].append(obj)

        # Rekursiv durch Subgoals
        for subproof in proof.subgoals:
            subfacts = self.extract_facts_from_proof(subproof)
            for relation, objects in subfacts.items():
                if relation not in facts:
                    facts[relation] = []
                facts[relation].extend([o for o in objects if o not in facts[relation]])

        return facts

    # ==================== RESONANCE-BASED QUESTION ANSWERING ====================

    def _handle_resonance_inference(
        self, query: str, context: Optional[Dict] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Beantwortet Frage via Activation Spreading (Resonance-Based QA).

        Example: "Kann ein Pinguin fliegen?"
        1. Extrahiere Konzepte: ["Pinguin", "fliegen"]
        2. Aktiviere beide Konzepte parallel
        3. Finde Überschneidungen in Activation Maps
        4. Prüfe auf Widersprüche (CAPABLE_OF vs. NOT_CAPABLE_OF)
        5. Generiere Antwort basierend auf Resonance

        Args:
            query: Die Frage als Text
            context: Optional context dict

        Returns:
            Dictionary mit answer, proof_tree, activation_maps, confidence
            oder None wenn Resonance-Inference nicht möglich
        """
        if not self._resonance_engine:
            logger.warning("[Resonance QA] ResonanceEngine nicht verfügbar")
            return None

        if context is None:
            context = {}

        logger.info(f"[Resonance QA] Starte Resonance-Based QA für: '{query}'")

        try:
            # 1. Extrahiere Key-Konzepte aus der Query
            concepts = self._extract_key_concepts(query)

            if not concepts:
                logger.warning("[Resonance QA] Keine Konzepte extrahiert")
                return None

            logger.debug(f"[Resonance QA] Extrahierte Konzepte: {concepts}")

            # 2. Aktiviere alle Konzepte parallel
            activation_maps = {}
            for concept in concepts:
                try:
                    activation_map = self._resonance_engine.activate_concept(
                        start_word=concept, query_context=context
                    )
                    activation_maps[concept] = activation_map
                    logger.debug(
                        f"[Resonance QA] {concept}: {activation_map.concepts_activated} Konzepte aktiviert, "
                        f"{len(activation_map.resonance_points)} Resonanz-Punkte"
                    )
                except Exception as e:
                    logger.warning(
                        f"[Resonance QA] Aktivierung fehlgeschlagen für '{concept}': {e}"
                    )
                    continue

            if not activation_maps:
                logger.warning("[Resonance QA] Keine Activation Maps erstellt")
                return None

            # 3. Finde semantische Überschneidungen
            overlap = self._find_activation_overlap(activation_maps)

            logger.debug(
                f"[Resonance QA] Overlap gefunden: {len(overlap)} gemeinsame Konzepte"
            )

            # 4. Prüfe auf Widersprüche
            contradictions = self._detect_contradictions(
                overlap, concepts, activation_maps
            )

            # 5. Reasoning basierend auf Overlap
            if overlap or contradictions:
                # Konzepte sind semantisch verbunden
                answer = self._generate_answer_from_overlap(
                    overlap, contradictions, query, concepts
                )
                confidence = self._calculate_resonance_confidence(
                    overlap, contradictions, activation_maps
                )
            else:
                # Keine Verbindung gefunden
                answer = self._generate_negative_answer(concepts)
                confidence = 0.3  # Low confidence für negative Antwort

            # 6. Build Proof Tree aus Activation Paths
            proof_tree = None
            if PROOF_SYSTEM_AVAILABLE:
                try:
                    proof_tree = self._build_proof_from_activation(
                        activation_maps, overlap, contradictions, query, answer
                    )
                    # Emittiere an UI
                    self.signals.proof_tree_update.emit(proof_tree)
                except Exception as e:
                    logger.warning(
                        f"[Resonance QA] Proof Tree Generierung fehlgeschlagen: {e}"
                    )

            # 7. Tracke in Working Memory
            self.working_memory.add_reasoning_state(
                step_type="resonance_qa",
                description=f"Resonance-Based QA für '{query}'",
                data={
                    "query": query,
                    "concepts": concepts,
                    "overlap_size": len(overlap),
                    "contradictions": len(contradictions),
                    "answer": answer,
                },
                confidence=confidence,
            )

            logger.info(
                f"[Resonance QA] [OK] Antwort generiert (Konfidenz: {confidence:.2f})"
            )

            return {
                "answer": answer,
                "proof_tree": proof_tree,
                "activation_maps": activation_maps,
                "confidence": confidence,
                "overlap": overlap,
                "contradictions": contradictions,
            }

        except Exception as e:
            logger.error(f"[Resonance QA] Fehler: {e}", exc_info=True)
            return None

    def _extract_key_concepts(self, query: str) -> List[str]:
        """
        Extrahiert Schlüssel-Konzepte aus einer Query.

        Uses spaCy linguistic engine to extract nouns and verbs.

        Args:
            query: Die Frage als Text

        Returns:
            Liste von Konzepten (lemmatisiert, lowercase)
        """
        try:
            # Nutze linguistik_engine falls verfügbar
            if hasattr(self, "linguistik_engine"):
                # Parse query
                doc = self.linguistik_engine.nlp(query)

                # Extrahiere Nomen und Verben
                concepts = []
                for token in doc:
                    # Nur Nomen und Verben
                    if token.pos_ in ["NOUN", "VERB", "PROPN"]:
                        # Lemmatisiert und lowercase
                        lemma = token.lemma_.lower()
                        if lemma and len(lemma) > 2:  # Min 3 Zeichen
                            concepts.append(lemma)

                return concepts

            else:
                # Fallback: Einfaches Word-Splitting
                logger.warning(
                    "[Resonance QA] Linguistik-Engine nicht verfügbar, verwende Fallback"
                )
                import string

                # Entferne Satzzeichen
                query_clean = query.translate(str.maketrans("", "", string.punctuation))
                words = query_clean.lower().split()
                # Filtere Stopwords (einfach)
                stopwords = {
                    "ein",
                    "eine",
                    "der",
                    "die",
                    "das",
                    "ist",
                    "kann",
                    "hat",
                    "was",
                    "wie",
                    "wer",
                    "wo",
                    "wann",
                    "warum",
                }
                concepts = [w for w in words if w not in stopwords and len(w) > 2]
                return concepts

        except Exception as e:
            logger.warning(f"[Resonance QA] Konzept-Extraktion fehlgeschlagen: {e}")
            return []

    def _find_activation_overlap(
        self, activation_maps: Dict[str, Any]
    ) -> Dict[str, Dict[str, float]]:
        """
        Findet semantische Überschneidungen zwischen Activation Maps.

        Args:
            activation_maps: Dict mit concept -> ActivationMap

        Returns:
            Dict mit overlapping concepts und deren Aktivierungen
            Format: {concept: {source1: activation1, source2: activation2, ...}}
        """
        if len(activation_maps) < 2:
            # Kein Overlap möglich mit nur einer Map
            return {}

        # Sammle alle aktivierten Konzepte pro Source
        concept_sources = {}  # concept -> {source: activation}

        for source_concept, activation_map in activation_maps.items():
            for concept, activation in activation_map.activations.items():
                if concept not in concept_sources:
                    concept_sources[concept] = {}
                concept_sources[concept][source_concept] = activation

        # Finde Konzepte die in mehreren Maps vorkommen
        overlap = {}
        for concept, sources in concept_sources.items():
            if len(sources) >= 2:  # Mindestens 2 Sources
                overlap[concept] = sources

        return overlap

    def _detect_contradictions(
        self,
        overlap: Dict[str, Dict[str, float]],
        query_concepts: List[str],
        activation_maps: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Erkennt Widersprüche in den Activation Maps.

        Prüft auf:
        - CAPABLE_OF vs. NOT_CAPABLE_OF
        - HAS_PROPERTY vs. NOT_HAS_PROPERTY
        - IS_A vs. NOT_IS_A

        Args:
            overlap: Overlapping concepts
            query_concepts: Konzepte aus der Query
            activation_maps: Alle Activation Maps

        Returns:
            Liste von Widersprüchen
        """
        contradictions = []

        # Prüfe für jedes Query-Konzept ob es widersprüchliche Paths gibt
        for concept in query_concepts:
            if concept not in activation_maps:
                continue

            activation_map = activation_maps[concept]

            # Sammle alle Relationstypen aus Paths
            positive_relations = {}  # target -> [paths]
            negative_relations = {}  # target -> [paths]

            for path in activation_map.reasoning_paths:
                for relation in path.relations:
                    # Prüfe ob negativ
                    is_negative = relation.startswith("NOT_")
                    _ = (
                        relation[4:] if is_negative else relation
                    )  # base_relation unused

                    target = path.target

                    if is_negative:
                        if target not in negative_relations:
                            negative_relations[target] = []
                        negative_relations[target].append(
                            {"relation": relation, "path": path, "source": concept}
                        )
                    else:
                        if target not in positive_relations:
                            positive_relations[target] = []
                        positive_relations[target].append(
                            {"relation": relation, "path": path, "source": concept}
                        )

            # Finde Widersprüche
            for target in positive_relations:
                if target in negative_relations:
                    # Widerspruch gefunden!
                    contradictions.append(
                        {
                            "concept": concept,
                            "target": target,
                            "positive_paths": positive_relations[target],
                            "negative_paths": negative_relations[target],
                            "type": "relation_negation",
                        }
                    )

        return contradictions

    def _generate_answer_from_overlap(
        self,
        overlap: Dict[str, Dict[str, float]],
        contradictions: List[Dict[str, Any]],
        query: str,
        concepts: List[str],
    ) -> str:
        """
        Generiert natürlichsprachliche Antwort aus Overlap und Widersprüchen.

        Args:
            overlap: Overlapping concepts
            contradictions: Erkannte Widersprüche
            query: Ursprüngliche Frage
            concepts: Extrahierte Konzepte

        Returns:
            Deutsche Antwort-Text
        """
        # Prüfe ob Ja/Nein-Frage
        is_yes_no = any(
            word in query.lower() for word in ["kann", "ist", "hat", "sind"]
        )

        # Fall 1: Widersprüche gefunden
        if contradictions:
            # Nehme ersten Widerspruch
            contr = contradictions[0]

            # Extrahiere Details
            concept = contr["concept"]
            target = contr["target"]
            _ = contr["positive_paths"]  # unused
            neg_paths = contr["negative_paths"]

            # Bevorzuge negative Paths (spezifischer)
            if neg_paths:
                neg_relation = neg_paths[0]["relation"]
                _ = (
                    neg_relation[4:]
                    if neg_relation.startswith("NOT_")
                    else neg_relation
                )  # base_relation unused

                answer = f"Nein, {concept} kann nicht {target}. "
                answer += f"Obwohl {concept} generell zur Kategorie gehören könnte, "
                answer += "gibt es eine explizite Ausnahme."
            else:
                answer = f"Es gibt widersprüchliche Informationen über {concept} und {target}."

            return answer

        # Fall 2: Starker Overlap (Konzepte sind verbunden)
        if overlap:
            # Finde Konzept mit höchster kombinierter Aktivierung
            best_overlap = max(overlap.items(), key=lambda x: sum(x[1].values()))
            overlap_concept, sources = best_overlap

            if is_yes_no:
                # Finde Pfade die die Relation bestätigen
                # (vereinfacht: schaue ob overlap_concept relevant ist)
                if any(concept in sources for concept in concepts):
                    answer = f"Ja, basierend auf semantischen Verbindungen über '{overlap_concept}'. "
                    answer += f"Aktivierung: {sum(sources.values()):.2f}"
                else:
                    answer = f"Die Konzepte sind semantisch verbunden über '{overlap_concept}', "
                    answer += "aber es gibt keine direkte Bestätigung."
            else:
                answer = (
                    f"Die Konzepte sind semantisch verbunden über '{overlap_concept}'. "
                )
                answer += f"{len(overlap)} gemeinsame Konzepte gefunden."

            return answer

        # Fall 3: Kein Overlap
        return self._generate_negative_answer(concepts)

    def _generate_negative_answer(self, concepts: List[str]) -> str:
        """
        Generiert negative Antwort bei fehlender semantischer Verbindung.

        Args:
            concepts: Die Query-Konzepte

        Returns:
            Deutsche Antwort
        """
        if len(concepts) >= 2:
            return f"Ich habe keine semantische Verbindung zwischen {' und '.join(concepts)} gefunden."
        elif concepts:
            return f"Ich habe nicht genügend Informationen über '{concepts[0]}'."
        else:
            return "Ich konnte die Frage nicht verstehen."

    def _build_proof_from_activation(
        self,
        activation_maps: Dict[str, Any],
        overlap: Dict[str, Dict[str, float]],
        contradictions: List[Dict[str, Any]],
        query: str,
        answer: str,
    ) -> Optional[Any]:
        """
        Baut Proof Tree aus Activation Maps.

        Args:
            activation_maps: Alle Activation Maps
            overlap: Overlapping concepts
            contradictions: Widersprüche
            query: Original-Query
            answer: Generierte Antwort

        Returns:
            ProofTree oder None
        """
        if not PROOF_SYSTEM_AVAILABLE:
            return None

        try:
            from component_17_proof_explanation import ProofStep, ProofTree, StepType

            proof_tree = ProofTree(query=query)

            # Root Step: Query
            root_step = ProofStep(
                step_id="resonance_root",
                step_type=StepType.QUERY,
                inputs=list(activation_maps.keys()),
                output=answer,
                confidence=self._calculate_resonance_confidence(
                    overlap, contradictions, activation_maps
                ),
                explanation_text=f"Resonance-Based Question Answering für: {query}",
                source_component="resonance_qa",
                metadata={
                    "concepts": list(activation_maps.keys()),
                    "overlap_size": len(overlap),
                    "contradictions": len(contradictions),
                },
            )
            proof_tree.add_root_step(root_step)

            # Add Activation Summaries
            for concept, activation_map in activation_maps.items():
                summary = self._resonance_engine.get_activation_summary(activation_map)

                activation_step = ProofStep(
                    step_id=f"activation_{concept}",
                    step_type=StepType.INFERENCE,
                    inputs=[concept],
                    output=f"{activation_map.concepts_activated} Konzepte aktiviert",
                    confidence=min(activation_map.max_activation, 1.0),
                    explanation_text=summary,
                    source_component="resonance_engine",
                    metadata={
                        "waves": activation_map.waves_executed,
                        "resonance_points": len(activation_map.resonance_points),
                    },
                )
                proof_tree.add_child_step("resonance_root", activation_step)

                # Add Resonance Points
                for rp in activation_map.resonance_points[:3]:  # Top 3
                    explanation = self._resonance_engine.explain_activation(
                        rp.concept, activation_map, max_paths=2
                    )
                    rp_step = ProofStep(
                        step_id=f"resonance_{concept}_{rp.concept}",
                        step_type=StepType.FACT_MATCH,
                        inputs=[concept],
                        output=f"{rp.concept} (Resonanz)",
                        confidence=min(rp.resonance_boost, 1.0),
                        explanation_text=explanation,
                        source_component="resonance_engine",
                        metadata={"num_paths": rp.num_paths},
                    )
                    proof_tree.add_child_step(f"activation_{concept}", rp_step)

            # Add Contradictions if any
            for i, contr in enumerate(contradictions[:2]):  # Max 2
                contr_step = ProofStep(
                    step_id=f"contradiction_{i}",
                    step_type=StepType.CONSTRAINT,
                    inputs=[contr["concept"], contr["target"]],
                    output=f"Widerspruch erkannt: {contr['type']}",
                    confidence=0.9,
                    explanation_text=f"Positive und negative Relationen zu {contr['target']} gefunden",
                    source_component="resonance_qa",
                    metadata=contr,
                )
                proof_tree.add_child_step("resonance_root", contr_step)

            return proof_tree

        except Exception as e:
            logger.warning(f"[Resonance QA] Proof Tree Build fehlgeschlagen: {e}")
            return None

    def _calculate_resonance_confidence(
        self,
        overlap: Dict[str, Dict[str, float]],
        contradictions: List[Dict[str, Any]],
        activation_maps: Dict[str, Any],
    ) -> float:
        """
        Berechnet Confidence-Score aus Resonance-Daten.

        Args:
            overlap: Overlapping concepts
            contradictions: Widersprüche
            activation_maps: Alle Activation Maps

        Returns:
            Confidence zwischen 0.0 und 1.0
        """
        # Base Confidence aus Overlap-Größe
        if not overlap:
            base_conf = 0.3
        else:
            # Normalisiere auf Anzahl Query-Konzepte
            overlap_ratio = len(overlap) / max(len(activation_maps), 1)
            base_conf = 0.5 + (overlap_ratio * 0.3)  # 0.5 bis 0.8

        # Boost für Widersprüche (höhere Confidence da spezifisch)
        if contradictions:
            base_conf = min(base_conf + 0.15, 0.95)

        # Boost für Resonanz-Punkte
        total_resonance_points = sum(
            len(am.resonance_points) for am in activation_maps.values()
        )
        if total_resonance_points > 0:
            resonance_boost = min(total_resonance_points * 0.05, 0.15)
            base_conf = min(base_conf + resonance_boost, 0.95)

        # Berücksichtige max Aktivierung
        max_activation = max(
            (am.max_activation for am in activation_maps.values()), default=0.0
        )
        if max_activation > 1.0:
            # Hohe Aktivierung deutet auf starke Verbindung
            activation_boost = min((max_activation - 1.0) * 0.1, 0.1)
            base_conf = min(base_conf + activation_boost, 0.95)

        return base_conf

    def _construct_query_from_topic(self, topic: str, relation_type: str) -> str:
        """
        Konstruiert natürlichsprachliche Query aus Topic und Relation.

        Args:
            topic: Das Thema (z.B. "pinguin")
            relation_type: Der Relationstyp (z.B. "CAPABLE_OF", "IS_A")

        Returns:
            Deutsche Frage als Text
        """
        topic_lower = topic.lower()

        # Mapping von Relationen zu Fragewörtern
        if relation_type == "IS_A":
            return f"Was ist ein {topic_lower}?"
        elif relation_type == "CAPABLE_OF":
            return f"Was kann ein {topic_lower}?"
        elif relation_type == "HAS_PROPERTY":
            return f"Welche Eigenschaften hat ein {topic_lower}?"
        elif relation_type == "PART_OF":
            return f"Teil von was ist {topic_lower}?"
        elif relation_type == "LOCATED_IN":
            return f"Wo befindet sich {topic_lower}?"
        else:
            # Fallback: Generic question
            return f"Was weißt du über {topic_lower}?"
