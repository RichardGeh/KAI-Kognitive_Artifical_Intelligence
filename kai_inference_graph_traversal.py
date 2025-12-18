# kai_inference_graph_traversal.py
"""
Graph Traversal Handler für KAI Inference System

Verantwortlichkeiten:
- Multi-Hop Graph Traversal für transitive Relationen
- Confidence-Berechnung für Pfade
- ProofTree-Generierung aus Graph-Pfaden
"""
import logging
from typing import Any, Dict, Optional

from component_1_netzwerk import KonzeptNetzwerk
from component_confidence_manager import get_confidence_manager
from kai_exceptions import (
    GraphTraversalError,
    InferenceError,
    get_user_friendly_message,
)

logger = logging.getLogger(__name__)

# Import Unified Proof Explanation System
try:
    from component_17_proof_explanation import ProofTree

    PROOF_SYSTEM_AVAILABLE = True
except ImportError:
    PROOF_SYSTEM_AVAILABLE = False
    logger.warning("Unified Proof Explanation System nicht verfügbar")


class GraphTraversalHandler:
    """
    Handler für Graph-Traversal basiertes Multi-Hop-Reasoning.

    Verwaltet:
    - Transitive Relationen-Suche
    - Pfad-basierte Confidence-Berechnung
    - ProofTree-Generierung
    """

    def __init__(
        self,
        netzwerk: KonzeptNetzwerk,
        graph_traversal,  # GraphTraversal
        working_memory,  # WorkingMemory
        signals,  # KaiSignals
    ):
        """
        Initialisiert den Graph-Traversal Handler.

        Args:
            netzwerk: KonzeptNetzwerk für Datenspeicherung
            graph_traversal: GraphTraversal-Engine für Multi-Hop-Reasoning
            working_memory: WorkingMemory für Reasoning-Trace
            signals: KaiSignals für UI-Kommunikation
        """
        self.netzwerk = netzwerk
        self.graph_traversal = graph_traversal
        self.working_memory = working_memory
        self.signals = signals

        # PHASE: Confidence-Based Learning
        self.confidence_manager = get_confidence_manager()
        logger.info("GraphTraversalHandler initialisiert mit ConfidenceManager")

    def try_graph_traversal(
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
                proof_tree = None
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
                        proof_tree = None
                    except Exception as e:
                        logger.warning(
                            f"[Proof Tree] Unerwarteter Fehler beim Generieren des ProofTree: {e}",
                            exc_info=True,
                        )
                        proof_tree = None

                return {
                    "inferred_facts": inferred_facts,
                    "proof_trace": proof_trace,
                    "confidence": confidence_metrics.value,
                    "proof_tree": proof_tree,
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
