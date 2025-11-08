# component_9_logik_engine_proof.py
"""
Mixin class for proof tracking and explanation in LogikEngine.

This module contains all proof-related methods extracted from component_9_logik_engine.py:
- Proof formatting and tracing
- Goal-based reasoning with tracking
- Episodic memory integration
- Proof tree persistence
- Probabilistic reasoning with uncertainty
- Unified proof explanations
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from component_15_logging_config import get_logger
from component_17_proof_explanation import Goal, ProofStep

if TYPE_CHECKING:
    from component_5_linguistik_strukturen import Fact, Rule

# Import Probabilistic Engine for integration
try:
    from component_16_probabilistic_engine import (
        ProbabilisticFact,
        convert_rule_to_conditional,
    )

    PROBABILISTIC_AVAILABLE = True
except ImportError:
    PROBABILISTIC_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "ProbabilisticEngine nicht verfügbar - läuft im deterministischen Modus"
    )

# Import Unified Proof Explanation System
try:
    from component_17_proof_explanation import (
        ProofTree,
        StepType,
        convert_logic_engine_proof,
        format_proof_step,
        generate_explanation_text,
    )

    UNIFIED_PROOFS_AVAILABLE = True
except ImportError:
    UNIFIED_PROOFS_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "Unified Proof Explanation System nicht verfügbar"
    )

logger = get_logger(__name__)
Binding = Dict[str, Any]


class ProofTrackingMixin:
    """
    Mixin class providing proof tracking, explanation, and persistence capabilities.

    Methods are designed to be mixed into LogikEngine class.
    Requires LogikEngine to have:
    - self.netzwerk: KonzeptNetzwerk instance
    - self.rules: List[Rule]
    - self.wm: List[Fact] (working memory)
    - self.use_probabilistic: bool flag
    - self.prob_engine: ProbabilisticEngine or None
    - self.current_inference_episode_id: str
    - run(): Forward-chaining method
    - prove_goal(goal): Backward-chaining method
    """

    def format_proof_trace(self, proof: "ProofStep", indent: int = 0) -> str:
        """
        Formatiert einen ProofStep als lesbaren Reasoning-Trail.

        Args:
            proof: ProofStep to format
            indent: Indentation level for nested output

        Returns:
            Formatted proof trace string
        """
        lines = []
        prefix = "  " * indent

        # Goal
        goal_str = f"{proof.goal.pred}({', '.join(f'{k}={v}' for k, v in proof.goal.args.items())})"
        lines.append(f"{prefix}[Goal] {goal_str}")

        # Method
        if proof.method == "fact":
            fact = proof.supporting_facts[0]
            lines.append(
                f"{prefix}  [OK] Direkt durch Fakt (confidence={fact.confidence:.2f})"
            )

        elif proof.method == "rule":
            lines.append(f"{prefix}  [OK] Durch Regel: {proof.rule_id}")
            lines.append(f"{prefix}    Bindings: {proof.bindings}")
            for subproof in proof.subgoals:
                lines.append(self.format_proof_trace(subproof, indent + 2))

        elif proof.method == "graph_traversal":
            lines.append(
                f"{prefix}  [OK] Durch Graph-Traversal (confidence={proof.confidence:.2f})"
            )
            for fact in proof.supporting_facts:
                lines.append(f"{prefix}    {fact.pred}: {fact.args}")

        elif proof.method == "constraint_satisfaction":
            lines.append(
                f"{prefix}  [OK] Durch Constraint-Satisfaction (confidence={proof.confidence:.2f})"
            )
            lines.append(f"{prefix}    Bindings: {proof.bindings}")
            if proof.supporting_facts:
                lines.append(f"{prefix}    Verwendete Fakten:")
                for fact in proof.supporting_facts:
                    lines.append(f"{prefix}      - {fact.pred}: {fact.args}")

        return "\n".join(lines)

    def run_with_goal(self, goal: "Goal", max_depth: int = 5) -> Optional["ProofStep"]:
        """
        Kombiniert Forward- und Backward-Chaining.

        Phases:
        1. Forward-Chaining zum Ableiten neuer Fakten
        2. Backward-Chaining zum Beweisen des Goals

        Args:
            goal: Das zu beweisende Ziel
            max_depth: Maximale Rekursionstiefe für Backward-Chaining

        Returns:
            ProofStep mit vollständigem Beweisbaum oder None
        """
        logger.info("=== Hybrid Reasoning: Forward + Backward ===")

        # Phase 1: Forward-Chaining
        logger.info("Phase 1: Forward-Chaining...")
        self.run()  # Leitet alle ableitbaren Fakten ab

        # Phase 2: Backward-Chaining
        logger.info(f"Phase 2: Backward-Chaining für Goal: {goal.pred}")
        proof = self.prove_goal(goal, max_depth)

        if proof:
            logger.info("=== Goal erfolgreich bewiesen ===")
            logger.info(self.format_proof_trace(proof))
        else:
            logger.info("=== Goal konnte nicht bewiesen werden ===")

        return proof

    def run_with_tracking(
        self,
        goal: "Goal",
        inference_type: str = "hybrid",
        query: str = "",
        max_depth: int = 5,
    ) -> Optional["ProofStep"]:
        """
        EPISODIC MEMORY FOR REASONING: Enhanced version of run_with_goal.

        Erstellt eine InferenceEpisode und verknüpft alle verwendeten Fakten und Regeln.
        Dies ermöglicht spätere Meta-Reasoning-Abfragen wie:
        - "Wann habe ich über X nachgedacht?"
        - "Welche Fakten habe ich für diese Schlussfolgerung verwendet?"

        Args:
            goal: Das zu beweisende Ziel
            inference_type: Art der Inferenz ("forward_chaining", "backward_chaining",
                           "graph_traversal", "hybrid")
            query: Die ursprüngliche Benutzerfrage (für Erklärungen)
            max_depth: Maximale Rekursionstiefe

        Returns:
            ProofStep mit vollständigem Beweisbaum oder None
        """
        import time

        start_time = time.time()

        logger.info(f"=== Tracking-fähiges Reasoning: {inference_type} ===")

        # SCHRITT 1: Erstelle InferenceEpisode
        goal_str = f"{goal.pred}({', '.join(f'{k}={v}' for k, v in goal.args.items())})"
        metadata = {"goal": goal_str, "max_depth": max_depth, "start_time": start_time}

        self.current_inference_episode_id = self.netzwerk.create_inference_episode(
            inference_type=inference_type, query=query or goal_str, metadata=metadata
        )

        if not self.current_inference_episode_id:
            logger.warning(
                "Konnte InferenceEpisode nicht erstellen, fahre ohne Tracking fort"
            )
            # Fallback auf normale Methode
            return self.run_with_goal(goal, max_depth)

        # SCHRITT 2: Führe Reasoning durch (mit Tracking)
        proof = self.run_with_goal(goal, max_depth)

        # SCHRITT 3: Persistiere ProofStep-Hierarchie und verknüpfe mit Episode
        if proof:
            root_step_id = self._persist_proof_tree(proof, parent_step_id=None)
            if root_step_id:
                # Verknüpfe Episode mit Root-ProofStep
                self.netzwerk.link_inference_to_proof(
                    self.current_inference_episode_id, root_step_id
                )

        # SCHRITT 4: Sammle verwendete Fakten und Regeln
        used_fact_ids = self._extract_used_fact_ids(proof) if proof else []
        applied_rule_ids = self._extract_applied_rule_ids(proof) if proof else []

        # Verknüpfe mit Episode
        if used_fact_ids:
            self.netzwerk.link_inference_to_facts(
                self.current_inference_episode_id, used_fact_ids
            )

        if applied_rule_ids:
            self.netzwerk.link_inference_to_rules(
                self.current_inference_episode_id, applied_rule_ids
            )

        # SCHRITT 5: Update Metadata mit Ergebnissen
        execution_time = time.time() - start_time
        logger.info(
            f"=== Tracked Reasoning abgeschlossen in {execution_time:.2f}s ===\n"
            f"Episode ID: {self.current_inference_episode_id[:8]}\n"
            f"Fakten verwendet: {len(used_fact_ids)}\n"
            f"Regeln angewendet: {len(applied_rule_ids)}"
        )

        return proof

    def _persist_proof_tree(
        self, proof: "ProofStep", parent_step_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Persistiert rekursiv den Beweisbaum in Neo4j.

        Args:
            proof: Der zu persistierende ProofStep
            parent_step_id: ID des übergeordneten Steps (für Hierarchie)

        Returns:
            ID des erstellten ProofStep-Knotens
        """
        # Erstelle Goal-String
        goal_str = f"{proof.goal.pred}({', '.join(f'{k}={v}' for k, v in proof.goal.args.items())})"

        # Erstelle ProofStep-Knoten
        step_id = self.netzwerk.create_proof_step(
            goal=goal_str,
            method=proof.method,
            confidence=proof.confidence,
            depth=proof.goal.depth,
            bindings=proof.bindings,
            parent_step_id=parent_step_id,
        )

        if not step_id:
            logger.warning(f"Konnte ProofStep nicht persistieren: {goal_str}")
            return None

        # Rekursiv Kinder persistieren
        for subproof in proof.subgoals:
            self._persist_proof_tree(subproof, parent_step_id=step_id)

        return step_id

    def _extract_used_fact_ids(self, proof: "ProofStep") -> List[str]:
        """
        Extrahiert alle Fact-IDs, die im Beweis verwendet wurden.

        Args:
            proof: Der ProofStep

        Returns:
            Liste von Fact-IDs
        """
        fact_ids = []

        # Fakten aus diesem Step
        for fact in proof.supporting_facts:
            fact_ids.append(fact.id)

        # Rekursiv aus Subgoals
        for subproof in proof.subgoals:
            fact_ids.extend(self._extract_used_fact_ids(subproof))

        return list(set(fact_ids))  # Duplikate entfernen

    def _extract_applied_rule_ids(self, proof: "ProofStep") -> List[str]:
        """
        Extrahiert alle Regel-IDs, die im Beweis angewendet wurden.

        Args:
            proof: Der ProofStep

        Returns:
            Liste von Regel-IDs
        """
        rule_ids = []

        # Regel aus diesem Step (wenn method == "rule")
        if proof.method == "rule" and proof.rule_id:
            rule_ids.append(proof.rule_id)

        # Rekursiv aus Subgoals
        for subproof in proof.subgoals:
            rule_ids.extend(self._extract_applied_rule_ids(subproof))

        return list(set(rule_ids))  # Duplikate entfernen

    # ==================== PROBABILISTIC REASONING (Phase 4) ====================

    def run_probabilistic(self, max_iterations: int = 10) -> List[ProbabilisticFact]:
        """
        Führt probabilistisches Forward-Chaining durch.

        Nutzt die ProbabilisticEngine für Bayesian Inference mit Unsicherheitspropagierung.

        Args:
            max_iterations: Maximale Anzahl Inferenz-Runden

        Returns:
            Liste abgeleiteter probabilistischer Fakten
        """
        if not self.use_probabilistic or not self.prob_engine:
            logger.warning("Probabilistische Inferenz nicht verfügbar.")
            return []

        logger.info("=== Probabilistische Inferenz gestartet ===")

        # Synchronisiere Regeln mit ProbabilisticEngine
        for rule in self.rules:
            cond_probs = convert_rule_to_conditional(rule)
            for cp in cond_probs:
                self.prob_engine.add_conditional(cp)

        # Führe Inferenz durch
        derived_facts = self.prob_engine.infer(max_iterations)

        # Konvertiere zurück zu deterministischen Fakten (für Kompatibilität)
        from component_9_logik_engine_core import Fact

        for prob_fact in derived_facts:
            det_fact = Fact(
                pred=prob_fact.pred,
                args=prob_fact.args,
                confidence=prob_fact.probability,
                source=prob_fact.source,
                support=prob_fact.evidence,
            )
            self.wm.append(det_fact)

        logger.info(f"=== {len(derived_facts)} probabilistische Fakten abgeleitet ===")
        return derived_facts

    def query_with_uncertainty(
        self, goal: "Goal", threshold_high: float = 0.8, threshold_low: float = 0.2
    ) -> Tuple[Optional["ProofStep"], str]:
        """
        Beantwortet eine Frage mit Unsicherheitsquantifizierung.

        Kombiniert:
        1. Deterministisches Backward-Chaining
        2. Probabilistische Inferenz
        3. Uncertainty-aware Response Generation

        Args:
            goal: Das zu beweisende Ziel
            threshold_high: Schwelle für sichere Bejahung
            threshold_low: Schwelle für sichere Verneinung

        Returns:
            Tuple (ProofStep oder None, natürlichsprachliche Antwort)
        """
        logger.info(f"=== Query mit Unsicherheitsquantifizierung: {goal.pred} ===")

        # Versuch 1: Deterministisches Backward-Chaining
        proof = self.prove_goal(goal)

        if proof:
            # Erfolgreicher deterministischer Beweis
            prob = proof.confidence
            response = f"Ja, bewiesen mit Confidence {prob:.2f}."

            if self.use_probabilistic and self.prob_engine:
                # Erweitere mit probabilistischer Erklärung
                goal_sig = self._goal_signature(goal)
                prob_query, conf = self.prob_engine.query(goal_sig)
                response += f" (Probabilistische Analyse: P={prob_query:.2f}, Konfidenz={conf:.2f})"

            return (proof, response)

        # Versuch 2: Probabilistische Inferenz
        if self.use_probabilistic and self.prob_engine:
            goal_sig = self._goal_signature(goal)
            prob, conf = self.prob_engine.query(goal_sig)

            # Generiere Antwort basierend auf Wahrscheinlichkeit
            response = self.prob_engine.generate_response(
                goal_sig, threshold_high, threshold_low
            )

            # Erstelle "virtuellen" ProofStep für Anzeige
            from component_9_logik_engine_core import ProofStep

            virtual_proof = ProofStep(
                goal=goal, method="probabilistic", bindings={}, confidence=prob
            )

            return (virtual_proof, response)

        # Keine Antwort gefunden
        return (None, "Unbekannt. Keine ausreichende Evidenz vorhanden.")

    def explain_uncertainty(self, goal: "Goal") -> Dict:
        """
        Erklärt die Unsicherheit einer Schlussfolgerung.

        Zeigt:
        - Wahrscheinlichkeit
        - Konfidenz (Inverse der Varianz)
        - Verwendete Evidenzen
        - Anwendbare Regeln

        Args:
            goal: Das zu erklärende Ziel

        Returns:
            Erklärungsdictionary
        """
        if not self.use_probabilistic or not self.prob_engine:
            return {"error": "Probabilistische Inferenz nicht verfügbar"}

        goal_sig = self._goal_signature(goal)
        return self.prob_engine.explain_belief(goal_sig)

    def suggest_information_need(self, top_k: int = 5) -> List[str]:
        """
        Schlägt vor, welche Information als nächstes benötigt wird.

        Strategie: Identifiziere Fakten mit höchster Unsicherheit (niedriger Konfidenz).
        Diese sind Kandidaten für aktives Lernen.

        Args:
            top_k: Anzahl Vorschläge

        Returns:
            Liste von Propositions-IDs mit hoher Unsicherheit
        """
        if not self.use_probabilistic or not self.prob_engine:
            return []

        uncertain_facts = self.prob_engine.get_most_uncertain_facts(top_k)

        suggestions = [
            f"{prop} (P={prob:.2f}, Konfidenz={conf:.2f})"
            for prop, prob, conf in uncertain_facts
        ]

        return suggestions

    # ==================== UNIFIED PROOF EXPLANATIONS (Phase 1) ====================

    def get_unified_proof_tree(
        self, proof: Optional["ProofStep"], query: str
    ) -> Optional[ProofTree]:
        """
        Konvertiert einen Logic Engine ProofStep in einen Unified ProofTree.

        Args:
            proof: ProofStep aus dem Logic Engine
            query: Die ursprüngliche Anfrage

        Returns:
            UnifiedProofTree oder None
        """
        if not UNIFIED_PROOFS_AVAILABLE or not proof:
            return None

        try:
            return convert_logic_engine_proof(proof)
        except Exception as e:
            logger.error(f"Fehler bei Konvertierung zu UnifiedProofTree: {e}")
            return None

    def format_proof_with_explanations(
        self, proof: "ProofStep", show_details: bool = True
    ) -> str:
        """
        Formatiert einen ProofStep mit detaillierten Erklärungen.

        Nutzt das Unified Proof Explanation System falls verfügbar,
        sonst Fallback auf alte format_proof_trace Methode.

        Args:
            proof: ProofStep zum Formatieren
            show_details: Ob Details angezeigt werden sollen

        Returns:
            Formatierter String für UI-Anzeige
        """
        if not UNIFIED_PROOFS_AVAILABLE:
            # Fallback auf alte Methode
            return self.format_proof_trace(proof)

        try:
            unified_step = convert_logic_engine_proof(proof)
            return format_proof_step(unified_step, indent=0, show_details=show_details)
        except Exception as e:
            logger.error(f"Fehler bei Formatierung mit Unified System: {e}")
            return self.format_proof_trace(proof)

    def _generate_rule_explanation(
        self,
        rule: "Rule",
        bindings: Binding,
        support_facts: List["Fact"],
        new_fact: "Fact",
    ) -> str:
        """
        Generiert eine natürlichsprachliche Erklärung für eine Regelanwendung.

        Args:
            rule: Die angewandte Regel
            bindings: Variable Bindings
            support_facts: Verwendete Fakten
            new_fact: Abgeleiteter neuer Fakt

        Returns:
            Natürlichsprachliche Erklärung
        """
        if UNIFIED_PROOFS_AVAILABLE:
            # Nutze Unified Explanation Generator
            inputs = [f"{f.pred}({f.args})" for f in support_facts]
            output = f"{new_fact.pred}({new_fact.args})"

            return generate_explanation_text(
                step_type=StepType.RULE_APPLICATION,
                inputs=inputs,
                output=output,
                rule_name=rule.id,
                bindings=bindings,
                metadata={
                    "salience": rule.salience,
                    "explain": rule.explain,
                    "num_premises": len(support_facts),
                },
            )
        else:
            # Fallback Erklärung
            premises_str = ", ".join(f"{f.pred}" for f in support_facts)
            return f"Wendete Regel '{rule.id}' mit Prämissen [{premises_str}] an -> {new_fact.pred}"

    def _generate_fact_match_explanation(self, fact: "Fact", goal: "Goal") -> str:
        """
        Generiert Erklärung für direkten Faktenmatch.

        Args:
            fact: Der gematchte Fakt
            goal: Das Goal

        Returns:
            Natürlichsprachliche Erklärung
        """
        if UNIFIED_PROOFS_AVAILABLE:
            output = f"{fact.pred}({fact.args})"
            return generate_explanation_text(
                step_type=StepType.FACT_MATCH,
                inputs=[],
                output=output,
                metadata={"source": fact.source, "confidence": fact.confidence},
            )
        else:
            return f"Fand Fakt direkt in Wissensbasis: {fact.pred}({fact.args})"

    def _generate_graph_traversal_explanation(
        self,
        subject: str,
        relation: str,
        target: str,
        hops: int,
        path: Optional[List[str]] = None,
    ) -> str:
        """
        Generiert Erklärung für Graph-Traversal.

        Args:
            subject: Start-Entität
            relation: Relationstyp
            target: Ziel-Entität
            hops: Anzahl Schritte
            path: Pfad (optional)

        Returns:
            Natürlichsprachliche Erklärung
        """
        if UNIFIED_PROOFS_AVAILABLE:
            output = f"{subject} {relation} {target}"
            return generate_explanation_text(
                step_type=StepType.GRAPH_TRAVERSAL,
                inputs=[subject],
                output=output,
                metadata={"hops": hops, "path": path or [], "relation": relation},
            )
        else:
            if path:
                path_str = " -> ".join(path)
                return f"Fand Pfad über {hops} Schritte: {path_str}"
            return f"Fand Verbindung über {hops} Schritte im Graphen: {subject} -> {target}"
