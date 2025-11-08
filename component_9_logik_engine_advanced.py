# component_9_logik_engine_advanced.py
"""
Advanced Reasoning Mixin for Logic Engine.

Contains all advanced reasoning methods:
- SAT Solving Methods (boolean reasoning, KB consistency)
- Consistency Checking (fact set validation, contradiction detection)
- Inference Chain Validation (proof verification, cycle detection)
- Contradiction Explanation (natural language explanations)
- Ontology-Based Semantic Constraints (domain knowledge integration)

This mixin is designed to be mixed into the LogikEngine class in component_9_logik_engine.py
to provide advanced reasoning capabilities without cluttering the main class.
"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from component_15_logging_config import get_logger

# Import SAT Solver components
try:
    from component_30_sat_solver import (
        Clause,
        CNFFormula,
        Literal,
        SATResult,
    )

    SAT_SOLVER_AVAILABLE = True
except ImportError:
    SAT_SOLVER_AVAILABLE = False
    logging.getLogger(__name__).warning("SAT Solver nicht verfügbar")

# Import Ontology Constraint Generator
try:
    pass

    ONTOLOGY_CONSTRAINTS_AVAILABLE = True
except ImportError:
    ONTOLOGY_CONSTRAINTS_AVAILABLE = False
    logging.getLogger(__name__).warning("Ontology Constraint Generator nicht verfügbar")

# Import Proof Structures
try:
    from component_17_proof_explanation import ProofStep

    PROOF_STRUCTURES_AVAILABLE = True
except ImportError:
    PROOF_STRUCTURES_AVAILABLE = False
    logging.getLogger(__name__).warning("Proof structures nicht verfügbar")

logger = get_logger(__name__)


# Import data structures from main component
# These are referenced by the mixin methods
class Fact:
    """Placeholder - will be provided by parent class"""


class Rule:
    """Placeholder - will be provided by parent class"""


class AdvancedReasoningMixin:
    """
    Mixin providing advanced reasoning capabilities to LogikEngine.

    This mixin assumes the parent LogikEngine class has the following attributes:
    - self.use_sat: bool - whether SAT solver is enabled
    - self.sat_solver: DPLLSolver - SAT solver instance
    - self.kb_checker: KnowledgeBaseChecker - KB consistency checker
    - self.use_ontology_constraints: bool - whether ontology constraints are enabled
    - self.ontology_generator: OntologyConstraintGenerator - ontology generator
    - self.rules: List[Rule] - list of reasoning rules
    - self.wm: List[Fact] - working memory (facts)
    - self.kb: List[Fact] - knowledge base (facts)
    - self.netzwerk: KonzeptNetzwerk - knowledge graph
    - self._goal_signature: callable - method to generate goal signatures
    """

    # ==================== SAT SOLVING METHODS ====================

    def check_kb_consistency(self) -> Tuple[bool, List[str]]:
        """
        Prüft Konsistenz der Wissensbasis mittels SAT-Solver.

        Konvertiert Facts und Rules in CNF und prüft Erfüllbarkeit.

        Returns:
            Tuple (is_consistent, conflicts)
            is_consistent: True wenn konsistent
            conflicts: Liste von Konfliktbeschreibungen
        """
        if not self.use_sat or not self.kb_checker:
            logger.warning("SAT-Solver nicht verfügbar für Konsistenzprüfung.")
            return True, []

        logger.info("=== Konsistenzprüfung der Wissensbasis gestartet ===")

        # Sammle Literale aus Facts
        fact_literals = []
        for fact in self.wm:
            # Konvertiere Fact zu Literal
            # Format: pred_arg1_arg2_...
            lit_name = self._fact_to_literal_name(fact)
            # Negative Fakten haben negated=True
            negated = fact.status == "contradicted" or fact.confidence < 0.5
            fact_literals.append(Literal(lit_name, negated))

        # Konvertiere Regeln zu Implikationen
        # Rule: (p1 ∧ p2) → conclusion
        # CNF: ¬p1 ∨ ¬p2 ∨ conclusion
        rule_implications = []
        for rule in self.rules:
            if not rule.when or not rule.then:
                continue

            # Extrahiere Prämissen
            premises = []
            for when_cond in rule.when:
                lit_name = f"{when_cond['pred']}"
                if when_cond.get("args"):
                    args_str = "_".join(str(v) for v in when_cond["args"].values())
                    lit_name = f"{lit_name}_{args_str}"
                premises.append(Literal(lit_name))

            # Extrahiere Konklusion (nur erste ASSERT-Action)
            conclusion_lit = None
            for then_action in rule.then:
                if "ASSERT" in then_action:
                    assert_dict = then_action["ASSERT"]
                    lit_name = f"{assert_dict.get('pred', 'unknown')}"
                    if assert_dict.get("args"):
                        args_str = "_".join(
                            str(v) for v in assert_dict["args"].values()
                        )
                        lit_name = f"{lit_name}_{args_str}"
                    conclusion_lit = Literal(lit_name)
                    break

            if conclusion_lit and premises:
                rule_implications.append((premises, conclusion_lit))

        # Prüfe Konsistenz
        conflicts = self.kb_checker.find_conflicts(fact_literals, rule_implications)

        is_consistent = len(conflicts) == 0

        if is_consistent:
            logger.info("✓ Wissensbasis ist konsistent.")
        else:
            logger.warning(f"✗ Wissensbasis hat {len(conflicts)} Konflikte:")
            for conflict in conflicts:
                logger.warning(f"  - {conflict}")

        return is_consistent, conflicts

    def solve_boolean_problem(
        self, formula: CNFFormula, problem_description: str = "Boolean Problem"
    ) -> Tuple[SATResult, Optional[Dict[str, bool]]]:
        """
        Löst ein propositionales Logik-Problem mittels SAT-Solver.

        Args:
            formula: CNF-Formel
            problem_description: Beschreibung des Problems (für Logging)

        Returns:
            Tuple (result, model)
            result: SATISFIABLE, UNSATISFIABLE, oder UNKNOWN
            model: Variable-Assignment (wenn satisfiable)
        """
        if not self.use_sat or not self.sat_solver:
            logger.warning("SAT-Solver nicht verfügbar.")
            return SATResult.UNKNOWN, None

        logger.info(f"=== Löse Boolean Problem: {problem_description} ===")
        logger.info(
            f"Formel: {len(formula.clauses)} Klauseln, {len(formula.variables)} Variablen"
        )

        result, model = self.sat_solver.solve(formula)

        logger.info(f"Ergebnis: {result.value}")
        if model:
            logger.info(f"Lösung gefunden: {len(model)} Variable(n) belegt")

        return result, model

    def verify_rule_consistency(
        self, rule_ids: Optional[List[str]] = None
    ) -> Tuple[bool, Optional[Dict[str, bool]]]:
        """
        Verifiziert, dass ein Set von Regeln konsistent ist.

        Args:
            rule_ids: Liste von Regel-IDs (None = alle Regeln)

        Returns:
            Tuple (is_consistent, model)
            is_consistent: True wenn konsistent
            model: Erfüllendes Assignment (wenn konsistent)
        """
        if not self.use_sat or not self.kb_checker:
            logger.warning("SAT-Solver nicht verfügbar für Regelverifizierung.")
            return True, None

        # Filtere Regeln
        rules_to_check = self.rules
        if rule_ids:
            rules_to_check = [r for r in self.rules if r.id in rule_ids]

        logger.info(
            f"=== Verifiziere Konsistenz von {len(rules_to_check)} Regel(n) ==="
        )

        # Konvertiere Regeln
        rule_implications = []
        for rule in rules_to_check:
            if not rule.when or not rule.then:
                continue

            premises = []
            for when_cond in rule.when:
                lit_name = f"{when_cond['pred']}"
                if when_cond.get("args"):
                    args_str = "_".join(str(v) for v in when_cond["args"].values())
                    lit_name = f"{lit_name}_{args_str}"
                premises.append(Literal(lit_name))

            for then_action in rule.then:
                if "ASSERT" in then_action:
                    assert_dict = then_action["ASSERT"]
                    lit_name = f"{assert_dict.get('pred', 'unknown')}"
                    if assert_dict.get("args"):
                        args_str = "_".join(
                            str(v) for v in assert_dict["args"].values()
                        )
                        lit_name = f"{lit_name}_{args_str}"
                    conclusion_lit = Literal(lit_name)
                    rule_implications.append((premises, conclusion_lit))
                    break

        # Prüfe Konsistenz
        is_consistent, model = self.kb_checker.check_rule_consistency(rule_implications)

        if is_consistent:
            logger.info("✓ Regeln sind konsistent.")
        else:
            logger.warning("✗ Regeln sind inkonsistent (widersprechen sich).")

        return is_consistent, model

    def _fact_to_literal_name(self, fact: Fact) -> str:
        """
        Konvertiert Fact zu Literal-Namen für SAT-Solver.

        Args:
            fact: Fact-Objekt

        Returns:
            String-Repräsentation als Literal-Name
        """
        lit_name = fact.pred
        if fact.args:
            args_str = "_".join(str(v) for k, v in sorted(fact.args.items()))
            lit_name = f"{lit_name}_{args_str}"
        # Normalisiere: Keine Sonderzeichen
        lit_name = lit_name.replace(" ", "_").replace("-", "_")
        return lit_name

    # ==================== CONSISTENCY CHECKING ====================

    def check_consistency(
        self, facts: List[Fact], include_graph_facts: bool = True
    ) -> bool:
        """
        Prüft Konsistenz einer Faktenmenge via SAT mit semantischen Constraints.

        Konvertiert Fakten → propositionale Formeln → CNF und
        nutzt SAT-Solver für Consistency-Check. Lädt zusätzlich
        semantische Constraints aus der Ontologie.

        Args:
            facts: Liste von Fakten, die auf Konsistenz geprüft werden sollen
            include_graph_facts: Lade relevante Fakten aus Neo4j-Graph

        Returns:
            True wenn Faktenmenge konsistent ist, False bei Widersprüchen

        Beispiel:
            facts = [
                Fact(pred="IS_A", args={"subject": "hund", "object": "tier"}),
                Fact(pred="IS_A", args={"subject": "hund", "object": "pflanze"})
            ]
            # Mit Ontologie-Constraints: False (tier und pflanze schließen sich aus)
        """
        if not self.use_sat or not self.sat_solver:
            logger.warning("SAT-Solver nicht verfügbar für Konsistenzprüfung.")
            return True  # Keine Prüfung möglich, assume consistent

        logger.info(f"=== Konsistenzprüfung für {len(facts)} Fakten gestartet ===")

        if not facts:
            logger.info("Leere Faktenmenge ist konsistent.")
            return True

        # PHASE 1: Lade relevante Fakten aus Graph (falls aktiviert)
        all_facts = facts.copy()
        if include_graph_facts:
            graph_facts = self._load_relevant_graph_facts(facts)
            all_facts.extend(graph_facts)
            if graph_facts:
                logger.info(
                    f"Hinzugefügt: {len(graph_facts)} relevante Fakten aus Graph"
                )

        # PHASE 2: Sammle alle Literale aus Facts
        fact_literals = []
        literal_to_fact_map = {}  # Mapping für spätere Konfliktanalyse

        for fact in all_facts:
            # Konvertiere Fact zu Literal
            lit_name = self._fact_to_literal_name(fact)

            # Prüfe Status des Fakts
            negated = fact.status == "contradicted" or fact.confidence < 0.5
            literal = Literal(lit_name, negated)

            fact_literals.append(literal)
            literal_to_fact_map[lit_name] = fact

        # Prüfe auf direkte Widersprüche (Fact und sein Negat)
        literal_vars = set()
        for lit in fact_literals:
            if lit.variable in literal_vars:
                # Gefunden: Literal existiert bereits
                # Prüfe ob es das Negat ist
                for other_lit in fact_literals:
                    if (
                        other_lit.variable == lit.variable
                        and other_lit.negated != lit.negated
                    ):
                        logger.warning(
                            f"Direkter Widerspruch gefunden: {lit} vs {other_lit}"
                        )
                        return False
            literal_vars.add(lit.variable)

        # Erstelle CNF-Formel aus Fakten
        # Jedes Fact wird als Unit Clause hinzugefügt
        formula = CNFFormula([])
        for lit in fact_literals:
            formula.add_clause(Clause({lit}))

        # Erweitere mit relevanten Regeln (falls vorhanden)
        # Regeln als Implikationen: (p1 ∧ p2) → conclusion
        # CNF: ¬p1 ∨ ¬p2 ∨ conclusion
        for rule in self.rules:
            if not rule.when or not rule.then:
                continue

            # Prüfe ob Regel relevant für die gegebenen Fakten ist
            rule_relevant = False
            for when_cond in rule.when:
                for fact in facts:
                    if fact.pred == when_cond.get("pred"):
                        rule_relevant = True
                        break
                if rule_relevant:
                    break

            if not rule_relevant:
                continue

            # Extrahiere Prämissen
            premises = []
            for when_cond in rule.when:
                lit_name = f"{when_cond['pred']}"
                if when_cond.get("args"):
                    args_str = "_".join(str(v) for v in when_cond["args"].values())
                    lit_name = f"{lit_name}_{args_str}"
                premises.append(Literal(lit_name))

            # Extrahiere Konklusion (nur erste ASSERT-Action)
            conclusion_lit = None
            for then_action in rule.then:
                if "assert" in then_action:
                    assert_dict = then_action["assert"]
                    lit_name = f"{assert_dict.get('pred', 'unknown')}"
                    if assert_dict.get("args"):
                        args_str = "_".join(
                            str(v) for v in assert_dict["args"].values()
                        )
                        lit_name = f"{lit_name}_{args_str}"
                    conclusion_lit = Literal(lit_name)
                    break

            if conclusion_lit and premises:
                # Konvertiere Regel zu CNF: ¬p1 ∨ ¬p2 ∨ ... ∨ conclusion
                clause_literals = {-p for p in premises}
                clause_literals.add(conclusion_lit)
                formula.add_clause(Clause(clause_literals))

        # PHASE 3: Füge semantische Ontologie-Constraints hinzu
        if self.use_ontology_constraints and self.ontology_generator:
            logger.info("Lade semantische Constraints aus Ontologie...")

            # Generiere Constraints basierend auf den Facts
            ontology_clauses = self._generate_semantic_constraints_for_facts(all_facts)

            if ontology_clauses:
                logger.info(
                    f"Hinzugefügt: {len(ontology_clauses)} semantische Constraints"
                )
                for clause in ontology_clauses:
                    formula.add_clause(clause)
            else:
                logger.debug("Keine zusätzlichen Ontologie-Constraints generiert")

        # PHASE 4: Löse mit SAT-Solver
        result, model = self.sat_solver.solve(formula)

        is_consistent = result == SATResult.SATISFIABLE

        if is_consistent:
            logger.info("✓ Faktenmenge ist konsistent.")
            if model:
                logger.debug(f"Satisfying assignment gefunden: {len(model)} Variablen")
        else:
            logger.warning("✗ Faktenmenge ist inkonsistent (enthält Widersprüche).")

        return is_consistent

    def find_contradictions(self) -> List[Tuple[Fact, Fact]]:
        """
        Findet widersprüchliche Fakten in der Wissensbasis.

        Nutzt SAT-Solver für Konflikt-Analyse und identifiziert
        Paare von Fakten, die sich gegenseitig ausschließen.

        Returns:
            Liste von (Fact, Fact) Tupeln mit widersprüchlichen Fakten

        Beispiele für Widersprüche:
            1. Direkte Negation:
               - Fact1: IS_A(hund, tier)
               - Fact2: IS_A(hund, tier) mit status="contradicted"

            2. Exklusive Eigenschaften:
               - Fact1: HAS_PROPERTY(apfel, rot)
               - Fact2: HAS_PROPERTY(apfel, grün)
               (wenn Regeln definieren, dass rot ∧ grün → FALSE)

            3. Zyklische Hierarchien:
               - Fact1: IS_A(A, B)
               - Fact2: IS_A(B, C)
               - Fact3: IS_A(C, A)
               (wenn Regeln transitive Antisymmetrie fordern)
        """
        if not self.use_sat or not self.kb_checker:
            logger.warning("SAT-Solver nicht verfügbar für Widerspruchsanalyse.")
            return []

        logger.info("=== Widerspruchsanalyse gestartet ===")

        all_facts = self.wm + self.kb
        if not all_facts:
            logger.info("Keine Fakten in Wissensbasis.")
            return []

        contradictions = []

        # Strategie 1: Direkte Widersprüche (Fact vs. negiertes Fact)
        logger.debug("Suche direkte Widersprüche...")
        fact_signatures = {}  # signature -> Fact

        for fact in all_facts:
            sig = self._fact_to_literal_name(fact)

            if sig in fact_signatures:
                # Gefunden: Fact mit gleicher Signatur
                other_fact = fact_signatures[sig]

                # Prüfe auf Widerspruch
                fact_negated = fact.status == "contradicted" or fact.confidence < 0.5
                other_negated = (
                    other_fact.status == "contradicted" or other_fact.confidence < 0.5
                )

                if fact_negated != other_negated:
                    # Widerspruch: Eines ist negiert, das andere nicht
                    contradictions.append((fact, other_fact))
                    logger.info(
                        f"Direkter Widerspruch gefunden: "
                        f"{fact.pred}({fact.args}) [conf={fact.confidence:.2f}] vs "
                        f"{other_fact.pred}({other_fact.args}) [conf={other_fact.confidence:.2f}]"
                    )
            else:
                fact_signatures[sig] = fact

        # Strategie 2: Konvertiere alle Fakten + Regeln zu CNF und finde UNSAT-Kerne
        logger.debug("Suche indirekte Widersprüche via SAT-Solver...")

        # Erstelle Literale für alle Fakten
        fact_literals = []
        for fact in all_facts:
            lit_name = self._fact_to_literal_name(fact)
            negated = fact.status == "contradicted" or fact.confidence < 0.5
            fact_literals.append(Literal(lit_name, negated))

        # Konvertiere Regeln zu Implikationen
        rule_implications = []
        for rule in self.rules:
            if not rule.when or not rule.then:
                continue

            premises = []
            for when_cond in rule.when:
                lit_name = f"{when_cond['pred']}"
                if when_cond.get("args"):
                    args_str = "_".join(str(v) for v in when_cond["args"].values())
                    lit_name = f"{lit_name}_{args_str}"
                premises.append(Literal(lit_name))

            for then_action in rule.then:
                if "assert" in then_action:
                    assert_dict = then_action["assert"]
                    lit_name = f"{assert_dict.get('pred', 'unknown')}"
                    if assert_dict.get("args"):
                        args_str = "_".join(
                            str(v) for v in assert_dict["args"].values()
                        )
                        lit_name = f"{lit_name}_{args_str}"
                    conclusion_lit = Literal(lit_name)
                    rule_implications.append((premises, conclusion_lit))
                    break

        # Nutze KnowledgeBaseChecker für Konflikt-Analyse
        conflicts = self.kb_checker.find_conflicts(fact_literals, rule_implications)

        if conflicts:
            logger.warning(f"Gefunden: {len(conflicts)} Konflikte in Wissensbasis")
            for conflict in conflicts:
                logger.warning(f"  - {conflict}")

            # Strategie 3: Systematische Paarweise Inkonsistenz-Tests
            # Teste jedes Paar von Fakten auf Inkonsistenz
            logger.debug("Teste Faktenpaare systematisch...")

            for i, fact1 in enumerate(all_facts):
                for j, fact2 in enumerate(all_facts):
                    if i >= j:
                        continue  # Vermeide Duplikate und Selbst-Vergleich

                    # Prüfe ob dieses Paar bereits als Widerspruch identifiziert wurde
                    if (fact1, fact2) in contradictions or (
                        fact2,
                        fact1,
                    ) in contradictions:
                        continue

                    # Teste ob fact1 + fact2 zusammen konsistent sind
                    if not self.check_consistency([fact1, fact2]):
                        contradictions.append((fact1, fact2))
                        logger.info(
                            f"Indirekter Widerspruch gefunden: "
                            f"{fact1.pred}({fact1.args}) ⊥ {fact2.pred}({fact2.args})"
                        )

        # Entferne Duplikate (berücksichtige Symmetrie: (A,B) == (B,A))
        unique_contradictions = []
        seen_pairs = set()

        for f1, f2 in contradictions:
            pair_sig = tuple(sorted([f1.id, f2.id]))
            if pair_sig not in seen_pairs:
                unique_contradictions.append((f1, f2))
                seen_pairs.add(pair_sig)

        logger.info(
            f"=== Widerspruchsanalyse abgeschlossen: "
            f"{len(unique_contradictions)} Widerspruchspaare gefunden ==="
        )

        return unique_contradictions

    # ==================== INFERENCE CHAIN VALIDATION ====================

    def validate_inference_chain(self, proof: ProofStep) -> List[str]:
        """
        Validiert Reasoning-Kette auf Konsistenz.

        Nutzt SAT-Solver (Phase 2) für Konsistenzprüfung der gesamten Beweiskette.
        Prüft ob alle Schritte logisch konsistent sind und keine Widersprüche enthalten.

        Args:
            proof: ProofStep mit vollständigem Beweisbaum

        Returns:
            Liste von Inkonsistenzen (Strings mit Beschreibungen)
            Leere Liste bedeutet: Beweis ist konsistent

        Beispiel:
            proof = engine.prove_goal(goal)
            inconsistencies = engine.validate_inference_chain(proof)
            if inconsistencies:
                print("Warnung: Inkonsistenzen gefunden:", inconsistencies)
        """
        if not self.use_sat or not self.sat_solver:
            logger.warning("SAT-Solver nicht verfügbar für Inferenzketten-Validierung.")
            return ["SAT-Solver nicht verfügbar - keine Validierung möglich"]

        logger.info("=== Validiere Inferenzkette auf Konsistenz ===")

        inconsistencies = []

        # SCHRITT 1: Sammle alle Fakten aus dem Beweisbaum
        all_facts_in_proof = self._extract_all_facts_from_proof(proof)

        if not all_facts_in_proof:
            logger.warning("Keine Fakten in Beweis gefunden.")
            return ["Beweis enthält keine verifizierbaren Fakten"]

        logger.debug(f"Extrahierte {len(all_facts_in_proof)} Fakten aus Beweisbaum")

        # SCHRITT 2: Prüfe Konsistenz der Faktenmenge
        is_consistent = self.check_consistency(all_facts_in_proof)

        if not is_consistent:
            inconsistencies.append(
                f"Beweis enthält widersprüchliche Fakten: {len(all_facts_in_proof)} Fakten sind nicht konsistent"
            )

        # SCHRITT 3: Prüfe jede Regelanwendung im Beweis
        rule_applications = self._extract_rule_applications_from_proof(proof)

        for rule_app in rule_applications:
            rule_id = rule_app.get("rule_id")
            premises = rule_app.get("premises", [])
            conclusion = rule_app.get("conclusion")

            # Prüfe ob Regelanwendung valide ist
            if not self._validate_rule_application(rule_id, premises, conclusion):
                inconsistencies.append(
                    f"Ungültige Regelanwendung: Regel '{rule_id}' mit Prämissen {[f.pred for f in premises]} "
                    f"-> {conclusion.pred if conclusion else 'None'}"
                )

        # SCHRITT 4: Prüfe Confidence-Werte (sollten monoton abnehmen)
        confidence_issues = self._check_confidence_monotonicity(proof)
        inconsistencies.extend(confidence_issues)

        # SCHRITT 5: Prüfe auf zyklische Abhängigkeiten
        cycle_issues = self._check_for_cycles(proof)
        if cycle_issues:
            inconsistencies.append(f"Zyklische Abhängigkeiten gefunden: {cycle_issues}")

        # SCHRITT 6: Nutze SAT-Solver für vollständige Konsistenzprüfung
        # Konvertiere alle Fakten + angewandte Regeln zu CNF
        try:
            sat_inconsistencies = self._sat_validate_proof_chain(proof)
            inconsistencies.extend(sat_inconsistencies)
        except Exception as e:
            logger.warning(f"SAT-Validierung fehlgeschlagen: {e}")
            inconsistencies.append(
                f"SAT-Validierung konnte nicht durchgeführt werden: {str(e)}"
            )

        if inconsistencies:
            logger.warning(
                f"=== Validierung abgeschlossen: {len(inconsistencies)} Inkonsistenzen gefunden ==="
            )
        else:
            logger.info("=== Validierung erfolgreich: Inferenzkette ist konsistent ===")

        return inconsistencies

    def _extract_all_facts_from_proof(self, proof: ProofStep) -> List[Fact]:
        """
        Extrahiert alle Fakten aus einem Beweisbaum (rekursiv).

        Args:
            proof: ProofStep

        Returns:
            Liste aller verwendeten Fakten
        """
        facts = []

        # Fakten aus diesem Step
        facts.extend(proof.supporting_facts)

        # Rekursiv aus Subgoals
        for subproof in proof.subgoals:
            facts.extend(self._extract_all_facts_from_proof(subproof))

        # Dedupliziere nach Fact-ID
        seen_ids = set()
        unique_facts = []
        for fact in facts:
            if fact.id not in seen_ids:
                unique_facts.append(fact)
                seen_ids.add(fact.id)

        return unique_facts

    def _extract_rule_applications_from_proof(self, proof: ProofStep) -> List[Dict]:
        """
        Extrahiert alle Regelanwendungen aus einem Beweisbaum.

        Args:
            proof: ProofStep

        Returns:
            Liste von Dicts mit {rule_id, premises, conclusion}
        """
        applications = []

        if proof.method == "rule" and proof.rule_id:
            # Extrahiere Prämissen (supporting_facts)
            premises = proof.supporting_facts

            # Extrahiere Konklusion (abgeleiteter Fakt)
            # Konklusion ist das Goal, das bewiesen wurde
            from component_9_logik_engine_core import Fact as FactClass

            conclusion_fact = FactClass(
                pred=proof.goal.pred,
                args=proof.goal.args,
                confidence=proof.confidence,
                source=f"rule:{proof.rule_id}",
            )

            applications.append(
                {
                    "rule_id": proof.rule_id,
                    "premises": premises,
                    "conclusion": conclusion_fact,
                }
            )

        # Rekursiv aus Subgoals
        for subproof in proof.subgoals:
            applications.extend(self._extract_rule_applications_from_proof(subproof))

        return applications

    def _validate_rule_application(
        self, rule_id: str, premises: List[Fact], conclusion: Fact
    ) -> bool:
        """
        Validiert ob eine Regelanwendung korrekt ist.

        Prüft ob:
        1. Regel existiert
        2. Prämissen die WHEN-Klauseln erfüllen
        3. Konklusion der THEN-Klausel entspricht

        Args:
            rule_id: ID der angewandte Regel
            premises: Liste der Prämissen-Fakten
            conclusion: Abgeleiteter Fakt

        Returns:
            True wenn Regelanwendung valide, False sonst
        """
        # Finde Regel
        rule = None
        for r in self.rules:
            if r.id == rule_id:
                rule = r
                break

        if not rule:
            logger.warning(f"Regel '{rule_id}' nicht gefunden")
            return False

        # Prüfe ob Anzahl der Prämissen passt
        if len(premises) < len(rule.when):
            logger.warning(
                f"Regel '{rule_id}': Zu wenige Prämissen ({len(premises)} vs {len(rule.when)} erwartet)"
            )
            return False

        # Prüfe ob Prämissen die WHEN-Klauseln erfüllen
        for when_clause in rule.when:
            matched = False
            for premise in premises:
                if premise.pred == when_clause.get("pred"):
                    matched = True
                    break
            if not matched:
                logger.warning(
                    f"Regel '{rule_id}': WHEN-Klausel {when_clause.get('pred')} nicht erfüllt"
                )
                return False

        # Prüfe ob Konklusion einer THEN-Klausel entspricht
        for then_action in rule.then:
            if "assert" in then_action:
                assert_dict = then_action["assert"]
                if assert_dict.get("pred") == conclusion.pred:
                    # Regelanwendung ist valide
                    return True

        logger.warning(
            f"Regel '{rule_id}': Konklusion {conclusion.pred} entspricht keiner THEN-Klausel"
        )
        return False

    def _check_confidence_monotonicity(self, proof: ProofStep) -> List[str]:
        """
        Prüft ob Confidence-Werte monoton abnehmen (Kind <= Eltern).

        In einem korrekten Beweis sollte die Confidence eines Subgoals
        nicht höher sein als die des übergeordneten Goals.

        Args:
            proof: ProofStep

        Returns:
            Liste von Confidence-Violations
        """
        violations = []

        parent_confidence = proof.confidence

        for subproof in proof.subgoals:
            if subproof.confidence > parent_confidence + 0.01:  # Toleranz für Rundung
                violations.append(
                    f"Confidence-Verletzung: Subgoal {subproof.goal.pred} hat Confidence {subproof.confidence:.2f}, "
                    f"aber Eltern-Goal {proof.goal.pred} nur {parent_confidence:.2f}"
                )

            # Rekursiv prüfen
            violations.extend(self._check_confidence_monotonicity(subproof))

        return violations

    def _check_for_cycles(
        self, proof: ProofStep, visited: Optional[Set[str]] = None
    ) -> Optional[str]:
        """
        Prüft auf zyklische Abhängigkeiten im Beweisbaum.

        Args:
            proof: ProofStep
            visited: Set von besuchten Goal-IDs

        Returns:
            Zyklusbeschreibung oder None
        """
        if visited is None:
            visited = set()

        goal_sig = self._goal_signature(proof.goal)

        if goal_sig in visited:
            return f"Zyklus gefunden: {goal_sig} wurde bereits besucht"

        visited.add(goal_sig)

        for subproof in proof.subgoals:
            cycle = self._check_for_cycles(subproof, visited.copy())
            if cycle:
                return cycle

        return None

    def _sat_validate_proof_chain(self, proof: ProofStep) -> List[str]:
        """
        Nutzt SAT-Solver für vollständige Konsistenzprüfung der Beweiskette.

        Konvertiert alle Fakten und Regelanwendungen zu CNF und
        prüft Erfüllbarkeit.

        Args:
            proof: ProofStep

        Returns:
            Liste von SAT-Inkonsistenzen
        """
        inconsistencies = []

        # Sammle alle Fakten
        all_facts = self._extract_all_facts_from_proof(proof)

        # Sammle alle Literale
        fact_literals = []
        for fact in all_facts:
            lit_name = self._fact_to_literal_name(fact)
            negated = fact.status == "contradicted" or fact.confidence < 0.5
            fact_literals.append(Literal(lit_name, negated))

        # Sammle Regelimplikationen
        rule_applications = self._extract_rule_applications_from_proof(proof)
        rule_implications = []

        for rule_app in rule_applications:
            _ = rule_app.get("rule_id")  # Mark as intentionally unused
            premises_facts = rule_app.get("premises", [])
            conclusion_fact = rule_app.get("conclusion")

            # Konvertiere zu Literalen
            premises_lits = []
            for prem_fact in premises_facts:
                lit_name = self._fact_to_literal_name(prem_fact)
                premises_lits.append(Literal(lit_name))

            if conclusion_fact:
                conclusion_lit_name = self._fact_to_literal_name(conclusion_fact)
                conclusion_lit = Literal(conclusion_lit_name)
                rule_implications.append((premises_lits, conclusion_lit))

        # Nutze KnowledgeBaseChecker
        if self.kb_checker:
            conflicts = self.kb_checker.find_conflicts(fact_literals, rule_implications)
            if conflicts:
                inconsistencies.extend(conflicts)

        return inconsistencies

    # ==================== CONTRADICTION EXPLANATION ====================

    def explain_contradiction(self, fact1: Fact, fact2: Fact) -> str:
        """
        Erklärt warum zwei Fakten widersprüchlich sind.

        Generiert natürlichsprachliche Erklärung basierend auf:
        1. Direkter Negation (Fact vs. negiertes Fact)
        2. Regelbasierten Konflikten
        3. Domänenwissen (exklusive Eigenschaften)

        Args:
            fact1: Erster Fakt
            fact2: Zweiter Fakt

        Returns:
            Natürlichsprachliche Erklärung des Widerspruchs

        Beispiel:
            fact1 = Fact(pred="HAS_PROPERTY", args={"subject": "apfel", "object": "rot"})
            fact2 = Fact(pred="HAS_PROPERTY", args={"subject": "apfel", "object": "grün"})
            explanation = engine.explain_contradiction(fact1, fact2)
            # "Widerspruch: 'apfel' kann nicht gleichzeitig 'rot' und 'grün' sein."
        """
        logger.info("=== Erkläre Widerspruch zwischen Fakten ===")
        logger.debug(f"Fact1: {fact1.pred}({fact1.args})")
        logger.debug(f"Fact2: {fact2.pred}({fact2.args})")

        # STRATEGIE 1: Direkte Negation
        if fact1.pred == fact2.pred and fact1.args == fact2.args:
            # Gleicher Fakt, aber unterschiedlicher Status
            if (fact1.status == "contradicted" or fact1.confidence < 0.5) != (
                fact2.status == "contradicted" or fact2.confidence < 0.5
            ):
                return (
                    f"**Direkter Widerspruch (Negation):**\n"
                    f"Fakt '{fact1.pred}({self._format_args(fact1.args)})' existiert sowohl als bestätigt "
                    f"(Konfidenz: {max(fact1.confidence, fact2.confidence):.2f}) als auch als negiert "
                    f"(Konfidenz: {min(fact1.confidence, fact2.confidence):.2f}).\n\n"
                    f"**Quelle:** {fact1.source} vs {fact2.source}"
                )

        # STRATEGIE 2: Exklusive Eigenschaften (HAS_PROPERTY)
        if fact1.pred == "HAS_PROPERTY" and fact2.pred == "HAS_PROPERTY":
            subj1 = fact1.args.get("subject")
            subj2 = fact2.args.get("subject")
            obj1 = fact1.args.get("object")
            obj2 = fact2.args.get("object")

            if subj1 == subj2 and obj1 != obj2:
                # Prüfe ob Eigenschaften exklusiv sind
                if self._are_properties_exclusive(obj1, obj2):
                    return (
                        f"**Exklusive Eigenschaften:**\n"
                        f"'{subj1}' kann nicht gleichzeitig '{obj1}' und '{obj2}' sein.\n\n"
                        f"**Grund:** Die Eigenschaften '{obj1}' und '{obj2}' schließen sich gegenseitig aus "
                        f"(z.B. Farben, Größen, Zustände).\n\n"
                        f"**Quellen:**\n"
                        f"  - Fakt 1: {fact1.source} (Konfidenz: {fact1.confidence:.2f})\n"
                        f"  - Fakt 2: {fact2.source} (Konfidenz: {fact2.confidence:.2f})"
                    )

        # STRATEGIE 3: Hierarchiewidersprüche (IS_A)
        if fact1.pred == "IS_A" and fact2.pred == "IS_A":
            subj1 = fact1.args.get("subject")
            subj2 = fact2.args.get("subject")
            obj1 = fact1.args.get("object")
            obj2 = fact2.args.get("object")

            if subj1 == subj2 and obj1 != obj2:
                # Prüfe ob Hierarchien widersprüchlich sind
                if self._are_categories_exclusive(obj1, obj2):
                    return (
                        f"**Hierarchiewiderspruch:**\n"
                        f"'{subj1}' kann nicht gleichzeitig '{obj1}' und '{obj2}' sein.\n\n"
                        f"**Grund:** Die Kategorien '{obj1}' und '{obj2}' sind disjunkt "
                        f"(keine gemeinsame Spezialisierung möglich).\n\n"
                        f"**Quellen:**\n"
                        f"  - {fact1.source}: '{subj1}' IS_A '{obj1}'\n"
                        f"  - {fact2.source}: '{subj1}' IS_A '{obj2}'"
                    )

        # STRATEGIE 4: Regelbasierter Konflikt
        rule_explanation = self._explain_rule_based_conflict(fact1, fact2)
        if rule_explanation:
            return rule_explanation

        # STRATEGIE 5: SAT-Solver basierte Erklärung
        if self.use_sat and self.sat_solver:
            sat_explanation = self._explain_sat_conflict(fact1, fact2)
            if sat_explanation:
                return sat_explanation

        # FALLBACK: Generische Erklärung
        return (
            f"**Widerspruch detektiert:**\n"
            f"Die folgenden Fakten sind inkonsistent:\n\n"
            f"  1. {fact1.pred}({self._format_args(fact1.args)}) "
            f"[Quelle: {fact1.source}, Konfidenz: {fact1.confidence:.2f}]\n"
            f"  2. {fact2.pred}({self._format_args(fact2.args)}) "
            f"[Quelle: {fact2.source}, Konfidenz: {fact2.confidence:.2f}]\n\n"
            f"**Grund:** Die Fakten können nicht gleichzeitig wahr sein, aber die genaue Ursache "
            f"konnte nicht automatisch bestimmt werden. Möglicherweise liegt ein logischer Konflikt vor, "
            f"der durch domänenspezifische Regeln entsteht."
        )

    def _format_args(self, args: Dict[str, Any]) -> str:
        """Formatiert Argumente für Ausgabe"""
        return ", ".join(f"{k}={v}" for k, v in args.items())

    def _are_properties_exclusive(self, prop1: str, prop2: str) -> bool:
        """
        Prüft ob zwei Eigenschaften exklusiv sind.

        Nutzt Domänenwissen über exklusive Kategorien.
        """
        prop1_lower = prop1.lower()
        prop2_lower = prop2.lower()

        # Identische Eigenschaften sind NICHT exklusiv
        if prop1_lower == prop2_lower:
            return False

        # Bekannte exklusive Kategorien
        exclusive_sets = [
            # Farben
            {
                "rot",
                "grün",
                "blau",
                "gelb",
                "schwarz",
                "weiß",
                "orange",
                "lila",
                "rosa",
                "braun",
            },
            # Größen
            {"groß", "klein", "mittel", "riesig", "winzig"},
            # Temperaturen
            {"heiß", "kalt", "warm", "kühl", "eiskalt", "glühend"},
            # Zustände
            {"lebendig", "tot"},
            {"an", "aus"},
            {"offen", "geschlossen"},
            # Geschmack
            {"süß", "sauer", "bitter", "salzig", "umami"},
        ]

        # Prüfe ob beide in derselben exklusiven Menge sind
        for exclusive_set in exclusive_sets:
            if prop1_lower in exclusive_set and prop2_lower in exclusive_set:
                return True

        # Prüfe ob eines das Negat des anderen ist (antonyms)
        antonym_pairs = [
            ("groß", "klein"),
            ("heiß", "kalt"),
            ("schnell", "langsam"),
            ("leicht", "schwer"),
            ("hell", "dunkel"),
            ("gut", "schlecht"),
        ]

        for ant1, ant2 in antonym_pairs:
            if (prop1_lower == ant1 and prop2_lower == ant2) or (
                prop1_lower == ant2 and prop2_lower == ant1
            ):
                return True

        return False

    def _are_categories_exclusive(self, cat1: str, cat2: str) -> bool:
        """
        Prüft ob zwei Kategorien disjunkt sind.

        Nutzt Graphtraversal um zu prüfen ob cat1 und cat2
        keine gemeinsame Spezialisierung haben.
        """
        # Bekannte disjunkte Kategorien
        disjoint_pairs = [
            # Biologie
            ("tier", "pflanze"),
            ("säugetier", "vogel"),
            ("säugetier", "fisch"),
            ("säugetier", "reptil"),
            ("vogel", "fisch"),
            # Objekte
            ("lebewesen", "gegenstand"),
            ("möbel", "tier"),
            ("fahrzeug", "gebäude"),
            # Abstrakt
            ("zahl", "buchstabe"),
            ("farbe", "form"),
        ]

        cat1_lower = cat1.lower()
        cat2_lower = cat2.lower()

        for disj1, disj2 in disjoint_pairs:
            if (cat1_lower == disj1 and cat2_lower == disj2) or (
                cat1_lower == disj2 and cat2_lower == disj1
            ):
                return True

        # Optional: Prüfe mit Neo4j ob gemeinsamer Supertyp existiert
        # (Nicht implementiert für Performance-Gründe)

        return False

    def _explain_rule_based_conflict(self, fact1: Fact, fact2: Fact) -> Optional[str]:
        """
        Erklärt Widerspruch basierend auf Regeln.

        Prüft ob eine Regel existiert, die den Konflikt definiert.
        """
        # Durchsuche Regeln nach Konfliktregeln
        for rule in self.rules:
            if rule.typ == "CONFLICT" or "conflict" in rule.id.lower():
                # Prüfe ob diese Regel auf fact1 und fact2 anwendbar ist
                if self._rule_applies_to_facts(rule, [fact1, fact2]):
                    return (
                        f"**Regelbasierter Konflikt:**\n"
                        f"Regel '{rule.id}' definiert, dass die folgenden Fakten sich widersprechen:\n\n"
                        f"  1. {fact1.pred}({self._format_args(fact1.args)})\n"
                        f"  2. {fact2.pred}({self._format_args(fact2.args)})\n\n"
                        f"**Regel-Erklärung:** {rule.explain if rule.explain else 'Keine Erklärung verfügbar'}"
                    )

        return None

    def _rule_applies_to_facts(self, rule: Rule, facts: List[Fact]) -> bool:
        """
        Prüft ob eine Regel auf eine Faktenmenge anwendbar ist.
        """
        # Einfache Heuristik: Prüfe ob alle WHEN-Klauseln von Fakten abgedeckt werden
        for when_clause in rule.when:
            matched = False
            for fact in facts:
                if fact.pred == when_clause.get("pred"):
                    matched = True
                    break
            if not matched:
                return False
        return True

    def _explain_sat_conflict(self, fact1: Fact, fact2: Fact) -> Optional[str]:
        """
        Nutzt SAT-Solver um Konflikt zu erklären.

        Erstellt minimale unerfüllbare Teilmenge (MUS).
        """
        if not self.sat_solver or not self.kb_checker:
            return None

        # Erstelle CNF mit nur diesen beiden Fakten
        fact_literals = [
            Literal(self._fact_to_literal_name(fact1), fact1.status == "contradicted"),
            Literal(self._fact_to_literal_name(fact2), fact2.status == "contradicted"),
        ]

        # Prüfe Konsistenz
        formula = CNFFormula([])
        for lit in fact_literals:
            formula.add_clause(Clause({lit}))

        # Füge relevante Regeln hinzu
        for rule in self.rules:
            # TODO: Filtere nur relevante Regeln
            pass

        result, _ = self.sat_solver.solve(formula)

        if result == SATResult.UNSATISFIABLE:
            return (
                f"**SAT-Solver Analyse:**\n"
                f"Der SAT-Solver bestätigt, dass die folgenden Fakten unvereinbar sind:\n\n"
                f"  1. {fact1.pred}({self._format_args(fact1.args)})\n"
                f"  2. {fact2.pred}({self._format_args(fact2.args)})\n\n"
                f"**Technische Details:** Die Formel ist UNSAT (unerfüllbar), d.h. es existiert "
                f"keine Belegung der Variablen, die beide Fakten gleichzeitig wahr macht."
            )

        return None

    # ==================== ONTOLOGY-BASED SEMANTIC CONSTRAINTS ====================

    def _load_relevant_graph_facts(self, facts: List[Fact]) -> List[Fact]:
        """
        Lädt relevante Fakten aus Neo4j-Graph basierend auf gegebenen Fakten.

        Strategie:
        1. Extrahiere alle Entitäten aus gegebenen Fakten
        2. Lade alle Fakten für diese Entitäten aus dem Graph
        3. Konvertiere zu Fact-Objekten

        Args:
            facts: Liste von Fakten, für die Graph-Fakten geladen werden sollen

        Returns:
            Liste von Fakten aus dem Graph
        """
        if not self.netzwerk or not self.netzwerk.driver:
            return []

        # Sammle alle erwähnten Entitäten
        entities = set()
        for fact in facts:
            if fact.pred in [
                "IS_A",
                "HAS_PROPERTY",
                "LOCATED_IN",
                "PART_OF",
                "CAPABLE_OF",
            ]:
                subject = fact.args.get("subject")
                obj = fact.args.get("object")
                if subject:
                    entities.add(subject.lower())
                if obj:
                    entities.add(obj.lower())

        if not entities:
            return []

        logger.debug(f"Lade Graph-Fakten für {len(entities)} Entitäten: {entities}")

        graph_facts = []

        for entity in entities:
            # Query Neo4j für alle Fakten über diese Entität
            entity_facts = self.netzwerk.query_graph_for_facts(entity)

            for rel_type, targets in entity_facts.items():
                for target in targets:
                    # Konvertiere zu Fact-Objekt
                    from component_9_logik_engine import Fact as FactClass

                    fact = FactClass(
                        pred=rel_type,
                        args={"subject": entity, "object": target},
                        confidence=1.0,
                        source="neo4j_graph",
                    )
                    graph_facts.append(fact)

        logger.debug(f"Geladen: {len(graph_facts)} Fakten aus Graph")

        return graph_facts

    def _generate_semantic_constraints_for_facts(
        self, facts: List[Fact]
    ) -> List[Clause]:
        """
        Generiert semantische Constraints aus Ontologie für gegebene Fakten.

        Analysiert die Fakten und generiert spezifische Constraints:
        - IS_A Exklusivität für Geschwister-Konzepte
        - Property Konflikte für widersprüchliche Eigenschaften
        - Location Konflikte für nicht-hierarchische Orte

        Args:
            facts: Liste von Fakten

        Returns:
            Liste von CNF-Clauses encoding semantischer Constraints

        Beispiel:
            facts = [
                Fact(pred="IS_A", args={"subject": "hund", "object": "tier"}),
                Fact(pred="IS_A", args={"subject": "hund", "object": "pflanze"})
            ]
            → Generiere Constraint: ¬(hund_IS_A_tier) ∨ ¬(hund_IS_A_pflanze)
        """
        if not self.use_ontology_constraints or not self.ontology_generator:
            return []

        clauses = []

        # Gruppiere Fakten nach Subject (für IS_A, HAS_PROPERTY, LOCATED_IN)
        subject_facts: Dict[str, List[Fact]] = {}

        for fact in facts:
            subject = fact.args.get("subject")
            if subject:
                subject = subject.lower()
                if subject not in subject_facts:
                    subject_facts[subject] = []
                subject_facts[subject].append(fact)

        # Für jeden Subject: Prüfe auf potenzielle Konflikte
        for subject, subject_fact_list in subject_facts.items():
            # IS_A Exklusivität
            is_a_facts = [f for f in subject_fact_list if f.pred == "IS_A"]
            if len(is_a_facts) > 1:
                # Prüfe ob Typen sich ausschließen
                for i, fact1 in enumerate(is_a_facts):
                    for fact2 in is_a_facts[i + 1 :]:
                        type1 = fact1.args.get("object", "").lower()
                        type2 = fact2.args.get("object", "").lower()

                        if self.ontology_generator.are_concepts_mutually_exclusive(
                            type1, type2
                        ):
                            # Erstelle Exklusivitäts-Constraint
                            lit1_name = self._fact_to_literal_name(fact1)
                            lit2_name = self._fact_to_literal_name(fact2)

                            lit1 = Literal(lit1_name, negated=False)
                            lit2 = Literal(lit2_name, negated=False)

                            # at-most-one: ¬lit1 ∨ ¬lit2
                            clause = Clause({-lit1, -lit2})
                            clauses.append(clause)

                            logger.debug(
                                f"IS_A Exclusivity Constraint: {subject} cannot be both "
                                f"'{type1}' and '{type2}' (siblings)"
                            )

            # HAS_PROPERTY Konflikte
            property_facts = [f for f in subject_fact_list if f.pred == "HAS_PROPERTY"]
            if len(property_facts) > 1:
                # Prüfe auf widersprüchliche Eigenschaften
                for i, fact1 in enumerate(property_facts):
                    for fact2 in property_facts[i + 1 :]:
                        prop1 = fact1.args.get("object", "").lower()
                        prop2 = fact2.args.get("object", "").lower()

                        # Prüfe ob Eigenschaften sich ausschließen
                        if self._are_properties_mutually_exclusive(prop1, prop2):
                            lit1_name = self._fact_to_literal_name(fact1)
                            lit2_name = self._fact_to_literal_name(fact2)

                            lit1 = Literal(lit1_name, negated=False)
                            lit2 = Literal(lit2_name, negated=False)

                            clause = Clause({-lit1, -lit2})
                            clauses.append(clause)

                            logger.debug(
                                f"Property Conflict Constraint: {subject} cannot have both "
                                f"'{prop1}' and '{prop2}' (mutually exclusive properties)"
                            )

            # LOCATED_IN Konflikte
            location_facts = [f for f in subject_fact_list if f.pred == "LOCATED_IN"]
            if len(location_facts) > 1:
                # Prüfe auf nicht-hierarchische Locations
                for i, fact1 in enumerate(location_facts):
                    for fact2 in location_facts[i + 1 :]:
                        loc1 = fact1.args.get("object", "").lower()
                        loc2 = fact2.args.get("object", "").lower()

                        # Prüfe ob Locations hierarchisch verwandt sind
                        if not self._is_location_hierarchical(loc1, loc2):
                            lit1_name = self._fact_to_literal_name(fact1)
                            lit2_name = self._fact_to_literal_name(fact2)

                            lit1 = Literal(lit1_name, negated=False)
                            lit2 = Literal(lit2_name, negated=False)

                            clause = Clause({-lit1, -lit2})
                            clauses.append(clause)

                            logger.debug(
                                f"Location Conflict Constraint: {subject} cannot be in both "
                                f"'{loc1}' and '{loc2}' (non-hierarchical locations)"
                            )

        return clauses

    def _are_properties_mutually_exclusive(self, prop1: str, prop2: str) -> bool:
        """
        Prüft ob zwei Eigenschaften sich gegenseitig ausschließen.

        Nutzt Ontologie-Generator für Property-Gruppen-Check.

        Args:
            prop1: Erste Eigenschaft
            prop2: Zweite Eigenschaft

        Returns:
            True wenn Eigenschaften sich ausschließen
        """
        if not self.ontology_generator:
            return False

        # Prüfe ob beide Eigenschaften in derselben Mutually-Exclusive-Gruppe sind
        for group_name, properties in self.ontology_generator.property_groups.items():
            if prop1 in properties and prop2 in properties:
                return True

        return False

    def _is_location_hierarchical(self, loc1: str, loc2: str) -> bool:
        """
        Prüft ob zwei Locations hierarchisch verwandt sind (PART_OF).

        Args:
            loc1: Erste Location
            loc2: Zweite Location

        Returns:
            True wenn Locations hierarchisch verwandt sind
        """
        if not self.netzwerk or not self.netzwerk.driver:
            return False

        # Query Neo4j für PART_OF Beziehung zwischen Locations
        with self.netzwerk.driver.session(database="neo4j") as session:
            # Prüfe beide Richtungen: loc1 PART_OF loc2 oder loc2 PART_OF loc1
            query = """
                MATCH path = (a:Konzept {name: $loc1})-[:PART_OF*1..3]-(b:Konzept {name: $loc2})
                RETURN count(path) AS count
            """
            result = session.run(query, loc1=loc1, loc2=loc2)
            record = result.single()

            if record and record["count"] > 0:
                return True

        return False
