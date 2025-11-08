# component_9_logik_engine_core.py
import json
import logging
import uuid
from dataclasses import dataclass, field
from time import time
from typing import Any, Dict, List, Optional, Set, Tuple

from component_1_netzwerk import KonzeptNetzwerk
from component_15_logging_config import get_logger

# Import Probabilistic Engine für Integration
try:
    from component_16_probabilistic_engine import (
        ProbabilisticEngine,
        convert_fact_to_probabilistic,
    )

    PROBABILISTIC_AVAILABLE = True
except ImportError:
    PROBABILISTIC_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "ProbabilisticEngine nicht verfügbar - läuft im deterministischen Modus"
    )

# Import Unified Proof Explanation System
try:
    UNIFIED_PROOFS_AVAILABLE = True
except ImportError:
    UNIFIED_PROOFS_AVAILABLE = False
    import logging

    logging.getLogger(__name__).warning(
        "Unified Proof Explanation System nicht verfügbar"
    )

# Import Constraint Reasoning Engine
try:
    CONSTRAINT_REASONING_AVAILABLE = True
except ImportError:
    CONSTRAINT_REASONING_AVAILABLE = False
    import logging

    logging.getLogger(__name__).warning("Constraint Reasoning Engine nicht verfügbar")

# Import SAT Solver for Boolean Reasoning
try:
    from component_30_sat_solver import DPLLSolver, KnowledgeBaseChecker

    SAT_SOLVER_AVAILABLE = True
except ImportError:
    SAT_SOLVER_AVAILABLE = False
    import logging

    logging.getLogger(__name__).warning("SAT Solver nicht verfügbar")

# Import Ontology Constraint Generator for Semantic Reasoning
try:
    from component_31_ontology_constraints import (
        OntologyConstraint,
        OntologyConstraintGenerator,
    )

    ONTOLOGY_CONSTRAINTS_AVAILABLE = True
except ImportError:
    ONTOLOGY_CONSTRAINTS_AVAILABLE = False
    import logging

    logging.getLogger(__name__).warning("Ontology Constraint Generator nicht verfügbar")

logger = get_logger(__name__)
Binding = Dict[str, Any]


@dataclass
class Fact:
    pred: str
    args: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    source: str = "kb"
    timestamp: float = field(default_factory=time)
    status: str = "believed"
    support: List[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: f"fact-{uuid.uuid4().hex[:6]}")


@dataclass
class Goal:
    """Represents a reasoning goal for backward-chaining."""

    pred: str
    args: Dict[str, Any] = field(default_factory=dict)
    depth: int = 0
    parent_id: Optional[str] = None
    id: str = field(default_factory=lambda: f"goal-{uuid.uuid4().hex[:6]}")


@dataclass
class ProofStep:
    """Represents a step in a proof/reasoning chain."""

    goal: Goal
    method: str  # "fact", "rule", "decomposition"
    bindings: Binding = field(default_factory=dict)
    subgoals: List["ProofStep"] = field(default_factory=list)
    supporting_facts: List[Fact] = field(default_factory=list)
    rule_id: Optional[str] = None
    confidence: float = 1.0


@dataclass
class Rule:
    id: str
    salience: int
    when: List[Dict[str, Any]]
    then: List[Dict[str, Any]]
    explain: str = ""
    weight: float = 1.0
    typ: str = "STANDARD"


def unify(
    pattern_args: Dict[str, Any], fact_args: Dict[str, Any], bindings: Binding
) -> Optional[Binding]:
    new_bindings = bindings.copy()
    for p_key, p_val in pattern_args.items():
        if isinstance(p_val, str) and p_val.startswith("?"):
            # Variable binding
            variable = p_val
            if p_key in fact_args:
                fact_value = fact_args[p_key]
                if variable in new_bindings and new_bindings[variable] != fact_value:
                    return None
                new_bindings[variable] = fact_value
            else:
                return None
        elif p_val is None:
            # None acts as wildcard - matches anything without binding
            if p_key not in fact_args:
                return None
            # Don't add to bindings, just allow the match
        else:
            # Concrete value - must match exactly
            if p_key not in fact_args or fact_args[p_key] != p_val:
                return None
    return new_bindings


def resolve(template: Any, bindings: Binding) -> Any:
    if isinstance(template, str) and template.startswith("?"):
        return bindings.get(template, template)
    if isinstance(template, dict):
        return {k: resolve(v, bindings) for k, v in template.items()}
    if isinstance(template, list):
        return [resolve(v, bindings) for v in template]
    return template


def match_rule(
    rule: Rule, facts: List[Fact]
) -> List[Tuple[Binding, List[Fact], float]]:
    """
    Matched eine Regel gegen die Faktenbasis.
    Unterstützt jetzt mehrere WHEN-Bedingungen (konjunktiv verknüpft).

    Returns:
        Liste von (Bindings, Support-Facts, Confidence) für alle möglichen Matches
    """
    if not rule.when:
        return []

    # Für Single-Pattern-Regeln: Alte, optimierte Logik beibehalten
    if len(rule.when) == 1:
        pattern = rule.when[0]
        pattern_pred = pattern.get("pred")
        pattern_args = pattern.get("args", {})

        all_matches = []
        for fact in facts:
            if fact.pred == pattern_pred:
                bindings = unify(pattern_args, fact.args, {})
                if bindings is not None:
                    all_matches.append((bindings, [fact], fact.confidence))
        return all_matches

    # Für Multi-Pattern-Regeln: Iteratives Matching
    return _match_multi_pattern_rule(rule, facts)


def _match_multi_pattern_rule(
    rule: Rule, facts: List[Fact]
) -> List[Tuple[Binding, List[Fact], float]]:
    """
    Matched eine Regel mit mehreren WHEN-Bedingungen.
    Alle Patterns müssen erfüllt sein (AND-Verknüpfung).
    """
    # Starte mit dem ersten Pattern
    first_pattern = rule.when[0]
    current_matches = []

    for fact in facts:
        if fact.pred == first_pattern.get("pred"):
            bindings = unify(first_pattern.get("args", {}), fact.args, {})
            if bindings is not None:
                current_matches.append((bindings, [fact], fact.confidence))

    # Erweitere schrittweise um weitere Patterns
    for pattern in rule.when[1:]:
        next_matches = []
        pattern_pred = pattern.get("pred")
        pattern_args = pattern.get("args", {})

        for bindings, support_facts, confidence in current_matches:
            # Suche Fakten, die mit den bisherigen Bindings kompatibel sind
            for fact in facts:
                if fact.pred == pattern_pred and fact not in support_facts:
                    new_bindings = unify(pattern_args, fact.args, bindings)
                    if new_bindings is not None:
                        # Kombiniere Confidence (Minimum oder Produkt, je nach Anforderung)
                        combined_confidence = min(confidence, fact.confidence)
                        next_matches.append(
                            (new_bindings, support_facts + [fact], combined_confidence)
                        )

        current_matches = next_matches
        if not current_matches:
            break  # Keine Matches mehr möglich

    return current_matches


# --- DIE ENGINE-KLASSE ---


class Engine:
    """Orchestriert den Inferenzprozess."""

    def __init__(
        self,
        netzwerk: KonzeptNetzwerk,
        use_probabilistic: bool = True,
        use_sat: bool = True,
    ):
        self.kb: List[Fact] = []
        self.wm: List[Fact] = []
        self.rules: List[Rule] = []
        self.trace: List[Dict[str, Any]] = []
        self.fired_signatures: set = set()
        self.netzwerk = netzwerk
        # EPISODIC MEMORY FOR REASONING: Tracking-Variablen
        self.current_inference_episode_id: Optional[str] = None
        self.proof_step_id_map: Dict[str, str] = (
            {}
        )  # Maps goal signatures to ProofStep IDs

        # PROBABILISTIC REASONING: Integration mit ProbabilisticEngine
        self.use_probabilistic = use_probabilistic and PROBABILISTIC_AVAILABLE
        self.prob_engine: Optional[ProbabilisticEngine] = None

        if self.use_probabilistic:
            self.prob_engine = ProbabilisticEngine()
            logger.info(
                "Logik-Engine initialisiert mit probabilistischer Unterstützung."
            )
        else:
            logger.info("Logik-Engine initialisiert im deterministischen Modus.")

        # SAT SOLVER: Integration for boolean reasoning and consistency checking
        self.use_sat = use_sat and SAT_SOLVER_AVAILABLE
        self.sat_solver: Optional[DPLLSolver] = None
        self.kb_checker: Optional[KnowledgeBaseChecker] = None

        if self.use_sat:
            self.sat_solver = DPLLSolver(use_watched_literals=True)
            self.kb_checker = KnowledgeBaseChecker(self.sat_solver)
            logger.info(
                "Logik-Engine initialisiert mit SAT-Solver für Boolean Reasoning."
            )
        else:
            logger.info("Logik-Engine läuft ohne SAT-Solver.")

        # ONTOLOGY CONSTRAINTS: Semantic constraint generation from knowledge graph
        self.use_ontology_constraints = self.use_sat and ONTOLOGY_CONSTRAINTS_AVAILABLE
        self.ontology_generator: Optional[OntologyConstraintGenerator] = None
        self._ontology_constraints_cached: List[OntologyConstraint] = []

        if self.use_ontology_constraints:
            self.ontology_generator = OntologyConstraintGenerator(
                netzwerk, enable_caching=True
            )
            logger.info(
                "Logik-Engine initialisiert mit Ontologie-Constraint-Generator."
            )
        else:
            logger.info("Logik-Engine läuft ohne Ontologie-Constraints.")

    def load_rules_from_graph(self, netzwerk: KonzeptNetzwerk):
        """Lädt alle Regeln aus dem Neo4j-Graphen."""
        logger.info("Lade Regeln aus dem Wissensgraphen...")
        if not netzwerk or not netzwerk.driver:
            logger.error("Keine Netzwerkverbindung, kann Regeln nicht laden.")
            return

        with netzwerk.driver.session(database="neo4j") as session:
            query = """
                MATCH (r:Regel)
                OPTIONAL MATCH (r)-[:WENN]->(when_node)
                WITH r, collect(properties(when_node)) AS when_clauses
                OPTIONAL MATCH (r)-[:DANN]->(then_node)
                WITH r, when_clauses, collect(properties(then_node)) AS then_clauses
                RETURN r.id AS id, r.salience AS salience, r.explain AS explain,
                       r.typ AS typ, when_clauses, then_clauses
            """
            results = session.run(query)

            rules_data = []
            for record in results:
                when_conditions = [
                    {"pred": c.get("pred"), "args": json.loads(c.get("args", "{}"))}
                    for c in record["when_clauses"]
                ]
                then_actions = [
                    {c.get("typ"): json.loads(c.get("content", "{}"))}
                    for c in record["then_clauses"]
                ]
                rules_data.append(
                    {
                        "id": record["id"],
                        "salience": record["salience"],
                        "explain": record["explain"],
                        "typ": record["typ"] or "STANDARD",
                        "when": when_conditions,
                        "then": then_actions,
                    }
                )

        self.rules = [Rule(**r) for r in rules_data]
        self.rules.sort(key=lambda r: r.salience, reverse=True)
        logger.info(f"{len(self.rules)} Regeln aus dem Graphen geladen und sortiert.")

    def add_fact(self, fact: Fact):
        self.wm.append(fact)

        # Wenn probabilistisch: Füge auch zur ProbabilisticEngine hinzu
        if self.use_probabilistic and self.prob_engine:
            prob_fact = convert_fact_to_probabilistic(fact)
            self.prob_engine.add_fact(prob_fact)

    def run(self):
        self.trace = []
        self.fired_signatures = set()
        has_changed = True
        while has_changed:
            has_changed = False
            agenda = self._populate_agenda()
            if not agenda:
                break

            agenda.sort(key=lambda x: x[0].salience, reverse=True)
            rule, bindings, support_facts, confidence = agenda[0]

            if self._apply_rule(rule, bindings, support_facts, confidence):
                has_changed = True

    def _populate_agenda(self) -> list:
        agenda = []
        for rule in self.rules:
            matches = match_rule(rule, self.wm + self.kb)
            for bindings, support_facts, confidence in matches:
                signature = (rule.id, tuple(sorted(f.id for f in support_facts)))
                if signature not in self.fired_signatures:
                    agenda.append((rule, bindings, support_facts, confidence))
        return agenda

    def _apply_rule(
        self,
        rule: Rule,
        bindings: Binding,
        support_facts: List[Fact],
        confidence: float,
    ) -> bool:
        logger.info(f"FEUERE REGEL: {rule.id} mit Bindings: {bindings}")
        signature = (rule.id, tuple(sorted(f.id for f in support_facts)))
        self.fired_signatures.add(signature)

        new_facts_created = False
        for action_template in rule.then:
            action = resolve(action_template, bindings)
            if "assert" in action:
                new_fact_data = action["assert"]
                new_fact = Fact(
                    pred=new_fact_data["pred"],
                    args=new_fact_data.get("args", {}),
                    confidence=confidence,
                    source=f"rule:{rule.id}",
                    support=[rule.id] + [f.id for f in support_facts],
                )
                self.add_fact(new_fact)
                logger.info(f"  => Neuer Fakt: {new_fact}")
                new_facts_created = True

        self.trace.append({"rule": rule.id, "bindings": bindings})
        return new_facts_created

    def get_goals(self) -> List[Fact]:
        return [f for f in self.wm if f.pred == "goal"]

    def clear_wm(self):
        self.wm = []

    # ==================== BACKWARD-CHAINING (Phase 3) ====================

    def prove_goal(
        self, goal: Goal, max_depth: int = 5, visited: Optional[Set[str]] = None
    ) -> Optional[ProofStep]:
        """
        Versucht ein Ziel (Goal) durch Backward-Chaining zu beweisen.

        Strategien:
        1. Direkte Faktensuche (Base Case)
        2. Regelanwendung mit Subgoal-Zerlegung (Recursive Case)
        3. Graph-Traversal für Multi-Hop-Reasoning

        Args:
            goal: Das zu beweisende Ziel
            max_depth: Maximale Rekursionstiefe
            visited: Set von besuchten Goal-IDs zur Zyklusvermeidung

        Returns:
            ProofStep mit vollständigem Beweistrail oder None
        """
        if visited is None:
            visited = set()

        # Zykluserkennung
        goal_signature = self._goal_signature(goal)
        if goal_signature in visited:
            logger.debug(f"Zyklus erkannt für Goal: {goal_signature}")
            return None

        if goal.depth > max_depth:
            logger.debug(f"Max depth {max_depth} erreicht für Goal: {goal.pred}")
            return None

        visited.add(goal_signature)
        logger.info(f"[BC] Beweise Goal (depth={goal.depth}): {goal.pred} {goal.args}")

        # Strategie 1: Suche direkt in Faktenbasis
        proof = self._try_prove_by_fact(goal)
        if proof:
            logger.info(f"  [OK] Direkt durch Fakt bewiesen: {goal.pred}")
            return proof

        # Strategie 2: Suche anwendbare Regeln
        proof = self._try_prove_by_rule(goal, max_depth, visited)
        if proof:
            logger.info(f"  [OK] Durch Regel bewiesen: {goal.pred}")
            return proof

        # Strategie 3: Graph-Traversal (Multi-Hop via Neo4j)
        proof = self._try_prove_by_graph(goal, max_depth, visited)
        if proof:
            logger.info(f"  [OK] Durch Graph-Traversal bewiesen: {goal.pred}")
            return proof

        logger.info(f"  [X] Goal nicht beweisbar: {goal.pred}")
        visited.remove(goal_signature)
        return None

    def _goal_signature(self, goal: Goal) -> str:
        """Erstellt eindeutige Signatur für Zykluserkennung."""
        args_str = json.dumps(goal.args, sort_keys=True)
        return f"{goal.pred}:{args_str}"

    def _try_prove_by_fact(self, goal: Goal) -> Optional[ProofStep]:
        """
        Versucht Goal durch direkten Faktenmatch zu beweisen.
        """
        all_facts = self.wm + self.kb

        for fact in all_facts:
            if fact.pred != goal.pred:
                continue

            # Versuche zu unifizieren
            bindings = unify(goal.args, fact.args, {})
            if bindings is not None:
                return ProofStep(
                    goal=goal,
                    method="fact",
                    bindings=bindings,
                    supporting_facts=[fact],
                    confidence=fact.confidence,
                )

        return None

    def _try_prove_by_rule(
        self, goal: Goal, max_depth: int, visited: Set[str]
    ) -> Optional[ProofStep]:
        """
        Versucht Goal durch Regelanwendung zu beweisen.
        Zerlegt Goal in Subgoals basierend auf WHEN-Klauseln.
        """
        for rule in self.rules:
            # Prüfe ob Regel das Goal produzieren kann
            for then_template in rule.then:
                if "assert" not in then_template:
                    continue

                then_fact = then_template["assert"]
                if then_fact["pred"] != goal.pred:
                    continue

                # Versuche THEN-Teil mit Goal zu unifizieren
                # Wichtig: Pattern ist THEN-Fakt (mit Variablen), Ziel ist Goal (mit Werten)
                initial_bindings = unify(then_fact.get("args", {}), goal.args, {})
                if initial_bindings is None:
                    continue

                # Erstelle Subgoals aus WHEN-Klauseln
                subgoals = []
                for when_clause in rule.when:
                    subgoal_args = resolve(
                        when_clause.get("args", {}), initial_bindings
                    )
                    subgoal = Goal(
                        pred=when_clause["pred"],
                        args=subgoal_args,
                        depth=goal.depth + 1,
                        parent_id=goal.id,
                    )
                    subgoals.append(subgoal)

                # Versuche alle Subgoals zu beweisen (konjunktiv)
                subproofs = []
                combined_bindings = initial_bindings.copy()
                min_confidence = 1.0

                for subgoal in subgoals:
                    # Aktualisiere Subgoal-Args mit bisherigen Bindings
                    subgoal.args = resolve(subgoal.args, combined_bindings)

                    subproof = self.prove_goal(subgoal, max_depth, visited.copy())
                    if subproof is None:
                        break  # Ein Subgoal fehlgeschlagen

                    subproofs.append(subproof)
                    # Merge bindings
                    combined_bindings.update(subproof.bindings)
                    min_confidence = min(min_confidence, subproof.confidence)
                else:
                    # Alle Subgoals erfolgreich bewiesen
                    return ProofStep(
                        goal=goal,
                        method="rule",
                        bindings=combined_bindings,
                        subgoals=subproofs,
                        rule_id=rule.id,
                        confidence=min_confidence,
                    )

        return None

    def _try_prove_by_graph(
        self, goal: Goal, max_depth: int, visited: Set[str]
    ) -> Optional[ProofStep]:
        """
        Versucht Goal durch Multi-Hop-Reasoning über den Neo4j-Graphen zu beweisen.

        Beispiel:
        Goal: IS_A(?x, "tier")
        Graph: hund -IS_A-> säugetier -IS_A-> tier
        => Findet transitive Beziehungen
        """
        if not self.netzwerk or not self.netzwerk.driver:
            return None

        # Nur für relationale Predicates sinnvoll
        if goal.pred not in ["IS_A", "PART_OF", "LOCATED_IN", "CAPABLE_OF"]:
            return None

        # Extrahiere Subject und Object aus Goal
        subject = goal.args.get("subject")
        object_entity = goal.args.get("object")

        if not subject and not object_entity:
            return None

        # Cypher-Query für Multi-Hop-Pfadsuche
        # Beziehungen existieren zwischen Konzept-Nodes
        with self.netzwerk.driver.session(database="neo4j") as session:
            # Fall 1: Subject und Object bekannt - prüfe Pfad
            if subject and object_entity:
                query = (
                    """
                    MATCH path = (start:Konzept {name: $subject})-[r:%s*1..3]->(end:Konzept {name: $object})
                    RETURN length(path) AS hops
                    ORDER BY hops ASC
                    LIMIT 1
                """
                    % goal.pred
                )
                result = session.run(
                    query, subject=subject.lower(), object=object_entity.lower()
                )

                for record in result:
                    # Pfad existiert
                    derived_fact = Fact(
                        pred=goal.pred,
                        args={"subject": subject, "object": object_entity},
                        confidence=1.0 / (1 + record["hops"]),
                        source=f"graph_traversal:{goal.pred}",
                    )

                    proof = ProofStep(
                        goal=goal,
                        method="graph_traversal",
                        bindings={"?x": subject, "?y": object_entity},
                        supporting_facts=[derived_fact],
                        confidence=derived_fact.confidence,
                    )

                    logger.info(
                        f"  Graph-Pfad gefunden: {subject} -[{goal.pred}*{record['hops']}]-> {object_entity}"
                    )
                    return proof

            # Fall 2: Subject bekannt, Object unbekannt
            elif subject and not object_entity:
                query = (
                    """
                    MATCH path = (start:Konzept {name: $subject})-[r:%s*1..3]->(end:Konzept)
                    RETURN end.name AS object, length(path) AS hops
                    ORDER BY hops ASC
                    LIMIT 5
                """
                    % goal.pred
                )
                result = session.run(query, subject=subject.lower())

                for record in result:
                    # Erstelle Fakt aus Graph-Pfad
                    derived_fact = Fact(
                        pred=goal.pred,
                        args={"subject": subject, "object": record["object"]},
                        confidence=1.0 / (1 + record["hops"]),  # Abnehmende Confidence
                        source=f"graph_traversal:{goal.pred}",
                    )

                    proof = ProofStep(
                        goal=goal,
                        method="graph_traversal",
                        bindings={"?x": subject, "?y": record["object"]},
                        supporting_facts=[derived_fact],
                        confidence=derived_fact.confidence,
                    )

                    logger.info(
                        f"  Graph-Pfad gefunden: {subject} -[{goal.pred}*{record['hops']}]-> {record['object']}"
                    )
                    return proof

            # Fall 3: Object bekannt, Subject unbekannt (Rückwärtssuche)
            elif object_entity and not subject:
                query = (
                    """
                    MATCH path = (start:Konzept)-[r:%s*1..3]->(end:Konzept {name: $object})
                    RETURN start.name AS subject, length(path) AS hops
                    ORDER BY hops ASC
                    LIMIT 5
                """
                    % goal.pred
                )
                result = session.run(query, object=object_entity.lower())

                for record in result:
                    derived_fact = Fact(
                        pred=goal.pred,
                        args={"subject": record["subject"], "object": object_entity},
                        confidence=1.0 / (1 + record["hops"]),
                        source=f"graph_traversal:{goal.pred}",
                    )

                    proof = ProofStep(
                        goal=goal,
                        method="graph_traversal",
                        bindings={"?x": record["subject"], "?y": object_entity},
                        supporting_facts=[derived_fact],
                        confidence=derived_fact.confidence,
                    )

                    return proof

        return None
