# component_14_abductive_engine.py
"""
Abductive Reasoning Engine for KAI

Generates explanatory hypotheses when deductive reasoning fails.
Implements three hypothesis generation strategies:
- Template-based: Match causal patterns
- Analogy-based: Transfer explanations from similar cases
- Causal chain: Backward reasoning from effects

Scores hypotheses using multiple criteria:
- Coverage: Explains all observations
- Simplicity: Occam's razor (fewer assumptions)
- Coherence: Fits existing knowledge
- Specificity: Generates testable predictions
"""

import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Import existing components
from component_9_logik_engine import Fact

logger = logging.getLogger(__name__)

# Import Unified Proof Explanation System
try:
    from component_17_proof_explanation import ProofStep as UnifiedProofStep
    from component_17_proof_explanation import (
        StepType,
        generate_explanation_text,
    )

    UNIFIED_PROOFS_AVAILABLE = True
except ImportError:
    UNIFIED_PROOFS_AVAILABLE = False


@dataclass
class Hypothesis:
    """Represents an abduced explanatory hypothesis."""

    id: str
    explanation: str  # Natural language explanation
    observations: List[str]  # What it explains
    abduced_facts: List[Fact]  # New facts it proposes
    strategy: str  # "template" | "analogy" | "causal_chain"
    confidence: float  # 0.0-1.0 overall score
    scores: Dict[str, float]  # {coverage, simplicity, coherence, specificity}
    timestamp: datetime = field(default_factory=datetime.now)
    sources: List[str] = field(default_factory=list)  # What knowledge was used
    reasoning_trace: str = ""  # How hypothesis was generated


@dataclass
class CausalPattern:
    """Template for causal reasoning."""

    pattern_type: str  # "CAUSES", "ENABLES", "PROPERTY_OF", "PART_OF"
    template: str  # Natural language template
    forward: str  # X -> Y
    backward: str  # Y observed -> hypothesize X


class AbductiveEngine:
    """
    Main engine for abductive reasoning.

    Generates and scores explanatory hypotheses when deductive reasoning fails.
    """

    def __init__(self, netzwerk, logic_engine=None):
        """
        Initialize abductive engine.

        Args:
            netzwerk: KonzeptNetzwerk instance for graph queries
            logic_engine: Optional Engine instance for deductive verification
        """
        self.netzwerk = netzwerk
        self.logic_engine = logic_engine

        # Define causal patterns for template-based reasoning
        self.causal_patterns = [
            CausalPattern(
                pattern_type="CAUSES",
                template="{X} verursacht {Y}",
                forward="Wenn {X}, dann {Y}",
                backward="Wenn {Y} beobachtet, dann möglicherweise {X}",
            ),
            CausalPattern(
                pattern_type="ENABLES",
                template="{X} ermöglicht {Y}",
                forward="Mit {X} kann {Y} passieren",
                backward="Wenn {Y} passiert, dann war möglicherweise {X} vorhanden",
            ),
            CausalPattern(
                pattern_type="HAS_PROPERTY",
                template="{X} hat Eigenschaft {Y}",
                forward="{X} ist {Y}",
                backward="Wenn etwas {Y} ist, könnte es {X} sein",
            ),
            CausalPattern(
                pattern_type="PART_OF",
                template="{X} ist Teil von {Y}",
                forward="{Y} besteht aus {X}",
                backward="Wenn {Y} existiert, dann auch {X}",
            ),
        ]

        # Scoring weights (can be tuned)
        self.score_weights = {
            "coverage": 0.3,
            "simplicity": 0.2,
            "coherence": 0.3,
            "specificity": 0.2,
        }

    def generate_hypotheses(
        self,
        observation: str,
        context_facts: List[Fact] = None,
        strategies: List[str] = None,
        max_hypotheses: int = 10,
    ) -> List[Hypothesis]:
        """
        Generate explanatory hypotheses for an observation.

        Args:
            observation: The observation to explain (e.g., "Der Boden ist nass")
            context_facts: Known facts for context
            strategies: Which strategies to use (default: all)
            max_hypotheses: Maximum number of hypotheses to return

        Returns:
            List of Hypothesis objects, ranked by confidence
        """
        if strategies is None:
            strategies = ["template", "analogy", "causal_chain"]

        if context_facts is None:
            context_facts = []

        all_hypotheses = []

        # Extract key concepts from observation
        key_concepts = self._extract_concepts(observation)

        # Strategy 1: Template-based
        if "template" in strategies:
            template_hypotheses = self._generate_template_hypotheses(
                observation, key_concepts, context_facts
            )
            all_hypotheses.extend(template_hypotheses)

        # Strategy 2: Analogy-based
        if "analogy" in strategies:
            analogy_hypotheses = self._generate_analogy_hypotheses(
                observation, key_concepts, context_facts
            )
            all_hypotheses.extend(analogy_hypotheses)

        # Strategy 3: Causal chain
        if "causal_chain" in strategies:
            causal_hypotheses = self._generate_causal_chain_hypotheses(
                observation, key_concepts, context_facts
            )
            all_hypotheses.extend(causal_hypotheses)

        # Score all hypotheses
        for hypothesis in all_hypotheses:
            self._score_hypothesis(hypothesis, context_facts)

        # Rank by confidence
        all_hypotheses.sort(key=lambda h: h.confidence, reverse=True)

        return all_hypotheses[:max_hypotheses]

    def _extract_concepts(self, text: str) -> List[str]:
        """
        Extract key concepts from text.
        Simple implementation: extract nouns and significant words.
        """
        # Remove common words
        stopwords = {
            "der",
            "die",
            "das",
            "ein",
            "eine",
            "ist",
            "sind",
            "hat",
            "haben",
            "war",
            "waren",
        }

        # Tokenize and filter
        words = re.findall(r"\b\w+\b", text.lower())
        concepts = [w for w in words if w not in stopwords and len(w) > 2]

        return concepts

    def _generate_template_hypotheses(
        self, observation: str, concepts: List[str], context_facts: List[Fact]
    ) -> List[Hypothesis]:
        """
        Generate hypotheses by matching causal patterns.

        Strategy: For each concept in observation, query graph for relations
        that could explain it (CAUSES, ENABLES, etc.).
        """
        hypotheses = []

        for concept in concepts:
            # Query graph for potential causes
            facts = self.netzwerk.query_graph_for_facts(concept)

            for pattern in self.causal_patterns:
                rel_type = pattern.pattern_type

                # Check if this relation type exists in facts
                if rel_type in facts:
                    for related_concept in facts[rel_type]:
                        # Generate hypothesis
                        explanation = pattern.backward.format(
                            X=related_concept, Y=concept
                        )

                        # Create abduced fact
                        abduced_fact = Fact(
                            pred=rel_type,
                            args={"subject": related_concept, "object": concept},
                            id=f"abduced_{uuid.uuid4().hex[:8]}",
                            confidence=0.7,  # Lower confidence for abduced facts
                        )

                        hypothesis = Hypothesis(
                            id=f"hyp_{uuid.uuid4().hex[:8]}",
                            explanation=explanation,
                            observations=[observation],
                            abduced_facts=[abduced_fact],
                            strategy="template",
                            confidence=0.0,  # Will be scored later
                            scores={},
                            sources=[f"Pattern: {pattern.pattern_type}"],
                            reasoning_trace=f"Matched pattern: {pattern.template}",
                        )

                        hypotheses.append(hypothesis)

        return hypotheses

    def _generate_analogy_hypotheses(
        self, observation: str, concepts: List[str], context_facts: List[Fact]
    ) -> List[Hypothesis]:
        """
        Generate hypotheses by finding similar cases in knowledge base.

        Strategy: Find concepts similar to those in observation,
        then transfer explanations from those cases.
        """
        hypotheses = []

        for concept in concepts:
            # Find similar concepts (same IS_A hierarchy)
            facts = self.netzwerk.query_graph_for_facts(concept)

            if "IS_A" in facts:
                # Get parent categories
                categories = facts["IS_A"]

                # Find other members of same category
                for category in categories:
                    # Query for other things in this category
                    # (This requires a reverse query - find X where X IS_A category)
                    similar_concepts = self._find_similar_concepts(category)

                    for similar in similar_concepts:
                        if similar == concept:
                            continue  # Skip self

                        # Get facts about similar concept
                        similar_facts = self.netzwerk.query_graph_for_facts(similar)

                        # Transfer properties
                        for rel_type, related in similar_facts.items():
                            if rel_type in ["HAS_PROPERTY", "CAPABLE_OF", "LOCATED_IN"]:
                                for prop in related:
                                    explanation = (
                                        f"{concept} könnte {prop} sein/haben, "
                                        f"wie {similar} (beide sind {category})"
                                    )

                                    abduced_fact = Fact(
                                        pred=rel_type,
                                        args={"subject": concept, "object": prop},
                                        id=f"abduced_{uuid.uuid4().hex[:8]}",
                                        confidence=0.6,  # Lower confidence for analogies
                                    )

                                    hypothesis = Hypothesis(
                                        id=f"hyp_{uuid.uuid4().hex[:8]}",
                                        explanation=explanation,
                                        observations=[observation],
                                        abduced_facts=[abduced_fact],
                                        strategy="analogy",
                                        confidence=0.0,
                                        scores={},
                                        sources=[f"Analogy: {similar}"],
                                        reasoning_trace=(
                                            f"Found similar concept '{similar}' "
                                            f"(both are '{category}')"
                                        ),
                                    )

                                    hypotheses.append(hypothesis)

        return hypotheses

    def _find_similar_concepts(self, category: str) -> List[str]:
        """
        Find concepts that are members of the given category.

        Runs reverse IS_A query: find X where (X)-[:IS_A]->(category)
        """
        if not self.netzwerk.driver:
            return []

        with self.netzwerk.driver.session(database="neo4j") as session:
            result = session.run(
                """
                MATCH (w:Wort)-[:IS_A]->(cat:Konzept {text: $category})
                RETURN w.text AS concept
                LIMIT 10
                """,
                category=category,
            )

            return [record["concept"] for record in result]

    def _generate_causal_chain_hypotheses(
        self, observation: str, concepts: List[str], context_facts: List[Fact]
    ) -> List[Hypothesis]:
        """
        Generate hypotheses by tracing causal chains backward.

        Strategy: Start from observed effect, follow causal relations
        backward to find potential root causes.
        """
        hypotheses = []

        for concept in concepts:
            # Find causal chains ending at this concept
            chains = self._find_causal_chains(concept, max_depth=3)

            for chain in chains:
                # chain is a list of (concept, relation) tuples
                # Example: [("regen", "CAUSES"), ("wolken", "CAUSES")]

                if len(chain) == 0:
                    continue

                # Build explanation from chain
                explanation_parts = []
                abduced_facts = []

                prev_concept = concept
                for cause, relation in chain:
                    explanation_parts.append(f"{cause} -> {relation} -> {prev_concept}")

                    abduced_fact = Fact(
                        pred=relation,
                        args={"subject": cause, "object": prev_concept},
                        id=f"abduced_{uuid.uuid4().hex[:8]}",
                        confidence=0.8 / len(chain),  # Longer chains less confident
                    )
                    abduced_facts.append(abduced_fact)

                    prev_concept = cause

                explanation = (
                    f"Kausale Kette: {' -> '.join(reversed(explanation_parts))}"
                )

                hypothesis = Hypothesis(
                    id=f"hyp_{uuid.uuid4().hex[:8]}",
                    explanation=explanation,
                    observations=[observation],
                    abduced_facts=abduced_facts,
                    strategy="causal_chain",
                    confidence=0.0,
                    scores={},
                    sources=[f"Causal chain of length {len(chain)}"],
                    reasoning_trace=f"Traced {len(chain)} causal steps backward",
                )

                hypotheses.append(hypothesis)

        return hypotheses

    def _find_causal_chains(
        self, effect: str, max_depth: int = 3
    ) -> List[List[Tuple[str, str]]]:
        """
        Find causal chains ending at the given effect.

        Returns list of chains, where each chain is a list of (cause, relation) tuples.
        """
        if not self.netzwerk.driver:
            return []

        chains = []

        # Query for immediate causes
        facts = self.netzwerk.query_graph_for_facts(effect)

        # Look for causal relations (reversed - things that cause this effect)
        for rel_type in ["CAUSES", "ENABLES", "LEADS_TO"]:
            if rel_type in facts:
                for cause in facts[rel_type]:
                    # Start a chain
                    chain = [(cause, rel_type)]

                    # Recursively extend chain if depth allows
                    if max_depth > 1:
                        sub_chains = self._find_causal_chains(cause, max_depth - 1)
                        if sub_chains:
                            for sub_chain in sub_chains:
                                extended_chain = chain + sub_chain
                                chains.append(extended_chain)
                        else:
                            chains.append(chain)
                    else:
                        chains.append(chain)

        return chains

    def _score_hypothesis(
        self, hypothesis: Hypothesis, context_facts: List[Fact]
    ) -> None:
        """
        Score hypothesis using multiple criteria.
        Updates hypothesis.scores and hypothesis.confidence in-place.
        """
        scores = {}

        # 1. Coverage: Does it explain all observations?
        scores["coverage"] = self._score_coverage(hypothesis)

        # 2. Simplicity: Occam's razor (fewer assumptions = better)
        scores["simplicity"] = self._score_simplicity(hypothesis)

        # 3. Coherence: Fits existing knowledge?
        scores["coherence"] = self._score_coherence(hypothesis, context_facts)

        # 4. Specificity: Generates testable predictions?
        scores["specificity"] = self._score_specificity(hypothesis)

        # Calculate weighted average
        confidence = sum(
            scores[criterion] * self.score_weights[criterion] for criterion in scores
        )

        hypothesis.scores = scores
        hypothesis.confidence = confidence

    def _score_coverage(self, hypothesis: Hypothesis) -> float:
        """
        Score how well hypothesis covers all observations.

        Simple version: 1.0 if at least one observation, can be extended.
        """
        if len(hypothesis.observations) > 0:
            return 1.0
        return 0.0

    def _score_simplicity(self, hypothesis: Hypothesis) -> float:
        """
        Score simplicity (Occam's razor).

        Fewer abduced facts = simpler = better.
        """
        num_facts = len(hypothesis.abduced_facts)

        if num_facts == 0:
            return 0.0  # Empty hypothesis
        elif num_facts == 1:
            return 1.0  # Single fact (simplest)
        elif num_facts <= 3:
            return 0.7  # Few facts
        else:
            return 0.4  # Many facts (complex)

    def _score_coherence(
        self, hypothesis: Hypothesis, context_facts: List[Fact]
    ) -> float:
        """
        Score coherence with existing knowledge.

        Check if abduced facts contradict or align with known facts.
        """
        if len(hypothesis.abduced_facts) == 0:
            return 0.5  # Neutral

        coherent_count = 0
        total_count = len(hypothesis.abduced_facts)

        for abduced in hypothesis.abduced_facts:
            # Check if this fact is already known
            is_known = self._is_fact_known(abduced)

            if is_known:
                coherent_count += 1  # Aligns with existing knowledge
            else:
                # Check if it contradicts
                contradicts = self._contradicts_knowledge(abduced)
                if not contradicts:
                    coherent_count += 0.5  # Doesn't contradict (neutral)
                # If contradicts, add 0 (penalize)

        return coherent_count / total_count

    def _score_specificity(self, hypothesis: Hypothesis) -> float:
        """
        Score specificity (generates testable predictions).

        More specific facts = more testable = better.
        """
        if len(hypothesis.abduced_facts) == 0:
            return 0.0

        # Count facts with concrete (non-variable) arguments
        specific_count = 0
        for fact in hypothesis.abduced_facts:
            has_variables = any(str(v).startswith("?") for v in fact.args.values())
            if not has_variables:
                specific_count += 1

        return specific_count / len(hypothesis.abduced_facts)

    def _is_fact_known(self, fact: Fact) -> bool:
        """
        Check if a fact is already in the knowledge base.
        """
        if "subject" in fact.args and "object" in fact.args:
            subject = fact.args["subject"]
            obj = fact.args["object"]

            # Query graph
            facts = self.netzwerk.query_graph_for_facts(subject)

            if fact.pred in facts:
                return obj in facts[fact.pred]

        return False

    def _get_facts_about_subject(self, subject: str) -> List[Fact]:
        """
        Hole alle Fakten über ein Subjekt aus der Knowledge Base.

        Konvertiert die Ergebnisse von query_graph_for_facts() in Fact-Objekte
        für SAT-basierte Konsistenzprüfung.

        Args:
            subject: Das Subjekt (z.B. "hund")

        Returns:
            Liste von Fact-Objekten über das Subjekt
        """
        facts_list = []

        # Query graph für alle Relationen des Subjekts
        facts_dict = self.netzwerk.query_graph_for_facts(subject)

        # Konvertiere zu Fact-Objekten
        for relation_type, objects in facts_dict.items():
            for obj in objects:
                fact = Fact(
                    pred=relation_type,
                    args={"subject": subject, "object": obj},
                    id=f"kb_{uuid.uuid4().hex[:8]}",
                    confidence=1.0,  # Existierende KB-Fakten haben hohe Konfidenz
                )
                facts_list.append(fact)

        logger.debug(
            f"Extrahierte {len(facts_list)} Fakten aus KB für Subjekt '{subject}'"
        )

        return facts_list

    def _contradicts_knowledge(self, fact: Fact) -> bool:
        """
        Check if a fact contradicts existing knowledge.

        **PHASE 4.2 ERWEITERUNG**: Nutzt SAT-Solver für robuste Konsistenzprüfung.

        Vorher: Heuristische Kategorie-Checks (IS_A, HAS_PROPERTY, LOCATED_IN)
        Nachher: Formale SAT-basierte Konsistenzprüfung + Heuristische Fallbacks

        Detects contradictions via:
        1. SAT-based consistency checking (primary method, if available)
        2. Heuristic category checks (fallback):
           - Mutually exclusive IS_A relations
           - Contradictory properties (colors, sizes, etc.)
           - Incompatible locations

        Args:
            fact: The fact to check for contradictions

        Returns:
            True if fact contradicts existing knowledge, False otherwise
        """
        if "subject" not in fact.args or "object" not in fact.args:
            return False

        subject = fact.args["subject"]
        obj = fact.args["object"]

        # ==================== PHASE 4.2: SAT-BASED CONSISTENCY CHECK ====================
        # Versuche SAT-basierte Konsistenzprüfung (falls Logic Engine verfügbar)
        if self.logic_engine and hasattr(self.logic_engine, "check_consistency"):
            try:
                logger.debug(
                    f"SAT-basierte Konsistenzprüfung für {fact.pred}({fact.args})"
                )

                # Hole alle relevanten Fakten aus der Knowledge Base
                existing_facts_list = self._get_facts_about_subject(subject)

                if existing_facts_list:
                    # Prüfe ob [existing_facts + new_fact] konsistent ist
                    all_facts = existing_facts_list + [fact]
                    is_consistent = self.logic_engine.check_consistency(all_facts)

                    if not is_consistent:
                        logger.info(
                            f"SAT-Solver: Widerspruch gefunden für "
                            f"{fact.pred}({subject} → {obj})"
                        )
                        return True
                    else:
                        # SAT sagt: konsistent → Kein Widerspruch
                        logger.debug(
                            f"SAT-Solver: Kein Widerspruch für "
                            f"{fact.pred}({subject} → {obj})"
                        )
                        return False

            except Exception as e:
                logger.warning(
                    f"SAT-basierte Konsistenzprüfung fehlgeschlagen: {e}. "
                    f"Fallback auf heuristische Prüfung."
                )
                # Fallback auf heuristische Prüfung (siehe unten)

        # ==================== FALLBACK: HEURISTIC CONSISTENCY CHECKS ====================
        # Query existing facts about the subject
        existing_facts = self.netzwerk.query_graph_for_facts(subject)

        # CATEGORY 1: Mutually Exclusive IS_A Relations
        # Ein Objekt kann nicht gleichzeitig verschiedene fundamentale Kategorien haben
        # z.B. "Hund" kann nicht gleichzeitig IS_A Katze sein
        if fact.pred == "IS_A":
            if "IS_A" in existing_facts:
                for existing_type in existing_facts["IS_A"]:
                    if existing_type != obj:
                        # Prüfe ob die beiden Typen inkompatibel sind
                        # (vereinfachte Version: unterschiedliche IS_A = potentieller Widerspruch)
                        # Ausnahme: Hierarchien (z.B. "Hund IS_A Säugetier" UND "Hund IS_A Tier" ist OK)
                        # Für jetzt: nur direkte Widersprüche auf gleicher Ebene
                        if self._are_types_mutually_exclusive(existing_type, obj):
                            logger.debug(
                                f"Widerspruch gefunden: {subject} kann nicht gleichzeitig "
                                f"'{existing_type}' UND '{obj}' sein (IS_A Konflikt)"
                            )
                            return True

        # CATEGORY 2: Contradictory Properties
        # Eigenschaften die sich gegenseitig ausschließen (z.B. Farben)
        if fact.pred == "HAS_PROPERTY":
            if "HAS_PROPERTY" in existing_facts:
                for existing_prop in existing_facts["HAS_PROPERTY"]:
                    if self._are_properties_contradictory(existing_prop, obj):
                        logger.debug(
                            f"Widerspruch gefunden: {subject} kann nicht gleichzeitig "
                            f"'{existing_prop}' UND '{obj}' haben (Property Konflikt)"
                        )
                        return True

        # CATEGORY 3: Incompatible Locations
        # Ein Objekt kann nicht gleichzeitig an mehreren Orten sein
        if fact.pred == "LOCATED_IN":
            if "LOCATED_IN" in existing_facts:
                for existing_location in existing_facts["LOCATED_IN"]:
                    if existing_location != obj:
                        # Ausnahme: Hierarchien (z.B. "Berlin" LOCATED_IN "Deutschland" ist OK)
                        # Für jetzt: unterschiedliche Locations = Widerspruch
                        if not self._is_location_hierarchy(existing_location, obj):
                            logger.debug(
                                f"Widerspruch gefunden: {subject} kann nicht gleichzeitig "
                                f"in '{existing_location}' UND '{obj}' sein (Location Konflikt)"
                            )
                            return True

        return False

    def _are_types_mutually_exclusive(self, type1: str, type2: str) -> bool:
        """
        Prüft ob zwei IS_A Typen sich gegenseitig ausschließen.

        Heuristik: Wenn beide Typen konkrete Objekt-Kategorien sind (keine abstrakten Kategorien),
        dann schließen sie sich aus (z.B. "Hund" vs "Katze").

        Abstrakte Kategorien wie "Tier", "Lebewesen", "Objekt" sind hierarchisch und OK.

        Args:
            type1: Erster Typ
            type2: Zweiter Typ

        Returns:
            True wenn die Typen sich ausschließen
        """
        # Liste abstrakter Kategorien (hierarchisch OK)
        abstract_categories = {
            "objekt",
            "ding",
            "sache",
            "entität",
            "lebewesen",
            "tier",
            "pflanze",
            "organismus",
            "konzept",
            "idee",
            "abstraktum",
        }

        type1_lower = type1.lower()
        type2_lower = type2.lower()

        # Wenn beide abstrakt sind, kein Widerspruch
        if type1_lower in abstract_categories and type2_lower in abstract_categories:
            return False

        # Wenn einer der beiden abstrakt ist, prüfe ob der andere davon abgeleitet ist
        # (Vereinfachung: wenn einer abstrakt ist, kein Widerspruch)
        if type1_lower in abstract_categories or type2_lower in abstract_categories:
            return False

        # Prüfe ob type2 in der Hierarchie von type1 ist (oder umgekehrt)
        # Dafür müssten wir die IS_A Hierarchie abfragen
        if self._is_subtype_of(type1, type2) or self._is_subtype_of(type2, type1):
            return False

        # Ansonsten: konkrete unterschiedliche Typen = potentieller Widerspruch
        return True

    def _is_subtype_of(self, subtype: str, supertype: str) -> bool:
        """
        Prüft ob subtype ein Untertyp von supertype ist (via IS_A Hierarchie).

        Args:
            subtype: Der potentielle Untertyp
            supertype: Der potentielle Obertyp

        Returns:
            True wenn subtype ein Untertyp von supertype ist
        """
        # Abfrage der IS_A Hierarchie
        facts = self.netzwerk.query_graph_for_facts(subtype)
        if "IS_A" in facts:
            # Wenn subtype direkt IS_A supertype ist
            if supertype in facts["IS_A"]:
                return True
            # Rekursiv prüfen (transitive IS_A)
            for parent in facts["IS_A"]:
                if self._is_subtype_of(parent, supertype):
                    return True
        return False

    def _are_properties_contradictory(self, prop1: str, prop2: str) -> bool:
        """
        Prüft ob zwei Eigenschaften sich widersprechen.

        Heuristik: Farben, Größen, und andere messbare Eigenschaften schließen sich aus.

        Args:
            prop1: Erste Eigenschaft
            prop2: Zweite Eigenschaft

        Returns:
            True wenn die Eigenschaften sich widersprechen
        """
        prop1_lower = prop1.lower()
        prop2_lower = prop2.lower()

        # Kategorie 1: Farben
        colors = {
            "rot",
            "blau",
            "grün",
            "gelb",
            "orange",
            "lila",
            "schwarz",
            "weiß",
            "grau",
            "braun",
        }
        if prop1_lower in colors and prop2_lower in colors:
            return prop1_lower != prop2_lower

        # Kategorie 2: Größen (relativ)
        sizes = {"groß", "klein", "mittel", "riesig", "winzig"}
        if prop1_lower in sizes and prop2_lower in sizes:
            return prop1_lower != prop2_lower

        # Kategorie 3: Temperaturen (relativ)
        temperatures = {"heiß", "kalt", "warm", "kühl", "eiskalt"}
        if prop1_lower in temperatures and prop2_lower in temperatures:
            # Einige Kombinationen sind OK (z.B. "warm" und "heiß" sind nicht widersprüchlich)
            # Aber "heiß" und "kalt" schon
            opposites = [
                ("heiß", "kalt"),
                ("warm", "kalt"),
                ("heiß", "kühl"),
                ("warm", "eiskalt"),
                ("heiß", "eiskalt"),
            ]
            for a, b in opposites:
                if (prop1_lower == a and prop2_lower == b) or (
                    prop1_lower == b and prop2_lower == a
                ):
                    return True

        # Kategorie 4: Zustände (binär)
        binary_states = {
            ("lebendig", "tot"),
            ("aktiv", "inaktiv"),
            ("offen", "geschlossen"),
            ("wahr", "falsch"),
            ("an", "aus"),
        }
        for state1, state2 in binary_states:
            if (prop1_lower == state1 and prop2_lower == state2) or (
                prop1_lower == state2 and prop2_lower == state1
            ):
                return True

        # Keine Widersprüche gefunden
        return False

    def _is_location_hierarchy(self, loc1: str, loc2: str) -> bool:
        """
        Prüft ob zwei Locations in einer Hierarchie stehen (z.B. Berlin in Deutschland).

        Args:
            loc1: Erste Location
            loc2: Zweite Location

        Returns:
            True wenn loc1 Teil von loc2 ist (oder umgekehrt)
        """
        # Prüfe ob loc1 PART_OF loc2 ist
        facts1 = self.netzwerk.query_graph_for_facts(loc1)
        if "PART_OF" in facts1 and loc2 in facts1["PART_OF"]:
            return True

        # Prüfe ob loc2 PART_OF loc1 ist
        facts2 = self.netzwerk.query_graph_for_facts(loc2)
        if "PART_OF" in facts2 and loc1 in facts2["PART_OF"]:
            return True

        # Prüfe transitive PART_OF Beziehungen
        if "PART_OF" in facts1:
            for parent in facts1["PART_OF"]:
                if self._is_location_hierarchy(parent, loc2):
                    return True

        if "PART_OF" in facts2:
            for parent in facts2["PART_OF"]:
                if self._is_location_hierarchy(parent, loc1):
                    return True

        return False

    def explain_hypothesis(self, hypothesis: Hypothesis) -> str:
        """
        Generate detailed natural language explanation of hypothesis.

        Args:
            hypothesis: The hypothesis to explain

        Returns:
            Natural language explanation with reasoning trace
        """
        parts = []

        parts.append(f"**Hypothese (ID: {hypothesis.id})**")
        parts.append(f"Erklärung: {hypothesis.explanation}")
        parts.append(f"Strategie: {hypothesis.strategy}")
        parts.append(f"Konfidenz: {hypothesis.confidence:.2f}")
        parts.append("")

        parts.append("**Bewertung:**")
        for criterion, score in hypothesis.scores.items():
            parts.append(f"  - {criterion}: {score:.2f}")
        parts.append("")

        if hypothesis.abduced_facts:
            parts.append("**Abgeleitete Fakten:**")
            for fact in hypothesis.abduced_facts:
                parts.append(
                    f"  - {fact.pred}({fact.args}) "
                    f"[Konfidenz: {fact.confidence:.2f}]"
                )
            parts.append("")

        if hypothesis.reasoning_trace:
            parts.append("**Reasoning Trace:**")
            parts.append(f"  {hypothesis.reasoning_trace}")
            parts.append("")

        return "\n".join(parts)

    # ==================== UNIFIED PROOF EXPLANATION INTEGRATION ====================

    def create_proof_step_from_hypothesis(
        self, hypothesis: Hypothesis, query: str = ""
    ) -> Optional[UnifiedProofStep]:
        """
        Konvertiert eine Hypothese in einen UnifiedProofStep.

        Args:
            hypothesis: Hypothesis-Objekt
            query: Die ursprüngliche Anfrage (optional)

        Returns:
            UnifiedProofStep oder None
        """
        if not UNIFIED_PROOFS_AVAILABLE or not hypothesis:
            return None

        # Erstelle Inputs aus Observationen
        inputs = hypothesis.observations

        # Output ist die Hypothesen-Erklärung
        output = hypothesis.explanation

        # Generiere erweiterte Erklärung
        explanation = generate_explanation_text(
            step_type=StepType.HYPOTHESIS,
            inputs=inputs,
            output=output,
            metadata={
                "strategy": hypothesis.strategy,
                "score": hypothesis.confidence,
                "scores": hypothesis.scores,
                "num_abduced_facts": len(hypothesis.abduced_facts),
            },
        )

        # Erstelle UnifiedProofStep
        step = UnifiedProofStep(
            step_id=hypothesis.id,
            step_type=StepType.HYPOTHESIS,
            inputs=inputs,
            rule_name=None,
            output=output,
            confidence=hypothesis.confidence,
            explanation_text=explanation,
            parent_steps=[],  # Hypothesen haben keine direkten Parents
            bindings={},
            metadata={
                "strategy": hypothesis.strategy,
                "scores": hypothesis.scores,
                "abduced_facts": [f.pred for f in hypothesis.abduced_facts],
                "sources": hypothesis.sources,
                "reasoning_trace": hypothesis.reasoning_trace,
                "observations": hypothesis.observations,
            },
            source_component="component_14_abductive_engine",
            timestamp=hypothesis.timestamp,
        )

        # Füge abgeleitete Fakten als Subgoals hinzu (optional)
        for fact in hypothesis.abduced_facts:
            fact_step = UnifiedProofStep(
                step_id=fact.id,
                step_type=StepType.HYPOTHESIS,  # Abgeleitete Fakten sind auch Hypothesen
                inputs=[],
                rule_name=None,
                output=f"{fact.pred}({fact.args})",
                confidence=fact.confidence,
                explanation_text=f"Abgeleiteter Fakt aus Hypothese: {fact.pred}",
                parent_steps=[hypothesis.id],
                bindings=fact.args,
                metadata={"abduced": True, "parent_hypothesis": hypothesis.id},
                source_component="component_14_abductive_engine",
            )
            step.add_subgoal(fact_step)

        return step

    def create_multi_hypothesis_proof_chain(
        self, hypotheses: List[Hypothesis], query: str = ""
    ) -> List[UnifiedProofStep]:
        """
        Erstellt eine Proof-Kette aus mehreren Hypothesen.

        Nützlich wenn mehrere alternative Hypothesen existieren.

        Args:
            hypotheses: Liste von Hypothesis-Objekten
            query: Die ursprüngliche Anfrage

        Returns:
            Liste von UnifiedProofStep-Objekten (sortiert nach Konfidenz)
        """
        if not UNIFIED_PROOFS_AVAILABLE:
            return []

        proof_steps = []
        for i, hypothesis in enumerate(hypotheses):
            step = self.create_proof_step_from_hypothesis(hypothesis, query)
            if step:
                # Markiere Rang in alternativen Hypothesen
                step.metadata["hypothesis_rank"] = i + 1
                step.metadata["total_hypotheses"] = len(hypotheses)
                proof_steps.append(step)

        return proof_steps

    def explain_with_proof_step(
        self,
        observation: str,
        context_facts: List[Fact] = None,
        max_hypotheses: int = 3,
    ) -> List[UnifiedProofStep]:
        """
        Generiert Hypothesen und gibt UnifiedProofSteps zurück.

        Dies ist die Hauptschnittstelle für Integration mit dem Reasoning System.

        Args:
            observation: Die zu erklärende Beobachtung
            context_facts: Bekannte Fakten für Kontext
            max_hypotheses: Maximale Anzahl Hypothesen

        Returns:
            Liste von UnifiedProofStep-Objekten mit Erklärungen
        """
        # Generiere Hypothesen
        hypotheses = self.generate_hypotheses(
            observation=observation,
            context_facts=context_facts,
            max_hypotheses=max_hypotheses,
        )

        if not UNIFIED_PROOFS_AVAILABLE:
            return []

        # Konvertiere zu UnifiedProofSteps
        proof_steps = self.create_multi_hypothesis_proof_chain(
            hypotheses, query=observation
        )

        return proof_steps

    def create_detailed_explanation(self, hypothesis: Hypothesis) -> str:
        """
        Erstellt eine detaillierte Erklärung mit Unified Explanation System.

        Args:
            hypothesis: Die zu erklärende Hypothese

        Returns:
            Detaillierte natürlichsprachliche Erklärung
        """
        if not UNIFIED_PROOFS_AVAILABLE:
            # Fallback auf alte Methode
            return self.explain_hypothesis(hypothesis)

        # Erstelle UnifiedProofStep
        proof_step = self.create_proof_step_from_hypothesis(hypothesis)

        if not proof_step:
            return self.explain_hypothesis(hypothesis)

        # Nutze Unified Formatter
        from component_17_proof_explanation import format_proof_step

        formatted = format_proof_step(proof_step, indent=0, show_details=True)

        # Füge zusätzliche Informationen hinzu
        parts = [formatted, ""]

        parts.append("=== Detaillierte Bewertung ===")
        for criterion, score in hypothesis.scores.items():
            parts.append(f"  {criterion.capitalize()}: {score:.2f}")

        if hypothesis.sources:
            parts.append("")
            parts.append("=== Quellen ===")
            for source in hypothesis.sources:
                parts.append(f"  - {source}")

        return "\n".join(parts)
