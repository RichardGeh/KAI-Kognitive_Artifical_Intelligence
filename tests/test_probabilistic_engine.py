# test_probabilistic_engine.py
"""
Comprehensive test suite for component_16_probabilistic_engine.py

Tests:
- Bayesian inference with forward-chaining
- Confidence propagation through reasoning chains
- Noisy-OR combination for redundant evidence
- Uncertainty-aware response generation
- Integration with deterministic logic engine
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import logging
from component_16_probabilistic_engine import (
    ProbabilisticEngine,
    ProbabilisticFact,
    ConditionalProbability,
    BeliefState,
    convert_fact_to_probabilistic,
    convert_rule_to_conditional,
)

# Import für Integration-Tests
from component_9_logik_engine import Engine, Fact, Rule, Goal
from component_1_netzwerk import KonzeptNetzwerk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestBeliefState:
    """Tests für die BeliefState-Klasse (Beta-Verteilung)."""

    def test_initial_belief_state(self):
        """Test: Initialer Glaubenszustand mit Laplace-Prior."""
        belief = BeliefState(proposition="test_prop")

        # Laplace-Prior: α=1, β=1 -> P = 0.5
        assert belief.alpha == 1.0
        assert belief.beta == 1.0
        assert belief.probability == 0.5

    def test_positive_update(self):
        """Test: Bayesian Update mit positiver Evidenz."""
        belief = BeliefState(proposition="regen")

        # 3 positive Beobachtungen
        belief.update(observation=True, weight=1.0)
        belief.update(observation=True, weight=1.0)
        belief.update(observation=True, weight=1.0)

        # α = 1 + 3 = 4, β = 1 -> P = 4/5 = 0.8
        assert belief.alpha == 4.0
        assert belief.beta == 1.0
        assert belief.probability == 0.8

    def test_negative_update(self):
        """Test: Bayesian Update mit negativer Evidenz."""
        belief = BeliefState(proposition="regen")

        # 2 negative Beobachtungen
        belief.update(observation=False, weight=1.0)
        belief.update(observation=False, weight=1.0)

        # α = 1, β = 1 + 2 = 3 -> P = 1/4 = 0.25
        assert belief.alpha == 1.0
        assert belief.beta == 3.0
        assert belief.probability == 0.25

    def test_mixed_updates(self):
        """Test: Gemischte positive und negative Updates."""
        belief = BeliefState(proposition="grippe")

        # 3 positive, 1 negative
        belief.update(observation=True, weight=1.0)
        belief.update(observation=True, weight=1.0)
        belief.update(observation=False, weight=1.0)
        belief.update(observation=True, weight=1.0)

        # α = 1 + 3 = 4, β = 1 + 1 = 2 -> P = 4/6 ≈ 0.667
        assert belief.alpha == 4.0
        assert belief.beta == 2.0
        assert abs(belief.probability - 0.667) < 0.01

    def test_weighted_update(self):
        """Test: Gewichtete Updates (verschiedene Evidenz-Stärken)."""
        belief = BeliefState(proposition="diagnose")

        # Starke positive Evidenz (weight=2.0)
        belief.update(observation=True, weight=2.0)
        # Schwache negative Evidenz (weight=0.5)
        belief.update(observation=False, weight=0.5)

        assert belief.alpha == 3.0  # 1 + 2
        assert belief.beta == 1.5  # 1 + 0.5
        assert abs(belief.probability - 0.667) < 0.01

    def test_confidence_increases_with_data(self):
        """Test: Konfidenz steigt mit mehr Beobachtungen."""
        belief = BeliefState(proposition="test")

        initial_confidence = belief.confidence

        # Viele Beobachtungen
        for _ in range(10):
            belief.update(observation=True, weight=1.0)

        final_confidence = belief.confidence

        # Konfidenz sollte mit mehr Daten steigen
        assert final_confidence > initial_confidence


class TestProbabilisticFact:
    """Tests für probabilistische Fakten."""

    def test_create_probabilistic_fact(self):
        """Test: Erstellen eines probabilistischen Fakts."""
        fact = ProbabilisticFact(
            pred="IS_A",
            args={"subject": "hund", "object": "tier"},
            probability=0.9,
            source="expert_knowledge",
        )

        assert fact.pred == "IS_A"
        assert fact.args["subject"] == "hund"
        assert fact.probability == 0.9

    def test_invalid_probability_raises_error(self):
        """Test: Ungültige Wahrscheinlichkeit wirft Fehler."""
        with pytest.raises(ValueError):
            ProbabilisticFact(pred="IS_A", args={}, probability=1.5)  # > 1.0

        with pytest.raises(ValueError):
            ProbabilisticFact(pred="IS_A", args={}, probability=-0.1)  # < 0.0


class TestProbabilisticEngine:
    """Tests für die Haupt-Engine."""

    @pytest.fixture
    def engine(self):
        """Fixture: Frische Engine-Instanz."""
        return ProbabilisticEngine()

    def test_engine_initialization(self, engine):
        """Test: Engine-Initialisierung."""
        assert len(engine.beliefs) == 0
        assert len(engine.conditional_probs) == 0
        assert len(engine.facts) == 0

    def test_add_fact_updates_belief(self, engine):
        """Test: Hinzufügen eines Fakts aktualisiert Glaubenszustand."""
        fact = ProbabilisticFact(
            pred="HAS_SYMPTOM",
            args={"patient": "max", "symptom": "fieber"},
            probability=0.8,
        )

        engine.add_fact(fact)

        # Belief sollte existieren
        prop_id = engine._fact_to_proposition(fact)
        assert prop_id in engine.beliefs

        # Wahrscheinlichkeit sollte nahe 0.8 sein (weighted Bayesian update)
        belief = engine.beliefs[prop_id]
        assert belief.probability > 0.5  # Sollte über Prior (0.5) liegen

    def test_add_conditional_probability(self, engine):
        """Test: Hinzufügen bedingter Wahrscheinlichkeiten."""
        cond_prob = ConditionalProbability(
            consequent="has_disease",
            antecedents=["has_symptom_fever", "has_symptom_cough"],
            probability=0.8,
            rule_id="rule_flu",
        )

        engine.add_conditional(cond_prob)

        assert len(engine.conditional_probs) == 1
        assert engine.conditional_probs[0].rule_id == "rule_flu"

    def test_simple_forward_chaining(self, engine):
        """Test: Einfaches Forward-Chaining mit einer Regel."""
        # Fakten: Symptome
        engine.add_fact(
            ProbabilisticFact(
                pred="symptom", args={}, probability=0.9, source="observation"
            )
        )

        # Regel: Symptom -> Krankheit (P=0.8)
        engine.add_conditional(
            ConditionalProbability(
                consequent="disease",
                antecedents=["symptom"],
                probability=0.8,
                rule_id="rule_1",
            )
        )

        # Inferenz
        derived = engine.infer(max_iterations=1)

        # Es sollte ein neuer Fakt abgeleitet worden sein
        assert len(derived) > 0
        assert derived[0].pred == "disease"

        # Wahrscheinlichkeit sollte kombiniert sein (min(0.9, ...) * 0.8)
        assert derived[0].probability > 0.0

    def test_multi_hop_reasoning(self, engine):
        """Test: Multi-Hop-Reasoning über mehrere Regeln."""
        # Fakten
        engine.add_fact(ProbabilisticFact(pred="symptom_A", args={}, probability=0.9))

        # Regel 1: symptom_A -> intermediate (P=0.8)
        engine.add_conditional(
            ConditionalProbability(
                consequent="intermediate",
                antecedents=["symptom_A"],
                probability=0.8,
                rule_id="rule_1",
            )
        )

        # Regel 2: intermediate -> conclusion (P=0.7)
        engine.add_conditional(
            ConditionalProbability(
                consequent="conclusion",
                antecedents=["intermediate"],
                probability=0.7,
                rule_id="rule_2",
            )
        )

        # Inferenz (2 Iterationen für 2-Hop)
        derived = engine.infer(max_iterations=3)

        # Es sollten 2 Fakten abgeleitet worden sein
        assert len(derived) >= 2

        # Prüfe dass "conclusion" abgeleitet wurde
        conclusion_facts = [f for f in derived if f.pred == "conclusion"]
        assert len(conclusion_facts) > 0

    def test_convergence_detection(self, engine):
        """Test: Engine erkennt Konvergenz."""
        # Einfaches Setup mit einer Regel
        engine.add_fact(ProbabilisticFact(pred="A", args={}, probability=1.0))
        engine.add_conditional(
            ConditionalProbability(
                consequent="B", antecedents=["A"], probability=1.0, rule_id="r1"
            )
        )

        # Inferenz mit hoher Iterationszahl
        derived = engine.infer(max_iterations=100)

        # Sollte nach 1 Iteration konvergieren (nur 1 Regel)
        assert len(derived) == 1

    def test_noisy_or_combination(self, engine):
        """Test: Noisy-OR für redundante Evidenz."""
        # 3 unabhängige Ursachen mit jeweils P=0.5
        probabilities = [0.5, 0.5, 0.5]

        combined = engine.noisy_or(probabilities)

        # P(E | C1, C2, C3) = 1 - (0.5^3) = 1 - 0.125 = 0.875
        expected = 0.875
        assert abs(combined - expected) < 0.01

    def test_noisy_or_with_different_strengths(self, engine):
        """Test: Noisy-OR mit unterschiedlich starker Evidenz."""
        # Starke (0.9), mittlere (0.5), schwache (0.2) Ursache
        probabilities = [0.9, 0.5, 0.2]

        combined = engine.noisy_or(probabilities)

        # P = 1 - (0.1 * 0.5 * 0.8) = 1 - 0.04 = 0.96
        expected = 0.96
        assert abs(combined - expected) < 0.01

    def test_confidence_propagation(self, engine):
        """Test: Confidence-Propagation durch Reasoning-Kette."""
        # Setup: Fakten mit verschiedenen Wahrscheinlichkeiten
        facts = [
            ProbabilisticFact(pred="F1", args={}, probability=0.9),
            ProbabilisticFact(pred="F2", args={}, probability=0.8),
        ]

        # Reasoning-Kette mit Regel-Stärken
        reasoning_chain = ["rule_weak", "rule_strong"]

        # Füge Regeln hinzu
        engine.add_conditional(
            ConditionalProbability(
                consequent="dummy",
                antecedents=[],
                probability=0.6,  # Schwache Regel
                rule_id="rule_weak",
            )
        )
        engine.add_conditional(
            ConditionalProbability(
                consequent="dummy",
                antecedents=[],
                probability=0.95,  # Starke Regel
                rule_id="rule_strong",
            )
        )

        # Propagiere Confidence
        final_conf = engine.propagate_confidence(facts, reasoning_chain)

        # Min(0.9, 0.8) * 0.6 * 0.95 = 0.8 * 0.6 * 0.95 = 0.456
        expected = 0.456
        assert abs(final_conf - expected) < 0.01

    def test_query_known_proposition(self, engine):
        """Test: Abfrage einer bekannten Proposition."""
        # Füge Fakt hinzu
        engine.add_fact(
            ProbabilisticFact(pred="test", args={"x": "y"}, probability=0.85)
        )

        # Frage ab
        prob, conf = engine.query("test(x=y)")

        # Wahrscheinlichkeit sollte nahe 0.85 sein
        assert prob > 0.7  # Aufgrund von Bayesian update
        # Konfidenz sollte > 0 sein
        assert conf > 0.0

    def test_query_unknown_proposition(self, engine):
        """Test: Abfrage einer unbekannten Proposition."""
        prob, conf = engine.query("unknown_prop")

        # Uninformativer Prior: P=0.5, Konfidenz=0
        assert prob == 0.5
        assert conf == 0.0

    def test_explain_belief(self, engine):
        """Test: Erklärung eines Glaubenszustands."""
        # Füge Fakt und Regel hinzu
        engine.add_fact(
            ProbabilisticFact(
                pred="A", args={}, probability=0.9, evidence=["observation"]
            )
        )

        engine.add_conditional(
            ConditionalProbability(
                consequent="A", antecedents=["B"], probability=0.8, rule_id="rule_A"
            )
        )

        # Erkläre Belief
        explanation = engine.explain_belief("A()")

        assert "probability" in explanation
        assert "confidence" in explanation
        assert explanation["direct_facts"] == 1
        assert "rule_A" in explanation["applicable_rules"]

    def test_generate_response_high_confidence(self, engine):
        """Test: Response-Generation bei hoher Wahrscheinlichkeit."""
        # Füge sicheres Wissen hinzu
        engine.add_fact(ProbabilisticFact(pred="test", args={}, probability=0.95))

        response = engine.generate_response(
            "test()", threshold_high=0.8, threshold_low=0.2
        )

        # Sollte Bejahung enthalten
        assert "ja" in response.lower() or "wahrscheinlich" in response.lower()

    def test_generate_response_low_confidence(self, engine):
        """Test: Response-Generation bei niedriger Wahrscheinlichkeit."""
        # Füge unsicheres Wissen hinzu
        engine.add_fact(ProbabilisticFact(pred="test", args={}, probability=0.1))

        # Update mehrmals negativ
        belief = engine.beliefs["test()"]
        for _ in range(5):
            belief.update(observation=False, weight=1.0)

        response = engine.generate_response(
            "test()", threshold_high=0.8, threshold_low=0.2
        )

        # Sollte Verneinung enthalten
        assert "nein" in response.lower() or "unwahrscheinlich" in response.lower()

    def test_generate_response_uncertain(self, engine):
        """Test: Response-Generation bei Unsicherheit."""
        # Füge mittelmäßiges Wissen hinzu
        engine.add_fact(ProbabilisticFact(pred="test", args={}, probability=0.5))

        response = engine.generate_response(
            "test()", threshold_high=0.8, threshold_low=0.2
        )

        # Sollte Unsicherheit ausdrücken
        assert "unsicher" in response.lower() or "weitere evidenz" in response.lower()

    def test_most_certain_facts(self, engine):
        """Test: Abruf der sichersten Fakten."""
        # Füge Fakten mit verschiedenen Konfidenz-Werten hinzu
        engine.add_fact(ProbabilisticFact(pred="certain", args={}, probability=0.99))

        # Viele Updates für hohe Konfidenz
        for _ in range(20):
            engine.beliefs["certain()"].update(observation=True, weight=1.0)

        engine.add_fact(ProbabilisticFact(pred="uncertain", args={}, probability=0.5))

        most_certain = engine.get_most_certain_facts(top_k=1)

        assert len(most_certain) > 0
        assert most_certain[0][0] == "certain()"

    def test_most_uncertain_facts(self, engine):
        """Test: Abruf der unsichersten Fakten (aktives Lernen)."""
        # Füge Fakten hinzu
        engine.add_fact(ProbabilisticFact(pred="certain", args={}, probability=0.99))
        for _ in range(20):
            engine.beliefs["certain()"].update(observation=True, weight=1.0)

        engine.add_fact(ProbabilisticFact(pred="uncertain", args={}, probability=0.5))

        most_uncertain = engine.get_most_uncertain_facts(top_k=1)

        assert len(most_uncertain) > 0
        assert most_uncertain[0][0] == "uncertain()"

    def test_reset_engine(self, engine):
        """Test: Zurücksetzen der Engine."""
        # Füge Daten hinzu
        engine.add_fact(ProbabilisticFact(pred="test", args={}, probability=0.9))
        engine.add_conditional(
            ConditionalProbability(
                consequent="B", antecedents=["A"], probability=0.8, rule_id="r1"
            )
        )

        # Reset
        engine.reset()

        assert len(engine.beliefs) == 0
        assert len(engine.conditional_probs) == 0
        assert len(engine.facts) == 0


class TestIntegrationWithLogicEngine:
    """Integration-Tests mit der deterministischen Logic Engine."""

    @pytest.fixture
    def netzwerk(self):
        """Fixture: Neo4j-Netzwerk (read-only für Tests)."""
        return KonzeptNetzwerk()

    @pytest.fixture
    def prob_engine(self, netzwerk):
        """Fixture: Engine mit probabilistischer Unterstützung."""
        return Engine(netzwerk, use_probabilistic=True)

    def test_conversion_fact_to_probabilistic(self):
        """Test: Konvertierung deterministischer Fakt -> probabilistischer Fakt."""
        det_fact = Fact(
            pred="IS_A",
            args={"subject": "hund", "object": "tier"},
            confidence=0.9,
            source="graph",
        )

        prob_fact = convert_fact_to_probabilistic(det_fact)

        assert prob_fact.pred == "IS_A"
        assert prob_fact.args == {"subject": "hund", "object": "tier"}
        assert prob_fact.probability == 0.9

    def test_conversion_rule_to_conditional(self):
        """Test: Konvertierung deterministische Regel -> bedingte Wahrscheinlichkeit."""
        rule = Rule(
            id="test_rule",
            salience=100,
            when=[
                {"pred": "A", "args": {"x": "?x"}},
                {"pred": "B", "args": {"x": "?x"}},
            ],
            then=[{"assert": {"pred": "C", "args": {"x": "?x"}}}],
            weight=0.85,
        )

        cond_probs = convert_rule_to_conditional(rule)

        assert len(cond_probs) == 1
        assert cond_probs[0].rule_id == "test_rule"
        assert cond_probs[0].probability == 0.85
        assert len(cond_probs[0].antecedents) == 2

    def test_probabilistic_engine_integration(self, prob_engine):
        """Test: Probabilistische Engine ist integriert."""
        if not prob_engine.use_probabilistic:
            pytest.skip("Probabilistische Engine nicht verfügbar")

        assert prob_engine.prob_engine is not None
        assert isinstance(prob_engine.prob_engine, ProbabilisticEngine)

    def test_add_fact_syncs_with_prob_engine(self, prob_engine):
        """Test: Fakten werden mit ProbabilisticEngine synchronisiert."""
        if not prob_engine.use_probabilistic:
            pytest.skip("Probabilistische Engine nicht verfügbar")

        # Füge Fakt hinzu
        fact = Fact(pred="test_sync", args={"x": "y"}, confidence=0.8)
        prob_engine.add_fact(fact)

        # Prüfe ob in ProbabilisticEngine vorhanden
        assert len(prob_engine.prob_engine.facts) > 0

    def test_query_with_uncertainty(self, prob_engine, netzwerk):
        """Test: Query mit Unsicherheitsquantifizierung."""
        if not prob_engine.use_probabilistic:
            pytest.skip("Probabilistische Engine nicht verfügbar")

        # Füge Testwissen hinzu
        fact = Fact(
            pred="IS_A",
            args={"subject": "test_tier", "object": "test_kategorie"},
            confidence=0.7,
        )
        prob_engine.add_fact(fact)

        # Query
        goal = Goal(
            pred="IS_A", args={"subject": "test_tier", "object": "test_kategorie"}
        )

        proof, response = prob_engine.query_with_uncertainty(goal)

        # Sollte Antwort generiert haben
        assert isinstance(response, str)
        assert len(response) > 0


class TestEdgeCases:
    """Edge-Cases und Fehlerbehandlung."""

    def test_empty_antecedents(self):
        """Test: Regel ohne Antecedents."""
        engine = ProbabilisticEngine()

        # Regel ohne Vorbedingungen
        engine.add_conditional(
            ConditionalProbability(
                consequent="always_true",
                antecedents=[],  # Leer
                probability=1.0,
                rule_id="empty_rule",
            )
        )

        derived = engine.infer(max_iterations=1)

        # Sollte nichts ableiten (keine Antecedents)
        assert len(derived) == 0

    def test_circular_rules(self):
        """Test: Zirkuläre Regeln (A -> B, B -> A)."""
        engine = ProbabilisticEngine()

        # A -> B
        engine.add_fact(ProbabilisticFact(pred="A", args={}, probability=1.0))
        engine.add_conditional(
            ConditionalProbability(
                consequent="B", antecedents=["A"], probability=0.9, rule_id="r1"
            )
        )

        # B -> A (zirkulär)
        engine.add_conditional(
            ConditionalProbability(
                consequent="A", antecedents=["B"], probability=0.9, rule_id="r2"
            )
        )

        # Sollte nicht endlos laufen (Konvergenz-Erkennung)
        derived = engine.infer(max_iterations=10)

        # Sollte B ableiten, aber nicht unendlich iterieren
        assert len(derived) > 0
        assert len(derived) < 10  # Keine Explosion

    def test_very_low_probabilities(self):
        """Test: Sehr niedrige Wahrscheinlichkeiten."""
        engine = ProbabilisticEngine()

        engine.add_fact(
            ProbabilisticFact(pred="rare_event", args={}, probability=0.001)
        )

        prob, conf = engine.query("rare_event()")

        # Sollte niedrig sein
        assert prob < 0.1

    def test_noisy_or_with_empty_list(self):
        """Test: Noisy-OR mit leerer Liste."""
        engine = ProbabilisticEngine()

        result = engine.noisy_or([])

        assert result == 0.0

    def test_noisy_or_with_single_probability(self):
        """Test: Noisy-OR mit nur einer Ursache."""
        engine = ProbabilisticEngine()

        result = engine.noisy_or([0.7])

        # Sollte gleich der einzelnen Wahrscheinlichkeit sein
        assert result == 0.7


# ==================== RUN TESTS ====================

if __name__ == "__main__":
    # Run mit pytest
    pytest.main([__file__, "-v", "--tb=short"])
