"""
test_sat_consistency.py
================================
Tests für SAT-basierte Konsistenzprüfung und Widerspruchserkennung.

Testet:
- check_consistency() für Faktenmenge-Konsistenz
- find_contradictions() für automatische Widerspruchserkennung
- Integration mit SAT-Solver und Logic Engine

Author: KAI Development Team
Date: 2025-10-30
"""

import pytest
from component_9_logik_engine import Engine, Fact, Rule
from component_1_netzwerk import KonzeptNetzwerk


@pytest.fixture
def netzwerk():
    """Erstelle Test-Netzwerk."""
    return KonzeptNetzwerk()


@pytest.fixture
def engine(netzwerk):
    """Erstelle Logic Engine mit SAT-Solver."""
    engine = Engine(netzwerk, use_probabilistic=False, use_sat=True)
    return engine


class TestConsistencyChecking:
    """Tests für check_consistency() Methode."""

    def test_empty_facts_consistent(self, engine):
        """Leere Faktenmenge ist konsistent."""
        assert engine.check_consistency([]) is True

    def test_single_fact_consistent(self, engine):
        """Einzelnes Fakt ist konsistent."""
        fact = Fact(pred="IS_A", args={"subject": "hund", "object": "tier"})
        assert engine.check_consistency([fact]) is True

    def test_consistent_facts(self, engine):
        """Konsistente Fakten werden als konsistent erkannt."""
        facts = [
            Fact(pred="IS_A", args={"subject": "hund", "object": "tier"}),
            Fact(pred="IS_A", args={"subject": "katze", "object": "tier"}),
            Fact(pred="HAS_PROPERTY", args={"subject": "hund", "object": "bellend"}),
        ]
        assert engine.check_consistency(facts) is True

    def test_contradictory_facts_direct_negation(self, engine):
        """Direkt widersprüchliche Fakten (negierte Version) werden erkannt."""
        fact1 = Fact(pred="IS_A", args={"subject": "hund", "object": "tier"})
        fact2 = Fact(
            pred="IS_A",
            args={"subject": "hund", "object": "tier"},
            status="contradicted",  # Negiert
        )
        # Diese sollten als inkonsistent erkannt werden
        assert engine.check_consistency([fact1, fact2]) is False

    def test_contradictory_facts_low_confidence(self, engine):
        """Fakten mit niedriger Confidence werden als negiert behandelt."""
        fact1 = Fact(
            pred="IS_A", args={"subject": "hund", "object": "tier"}, confidence=0.9
        )
        fact2 = Fact(
            pred="IS_A",
            args={"subject": "hund", "object": "tier"},
            confidence=0.3,  # < 0.5 wird als negiert behandelt
        )
        assert engine.check_consistency([fact1, fact2]) is False

    def test_consistency_with_rules(self, engine):
        """Konsistenz unter Berücksichtigung von Regeln."""
        # Regel: Wenn X ein Hund ist, dann ist X ein Säugetier
        rule = Rule(
            id="rule_hund_saeugetier",
            salience=100,
            when=[{"pred": "IS_A", "args": {"subject": "?x", "object": "hund"}}],
            then=[
                {
                    "assert": {
                        "pred": "IS_A",
                        "args": {"subject": "?x", "object": "saeugetier"},
                    }
                }
            ],
        )
        engine.rules.append(rule)

        # Fakten: Bello ist ein Hund
        facts = [Fact(pred="IS_A", args={"subject": "bello", "object": "hund"})]

        # Dies sollte konsistent sein (Regel leitet ab, dass Bello ein Säugetier ist)
        assert engine.check_consistency(facts) is True


class TestContradictionFinding:
    """Tests für find_contradictions() Methode."""

    def test_no_contradictions_empty_kb(self, engine):
        """Leere Wissensbasis hat keine Widersprüche."""
        contradictions = engine.find_contradictions()
        assert len(contradictions) == 0

    def test_no_contradictions_consistent_facts(self, engine):
        """Konsistente Fakten haben keine Widersprüche."""
        engine.add_fact(Fact(pred="IS_A", args={"subject": "hund", "object": "tier"}))
        engine.add_fact(Fact(pred="IS_A", args={"subject": "katze", "object": "tier"}))

        contradictions = engine.find_contradictions()
        assert len(contradictions) == 0

    def test_find_direct_contradiction(self, engine):
        """Direkte Widersprüche werden gefunden."""
        fact1 = Fact(pred="IS_A", args={"subject": "hund", "object": "tier"})
        fact2 = Fact(
            pred="IS_A",
            args={"subject": "hund", "object": "tier"},
            status="contradicted",
        )

        engine.add_fact(fact1)
        engine.add_fact(fact2)

        contradictions = engine.find_contradictions()
        assert len(contradictions) > 0

        # Prüfe ob die gefundenen Widersprüche fact1 und fact2 enthalten
        found = False
        for f1, f2 in contradictions:
            if (f1.id == fact1.id and f2.id == fact2.id) or (
                f1.id == fact2.id and f2.id == fact1.id
            ):
                found = True
                break
        assert found

    def test_find_contradiction_low_confidence(self, engine):
        """Widersprüche durch niedrige Confidence werden gefunden."""
        fact1 = Fact(
            pred="IS_A", args={"subject": "hund", "object": "tier"}, confidence=0.9
        )
        fact2 = Fact(
            pred="IS_A",
            args={"subject": "hund", "object": "tier"},
            confidence=0.2,  # < 0.5 = negiert
        )

        engine.add_fact(fact1)
        engine.add_fact(fact2)

        contradictions = engine.find_contradictions()
        assert len(contradictions) > 0

    def test_no_duplicate_contradictions(self, engine):
        """Keine doppelten Widersprüche (Symmetrie: (A,B) == (B,A))."""
        fact1 = Fact(
            pred="IS_A", args={"subject": "hund", "object": "tier"}, confidence=0.9
        )
        fact2 = Fact(
            pred="IS_A", args={"subject": "hund", "object": "tier"}, confidence=0.1
        )

        engine.add_fact(fact1)
        engine.add_fact(fact2)

        contradictions = engine.find_contradictions()

        # Prüfe dass es nur ein Widerspruchspaar gibt (nicht zwei: (A,B) und (B,A))
        assert len(contradictions) == 1

    def test_multiple_contradictions(self, engine):
        """Mehrere Widersprüche werden alle gefunden."""
        # Widerspruch 1: hund ist/ist nicht tier
        fact1a = Fact(
            pred="IS_A", args={"subject": "hund", "object": "tier"}, confidence=0.9
        )
        fact1b = Fact(
            pred="IS_A", args={"subject": "hund", "object": "tier"}, confidence=0.1
        )

        # Widerspruch 2: katze ist/ist nicht tier
        fact2a = Fact(
            pred="IS_A", args={"subject": "katze", "object": "tier"}, confidence=0.9
        )
        fact2b = Fact(
            pred="IS_A", args={"subject": "katze", "object": "tier"}, confidence=0.1
        )

        engine.add_fact(fact1a)
        engine.add_fact(fact1b)
        engine.add_fact(fact2a)
        engine.add_fact(fact2b)

        contradictions = engine.find_contradictions()

        # Sollte 2 Widerspruchspaare finden
        assert len(contradictions) >= 2


class TestIntegrationWithSAT:
    """Integrationstests mit SAT-Solver."""

    def test_sat_solver_available(self, engine):
        """SAT-Solver ist verfügbar und initialisiert."""
        assert engine.use_sat is True
        assert engine.sat_solver is not None
        assert engine.kb_checker is not None

    def test_consistency_uses_sat_solver(self, engine):
        """check_consistency() nutzt den SAT-Solver."""
        # Wir können nicht direkt testen ob SAT aufgerufen wird,
        # aber wir können prüfen ob das Ergebnis korrekt ist

        # UNSAT Fall
        fact1 = Fact(pred="test", args={"x": "1"}, confidence=0.9)
        fact2 = Fact(pred="test", args={"x": "1"}, confidence=0.1)

        result = engine.check_consistency([fact1, fact2])
        assert result is False  # Sollte UNSAT sein

        # SAT Fall
        fact3 = Fact(pred="test", args={"x": "2"}, confidence=0.9)
        result = engine.check_consistency([fact3])
        assert result is True  # Sollte SAT sein

    def test_contradiction_finding_with_complex_kb(self, engine):
        """Widerspruchserkennung in komplexer Wissensbasis."""
        # Erstelle komplexere Wissensbasis
        facts = [
            Fact(pred="IS_A", args={"subject": "bello", "object": "hund"}),
            Fact(pred="IS_A", args={"subject": "hund", "object": "tier"}),
            Fact(pred="HAS_PROPERTY", args={"subject": "bello", "object": "bellend"}),
            Fact(pred="IS_A", args={"subject": "minka", "object": "katze"}),
            Fact(pred="IS_A", args={"subject": "katze", "object": "tier"}),
            # Widerspruch: minka ist und ist nicht eine Katze
            Fact(
                pred="IS_A",
                args={"subject": "minka", "object": "katze"},
                confidence=0.1,
            ),
        ]

        for fact in facts:
            engine.add_fact(fact)

        contradictions = engine.find_contradictions()

        # Sollte mindestens einen Widerspruch finden (minka)
        assert len(contradictions) > 0

        # Prüfe ob minka-Widerspruch gefunden wurde
        found_minka_contradiction = False
        for f1, f2 in contradictions:
            if "minka" in str(f1.args) and "minka" in str(f2.args):
                found_minka_contradiction = True
                break

        assert found_minka_contradiction


class TestEdgeCases:
    """Tests für Edge Cases."""

    def test_consistency_without_sat_solver(self, netzwerk):
        """check_consistency() ohne SAT-Solver gibt True zurück (Fallback)."""
        engine_no_sat = Engine(netzwerk, use_probabilistic=False, use_sat=False)

        facts = [
            Fact(pred="test", args={"x": "1"}, confidence=0.9),
            Fact(pred="test", args={"x": "1"}, confidence=0.1),
        ]

        # Ohne SAT-Solver kann keine Prüfung durchgeführt werden
        # Sollte True zurückgeben (assume consistent)
        result = engine_no_sat.check_consistency(facts)
        assert result is True

    def test_contradictions_without_sat_solver(self, netzwerk):
        """find_contradictions() ohne SAT-Solver gibt leere Liste zurück."""
        engine_no_sat = Engine(netzwerk, use_probabilistic=False, use_sat=False)

        engine_no_sat.add_fact(Fact(pred="test", args={"x": "1"}, confidence=0.9))
        engine_no_sat.add_fact(Fact(pred="test", args={"x": "1"}, confidence=0.1))

        contradictions = engine_no_sat.find_contradictions()
        assert len(contradictions) == 0

    def test_consistency_with_special_characters(self, engine):
        """Konsistenzprüfung mit Sonderzeichen in Fact-Namen."""
        facts = [
            Fact(pred="IS_A", args={"subject": "haus-tür", "object": "tür"}),
            Fact(pred="IS_A", args={"subject": "auto tür", "object": "tür"}),
        ]

        # Sollte trotz Sonderzeichen funktionieren (werden normalisiert)
        result = engine.check_consistency(facts)
        assert result is True

    def test_self_contradiction(self, engine):
        """Ein Fakt kann nicht mit sich selbst im Widerspruch stehen."""
        fact = Fact(pred="test", args={"x": "1"})
        engine.add_fact(fact)

        contradictions = engine.find_contradictions()

        # Keine Selbst-Widersprüche
        for f1, f2 in contradictions:
            assert f1.id != f2.id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
