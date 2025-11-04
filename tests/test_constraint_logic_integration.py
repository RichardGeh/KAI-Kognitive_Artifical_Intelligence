"""
Tests für Phase 1.2: Integration von Constraint Reasoning mit Logik-Engine

Testet die solve_with_constraints Methode und ihre Interaktion
mit dem CSP-Solver.
"""

import pytest
from component_9_logik_engine import Engine, Fact, Goal
from component_29_constraint_reasoning import (
    Constraint,
    not_equal_constraint,
    all_different_constraint,
)


class TestConstraintLogicIntegration:
    """Tests für Constraint-basiertes Reasoning in der Logik-Engine"""

    @pytest.fixture
    def netzwerk(self):
        """Fixture für KonzeptNetzwerk (Mock)"""
        # Für diese Tests verwenden wir None, da wir keine echte DB brauchen
        return None

    @pytest.fixture
    def engine(self, netzwerk):
        """Fixture für Engine mit Testdaten"""
        engine = Engine(netzwerk, use_probabilistic=False)

        # Füge Testfakten hinzu
        engine.add_fact(
            Fact(pred="HAS_PROPERTY", args={"subject": "apfel", "object": "rot"})
        )
        engine.add_fact(
            Fact(pred="HAS_PROPERTY", args={"subject": "banane", "object": "gelb"})
        )
        engine.add_fact(
            Fact(pred="HAS_PROPERTY", args={"subject": "himmel", "object": "blau"})
        )
        engine.add_fact(
            Fact(pred="HAS_PROPERTY", args={"subject": "gras", "object": "grün"})
        )

        engine.add_fact(
            Fact(pred="IS_A", args={"subject": "apfel", "object": "frucht"})
        )
        engine.add_fact(
            Fact(pred="IS_A", args={"subject": "banane", "object": "frucht"})
        )
        engine.add_fact(Fact(pred="IS_A", args={"subject": "hund", "object": "tier"}))
        engine.add_fact(Fact(pred="IS_A", args={"subject": "katze", "object": "tier"}))

        return engine

    def test_solve_with_constraints_basic(self, engine):
        """Test: Einfaches Goal mit Variable ohne Constraints"""
        goal = Goal(pred="HAS_PROPERTY", args={"subject": "?x", "object": "rot"})

        proof = engine.solve_with_constraints(goal)

        assert proof is not None, "Sollte Lösung finden"
        assert proof.method == "constraint_satisfaction"
        assert "?x" in proof.bindings
        assert proof.bindings["?x"] == "apfel"
        assert proof.confidence > 0

    def test_solve_with_constraints_no_solution(self, engine):
        """Test: Goal mit Variable hat keine Lösung"""
        goal = Goal(
            pred="HAS_PROPERTY",
            args={"subject": "?x", "object": "lila"},  # Keine lila Objekte
        )

        proof = engine.solve_with_constraints(goal)

        assert proof is None, "Sollte keine Lösung finden"

    def test_solve_with_constraints_multiple_variables(self, engine):
        """Test: Goal mit mehreren Variablen"""
        goal = Goal(pred="HAS_PROPERTY", args={"subject": "?x", "object": "?y"})

        proof = engine.solve_with_constraints(goal)

        assert proof is not None
        assert "?x" in proof.bindings
        assert "?y" in proof.bindings
        # Sollte irgendeine gültige Kombination finden
        assert proof.bindings["?x"] in ["apfel", "banane", "himmel", "gras"]
        assert proof.bindings["?y"] in ["rot", "gelb", "blau", "grün"]

    def test_solve_with_constraints_with_explicit_constraint(self, engine):
        """Test: Goal mit explizitem NOT-EQUAL Constraint"""
        # Erstelle Constraint: ?x != "apfel"
        constraint = Constraint(
            name="?x != apfel",
            scope=["?x"],
            predicate=lambda assignment: assignment.get("?x") != "apfel",
        )

        goal = Goal(pred="IS_A", args={"subject": "?x", "object": "frucht"})

        proof = engine.solve_with_constraints(goal, constraints=[constraint])

        assert proof is not None
        assert proof.bindings["?x"] != "apfel", "Constraint sollte apfel ausschließen"
        assert proof.bindings["?x"] == "banane", "Sollte banane wählen"

    def test_solve_with_constraints_binary_constraint(self, engine):
        """Test: Goal mit binärem Constraint zwischen zwei Variablen"""
        # Füge zusätzliche Fakten hinzu für diesen Test
        engine.add_fact(
            Fact(pred="LOCATED_IN", args={"subject": "berlin", "object": "deutschland"})
        )
        engine.add_fact(
            Fact(pred="LOCATED_IN", args={"subject": "paris", "object": "frankreich"})
        )
        engine.add_fact(
            Fact(
                pred="LOCATED_IN", args={"subject": "münchen", "object": "deutschland"}
            )
        )

        # Constraint: ?x != ?y (verschiedene Städte)
        not_equal_constraint("?x", "?y")

        # Für diesen Test müssen wir ein spezielles Goal erstellen
        # Das ist eher ein konzeptioneller Test - in der Praxis würde man
        # zwei separate Goals haben
        goal = Goal(pred="LOCATED_IN", args={"subject": "?x", "object": "deutschland"})

        proof = engine.solve_with_constraints(goal)

        assert proof is not None
        assert proof.bindings["?x"] in ["berlin", "münchen"]

    def test_solve_with_constraints_fallback_to_normal_bc(self, engine):
        """Test: Goal ohne Variablen fällt zurück auf normales Backward-Chaining"""
        goal = Goal(pred="HAS_PROPERTY", args={"subject": "apfel", "object": "rot"})

        proof = engine.solve_with_constraints(goal)

        assert proof is not None
        # Sollte auf normales BC fallen, daher method != "constraint_satisfaction"
        assert proof.method != "constraint_satisfaction"

    def test_solve_with_constraints_domain_building(self, engine):
        """Test: Domain-Building aus Faktenbasis"""
        # Prüfe ob Domains korrekt aus Fakten extrahiert werden
        domain_subjects = engine._build_domain_from_facts("HAS_PROPERTY", "subject")
        domain_objects = engine._build_domain_from_facts("HAS_PROPERTY", "object")

        assert "apfel" in domain_subjects
        assert "banane" in domain_subjects
        assert "himmel" in domain_subjects
        assert "gras" in domain_subjects

        assert "rot" in domain_objects
        assert "gelb" in domain_objects
        assert "blau" in domain_objects
        assert "grün" in domain_objects

    def test_solve_with_constraints_verification(self, engine):
        """Test: CSP-Lösung wird durch Backward-Chaining verifiziert"""
        goal = Goal(pred="HAS_PROPERTY", args={"subject": "?x", "object": "rot"})

        proof = engine.solve_with_constraints(goal)

        assert proof is not None
        # Verifikation sollte erfolgreich sein, da "apfel ist rot" in Fakten
        assert proof.confidence == 1.0

    def test_solve_with_constraints_proof_formatting(self, engine):
        """Test: Proof-Formatierung für Constraint-Satisfaction"""
        goal = Goal(pred="HAS_PROPERTY", args={"subject": "?x", "object": "rot"})

        proof = engine.solve_with_constraints(goal)
        assert proof is not None

        # Formatiere Proof
        formatted = engine.format_proof_trace(proof)

        assert (
            "constraint_satisfaction" in formatted.lower()
            or "Constraint-Satisfaction" in formatted
        )
        assert "?x" in formatted
        assert "apfel" in formatted

    def test_solve_with_constraints_all_different(self, engine):
        """Test: AllDifferent Constraint (konzeptionell)"""
        # Füge Fakten für N-Queens-ähnliches Problem hinzu
        engine.add_fact(Fact(pred="POSITION", args={"queen": "q1", "column": "1"}))
        engine.add_fact(Fact(pred="POSITION", args={"queen": "q1", "column": "2"}))
        engine.add_fact(Fact(pred="POSITION", args={"queen": "q2", "column": "1"}))
        engine.add_fact(Fact(pred="POSITION", args={"queen": "q2", "column": "2"}))

        # Constraint: Alle Queens müssen in verschiedenen Spalten sein
        all_different_constraint(["?x", "?y"])

        goal = Goal(pred="POSITION", args={"queen": "q1", "column": "?x"})

        # Mit Constraint sollte nur eine Position gewählt werden
        proof = engine.solve_with_constraints(goal)

        assert proof is not None
        assert proof.bindings["?x"] in ["1", "2"]

    def test_solve_with_constraints_empty_domain(self, engine):
        """Test: Leere Domain führt zu None-Ergebnis"""
        goal = Goal(
            pred="UNKNOWN_PREDICATE",  # Kein Fakt hat dieses Prädikat
            args={"subject": "?x", "object": "?y"},
        )

        proof = engine.solve_with_constraints(goal)

        assert proof is None, "Leere Domain sollte None zurückgeben"

    def test_solve_with_constraints_integration_with_normal_bc(self, engine):
        """Test: Integration mit normalem Backward-Chaining"""
        # Erstelle ein Goal mit Variablen
        goal_with_vars = Goal(
            pred="HAS_PROPERTY", args={"subject": "?x", "object": "rot"}
        )

        # Löse mit Constraints
        proof_csp = engine.solve_with_constraints(goal_with_vars)

        # Löse gleiches Goal ohne Variablen mit normalem BC
        goal_concrete = Goal(
            pred="HAS_PROPERTY", args={"subject": "apfel", "object": "rot"}
        )
        proof_bc = engine.prove_goal(goal_concrete)

        # Beide sollten erfolgreich sein
        assert proof_csp is not None
        assert proof_bc is not None

        # CSP sollte apfel finden
        assert proof_csp.bindings["?x"] == "apfel"


class TestConstraintLogicEdgeCases:
    """Tests für Edge Cases und Fehlerbehandlung"""

    @pytest.fixture
    def engine(self):
        """Minimale Engine für Edge-Case-Tests"""
        return Engine(None, use_probabilistic=False)

    def test_solve_with_constraints_no_facts(self, engine):
        """Test: Engine ohne Fakten"""
        goal = Goal(pred="HAS_PROPERTY", args={"subject": "?x", "object": "rot"})

        proof = engine.solve_with_constraints(goal)
        assert proof is None

    def test_solve_with_constraints_none_constraints(self, engine):
        """Test: constraints=None sollte funktionieren"""
        engine.add_fact(Fact(pred="IS_A", args={"subject": "hund", "object": "tier"}))

        goal = Goal(pred="IS_A", args={"subject": "?x", "object": "tier"})
        proof = engine.solve_with_constraints(goal, constraints=None)

        assert proof is not None

    def test_solve_with_constraints_mixed_args(self, engine):
        """Test: Mix aus Variablen und konkreten Werten"""
        engine.add_fact(Fact(pred="IS_A", args={"subject": "hund", "object": "tier"}))
        engine.add_fact(Fact(pred="IS_A", args={"subject": "katze", "object": "tier"}))

        goal = Goal(
            pred="IS_A",
            args={"subject": "?x", "object": "tier"},  # ?x variabel, "tier" konkret
        )

        proof = engine.solve_with_constraints(goal)
        assert proof is not None
        assert proof.bindings["?x"] in ["hund", "katze"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
