"""
KAI Test Suite - Backward Chaining Tests
Extrahiert aus test_kai_worker.py fuer bessere Wartbarkeit.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

logger = logging.getLogger(__name__)
TEST_VECTOR_DIM = 384


class TestBackwardChaining:
    """
    Tests für Goal-Driven Reasoning und Multi-Hop-Schlussfolgerungen.
    Phase 3: Backward-Chaining & Autonomes Lernen
    """

    def test_prove_goal_by_direct_fact(self, netzwerk_session, clean_test_concepts):
        """
        Testet Backward-Chaining mit direktem Faktenmatch.
        Base Case: Goal kann direkt durch einen Fakt bewiesen werden.
        """
        from component_9_logik_engine import Engine, Goal, Fact

        engine = Engine(netzwerk_session)

        # Setup: Füge einen Fakt hinzu
        fact = Fact(
            pred="IS_A",
            args={"subject": "test_bc_hund", "object": "test_bc_tier"},
            confidence=1.0,
            source="test",
        )
        engine.add_fact(fact)

        # Goal: Beweise dass test_bc_hund ein test_bc_tier ist
        goal = Goal(
            pred="IS_A", args={"subject": "test_bc_hund", "object": "test_bc_tier"}
        )

        # Aktion: Beweise Goal
        proof = engine.prove_goal(goal)

        # Verifikation
        assert proof is not None, "Goal sollte beweisbar sein"
        assert proof.method == "fact", "Sollte durch direkten Fakt bewiesen werden"
        assert proof.confidence == 1.0
        assert len(proof.supporting_facts) == 1
        assert proof.supporting_facts[0].pred == "IS_A"

        logger.info(f"[SUCCESS] Direct Fact Proof erfolgreich")
        logger.info(f"Proof Trace:\n{engine.format_proof_trace(proof)}")

    def test_prove_goal_by_rule_with_subgoals(
        self, netzwerk_session, clean_test_concepts
    ):
        """
        Testet Backward-Chaining mit Regelanwendung und Subgoal-Zerlegung.
        Recursive Case: Goal erfordert Anwendung einer Regel.
        """
        from component_9_logik_engine import Engine, Goal, Fact, Rule

        engine = Engine(netzwerk_session)

        # Setup: Erstelle Regel "IF IS_A(?x, mammal) THEN IS_A(?x, animal)"
        rule = Rule(
            id="test_bc_rule_mammal",
            salience=100,
            when=[
                {"pred": "IS_A", "args": {"subject": "?x", "object": "test_bc_mammal"}}
            ],
            then=[
                {
                    "assert": {
                        "pred": "IS_A",
                        "args": {"subject": "?x", "object": "test_bc_animal"},
                    }
                }
            ],
            explain="Mammals are animals",
        )
        engine.rules.append(rule)

        # Setup: Füge Fakt hinzu, der das Subgoal erfüllt
        fact = Fact(
            pred="IS_A",
            args={"subject": "test_bc_whale", "object": "test_bc_mammal"},
            confidence=1.0,
            source="test",
        )
        engine.add_fact(fact)

        # Goal: Beweise dass test_bc_whale ein test_bc_animal ist
        goal = Goal(
            pred="IS_A", args={"subject": "test_bc_whale", "object": "test_bc_animal"}
        )

        # Aktion: Beweise Goal
        proof = engine.prove_goal(goal)

        # Verifikation
        assert proof is not None, "Goal sollte durch Regel beweisbar sein"
        assert proof.method == "rule", "Sollte durch Regel bewiesen werden"
        assert proof.rule_id == "test_bc_rule_mammal"
        assert len(proof.subgoals) == 1, "Sollte ein Subgoal haben"
        assert (
            proof.subgoals[0].method == "fact"
        ), "Subgoal sollte durch Fakt bewiesen werden"

        logger.info(f"[SUCCESS] Rule-Based Proof mit Subgoals erfolgreich")
        logger.info(f"Proof Trace:\n{engine.format_proof_trace(proof)}")

    def test_prove_goal_with_multi_hop_graph_traversal(
        self, netzwerk_session, clean_test_concepts
    ):
        """
        Testet Multi-Hop-Reasoning durch Graph-Traversal.
        Findet transitive Beziehungen im Neo4j-Graphen.

        Beispiel:
        test_bc_collie -IS_A-> test_bc_dog -IS_A-> test_bc_mammal -IS_A-> test_bc_creature
        Goal: IS_A(test_bc_collie, test_bc_creature) -> Sollte 3-Hop-Pfad finden
        """
        from component_9_logik_engine import Engine, Goal

        engine = Engine(netzwerk_session)

        # Setup: Erstelle Beziehungskette im Graph
        netzwerk_session.ensure_wort_und_konzept("test_bc_collie")
        netzwerk_session.ensure_wort_und_konzept("test_bc_dog")
        netzwerk_session.ensure_wort_und_konzept("test_bc_mammal")
        netzwerk_session.ensure_wort_und_konzept("test_bc_creature")

        netzwerk_session.assert_relation(
            "test_bc_collie", "IS_A", "test_bc_dog", "test"
        )
        netzwerk_session.assert_relation(
            "test_bc_dog", "IS_A", "test_bc_mammal", "test"
        )
        netzwerk_session.assert_relation(
            "test_bc_mammal", "IS_A", "test_bc_creature", "test"
        )

        # Goal: Beweise dass test_bc_collie ein test_bc_creature ist (3 Hops)
        goal = Goal(
            pred="IS_A",
            args={"subject": "test_bc_collie", "object": "test_bc_creature"},
        )

        # Aktion: Beweise Goal durch Graph-Traversal
        proof = engine.prove_goal(goal, max_depth=5)

        # Verifikation
        assert proof is not None, "Goal sollte durch Graph-Traversal beweisbar sein"
        assert (
            proof.method == "graph_traversal"
        ), "Sollte durch Graph-Traversal bewiesen werden"
        assert proof.confidence < 1.0, "Multi-hop sollte niedrigere Confidence haben"
        assert len(proof.supporting_facts) > 0

        logger.info(f"[SUCCESS] Multi-Hop Graph Traversal erfolgreich")
        logger.info(f"Confidence: {proof.confidence:.2f}")
        logger.info(f"Proof Trace:\n{engine.format_proof_trace(proof)}")

    def test_prove_goal_with_nested_subgoals(
        self, netzwerk_session, clean_test_concepts
    ):
        """
        Testet verschachtelte Subgoal-Zerlegung (Deep Reasoning).

        Regel 1: IF HAS_PROPERTY(?x, warm-blooded) AND NURSES_YOUNG(?x) THEN IS_A(?x, mammal)
        Regel 2: IF IS_A(?x, mammal) THEN IS_A(?x, animal)

        Goal: IS_A(test_bc_elephant, animal)
        => Sollte zwei Ebenen von Subgoals durchlaufen
        """
        from component_9_logik_engine import Engine, Goal, Fact, Rule

        engine = Engine(netzwerk_session)

        # Regel 1: warm-blooded + nurses_young => mammal
        rule1 = Rule(
            id="test_bc_define_mammal",
            salience=100,
            when=[
                {
                    "pred": "HAS_PROPERTY",
                    "args": {"subject": "?x", "property": "test_bc_warm_blooded"},
                },
                {
                    "pred": "CAPABLE_OF",
                    "args": {"subject": "?x", "action": "test_bc_nurse_young"},
                },
            ],
            then=[
                {
                    "assert": {
                        "pred": "IS_A",
                        "args": {"subject": "?x", "object": "test_bc_mammal"},
                    }
                }
            ],
        )

        # Regel 2: mammal => animal
        rule2 = Rule(
            id="test_bc_mammal_to_animal",
            salience=90,
            when=[
                {"pred": "IS_A", "args": {"subject": "?x", "object": "test_bc_mammal"}}
            ],
            then=[
                {
                    "assert": {
                        "pred": "IS_A",
                        "args": {"subject": "?x", "object": "test_bc_animal"},
                    }
                }
            ],
        )

        engine.rules.extend([rule1, rule2])

        # Fakten: Elephant hat die Eigenschaften
        fact1 = Fact(
            pred="HAS_PROPERTY",
            args={"subject": "test_bc_elephant", "property": "test_bc_warm_blooded"},
            confidence=1.0,
        )
        fact2 = Fact(
            pred="CAPABLE_OF",
            args={"subject": "test_bc_elephant", "action": "test_bc_nurse_young"},
            confidence=1.0,
        )
        engine.add_fact(fact1)
        engine.add_fact(fact2)

        # Goal: Beweise dass Elephant ein Animal ist (2-Level Reasoning)
        goal = Goal(
            pred="IS_A",
            args={"subject": "test_bc_elephant", "object": "test_bc_animal"},
        )

        # Aktion: Beweise Goal
        proof = engine.prove_goal(goal, max_depth=5)

        # Verifikation
        assert (
            proof is not None
        ), "Goal sollte durch verschachtelte Regeln beweisbar sein"
        assert proof.method == "rule"
        assert proof.rule_id == "test_bc_mammal_to_animal"
        assert len(proof.subgoals) == 1, "Sollte ein Subgoal (IS_A mammal) haben"

        # Prüfe verschachtelte Ebene
        subgoal_proof = proof.subgoals[0]
        assert subgoal_proof.method == "rule"
        assert subgoal_proof.rule_id == "test_bc_define_mammal"
        assert len(subgoal_proof.subgoals) == 2, "Innere Regel sollte 2 Subgoals haben"

        logger.info(f"[SUCCESS] Nested Subgoal Reasoning erfolgreich")
        logger.info(f"Proof Trace:\n{engine.format_proof_trace(proof)}")

    def test_hybrid_reasoning_forward_and_backward(
        self, netzwerk_session, clean_test_concepts
    ):
        """
        Testet Kombination von Forward- und Backward-Chaining.

        Forward-Chaining leitet neue Fakten ab.
        Backward-Chaining nutzt diese, um ein Goal zu beweisen.
        """
        from component_9_logik_engine import Engine, Goal, Fact, Rule

        engine = Engine(netzwerk_session)

        # Regel: IF IS_A(?x, dog) THEN IS_A(?x, mammal)
        rule = Rule(
            id="test_bc_hybrid_rule",
            salience=100,
            when=[
                {"pred": "IS_A", "args": {"subject": "?x", "object": "test_bc_h_dog"}}
            ],
            then=[
                {
                    "assert": {
                        "pred": "IS_A",
                        "args": {"subject": "?x", "object": "test_bc_h_mammal"},
                    }
                }
            ],
        )
        engine.rules.append(rule)

        # Basis-Fakt
        fact = Fact(
            pred="IS_A",
            args={"subject": "test_bc_h_rex", "object": "test_bc_h_dog"},
            confidence=1.0,
        )
        engine.add_fact(fact)

        # Goal: Beweise dass Rex ein Mammal ist
        goal = Goal(
            pred="IS_A", args={"subject": "test_bc_h_rex", "object": "test_bc_h_mammal"}
        )

        # Aktion: Hybrid Reasoning
        proof = engine.run_with_goal(goal, max_depth=5)

        # Verifikation
        assert proof is not None, "Goal sollte durch Hybrid Reasoning beweisbar sein"

        # Prüfe dass Forward-Chaining den Fakt abgeleitet hat
        derived_facts = [f for f in engine.wm if f.source.startswith("rule:")]
        assert len(derived_facts) > 0, "Forward-Chaining sollte neue Fakten ableiten"

        logger.info(f"[SUCCESS] Hybrid Reasoning (Forward + Backward) erfolgreich")
        logger.info(f"Abgeleitete Fakten: {len(derived_facts)}")
        logger.info(f"Proof Trace:\n{engine.format_proof_trace(proof)}")

    def test_cycle_detection_prevents_infinite_loops(
        self, netzwerk_session, clean_test_concepts
    ):
        """
        Testet dass Zykluserkennung infinite Loops verhindert.

        Zyklische Regel: IF IS_A(?x, A) THEN IS_A(?x, B)
                         IF IS_A(?x, B) THEN IS_A(?x, A)
        """
        from component_9_logik_engine import Engine, Goal, Rule

        engine = Engine(netzwerk_session)

        # Zyklische Regeln
        rule1 = Rule(
            id="test_bc_cycle_1",
            salience=100,
            when=[{"pred": "IS_A", "args": {"subject": "?x", "object": "test_bc_c_a"}}],
            then=[
                {
                    "assert": {
                        "pred": "IS_A",
                        "args": {"subject": "?x", "object": "test_bc_c_b"},
                    }
                }
            ],
        )
        rule2 = Rule(
            id="test_bc_cycle_2",
            salience=100,
            when=[{"pred": "IS_A", "args": {"subject": "?x", "object": "test_bc_c_b"}}],
            then=[
                {
                    "assert": {
                        "pred": "IS_A",
                        "args": {"subject": "?x", "object": "test_bc_c_a"},
                    }
                }
            ],
        )
        engine.rules.extend([rule1, rule2])

        # Goal: Sollte nicht in Infinite Loop geraten
        goal = Goal(
            pred="IS_A", args={"subject": "test_bc_c_entity", "object": "test_bc_c_a"}
        )

        # Aktion: Versuche zu beweisen (sollte terminieren)
        proof = engine.prove_goal(goal, max_depth=3)

        # Verifikation: Sollte None zurückgeben (nicht beweisbar)
        assert proof is None, "Goal sollte nicht beweisbar sein (keine Basis-Fakten)"

        logger.info(f"[SUCCESS] Zykluserkennung funktioniert - kein Infinite Loop")

    def test_confidence_propagation_in_proof_chain(
        self, netzwerk_session, clean_test_concepts
    ):
        """
        Testet dass Confidence korrekt durch Proof-Chain propagiert wird.

        Multi-Hop: Confidence sollte mit Distanz abnehmen.
        Rule-Chain: Confidence = min(subgoal_confidences)
        """
        from component_9_logik_engine import Engine, Goal, Fact, Rule

        engine = Engine(netzwerk_session)

        # Regel mit niedrigerer Confidence
        rule = Rule(
            id="test_bc_conf_rule",
            salience=100,
            when=[
                {
                    "pred": "IS_A",
                    "args": {"subject": "?x", "object": "test_bc_conf_bird"},
                }
            ],
            then=[
                {
                    "assert": {
                        "pred": "IS_A",
                        "args": {"subject": "?x", "object": "test_bc_conf_animal"},
                    }
                }
            ],
        )
        engine.rules.append(rule)

        # Fakt mit Confidence 0.8
        fact = Fact(
            pred="IS_A",
            args={"subject": "test_bc_conf_penguin", "object": "test_bc_conf_bird"},
            confidence=0.8,
        )
        engine.add_fact(fact)

        # Goal
        goal = Goal(
            pred="IS_A",
            args={"subject": "test_bc_conf_penguin", "object": "test_bc_conf_animal"},
        )

        # Aktion
        proof = engine.prove_goal(goal)

        # Verifikation: Confidence sollte <= 0.8 sein (min der Kette)
        assert proof is not None
        assert (
            proof.confidence <= 0.8
        ), f"Confidence sollte <= 0.8 sein, ist aber {proof.confidence}"

        logger.info(f"[SUCCESS] Confidence Propagation korrekt: {proof.confidence:.2f}")


# ============================================================================
# TESTS FÜR W-FRAGEN (Erweiterte Frageverarbeitung)
# ============================================================================
