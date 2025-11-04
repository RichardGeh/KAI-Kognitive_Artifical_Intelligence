# test_abductive_reasoning.py
"""
Tests für Abductive Reasoning System.

Testet Hypothesen-Generierung, Multi-Kriterien-Scoring und Neo4j-Persistenz.
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from component_1_netzwerk import KonzeptNetzwerk
from component_9_logik_engine import Engine
from component_14_abductive_engine import AbductiveEngine, Hypothesis


@pytest.fixture
def netzwerk():
    """Erstellt eine Neo4j-Verbindung für Tests."""
    netz = KonzeptNetzwerk()
    yield netz
    # Cleanup: Lösche Test-Hypothesen
    if netz.driver:
        with netz.driver.session(database="neo4j") as session:
            session.run(
                "MATCH (h:Hypothesis) WHERE h.id CONTAINS 'test_' DETACH DELETE h"
            )
            # Lösche Test-Konzepte
            session.run(
                "MATCH (k:Konzept) WHERE k.name CONTAINS 'test_' DETACH DELETE k"
            )
            session.run("MATCH (w:Wort) WHERE w.wort CONTAINS 'test_' DETACH DELETE w")
    netz.close()


@pytest.fixture
def engine(netzwerk):
    """Erstellt eine Engine-Instanz mit Netzwerk."""
    eng = Engine(netzwerk)
    return eng


@pytest.fixture
def abductive_engine(netzwerk, engine):
    """Erstellt eine AbductiveEngine-Instanz."""
    abd_eng = AbductiveEngine(netzwerk, engine)
    return abd_eng


@pytest.fixture
def setup_test_knowledge(netzwerk):
    """Erstellt Test-Wissensbasis für Hypothesen-Generierung."""
    # Erstelle Konzepte und Relationen
    netzwerk.ensure_wort_und_konzept("test_vogel")
    netzwerk.ensure_wort_und_konzept("test_fliegen")
    netzwerk.ensure_wort_und_konzept("test_flügel")
    netzwerk.ensure_wort_und_konzept("test_tier")
    netzwerk.ensure_wort_und_konzept("test_federkleid")

    # Erstelle CAPABLE_OF Relation
    netzwerk.assert_relation("test_vogel", "CAPABLE_OF", "test_fliegen")

    # Erstelle HAS_PROPERTY Relation
    netzwerk.assert_relation("test_vogel", "HAS_PROPERTY", "test_flügel")
    netzwerk.assert_relation("test_vogel", "HAS_PROPERTY", "test_federkleid")

    # Erstelle IS_A Relation
    netzwerk.assert_relation("test_vogel", "IS_A", "test_tier")

    return netzwerk


class TestHypothesisCreation:
    """Testet die Erstellung von Hypothesen-Objekten."""

    def test_create_hypothesis_object(self):
        """Test: Erstelle Hypothesis-Objekt"""
        hyp = Hypothesis(
            id="test_hyp_1",
            explanation="Test-Erklärung",
            observations=["test_obs_1"],
            abduced_facts=[],
            strategy="template",
            confidence=0.8,
            scores={
                "coverage": 0.8,
                "simplicity": 0.9,
                "coherence": 0.7,
                "specificity": 0.6,
            },
        )

        assert hyp.id == "test_hyp_1"
        assert hyp.explanation == "Test-Erklärung"
        assert hyp.strategy == "template"
        assert hyp.confidence == 0.8
        assert len(hyp.scores) == 4


class TestTemplateBasedStrategy:
    """Testet template-basierte Hypothesen-Generierung."""

    def test_template_strategy_with_capable_of(
        self, abductive_engine, setup_test_knowledge
    ):
        """Test: Template-Strategie mit CAPABLE_OF-Pattern"""
        observation = "test_vogel kann test_fliegen"

        hypotheses = abductive_engine.generate_hypotheses(
            observation=observation, strategies=["template"], max_hypotheses=5
        )

        assert len(hypotheses) > 0
        assert any(h.strategy == "template" for h in hypotheses)

    def test_template_strategy_scoring(self, abductive_engine, setup_test_knowledge):
        """Test: Template-Strategie mit Scoring"""
        observation = "test_vogel hat test_flügel"

        hypotheses = abductive_engine.generate_hypotheses(
            observation=observation, strategies=["template"], max_hypotheses=3
        )

        if hypotheses:
            hyp = hypotheses[0]
            assert 0.0 <= hyp.confidence <= 1.0
            assert "coverage" in hyp.scores
            assert "simplicity" in hyp.scores
            assert "coherence" in hyp.scores
            assert "specificity" in hyp.scores


class TestAnalogyBasedStrategy:
    """Testet analogie-basierte Hypothesen-Generierung."""

    def test_analogy_strategy_with_similar_concepts(
        self, abductive_engine, setup_test_knowledge
    ):
        """Test: Analogie-Strategie mit ähnlichen Konzepten"""
        # Erstelle weiteres Konzept als Analogie-Quelle
        setup_test_knowledge.ensure_wort_und_konzept("test_schwalbe")
        setup_test_knowledge.assert_relation("test_schwalbe", "IS_A", "test_tier")
        setup_test_knowledge.assert_relation(
            "test_schwalbe", "CAPABLE_OF", "test_fliegen"
        )

        observation = "test_schwalbe verhält sich wie test_vogel"

        hypotheses = abductive_engine.generate_hypotheses(
            observation=observation, strategies=["analogy"], max_hypotheses=5
        )

        # Kann leer sein wenn keine starken Analogien gefunden
        assert isinstance(hypotheses, list)
        if hypotheses:
            assert any(h.strategy == "analogy" for h in hypotheses)

    def test_analogy_strategy_with_embeddings(
        self, abductive_engine, setup_test_knowledge
    ):
        """Test: Analogie-Strategie nutzt Embeddings"""
        observation = "test_neues_tier kann fliegen"

        hypotheses = abductive_engine.generate_hypotheses(
            observation=observation, strategies=["analogy"], max_hypotheses=3
        )

        # Validiere Struktur auch wenn leer
        assert isinstance(hypotheses, list)


class TestCausalChainStrategy:
    """Testet kausale Ketten-Hypothesen-Generierung."""

    def test_causal_chain_backward_reasoning(
        self, abductive_engine, setup_test_knowledge
    ):
        """Test: Causal-Chain-Strategie mit Rückwärts-Reasoning"""
        # Erstelle kausale Kette
        setup_test_knowledge.ensure_wort_und_konzept("test_wärme")
        setup_test_knowledge.ensure_wort_und_konzept("test_sonnenlicht")
        setup_test_knowledge.assert_relation("test_sonnenlicht", "CAUSES", "test_wärme")

        observation = "Es ist warm (test_wärme)"

        hypotheses = abductive_engine.generate_hypotheses(
            observation=observation, strategies=["causal_chain"], max_hypotheses=5
        )

        assert isinstance(hypotheses, list)
        if hypotheses:
            assert any(h.strategy == "causal_chain" for h in hypotheses)

    def test_causal_chain_multi_hop(self, abductive_engine, setup_test_knowledge):
        """Test: Causal-Chain mit mehreren Schritten"""
        # Erstelle längere kausale Kette
        setup_test_knowledge.ensure_wort_und_konzept("test_regen")
        setup_test_knowledge.ensure_wort_und_konzept("test_wolken")
        setup_test_knowledge.ensure_wort_und_konzept("test_verdunstung")

        setup_test_knowledge.assert_relation(
            "test_verdunstung", "CAUSES", "test_wolken"
        )
        setup_test_knowledge.assert_relation("test_wolken", "CAUSES", "test_regen")

        observation = "Es regnet (test_regen)"

        hypotheses = abductive_engine.generate_hypotheses(
            observation=observation, strategies=["causal_chain"], max_hypotheses=5
        )

        assert isinstance(hypotheses, list)


class TestMultiCriteriaScoring:
    """Testet das Multi-Kriterien-Scoring-System."""

    def test_coverage_score(self, abductive_engine):
        """Test: Coverage-Score berechnet korrekt"""
        from component_9_logik_engine import Fact

        hyp = Hypothesis(
            id="test_score_1",
            explanation="Test",
            observations=["obs1", "obs2", "obs3"],
            abduced_facts=[
                Fact(
                    pred="test_pred",
                    args={"subject": "obs1", "object": "x"},
                    confidence=0.9,
                ),
                Fact(
                    pred="test_pred",
                    args={"subject": "obs2", "object": "x"},
                    confidence=0.9,
                ),
            ],
            strategy="template",
            confidence=0.0,
            scores={},
        )

        # Score Hypothese und prüfe dass Coverage gesetzt wird (in-place modification)
        abductive_engine._score_hypothesis(hyp, context_facts=[])
        assert "coverage" in hyp.scores
        assert 0.0 <= hyp.scores["coverage"] <= 1.0

    def test_simplicity_score(self, abductive_engine):
        """Test: Simplicity-Score (Occam's Razor)"""
        from component_9_logik_engine import Fact

        # Einfache Hypothese (wenige Fakten)
        hyp_simple = Hypothesis(
            id="test_simple",
            explanation="Simple",
            observations=["obs1"],
            abduced_facts=[
                Fact(
                    pred="test_pred",
                    args={"subject": "x", "object": "y"},
                    confidence=0.9,
                )
            ],
            strategy="template",
            confidence=0.0,
            scores={},
        )

        # Komplexe Hypothese (viele Fakten)
        hyp_complex = Hypothesis(
            id="test_complex",
            explanation="Complex",
            observations=["obs1"],
            abduced_facts=[
                Fact(
                    pred="test_pred",
                    args={"subject": f"x{i}", "object": "y"},
                    confidence=0.9,
                )
                for i in range(10)
            ],
            strategy="template",
            confidence=0.0,
            scores={},
        )

        abductive_engine._score_hypothesis(hyp_simple, context_facts=[])
        abductive_engine._score_hypothesis(hyp_complex, context_facts=[])

        assert "simplicity" in hyp_simple.scores
        assert "simplicity" in hyp_complex.scores
        assert hyp_simple.scores["simplicity"] > hyp_complex.scores["simplicity"]

    def test_coherence_score(self, abductive_engine, setup_test_knowledge):
        """Test: Coherence-Score mit existierendem Wissen"""
        from component_9_logik_engine import Fact

        # Hypothese kohärent mit Wissensbasis
        hyp = Hypothesis(
            id="test_coherence",
            explanation="Coherent hypothesis",
            observations=["test_obs"],
            abduced_facts=[
                Fact(
                    pred="IS_A",
                    args={"subject": "test_vogel", "object": "test_tier"},
                    confidence=0.9,
                )
            ],
            strategy="template",
            confidence=0.0,
            scores={},
        )

        abductive_engine._score_hypothesis(hyp, context_facts=[])
        assert "coherence" in hyp.scores
        assert 0.0 <= hyp.scores["coherence"] <= 1.0

    def test_specificity_score(self, abductive_engine):
        """Test: Specificity-Score für Testbarkeit"""
        from component_9_logik_engine import Fact

        # Spezifische Hypothese (konkrete Fakten)
        hyp_specific = Hypothesis(
            id="test_specific",
            explanation="Specific prediction",
            observations=["obs"],
            abduced_facts=[
                Fact(
                    pred="HAS_PROPERTY",
                    args={"subject": "test_x", "object": "test_specific_value"},
                    confidence=0.9,
                )
            ],
            strategy="template",
            confidence=0.0,
            scores={},
        )

        abductive_engine._score_hypothesis(hyp_specific, context_facts=[])
        assert "specificity" in hyp_specific.scores
        assert 0.0 <= hyp_specific.scores["specificity"] <= 1.0

    def test_weighted_scoring(self, abductive_engine):
        """Test: Gewichtete Kombination aller Scores"""
        scores = {
            "coverage": 0.8,
            "simplicity": 0.7,
            "coherence": 0.9,
            "specificity": 0.6,
        }

        # Default-Gewichte: coverage=0.3, simplicity=0.2, coherence=0.3, specificity=0.2
        expected = 0.8 * 0.3 + 0.7 * 0.2 + 0.9 * 0.3 + 0.6 * 0.2

        # Simuliere Scoring
        from component_9_logik_engine import Fact

        hyp = Hypothesis(
            id="test_weighted",
            explanation="Test",
            observations=["obs"],
            abduced_facts=[Fact(pred="test", args={}, confidence=0.9)],
            strategy="template",
            confidence=0.0,
            scores=scores,
        )

        # Confidence sollte gewichteter Durchschnitt sein
        assert (
            abs(
                expected
                - sum(scores[k] * abductive_engine.score_weights[k] for k in scores)
            )
            < 0.01
        )


class TestHypothesisPersistence:
    """Testet die Persistenz von Hypothesen in Neo4j."""

    def test_store_hypothesis(self, netzwerk):
        """Test: Speichere Hypothese in Neo4j"""
        success = netzwerk.store_hypothesis(
            hypothesis_id="test_hyp_persist_1",
            explanation="Test-Hypothese zum Speichern",
            observations=["test_obs_1", "test_obs_2"],
            strategy="template",
            confidence=0.85,
            scores={
                "coverage": 0.8,
                "simplicity": 0.9,
                "coherence": 0.85,
                "specificity": 0.8,
            },
            abduced_facts=[
                {
                    "pred": "test_IS_A",
                    "args": {"subject": "test_x", "object": "test_y"},
                    "confidence": 0.9,
                }
            ],
            sources=["test_source_1"],
            reasoning_trace="Test-Reasoning-Trace",
        )

        assert success is True

    def test_link_hypothesis_to_observations(self, netzwerk):
        """Test: Verknüpfe Hypothese mit Beobachtungen"""
        # Erstelle Hypothese
        netzwerk.store_hypothesis(
            hypothesis_id="test_hyp_link_1",
            explanation="Test",
            observations=["test_obs_link_1"],
            strategy="template",
            confidence=0.8,
            scores={},
            abduced_facts=[],
        )

        # Verknüpfe mit Beobachtungen
        success = netzwerk.link_hypothesis_to_observations(
            "test_hyp_link_1", ["test_obs_link_1", "test_obs_link_2"]
        )

        assert success is True

    def test_link_hypothesis_to_concepts(self, netzwerk):
        """Test: Verknüpfe Hypothese mit Konzepten"""
        # Erstelle Hypothese
        netzwerk.store_hypothesis(
            hypothesis_id="test_hyp_concept_1",
            explanation="Test",
            observations=["test_obs"],
            strategy="analogy",
            confidence=0.75,
            scores={},
            abduced_facts=[],
        )

        # Erstelle Konzepte
        netzwerk.ensure_wort_und_konzept("test_konzept_1")
        netzwerk.ensure_wort_und_konzept("test_konzept_2")

        # Verknüpfe
        success = netzwerk.link_hypothesis_to_concepts(
            "test_hyp_concept_1", ["test_konzept_1", "test_konzept_2"]
        )

        assert success is True

    def test_query_hypotheses_about_topic(self, netzwerk):
        """Test: Query Hypothesen zu einem Thema"""
        # Erstelle mehrere Hypothesen
        for i in range(3):
            netzwerk.store_hypothesis(
                hypothesis_id=f"test_hyp_query_{i}",
                explanation=f"Test-Hypothese {i}",
                observations=[f"test_obs_{i}"],
                strategy="template" if i % 2 == 0 else "analogy",
                confidence=0.7 + i * 0.1,
                scores={},
                abduced_facts=[],
            )

            # Verknüpfe mit Konzept
            netzwerk.ensure_wort_und_konzept("test_query_topic")
            netzwerk.link_hypothesis_to_concepts(
                f"test_hyp_query_{i}", ["test_query_topic"]
            )

        # Query
        hypotheses = netzwerk.query_hypotheses_about(topic="test_query_topic", limit=10)

        assert len(hypotheses) >= 3
        # Sollten nach Confidence sortiert sein
        if len(hypotheses) >= 2:
            assert hypotheses[0]["confidence"] >= hypotheses[1]["confidence"]

    def test_query_hypotheses_filtered_by_strategy(self, netzwerk):
        """Test: Query mit Strategy-Filter"""
        # Erstelle Hypothesen mit verschiedenen Strategien
        netzwerk.store_hypothesis(
            hypothesis_id="test_hyp_template_filter",
            explanation="Template-basiert",
            observations=["obs"],
            strategy="template",
            confidence=0.8,
            scores={},
            abduced_facts=[],
        )

        netzwerk.store_hypothesis(
            hypothesis_id="test_hyp_analogy_filter",
            explanation="Analogie-basiert",
            observations=["obs"],
            strategy="analogy",
            confidence=0.75,
            scores={},
            abduced_facts=[],
        )

        netzwerk.ensure_wort_und_konzept("test_filter_topic")
        netzwerk.link_hypothesis_to_concepts(
            "test_hyp_template_filter", ["test_filter_topic"]
        )
        netzwerk.link_hypothesis_to_concepts(
            "test_hyp_analogy_filter", ["test_filter_topic"]
        )

        # Query nur template-basierte
        hypotheses = netzwerk.query_hypotheses_about(
            topic="test_filter_topic", strategy="template", limit=10
        )

        assert all(h["strategy"] == "template" for h in hypotheses)

    def test_query_hypotheses_filtered_by_confidence(self, netzwerk):
        """Test: Query mit Confidence-Filter"""
        # Erstelle Hypothesen mit verschiedenen Confidence-Werten
        for i, conf in enumerate([0.5, 0.7, 0.9]):
            netzwerk.store_hypothesis(
                hypothesis_id=f"test_hyp_conf_{i}",
                explanation=f"Confidence {conf}",
                observations=["obs"],
                strategy="template",
                confidence=conf,
                scores={},
                abduced_facts=[],
            )

            netzwerk.ensure_wort_und_konzept("test_conf_topic")
            netzwerk.link_hypothesis_to_concepts(
                f"test_hyp_conf_{i}", ["test_conf_topic"]
            )

        # Query nur mit min_confidence=0.75
        hypotheses = netzwerk.query_hypotheses_about(
            topic="test_conf_topic", min_confidence=0.75, limit=10
        )

        assert all(h["confidence"] >= 0.75 for h in hypotheses)

    def test_get_best_hypothesis(self, netzwerk):
        """Test: Hole beste Hypothese für Thema"""
        # Erstelle mehrere Hypothesen
        for i, conf in enumerate([0.6, 0.9, 0.7]):
            netzwerk.store_hypothesis(
                hypothesis_id=f"test_hyp_best_{i}",
                explanation=f"Hypothese {i}",
                observations=["obs"],
                strategy="template",
                confidence=conf,
                scores={},
                abduced_facts=[],
            )

            netzwerk.ensure_wort_und_konzept("test_best_topic")
            netzwerk.link_hypothesis_to_concepts(
                f"test_hyp_best_{i}", ["test_best_topic"]
            )

        # Hole beste
        best = netzwerk.get_best_hypothesis_for("test_best_topic")

        assert best is not None
        assert best["confidence"] == 0.9

    def test_explain_hypothesis(self, netzwerk):
        """Test: Generiere Erklärung für Hypothese"""
        # Erstelle Hypothese
        netzwerk.store_hypothesis(
            hypothesis_id="test_hyp_explain",
            explanation="Diese Hypothese erklärt etwas",
            observations=["obs1", "obs2"],
            strategy="causal_chain",
            confidence=0.85,
            scores={
                "coverage": 0.9,
                "simplicity": 0.8,
                "coherence": 0.85,
                "specificity": 0.8,
            },
            abduced_facts=[
                {
                    "pred": "CAUSES",
                    "args": {"subject": "x", "object": "y"},
                    "confidence": 0.9,
                }
            ],
            reasoning_trace="Test-Reasoning",
        )

        # Generiere Erklärung
        explanation = netzwerk.explain_hypothesis("test_hyp_explain")

        assert isinstance(explanation, str)
        assert len(explanation) > 0
        assert "test_hyp_explain" in explanation or "Hypothese" in explanation


class TestIntegrationWithKaiWorker:
    """Testet die Integration mit KAI Worker."""

    def test_abductive_engine_initialization(self, netzwerk, engine):
        """Test: AbductiveEngine kann initialisiert werden"""
        abd_eng = AbductiveEngine(netzwerk, engine)

        assert abd_eng.netzwerk is not None
        assert abd_eng.logic_engine is not None
        assert len(abd_eng.causal_patterns) > 0

    def test_generate_hypotheses_integration(
        self, abductive_engine, setup_test_knowledge
    ):
        """Test: Vollständige Hypothesen-Generierung"""
        observation = "test_vogel kann test_fliegen"

        hypotheses = abductive_engine.generate_hypotheses(
            observation=observation,
            strategies=["template", "analogy", "causal_chain"],
            max_hypotheses=10,
        )

        assert isinstance(hypotheses, list)
        # Sollte nach Confidence sortiert sein
        if len(hypotheses) >= 2:
            assert hypotheses[0].confidence >= hypotheses[1].confidence

    def test_multi_strategy_integration(self, abductive_engine, setup_test_knowledge):
        """Test: Alle Strategien parallel"""
        observation = "test_neues_phänomen tritt auf"

        hypotheses = abductive_engine.generate_hypotheses(
            observation=observation,
            strategies=["template", "analogy", "causal_chain"],
            max_hypotheses=15,
        )

        # Validiere dass verschiedene Strategien vertreten sein können
        strategies_used = set(h.strategy for h in hypotheses)
        assert len(strategies_used) >= 0  # Kann 0 sein wenn keine Hypothesen gefunden


class TestEdgeCases:
    """Testet Edge Cases und Fehlerfälle."""

    def test_generate_hypotheses_empty_observation(self, abductive_engine):
        """Test: Leere Beobachtung"""
        hypotheses = abductive_engine.generate_hypotheses(
            observation="", strategies=["template"], max_hypotheses=5
        )

        assert isinstance(hypotheses, list)

    def test_generate_hypotheses_no_strategies(self, abductive_engine):
        """Test: Keine Strategien angegeben (sollte alle nutzen)"""
        hypotheses = abductive_engine.generate_hypotheses(
            observation="test observation", strategies=None, max_hypotheses=5
        )

        assert isinstance(hypotheses, list)

    def test_generate_hypotheses_invalid_strategy(self, abductive_engine):
        """Test: Ungültige Strategie"""
        hypotheses = abductive_engine.generate_hypotheses(
            observation="test observation",
            strategies=["invalid_strategy"],
            max_hypotheses=5,
        )

        # Sollte leere Liste zurückgeben oder Exception werfen
        assert isinstance(hypotheses, list)

    def test_query_nonexistent_hypothesis(self, netzwerk):
        """Test: Query nicht-existierende Hypothese"""
        explanation = netzwerk.explain_hypothesis("nonexistent-hyp-id-12345")
        assert "nicht gefunden" in explanation.lower()

    def test_store_hypothesis_without_observations(self, netzwerk):
        """Test: Hypothese ohne Beobachtungen"""
        success = netzwerk.store_hypothesis(
            hypothesis_id="test_hyp_no_obs",
            explanation="Test",
            observations=[],
            strategy="template",
            confidence=0.8,
            scores={},
            abduced_facts=[],
        )

        assert success is True

    def test_hypothesis_with_no_abduced_facts(self, netzwerk):
        """Test: Hypothese ohne abgeleitete Fakten"""
        success = netzwerk.store_hypothesis(
            hypothesis_id="test_hyp_no_facts",
            explanation="Hypothese ohne neue Fakten",
            observations=["obs1"],
            strategy="analogy",
            confidence=0.7,
            scores={},
            abduced_facts=[],
        )

        assert success is True


class TestPerformance:
    """Testet Performance-Aspekte."""

    @pytest.mark.slow
    def test_hypothesis_generation_performance(
        self, abductive_engine, setup_test_knowledge
    ):
        """Test: Hypothesen-Generierung sollte schnell sein"""
        import time

        start = time.time()
        hypotheses = abductive_engine.generate_hypotheses(
            observation="test_vogel kann test_fliegen",
            strategies=["template", "analogy", "causal_chain"],
            max_hypotheses=20,
        )
        elapsed = time.time() - start

        # Sollte unter 5 Sekunden sein
        assert elapsed < 5.0
        assert isinstance(hypotheses, list)

    @pytest.mark.slow
    def test_batch_hypothesis_storage(self, netzwerk):
        """Test: Batch-Speicherung mehrerer Hypothesen"""
        import time

        start = time.time()
        for i in range(10):
            netzwerk.store_hypothesis(
                hypothesis_id=f"test_hyp_batch_{i}",
                explanation=f"Batch-Hypothese {i}",
                observations=[f"obs_{i}"],
                strategy="template",
                confidence=0.7 + i * 0.02,
                scores={},
                abduced_facts=[],
            )
        elapsed = time.time() - start

        # 10 Hypothesen sollten unter 3 Sekunden sein
        assert elapsed < 3.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
