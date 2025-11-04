# test_episodic_reasoning.py
"""
Tests für Episodic Memory for Reasoning System.

Testet die Tracking-Fähigkeiten von InferenceEpisodes und ProofSteps.
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from component_1_netzwerk import KonzeptNetzwerk
from component_9_logik_engine import Engine, Goal, Fact


@pytest.fixture
def netzwerk():
    """Erstellt eine Neo4j-Verbindung für Tests."""
    netz = KonzeptNetzwerk()
    yield netz
    # Cleanup: Lösche Test-InferenceEpisodes
    if netz.driver:
        with netz.driver.session(database="neo4j") as session:
            session.run(
                "MATCH (ie:InferenceEpisode) WHERE ie.query CONTAINS 'test_' DETACH DELETE ie"
            )
            session.run(
                "MATCH (ps:ProofStep) WHERE ps.goal CONTAINS 'test_' DETACH DELETE ps"
            )
    netz.close()


@pytest.fixture
def engine(netzwerk):
    """Erstellt eine Engine-Instanz mit Netzwerk."""
    eng = Engine(netzwerk)
    return eng


class TestInferenceEpisodeCreation:
    """Testet die Erstellung von InferenceEpisodes."""

    def test_create_inference_episode(self, netzwerk):
        """Test: Erstelle InferenceEpisode"""
        episode_id = netzwerk.create_inference_episode(
            inference_type="test_backward_chaining",
            query="test_query: Was ist ein Hund?",
            metadata={"topic": "test_hund", "max_depth": 5},
        )

        assert episode_id is not None
        assert len(episode_id) > 0

    def test_create_proof_step(self, netzwerk):
        """Test: Erstelle ProofStep"""
        step_id = netzwerk.create_proof_step(
            goal="test_IS_A(test_hund, ?x)",
            method="fact",
            confidence=1.0,
            depth=0,
            bindings={"?x": "test_tier"},
        )

        assert step_id is not None
        assert len(step_id) > 0

    def test_link_inference_to_proof(self, netzwerk):
        """Test: Verknüpfe InferenceEpisode mit ProofStep"""
        # Erstelle Episode und ProofStep
        episode_id = netzwerk.create_inference_episode(
            inference_type="test_hybrid", query="test_link_query", metadata={}
        )

        step_id = netzwerk.create_proof_step(
            goal="test_goal", method="rule", confidence=0.9, depth=0
        )

        # Verknüpfe
        success = netzwerk.link_inference_to_proof(episode_id, step_id)
        assert success is True


class TestProofStepHierarchy:
    """Testet die Hierarchie von ProofSteps."""

    def test_create_proof_tree(self, netzwerk):
        """Test: Erstelle hierarchischen Beweisbaum"""
        # Root ProofStep
        root_id = netzwerk.create_proof_step(
            goal="test_root_goal", method="rule", confidence=1.0, depth=0
        )

        # Child ProofSteps
        child1_id = netzwerk.create_proof_step(
            goal="test_child1_goal",
            method="fact",
            confidence=1.0,
            depth=1,
            parent_step_id=root_id,
        )

        child2_id = netzwerk.create_proof_step(
            goal="test_child2_goal",
            method="fact",
            confidence=0.9,
            depth=1,
            parent_step_id=root_id,
        )

        assert root_id is not None
        assert child1_id is not None
        assert child2_id is not None

        # Rekonstruiere Baum
        tree = netzwerk.get_proof_tree(root_id)
        assert tree is not None
        assert tree["goal"] == "test_root_goal"
        assert len(tree["children"]) == 2


class TestInferenceTracking:
    """Testet das vollständige Inference-Tracking."""

    def test_run_with_tracking(self, engine, netzwerk):
        """Test: Führe tracked reasoning aus"""
        # Füge Testfakten hinzu
        engine.add_fact(
            Fact(
                pred="test_IS_A",
                args={"subject": "test_hund", "object": "test_säugetier"},
                id="test_fact_1",
            )
        )

        engine.add_fact(
            Fact(
                pred="test_IS_A",
                args={"subject": "test_säugetier", "object": "test_tier"},
                id="test_fact_2",
            )
        )

        # Erstelle Goal
        goal = Goal(pred="test_IS_A", args={"subject": "test_hund", "object": None})

        # Führe tracked reasoning aus
        proof = engine.run_with_tracking(
            goal=goal,
            inference_type="test_backward_chaining",
            query="test_Was ist ein Hund?",
            max_depth=5,
        )

        # Assertions
        assert proof is not None
        assert engine.current_inference_episode_id is not None

        # Prüfe, ob Episode erstellt wurde
        episodes = netzwerk.query_inference_history(topic="test_hund", limit=5)

        assert len(episodes) > 0
        assert episodes[0]["inference_type"] == "test_backward_chaining"

    def test_inference_history_query(self, netzwerk):
        """Test: Query inference history"""
        # Erstelle mehrere Test-Episoden
        for i in range(3):
            netzwerk.create_inference_episode(
                inference_type="test_type",
                query=f"test_query_{i}: test_topic",
                metadata={"index": i},
            )

        # Query
        episodes = netzwerk.query_inference_history(topic="test_topic", limit=10)

        assert len(episodes) >= 3

    def test_explain_inference(self, netzwerk):
        """Test: Generiere Erklärung für Inferenz"""
        # Erstelle Episode
        episode_id = netzwerk.create_inference_episode(
            inference_type="test_explanation", query="test_explain_query", metadata={}
        )

        # Generiere Erklärung
        explanation = netzwerk.explain_inference(episode_id)

        assert isinstance(explanation, str)
        assert len(explanation) > 0
        assert "test_explain_query" in explanation


class TestEdgeCases:
    """Testet Edge Cases."""

    def test_create_episode_without_metadata(self, netzwerk):
        """Test: Episode ohne Metadata"""
        episode_id = netzwerk.create_inference_episode(
            inference_type="test_no_metadata", query="test_query_no_meta"
        )

        assert episode_id is not None

    def test_query_nonexistent_episode(self, netzwerk):
        """Test: Query nicht-existierende Episode"""
        explanation = netzwerk.explain_inference("nonexistent-id-12345")
        assert "nicht gefunden" in explanation.lower()

    def test_get_proof_tree_nonexistent(self, netzwerk):
        """Test: Hole nicht-existierenden Proof-Tree"""
        tree = netzwerk.get_proof_tree("nonexistent-step-id")
        assert tree is None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
