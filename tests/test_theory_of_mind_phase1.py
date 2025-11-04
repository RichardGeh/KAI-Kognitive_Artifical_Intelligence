"""
KAI Test Suite - Theory of Mind Phase 1 Tests
Tests für Agent, Belief, und MetaBelief Nodes und Relations.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

logger = logging.getLogger(__name__)


class TestTheoryOfMindPhase1:
    """Tests für Phase 1: Foundation & Graph Schema."""

    def test_create_agent_basic(self, netzwerk_session, clean_test_concepts):
        """Testet das Erstellen eines Agent Nodes."""
        agent_id = f"{clean_test_concepts}alice"
        name = "Alice"

        # Erstelle Agent
        success = netzwerk_session.create_agent(agent_id, name, reasoning_capacity=5)
        assert success, "create_agent sollte True zurückgeben"

        # Verifiziere Agent in Neo4j
        with netzwerk_session.driver.session(database="neo4j") as session:
            result = session.run(
                """
                MATCH (a:Agent {id: $agent_id})
                RETURN a.id AS id, a.name AS name, a.reasoning_capacity AS capacity
            """,
                agent_id=agent_id,
            )
            record = result.single()

            assert record is not None, "Agent Node sollte existieren"
            assert record["id"] == agent_id
            assert record["name"] == name
            assert record["capacity"] == 5

    def test_create_agent_idempotent(self, netzwerk_session, clean_test_concepts):
        """Testet dass create_agent idempotent ist (kein Duplikat bei wiederholtem Aufruf)."""
        agent_id = f"{clean_test_concepts}bob"
        name = "Bob"

        # Erstelle Agent zweimal
        netzwerk_session.create_agent(agent_id, name)
        netzwerk_session.create_agent(agent_id, name)

        # Verifiziere nur ein Agent Node existiert
        with netzwerk_session.driver.session(database="neo4j") as session:
            result = session.run(
                """
                MATCH (a:Agent {id: $agent_id})
                RETURN count(a) AS count
            """,
                agent_id=agent_id,
            )
            count = result.single()["count"]

            assert count == 1, "Doppeltes Erstellen sollte keine Duplikate erzeugen"

    def test_add_belief_basic(self, netzwerk_session, clean_test_concepts):
        """Testet das Erstellen eines Belief Nodes mit KNOWS Relation."""
        agent_id = f"{clean_test_concepts}charlie"
        name = "Charlie"
        proposition = "hund IS_A tier"

        # Erstelle Agent
        netzwerk_session.create_agent(agent_id, name)

        # Erstelle Belief
        success = netzwerk_session.add_belief(agent_id, proposition, certainty=0.9)
        assert success, "add_belief sollte True zurückgeben"

        # Verifiziere Belief und KNOWS Relation in Neo4j
        with netzwerk_session.driver.session(database="neo4j") as session:
            result = session.run(
                """
                MATCH (a:Agent {id: $agent_id})-[:KNOWS]->(b:Belief)
                RETURN b.proposition AS proposition, b.certainty AS certainty
            """,
                agent_id=agent_id,
            )
            record = result.single()

            assert (
                record is not None
            ), "Belief Node mit KNOWS Relation sollte existieren"
            assert record["proposition"] == proposition
            assert record["certainty"] == 0.9

    def test_add_belief_without_agent(self, netzwerk_session, clean_test_concepts):
        """Testet dass add_belief fehlschlägt wenn Agent nicht existiert."""
        agent_id = f"{clean_test_concepts}nonexistent"
        proposition = "test proposition"

        # Versuche Belief für nicht-existierenden Agent zu erstellen
        success = netzwerk_session.add_belief(agent_id, proposition)
        assert (
            not success
        ), "add_belief sollte False zurückgeben wenn Agent nicht existiert"

    def test_add_meta_belief_basic(self, netzwerk_session, clean_test_concepts):
        """Testet das Erstellen eines MetaBelief Nodes mit KNOWS_THAT und ABOUT_AGENT Relations."""
        observer_id = f"{clean_test_concepts}alice_mb"
        subject_id = f"{clean_test_concepts}bob_mb"
        proposition = "katze IS_A tier"
        meta_level = 1

        # Erstelle beide Agents
        netzwerk_session.create_agent(observer_id, "Alice")
        netzwerk_session.create_agent(subject_id, "Bob")

        # Erstelle MetaBelief
        success = netzwerk_session.add_meta_belief(
            observer_id, subject_id, proposition, meta_level
        )
        assert success, "add_meta_belief sollte True zurückgeben"

        # Verifiziere MetaBelief und Relations in Neo4j
        with netzwerk_session.driver.session(database="neo4j") as session:
            result = session.run(
                """
                MATCH (observer:Agent {id: $observer_id})-[:KNOWS_THAT]->(mb:MetaBelief)-[:ABOUT_AGENT]->(subject:Agent {id: $subject_id})
                RETURN mb.proposition AS proposition, mb.meta_level AS meta_level, subject.id AS subject_id
            """,
                observer_id=observer_id,
                subject_id=subject_id,
            )
            record = result.single()

            assert (
                record is not None
            ), "MetaBelief mit KNOWS_THAT und ABOUT_AGENT Relations sollte existieren"
            assert record["proposition"] == proposition
            assert record["meta_level"] == meta_level
            assert record["subject_id"] == subject_id

    def test_add_meta_belief_without_agents(
        self, netzwerk_session, clean_test_concepts
    ):
        """Testet dass add_meta_belief fehlschlägt wenn Agents nicht existieren."""
        observer_id = f"{clean_test_concepts}nonexistent_observer"
        subject_id = f"{clean_test_concepts}nonexistent_subject"
        proposition = "test proposition"

        # Versuche MetaBelief für nicht-existierende Agents zu erstellen
        success = netzwerk_session.add_meta_belief(
            observer_id, subject_id, proposition, meta_level=1
        )
        assert (
            not success
        ), "add_meta_belief sollte False zurückgeben wenn Agents nicht existieren"

    def test_multiple_beliefs_per_agent(self, netzwerk_session, clean_test_concepts):
        """Testet dass ein Agent mehrere Beliefs haben kann."""
        agent_id = f"{clean_test_concepts}diana"
        name = "Diana"

        # Erstelle Agent
        netzwerk_session.create_agent(agent_id, name)

        # Erstelle mehrere Beliefs
        propositions = ["hund IS_A tier", "katze IS_A tier", "vogel IS_A tier"]

        for prop in propositions:
            success = netzwerk_session.add_belief(agent_id, prop)
            assert success, f"add_belief sollte für {prop} erfolgreich sein"

        # Verifiziere alle Beliefs in Neo4j
        with netzwerk_session.driver.session(database="neo4j") as session:
            result = session.run(
                """
                MATCH (a:Agent {id: $agent_id})-[:KNOWS]->(b:Belief)
                RETURN count(b) AS belief_count
            """,
                agent_id=agent_id,
            )
            count = result.single()["belief_count"]

            assert count == 3, "Agent sollte 3 Beliefs haben"

    def test_constraints_exist(self, netzwerk_session):
        """Testet dass die neuen Constraints in Neo4j aktiv sind."""
        with netzwerk_session.driver.session(database="neo4j") as session:
            result = session.run("SHOW CONSTRAINTS")
            constraints = [record["name"] for record in result]

            # Prüfe dass die drei neuen Constraints existieren
            assert "AgentId" in constraints, "AgentId Constraint sollte existieren"
            assert "BeliefId" in constraints, "BeliefId Constraint sollte existieren"
            assert (
                "MetaBeliefId" in constraints
            ), "MetaBeliefId Constraint sollte existieren"

    def test_agent_with_custom_reasoning_capacity(
        self, netzwerk_session, clean_test_concepts
    ):
        """Testet dass reasoning_capacity korrekt gesetzt wird."""
        agent_id = f"{clean_test_concepts}eve"
        name = "Eve"
        reasoning_capacity = 10

        # Erstelle Agent mit custom reasoning_capacity
        netzwerk_session.create_agent(
            agent_id, name, reasoning_capacity=reasoning_capacity
        )

        # Verifiziere reasoning_capacity
        with netzwerk_session.driver.session(database="neo4j") as session:
            result = session.run(
                """
                MATCH (a:Agent {id: $agent_id})
                RETURN a.reasoning_capacity AS capacity
            """,
                agent_id=agent_id,
            )
            capacity = result.single()["capacity"]

            assert (
                capacity == reasoning_capacity
            ), f"reasoning_capacity sollte {reasoning_capacity} sein"

    def test_belief_with_custom_certainty(self, netzwerk_session, clean_test_concepts):
        """Testet dass certainty korrekt gesetzt wird."""
        agent_id = f"{clean_test_concepts}frank"
        name = "Frank"
        proposition = "test proposition"
        certainty = 0.7

        # Erstelle Agent und Belief
        netzwerk_session.create_agent(agent_id, name)
        netzwerk_session.add_belief(agent_id, proposition, certainty=certainty)

        # Verifiziere certainty
        with netzwerk_session.driver.session(database="neo4j") as session:
            result = session.run(
                """
                MATCH (a:Agent {id: $agent_id})-[:KNOWS]->(b:Belief)
                RETURN b.certainty AS certainty
            """,
                agent_id=agent_id,
            )
            retrieved_certainty = result.single()["certainty"]

            assert (
                retrieved_certainty == certainty
            ), f"certainty sollte {certainty} sein"
