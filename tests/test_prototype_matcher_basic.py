"""
KAI Test Suite - Prototype Matcher Basic Tests
Basis-Tests aus test_kai_worker.py extrahiert.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import pytest

logger = logging.getLogger(__name__)
TEST_VECTOR_DIM = 384


class TestPrototypingEngine:
    """Tests für das Prototyp-Matching mit semantischen Embeddings."""

    def test_create_new_prototype(self, netzwerk_session, embedding_service_session):
        """Testet das Erstellen eines neuen Prototyps."""
        from component_8_prototype_matcher import PrototypingEngine

        engine = PrototypingEngine(netzwerk_session, embedding_service_session)

        test_vector = [0.5] * TEST_VECTOR_DIM  # 384D
        prototype_id = engine.process_vector(test_vector, "TEST_CATEGORY")

        assert prototype_id is not None

        # Cleanup
        with netzwerk_session.driver.session(database="neo4j") as session:
            session.run(
                """
                MATCH (p:PatternPrototype {id: $id})
                DETACH DELETE p
            """,
                id=prototype_id,
            )

    def test_update_existing_prototype(
        self, netzwerk_session, embedding_service_session
    ):
        """Testet das Aktualisieren eines existierenden Prototyps."""
        from component_8_prototype_matcher import PrototypingEngine

        engine = PrototypingEngine(netzwerk_session, embedding_service_session)

        # Verwende echte Embeddings für ähnliche Sätze
        test_vector1 = embedding_service_session.get_embedding("Ein Hund ist ein Tier.")
        test_vector2 = embedding_service_session.get_embedding(
            "Ein Hund ist ein Lebewesen."
        )

        try:
            prototype_id1 = engine.process_vector(test_vector1, "TEST_UPDATE")
            prototype_id2 = engine.process_vector(test_vector2, "TEST_UPDATE")

            # Sollten denselben Prototyp verwenden (Update) da semantisch ähnlich
            assert prototype_id1 == prototype_id2

            # Prüfe, ob count erhöht wurde
            prototypes = netzwerk_session.get_all_pattern_prototypes()
            test_proto = next((p for p in prototypes if p["id"] == prototype_id1), None)
            assert test_proto is not None
            assert test_proto["count"] == 2
        finally:
            with netzwerk_session.driver.session(database="neo4j") as session:
                session.run(
                    """
                    MATCH (p:PatternPrototype)
                    WHERE p.category = 'TEST_UPDATE'
                    DETACH DELETE p
                """
                )

    def test_find_best_match_returns_closest_prototype(
        self, netzwerk_session, embedding_service_session
    ):
        """
        Stellt sicher, dass der Prototyp mit der geringsten Distanz zurückgegeben wird.
        """
        from component_8_prototype_matcher import PrototypingEngine

        # Prüfe, ob die Methode existiert
        if not hasattr(PrototypingEngine, "find_best_match"):
            pytest.skip(
                "find_best_match() Methode existiert nicht in PrototypingEngine"
            )

        engine = PrototypingEngine(netzwerk_session, embedding_service_session)

        try:
            # SETUP: Erstelle zwei Prototypen mit unterschiedlicher Distanz
            close_vector = embedding_service_session.get_embedding(
                "Ein Hund ist ein Tier."
            )
            far_vector = embedding_service_session.get_embedding(
                "Das Wetter ist heute sonnig."
            )
            close_id = netzwerk_session.create_pattern_prototype(
                close_vector, "TEST_CLOSE"
            )
            far_id = netzwerk_session.create_pattern_prototype(far_vector, "TEST_FAR")

            assert close_id is not None, "Konnte close_id Prototyp nicht erstellen"
            assert far_id is not None, "Konnte far_id Prototyp nicht erstellen"

            # AKTION: Finde den besten Match für einen Vektor nahe am ersten
            test_vector = embedding_service_session.get_embedding(
                "Ein Hund ist ein Lebewesen."
            )
            result = engine.find_best_match(test_vector)

            # VERIFIKATION
            assert result is not None, "find_best_match sollte ein Ergebnis zurückgeben"
            matched_prototype, distance = result

            assert (
                matched_prototype["id"] == close_id
            ), f"Sollte close_id ({close_id}) matchen, nicht {matched_prototype['id']}"
            assert matched_prototype["category"] == "TEST_CLOSE"
            assert (
                distance < 20.0
            ), f"Distanz sollte kleiner als 20 sein, war aber {distance}"

            logger.info(
                f"[SUCCESS] Best match gefunden: ID={close_id}, Distance={distance:.4f}"
            )

        finally:
            # Cleanup
            with netzwerk_session.driver.session(database="neo4j") as session:
                session.run(
                    """
                    MATCH (p:PatternPrototype)
                    WHERE p.category IN ['TEST_CLOSE', 'TEST_FAR']
                    DETACH DELETE p
                """
                )

    def test_find_best_match_returns_none_if_no_prototypes(
        self, netzwerk_session, embedding_service_session
    ):
        """
        Stellt sicher, dass None zurückgegeben wird, wenn keine Prototypen existieren.
        """
        from component_8_prototype_matcher import PrototypingEngine

        # Prüfe, ob die Methode existiert
        if not hasattr(PrototypingEngine, "find_best_match"):
            pytest.skip(
                "find_best_match() Methode existiert nicht in PrototypingEngine"
            )

        engine = PrototypingEngine(netzwerk_session, embedding_service_session)

        # Stelle sicher, dass keine TEST-Prototypen existieren
        with netzwerk_session.driver.session(database="neo4j") as session:
            session.run(
                """
                MATCH (p:PatternPrototype)
                WHERE p.category STARTS WITH 'TEST_'
                DETACH DELETE p
            """
            )

        # AKTION: Versuche Match zu finden in leerer DB (für TEST-Kategorien)
        test_vector = embedding_service_session.get_embedding("Ein Test.")
        result = engine.find_best_match(test_vector, category_filter="TEST_EMPTY")

        # VERIFIKATION
        assert (
            result is None
        ), "Sollte None zurückgeben wenn keine Prototypen existieren"
        logger.info("[SUCCESS] Korrekt: None zurückgegeben bei fehlenden Prototypen")


# ============================================================================
# TESTS FÜR KAI WORKER - INTEGRIERTE FUNKTIONALITÄT
# ============================================================================
