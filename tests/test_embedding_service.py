"""
KAI Test Suite - Embedding Service Tests
Extrahiert aus test_kai_worker.py fuer bessere Wartbarkeit.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

logger = logging.getLogger(__name__)
TEST_VECTOR_DIM = 384


class TestEmbeddingService:
    """Tests für semantische Embeddings (ersetzt Featurizer)."""

    def test_embedding_service_initialization(self, embedding_service_session):
        """Prüft, ob der EmbeddingService korrekt initialisiert wird."""
        assert embedding_service_session.is_available()

    def test_embedding_vector_dimensions(self, embedding_service_session):
        """Prüft, ob Embeddings die richtige Dimension haben."""
        vector = embedding_service_session.get_embedding("Dies ist ein Test.")
        assert vector is not None
        assert len(vector) == 384  # Semantische Embeddings
        assert all(isinstance(v, float) for v in vector)

    def test_embedding_similarity(self, embedding_service_session):
        """Prüft, ob semantisch ähnliche Sätze ähnliche Embeddings haben mit quantitativen Schwellwerten."""
        vec1 = embedding_service_session.get_embedding("Ein Hund ist ein Tier.")
        vec2 = embedding_service_session.get_embedding("Ein Hund ist ein Lebewesen.")
        vec3 = embedding_service_session.get_embedding("Das Wetter ist schön.")

        # Berechne Kosinusähnlichkeit
        import numpy as np

        similarity_12 = np.dot(vec1, vec2) / (
            np.linalg.norm(vec1) * np.linalg.norm(vec2)
        )
        similarity_13 = np.dot(vec1, vec3) / (
            np.linalg.norm(vec1) * np.linalg.norm(vec3)
        )

        # Ähnliche Sätze sollten höhere Ähnlichkeit haben
        assert (
            similarity_12 > similarity_13
        ), f"Ähnliche Sätze sollten ähnlichere Embeddings haben: {similarity_12:.4f} vs {similarity_13:.4f}"

        # Quantitative Schwellwerte: Ähnliche Sätze sollten > 0.7 Ähnlichkeit haben
        assert (
            similarity_12 > 0.7
        ), f"Semantisch ähnliche Sätze sollten Ähnlichkeit > 0.7 haben, ist aber {similarity_12:.4f}"

        # Unähnliche Sätze sollten < 0.6 Ähnlichkeit haben
        assert (
            similarity_13 < 0.6
        ), f"Semantisch unähnliche Sätze sollten Ähnlichkeit < 0.6 haben, ist aber {similarity_13:.4f}"

        # Edge Case: Identische Sätze sollten Ähnlichkeit ~1.0 haben
        vec1_duplicate = embedding_service_session.get_embedding(
            "Ein Hund ist ein Tier."
        )
        similarity_identical = np.dot(vec1, vec1_duplicate) / (
            np.linalg.norm(vec1) * np.linalg.norm(vec1_duplicate)
        )
        assert (
            similarity_identical > 0.99
        ), f"Identische Sätze sollten nahezu 1.0 Ähnlichkeit haben, ist aber {similarity_identical:.4f}"

        # Edge Case: Leerer Text sollte Exception werfen
        try:
            embedding_service_session.get_embedding("")
            assert False, "Leerer Text sollte ValueError werfen"
        except ValueError as e:
            logger.info(f"[SUCCESS] Leerer Text korrekt abgefangen: {e}")

        logger.info(
            f"[SUCCESS] Ähnlichkeiten: ähnlich={similarity_12:.4f}, unähnlich={similarity_13:.4f}, identisch={similarity_identical:.4f}"
        )


# ============================================================================
# TESTS FÜR GOAL PLANNER (component_4_goal_planner.py)
# ============================================================================
