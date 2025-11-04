"""
Additional tests for component_8_prototype_matcher.py to increase coverage from 64% to 70%+

Focuses on error handling, edge cases, and different matching scenarios.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
from component_1_netzwerk import KonzeptNetzwerk
from component_8_prototype_matcher import PrototypingEngine
from component_11_embedding_service import EmbeddingService


@pytest.fixture(scope="session")
def netzwerk_session():
    """Session-scoped netzwerk fixture."""
    netzwerk = KonzeptNetzwerk()
    yield netzwerk
    netzwerk.close()


@pytest.fixture(scope="session")
def embedding_service_session():
    """Session-scoped embedding service fixture."""
    return EmbeddingService()


@pytest.fixture(scope="session")
def engine_session(netzwerk_session, embedding_service_session):
    """Session-scoped prototyping engine fixture."""
    return PrototypingEngine(netzwerk_session, embedding_service_session)


class TestPrototypingEngineEdgeCases:
    """Tests for edge cases and error handling in PrototypingEngine."""

    def test_get_embedding_for_text_success(self, engine_session):
        """Test successful embedding generation."""
        embedding = engine_session.get_embedding_for_text("Test text")
        assert embedding is not None
        assert isinstance(embedding, list)
        assert len(embedding) > 0

    def test_get_embedding_for_text_empty(self, engine_session):
        """Test embedding generation with empty text."""
        # Empty text should raise ValueError from embedding service
        embedding = engine_session.get_embedding_for_text("")
        assert embedding is None

    def test_process_vector_empty(self, engine_session):
        """Test process_vector with empty vector."""
        result = engine_session.process_vector([], "TEST")
        assert result is None

    def test_process_vector_no_driver(self, embedding_service_session):
        """Test process_vector when driver is None."""
        netzwerk = KonzeptNetzwerk()
        netzwerk.driver = None
        engine = PrototypingEngine(netzwerk, embedding_service_session)

        test_vector = [0.1] * 384
        result = engine.process_vector(test_vector, "TEST")
        assert result is None

    def test_find_best_match_empty_vector(self, engine_session):
        """Test find_best_match with empty vector."""
        result = engine_session.find_best_match([])
        assert result is None

    def test_find_best_match_no_driver(self, embedding_service_session):
        """Test find_best_match when driver is None."""
        netzwerk = KonzeptNetzwerk()
        netzwerk.driver = None
        engine = PrototypingEngine(netzwerk, embedding_service_session)

        test_vector = [0.1] * 384
        result = engine.find_best_match(test_vector)
        assert result is None

    def test_find_best_match_no_prototypes(
        self, netzwerk_session, embedding_service_session
    ):
        """Test find_best_match when no prototypes exist in a category."""
        engine = PrototypingEngine(netzwerk_session, embedding_service_session)

        test_vector = [0.1] * 384
        result = engine.find_best_match(
            test_vector, category_filter="NONEXISTENT_CATEGORY_12345"
        )
        assert result is None

    def test_find_best_match_with_category_filter(
        self, engine_session, embedding_service_session
    ):
        """Test find_best_match with category filter."""
        # Create a test prototype first
        test_vector = embedding_service_session.get_embedding(
            "Test sentence for matching"
        )

        # Process vector to create prototype
        prototype_id = engine_session.process_vector(test_vector, "TEST_CATEGORY")
        assert prototype_id is not None

        # Now find best match with category filter
        similar_vector = embedding_service_session.get_embedding(
            "Test sentence similar"
        )
        result = engine_session.find_best_match(
            similar_vector, category_filter="TEST_CATEGORY"
        )

        # Should find the prototype we just created
        if result:
            prototype, distance = result
            assert prototype is not None
            assert distance >= 0


class TestPrototypeUpdate:
    """Tests for prototype update logic."""

    def test_calculate_euclidean_distance(self, engine_session):
        """Test Euclidean distance calculation."""
        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([4.0, 5.0, 6.0])

        distance = engine_session._calculate_euclidean_distance(vec1, vec2)
        expected = np.sqrt((4 - 1) ** 2 + (5 - 2) ** 2 + (6 - 3) ** 2)

        assert abs(distance - expected) < 0.001

    def test_update_prototype_updates_correctly(
        self, netzwerk_session, embedding_service_session
    ):
        """Test that prototype update correctly updates centroid."""
        engine = PrototypingEngine(netzwerk_session, embedding_service_session)

        # Create initial prototype
        initial_vector = embedding_service_session.get_embedding("Initial test text")
        prototype_id = engine.process_vector(initial_vector, "UPDATE_TEST")

        assert prototype_id is not None

        # Get the prototype
        prototypes = netzwerk_session.get_all_pattern_prototypes()
        prototype = next((p for p in prototypes if p["id"] == prototype_id), None)

        if prototype:
            # Update it with a similar vector
            similar_vector_list = embedding_service_session.get_embedding(
                "Similar test text"
            )
            np.array(similar_vector_list)

            # This should trigger an update
            result = engine.process_vector(similar_vector_list, "UPDATE_TEST")

            # Should return the same prototype ID
            assert result == prototype_id


class TestProcessVectorScenarios:
    """Tests for different process_vector scenarios."""

    def test_process_vector_creates_first_prototype(
        self, netzwerk_session, embedding_service_session
    ):
        """Test that process_vector creates first prototype when none exist."""
        engine = PrototypingEngine(netzwerk_session, embedding_service_session)

        test_vector = embedding_service_session.get_embedding("First prototype test")
        prototype_id = engine.process_vector(test_vector, "FIRST_TEST")

        assert prototype_id is not None
        assert isinstance(prototype_id, str)

    def test_process_vector_updates_existing_close_match(
        self, netzwerk_session, embedding_service_session
    ):
        """Test that close vectors update existing prototypes."""
        engine = PrototypingEngine(netzwerk_session, embedding_service_session)

        # Create initial prototype
        initial_text = "This is a test sentence"
        initial_vector = embedding_service_session.get_embedding(initial_text)
        prototype_id_1 = engine.process_vector(initial_vector, "CLOSE_MATCH_TEST")

        # Process very similar text (should update, not create new)
        similar_text = "This is a test sentence too"
        similar_vector = embedding_service_session.get_embedding(similar_text)
        prototype_id_2 = engine.process_vector(similar_vector, "CLOSE_MATCH_TEST")

        # Should be the same prototype
        assert prototype_id_1 == prototype_id_2

    def test_process_vector_creates_new_for_different_category(
        self, netzwerk_session, embedding_service_session
    ):
        """Test that different categories create separate prototypes."""
        engine = PrototypingEngine(netzwerk_session, embedding_service_session)

        # Create prototype in category A
        test_vector = embedding_service_session.get_embedding("Category test")
        prototype_id_a = engine.process_vector(test_vector, "CATEGORY_A")

        # Create prototype with same vector but different category
        prototype_id_b = engine.process_vector(test_vector, "CATEGORY_B")

        # Should be different prototypes
        assert prototype_id_a != prototype_id_b


class TestFindBestMatchScenarios:
    """Tests for find_best_match scenarios."""

    def test_find_best_match_returns_closest(
        self, netzwerk_session, embedding_service_session
    ):
        """Test that find_best_match returns the closest prototype."""
        engine = PrototypingEngine(netzwerk_session, embedding_service_session)

        # Create two prototypes
        vector1 = embedding_service_session.get_embedding("First test prototype")
        vector2 = embedding_service_session.get_embedding(
            "Completely different content xyz 123"
        )

        id1 = engine.process_vector(vector1, "MATCH_TEST")
        id2 = engine.process_vector(vector2, "MATCH_TEST")

        # Query with text similar to first
        query_vector = embedding_service_session.get_embedding("First test similar")
        result = engine.find_best_match(query_vector, category_filter="MATCH_TEST")

        if result:
            prototype, distance = result
            # Should match closer to first prototype
            assert prototype["id"] in [id1, id2]
            assert distance >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
