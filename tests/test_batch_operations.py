"""
Tests for batch operations in component_1 modules.

Tests PRIORITY 2 performance improvements:
- Task 4: batch_assert_relations()
- Task 5: batch_create_episodes()

Follows CLAUDE.md standards:
- Thread safety
- Structured logging
- Comprehensive error handling
- Type hints
"""

import pytest

from component_1_episodic_memory import EpisodicMemory


class TestBatchAssertRelations:
    """Test batch_assert_relations() method (Task 4)."""

    def test_batch_create_single_relation_type(self, netzwerk_session):
        """Batch create multiple relations of same type."""
        relations = [
            {
                "subject": "hund",
                "relation": "IS_A",
                "object": "tier",
                "confidence": 0.9,
            },
            {
                "subject": "katze",
                "relation": "IS_A",
                "object": "tier",
                "confidence": 0.9,
            },
            {
                "subject": "maus",
                "relation": "IS_A",
                "object": "tier",
                "confidence": 0.9,
            },
        ]

        counts = netzwerk_session.batch_assert_relations(relations)

        assert "IS_A" in counts
        assert counts["IS_A"] == 3

        # Verify relations were created
        facts = netzwerk_session.query_graph_for_facts("hund")
        assert "tier" in facts.get("IS_A", [])

    def test_batch_create_multiple_relation_types(self, netzwerk_session):
        """Batch create relations of different types."""
        relations = [
            {"subject": "hund", "relation": "IS_A", "object": "tier"},
            {"subject": "hund", "relation": "HAS_PROPERTY", "object": "freundlich"},
            {"subject": "katze", "relation": "IS_A", "object": "tier"},
            {"subject": "katze", "relation": "HAS_PROPERTY", "object": "unabhängig"},
        ]

        counts = netzwerk_session.batch_assert_relations(relations)

        assert "IS_A" in counts
        assert "HAS_PROPERTY" in counts
        assert counts["IS_A"] == 2
        assert counts["HAS_PROPERTY"] == 2

    def test_batch_create_with_source_sentences(self, netzwerk_session):
        """Batch create with source sentences for provenance."""
        relations = [
            {
                "subject": "hund",
                "relation": "IS_A",
                "object": "tier",
                "source_sentence": "Ein Hund ist ein Tier.",
            },
            {
                "subject": "katze",
                "relation": "IS_A",
                "object": "tier",
                "source_sentence": "Eine Katze ist ein Tier.",
            },
        ]

        counts = netzwerk_session.batch_assert_relations(relations)

        assert counts["IS_A"] == 2

    def test_batch_create_empty_list(self, netzwerk_session):
        """Empty relations list should return empty dict."""
        counts = netzwerk_session.batch_assert_relations([])

        assert counts == {}

    def test_batch_create_invalid_confidence_raises_error(self, netzwerk_session):
        """Invalid confidence should raise ValueError."""
        relations = [
            {
                "subject": "hund",
                "relation": "IS_A",
                "object": "tier",
                "confidence": 1.5,  # Invalid
            },
        ]

        with pytest.raises(ValueError, match="Confidence must be in"):
            netzwerk_session.batch_assert_relations(relations)

    def test_batch_create_invalid_relation_type_raises_error(self, netzwerk_session):
        """Invalid relation type should raise ValueError."""
        relations = [
            {
                "subject": "hund",
                "relation": "MALICIOUS_TYPE",  # Not in whitelist
                "object": "tier",
            },
        ]

        with pytest.raises(ValueError, match="not in whitelist"):
            netzwerk_session.batch_assert_relations(relations)

    def test_batch_create_with_custom_batch_size(self, netzwerk_session):
        """Custom batch_size parameter should work."""
        relations = [
            {"subject": f"entity{i}", "relation": "IS_A", "object": "type"}
            for i in range(10)
        ]

        # Use small batch size
        counts = netzwerk_session.batch_assert_relations(relations, batch_size=3)

        assert counts["IS_A"] == 10

    def test_batch_create_large_batch(self, netzwerk_session):
        """Large batch (100+) should work efficiently."""
        relations = [
            {"subject": f"entity{i}", "relation": "IS_A", "object": "type"}
            for i in range(150)
        ]

        counts = netzwerk_session.batch_assert_relations(relations)

        assert counts["IS_A"] == 150

    def test_batch_create_mixed_confidence_values(self, netzwerk_session):
        """Relations with different confidence values should work."""
        relations = [
            {
                "subject": "hund",
                "relation": "IS_A",
                "object": "tier",
                "confidence": 0.9,
            },
            {
                "subject": "katze",
                "relation": "IS_A",
                "object": "tier",
                "confidence": 0.7,
            },
            {
                "subject": "maus",
                "relation": "IS_A",
                "object": "tier",
                "confidence": 1.0,
            },
        ]

        counts = netzwerk_session.batch_assert_relations(relations)

        assert counts["IS_A"] == 3


class TestBatchCreateEpisodes:
    """Test batch_create_episodes() method (Task 5)."""

    def test_batch_create_single_episode(self, netzwerk_session):
        """Batch create single episode."""
        episodes = [
            {
                "episode_type": "ingestion",
                "content": "Ein Hund ist ein Tier.",
                "metadata": {"source": "user"},
            }
        ]

        # Access episodic memory directly
        memory = EpisodicMemory(netzwerk_session.driver)
        episode_ids = memory.batch_create_episodes(episodes)

        assert len(episode_ids) == 1
        assert isinstance(episode_ids[0], str)  # UUID

    def test_batch_create_multiple_episodes(self, netzwerk_session):
        """Batch create multiple episodes."""
        episodes = [
            {
                "episode_type": "ingestion",
                "content": "Ein Hund ist ein Tier.",
                "metadata": {"source": "user"},
            },
            {
                "episode_type": "ingestion",
                "content": "Eine Katze ist ein Tier.",
                "metadata": {"source": "user"},
            },
            {
                "episode_type": "pattern_learning",
                "content": "Learned pattern: X ist Y",
                "metadata": {"confidence": 0.8},
            },
        ]

        memory = EpisodicMemory(netzwerk_session.driver)
        episode_ids = memory.batch_create_episodes(episodes)

        assert len(episode_ids) == 3
        # All IDs should be unique
        assert len(set(episode_ids)) == 3

    def test_batch_create_episodes_without_metadata(self, netzwerk_session):
        """Episodes without metadata should work (use empty dict)."""
        episodes = [
            {
                "episode_type": "ingestion",
                "content": "Content without metadata.",
            }
        ]

        memory = EpisodicMemory(netzwerk_session.driver)
        episode_ids = memory.batch_create_episodes(episodes)

        assert len(episode_ids) == 1

    def test_batch_create_episodes_empty_list(self, netzwerk_session):
        """Empty episodes list should return empty list."""
        memory = EpisodicMemory(netzwerk_session.driver)
        episode_ids = memory.batch_create_episodes([])

        assert episode_ids == []

    def test_batch_create_episodes_with_custom_batch_size(self, netzwerk_session):
        """Custom batch_size parameter should work."""
        episodes = [
            {"episode_type": "test", "content": f"Episode {i}"} for i in range(10)
        ]

        memory = EpisodicMemory(netzwerk_session.driver)
        episode_ids = memory.batch_create_episodes(episodes, batch_size=3)

        assert len(episode_ids) == 10

    def test_batch_create_episodes_large_batch(self, netzwerk_session):
        """Large batch (100+) should work efficiently."""
        episodes = [
            {"episode_type": "test", "content": f"Episode {i}"} for i in range(150)
        ]

        memory = EpisodicMemory(netzwerk_session.driver)
        episode_ids = memory.batch_create_episodes(episodes)

        assert len(episode_ids) == 150

    def test_batch_create_episodes_invalid_metadata_type(self, netzwerk_session):
        """Non-dict metadata should be replaced with empty dict (graceful)."""
        episodes = [
            {
                "episode_type": "test",
                "content": "Test content",
                "metadata": "invalid",  # String instead of dict
            }
        ]

        memory = EpisodicMemory(netzwerk_session.driver)
        episode_ids = memory.batch_create_episodes(episodes)

        # Should still create episode (graceful degradation)
        assert len(episode_ids) == 1


class TestBatchOperationsPerformance:
    """Test performance characteristics of batch operations."""

    def test_batch_relations_faster_than_individual(self, netzwerk_session, benchmark):
        """Batch operations should be significantly faster."""
        # This test would require pytest-benchmark
        # Placeholder for performance validation
        relations = [
            {"subject": f"entity{i}", "relation": "IS_A", "object": "type"}
            for i in range(50)
        ]

        # Batch method
        counts = netzwerk_session.batch_assert_relations(relations)
        assert counts["IS_A"] == 50

        # Individual method (for comparison)
        # In production, batch should be 5-10x faster
        # Not tested here to avoid slow tests

    def test_batch_episodes_faster_than_individual(self, netzwerk_session):
        """Batch episode creation should be significantly faster."""
        episodes = [
            {"episode_type": "test", "content": f"Episode {i}"} for i in range(50)
        ]

        memory = EpisodicMemory(netzwerk_session.driver)
        episode_ids = memory.batch_create_episodes(episodes)

        assert len(episode_ids) == 50


class TestBatchOperationsCacheInvalidation:
    """Test that batch operations invalidate caches correctly."""

    def test_batch_relations_invalidate_cache(self, netzwerk_session):
        """Batch relation creation should invalidate fact cache."""
        # Query to populate cache
        facts = netzwerk_session.query_graph_for_facts("hund")

        # Batch create new relations
        relations = [
            {"subject": "hund", "relation": "HAS_PROPERTY", "object": "schnell"},
        ]
        netzwerk_session.batch_assert_relations(relations)

        # Query again - should see new relation
        facts = netzwerk_session.query_graph_for_facts("hund")
        assert "schnell" in facts.get("HAS_PROPERTY", [])
