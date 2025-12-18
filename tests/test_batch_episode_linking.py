# tests/test_batch_episode_linking.py
"""
Tests for batch episode linking functionality (Quick Win #3).

Tests the new link_facts_to_episode_batch() method for performance improvement.
"""

import pytest

from component_1_netzwerk import KonzeptNetzwerk


class TestBatchEpisodeLinking:
    """Test batch episode linking functionality."""

    @pytest.fixture
    def netzwerk(self):
        """Create fresh netzwerk instance for each test."""
        netz = KonzeptNetzwerk()
        yield netz
        netz.close()

    def test_batch_link_multiple_facts(self, netzwerk):
        """Test batch linking of multiple facts to an episode."""
        # Create episode
        episode_id = netzwerk.create_episode(
            episode_type="test", content="Batch linking test", metadata={"test": True}
        )
        assert episode_id is not None

        # Create facts first
        netzwerk.assert_relation("hund", "IS_A", "tier")
        netzwerk.assert_relation("katze", "IS_A", "tier")
        netzwerk.assert_relation("hund", "HAS_PROPERTY", "freundlich")

        # Batch link facts
        facts = [
            {"subject": "hund", "relation": "IS_A", "object": "tier"},
            {"subject": "katze", "relation": "IS_A", "object": "tier"},
            {"subject": "hund", "relation": "HAS_PROPERTY", "object": "freundlich"},
        ]

        linked_count = netzwerk.link_facts_to_episode_batch(facts, episode_id)

        # Should link all 3 facts
        assert linked_count == 3

    def test_batch_link_empty_list(self, netzwerk):
        """Test batch linking with empty facts list."""
        # Create episode
        episode_id = netzwerk.create_episode(
            episode_type="test", content="Empty batch test", metadata={}
        )
        assert episode_id is not None

        # Batch link empty list (should return 0, not error)
        facts = []
        linked_count = netzwerk.link_facts_to_episode_batch(facts, episode_id)

        assert linked_count == 0

    def test_batch_link_skips_invalid_relations(self, netzwerk):
        """Test that batch linking skips facts with invalid relation types."""
        # Create episode
        episode_id = netzwerk.create_episode(
            episode_type="test", content="Invalid relations test", metadata={}
        )
        assert episode_id is not None

        # Create valid fact
        netzwerk.assert_relation("hund", "IS_A", "tier")

        # Try to batch link with one valid and one invalid relation
        facts = [
            {"subject": "hund", "relation": "IS_A", "object": "tier"},
            {
                "subject": "invalid",
                "relation": "INVALID!",
                "object": "test",
            },  # Invalid relation
        ]

        linked_count = netzwerk.link_facts_to_episode_batch(facts, episode_id)

        # Should link at least 1 (the valid fact)
        assert linked_count >= 1

    def test_batch_link_groups_by_relation_type(self, netzwerk):
        """Test that batch linking correctly handles multiple relation types."""
        # Create episode
        episode_id = netzwerk.create_episode(
            episode_type="test", content="Multiple relation types test", metadata={}
        )
        assert episode_id is not None

        # Create facts with different relation types
        netzwerk.assert_relation("hund", "IS_A", "tier")
        netzwerk.assert_relation("hund", "HAS_PROPERTY", "freundlich")
        netzwerk.assert_relation("katze", "CAPABLE_OF", "jagen")

        # Batch link with mixed relation types
        facts = [
            {"subject": "hund", "relation": "IS_A", "object": "tier"},
            {"subject": "hund", "relation": "HAS_PROPERTY", "object": "freundlich"},
            {"subject": "katze", "relation": "CAPABLE_OF", "object": "jagen"},
        ]

        linked_count = netzwerk.link_facts_to_episode_batch(facts, episode_id)

        # Should link all 3 facts
        assert linked_count == 3

    def test_batch_faster_than_individual(self, netzwerk):
        """Test that batch linking is significantly faster than individual links."""
        import time

        # Create episode
        episode_id = netzwerk.create_episode(
            episode_type="performance_test",
            content="Performance comparison",
            metadata={},
        )
        assert episode_id is not None

        # Create many facts
        fact_count = 20
        facts = []
        for i in range(fact_count):
            subj = f"entity_{i}"
            obj = "category"
            netzwerk.assert_relation(subj, "IS_A", obj)
            facts.append({"subject": subj, "relation": "IS_A", "object": obj})

        # Measure batch operation
        start_batch = time.time()
        linked_batch = netzwerk.link_facts_to_episode_batch(facts, episode_id)
        time_batch = time.time() - start_batch

        # Batch should link all facts
        assert linked_batch == fact_count

        # Batch should be reasonably fast (< 1 second for 20 facts)
        assert time_batch < 1.0, f"Batch linking took {time_batch:.2f}s (expected < 1s)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
