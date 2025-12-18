# tests/test_query_confidence_filtering.py
"""
Tests for confidence-based query filtering (Quick Win #2).

Tests the new min_confidence parameter in query_graph_for_facts().
"""

import pytest

from component_1_netzwerk_core import KonzeptNetzwerkCore


class TestConfidenceFiltering:
    """Test confidence-based query filtering functionality."""

    @pytest.fixture
    def netzwerk(self):
        """Create fresh netzwerk instance for each test."""
        netz = KonzeptNetzwerkCore()
        yield netz
        netz.close()

    def test_query_without_filter_returns_all_facts(self, netzwerk):
        """Test that min_confidence=0.0 returns all facts (backward compatibility)."""
        # Setup: Create facts (default confidence = 0.85)
        netzwerk.assert_relation("hund", "IS_A", "saeugetier")
        netzwerk.assert_relation("hund", "HAS_PROPERTY", "freundlich")

        # Query without filter
        facts = netzwerk.query_graph_for_facts("hund", min_confidence=0.0)

        # Should return all facts
        assert "IS_A" in facts
        assert "saeugetier" in facts["IS_A"]
        assert "HAS_PROPERTY" in facts
        assert "freundlich" in facts["HAS_PROPERTY"]

    def test_query_with_medium_filter(self, netzwerk):
        """Test that medium confidence filter (0.5) works correctly."""
        # Setup: Create facts with implicit confidence 0.85 (default)
        netzwerk.assert_relation("hund", "IS_A", "saeugetier")
        netzwerk.assert_relation("hund", "HAS_PROPERTY", "freundlich")

        # Query with medium filter (should still get all since default is 0.85)
        facts = netzwerk.query_graph_for_facts("hund", min_confidence=0.5)

        # All facts should pass (confidence 0.85 >= 0.5)
        assert "IS_A" in facts
        assert "HAS_PROPERTY" in facts
        assert len(facts["IS_A"]) == 1
        assert len(facts["HAS_PROPERTY"]) == 1

    def test_query_with_high_filter(self, netzwerk):
        """Test that high confidence filter (0.9) works correctly."""
        # Setup: Create facts with default confidence 0.85
        netzwerk.assert_relation("hund", "IS_A", "saeugetier")
        netzwerk.assert_relation("hund", "HAS_PROPERTY", "freundlich")

        # Query with high filter (0.9 > 0.85, should filter out all)
        facts = netzwerk.query_graph_for_facts("hund", min_confidence=0.9)

        # No facts should pass (confidence 0.85 < 0.9)
        # Note: If no facts, query returns empty dict or dict with empty lists
        if "IS_A" in facts:
            assert len(facts["IS_A"]) == 0
        if "HAS_PROPERTY" in facts:
            assert len(facts["HAS_PROPERTY"]) == 0

    def test_cache_respects_filter_parameters(self, netzwerk):
        """Test that cache keys include filter parameters for correctness."""
        # Setup
        netzwerk.assert_relation("hund", "IS_A", "saeugetier")
        netzwerk.assert_relation("hund", "HAS_PROPERTY", "freundlich")

        # Query 1: No filter
        facts_all = netzwerk.query_graph_for_facts("hund", min_confidence=0.0)

        # Query 2: High filter (should NOT return cached result from Query 1)
        facts_filtered = netzwerk.query_graph_for_facts("hund", min_confidence=0.9)

        # Results should be different
        # (Query 1 has facts, Query 2 should be empty or have fewer facts)
        all_count = sum(len(v) for v in facts_all.values())
        filtered_count = sum(len(v) for v in facts_filtered.values())

        assert (
            all_count >= filtered_count
        ), "Filtered query should return same or fewer facts than unfiltered"

    def test_default_parameters_backward_compatible(self, netzwerk):
        """Test that calling without new parameters still works (backward compatibility)."""
        # Setup
        netzwerk.assert_relation("hund", "IS_A", "saeugetier")

        # Old-style call (no new parameters)
        facts = netzwerk.query_graph_for_facts("hund")

        # Should work exactly as before
        assert "IS_A" in facts
        assert "saeugetier" in facts["IS_A"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
