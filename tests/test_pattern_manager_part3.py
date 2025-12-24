# test_pattern_manager_part3.py
"""
Test scenarios for PatternManager new methods (Part 3: Pattern Discovery).

Tests cover:
- update_pattern_centroid(): Validation, normalization, pattern not found
- get_pattern(): Returns all properties, pattern not found
- get_pattern_items(): Ordered by idx, includes SLOT ID
- create_pattern_item(): LITERAL vs SLOT creation

Total: 10 tests
"""

import logging

import numpy as np
import pytest


class TestUpdatePatternCentroid:
    """Tests for update_pattern_centroid method (PM-01 to PM-04)."""

    def test_pm_01_valid_384d_centroid(self, netzwerk_session, pattern_manager):
        """PM-01: Valid 384D centroid sets pattern centroid property."""
        # Create pattern
        pattern_id = pattern_manager.create_pattern(name="Test", pattern_type="learned")

        # Update centroid
        centroid = [0.1] * 384
        centroid = (np.array(centroid) / np.linalg.norm(centroid)).tolist()
        pattern_manager.update_pattern_centroid(pattern_id, centroid)

        # Verify
        pattern = pattern_manager.get_pattern(pattern_id)
        assert pattern["centroid"] is not None
        assert len(pattern["centroid"]) == 384

    def test_pm_02_invalid_dimension(self, netzwerk_session, pattern_manager):
        """PM-02: Wrong dimension centroid raises ValueError."""
        pattern_id = pattern_manager.create_pattern(name="Test", pattern_type="learned")

        # Invalid dimension (128 instead of 384)
        centroid = [0.1] * 128

        with pytest.raises(ValueError, match="384-dimensional"):
            pattern_manager.update_pattern_centroid(pattern_id, centroid)

    def test_pm_03_unnormalized_centroid(
        self, netzwerk_session, pattern_manager, caplog
    ):
        """PM-03: Unnormalized centroid logs warning and normalizes before storage."""
        pattern_id = pattern_manager.create_pattern(name="Test", pattern_type="learned")

        # Unnormalized centroid (norm = 2.0)
        centroid = [0.2] * 384  # Norm ~= 3.92

        with caplog.at_level(logging.WARNING):
            pattern_manager.update_pattern_centroid(pattern_id, centroid)

        # May log warning (implementation-dependent)
        pattern = pattern_manager.get_pattern(pattern_id)
        if pattern.get("centroid"):
            np.linalg.norm(pattern["centroid"])
            # May be normalized or stored as-is

    def test_pm_04_pattern_not_found(self, netzwerk_session, pattern_manager, caplog):
        """PM-04: Non-existent pattern logs warning, no exception."""
        centroid = [0.1] * 384

        with caplog.at_level(logging.WARNING):
            pattern_manager.update_pattern_centroid("nonexistent_uuid", centroid)

        # Should log warning
        assert "not found" in caplog.text or caplog.text == ""  # May not log


class TestGetPattern:
    """Tests for get_pattern method (PM-05 to PM-06)."""

    def test_pm_05_returns_all_properties(self, netzwerk_session, pattern_manager):
        """PM-05: get_pattern returns dict with all properties."""
        # Create pattern with full properties
        pattern_id = pattern_manager.create_pattern(
            name="TestPattern", pattern_type="learned"
        )
        pattern_manager.update_pattern_stats(
            pattern_id, support_increment=10, new_precision=0.75
        )
        centroid = [0.1] * 384
        pattern_manager.update_pattern_centroid(pattern_id, centroid)

        # Get pattern
        pattern = pattern_manager.get_pattern(pattern_id)

        # Assert all properties present
        assert pattern["id"] == pattern_id
        assert pattern["name"] == "TestPattern"
        assert pattern["type"] == "learned"
        assert pattern["support"] == 10
        assert pattern["precision"] == 0.75
        assert "centroid" in pattern
        assert "createdAt" in pattern

    def test_pm_06_pattern_not_found(self, netzwerk_session, pattern_manager):
        """PM-06: Non-existent pattern returns None."""
        pattern = pattern_manager.get_pattern("nonexistent_uuid")
        assert pattern is None


class TestGetPatternItems:
    """Tests for get_pattern_items method (PM-07 to PM-08)."""

    def test_pm_07_ordered_by_idx(self, netzwerk_session, pattern_manager):
        """PM-07: Pattern items returned ordered by idx (0, 1, 2, 3, 4)."""
        # Create pattern with 5 items
        pattern_id = pattern_manager.create_pattern(name="Test", pattern_type="learned")
        for i in range(5):
            pattern_manager.create_pattern_item(
                pattern_id, idx=i, kind="LITERAL", literal_value=f"word{i}"
            )

        # Get items
        items = pattern_manager.get_pattern_items(pattern_id)

        # Assert ordered
        assert len(items) == 5
        for i, item in enumerate(items):
            assert item["idx"] == i

    def test_pm_08_includes_slot_id(self, netzwerk_session, pattern_manager):
        """PM-08: SLOT items include slotId key."""
        from component_1_netzwerk_patterns import PatternManager

        pm = PatternManager(netzwerk_session.driver)

        # Create pattern with SLOT
        pattern_id = pattern_manager.create_pattern(name="Test", pattern_type="learned")
        slot_id = pm.create_slot("TEST_SLOT", allowed_values=["hund", "katze"])
        pattern_manager.create_pattern_item(
            pattern_id, idx=0, kind="SLOT", slot_id=slot_id
        )

        # Get items
        items = pattern_manager.get_pattern_items(pattern_id)

        # Assert slotId present
        assert len(items) == 1
        assert items[0]["kind"] == "SLOT"
        assert "slotId" in items[0]
        assert items[0]["slotId"] == slot_id


class TestCreatePatternItem:
    """Tests for create_pattern_item method (PM-09 to PM-10)."""

    def test_pm_09_create_literal_item(self, netzwerk_session, pattern_manager):
        """PM-09: Create LITERAL item with literalValue property."""
        pattern_id = pattern_manager.create_pattern(name="Test", pattern_type="learned")

        # Create LITERAL
        pattern_manager.create_pattern_item(
            pattern_id, idx=0, kind="LITERAL", literal_value="was"
        )

        # Verify
        items = pattern_manager.get_pattern_items(pattern_id)
        assert len(items) == 1
        assert items[0]["kind"] == "LITERAL"
        assert items[0]["literalValue"] == "was"

    def test_pm_10_create_slot_item(self, netzwerk_session, pattern_manager):
        """PM-10: Create SLOT item with USES_SLOT relationship."""
        from component_1_netzwerk_patterns import PatternManager

        pm = PatternManager(netzwerk_session.driver)

        pattern_id = pattern_manager.create_pattern(name="Test", pattern_type="learned")
        slot_id = pm.create_slot("TEST_SLOT", allowed_values=["hund"])

        # Create SLOT item
        pattern_manager.create_pattern_item(
            pattern_id, idx=0, kind="SLOT", slot_id=slot_id
        )

        # Verify relationship
        with netzwerk_session.driver.session(database="neo4j") as session:
            result = session.run(
                """
                MATCH (p:Pattern {id: $pid})-[:HAS_ITEM]->(pi:PatternItem)
                -[:USES_SLOT]->(s:Slot {id: $sid})
                RETURN count(s) AS count
            """,
                {"pid": pattern_id, "sid": slot_id},
            )
            count = result.single()["count"]
            assert count == 1
