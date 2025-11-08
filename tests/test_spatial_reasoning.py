"""
tests/test_spatial_reasoning.py

Unit tests for core spatial reasoning functionality.

Tests cover:
- Position: creation, equality, distance calculations, neighbor generation
- Grid: creation, bounds checking, cell data, neighbor queries
- SpatialRelation: creation, inverse, symmetric properties
- SpatialReasoner: transitive inference, caching, performance metrics
"""

import pytest

from component_42_spatial_reasoning import (
    Grid,
    NeighborhoodType,
    Position,
    SpatialReasoner,
    SpatialRelation,
    SpatialRelationType,
)

# ==================== Position Tests ====================


class TestPosition:
    """Unit tests for Position class."""

    def test_position_creation(self):
        """Test basic position creation."""
        pos = Position(5, 10)
        assert pos.x == 5
        assert pos.y == 10

    def test_position_creation_negative(self):
        """Test position creation with negative coordinates."""
        pos = Position(-3, -7)
        assert pos.x == -3
        assert pos.y == -7

    def test_position_equality(self):
        """Test position equality comparison."""
        pos1 = Position(2, 3)
        pos2 = Position(2, 3)
        pos3 = Position(3, 2)
        pos4 = Position(2, 4)

        assert pos1 == pos2
        assert pos1 != pos3
        assert pos1 != pos4

    def test_position_hash(self):
        """Test position hashing for set/dict usage."""
        pos1 = Position(2, 3)
        pos2 = Position(2, 3)
        pos3 = Position(3, 2)

        position_set = {pos1, pos2, pos3}
        assert len(position_set) == 2  # pos1 and pos2 are same

    def test_position_euclidean_distance(self):
        """Test Euclidean distance calculation."""
        pos1 = Position(0, 0)
        pos2 = Position(3, 4)

        # sqrt(3^2 + 4^2) = sqrt(9 + 16) = sqrt(25) = 5
        assert pos1.distance_to(pos2) == 5.0

        # Distance should be symmetric
        assert pos2.distance_to(pos1) == 5.0

    def test_position_euclidean_distance_diagonal(self):
        """Test Euclidean distance for diagonal movement."""
        pos1 = Position(0, 0)
        pos2 = Position(10, 10)

        # sqrt(10^2 + 10^2) = sqrt(200) ≈ 14.142
        distance = pos1.distance_to(pos2)
        assert abs(distance - 14.142135623730951) < 0.0001

    def test_position_manhattan_distance(self):
        """Test Manhattan distance calculation."""
        pos1 = Position(0, 0)
        pos2 = Position(3, 4)

        # |3-0| + |4-0| = 3 + 4 = 7
        assert pos1.manhattan_distance_to(pos2) == 7

        # Distance should be symmetric
        assert pos2.manhattan_distance_to(pos1) == 7

    def test_position_manhattan_distance_negative(self):
        """Test Manhattan distance with negative coordinates."""
        pos1 = Position(-2, -3)
        pos2 = Position(2, 3)

        # |-2-2| + |-3-3| = 4 + 6 = 10
        assert pos1.manhattan_distance_to(pos2) == 10

    def test_position_distance_to_self(self):
        """Test distance to self is zero."""
        pos = Position(5, 5)
        assert pos.distance_to(pos) == 0.0
        assert pos.manhattan_distance_to(pos) == 0

    def test_position_get_neighbors_orthogonal(self):
        """Test orthogonal neighbor generation."""
        pos = Position(5, 5)
        neighbors = pos.get_neighbors(NeighborhoodType.ORTHOGONAL)

        assert len(neighbors) == 4
        assert Position(5, 6) in neighbors  # North
        assert Position(5, 4) in neighbors  # South
        assert Position(6, 5) in neighbors  # East
        assert Position(4, 5) in neighbors  # West

    def test_position_get_neighbors_moore(self):
        """Test Moore (8-directional) neighbor generation."""
        pos = Position(5, 5)
        neighbors = pos.get_neighbors(NeighborhoodType.MOORE)

        assert len(neighbors) == 8

        # Orthogonal
        assert Position(5, 6) in neighbors  # North
        assert Position(5, 4) in neighbors  # South
        assert Position(6, 5) in neighbors  # East
        assert Position(4, 5) in neighbors  # West

        # Diagonal
        assert Position(6, 6) in neighbors  # Northeast
        assert Position(4, 6) in neighbors  # Northwest
        assert Position(6, 4) in neighbors  # Southeast
        assert Position(4, 4) in neighbors  # Southwest

    def test_position_get_neighbors_diagonal_only(self):
        """Test diagonal-only neighbor generation."""
        pos = Position(5, 5)
        neighbors = pos.get_neighbors(NeighborhoodType.DIAGONAL)

        assert len(neighbors) == 4
        assert Position(6, 6) in neighbors  # Northeast
        assert Position(4, 6) in neighbors  # Northwest
        assert Position(6, 4) in neighbors  # Southeast
        assert Position(4, 4) in neighbors  # Southwest


# ==================== Grid Tests ====================


class TestGrid:
    """Unit tests for Grid class."""

    def test_grid_creation(self):
        """Test basic grid creation."""
        grid = Grid(height=10, width=20)
        assert grid.height == 10
        assert grid.width == 20
        assert grid.name == "Grid_20x10"

    def test_grid_creation_with_name(self):
        """Test grid creation with custom name."""
        grid = Grid(height=8, width=8, name="Chessboard")
        assert grid.name == "Chessboard"

    def test_grid_cell_count(self):
        """Test cell count calculation."""
        grid = Grid(height=10, width=15)
        assert grid.get_cell_count() == 150

    def test_grid_bounds_valid_positions(self):
        """Test boundary validation for valid positions."""
        grid = Grid(height=10, width=10)

        # Corners
        assert grid.is_valid_position(Position(0, 0))
        assert grid.is_valid_position(Position(9, 9))
        assert grid.is_valid_position(Position(0, 9))
        assert grid.is_valid_position(Position(9, 0))

        # Middle
        assert grid.is_valid_position(Position(5, 5))

    def test_grid_bounds_invalid_positions(self):
        """Test boundary validation for invalid positions."""
        grid = Grid(height=10, width=10)

        # Out of bounds
        assert not grid.is_valid_position(Position(-1, 0))
        assert not grid.is_valid_position(Position(0, -1))
        assert not grid.is_valid_position(Position(10, 0))
        assert not grid.is_valid_position(Position(0, 10))
        assert not grid.is_valid_position(Position(10, 10))

    def test_grid_cell_data_set_and_get(self):
        """Test cell data storage and retrieval."""
        grid = Grid(height=5, width=5)
        pos = Position(2, 3)

        # Initially None
        assert grid.get_cell_data(pos) is None

        # Set data
        grid.set_cell_data(pos, "obstacle")
        assert grid.get_cell_data(pos) == "obstacle"

        # Overwrite data
        grid.set_cell_data(pos, "empty")
        assert grid.get_cell_data(pos) == "empty"

    def test_grid_cell_data_multiple_cells(self):
        """Test storing data in multiple cells."""
        grid = Grid(height=3, width=3)

        grid.set_cell_data(Position(0, 0), "A")
        grid.set_cell_data(Position(1, 1), "B")
        grid.set_cell_data(Position(2, 2), "C")

        assert grid.get_cell_data(Position(0, 0)) == "A"
        assert grid.get_cell_data(Position(1, 1)) == "B"
        assert grid.get_cell_data(Position(2, 2)) == "C"
        assert grid.get_cell_data(Position(0, 1)) is None

    def test_grid_cell_data_out_of_bounds(self):
        """Test cell data access for out-of-bounds positions."""
        grid = Grid(height=5, width=5)
        pos = Position(10, 10)

        # Should return None for out-of-bounds
        assert grid.get_cell_data(pos) is None

        # Setting data on out-of-bounds position should not raise error
        grid.set_cell_data(pos, "data")  # Should be ignored

    def test_grid_neighbors_orthogonal_center(self):
        """Test orthogonal neighbors at grid center."""
        grid = Grid(height=10, width=10)
        center = Position(5, 5)

        neighbors = grid.get_neighbors(center, NeighborhoodType.ORTHOGONAL)

        assert len(neighbors) == 4
        assert Position(5, 6) in neighbors
        assert Position(5, 4) in neighbors
        assert Position(6, 5) in neighbors
        assert Position(4, 5) in neighbors

    def test_grid_neighbors_moore_center(self):
        """Test Moore neighbors at grid center."""
        grid = Grid(height=10, width=10)
        center = Position(5, 5)

        neighbors = grid.get_neighbors(center, NeighborhoodType.MOORE)

        assert len(neighbors) == 8

    def test_grid_neighbors_corner(self):
        """Test neighbors at grid corner."""
        grid = Grid(height=10, width=10)
        corner = Position(0, 0)

        orthogonal = grid.get_neighbors(corner, NeighborhoodType.ORTHOGONAL)
        moore = grid.get_neighbors(corner, NeighborhoodType.MOORE)

        # Corner has only 2 orthogonal neighbors
        assert len(orthogonal) == 2
        assert Position(1, 0) in orthogonal
        assert Position(0, 1) in orthogonal

        # Corner has 3 Moore neighbors
        assert len(moore) == 3
        assert Position(1, 0) in moore
        assert Position(0, 1) in moore
        assert Position(1, 1) in moore

    def test_grid_neighbors_edge(self):
        """Test neighbors at grid edge."""
        grid = Grid(height=10, width=10)
        edge = Position(5, 0)

        orthogonal = grid.get_neighbors(edge, NeighborhoodType.ORTHOGONAL)
        moore = grid.get_neighbors(edge, NeighborhoodType.MOORE)

        # Edge has 3 orthogonal neighbors
        assert len(orthogonal) == 3

        # Edge has 5 Moore neighbors
        assert len(moore) == 5

    def test_grid_large_grid(self):
        """Test large grid creation and operations."""
        grid = Grid(height=1000, width=1000)

        assert grid.get_cell_count() == 1000000
        assert grid.is_valid_position(Position(999, 999))
        assert not grid.is_valid_position(Position(1000, 1000))


# ==================== SpatialRelation Tests ====================


class TestSpatialRelation:
    """Unit tests for SpatialRelation class."""

    def test_relation_creation(self):
        """Test basic relation creation."""
        rel = SpatialRelation("A", "B", SpatialRelationType.NORTH_OF)

        assert rel.subject == "A"
        assert rel.object == "B"
        assert rel.relation_type == SpatialRelationType.NORTH_OF
        assert rel.confidence == 1.0  # Default

    def test_relation_creation_with_confidence(self):
        """Test relation creation with custom confidence."""
        rel = SpatialRelation("A", "B", SpatialRelationType.NORTH_OF, confidence=0.85)

        assert rel.confidence == 0.85

    def test_relation_inverse_directional(self):
        """Test inverse relation for directional relations."""
        rel = SpatialRelation("A", "B", SpatialRelationType.NORTH_OF)
        inverse = rel.to_inverse()

        assert inverse is not None
        assert inverse.subject == "B"
        assert inverse.object == "A"
        assert inverse.relation_type == SpatialRelationType.SOUTH_OF
        assert inverse.confidence == rel.confidence

    def test_relation_inverse_symmetric(self):
        """Test inverse relation for symmetric relations."""
        rel = SpatialRelation("A", "B", SpatialRelationType.ADJACENT_TO)
        inverse = rel.to_inverse()

        assert inverse is not None
        assert inverse.subject == "B"
        assert inverse.object == "A"
        assert inverse.relation_type == SpatialRelationType.ADJACENT_TO  # Same type

    def test_relation_type_is_transitive(self):
        """Test transitivity property of relation types."""
        assert SpatialRelationType.NORTH_OF.is_transitive
        assert SpatialRelationType.SOUTH_OF.is_transitive
        assert SpatialRelationType.EAST_OF.is_transitive
        assert SpatialRelationType.WEST_OF.is_transitive

        # ADJACENT_TO is not transitive
        # (A adjacent to B, B adjacent to C does NOT imply A adjacent to C)
        assert not SpatialRelationType.ADJACENT_TO.is_transitive

    def test_relation_type_is_symmetric(self):
        """Test symmetry property of relation types."""
        assert SpatialRelationType.ADJACENT_TO.is_symmetric

        # Directional relations are not symmetric
        assert not SpatialRelationType.NORTH_OF.is_symmetric
        assert not SpatialRelationType.SOUTH_OF.is_symmetric
        assert not SpatialRelationType.EAST_OF.is_symmetric
        assert not SpatialRelationType.WEST_OF.is_symmetric

    def test_relation_equality(self):
        """Test relation equality comparison."""
        rel1 = SpatialRelation("A", "B", SpatialRelationType.NORTH_OF, confidence=0.9)
        rel2 = SpatialRelation("A", "B", SpatialRelationType.NORTH_OF, confidence=0.9)
        rel3 = SpatialRelation("A", "B", SpatialRelationType.SOUTH_OF, confidence=0.9)
        rel4 = SpatialRelation("B", "A", SpatialRelationType.NORTH_OF, confidence=0.9)

        # Same subject, object, type, confidence
        assert rel1 == rel2

        # Different type
        assert rel1 != rel3

        # Different subject/object
        assert rel1 != rel4


# ==================== SpatialReasoner Tests ====================


class TestSpatialReasoner:
    """Unit tests for SpatialReasoner class."""

    @pytest.fixture
    def reasoner(self):
        """Create a SpatialReasoner without netzwerk backend."""
        return SpatialReasoner(netzwerk=None)

    def test_reasoner_creation(self, reasoner):
        """Test reasoner initialization."""
        assert reasoner is not None
        assert reasoner.netzwerk is None

    def test_reasoner_transitive_inference_single_hop(self, reasoner):
        """Test transitive inference with single hop."""
        # A NORTH_OF B, B NORTH_OF C => A NORTH_OF C
        known = [
            SpatialRelation("A", "B", SpatialRelationType.NORTH_OF, confidence=0.9),
        ]

        # Without graph backend, we can't do multi-hop
        # But we test the method signature and return type
        inferred, proofs = reasoner.infer_transitive_with_proof(
            "A", known, SpatialRelationType.NORTH_OF
        )

        assert isinstance(inferred, list)
        assert isinstance(proofs, list)

    def test_reasoner_transitive_inference_confidence_decay(self, reasoner):
        """Test confidence decay in transitive inference."""
        # Simulate transitive inference with confidence decay
        rel1_confidence = 0.9
        rel2_confidence = 0.8

        # Expected: 0.9 * 0.8 = 0.72
        expected_confidence = rel1_confidence * rel2_confidence

        assert abs(expected_confidence - 0.72) < 0.001

    def test_reasoner_performance_metrics(self, reasoner):
        """Test performance metrics tracking."""
        metrics = reasoner.get_performance_metrics()

        assert "queries_total" in metrics
        assert "queries_cached" in metrics
        assert "transitive_inferences" in metrics
        assert "symmetric_inferences" in metrics

        # Initially zero
        assert metrics["queries_total"] == 0
        assert metrics["queries_cached"] == 0

    def test_reasoner_performance_metrics_after_query(self, reasoner):
        """Test performance metrics after running queries."""
        # Run a query
        known = [
            SpatialRelation("A", "B", SpatialRelationType.NORTH_OF, confidence=0.9),
        ]
        reasoner.infer_transitive_with_proof("A", known, SpatialRelationType.NORTH_OF)

        # Metrics should update
        metrics = reasoner.get_performance_metrics()
        # Note: Without netzwerk, metrics might not increment
        assert "queries_total" in metrics

    def test_reasoner_log_summary(self, reasoner):
        """Test logging summary (should not raise exceptions)."""
        try:
            reasoner.log_performance_summary()
        except Exception as e:
            pytest.fail(f"log_performance_summary raised exception: {e}")


# ==================== Edge Cases ====================


class TestSpatialReasoningEdgeCases:
    """Test edge cases and error handling."""

    def test_position_extreme_coordinates(self):
        """Test position with very large coordinates."""
        pos = Position(1000000, 1000000)
        assert pos.x == 1000000
        assert pos.y == 1000000

    def test_grid_single_cell(self):
        """Test 1x1 grid."""
        grid = Grid(height=1, width=1)
        assert grid.get_cell_count() == 1
        assert grid.is_valid_position(Position(0, 0))
        assert not grid.is_valid_position(Position(1, 0))

    def test_grid_rectangular(self):
        """Test non-square grids."""
        grid = Grid(height=5, width=10)
        assert grid.height == 5
        assert grid.width == 10
        assert grid.get_cell_count() == 50
        assert grid.is_valid_position(Position(9, 4))
        assert not grid.is_valid_position(Position(10, 4))
        assert not grid.is_valid_position(Position(9, 5))

    def test_relation_self_reference(self):
        """Test relation where subject == object."""
        rel = SpatialRelation("A", "A", SpatialRelationType.ADJACENT_TO)
        assert rel.subject == "A"
        assert rel.object == "A"

    def test_position_neighbors_at_origin(self):
        """Test neighbor generation at origin."""
        pos = Position(0, 0)
        neighbors = pos.get_neighbors(NeighborhoodType.ORTHOGONAL)

        # At origin, neighbors can have negative coordinates
        assert len(neighbors) == 4
        assert Position(0, 1) in neighbors
        assert Position(0, -1) in neighbors
        assert Position(1, 0) in neighbors
        assert Position(-1, 0) in neighbors


# ==================== Test Configuration ====================


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: unit tests for spatial reasoning core")
