"""
Component 42: Spatial Reasoning

Provides spatial reasoning capabilities for KAI, including:
- 2D grid-based representations (chess boards, Sudoku grids, custom N×M grids)
- Geometric shapes (triangles, quadrilaterals, circles) with properties and calculations
- Coordinate systems with position tracking and neighborhood logic (orthogonal/diagonal)
- Spatial relations and transitive reasoning

This component is designed to be general-purpose and domain-agnostic.
Specific applications (chess, Sudoku, etc.) are taught via rules, not hard-coded.

Author: KAI Development Team
Date: 2025-11-05
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from cachetools import TTLCache

from component_15_logging_config import get_logger

logger = get_logger(__name__)


# ============================================================================
# Enums and Constants
# ============================================================================


class SpatialRelationType(Enum):
    """Types of spatial relations supported by the reasoning engine."""

    # Directional relations (cardinal directions)
    NORTH_OF = "NORTH_OF"
    SOUTH_OF = "SOUTH_OF"
    EAST_OF = "EAST_OF"
    WEST_OF = "WEST_OF"

    # Adjacency relations
    ADJACENT_TO = "ADJACENT_TO"  # General neighbor (symmetric)
    NEIGHBOR_ORTHOGONAL = "NEIGHBOR_ORTHOGONAL"  # 4-directional neighbor
    NEIGHBOR_DIAGONAL = "NEIGHBOR_DIAGONAL"  # Diagonal neighbor

    # Hierarchical/containment relations
    INSIDE = "INSIDE"  # A is inside B
    CONTAINS = "CONTAINS"  # A contains B

    # Vertical relations
    ABOVE = "ABOVE"
    BELOW = "BELOW"

    # Positional relations
    BETWEEN = "BETWEEN"  # A is between B and C
    LOCATED_AT = "LOCATED_AT"  # Object is at specific position

    @property
    def is_symmetric(self) -> bool:
        """Check if this relation is symmetric (A R B => B R A)."""
        return self in {
            SpatialRelationType.ADJACENT_TO,
            SpatialRelationType.NEIGHBOR_ORTHOGONAL,
            SpatialRelationType.NEIGHBOR_DIAGONAL,
        }

    @property
    def is_transitive(self) -> bool:
        """Check if this relation is transitive (A R B, B R C => A R C)."""
        return self in {
            SpatialRelationType.NORTH_OF,
            SpatialRelationType.SOUTH_OF,
            SpatialRelationType.EAST_OF,
            SpatialRelationType.WEST_OF,
            SpatialRelationType.INSIDE,
            SpatialRelationType.CONTAINS,
            SpatialRelationType.ABOVE,
            SpatialRelationType.BELOW,
        }

    @property
    def inverse(self) -> Optional["SpatialRelationType"]:
        """Get the inverse relation (if exists)."""
        inverses = {
            SpatialRelationType.NORTH_OF: SpatialRelationType.SOUTH_OF,
            SpatialRelationType.SOUTH_OF: SpatialRelationType.NORTH_OF,
            SpatialRelationType.EAST_OF: SpatialRelationType.WEST_OF,
            SpatialRelationType.WEST_OF: SpatialRelationType.EAST_OF,
            SpatialRelationType.INSIDE: SpatialRelationType.CONTAINS,
            SpatialRelationType.CONTAINS: SpatialRelationType.INSIDE,
            SpatialRelationType.ABOVE: SpatialRelationType.BELOW,
            SpatialRelationType.BELOW: SpatialRelationType.ABOVE,
        }
        return inverses.get(self)


class NeighborhoodType(Enum):
    """Types of neighborhood definitions for grid-based reasoning."""

    ORTHOGONAL = "orthogonal"  # 4-directional (N, S, E, W)
    DIAGONAL = "diagonal"  # Diagonal only (NE, NW, SE, SW)
    MOORE = "moore"  # 8-directional (orthogonal + diagonal)
    CUSTOM = "custom"  # Custom neighborhood pattern (e.g., knight moves in chess)


# ============================================================================
# Data Structures
# ============================================================================


@dataclass(frozen=True, order=True)
class Position:
    """
    Represents a position in a 2D coordinate system.

    Immutable and hashable for use in sets and dictionaries.
    Coordinates are 0-indexed internally but can represent any coordinate system.
    """

    x: int
    y: int

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"

    def distance_to(self, other: "Position", metric: str = "euclidean") -> float:
        """
        Calculate distance to another position.

        Args:
            other: Target position
            metric: Distance metric ('euclidean', 'manhattan', 'chebyshev')

        Returns:
            Distance value
        """
        dx = abs(self.x - other.x)
        dy = abs(self.y - other.y)

        if metric == "manhattan":
            return dx + dy
        elif metric == "chebyshev":
            return max(dx, dy)
        else:  # euclidean
            return math.sqrt(dx**2 + dy**2)

    def manhattan_distance_to(self, other: "Position") -> float:
        """Calculate Manhattan distance to another position (convenience method)."""
        return self.distance_to(other, metric="manhattan")

    def euclidean_distance_to(self, other: "Position") -> float:
        """Calculate Euclidean distance to another position (convenience method)."""
        return self.distance_to(other, metric="euclidean")

    def direction_to(self, other: "Position") -> Optional[SpatialRelationType]:
        """
        Determine the cardinal direction from this position to another.

        Returns None if positions are diagonal or identical.
        """
        dx = other.x - self.x
        dy = other.y - self.y

        if dx == 0 and dy > 0:
            return SpatialRelationType.NORTH_OF
        elif dx == 0 and dy < 0:
            return SpatialRelationType.SOUTH_OF
        elif dx > 0 and dy == 0:
            return SpatialRelationType.EAST_OF
        elif dx < 0 and dy == 0:
            return SpatialRelationType.WEST_OF

        return None

    def get_neighbors(
        self,
        neighborhood_type: NeighborhoodType = NeighborhoodType.ORTHOGONAL,
        custom_offsets: Optional[List[Tuple[int, int]]] = None,
    ) -> List["Position"]:
        """
        Get neighboring positions based on neighborhood type.

        Args:
            neighborhood_type: Type of neighborhood
            custom_offsets: Custom offset list for CUSTOM neighborhood type
                           e.g., [(-2, -1), (-2, 1), ...] for knight moves

        Returns:
            List of neighboring positions
        """
        if neighborhood_type == NeighborhoodType.CUSTOM and custom_offsets:
            return [Position(self.x + dx, self.y + dy) for dx, dy in custom_offsets]

        # Standard offsets
        orthogonal = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # N, S, E, W
        diagonal = [(1, 1), (1, -1), (-1, 1), (-1, -1)]  # NE, SE, NE, NW

        if neighborhood_type == NeighborhoodType.ORTHOGONAL:
            offsets = orthogonal
        elif neighborhood_type == NeighborhoodType.DIAGONAL:
            offsets = diagonal
        else:  # MOORE
            offsets = orthogonal + diagonal

        return [Position(self.x + dx, self.y + dy) for dx, dy in offsets]


@dataclass
class SpatialRelation:
    """
    Represents a spatial relation between entities.

    Examples:
    - SpatialRelation("König", "Turm", ADJACENT_TO, 0.95)
    - SpatialRelation("Feld_A1", "Feld_A2", NORTH_OF, 1.0)
    """

    subject: str  # Entity A
    object: str  # Entity B
    relation_type: SpatialRelationType
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.subject} {self.relation_type.value} {self.object} (conf={self.confidence:.2f})"

    def to_inverse(self) -> Optional["SpatialRelation"]:
        """Create the inverse relation if one exists."""
        # For symmetric relations, return the same relation with swapped subject/object
        if self.relation_type.is_symmetric:
            return SpatialRelation(
                subject=self.object,
                object=self.subject,
                relation_type=self.relation_type,
                confidence=self.confidence,
                metadata=self.metadata.copy(),
            )

        # For directional relations, get the inverse type
        inverse_type = self.relation_type.inverse
        if inverse_type:
            return SpatialRelation(
                subject=self.object,
                object=self.subject,
                relation_type=inverse_type,
                confidence=self.confidence,
                metadata=self.metadata.copy(),
            )
        return None


@dataclass
class SpatialReasoningResult:
    """
    Result of a spatial reasoning query.

    Contains inferred facts, confidence scores, and reasoning trace.
    """

    query: str
    relations: List[SpatialRelation] = field(default_factory=list)
    confidence: float = 0.0
    reasoning_steps: List[str] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        """Check if reasoning was successful."""
        return self.error is None and len(self.relations) > 0


@dataclass
class Grid:
    """
    Represents a 2D grid structure.

    Grids are general-purpose and not tied to specific applications.
    They can represent chess boards, Sudoku grids, or any N×M structure.
    """

    width: int  # Number of columns (N)
    height: int  # Number of rows (M)
    name: str = ""  # Unique identifier (e.g., "Schachbrett_1", "Sudoku_9x9")
    neighborhood_type: NeighborhoodType = NeighborhoodType.ORTHOGONAL
    custom_offsets: Optional[List[Tuple[int, int]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate grid parameters."""
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                f"Grid dimensions must be positive: {self.width}×{self.height}"
            )

        if (
            self.neighborhood_type == NeighborhoodType.CUSTOM
            and not self.custom_offsets
        ):
            raise ValueError("Custom neighborhood type requires custom_offsets")

        # Generate default name if not provided
        if not self.name:
            object.__setattr__(self, "name", f"Grid_{self.width}x{self.height}")

        # Initialize cell data storage
        object.__setattr__(self, "_cell_data", {})

    @property
    def size(self) -> int:
        """Total number of positions in the grid."""
        return self.width * self.height

    def is_valid_position(self, pos: Position) -> bool:
        """Check if a position is within grid bounds."""
        return 0 <= pos.x < self.width and 0 <= pos.y < self.height

    def get_all_positions(self) -> List[Position]:
        """Get all positions in the grid."""
        return [Position(x, y) for x in range(self.width) for y in range(self.height)]

    def get_position_name(self, pos: Position) -> str:
        """
        Generate a unique name for a position in this grid.

        Format: "{grid_name}_Pos_{x}_{y}"
        Example: "Schachbrett_Pos_3_4"
        """
        return f"{self.name}_Pos_{pos.x}_{pos.y}"

    def get_cell_count(self) -> int:
        """Get total number of cells in the grid (alias for size property)."""
        return self.size

    def set_cell_data(self, pos: Position, data: Any) -> None:
        """Set data for a specific cell in the grid. Silently ignores out-of-bounds positions."""
        if not hasattr(self, "_cell_data"):
            object.__setattr__(self, "_cell_data", {})
        if not self.is_valid_position(pos):
            return  # Silently ignore out-of-bounds positions
        self._cell_data[pos] = data

    def get_cell_data(self, pos: Position, default: Any = None) -> Any:
        """Get data for a specific cell in the grid."""
        if not hasattr(self, "_cell_data"):
            object.__setattr__(self, "_cell_data", {})
        return self._cell_data.get(pos, default)

    def get_neighbors(
        self, pos: Position, neighborhood_type: Optional[NeighborhoodType] = None
    ) -> List[Position]:
        """
        Get valid neighbors of a position within grid bounds.

        Args:
            pos: Position to get neighbors for
            neighborhood_type: Type of neighborhood (defaults to grid's neighborhood_type)

        Returns:
            List of valid neighboring positions
        """
        nh_type = neighborhood_type or self.neighborhood_type
        neighbors = pos.get_neighbors(nh_type, self.custom_offsets)
        return [n for n in neighbors if self.is_valid_position(n)]

    def __str__(self) -> str:
        return f"Grid({self.name}, {self.width}×{self.height})"


@dataclass
class GeometricShape:
    """
    Base class for geometric shapes.

    All shapes can have:
    - Name/identifier
    - Properties (color, size, etc.)
    - Position on a grid (optional)
    """

    name: str
    shape_type: str = ""  # e.g., "Dreieck", "Viereck", "Kreis"
    properties: Dict[str, Any] = field(default_factory=dict)

    def calculate_area(self) -> Optional[float]:
        """Calculate the area of the shape. Override in subclasses."""
        return None

    def calculate_perimeter(self) -> Optional[float]:
        """Calculate the perimeter of the shape. Override in subclasses."""
        return None

    def __str__(self) -> str:
        return f"{self.shape_type}({self.name})"


@dataclass
class Polygon(GeometricShape):
    """
    Represents a polygon (multi-sided shape).

    Properties:
    - num_sides: Number of sides
    - vertices: List of vertex positions (optional)
    """

    num_sides: int = 0
    vertices: List[Position] = field(default_factory=list)

    def __post_init__(self):
        """Set default shape_type for polygons."""
        if not self.shape_type:
            self.shape_type = f"Polygon_{self.num_sides}"

        # Validate vertices match num_sides
        if self.vertices and len(self.vertices) != self.num_sides:
            raise ValueError(
                f"Number of vertices ({len(self.vertices)}) doesn't match num_sides ({self.num_sides})"
            )


@dataclass
class Triangle(Polygon):
    """Triangle (3-sided polygon)."""

    def __post_init__(self):
        self.num_sides = 3
        self.shape_type = "Dreieck"
        super().__post_init__()

    def calculate_area(self) -> Optional[float]:
        """Calculate area using Heron's formula if vertices are provided."""
        if len(self.vertices) != 3:
            return None

        # Calculate side lengths
        a = self.vertices[0].distance_to(self.vertices[1])
        b = self.vertices[1].distance_to(self.vertices[2])
        c = self.vertices[2].distance_to(self.vertices[0])

        # Heron's formula
        s = (a + b + c) / 2
        area = math.sqrt(s * (s - a) * (s - b) * (s - c))
        return area

    def calculate_perimeter(self) -> Optional[float]:
        """Calculate perimeter as sum of side lengths."""
        if len(self.vertices) != 3:
            return None

        a = self.vertices[0].distance_to(self.vertices[1])
        b = self.vertices[1].distance_to(self.vertices[2])
        c = self.vertices[2].distance_to(self.vertices[0])

        return a + b + c


@dataclass
class Quadrilateral(Polygon):
    """Quadrilateral (4-sided polygon)."""

    def __post_init__(self):
        self.num_sides = 4
        self.shape_type = "Viereck"
        super().__post_init__()

    def is_rectangle(self) -> bool:
        """Check if this quadrilateral is a rectangle (all angles 90°)."""
        if len(self.vertices) != 4:
            return False

        # Check if all angles are approximately 90 degrees
        # For a rectangle, opposite sides should be parallel and equal
        # We check if diagonals are equal (property of rectangles)
        d1 = self.vertices[0].distance_to(self.vertices[2])
        d2 = self.vertices[1].distance_to(self.vertices[3])

        return abs(d1 - d2) < 0.001  # Tolerance for floating point

    def calculate_area(self) -> Optional[float]:
        """Calculate area for rectangle (if applicable)."""
        if not self.is_rectangle() or len(self.vertices) != 4:
            return None

        # For a rectangle, area = width × height
        width = self.vertices[0].distance_to(self.vertices[1])
        height = self.vertices[1].distance_to(self.vertices[2])

        return width * height

    def calculate_perimeter(self) -> Optional[float]:
        """Calculate perimeter as sum of side lengths."""
        if len(self.vertices) != 4:
            return None

        perimeter = 0
        for i in range(4):
            next_i = (i + 1) % 4
            perimeter += self.vertices[i].distance_to(self.vertices[next_i])

        return perimeter


@dataclass
class Circle(GeometricShape):
    """Circle with center and radius."""

    center: Optional[Position] = None
    radius: float = 0.0

    def __post_init__(self):
        self.shape_type = "Kreis"

    def calculate_area(self) -> Optional[float]:
        """Calculate area: π × r²"""
        if self.radius <= 0:
            return None
        return math.pi * self.radius**2

    def calculate_perimeter(self) -> Optional[float]:
        """Calculate circumference: 2 × π × r"""
        if self.radius <= 0:
            return None
        return 2 * math.pi * self.radius


# ============================================================================
# Main Spatial Reasoning Engine
# ============================================================================


class SpatialReasoner:
    """
    Main spatial reasoning engine for KAI.

    Provides methods for:
    - Spatial relation inference (transitive, symmetric)
    - Position-based reasoning
    - Grid topology analysis
    - Integration with KAI's knowledge graph
    """

    def __init__(self, netzwerk=None):
        """
        Initialize the spatial reasoner.

        Args:
            netzwerk: KonzeptNetzwerk instance for graph access
        """
        self.netzwerk = netzwerk

        # Cache for spatial queries (5 minute TTL)
        self._query_cache = TTLCache(maxsize=100, ttl=300)

        # Supported spatial relations
        self.spatial_relation_types = {rt.value for rt in SpatialRelationType}

        # Performance metrics
        self._performance_metrics = {
            "queries_total": 0,
            "queries_cached": 0,
            "transitive_inferences": 0,
            "symmetric_inferences": 0,
        }

        logger.info(
            "SpatialReasoner initialized",
            extra={
                "relation_types": len(self.spatial_relation_types),
                "cache_size": 100,
                "cache_ttl": 300,
            },
        )

    def infer_spatial_relations(
        self, subject: str, relation_type: Optional[SpatialRelationType] = None
    ) -> SpatialReasoningResult:
        """
        Infer spatial relations for a given subject.

        Args:
            subject: The entity to reason about
            relation_type: Optional filter for specific relation type

        Returns:
            SpatialReasoningResult with inferred relations
        """
        cache_key = f"{subject}:{relation_type.value if relation_type else 'ALL'}"

        # Update metrics
        self._performance_metrics["queries_total"] += 1

        # Check cache
        if cache_key in self._query_cache:
            self._performance_metrics["queries_cached"] += 1
            logger.debug(
                "Cache hit for spatial query",
                extra={
                    "cache_key": cache_key,
                    "cache_hit_rate": self._performance_metrics["queries_cached"]
                    / self._performance_metrics["queries_total"],
                },
            )
            return self._query_cache[cache_key]

        logger.info(
            "Inferring spatial relations",
            extra={
                "subject": subject,
                "relation_type": relation_type.value if relation_type else "ALL",
                "cache_key": cache_key,
            },
        )

        result = SpatialReasoningResult(query=subject)

        try:
            # Query direct relations from graph
            direct_relations = self._query_direct_relations(subject, relation_type)
            result.relations.extend(direct_relations)
            result.reasoning_steps.append(
                f"Found {len(direct_relations)} direct spatial relations"
            )

            # Apply transitive inference
            transitive_relations = self._infer_transitive_relations(
                subject, direct_relations, relation_type
            )
            result.relations.extend(transitive_relations)
            if transitive_relations:
                result.reasoning_steps.append(
                    f"Inferred {len(transitive_relations)} transitive relations"
                )

            # Apply symmetric inference
            symmetric_relations = self._infer_symmetric_relations(
                subject, result.relations
            )
            result.relations.extend(symmetric_relations)
            if symmetric_relations:
                result.reasoning_steps.append(
                    f"Inferred {len(symmetric_relations)} symmetric relations"
                )

            # Calculate overall confidence
            if result.relations:
                result.confidence = sum(r.confidence for r in result.relations) / len(
                    result.relations
                )

            logger.info(
                "Spatial reasoning complete: %d relations, confidence=%.2f",
                len(result.relations),
                result.confidence,
            )

        except Exception as e:
            logger.error("Error during spatial reasoning: %s", str(e), exc_info=True)
            result.error = str(e)
            result.confidence = 0.0

        # Cache result
        self._query_cache[cache_key] = result

        return result

    def _query_direct_relations(
        self, subject: str, relation_type: Optional[SpatialRelationType]
    ) -> List[SpatialRelation]:
        """
        Query direct spatial relations from the knowledge graph.

        Args:
            subject: Entity to query
            relation_type: Optional relation type filter

        Returns:
            List of direct spatial relations
        """
        relations = []

        # Determine which relation types to query
        if relation_type:
            relation_types_to_query = [relation_type.value]
        else:
            relation_types_to_query = list(self.spatial_relation_types)

        # Query graph for each relation type
        for rel_type_str in relation_types_to_query:
            try:
                # Query facts from graph
                facts = self.netzwerk.query_graph_for_facts(subject)

                # Extract spatial relations
                if rel_type_str in facts:
                    for obj in facts[rel_type_str]:
                        rel_type_enum = SpatialRelationType(rel_type_str)
                        relations.append(
                            SpatialRelation(
                                subject=subject,
                                object=obj,
                                relation_type=rel_type_enum,
                                confidence=1.0,  # Direct facts have full confidence
                            )
                        )
            except Exception as e:
                logger.warning("Error querying relation %s: %s", rel_type_str, str(e))

        return relations

    def _infer_transitive_relations(
        self,
        subject: str,
        known_relations: List[SpatialRelation],
        relation_type_filter: Optional[SpatialRelationType],
    ) -> List[SpatialRelation]:
        """
        Infer transitive spatial relations.

        For transitive relations R: If A R B and B R C, then A R C.
        Examples: NORTH_OF, INSIDE, CONTAINS, etc.

        Args:
            subject: Starting entity
            known_relations: Already known relations
            relation_type_filter: Optional filter

        Returns:
            List of inferred transitive relations
        """
        inferred = []

        # Only process transitive relation types
        transitive_types = {
            rt
            for rt in SpatialRelationType
            if rt.is_transitive
            and (not relation_type_filter or rt == relation_type_filter)
        }

        for rel_type in transitive_types:
            # Find all A R B relations
            for rel in known_relations:
                if rel.relation_type == rel_type:
                    # Query for B R C relations
                    second_hop = self._query_direct_relations(rel.object, rel_type)

                    for second_rel in second_hop:
                        # A R B, B R C => A R C
                        inferred_rel = SpatialRelation(
                            subject=subject,
                            object=second_rel.object,
                            relation_type=rel_type,
                            confidence=min(rel.confidence, second_rel.confidence)
                            * 0.9,  # Decay
                            metadata={
                                "inferred_via": "transitivity",
                                "intermediate": rel.object,
                            },
                        )
                        inferred.append(inferred_rel)

        return inferred

    def infer_transitive_with_proof(
        self,
        subject: str,
        known_relations: List[SpatialRelation],
        relation_type_filter: Optional[SpatialRelationType] = None,
    ) -> Tuple[List[SpatialRelation], List]:
        """
        Infer transitive spatial relations WITH ProofStep generation.

        For transitive relations R: If A R B and B R C, then A R C.

        Args:
            subject: Starting entity
            known_relations: Already known relations
            relation_type_filter: Optional filter

        Returns:
            Tuple (inferred_relations, proof_steps)
        """
        from component_17_proof_explanation import ProofStep, StepType

        inferred = []
        proof_steps = []

        # Only process transitive relation types
        transitive_types = {
            rt
            for rt in SpatialRelationType
            if rt.is_transitive
            and (not relation_type_filter or rt == relation_type_filter)
        }

        for rel_type in transitive_types:
            # Find all A R B relations
            for rel in known_relations:
                if rel.relation_type == rel_type:
                    # Query for B R C relations
                    second_hop = self._query_direct_relations(rel.object, rel_type)

                    for second_rel in second_hop:
                        # A R B, B R C => A R C
                        intermediate = rel.object
                        final_object = second_rel.object

                        inferred_rel = SpatialRelation(
                            subject=subject,
                            object=final_object,
                            relation_type=rel_type,
                            confidence=min(rel.confidence, second_rel.confidence) * 0.9,
                            metadata={
                                "inferred_via": "transitivity",
                                "intermediate": intermediate,
                            },
                        )
                        inferred.append(inferred_rel)

                        # Erstelle ProofStep für diese transitive Inferenz
                        proof_step = ProofStep(
                            step_id=f"spatial_transitive_{subject}_{rel_type.value}_{final_object}",
                            step_type=StepType.SPATIAL_TRANSITIVE_INFERENCE,
                            inputs=[
                                f"{subject} {rel_type.value} {intermediate}",
                                f"{intermediate} {rel_type.value} {final_object}",
                            ],
                            rule_name=f"Transitivity({rel_type.value})",
                            output=f"{subject} {rel_type.value} {final_object}",
                            confidence=inferred_rel.confidence,
                            explanation_text=(
                                f"Durch transitive Schlussfolgerung: "
                                f"Da '{subject}' {rel_type.value} '{intermediate}' liegt, "
                                f"und '{intermediate}' {rel_type.value} '{final_object}' liegt, "
                                f"muss '{subject}' auch {rel_type.value} '{final_object}' liegen."
                            ),
                            metadata={
                                "inference_type": "transitivity",
                                "intermediate_entity": intermediate,
                                "relation_type": rel_type.value,
                            },
                            source_component="spatial_reasoning",
                        )
                        proof_steps.append(proof_step)

        # Update metrics
        if hasattr(self, "_performance_metrics"):
            self._performance_metrics["transitive_inferences"] += len(inferred)

        logger.info(
            "Transitive inference complete",
            extra={
                "relations_inferred": len(inferred),
                "proof_steps_generated": len(proof_steps),
                "total_transitive_inferences": (
                    self._performance_metrics.get("transitive_inferences", 0)
                    if hasattr(self, "_performance_metrics")
                    else 0
                ),
            },
        )

        return inferred, proof_steps

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Returns performance metrics for the spatial reasoner.

        Returns:
            Dictionary with performance metrics
        """
        metrics = self._performance_metrics.copy()

        # Add cache statistics
        metrics["cache_size"] = len(self._query_cache)
        metrics["cache_max_size"] = self._query_cache.maxsize

        if metrics["queries_total"] > 0:
            metrics["cache_hit_rate"] = (
                metrics["queries_cached"] / metrics["queries_total"]
            )
        else:
            metrics["cache_hit_rate"] = 0.0

        return metrics

    def log_performance_summary(self) -> None:
        """Logs a summary of performance metrics."""
        metrics = self.get_performance_metrics()

        logger.info(
            "SpatialReasoner Performance Summary",
            extra={
                "queries_total": metrics["queries_total"],
                "cache_hit_rate": f"{metrics['cache_hit_rate']:.2%}",
                "transitive_inferences": metrics["transitive_inferences"],
                "symmetric_inferences": metrics["symmetric_inferences"],
                "cache_usage": f"{metrics['cache_size']}/{metrics['cache_max_size']}",
            },
        )

    def _infer_symmetric_relations(
        self, subject: str, known_relations: List[SpatialRelation]
    ) -> List[SpatialRelation]:
        """
        Infer symmetric spatial relations.

        For symmetric relations R: If A R B, then B R A.
        Examples: ADJACENT_TO, NEIGHBOR_ORTHOGONAL, etc.

        Args:
            subject: Entity to reason about
            known_relations: Known relations

        Returns:
            List of inferred symmetric relations (from subject's perspective)
        """
        inferred = []

        # Query relations where subject is the object (B R A)
        for rel_type in SpatialRelationType:
            if not rel_type.is_symmetric:
                continue

            try:
                # Find all X R subject relations
                # This requires a reverse query (find nodes that have relation to subject)
                # We'll use the graph traversal for this
                reverse_facts = self.netzwerk.find_incoming_relations(
                    subject, rel_type.value
                )

                for source_entity in reverse_facts:
                    # X R subject => subject R X (symmetric)
                    inferred_rel = SpatialRelation(
                        subject=subject,
                        object=source_entity,
                        relation_type=rel_type,
                        confidence=1.0,  # Symmetric relations preserve confidence
                        metadata={"inferred_via": "symmetry"},
                    )
                    inferred.append(inferred_rel)

            except AttributeError:
                # If find_incoming_relations doesn't exist, skip
                # We'll implement this in the netzwerk integration
                logger.debug("Reverse query not available for symmetry inference")
                continue

        return inferred

    def check_spatial_consistency(
        self, relations: List[SpatialRelation]
    ) -> Tuple[bool, List[str]]:
        """
        Check if a set of spatial relations is consistent.

        Detects contradictions like:
        - A NORTH_OF B and B NORTH_OF A (impossible)
        - A INSIDE B and B INSIDE A (impossible)
        - Circular transitive relations

        Args:
            relations: Set of spatial relations to check

        Returns:
            Tuple of (is_consistent, list_of_violations)
        """
        violations = []

        # Build relation graph
        relation_graph: Dict[Tuple[str, str], Set[SpatialRelationType]] = {}

        for rel in relations:
            key = (rel.subject, rel.object)
            if key not in relation_graph:
                relation_graph[key] = set()
            relation_graph[key].add(rel.relation_type)

        # Check for contradictions
        for (subj, obj), rel_types in relation_graph.items():
            # Check inverse key
            inverse_key = (obj, subj)
            if inverse_key in relation_graph:
                inverse_rels = relation_graph[inverse_key]

                # For each relation type, check for contradictions
                for rel_type in rel_types:
                    # Non-symmetric relations with their inverse is a contradiction
                    if not rel_type.is_symmetric and rel_type.inverse in inverse_rels:
                        violations.append(
                            f"Contradiction: {subj} {rel_type.value} {obj} AND "
                            f"{obj} {rel_type.inverse.value} {subj}"
                        )

                    # Same non-symmetric relation in both directions is also a contradiction
                    if not rel_type.is_symmetric and rel_type in inverse_rels:
                        violations.append(
                            f"Contradiction: {subj} {rel_type.value} {obj} AND "
                            f"{obj} {rel_type.value} {subj}"
                        )

        is_consistent = len(violations) == 0

        if not is_consistent:
            logger.warning(
                "Spatial consistency check failed: %d violations", len(violations)
            )

        return is_consistent, violations

    def clear_cache(self):
        """Clear the query cache."""
        self._query_cache.clear()
        logger.info("Spatial reasoning cache cleared")

    def add_position(
        self, object_name: str, position: Position, store_in_graph: bool = True
    ) -> bool:
        """
        Store an object's position in the knowledge graph (without requiring a grid).

        This is a simplified position storage for cases where you don't need
        a full grid structure.

        Args:
            object_name: Name of the object
            position: Position (x, y coordinates)
            store_in_graph: If True, store in Neo4j; if False, just validate

        Returns:
            True if successful, False otherwise
        """
        if not self.netzwerk or not store_in_graph:
            logger.debug(
                "Position for %s at %s recorded (not stored in graph)",
                object_name,
                position,
            )
            return True

        try:
            # Create object node with position attributes
            self.netzwerk.create_wort_if_not_exists(
                lemma=object_name,
                pos="NOUN",
                type="SpatialObject",
                position_x=position.x,
                position_y=position.y,
            )

            logger.info("Stored position for %s at %s", object_name, position)
            return True

        except Exception as e:
            logger.error(
                "Failed to store position for %s: %s",
                object_name,
                str(e),
                exc_info=True,
            )
            return False

    def add_spatial_relation(
        self,
        subject: str,
        relation_type: SpatialRelationType,
        target: str,
        confidence: float = 1.0,
    ) -> bool:
        """
        Store a spatial relation between two objects in the knowledge graph.

        Args:
            subject: The subject entity (e.g., "house")
            relation_type: Type of spatial relation (e.g., NORTH_OF, ADJACENT_TO)
            target: The target entity (e.g., "tree")
            confidence: Confidence in this relation (0.0-1.0)

        Returns:
            True if successful, False otherwise
        """
        if not self.netzwerk:
            logger.warning("No knowledge graph available for storing spatial relation")
            return False

        try:
            # Ensure both entities exist in the graph
            self.netzwerk.create_wort_if_not_exists(
                lemma=subject, pos="NOUN", type="SpatialEntity"
            )
            self.netzwerk.create_wort_if_not_exists(
                lemma=target, pos="NOUN", type="SpatialEntity"
            )

            # Store the spatial relation
            self.netzwerk.assert_relation(
                from_lemma=subject,
                to_lemma=target,
                relation_type=relation_type.value,
                confidence=confidence,
            )

            logger.info(
                "Stored spatial relation: %s %s %s (confidence: %.2f)",
                subject,
                relation_type.value,
                target,
                confidence,
            )

            # Update metrics
            self._performance_metrics["transitive_inferences"] += 1

            return True

        except Exception as e:
            logger.error(
                "Failed to store spatial relation %s %s %s: %s",
                subject,
                relation_type.value,
                target,
                str(e),
                exc_info=True,
            )
            return False

    def register_spatial_extraction_rules(self) -> int:
        """
        Register extraction rules for spatial relations in the knowledge graph.

        This enables KAI to learn spatial relations from natural language input.
        Uses German language patterns.

        Returns:
            Number of rules successfully registered
        """
        logger.info("Registering spatial extraction rules in knowledge graph")

        rules_registered = 0

        # Define extraction rules for spatial relations (German patterns)
        spatial_patterns = [
            # Cardinal directions
            ("NORTH_OF", r"^(.+) liegt nördlich von (.+)$"),
            ("NORTH_OF", r"^(.+) ist nördlich von (.+)$"),
            ("SOUTH_OF", r"^(.+) liegt südlich von (.+)$"),
            ("SOUTH_OF", r"^(.+) ist südlich von (.+)$"),
            ("EAST_OF", r"^(.+) liegt östlich von (.+)$"),
            ("EAST_OF", r"^(.+) ist östlich von (.+)$"),
            ("WEST_OF", r"^(.+) liegt westlich von (.+)$"),
            ("WEST_OF", r"^(.+) ist westlich von (.+)$"),
            # Adjacency
            ("ADJACENT_TO", r"^(.+) liegt neben (.+)$"),
            ("ADJACENT_TO", r"^(.+) ist neben (.+)$"),
            ("ADJACENT_TO", r"^(.+) grenzt an (.+)$"),
            ("NEIGHBOR_ORTHOGONAL", r"^(.+) ist direkter Nachbar von (.+)$"),
            ("NEIGHBOR_DIAGONAL", r"^(.+) ist diagonaler Nachbar von (.+)$"),
            # Containment
            ("INSIDE", r"^(.+) ist in (.+)$"),
            ("INSIDE", r"^(.+) liegt in (.+)$"),
            ("INSIDE", r"^(.+) befindet sich in (.+)$"),
            ("CONTAINS", r"^(.+) enthält (.+)$"),
            ("CONTAINS", r"^(.+) beinhaltet (.+)$"),
            # Vertical relations
            ("ABOVE", r"^(.+) ist über (.+)$"),
            ("ABOVE", r"^(.+) liegt über (.+)$"),
            ("BELOW", r"^(.+) ist unter (.+)$"),
            ("BELOW", r"^(.+) liegt unter (.+)$"),
            # Position
            ("LOCATED_AT", r"^(.+) ist bei (.+)$"),
            ("LOCATED_AT", r"^(.+) ist an (.+)$"),
            ("LOCATED_AT", r"^(.+) steht auf (.+)$"),
        ]

        # Register each extraction rule
        for relation_type, pattern in spatial_patterns:
            try:
                self.netzwerk.create_extraction_rule(
                    relation_type=relation_type, regex_pattern=pattern
                )
                rules_registered += 1
                logger.debug(
                    "Registered extraction rule: %s -> %s", relation_type, pattern
                )

            except Exception as e:
                # Rule might already exist, continue
                logger.debug("Could not register rule %s: %s", relation_type, str(e))
                continue

        logger.info(
            "Successfully registered %d spatial extraction rules", rules_registered
        )
        return rules_registered

    # ========================================================================
    # Grid Management (Phase 2.1)
    # ========================================================================

    def create_grid(self, grid: Grid) -> bool:
        """
        Create a grid in the knowledge graph.

        Creates:
        - Grid concept node with metadata
        - Position nodes for each cell
        - Neighborhood relationships between positions

        Args:
            grid: Grid specification

        Returns:
            True if successful, False otherwise
        """
        logger.info("Creating grid in knowledge graph: %s", grid)

        try:
            # Step 1: Create Grid concept node
            grid_properties = {
                "width": grid.width,
                "height": grid.height,
                "size": grid.size,
                "neighborhood_type": grid.neighborhood_type.value,
                "type": "Grid",
            }

            # Add custom metadata
            grid_properties.update(grid.metadata)

            # Create grid node
            self.netzwerk.create_wort_if_not_exists(
                lemma=grid.name, pos="NOUN", **grid_properties
            )

            logger.debug("Created grid node: %s", grid.name)

            # Step 2: Create position nodes for each cell
            positions_created = 0
            for pos in grid.get_all_positions():
                pos_name = grid.get_position_name(pos)

                # Create position node with coordinates
                self.netzwerk.create_wort_if_not_exists(
                    lemma=pos_name, pos="NOUN", x=pos.x, y=pos.y, type="GridPosition"
                )

                # Link position to grid
                self.netzwerk.assert_relation(
                    from_lemma=pos_name, to_lemma=grid.name, relation_type="PART_OF"
                )

                positions_created += 1

            logger.debug("Created %d position nodes", positions_created)

            # Step 3: Create neighborhood relationships
            neighbors_created = self._create_grid_neighbors(grid)
            logger.debug("Created %d neighborhood relationships", neighbors_created)

            logger.info(
                "Grid created successfully: %s with %d positions and %d neighbors",
                grid.name,
                positions_created,
                neighbors_created,
            )

            return True

        except Exception as e:
            logger.error(
                "Failed to create grid %s: %s", grid.name, str(e), exc_info=True
            )
            return False

    def _create_grid_neighbors(self, grid: Grid) -> int:
        """
        Create neighborhood relationships for all grid positions.

        Args:
            grid: Grid specification

        Returns:
            Number of neighbor relationships created
        """
        neighbors_created = 0

        for pos in grid.get_all_positions():
            # Get neighbors based on grid's neighborhood type
            neighbors = pos.get_neighbors(grid.neighborhood_type, grid.custom_offsets)

            # Filter to only valid grid positions
            valid_neighbors = [n for n in neighbors if grid.is_valid_position(n)]

            # Create relationships
            pos_name = grid.get_position_name(pos)

            for neighbor_pos in valid_neighbors:
                neighbor_name = grid.get_position_name(neighbor_pos)

                # Determine relation type based on direction
                direction = pos.direction_to(neighbor_pos)

                if direction:
                    # Cardinal direction relationship
                    self.netzwerk.assert_relation(
                        from_lemma=pos_name,
                        to_lemma=neighbor_name,
                        relation_type=direction.value,
                    )
                else:
                    # Diagonal or custom - use generic ADJACENT_TO
                    self.netzwerk.assert_relation(
                        from_lemma=pos_name,
                        to_lemma=neighbor_name,
                        relation_type="ADJACENT_TO",
                    )

                neighbors_created += 1

        return neighbors_created

    def get_grid(self, grid_name: str) -> Optional[Grid]:
        """
        Retrieve a grid from the knowledge graph.

        Args:
            grid_name: Name of the grid

        Returns:
            Grid object if found, None otherwise
        """
        try:
            # Query grid node
            grid_node = self.netzwerk.find_wort_node(grid_name)

            if not grid_node:
                logger.warning("Grid not found: %s", grid_name)
                return None

            # Extract properties
            props = dict(grid_node)

            # Reconstruct grid
            grid = Grid(
                name=grid_name,
                width=props.get("width", 0),
                height=props.get("height", 0),
                neighborhood_type=NeighborhoodType(
                    props.get("neighborhood_type", "orthogonal")
                ),
                metadata={
                    k: v
                    for k, v in props.items()
                    if k not in ["width", "height", "size", "neighborhood_type", "type"]
                },
            )

            logger.debug("Retrieved grid: %s", grid)
            return grid

        except Exception as e:
            logger.error("Error retrieving grid %s: %s", grid_name, str(e))
            return None

    def delete_grid(self, grid_name: str) -> bool:
        """
        Delete a grid and all its positions from the knowledge graph.

        Args:
            grid_name: Name of the grid to delete

        Returns:
            True if successful, False otherwise
        """
        logger.info("Deleting grid: %s", grid_name)

        try:
            # Get grid to find all positions
            grid = self.get_grid(grid_name)
            if not grid:
                logger.warning("Grid not found for deletion: %s", grid_name)
                return False

            # Delete all position nodes
            positions_deleted = 0
            for pos in grid.get_all_positions():
                pos_name = grid.get_position_name(pos)
                # Delete position node and all its relationships
                self.netzwerk.delete_wort_node(pos_name)
                positions_deleted += 1

            # Delete grid node
            self.netzwerk.delete_wort_node(grid_name)

            logger.info(
                "Deleted grid %s with %d positions", grid_name, positions_deleted
            )
            return True

        except Exception as e:
            logger.error("Error deleting grid %s: %s", grid_name, str(e), exc_info=True)
            return False

    # ========================================================================
    # Position Tracking (Phase 2.2)
    # ========================================================================

    def place_object(
        self, object_name: str, grid_name: str, position: Position
    ) -> bool:
        """
        Place an object at a specific position on a grid.

        Args:
            object_name: Name of the object to place
            grid_name: Name of the grid
            position: Position to place the object

        Returns:
            True if successful, False otherwise
        """
        logger.info(
            "Placing object %s on grid %s at position %s",
            object_name,
            grid_name,
            position,
        )

        try:
            # Verify grid exists
            grid = self.get_grid(grid_name)
            if not grid:
                logger.error("Grid not found: %s", grid_name)
                return False

            # Verify position is valid
            if not grid.is_valid_position(position):
                logger.error(
                    "Position %s is out of bounds for grid %s", position, grid_name
                )
                return False

            # Create object node if it doesn't exist
            self.netzwerk.create_wort_if_not_exists(
                lemma=object_name, pos="NOUN", type="GridObject"
            )

            # Get position name
            pos_name = grid.get_position_name(position)

            # Create LOCATED_AT relation
            self.netzwerk.assert_relation(
                from_lemma=object_name, to_lemma=pos_name, relation_type="LOCATED_AT"
            )

            logger.info("Object %s placed at %s", object_name, pos_name)
            return True

        except Exception as e:
            logger.error(
                "Error placing object %s: %s", object_name, str(e), exc_info=True
            )
            return False

    def move_object(
        self,
        object_name: str,
        grid_name: str,
        from_position: Position,
        to_position: Position,
    ) -> bool:
        """
        Move an object from one position to another on a grid.

        Args:
            object_name: Name of the object to move
            grid_name: Name of the grid
            from_position: Current position
            to_position: Target position

        Returns:
            True if successful, False otherwise
        """
        logger.info(
            "Moving object %s from %s to %s on grid %s",
            object_name,
            from_position,
            to_position,
            grid_name,
        )

        try:
            # Verify grid exists
            grid = self.get_grid(grid_name)
            if not grid:
                logger.error("Grid not found: %s", grid_name)
                return False

            # Verify positions are valid
            if not grid.is_valid_position(from_position):
                logger.error("From position %s is out of bounds", from_position)
                return False

            if not grid.is_valid_position(to_position):
                logger.error("To position %s is out of bounds", to_position)
                return False

            # Remove from old position
            from_pos_name = grid.get_position_name(from_position)
            self.netzwerk.delete_relation(
                from_lemma=object_name,
                to_lemma=from_pos_name,
                relation_type="LOCATED_AT",
            )

            # Add to new position
            to_pos_name = grid.get_position_name(to_position)
            self.netzwerk.assert_relation(
                from_lemma=object_name, to_lemma=to_pos_name, relation_type="LOCATED_AT"
            )

            logger.info(
                "Object %s moved from %s to %s", object_name, from_pos_name, to_pos_name
            )
            return True

        except Exception as e:
            logger.error(
                "Error moving object %s: %s", object_name, str(e), exc_info=True
            )
            return False

    def remove_object(
        self, object_name: str, grid_name: str, position: Position
    ) -> bool:
        """
        Remove an object from a position on a grid.

        Args:
            object_name: Name of the object to remove
            grid_name: Name of the grid
            position: Position to remove from

        Returns:
            True if successful, False otherwise
        """
        logger.info(
            "Removing object %s from position %s on grid %s",
            object_name,
            position,
            grid_name,
        )

        try:
            # Verify grid exists
            grid = self.get_grid(grid_name)
            if not grid:
                logger.error("Grid not found: %s", grid_name)
                return False

            # Get position name
            pos_name = grid.get_position_name(position)

            # Remove LOCATED_AT relation
            self.netzwerk.delete_relation(
                from_lemma=object_name, to_lemma=pos_name, relation_type="LOCATED_AT"
            )

            logger.info("Object %s removed from %s", object_name, pos_name)
            return True

        except Exception as e:
            logger.error(
                "Error removing object %s: %s", object_name, str(e), exc_info=True
            )
            return False

    def get_object_position(
        self, object_name: str, grid_name: str
    ) -> Optional[Position]:
        """
        Get the current position of an object on a grid.

        Args:
            object_name: Name of the object
            grid_name: Name of the grid

        Returns:
            Position if found, None otherwise
        """
        try:
            # Query LOCATED_AT relations
            facts = self.netzwerk.query_graph_for_facts(object_name)

            if "LOCATED_AT" not in facts:
                logger.debug("Object %s has no position", object_name)
                return None

            # Get grid to parse position names
            grid = self.get_grid(grid_name)
            if not grid:
                logger.error("Grid not found: %s", grid_name)
                return None

            # Find position belonging to this grid
            for pos_name in facts["LOCATED_AT"]:
                if pos_name.startswith(grid.name + "_Pos_"):
                    # Parse coordinates from name
                    parts = pos_name.split("_")
                    if len(parts) >= 4:
                        x = int(parts[-2])
                        y = int(parts[-1])
                        return Position(x, y)

            logger.debug("Object %s not found on grid %s", object_name, grid_name)
            return None

        except Exception as e:
            logger.error("Error getting position for %s: %s", object_name, str(e))
            return None

    def get_objects_at_position(self, grid_name: str, position: Position) -> List[str]:
        """
        Get all objects at a specific position on a grid.

        Args:
            grid_name: Name of the grid
            position: Position to query

        Returns:
            List of object names at that position
        """
        try:
            # Verify grid exists
            grid = self.get_grid(grid_name)
            if not grid:
                logger.error("Grid not found: %s", grid_name)
                return []

            # Get position name
            pos_name = grid.get_position_name(position)

            # Query incoming LOCATED_AT relations
            # We need to find all nodes that have LOCATED_AT -> pos_name
            # This requires a Cypher query
            query = """
            MATCH (obj)-[:LOCATED_AT]->(pos)
            WHERE pos.lemma = $pos_name
            RETURN obj.lemma as object_name
            """

            result = self.netzwerk.session.run(query, pos_name=pos_name)
            objects = [record["object_name"] for record in result]

            logger.debug("Found %d objects at position %s", len(objects), pos_name)
            return objects

        except Exception as e:
            logger.error("Error getting objects at position %s: %s", position, str(e))
            return []

    # ========================================================================
    # Neighborhood Logic (Phase 2.3)
    # ========================================================================

    def get_neighbors(self, grid_name: str, position: Position) -> List[Position]:
        """
        Get all neighboring positions for a given position on a grid.

        Args:
            grid_name: Name of the grid
            position: Position to find neighbors for

        Returns:
            List of neighboring positions
        """
        try:
            # Get grid
            grid = self.get_grid(grid_name)
            if not grid:
                logger.error("Grid not found: %s", grid_name)
                return []

            # Verify position is valid
            if not grid.is_valid_position(position):
                logger.error("Position %s is out of bounds", position)
                return []

            # Get neighbors using Position.get_neighbors() and filter by grid bounds
            all_neighbors = position.get_neighbors(
                grid.neighborhood_type, grid.custom_offsets
            )

            valid_neighbors = [n for n in all_neighbors if grid.is_valid_position(n)]

            logger.debug(
                "Position %s has %d neighbors on grid %s",
                position,
                len(valid_neighbors),
                grid_name,
            )

            return valid_neighbors

        except Exception as e:
            logger.error(
                "Error getting neighbors for position %s: %s", position, str(e)
            )
            return []

    def find_path(
        self,
        grid_name: str,
        start: Position,
        goal: Position,
        allow_diagonal: bool = False,
    ) -> Optional[List[Position]]:
        """
        Find a path between two positions on a grid using A* algorithm.

        Args:
            grid_name: Name of the grid
            start: Starting position
            goal: Goal position
            allow_diagonal: Whether diagonal moves are allowed

        Returns:
            List of positions representing the path (including start and goal),
            or None if no path exists
        """
        try:
            # Get grid
            grid = self.get_grid(grid_name)
            if not grid:
                logger.error("Grid not found: %s", grid_name)
                return None

            # Verify positions are valid
            if not grid.is_valid_position(start) or not grid.is_valid_position(goal):
                logger.error("Invalid start or goal position")
                return None

            # A* pathfinding
            from heapq import heappop, heappush

            # Priority queue: (f_score, position)
            open_set = [(0, start)]
            came_from = {}
            g_score = {start: 0}
            f_score = {start: start.distance_to(goal, metric="manhattan")}

            while open_set:
                current_f, current = heappop(open_set)

                # Goal reached
                if current == goal:
                    # Reconstruct path
                    path = [current]
                    while current in came_from:
                        current = came_from[current]
                        path.append(current)
                    path.reverse()

                    logger.debug(
                        "Found path from %s to %s: %d steps", start, goal, len(path)
                    )
                    return path

                # Get neighbors
                neighbor_type = (
                    NeighborhoodType.MOORE
                    if allow_diagonal
                    else NeighborhoodType.ORTHOGONAL
                )
                neighbors = current.get_neighbors(neighbor_type)
                neighbors = [n for n in neighbors if grid.is_valid_position(n)]

                for neighbor in neighbors:
                    # Tentative g_score
                    tentative_g = g_score[current] + 1

                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        # This path is better
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score[neighbor] = tentative_g + neighbor.distance_to(
                            goal, metric="manhattan"
                        )

                        # Add to open set if not already there
                        if neighbor not in [pos for _, pos in open_set]:
                            heappush(open_set, (f_score[neighbor], neighbor))

            # No path found
            logger.debug(
                "No path found from %s to %s on grid %s", start, goal, grid_name
            )
            return None

        except Exception as e:
            logger.error("Error finding path: %s", str(e), exc_info=True)
            return None

    def get_distance_between_positions(
        self, position1: Position, position2: Position, metric: str = "manhattan"
    ) -> float:
        """
        Calculate distance between two positions.

        Args:
            position1: First position
            position2: Second position
            metric: Distance metric ('manhattan', 'euclidean', 'chebyshev')

        Returns:
            Distance value
        """
        return position1.distance_to(position2, metric=metric)

    def get_objects_in_neighborhood(
        self, grid_name: str, position: Position, radius: int = 1
    ) -> Dict[Position, List[str]]:
        """
        Get all objects within a certain radius of a position.

        Args:
            grid_name: Name of the grid
            position: Center position
            radius: Search radius (in grid cells)

        Returns:
            Dictionary mapping positions to lists of objects at those positions
        """
        try:
            # Get grid
            grid = self.get_grid(grid_name)
            if not grid:
                logger.error("Grid not found: %s", grid_name)
                return {}

            result = {}

            # Check all positions within radius
            for x in range(
                max(0, position.x - radius), min(grid.width, position.x + radius + 1)
            ):
                for y in range(
                    max(0, position.y - radius),
                    min(grid.height, position.y + radius + 1),
                ):
                    check_pos = Position(x, y)

                    # Calculate actual distance
                    if check_pos.distance_to(position, metric="chebyshev") <= radius:
                        objects = self.get_objects_at_position(grid_name, check_pos)
                        if objects:
                            result[check_pos] = objects

            logger.debug(
                "Found %d positions with objects within radius %d of %s",
                len(result),
                radius,
                position,
            )

            return result

        except Exception as e:
            logger.error("Error getting objects in neighborhood: %s", str(e))
            return {}

    # ========================================================================
    # Geometric Shapes (Phase 3)
    # ========================================================================

    def create_shape(self, shape: GeometricShape) -> bool:
        """
        Create a geometric shape in the knowledge graph.

        Args:
            shape: GeometricShape instance (Triangle, Quadrilateral, Circle, etc.)

        Returns:
            True if successful, False otherwise
        """
        logger.info("Creating shape in knowledge graph: %s", shape)

        try:
            # Step 1: Create shape node with basic properties
            shape_properties = {
                "shape_type": shape.shape_type,
                "type": "GeometricShape",
            }

            # Add custom properties
            shape_properties.update(shape.properties)

            # Add shape-specific properties
            if isinstance(shape, Polygon):
                shape_properties["num_sides"] = shape.num_sides

            if isinstance(shape, Circle):
                shape_properties["radius"] = shape.radius
                if shape.center:
                    shape_properties["center_x"] = shape.center.x
                    shape_properties["center_y"] = shape.center.y

            # Create shape node
            self.netzwerk.create_wort_if_not_exists(
                lemma=shape.name, pos="NOUN", **shape_properties
            )

            # Step 2: Create IS_A hierarchy
            # e.g., "Dreieck1" IS_A "Dreieck" IS_A "Polygon" IS_A "Form"
            self.netzwerk.assert_relation(
                from_lemma=shape.name, to_lemma=shape.shape_type, relation_type="IS_A"
            )

            # Step 3: Add HAS_PROPERTY relations for computed properties
            area = shape.calculate_area()
            if area is not None:
                # Create property node for area
                area_node_name = f"{shape.name}_Fläche"
                self.netzwerk.create_wort_if_not_exists(
                    lemma=area_node_name,
                    pos="NOUN",
                    value=area,
                    unit="quadrat_einheiten",
                )
                self.netzwerk.assert_relation(
                    from_lemma=shape.name,
                    to_lemma=area_node_name,
                    relation_type="HAS_PROPERTY",
                )

            perimeter = shape.calculate_perimeter()
            if perimeter is not None:
                # Create property node for perimeter
                perimeter_node_name = f"{shape.name}_Umfang"
                self.netzwerk.create_wort_if_not_exists(
                    lemma=perimeter_node_name,
                    pos="NOUN",
                    value=perimeter,
                    unit="einheiten",
                )
                self.netzwerk.assert_relation(
                    from_lemma=shape.name,
                    to_lemma=perimeter_node_name,
                    relation_type="HAS_PROPERTY",
                )

            # Step 4: Store vertices if polygon
            if isinstance(shape, Polygon) and shape.vertices:
                for i, vertex in enumerate(shape.vertices):
                    vertex_node_name = f"{shape.name}_Ecke_{i}"
                    self.netzwerk.create_wort_if_not_exists(
                        lemma=vertex_node_name,
                        pos="NOUN",
                        x=vertex.x,
                        y=vertex.y,
                        index=i,
                        type="Vertex",
                    )
                    self.netzwerk.assert_relation(
                        from_lemma=shape.name,
                        to_lemma=vertex_node_name,
                        relation_type="HAS_VERTEX",
                    )

            logger.info("Shape created successfully: %s", shape.name)
            return True

        except Exception as e:
            logger.error(
                "Failed to create shape %s: %s", shape.name, str(e), exc_info=True
            )
            return False

    def get_shape(self, shape_name: str) -> Optional[GeometricShape]:
        """
        Retrieve a geometric shape from the knowledge graph.

        Args:
            shape_name: Name of the shape

        Returns:
            GeometricShape instance if found, None otherwise
        """
        try:
            # Query shape node
            shape_node = self.netzwerk.find_wort_node(shape_name)

            if not shape_node:
                logger.warning("Shape not found: %s", shape_name)
                return None

            # Extract properties
            props = dict(shape_node)
            shape_type = props.get("shape_type", "Unknown")

            # Reconstruct shape based on type
            if shape_type == "Dreieck":
                # Get vertices
                vertices = self._get_shape_vertices(shape_name)
                return Triangle(
                    name=shape_name,
                    vertices=vertices,
                    properties={
                        k: v
                        for k, v in props.items()
                        if k not in ["shape_type", "type", "num_sides"]
                    },
                )

            elif shape_type == "Viereck":
                vertices = self._get_shape_vertices(shape_name)
                return Quadrilateral(
                    name=shape_name,
                    vertices=vertices,
                    properties={
                        k: v
                        for k, v in props.items()
                        if k not in ["shape_type", "type", "num_sides"]
                    },
                )

            elif shape_type == "Kreis":
                center = None
                if "center_x" in props and "center_y" in props:
                    center = Position(props["center_x"], props["center_y"])

                return Circle(
                    name=shape_name,
                    center=center,
                    radius=props.get("radius", 0.0),
                    properties={
                        k: v
                        for k, v in props.items()
                        if k
                        not in ["shape_type", "type", "radius", "center_x", "center_y"]
                    },
                )

            else:
                # Generic polygon
                num_sides = props.get("num_sides", 0)
                vertices = self._get_shape_vertices(shape_name)
                return Polygon(
                    name=shape_name,
                    shape_type=shape_type,
                    num_sides=num_sides,
                    vertices=vertices,
                    properties={
                        k: v
                        for k, v in props.items()
                        if k not in ["shape_type", "type", "num_sides"]
                    },
                )

        except Exception as e:
            logger.error("Error retrieving shape %s: %s", shape_name, str(e))
            return None

    def _get_shape_vertices(self, shape_name: str) -> List[Position]:
        """
        Get vertices of a shape from the knowledge graph.

        Args:
            shape_name: Name of the shape

        Returns:
            List of vertex positions sorted by index
        """
        try:
            # Query HAS_VERTEX relations
            query = """
            MATCH (shape)-[:HAS_VERTEX]->(vertex)
            WHERE shape.lemma = $shape_name
            RETURN vertex.x as x, vertex.y as y, vertex.index as index
            ORDER BY vertex.index
            """

            result = self.netzwerk.session.run(query, shape_name=shape_name)
            vertices = [Position(record["x"], record["y"]) for record in result]

            return vertices

        except Exception as e:
            logger.error("Error getting vertices for %s: %s", shape_name, str(e))
            return []

    def classify_shape(
        self,
        num_sides: Optional[int] = None,
        vertices: Optional[List[Position]] = None,
        properties: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Classify a shape based on its properties.

        Args:
            num_sides: Number of sides (for polygons)
            vertices: Vertex positions (for detailed classification)
            properties: Additional properties

        Returns:
            Shape classification as string
        """
        properties = properties or {}

        # Classify by number of sides
        if num_sides is not None:
            if num_sides == 3:
                return "Dreieck"
            elif num_sides == 4:
                # Further classify quadrilaterals
                if vertices and len(vertices) == 4:
                    quad = Quadrilateral(name="temp", vertices=vertices)
                    if quad.is_rectangle():
                        return "Rechteck"
                return "Viereck"
            elif num_sides == 5:
                return "Fünfeck"
            elif num_sides == 6:
                return "Sechseck"
            else:
                return f"Polygon_{num_sides}"

        # Classify by properties
        if properties.get("radius"):
            return "Kreis"

        return "Unbekannte Form"

    def calculate_shape_properties(self, shape_name: str) -> Dict[str, float]:
        """
        Calculate geometric properties of a shape.

        Args:
            shape_name: Name of the shape

        Returns:
            Dictionary with calculated properties (area, perimeter, etc.)
        """
        try:
            shape = self.get_shape(shape_name)
            if not shape:
                logger.warning("Shape not found: %s", shape_name)
                return {}

            properties = {}

            area = shape.calculate_area()
            if area is not None:
                properties["area"] = area

            perimeter = shape.calculate_perimeter()
            if perimeter is not None:
                properties["perimeter"] = perimeter

            # Shape-specific properties
            if isinstance(shape, Quadrilateral):
                properties["is_rectangle"] = shape.is_rectangle()

            if isinstance(shape, Circle):
                properties["diameter"] = 2 * shape.radius if shape.radius > 0 else 0

            logger.debug("Calculated properties for %s: %s", shape_name, properties)
            return properties

        except Exception as e:
            logger.error("Error calculating properties for %s: %s", shape_name, str(e))
            return {}

    # ========================================================================
    # Spatial Constraints (Phase 4.1)
    # ========================================================================

    def add_spatial_constraint(
        self,
        constraint_name: str,
        objects: List[str],
        constraint_predicate: Callable[[Dict[str, Position]], bool],
        description: str = "",
    ) -> bool:
        """
        Add a spatial constraint for objects on a grid.

        Args:
            constraint_name: Unique identifier for this constraint
            objects: List of object names involved
            constraint_predicate: Function that takes {object: position} dict and returns True if valid
            description: Human-readable description

        Returns:
            True if constraint added successfully

        Example:
            # Constraint: "König und Turm dürfen nicht auf gleicher Position sein"
            reasoner.add_spatial_constraint(
                "no_same_position",
                ["König", "Turm"],
                lambda pos: pos["König"] != pos["Turm"],
                "König und Turm nicht auf gleicher Position"
            )
        """
        try:
            if not hasattr(self, "_spatial_constraints"):
                self._spatial_constraints = {}

            self._spatial_constraints[constraint_name] = {
                "objects": objects,
                "predicate": constraint_predicate,
                "description": description,
            }

            logger.info(
                "Added spatial constraint: %s for objects %s", constraint_name, objects
            )
            return True

        except Exception as e:
            logger.error("Error adding constraint %s: %s", constraint_name, str(e))
            return False

    def check_spatial_constraints(
        self, grid_name: str, object_positions: Optional[Dict[str, Position]] = None
    ) -> Tuple[bool, List[str]]:
        """
        Check if current object positions satisfy all spatial constraints.

        Args:
            grid_name: Name of the grid
            object_positions: Optional dict of {object_name: position}.
                            If None, queries current positions from graph.

        Returns:
            Tuple of (all_satisfied, list_of_violations)
        """
        if not hasattr(self, "_spatial_constraints"):
            return True, []

        violations = []

        # Get current positions if not provided
        if object_positions is None:
            object_positions = {}
            # Query all objects on this grid
            # For now, we'll need the caller to provide positions
            logger.warning(
                "check_spatial_constraints requires object_positions parameter"
            )
            return True, []

        # Check each constraint
        for constraint_name, constraint_info in self._spatial_constraints.items():
            objects = constraint_info["objects"]
            predicate = constraint_info["predicate"]

            # Check if all objects in constraint have positions
            relevant_positions = {}
            missing_objects = []

            for obj in objects:
                if obj in object_positions:
                    relevant_positions[obj] = object_positions[obj]
                else:
                    missing_objects.append(obj)

            # Skip constraint if not all objects are placed
            if missing_objects:
                logger.debug(
                    "Skipping constraint %s: missing objects %s",
                    constraint_name,
                    missing_objects,
                )
                continue

            # Check constraint predicate
            try:
                if not predicate(relevant_positions):
                    violations.append(
                        f"Constraint '{constraint_name}' violated: {constraint_info['description']}"
                    )
            except Exception as e:
                logger.error(
                    "Error evaluating constraint %s: %s", constraint_name, str(e)
                )
                violations.append(
                    f"Constraint '{constraint_name}' evaluation error: {str(e)}"
                )

        all_satisfied = len(violations) == 0

        if not all_satisfied:
            logger.warning(
                "Spatial constraints violated: %d violations", len(violations)
            )

        return all_satisfied, violations

    def find_valid_positions(
        self, grid_name: str, objects: List[str], max_solutions: int = 10
    ) -> List[Dict[str, Position]]:
        """
        Find valid positions for objects that satisfy all spatial constraints.

        Uses backtracking search to find assignments that satisfy constraints.

        Args:
            grid_name: Name of the grid
            objects: List of objects to place
            max_solutions: Maximum number of solutions to return

        Returns:
            List of valid position assignments (dicts of {object: position})
        """
        try:
            grid = self.get_grid(grid_name)
            if not grid:
                logger.error("Grid not found: %s", grid_name)
                return []

            solutions = []
            all_positions = grid.get_all_positions()

            # Simple backtracking search
            def backtrack(
                assignment: Dict[str, Position], remaining_objects: List[str]
            ):
                if len(solutions) >= max_solutions:
                    return

                # Base case: all objects assigned
                if not remaining_objects:
                    # Check constraints
                    satisfied, _ = self.check_spatial_constraints(grid_name, assignment)
                    if satisfied:
                        solutions.append(assignment.copy())
                    return

                # Recursive case: assign next object
                obj = remaining_objects[0]
                rest = remaining_objects[1:]

                for pos in all_positions:
                    # Check if position already used
                    if pos in assignment.values():
                        continue

                    # Try assigning this position
                    assignment[obj] = pos

                    # Check if still valid
                    satisfied, _ = self.check_spatial_constraints(grid_name, assignment)
                    if satisfied:
                        backtrack(assignment, rest)

                    # Backtrack
                    del assignment[obj]

            # Start search
            backtrack({}, objects)

            logger.info(
                "Found %d valid position assignments for %d objects",
                len(solutions),
                len(objects),
            )

            return solutions

        except Exception as e:
            logger.error("Error finding valid positions: %s", str(e), exc_info=True)
            return []

    def clear_spatial_constraints(self):
        """Clear all spatial constraints."""
        if hasattr(self, "_spatial_constraints"):
            self._spatial_constraints = {}
            logger.info("Cleared all spatial constraints")

    # ========================================================================
    # Phase 4.2: Movement Planning
    # ========================================================================

    def plan_movement(
        self,
        object_name: str,
        grid_name: str,
        start_pos: Position,
        goal_pos: Position,
        movement_rules: Optional[Callable[[Position, Position], bool]] = None,
        avoid_objects: bool = True,
    ) -> Optional["MovementPlan"]:
        """
        Plan a sequence of moves to get an object from start to goal.

        Args:
            object_name: Name of object to move
            grid_name: Name of grid
            start_pos: Starting position
            goal_pos: Goal position
            movement_rules: Optional function to validate individual moves
            avoid_objects: If True, avoid positions occupied by other objects

        Returns:
            MovementPlan with sequence of actions, or None if no path exists
        """
        try:
            grid = self.get_grid(grid_name)
            if not grid:
                logger.error("Grid not found: %s", grid_name)
                return None

            # Validate start and goal positions
            if not grid.is_valid_position(start_pos) or not grid.is_valid_position(
                goal_pos
            ):
                logger.error("Start or goal position out of bounds")
                return None

            # Get blocked positions if avoiding objects
            blocked = set()
            if avoid_objects:
                for pos in grid.get_all_positions():
                    objects = self.get_objects_at_position(grid_name, pos)
                    if objects and object_name not in objects:
                        blocked.add(pos)

            # Find path using A* with custom movement rules
            path = self._find_path_with_rules(
                grid, start_pos, goal_pos, movement_rules, blocked
            )

            if not path:
                logger.warning("No valid path found from %s to %s", start_pos, goal_pos)
                return None

            # Convert path to movement actions
            actions = []
            for i in range(len(path) - 1):
                from_pos = path[i]
                to_pos = path[i + 1]
                action = MovementAction(
                    object_name=object_name,
                    from_position=from_pos,
                    to_position=to_pos,
                    step_number=i + 1,
                )
                actions.append(action)

            plan = MovementPlan(
                object_name=object_name,
                grid_name=grid_name,
                actions=actions,
                total_steps=len(actions),
                path_length=len(path),
            )

            logger.info(
                "Created movement plan for %s: %d steps from %s to %s",
                object_name,
                len(actions),
                start_pos,
                goal_pos,
            )

            return plan

        except Exception as e:
            logger.error("Error planning movement: %s", str(e), exc_info=True)
            return None

    def _find_path_with_rules(
        self,
        grid: Grid,
        start: Position,
        goal: Position,
        movement_rules: Optional[Callable[[Position, Position], bool]],
        blocked: Set[Position],
    ) -> Optional[List[Position]]:
        """
        A* pathfinding with custom movement rules and blocked positions.

        Args:
            grid: Grid object
            start: Start position
            goal: Goal position
            movement_rules: Optional function (from_pos, to_pos) -> bool
            blocked: Set of blocked positions

        Returns:
            List of positions from start to goal, or None
        """
        from heapq import heappop, heappush

        def heuristic(pos: Position) -> float:
            return pos.distance_to(goal, metric="manhattan")

        # Priority queue: (f_score, position)
        open_set = []
        heappush(open_set, (heuristic(start), start))

        came_from: Dict[Position, Position] = {}
        g_score: Dict[Position, float] = {start: 0}
        f_score: Dict[Position, float] = {start: heuristic(start)}

        while open_set:
            _, current = heappop(open_set)

            if current == goal:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            # Get neighbors based on grid type
            neighbors = self.get_neighbors(grid.name, current)

            for neighbor in neighbors:
                # Skip blocked positions
                if neighbor in blocked and neighbor != goal:
                    continue

                # Check custom movement rules
                if movement_rules and not movement_rules(current, neighbor):
                    continue

                tentative_g = g_score[current] + current.distance_to(neighbor)

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + heuristic(neighbor)
                    f_score[neighbor] = f
                    heappush(open_set, (f, neighbor))

        return None  # No path found

    def validate_movement_plan(
        self, plan: "MovementPlan", check_constraints: bool = True
    ) -> Tuple[bool, List[str]]:
        """
        Validate a movement plan.

        Args:
            plan: MovementPlan to validate
            check_constraints: If True, check spatial constraints

        Returns:
            (is_valid, list_of_errors)
        """
        errors = []

        try:
            grid = self.get_grid(plan.grid_name)
            if not grid:
                errors.append(f"Grid not found: {plan.grid_name}")
                return False, errors

            # Validate each action
            for action in plan.actions:
                # Check positions are valid
                if not grid.is_valid_position(action.from_position):
                    errors.append(
                        f"Step {action.step_number}: Invalid from_position {action.from_position}"
                    )

                if not grid.is_valid_position(action.to_position):
                    errors.append(
                        f"Step {action.step_number}: Invalid to_position {action.to_position}"
                    )

                # Check positions are neighbors
                neighbors = self.get_neighbors(plan.grid_name, action.from_position)
                if action.to_position not in neighbors:
                    errors.append(
                        f"Step {action.step_number}: Position {action.to_position} "
                        f"is not a neighbor of {action.from_position}"
                    )

            # Check action sequence is continuous
            for i in range(len(plan.actions) - 1):
                if plan.actions[i].to_position != plan.actions[i + 1].from_position:
                    errors.append(
                        f"Discontinuous path at step {plan.actions[i].step_number}"
                    )

            # Check spatial constraints if requested
            if check_constraints and hasattr(self, "_spatial_constraints"):
                if self._spatial_constraints:
                    final_pos = plan.actions[-1].to_position if plan.actions else None
                    if final_pos:
                        positions = {plan.object_name: final_pos}
                        satisfied, violations = self.check_spatial_constraints(
                            plan.grid_name, positions
                        )
                        if not satisfied:
                            for v in violations:
                                errors.append(f"Constraint violation: {v}")

            is_valid = len(errors) == 0

            if is_valid:
                logger.info(
                    "Movement plan validated successfully: %s", plan.object_name
                )
            else:
                logger.warning(
                    "Movement plan validation failed with %d errors", len(errors)
                )

            return is_valid, errors

        except Exception as e:
            logger.error("Error validating movement plan: %s", str(e), exc_info=True)
            errors.append(f"Validation error: {str(e)}")
            return False, errors

    def execute_movement_plan(
        self, plan: "MovementPlan", validate: bool = True
    ) -> bool:
        """
        Execute a movement plan by updating object positions in graph.

        Args:
            plan: MovementPlan to execute
            validate: If True, validate plan before execution

        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate if requested
            if validate:
                is_valid, errors = self.validate_movement_plan(plan)
                if not is_valid:
                    logger.error(
                        "Cannot execute invalid plan. Errors: %s", "; ".join(errors)
                    )
                    return False

            # Execute each action
            for action in plan.actions:
                success = self.move_object(
                    action.object_name,
                    plan.grid_name,
                    action.from_position,
                    action.to_position,
                )

                if not success:
                    logger.error(
                        "Failed to execute step %d of movement plan", action.step_number
                    )
                    return False

            logger.info(
                "Successfully executed movement plan: %s moved %d steps",
                plan.object_name,
                plan.total_steps,
            )

            return True

        except Exception as e:
            logger.error("Error executing movement plan: %s", str(e), exc_info=True)
            return False

    def create_movement_rule(
        self,
        name: str,
        rule_function: Callable[[Position, Position], bool],
        description: str = "",
    ) -> bool:
        """
        Create a named movement rule that can be reused.

        Args:
            name: Unique name for the rule
            rule_function: Function (from_pos, to_pos) -> bool
            description: Optional description

        Returns:
            True if successful
        """
        try:
            if not hasattr(self, "_movement_rules"):
                self._movement_rules = {}

            self._movement_rules[name] = {
                "function": rule_function,
                "description": description,
            }

            logger.info("Created movement rule: %s", name)
            return True

        except Exception as e:
            logger.error("Error creating movement rule: %s", str(e), exc_info=True)
            return False

    def get_movement_rule(
        self, name: str
    ) -> Optional[Callable[[Position, Position], bool]]:
        """Get a movement rule by name."""
        if hasattr(self, "_movement_rules") and name in self._movement_rules:
            return self._movement_rules[name]["function"]
        return None

    def plan_multi_object_movement(
        self,
        grid_name: str,
        movements: List[Tuple[str, Position, Position]],
        avoid_collisions: bool = True,
    ) -> Optional[Dict[str, "MovementPlan"]]:
        """
        Plan movements for multiple objects, avoiding collisions.

        Args:
            grid_name: Name of grid
            movements: List of (object_name, start_pos, goal_pos) tuples
            avoid_collisions: If True, ensure objects don't collide

        Returns:
            Dict mapping object_name to MovementPlan, or None if impossible
        """
        try:
            plans = {}
            occupied_positions: Dict[int, Set[Position]] = {}  # step -> positions

            # Plan each movement
            for obj_name, start_pos, goal_pos in movements:
                # Get blocked positions from other planned movements
                blocked = set()
                if avoid_collisions:
                    # Collect all positions that will be occupied during other plans
                    for step_num in occupied_positions:
                        blocked.update(occupied_positions[step_num])

                # Plan this object's movement
                plan = self.plan_movement(
                    obj_name,
                    grid_name,
                    start_pos,
                    goal_pos,
                    avoid_objects=avoid_collisions,
                )

                if not plan:
                    logger.error(
                        "Cannot plan movement for %s, multi-object planning failed",
                        obj_name,
                    )
                    return None

                plans[obj_name] = plan

                # Record occupied positions at each step
                if avoid_collisions:
                    for action in plan.actions:
                        step = action.step_number
                        if step not in occupied_positions:
                            occupied_positions[step] = set()
                        occupied_positions[step].add(action.to_position)

            logger.info("Successfully planned movements for %d objects", len(movements))

            return plans

        except Exception as e:
            logger.error(
                "Error planning multi-object movement: %s", str(e), exc_info=True
            )
            return None

    # ========================================================================
    # Phase 4.3: Rule Learning System for Spatial Domains
    # ========================================================================

    def observe_movement_pattern(
        self,
        object_name: str,
        movements: List[Tuple[Position, Position]],
        pattern_name: Optional[str] = None,
    ) -> bool:
        """
        Observe a sequence of movements and learn the movement pattern.

        Args:
            object_name: Name of object type (e.g., "Knight", "Rook")
            movements: List of (from_pos, to_pos) tuples
            pattern_name: Optional name for the pattern

        Returns:
            True if pattern was learned
        """
        try:
            if not movements:
                return False

            # Analyze movement vectors
            vectors = []
            for from_pos, to_pos in movements:
                dx = to_pos.x - from_pos.x
                dy = to_pos.y - from_pos.y
                vectors.append((dx, dy))

            # Store in knowledge graph as learned pattern
            pattern_id = pattern_name or f"{object_name}_movement_pattern"

            self.netzwerk.create_wort_if_not_exists(pattern_id)
            self.netzwerk.set_wort_attribut(
                pattern_id, "type", "SpatialMovementPattern"
            )
            self.netzwerk.set_wort_attribut(pattern_id, "object_type", object_name)
            self.netzwerk.set_wort_attribut(pattern_id, "num_examples", len(movements))

            # Store observed vectors
            vector_set = set(vectors)
            for i, (dx, dy) in enumerate(vector_set):
                vector_name = f"{pattern_id}_vector_{i}"
                self.netzwerk.create_wort_if_not_exists(vector_name)
                self.netzwerk.set_wort_attribut(vector_name, "dx", dx)
                self.netzwerk.set_wort_attribut(vector_name, "dy", dy)
                self.netzwerk.assert_relation(
                    pattern_id, "HAS_MOVEMENT_VECTOR", vector_name
                )

            logger.info(
                "Learned movement pattern for %s: %d unique vectors from %d examples",
                object_name,
                len(vector_set),
                len(movements),
            )

            return True

        except Exception as e:
            logger.error("Error observing movement pattern: %s", str(e), exc_info=True)
            return False

    def get_learned_movement_pattern(
        self, object_name: str
    ) -> Optional[Callable[[Position, Position], bool]]:
        """
        Retrieve a learned movement pattern as a validation function.

        Args:
            object_name: Name of object type

        Returns:
            Function (from_pos, to_pos) -> bool, or None
        """
        try:
            pattern_id = f"{object_name}_movement_pattern"
            pattern_node = self.netzwerk.find_wort_node(pattern_id)

            if not pattern_node:
                return None

            # Get learned movement vectors
            vectors = []
            query = """
                MATCH (pattern {lemma: $pattern_id})-[:HAS_MOVEMENT_VECTOR]->(vector)
                RETURN vector.dx as dx, vector.dy as dy
            """
            results = self.netzwerk.session.run(query, pattern_id=pattern_id)

            for record in results:
                vectors.append((record["dx"], record["dy"]))

            if not vectors:
                return None

            # Create validation function
            def validate_move(from_pos: Position, to_pos: Position) -> bool:
                dx = to_pos.x - from_pos.x
                dy = to_pos.y - from_pos.y
                return (dx, dy) in vectors

            logger.info(
                "Retrieved learned pattern for %s with %d movement vectors",
                object_name,
                len(vectors),
            )

            return validate_move

        except Exception as e:
            logger.error("Error retrieving learned pattern: %s", str(e), exc_info=True)
            return None

    def observe_spatial_configuration(
        self,
        configuration_name: str,
        objects_and_positions: Dict[str, Position],
        grid_name: Optional[str] = None,
    ) -> bool:
        """
        Observe and learn a spatial configuration pattern.

        Args:
            configuration_name: Name for this configuration
            objects_and_positions: Dict mapping object names to positions
            grid_name: Optional grid context

        Returns:
            True if pattern was learned
        """
        try:
            # Create configuration node
            config_id = f"SpatialConfig_{configuration_name}"
            self.netzwerk.create_wort_if_not_exists(config_id)
            self.netzwerk.set_wort_attribut(config_id, "type", "SpatialConfiguration")
            self.netzwerk.set_wort_attribut(
                config_id, "num_objects", len(objects_and_positions)
            )

            if grid_name:
                self.netzwerk.set_wort_attribut(config_id, "grid", grid_name)

            # Analyze and store relative positions
            objects_list = list(objects_and_positions.items())

            for i, (obj1, pos1) in enumerate(objects_list):
                for j, (obj2, pos2) in enumerate(objects_list):
                    if i >= j:
                        continue

                    # Calculate relative position
                    dx = pos2.x - pos1.x
                    dy = pos2.y - pos1.y
                    distance = pos1.distance_to(pos2, metric="manhattan")

                    # Store relative relationship
                    rel_id = f"{config_id}_{obj1}_{obj2}"
                    self.netzwerk.create_wort_if_not_exists(rel_id)
                    self.netzwerk.set_wort_attribut(rel_id, "object1", obj1)
                    self.netzwerk.set_wort_attribut(rel_id, "object2", obj2)
                    self.netzwerk.set_wort_attribut(rel_id, "dx", dx)
                    self.netzwerk.set_wort_attribut(rel_id, "dy", dy)
                    self.netzwerk.set_wort_attribut(rel_id, "distance", distance)

                    self.netzwerk.assert_relation(
                        config_id, "HAS_RELATIVE_POSITION", rel_id
                    )

            logger.info(
                "Learned spatial configuration '%s' with %d objects",
                configuration_name,
                len(objects_and_positions),
            )

            return True

        except Exception as e:
            logger.error(
                "Error observing spatial configuration: %s", str(e), exc_info=True
            )
            return False

    def detect_spatial_pattern_in_configuration(
        self, objects_and_positions: Dict[str, Position]
    ) -> List[str]:
        """
        Detect which learned patterns match the current configuration.

        Args:
            objects_and_positions: Current object positions

        Returns:
            List of matching pattern names
        """
        try:
            matches = []

            # Query all stored configurations
            query = """
                MATCH (config)
                WHERE config.type = 'SpatialConfiguration'
                RETURN config.lemma as name, config.num_objects as num_objects
            """
            results = self.netzwerk.session.run(query)

            for record in results:
                config_name = record["name"]
                expected_num = record["num_objects"]

                # Quick filter by number of objects
                if len(objects_and_positions) != expected_num:
                    continue

                # Check if relative positions match
                if self._check_configuration_match(config_name, objects_and_positions):
                    # Extract original name (remove "SpatialConfig_" prefix)
                    pattern_name = config_name.replace("SpatialConfig_", "")
                    matches.append(pattern_name)

            logger.info(
                "Detected %d matching spatial patterns in configuration", len(matches)
            )

            return matches

        except Exception as e:
            logger.error("Error detecting spatial patterns: %s", str(e), exc_info=True)
            return []

    def _check_configuration_match(
        self,
        config_id: str,
        objects_and_positions: Dict[str, Position],
        tolerance: float = 0.5,
    ) -> bool:
        """
        Check if current configuration matches a stored pattern.

        Args:
            config_id: ID of stored configuration
            objects_and_positions: Current positions
            tolerance: Allowed deviation in relative positions

        Returns:
            True if configuration matches pattern
        """
        try:
            # Get stored relative positions
            query = """
                MATCH (config {lemma: $config_id})-[:HAS_RELATIVE_POSITION]->(rel)
                RETURN rel.object1 as obj1, rel.object2 as obj2,
                       rel.dx as dx, rel.dy as dy
            """
            results = self.netzwerk.session.run(query, config_id=config_id)

            # Check each relative position
            for record in results:
                obj1 = record["obj1"]
                obj2 = record["obj2"]
                expected_dx = record["dx"]
                expected_dy = record["dy"]

                # Find matching objects in current configuration
                # (Simple approach: match by type/name pattern)
                current_obj1 = self._find_matching_object(obj1, objects_and_positions)
                current_obj2 = self._find_matching_object(obj2, objects_and_positions)

                if not current_obj1 or not current_obj2:
                    return False

                pos1 = objects_and_positions[current_obj1]
                pos2 = objects_and_positions[current_obj2]

                actual_dx = pos2.x - pos1.x
                actual_dy = pos2.y - pos1.y

                # Check if within tolerance
                if (
                    abs(actual_dx - expected_dx) > tolerance
                    or abs(actual_dy - expected_dy) > tolerance
                ):
                    return False

            return True

        except Exception as e:
            logger.error("Error checking configuration match: %s", str(e))
            return False

    def _find_matching_object(
        self, pattern_obj: str, current_objects: Dict[str, Position]
    ) -> Optional[str]:
        """Find an object in current configuration that matches the pattern."""
        # Simple implementation: direct match or type-based match
        if pattern_obj in current_objects:
            return pattern_obj

        # Try to match by type (e.g., "König1" matches "König2")
        for obj_name in current_objects:
            # Remove trailing numbers for type comparison
            pattern_type = "".join(c for c in pattern_obj if not c.isdigit())
            obj_type = "".join(c for c in obj_name if not c.isdigit())

            if pattern_type == obj_type:
                return obj_name

        return None

    def learn_spatial_rule_from_examples(
        self,
        rule_name: str,
        positive_examples: List[Dict[str, Position]],
        negative_examples: Optional[List[Dict[str, Position]]] = None,
    ) -> bool:
        """
        Learn a spatial rule from positive and negative examples.

        Args:
            rule_name: Name for the rule
            positive_examples: Examples that satisfy the rule
            negative_examples: Examples that violate the rule (optional)

        Returns:
            True if rule was learned
        """
        try:
            if not positive_examples:
                return False

            # Analyze positive examples to find common patterns
            # Simple approach: find constraints that all positives satisfy

            # Extract common structure
            len(positive_examples[0])

            # Find constraints that hold for all positive examples
            constraints = []

            # Check for consistent relative positions
            object_pairs = list(positive_examples[0].keys())

            for i, obj1 in enumerate(object_pairs):
                for j, obj2 in enumerate(object_pairs):
                    if i >= j:
                        continue

                    # Check if this pair has consistent relative position across examples
                    relative_positions = []
                    for example in positive_examples:
                        if obj1 in example and obj2 in example:
                            pos1 = example[obj1]
                            pos2 = example[obj2]
                            dx = pos2.x - pos1.x
                            dy = pos2.y - pos1.y
                            relative_positions.append((dx, dy))

                    # If all examples have same relative position, it's a constraint
                    if len(set(relative_positions)) == 1:
                        dx, dy = relative_positions[0]
                        constraints.append(
                            {
                                "type": "relative_position",
                                "obj1": obj1,
                                "obj2": obj2,
                                "dx": dx,
                                "dy": dy,
                            }
                        )

            # Store learned rule
            rule_id = f"SpatialRule_{rule_name}"
            self.netzwerk.create_wort_if_not_exists(rule_id)
            self.netzwerk.set_wort_attribut(rule_id, "type", "LearnedSpatialRule")
            self.netzwerk.set_wort_attribut(
                rule_id, "num_constraints", len(constraints)
            )
            self.netzwerk.set_wort_attribut(
                rule_id, "num_examples", len(positive_examples)
            )

            # Store constraints
            for i, constraint in enumerate(constraints):
                constraint_id = f"{rule_id}_constraint_{i}"
                self.netzwerk.create_wort_if_not_exists(constraint_id)

                for key, value in constraint.items():
                    self.netzwerk.set_wort_attribut(constraint_id, key, value)

                self.netzwerk.assert_relation(rule_id, "HAS_CONSTRAINT", constraint_id)

            logger.info(
                "Learned spatial rule '%s' with %d constraints from %d positive examples",
                rule_name,
                len(constraints),
                len(positive_examples),
            )

            return True

        except Exception as e:
            logger.error("Error learning spatial rule: %s", str(e), exc_info=True)
            return False


# ============================================================================
# Data Structures for Movement Planning (Phase 4.2)
# ============================================================================


@dataclass(frozen=True)
class MovementAction:
    """Represents a single movement action."""

    object_name: str
    from_position: Position
    to_position: Position
    step_number: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"Step {self.step_number}: Move {self.object_name} "
            f"from {self.from_position} to {self.to_position}"
        )


@dataclass
class MovementPlan:
    """Represents a complete movement plan for an object."""

    object_name: str
    grid_name: str
    actions: List[MovementAction]
    total_steps: int
    path_length: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"MovementPlan({self.object_name} on {self.grid_name}: "
            f"{self.total_steps} steps, path length {self.path_length})"
        )

    def get_final_position(self) -> Optional[Position]:
        """Get the final position after executing all actions."""
        if self.actions:
            return self.actions[-1].to_position
        return None

    def get_path(self) -> List[Position]:
        """Get the complete path as a list of positions."""
        if not self.actions:
            return []
        path = [self.actions[0].from_position]
        for action in self.actions:
            path.append(action.to_position)
        return path


# Alias for backwards compatibility
SpatialReasoningEngine = SpatialReasoner
