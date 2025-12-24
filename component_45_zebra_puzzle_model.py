"""
component_45_zebra_puzzle_model.py
==================================
Data models for Zebra (Einstein) puzzles.

This module defines the data structures used to represent Zebra puzzles:
- ZebraConstraintType: Enum of constraint types (SAME_ENTITY, DIRECTLY_LEFT_OF, etc.)
- ZebraConstraint: A single constraint in the puzzle
- ZebraPuzzle: Complete puzzle definition with categories, values, and constraints

The Zebra puzzle is a classic logic puzzle with multiple categories (nationality,
house color, pet, drink, cigarette brand) where each entity has exactly one value
in each category, and clues constrain the relationships between values.

Example Puzzle Structure:
- 5 houses in a row (positions 1-5)
- 5 nationalities: Brit, Swede, Dane, Norwegian, German
- 5 house colors: Red, Green, White, Yellow, Blue
- 5 drinks: Tea, Coffee, Milk, Beer, Water
- 5 pets: Dog, Bird, Cat, Horse, Zebra
- 5 cigarettes: Pall Mall, Dunhill, Blend, Blue Master, Prince

Author: KAI Development Team
Date: 2025-12-24
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class ZebraConstraintType(Enum):
    """Types of constraints in Zebra puzzles."""

    # Two values belong to the same entity (same house position)
    # "Der Brite lebt im roten Haus" -> SAME_ENTITY(brit, red)
    SAME_ENTITY = "same_entity"

    # One entity is directly left of another
    # "Das gruene Haus steht direkt links vom weissen Haus"
    # -> DIRECTLY_LEFT_OF(green, white)
    DIRECTLY_LEFT_OF = "directly_left_of"

    # One entity is directly right of another
    # -> DIRECTLY_RIGHT_OF(white, green) equivalent to DIRECTLY_LEFT_OF(green, white)
    DIRECTLY_RIGHT_OF = "directly_right_of"

    # Two entities are adjacent (neighbors)
    # "Die Person, die Blend raucht, lebt neben der Person mit der Katze"
    # -> ADJACENT_TO(blend, cat)
    ADJACENT_TO = "adjacent_to"

    # Entity is at a specific position (1-indexed)
    # "Der Norweger lebt im ersten Haus" -> AT_POSITION(norwegian, 1)
    # "Die Person im mittleren Haus trinkt Milch" -> AT_POSITION(milk, 3)
    AT_POSITION = "at_position"

    # Entity is left of another (not necessarily adjacent)
    # Used less commonly, but supported
    LEFT_OF = "left_of"

    # Entity is right of another (not necessarily adjacent)
    RIGHT_OF = "right_of"

    # Two values belong to DIFFERENT entities (negation constraint)
    # "Clara ist nicht die Anwaeltin" -> DIFFERENT_ENTITY(clara, anwaeltin)
    DIFFERENT_ENTITY = "different_entity"

    # Ordering/Comparison constraints for transitive ordering puzzles
    # Position 1 = greatest, Position N = smallest
    # "A ist groesser als B" -> GREATER_THAN(A, B) means pos(A) < pos(B)
    GREATER_THAN = "greater_than"

    # "A ist kleiner als B" -> LESS_THAN(A, B) means pos(A) > pos(B)
    LESS_THAN = "less_than"


@dataclass
class ZebraConstraint:
    """
    A single constraint in a Zebra puzzle.

    Attributes:
        constraint_type: Type of constraint (SAME_ENTITY, ADJACENT_TO, etc.)
        values: List of values involved (1-2 depending on type)
        position: Optional position for AT_POSITION constraints
        original_text: Original German text for proof tree
    """

    constraint_type: ZebraConstraintType
    values: List[str]
    position: Optional[int] = None
    original_text: str = ""

    def __post_init__(self):
        """Validate constraint."""
        if self.constraint_type == ZebraConstraintType.AT_POSITION:
            if self.position is None:
                raise ValueError("AT_POSITION constraint requires position")
            if len(self.values) != 1:
                raise ValueError("AT_POSITION constraint requires exactly 1 value")
        elif self.constraint_type in (
            ZebraConstraintType.SAME_ENTITY,
            ZebraConstraintType.DIRECTLY_LEFT_OF,
            ZebraConstraintType.DIRECTLY_RIGHT_OF,
            ZebraConstraintType.ADJACENT_TO,
            ZebraConstraintType.LEFT_OF,
            ZebraConstraintType.RIGHT_OF,
            ZebraConstraintType.DIFFERENT_ENTITY,
            ZebraConstraintType.GREATER_THAN,
            ZebraConstraintType.LESS_THAN,
        ):
            if len(self.values) != 2:
                raise ValueError(
                    f"{self.constraint_type.value} constraint requires exactly 2 values"
                )

    def __str__(self) -> str:
        """Human-readable representation."""
        if self.constraint_type == ZebraConstraintType.AT_POSITION:
            return f"{self.values[0]} at position {self.position}"
        else:
            return f"{self.values[0]} {self.constraint_type.value} {self.values[1]}"


@dataclass
class ZebraPuzzle:
    """
    Complete Zebra puzzle definition.

    Attributes:
        num_entities: Number of entities (houses) - typically 5
        categories: Dict mapping category name to list of values
                   e.g., {"nationality": ["brit", "swede", ...], "color": ["red", ...]}
        constraints: List of ZebraConstraint objects
        question: Optional question to answer
        original_text: Original puzzle text for reference
    """

    num_entities: int
    categories: Dict[str, List[str]]
    constraints: List[ZebraConstraint] = field(default_factory=list)
    question: Optional[str] = None
    original_text: str = ""

    def __post_init__(self):
        """Validate puzzle structure."""
        for category, values in self.categories.items():
            if len(values) != self.num_entities:
                raise ValueError(
                    f"Category '{category}' has {len(values)} values, "
                    f"expected {self.num_entities}"
                )

    def get_all_values(self) -> Set[str]:
        """Get all values across all categories."""
        all_values = set()
        for values in self.categories.values():
            all_values.update(values)
        return all_values

    def get_category_for_value(self, value: str) -> Optional[str]:
        """Find which category a value belongs to."""
        value_lower = value.lower()
        for category, values in self.categories.items():
            if value_lower in [v.lower() for v in values]:
                return category
        return None

    def get_positions(self) -> List[int]:
        """Get list of valid positions (1-indexed)."""
        return list(range(1, self.num_entities + 1))

    def __str__(self) -> str:
        """Human-readable summary."""
        lines = [
            f"ZebraPuzzle with {self.num_entities} entities",
            f"Categories: {list(self.categories.keys())}",
            f"Constraints: {len(self.constraints)}",
        ]
        if self.question:
            lines.append(f"Question: {self.question}")
        return "\n".join(lines)


@dataclass
class ZebraSolution:
    """
    Solution to a Zebra puzzle.

    Attributes:
        assignments: Dict mapping position -> category -> value
                    e.g., {1: {"nationality": "norwegian", "color": "yellow", ...}, ...}
        confidence: Confidence score (1.0 for exact CSP solution)
        proof_steps: List of reasoning steps for proof tree
    """

    assignments: Dict[int, Dict[str, str]] = field(default_factory=dict)
    confidence: float = 1.0
    proof_steps: List[str] = field(default_factory=list)

    def get_value_at_position(self, position: int, category: str) -> Optional[str]:
        """Get the value for a category at a specific position."""
        if position in self.assignments:
            return self.assignments[position].get(category)
        return None

    def find_position_for_value(self, value: str) -> Optional[int]:
        """Find which position has a given value."""
        value_lower = value.lower()
        for pos, categories in self.assignments.items():
            for cat_value in categories.values():
                if cat_value.lower() == value_lower:
                    return pos
        return None

    def get_entity_with_value(self, value: str, target_category: str) -> Optional[str]:
        """
        Find what value in target_category belongs to same entity as given value.

        Example: get_entity_with_value("zebra", "nationality") -> "german"
        (The German owns the zebra)
        """
        position = self.find_position_for_value(value)
        if position is not None:
            return self.get_value_at_position(position, target_category)
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "assignments": self.assignments,
            "confidence": self.confidence,
            "proof_steps": self.proof_steps,
        }

    def __str__(self) -> str:
        """Human-readable solution."""
        lines = ["Zebra Puzzle Solution:"]
        for pos in sorted(self.assignments.keys()):
            cats = self.assignments[pos]
            cat_str = ", ".join(f"{k}={v}" for k, v in cats.items())
            lines.append(f"  Position {pos}: {cat_str}")
        return "\n".join(lines)
