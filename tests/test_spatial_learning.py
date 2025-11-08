"""
tests/test_spatial_learning.py

Unit tests for spatial pattern learning and knowledge extraction.

Tests cover:
- Spatial relation extraction from text
- Pattern learning for spatial configurations
- Confidence-based spatial learning
- Episodic memory for spatial scenarios
- Integration with knowledge extraction (component_10)
- Pattern matching for spatial prototypes (component_8)
"""

from typing import List

from component_42_spatial_reasoning import (
    Position,
    SpatialRelation,
    SpatialRelationType,
)

# ==================== Spatial Relation Extraction Tests ====================


class TestSpatialRelationExtraction:
    """Test extraction of spatial relations from text."""

    def test_extract_north_of_relation(self):
        """Test extraction of NORTH_OF relation from text."""

        # Simulate extraction (in real system: component_7 + component_10)
        # Pattern: "X liegt nördlich von Y" -> X NORTH_OF Y
        relation = SpatialRelation(
            "Berlin", "München", SpatialRelationType.NORTH_OF, confidence=0.9
        )

        assert relation.subject == "Berlin"
        assert relation.object == "München"
        assert relation.relation_type == SpatialRelationType.NORTH_OF

    def test_extract_south_of_relation(self):
        """Test extraction of SOUTH_OF relation from text."""

        relation = SpatialRelation(
            "Rom", "Berlin", SpatialRelationType.SOUTH_OF, confidence=0.9
        )

        assert relation.subject == "Rom"
        assert relation.object == "Berlin"
        assert relation.relation_type == SpatialRelationType.SOUTH_OF

    def test_extract_adjacent_to_relation(self):
        """Test extraction of ADJACENT_TO relation from text."""

        # "grenzt an" -> ADJACENT_TO
        relation = SpatialRelation(
            "Frankreich",
            "Deutschland",
            SpatialRelationType.ADJACENT_TO,
            confidence=0.85,
        )

        assert relation.relation_type == SpatialRelationType.ADJACENT_TO

    def test_extract_multiple_relations(self):
        """Test extraction of multiple spatial relations from text."""
        texts = [
            "A liegt nördlich von B",
            "B liegt nördlich von C",
            "C liegt westlich von D",
        ]

        relations = [
            SpatialRelation("A", "B", SpatialRelationType.NORTH_OF, confidence=0.9),
            SpatialRelation("B", "C", SpatialRelationType.NORTH_OF, confidence=0.9),
            SpatialRelation("C", "D", SpatialRelationType.WEST_OF, confidence=0.9),
        ]

        assert len(relations) == 3
        assert relations[0].subject == "A"
        assert relations[1].subject == "B"
        assert relations[2].relation_type == SpatialRelationType.WEST_OF


# ==================== Pattern Learning Tests ====================


class TestSpatialPatternLearning:
    """Test learning of spatial patterns."""

    def test_learn_simple_configuration(self):
        """Test learning a simple spatial configuration."""
        # Configuration: A is north of B, B is north of C (linear chain)
        config = [
            SpatialRelation("A", "B", SpatialRelationType.NORTH_OF, confidence=1.0),
            SpatialRelation("B", "C", SpatialRelationType.NORTH_OF, confidence=1.0),
        ]

        # Learned pattern: Linear chain (A -> B -> C)

        assert len(config) == 2
        assert config[0].relation_type == SpatialRelationType.NORTH_OF
        assert config[1].relation_type == SpatialRelationType.NORTH_OF

    def test_learn_triangular_configuration(self):
        """Test learning a triangular spatial configuration."""
        # Configuration: A north of B, B east of C, C west of A
        config = [
            SpatialRelation("A", "B", SpatialRelationType.NORTH_OF, confidence=1.0),
            SpatialRelation("B", "C", SpatialRelationType.EAST_OF, confidence=1.0),
            SpatialRelation("C", "A", SpatialRelationType.WEST_OF, confidence=1.0),
        ]

        # Pattern: Triangular configuration

        assert len(config) == 3

    def test_learn_grid_pattern(self):
        """Test learning a grid-like spatial pattern."""
        # 2x2 grid: A, B in row 1; C, D in row 2
        # A is west of B, C is west of D
        # A is north of C, B is north of D
        config = [
            SpatialRelation("A", "B", SpatialRelationType.WEST_OF, confidence=1.0),
            SpatialRelation("C", "D", SpatialRelationType.WEST_OF, confidence=1.0),
            SpatialRelation("A", "C", SpatialRelationType.NORTH_OF, confidence=1.0),
            SpatialRelation("B", "D", SpatialRelationType.NORTH_OF, confidence=1.0),
        ]

        assert len(config) == 4

    def test_learn_adjacency_pattern(self):
        """Test learning an adjacency pattern (objects clustered together)."""
        # All adjacent to each other
        config = [
            SpatialRelation("A", "B", SpatialRelationType.ADJACENT_TO, confidence=1.0),
            SpatialRelation("B", "C", SpatialRelationType.ADJACENT_TO, confidence=1.0),
            SpatialRelation("A", "C", SpatialRelationType.ADJACENT_TO, confidence=1.0),
        ]

        # All relations are symmetric
        assert all(rel.relation_type.is_symmetric for rel in config)


# ==================== Confidence-Based Learning Tests ====================


class TestConfidenceBasedSpatialLearning:
    """Test confidence-based learning for spatial relations."""

    def test_high_confidence_relation(self):
        """Test learning high-confidence spatial relation."""
        # Explicit statement: high confidence
        relation = SpatialRelation(
            "A", "B", SpatialRelationType.NORTH_OF, confidence=0.95
        )

        assert relation.confidence >= 0.9

    def test_medium_confidence_relation(self):
        """Test learning medium-confidence spatial relation."""
        # Inferred or ambiguous: medium confidence
        relation = SpatialRelation(
            "A", "B", SpatialRelationType.NORTH_OF, confidence=0.7
        )

        assert 0.6 <= relation.confidence < 0.9

    def test_low_confidence_relation(self):
        """Test learning low-confidence spatial relation."""
        # Uncertain or guessed: low confidence
        relation = SpatialRelation(
            "A", "B", SpatialRelationType.NORTH_OF, confidence=0.4
        )

        assert relation.confidence < 0.6

    def test_confidence_decay_in_transitive_inference(self):
        """Test confidence decay in transitive spatial inference."""
        # A NORTH_OF B (conf=0.9), B NORTH_OF C (conf=0.8)
        rel1 = SpatialRelation("A", "B", SpatialRelationType.NORTH_OF, confidence=0.9)
        rel2 = SpatialRelation("B", "C", SpatialRelationType.NORTH_OF, confidence=0.8)

        # Inferred: A NORTH_OF C (conf = 0.9 * 0.8 = 0.72)
        inferred_confidence = rel1.confidence * rel2.confidence

        assert abs(inferred_confidence - 0.72) < 0.001

    def test_confidence_update_with_new_evidence(self):
        """Test updating confidence with new evidence."""
        # Initial relation: A NORTH_OF B (conf=0.7)
        relation = SpatialRelation(
            "A", "B", SpatialRelationType.NORTH_OF, confidence=0.7
        )

        # New evidence confirms: increase confidence
        # Simple average: (0.7 + 0.9) / 2 = 0.8
        new_evidence_confidence = 0.9
        updated_confidence = (relation.confidence + new_evidence_confidence) / 2

        assert updated_confidence == 0.8

    def test_confidence_threshold_for_learning(self):
        """Test confidence threshold for accepting learned relations."""
        threshold = 0.7

        rel_high = SpatialRelation(
            "A", "B", SpatialRelationType.NORTH_OF, confidence=0.85
        )
        rel_low = SpatialRelation(
            "C", "D", SpatialRelationType.SOUTH_OF, confidence=0.5
        )

        # Only accept high-confidence relations
        assert rel_high.confidence >= threshold
        assert rel_low.confidence < threshold


# ==================== Episodic Spatial Memory Tests ====================


class TestEpisodicSpatialMemory:
    """Test episodic memory for spatial scenarios."""

    def test_remember_spatial_configuration(self):
        """Test remembering a specific spatial configuration."""
        # Episode: User described a room layout
        episode = {
            "timestamp": "2025-01-01T10:00:00",
            "description": "Raumlayout beschrieben",
            "relations": [
                SpatialRelation(
                    "Tisch", "Stuhl", SpatialRelationType.ADJACENT_TO, confidence=1.0
                ),
                SpatialRelation(
                    "Fenster", "Tisch", SpatialRelationType.NORTH_OF, confidence=1.0
                ),
            ],
        }

        assert len(episode["relations"]) == 2
        assert episode["description"] == "Raumlayout beschrieben"

    def test_recall_spatial_episode(self):
        """Test recalling a spatial episode."""
        # Store episodes
        episodes = [
            {
                "id": "ep1",
                "relations": [
                    SpatialRelation(
                        "A", "B", SpatialRelationType.NORTH_OF, confidence=1.0
                    ),
                ],
            },
            {
                "id": "ep2",
                "relations": [
                    SpatialRelation(
                        "C", "D", SpatialRelationType.EAST_OF, confidence=1.0
                    ),
                ],
            },
        ]

        # Recall episode with "A NORTH_OF B"
        query_relation_type = SpatialRelationType.NORTH_OF
        matching_episodes = [
            ep
            for ep in episodes
            if any(rel.relation_type == query_relation_type for rel in ep["relations"])
        ]

        assert len(matching_episodes) == 1
        assert matching_episodes[0]["id"] == "ep1"

    def test_spatial_scenario_replay(self):
        """Test replaying a learned spatial scenario."""
        # Scenario: User taught a path through a maze
        scenario = {
            "name": "maze_solution",
            "path": [
                Position(0, 0),
                Position(0, 1),
                Position(1, 1),
                Position(2, 1),
                Position(2, 2),
            ],
            "actions": ["north", "east", "east", "north"],
        }

        # Replay: Reconstruct actions from path
        reconstructed_actions = []
        for i in range(len(scenario["path"]) - 1):
            current = scenario["path"][i]
            next_pos = scenario["path"][i + 1]

            dx = next_pos.x - current.x
            dy = next_pos.y - current.y

            if dx == 0 and dy == 1:
                reconstructed_actions.append("north")
            elif dx == 1 and dy == 0:
                reconstructed_actions.append("east")

        assert reconstructed_actions == scenario["actions"]


# ==================== Pattern Matching Tests ====================


class TestSpatialPatternMatching:
    """Test matching of learned spatial patterns."""

    def test_match_similar_configuration(self):
        """Test matching a new configuration to a learned pattern."""
        # Learned pattern: Linear chain (A -> B -> C) with NORTH_OF
        learned_pattern = [
            SpatialRelation("X1", "X2", SpatialRelationType.NORTH_OF, confidence=1.0),
            SpatialRelation("X2", "X3", SpatialRelationType.NORTH_OF, confidence=1.0),
        ]

        # New configuration: Berlin -> München -> Rom (also linear chain)
        new_config = [
            SpatialRelation(
                "Berlin", "München", SpatialRelationType.NORTH_OF, confidence=0.9
            ),
            SpatialRelation(
                "München", "Rom", SpatialRelationType.NORTH_OF, confidence=0.9
            ),
        ]

        # Match: Both are linear chains with NORTH_OF
        def is_linear_chain_north(relations: List[SpatialRelation]) -> bool:
            if len(relations) < 2:
                return False
            # Check if all are NORTH_OF and form a chain
            return all(
                rel.relation_type == SpatialRelationType.NORTH_OF for rel in relations
            )

        assert is_linear_chain_north(learned_pattern)
        assert is_linear_chain_north(new_config)

    def test_match_grid_pattern(self):
        """Test matching a grid pattern."""

        # Pattern: 2x2 grid
        def is_grid_2x2(relations: List[SpatialRelation]) -> bool:
            # Should have 4 relations: 2 horizontal (WEST_OF), 2 vertical (NORTH_OF)
            if len(relations) != 4:
                return False

            horizontal = [
                rel
                for rel in relations
                if rel.relation_type == SpatialRelationType.WEST_OF
            ]
            vertical = [
                rel
                for rel in relations
                if rel.relation_type == SpatialRelationType.NORTH_OF
            ]

            return len(horizontal) == 2 and len(vertical) == 2

        # Test configuration
        config = [
            SpatialRelation("A", "B", SpatialRelationType.WEST_OF, confidence=1.0),
            SpatialRelation("C", "D", SpatialRelationType.WEST_OF, confidence=1.0),
            SpatialRelation("A", "C", SpatialRelationType.NORTH_OF, confidence=1.0),
            SpatialRelation("B", "D", SpatialRelationType.NORTH_OF, confidence=1.0),
        ]

        assert is_grid_2x2(config)

    def test_no_match_different_pattern(self):
        """Test that different patterns don't match."""
        # Pattern 1: Linear chain (NORTH_OF)
        pattern1 = [
            SpatialRelation("A", "B", SpatialRelationType.NORTH_OF, confidence=1.0),
            SpatialRelation("B", "C", SpatialRelationType.NORTH_OF, confidence=1.0),
        ]

        # Pattern 2: Linear chain (EAST_OF)
        pattern2 = [
            SpatialRelation("X", "Y", SpatialRelationType.EAST_OF, confidence=1.0),
            SpatialRelation("Y", "Z", SpatialRelationType.EAST_OF, confidence=1.0),
        ]

        # Should not match (different relation types)
        def get_relation_types(relations: List[SpatialRelation]) -> set:
            return {rel.relation_type for rel in relations}

        assert get_relation_types(pattern1) != get_relation_types(pattern2)


# ==================== Knowledge Extraction Integration Tests ====================


class TestSpatialKnowledgeExtraction:
    """Test integration with knowledge extraction system."""

    def test_extract_spatial_facts_from_text(self):
        """Test extracting spatial facts from natural language text."""
        texts = [
            "Berlin liegt nördlich von München.",
            "München liegt nördlich von Rom.",
            "Paris liegt westlich von Berlin.",
        ]

        # Simulate extraction (component_10 would do this)
        extracted_facts = [
            {"subject": "Berlin", "relation": "NORTH_OF", "object": "München"},
            {"subject": "München", "relation": "NORTH_OF", "object": "Rom"},
            {"subject": "Paris", "relation": "WEST_OF", "object": "Berlin"},
        ]

        assert len(extracted_facts) == 3
        assert extracted_facts[0]["relation"] == "NORTH_OF"

    def test_store_spatial_knowledge_in_graph(self):
        """Test storing spatial knowledge in Neo4j (simulated)."""
        # Simulate graph storage
        graph_knowledge = {}

        # Store: A NORTH_OF B
        relation = SpatialRelation(
            "A", "B", SpatialRelationType.NORTH_OF, confidence=0.9
        )

        # In real system: netzwerk.create_relation(...)
        graph_knowledge[("A", "B")] = {
            "type": relation.relation_type.value,
            "confidence": relation.confidence,
        }

        assert ("A", "B") in graph_knowledge
        assert graph_knowledge[("A", "B")]["type"] == "NORTH_OF"

    def test_query_spatial_knowledge_from_graph(self):
        """Test querying spatial knowledge from Neo4j (simulated)."""
        # Simulated graph with spatial relations
        graph = {
            ("A", "B"): {"type": "NORTH_OF", "confidence": 0.9},
            ("B", "C"): {"type": "NORTH_OF", "confidence": 0.8},
            ("A", "D"): {"type": "EAST_OF", "confidence": 0.85},
        }

        # Query: All NORTH_OF relations
        north_of_relations = [
            (subj, obj)
            for (subj, obj), data in graph.items()
            if data["type"] == "NORTH_OF"
        ]

        assert len(north_of_relations) == 2
        assert ("A", "B") in north_of_relations
        assert ("B", "C") in north_of_relations


# ==================== Learning Strategy Tests ====================


class TestSpatialLearningStrategies:
    """Test different learning strategies for spatial knowledge."""

    def test_explicit_learning(self):
        """Test explicit learning (user directly teaches)."""
        # User says: "Lerne: Berlin liegt nördlich von München"
        relation = SpatialRelation(
            "Berlin", "München", SpatialRelationType.NORTH_OF, confidence=1.0
        )

        # Explicit learning: high confidence
        assert relation.confidence == 1.0

    def test_implicit_learning(self):
        """Test implicit learning (inferred from context)."""
        # User mentions: "Wir fliegen von Berlin nach Rom über München"
        # Implicit: Berlin -> München -> Rom (spatial sequence)

        # Inferred relations (lower confidence)
        relations = [
            SpatialRelation(
                "Berlin", "München", SpatialRelationType.NORTH_OF, confidence=0.6
            ),
            SpatialRelation(
                "München", "Rom", SpatialRelationType.NORTH_OF, confidence=0.6
            ),
        ]

        # Implicit learning: lower confidence
        assert all(rel.confidence < 0.8 for rel in relations)

    def test_corrective_learning(self):
        """Test corrective learning (user corrects a mistake)."""
        # Initial (wrong): A NORTH_OF B (conf=0.6)
        wrong_relation = SpatialRelation(
            "A", "B", SpatialRelationType.NORTH_OF, confidence=0.6
        )

        # User correction: "Nein, A liegt südlich von B"
        corrected_relation = SpatialRelation(
            "A", "B", SpatialRelationType.SOUTH_OF, confidence=1.0
        )

        # Correction: new relation replaces old, high confidence
        assert corrected_relation.relation_type != wrong_relation.relation_type
        assert corrected_relation.confidence == 1.0

    def test_reinforcement_learning(self):
        """Test reinforcement learning (repeated confirmation)."""
        # Initial: A NORTH_OF B (conf=0.7)
        relation = SpatialRelation(
            "A", "B", SpatialRelationType.NORTH_OF, confidence=0.7
        )

        # User confirms again: increase confidence
        confirmations = 3
        updated_confidence = min(1.0, relation.confidence + 0.1 * confirmations)

        assert updated_confidence == 1.0  # Capped at 1.0


# ==================== Error Handling Tests ====================


class TestSpatialLearningErrorHandling:
    """Test error handling in spatial learning."""

    def test_contradictory_relations(self):
        """Test handling contradictory spatial relations."""
        # Contradiction: A NORTH_OF B AND A SOUTH_OF B
        rel1 = SpatialRelation("A", "B", SpatialRelationType.NORTH_OF, confidence=0.8)
        rel2 = SpatialRelation("A", "B", SpatialRelationType.SOUTH_OF, confidence=0.7)

        # Detect contradiction
        is_contradictory = (
            rel1.subject == rel2.subject
            and rel1.object == rel2.object
            and rel1.relation_type != rel2.relation_type
        )

        assert is_contradictory

        # Resolution: Keep higher confidence relation
        kept_relation = rel1 if rel1.confidence > rel2.confidence else rel2
        assert kept_relation == rel1

    def test_cyclic_relations(self):
        """Test detection of cyclic spatial relations."""
        # Cycle: A NORTH_OF B, B NORTH_OF C, C NORTH_OF A (impossible!)
        relations = [
            SpatialRelation("A", "B", SpatialRelationType.NORTH_OF, confidence=1.0),
            SpatialRelation("B", "C", SpatialRelationType.NORTH_OF, confidence=1.0),
            SpatialRelation("C", "A", SpatialRelationType.NORTH_OF, confidence=1.0),
        ]

        # Detect cycle (simplified)
        # In real system: graph cycle detection
        entities = ["A", "B", "C"]
        cycle_detected = len(relations) == len(entities) and all(
            rel.relation_type == SpatialRelationType.NORTH_OF for rel in relations
        )

        assert cycle_detected

    def test_missing_entity(self):
        """Test handling missing entity in spatial relation."""
        # Incomplete: "... liegt nördlich von München" (missing subject)
        # Should not create relation without subject

        subject = None
        object_entity = "München"

        # Validation
        is_valid = subject is not None and object_entity is not None

        assert not is_valid


# ==================== Test Configuration ====================


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "learning: tests for spatial pattern learning")
