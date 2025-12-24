# test_pattern_discovery_part2_bootstrap.py
"""
Test scenarios for Pattern Bootstrap (Part 2: Pattern Discovery).

Tests cover:
- Happy Path: Load seed templates, bootstrap status checks
- Edge Cases: Missing files, invalid YAML, missing fields
- Error Handling: YAML syntax errors, invalid values, duplicate IDs

Total: 15 tests
"""


import pytest
import yaml

from component_61_pattern_bootstrap import check_bootstrap_status, load_seed_templates
from kai_exceptions import ConfigurationException


class TestHappyPathBootstrap:
    """Happy path scenarios for pattern bootstrap."""

    def test_hp03_bootstrap_seed_templates(
        self, netzwerk_session, test_templates_yml_path, clean_pattern_data
    ):
        """HP-03: Load 3 templates from test YAML file."""
        stats = load_seed_templates(netzwerk_session, test_templates_yml_path)

        # Assertions
        assert stats["patterns_created"] == 3
        assert stats["items_created"] >= 7  # At least 2-3 items per pattern
        assert stats["slots_created"] >= 5  # Multiple slots

    def test_hp11_bootstrap_status_not_bootstrapped(
        self, netzwerk_session, clean_pattern_data
    ):
        """HP-11: Check returns false before first load."""
        status = check_bootstrap_status(netzwerk_session)

        assert status["bootstrapped"] is False
        assert status["seed_pattern_count"] == 0
        assert status["total_pattern_count"] == 0

    def test_hp12_bootstrap_status_bootstrapped(
        self, netzwerk_session, test_templates_yml_path, clean_pattern_data
    ):
        """HP-12: Check returns true after load."""
        load_seed_templates(netzwerk_session, test_templates_yml_path)
        status = check_bootstrap_status(netzwerk_session)

        assert status["bootstrapped"] is True
        assert status["seed_pattern_count"] == 3
        assert status["total_pattern_count"] == 3

    def test_hp14_pattern_creation_with_metadata(
        self, pattern_manager, netzwerk_session
    ):
        """HP-14: Pattern stores category and template_id."""
        pattern_id = pattern_manager.create_pattern(
            name="Test Pattern",
            pattern_type="seed",
            metadata={"category": "Question", "template_id": "test_001"},
        )

        # Verify pattern properties
        with netzwerk_session.driver.session(database="neo4j") as session:
            result = session.run(
                """
                MATCH (p:Pattern {id: $pid})
                RETURN p
            """,
                {"pid": pattern_id},
            )
            pattern = dict(result.single()["p"])

            assert pattern["type"] == "seed"
            assert pattern["support"] == 0
            assert pattern["precision"] == 0.5
            assert pattern["createdAt"] is not None

    def test_hp15_update_pattern_stats(self, pattern_manager, netzwerk_session):
        """HP-15: Pattern stats updated after successful match."""
        # Create pattern
        pattern_id = pattern_manager.create_pattern(
            name="Test Pattern", pattern_type="seed"
        )

        # Update stats
        pattern_manager.update_pattern_stats(
            pattern_id=pattern_id, support_increment=5, new_precision=0.80
        )

        # Verify
        with netzwerk_session.driver.session(database="neo4j") as session:
            result = session.run(
                """
                MATCH (p:Pattern {id: $pid})
                RETURN p.support AS support, p.precision AS precision
            """,
                {"pid": pattern_id},
            )
            record = result.single()

            assert record["support"] == 5
            assert abs(record["precision"] - 0.80) < 0.01

    def test_hp17_template_all_literals(self, netzwerk_session, tmp_path):
        """HP-17: Template with all LITERAL items (no slots)."""
        yaml_path = tmp_path / "literal_only.yml"
        data = {
            "templates": [
                {
                    "id": "literal_test",
                    "name": "Literal Test",
                    "category": "Command",
                    "pattern": [
                        {"kind": "LITERAL", "value": "lerne"},
                        {"kind": "LITERAL", "value": ":"},
                    ],
                }
            ]
        }

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f)

        stats = load_seed_templates(netzwerk_session, str(yaml_path))

        assert stats["patterns_created"] == 1
        assert stats["slots_created"] == 0
        assert stats["items_created"] == 2

    def test_hp18_template_all_slots(self, netzwerk_session, tmp_path):
        """HP-18: Template with all SLOT items (very flexible)."""
        yaml_path = tmp_path / "slot_only.yml"
        data = {
            "templates": [
                {
                    "id": "slot_test",
                    "name": "Slot Test",
                    "category": "Statement",
                    "pattern": [
                        {"kind": "SLOT", "slot_type": "SUBJECT", "allowed": []},
                        {"kind": "SLOT", "slot_type": "VERB", "allowed": []},
                        {"kind": "SLOT", "slot_type": "OBJECT", "allowed": []},
                    ],
                }
            ]
        }

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f)

        stats = load_seed_templates(netzwerk_session, str(yaml_path))

        assert stats["patterns_created"] == 1
        assert stats["slots_created"] == 3
        assert stats["items_created"] == 3


class TestEdgeCasesBootstrap:
    """Edge case scenarios for pattern bootstrap."""

    def test_ec10_bootstrap_missing_yaml_file(self, netzwerk_session):
        """EC-10: Missing YAML file raises ConfigurationException."""
        with pytest.raises(ConfigurationException, match="file not found"):
            load_seed_templates(netzwerk_session, "nonexistent.yml")

    def test_ec12_bootstrap_missing_templates_key(self, netzwerk_session, tmp_path):
        """EC-12: YAML without 'templates' key raises ConfigurationException."""
        yaml_path = tmp_path / "no_templates_key.yml"
        data = [{"id": "test", "name": "Test"}]  # Missing templates wrapper

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f)

        with pytest.raises(ConfigurationException, match="'templates' key"):
            load_seed_templates(netzwerk_session, str(yaml_path))

    def test_ec13_template_missing_required_field(self, netzwerk_session, tmp_path):
        """EC-13: Template without 'id' field - resilient, logs error and continues."""
        yaml_path = tmp_path / "missing_id.yml"
        data = {
            "templates": [
                {
                    "name": "Test",  # Missing 'id'
                    "category": "Question",
                    "pattern": [{"kind": "LITERAL", "value": "test"}],
                }
            ]
        }

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f)

        stats = load_seed_templates(netzwerk_session, str(yaml_path))

        # Resilient: logs error and continues, no patterns created
        assert stats["patterns_created"] == 0
        assert stats["items_created"] == 0
        assert stats["slots_created"] == 0

    def test_ec14_template_empty_pattern(self, netzwerk_session, tmp_path):
        """EC-14: Template with empty pattern list - resilient, logs error and continues."""
        yaml_path = tmp_path / "empty_pattern.yml"
        data = {
            "templates": [
                {
                    "id": "test",
                    "name": "Test",
                    "category": "Question",
                    "pattern": [],  # Empty
                }
            ]
        }

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f)

        stats = load_seed_templates(netzwerk_session, str(yaml_path))

        # Resilient: logs error and continues, no patterns created
        assert stats["patterns_created"] == 0
        assert stats["items_created"] == 0
        assert stats["slots_created"] == 0

    def test_ec16_pattern_item_invalid_kind(self, netzwerk_session, tmp_path):
        """EC-16: PatternItem with invalid kind - resilient, logs error and continues."""
        yaml_path = tmp_path / "invalid_kind.yml"
        data = {
            "templates": [
                {
                    "id": "test",
                    "name": "Test",
                    "category": "Question",
                    "pattern": [{"kind": "INVALID", "value": "test"}],
                }
            ]
        }

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f)

        stats = load_seed_templates(netzwerk_session, str(yaml_path))

        # Resilient: Pattern created but items failed
        assert stats["patterns_created"] == 1
        assert stats["items_created"] == 0
        assert stats["slots_created"] == 0

    def test_ec17_literal_missing_value(self, netzwerk_session, tmp_path):
        """EC-17: LITERAL item without 'value' field - resilient, logs error and continues."""
        yaml_path = tmp_path / "literal_no_value.yml"
        data = {
            "templates": [
                {
                    "id": "test",
                    "name": "Test",
                    "category": "Question",
                    "pattern": [{"kind": "LITERAL"}],  # Missing value
                }
            ]
        }

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f)

        stats = load_seed_templates(netzwerk_session, str(yaml_path))

        # Resilient: Pattern created but items failed
        assert stats["patterns_created"] == 1
        assert stats["items_created"] == 0
        assert stats["slots_created"] == 0

    def test_ec18_slot_missing_type(self, netzwerk_session, tmp_path):
        """EC-18: SLOT item without 'slot_type' field - resilient, logs error and continues."""
        yaml_path = tmp_path / "slot_no_type.yml"
        data = {
            "templates": [
                {
                    "id": "test",
                    "name": "Test",
                    "category": "Question",
                    "pattern": [
                        {"kind": "SLOT", "allowed": ["a", "b"]}
                    ],  # Missing slot_type
                }
            ]
        }

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f)

        stats = load_seed_templates(netzwerk_session, str(yaml_path))

        # Resilient: Pattern created but items failed
        assert stats["patterns_created"] == 1
        assert stats["items_created"] == 0
        assert stats["slots_created"] == 0

    def test_ec19_slot_empty_allowed_values(self, netzwerk_session, tmp_path):
        """EC-19: SLOT with empty allowed values (learns through experience)."""
        yaml_path = tmp_path / "empty_allowed.yml"
        data = {
            "templates": [
                {
                    "id": "test",
                    "name": "Test",
                    "category": "Question",
                    "pattern": [
                        {"kind": "SLOT", "slot_type": "SUBJECT", "allowed": []}
                    ],
                }
            ]
        }

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f)

        stats = load_seed_templates(netzwerk_session, str(yaml_path))

        assert stats["slots_created"] == 1
        # Slot should have 0 ALLOWS relations initially


class TestErrorHandlingBootstrap:
    """Error handling scenarios for pattern bootstrap."""

    def test_err09_duplicate_template_ids(self, netzwerk_session, tmp_path):
        """ERR-09: Two templates with same ID (both created, warning logged)."""
        yaml_path = tmp_path / "duplicate_ids.yml"
        data = {
            "templates": [
                {
                    "id": "duplicate",
                    "name": "Test 1",
                    "category": "Question",
                    "pattern": [{"kind": "LITERAL", "value": "test1"}],
                },
                {
                    "id": "duplicate",  # Same ID
                    "name": "Test 2",
                    "category": "Question",
                    "pattern": [{"kind": "LITERAL", "value": "test2"}],
                },
            ]
        }

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f)

        stats = load_seed_templates(netzwerk_session, str(yaml_path))

        # Both patterns created (Neo4j generates unique node IDs)
        assert stats["patterns_created"] == 2

    def test_err14_bootstrap_idempotency(
        self, netzwerk_session, test_templates_yml_path, clean_pattern_data
    ):
        """ERR-14: Load seed templates twice (creates duplicates)."""
        stats1 = load_seed_templates(netzwerk_session, test_templates_yml_path)
        stats2 = load_seed_templates(netzwerk_session, test_templates_yml_path)

        assert stats1["patterns_created"] == 3
        assert stats2["patterns_created"] == 3

        # Query total patterns
        from component_1_netzwerk_patterns import PatternManager

        manager = PatternManager(netzwerk_session.driver)
        all_patterns = manager.get_all_patterns(limit=100)
        assert len(all_patterns) == 6  # Duplicates created
