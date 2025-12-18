"""
Tests for security fixes in component_1 modules.

Tests PRIORITY 1 security fixes:
- Task 1: F-string Cypher injection prevention in assert_relation()
- Task 2: F-string Cypher injection prevention in set_wort_attribut()
- Task 3: Confidence range validation

Follows CLAUDE.md standards:
- Thread safety
- Structured logging
- Comprehensive error handling
- Type hints
"""

import pytest

from component_1_relation_management import ALLOWED_RELATIONS
from component_1_word_management import ALLOWED_ATTRIBUTES


class TestRelationTypeWhitelist:
    """Test relation type whitelist validation (Task 1)."""

    def test_valid_relation_type(self, netzwerk_session):
        """Valid relation types should succeed."""
        success = netzwerk_session.assert_relation(
            subject="hund", relation="IS_A", object="tier"
        )
        assert success is True

    def test_invalid_relation_type_raises_error(self, netzwerk_session):
        """Invalid relation types should raise ValueError."""
        with pytest.raises(ValueError, match="not in whitelist"):
            netzwerk_session.assert_relation(
                subject="hund",
                relation="MALICIOUS_INJECTION",  # Not in whitelist
                object="tier",
            )

    def test_injection_attempt_blocked(self, netzwerk_session):
        """SQL-like injection attempts should be blocked."""
        with pytest.raises(ValueError, match="not in whitelist"):
            netzwerk_session.assert_relation(
                subject="hund",
                relation="'; DROP TABLE; --",  # Injection attempt
                object="tier",
            )

    def test_all_whitelist_relations_allowed(self, netzwerk_session):
        """All relations in whitelist should be allowed."""
        # Sample of key relation types
        for relation in ["IS_A", "HAS_PROPERTY", "CAPABLE_OF", "PART_OF"]:
            assert relation in ALLOWED_RELATIONS
            success = netzwerk_session.assert_relation(
                subject="test", relation=relation, object="obj"
            )
            assert success is True


class TestAttributeWhitelist:
    """Test attribute whitelist validation (Task 2)."""

    def test_valid_attribute_name(self, netzwerk_session):
        """Valid attribute names should succeed."""
        success = netzwerk_session.set_wort_attribut("hund", "pos", "NOUN")
        assert success is True

    def test_invalid_attribute_raises_error(self, netzwerk_session):
        """Invalid attribute names should raise ValueError."""
        with pytest.raises(ValueError, match="not in whitelist"):
            netzwerk_session.set_wort_attribut(
                "hund",
                "malicious_attr",  # Not in whitelist
                "value",
            )

    def test_injection_attempt_blocked_attribute(self, netzwerk_session):
        """SQL-like injection in attribute names should be blocked."""
        with pytest.raises(ValueError, match="not in whitelist"):
            netzwerk_session.set_wort_attribut(
                "hund",
                "'; DROP TABLE; --",  # Injection attempt
                "value",
            )

    def test_all_whitelist_attributes_allowed(self, netzwerk_session):
        """All attributes in whitelist should be allowed."""
        # Sample of key attributes
        for attr in ["pos", "frequency", "importance"]:
            assert attr in ALLOWED_ATTRIBUTES
            success = netzwerk_session.set_wort_attribut("test", attr, "value")
            assert success is True


class TestConfidenceValidation:
    """Test confidence range validation (Task 3)."""

    def test_valid_confidence_values(self, netzwerk_session):
        """Valid confidence values [0.0, 1.0] should succeed."""
        for conf in [0.0, 0.5, 0.85, 1.0]:
            success = netzwerk_session.assert_relation(
                subject="hund",
                relation="IS_A",
                object="tier",
                confidence=conf,
            )
            assert success is True

    def test_confidence_below_zero_raises_error(self, netzwerk_session):
        """Confidence < 0.0 should raise ValueError."""
        with pytest.raises(ValueError, match="Confidence must be in"):
            netzwerk_session.assert_relation(
                subject="hund",
                relation="IS_A",
                object="tier",
                confidence=-0.1,
            )

    def test_confidence_above_one_raises_error(self, netzwerk_session):
        """Confidence > 1.0 should raise ValueError."""
        with pytest.raises(ValueError, match="Confidence must be in"):
            netzwerk_session.assert_relation(
                subject="hund",
                relation="IS_A",
                object="tier",
                confidence=1.5,
            )

    def test_min_confidence_validation_in_query(self, netzwerk_session):
        """min_confidence parameter should be validated in query methods."""
        # Valid min_confidence
        facts = netzwerk_session.query_graph_for_facts("hund", min_confidence=0.7)
        assert isinstance(facts, dict)

        # Invalid min_confidence < 0.0
        with pytest.raises(ValueError, match="min_confidence must be in"):
            netzwerk_session.query_graph_for_facts("hund", min_confidence=-0.1)

        # Invalid min_confidence > 1.0
        with pytest.raises(ValueError, match="min_confidence must be in"):
            netzwerk_session.query_graph_for_facts("hund", min_confidence=1.5)


class TestSecurityEdgeCases:
    """Test edge cases and security scenarios."""

    def test_empty_relation_type_rejected(self, netzwerk_session):
        """Empty relation types should be rejected."""
        # Empty string after sanitization
        success = netzwerk_session.assert_relation(
            subject="hund",
            relation="",  # Empty
            object="tier",
        )
        assert success is False

    def test_special_chars_sanitized_then_validated(self, netzwerk_session):
        """Special characters should be sanitized, then whitelist-validated."""
        # "IS-A" -> sanitized to "ISA" -> not in whitelist -> error
        with pytest.raises(ValueError, match="not in whitelist"):
            netzwerk_session.assert_relation(
                subject="hund",
                relation="IS-A",  # Contains hyphen
                object="tier",
            )

    def test_unicode_relation_type_sanitized(self, netzwerk_session):
        """Unicode characters should be sanitized."""
        # Unicode arrow -> sanitized to empty -> rejected
        success = netzwerk_session.assert_relation(
            subject="hund",
            relation="→",  # Unicode arrow
            object="tier",
        )
        assert success is False

    def test_confidence_type_validation(self, netzwerk_session):
        """Non-numeric confidence values should be handled."""
        # Type coercion should work for int
        success = netzwerk_session.assert_relation(
            subject="hund",
            relation="IS_A",
            object="tier",
            confidence=1,  # int instead of float
        )
        assert success is True

        # String should raise TypeError/ValueError
        with pytest.raises((TypeError, ValueError)):
            netzwerk_session.assert_relation(
                subject="hund",
                relation="IS_A",
                object="tier",
                confidence="high",  # type: ignore
            )
