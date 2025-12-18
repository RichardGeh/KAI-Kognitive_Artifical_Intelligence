# tests/test_negation_basic.py
"""
Simplified test suite for Negation Support (Quick Win #5)

Focus on core functionality with minimal fixtures.
"""

import pytest


class TestNegationBasic:
    """Test basic negation functionality."""

    def test_assert_negation_cannot_do(self, netzwerk_session, clean_test_concepts):
        """Test creating CANNOT_DO negation relation."""
        netzwerk = netzwerk_session

        # Use fixture prefix for proper cleanup
        subject = f"{clean_test_concepts}pinguin"
        obj = f"{clean_test_concepts}fliegen"

        # Create negation relation
        success = netzwerk.assert_negation(
            subject=subject,
            base_relation="CAPABLE_OF",
            object=obj,
            source_sentence="Ein Pinguin kann nicht fliegen",
        )

        assert success, "Negation relation should be created"

        # Verify negation was stored
        facts = netzwerk.query_graph_for_facts(subject)
        assert "CANNOT_DO" in facts, "CANNOT_DO relation should exist"
        assert obj in facts["CANNOT_DO"], f"{obj} should be in CANNOT_DO facts"

    def test_assert_negation_not_is_a(self, netzwerk_session, clean_test_concepts):
        """Test creating NOT_IS_A negation relation."""
        netzwerk = netzwerk_session

        # Use fixture prefix for proper cleanup
        subject = f"{clean_test_concepts}wal"
        obj = f"{clean_test_concepts}fisch"

        success = netzwerk.assert_negation(
            subject=subject,
            base_relation="IS_A",
            object=obj,
            source_sentence="Ein Wal ist kein Fisch",
        )

        assert success

        facts = netzwerk.query_graph_for_facts(subject)
        assert "NOT_IS_A" in facts
        assert obj in facts["NOT_IS_A"]

    def test_negation_priority_simple(self, netzwerk_session, clean_test_concepts):
        """
        Test that negation is stored correctly alongside positive facts.

        Setup:
        - vogel CAPABLE_OF fliegen
        - pinguin IS_A vogel
        - pinguin CANNOT_DO fliegen

        Verify all facts are stored.
        """
        netzwerk = netzwerk_session

        # Use fixture prefix for proper cleanup
        vogel = f"{clean_test_concepts}vogel"
        pinguin = f"{clean_test_concepts}pinguin2"
        fliegen = f"{clean_test_concepts}fliegen"

        # Setup hierarchy
        netzwerk.assert_relation(vogel, "CAPABLE_OF", fliegen, "Vögel können fliegen")
        netzwerk.assert_relation(pinguin, "IS_A", vogel, "Ein Pinguin ist ein Vogel")

        # Add explicit negation
        netzwerk.assert_negation(
            pinguin,
            "CAPABLE_OF",
            fliegen,
            "Ein Pinguin kann nicht fliegen",
        )

        # Verify all facts exist
        vogel_facts = netzwerk.query_graph_for_facts(vogel)
        assert "CAPABLE_OF" in vogel_facts, "Vogel should CAPABLE_OF fliegen"

        pinguin_facts = netzwerk.query_graph_for_facts(pinguin)
        assert "IS_A" in pinguin_facts, "Pinguin should IS_A vogel"
        assert "CANNOT_DO" in pinguin_facts, "Pinguin should CANNOT_DO fliegen"

    def test_multiple_negations(self, netzwerk_session, clean_test_concepts):
        """Test multiple negations for the same subject."""
        netzwerk = netzwerk_session

        # Use fixture prefix for proper cleanup
        subject = f"{clean_test_concepts}pinguin3"
        fliegen = f"{clean_test_concepts}fliegen"
        rennen = f"{clean_test_concepts}rennen"

        netzwerk.assert_negation(subject, "CAPABLE_OF", fliegen, "Kann nicht fliegen")
        netzwerk.assert_negation(subject, "CAPABLE_OF", rennen, "Kann nicht rennen")

        facts = netzwerk.query_graph_for_facts(subject)

        assert "CANNOT_DO" in facts
        assert fliegen in facts["CANNOT_DO"]
        assert rennen in facts["CANNOT_DO"]


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
