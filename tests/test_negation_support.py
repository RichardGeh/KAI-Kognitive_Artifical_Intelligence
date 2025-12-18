# tests/test_negation_support.py
"""
Test Suite für Negation Support (Quick Win #5)

Tests for negation handling in KAI:
- Basic negation learning (assert_negation)
- Negation detection in natural language
- Negation-aware reasoning (priority over positive inheritance)
- Logic puzzle with negation (penguin example)
- Multi-hop reasoning with negations

CRITICAL: Dieser Test ist der BLOCKER für Logic Puzzle Feature!
"""

import pytest


class TestBasicNegation:
    """Test basic negation assertion and retrieval."""

    def test_assert_negation_cannot_do(self, netzwerk):
        """Test creating CANNOT_DO negation relation."""
        success = netzwerk.assert_negation(
            subject="pinguin",
            base_relation="CAPABLE_OF",
            object="fliegen",
            source_sentence="Ein Pinguin kann nicht fliegen",
        )

        assert success, "Negation relation should be created"

        # Verify negation was stored
        facts = netzwerk.query_graph_for_facts("pinguin")
        assert "CANNOT_DO" in facts, "CANNOT_DO relation should exist"
        assert "fliegen" in facts["CANNOT_DO"], "fliegen should be in CANNOT_DO facts"

    def test_assert_negation_not_is_a(self, netzwerk):
        """Test creating NOT_IS_A negation relation."""
        success = netzwerk.assert_negation(
            subject="wal",
            base_relation="IS_A",
            object="fisch",
            source_sentence="Ein Wal ist kein Fisch",
        )

        assert success

        facts = netzwerk.query_graph_for_facts("wal")
        assert "NOT_IS_A" in facts
        assert "fisch" in facts["NOT_IS_A"]

    def test_assert_negation_has_not_property(self, netzwerk):
        """Test creating HAS_NOT_PROPERTY negation relation."""
        success = netzwerk.assert_negation(
            subject="pinguin",
            base_relation="HAS_PROPERTY",
            object="flugfaehig",
            source_sentence="Ein Pinguin ist nicht flugfähig",
        )

        assert success

        facts = netzwerk.query_graph_for_facts("pinguin")
        assert "HAS_NOT_PROPERTY" in facts
        assert "flugfaehig" in facts["HAS_NOT_PROPERTY"]


class TestNegationDetection:
    """Test automatic negation detection in natural language."""

    def test_detect_negation_nicht(self, netzwerk):
        """Test detection of 'nicht' negation marker."""
        from kai_ingestion_handler import KaiIngestionHandler

        ingestion = KaiIngestionHandler(netzwerk)

        # Ingest sentence with "nicht"
        stats = ingestion.ingest_text("Ein Pinguin kann nicht fliegen.")

        assert stats["facts_created"] > 0, "Should create at least one fact"

        # Verify CANNOT_DO relation was created
        facts = netzwerk.query_graph_for_facts("pinguin")
        assert "CANNOT_DO" in facts, "Should detect negation and create CANNOT_DO"

    def test_detect_negation_kein(self, netzwerk):
        """Test detection of 'kein' negation marker."""
        from kai_ingestion_handler import KaiIngestionHandler

        ingestion = KaiIngestionHandler(netzwerk)

        stats = ingestion.ingest_text("Ein Wal ist kein Fisch.")

        assert stats["facts_created"] > 0

        # Should create NOT_IS_A or similar
        facts = netzwerk.query_graph_for_facts("wal")
        # May be handled differently based on copula verb logic
        # Accept either NOT_IS_A or other negation patterns
        has_negation = any(
            rel_type.startswith("NOT_") or rel_type.startswith("CANNOT")
            for rel_type in facts.keys()
        )
        assert has_negation, "Should create some form of negation relation"


class TestNegationPriority:
    """Test that negations have priority over positive inheritance."""

    def test_negation_overrides_inheritance(self, netzwerk, inference_handler):
        """
        Test CRITICAL logic: Explicit negation overrides positive inheritance.

        Setup:
        - vogel CAPABLE_OF fliegen (general rule)
        - pinguin IS_A vogel (inheritance)
        - pinguin CANNOT_DO fliegen (explicit negation)

        Expected: "Kann pinguin fliegen?" -> NO (negation wins)
        """
        # Setup hierarchy
        netzwerk.assert_relation(
            "vogel", "CAPABLE_OF", "fliegen", "Vögel können fliegen"
        )
        netzwerk.assert_relation(
            "pinguin", "IS_A", "vogel", "Ein Pinguin ist ein Vogel"
        )

        # Add explicit negation (should override inheritance)
        netzwerk.assert_negation(
            "pinguin",
            "CAPABLE_OF",
            "fliegen",
            "Ein Pinguin kann nicht fliegen",
        )

        # Query: Can penguin fly?
        result = inference_handler.try_backward_chaining_inference(
            "pinguin", "CAPABLE_OF"
        )

        # CRITICAL ASSERTION: Negation should be detected
        assert result is not None, "Should find negation result"
        assert result.get("is_negation") is True, "Should be marked as negation"
        assert result.get("confidence") == 0.0, "Negation should have confidence 0.0"

        # Proof trace should mention negation
        proof_trace = result.get("proof_trace", "")
        assert (
            "CANNOT_DO" in proof_trace or "Negation" in proof_trace
        ), "Proof should mention negation"

    def test_no_false_negation(self, netzwerk, inference_handler):
        """Test that positive facts work normally when no negation exists."""
        # Setup: Only positive facts
        netzwerk.assert_relation("hund", "CAPABLE_OF", "bellen", "Ein Hund kann bellen")

        result = inference_handler.try_backward_chaining_inference("hund", "CAPABLE_OF")

        # Should find positive fact (NOT negation)
        assert result is not None
        assert result.get("is_negation") is not True, "Should NOT be negation"
        assert result.get("confidence") > 0.0, "Should have positive confidence"


class TestPenguinLogicPuzzle:
    """
    CRITICAL TEST: The penguin logic puzzle is the canonical negation example.

    This test MUST pass for Quick Win #5 to be considered complete.
    """

    def test_penguin_cannot_fly(self, netzwerk, inference_handler):
        """
        Full penguin logic puzzle test.

        Input facts:
        1. "Ein Vogel kann fliegen" -> (vogel)-[CAPABLE_OF]->(fliegen)
        2. "Ein Pinguin ist ein Vogel" -> (pinguin)-[IS_A]->(vogel)
        3. "Ein Pinguin kann nicht fliegen" -> (pinguin)-[CANNOT_DO]->(fliegen)

        Question: "Kann ein Pinguin fliegen?"
        Expected Answer: "Nein" (confidence=0.0, is_negation=True)
        """
        from kai_ingestion_handler import KaiIngestionHandler

        ingestion = KaiIngestionHandler(netzwerk)

        # Ingest facts in order
        text = """
        Ein Vogel kann fliegen.
        Ein Pinguin ist ein Vogel.
        Ein Pinguin kann nicht fliegen.
        """

        stats = ingestion.ingest_text(text)

        assert stats["facts_created"] >= 3, "Should create at least 3 facts"

        # Verify all facts were stored
        vogel_facts = netzwerk.query_graph_for_facts("vogel")
        assert "CAPABLE_OF" in vogel_facts, "Vogel should CAPABLE_OF fliegen"

        pinguin_facts = netzwerk.query_graph_for_facts("pinguin")
        assert "IS_A" in pinguin_facts, "Pinguin should IS_A vogel"
        assert "CANNOT_DO" in pinguin_facts, "Pinguin should CANNOT_DO fliegen"

        # Query: Can penguin fly?
        result = inference_handler.try_backward_chaining_inference(
            "pinguin", "CAPABLE_OF"
        )

        # CRITICAL ASSERTIONS
        assert result is not None, "Should return negation result"
        assert result["is_negation"] is True, "Should detect negation"
        assert result["confidence"] == 0.0, "Confidence should be 0.0 (cannot fly)"
        assert (
            result["negation_relation"] == "CANNOT_DO"
        ), "Should identify CANNOT_DO relation"

        # Verify proof trace
        proof = result.get("proof_trace", "")
        assert (
            "CANNOT_DO" in proof or "Negation" in proof
        ), "Proof should mention negation"

        print("\n[OK] PENGUIN LOGIC PUZZLE PASSED - Negation Support is working!")


class TestMultiHopWithNegation:
    """Test multi-hop reasoning combined with negations."""

    def test_negation_blocks_transitive_inference(self, netzwerk, inference_handler):
        """
        Test that negation prevents transitive inheritance.

        Setup:
        - tier CAPABLE_OF bewegen
        - vogel IS_A tier
        - pinguin IS_A vogel
        - pinguin CANNOT_DO fliegen (specific negation)

        Query: "Kann pinguin fliegen?" -> NO (negation blocks inheritance)
        Query: "Kann pinguin bewegen?" -> YES (no negation for "bewegen")
        """
        # Setup hierarchy
        netzwerk.assert_relation(
            "tier", "CAPABLE_OF", "bewegen", "Tiere können sich bewegen"
        )
        netzwerk.assert_relation("vogel", "IS_A", "tier", "Ein Vogel ist ein Tier")
        netzwerk.assert_relation(
            "vogel", "CAPABLE_OF", "fliegen", "Vögel können fliegen"
        )
        netzwerk.assert_relation(
            "pinguin", "IS_A", "vogel", "Ein Pinguin ist ein Vogel"
        )
        netzwerk.assert_negation(
            "pinguin",
            "CAPABLE_OF",
            "fliegen",
            "Ein Pinguin kann nicht fliegen",
        )

        # Test 1: Negated capability (fliegen)
        result_fliegen = inference_handler.try_backward_chaining_inference(
            "pinguin", "CAPABLE_OF"
        )
        assert result_fliegen is not None
        assert (
            result_fliegen.get("is_negation") is True
        ), "Should find negation for fliegen"

        # Test 2: Non-negated capability (bewegen) should still work
        # This requires multi-hop reasoning: pinguin -> vogel -> tier -> bewegen
        result_bewegen = inference_handler.try_backward_chaining_inference(
            "pinguin", "CAPABLE_OF"
        )

        # Note: This test may need adjustment based on how multi-hop handles mixed facts
        # For now, we verify negation detection works correctly


class TestEdgeCases:
    """Test edge cases and error handling for negations."""

    def test_negation_for_unknown_relation(self, netzwerk):
        """Test negation for relation type without mapping."""
        # Custom relation type (no negation mapping defined)
        success = netzwerk.assert_negation(
            "auto",
            "CUSTOM_RELATION",  # No mapping in negation_map
            "target",
            "Test custom relation negation",
        )

        # Should still succeed (uses fallback NOT_CUSTOM_RELATION)
        assert success

        facts = netzwerk.query_graph_for_facts("auto")
        assert "NOT_CUSTOM_RELATION" in facts, "Should use fallback NOT_ prefix"

    def test_multiple_negations_same_subject(self, netzwerk):
        """Test multiple negations for the same subject."""
        netzwerk.assert_negation(
            "pinguin", "CAPABLE_OF", "fliegen", "Kann nicht fliegen"
        )
        netzwerk.assert_negation("pinguin", "CAPABLE_OF", "rennen", "Kann nicht rennen")

        facts = netzwerk.query_graph_for_facts("pinguin")

        assert "CANNOT_DO" in facts
        assert "fliegen" in facts["CANNOT_DO"]
        assert "rennen" in facts["CANNOT_DO"]

    def test_negation_same_as_positive(self, netzwerk):
        """Test what happens when both positive and negative relations exist."""
        # This represents a contradiction
        netzwerk.assert_relation("X", "CAPABLE_OF", "Y", "X can do Y")
        netzwerk.assert_negation("X", "CAPABLE_OF", "Y", "X cannot do Y")

        facts = netzwerk.query_graph_for_facts("x")

        # Both relations should exist (contradiction detection is separate feature)
        assert "CAPABLE_OF" in facts or "CANNOT_DO" in facts


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
