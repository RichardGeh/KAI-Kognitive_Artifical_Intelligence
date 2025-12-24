"""
Unit test for entity extraction punctuation bug fix.

Tests the fix for the bug where punctuation-suffixed words (e.g., "Rot,")
bypassed the exclusion list check in kai_strategy_dispatcher.py.

Bug Report: Entity extraction didn't strip punctuation before checking
exclusion list, causing colors like "Rot," to be incorrectly identified
as entities.

Fix: Strip punctuation BEFORE exclusion check (lines 1254-1259).
"""

from unittest.mock import MagicMock

import pytest

from kai_strategy_dispatcher import StrategyDispatcher


class TestEntityExtractionPunctuationBugfix:
    """Test suite for entity extraction punctuation handling."""

    @pytest.fixture
    def dispatcher(self):
        """Create dispatcher with minimal mocks."""
        # Create all required mocks
        mock_netzwerk = MagicMock()
        mock_logic_engine = MagicMock()
        mock_graph_traversal = MagicMock()
        mock_working_memory = MagicMock()
        mock_signals = MagicMock()

        # Initialize lazy-loaded components as None
        dispatcher = StrategyDispatcher(
            netzwerk=mock_netzwerk,
            logic_engine=mock_logic_engine,
            graph_traversal=mock_graph_traversal,
            working_memory=mock_working_memory,
            signals=mock_signals,
        )

        # Mock lazy-loaded logic_puzzle_solver to avoid initialization
        dispatcher.logic_puzzle_solver = MagicMock()

        return dispatcher

    def test_entity_extraction_strips_punctuation_before_exclusion(self, dispatcher):
        """
        Test that punctuation is stripped BEFORE checking exclusion list.

        Regression test for bug where "Rot," bypassed exclusion filter
        because it didn't match "Rot" in the exclusion list.
        """
        # Input with punctuation-suffixed colors and entity names
        query = "Vier Personen (Anna, Ben, Clara, Daniel) tragen jeweils eine Farbe: Rot, Blau, Gruen, Gelb."

        entities = dispatcher._extract_entities_from_query(query)

        # Should extract only entity names (Anna, Ben, Clara, Daniel)
        # NOT colors (Rot, Blau, Gruen, Gelb) even with punctuation
        assert set(entities) == {"Anna", "Ben", "Clara", "Daniel"}

        # Colors should be filtered out
        assert "Rot" not in entities
        assert "Blau" not in entities
        assert "Gruen" not in entities
        assert "Gelb" not in entities

    def test_entity_extraction_handles_various_punctuation(self, dispatcher):
        """Test that various punctuation marks are correctly stripped."""
        test_cases = [
            ("Test Person: Alice.", ["Alice"]),  # Period
            ("Test Person, Bob!", ["Bob"]),  # Exclamation
            ("Test Person? Charlie", ["Charlie"]),  # Question mark
            ("Test (David) Person", ["David"]),  # Parentheses
            ("Test Person; Eve", ["Eve"]),  # Semicolon
            ('Test "Frank" Person', ["Frank"]),  # Quotes
        ]

        for query, expected_entities in test_cases:
            entities = dispatcher._extract_entities_from_query(query)
            assert entities == expected_entities, f"Failed for query: {query}"

    def test_entity_extraction_filters_german_articles(self, dispatcher):
        """Test that German articles are correctly filtered."""
        query = "Der Mann und Die Frau haben Das Buch"
        entities = dispatcher._extract_entities_from_query(query)

        # Articles should be filtered
        assert "Der" not in entities
        assert "Die" not in entities
        assert "Das" not in entities

        # Valid entities should remain
        assert "Mann" in entities
        assert "Frau" in entities
        assert "Buch" in entities

    def test_entity_extraction_filters_color_words(self, dispatcher):
        """Test that all color words are correctly filtered."""
        # Test all colors from the exclusion list
        colors = [
            "Rot",
            "Blau",
            "Gruen",
            "Gelb",
            "Schwarz",
            "Weiss",
            "Grau",
            "Orange",
            "Lila",
            "Rosa",
            "Braun",
        ]

        for color in colors:
            # Test with punctuation
            query = f"Test Person Anna hat {color}, und Bob hat {color}."
            entities = dispatcher._extract_entities_from_query(query)

            assert color not in entities, f"Color {color} should be filtered"
            assert "Anna" in entities
            assert "Bob" in entities

    def test_entity_extraction_filters_category_words(self, dispatcher):
        """Test that category words (Farbe, Beruf, Person) are filtered."""
        query = "Person Anna hat Farbe Rot und Beruf Lehrer"
        entities = dispatcher._extract_entities_from_query(query)

        # Category words should be filtered
        assert "Person" not in entities
        assert "Farbe" not in entities
        assert "Beruf" not in entities

        # Valid entities should remain
        assert "Anna" in entities
        assert "Lehrer" in entities

    def test_entity_extraction_skips_first_word(self, dispatcher):
        """Test that the first word is always skipped (question word)."""
        # First word should be skipped even if it's a valid entity name
        query = "Anna ist eine Person und Bob ist auch eine Person"
        entities = dispatcher._extract_entities_from_query(query)

        # First "Anna" should be skipped, but "Bob" should be extracted
        # Note: If "Anna" appears later, it would be extracted
        assert "Bob" in entities

    def test_entity_extraction_handles_empty_strings_after_stripping(self, dispatcher):
        """Test that empty strings after punctuation removal are skipped."""
        query = "Test Person: ., !, ? Anna"
        entities = dispatcher._extract_entities_from_query(query)

        # Only "Anna" should be extracted (punctuation-only tokens skipped)
        assert "Anna" in entities
        assert len([e for e in entities if e]) == len(entities)  # No empty strings

    def test_entity_extraction_preserves_case_sensitivity(self, dispatcher):
        """Test that entity names maintain their original case."""
        query = "Test Person Anna und ALICE und Bob"
        entities = dispatcher._extract_entities_from_query(query)

        # Case should be preserved
        assert "Anna" in entities
        assert "ALICE" in entities
        assert "Bob" in entities

    def test_entity_extraction_removes_duplicates(self, dispatcher):
        """Test that duplicate entities are removed."""
        query = "Test Person Anna, Anna, und Anna"
        entities = dispatcher._extract_entities_from_query(query)

        # Should only have one "Anna"
        assert entities.count("Anna") == 1
