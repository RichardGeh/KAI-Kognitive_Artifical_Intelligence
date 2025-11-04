"""
Additional tests for component_7_meaning_extractor.py to increase coverage from 60% to 70%+

Focuses on declarative statements, heuristics, and edge cases.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from component_6_linguistik_engine import LinguisticPreprocessor
from component_7_meaning_extractor import MeaningPointExtractor
from component_11_embedding_service import EmbeddingService
from component_5_linguistik_strukturen import MeaningPointCategory
from component_utils_text_normalization import TextNormalizer


@pytest.fixture(scope="module")
def preprocessor():
    """Linguistic preprocessor fixture."""
    return LinguisticPreprocessor()


@pytest.fixture(scope="module")
def embedding_service():
    """Embedding service fixture."""
    return EmbeddingService()


@pytest.fixture(scope="module")
def extractor(embedding_service, preprocessor):
    """Meaning point extractor fixture."""
    return MeaningPointExtractor(embedding_service, preprocessor)


class TestDeclarativeStatements:
    """Tests for declarative statement detection patterns."""

    def test_detect_is_a_singular(self, extractor, preprocessor):
        """Test IS_A pattern: 'X ist ein Y'."""
        doc = preprocessor.process("Ein Hund ist ein Tier")
        results = extractor.extract(doc)

        assert len(results) == 1
        assert results[0].category == MeaningPointCategory.DEFINITION
        assert results[0].arguments.get("subject") == "hund"
        assert results[0].arguments.get("relation_type") == "IS_A"
        assert results[0].arguments.get("object") == "tier"

    def test_detect_is_a_plural(self, extractor, preprocessor):
        """Test IS_A plural pattern: 'X sind Y'."""
        doc = preprocessor.process("Katzen sind Tiere")
        results = extractor.extract(doc)

        assert len(results) == 1
        assert results[0].category == MeaningPointCategory.DEFINITION
        assert results[0].arguments.get("relation_type") == "IS_A"

    def test_detect_has_property(self, extractor, preprocessor):
        """Test HAS_PROPERTY pattern: 'X ist Y' (adjective)."""
        doc = preprocessor.process("Der Apfel ist rot")
        results = extractor.extract(doc)

        assert len(results) == 1
        assert results[0].category == MeaningPointCategory.DEFINITION
        assert results[0].arguments.get("relation_type") == "HAS_PROPERTY"

    def test_detect_capable_of(self, extractor, preprocessor):
        """Test CAPABLE_OF pattern: 'X kann Y'."""
        doc = preprocessor.process("Der Vogel kann fliegen")
        results = extractor.extract(doc)

        assert len(results) == 1
        assert results[0].category == MeaningPointCategory.DEFINITION
        assert results[0].arguments.get("relation_type") == "CAPABLE_OF"

    def test_detect_part_of(self, extractor, preprocessor):
        """Test PART_OF pattern: 'X hat Y'."""
        doc = preprocessor.process("Das Auto hat Räder")
        results = extractor.extract(doc)

        assert len(results) == 1
        assert results[0].category == MeaningPointCategory.DEFINITION
        assert results[0].arguments.get("relation_type") == "PART_OF"

    def test_detect_located_in(self, extractor, preprocessor):
        """Test LOCATED_IN pattern: 'X liegt in Y'."""
        doc = preprocessor.process("Berlin liegt in Deutschland")
        results = extractor.extract(doc)

        assert len(results) == 1
        assert results[0].category == MeaningPointCategory.DEFINITION
        assert results[0].arguments.get("relation_type") == "LOCATED_IN"


class TestQuestionHeuristics:
    """Tests for question detection heuristics."""

    def test_heuristic_what_is(self, extractor, preprocessor):
        """Test 'Was ist X?' pattern."""
        doc = preprocessor.process("Was ist ein Hund?")
        results = extractor.extract(doc)

        assert len(results) == 1
        assert results[0].category == MeaningPointCategory.QUESTION
        assert "hund" in results[0].arguments.get("topic", "")

    def test_heuristic_where_is(self, extractor, preprocessor):
        """Test 'Wo ist X?' pattern."""
        doc = preprocessor.process("Wo liegt Berlin?")
        results = extractor.extract(doc)

        assert len(results) == 1
        assert results[0].category == MeaningPointCategory.QUESTION
        assert results[0].arguments.get("property_name") == "LOCATED_IN"

    def test_heuristic_wer(self, extractor, preprocessor):
        """Test 'Wer...' pattern."""
        doc = preprocessor.process("Wer ist der Chef?")
        results = extractor.extract(doc)

        assert len(results) == 1
        assert results[0].category == MeaningPointCategory.QUESTION
        assert results[0].arguments.get("question_word") == "wer"

    def test_heuristic_wie(self, extractor, preprocessor):
        """Test 'Wie...' pattern."""
        doc = preprocessor.process("Wie funktioniert das?")
        results = extractor.extract(doc)

        assert len(results) == 1
        assert results[0].category == MeaningPointCategory.QUESTION
        assert results[0].arguments.get("question_word") == "wie"

    def test_heuristic_warum(self, extractor, preprocessor):
        """Test 'Warum/Wieso/Weshalb...' patterns."""
        for word in ["Warum", "Wieso", "Weshalb"]:
            doc = preprocessor.process(f"{word} ist das so?")
            results = extractor.extract(doc)

            assert len(results) == 1
            assert results[0].category == MeaningPointCategory.QUESTION
            assert results[0].arguments.get("question_word") == word.lower()

    def test_heuristic_wann(self, extractor, preprocessor):
        """Test 'Wann...' pattern."""
        doc = preprocessor.process("Wann passiert das?")
        results = extractor.extract(doc)

        assert len(results) == 1
        assert results[0].category == MeaningPointCategory.QUESTION
        assert results[0].arguments.get("question_word") == "wann"

    def test_heuristic_flexible_wh(self, extractor, preprocessor):
        """Test flexible WH-word patterns (welche, wozu, etc.)."""
        for word in ["Welcher", "Welche", "Welches", "Wozu"]:
            doc = preprocessor.process(f"{word} ist das Beste?")
            results = extractor.extract(doc)

            assert len(results) == 1
            assert results[0].category == MeaningPointCategory.QUESTION


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_input(self, extractor, preprocessor):
        """Test extraction with empty input."""
        doc = preprocessor.process("")
        results = extractor.extract(doc)

        assert len(results) == 1
        assert results[0].category == MeaningPointCategory.UNKNOWN
        assert results[0].confidence == 0.0

    def test_whitespace_only(self, extractor, preprocessor):
        """Test extraction with whitespace only."""
        doc = preprocessor.process("   ")
        results = extractor.extract(doc)

        assert len(results) == 1
        assert results[0].category == MeaningPointCategory.UNKNOWN

    def test_no_match_returns_unknown(self, extractor, preprocessor):
        """Test that unmatched input returns UNKNOWN."""
        doc = preprocessor.process("Completely unrecognizable gibberish xyz123")
        results = extractor.extract(doc)

        assert len(results) == 1
        assert results[0].category == MeaningPointCategory.UNKNOWN
        assert results[0].confidence == 0.0


class TestPluralNormalization:
    """Tests for plural-to-singular normalization using TextNormalizer."""

    def test_clean_entity_removes_articles(self):
        """Test that articles are removed."""
        normalizer = TextNormalizer()
        cleaned = normalizer.clean_entity("der Hund")
        assert cleaned == "hund"

        cleaned = normalizer.clean_entity("ein Auto")
        assert cleaned == "auto"

    def test_clean_entity_removes_punctuation(self):
        """Test that punctuation is removed."""
        normalizer = TextNormalizer()
        cleaned = normalizer.clean_entity("Hund.")
        assert cleaned == "hund"

    def test_normalize_plural_en(self, preprocessor):
        """Test plural normalization for -en ending."""
        normalizer = TextNormalizer(preprocessor)
        result = normalizer.normalize_plural_to_singular("hunden")
        assert result == "hund"

    def test_normalize_plural_e(self, preprocessor):
        """Test plural normalization for -e ending."""
        normalizer = TextNormalizer(preprocessor)
        result = normalizer.normalize_plural_to_singular("tage")
        assert result == "tag"

    def test_normalize_plural_s(self, preprocessor):
        """Test plural normalization for -s ending."""
        normalizer = TextNormalizer(preprocessor)
        result = normalizer.normalize_plural_to_singular("autos")
        assert result == "auto"

    def test_normalize_plural_ionen(self, preprocessor):
        """Test plural normalization for -ionen ending."""
        normalizer = TextNormalizer(preprocessor)
        result = normalizer.normalize_plural_to_singular("aktionen")
        assert result == "aktion"

    def test_normalize_short_word(self, preprocessor):
        """Test that short words aren't normalized."""
        normalizer = TextNormalizer(preprocessor)
        result = normalizer.normalize_plural_to_singular("ab")
        assert result == "ab"


class TestExplicitCommands:
    """Tests for explicit command parsing."""

    def test_parse_definiere_command(self, extractor, preprocessor):
        """Test 'Definiere:' command parsing."""
        doc = preprocessor.process("Definiere: test/bedeutung = Eine Testbedeutung")
        results = extractor.extract(doc)

        assert len(results) == 1
        assert results[0].category == MeaningPointCategory.COMMAND
        assert results[0].confidence == 1.0
        assert results[0].arguments.get("command") == "definiere"

    def test_parse_lerne_muster_command(self, extractor, preprocessor):
        """Test 'Lerne Muster:' command parsing."""
        doc = preprocessor.process('Lerne Muster: "Ein Hund ist ein Tier"')
        results = extractor.extract(doc)

        assert len(results) == 1
        assert results[0].category == MeaningPointCategory.COMMAND
        assert results[0].confidence == 1.0
        assert results[0].arguments.get("command") == "learn_pattern"

    def test_parse_ingestiere_command(self, extractor, preprocessor):
        """Test 'Ingestiere Text:' command parsing."""
        doc = preprocessor.process('Ingestiere Text: "Ein Hund ist ein Tier."')
        results = extractor.extract(doc)

        assert len(results) == 1
        assert results[0].category == MeaningPointCategory.COMMAND
        assert results[0].confidence == 1.0
        assert results[0].arguments.get("command") == "ingest_text"


class TestArgumentExtraction:
    """Tests for argument extraction from different categories."""

    def test_extract_arguments_question(self, extractor, preprocessor):
        """Test argument extraction for questions."""
        doc = preprocessor.process("Was ist ein Hund?")
        results = extractor.extract(doc)

        # Arguments should contain question_word
        assert (
            "question_word" in results[0].arguments or "topic" in results[0].arguments
        )

    def test_extract_arguments_command(self, extractor, preprocessor):
        """Test argument extraction for commands."""
        doc = preprocessor.process('Lerne Muster: "Test"')
        results = extractor.extract(doc)

        assert "command" in results[0].arguments

    def test_extract_arguments_definition(self, extractor, preprocessor):
        """Test argument extraction for definitions."""
        doc = preprocessor.process("Ein Hund ist ein Tier")
        results = extractor.extract(doc)

        assert "subject" in results[0].arguments
        assert "object" in results[0].arguments


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
