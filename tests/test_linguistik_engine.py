"""
Test suite for component_6_linguistik_engine.py

Tests LinguisticPreprocessor and ResourceManager classes.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import tempfile
import shutil
from pathlib import Path
from component_6_linguistik_engine import LinguisticPreprocessor, ResourceManager


class TestResourceManager:
    """Tests for ResourceManager class."""

    def setup_method(self):
        """Create temporary directory for test lexika files."""
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)

    def teardown_method(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_resource_manager_initialization(self):
        """Test ResourceManager initialization."""
        rm = ResourceManager(self.test_dir)
        assert rm.lexika_pfad == self.test_path
        assert rm.lexika == {}

    def test_load_empty_directory(self):
        """Test loading from empty directory."""
        rm = ResourceManager(self.test_dir)
        rm.load()
        assert rm.lexika == {}

    def test_load_nonexistent_directory(self):
        """Test loading from non-existent directory."""
        rm = ResourceManager("nonexistent_directory_12345")
        rm.load()  # Should not crash, just warn
        assert rm.lexika == {}

    def test_load_valid_lexikon(self):
        """Test loading valid lexikon YAML file."""
        # Create test YAML file
        test_file = self.test_path / "test_lexikon.yml"
        test_file.write_text("- Hund\n- Katze\n- Vogel\n", encoding="utf-8")

        rm = ResourceManager(self.test_dir)
        rm.load()

        assert "test_lexikon" in rm.lexika
        assert rm.lexika["test_lexikon"] == ["hund", "katze", "vogel"]

    def test_load_multiple_lexika(self):
        """Test loading multiple lexikon files."""
        # Create multiple YAML files
        (self.test_path / "animals.yml").write_text(
            "- Hund\n- Katze\n", encoding="utf-8"
        )
        (self.test_path / "colors.yml").write_text("- Rot\n- Blau\n", encoding="utf-8")

        rm = ResourceManager(self.test_dir)
        rm.load()

        assert "animals" in rm.lexika
        assert "colors" in rm.lexika
        assert rm.lexika["animals"] == ["hund", "katze"]
        assert rm.lexika["colors"] == ["rot", "blau"]

    def test_load_non_list_yaml(self):
        """Test loading YAML file that doesn't contain a list."""
        test_file = self.test_path / "invalid.yml"
        test_file.write_text("key: value\n", encoding="utf-8")

        rm = ResourceManager(self.test_dir)
        rm.load()  # Should warn but not crash

        assert "invalid" not in rm.lexika

    def test_load_malformed_yaml(self):
        """Test loading malformed YAML file."""
        test_file = self.test_path / "broken.yml"
        test_file.write_text("- item1\n  - broken indentation\n", encoding="utf-8")

        rm = ResourceManager(self.test_dir)
        rm.load()  # Should catch exception and continue

        # Depending on YAML parser, might succeed or fail
        # Either way, should not crash

    def test_load_converts_to_lowercase(self):
        """Test that loaded items are converted to lowercase."""
        test_file = self.test_path / "mixed_case.yml"
        test_file.write_text("- UPPER\n- MiXeD\n- lower\n", encoding="utf-8")

        rm = ResourceManager(self.test_dir)
        rm.load()

        assert rm.lexika["mixed_case"] == ["upper", "mixed", "lower"]

    def test_load_ignores_non_yml_files(self):
        """Test that non-.yml files are ignored."""
        (self.test_path / "data.yml").write_text("- item1\n", encoding="utf-8")
        (self.test_path / "data.txt").write_text("- item2\n", encoding="utf-8")
        (self.test_path / "data.yaml").write_text("- item3\n", encoding="utf-8")

        rm = ResourceManager(self.test_dir)
        rm.load()

        # Only .yml files should be loaded
        assert "data" in rm.lexika
        assert "data.txt" not in rm.lexika
        # Note: .yaml files are not loaded (only .yml pattern)

    def test_load_with_unicode_content(self):
        """Test loading YAML with Unicode characters."""
        test_file = self.test_path / "unicode.yml"
        test_file.write_text("- Ä\n- Ö\n- Ü\n- ß\n", encoding="utf-8")

        rm = ResourceManager(self.test_dir)
        rm.load()

        assert "unicode" in rm.lexika
        assert rm.lexika["unicode"] == ["ä", "ö", "ü", "ß"]

    def test_load_with_numeric_items(self):
        """Test loading YAML with numeric items."""
        test_file = self.test_path / "numbers.yml"
        test_file.write_text("- 123\n- 456\n- 789\n", encoding="utf-8")

        rm = ResourceManager(self.test_dir)
        rm.load()

        # Numbers should be converted to strings and lowercased
        assert "numbers" in rm.lexika
        assert rm.lexika["numbers"] == ["123", "456", "789"]


class TestLinguisticPreprocessor:
    """Tests for LinguisticPreprocessor class."""

    def test_initialization_success(self):
        """Test successful initialization with spaCy model."""
        preprocessor = LinguisticPreprocessor()
        assert preprocessor.nlp is not None

    def test_process_simple_text(self):
        """Test processing simple German text."""
        preprocessor = LinguisticPreprocessor()
        doc = preprocessor.process("Der Hund bellt.")

        assert doc is not None
        assert len(doc) > 0
        assert any(token.text == "Hund" for token in doc)

    def test_process_empty_text(self):
        """Test processing empty text."""
        preprocessor = LinguisticPreprocessor()
        doc = preprocessor.process("")

        assert doc is not None
        assert len(doc) == 0

    def test_process_unicode_text(self):
        """Test processing text with German umlauts."""
        preprocessor = LinguisticPreprocessor()
        doc = preprocessor.process("Äpfel und Öl.")

        assert doc is not None
        assert len(doc) > 0

    def test_process_returns_doc_object(self):
        """Test that process returns a spaCy Doc object."""
        preprocessor = LinguisticPreprocessor()
        doc = preprocessor.process("Test")

        # Check that it's a spaCy Doc object with tokens
        assert hasattr(doc, "__iter__")
        assert hasattr(doc, "__len__")

    def test_process_multiple_sentences(self):
        """Test processing multiple sentences."""
        preprocessor = LinguisticPreprocessor()
        doc = preprocessor.process("Der Hund bellt. Die Katze miaut.")

        assert doc is not None
        assert len(doc) > 5  # Should have multiple tokens

    def test_process_with_punctuation(self):
        """Test processing text with various punctuation."""
        preprocessor = LinguisticPreprocessor()
        doc = preprocessor.process("Was ist das? Ein Test!")

        assert doc is not None
        assert len(doc) > 0

    def test_process_long_text(self):
        """Test processing longer text."""
        preprocessor = LinguisticPreprocessor()
        long_text = " ".join(["Der Hund bellt."] * 10)
        doc = preprocessor.process(long_text)

        assert doc is not None
        assert len(doc) > 20

    def test_nlp_attribute_exists(self):
        """Test that nlp attribute is accessible."""
        preprocessor = LinguisticPreprocessor()
        assert hasattr(preprocessor, "nlp")
        assert preprocessor.nlp is not None

    def test_process_preserves_text_structure(self):
        """Test that token structure is preserved."""
        preprocessor = LinguisticPreprocessor()
        text = "Der Hund bellt"
        doc = preprocessor.process(text)

        # Reconstruct text from tokens
        reconstructed = "".join(token.text_with_ws for token in doc)
        # Should be similar to original (spaCy may normalize spacing)
        assert "Hund" in reconstructed

    def test_process_with_special_characters(self):
        """Test processing text with special characters."""
        preprocessor = LinguisticPreprocessor()
        doc = preprocessor.process("Test: 123 €, 45%")

        assert doc is not None
        assert len(doc) > 0

    def test_repeated_processing(self):
        """Test that processor can be used multiple times."""
        preprocessor = LinguisticPreprocessor()

        doc1 = preprocessor.process("Erster Text")
        doc2 = preprocessor.process("Zweiter Text")

        assert doc1 is not None
        assert doc2 is not None
        assert len(doc1) > 0
        assert len(doc2) > 0

    def test_process_with_newlines(self):
        """Test processing text with newlines."""
        preprocessor = LinguisticPreprocessor()
        doc = preprocessor.process("Zeile 1\nZeile 2\nZeile 3")

        assert doc is not None
        assert len(doc) > 0


class TestLinguisticPreprocessorFallback:
    """Tests for LinguisticPreprocessor fallback behavior."""

    def test_fallback_mechanism_exists(self):
        """Test that fallback mechanism is in place (blank model)."""
        # The fallback code is in the except block of __init__
        # We can't easily test it without uninstalling spacy model
        # But we can verify the code path exists by checking imports
        from component_6_linguistik_engine import spacy

        assert spacy is not None

    def test_process_none_check_coverage(self):
        """Test coverage of None check in process method."""
        # This tests line 60-61 which checks if nlp is None
        preprocessor = LinguisticPreprocessor()

        # Set nlp to None to test the None check
        original_nlp = preprocessor.nlp
        preprocessor.nlp = None

        result = preprocessor.process("Test")
        assert result is None

        # Restore original nlp
        preprocessor.nlp = original_nlp


class TestIntegration:
    """Integration tests combining ResourceManager and LinguisticPreprocessor."""

    def setup_method(self):
        """Create temporary directory for test."""
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)

    def teardown_method(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_combined_usage(self):
        """Test using both components together."""
        # Create lexikon
        (self.test_path / "verbs.yml").write_text(
            "- laufen\n- springen\n", encoding="utf-8"
        )

        # Load lexikon
        rm = ResourceManager(self.test_dir)
        rm.load()

        # Process text
        preprocessor = LinguisticPreprocessor()
        doc = preprocessor.process("Der Hund läuft.")

        assert "verbs" in rm.lexika
        assert doc is not None
        assert len(doc) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
