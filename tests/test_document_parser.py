"""
Test suite for component_28_document_parser.py

Tests all parser classes, factory functionality, error handling,
and utility functions.

Note: This test suite uses sys.modules mocks for external dependencies
to allow testing without requiring python-docx and pdfplumber to be installed.
"""

import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock the external dependencies BEFORE importing the module
mock_docx = MagicMock()
mock_pdfplumber = MagicMock()
sys.modules["docx"] = mock_docx
sys.modules["pdfplumber"] = mock_pdfplumber

# Now import the module
from component_28_document_parser import (
    DocumentParser,
    DocxParser,
    PdfParser,
    DocumentParserFactory,
    extract_text_from_document,
    get_document_info,
)
from kai_exceptions import DocumentParseError


class TestDocumentParserBase:
    """Tests for the abstract DocumentParser base class."""

    def test_validate_file_not_found(self, tmp_path):
        """Test file validation with non-existent file."""

        class ConcreteParser(DocumentParser):
            def extract_text(self, filepath: str) -> str:
                return ""

            def get_supported_extensions(self):
                return [".txt"]

        parser = ConcreteParser()
        non_existent = tmp_path / "does_not_exist.txt"

        with pytest.raises(FileNotFoundError):
            parser.validate_file(str(non_existent))

    def test_validate_file_not_readable(self, tmp_path):
        """Test file validation with directory instead of file."""

        class ConcreteParser(DocumentParser):
            def extract_text(self, filepath: str) -> str:
                return ""

            def get_supported_extensions(self):
                return [".txt"]

        parser = ConcreteParser()
        directory = tmp_path / "test_dir"
        directory.mkdir()

        with pytest.raises(DocumentParseError) as exc_info:
            parser.validate_file(str(directory))

        assert "keine Datei" in str(exc_info.value)


class TestDocxParser:
    """Tests for DocxParser class."""

    def test_docx_parser_initialization(self):
        """Test successful DocxParser initialization."""
        parser = DocxParser()
        assert parser._docx_module is not None
        assert parser._docx_module == mock_docx

    def test_get_supported_extensions(self):
        """Test supported extensions for DocxParser."""
        parser = DocxParser()
        extensions = parser.get_supported_extensions()
        assert ".docx" in extensions
        assert ".DOCX" in extensions
        assert len(extensions) == 2

    def test_extract_text_success(self, tmp_path):
        """Test successful text extraction from DOCX."""
        # Create test file
        test_file = tmp_path / "test.docx"
        test_file.write_text("dummy", encoding="utf-8")

        # Mock document with paragraphs
        mock_paragraph1 = Mock()
        mock_paragraph1.text = "First paragraph"
        mock_paragraph2 = Mock()
        mock_paragraph2.text = "Second paragraph"
        mock_paragraph3 = Mock()
        mock_paragraph3.text = ""  # Empty paragraph (should be skipped)

        mock_doc = Mock()
        mock_doc.paragraphs = [mock_paragraph1, mock_paragraph2, mock_paragraph3]
        mock_docx.Document.return_value = mock_doc

        # Extract text
        parser = DocxParser()
        text = parser.extract_text(str(test_file))

        # Verify
        assert "First paragraph" in text
        assert "Second paragraph" in text
        assert text == "First paragraph\n\nSecond paragraph"

        # Reset mock
        mock_docx.Document.reset_mock()

    def test_extract_text_error(self, tmp_path):
        """Test error handling during DOCX extraction."""
        # Create test file
        test_file = tmp_path / "corrupt.docx"
        test_file.write_text("dummy", encoding="utf-8")

        # Mock error
        mock_docx.Document.side_effect = Exception("Corrupt file")

        parser = DocxParser()

        with pytest.raises(DocumentParseError) as exc_info:
            parser.extract_text(str(test_file))

        assert "Konnte DOCX-Datei nicht parsen" in str(exc_info.value)
        assert exc_info.value.context["file_path"] == str(test_file)
        assert exc_info.value.context["file_format"] == "docx"

        # Reset mock
        mock_docx.Document.side_effect = None
        mock_docx.Document.reset_mock()


class TestPdfParser:
    """Tests for PdfParser class."""

    def test_pdf_parser_initialization(self):
        """Test successful PdfParser initialization."""
        parser = PdfParser()
        assert parser._pdfplumber_module is not None
        assert parser._pdfplumber_module == mock_pdfplumber

    def test_get_supported_extensions(self):
        """Test supported extensions for PdfParser."""
        parser = PdfParser()
        extensions = parser.get_supported_extensions()
        assert ".pdf" in extensions
        assert ".PDF" in extensions
        assert len(extensions) == 2

    def test_extract_text_success(self, tmp_path):
        """Test successful text extraction from PDF."""
        # Create test file
        test_file = tmp_path / "test.pdf"
        test_file.write_text("dummy", encoding="utf-8")

        # Mock PDF pages
        mock_page1 = Mock()
        mock_page1.extract_text.return_value = "Page 1 content"
        mock_page2 = Mock()
        mock_page2.extract_text.return_value = "Page 2 content"

        mock_pdf = Mock()
        mock_pdf.pages = [mock_page1, mock_page2]
        mock_pdf.__enter__ = Mock(return_value=mock_pdf)
        mock_pdf.__exit__ = Mock(return_value=False)
        mock_pdfplumber.open.return_value = mock_pdf

        # Extract text
        parser = PdfParser()
        text = parser.extract_text(str(test_file))

        # Verify
        assert "Page 1 content" in text
        assert "Page 2 content" in text
        assert text == "Page 1 content\n\nPage 2 content"

        # Reset mock
        mock_pdfplumber.open.reset_mock()

    def test_extract_text_empty_page(self, tmp_path):
        """Test extraction with empty page (e.g., image-only page)."""
        # Create test file
        test_file = tmp_path / "test.pdf"
        test_file.write_text("dummy", encoding="utf-8")

        # Mock PDF pages (page 2 is empty)
        mock_page1 = Mock()
        mock_page1.extract_text.return_value = "Page 1 content"
        mock_page2 = Mock()
        mock_page2.extract_text.return_value = None  # Empty page

        mock_pdf = Mock()
        mock_pdf.pages = [mock_page1, mock_page2]
        mock_pdf.__enter__ = Mock(return_value=mock_pdf)
        mock_pdf.__exit__ = Mock(return_value=False)
        mock_pdfplumber.open.return_value = mock_pdf

        # Extract text
        parser = PdfParser()
        text = parser.extract_text(str(test_file))

        # Verify (empty page should be skipped)
        assert "Page 1 content" in text
        assert text == "Page 1 content"

        # Reset mock
        mock_pdfplumber.open.reset_mock()

    def test_extract_text_error(self, tmp_path):
        """Test error handling during PDF extraction."""
        # Create test file
        test_file = tmp_path / "corrupt.pdf"
        test_file.write_text("dummy", encoding="utf-8")

        # Mock error
        mock_pdfplumber.open.side_effect = Exception("Corrupt file")

        parser = PdfParser()

        with pytest.raises(DocumentParseError) as exc_info:
            parser.extract_text(str(test_file))

        assert "Konnte PDF-Datei nicht parsen" in str(exc_info.value)
        assert exc_info.value.context["file_path"] == str(test_file)
        assert exc_info.value.context["file_format"] == "pdf"

        # Reset mock
        mock_pdfplumber.open.side_effect = None
        mock_pdfplumber.open.reset_mock()


class TestDocumentParserFactory:
    """Tests for DocumentParserFactory class."""

    def test_get_supported_extensions(self):
        """Test retrieval of supported extensions."""
        extensions = DocumentParserFactory.get_supported_extensions()
        assert ".docx" in extensions or ".DOCX" in extensions
        assert ".pdf" in extensions or ".PDF" in extensions
        # Check that we have at least the original 4 extensions (.docx, .DOCX, .pdf, .PDF)
        assert len(extensions) >= 4

    def test_is_supported_pdf(self):
        """Test format check for PDF."""
        assert DocumentParserFactory.is_supported("test.pdf") is True
        assert DocumentParserFactory.is_supported("test.PDF") is True

    def test_is_supported_docx(self):
        """Test format check for DOCX."""
        assert DocumentParserFactory.is_supported("test.docx") is True
        assert DocumentParserFactory.is_supported("test.DOCX") is True

    def test_is_supported_unsupported(self):
        """Test format check for unsupported format."""
        assert DocumentParserFactory.is_supported("test.xlsx") is False
        assert DocumentParserFactory.is_supported("test.doc") is False
        assert DocumentParserFactory.is_supported("test.pptx") is False

    def test_create_parser_pdf(self):
        """Test parser creation for PDF file."""
        parser = DocumentParserFactory.create_parser("document.pdf")
        assert isinstance(parser, PdfParser)

    def test_create_parser_docx(self):
        """Test parser creation for DOCX file."""
        parser = DocumentParserFactory.create_parser("document.docx")
        assert isinstance(parser, DocxParser)

    def test_create_parser_case_insensitive(self):
        """Test parser creation is case-insensitive."""
        parser_lower = DocumentParserFactory.create_parser("document.pdf")
        parser_upper = DocumentParserFactory.create_parser("document.PDF")
        assert type(parser_lower) == type(parser_upper)

    def test_create_parser_no_extension(self):
        """Test parser creation with file without extension."""
        with pytest.raises(DocumentParseError) as exc_info:
            DocumentParserFactory.create_parser("file_without_extension")

        assert "keine Erweiterung" in str(exc_info.value)

    def test_create_parser_unsupported(self):
        """Test parser creation with unsupported extension."""
        with pytest.raises(DocumentParseError) as exc_info:
            DocumentParserFactory.create_parser("document.xlsx")

        assert "nicht unterstützt" in str(exc_info.value)
        assert exc_info.value.context["file_format"] == ".xlsx"


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_extract_text_from_document_docx(self, tmp_path):
        """Test convenience function for DOCX extraction."""
        # Create test file
        test_file = tmp_path / "test.docx"
        test_file.write_text("dummy", encoding="utf-8")

        # Mock document
        mock_para = Mock()
        mock_para.text = "Test content"
        mock_doc = Mock()
        mock_doc.paragraphs = [mock_para]
        mock_docx.Document.return_value = mock_doc

        # Extract
        text = extract_text_from_document(str(test_file))

        # Verify
        assert "Test content" in text

        # Reset
        mock_docx.Document.reset_mock()

    def test_extract_text_from_document_pdf(self, tmp_path):
        """Test convenience function for PDF extraction."""
        # Create test file
        test_file = tmp_path / "test.pdf"
        test_file.write_text("dummy", encoding="utf-8")

        # Mock PDF
        mock_page = Mock()
        mock_page.extract_text.return_value = "Test PDF content"
        mock_pdf = Mock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__ = Mock(return_value=mock_pdf)
        mock_pdf.__exit__ = Mock(return_value=False)
        mock_pdfplumber.open.return_value = mock_pdf

        # Extract
        text = extract_text_from_document(str(test_file))

        # Verify
        assert "Test PDF content" in text

        # Reset
        mock_pdfplumber.open.reset_mock()

    def test_get_document_info(self, tmp_path):
        """Test document info retrieval."""
        # Create test file
        test_file = tmp_path / "test.pdf"
        test_file.write_text("dummy content", encoding="utf-8")

        # Get info
        info = get_document_info(str(test_file))

        # Verify
        assert info["filename"] == "test.pdf"
        assert info["format"] == ".pdf"
        assert info["size_bytes"] > 0
        assert info["is_supported"] is True
        assert str(test_file.absolute()) in info["filepath"]

    def test_get_document_info_not_found(self, tmp_path):
        """Test document info for non-existent file."""
        non_existent = tmp_path / "does_not_exist.pdf"

        with pytest.raises(FileNotFoundError):
            get_document_info(str(non_existent))

    def test_get_document_info_unsupported(self, tmp_path):
        """Test document info for unsupported format."""
        test_file = tmp_path / "test.xlsx"
        test_file.write_text("dummy", encoding="utf-8")

        info = get_document_info(str(test_file))
        assert info["is_supported"] is False
        assert info["format"] == ".xlsx"


class TestErrorHandling:
    """Tests for comprehensive error handling."""

    def test_exception_context_preservation_docx(self, tmp_path):
        """Test that exception context is preserved through wrapping for DOCX."""
        # Create test file
        test_file = tmp_path / "test.docx"
        test_file.write_text("dummy", encoding="utf-8")

        # Mock error with specific message
        original_error = ValueError("Specific error message")
        mock_docx.Document.side_effect = original_error

        parser = DocxParser()

        with pytest.raises(DocumentParseError) as exc_info:
            parser.extract_text(str(test_file))

        # Verify context
        exception = exc_info.value
        assert exception.context["file_path"] == str(test_file)
        assert exception.context["file_format"] == "docx"
        assert exception.original_exception == original_error

        # Reset
        mock_docx.Document.side_effect = None
        mock_docx.Document.reset_mock()

    def test_pdf_error_context(self, tmp_path):
        """Test PDF error context preservation."""
        # Create test file
        test_file = tmp_path / "test.pdf"
        test_file.write_text("dummy", encoding="utf-8")

        # Mock error
        original_error = IOError("File corrupted")
        mock_pdfplumber.open.side_effect = original_error

        parser = PdfParser()

        with pytest.raises(DocumentParseError) as exc_info:
            parser.extract_text(str(test_file))

        # Verify context
        exception = exc_info.value
        assert exception.context["file_path"] == str(test_file)
        assert exception.context["file_format"] == "pdf"
        assert exception.original_exception == original_error

        # Reset
        mock_pdfplumber.open.side_effect = None
        mock_pdfplumber.open.reset_mock()


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_workflow_docx(self, tmp_path):
        """Test complete workflow from factory to extraction for DOCX."""
        # Create test file
        test_file = tmp_path / "integration_test.docx"
        test_file.write_text("dummy", encoding="utf-8")

        # Mock document
        mock_para = Mock()
        mock_para.text = "Integration test content"
        mock_doc = Mock()
        mock_doc.paragraphs = [mock_para]
        mock_docx.Document.return_value = mock_doc

        # Full workflow
        text = extract_text_from_document(str(test_file))

        # Verify
        assert "Integration test content" in text

        # Reset
        mock_docx.Document.reset_mock()

    def test_full_workflow_pdf(self, tmp_path):
        """Test complete workflow from factory to extraction for PDF."""
        # Create test file
        test_file = tmp_path / "integration_test.pdf"
        test_file.write_text("dummy", encoding="utf-8")

        # Mock PDF
        mock_page = Mock()
        mock_page.extract_text.return_value = "Integration test PDF content"
        mock_pdf = Mock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__ = Mock(return_value=mock_pdf)
        mock_pdf.__exit__ = Mock(return_value=False)
        mock_pdfplumber.open.return_value = mock_pdf

        # Full workflow
        text = extract_text_from_document(str(test_file))

        # Verify
        assert "Integration test PDF content" in text

        # Reset
        mock_pdfplumber.open.reset_mock()

    def test_mixed_format_detection(self, tmp_path):
        """Test that factory correctly handles different formats."""
        # Create files
        pdf_file = tmp_path / "document.pdf"
        docx_file = tmp_path / "document.docx"
        pdf_file.write_text("dummy", encoding="utf-8")
        docx_file.write_text("dummy", encoding="utf-8")

        # Create parsers
        pdf_parser = DocumentParserFactory.create_parser(str(pdf_file))
        docx_parser = DocumentParserFactory.create_parser(str(docx_file))

        # Verify correct types
        assert isinstance(pdf_parser, PdfParser)
        assert isinstance(docx_parser, DocxParser)
        assert type(pdf_parser) != type(docx_parser)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
