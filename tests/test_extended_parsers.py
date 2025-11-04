"""
test_extended_parsers.py

Schnelltest für erweiterte Document Parser (TXT, MD, HTML).
"""

# Fix Windows cp1252 encoding issue
import kai_encoding_fix  # noqa: F401

import tempfile
import os
from component_28_document_parser import (
    DocumentParserFactory,
    TxtParser,
    MarkdownParser,
    HtmlParser,
)


def test_txt_parser():
    """Test TxtParser mit temporärer Datei."""
    print("=== Test TxtParser ===")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write("Dies ist ein Test.\nZweite Zeile.\n")
        temp_path = f.name

    try:
        parser = TxtParser()
        text = parser.extract_text(temp_path)
        print(f"✓ Text extrahiert: {len(text)} Zeichen")
        print(f"  Inhalt: {text[:50]}...")
        assert "Dies ist ein Test" in text
        print("✓ TxtParser funktioniert!\n")
    finally:
        os.unlink(temp_path)


def test_markdown_parser():
    """Test MarkdownParser mit temporärer Datei."""
    print("=== Test MarkdownParser ===")

    markdown_content = """
# Überschrift

Dies ist ein **fetter** Text mit einem [Link](http://example.com).

- Liste Item 1
- Liste Item 2

```python
code block
```
"""

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, encoding="utf-8"
    ) as f:
        f.write(markdown_content)
        temp_path = f.name

    try:
        parser = MarkdownParser()
        text = parser.extract_text(temp_path)
        print(f"✓ Text extrahiert: {len(text)} Zeichen")
        print(f"  Inhalt: {text[:100]}...")
        assert "Überschrift" in text
        assert "fetter" in text  # Markdown ** sollte entfernt sein
        assert "Link" in text  # Link-Text sollte erhalten bleiben
        assert "code block" not in text  # Code-Blöcke sollten entfernt sein
        print("✓ MarkdownParser funktioniert!\n")
    finally:
        os.unlink(temp_path)


def test_html_parser():
    """Test HtmlParser mit temporärer Datei."""
    print("=== Test HtmlParser ===")

    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Test</title>
    <script>var x = 1;</script>
</head>
<body>
    <h1>Überschrift</h1>
    <p>Dies ist ein Absatz.</p>
    <div>
        <p>Zweiter Absatz.</p>
    </div>
</body>
</html>
"""

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".html", delete=False, encoding="utf-8"
    ) as f:
        f.write(html_content)
        temp_path = f.name

    try:
        parser = HtmlParser()
        text = parser.extract_text(temp_path)
        print(f"✓ Text extrahiert: {len(text)} Zeichen")
        print(f"  Inhalt: {text[:100]}...")
        assert "Überschrift" in text
        assert "Dies ist ein Absatz" in text
        assert "var x = 1" not in text  # JavaScript sollte entfernt sein
        print("✓ HtmlParser funktioniert!\n")
    finally:
        os.unlink(temp_path)


def test_factory():
    """Test DocumentParserFactory."""
    print("=== Test DocumentParserFactory ===")

    extensions = DocumentParserFactory.get_supported_extensions()
    print(f"Unterstützte Formate: {len(extensions)}")
    print(f"  {', '.join(extensions)}")

    assert ".txt" in extensions
    assert ".md" in extensions
    assert ".html" in extensions
    assert ".pdf" in extensions
    assert ".docx" in extensions

    print("✓ Factory unterstützt alle Formate!\n")


if __name__ == "__main__":
    print("=== Extended Document Parser Test ===\n")

    try:
        test_factory()
        test_txt_parser()
        test_markdown_parser()
        test_html_parser()

        print("=" * 50)
        print("✓ ALLE TESTS BESTANDEN!")
        print("=" * 50)

    except AssertionError as e:
        print(f"\n✗ TEST FEHLGESCHLAGEN: {e}")
    except Exception as e:
        print(f"\n✗ FEHLER: {e}")
        import traceback

        traceback.print_exc()
