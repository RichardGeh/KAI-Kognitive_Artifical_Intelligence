"""
fix_unicode_in_tests.py

Ersetzt Unicode-Zeichen in Test-Dateien durch ASCII-Alternativen.

Problem:
- Windows-Konsole (cp1252) kann Unicode-Zeichen wie ✓, ✗, → nicht encodieren
- pytest deaktiviert den encoding fix, um nicht mit Output-Capturing zu interferieren
- Dies führt zu UnicodeEncodeError in Test-Logs

Lösung:
- Ersetzt alle problematischen Unicode-Zeichen durch ASCII-Alternativen
"""

import io

# WICHTIG: Encoding fix ZUERST importieren, vor allen anderen Imports
import sys

# Erzwinge UTF-8 für stdout/stderr
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )

from pathlib import Path

# Mapping von Unicode-Zeichen zu ASCII-Alternativen
UNICODE_REPLACEMENTS = {
    "✓": "[OK]",
    "✗": "[X]",
    "→": "->",
    "←": "<-",
    "↔": "<->",
    "📋": "[INFO]",
    "❌": "[ERROR]",
    "⚠️": "[WARNING]",
    "✅": "[SUCCESS]",
    "ℹ️": "[INFO]",
    "🤖": "[AI]",
    "•": "*",
    "·": "-",
    "…": "...",
}


def fix_unicode_in_file(file_path: Path) -> tuple[bool, int]:
    """
    Ersetzt Unicode-Zeichen in einer Datei durch ASCII-Alternativen.

    Args:
        file_path: Pfad zur zu bearbeitenden Datei

    Returns:
        (wurde_geändert, anzahl_ersetzungen)
    """
    try:
        # Lese Datei mit UTF-8
        content = file_path.read_text(encoding="utf-8")
        original_content = content
        replacements_made = 0

        # Ersetze alle Unicode-Zeichen
        for unicode_char, ascii_char in UNICODE_REPLACEMENTS.items():
            count = content.count(unicode_char)
            if count > 0:
                content = content.replace(unicode_char, ascii_char)
                replacements_made += count
                print(
                    f"  {file_path.name}: '{unicode_char}' -> '{ascii_char}' ({count}x)"
                )

        # Schreibe nur, wenn Änderungen vorgenommen wurden
        if content != original_content:
            file_path.write_text(content, encoding="utf-8")
            return True, replacements_made

        return False, 0

    except Exception as e:
        print(f"FEHLER bei {file_path}: {e}")
        return False, 0


def main():
    """Hauptfunktion zum Fixen aller Test-Dateien."""
    tests_dir = Path(__file__).parent / "tests"

    # Finde alle Python-Dateien in tests/
    test_files = list(tests_dir.glob("test_*.py"))

    print(f"Durchsuche {len(test_files)} Test-Dateien...\n")

    files_modified = 0
    total_replacements = 0

    for test_file in test_files:
        was_modified, replacements = fix_unicode_in_file(test_file)
        if was_modified:
            files_modified += 1
            total_replacements += replacements

    print(f"\n{'='*60}")
    print(f"ZUSAMMENFASSUNG:")
    print(f"  Dateien durchsucht: {len(test_files)}")
    print(f"  Dateien geändert:   {files_modified}")
    print(f"  Ersetzungen gesamt: {total_replacements}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
