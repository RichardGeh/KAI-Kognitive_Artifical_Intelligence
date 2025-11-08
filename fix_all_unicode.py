"""
fix_all_unicode.py

Ersetzt ALLE Unicode-Zeichen in allen Python-Dateien durch ASCII-Alternativen.
Einfacher Ansatz ohne komplexes Regex-Matching.
"""

import io

# WICHTIG: Encoding fix ZUERST
import sys

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


def fix_file(file_path: Path) -> tuple[bool, int]:
    """Ersetzt alle Unicode-Zeichen in einer Datei."""
    try:
        content = file_path.read_text(encoding="utf-8")
        original_content = content
        replacements_made = 0

        for unicode_char, ascii_char in UNICODE_REPLACEMENTS.items():
            count = content.count(unicode_char)
            if count > 0:
                content = content.replace(unicode_char, ascii_char)
                replacements_made += count

        if content != original_content and replacements_made > 0:
            file_path.write_text(content, encoding="utf-8")
            return True, replacements_made

        return False, 0

    except Exception as e:
        print(f"FEHLER bei {file_path.name}: {e}")
        return False, 0


def main():
    """Hauptfunktion."""
    base_dir = Path(__file__).parent

    # Alle Python-Dateien (außer Fix-Skripte)
    all_files = list(base_dir.glob("*.py")) + list((base_dir / "tests").glob("*.py"))
    all_files = [
        f
        for f in all_files
        if "fix_unicode" not in f.name and "fix_all_unicode" not in f.name
    ]

    print(f"Durchsuche {len(all_files)} Python-Dateien...\n")

    files_modified = 0
    total_replacements = 0

    for file_path in sorted(all_files):
        was_modified, replacements = fix_file(file_path)
        if was_modified:
            print(f"  {file_path.name}: {replacements} Ersetzungen")
            files_modified += 1
            total_replacements += replacements

    print(f"\n{'='*60}")
    print(f"ZUSAMMENFASSUNG:")
    print(f"  Dateien durchsucht: {len(all_files)}")
    print(f"  Dateien geändert:   {files_modified}")
    print(f"  Ersetzungen gesamt: {total_replacements}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
