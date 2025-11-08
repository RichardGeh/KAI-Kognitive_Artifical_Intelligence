"""
fix_unicode_in_components.py

Ersetzt Unicode-Zeichen in Logger-Ausgaben von Component-Dateien.

WICHTIG:
- Ersetzt nur in logger.xxx() Aufrufen
- Lässt UI-Strings und andere Strings unverändert
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

import re
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


def fix_unicode_in_logger_calls(file_path: Path) -> tuple[bool, int]:
    """
    Ersetzt Unicode-Zeichen nur in logger-Aufrufen.

    Args:
        file_path: Pfad zur zu bearbeitenden Datei

    Returns:
        (wurde_geändert, anzahl_ersetzungen)
    """
    try:
        content = file_path.read_text(encoding="utf-8")
        original_content = content
        replacements_made = 0

        # Regex-Muster für logger-Aufrufe
        # Matcht: logger.debug(...), logger.info(...), etc.
        logger_pattern = re.compile(
            r"(logger\.(debug|info|warning|error|critical)\s*\([^)]+\))", re.DOTALL
        )

        def replace_in_match(match):
            """Ersetzt Unicode-Zeichen nur innerhalb eines logger-Aufrufs."""
            nonlocal replacements_made
            logger_call = match.group(0)
            modified_call = logger_call

            for unicode_char, ascii_char in UNICODE_REPLACEMENTS.items():
                count = modified_call.count(unicode_char)
                if count > 0:
                    modified_call = modified_call.replace(unicode_char, ascii_char)
                    replacements_made += count

            return modified_call

        # Ersetze nur in logger-Aufrufen
        content = logger_pattern.sub(replace_in_match, content)

        # Schreibe nur, wenn Änderungen vorgenommen wurden
        if content != original_content and replacements_made > 0:
            file_path.write_text(content, encoding="utf-8")
            return True, replacements_made

        return False, 0

    except Exception as e:
        print(f"FEHLER bei {file_path}: {e}")
        return False, 0


def main():
    """Hauptfunktion zum Fixen aller Component-Dateien."""
    base_dir = Path(__file__).parent

    # Liste von Dateien, die bearbeitet werden sollen
    files_to_fix = [
        "component_*.py",
        "kai_*.py",
        "main_ui_graphical.py",
        "settings_ui.py",
        "logging_ui.py",
        "setup_initial_knowledge.py",
    ]

    # Sammle alle Dateien
    all_files = []
    for pattern in files_to_fix:
        all_files.extend(base_dir.glob(pattern))

    # Entferne Duplikate
    all_files = list(set(all_files))

    # Exkludiere das Fix-Skript selbst
    all_files = [f for f in all_files if "fix_unicode" not in f.name]

    print(f"Durchsuche {len(all_files)} Component-Dateien...\n")

    files_modified = 0
    total_replacements = 0

    for file_path in sorted(all_files):
        was_modified, replacements = fix_unicode_in_logger_calls(file_path)
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
