"""
kai_encoding_fix.py

Zentrale Lösung für Unicode-Encoding-Probleme unter Windows.

Problem:
- Windows-Konsole verwendet standardmäßig cp1252 statt UTF-8
- Unicode-Zeichen ([OK], [X], ->, [INFO], [ERROR], etc.) führen zu UnicodeEncodeError
- Python 3.7+ hat UTF-8 als Standard, aber Windows überschreibt dies

Lösung:
- Erzwingt UTF-8 für stdout/stderr mit errors='replace' Fallback
- Automatische Anwendung beim Import
- Kompatibel mit Logging-System und PySide6

Verwendung:
    # Am Anfang jeder Entry-Point-Datei:
    import kai_encoding_fix  # Automatische Aktivierung beim Import

    # Manuelle Aktivierung (optional):
    kai_encoding_fix.ensure_utf8_encoding()
"""

import sys
import io
from typing import Optional


def ensure_utf8_encoding(force: bool = False) -> bool:
    """
    Stellt sicher, dass stdout und stderr UTF-8 verwenden.

    Args:
        force: Wenn True, erzwingt Rekonfiguration auch wenn bereits UTF-8

    Returns:
        True wenn erfolgreich rekonfiguriert, False wenn bereits UTF-8
    """
    # Prüfe, ob bereits UTF-8 verwendet wird
    if (
        not force
        and hasattr(sys.stdout, "encoding")
        and sys.stdout.encoding.lower().startswith("utf")
    ):
        return False

    try:
        # Rekonfiguriere stdout für UTF-8 mit errors='replace' Fallback
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        else:
            # Fallback für ältere Python-Versionen
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer,
                encoding="utf-8",
                errors="replace",
                line_buffering=True,
            )

        # Rekonfiguriere stderr für UTF-8
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        else:
            # Fallback für ältere Python-Versionen
            sys.stderr = io.TextIOWrapper(
                sys.stderr.buffer,
                encoding="utf-8",
                errors="replace",
                line_buffering=True,
            )

        return True

    except Exception as e:
        # Falls Rekonfiguration fehlschlägt, logge Warnung (ohne Crash)
        # Verwende print statt logging, da Logging möglicherweise noch nicht initialisiert ist
        print(
            f"[WARNING] Konnte stdout/stderr nicht auf UTF-8 umstellen: {e}",
            file=sys.stderr,
        )
        return False


def get_safe_unicode_string(
    text: str, fallback_map: Optional[dict[str, str]] = None
) -> str:
    """
    Konvertiert Unicode-Zeichen in sichere ASCII-Alternativen für nicht-UTF-8 Terminals.

    Args:
        text: Der zu konvertierende Text
        fallback_map: Optional - Dict mit Unicode -> ASCII Mappings

    Returns:
        Text mit sicheren Zeichen
    """
    # Standard-Fallback-Mappings für häufige Unicode-Zeichen
    default_map = {
        "[OK]": "[OK]",
        "[X]": "[X]",
        "->": "->",
        "<-": "<-",
        "<->": "<->",
        "[INFO]": "[INFO]",
        "[ERROR]": "[ERROR]",
        "[WARNING]": "[WARNING]",
        "[SUCCESS]": "[SUCCESS]",
        "[INFO]": "[INFO]",
        "[AI]": "[AI]",
        "*": "*",
        "-": "-",
        "...": "...",
        '"': '"',
        '"': '"',
        """: "'",
        """: "'",
    }

    # Merge mit benutzerdefinierten Mappings
    if fallback_map:
        default_map.update(fallback_map)

    # Ersetze alle gemappten Zeichen
    result = text
    for unicode_char, ascii_char in default_map.items():
        result = result.replace(unicode_char, ascii_char)

    return result


def is_utf8_capable() -> bool:
    """
    Prüft, ob das aktuelle Terminal UTF-8 unterstützt.

    Returns:
        True wenn UTF-8 unterstützt wird
    """
    try:
        return sys.stdout.encoding.lower().startswith("utf")
    except AttributeError:
        return False


def print_safe(text: str, *args, **kwargs) -> None:
    """
    Sichere Print-Funktion, die automatisch auf ASCII-Fallback umschaltet wenn nötig.

    Args:
        text: Der zu druckende Text
        *args, **kwargs: Weitergeleitet an print()
    """
    if not is_utf8_capable():
        text = get_safe_unicode_string(text)

    print(text, *args, **kwargs)


# === Automatische Aktivierung beim Import ===

# Deaktiviere Encoding-Fix bei pytest, da es mit Output-Capturing interferiert
_is_pytest = (
    "pytest" in sys.modules or "PYTEST_CURRENT_TEST" in __import__("os").environ
)

# Versuche UTF-8 zu erzwingen (außer bei pytest)
_was_reconfigured = False if _is_pytest else ensure_utf8_encoding()

# Nur loggen, wenn tatsächlich rekonfiguriert wurde (sonst Spam bei jedem Import)
if _was_reconfigured:
    # Verwende print statt logging, da logging_config.py möglicherweise noch nicht geladen ist
    print(
        f"[kai_encoding_fix] stdout/stderr auf UTF-8 umgestellt "
        f"(vorher: {getattr(sys.stdout, 'encoding', 'unknown')})",
        file=sys.stderr,
    )


# === Öffentliche API ===

__all__ = [
    "ensure_utf8_encoding",
    "get_safe_unicode_string",
    "is_utf8_capable",
    "print_safe",
]


if __name__ == "__main__":
    # Test-Code
    print("\n=== Unicode Encoding Test ===")
    print(f"Current stdout encoding: {sys.stdout.encoding}")
    print(f"Current stderr encoding: {sys.stderr.encoding}")
    print(f"UTF-8 capable: {is_utf8_capable()}")

    print("\n=== Unicode Characters Test ===")
    test_strings = [
        "[OK] Success",
        "[X] Failure",
        "Hund -> Säugetier -> Tier",
        "[INFO] Information",
        "[ERROR] Error",
        "[WARNING] Warning",
        "[AI] Generated with Claude Code",
    ]

    for s in test_strings:
        try:
            print(f"  {s}")
        except UnicodeEncodeError as e:
            print(f"  [FAILED] {get_safe_unicode_string(s)} (Error: {e})")

    print("\n=== Fallback Test ===")
    print("Original:", "[OK] -> [ERROR]")
    print("Fallback:", get_safe_unicode_string("[OK] -> [ERROR]"))
