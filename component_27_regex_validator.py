# component_27_regex_validator.py
"""
Regex-Validierung für Extraktionsregeln.

Prüft Regex-Muster auf Syntaxfehler und Kompatibilität mit KAI-Anforderungen.
"""

import re
from typing import Dict, List, Optional, Tuple

from component_15_logging_config import get_logger

logger = get_logger(__name__)


class RegexValidator:
    """
    Validiert Regex-Muster für Extraktionsregeln.

    KAI-Anforderungen:
    - Muster müssen genau 2 Capture-Groups haben (Subject, Object)
    - Muster dürfen keine fehlerhaften Syntax haben
    - Muster sollten nicht zu restriktiv/permissiv sein
    """

    def __init__(self):
        logger.info("RegexValidator initialisiert")

    def validate_pattern(
        self, pattern: str
    ) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """
        Validiert ein Regex-Muster umfassend.

        Args:
            pattern: Das zu validierende Regex-Muster

        Returns:
            (is_valid, error_message, details)
            - is_valid: True wenn Muster gültig ist
            - error_message: Nutzerfreundliche Fehlermeldung oder None
            - details: Dict mit technischen Details (für Debugging)
        """
        details = {"pattern": pattern, "length": len(pattern), "capture_groups": 0}

        # 1. Prüfe auf leeres Pattern
        if not pattern or not pattern.strip():
            return (False, "[ERROR] Das Muster darf nicht leer sein.", details)

        # 2. Prüfe Regex-Syntax
        try:
            compiled = re.compile(pattern, re.IGNORECASE)
            details["compiled"] = True
        except re.error as e:
            return (
                False,
                f"[ERROR] Ungültige Regex-Syntax: {e}\n\n💡 Tipp: Prüfe Klammern, Backslashes und Sonderzeichen.",
                details,
            )

        # 3. Zähle Capture-Groups
        capture_groups = compiled.groups
        details["capture_groups"] = capture_groups

        if capture_groups < 2:
            return (
                False,
                f"[ERROR] Zu wenige Capture-Groups: {capture_groups} gefunden, 2 benötigt.\n\n"
                f"💡 Tipp: Extraktionsregeln brauchen genau 2 Gruppen:\n"
                f"   - Gruppe 1: Subject (was/wer)\n"
                f"   - Gruppe 2: Object (was/wohin/wie)\n\n"
                f"Beispiel: r'^(.+) ist (ein|eine) (.+)$' hat 3 Gruppen (zu viele)\n"
                f"Richtig:  r'^(.+) ist (?:ein|eine) (.+)$' hat 2 Gruppen (mit non-capturing group (?:...))",
                details,
            )

        if capture_groups > 2:
            return (
                False,
                f"[ERROR] Zu viele Capture-Groups: {capture_groups} gefunden, 2 benötigt.\n\n"
                f"💡 Tipp: Verwende non-capturing groups (?:...) für optionale Teile:\n"
                f"   Falsch: r'^(.+) (ein|eine) (.+)$' -> 3 Gruppen\n"
                f"   Richtig: r'^(.+) (?:ein|eine) (.+)$' -> 2 Gruppen",
                details,
            )

        # 4. Prüfe auf sinnvolle Muster (Warnings, nicht blocking)
        warnings = self._check_pattern_warnings(pattern, compiled)
        if warnings:
            details["warnings"] = warnings
            logger.info(
                "Regex-Warnungen", extra={"pattern": pattern, "warnings": warnings}
            )

        # 5. Teste gegen Beispiele
        test_results = self._test_pattern_examples(pattern, compiled)
        details["test_results"] = test_results

        return (True, None, details)

    def _check_pattern_warnings(self, pattern: str, compiled: re.Pattern) -> List[str]:
        """
        Prüft auf potentielle Probleme (nicht blocking).

        Returns:
            Liste von Warnungen
        """
        warnings = []

        # Warning 1: Zu permissive Patterns (.*) statt (.+)
        if ".*" in pattern and ".+" not in pattern:
            warnings.append(
                "[WARNING] Pattern verwendet '.*' (erlaubt leere Matches). "
                "Empfehlung: '.+' verwenden für nicht-leere Matches."
            )

        # Warning 2: Kein Anker (^ oder $)
        if not pattern.startswith("^") and not pattern.endswith("$"):
            warnings.append(
                "[WARNING] Pattern hat keine Anker (^ und $). "
                "Empfehlung: Anker verwenden für eindeutige Matches."
            )

        # Warning 3: Sehr kurzes Pattern (zu allgemein)
        if len(pattern) < 10:
            warnings.append(
                "[WARNING] Pattern ist sehr kurz (< 10 Zeichen). "
                "Prüfe ob es nicht zu allgemein ist."
            )

        return warnings

    def _test_pattern_examples(
        self, pattern: str, compiled: re.Pattern
    ) -> Dict[str, any]:
        """
        Testet Pattern gegen typische Beispielsätze.

        Returns:
            Dict mit Test-Ergebnissen
        """
        test_sentences = [
            "Ein Hund ist ein Tier",
            "Der Apfel ist rot",
            "Hunde sind Tiere",
            "X schmeckt Y",
            "A befindet sich in B",
        ]

        results = {"total": len(test_sentences), "matches": 0, "examples": []}

        for sentence in test_sentences:
            match = compiled.match(sentence, re.IGNORECASE)
            if match:
                results["matches"] += 1
                results["examples"].append(
                    {"sentence": sentence, "groups": match.groups()}
                )

        return results

    def suggest_fixes(self, pattern: str) -> List[str]:
        """
        Schlägt Korrekturen für häufige Fehler vor.

        Args:
            pattern: Das fehlerhafte Pattern

        Returns:
            Liste von Vorschlägen
        """
        suggestions = []

        # Häufiger Fehler 1: Zu viele Gruppen wegen (ein|eine)
        if "(ein|eine)" in pattern:
            fixed = pattern.replace("(ein|eine)", "(?:ein|eine)")
            suggestions.append(f"Ersetze '(ein|eine)' durch '(?:ein|eine)': {fixed}")

        # Häufiger Fehler 2: Zu viele Gruppen wegen (der|die|das)
        if "(der|die|das)" in pattern:
            fixed = pattern.replace("(der|die|das)", "(?:der|die|das)")
            suggestions.append(
                f"Ersetze '(der|die|das)' durch '(?:der|die|das)': {fixed}"
            )

        # Häufiger Fehler 3: Fehlende Anker
        if not pattern.startswith("^"):
            suggestions.append(f"Füge Anker am Anfang hinzu: ^{pattern}")

        if not pattern.endswith("$"):
            suggestions.append(f"Füge Anker am Ende hinzu: {pattern}$")

        # Häufiger Fehler 4: .* statt .+
        if ".*" in pattern:
            fixed = pattern.replace(".*", ".+")
            suggestions.append(
                f"Ersetze '.*' durch '.+' für nicht-leere Matches: {fixed}"
            )

        return suggestions

    def format_validation_result(
        self, is_valid: bool, error_message: Optional[str], details: Optional[Dict]
    ) -> str:
        """
        Formatiert Validierungsergebnis für Nutzer-Ausgabe.

        Args:
            is_valid: Ob Pattern gültig ist
            error_message: Fehlermeldung
            details: Zusätzliche Details

        Returns:
            Formatierte Ausgabe
        """
        if is_valid:
            output = "[OK] Regex-Muster ist gültig!\n\n"

            if details:
                output += "**Details:**\n"
                output += f"  - Capture-Groups: {details.get('capture_groups', 0)}\n"

                warnings = details.get("warnings", [])
                if warnings:
                    output += "\n**Warnungen:**\n"
                    for warning in warnings:
                        output += f"  {warning}\n"

                test_results = details.get("test_results", {})
                if test_results:
                    matches = test_results.get("matches", 0)
                    total = test_results.get("total", 0)
                    output += f"\n**Test-Matches:** {matches}/{total} Beispiele\n"

                    examples = test_results.get("examples", [])
                    if examples:
                        output += "\n**Beispiel-Matches:**\n"
                        for example in examples[:3]:  # Max 3 Beispiele
                            output += (
                                f"  '{example['sentence']}' -> {example['groups']}\n"
                            )

            return output
        else:
            output = error_message or "[ERROR] Regex-Muster ist ungültig."

            if details and details.get("pattern"):
                output += f"\n\nDein Muster: {details['pattern']}"

            return output


# Singleton-Instanz für globalen Zugriff
_regex_validator_instance: Optional[RegexValidator] = None


def get_regex_validator() -> RegexValidator:
    """
    Gibt Singleton-Instanz des RegexValidators zurück.
    """
    global _regex_validator_instance
    if _regex_validator_instance is None:
        _regex_validator_instance = RegexValidator()
    return _regex_validator_instance


if __name__ == "__main__":
    # Test-Code
    validator = RegexValidator()

    test_patterns = [
        # Gültige Muster
        (r"^(.+) ist (?:ein|eine) (.+)$", "Gültiges Muster mit non-capturing group"),
        (r"^(.+) hat (.+)$", "Einfaches gültiges Muster"),
        # Ungültige Muster
        (r"^(.+) ist (ein|eine) (.+)$", "Zu viele Gruppen (3 statt 2)"),
        (r"^(.+)$", "Zu wenige Gruppen (1 statt 2)"),
        (r"^[(.+) ist (.+)$", "Syntax-Fehler (unbalanced bracket)"),
        ("", "Leeres Muster"),
        # Muster mit Warnungen
        (r"(.*) ist (.*)", "Keine Anker + .* statt .+"),
    ]

    print("=== Testing RegexValidator ===\n")

    for pattern, description in test_patterns:
        print(f"Test: {description}")
        print(f"Pattern: {pattern}")

        is_valid, error_msg, details = validator.validate_pattern(pattern)

        if is_valid:
            print("[OK] GÜLTIG")
            if details.get("warnings"):
                print(f"Warnungen: {details['warnings']}")
        else:
            print(f"[X] UNGÜLTIG: {error_msg}")

            suggestions = validator.suggest_fixes(pattern)
            if suggestions:
                print("\n💡 Vorschläge:")
                for suggestion in suggestions:
                    print(f"  - {suggestion}")

        print("\n" + "-" * 80 + "\n")
