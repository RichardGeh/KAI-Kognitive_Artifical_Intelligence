"""Test if PatternOrchestrator early exit works for logic puzzles"""
import re

puzzle = """
Vier Personen (Anna, Ben, Clara, Daniel) tragen jeweils eine Farbe: Rot, Blau, Gruen, Gelb.
1. Anna traegt nicht Rot.
2. Ben traegt Blau.
3. Clara traegt weder Gruen noch Gelb.
4. Daniel traegt nicht die gleiche Farbe wie Anna.
5. Genau eine Person traegt Rot.

Wer traegt welche Farbe?
"""

# Simulate the early exit logic from component_24_pattern_orchestrator.py

# Early Exit: Commands
command_prefixes = [
    r"^\s*definiere:",
    r"^\s*lerne muster:",
    r"^\s*ingestiere text:",
    r"^\s*lerne:",
    r"^\s*(?:lese datei|ingestiere dokument|verarbeite pdf|lade datei):",
]

for pattern in command_prefixes:
    if re.match(pattern, puzzle, re.IGNORECASE):
        print(f"[SKIP] Would skip on command pattern: {pattern}")
        exit(0)

# Early Exit: Questions (single-line OR multi-line)
question_patterns = [
    r"(?:^|\n)\s*(?:was|wer|wie|wo|wann|warum|wieso|weshalb|wozu|welche)\s+",
    r"\?\s*$",
]

for pattern in question_patterns:
    if re.search(pattern, puzzle, re.IGNORECASE | re.MULTILINE):
        print(f"[SKIP] Would skip on question pattern: {pattern}")
        print("[SUCCESS] Early exit would be triggered - typo detection SKIPPED")
        exit(0)

# Early Exit: Multi-line structured inputs
if re.search(r"(?:^|\n)\s*\d+\.\s+.+(?:\n\s*\d+\.\s+.+)+", puzzle, re.MULTILINE):
    print("[SKIP] Would skip on numbered list pattern")
    print("[SUCCESS] Early exit would be triggered - typo detection SKIPPED")
    exit(0)

print("[FAIL] No early exit triggered - typo detection would RUN")
exit(1)
