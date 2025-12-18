"""Quick test to verify pattern recognition fix"""
import re

puzzle_text = """
Vier Personen (Anna, Ben, Clara, Daniel) tragen jeweils eine Farbe: Rot, Blau, Gruen, Gelb.
1. Anna traegt nicht Rot.
2. Ben traegt Blau.
3. Clara traegt weder Gruen noch Gelb.
4. Daniel traegt nicht die gleiche Farbe wie Anna.
5. Genau eine Person traegt Rot.

Wer traegt welche Farbe?
"""

# Test 1: Check for question patterns
question_patterns = [
    r"(?:^|\n)\s*(?:was|wer|wie|wo|wann|warum|wieso|weshalb|wozu|welche)\s+",
    r"\?\s*$",
]

for pattern in question_patterns:
    if re.search(pattern, puzzle_text, re.IGNORECASE | re.MULTILINE):
        print(f"[OK] Question detected with pattern: {pattern}")
        break
else:
    print("[FAIL] No question pattern matched!")

# Test 2: Check for numbered list
if re.search(r"(?:^|\n)\s*\d+\.\s+.+(?:\n\s*\d+\.\s+.+)+", puzzle_text, re.MULTILINE):
    print("[OK] Numbered list detected")
else:
    print("[FAIL] Numbered list NOT detected")

# Test 3: Just question mark at end
if re.search(r"\?\s*$", puzzle_text, re.MULTILINE):
    print("[OK] Question mark at end detected")
else:
    print("[FAIL] Question mark at end NOT detected")

# Test 4: Wer at beginning of any line
if re.search(r"(?:^|\n)\s*wer\s+", puzzle_text, re.IGNORECASE | re.MULTILINE):
    print("[OK] 'Wer' detected at line start")
else:
    print("[FAIL] 'Wer' NOT detected at line start")
