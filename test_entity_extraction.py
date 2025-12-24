"""
Test entity extraction from color puzzle.
"""

import re

query_text = """
Vier Personen (Anna, Ben, Clara, Daniel) tragen jeweils eine Farbe: Rot, Blau, Gruen, Gelb.
1. Anna traegt nicht Rot.
2. Ben traegt Blau.
3. Clara traegt weder Gruen noch Gelb.
4. Daniel traegt nicht die gleiche Farbe wie Anna.
5. Genau eine Person traegt Rot.

Wer traegt welche Farbe?
"""

def extract_entities(query_text):
    """Reproduce the _extract_entities_from_query logic."""
    words = query_text.split()
    entities = []

    for i, word in enumerate(words):
        # Skip first word
        if i == 0:
            continue
        # Check if capitalized and not common German words
        if word[0].isupper() and word not in [
            "Der", "Die", "Das", "Ein", "Eine", "Einen",
            "Farbe", "Farben", "Beruf", "Berufe", "Person", "Personen",
            # Colors
            "Rot", "Blau", "Gruen", "Gelb", "Schwarz", "Weiss", "Grau",
            "Orange", "Lila", "Rosa", "Braun",
        ]:
            # Remove punctuation
            clean_word = re.sub(r"[^\w]", "", word)
            if clean_word and clean_word not in entities:
                entities.append(clean_word)

    return entities

entities = extract_entities(query_text)
print(f"Extracted entities ({len(entities)}): {entities}")

# Now test with the solver
from component_45_logic_puzzle_solver_core import LogicPuzzleSolver

solver = LogicPuzzleSolver()
result = solver.solve(
    conditions_text=query_text,
    entities=entities,
    question="Wer traegt welche Farbe?"
)

print(f"\nResult: {result.get('result')}")
print(f"Answer: {result.get('answer', 'N/A')}")
if result.get('result') == 'SATISFIABLE':
    print(f"Solution: {result.get('solution')}")
else:
    print(f"Diagnostic: {result.get('diagnostic')}")
