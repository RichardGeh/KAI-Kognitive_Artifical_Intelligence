"""
Test für LogicPuzzleSolver mit Brandy-Rätsel
"""

from component_45_logic_puzzle_solver import LogicPuzzleSolver

# Brandy-Rätsel Bedingungen
conditions_text = """
Wenn Leo einen Brandy bestellt, bestellt auch Mark einen.
Es kann vorkommen, dass Mark oder Nick einen Brandy bestellen, aber nie beide zusammen.
Hingegen geschieht es, dass Leo und Nick einzeln oder gleichzeitig einen Brandy bestellen.
Wenn Nick einen Brandy bestellt, will Leo auch einen.
"""

entities = ["Leo", "Mark", "Nick"]
question = "Wer von den dreien trinkt also gerne einen Brandy?"

print("=" * 80)
print("BRANDY-RÄTSEL TEST MIT LOGICPUZZLESOLVER")
print("=" * 80)
print(f"\nEntitäten: {entities}")
print(f"\nBedingungen:\n{conditions_text}")
print(f"\nFrage: {question}")
print("\n" + "=" * 80)

# Erstelle Solver
solver = LogicPuzzleSolver()

# Löse das Rätsel
result = solver.solve(conditions_text, entities, question)

print("\nERGEBNIS:")
print("=" * 80)
print(f"Status: {result['result']}")
print(f"\nLösung:")
for var, value in sorted(result["solution"].items()):
    if value:
        print(f"  [OK] {var} = TRUE")
    else:
        print(f"  [ ] {var} = FALSE")

print(f"\nAntwort:")
print(f"  {result['answer']}")

print("\n" + "=" * 80)
