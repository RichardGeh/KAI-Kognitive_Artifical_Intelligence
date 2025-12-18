"""
Quick debugging script to trace the color assignment puzzle parsing
"""

from component_45_logic_puzzle_parser import LogicConditionParser
from component_45_logic_puzzle_solver_core import LogicPuzzleSolver

puzzle_text = """
Vier Personen (Anna, Ben, Clara, Daniel) tragen jeweils eine Farbe: Rot, Blau, Gruen, Gelb.
1. Anna traegt nicht Rot.
2. Ben traegt Blau.
3. Clara traegt weder Gruen noch Gelb.
4. Daniel traegt nicht die gleiche Farbe wie Anna.
5. Genau eine Person traegt Rot.

Wer traegt welche Farbe?
"""

entities = ["Anna", "Ben", "Clara", "Daniel"]

print("="*80)
print("DEBUGGING COLOR ASSIGNMENT PUZZLE")
print("="*80)

# Test 1: Parse conditions
parser = LogicConditionParser()
conditions = parser.parse_conditions(puzzle_text, entities)

print(f"\n[STEP 1] Parsed {len(conditions)} conditions:")
print(f"Detected entities: {parser.entities}")
print(f"Detected objects: {parser._detected_objects}")
print(f"Registered variables: {list(parser.variables.keys())}")

print("\n" + "-"*80)
print("PARSED CONDITIONS:")
for i, cond in enumerate(conditions, 1):
    print(f"\n{i}. Type: {cond.condition_type}")
    print(f"   Operands: {cond.operands}")
    print(f"   Text: {cond.text}")

# Test 2: Convert to CNF and solve
solver = LogicPuzzleSolver()
result = solver.solve(puzzle_text, entities, question="Wer traegt welche Farbe?")

print("\n" + "="*80)
print("SOLVER RESULT:")
print(f"Result: {result['result']}")
print(f"Solution: {result.get('solution', {})}")
print(f"Answer: {result.get('answer', 'N/A')}")

if result['result'] == 'SATISFIABLE':
    print("\n[OK] Puzzle is SATISFIABLE")
    print("True variables:")
    for var, val in result['solution'].items():
        if val:
            print(f"  - {var} = TRUE")
else:
    print("\n[FEHLER] Puzzle is UNSATISFIABLE (should be solvable!)")
    print("\nDEBUGGING CNF CONVERSION:")

    # Rebuild CNF to inspect
    cnf = solver._build_cnf(conditions)
    print(f"\nCNF has {len(cnf.clauses)} clauses:")
    for i, clause in enumerate(cnf.clauses[:20], 1):  # First 20 clauses
        literals_str = []
        for lit in clause.literals:
            sign = "NOT " if lit.negated else ""
            literals_str.append(f"{sign}{lit.name}")
        print(f"  {i}. {' OR '.join(literals_str)}")

    if len(cnf.clauses) > 20:
        print(f"  ... ({len(cnf.clauses) - 20} more clauses)")

print("\n" + "="*80)
