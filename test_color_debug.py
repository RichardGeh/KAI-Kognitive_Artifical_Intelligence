"""
Debug script for color assignment logic puzzle.

Reproduces the UNSAT issue without requiring Neo4j.
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

print("=" * 80)
print("DEBUG: Color Assignment Puzzle - UNSAT Investigation")
print("=" * 80)

# Step 1: Parse conditions
solver = LogicPuzzleSolver()
parser = solver.parser

print("\n[STEP 1] Parsing conditions...")
conditions = parser.parse_conditions(puzzle_text, entities)

print(f"\n[INFO] Detected {len(conditions)} conditions:")
for i, cond in enumerate(conditions, 1):
    print(f"  Condition {i}: {cond.condition_type} - {cond.text}")
    print(f"    Operands: {cond.operands}")

# Step 2: Show detected objects
print(f"\n[INFO] Detected objects: {parser._detected_objects}")
print(f"[INFO] Total variables: {len(parser.variables)}")

# Step 3: Build CNF
print("\n[STEP 2] Building CNF...")
cnf = solver._build_cnf(conditions)
print(f"[INFO] CNF has {len(cnf.clauses)} clauses")

# Step 4: Show critical clauses
print("\n[CRITICAL CLAUSES]")
for i, clause in enumerate(cnf.clauses[:30], 1):  # First 30 clauses
    literals = [f"{'NOT ' if lit.negated else ''}{lit.variable}" for lit in clause.literals]
    print(f"  Clause {i}: {' OR '.join(literals)}")

# Step 5: Analyze constraint 4 (Daniel != Anna)
print("\n[CONSTRAINT 4 ANALYSIS]: 'Daniel traegt nicht die gleiche Farbe wie Anna'")
print("Expected: For each color, NOT(daniel_hat_color AND anna_hat_color)")
print("\nActual clauses generated:")
constraint_4_clauses = [cond for cond in conditions if "daniel" in cond.text.lower() and "anna" in cond.text.lower()]
for cond in constraint_4_clauses:
    print(f"  {cond.condition_type}: {cond.text}")
    print(f"    Operands: {cond.operands}")

# Step 6: Try solving
print("\n[STEP 3] Attempting to solve with SAT solver...")
model = solver.solver.solve(cnf)

if model:
    print(f"\n[SUCCESS] Solution found!")
    true_vars = [var for var, val in model.items() if val]
    print(f"  True variables: {true_vars}")
else:
    print(f"\n[FAILURE] UNSATISFIABLE - No solution found")
    print("\n[DIAGNOSTIC] This is a FALSE UNSAT!")
    print("Expected solution:")
    print("  - Ben = Blau (given)")
    print("  - Clara = Rot (from constraint 3: Clara weder Gruen noch Gelb + constraint 5: exactly one has Rot)")
    print("  - Anna = Gruen or Gelb")
    print("  - Daniel = Gelb or Gruen (opposite of Anna)")

    print("\n[HYPOTHESIS] Possible causes:")
    print("  1. Over-constrained uniqueness constraints")
    print("  2. Incorrect encoding of constraint 4 (Daniel != Anna)")
    print("  3. Constraint 5 (exactly one Rot) incorrectly encoded")
    print("  4. Interaction between constraints creating contradiction")
