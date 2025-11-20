"""
Debug-Test für LogicConditionParser - Zeigt genau, welche Bedingungen wie geparst werden
"""

from component_45_logic_puzzle_solver import LogicConditionParser

# Brandy-Rätsel Bedingungen (einzeln)
conditions = [
    "Wenn Leo einen Brandy bestellt, bestellt auch Mark einen",
    "Es kann vorkommen, dass Mark oder Nick einen Brandy bestellen, aber nie beide zusammen",
    "Hingegen geschieht es, dass Leo und Nick einzeln oder gleichzeitig einen Brandy bestellen",
    "Wenn Nick einen Brandy bestellt, will Leo auch einen",
]

entities = ["Leo", "Mark", "Nick"]

print("=" * 80)
print("PARSER DEBUG - Bedingung für Bedingung")
print("=" * 80)

parser = LogicConditionParser()

for i, cond_text in enumerate(conditions, 1):
    print(f"\n[{i}] INPUT:")
    print(f"    '{cond_text}'")
    print()

    # Parse nur diese eine Bedingung
    parsed = parser.parse_conditions(cond_text, entities)

    if parsed:
        for cond in parsed:
            print(f"    [OK] Type: {cond.condition_type}")
            print(f"         Operands: {cond.operands}")

            # Zeige die tatsächlichen Variablen
            for op in cond.operands:
                var = parser.get_variable(op)
                if var:
                    print(f"           {op} = {var.entity}:{var.property}")
    else:
        print(f"    [FAIL] Keine Bedingung geparst!")

    print()
    print("-" * 80)

print("\n" + "=" * 80)
print("ZUSAMMENFASSUNG:")
print("=" * 80)
all_parsed = parser.parse_conditions("\n".join(conditions), entities)
print(f"Gesamt geparst: {len(all_parsed)}/{len(conditions)} Bedingungen")

for i, cond in enumerate(all_parsed, 1):
    print(f"  {i}. {cond.condition_type}: {cond.operands}")
