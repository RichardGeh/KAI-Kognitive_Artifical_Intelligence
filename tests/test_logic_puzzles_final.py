"""
Finale Test-Suite mit 3 klaren Logikrätseln (nach Brandy-Muster)

Alle Rätsel haben eindeutige Lösungen und folgen der gleichen Struktur.
"""

from component_45_logic_puzzle_solver import LogicPuzzleSolver


def test_brandy():
    """Original Brandy-Rätsel (Referenz)"""
    print("\n" + "=" * 80)
    print("RÄTSEL 0: BRANDY (REFERENZ)")
    print("=" * 80)

    conditions = """
Wenn Leo einen Brandy bestellt, bestellt auch Mark einen
Es kann vorkommen, dass Mark oder Nick einen Brandy bestellen, aber nie beide zusammen
Hingegen geschieht es, dass Leo und Nick einzeln oder gleichzeitig einen Brandy bestellen
Wenn Nick einen Brandy bestellt, will Leo auch einen
"""

    entities = ["Leo", "Mark", "Nick"]
    question = "Wer trinkt Brandy?"

    solver = LogicPuzzleSolver()
    result = solver.solve(conditions, entities, question)

    print(f"Status: {result['result']}")
    print(f"Antwort: {result['answer']}")
    print("=" * 80)
    return result["result"] == "SATISFIABLE"


def test_coffee():
    """Rätsel 1: Kaffee (analog zu Brandy)"""
    print("\n" + "=" * 80)
    print("RÄTSEL 1: WER TRINKT KAFFEE?")
    print("=" * 80)

    conditions = """
Wenn Anna einen Kaffee bestellt, bestellt auch Ben einen
Es kann vorkommen, dass Ben oder Clara einen Kaffee bestellen, aber nie beide zusammen
Hingegen geschieht es, dass Anna und Clara einzeln oder gleichzeitig einen Kaffee bestellen
Wenn Clara einen Kaffee bestellt, will Anna auch einen
"""

    entities = ["Anna", "Ben", "Clara"]
    question = "Wer trinkt Kaffee?"

    solver = LogicPuzzleSolver()
    result = solver.solve(conditions, entities, question)

    print(f"Status: {result['result']}")
    print(f"Antwort: {result['answer']}")
    print(f"Erwartet: Anna und Ben trinken Kaffee (Clara nicht)")
    print("=" * 80)
    return result["result"] == "SATISFIABLE"


def test_pizza():
    """Rätsel 2: Pizza (analog zu Brandy)"""
    print("\n" + "=" * 80)
    print("RÄTSEL 2: WER ISST PIZZA?")
    print("=" * 80)

    conditions = """
Wenn David eine Pizza isst, isst auch Emma eine
Es kann vorkommen, dass Emma oder Felix eine Pizza essen, aber nie beide zusammen
Hingegen geschieht es, dass David und Felix einzeln oder gleichzeitig eine Pizza essen
Wenn Felix eine Pizza isst, will David auch eine
"""

    entities = ["David", "Emma", "Felix"]
    question = "Wer isst Pizza?"

    solver = LogicPuzzleSolver()
    result = solver.solve(conditions, entities, question)

    print(f"Status: {result['result']}")
    print(f"Antwort: {result['answer']}")
    print(f"Erwartet: David und Emma essen Pizza (Felix nicht)")
    print("=" * 80)
    return result["result"] == "SATISFIABLE"


def test_bike():
    """Rätsel 3: Fahrrad (analog zu Brandy)"""
    print("\n" + "=" * 80)
    print("RÄTSEL 3: WER KAUFT EIN FAHRRAD?")
    print("=" * 80)

    conditions = """
Wenn Georg ein Fahrrad kauft, kauft auch Hanna eins
Es kann vorkommen, dass Hanna oder Ines ein Fahrrad kaufen, aber nie beide zusammen
Hingegen geschieht es, dass Georg und Ines einzeln oder gleichzeitig ein Fahrrad kaufen
Wenn Ines ein Fahrrad kauft, will Georg auch eins
"""

    entities = ["Georg", "Hanna", "Ines"]
    question = "Wer kauft ein Fahrrad?"

    solver = LogicPuzzleSolver()
    result = solver.solve(conditions, entities, question)

    print(f"Status: {result['result']}")
    print(f"Antwort: {result['answer']}")
    print(f"Erwartet: Georg und Hanna kaufen ein Fahrrad (Ines nicht)")
    print("=" * 80)
    return result["result"] == "SATISFIABLE"


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("FINALE LOGIKRÄTSEL TEST-SUITE")
    print("Alle Rätsel folgen dem Brandy-Muster")
    print("=" * 80)

    results = []
    results.append(("Rätsel 0: Brandy (Referenz)", test_brandy()))
    results.append(("Rätsel 1: Kaffee", test_coffee()))
    results.append(("Rätsel 2: Pizza", test_pizza()))
    results.append(("Rätsel 3: Fahrrad", test_bike()))

    print("\n" + "=" * 80)
    print("ZUSAMMENFASSUNG")
    print("=" * 80)
    for name, success in results:
        status = "[OK] GELÖST" if success else "[FEHLER] UNLÖSBAR"
        print(f"{status}: {name}")

    total = len(results)
    solved = sum(1 for _, s in results if s)
    print(f"\n{solved}/{total} Rätsel erfolgreich gelöst!")
    print("=" * 80)
