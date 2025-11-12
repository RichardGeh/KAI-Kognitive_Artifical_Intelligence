"""
Arithmetisches Reasoning für KAI
Unterstützt: Grundrechenarten, Vergleiche, Eigenschaften, Brüche
"""

import math
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from decimal import Decimal, getcontext
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple

from component_1_netzwerk_core import KonzeptNetzwerkCore
from component_17_proof_explanation import ProofStep, ProofTree, StepType


@dataclass
class ArithmeticResult:
    """Ergebnis einer arithmetischen Operation"""

    value: Any  # int, float, Fraction, Decimal
    proof_tree: ProofTree
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseOperation(ABC):
    """Abstract Base Class für arithmetische Operationen"""

    def __init__(self, symbol: str, name: str, arity: int):
        self.symbol = symbol
        self.name = name
        self.arity = arity

    @abstractmethod
    def execute(self, *operands) -> ArithmeticResult:
        """Führt Operation aus und erstellt Proof"""

    @abstractmethod
    def validate(self, *operands) -> Tuple[bool, Optional[str]]:
        """Validiert Operanden (z.B. Division durch 0)"""


class Addition(BaseOperation):
    """Addition von Zahlen"""

    def __init__(self):
        super().__init__("+", "addition", 2)

    def validate(self, *operands) -> Tuple[bool, Optional[str]]:
        """Validiert Operanden für Addition"""
        if len(operands) != self.arity:
            return (
                False,
                f"Addition benötigt {self.arity} Operanden, {len(operands)} gegeben",
            )

        for i, op in enumerate(operands):
            if not isinstance(op, (int, float, Decimal, Fraction)):
                return False, f"Operand {i+1} ist keine Zahl: {type(op)}"

        return True, None

    def execute(self, *operands) -> ArithmeticResult:
        """Führt Addition aus"""
        a, b = operands
        result = a + b

        # Erstelle detaillierten Proof Tree
        proof = ProofTree(query=f"{a} + {b} = ?")

        # Schritt 1: Operanden identifizieren (PREMISE)
        step1 = ProofStep(
            step_id=str(uuid.uuid4()),
            step_type=StepType.PREMISE,
            inputs=[],
            output=f"a={a}, b={b}",
            explanation_text=f"Gegeben: Operanden a={a} und b={b}",
            confidence=1.0,
            source_component="arithmetic_engine",
            metadata={"operands": [a, b]},
        )
        proof.add_root_step(step1)

        # Schritt 2: Operation anwenden (RULE_APPLICATION)
        step2 = ProofStep(
            step_id=str(uuid.uuid4()),
            step_type=StepType.RULE_APPLICATION,
            inputs=[str(a), str(b)],
            output=f"{a} + {b}",
            explanation_text=f"Wende Addition an: {a} + {b}",
            confidence=1.0,
            source_component="arithmetic_engine",
            metadata={"rule": "Arithmetik: Addition", "operation": "+"},
        )
        step1.add_subgoal(step2)

        # Schritt 3: Ergebnis berechnen (CONCLUSION)
        step3 = ProofStep(
            step_id=str(uuid.uuid4()),
            step_type=StepType.CONCLUSION,
            inputs=[f"{a} + {b}"],
            output=str(result),
            explanation_text=f"Ergebnis: {a} + {b} = {result}",
            confidence=1.0,
            source_component="arithmetic_engine",
            metadata={"result": result},
        )
        step2.add_subgoal(step3)

        return ArithmeticResult(
            value=result,
            proof_tree=proof,
            confidence=1.0,
            metadata={"operation": self.name, "operands": [a, b]},
        )


class Subtraction(BaseOperation):
    """Subtraktion von Zahlen"""

    def __init__(self):
        super().__init__("-", "subtraction", 2)

    def validate(self, *operands) -> Tuple[bool, Optional[str]]:
        """Validiert Operanden für Subtraktion"""
        if len(operands) != self.arity:
            return (
                False,
                f"Subtraktion benötigt {self.arity} Operanden, {len(operands)} gegeben",
            )

        for i, op in enumerate(operands):
            if not isinstance(op, (int, float, Decimal, Fraction)):
                return False, f"Operand {i+1} ist keine Zahl: {type(op)}"

        return True, None

    def execute(self, *operands) -> ArithmeticResult:
        """Führt Subtraktion aus"""
        a, b = operands
        result = a - b

        # Erstelle detaillierten Proof Tree
        proof = ProofTree(query=f"{a} - {b} = ?")

        # Schritt 1: Operanden identifizieren (PREMISE)
        step1 = ProofStep(
            step_id=str(uuid.uuid4()),
            step_type=StepType.PREMISE,
            inputs=[],
            output=f"a={a}, b={b}",
            explanation_text=f"Gegeben: Minuend a={a} und Subtrahend b={b}",
            confidence=1.0,
            source_component="arithmetic_engine",
            metadata={"operands": [a, b]},
        )
        proof.add_root_step(step1)

        # Schritt 2: Operation anwenden (RULE_APPLICATION)
        step2 = ProofStep(
            step_id=str(uuid.uuid4()),
            step_type=StepType.RULE_APPLICATION,
            inputs=[str(a), str(b)],
            output=f"{a} - {b}",
            explanation_text=f"Wende Subtraktion an: {a} - {b}",
            confidence=1.0,
            source_component="arithmetic_engine",
            metadata={"rule": "Arithmetik: Subtraktion", "operation": "-"},
        )
        step1.add_subgoal(step2)

        # Schritt 3: Ergebnis berechnen (CONCLUSION)
        step3 = ProofStep(
            step_id=str(uuid.uuid4()),
            step_type=StepType.CONCLUSION,
            inputs=[f"{a} - {b}"],
            output=str(result),
            explanation_text=f"Ergebnis: {a} - {b} = {result}",
            confidence=1.0,
            source_component="arithmetic_engine",
            metadata={"result": result},
        )
        step2.add_subgoal(step3)

        return ArithmeticResult(
            value=result,
            proof_tree=proof,
            confidence=1.0,
            metadata={"operation": self.name, "operands": [a, b]},
        )


class Multiplication(BaseOperation):
    """Multiplikation von Zahlen"""

    def __init__(self):
        super().__init__("*", "multiplication", 2)

    def validate(self, *operands) -> Tuple[bool, Optional[str]]:
        """Validiert Operanden für Multiplikation"""
        if len(operands) != self.arity:
            return (
                False,
                f"Multiplikation benötigt {self.arity} Operanden, {len(operands)} gegeben",
            )

        for i, op in enumerate(operands):
            if not isinstance(op, (int, float, Decimal, Fraction)):
                return False, f"Operand {i+1} ist keine Zahl: {type(op)}"

        return True, None

    def execute(self, *operands) -> ArithmeticResult:
        """Führt Multiplikation aus"""
        a, b = operands
        result = a * b

        # Erstelle detaillierten Proof Tree
        proof = ProofTree(query=f"{a} * {b} = ?")

        # Schritt 1: Operanden identifizieren (PREMISE)
        step1 = ProofStep(
            step_id=str(uuid.uuid4()),
            step_type=StepType.PREMISE,
            inputs=[],
            output=f"a={a}, b={b}",
            explanation_text=f"Gegeben: Faktoren a={a} und b={b}",
            confidence=1.0,
            source_component="arithmetic_engine",
            metadata={"operands": [a, b]},
        )
        proof.add_root_step(step1)

        # Schritt 2: Operation anwenden (RULE_APPLICATION)
        step2 = ProofStep(
            step_id=str(uuid.uuid4()),
            step_type=StepType.RULE_APPLICATION,
            inputs=[str(a), str(b)],
            output=f"{a} * {b}",
            explanation_text=f"Wende Multiplikation an: {a} * {b}",
            confidence=1.0,
            source_component="arithmetic_engine",
            metadata={"rule": "Arithmetik: Multiplikation", "operation": "*"},
        )
        step1.add_subgoal(step2)

        # Schritt 3: Ergebnis berechnen (CONCLUSION)
        step3 = ProofStep(
            step_id=str(uuid.uuid4()),
            step_type=StepType.CONCLUSION,
            inputs=[f"{a} * {b}"],
            output=str(result),
            explanation_text=f"Ergebnis: {a} * {b} = {result}",
            confidence=1.0,
            source_component="arithmetic_engine",
            metadata={"result": result},
        )
        step2.add_subgoal(step3)

        return ArithmeticResult(
            value=result,
            proof_tree=proof,
            confidence=1.0,
            metadata={"operation": self.name, "operands": [a, b]},
        )


class Division(BaseOperation):
    """Division von Zahlen"""

    def __init__(self):
        super().__init__("/", "division", 2)

    def validate(self, *operands) -> Tuple[bool, Optional[str]]:
        """Validiert Operanden für Division"""
        if len(operands) != self.arity:
            return (
                False,
                f"Division benötigt {self.arity} Operanden, {len(operands)} gegeben",
            )

        for i, op in enumerate(operands):
            if not isinstance(op, (int, float, Decimal, Fraction)):
                return False, f"Operand {i+1} ist keine Zahl: {type(op)}"

        # Prüfe Division durch Null
        if operands[1] == 0:
            return False, "Division durch Null ist nicht erlaubt"

        return True, None

    def execute(self, *operands) -> ArithmeticResult:
        """Führt Division aus (mit Fraction für exakte Brüche)"""
        a, b = operands

        # Verwende Fraction für exakte Brüche (wenn beide integer sind)
        if isinstance(a, int) and isinstance(b, int):
            result = Fraction(a, b)
            result_str = str(result)
        else:
            result = a / b
            result_str = str(result)

        # Erstelle detaillierten Proof Tree
        proof = ProofTree(query=f"{a} / {b} = ?")

        # Schritt 1: Operanden identifizieren (PREMISE)
        step1 = ProofStep(
            step_id=str(uuid.uuid4()),
            step_type=StepType.PREMISE,
            inputs=[],
            output=f"a={a}, b={b}",
            explanation_text=f"Gegeben: Dividend a={a} und Divisor b={b}",
            confidence=1.0,
            source_component="arithmetic_engine",
            metadata={"operands": [a, b]},
        )
        proof.add_root_step(step1)

        # Schritt 2: Constraint Check - Division durch Null (PREMISE für Constraint)
        step2 = ProofStep(
            step_id=str(uuid.uuid4()),
            step_type=StepType.PREMISE,
            inputs=[str(b)],
            output=f"{b} ≠ 0: ✓",
            explanation_text=f"Prüfe Constraint: Divisor {b} ≠ 0",
            confidence=1.0,
            source_component="arithmetic_engine",
            metadata={"constraint": "division_by_zero", "check_passed": True},
        )
        step1.add_subgoal(step2)

        # Schritt 3: Operation anwenden (RULE_APPLICATION)
        step3 = ProofStep(
            step_id=str(uuid.uuid4()),
            step_type=StepType.RULE_APPLICATION,
            inputs=[str(a), str(b)],
            output=f"{a} / {b}",
            explanation_text=f"Wende Division an: {a} / {b}",
            confidence=1.0,
            source_component="arithmetic_engine",
            metadata={"rule": "Arithmetik: Division", "operation": "/"},
        )
        step2.add_subgoal(step3)

        # Schritt 4: Ergebnis berechnen (CONCLUSION)
        if isinstance(result, Fraction):
            explanation = f"Ergebnis: {a} / {b} = {result_str} (exakter Bruch)"
        else:
            explanation = f"Ergebnis: {a} / {b} = {result_str}"

        step4 = ProofStep(
            step_id=str(uuid.uuid4()),
            step_type=StepType.CONCLUSION,
            inputs=[f"{a} / {b}"],
            output=result_str,
            explanation_text=explanation,
            confidence=1.0,
            source_component="arithmetic_engine",
            metadata={"result": result, "result_type": type(result).__name__},
        )
        step3.add_subgoal(step4)

        return ArithmeticResult(
            value=result,
            proof_tree=proof,
            confidence=1.0,
            metadata={
                "operation": self.name,
                "operands": [a, b],
                "result_type": type(result).__name__,
            },
        )


class OperationRegistry:
    """Registry für alle verfügbaren Operationen"""

    def __init__(self):
        self._operations: Dict[str, BaseOperation] = {}

    def register(self, operation: BaseOperation):
        """Registriert eine Operation"""
        self._operations[operation.symbol] = operation
        self._operations[operation.name] = operation

    def get(self, key: str) -> Optional[BaseOperation]:
        """Holt Operation nach Symbol oder Name"""
        return self._operations.get(key)

    def list_operations(self) -> List[str]:
        """Listet alle registrierten Operationen"""
        return list(set(self._operations.keys()))


class ComparisonEngine:
    """Engine für Vergleichsoperationen mit transitiven Schlüssen"""

    def __init__(self, netzwerk: KonzeptNetzwerkCore):
        self.netzwerk = netzwerk
        self._comparison_ops = {
            "<": lambda x, y: x < y,
            ">": lambda x, y: x > y,
            "=": lambda x, y: x == y,
            "<=": lambda x, y: x <= y,
            ">=": lambda x, y: x >= y,
        }
        self._op_names = {
            "<": "kleiner als",
            ">": "größer als",
            "=": "gleich",
            "<=": "kleiner gleich",
            ">=": "größer gleich",
        }

    def compare(self, a, b, operator: str) -> ArithmeticResult:
        """
        Vergleicht zwei Zahlen

        Args:
            a, b: Zahlen zum Vergleichen
            operator: "<", ">", "=", "<=", ">="

        Returns:
            ArithmeticResult mit bool-Wert und Proof
        """
        if operator not in self._comparison_ops:
            raise ValueError(f"Unbekannter Vergleichsoperator: {operator}")

        result_value = self._comparison_ops[operator](a, b)
        op_name = self._op_names[operator]

        # Erstelle Proof Tree
        proof = ProofTree(query=f"{a} {operator} {b} = ?")

        # Schritt 1: Operanden gegeben (PREMISE)
        step1 = ProofStep(
            step_id=str(uuid.uuid4()),
            step_type=StepType.PREMISE,
            inputs=[],
            output=f"a={a}, b={b}",
            explanation_text=f"Gegeben: a={a} und b={b}",
            confidence=1.0,
            source_component="comparison_engine",
            metadata={"operands": [a, b]},
        )
        proof.add_root_step(step1)

        # Schritt 2: Vergleich durchführen (RULE_APPLICATION)
        step2 = ProofStep(
            step_id=str(uuid.uuid4()),
            step_type=StepType.RULE_APPLICATION,
            inputs=[str(a), str(b)],
            output=f"{a} {operator} {b}",
            explanation_text=f"Vergleiche: Ist {a} {op_name} {b}?",
            confidence=1.0,
            source_component="comparison_engine",
            metadata={
                "rule": f"Arithmetik: Vergleich {operator}",
                "operator": operator,
            },
        )
        step1.add_subgoal(step2)

        # Schritt 3: Ergebnis (CONCLUSION)
        step3 = ProofStep(
            step_id=str(uuid.uuid4()),
            step_type=StepType.CONCLUSION,
            inputs=[f"{a} {operator} {b}"],
            output=str(result_value),
            explanation_text=f"Ergebnis: {a} {operator} {b} ist {result_value}",
            confidence=1.0,
            source_component="comparison_engine",
            metadata={"result": result_value},
        )
        step2.add_subgoal(step3)

        return ArithmeticResult(
            value=result_value,
            proof_tree=proof,
            confidence=1.0,
            metadata={
                "operation": "comparison",
                "operator": operator,
                "op_name": op_name,
            },
        )

    def transitive_inference(
        self, relations: List[Tuple[Any, str, Any]]
    ) -> List[Tuple[Any, str, Any]]:
        """
        Leitet transitive Relationen ab (mit mehreren Runden)

        Beispiel:
            Input: [(3, "<", 5), (5, "<", 7)]
            Output: [(3, "<", 7)]

        Unterstützt transitive Operatoren: <, >, <=, >=

        Args:
            relations: Liste von (a, operator, b) Tupeln

        Returns:
            Liste von abgeleiteten Relationen
        """
        inferred = []
        transitive_ops = {"<", ">", "<=", ">="}

        # Kombiniere existierende und abgeleitete Relationen für mehrere Runden
        all_relations = list(relations)
        max_rounds = 10  # Verhindere Endlosschleifen

        for round_num in range(max_rounds):
            new_in_round = []

            # Prüfe alle Paare von Relationen
            for i, (a1, op1, b1) in enumerate(all_relations):
                if op1 not in transitive_ops:
                    continue

                for j, (a2, op2, b2) in enumerate(all_relations):
                    if i >= j or op2 not in transitive_ops:
                        continue

                    # A < B ∧ B < C → A < C (und Varianten)
                    if b1 == a2 and op1 == op2:
                        new_relation = (a1, op1, b2)
                        if (
                            new_relation not in relations
                            and new_relation not in inferred
                            and new_relation not in new_in_round
                        ):
                            new_in_round.append(new_relation)

                    # A > B ∧ B > C → A > C (und Varianten)
                    elif a1 == b2 and op1 == op2:
                        new_relation = (a2, op1, b1)
                        if (
                            new_relation not in relations
                            and new_relation not in inferred
                            and new_relation not in new_in_round
                        ):
                            new_in_round.append(new_relation)

            # Wenn keine neuen Relationen, breche ab
            if not new_in_round:
                break

            # Füge neue Relationen hinzu
            inferred.extend(new_in_round)
            all_relations.extend(new_in_round)

        return inferred

    def build_transitive_proof(
        self, relations: List[Tuple[Any, str, Any]]
    ) -> ArithmeticResult:
        """
        Baut Proof Tree für transitive Inferenz

        Args:
            relations: Liste von (a, operator, b) Tupeln

        Returns:
            ArithmeticResult mit abgeleiteten Relationen und Proof
        """
        inferred = self.transitive_inference(relations)

        # Erstelle Proof Tree
        proof = ProofTree(query=f"Transitive Inferenz aus {len(relations)} Relationen")

        # Schritt 1: Gegebene Relationen (PREMISE)
        step1 = ProofStep(
            step_id=str(uuid.uuid4()),
            step_type=StepType.PREMISE,
            inputs=[],
            output="; ".join([f"{a} {op} {b}" for a, op, b in relations]),
            explanation_text=f"Gegeben: {len(relations)} Relationen",
            confidence=1.0,
            source_component="comparison_engine",
            metadata={"relations": relations},
        )
        proof.add_root_step(step1)

        # Schritt 2: Transitive Regel anwenden (RULE_APPLICATION)
        step2 = ProofStep(
            step_id=str(uuid.uuid4()),
            step_type=StepType.RULE_APPLICATION,
            inputs=[str(r) for r in relations],
            output="Transitivitätsregel",
            explanation_text="Wende Transitivitätsregel an: (A op B) ∧ (B op C) → (A op C)",
            confidence=1.0,
            source_component="comparison_engine",
            metadata={"rule": "Transitivität"},
        )
        step1.add_subgoal(step2)

        # Schritt 3: Abgeleitete Relationen (CONCLUSION)
        if inferred:
            conclusion_text = "; ".join([f"{a} {op} {b}" for a, op, b in inferred])
        else:
            conclusion_text = "Keine neuen Relationen ableitbar"

        step3 = ProofStep(
            step_id=str(uuid.uuid4()),
            step_type=StepType.CONCLUSION,
            inputs=["Transitivitätsregel"],
            output=conclusion_text,
            explanation_text=f"Abgeleitet: {len(inferred)} neue Relationen",
            confidence=1.0,
            source_component="comparison_engine",
            metadata={"inferred": inferred},
        )
        step2.add_subgoal(step3)

        return ArithmeticResult(
            value=inferred,
            proof_tree=proof,
            confidence=1.0,
            metadata={"operation": "transitive_inference"},
        )


class PropertyChecker:
    """Prüft mathematische Eigenschaften von Zahlen"""

    def __init__(self, netzwerk: KonzeptNetzwerkCore):
        self.netzwerk = netzwerk

    def is_even(self, n: int) -> ArithmeticResult:
        """Prüft ob Zahl gerade ist"""
        if not isinstance(n, int):
            raise ValueError(f"is_even benötigt Integer, nicht {type(n)}")

        result_value = n % 2 == 0

        # Erstelle Proof Tree
        proof = ProofTree(query=f"Ist {n} gerade?")

        # Schritt 1: Zahl gegeben (PREMISE)
        step1 = ProofStep(
            step_id=str(uuid.uuid4()),
            step_type=StepType.PREMISE,
            inputs=[],
            output=f"n={n}",
            explanation_text=f"Gegeben: Zahl n={n}",
            confidence=1.0,
            source_component="property_checker",
            metadata={"number": n},
        )
        proof.add_root_step(step1)

        # Schritt 2: Modulo-Berechnung (RULE_APPLICATION)
        remainder = n % 2
        step2 = ProofStep(
            step_id=str(uuid.uuid4()),
            step_type=StepType.RULE_APPLICATION,
            inputs=[str(n)],
            output=f"{n} % 2 = {remainder}",
            explanation_text=f"Berechne Modulo: {n} % 2 = {remainder}",
            confidence=1.0,
            source_component="property_checker",
            metadata={
                "rule": "Definition: n ist gerade ⟺ n % 2 = 0",
                "remainder": remainder,
            },
        )
        step1.add_subgoal(step2)

        # Schritt 3: Ergebnis (CONCLUSION)
        step3 = ProofStep(
            step_id=str(uuid.uuid4()),
            step_type=StepType.CONCLUSION,
            inputs=[f"{n} % 2 = {remainder}"],
            output=str(result_value),
            explanation_text=f"{n} ist {'gerade' if result_value else 'ungerade'}",
            confidence=1.0,
            source_component="property_checker",
            metadata={
                "result": result_value,
                "property": "even" if result_value else "odd",
            },
        )
        step2.add_subgoal(step3)

        # Speichere in Neo4j
        self._persist_property(n, "gerade" if result_value else "ungerade")

        return ArithmeticResult(
            value=result_value,
            proof_tree=proof,
            confidence=1.0,
            metadata={
                "operation": "is_even",
                "property": "even" if result_value else "odd",
            },
        )

    def is_odd(self, n: int) -> ArithmeticResult:
        """Prüft ob Zahl ungerade ist"""
        result = self.is_even(n)
        # Negiere Ergebnis
        result.value = not result.value
        result.metadata["operation"] = "is_odd"
        result.metadata["property"] = "odd" if result.value else "even"

        # Aktualisiere Proof Tree Conclusion
        if result.proof_tree.root_steps:
            root = result.proof_tree.root_steps[0]
            if root.subgoals:
                for step in root.subgoals:
                    if step.subgoals:
                        conclusion = step.subgoals[0]
                        conclusion.explanation_text = (
                            f"{n} ist {'ungerade' if result.value else 'gerade'}"
                        )

        return result

    def is_prime(self, n: int) -> ArithmeticResult:
        """Prüft ob Zahl eine Primzahl ist"""
        if not isinstance(n, int):
            raise ValueError(f"is_prime benötigt Integer, nicht {type(n)}")

        # Primzahl-Check
        if n < 2:
            return self._build_prime_result(n, False, reason=f"{n} < 2")

        if n == 2:
            return self._build_prime_result(
                n, True, reason="2 ist die kleinste Primzahl"
            )

        if n % 2 == 0:
            return self._build_prime_result(n, False, divisor=2)

        # Prüfe ungerade Teiler bis √n
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return self._build_prime_result(n, False, divisor=i)

        return self._build_prime_result(n, True)

    def _build_prime_result(
        self, n: int, is_prime: bool, divisor: int = None, reason: str = None
    ) -> ArithmeticResult:
        """Erstellt ArithmeticResult für Primzahl-Check"""
        proof = ProofTree(query=f"Ist {n} eine Primzahl?")

        # Schritt 1: Zahl gegeben (PREMISE)
        step1 = ProofStep(
            step_id=str(uuid.uuid4()),
            step_type=StepType.PREMISE,
            inputs=[],
            output=f"n={n}",
            explanation_text=f"Gegeben: Zahl n={n}",
            confidence=1.0,
            source_component="property_checker",
            metadata={"number": n},
        )
        proof.add_root_step(step1)

        # Schritt 2: Primzahl-Kriterium prüfen (RULE_APPLICATION)
        if reason:
            explanation = reason
        elif divisor:
            explanation = f"{n} ist durch {divisor} teilbar: {n} % {divisor} = 0"
        else:
            explanation = f"Keine Teiler zwischen 2 und {int(n**0.5)} gefunden"

        step2 = ProofStep(
            step_id=str(uuid.uuid4()),
            step_type=StepType.RULE_APPLICATION,
            inputs=[str(n)],
            output=explanation,
            explanation_text=f"Prüfe Primzahl-Kriterium: {explanation}",
            confidence=1.0,
            source_component="property_checker",
            metadata={
                "rule": "Definition: Primzahl hat nur 1 und sich selbst als Teiler",
                "divisor": divisor,
            },
        )
        step1.add_subgoal(step2)

        # Schritt 3: Ergebnis (CONCLUSION)
        step3 = ProofStep(
            step_id=str(uuid.uuid4()),
            step_type=StepType.CONCLUSION,
            inputs=[explanation],
            output=str(is_prime),
            explanation_text=f"{n} ist {'eine Primzahl' if is_prime else 'keine Primzahl'}",
            confidence=1.0,
            source_component="property_checker",
            metadata={
                "result": is_prime,
                "property": "prime" if is_prime else "composite",
            },
        )
        step2.add_subgoal(step3)

        # Speichere in Neo4j
        if is_prime:
            self._persist_property(n, "primzahl")

        return ArithmeticResult(
            value=is_prime,
            proof_tree=proof,
            confidence=1.0,
            metadata={
                "operation": "is_prime",
                "property": "prime" if is_prime else "composite",
            },
        )

    def find_divisors(self, n: int) -> ArithmeticResult:
        """Findet alle Teiler einer Zahl"""
        if not isinstance(n, int):
            raise ValueError(f"find_divisors benötigt Integer, nicht {type(n)}")

        if n == 0:
            raise ValueError("0 hat unendlich viele Teiler")

        n_abs = abs(n)
        divisors = [i for i in range(1, n_abs + 1) if n_abs % i == 0]

        # Erstelle Proof Tree
        proof = ProofTree(query=f"Finde alle Teiler von {n}")

        # Schritt 1: Zahl gegeben (PREMISE)
        step1 = ProofStep(
            step_id=str(uuid.uuid4()),
            step_type=StepType.PREMISE,
            inputs=[],
            output=f"n={n}",
            explanation_text=f"Gegeben: Zahl n={n}",
            confidence=1.0,
            source_component="property_checker",
            metadata={"number": n},
        )
        proof.add_root_step(step1)

        # Schritt 2: Teiler suchen (RULE_APPLICATION)
        step2 = ProofStep(
            step_id=str(uuid.uuid4()),
            step_type=StepType.RULE_APPLICATION,
            inputs=[str(n)],
            output=f"Prüfe alle Zahlen von 1 bis {n_abs}",
            explanation_text=f"Suche Teiler: Prüfe i ∈ [1, {n_abs}] mit {n_abs} % i = 0",
            confidence=1.0,
            source_component="property_checker",
            metadata={"rule": "Definition: d teilt n ⟺ n % d = 0", "range": [1, n_abs]},
        )
        step1.add_subgoal(step2)

        # Schritt 3: Ergebnis (CONCLUSION)
        step3 = ProofStep(
            step_id=str(uuid.uuid4()),
            step_type=StepType.CONCLUSION,
            inputs=[f"Prüfe 1 bis {n_abs}"],
            output=str(divisors),
            explanation_text=f"Teiler von {n}: {divisors} ({len(divisors)} Teiler)",
            confidence=1.0,
            source_component="property_checker",
            metadata={"result": divisors, "count": len(divisors)},
        )
        step2.add_subgoal(step3)

        # Speichere in Neo4j
        self._persist_divisors(n, divisors)

        return ArithmeticResult(
            value=divisors,
            proof_tree=proof,
            confidence=1.0,
            metadata={"operation": "find_divisors", "count": len(divisors)},
        )

    def _persist_property(self, number: int, property_name: str):
        """Speichert Eigenschaft in Neo4j"""
        try:
            # Erstelle Wort für Zahl
            number_word = self.netzwerk.get_or_create_wort(str(number), pos="NUM")

            # Erstelle Eigenschafts-Relation
            self.netzwerk.create_relation(
                start_node_id=number_word,
                end_node_id=self.netzwerk.get_or_create_wort(property_name, pos="ADJ"),
                relation_type="HAS_PROPERTY",
                confidence=1.0,
                provenance="arithmetic_reasoning",
            )
        except Exception as e:
            # Logging, aber nicht kritisch
            print(f"Warnung: Konnte Eigenschaft nicht speichern: {e}")

    def _persist_divisors(self, number: int, divisors: List[int]):
        """Speichert Teiler-Relationen in Neo4j"""
        try:
            number_word = self.netzwerk.get_or_create_wort(str(number), pos="NUM")

            for divisor in divisors:
                divisor_word = self.netzwerk.get_or_create_wort(str(divisor), pos="NUM")
                self.netzwerk.create_relation(
                    start_node_id=divisor_word,
                    end_node_id=number_word,
                    relation_type="DIVIDES",
                    confidence=1.0,
                    provenance="arithmetic_reasoning",
                )
        except Exception as e:
            print(f"Warnung: Konnte Teiler nicht speichern: {e}")


class RationalArithmetic:
    """Bruchrechnung mit exakten Ergebnissen (Python fractions)"""

    def __init__(self):
        pass

    def add(self, a: Fraction, b: Fraction) -> Fraction:
        """Addiert zwei Brüche"""
        return a + b  # Python Fraction unterstützt + direkt

    def subtract(self, a: Fraction, b: Fraction) -> Fraction:
        """Subtrahiert zwei Brüche"""
        return a - b

    def multiply(self, a: Fraction, b: Fraction) -> Fraction:
        """Multipliziert zwei Brüche"""
        return a * b

    def divide(self, a: Fraction, b: Fraction) -> Fraction:
        """Dividiert zwei Brüche"""
        if b == 0:
            raise ValueError("Division durch Null ist nicht erlaubt")
        return a / b

    def simplify(self, fraction: Fraction) -> Fraction:
        """Kürzt Bruch (automatisch durch Fraction)"""
        return fraction

    def to_mixed_number(self, fraction: Fraction) -> Tuple[int, Fraction]:
        """
        Konvertiert zu gemischter Zahl

        Beispiel: 7/3 → (2, 1/3)

        Args:
            fraction: Bruch zum Konvertieren

        Returns:
            Tuple (ganzer Teil, Bruch-Rest)
        """
        whole = fraction.numerator // fraction.denominator
        remainder = fraction.numerator % fraction.denominator
        return whole, Fraction(remainder, fraction.denominator)

    def from_mixed_number(self, whole: int, fraction: Fraction) -> Fraction:
        """
        Konvertiert gemischte Zahl zu Bruch

        Beispiel: (2, 1/3) → 7/3

        Args:
            whole: Ganzer Teil
            fraction: Bruch-Teil

        Returns:
            Bruch
        """
        return Fraction(
            whole * fraction.denominator + fraction.numerator, fraction.denominator
        )

    def gcd(self, a: int, b: int) -> int:
        """Größter gemeinsamer Teiler"""
        return math.gcd(a, b)

    def lcm(self, a: int, b: int) -> int:
        """Kleinstes gemeinsames Vielfaches"""
        if a == 0 or b == 0:
            return 0
        return abs(a * b) // math.gcd(a, b)

    def compare(self, a: Fraction, b: Fraction, operator: str) -> bool:
        """Vergleicht zwei Brüche"""
        ops = {
            "<": lambda x, y: x < y,
            ">": lambda x, y: x > y,
            "=": lambda x, y: x == y,
            "==": lambda x, y: x == y,
            "<=": lambda x, y: x <= y,
            ">=": lambda x, y: x >= y,
        }
        if operator not in ops:
            raise ValueError(f"Unbekannter Operator: {operator}")
        return ops[operator](a, b)


class DecimalArithmetic:
    """Dezimalzahl-Arithmetik mit konfigurierbarer Präzision"""

    def __init__(self, precision: int = 10):
        """
        Initialisiert DecimalArithmetic mit Präzision

        Args:
            precision: Anzahl signifikanter Stellen (Standard: 10)
        """
        self.precision = precision
        getcontext().prec = precision

    def set_precision(self, precision: int):
        """Setzt neue Präzision"""
        self.precision = precision
        getcontext().prec = precision

    def calculate(self, operation: str, *operands) -> Decimal:
        """
        Führt Operation mit Decimal-Präzision aus

        Args:
            operation: Operation ("+", "-", "*", "/")
            operands: Operanden (werden zu Decimal konvertiert)

        Returns:
            Decimal-Ergebnis
        """
        # Konvertiere zu Decimal
        operands_decimal = [Decimal(str(op)) for op in operands]

        if operation == "+":
            return sum(operands_decimal)
        elif operation == "-":
            result = operands_decimal[0]
            for op in operands_decimal[1:]:
                result -= op
            return result
        elif operation == "*":
            result = operands_decimal[0]
            for op in operands_decimal[1:]:
                result *= op
            return result
        elif operation == "/":
            result = operands_decimal[0]
            for op in operands_decimal[1:]:
                if op == 0:
                    raise ValueError("Division durch Null ist nicht erlaubt")
                result /= op
            return result
        else:
            raise ValueError(f"Unbekannte Operation: {operation}")

    def round(self, value: float, decimals: int) -> float:
        """
        Rundet auf n Dezimalstellen

        Args:
            value: Zu rundender Wert
            decimals: Anzahl Dezimalstellen

        Returns:
            Gerundeter Wert
        """
        return round(value, decimals)

    def round_decimal(self, value: Decimal, decimals: int) -> Decimal:
        """
        Rundet Decimal auf n Dezimalstellen

        Args:
            value: Decimal-Wert
            decimals: Anzahl Dezimalstellen

        Returns:
            Gerundeter Decimal
        """
        quantize_str = "1." + "0" * decimals if decimals > 0 else "1"
        return value.quantize(Decimal(quantize_str))

    def to_fraction(self, value: Decimal) -> Fraction:
        """Konvertiert Decimal zu Fraction (exakt)"""
        return Fraction(str(value))

    def from_fraction(self, fraction: Fraction) -> Decimal:
        """Konvertiert Fraction zu Decimal (mit aktueller Präzision)"""
        return Decimal(fraction.numerator) / Decimal(fraction.denominator)


class PowerArithmetic:
    """Potenzen, Wurzeln und verwandte Operationen"""

    def __init__(self):
        pass

    def power(self, base: float, exponent: float) -> float:
        """
        Berechnet base^exponent

        Args:
            base: Basis
            exponent: Exponent

        Returns:
            Ergebnis der Potenzierung

        Examples:
            power(2, 3) = 8
            power(4, 0.5) = 2.0 (Quadratwurzel)
            power(2, -1) = 0.5
        """
        if base == 0 and exponent < 0:
            raise ValueError("0 kann nicht zu einer negativen Potenz erhoben werden")

        if base < 0 and not self._is_integer(exponent):
            raise ValueError(
                "Negative Basis mit nicht-ganzzahligem Exponenten "
                "ist nicht definiert (komplexe Zahl)"
            )

        return base**exponent

    def square(self, n: float) -> float:
        """Quadrat: n²"""
        return n**2

    def cube(self, n: float) -> float:
        """Kubik: n³"""
        return n**3

    def sqrt(self, n: float) -> float:
        """
        Quadratwurzel: √n

        Args:
            n: Zahl (muss >= 0 sein)

        Returns:
            Quadratwurzel von n
        """
        if n < 0:
            raise ValueError(
                "Quadratwurzel von negativen Zahlen ist nicht definiert (komplexe Zahl)"
            )

        return math.sqrt(n)

    def cbrt(self, n: float) -> float:
        """
        Kubikwurzel: ∛n

        Args:
            n: Zahl (kann negativ sein)

        Returns:
            Kubikwurzel von n
        """
        if n >= 0:
            return n ** (1 / 3)
        else:
            # Negative Kubikwurzel
            return -((-n) ** (1 / 3))

    def nth_root(self, n: float, root: int) -> float:
        """
        n-te Wurzel

        Args:
            n: Zahl
            root: Wurzelgrad (muss > 0 sein)

        Returns:
            root-te Wurzel von n

        Examples:
            nth_root(8, 3) = 2.0
            nth_root(16, 4) = 2.0
        """
        if root <= 0:
            raise ValueError("Wurzelgrad muss > 0 sein")

        if root == 1:
            return n

        # Gerade Wurzel von negativer Zahl
        if root % 2 == 0 and n < 0:
            raise ValueError(
                f"{root}-te Wurzel von negativen Zahlen ist nicht definiert (komplexe Zahl)"
            )

        # Ungerade Wurzel (auch für negative Zahlen definiert)
        if root % 2 == 1 and n < 0:
            return -((-n) ** (1 / root))

        return n ** (1 / root)

    def exp(self, x: float) -> float:
        """
        Exponentialfunktion: e^x

        Args:
            x: Exponent

        Returns:
            e^x
        """
        return math.exp(x)

    def log(self, x: float, base: Optional[float] = None) -> float:
        """
        Logarithmus

        Args:
            x: Argument (muss > 0 sein)
            base: Basis (optional, Standard: e für natürlichen Logarithmus)

        Returns:
            Logarithmus von x zur Basis base

        Examples:
            log(10) = ln(10) ≈ 2.302585
            log(100, 10) = log₁₀(100) = 2.0
        """
        if x <= 0:
            raise ValueError("Logarithmus ist nur für positive Zahlen definiert")

        if base is None:
            return math.log(x)  # Natürlicher Logarithmus

        if base <= 0 or base == 1:
            raise ValueError("Logarithmus-Basis muss > 0 und ≠ 1 sein")

        return math.log(x, base)

    def log10(self, x: float) -> float:
        """Logarithmus zur Basis 10"""
        if x <= 0:
            raise ValueError("Logarithmus ist nur für positive Zahlen definiert")
        return math.log10(x)

    def log2(self, x: float) -> float:
        """Logarithmus zur Basis 2"""
        if x <= 0:
            raise ValueError("Logarithmus ist nur für positive Zahlen definiert")
        return math.log2(x)

    def _is_integer(self, n: float) -> bool:
        """Prüft ob Zahl ganzzahlig ist"""
        return n == int(n)


class ModuloArithmetic:
    """Modulo- und Rest-Arithmetik"""

    def __init__(self):
        pass

    def modulo(self, a: int, m: int) -> int:
        """
        Modulo-Operation: a mod m

        Args:
            a: Dividend
            m: Modul (muss ≠ 0 sein)

        Returns:
            Rest der Division a / m (immer nicht-negativ)

        Examples:
            modulo(7, 3) = 1
            modulo(-7, 3) = 2 (Python-Konvention)
        """
        if m == 0:
            raise ValueError("Modulo durch Null ist nicht erlaubt")

        return a % m

    def remainder(self, a: int, m: int) -> int:
        """
        Rest-Operation (gleich wie modulo in Python)

        Args:
            a: Dividend
            m: Divisor

        Returns:
            Rest der Division
        """
        return self.modulo(a, m)

    def divmod_op(self, a: int, m: int) -> Tuple[int, int]:
        """
        Quotient und Rest gleichzeitig

        Args:
            a: Dividend
            m: Divisor (muss ≠ 0 sein)

        Returns:
            Tuple (Quotient, Rest)

        Examples:
            divmod_op(7, 3) = (2, 1)  # 7 = 2*3 + 1
        """
        if m == 0:
            raise ValueError("Division durch Null ist nicht erlaubt")

        return divmod(a, m)

    def mod_add(self, a: int, b: int, m: int) -> int:
        """
        Modulare Addition: (a + b) mod m

        Args:
            a, b: Summanden
            m: Modul

        Returns:
            (a + b) mod m
        """
        return (a + b) % m

    def mod_subtract(self, a: int, b: int, m: int) -> int:
        """
        Modulare Subtraktion: (a - b) mod m

        Args:
            a, b: Minuend, Subtrahend
            m: Modul

        Returns:
            (a - b) mod m
        """
        return (a - b) % m

    def mod_multiply(self, a: int, b: int, m: int) -> int:
        """
        Modulare Multiplikation: (a * b) mod m

        Args:
            a, b: Faktoren
            m: Modul

        Returns:
            (a * b) mod m
        """
        return (a * b) % m

    def mod_power(self, base: int, exponent: int, m: int) -> int:
        """
        Modulare Potenzierung: base^exponent mod m
        Verwendet effizientes Square-and-Multiply (Python built-in pow)

        Args:
            base: Basis
            exponent: Exponent (muss >= 0 sein)
            m: Modul

        Returns:
            base^exponent mod m

        Examples:
            mod_power(2, 10, 1000) = 24
            mod_power(3, 100, 7) = 4
        """
        if exponent < 0:
            raise ValueError(
                "Negative Exponenten nicht unterstützt (benötigt modulares Inverse)"
            )

        if m == 0:
            raise ValueError("Modulo durch Null ist nicht erlaubt")

        return pow(base, exponent, m)

    def is_congruent(self, a: int, b: int, m: int) -> bool:
        """
        Prüft Kongruenz: a ≡ b (mod m)

        Args:
            a, b: Zahlen zum Vergleichen
            m: Modul

        Returns:
            True wenn a ≡ b (mod m)

        Examples:
            is_congruent(7, 1, 3) = True  # 7 ≡ 1 (mod 3)
            is_congruent(10, 2, 4) = True  # 10 ≡ 2 (mod 4)
        """
        return (a % m) == (b % m)

    def mod_inverse(self, a: int, m: int) -> Optional[int]:
        """
        Modulares Inverse: Findet x mit a*x ≡ 1 (mod m)
        Verwendet erweiterten euklidischen Algorithmus

        Args:
            a: Zahl
            m: Modul

        Returns:
            Modulares Inverse von a mod m, oder None wenn nicht existiert

        Examples:
            mod_inverse(3, 7) = 5  # 3*5 = 15 ≡ 1 (mod 7)
        """

        def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
            """Erweiterter euklidischer Algorithmus"""
            if a == 0:
                return b, 0, 1

            gcd, x1, y1 = extended_gcd(b % a, a)
            x = y1 - (b // a) * x1
            y = x1

            return gcd, x, y

        gcd, x, _ = extended_gcd(a % m, m)

        # Inverse existiert nur wenn gcd(a, m) = 1
        if gcd != 1:
            return None

        return (x % m + m) % m


class MathematicalConstants:
    """Mathematische Konstanten und Funktionen"""

    # Konstanten (mit hoher Präzision)
    PI = Decimal("3.1415926535897932384626433832795028841971693993751")
    E = Decimal("2.7182818284590452353602874713526624977572470936999")
    GOLDEN_RATIO = Decimal("1.6180339887498948482045868343656381177203091798057")
    SQRT_2 = Decimal("1.4142135623730950488016887242096980785696718753769")
    SQRT_3 = Decimal("1.7320508075688772935274463415058723669428052538103")
    SQRT_5 = Decimal("2.2360679774997896964091736687312762354406183596115")

    # Float-Versionen für schnellere Berechnungen
    PI_FLOAT = math.pi
    E_FLOAT = math.e
    TAU_FLOAT = 2 * math.pi  # τ = 2π

    def __init__(self, use_decimal: bool = False):
        """
        Initialisiert MathematicalConstants

        Args:
            use_decimal: Wenn True, nutze Decimal-Präzision, sonst Float
        """
        self.use_decimal = use_decimal

    def pi(self) -> Any:
        """Gibt π zurück (Decimal oder Float)"""
        return self.PI if self.use_decimal else self.PI_FLOAT

    def e(self) -> Any:
        """Gibt e zurück (Decimal oder Float)"""
        return self.E if self.use_decimal else self.E_FLOAT

    def tau(self) -> Any:
        """Gibt τ = 2π zurück"""
        return self.PI * 2 if self.use_decimal else self.TAU_FLOAT

    def golden_ratio(self) -> Any:
        """Gibt goldenen Schnitt φ zurück"""
        return self.GOLDEN_RATIO if self.use_decimal else float(self.GOLDEN_RATIO)

    def sqrt_2(self) -> Any:
        """Gibt √2 zurück"""
        return self.SQRT_2 if self.use_decimal else float(self.SQRT_2)

    def sqrt_3(self) -> Any:
        """Gibt √3 zurück"""
        return self.SQRT_3 if self.use_decimal else float(self.SQRT_3)

    def sqrt_5(self) -> Any:
        """Gibt √5 zurück"""
        return self.SQRT_5 if self.use_decimal else float(self.SQRT_5)

    def circle_area(self, radius: float) -> float:
        """Berechnet Kreisfläche: A = πr²"""
        return self.PI_FLOAT * radius**2

    def circle_circumference(self, radius: float) -> float:
        """Berechnet Kreisumfang: C = 2πr"""
        return 2 * self.PI_FLOAT * radius

    def sphere_volume(self, radius: float) -> float:
        """Berechnet Kugelvolumen: V = 4/3 πr³"""
        return (4 / 3) * self.PI_FLOAT * radius**3

    def sphere_surface(self, radius: float) -> float:
        """Berechnet Kugeloberfläche: A = 4πr²"""
        return 4 * self.PI_FLOAT * radius**2

    def cylinder_volume(self, radius: float, height: float) -> float:
        """Berechnet Zylindervolumen: V = πr²h"""
        return self.PI_FLOAT * radius**2 * height

    def degrees_to_radians(self, degrees: float) -> float:
        """Konvertiert Grad zu Radiant"""
        return math.radians(degrees)

    def radians_to_degrees(self, radians: float) -> float:
        """Konvertiert Radiant zu Grad"""
        return math.degrees(radians)

    def sin(self, x: float, use_degrees: bool = False) -> float:
        """Sinus (Eingabe in Radiant oder Grad)"""
        if use_degrees:
            x = self.degrees_to_radians(x)
        return math.sin(x)

    def cos(self, x: float, use_degrees: bool = False) -> float:
        """Kosinus (Eingabe in Radiant oder Grad)"""
        if use_degrees:
            x = self.degrees_to_radians(x)
        return math.cos(x)

    def tan(self, x: float, use_degrees: bool = False) -> float:
        """Tangens (Eingabe in Radiant oder Grad)"""
        if use_degrees:
            x = self.degrees_to_radians(x)
        return math.tan(x)


class ArithmeticEngine:
    """Haupt-Engine für arithmetisches Reasoning"""

    def __init__(self, netzwerk: KonzeptNetzwerkCore):
        self.netzwerk = netzwerk
        self.registry = OperationRegistry()
        self.comparison_engine = ComparisonEngine(netzwerk)
        self.property_checker = PropertyChecker(netzwerk)
        self.rational_arithmetic = RationalArithmetic()
        self.decimal_arithmetic = DecimalArithmetic()
        self.power_arithmetic = PowerArithmetic()
        self.modulo_arithmetic = ModuloArithmetic()
        self.math_constants = MathematicalConstants()
        self._register_operations()

    def _register_operations(self):
        """Registriert alle Standard-Operationen"""
        self.registry.register(Addition())
        self.registry.register(Subtraction())
        self.registry.register(Multiplication())
        self.registry.register(Division())

    def calculate(self, operation: str, *operands) -> ArithmeticResult:
        """
        Führt Berechnung aus

        Args:
            operation: Operation (Symbol oder Name)
            operands: Operanden (bereits als Zahlen konvertiert)

        Returns:
            ArithmeticResult mit Wert, Proof und Confidence
        """
        op = self.registry.get(operation)
        if not op:
            raise ValueError(f"Unbekannte Operation: {operation}")

        # Validierung
        valid, error = op.validate(*operands)
        if not valid:
            raise ValueError(f"Validierung fehlgeschlagen: {error}")

        # Ausführung
        result = op.execute(*operands)

        # Optional: In Neo4j speichern
        self._persist_calculation(operation, operands, result)

        return result

    def compare(self, a, b, operator: str) -> ArithmeticResult:
        """
        Vergleicht zwei Zahlen (Delegiert an ComparisonEngine)

        Args:
            a, b: Zahlen
            operator: "<", ">", "=", "<=", ">="

        Returns:
            ArithmeticResult mit bool-Wert
        """
        return self.comparison_engine.compare(a, b, operator)

    def check_property(self, n: int, property_name: str) -> ArithmeticResult:
        """
        Prüft Eigenschaft einer Zahl (Delegiert an PropertyChecker)

        Args:
            n: Zahl
            property_name: "even", "odd", "prime"

        Returns:
            ArithmeticResult mit bool-Wert
        """
        property_methods = {
            "even": self.property_checker.is_even,
            "odd": self.property_checker.is_odd,
            "prime": self.property_checker.is_prime,
        }

        if property_name not in property_methods:
            raise ValueError(f"Unbekannte Eigenschaft: {property_name}")

        return property_methods[property_name](n)

    def find_divisors(self, n: int) -> ArithmeticResult:
        """Findet alle Teiler (Delegiert an PropertyChecker)"""
        return self.property_checker.find_divisors(n)

    def transitive_inference(
        self, relations: List[Tuple[Any, str, Any]]
    ) -> ArithmeticResult:
        """Transitive Inferenz (Delegiert an ComparisonEngine)"""
        return self.comparison_engine.build_transitive_proof(relations)

    def _persist_calculation(
        self, operation: str, operands: tuple, result: ArithmeticResult
    ):
        """Speichert Berechnung in Neo4j (optional)"""
        # TODO: Implementieren
