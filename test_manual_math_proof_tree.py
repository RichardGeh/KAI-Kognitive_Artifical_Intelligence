"""
Manueller Test für Math Proof Tree Integration

Testet:
- ArithmeticEngine erstellt ProofTree
- ProofTreeWidget zeigt mathematische Proofs an
- Unicode-Symbole werden korrekt dargestellt
"""

import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QLabel
from component_1_netzwerk_core import KonzeptNetzwerkCore
from component_52_arithmetic_reasoning import ArithmeticEngine
from component_18_proof_tree_widget import ProofTreeWidget


class MathProofTestWindow(QMainWindow):
    """Test-Fenster für Math Proof Tree Visualisierung"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Math Proof Tree Test")
        self.setGeometry(100, 100, 1200, 800)

        # Central Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Label
        label = QLabel("Mathematische Proof Trees - Test")
        label.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;")
        layout.addWidget(label)

        # Test Buttons
        button_layout = QVBoxLayout()

        btn_addition = QPushButton("Test Addition: 3 + 5")
        btn_addition.clicked.connect(lambda: self.test_operation("+", 3, 5))
        button_layout.addWidget(btn_addition)

        btn_subtraction = QPushButton("Test Subtraktion: 10 - 4")
        btn_subtraction.clicked.connect(lambda: self.test_operation("-", 10, 4))
        button_layout.addWidget(btn_subtraction)

        btn_multiplication = QPushButton("Test Multiplikation: 6 × 7")
        btn_multiplication.clicked.connect(lambda: self.test_operation("*", 6, 7))
        button_layout.addWidget(btn_multiplication)

        btn_division = QPushButton("Test Division: 15 ÷ 3")
        btn_division.clicked.connect(lambda: self.test_operation("/", 15, 3))
        button_layout.addWidget(btn_division)

        btn_complex = QPushButton("Test Komplexe Berechnung")
        btn_complex.clicked.connect(self.test_complex_calculation)
        button_layout.addWidget(btn_complex)

        layout.addLayout(button_layout)

        # ProofTreeWidget
        self.proof_tree_widget = ProofTreeWidget()
        layout.addWidget(self.proof_tree_widget)

        # Initialize Engine
        try:
            self.netzwerk = KonzeptNetzwerkCore()
            self.arithmetic_engine = ArithmeticEngine(self.netzwerk)
            print("[OK] ArithmeticEngine initialisiert")
        except Exception as e:
            print(f"[ERROR] Fehler bei Initialisierung: {e}")
            self.arithmetic_engine = None

    def test_operation(self, operator: str, operand1, operand2):
        """Testet einzelne Operation"""
        if not self.arithmetic_engine:
            print("[ERROR] ArithmeticEngine nicht verfügbar")
            return

        try:
            # Berechnung durchführen
            result = self.arithmetic_engine.calculate(operator, operand1, operand2)

            print(f"\n{'='*60}")
            print(f"Test: {operand1} {operator} {operand2}")
            print(f"Ergebnis: {result.value}")
            print(f"Confidence: {result.confidence:.2f}")
            print(f"Proof Tree Steps: {len(result.proof_tree.get_all_steps())}")

            # ProofTree visualisieren
            self.proof_tree_widget.set_proof_tree(result.proof_tree)

            # Proof Tree Text ausgeben
            from component_17_proof_explanation import format_proof_tree
            proof_text = format_proof_tree(result.proof_tree, show_details=True)
            print(f"\n{proof_text}")

        except Exception as e:
            print(f"[ERROR] Fehler bei Test: {e}")
            import traceback
            traceback.print_exc()

    def test_complex_calculation(self):
        """Testet komplexe Berechnung mit mehreren Schritten"""
        if not self.arithmetic_engine:
            print("[ERROR] ArithmeticEngine nicht verfügbar")
            return

        try:
            print(f"\n{'='*60}")
            print("Test: Komplexe Berechnung (mehrere Schritte)")

            # Schritt 1: 3 + 5 = 8
            result1 = self.arithmetic_engine.calculate("+", 3, 5)
            print(f"Schritt 1: 3 + 5 = {result1.value}")

            # Schritt 2: 8 * 2 = 16
            result2 = self.arithmetic_engine.calculate("*", result1.value, 2)
            print(f"Schritt 2: {result1.value} * 2 = {result2.value}")

            # Schritt 3: 16 / 4 = 4
            result3 = self.arithmetic_engine.calculate("/", result2.value, 4)
            print(f"Schritt 3: {result2.value} / 4 = {result3.value}")

            # Zeige letzten Proof Tree
            self.proof_tree_widget.set_proof_tree(result3.proof_tree)

            print(f"\nFinale: ((3 + 5) * 2) / 4 = {result3.value}")

        except Exception as e:
            print(f"[ERROR] Fehler bei komplexer Berechnung: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Hauptfunktion"""
    app = QApplication(sys.argv)

    # Dark Theme
    app.setStyleSheet("""
        QWidget {
            background-color: #2c3e50;
            color: #ecf0f1;
            font-size: 14px;
        }
        QPushButton {
            background-color: #34495e;
            color: #ecf0f1;
            border: 2px solid #7f8c8d;
            border-radius: 6px;
            padding: 8px;
            font-size: 13px;
        }
        QPushButton:hover {
            background-color: #3498db;
            border: 2px solid #2980b9;
        }
        QPushButton:pressed {
            background-color: #2980b9;
        }
        QLabel {
            color: #3498db;
        }
    """)

    window = MathProofTestWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
