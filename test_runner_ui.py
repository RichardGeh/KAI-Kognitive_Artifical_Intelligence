"""
test_runner_ui.py

Test-Runner UI für KAI - Hierarchische Test-Ausführung mit Live-Updates.

Separiert aus settings_ui.py für eigenständige Verwendung.
"""

import subprocess
import sys
import traceback
from typing import Dict, List

from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)


class TestRunnerThread(QThread):
    """
    Thread zur Ausführung von pytest-Tests mit Live-Updates.

    Signals:
        test_started: Emitted wenn ein Test startet (test_name)
        test_passed: Emitted wenn ein Test erfolgreich ist (test_name)
        test_failed: Emitted wenn ein Test fehlschlägt (test_name, error_msg)
        progress_updated: Emitted bei Fortschritt (current, total, percentage)
        finished_all: Emitted wenn alle Tests abgeschlossen sind (passed, failed, total)
        output_line: Emitted für jede Ausgabezeile (line)
    """

    test_started = Signal(str)
    test_passed = Signal(str)
    test_failed = Signal(str, str)  # test_name, error_message
    progress_updated = Signal(int, int, float)  # current, total, percentage
    finished_all = Signal(int, int, int)  # passed, failed, total
    output_line = Signal(str)  # Ausgabezeile

    def __init__(self, selected_tests: List[tuple]):
        super().__init__()
        self.selected_tests = selected_tests  # List of (test_file, test_class) tuples
        self.should_stop = False

    def run(self):
        """Führt ausgewählte Tests aus und parst die Ausgabe"""
        if not self.selected_tests:
            # Keine Tests ausgewählt
            self.output_line.emit("⚠ Keine Tests ausgewählt")
            self.finished_all.emit(0, 0, 0)
            return

        # Gruppiere Tests nach Datei
        tests_by_file: Dict[str, List[str]] = {}
        for test_file, test_class in self.selected_tests:
            if test_file not in tests_by_file:
                tests_by_file[test_file] = []
            tests_by_file[test_file].append(test_class)

        # Baue Test-Spezifikationen
        # Für jede Datei: tests/test_file.py -k "TestClass1 or TestClass2"
        test_specs = []
        for test_file, test_classes in tests_by_file.items():
            test_filter = " or ".join(test_classes)
            test_specs.append((f"tests/{test_file}", test_filter))

        # Führe pytest mit verbose und kurzem traceback aus
        # Run all test files in a single pytest call for efficiency
        cmd = [sys.executable, "-m", "pytest"]

        # Add all test files
        for test_file_path, _ in test_specs:
            cmd.append(test_file_path)

        # Add combined filter for all test classes
        all_test_classes = [
            tc for _, test_classes in tests_by_file.items() for tc in test_classes
        ]
        test_filter = " or ".join(all_test_classes)

        cmd.extend(
            [
                "-k",
                test_filter,
                "-v",
                "--tb=short",
                "--no-header",
                "-p",
                "no:warnings",  # Unterdrücke Warning-Summary für saubere Ausgabe
            ]
        )

        self.output_line.emit(f"Kommando: {' '.join(cmd)}\n")
        self.output_line.emit("=" * 80)

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )

            passed_count = 0
            failed_count = 0
            total_count = 0

            # Parse Output Zeile für Zeile
            for line in process.stdout:
                if self.should_stop:
                    process.terminate()
                    self.output_line.emit("\n⏹ Test-Ausführung gestoppt")
                    break

                # Emit jede Zeile für Konsolen-Ausgabe
                self.output_line.emit(line.rstrip())

                line = line.strip()

                # Erkenne Test-Start (pytest -v Format)
                if "::" in line and (" PASSED" in line or " FAILED" in line):
                    # Beispiel: test_kai_worker.py::TestGoalPlanner::test_high_confidence_direct_execution PASSED
                    parts = line.split("::")
                    if len(parts) >= 3:
                        test_class = parts[1]
                        test_name_with_status = parts[2]
                        test_name = test_name_with_status.split()[0]
                        full_test_name = f"{test_class}::{test_name}"

                        total_count += 1
                        self.test_started.emit(full_test_name)

                        if "PASSED" in line:
                            passed_count += 1
                            self.test_passed.emit(full_test_name)
                        elif "FAILED" in line:
                            failed_count += 1
                            # Fehlermeldung kommt in nächsten Zeilen - hier erstmal kurz
                            self.test_failed.emit(
                                full_test_name, "Siehe Test-Output für Details"
                            )

                        # Update Progress
                        percentage = (
                            total_count / max(1, total_count)
                        ) * 100  # Wird später korrigiert
                        self.progress_updated.emit(total_count, total_count, percentage)

            return_code = process.wait()

            # Final Update
            self.output_line.emit("=" * 80)
            self.output_line.emit(f"Tests abgeschlossen. Return Code: {return_code}")
            self.output_line.emit(
                f"Bestanden: {passed_count}, Fehlgeschlagen: {failed_count}, Gesamt: {total_count}"
            )
            self.finished_all.emit(passed_count, failed_count, total_count)

        except FileNotFoundError as e:
            error_msg = f"Fehler: pytest nicht gefunden. Bitte installieren Sie pytest: pip install pytest\n\nDetails: {e}"
            self.output_line.emit(f"\n[ERROR] {error_msg}")
            self.test_failed.emit("System", error_msg)
            self.finished_all.emit(0, 1, 1)
        except Exception as e:
            # Bei Fehler: Alle als fehlgeschlagen markieren
            error_msg = f"Fehler beim Ausführen der Tests:\n{e}\n\nTraceback:\n{traceback.format_exc()}"
            self.output_line.emit(f"\n[ERROR] {error_msg}")
            self.test_failed.emit("System", error_msg)
            self.finished_all.emit(0, 1, 1)

    def stop(self):
        """Stoppt die Test-Ausführung"""
        self.should_stop = True



class TestRunnerTab(QWidget):
    """Tab für Test-Ausführung mit hierarchischer Tree-Ansicht"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.test_runner_thread = None
        self.test_tree_items = {}  # Maps (test_file, test_class) -> QTreeWidgetItem
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # === Such-/Filterzeile ===
        search_layout = QHBoxLayout()
        search_label = QLabel("🔍 Filter:")
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Test-Namen filtern...")
        self.search_input.textChanged.connect(self.filter_tests)

        search_layout.addWidget(search_label)
        search_layout.addWidget(self.search_input)

        # === Buttons ===
        button_row = QHBoxLayout()
        select_all_btn = QPushButton("Alle auswählen")
        select_all_btn.clicked.connect(self.select_all_tests)
        select_none_btn = QPushButton("Keine auswählen")
        select_none_btn.clicked.connect(self.select_no_tests)
        expand_all_btn = QPushButton("Alle ausklappen")
        expand_all_btn.clicked.connect(self.expand_all)
        collapse_all_btn = QPushButton("Alle einklappen")
        collapse_all_btn.clicked.connect(self.collapse_all)

        button_row.addWidget(select_all_btn)
        button_row.addWidget(select_none_btn)
        button_row.addWidget(expand_all_btn)
        button_row.addWidget(collapse_all_btn)
        button_row.addStretch()

        # === Test Tree ===
        selection_group = QGroupBox("Test-Auswahl")
        selection_layout = QVBoxLayout()

        self.test_tree = QTreeWidget()
        self.test_tree.setHeaderLabels(["Test", "Status"])
        self.test_tree.setColumnWidth(0, 400)
        self.test_tree.setAlternatingRowColors(True)
        self.test_tree.itemChanged.connect(self.on_item_changed)

        # Populate tree
        self.populate_tree()

        selection_layout.addLayout(search_layout)
        selection_layout.addLayout(button_row)
        selection_layout.addWidget(self.test_tree)
        selection_group.setLayout(selection_layout)

        # === Ausführungs-Steuerung ===
        control_layout = QHBoxLayout()

        self.run_button = QPushButton("▶ Tests ausführen")
        self.run_button.clicked.connect(self.run_tests)
        self.run_button.setStyleSheet(
            "background-color: #27ae60; font-weight: bold; padding: 10px;"
        )

        self.stop_button = QPushButton("⏹ Stoppen")
        self.stop_button.clicked.connect(self.stop_tests)
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet("background-color: #e74c3c; padding: 10px;")

        control_layout.addWidget(self.run_button)
        control_layout.addWidget(self.stop_button)

        # === Progress-Anzeige ===
        progress_group = QGroupBox("Fortschritt")
        progress_layout = QVBoxLayout()

        self.progress_label = QLabel("Bereit")
        self.progress_label.setStyleSheet("font-size: 14px; font-weight: bold;")

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)

        progress_layout.addWidget(self.progress_label)
        progress_layout.addWidget(self.progress_bar)
        progress_group.setLayout(progress_layout)

        # === Test-Ausgabe (Konsole) ===
        output_group = QGroupBox("Test-Ausgabe")
        output_layout = QVBoxLayout()

        self.output_console = QTextEdit()
        self.output_console.setReadOnly(True)
        self.output_console.setMaximumHeight(200)
        self.output_console.setStyleSheet(
            "QTextEdit { background-color: #2c3e50; color: #ecf0f1; font-family: 'Courier New', monospace; font-size: 11px; }"
        )
        self.output_console.setPlaceholderText("Test-Ausgabe erscheint hier...")

        output_layout.addWidget(self.output_console)
        output_group.setLayout(output_layout)

        # === Fehlgeschlagene Tests ===
        failed_group = QGroupBox("Fehlgeschlagene Tests")
        failed_layout = QVBoxLayout()

        self.failed_tests_list = QListWidget()
        self.failed_tests_list.setMaximumHeight(100)
        self.failed_tests_list.setStyleSheet(
            "QListWidget::item { color: #e74c3c; padding: 5px; }"
        )

        failed_layout.addWidget(self.failed_tests_list)
        failed_group.setLayout(failed_layout)

        # === Zusammensetzen ===
        layout.addWidget(selection_group)
        layout.addLayout(control_layout)
        layout.addWidget(progress_group)
        layout.addWidget(output_group)
        layout.addWidget(failed_group)
        layout.addStretch()

    def populate_tree(self):
        """Füllt den Tree mit allen Tests, gruppiert nach Kategorien und Dateien"""

        # Test-Daten: (Kategorie, DisplayName, test_file, test_class)
        test_data = self.get_test_data()

        # Gruppiere nach Kategorie -> Datei -> Test-Klassen
        categories = {}
        for category, display_name, test_file, test_class in test_data:
            if category not in categories:
                categories[category] = {}
            if test_file not in categories[category]:
                categories[category][test_file] = []
            categories[category][test_file].append((display_name, test_class))

        # Erstelle Tree
        self.test_tree.blockSignals(True)  # Verhindere Events während Aufbau

        for category_name in sorted(categories.keys()):
            category_item = QTreeWidgetItem(self.test_tree, [f"📂 {category_name}", ""])
            category_item.setFlags(
                category_item.flags()
                | Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsAutoTristate
            )
            category_item.setCheckState(0, Qt.CheckState.Checked)
            category_item.setExpanded(False)  # Standardmäßig eingeklappt
            category_item.setFont(0, self.get_bold_font())

            for test_file in sorted(categories[category_name].keys()):
                file_item = QTreeWidgetItem(category_item, [f"📄 {test_file}", ""])
                file_item.setFlags(
                    file_item.flags()
                    | Qt.ItemFlag.ItemIsUserCheckable
                    | Qt.ItemFlag.ItemIsAutoTristate
                )
                file_item.setCheckState(0, Qt.CheckState.Checked)
                file_item.setFont(0, self.get_semi_bold_font())

                for display_name, test_class in sorted(
                    categories[category_name][test_file]
                ):
                    test_item = QTreeWidgetItem(file_item, [f"  {display_name}", ""])
                    test_item.setFlags(
                        test_item.flags() | Qt.ItemFlag.ItemIsUserCheckable
                    )
                    test_item.setCheckState(0, Qt.CheckState.Checked)
                    test_item.setData(
                        0, Qt.ItemDataRole.UserRole, (test_file, test_class)
                    )

                    # Speichere Referenz für spätere Updates
                    self.test_tree_items[(test_file, test_class)] = test_item

        self.test_tree.blockSignals(False)

    def get_bold_font(self):
        """Gibt eine fette Schrift zurück"""
        from PySide6.QtGui import QFont

        font = QFont()
        font.setBold(True)
        font.setPointSize(11)
        return font

    def get_semi_bold_font(self):
        """Gibt eine halbfette Schrift zurück"""
        from PySide6.QtGui import QFont

        font = QFont()
        font.setWeight(QFont.Weight.DemiBold)
        return font

    def on_item_changed(self, item, column):
        """Wird aufgerufen wenn ein Item geändert wurde"""
        # Diese Methode wird automatisch von Qt aufgerufen bei Checkbox-Änderungen
        # Tristate-Verhalten wird automatisch gehandhabt

    def filter_tests(self, search_text):
        """Filtert Tests basierend auf Suchtext"""
        search_text = search_text.lower()

        def set_item_visibility(item, visible):
            """Setzt Sichtbarkeit eines Items und aller Kinder"""
            item.setHidden(not visible)
            for i in range(item.childCount()):
                set_item_visibility(item.child(i), visible)

        if not search_text:
            # Zeige alles
            for i in range(self.test_tree.topLevelItemCount()):
                set_item_visibility(self.test_tree.topLevelItem(i), True)
            return

        # Durchsuche alle Items
        for i in range(self.test_tree.topLevelItemCount()):
            category_item = self.test_tree.topLevelItem(i)
            category_visible = False

            for j in range(category_item.childCount()):
                file_item = category_item.child(j)
                file_visible = False

                for k in range(file_item.childCount()):
                    test_item = file_item.child(k)
                    test_text = test_item.text(0).lower()

                    if search_text in test_text:
                        test_item.setHidden(False)
                        file_visible = True
                        category_visible = True
                    else:
                        test_item.setHidden(True)

                file_item.setHidden(not file_visible)

            category_item.setHidden(not category_visible)

    def select_all_tests(self):
        """Wählt alle Tests aus"""
        for i in range(self.test_tree.topLevelItemCount()):
            self.test_tree.topLevelItem(i).setCheckState(0, Qt.CheckState.Checked)

    def select_no_tests(self):
        """Wählt keine Tests aus"""
        for i in range(self.test_tree.topLevelItemCount()):
            self.test_tree.topLevelItem(i).setCheckState(0, Qt.CheckState.Unchecked)

    def expand_all(self):
        """Klappt alle Items aus"""
        self.test_tree.expandAll()

    def collapse_all(self):
        """Klappt alle Items ein"""
        self.test_tree.collapseAll()

    def get_selected_tests(self) -> List[tuple]:
        """Gibt Liste der ausgewählten Test-Klassen zurück als (test_file, test_class) Tupel"""
        selected = []

        def collect_checked(item):
            """Sammelt rekursiv alle gecheckte Test-Items"""
            # Nur Test-Items (Blätter) haben UserRole-Daten
            data = item.data(0, Qt.ItemDataRole.UserRole)
            if data is not None and item.checkState(0) == Qt.CheckState.Checked:
                selected.append(data)

            # Rekursiv Kinder durchgehen
            for i in range(item.childCount()):
                collect_checked(item.child(i))

        for i in range(self.test_tree.topLevelItemCount()):
            collect_checked(self.test_tree.topLevelItem(i))

        return selected

    def run_tests(self):
        """Startet Test-Ausführung"""
        selected = self.get_selected_tests()

        if not selected:
            self.progress_label.setText("⚠ Keine Tests ausgewählt")
            return

        # Reset Status in Tree
        for item in self.test_tree_items.values():
            item.setText(1, "")

        # UI-Updates
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.failed_tests_list.clear()
        self.output_console.clear()
        self.progress_bar.setValue(0)
        self.progress_label.setText(f"Starte {len(selected)} Test-Klassen...")

        # Starte Thread
        self.test_runner_thread = TestRunnerThread(selected)
        self.test_runner_thread.test_started.connect(self.on_test_started)
        self.test_runner_thread.test_passed.connect(self.on_test_passed)
        self.test_runner_thread.test_failed.connect(self.on_test_failed)
        self.test_runner_thread.progress_updated.connect(self.on_progress_updated)
        self.test_runner_thread.finished_all.connect(self.on_all_tests_finished)
        self.test_runner_thread.output_line.connect(self.on_output_line)
        self.test_runner_thread.start()

    def stop_tests(self):
        """Stoppt laufende Tests"""
        if self.test_runner_thread and self.test_runner_thread.isRunning():
            self.test_runner_thread.stop()
            self.progress_label.setText("⏹ Gestoppt...")

    @Slot(str)
    def on_test_started(self, test_name: str):
        """Wird aufgerufen wenn ein Test startet"""
        self.progress_label.setText(f"⏳ Läuft: {test_name}")

    @Slot(str)
    def on_test_passed(self, test_name: str):
        """Wird aufgerufen wenn ein Test erfolgreich ist"""
        # Finde Item und update Status
        for (test_file, test_class), item in self.test_tree_items.items():
            if f"{test_class}::" in test_name or test_class in test_name:
                item.setText(1, "✓")
                item.setForeground(1, Qt.GlobalColor.green)
                break

    @Slot(str, str)
    def on_test_failed(self, test_name: str, error_msg: str):
        """Wird aufgerufen wenn ein Test fehlschlägt"""
        # Finde Item und update Status
        for (test_file, test_class), item in self.test_tree_items.items():
            if f"{test_class}::" in test_name or test_class in test_name:
                item.setText(1, "✗")
                item.setForeground(1, Qt.GlobalColor.red)
                break

        # Füge zu Failed-Liste hinzu
        list_item = QListWidgetItem(f"[ERROR] {test_name}")
        list_item.setToolTip(error_msg)
        self.failed_tests_list.addItem(list_item)

    @Slot(str)
    def on_output_line(self, line: str):
        """Wird aufgerufen für jede Ausgabezeile"""
        self.output_console.append(line)
        # Auto-scroll nach unten
        scrollbar = self.output_console.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    @Slot(int, int, float)
    def on_progress_updated(self, current: int, total: int, percentage: float):
        """Wird aufgerufen bei Fortschritt"""
        # Grobe Schätzung: Durchschnittlich 6 Tests pro Klasse
        estimated_total = len(self.get_selected_tests()) * 6
        actual_percentage = (current / max(1, estimated_total)) * 100

        self.progress_bar.setValue(int(actual_percentage))
        self.progress_label.setText(f"🔄 {current} Tests durchlaufen...")

    @Slot(int, int, int)
    def on_all_tests_finished(self, passed: int, failed: int, total: int):
        """Wird aufgerufen wenn alle Tests abgeschlossen sind"""
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setValue(100)

        if failed == 0:
            self.progress_label.setText(f"✅ Alle {passed} Tests erfolgreich!")
            self.progress_label.setStyleSheet(
                "font-size: 14px; font-weight: bold; color: #27ae60;"
            )
        else:
            self.progress_label.setText(
                f"⚠ {passed} erfolgreich, {failed} fehlgeschlagen (Gesamt: {total})"
            )
            self.progress_label.setStyleSheet(
                "font-size: 14px; font-weight: bold; color: #e74c3c;"
            )

    def get_test_data(self):
        """Gibt alle Test-Daten zurück - COMPLETE LIST"""
        return [
            # System & Integration
            (
                "System & Integration",
                "SystemSetup",
                "test_system_setup.py",
                "TestSystemSetup",
            ),
            (
                "System & Integration",
                "KaiWorkerIntegration",
                "test_kai_integration.py",
                "TestKaiWorkerIntegration",
            ),
            (
                "System & Integration",
                "EdgeCases",
                "test_kai_integration.py",
                "TestEdgeCases",
            ),
            (
                "System & Integration",
                "DatabaseConsistency",
                "test_kai_integration.py",
                "TestDatabaseConsistency",
            ),
            (
                "System & Integration",
                "LimitsAndPerformance",
                "test_kai_integration.py",
                "TestLimitsAndPerformance",
            ),
            # KAI Worker
            (
                "KAI Worker",
                "Initialization",
                "test_kai_worker_extended.py",
                "TestKaiWorkerInitialization",
            ),
            (
                "KAI Worker",
                "Command Suggestions",
                "test_kai_worker_extended.py",
                "TestKaiWorkerCommandSuggestions",
            ),
            (
                "KAI Worker",
                "Pattern Recognition",
                "test_kai_worker_extended.py",
                "TestKaiWorkerPatternRecognition",
            ),
            (
                "KAI Worker",
                "Context Handling",
                "test_kai_worker_extended.py",
                "TestKaiWorkerContextHandling",
            ),
            (
                "KAI Worker",
                "Execute Plan",
                "test_kai_worker_extended.py",
                "TestKaiWorkerExecutePlan",
            ),
            (
                "KAI Worker",
                "Helper Methods",
                "test_kai_worker_extended.py",
                "TestKaiWorkerHelperMethods",
            ),
            # Netzwerk (Knowledge Graph)
            (
                "Knowledge Graph",
                "KonzeptNetzwerk Basic",
                "test_netzwerk_basic.py",
                "TestKonzeptNetzwerk",
            ),
            (
                "Knowledge Graph",
                "Core Extended",
                "test_netzwerk_extended.py",
                "TestKonzeptNetzwerkCore",
            ),
            (
                "Knowledge Graph",
                "Patterns Extended",
                "test_netzwerk_extended.py",
                "TestKonzeptNetzwerkPatterns",
            ),
            (
                "Knowledge Graph",
                "Memory Extended",
                "test_netzwerk_extended.py",
                "TestKonzeptNetzwerkMemory",
            ),
            (
                "Knowledge Graph",
                "Word Usage Extended",
                "test_netzwerk_extended.py",
                "TestKonzeptNetzwerkWordUsage",
            ),
            (
                "Knowledge Graph",
                "Feedback Extended",
                "test_netzwerk_extended.py",
                "TestKonzeptNetzwerkFeedback",
            ),
            (
                "Knowledge Graph",
                "Edge Cases",
                "test_netzwerk_extended.py",
                "TestKonzeptNetzwerkEdgeCases",
            ),
            (
                "Knowledge Graph",
                "Error Handling",
                "test_netzwerk_additional.py",
                "TestErrorHandling",
            ),
            (
                "Knowledge Graph",
                "Pattern Prototypes",
                "test_netzwerk_additional.py",
                "TestPatternPrototypes",
            ),
            (
                "Knowledge Graph",
                "Lexical Triggers",
                "test_netzwerk_additional.py",
                "TestLexicalTriggers",
            ),
            (
                "Knowledge Graph",
                "Other Queries",
                "test_netzwerk_additional.py",
                "TestOtherQueries",
            ),
            (
                "Knowledge Graph",
                "Similar Word Finding",
                "test_netzwerk_additional.py",
                "TestSimilarWordFinding",
            ),
            (
                "Knowledge Graph",
                "Episodic Memory",
                "test_netzwerk_additional.py",
                "TestEpisodicMemory",
            ),
            (
                "Knowledge Graph",
                "Inference Episodes",
                "test_netzwerk_additional.py",
                "TestInferenceEpisodes",
            ),
            (
                "Knowledge Graph",
                "Hypothesis Storage",
                "test_netzwerk_additional.py",
                "TestHypothesisStorage",
            ),
            (
                "Knowledge Graph",
                "Query Facts With Synonyms",
                "test_netzwerk_additional.py",
                "TestQueryFactsWithSynonyms",
            ),
            # NLP & Linguistic Processing
            (
                "NLP & Linguistics",
                "MeaningExtractor Basic",
                "test_meaning_extractor_basic.py",
                "TestMeaningExtractor",
            ),
            (
                "NLP & Linguistics",
                "Declarative Statements",
                "test_meaning_extractor_additional.py",
                "TestDeclarativeStatements",
            ),
            (
                "NLP & Linguistics",
                "Question Heuristics",
                "test_meaning_extractor_additional.py",
                "TestQuestionHeuristics",
            ),
            (
                "NLP & Linguistics",
                "Edge Cases",
                "test_meaning_extractor_additional.py",
                "TestEdgeCases",
            ),
            (
                "NLP & Linguistics",
                "Plural Normalization",
                "test_meaning_extractor_additional.py",
                "TestPluralNormalization",
            ),
            (
                "NLP & Linguistics",
                "Explicit Commands",
                "test_meaning_extractor_additional.py",
                "TestExplicitCommands",
            ),
            (
                "NLP & Linguistics",
                "Argument Extraction",
                "test_meaning_extractor_additional.py",
                "TestArgumentExtraction",
            ),
            (
                "NLP & Linguistics",
                "File Commands",
                "test_meaning_extractor_file_commands.py",
                "TestMeaningExtractorFileCommands",
            ),
            (
                "NLP & Linguistics",
                "Linguistik Engine Resource",
                "test_linguistik_engine.py",
                "TestResourceManager",
            ),
            (
                "NLP & Linguistics",
                "Linguistic Preprocessor",
                "test_linguistik_engine.py",
                "TestLinguisticPreprocessor",
            ),
            (
                "NLP & Linguistics",
                "Preprocessor Fallback",
                "test_linguistik_engine.py",
                "TestLinguisticPreprocessorFallback",
            ),
            (
                "NLP & Linguistics",
                "Linguistik Integration",
                "test_linguistik_engine.py",
                "TestIntegration",
            ),
            (
                "NLP & Linguistics",
                "Enhanced NLP",
                "test_enhanced_nlp.py",
                "TestEnhancedNLP",
            ),
            (
                "NLP & Linguistics",
                "Text Normalization Basic",
                "test_text_normalization.py",
                "TestTextNormalizerWithoutSpacy",
            ),
            (
                "NLP & Linguistics",
                "Text Normalization spaCy",
                "test_text_normalization.py",
                "TestTextNormalizerWithSpacy",
            ),
            (
                "NLP & Linguistics",
                "Text Convenience",
                "test_text_normalization.py",
                "TestConvenienceFunctions",
            ),
            (
                "NLP & Linguistics",
                "Text Regression",
                "test_text_normalization.py",
                "TestRegressionSuite",
            ),
            (
                "NLP & Linguistics",
                "Property-Based Normalization",
                "test_property_based_text_normalization.py",
                "test_",
            ),
            # Pattern Matching & Recognition
            (
                "Pattern Recognition",
                "Prototype Basic",
                "test_prototype_matcher_basic.py",
                "TestPrototypingEngine",
            ),
            (
                "Pattern Recognition",
                "Prototype Edge Cases",
                "test_prototype_matcher_additional.py",
                "TestPrototypingEngineEdgeCases",
            ),
            (
                "Pattern Recognition",
                "Prototype Update",
                "test_prototype_matcher_additional.py",
                "TestPrototypeUpdate",
            ),
            (
                "Pattern Recognition",
                "Process Vector",
                "test_prototype_matcher_additional.py",
                "TestProcessVectorScenarios",
            ),
            (
                "Pattern Recognition",
                "Find Best Match",
                "test_prototype_matcher_additional.py",
                "TestFindBestMatchScenarios",
            ),
            (
                "Pattern Recognition",
                "Keyboard Distance",
                "test_pattern_recognition_char.py",
                "TestKeyboardDistance",
            ),
            (
                "Pattern Recognition",
                "Weighted Levenshtein",
                "test_pattern_recognition_char.py",
                "TestWeightedLevenshtein",
            ),
            (
                "Pattern Recognition",
                "Typo Candidate Finder",
                "test_pattern_recognition_char.py",
                "TestTypoCandidateFinder",
            ),
            (
                "Pattern Recognition",
                "Feedback Integration",
                "test_pattern_recognition_char.py",
                "TestFeedbackIntegration",
            ),
            (
                "Pattern Recognition",
                "Sequence Predictor",
                "test_pattern_recognition_sequence.py",
                "TestSequencePredictor",
            ),
            ("Pattern Recognition", "E2E", "test_pattern_recognition_e2e.py", "test_"),
            (
                "Pattern Recognition",
                "Adaptive Thresholds",
                "test_adaptive_pattern_recognition.py",
                "TestAdaptiveThresholds",
            ),
            (
                "Pattern Recognition",
                "Word Frequency",
                "test_adaptive_pattern_recognition.py",
                "TestWordFrequency",
            ),
            (
                "Pattern Recognition",
                "Bayesian Quality",
                "test_adaptive_pattern_recognition.py",
                "TestBayesianPatternQuality",
            ),
            (
                "Pattern Recognition",
                "False Positive Reduction",
                "test_adaptive_pattern_recognition.py",
                "TestFalsePositiveReduction",
            ),
            (
                "Pattern Recognition",
                "Typo Feedback Recording",
                "test_adaptive_pattern_recognition.py",
                "TestTypoFeedbackRecording",
            ),
            # Learning & Knowledge Acquisition
            ("Learning", "Goal Planner", "test_goal_planner.py", "TestGoalPlanner"),
            (
                "Learning",
                "Interactive Learning",
                "test_interactive_learning.py",
                "TestInteractiveLearning",
            ),
            (
                "Learning",
                "Intelligent Ingestion",
                "test_intelligent_ingestion.py",
                "TestIntelligentIngestion",
            ),
            ("Learning", "Autonomous Learning", "test_autonomous_learning.py", "test_"),
            ("Learning", "Lerne Command", "test_lerne_command.py", "test_"),
            (
                "Learning",
                "Auto Detect Definitions",
                "test_auto_detect_definitions.py",
                "test_",
            ),
            ("Learning", "Auto Detect E2E", "test_auto_detect_e2e.py", "test_"),
            ("Learning", "Definition Strategy", "test_definition_strategy.py", "test_"),
            (
                "Learning",
                "Plural & Definitions",
                "test_plural_and_definitions.py",
                "TestPluralNormalization",
            ),
            (
                "Learning",
                "Definition Storage",
                "test_plural_and_definitions.py",
                "TestDefinitionStorage",
            ),
            (
                "Learning",
                "Integration Scenario",
                "test_plural_and_definitions.py",
                "TestIntegrationScenario",
            ),
            # Reasoning Engines
            (
                "Reasoning",
                "Backward Chaining",
                "test_backward_chaining.py",
                "TestBackwardChaining",
            ),
            (
                "Reasoning",
                "Graph Traversal",
                "test_graph_traversal.py",
                "TestTransitiveRelations",
            ),
            ("Reasoning", "Path Finding", "test_graph_traversal.py", "TestPathFinding"),
            (
                "Reasoning",
                "Multiple Paths",
                "test_graph_traversal.py",
                "TestMultiplePaths",
            ),
            (
                "Reasoning",
                "Inference Explanation",
                "test_graph_traversal.py",
                "TestInferenceExplanation",
            ),
            (
                "Reasoning",
                "Concept Hierarchy",
                "test_graph_traversal.py",
                "TestConceptHierarchy",
            ),
            (
                "Reasoning",
                "Inverse Traversal",
                "test_graph_traversal.py",
                "TestInverseTraversal",
            ),
            (
                "Reasoning",
                "Graph Edge Cases",
                "test_graph_traversal.py",
                "TestEdgeCases",
            ),
            (
                "Reasoning",
                "German Explanations",
                "test_graph_traversal.py",
                "TestGermanExplanations",
            ),
            (
                "Reasoning",
                "Confidence Propagation",
                "test_graph_traversal.py",
                "TestConfidencePropagation",
            ),
            (
                "Reasoning",
                "Abductive Hypothesis",
                "test_abductive_reasoning.py",
                "TestHypothesisGeneration",
            ),
            (
                "Reasoning",
                "Hypothesis Creation",
                "test_abductive_reasoning.py",
                "TestHypothesisCreation",
            ),
            (
                "Reasoning",
                "Template Strategy",
                "test_abductive_reasoning.py",
                "TestTemplateBasedStrategy",
            ),
            (
                "Reasoning",
                "Analogy Strategy",
                "test_abductive_reasoning.py",
                "TestAnalogyBasedStrategy",
            ),
            (
                "Reasoning",
                "Causal Chain",
                "test_abductive_reasoning.py",
                "TestCausalChainStrategy",
            ),
            (
                "Reasoning",
                "Multi Criteria Scoring",
                "test_abductive_reasoning.py",
                "TestMultiCriteriaScoring",
            ),
            (
                "Reasoning",
                "Hypothesis Persistence",
                "test_abductive_reasoning.py",
                "TestHypothesisPersistence",
            ),
            (
                "Reasoning",
                "Abductive Integration",
                "test_abductive_reasoning.py",
                "TestIntegrationWithKaiWorker",
            ),
            (
                "Reasoning",
                "Abductive Edge Cases",
                "test_abductive_reasoning.py",
                "TestEdgeCases",
            ),
            (
                "Reasoning",
                "Abductive Performance",
                "test_abductive_reasoning.py",
                "TestPerformance",
            ),
            (
                "Reasoning",
                "Probabilistic Fact",
                "test_probabilistic_engine.py",
                "TestBeliefState",
            ),
            (
                "Reasoning",
                "Probabilistic Engine",
                "test_probabilistic_engine.py",
                "TestProbabilisticFact",
            ),
            (
                "Reasoning",
                "Probabilistic Engine Core",
                "test_probabilistic_engine.py",
                "TestProbabilisticEngine",
            ),
            (
                "Reasoning",
                "Probabilistic+Logic",
                "test_probabilistic_engine.py",
                "TestIntegrationWithLogicEngine",
            ),
            (
                "Reasoning",
                "Probabilistic Edge Cases",
                "test_probabilistic_engine.py",
                "TestEdgeCases",
            ),
            # Hybrid Reasoning
            (
                "Hybrid Reasoning",
                "Orchestrator",
                "test_hybrid_reasoning.py",
                "TestReasoningOrchestrator",
            ),
            (
                "Hybrid Reasoning",
                "Proof Tree Generation",
                "test_hybrid_reasoning.py",
                "TestHybridProofTreeGeneration",
            ),
            (
                "Hybrid Reasoning",
                "Handler Integration",
                "test_hybrid_reasoning.py",
                "TestInferenceHandlerIntegration",
            ),
            (
                "Hybrid Reasoning",
                "Performance",
                "test_hybrid_reasoning.py",
                "TestPerformance",
            ),
            (
                "Hybrid Reasoning",
                "Edge Cases",
                "test_hybrid_reasoning.py",
                "TestEdgeCases",
            ),
            (
                "Hybrid Reasoning",
                "E2E",
                "test_hybrid_reasoning.py",
                "TestE2EHybridReasoning",
            ),
            (
                "Hybrid Reasoning",
                "Aggregation Methods",
                "test_hybrid_reasoning_phase2.py",
                "TestAggregationMethods",
            ),
            (
                "Hybrid Reasoning",
                "YAML Configuration",
                "test_hybrid_reasoning_phase2.py",
                "TestYAMLConfiguration",
            ),
            (
                "Hybrid Reasoning",
                "Result Caching",
                "test_hybrid_reasoning_phase2.py",
                "TestResultCaching",
            ),
            (
                "Hybrid Reasoning",
                "Parallel Execution",
                "test_hybrid_reasoning_phase2.py",
                "TestParallelExecution",
            ),
            (
                "Hybrid Reasoning",
                "Optimizations",
                "test_hybrid_reasoning_phase2.py",
                "TestPerformanceOptimizations",
            ),
            (
                "Hybrid Reasoning",
                "Strategy Weights",
                "test_hybrid_reasoning_phase2.py",
                "TestStrategyWeights",
            ),
            (
                "Hybrid Reasoning",
                "Phase2 Integration",
                "test_hybrid_reasoning_phase2.py",
                "TestPhase2Integration",
            ),
            # Constraint Reasoning & SAT
            (
                "Constraint & SAT",
                "Constraint Variable",
                "test_constraint_reasoning.py",
                "TestVariable",
            ),
            (
                "Constraint & SAT",
                "Constraint Definition",
                "test_constraint_reasoning.py",
                "TestConstraint",
            ),
            (
                "Constraint & SAT",
                "Constraint Problem",
                "test_constraint_reasoning.py",
                "TestConstraintProblem",
            ),
            (
                "Constraint & SAT",
                "Constraint Solver",
                "test_constraint_reasoning.py",
                "TestConstraintSolver",
            ),
            (
                "Constraint & SAT",
                "N-Queens Problem",
                "test_constraint_reasoning.py",
                "TestNQueensProblem",
            ),
            (
                "Constraint & SAT",
                "Graph Coloring",
                "test_constraint_reasoning.py",
                "TestGraphColoringProblem",
            ),
            (
                "Constraint & SAT",
                "Logic Grid Puzzle",
                "test_constraint_reasoning.py",
                "TestLogicGridPuzzle",
            ),
            (
                "Constraint & SAT",
                "ProofTree Integration",
                "test_constraint_reasoning.py",
                "TestProofTreeIntegration",
            ),
            (
                "Constraint & SAT",
                "Performance Metrics",
                "test_constraint_reasoning.py",
                "TestPerformanceMetrics",
            ),
            (
                "Constraint & SAT",
                "Logic Integration",
                "test_constraint_logic_integration.py",
                "TestConstraintLogicIntegration",
            ),
            (
                "Constraint & SAT",
                "Logic Edge Cases",
                "test_constraint_logic_integration.py",
                "TestConstraintLogicEdgeCases",
            ),
            ("Constraint & SAT", "SAT Literal", "test_sat_solver.py", "TestLiteral"),
            ("Constraint & SAT", "SAT Clause", "test_sat_solver.py", "TestClause"),
            (
                "Constraint & SAT",
                "SAT CNF Formula",
                "test_sat_solver.py",
                "TestCNFFormula",
            ),
            (
                "Constraint & SAT",
                "SAT DPLL Solver",
                "test_sat_solver.py",
                "TestDPLLSolver",
            ),
            ("Constraint & SAT", "SAT Encoder", "test_sat_solver.py", "TestSATEncoder"),
            (
                "Constraint & SAT",
                "Knights and Knaves",
                "test_sat_solver.py",
                "TestKnightsAndKnaves",
            ),
            (
                "Constraint & SAT",
                "Knowledge Base Checker",
                "test_sat_solver.py",
                "TestKnowledgeBaseChecker",
            ),
            (
                "Constraint & SAT",
                "Watched Literals",
                "test_sat_solver.py",
                "TestWatchedLiterals",
            ),
            (
                "Constraint & SAT",
                "Propositional Formula",
                "test_sat_solver.py",
                "TestPropositionalFormula",
            ),
            (
                "Constraint & SAT",
                "CNF Converter",
                "test_sat_solver.py",
                "TestCNFConverter",
            ),
            (
                "Constraint & SAT",
                "SAT Solver Wrapper",
                "test_sat_solver.py",
                "TestSATSolverWrapper",
            ),
            (
                "Constraint & SAT",
                "SAT Convenience",
                "test_sat_solver.py",
                "TestConvenienceFunctions",
            ),
            (
                "Constraint & SAT",
                "SAT Proof Generation",
                "test_sat_solver.py",
                "TestProofGeneration",
            ),
            (
                "Constraint & SAT",
                "SAT Logic Integration",
                "test_sat_solver.py",
                "TestIntegrationWithLogicEngine",
            ),
            (
                "Constraint & SAT",
                "SAT Performance",
                "test_sat_solver.py",
                "TestPerformance",
            ),
            (
                "Constraint & SAT",
                "SAT Consistency",
                "test_sat_consistency.py",
                "TestConsistencyChecking",
            ),
            (
                "Constraint & SAT",
                "SAT Contradiction",
                "test_sat_consistency.py",
                "TestContradictionFinding",
            ),
            (
                "Constraint & SAT",
                "SAT Integration",
                "test_sat_consistency.py",
                "TestIntegrationWithSAT",
            ),
            (
                "Constraint & SAT",
                "SAT Edge Cases",
                "test_sat_consistency.py",
                "TestEdgeCases",
            ),
            (
                "Constraint & SAT",
                "SAT Reasoning Consistency",
                "test_sat_reasoning.py",
                "TestConsistencyChecking",
            ),
            (
                "Constraint & SAT",
                "SAT Contradiction Detection",
                "test_sat_reasoning.py",
                "TestContradictionDetection",
            ),
            (
                "Constraint & SAT",
                "Knights and Knaves Reasoning",
                "test_sat_reasoning.py",
                "TestKnightsAndKnavesSAT",
            ),
            (
                "Constraint & SAT",
                "SAT Generic Capabilities",
                "test_sat_reasoning.py",
                "TestGenericSATCapabilities",
            ),
            (
                "Constraint & SAT",
                "SAT ProofTree",
                "test_sat_reasoning.py",
                "TestProofTreeIntegration",
            ),
            (
                "Constraint & SAT",
                "SAT Reasoning Performance",
                "test_sat_reasoning.py",
                "TestPerformanceAndEdgeCases",
            ),
            (
                "Constraint & SAT",
                "Contradiction Detection",
                "test_contradiction_detection.py",
                "test_",
            ),
            # State-Space Planning
            (
                "State-Space Planning",
                "State Representation",
                "test_state_space_planner.py",
                "TestStateRepresentation",
            ),
            (
                "State-Space Planning",
                "Action Model",
                "test_state_space_planner.py",
                "TestActionModel",
            ),
            (
                "State-Space Planning",
                "Blocks World",
                "test_state_space_planner.py",
                "TestBlocksWorld",
            ),
            (
                "State-Space Planning",
                "Grid Navigation",
                "test_state_space_planner.py",
                "TestGridNavigation",
            ),
            (
                "State-Space Planning",
                "River Crossing",
                "test_state_space_planner.py",
                "TestRiverCrossing",
            ),
            (
                "State-Space Planning",
                "Diagnostic Reasoning",
                "test_state_space_planner.py",
                "TestDiagnosticReasoning",
            ),
            (
                "State-Space Planning",
                "Heuristics",
                "test_state_space_planner.py",
                "TestHeuristics",
            ),
            (
                "State-Space Planning",
                "Temporal Reasoning",
                "test_state_space_planner.py",
                "TestTemporalReasoning",
            ),
            (
                "State-Space Planning",
                "Plan Simulation",
                "test_state_space_planner.py",
                "TestPlanSimulation",
            ),
            (
                "State-Space Planning",
                "Performance",
                "test_state_space_planner.py",
                "TestPerformance",
            ),
            (
                "State-Space Planning",
                "Hybrid Planning",
                "test_state_space_planner.py",
                "TestHybridPlanning",
            ),
            (
                "State-Space Planning",
                "Integration",
                "test_state_space_planner.py",
                "TestIntegration",
            ),
            (
                "State-Space Planning",
                "Blocks World State",
                "test_state_reasoning.py",
                "TestBlocksWorld",
            ),
            (
                "State-Space Planning",
                "Grid Navigation State",
                "test_state_reasoning.py",
                "TestGridNavigation",
            ),
            (
                "State-Space Planning",
                "River Crossing State",
                "test_state_reasoning.py",
                "TestRiverCrossing",
            ),
            (
                "State-Space Planning",
                "State Integration 31",
                "test_state_reasoning.py",
                "TestStateReasoningIntegration31",
            ),
            (
                "State-Space Planning",
                "State Performance 31",
                "test_state_reasoning.py",
                "TestPerformance31",
            ),
            (
                "State-Space Planning",
                "State Object",
                "test_state_reasoning.py",
                "TestState",
            ),
            (
                "State-Space Planning",
                "Action Object",
                "test_state_reasoning.py",
                "TestAction",
            ),
            (
                "State-Space Planning",
                "Planner Object",
                "test_state_reasoning.py",
                "TestStateSpacePlanner",
            ),
            (
                "State-Space Planning",
                "BFS Planning",
                "test_state_reasoning.py",
                "TestBFSPlanning",
            ),
            (
                "State-Space Planning",
                "Utility Functions",
                "test_state_reasoning.py",
                "TestUtilityFunctions",
            ),
            (
                "State-Space Planning",
                "Constraint Integration",
                "test_state_reasoning.py",
                "TestConstraintIntegration",
            ),
            (
                "State-Space Planning",
                "Complex Planning",
                "test_state_reasoning.py",
                "TestComplexPlanning",
            ),
            # Combinatorial Reasoning
            (
                "Combinatorial Reasoning",
                "Permutations & Cycles",
                "test_combinatorial_reasoning_prisoners.py",
                "TestCombinatorialReasoningPrisoners::test_permutation_creation_and_cycles",
            ),
            (
                "Combinatorial Reasoning",
                "Cycle Distribution",
                "test_combinatorial_reasoning_prisoners.py",
                "TestCombinatorialReasoningPrisoners::test_cycle_length_distribution",
            ),
            (
                "Combinatorial Reasoning",
                "Max Cycle Length",
                "test_combinatorial_reasoning_prisoners.py",
                "TestCombinatorialReasoningPrisoners::test_max_cycle_length",
            ),
            (
                "Combinatorial Reasoning",
                "Element Cycle Finder",
                "test_combinatorial_reasoning_prisoners.py",
                "TestCombinatorialReasoningPrisoners::test_find_element_cycle",
            ),
            (
                "Combinatorial Reasoning",
                "Probability Small",
                "test_combinatorial_reasoning_prisoners.py",
                "TestCombinatorialReasoningPrisoners::test_probability_max_cycle_exceeds_threshold_small",
            ),
            (
                "Combinatorial Reasoning",
                "Probability Large",
                "test_combinatorial_reasoning_prisoners.py",
                "TestCombinatorialReasoningPrisoners::test_probability_max_cycle_exceeds_threshold_large",
            ),
            (
                "Combinatorial Reasoning",
                "Expected Cycle Length",
                "test_combinatorial_reasoning_prisoners.py",
                "TestCombinatorialReasoningPrisoners::test_expected_max_cycle_length",
            ),
            (
                "Combinatorial Reasoning",
                "Random Strategy",
                "test_combinatorial_reasoning_prisoners.py",
                "TestCombinatorialReasoningPrisoners::test_strategy_random_evaluation",
            ),
            (
                "Combinatorial Reasoning",
                "Cycle Strategy",
                "test_combinatorial_reasoning_prisoners.py",
                "TestCombinatorialReasoningPrisoners::test_strategy_cycle_following_evaluation",
            ),
            (
                "Combinatorial Reasoning",
                "Strategy Comparison",
                "test_combinatorial_reasoning_prisoners.py",
                "TestCombinatorialReasoningPrisoners::test_compare_strategies_and_find_optimal",
            ),
            (
                "Combinatorial Reasoning",
                "Permutation Analysis",
                "test_combinatorial_reasoning_prisoners.py",
                "TestCombinatorialReasoningPrisoners::test_analyze_specific_permutation",
            ),
            (
                "Combinatorial Reasoning",
                "Permutation Composition",
                "test_combinatorial_reasoning_prisoners.py",
                "TestCombinatorialReasoningPrisoners::test_permutation_composition",
            ),
            (
                "Combinatorial Reasoning",
                "Strategy Parameters",
                "test_combinatorial_reasoning_prisoners.py",
                "TestCombinatorialReasoningPrisoners::test_strategy_parameter_access",
            ),
            (
                "Combinatorial Reasoning",
                "100 Prisoners Problem",
                "test_combinatorial_reasoning_prisoners.py",
                "TestCombinatorialReasoningPrisoners::test_comprehensive_prisoners_problem_solution",
            ),
            # Memory Systems
            (
                "Memory",
                "Episodic Memory Basic",
                "test_episodic_memory_basic.py",
                "TestEpisodicMemory",
            ),
            (
                "Memory",
                "Inference Episodes",
                "test_episodic_reasoning.py",
                "TestInferenceEpisodeCreation",
            ),
            (
                "Memory",
                "Proof Step Hierarchy",
                "test_episodic_reasoning.py",
                "TestProofStepHierarchy",
            ),
            (
                "Memory",
                "Inference Tracking",
                "test_episodic_reasoning.py",
                "TestInferenceTracking",
            ),
            (
                "Memory",
                "Episodic Edge Cases",
                "test_episodic_reasoning.py",
                "TestEdgeCases",
            ),
            (
                "Memory",
                "Working Memory State",
                "test_working_memory.py",
                "TestReasoningState",
            ),
            ("Memory", "Context Frame", "test_working_memory.py", "TestContextFrame"),
            ("Memory", "Working Memory", "test_working_memory.py", "TestWorkingMemory"),
            ("Memory", "WM Helpers", "test_working_memory.py", "TestHelperFunctions"),
            ("Memory", "WM Complex", "test_working_memory.py", "TestComplexScenarios"),
            (
                "Memory",
                "Deep Nested Contexts",
                "test_working_memory_stress.py",
                "TestDeepNestedContexts",
            ),
            (
                "Memory",
                "Large Reasoning Traces",
                "test_working_memory_stress.py",
                "TestLargeReasoningTraces",
            ),
            (
                "Memory",
                "Memory Leak Prevention",
                "test_working_memory_stress.py",
                "TestMemoryLeakPrevention",
            ),
            (
                "Memory",
                "Multi-Turn Scenarios",
                "test_working_memory_stress.py",
                "TestComplexMultiTurnScenarios",
            ),
            (
                "Memory",
                "Context Summary",
                "test_working_memory_stress.py",
                "TestContextSummaryAndExport",
            ),
            (
                "Memory",
                "Idle Cleanup",
                "test_working_memory_stress.py",
                "TestIdleCleanup",
            ),
            (
                "Memory",
                "Export Import",
                "test_working_memory_stress.py",
                "TestExportImport",
            ),
            (
                "Memory",
                "Word Usage Tracking",
                "test_word_usage_tracking.py",
                "TestTextFragmentation",
            ),
            (
                "Memory",
                "Word Usage Storage",
                "test_word_usage_tracking.py",
                "TestWordUsageStorage",
            ),
            (
                "Memory",
                "Fragment Similarity",
                "test_word_usage_tracking.py",
                "TestFragmentSimilarity",
            ),
            (
                "Memory",
                "Word Usage Integration",
                "test_word_usage_tracking.py",
                "TestWordUsageIntegration",
            ),
            (
                "Memory",
                "Config Integration",
                "test_word_usage_tracking.py",
                "TestConfigIntegration",
            ),
            # Proof & Explanation
            (
                "Proof & Explanation",
                "ProofStep Data",
                "test_proof_explanation.py",
                "TestProofStep",
            ),
            (
                "Proof & Explanation",
                "ProofTreeNode Data",
                "test_proof_explanation.py",
                "TestProofTreeNode",
            ),
            (
                "Proof & Explanation",
                "ProofTree Data",
                "test_proof_explanation.py",
                "TestProofTree",
            ),
            (
                "Proof & Explanation",
                "Explanation Generators",
                "test_proof_explanation.py",
                "TestExplanationGenerators",
            ),
            (
                "Proof & Explanation",
                "NL Generation",
                "test_proof_explanation.py",
                "TestNaturalLanguageGeneration",
            ),
            (
                "Proof & Explanation",
                "Helper Functions",
                "test_proof_explanation.py",
                "TestHelperFunctions",
            ),
            (
                "Proof & Explanation",
                "Integration Functions",
                "test_proof_explanation.py",
                "TestIntegrationFunctions",
            ),
            (
                "Proof & Explanation",
                "Export/Import",
                "test_proof_explanation.py",
                "TestExportImport",
            ),
            (
                "Proof & Explanation",
                "Complex Scenarios",
                "test_proof_explanation.py",
                "TestComplexScenarios",
            ),
            (
                "Proof & Explanation",
                "ProofNodeItem UI",
                "test_proof_tree_widget.py",
                "TestProofNodeItem",
            ),
            (
                "Proof & Explanation",
                "ProofEdgeItem UI",
                "test_proof_tree_widget.py",
                "TestProofEdgeItem",
            ),
            (
                "Proof & Explanation",
                "ProofTreeWidget UI",
                "test_proof_tree_widget.py",
                "TestProofTreeWidget",
            ),
            (
                "Proof & Explanation",
                "Tree Layout",
                "test_proof_tree_widget.py",
                "TestTreeLayoutAlgorithm",
            ),
            (
                "Proof & Explanation",
                "Proof Performance",
                "test_proof_tree_widget.py",
                "TestPerformance",
            ),
            (
                "Proof & Explanation",
                "Proof UI Integration",
                "test_proof_tree_widget.py",
                "TestIntegration",
            ),
            # Dialogue & Context
            (
                "Dialogue",
                "Multi-Turn Dialog",
                "test_dialog_system.py",
                "TestMultiTurnDialogSystem",
            ),
            ("Dialogue", "W-Fragen", "test_w_fragen.py", "TestWFragenVerarbeitung"),
            (
                "Dialogue",
                "All Relation Types",
                "test_all_relation_types.py",
                "TestAllRelationTypes",
            ),
            (
                "Dialogue",
                "Episodic Query Recognition",
                "test_episodic_query_ui.py",
                "TestEpisodicQueryRecognition",
            ),
            (
                "Dialogue",
                "Episodic Query Planning",
                "test_episodic_query_ui.py",
                "TestEpisodicQueryPlanning",
            ),
            (
                "Dialogue",
                "Episodic Response Formatting",
                "test_episodic_query_ui.py",
                "TestEpisodicResponseFormatting",
            ),
            (
                "Dialogue",
                "Episodic Memory Strategy",
                "test_episodic_query_ui.py",
                "TestEpisodicMemoryStrategy",
            ),
            # Document Processing
            (
                "Document Processing",
                "Document Parser",
                "test_document_parser.py",
                "TestDocumentParser",
            ),
            (
                "Document Processing",
                "File Reader Strategy",
                "test_file_reader_strategy.py",
                "TestFileReaderStrategy",
            ),
            (
                "Document Processing",
                "Goal Planner File",
                "test_goal_planner_file_reading.py",
                "TestGoalPlannerFileReading",
            ),
            (
                "Document Processing",
                "File Ingestion E2E",
                "test_file_ingestion_e2e.py",
                "TestFileIngestionE2E",
            ),
            # Confidence & Feedback
            (
                "Confidence",
                "Classification",
                "test_confidence_manager.py",
                "TestConfidenceClassification",
            ),
            (
                "Confidence",
                "Threshold Decisions",
                "test_confidence_manager.py",
                "TestThresholdDecisions",
            ),
            (
                "Confidence",
                "Combination",
                "test_confidence_manager.py",
                "TestConfidenceCombination",
            ),
            (
                "Confidence",
                "Decay",
                "test_confidence_manager.py",
                "TestConfidenceDecay",
            ),
            (
                "Confidence",
                "Specialized Methods",
                "test_confidence_manager.py",
                "TestSpecializedMethods",
            ),
            (
                "Confidence",
                "Metrics",
                "test_confidence_manager.py",
                "TestConfidenceMetrics",
            ),
            (
                "Confidence",
                "UI Feedback",
                "test_confidence_manager.py",
                "TestUIFeedback",
            ),
            (
                "Confidence",
                "Global Instance",
                "test_confidence_manager.py",
                "TestGlobalInstance",
            ),
            ("Confidence", "Edge Cases", "test_confidence_manager.py", "TestEdgeCases"),
            (
                "Confidence",
                "Feedback Collection",
                "test_confidence_feedback.py",
                "TestFeedbackCollection",
            ),
            (
                "Confidence",
                "Adjustment",
                "test_confidence_feedback.py",
                "TestConfidenceAdjustment",
            ),
            (
                "Confidence",
                "Historical Consensus",
                "test_confidence_feedback.py",
                "TestHistoricalConsensus",
            ),
            (
                "Confidence",
                "Persistence",
                "test_confidence_feedback.py",
                "TestFeedbackPersistence",
            ),
            (
                "Confidence",
                "Statistics",
                "test_confidence_feedback.py",
                "TestStatisticsAndReporting",
            ),
            (
                "Confidence",
                "Feedback Global",
                "test_confidence_feedback.py",
                "TestGlobalInstance",
            ),
            (
                "Confidence",
                "Feedback Edge Cases",
                "test_confidence_feedback.py",
                "TestEdgeCases",
            ),
            # Exception Handling
            (
                "Exceptions",
                "KAI Exception Base",
                "test_kai_exceptions.py",
                "TestKAIExceptionBase",
            ),
            (
                "Exceptions",
                "Database Exceptions",
                "test_kai_exceptions.py",
                "TestDatabaseExceptions",
            ),
            (
                "Exceptions",
                "Linguistic Exceptions",
                "test_kai_exceptions.py",
                "TestLinguisticExceptions",
            ),
            (
                "Exceptions",
                "Knowledge Exceptions",
                "test_kai_exceptions.py",
                "TestKnowledgeExceptions",
            ),
            (
                "Exceptions",
                "Reasoning Exceptions",
                "test_kai_exceptions.py",
                "TestReasoningExceptions",
            ),
            (
                "Exceptions",
                "Planning Exceptions",
                "test_kai_exceptions.py",
                "TestPlanningExceptions",
            ),
            (
                "Exceptions",
                "Configuration Exceptions",
                "test_kai_exceptions.py",
                "TestConfigurationExceptions",
            ),
            (
                "Exceptions",
                "Exception Wrapping",
                "test_kai_exceptions.py",
                "TestExceptionWrapping",
            ),
            (
                "Exceptions",
                "Exception Hierarchy",
                "test_kai_exceptions.py",
                "TestExceptionHierarchy",
            ),
            (
                "Exceptions",
                "Exception Catching",
                "test_kai_exceptions.py",
                "TestExceptionCatching",
            ),
            (
                "Exceptions",
                "Exception Handling",
                "test_exception_handling.py",
                "TestKaiExceptionHandling",
            ),
            (
                "Exceptions",
                "Hierarchy Basic",
                "test_exception_handling.py",
                "TestExceptionHierarchy",
            ),
            (
                "Exceptions",
                "Wrap Exception",
                "test_exception_handling.py",
                "TestWrapException",
            ),
            (
                "Exceptions",
                "User Friendly Messages",
                "test_exception_handling.py",
                "TestUserFriendlyMessages",
            ),
            (
                "Exceptions",
                "Exception Inheritance",
                "test_exception_handling.py",
                "TestExceptionInheritance",
            ),
            (
                "Exceptions",
                "Logging System",
                "test_logging_and_exceptions.py",
                "TestLoggingSystem",
            ),
            (
                "Exceptions",
                "Hierarchy Logging",
                "test_logging_and_exceptions.py",
                "TestExceptionHierarchy",
            ),
            (
                "Exceptions",
                "Logging Integration",
                "test_logging_and_exceptions.py",
                "TestIntegrationLoggingAndExceptions",
            ),
            # Performance & Infrastructure
            (
                "Performance",
                "Embedding Service",
                "test_embedding_service.py",
                "TestEmbeddingService",
            ),
            (
                "Performance",
                "Batch Embeddings",
                "test_performance_optimizations.py",
                "TestBatchEmbeddings",
            ),
            (
                "Performance",
                "Batch Processing",
                "test_performance_optimizations.py",
                "TestIngestionBatchProcessing",
            ),
            (
                "Performance",
                "Proof Tree Lazy Loading",
                "test_performance_optimizations.py",
                "TestProofTreeLazyLoading",
            ),
            (
                "Performance",
                "Neo4j Profiler",
                "test_performance_optimizations.py",
                "TestNeo4jQueryProfiler",
            ),
            (
                "Performance",
                "Performance Regression",
                "test_performance_optimizations.py",
                "TestPerformanceRegression",
            ),
            ("Performance", "Cache Performance", "test_cache_performance.py", "test_"),
            (
                "Performance",
                "Error Paths",
                "test_error_paths_and_resilience.py",
                "test_",
            ),
            # Arithmetic & Mathematics
            (
                "Arithmetic & Mathematics",
                "Basic Operations",
                "test_arithmetic_basic_operations.py",
                "test_",
            ),
            (
                "Arithmetic & Mathematics",
                "Comparisons",
                "test_arithmetic_comparisons.py",
                "test_",
            ),
            (
                "Arithmetic & Mathematics",
                "Properties",
                "test_arithmetic_properties.py",
                "test_",
            ),
            (
                "Arithmetic & Mathematics",
                "Concepts",
                "test_arithmetic_concepts.py",
                "test_",
            ),
            (
                "Arithmetic & Mathematics",
                "Fractions & Decimals",
                "test_arithmetic_fractions_decimals.py",
                "test_",
            ),
            (
                "Arithmetic & Mathematics",
                "Advanced Operations",
                "test_arithmetic_advanced.py",
                "test_",
            ),
            (
                "Arithmetic & Mathematics",
                "Number Language",
                "test_number_language.py",
                "test_",
            ),
        ]


# ============================================================================
# TEST RUNNER WINDOW (Standalone)
# ============================================================================


class TestRunnerWindow(QDialog):
    """Eigenständiges Test-Runner-Fenster"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("KAI Test-Runner")
        self.setModal(False)
        self.setMinimumSize(900, 700)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Hauptinhalt: TestRunnerTab
        self.test_runner_tab = TestRunnerTab(self)
        layout.addWidget(self.test_runner_tab)

        # Schließen-Button
        close_layout = QHBoxLayout()
        close_layout.addStretch()
        self.close_button = QPushButton("Schließen")
        self.close_button.clicked.connect(self.accept)
        close_layout.addWidget(self.close_button)

        layout.addLayout(close_layout)

