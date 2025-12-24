"""
test_runner_widget.py

UI-Widgets fuer den KAI Test-Runner.

Enthaelt:
- TestRunnerTab: Haupt-Widget mit Tree-Ansicht und Kontrollen
- TestRunnerWindow: Eigenstendiges Dialog-Fenster

Extrahiert aus test_runner_ui.py fuer bessere Modularitaet.
"""

from typing import List

from PySide6.QtCore import Qt, Slot
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

from test_runner_data import get_all_test_data
from test_runner_thread import TestRunnerThread


class TestRunnerTab(QWidget):
    """Tab fuer Test-Ausfuehrung mit hierarchischer Tree-Ansicht"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.test_runner_thread = None
        self.test_tree_items = {}  # Maps (test_file, test_class) -> QTreeWidgetItem
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # === Such-/Filterzeile ===
        search_layout = QHBoxLayout()
        search_label = QLabel("[Search] Filter:")
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Test-Namen filtern...")
        self.search_input.textChanged.connect(self.filter_tests)

        search_layout.addWidget(search_label)
        search_layout.addWidget(self.search_input)

        # === Buttons ===
        button_row = QHBoxLayout()
        select_all_btn = QPushButton("Alle auswaehlen")
        select_all_btn.clicked.connect(self.select_all_tests)
        select_none_btn = QPushButton("Keine auswaehlen")
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

        # === Ausfuehrungs-Steuerung ===
        control_layout = QHBoxLayout()

        self.run_button = QPushButton("[>] Tests ausfuehren")
        self.run_button.clicked.connect(self.run_tests)
        self.run_button.setStyleSheet(
            "background-color: #27ae60; font-weight: bold; padding: 10px;"
        )

        self.stop_button = QPushButton("[X] Stoppen")
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
        """Fuellt den Tree mit allen Tests, gruppiert nach Kategorien und Dateien"""

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
        self.test_tree.blockSignals(True)  # Verhindere Events waehrend Aufbau

        for category_name in sorted(categories.keys()):
            category_item = QTreeWidgetItem(self.test_tree, [f"[Folder] {category_name}", ""])
            category_item.setFlags(
                category_item.flags()
                | Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsAutoTristate
            )
            category_item.setCheckState(0, Qt.CheckState.Checked)
            category_item.setExpanded(False)  # Standardmaessig eingeklappt
            category_item.setFont(0, self.get_bold_font())

            for test_file in sorted(categories[category_name].keys()):
                file_item = QTreeWidgetItem(category_item, [f"[File] {test_file}", ""])
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

                    # Speichere Referenz fuer spaetere Updates
                    self.test_tree_items[(test_file, test_class)] = test_item

        self.test_tree.blockSignals(False)

    def get_bold_font(self):
        """Gibt eine fette Schrift zurueck"""
        from PySide6.QtGui import QFont

        font = QFont()
        font.setBold(True)
        font.setPointSize(11)
        return font

    def get_semi_bold_font(self):
        """Gibt eine halbfette Schrift zurueck"""
        from PySide6.QtGui import QFont

        font = QFont()
        font.setWeight(QFont.Weight.DemiBold)
        return font

    def on_item_changed(self, item, column):
        """Wird aufgerufen wenn ein Item geaendert wurde"""
        # Diese Methode wird automatisch von Qt aufgerufen bei Checkbox-Aenderungen
        # Tristate-Verhalten wird automatisch gehandhabt
        pass

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
        """Waehlt alle Tests aus"""
        for i in range(self.test_tree.topLevelItemCount()):
            self.test_tree.topLevelItem(i).setCheckState(0, Qt.CheckState.Checked)

    def select_no_tests(self):
        """Waehlt keine Tests aus"""
        for i in range(self.test_tree.topLevelItemCount()):
            self.test_tree.topLevelItem(i).setCheckState(0, Qt.CheckState.Unchecked)

    def expand_all(self):
        """Klappt alle Items aus"""
        self.test_tree.expandAll()

    def collapse_all(self):
        """Klappt alle Items ein"""
        self.test_tree.collapseAll()

    def get_selected_tests(self) -> List[tuple]:
        """Gibt Liste der ausgewaehlten Test-Klassen zurueck als (test_file, test_class) Tupel"""
        selected = []

        def collect_checked(item):
            """Sammelt rekursiv alle gecheckte Test-Items"""
            # Nur Test-Items (Blaetter) haben UserRole-Daten
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
        """Startet Test-Ausfuehrung"""
        selected = self.get_selected_tests()

        if not selected:
            self.progress_label.setText("[WARNING] Keine Tests ausgewaehlt")
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
            self.progress_label.setText("[STOPPED] Gestoppt...")

    @Slot(str)
    def on_test_started(self, test_name: str):
        """Wird aufgerufen wenn ein Test startet"""
        self.progress_label.setText(f"[RUNNING] Laeuft: {test_name}")

    @Slot(str)
    def on_test_passed(self, test_name: str):
        """Wird aufgerufen wenn ein Test erfolgreich ist"""
        # Finde Item und update Status
        for (test_file, test_class), item in self.test_tree_items.items():
            if f"{test_class}::" in test_name or test_class in test_name:
                item.setText(1, "[OK]")
                item.setForeground(1, Qt.GlobalColor.green)
                break

    @Slot(str, str)
    def on_test_failed(self, test_name: str, error_msg: str):
        """Wird aufgerufen wenn ein Test fehlschlaegt"""
        # Finde Item und update Status
        for (test_file, test_class), item in self.test_tree_items.items():
            if f"{test_class}::" in test_name or test_class in test_name:
                item.setText(1, "[FAIL]")
                item.setForeground(1, Qt.GlobalColor.red)
                break

        # Fuege zu Failed-Liste hinzu
        list_item = QListWidgetItem(f"[ERROR] {test_name}")
        list_item.setToolTip(error_msg)
        self.failed_tests_list.addItem(list_item)

    @Slot(str)
    def on_output_line(self, line: str):
        """Wird aufgerufen fuer jede Ausgabezeile"""
        self.output_console.append(line)
        # Auto-scroll nach unten
        scrollbar = self.output_console.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    @Slot(int, int, float)
    def on_progress_updated(self, current: int, total: int, percentage: float):
        """Wird aufgerufen bei Fortschritt"""
        # Grobe Schaetzung: Durchschnittlich 6 Tests pro Klasse
        estimated_total = len(self.get_selected_tests()) * 6
        actual_percentage = (current / max(1, estimated_total)) * 100

        self.progress_bar.setValue(int(actual_percentage))
        self.progress_label.setText(f"[PROGRESS] {current} Tests durchlaufen...")

    @Slot(int, int, int)
    def on_all_tests_finished(self, passed: int, failed: int, total: int):
        """Wird aufgerufen wenn alle Tests abgeschlossen sind"""
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setValue(100)

        if failed == 0:
            self.progress_label.setText(f"[SUCCESS] Alle {passed} Tests erfolgreich!")
            self.progress_label.setStyleSheet(
                "font-size: 14px; font-weight: bold; color: #27ae60;"
            )
        else:
            self.progress_label.setText(
                f"[WARNING] {passed} erfolgreich, {failed} fehlgeschlagen (Gesamt: {total})"
            )
            self.progress_label.setStyleSheet(
                "font-size: 14px; font-weight: bold; color: #e74c3c;"
            )

    def get_test_data(self):
        """Gibt alle Test-Daten zurueck - delegiert an test_runner_data Modul"""
        return get_all_test_data()


# ============================================================================
# TEST RUNNER WINDOW (Standalone)
# ============================================================================


class TestRunnerWindow(QDialog):
    """Eigenstaendiges Test-Runner-Fenster"""

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

        # Schliessen-Button
        close_layout = QHBoxLayout()
        close_layout.addStretch()
        self.close_button = QPushButton("Schliessen")
        self.close_button.clicked.connect(self.accept)
        close_layout.addWidget(self.close_button)

        layout.addLayout(close_layout)
