"""
settings_ui.py

Vereinheitlichter Einstellungen-Dialog für KAI mit mehreren Tabs:
- Logging-Einstellungen
- Test-Runner mit Progress-Anzeige

Ersetzt den alten "Logging"-Menü-Eintrag durch "Einstellungen".
"""

import logging
import subprocess
import sys
import traceback
from typing import List, Dict

from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QPushButton,
    QTextEdit,
    QGroupBox,
    QCheckBox,
    QWidget,
    QTabWidget,
    QProgressBar,
    QListWidget,
    QListWidgetItem,
    QScrollArea,
    QFrame,
    QLineEdit,
    QSlider,
    QDoubleSpinBox,
    QSpinBox,
    QTreeWidget,
    QTreeWidgetItem,
)

from component_15_logging_config import setup_logging, LOG_DIR, PERFORMANCE_LOG_FILE
from kai_config import get_config


# ============================================================================
# TEST RUNNER THREAD
# ============================================================================


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


# ============================================================================
# KAI SETTINGS TAB
# ============================================================================


class KaiSettingsTab(QWidget):
    """Tab für allgemeine KAI-Einstellungen (Wortverwendungs-Tracking, etc.)"""

    settings_changed = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)

        # Lade aktuelle Einstellungen aus Config
        cfg = get_config()
        self.current_settings = {
            "word_usage_tracking": cfg.get("word_usage_tracking", True),
            "usage_similarity_threshold": cfg.get("usage_similarity_threshold", 80),
            "context_window_size": cfg.get("context_window_size", 3),
            "max_words_to_comma": cfg.get("max_words_to_comma", 3),
        }

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # === Wortverwendungs-Tracking ===
        usage_group = QGroupBox("Wortverwendungs-Tracking")
        usage_layout = QVBoxLayout()

        self.usage_tracking_checkbox = QCheckBox(
            "Automatisch Wortverwendungen speichern"
        )
        self.usage_tracking_checkbox.setChecked(
            self.current_settings["word_usage_tracking"]
        )
        self.usage_tracking_checkbox.setToolTip(
            "Speichert Kontext-Fragmente für jedes Wort, um später Muster zu erkennen.\n"
            "Dies hilft KAI, typische Wortverbindungen zu lernen."
        )

        usage_info_label = QLabel(
            "Wenn aktiviert, speichert KAI bei jedem 'Lerne:' und 'Ingestiere Text:' Befehl "
            "typische Wortverbindungen mit Häufigkeits-Zählern. Dies ermöglicht spätere Mustererkennung."
        )
        usage_info_label.setWordWrap(True)
        usage_info_label.setStyleSheet(
            "color: #95a5a6; font-size: 11px; margin-top: 5px;"
        )

        usage_layout.addWidget(self.usage_tracking_checkbox)
        usage_layout.addWidget(usage_info_label)
        usage_group.setLayout(usage_layout)

        # === Ähnlichkeits-Schwellenwert ===
        similarity_group = QGroupBox("Ähnlichkeits-Schwellenwert")
        similarity_layout = QVBoxLayout()

        similarity_label = QLabel("Schwellenwert für ähnliche Kontext-Fragmente (%):")
        self.similarity_combo = QComboBox()
        self.similarity_combo.addItems(["70", "75", "80", "85", "90", "95", "100"])
        self.similarity_combo.setCurrentText(
            str(self.current_settings["usage_similarity_threshold"])
        )
        self.similarity_combo.setToolTip(
            "Bei welcher Ähnlichkeit sollen beide Counter erhöht werden?\n"
            "80% = 1-2 Wörter Unterschied OK, 100% = nur exakte Übereinstimmung"
        )

        similarity_info_label = QLabel(
            "Bestimmt, ab wann zwei Kontext-Fragmente als 'ähnlich genug' gelten.\n"
            "Beispiel bei 80%: 'im großen Haus' ≈ 'im großen alten Haus' (beide Counter werden erhöht)"
        )
        similarity_info_label.setWordWrap(True)
        similarity_info_label.setStyleSheet(
            "color: #95a5a6; font-size: 11px; margin-top: 5px;"
        )

        similarity_layout.addWidget(similarity_label)
        similarity_layout.addWidget(self.similarity_combo)
        similarity_layout.addWidget(similarity_info_label)
        similarity_group.setLayout(similarity_layout)

        # === Kontext-Fenster ===
        context_group = QGroupBox("Kontext-Extraktion")
        context_layout = QVBoxLayout()

        # Context Window Size
        window_label = QLabel("Kontext-Fenster-Größe (±N Wörter):")
        self.window_combo = QComboBox()
        self.window_combo.addItems(["1", "2", "3", "4", "5"])
        self.window_combo.setCurrentText(
            str(self.current_settings["context_window_size"])
        )
        self.window_combo.setToolTip(
            "Wie viele Wörter vor und nach dem Zielwort sollen gespeichert werden?"
        )

        # Max Words to Comma
        comma_label = QLabel("Max. Wörter bis Komma:")
        self.comma_combo = QComboBox()
        self.comma_combo.addItems(["2", "3", "4", "5", "6"])
        self.comma_combo.setCurrentText(
            str(self.current_settings["max_words_to_comma"])
        )
        self.comma_combo.setToolTip(
            "Wenn weniger als N Wörter bis zum Komma, wird bis Komma gespeichert.\n"
            "Sonst wird auf ±N Fenster begrenzt."
        )

        context_info_label = QLabel(
            "Kontext-Fragmente werden entweder bis zum nächsten Komma ODER ±N Wörter gespeichert.\n"
            "Dynamischer Schwellenwert verhindert zu lange Fragmente."
        )
        context_info_label.setWordWrap(True)
        context_info_label.setStyleSheet(
            "color: #95a5a6; font-size: 11px; margin-top: 5px;"
        )

        context_layout.addWidget(window_label)
        context_layout.addWidget(self.window_combo)
        context_layout.addWidget(comma_label)
        context_layout.addWidget(self.comma_combo)
        context_layout.addWidget(context_info_label)
        context_group.setLayout(context_layout)

        # === Beispiel-Visualisierung ===
        example_group = QGroupBox("Beispiel")
        example_layout = QVBoxLayout()

        example_text = QLabel(
            "Beispiel-Satz: 'Das Haus steht im großen Park.'\n\n"
            "Für 'großen' wird gespeichert:\n"
            "* CONNECTION Edges: 'im' -> 'großen' (distance=1, count++)\n"
            "                    'großen' -> 'Park' (distance=1, count++)\n"
            "* UsageContext: 'im großen Park' (±3 Fenster, word_position=1, count++)\n\n"
            "Bei erneutem Auftreten von 'im großen Park' -> count wird erhöht statt neuen Node zu erstellen."
        )
        example_text.setWordWrap(True)
        example_text.setStyleSheet(
            "color: #3498db; font-size: 11px; font-family: 'Courier New', monospace;"
        )

        example_layout.addWidget(example_text)
        example_group.setLayout(example_layout)

        # Layout zusammensetzen
        layout.addWidget(usage_group)
        layout.addWidget(similarity_group)
        layout.addWidget(context_group)
        layout.addWidget(example_group)
        layout.addStretch()

    def get_settings(self) -> dict:
        """Gibt aktuelle Einstellungen zurück"""
        return {
            "word_usage_tracking": self.usage_tracking_checkbox.isChecked(),
            "usage_similarity_threshold": int(self.similarity_combo.currentText()),
            "context_window_size": int(self.window_combo.currentText()),
            "max_words_to_comma": int(self.comma_combo.currentText()),
        }

    def apply_settings(self):
        """Speichert Einstellungen in persistenter Config"""
        new_settings = self.get_settings()
        self.current_settings = new_settings

        # Speichern in kai_config.py
        cfg = get_config()
        cfg.update(new_settings)

        self.settings_changed.emit(new_settings)


# ============================================================================
# LOGGING SETTINGS TAB
# ============================================================================


class LoggingSettingsTab(QWidget):
    """Tab für Logging-Einstellungen (aus logging_ui.py übernommen)"""

    settings_changed = Signal(dict)

    LOG_LEVELS = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    def __init__(self, parent=None):
        super().__init__(parent)

        self.current_settings = {
            "console_level": "INFO",
            "file_level": "DEBUG",
            "performance_logging": True,
        }

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # === Konsolen-Logging ===
        console_group = QGroupBox("Konsolen-Logging")
        console_layout = QVBoxLayout()

        console_label = QLabel("Log-Level für Konsolen-Ausgabe:")
        self.console_level_combo = QComboBox()
        self.console_level_combo.addItems(self.LOG_LEVELS.keys())
        self.console_level_combo.setCurrentText(self.current_settings["console_level"])

        console_layout.addWidget(console_label)
        console_layout.addWidget(self.console_level_combo)
        console_group.setLayout(console_layout)

        # === Datei-Logging ===
        file_group = QGroupBox("Datei-Logging")
        file_layout = QVBoxLayout()

        file_label = QLabel("Log-Level für Datei-Ausgabe:")
        self.file_level_combo = QComboBox()
        self.file_level_combo.addItems(self.LOG_LEVELS.keys())
        self.file_level_combo.setCurrentText(self.current_settings["file_level"])

        file_info_label = QLabel(
            f"Log-Dateien werden gespeichert in:\n{LOG_DIR.absolute()}"
        )
        file_info_label.setStyleSheet(
            "color: #95a5a6; font-size: 11px; margin-top: 10px;"
        )
        file_info_label.setWordWrap(True)

        file_layout.addWidget(file_label)
        file_layout.addWidget(self.file_level_combo)
        file_layout.addWidget(file_info_label)
        file_group.setLayout(file_layout)

        # === Performance-Logging ===
        perf_group = QGroupBox("Performance-Logging")
        perf_layout = QVBoxLayout()

        self.perf_checkbox = QCheckBox("Performance-Metriken in separate Datei loggen")
        self.perf_checkbox.setChecked(self.current_settings["performance_logging"])

        perf_info_label = QLabel(
            "Performance-Logs enthalten Zeitstempel für kritische Operationen.\n"
            f"Datei: {PERFORMANCE_LOG_FILE.name}"
        )
        perf_info_label.setStyleSheet(
            "color: #95a5a6; font-size: 11px; margin-top: 5px;"
        )
        perf_info_label.setWordWrap(True)

        perf_layout.addWidget(self.perf_checkbox)
        perf_layout.addWidget(perf_info_label)
        perf_group.setLayout(perf_layout)

        # === Info ===
        info_group = QGroupBox("Hinweise")
        info_layout = QVBoxLayout()

        info_text = QLabel(
            "* Niedrigere Log-Levels zeigen mehr Details (DEBUG > INFO > WARNING > ERROR > CRITICAL)\n"
            "* Konsolen-Level: Ausgabe im Terminal während KAI läuft\n"
            "* Datei-Level: Persistente Logs für spätere Analyse\n"
            "* Änderungen werden beim Klick auf 'Anwenden' aktiv"
        )
        info_text.setWordWrap(True)
        info_text.setStyleSheet("color: #3498db; font-size: 11px;")

        info_layout.addWidget(info_text)
        info_group.setLayout(info_layout)

        # Layout zusammensetzen
        layout.addWidget(console_group)
        layout.addWidget(file_group)
        layout.addWidget(perf_group)
        layout.addWidget(info_group)
        layout.addStretch()

    def get_settings(self) -> dict:
        """Gibt aktuelle Einstellungen zurück"""
        return {
            "console_level": self.console_level_combo.currentText(),
            "file_level": self.file_level_combo.currentText(),
            "performance_logging": self.perf_checkbox.isChecked(),
        }

    def apply_settings(self):
        """Wendet Einstellungen an"""
        new_settings = self.get_settings()

        setup_logging(
            console_level=self.LOG_LEVELS[new_settings["console_level"]],
            file_level=self.LOG_LEVELS[new_settings["file_level"]],
            enable_performance_logging=new_settings["performance_logging"],
        )

        self.current_settings = new_settings
        self.settings_changed.emit(new_settings)


# ============================================================================
# NEO4J CONNECTION TAB
# ============================================================================


class Neo4jConnectionTab(QWidget):
    """Tab für Neo4j-Verbindungseinstellungen"""

    settings_changed = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)

        # Lade aktuelle Einstellungen
        cfg = get_config()
        self.current_settings = {
            "neo4j_uri": cfg.get("neo4j_uri", "bolt://127.0.0.1:7687"),
            "neo4j_user": cfg.get("neo4j_user", "neo4j"),
            "neo4j_password": cfg.get("neo4j_password", "password"),
        }

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # === Connection Parameters ===
        connection_group = QGroupBox("Verbindungsparameter")
        connection_layout = QVBoxLayout()

        # URI
        uri_label = QLabel("Neo4j URI:")
        self.uri_edit = QLineEdit()
        self.uri_edit.setText(self.current_settings["neo4j_uri"])
        self.uri_edit.setPlaceholderText("bolt://127.0.0.1:7687")
        self.uri_edit.setToolTip("Format: bolt://host:port oder neo4j://host:port")

        # User
        user_label = QLabel("Benutzername:")
        self.user_edit = QLineEdit()
        self.user_edit.setText(self.current_settings["neo4j_user"])
        self.user_edit.setPlaceholderText("neo4j")

        # Password
        password_label = QLabel("Passwort:")
        self.password_edit = QLineEdit()
        self.password_edit.setText(self.current_settings["neo4j_password"])
        self.password_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.password_edit.setPlaceholderText("password")

        # Show Password Checkbox
        self.show_password_checkbox = QCheckBox("Passwort anzeigen")
        self.show_password_checkbox.stateChanged.connect(
            self.toggle_password_visibility
        )

        connection_layout.addWidget(uri_label)
        connection_layout.addWidget(self.uri_edit)
        connection_layout.addWidget(user_label)
        connection_layout.addWidget(self.user_edit)
        connection_layout.addWidget(password_label)
        connection_layout.addWidget(self.password_edit)
        connection_layout.addWidget(self.show_password_checkbox)
        connection_group.setLayout(connection_layout)

        # === Hinweis ===
        info_group = QGroupBox("Hinweise")
        info_layout = QVBoxLayout()

        info_text = QLabel(
            "* Änderungen werden erst nach Neustart von KAI aktiv\n"
            "* Stelle sicher, dass Neo4j läuft und die Zugangsdaten korrekt sind\n"
            "* Standard-Port: 7687 (bolt protocol)\n"
            "* Die Verbindung wird beim Start von KAI getestet"
        )
        info_text.setWordWrap(True)
        info_text.setStyleSheet("color: #3498db; font-size: 11px;")

        info_layout.addWidget(info_text)
        info_group.setLayout(info_layout)

        # Layout zusammensetzen
        layout.addWidget(connection_group)
        layout.addWidget(info_group)
        layout.addStretch()

    def toggle_password_visibility(self, state):
        """Schaltet Passwort-Sichtbarkeit um"""
        if state == 2:  # Qt.CheckState.Checked
            self.password_edit.setEchoMode(QLineEdit.EchoMode.Normal)
        else:
            self.password_edit.setEchoMode(QLineEdit.EchoMode.Password)

    def get_settings(self) -> dict:
        """Gibt aktuelle Einstellungen zurück"""
        return {
            "neo4j_uri": self.uri_edit.text().strip(),
            "neo4j_user": self.user_edit.text().strip(),
            "neo4j_password": self.password_edit.text(),
        }

    def apply_settings(self):
        """Speichert Einstellungen"""
        new_settings = self.get_settings()
        self.current_settings = new_settings

        cfg = get_config()
        cfg.update(new_settings)

        self.settings_changed.emit(new_settings)


# ============================================================================
# CONFIDENCE THRESHOLDS TAB
# ============================================================================


class ConfidenceThresholdsTab(QWidget):
    """Tab für Konfidenz-Schwellenwerte (GoalPlanner)"""

    settings_changed = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)

        cfg = get_config()
        self.current_settings = {
            "confidence_low_threshold": cfg.get("confidence_low_threshold", 0.40),
            "confidence_medium_threshold": cfg.get("confidence_medium_threshold", 0.85),
        }

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # === Low Confidence Threshold ===
        low_group = QGroupBox("Niedrige Konfidenz (Rückfrage)")
        low_layout = QVBoxLayout()

        low_label = QLabel("Schwellenwert für Klärungsbedarf:")
        self.low_slider = QSlider(Qt.Orientation.Horizontal)
        self.low_slider.setMinimum(0)
        self.low_slider.setMaximum(100)
        self.low_slider.setValue(
            int(self.current_settings["confidence_low_threshold"] * 100)
        )
        self.low_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.low_slider.setTickInterval(10)

        self.low_value_label = QLabel(
            f"{self.current_settings['confidence_low_threshold']:.2f}"
        )
        self.low_value_label.setStyleSheet("font-weight: bold; color: #e74c3c;")
        self.low_slider.valueChanged.connect(
            lambda v: self.low_value_label.setText(f"{v/100:.2f}")
        )

        low_info = QLabel(
            "Unter diesem Wert fragt KAI nach, was gemeint ist.\n"
            "Beispiel: Bei 0.40 werden sehr unsichere Interpretationen hinterfragt."
        )
        low_info.setWordWrap(True)
        low_info.setStyleSheet("color: #95a5a6; font-size: 11px; margin-top: 5px;")

        low_layout.addWidget(low_label)
        low_layout.addWidget(self.low_slider)
        low_layout.addWidget(self.low_value_label)
        low_layout.addWidget(low_info)
        low_group.setLayout(low_layout)

        # === Medium Confidence Threshold ===
        medium_group = QGroupBox("Mittlere Konfidenz (Bestätigung)")
        medium_layout = QVBoxLayout()

        medium_label = QLabel("Schwellenwert für Bestätigungsanfrage:")
        self.medium_slider = QSlider(Qt.Orientation.Horizontal)
        self.medium_slider.setMinimum(0)
        self.medium_slider.setMaximum(100)
        self.medium_slider.setValue(
            int(self.current_settings["confidence_medium_threshold"] * 100)
        )
        self.medium_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.medium_slider.setTickInterval(10)

        self.medium_value_label = QLabel(
            f"{self.current_settings['confidence_medium_threshold']:.2f}"
        )
        self.medium_value_label.setStyleSheet("font-weight: bold; color: #f39c12;")
        self.medium_slider.valueChanged.connect(
            lambda v: self.medium_value_label.setText(f"{v/100:.2f}")
        )

        medium_info = QLabel(
            "Unter diesem Wert fragt KAI um Bestätigung.\n"
            "Über diesem Wert führt KAI die Aktion direkt aus.\n"
            "Beispiel: Bei 0.85 werden nur sehr sichere Interpretationen direkt ausgeführt."
        )
        medium_info.setWordWrap(True)
        medium_info.setStyleSheet("color: #95a5a6; font-size: 11px; margin-top: 5px;")

        medium_layout.addWidget(medium_label)
        medium_layout.addWidget(self.medium_slider)
        medium_layout.addWidget(self.medium_value_label)
        medium_layout.addWidget(medium_info)
        medium_group.setLayout(medium_layout)

        # === Beispiel ===
        example_group = QGroupBox("Beispiel")
        example_layout = QVBoxLayout()

        example_text = QLabel(
            "Eingabe: 'Ein Hund ist ein Tier'\n\n"
            "* Konfidenz: 0.92 -> Direkte Ausführung (≥ Medium Threshold)\n"
            "* Konfidenz: 0.78 -> Bestätigungsanfrage (< Medium Threshold)\n"
            "* Konfidenz: 0.35 -> Klärungsfrage (< Low Threshold)"
        )
        example_text.setWordWrap(True)
        example_text.setStyleSheet(
            "color: #3498db; font-size: 11px; font-family: 'Courier New', monospace;"
        )

        example_layout.addWidget(example_text)
        example_group.setLayout(example_layout)

        # Layout zusammensetzen
        layout.addWidget(low_group)
        layout.addWidget(medium_group)
        layout.addWidget(example_group)
        layout.addStretch()

    def get_settings(self) -> dict:
        """Gibt aktuelle Einstellungen zurück"""
        return {
            "confidence_low_threshold": self.low_slider.value() / 100,
            "confidence_medium_threshold": self.medium_slider.value() / 100,
        }

    def apply_settings(self):
        """Speichert Einstellungen"""
        new_settings = self.get_settings()
        self.current_settings = new_settings

        cfg = get_config()
        cfg.update(new_settings)

        self.settings_changed.emit(new_settings)


# ============================================================================
# PATTERN MATCHING TAB
# ============================================================================


class PatternMatchingTab(QWidget):
    """Tab für Pattern-Matching Thresholds"""

    settings_changed = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)

        cfg = get_config()
        self.current_settings = {
            "prototype_novelty_threshold": cfg.get("prototype_novelty_threshold", 15.0),
            "typo_min_threshold": cfg.get("typo_min_threshold", 3),
            "typo_max_threshold": cfg.get("typo_max_threshold", 10),
            "sequence_min_threshold": cfg.get("sequence_min_threshold", 2),
            "sequence_max_threshold": cfg.get("sequence_max_threshold", 5),
        }

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # === Prototype Novelty Threshold ===
        novelty_group = QGroupBox("Prototype Novelty Threshold")
        novelty_layout = QVBoxLayout()

        novelty_label = QLabel(
            "Schwellenwert für neue Prototypen (Euklidische Distanz):"
        )
        self.novelty_spinbox = QDoubleSpinBox()
        self.novelty_spinbox.setMinimum(5.0)
        self.novelty_spinbox.setMaximum(30.0)
        self.novelty_spinbox.setSingleStep(0.5)
        self.novelty_spinbox.setValue(
            self.current_settings["prototype_novelty_threshold"]
        )
        self.novelty_spinbox.setToolTip(
            "Distanz in 384D semantischem Raum.\n"
            "< Threshold: Update existierenden Prototyp\n"
            "≥ Threshold: Erstelle neuen Prototyp"
        )

        novelty_info = QLabel(
            "Bestimmt, wann ein Satz als 'neu genug' gilt, um einen eigenen Prototyp zu erhalten.\n"
            "Standard: 15.0 (ausgewogen zwischen Clustering und Granularität)"
        )
        novelty_info.setWordWrap(True)
        novelty_info.setStyleSheet("color: #95a5a6; font-size: 11px; margin-top: 5px;")

        novelty_layout.addWidget(novelty_label)
        novelty_layout.addWidget(self.novelty_spinbox)
        novelty_layout.addWidget(novelty_info)
        novelty_group.setLayout(novelty_layout)

        # === Typo Detection Thresholds ===
        typo_group = QGroupBox("Typo Detection (Adaptive)")
        typo_layout = QVBoxLayout()

        typo_min_label = QLabel("Minimum Wort-Vorkommen:")
        self.typo_min_spinbox = QSpinBox()
        self.typo_min_spinbox.setMinimum(1)
        self.typo_min_spinbox.setMaximum(10)
        self.typo_min_spinbox.setValue(self.current_settings["typo_min_threshold"])

        typo_max_label = QLabel("Maximum Wort-Vorkommen:")
        self.typo_max_spinbox = QSpinBox()
        self.typo_max_spinbox.setMinimum(5)
        self.typo_max_spinbox.setMaximum(20)
        self.typo_max_spinbox.setValue(self.current_settings["typo_max_threshold"])

        typo_info = QLabel(
            "Adaptive Thresholds basierend auf Vocabulary-Größe.\n"
            "Formel: min(MAX, max(MIN, vocab_size^0.4))\n"
            "Beispiel: Bei 100 Wörtern im Vocab -> Threshold = 4"
        )
        typo_info.setWordWrap(True)
        typo_info.setStyleSheet("color: #95a5a6; font-size: 11px; margin-top: 5px;")

        typo_layout.addWidget(typo_min_label)
        typo_layout.addWidget(self.typo_min_spinbox)
        typo_layout.addWidget(typo_max_label)
        typo_layout.addWidget(self.typo_max_spinbox)
        typo_layout.addWidget(typo_info)
        typo_group.setLayout(typo_layout)

        # === Sequence Prediction Thresholds ===
        sequence_group = QGroupBox("Sequence Prediction (Adaptive)")
        sequence_layout = QVBoxLayout()

        seq_min_label = QLabel("Minimum Sequenz-Vorkommen:")
        self.seq_min_spinbox = QSpinBox()
        self.seq_min_spinbox.setMinimum(1)
        self.seq_min_spinbox.setMaximum(5)
        self.seq_min_spinbox.setValue(self.current_settings["sequence_min_threshold"])

        seq_max_label = QLabel("Maximum Sequenz-Vorkommen:")
        self.seq_max_spinbox = QSpinBox()
        self.seq_max_spinbox.setMinimum(3)
        self.seq_max_spinbox.setMaximum(10)
        self.seq_max_spinbox.setValue(self.current_settings["sequence_max_threshold"])

        seq_info = QLabel(
            "Adaptive Thresholds basierend auf CONNECTION-Dichte.\n"
            "Formel: min(MAX, max(MIN, connection_count^0.35))\n"
            "Beispiel: Bei 100 Connections -> Threshold = 3"
        )
        seq_info.setWordWrap(True)
        seq_info.setStyleSheet("color: #95a5a6; font-size: 11px; margin-top: 5px;")

        sequence_layout.addWidget(seq_min_label)
        sequence_layout.addWidget(self.seq_min_spinbox)
        sequence_layout.addWidget(seq_max_label)
        sequence_layout.addWidget(self.seq_max_spinbox)
        sequence_layout.addWidget(seq_info)
        sequence_group.setLayout(sequence_layout)

        # Layout zusammensetzen
        layout.addWidget(novelty_group)
        layout.addWidget(typo_group)
        layout.addWidget(sequence_group)
        layout.addStretch()

    def get_settings(self) -> dict:
        """Gibt aktuelle Einstellungen zurück"""
        return {
            "prototype_novelty_threshold": self.novelty_spinbox.value(),
            "typo_min_threshold": self.typo_min_spinbox.value(),
            "typo_max_threshold": self.typo_max_spinbox.value(),
            "sequence_min_threshold": self.seq_min_spinbox.value(),
            "sequence_max_threshold": self.seq_max_spinbox.value(),
        }

    def apply_settings(self):
        """Speichert Einstellungen"""
        new_settings = self.get_settings()
        self.current_settings = new_settings

        cfg = get_config()
        cfg.update(new_settings)

        self.settings_changed.emit(new_settings)


# ============================================================================
# APPEARANCE TAB
# ============================================================================


class AppearanceTab(QWidget):
    """Tab für UI-Einstellungen (Theme, etc.)"""

    settings_changed = Signal(dict)
    theme_changed = Signal(str)  # Spezial-Signal für Theme-Änderung

    def __init__(self, parent=None):
        super().__init__(parent)

        cfg = get_config()
        self.current_settings = {
            "theme": cfg.get("theme", "dark"),
        }

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # === Theme Selection ===
        theme_group = QGroupBox("Farbschema")
        theme_layout = QVBoxLayout()

        theme_label = QLabel("Theme auswählen:")
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["dark", "light"])
        self.theme_combo.setCurrentText(self.current_settings["theme"])
        self.theme_combo.setToolTip("Wähle zwischen Dark Mode und Light Mode")

        theme_info = QLabel(
            "* Dark Mode: Dunkles Farbschema (Standard)\n"
            "* Light Mode: Helles Farbschema\n"
            "* Änderungen werden sofort nach 'Anwenden' aktiv"
        )
        theme_info.setWordWrap(True)
        theme_info.setStyleSheet("color: #95a5a6; font-size: 11px; margin-top: 5px;")

        theme_layout.addWidget(theme_label)
        theme_layout.addWidget(self.theme_combo)
        theme_layout.addWidget(theme_info)
        theme_group.setLayout(theme_layout)

        # === Vorschau ===
        preview_group = QGroupBox("Vorschau")
        preview_layout = QVBoxLayout()

        self.preview_label = QLabel("Dies ist ein Beispiel-Text im aktuellen Theme")
        self.preview_label.setStyleSheet(
            "padding: 20px; border: 2px solid #7f8c8d; border-radius: 5px; font-size: 14px;"
        )
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        preview_layout.addWidget(self.preview_label)
        preview_group.setLayout(preview_layout)

        # === Hinweis ===
        info_group = QGroupBox("Hinweise")
        info_layout = QVBoxLayout()

        info_text = QLabel(
            "* Das Theme wird beim nächsten Start von KAI vollständig angewendet\n"
            "* Einige UI-Elemente aktualisieren sich sofort nach 'Anwenden'"
        )
        info_text.setWordWrap(True)
        info_text.setStyleSheet("color: #3498db; font-size: 11px;")

        info_layout.addWidget(info_text)
        info_group.setLayout(info_layout)

        # Layout zusammensetzen
        layout.addWidget(theme_group)
        layout.addWidget(preview_group)
        layout.addWidget(info_group)
        layout.addStretch()

        # Connect theme combo to preview update
        self.theme_combo.currentTextChanged.connect(self.update_preview)
        self.update_preview(self.current_settings["theme"])

    def update_preview(self, theme: str):
        """Aktualisiert Vorschau basierend auf ausgewähltem Theme"""
        if theme == "dark":
            self.preview_label.setStyleSheet(
                "background-color: #2c3e50; color: #ecf0f1; "
                "padding: 20px; border: 2px solid #7f8c8d; border-radius: 5px; font-size: 14px;"
            )
        else:  # light
            self.preview_label.setStyleSheet(
                "background-color: #ecf0f1; color: #2c3e50; "
                "padding: 20px; border: 2px solid #7f8c8d; border-radius: 5px; font-size: 14px;"
            )

    def get_settings(self) -> dict:
        """Gibt aktuelle Einstellungen zurück"""
        return {
            "theme": self.theme_combo.currentText(),
        }

    def apply_settings(self):
        """Speichert Einstellungen"""
        new_settings = self.get_settings()
        old_theme = self.current_settings["theme"]
        new_theme = new_settings["theme"]

        self.current_settings = new_settings

        cfg = get_config()
        cfg.update(new_settings)

        self.settings_changed.emit(new_settings)

        # Emit theme_changed signal if theme was changed
        if old_theme != new_theme:
            self.theme_changed.emit(new_theme)


# ============================================================================
# TEST RUNNER TAB
# ============================================================================


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
                | Qt.ItemFlag.ItemIsTristate
            )
            category_item.setCheckState(0, Qt.CheckState.Checked)
            category_item.setExpanded(False)  # Standardmäßig eingeklappt
            category_item.setFont(0, self.get_bold_font())

            for test_file in sorted(categories[category_name].keys()):
                file_item = QTreeWidgetItem(category_item, [f"📄 {test_file}", ""])
                file_item.setFlags(
                    file_item.flags()
                    | Qt.ItemFlag.ItemIsUserCheckable
                    | Qt.ItemFlag.ItemIsTristate
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
        ]


"""
settings_ui.py

Vereinheitlichter Einstellungen-Dialog für KAI mit mehreren Tabs:
- Logging-Einstellungen
- Test-Runner mit Progress-Anzeige

Ersetzt den alten "Logging"-Menü-Eintrag durch "Einstellungen".
"""

import logging
import subprocess
import sys
import traceback
from typing import List, Dict

from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QPushButton,
    QTextEdit,
    QGroupBox,
    QCheckBox,
    QWidget,
    QTabWidget,
    QProgressBar,
    QListWidget,
    QListWidgetItem,
    QScrollArea,
    QFrame,
    QLineEdit,
    QSlider,
    QDoubleSpinBox,
    QSpinBox,
    QTreeWidget,
    QTreeWidgetItem,
)

from component_15_logging_config import setup_logging, LOG_DIR, PERFORMANCE_LOG_FILE
from kai_config import get_config


# ============================================================================
# TEST RUNNER THREAD
# ============================================================================


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


# ============================================================================
# KAI SETTINGS TAB
# ============================================================================


class KaiSettingsTab(QWidget):
    """Tab für allgemeine KAI-Einstellungen (Wortverwendungs-Tracking, etc.)"""

    settings_changed = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)

        # Lade aktuelle Einstellungen aus Config
        cfg = get_config()
        self.current_settings = {
            "word_usage_tracking": cfg.get("word_usage_tracking", True),
            "usage_similarity_threshold": cfg.get("usage_similarity_threshold", 80),
            "context_window_size": cfg.get("context_window_size", 3),
            "max_words_to_comma": cfg.get("max_words_to_comma", 3),
        }

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # === Wortverwendungs-Tracking ===
        usage_group = QGroupBox("Wortverwendungs-Tracking")
        usage_layout = QVBoxLayout()

        self.usage_tracking_checkbox = QCheckBox(
            "Automatisch Wortverwendungen speichern"
        )
        self.usage_tracking_checkbox.setChecked(
            self.current_settings["word_usage_tracking"]
        )
        self.usage_tracking_checkbox.setToolTip(
            "Speichert Kontext-Fragmente für jedes Wort, um später Muster zu erkennen.\n"
            "Dies hilft KAI, typische Wortverbindungen zu lernen."
        )

        usage_info_label = QLabel(
            "Wenn aktiviert, speichert KAI bei jedem 'Lerne:' und 'Ingestiere Text:' Befehl "
            "typische Wortverbindungen mit Häufigkeits-Zählern. Dies ermöglicht spätere Mustererkennung."
        )
        usage_info_label.setWordWrap(True)
        usage_info_label.setStyleSheet(
            "color: #95a5a6; font-size: 11px; margin-top: 5px;"
        )

        usage_layout.addWidget(self.usage_tracking_checkbox)
        usage_layout.addWidget(usage_info_label)
        usage_group.setLayout(usage_layout)

        # === Ähnlichkeits-Schwellenwert ===
        similarity_group = QGroupBox("Ähnlichkeits-Schwellenwert")
        similarity_layout = QVBoxLayout()

        similarity_label = QLabel("Schwellenwert für ähnliche Kontext-Fragmente (%):")
        self.similarity_combo = QComboBox()
        self.similarity_combo.addItems(["70", "75", "80", "85", "90", "95", "100"])
        self.similarity_combo.setCurrentText(
            str(self.current_settings["usage_similarity_threshold"])
        )
        self.similarity_combo.setToolTip(
            "Bei welcher Ähnlichkeit sollen beide Counter erhöht werden?\n"
            "80% = 1-2 Wörter Unterschied OK, 100% = nur exakte Übereinstimmung"
        )

        similarity_info_label = QLabel(
            "Bestimmt, ab wann zwei Kontext-Fragmente als 'ähnlich genug' gelten.\n"
            "Beispiel bei 80%: 'im großen Haus' ≈ 'im großen alten Haus' (beide Counter werden erhöht)"
        )
        similarity_info_label.setWordWrap(True)
        similarity_info_label.setStyleSheet(
            "color: #95a5a6; font-size: 11px; margin-top: 5px;"
        )

        similarity_layout.addWidget(similarity_label)
        similarity_layout.addWidget(self.similarity_combo)
        similarity_layout.addWidget(similarity_info_label)
        similarity_group.setLayout(similarity_layout)

        # === Kontext-Fenster ===
        context_group = QGroupBox("Kontext-Extraktion")
        context_layout = QVBoxLayout()

        # Context Window Size
        window_label = QLabel("Kontext-Fenster-Größe (±N Wörter):")
        self.window_combo = QComboBox()
        self.window_combo.addItems(["1", "2", "3", "4", "5"])
        self.window_combo.setCurrentText(
            str(self.current_settings["context_window_size"])
        )
        self.window_combo.setToolTip(
            "Wie viele Wörter vor und nach dem Zielwort sollen gespeichert werden?"
        )

        # Max Words to Comma
        comma_label = QLabel("Max. Wörter bis Komma:")
        self.comma_combo = QComboBox()
        self.comma_combo.addItems(["2", "3", "4", "5", "6"])
        self.comma_combo.setCurrentText(
            str(self.current_settings["max_words_to_comma"])
        )
        self.comma_combo.setToolTip(
            "Wenn weniger als N Wörter bis zum Komma, wird bis Komma gespeichert.\n"
            "Sonst wird auf ±N Fenster begrenzt."
        )

        context_info_label = QLabel(
            "Kontext-Fragmente werden entweder bis zum nächsten Komma ODER ±N Wörter gespeichert.\n"
            "Dynamischer Schwellenwert verhindert zu lange Fragmente."
        )
        context_info_label.setWordWrap(True)
        context_info_label.setStyleSheet(
            "color: #95a5a6; font-size: 11px; margin-top: 5px;"
        )

        context_layout.addWidget(window_label)
        context_layout.addWidget(self.window_combo)
        context_layout.addWidget(comma_label)
        context_layout.addWidget(self.comma_combo)
        context_layout.addWidget(context_info_label)
        context_group.setLayout(context_layout)

        # === Beispiel-Visualisierung ===
        example_group = QGroupBox("Beispiel")
        example_layout = QVBoxLayout()

        example_text = QLabel(
            "Beispiel-Satz: 'Das Haus steht im großen Park.'\n\n"
            "Für 'großen' wird gespeichert:\n"
            "* CONNECTION Edges: 'im' -> 'großen' (distance=1, count++)\n"
            "                    'großen' -> 'Park' (distance=1, count++)\n"
            "* UsageContext: 'im großen Park' (±3 Fenster, word_position=1, count++)\n\n"
            "Bei erneutem Auftreten von 'im großen Park' -> count wird erhöht statt neuen Node zu erstellen."
        )
        example_text.setWordWrap(True)
        example_text.setStyleSheet(
            "color: #3498db; font-size: 11px; font-family: 'Courier New', monospace;"
        )

        example_layout.addWidget(example_text)
        example_group.setLayout(example_layout)

        # Layout zusammensetzen
        layout.addWidget(usage_group)
        layout.addWidget(similarity_group)
        layout.addWidget(context_group)
        layout.addWidget(example_group)
        layout.addStretch()

    def get_settings(self) -> dict:
        """Gibt aktuelle Einstellungen zurück"""
        return {
            "word_usage_tracking": self.usage_tracking_checkbox.isChecked(),
            "usage_similarity_threshold": int(self.similarity_combo.currentText()),
            "context_window_size": int(self.window_combo.currentText()),
            "max_words_to_comma": int(self.comma_combo.currentText()),
        }

    def apply_settings(self):
        """Speichert Einstellungen in persistenter Config"""
        new_settings = self.get_settings()
        self.current_settings = new_settings

        # Speichern in kai_config.py
        cfg = get_config()
        cfg.update(new_settings)

        self.settings_changed.emit(new_settings)


# ============================================================================
# LOGGING SETTINGS TAB
# ============================================================================


class LoggingSettingsTab(QWidget):
    """Tab für Logging-Einstellungen (aus logging_ui.py übernommen)"""

    settings_changed = Signal(dict)

    LOG_LEVELS = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    def __init__(self, parent=None):
        super().__init__(parent)

        self.current_settings = {
            "console_level": "INFO",
            "file_level": "DEBUG",
            "performance_logging": True,
        }

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # === Konsolen-Logging ===
        console_group = QGroupBox("Konsolen-Logging")
        console_layout = QVBoxLayout()

        console_label = QLabel("Log-Level für Konsolen-Ausgabe:")
        self.console_level_combo = QComboBox()
        self.console_level_combo.addItems(self.LOG_LEVELS.keys())
        self.console_level_combo.setCurrentText(self.current_settings["console_level"])

        console_layout.addWidget(console_label)
        console_layout.addWidget(self.console_level_combo)
        console_group.setLayout(console_layout)

        # === Datei-Logging ===
        file_group = QGroupBox("Datei-Logging")
        file_layout = QVBoxLayout()

        file_label = QLabel("Log-Level für Datei-Ausgabe:")
        self.file_level_combo = QComboBox()
        self.file_level_combo.addItems(self.LOG_LEVELS.keys())
        self.file_level_combo.setCurrentText(self.current_settings["file_level"])

        file_info_label = QLabel(
            f"Log-Dateien werden gespeichert in:\n{LOG_DIR.absolute()}"
        )
        file_info_label.setStyleSheet(
            "color: #95a5a6; font-size: 11px; margin-top: 10px;"
        )
        file_info_label.setWordWrap(True)

        file_layout.addWidget(file_label)
        file_layout.addWidget(self.file_level_combo)
        file_layout.addWidget(file_info_label)
        file_group.setLayout(file_layout)

        # === Performance-Logging ===
        perf_group = QGroupBox("Performance-Logging")
        perf_layout = QVBoxLayout()

        self.perf_checkbox = QCheckBox("Performance-Metriken in separate Datei loggen")
        self.perf_checkbox.setChecked(self.current_settings["performance_logging"])

        perf_info_label = QLabel(
            "Performance-Logs enthalten Zeitstempel für kritische Operationen.\n"
            f"Datei: {PERFORMANCE_LOG_FILE.name}"
        )
        perf_info_label.setStyleSheet(
            "color: #95a5a6; font-size: 11px; margin-top: 5px;"
        )
        perf_info_label.setWordWrap(True)

        perf_layout.addWidget(self.perf_checkbox)
        perf_layout.addWidget(perf_info_label)
        perf_group.setLayout(perf_layout)

        # === Info ===
        info_group = QGroupBox("Hinweise")
        info_layout = QVBoxLayout()

        info_text = QLabel(
            "* Niedrigere Log-Levels zeigen mehr Details (DEBUG > INFO > WARNING > ERROR > CRITICAL)\n"
            "* Konsolen-Level: Ausgabe im Terminal während KAI läuft\n"
            "* Datei-Level: Persistente Logs für spätere Analyse\n"
            "* Änderungen werden beim Klick auf 'Anwenden' aktiv"
        )
        info_text.setWordWrap(True)
        info_text.setStyleSheet("color: #3498db; font-size: 11px;")

        info_layout.addWidget(info_text)
        info_group.setLayout(info_layout)

        # Layout zusammensetzen
        layout.addWidget(console_group)
        layout.addWidget(file_group)
        layout.addWidget(perf_group)
        layout.addWidget(info_group)
        layout.addStretch()

    def get_settings(self) -> dict:
        """Gibt aktuelle Einstellungen zurück"""
        return {
            "console_level": self.console_level_combo.currentText(),
            "file_level": self.file_level_combo.currentText(),
            "performance_logging": self.perf_checkbox.isChecked(),
        }

    def apply_settings(self):
        """Wendet Einstellungen an"""
        new_settings = self.get_settings()

        setup_logging(
            console_level=self.LOG_LEVELS[new_settings["console_level"]],
            file_level=self.LOG_LEVELS[new_settings["file_level"]],
            enable_performance_logging=new_settings["performance_logging"],
        )

        self.current_settings = new_settings
        self.settings_changed.emit(new_settings)


# ============================================================================
# NEO4J CONNECTION TAB
# ============================================================================


class Neo4jConnectionTab(QWidget):
    """Tab für Neo4j-Verbindungseinstellungen"""

    settings_changed = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)

        # Lade aktuelle Einstellungen
        cfg = get_config()
        self.current_settings = {
            "neo4j_uri": cfg.get("neo4j_uri", "bolt://127.0.0.1:7687"),
            "neo4j_user": cfg.get("neo4j_user", "neo4j"),
            "neo4j_password": cfg.get("neo4j_password", "password"),
        }

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # === Connection Parameters ===
        connection_group = QGroupBox("Verbindungsparameter")
        connection_layout = QVBoxLayout()

        # URI
        uri_label = QLabel("Neo4j URI:")
        self.uri_edit = QLineEdit()
        self.uri_edit.setText(self.current_settings["neo4j_uri"])
        self.uri_edit.setPlaceholderText("bolt://127.0.0.1:7687")
        self.uri_edit.setToolTip("Format: bolt://host:port oder neo4j://host:port")

        # User
        user_label = QLabel("Benutzername:")
        self.user_edit = QLineEdit()
        self.user_edit.setText(self.current_settings["neo4j_user"])
        self.user_edit.setPlaceholderText("neo4j")

        # Password
        password_label = QLabel("Passwort:")
        self.password_edit = QLineEdit()
        self.password_edit.setText(self.current_settings["neo4j_password"])
        self.password_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.password_edit.setPlaceholderText("password")

        # Show Password Checkbox
        self.show_password_checkbox = QCheckBox("Passwort anzeigen")
        self.show_password_checkbox.stateChanged.connect(
            self.toggle_password_visibility
        )

        connection_layout.addWidget(uri_label)
        connection_layout.addWidget(self.uri_edit)
        connection_layout.addWidget(user_label)
        connection_layout.addWidget(self.user_edit)
        connection_layout.addWidget(password_label)
        connection_layout.addWidget(self.password_edit)
        connection_layout.addWidget(self.show_password_checkbox)
        connection_group.setLayout(connection_layout)

        # === Hinweis ===
        info_group = QGroupBox("Hinweise")
        info_layout = QVBoxLayout()

        info_text = QLabel(
            "* Änderungen werden erst nach Neustart von KAI aktiv\n"
            "* Stelle sicher, dass Neo4j läuft und die Zugangsdaten korrekt sind\n"
            "* Standard-Port: 7687 (bolt protocol)\n"
            "* Die Verbindung wird beim Start von KAI getestet"
        )
        info_text.setWordWrap(True)
        info_text.setStyleSheet("color: #3498db; font-size: 11px;")

        info_layout.addWidget(info_text)
        info_group.setLayout(info_layout)

        # Layout zusammensetzen
        layout.addWidget(connection_group)
        layout.addWidget(info_group)
        layout.addStretch()

    def toggle_password_visibility(self, state):
        """Schaltet Passwort-Sichtbarkeit um"""
        if state == 2:  # Qt.CheckState.Checked
            self.password_edit.setEchoMode(QLineEdit.EchoMode.Normal)
        else:
            self.password_edit.setEchoMode(QLineEdit.EchoMode.Password)

    def get_settings(self) -> dict:
        """Gibt aktuelle Einstellungen zurück"""
        return {
            "neo4j_uri": self.uri_edit.text().strip(),
            "neo4j_user": self.user_edit.text().strip(),
            "neo4j_password": self.password_edit.text(),
        }

    def apply_settings(self):
        """Speichert Einstellungen"""
        new_settings = self.get_settings()
        self.current_settings = new_settings

        cfg = get_config()
        cfg.update(new_settings)

        self.settings_changed.emit(new_settings)


# ============================================================================
# CONFIDENCE THRESHOLDS TAB
# ============================================================================


class ConfidenceThresholdsTab(QWidget):
    """Tab für Konfidenz-Schwellenwerte (GoalPlanner)"""

    settings_changed = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)

        cfg = get_config()
        self.current_settings = {
            "confidence_low_threshold": cfg.get("confidence_low_threshold", 0.40),
            "confidence_medium_threshold": cfg.get("confidence_medium_threshold", 0.85),
        }

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # === Low Confidence Threshold ===
        low_group = QGroupBox("Niedrige Konfidenz (Rückfrage)")
        low_layout = QVBoxLayout()

        low_label = QLabel("Schwellenwert für Klärungsbedarf:")
        self.low_slider = QSlider(Qt.Orientation.Horizontal)
        self.low_slider.setMinimum(0)
        self.low_slider.setMaximum(100)
        self.low_slider.setValue(
            int(self.current_settings["confidence_low_threshold"] * 100)
        )
        self.low_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.low_slider.setTickInterval(10)

        self.low_value_label = QLabel(
            f"{self.current_settings['confidence_low_threshold']:.2f}"
        )
        self.low_value_label.setStyleSheet("font-weight: bold; color: #e74c3c;")
        self.low_slider.valueChanged.connect(
            lambda v: self.low_value_label.setText(f"{v/100:.2f}")
        )

        low_info = QLabel(
            "Unter diesem Wert fragt KAI nach, was gemeint ist.\n"
            "Beispiel: Bei 0.40 werden sehr unsichere Interpretationen hinterfragt."
        )
        low_info.setWordWrap(True)
        low_info.setStyleSheet("color: #95a5a6; font-size: 11px; margin-top: 5px;")

        low_layout.addWidget(low_label)
        low_layout.addWidget(self.low_slider)
        low_layout.addWidget(self.low_value_label)
        low_layout.addWidget(low_info)
        low_group.setLayout(low_layout)

        # === Medium Confidence Threshold ===
        medium_group = QGroupBox("Mittlere Konfidenz (Bestätigung)")
        medium_layout = QVBoxLayout()

        medium_label = QLabel("Schwellenwert für Bestätigungsanfrage:")
        self.medium_slider = QSlider(Qt.Orientation.Horizontal)
        self.medium_slider.setMinimum(0)
        self.medium_slider.setMaximum(100)
        self.medium_slider.setValue(
            int(self.current_settings["confidence_medium_threshold"] * 100)
        )
        self.medium_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.medium_slider.setTickInterval(10)

        self.medium_value_label = QLabel(
            f"{self.current_settings['confidence_medium_threshold']:.2f}"
        )
        self.medium_value_label.setStyleSheet("font-weight: bold; color: #f39c12;")
        self.medium_slider.valueChanged.connect(
            lambda v: self.medium_value_label.setText(f"{v/100:.2f}")
        )

        medium_info = QLabel(
            "Unter diesem Wert fragt KAI um Bestätigung.\n"
            "Über diesem Wert führt KAI die Aktion direkt aus.\n"
            "Beispiel: Bei 0.85 werden nur sehr sichere Interpretationen direkt ausgeführt."
        )
        medium_info.setWordWrap(True)
        medium_info.setStyleSheet("color: #95a5a6; font-size: 11px; margin-top: 5px;")

        medium_layout.addWidget(medium_label)
        medium_layout.addWidget(self.medium_slider)
        medium_layout.addWidget(self.medium_value_label)
        medium_layout.addWidget(medium_info)
        medium_group.setLayout(medium_layout)

        # === Beispiel ===
        example_group = QGroupBox("Beispiel")
        example_layout = QVBoxLayout()

        example_text = QLabel(
            "Eingabe: 'Ein Hund ist ein Tier'\n\n"
            "* Konfidenz: 0.92 -> Direkte Ausführung (≥ Medium Threshold)\n"
            "* Konfidenz: 0.78 -> Bestätigungsanfrage (< Medium Threshold)\n"
            "* Konfidenz: 0.35 -> Klärungsfrage (< Low Threshold)"
        )
        example_text.setWordWrap(True)
        example_text.setStyleSheet(
            "color: #3498db; font-size: 11px; font-family: 'Courier New', monospace;"
        )

        example_layout.addWidget(example_text)
        example_group.setLayout(example_layout)

        # Layout zusammensetzen
        layout.addWidget(low_group)
        layout.addWidget(medium_group)
        layout.addWidget(example_group)
        layout.addStretch()

    def get_settings(self) -> dict:
        """Gibt aktuelle Einstellungen zurück"""
        return {
            "confidence_low_threshold": self.low_slider.value() / 100,
            "confidence_medium_threshold": self.medium_slider.value() / 100,
        }

    def apply_settings(self):
        """Speichert Einstellungen"""
        new_settings = self.get_settings()
        self.current_settings = new_settings

        cfg = get_config()
        cfg.update(new_settings)

        self.settings_changed.emit(new_settings)


# ============================================================================
# PATTERN MATCHING TAB
# ============================================================================


class PatternMatchingTab(QWidget):
    """Tab für Pattern-Matching Thresholds"""

    settings_changed = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)

        cfg = get_config()
        self.current_settings = {
            "prototype_novelty_threshold": cfg.get("prototype_novelty_threshold", 15.0),
            "typo_min_threshold": cfg.get("typo_min_threshold", 3),
            "typo_max_threshold": cfg.get("typo_max_threshold", 10),
            "sequence_min_threshold": cfg.get("sequence_min_threshold", 2),
            "sequence_max_threshold": cfg.get("sequence_max_threshold", 5),
        }

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # === Prototype Novelty Threshold ===
        novelty_group = QGroupBox("Prototype Novelty Threshold")
        novelty_layout = QVBoxLayout()

        novelty_label = QLabel(
            "Schwellenwert für neue Prototypen (Euklidische Distanz):"
        )
        self.novelty_spinbox = QDoubleSpinBox()
        self.novelty_spinbox.setMinimum(5.0)
        self.novelty_spinbox.setMaximum(30.0)
        self.novelty_spinbox.setSingleStep(0.5)
        self.novelty_spinbox.setValue(
            self.current_settings["prototype_novelty_threshold"]
        )
        self.novelty_spinbox.setToolTip(
            "Distanz in 384D semantischem Raum.\n"
            "< Threshold: Update existierenden Prototyp\n"
            "≥ Threshold: Erstelle neuen Prototyp"
        )

        novelty_info = QLabel(
            "Bestimmt, wann ein Satz als 'neu genug' gilt, um einen eigenen Prototyp zu erhalten.\n"
            "Standard: 15.0 (ausgewogen zwischen Clustering und Granularität)"
        )
        novelty_info.setWordWrap(True)
        novelty_info.setStyleSheet("color: #95a5a6; font-size: 11px; margin-top: 5px;")

        novelty_layout.addWidget(novelty_label)
        novelty_layout.addWidget(self.novelty_spinbox)
        novelty_layout.addWidget(novelty_info)
        novelty_group.setLayout(novelty_layout)

        # === Typo Detection Thresholds ===
        typo_group = QGroupBox("Typo Detection (Adaptive)")
        typo_layout = QVBoxLayout()

        typo_min_label = QLabel("Minimum Wort-Vorkommen:")
        self.typo_min_spinbox = QSpinBox()
        self.typo_min_spinbox.setMinimum(1)
        self.typo_min_spinbox.setMaximum(10)
        self.typo_min_spinbox.setValue(self.current_settings["typo_min_threshold"])

        typo_max_label = QLabel("Maximum Wort-Vorkommen:")
        self.typo_max_spinbox = QSpinBox()
        self.typo_max_spinbox.setMinimum(5)
        self.typo_max_spinbox.setMaximum(20)
        self.typo_max_spinbox.setValue(self.current_settings["typo_max_threshold"])

        typo_info = QLabel(
            "Adaptive Thresholds basierend auf Vocabulary-Größe.\n"
            "Formel: min(MAX, max(MIN, vocab_size^0.4))\n"
            "Beispiel: Bei 100 Wörtern im Vocab -> Threshold = 4"
        )
        typo_info.setWordWrap(True)
        typo_info.setStyleSheet("color: #95a5a6; font-size: 11px; margin-top: 5px;")

        typo_layout.addWidget(typo_min_label)
        typo_layout.addWidget(self.typo_min_spinbox)
        typo_layout.addWidget(typo_max_label)
        typo_layout.addWidget(self.typo_max_spinbox)
        typo_layout.addWidget(typo_info)
        typo_group.setLayout(typo_layout)

        # === Sequence Prediction Thresholds ===
        sequence_group = QGroupBox("Sequence Prediction (Adaptive)")
        sequence_layout = QVBoxLayout()

        seq_min_label = QLabel("Minimum Sequenz-Vorkommen:")
        self.seq_min_spinbox = QSpinBox()
        self.seq_min_spinbox.setMinimum(1)
        self.seq_min_spinbox.setMaximum(5)
        self.seq_min_spinbox.setValue(self.current_settings["sequence_min_threshold"])

        seq_max_label = QLabel("Maximum Sequenz-Vorkommen:")
        self.seq_max_spinbox = QSpinBox()
        self.seq_max_spinbox.setMinimum(3)
        self.seq_max_spinbox.setMaximum(10)
        self.seq_max_spinbox.setValue(self.current_settings["sequence_max_threshold"])

        seq_info = QLabel(
            "Adaptive Thresholds basierend auf CONNECTION-Dichte.\n"
            "Formel: min(MAX, max(MIN, connection_count^0.35))\n"
            "Beispiel: Bei 100 Connections -> Threshold = 3"
        )
        seq_info.setWordWrap(True)
        seq_info.setStyleSheet("color: #95a5a6; font-size: 11px; margin-top: 5px;")

        sequence_layout.addWidget(seq_min_label)
        sequence_layout.addWidget(self.seq_min_spinbox)
        sequence_layout.addWidget(seq_max_label)
        sequence_layout.addWidget(self.seq_max_spinbox)
        sequence_layout.addWidget(seq_info)
        sequence_group.setLayout(sequence_layout)

        # Layout zusammensetzen
        layout.addWidget(novelty_group)
        layout.addWidget(typo_group)
        layout.addWidget(sequence_group)
        layout.addStretch()

    def get_settings(self) -> dict:
        """Gibt aktuelle Einstellungen zurück"""
        return {
            "prototype_novelty_threshold": self.novelty_spinbox.value(),
            "typo_min_threshold": self.typo_min_spinbox.value(),
            "typo_max_threshold": self.typo_max_spinbox.value(),
            "sequence_min_threshold": self.seq_min_spinbox.value(),
            "sequence_max_threshold": self.seq_max_spinbox.value(),
        }

    def apply_settings(self):
        """Speichert Einstellungen"""
        new_settings = self.get_settings()
        self.current_settings = new_settings

        cfg = get_config()
        cfg.update(new_settings)

        self.settings_changed.emit(new_settings)


# ============================================================================
# APPEARANCE TAB
# ============================================================================


class AppearanceTab(QWidget):
    """Tab für UI-Einstellungen (Theme, etc.)"""

    settings_changed = Signal(dict)
    theme_changed = Signal(str)  # Spezial-Signal für Theme-Änderung

    def __init__(self, parent=None):
        super().__init__(parent)

        cfg = get_config()
        self.current_settings = {
            "theme": cfg.get("theme", "dark"),
        }

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # === Theme Selection ===
        theme_group = QGroupBox("Farbschema")
        theme_layout = QVBoxLayout()

        theme_label = QLabel("Theme auswählen:")
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["dark", "light"])
        self.theme_combo.setCurrentText(self.current_settings["theme"])
        self.theme_combo.setToolTip("Wähle zwischen Dark Mode und Light Mode")

        theme_info = QLabel(
            "* Dark Mode: Dunkles Farbschema (Standard)\n"
            "* Light Mode: Helles Farbschema\n"
            "* Änderungen werden sofort nach 'Anwenden' aktiv"
        )
        theme_info.setWordWrap(True)
        theme_info.setStyleSheet("color: #95a5a6; font-size: 11px; margin-top: 5px;")

        theme_layout.addWidget(theme_label)
        theme_layout.addWidget(self.theme_combo)
        theme_layout.addWidget(theme_info)
        theme_group.setLayout(theme_layout)

        # === Vorschau ===
        preview_group = QGroupBox("Vorschau")
        preview_layout = QVBoxLayout()

        self.preview_label = QLabel("Dies ist ein Beispiel-Text im aktuellen Theme")
        self.preview_label.setStyleSheet(
            "padding: 20px; border: 2px solid #7f8c8d; border-radius: 5px; font-size: 14px;"
        )
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        preview_layout.addWidget(self.preview_label)
        preview_group.setLayout(preview_layout)

        # === Hinweis ===
        info_group = QGroupBox("Hinweise")
        info_layout = QVBoxLayout()

        info_text = QLabel(
            "* Das Theme wird beim nächsten Start von KAI vollständig angewendet\n"
            "* Einige UI-Elemente aktualisieren sich sofort nach 'Anwenden'"
        )
        info_text.setWordWrap(True)
        info_text.setStyleSheet("color: #3498db; font-size: 11px;")

        info_layout.addWidget(info_text)
        info_group.setLayout(info_layout)

        # Layout zusammensetzen
        layout.addWidget(theme_group)
        layout.addWidget(preview_group)
        layout.addWidget(info_group)
        layout.addStretch()

        # Connect theme combo to preview update
        self.theme_combo.currentTextChanged.connect(self.update_preview)
        self.update_preview(self.current_settings["theme"])

    def update_preview(self, theme: str):
        """Aktualisiert Vorschau basierend auf ausgewähltem Theme"""
        if theme == "dark":
            self.preview_label.setStyleSheet(
                "background-color: #2c3e50; color: #ecf0f1; "
                "padding: 20px; border: 2px solid #7f8c8d; border-radius: 5px; font-size: 14px;"
            )
        else:  # light
            self.preview_label.setStyleSheet(
                "background-color: #ecf0f1; color: #2c3e50; "
                "padding: 20px; border: 2px solid #7f8c8d; border-radius: 5px; font-size: 14px;"
            )

    def get_settings(self) -> dict:
        """Gibt aktuelle Einstellungen zurück"""
        return {
            "theme": self.theme_combo.currentText(),
        }

    def apply_settings(self):
        """Speichert Einstellungen"""
        new_settings = self.get_settings()
        old_theme = self.current_settings["theme"]
        new_theme = new_settings["theme"]

        self.current_settings = new_settings

        cfg = get_config()
        cfg.update(new_settings)

        self.settings_changed.emit(new_settings)

        # Emit theme_changed signal if theme was changed
        if old_theme != new_theme:
            self.theme_changed.emit(new_theme)


# ============================================================================
# TEST RUNNER TAB
# ============================================================================


class TestRunnerTab(QWidget):
    """Tab für Test-Ausführung mit Live-Progress-Anzeige und hierarchischer Ansicht"""

    # Verfügbare Test-Klassen
    # Format: (category, display_name, test_file, test_class)
    TEST_CLASSES = [
        # === System & Integration ===
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
        # === KAI Worker Extended Tests (Coverage) ===
        (
            "KaiWorker Initialization",
            "test_kai_worker_extended.py",
            "TestKaiWorkerInitialization",
        ),
        (
            "KaiWorker Command Suggestions",
            "test_kai_worker_extended.py",
            "TestKaiWorkerCommandSuggestions",
        ),
        (
            "KaiWorker Pattern Recognition",
            "test_kai_worker_extended.py",
            "TestKaiWorkerPatternRecognition",
        ),
        (
            "KaiWorker Context Handling",
            "test_kai_worker_extended.py",
            "TestKaiWorkerContextHandling",
        ),
        (
            "KaiWorker Execute Plan",
            "test_kai_worker_extended.py",
            "TestKaiWorkerExecutePlan",
        ),
        (
            "KaiWorker Helper Methods",
            "test_kai_worker_extended.py",
            "TestKaiWorkerHelperMethods",
        ),
        # === Basic Component Tests ===
        ("KonzeptNetzwerk (Basic)", "test_netzwerk_basic.py", "TestKonzeptNetzwerk"),
        (
            "KonzeptNetzwerk Core (Extended)",
            "test_netzwerk_extended.py",
            "TestKonzeptNetzwerkCore",
        ),
        (
            "KonzeptNetzwerk Patterns (Extended)",
            "test_netzwerk_extended.py",
            "TestKonzeptNetzwerkPatterns",
        ),
        (
            "KonzeptNetzwerk Memory (Extended)",
            "test_netzwerk_extended.py",
            "TestKonzeptNetzwerkMemory",
        ),
        (
            "KonzeptNetzwerk Word Usage (Extended)",
            "test_netzwerk_extended.py",
            "TestKonzeptNetzwerkWordUsage",
        ),
        (
            "KonzeptNetzwerk Feedback (Extended)",
            "test_netzwerk_extended.py",
            "TestKonzeptNetzwerkFeedback",
        ),
        (
            "KonzeptNetzwerk Edge Cases",
            "test_netzwerk_extended.py",
            "TestKonzeptNetzwerkEdgeCases",
        ),
        (
            "MeaningExtractor (Basic)",
            "test_meaning_extractor_basic.py",
            "TestMeaningExtractor",
        ),
        (
            "PrototypingEngine (Basic)",
            "test_prototype_matcher_basic.py",
            "TestPrototypingEngine",
        ),
        (
            "EpisodicMemory (Basic)",
            "test_episodic_memory_basic.py",
            "TestEpisodicMemory",
        ),
        ("EmbeddingService", "test_embedding_service.py", "TestEmbeddingService"),
        ("EnhancedNLP", "test_enhanced_nlp.py", "TestEnhancedNLP"),
        # === Additional Component Tests ===
        ("Netzwerk (Additional)", "test_netzwerk_additional.py", "TestErrorHandling"),
        ("Pattern Prototypes", "test_netzwerk_additional.py", "TestPatternPrototypes"),
        (
            "MeaningExtractor (Additional)",
            "test_meaning_extractor_additional.py",
            "TestDeclarativeStatements",
        ),
        (
            "PrototypeMatcher (Additional)",
            "test_prototype_matcher_additional.py",
            "TestPrototypingEngineEdgeCases",
        ),
        # === Feature Tests ===
        ("GoalPlanner", "test_goal_planner.py", "TestGoalPlanner"),
        ("MultiTurnDialogSystem", "test_dialog_system.py", "TestMultiTurnDialogSystem"),
        (
            "InteractiveLearning",
            "test_interactive_learning.py",
            "TestInteractiveLearning",
        ),
        (
            "IntelligentIngestion",
            "test_intelligent_ingestion.py",
            "TestIntelligentIngestion",
        ),
        ("WFragenVerarbeitung", "test_w_fragen.py", "TestWFragenVerarbeitung"),
        ("AllRelationTypes", "test_all_relation_types.py", "TestAllRelationTypes"),
        # === Reasoning Tests ===
        ("BackwardChaining", "test_backward_chaining.py", "TestBackwardChaining"),
        ("GraphTraversal", "test_graph_traversal.py", "TestTransitiveRelations"),
        ("WorkingMemory", "test_working_memory.py", "TestReasoningState"),
        (
            "AbductiveReasoning",
            "test_abductive_reasoning.py",
            "TestHypothesisGeneration",
        ),
        ("ProbabilisticEngine", "test_probabilistic_engine.py", "TestBeliefState"),
        # === Proof Explanation & Visualization ===
        ("ProofStep (Data)", "test_proof_explanation.py", "TestProofStep"),
        ("ProofTreeNode (Data)", "test_proof_explanation.py", "TestProofTreeNode"),
        ("ProofTree (Data)", "test_proof_explanation.py", "TestProofTree"),
        (
            "Explanation Generators",
            "test_proof_explanation.py",
            "TestExplanationGenerators",
        ),
        (
            "NL Generation (German)",
            "test_proof_explanation.py",
            "TestNaturalLanguageGeneration",
        ),
        ("Proof Helper Functions", "test_proof_explanation.py", "TestHelperFunctions"),
        ("Proof Integration", "test_proof_explanation.py", "TestIntegrationFunctions"),
        ("Proof Export/Import", "test_proof_explanation.py", "TestExportImport"),
        (
            "Proof Complex Scenarios",
            "test_proof_explanation.py",
            "TestComplexScenarios",
        ),
        ("ProofNodeItem (UI)", "test_proof_tree_widget.py", "TestProofNodeItem"),
        ("ProofEdgeItem (UI)", "test_proof_tree_widget.py", "TestProofEdgeItem"),
        ("ProofTreeWidget (UI)", "test_proof_tree_widget.py", "TestProofTreeWidget"),
        (
            "Tree Layout Algorithm",
            "test_proof_tree_widget.py",
            "TestTreeLayoutAlgorithm",
        ),
        ("Proof Performance", "test_proof_tree_widget.py", "TestPerformance"),
        ("Proof UI Integration", "test_proof_tree_widget.py", "TestIntegration"),
        # === Utilities & Infrastructure ===
        ("Linguistik Engine", "test_linguistik_engine.py", "TestResourceManager"),
        (
            "Linguistik Engine Preprocessor",
            "test_linguistik_engine.py",
            "TestLinguisticPreprocessor",
        ),
        (
            "Linguistik Engine Fallback",
            "test_linguistik_engine.py",
            "TestLinguisticPreprocessorFallback",
        ),
        ("Linguistik Integration", "test_linguistik_engine.py", "TestIntegration"),
        (
            "Exception Handling",
            "test_exception_handling.py",
            "TestKaiExceptionHandling",
        ),
        (
            "Text Normalization (Basic)",
            "test_text_normalization.py",
            "TestTextNormalizerWithoutSpacy",
        ),
        (
            "Text Normalization (With spaCy)",
            "test_text_normalization.py",
            "TestTextNormalizerWithSpacy",
        ),
        (
            "Text Convenience Functions",
            "test_text_normalization.py",
            "TestConvenienceFunctions",
        ),
        (
            "Text Normalization Regression",
            "test_text_normalization.py",
            "TestRegressionSuite",
        ),
        # === Word Usage Tracking ===
        ("Text Fragmentation", "test_word_usage_tracking.py", "TestTextFragmentation"),
        ("Word Usage Storage", "test_word_usage_tracking.py", "TestWordUsageStorage"),
        (
            "Fragment Similarity",
            "test_word_usage_tracking.py",
            "TestFragmentSimilarity",
        ),
        (
            "Word Usage Integration",
            "test_word_usage_tracking.py",
            "TestWordUsageIntegration",
        ),
        ("Config Integration", "test_word_usage_tracking.py", "TestConfigIntegration"),
        # === Pattern Recognition ===
        (
            "Keyboard Distance",
            "test_pattern_recognition_char.py",
            "TestKeyboardDistance",
        ),
        (
            "Weighted Levenshtein",
            "test_pattern_recognition_char.py",
            "TestWeightedLevenshtein",
        ),
        (
            "Typo Candidate Finder",
            "test_pattern_recognition_char.py",
            "TestTypoCandidateFinder",
        ),
        (
            "Pattern Feedback Integration",
            "test_pattern_recognition_char.py",
            "TestFeedbackIntegration",
        ),
        (
            "Sequence Predictor",
            "test_pattern_recognition_sequence.py",
            "TestSequencePredictor",
        ),
        # === Episodic Reasoning ===
        (
            "Inference Episode Creation",
            "test_episodic_reasoning.py",
            "TestInferenceEpisodeCreation",
        ),
        (
            "Proof Step Hierarchy",
            "test_episodic_reasoning.py",
            "TestProofStepHierarchy",
        ),
        ("Inference Tracking", "test_episodic_reasoning.py", "TestInferenceTracking"),
        ("Episodic Edge Cases", "test_episodic_reasoning.py", "TestEdgeCases"),
        # === Confidence Management ===
        (
            "Confidence Classification",
            "test_confidence_manager.py",
            "TestConfidenceClassification",
        ),
        ("Threshold Decisions", "test_confidence_manager.py", "TestThresholdDecisions"),
        (
            "Confidence Combination",
            "test_confidence_manager.py",
            "TestConfidenceCombination",
        ),
        ("Confidence Decay", "test_confidence_manager.py", "TestConfidenceDecay"),
        ("Specialized Methods", "test_confidence_manager.py", "TestSpecializedMethods"),
        ("Confidence Metrics", "test_confidence_manager.py", "TestConfidenceMetrics"),
        ("Confidence UI Feedback", "test_confidence_manager.py", "TestUIFeedback"),
        (
            "Confidence Global Instance",
            "test_confidence_manager.py",
            "TestGlobalInstance",
        ),
        ("Confidence Edge Cases", "test_confidence_manager.py", "TestEdgeCases"),
        # === Confidence Feedback ===
        (
            "Feedback Collection",
            "test_confidence_feedback.py",
            "TestFeedbackCollection",
        ),
        (
            "Confidence Adjustment",
            "test_confidence_feedback.py",
            "TestConfidenceAdjustment",
        ),
        (
            "Historical Consensus",
            "test_confidence_feedback.py",
            "TestHistoricalConsensus",
        ),
        (
            "Feedback Persistence",
            "test_confidence_feedback.py",
            "TestFeedbackPersistence",
        ),
        (
            "Statistics And Reporting",
            "test_confidence_feedback.py",
            "TestStatisticsAndReporting",
        ),
        (
            "Feedback Global Instance",
            "test_confidence_feedback.py",
            "TestGlobalInstance",
        ),
        ("Feedback Edge Cases", "test_confidence_feedback.py", "TestEdgeCases"),
        # === Advanced Exception Handling ===
        ("KAI Exception Base", "test_kai_exceptions.py", "TestKAIExceptionBase"),
        ("Database Exceptions", "test_kai_exceptions.py", "TestDatabaseExceptions"),
        ("Linguistic Exceptions", "test_kai_exceptions.py", "TestLinguisticExceptions"),
        ("Knowledge Exceptions", "test_kai_exceptions.py", "TestKnowledgeExceptions"),
        ("Reasoning Exceptions", "test_kai_exceptions.py", "TestReasoningExceptions"),
        ("Planning Exceptions", "test_kai_exceptions.py", "TestPlanningExceptions"),
        (
            "Configuration Exceptions",
            "test_kai_exceptions.py",
            "TestConfigurationExceptions",
        ),
        ("Exception Wrapping", "test_kai_exceptions.py", "TestExceptionWrapping"),
        ("Exception Hierarchy", "test_kai_exceptions.py", "TestExceptionHierarchy"),
        ("Exception Catching", "test_kai_exceptions.py", "TestExceptionCatching"),
        # === Logging & Exceptions Integration ===
        ("Logging System", "test_logging_and_exceptions.py", "TestLoggingSystem"),
        (
            "Exception Hierarchy (Logging)",
            "test_logging_and_exceptions.py",
            "TestExceptionHierarchy",
        ),
        (
            "Logging+Exceptions Integration",
            "test_logging_and_exceptions.py",
            "TestIntegrationLoggingAndExceptions",
        ),
        # === Plural & Definitions ===
        (
            "Plural Normalization",
            "test_plural_and_definitions.py",
            "TestPluralNormalization",
        ),
        (
            "Definition Storage",
            "test_plural_and_definitions.py",
            "TestDefinitionStorage",
        ),
        (
            "Integration Scenario",
            "test_plural_and_definitions.py",
            "TestIntegrationScenario",
        ),
        # === Extended Netzwerk Tests ===
        ("Lexical Triggers", "test_netzwerk_additional.py", "TestLexicalTriggers"),
        ("Other Queries", "test_netzwerk_additional.py", "TestOtherQueries"),
        (
            "Similar Word Finding",
            "test_netzwerk_additional.py",
            "TestSimilarWordFinding",
        ),
        (
            "Episodic Memory (Additional)",
            "test_netzwerk_additional.py",
            "TestEpisodicMemory",
        ),
        ("Inference Episodes", "test_netzwerk_additional.py", "TestInferenceEpisodes"),
        ("Hypothesis Storage", "test_netzwerk_additional.py", "TestHypothesisStorage"),
        (
            "Query Facts With Synonyms",
            "test_netzwerk_additional.py",
            "TestQueryFactsWithSynonyms",
        ),
        # === Extended Meaning Extractor Tests ===
        (
            "Question Heuristics",
            "test_meaning_extractor_additional.py",
            "TestQuestionHeuristics",
        ),
        ("Meaning Edge Cases", "test_meaning_extractor_additional.py", "TestEdgeCases"),
        (
            "Plural Normalization (Meaning)",
            "test_meaning_extractor_additional.py",
            "TestPluralNormalization",
        ),
        (
            "Explicit Commands",
            "test_meaning_extractor_additional.py",
            "TestExplicitCommands",
        ),
        (
            "Argument Extraction",
            "test_meaning_extractor_additional.py",
            "TestArgumentExtraction",
        ),
        # === Extended Prototype Matcher Tests ===
        (
            "Prototype Update",
            "test_prototype_matcher_additional.py",
            "TestPrototypeUpdate",
        ),
        (
            "Process Vector Scenarios",
            "test_prototype_matcher_additional.py",
            "TestProcessVectorScenarios",
        ),
        (
            "Find Best Match Scenarios",
            "test_prototype_matcher_additional.py",
            "TestFindBestMatchScenarios",
        ),
        # === Extended Working Memory Tests ===
        ("Context Frame", "test_working_memory.py", "TestContextFrame"),
        ("Working Memory", "test_working_memory.py", "TestWorkingMemory"),
        ("Working Memory Helpers", "test_working_memory.py", "TestHelperFunctions"),
        ("Working Memory Complex", "test_working_memory.py", "TestComplexScenarios"),
        # === Extended Graph Traversal Tests ===
        ("Path Finding", "test_graph_traversal.py", "TestPathFinding"),
        ("Multiple Paths", "test_graph_traversal.py", "TestMultiplePaths"),
        (
            "Inference Explanation",
            "test_graph_traversal.py",
            "TestInferenceExplanation",
        ),
        ("Concept Hierarchy", "test_graph_traversal.py", "TestConceptHierarchy"),
        ("Inverse Traversal", "test_graph_traversal.py", "TestInverseTraversal"),
        ("Graph Edge Cases", "test_graph_traversal.py", "TestEdgeCases"),
        ("German Explanations", "test_graph_traversal.py", "TestGermanExplanations"),
        (
            "Confidence Propagation",
            "test_graph_traversal.py",
            "TestConfidencePropagation",
        ),
        # === Extended Abductive Reasoning Tests ===
        (
            "Hypothesis Creation",
            "test_abductive_reasoning.py",
            "TestHypothesisCreation",
        ),
        (
            "Template Based Strategy",
            "test_abductive_reasoning.py",
            "TestTemplateBasedStrategy",
        ),
        (
            "Analogy Based Strategy",
            "test_abductive_reasoning.py",
            "TestAnalogyBasedStrategy",
        ),
        (
            "Causal Chain Strategy",
            "test_abductive_reasoning.py",
            "TestCausalChainStrategy",
        ),
        (
            "Multi Criteria Scoring",
            "test_abductive_reasoning.py",
            "TestMultiCriteriaScoring",
        ),
        (
            "Hypothesis Persistence",
            "test_abductive_reasoning.py",
            "TestHypothesisPersistence",
        ),
        (
            "Abductive Integration",
            "test_abductive_reasoning.py",
            "TestIntegrationWithKaiWorker",
        ),
        ("Abductive Edge Cases", "test_abductive_reasoning.py", "TestEdgeCases"),
        ("Abductive Performance", "test_abductive_reasoning.py", "TestPerformance"),
        # === Extended Probabilistic Engine Tests ===
        ("Probabilistic Fact", "test_probabilistic_engine.py", "TestProbabilisticFact"),
        (
            "Probabilistic Engine",
            "test_probabilistic_engine.py",
            "TestProbabilisticEngine",
        ),
        (
            "Probabilistic+Logic Integration",
            "test_probabilistic_engine.py",
            "TestIntegrationWithLogicEngine",
        ),
        ("Probabilistic Edge Cases", "test_probabilistic_engine.py", "TestEdgeCases"),
        # === Extended Exception Handling Tests ===
        (
            "Exception Hierarchy (Basic)",
            "test_exception_handling.py",
            "TestExceptionHierarchy",
        ),
        ("Wrap Exception", "test_exception_handling.py", "TestWrapException"),
        (
            "User Friendly Messages",
            "test_exception_handling.py",
            "TestUserFriendlyMessages",
        ),
        (
            "Exception Inheritance",
            "test_exception_handling.py",
            "TestExceptionInheritance",
        ),
        # === Autonomous Learning Tests ===
        ("Autonomous Learning (All)", "test_autonomous_learning.py", "test_"),
        ("Lerne Command (All)", "test_lerne_command.py", "test_"),
        ("Auto Detect Definitions (All)", "test_auto_detect_definitions.py", "test_"),
        ("Auto Detect E2E (All)", "test_auto_detect_e2e.py", "test_"),
        ("Definition Strategy (All)", "test_definition_strategy.py", "test_"),
        # === Pattern Recognition E2E ===
        ("Pattern Recognition E2E (All)", "test_pattern_recognition_e2e.py", "test_"),
        # === Performance & Caching ===
        ("Cache Performance (All)", "test_cache_performance.py", "test_"),
        # === Robustness & Quality Tests ===
        ("Contradiction Detection (All)", "test_contradiction_detection.py", "test_"),
        (
            "Error Paths & Resilience (All)",
            "test_error_paths_and_resilience.py",
            "test_",
        ),
        (
            "Property-Based Text Normalization (All)",
            "test_property_based_text_normalization.py",
            "test_",
        ),
        # === Episodic Query UI ===
        (
            "Episodic Query Recognition",
            "test_episodic_query_ui.py",
            "TestEpisodicQueryRecognition",
        ),
        (
            "Episodic Query Planning",
            "test_episodic_query_ui.py",
            "TestEpisodicQueryPlanning",
        ),
        (
            "Episodic Response Formatting",
            "test_episodic_query_ui.py",
            "TestEpisodicResponseFormatting",
        ),
        (
            "Episodic Memory Strategy",
            "test_episodic_query_ui.py",
            "TestEpisodicMemoryStrategy",
        ),
        # === Document Processing (File Ingestion) ===
        ("Document Parser", "test_document_parser.py", "TestDocumentParser"),
        (
            "File Reader Strategy",
            "test_file_reader_strategy.py",
            "TestFileReaderStrategy",
        ),
        (
            "Goal Planner File Reading",
            "test_goal_planner_file_reading.py",
            "TestGoalPlannerFileReading",
        ),
        (
            "Meaning Extractor File Commands",
            "test_meaning_extractor_file_commands.py",
            "TestMeaningExtractorFileCommands",
        ),
        ("File Ingestion E2E", "test_file_ingestion_e2e.py", "TestFileIngestionE2E"),
        # === Adaptive Pattern Recognition ===
        (
            "Adaptive Thresholds",
            "test_adaptive_pattern_recognition.py",
            "TestAdaptiveThresholds",
        ),
        ("Word Frequency", "test_adaptive_pattern_recognition.py", "TestWordFrequency"),
        (
            "Bayesian Pattern Quality",
            "test_adaptive_pattern_recognition.py",
            "TestBayesianPatternQuality",
        ),
        (
            "False Positive Reduction",
            "test_adaptive_pattern_recognition.py",
            "TestFalsePositiveReduction",
        ),
        (
            "Typo Feedback Recording",
            "test_adaptive_pattern_recognition.py",
            "TestTypoFeedbackRecording",
        ),
        # === Hybrid Reasoning ===
        (
            "Reasoning Orchestrator",
            "test_hybrid_reasoning.py",
            "TestReasoningOrchestrator",
        ),
        (
            "Hybrid Proof Tree Generation",
            "test_hybrid_reasoning.py",
            "TestHybridProofTreeGeneration",
        ),
        (
            "Inference Handler Integration",
            "test_hybrid_reasoning.py",
            "TestInferenceHandlerIntegration",
        ),
        ("Hybrid Reasoning Performance", "test_hybrid_reasoning.py", "TestPerformance"),
        ("Hybrid Reasoning Edge Cases", "test_hybrid_reasoning.py", "TestEdgeCases"),
        ("E2E Hybrid Reasoning", "test_hybrid_reasoning.py", "TestE2EHybridReasoning"),
        # === Hybrid Reasoning Phase 2 ===
        (
            "Aggregation Methods",
            "test_hybrid_reasoning_phase2.py",
            "TestAggregationMethods",
        ),
        (
            "YAML Configuration",
            "test_hybrid_reasoning_phase2.py",
            "TestYAMLConfiguration",
        ),
        ("Result Caching", "test_hybrid_reasoning_phase2.py", "TestResultCaching"),
        (
            "Parallel Execution",
            "test_hybrid_reasoning_phase2.py",
            "TestParallelExecution",
        ),
        (
            "Performance Optimizations",
            "test_hybrid_reasoning_phase2.py",
            "TestPerformanceOptimizations",
        ),
        ("Strategy Weights", "test_hybrid_reasoning_phase2.py", "TestStrategyWeights"),
        (
            "Phase2 Integration",
            "test_hybrid_reasoning_phase2.py",
            "TestPhase2Integration",
        ),
        # === Working Memory Stress Tests ===
        (
            "Deep Nested Contexts",
            "test_working_memory_stress.py",
            "TestDeepNestedContexts",
        ),
        (
            "Large Reasoning Traces",
            "test_working_memory_stress.py",
            "TestLargeReasoningTraces",
        ),
        (
            "Memory Leak Prevention",
            "test_working_memory_stress.py",
            "TestMemoryLeakPrevention",
        ),
        (
            "Complex Multi-Turn Scenarios",
            "test_working_memory_stress.py",
            "TestComplexMultiTurnScenarios",
        ),
        (
            "Context Summary And Export",
            "test_working_memory_stress.py",
            "TestContextSummaryAndExport",
        ),
        ("Idle Cleanup", "test_working_memory_stress.py", "TestIdleCleanup"),
        ("Export Import", "test_working_memory_stress.py", "TestExportImport"),
        # === Constraint Reasoning ===
        ("Constraint Variable", "test_constraint_reasoning.py", "TestVariable"),
        ("Constraint Definition", "test_constraint_reasoning.py", "TestConstraint"),
        ("Constraint Problem", "test_constraint_reasoning.py", "TestConstraintProblem"),
        ("Constraint Solver", "test_constraint_reasoning.py", "TestConstraintSolver"),
        ("N-Queens Problem", "test_constraint_reasoning.py", "TestNQueensProblem"),
        (
            "Graph Coloring Problem",
            "test_constraint_reasoning.py",
            "TestGraphColoringProblem",
        ),
        ("Logic Grid Puzzle", "test_constraint_reasoning.py", "TestLogicGridPuzzle"),
        (
            "Constraint ProofTree Integration",
            "test_constraint_reasoning.py",
            "TestProofTreeIntegration",
        ),
        (
            "Constraint Performance Metrics",
            "test_constraint_reasoning.py",
            "TestPerformanceMetrics",
        ),
        (
            "Constraint Logic Integration",
            "test_constraint_logic_integration.py",
            "TestConstraintLogicIntegration",
        ),
        (
            "Constraint Logic Edge Cases",
            "test_constraint_logic_integration.py",
            "TestConstraintLogicEdgeCases",
        ),
        # === SAT Solver ===
        ("SAT Literal", "test_sat_solver.py", "TestLiteral"),
        ("SAT Clause", "test_sat_solver.py", "TestClause"),
        ("SAT CNF Formula", "test_sat_solver.py", "TestCNFFormula"),
        ("SAT DPLL Solver", "test_sat_solver.py", "TestDPLLSolver"),
        ("SAT Encoder", "test_sat_solver.py", "TestSATEncoder"),
        ("SAT Knights and Knaves", "test_sat_solver.py", "TestKnightsAndKnaves"),
        (
            "SAT Knowledge Base Checker",
            "test_sat_solver.py",
            "TestKnowledgeBaseChecker",
        ),
        ("SAT Watched Literals", "test_sat_solver.py", "TestWatchedLiterals"),
        ("SAT Propositional Formula", "test_sat_solver.py", "TestPropositionalFormula"),
        ("SAT CNF Converter", "test_sat_solver.py", "TestCNFConverter"),
        ("SAT Solver Wrapper", "test_sat_solver.py", "TestSATSolverWrapper"),
        ("SAT Convenience Functions", "test_sat_solver.py", "TestConvenienceFunctions"),
        ("SAT Proof Generation", "test_sat_solver.py", "TestProofGeneration"),
        (
            "SAT Logic Engine Integration",
            "test_sat_solver.py",
            "TestIntegrationWithLogicEngine",
        ),
        ("SAT Solver Performance", "test_sat_solver.py", "TestPerformance"),
        # === SAT Consistency ===
        (
            "SAT Consistency Checking",
            "test_sat_consistency.py",
            "TestConsistencyChecking",
        ),
        (
            "SAT Contradiction Finding",
            "test_sat_consistency.py",
            "TestContradictionFinding",
        ),
        ("SAT Integration", "test_sat_consistency.py", "TestIntegrationWithSAT"),
        ("SAT Consistency Edge Cases", "test_sat_consistency.py", "TestEdgeCases"),
        # === SAT Reasoning ===
        (
            "SAT Reasoning Consistency",
            "test_sat_reasoning.py",
            "TestConsistencyChecking",
        ),
        (
            "SAT Contradiction Detection",
            "test_sat_reasoning.py",
            "TestContradictionDetection",
        ),
        (
            "SAT Knights and Knaves Reasoning",
            "test_sat_reasoning.py",
            "TestKnightsAndKnavesSAT",
        ),
        (
            "SAT Generic Capabilities",
            "test_sat_reasoning.py",
            "TestGenericSATCapabilities",
        ),
        (
            "SAT ProofTree Integration",
            "test_sat_reasoning.py",
            "TestProofTreeIntegration",
        ),
        (
            "SAT Reasoning Performance",
            "test_sat_reasoning.py",
            "TestPerformanceAndEdgeCases",
        ),
        # === State Space Planner ===
        (
            "State Representation",
            "test_state_space_planner.py",
            "TestStateRepresentation",
        ),
        ("Action Model", "test_state_space_planner.py", "TestActionModel"),
        ("Blocks World (Planner)", "test_state_space_planner.py", "TestBlocksWorld"),
        (
            "Grid Navigation (Planner)",
            "test_state_space_planner.py",
            "TestGridNavigation",
        ),
        (
            "River Crossing (Planner)",
            "test_state_space_planner.py",
            "TestRiverCrossing",
        ),
        (
            "Diagnostic Reasoning",
            "test_state_space_planner.py",
            "TestDiagnosticReasoning",
        ),
        ("Planning Heuristics", "test_state_space_planner.py", "TestHeuristics"),
        ("Temporal Reasoning", "test_state_space_planner.py", "TestTemporalReasoning"),
        ("Plan Simulation", "test_state_space_planner.py", "TestPlanSimulation"),
        ("State Space Performance", "test_state_space_planner.py", "TestPerformance"),
        ("Hybrid Planning", "test_state_space_planner.py", "TestHybridPlanning"),
        ("State Space Integration", "test_state_space_planner.py", "TestIntegration"),
        # === State Reasoning ===
        ("Blocks World (State)", "test_state_reasoning.py", "TestBlocksWorld"),
        ("Grid Navigation (State)", "test_state_reasoning.py", "TestGridNavigation"),
        ("River Crossing (State)", "test_state_reasoning.py", "TestRiverCrossing"),
        (
            "State Reasoning Integration 31",
            "test_state_reasoning.py",
            "TestStateReasoningIntegration31",
        ),
        (
            "State Reasoning Performance 31",
            "test_state_reasoning.py",
            "TestPerformance31",
        ),
        ("State Object", "test_state_reasoning.py", "TestState"),
        ("Action Object", "test_state_reasoning.py", "TestAction"),
        (
            "State Space Planner Object",
            "test_state_reasoning.py",
            "TestStateSpacePlanner",
        ),
        ("BFS Planning", "test_state_reasoning.py", "TestBFSPlanning"),
        (
            "Planning Utility Functions",
            "test_state_reasoning.py",
            "TestUtilityFunctions",
        ),
        (
            "State Constraint Integration",
            "test_state_reasoning.py",
            "TestConstraintIntegration",
        ),
        ("Complex Planning", "test_state_reasoning.py", "TestComplexPlanning"),
        # === Performance Optimizations ===
        (
            "Batch Embeddings",
            "test_performance_optimizations.py",
            "TestBatchEmbeddings",
        ),
        (
            "Ingestion Batch Processing",
            "test_performance_optimizations.py",
            "TestIngestionBatchProcessing",
        ),
        (
            "Proof Tree Lazy Loading",
            "test_performance_optimizations.py",
            "TestProofTreeLazyLoading",
        ),
        (
            "Neo4j Query Profiler",
            "test_performance_optimizations.py",
            "TestNeo4jQueryProfiler",
        ),
        (
            "Performance Regression",
            "test_performance_optimizations.py",
            "TestPerformanceRegression",
        ),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.test_runner_thread = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # === Test-Auswahl ===
        selection_group = QGroupBox("Test-Auswahl")
        selection_layout = QVBoxLayout()

        # Alle/Keine Buttons
        button_row = QHBoxLayout()
        select_all_btn = QPushButton("Alle auswählen")
        select_all_btn.clicked.connect(self.select_all_tests)
        select_none_btn = QPushButton("Keine auswählen")
        select_none_btn.clicked.connect(self.select_no_tests)
        button_row.addWidget(select_all_btn)
        button_row.addWidget(select_none_btn)
        button_row.addStretch()

        # Checkboxen für Test-Klassen
        self.test_checkboxes: Dict[tuple, QCheckBox] = {}
        self.file_checkboxes: Dict[str, QCheckBox] = {}  # Für Datei-Header
        checkbox_container = QWidget()
        checkbox_layout = QVBoxLayout(checkbox_container)

        current_file = None
        current_file_tests = []  # Sammelt alle Test-Checkboxen für die aktuelle Datei

        for entry in self.TEST_CLASSES:
            # Handle both 3-tuple and 4-tuple formats
            if len(entry) == 4:
                category, display_name, test_file, test_class = entry
            else:
                display_name, test_file, test_class = entry
            # Add file separator when switching to new test file
            if test_file != current_file:
                if current_file is not None:
                    # Add separator line
                    separator = QFrame()
                    separator.setFrameShape(QFrame.Shape.HLine)
                    separator.setFrameShadow(QFrame.Shadow.Sunken)
                    separator.setStyleSheet("color: #7f8c8d;")
                    checkbox_layout.addWidget(separator)

                # Add file checkbox (anklickbar zum Auswählen aller Tests in dieser Datei)
                file_cb = QCheckBox(f"📁 {test_file}")
                file_cb.setStyleSheet(
                    "font-weight: bold; color: #3498db; margin-top: 5px;"
                )
                file_cb.setChecked(True)
                self.file_checkboxes[test_file] = file_cb
                checkbox_layout.addWidget(file_cb)
                current_file = test_file
                current_file_tests = []  # Reset für neue Datei

                # Connect file checkbox to toggle all tests in this file
                file_cb.stateChanged.connect(
                    lambda state, tf=test_file: self._toggle_file_tests(tf, state)
                )

            cb = QCheckBox(f"    {display_name}")  # Einrückung für Test-Einträge
            cb.setChecked(True)  # Standardmäßig alle ausgewählt
            # Store the tuple as key for later retrieval
            self.test_checkboxes[(test_file, test_class)] = cb
            checkbox_layout.addWidget(cb)

            # Connect test checkbox to update file checkbox state
            cb.stateChanged.connect(
                lambda state, tf=test_file: self._update_file_checkbox(tf)
            )

        # Scrollable Area für Checkboxen
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(checkbox_container)
        scroll_area.setMaximumHeight(200)

        selection_layout.addLayout(button_row)
        selection_layout.addWidget(scroll_area)
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

    def select_all_tests(self):
        """Wählt alle Tests aus"""
        # Blockiere Signale während Batch-Update
        for cb in self.test_checkboxes.values():
            cb.blockSignals(True)
            cb.setChecked(True)
            cb.blockSignals(False)

        # Update file checkboxes
        for file_cb in self.file_checkboxes.values():
            file_cb.blockSignals(True)
            file_cb.setChecked(True)
            file_cb.blockSignals(False)

    def select_no_tests(self):
        """Wählt keine Tests aus"""
        # Blockiere Signale während Batch-Update
        for cb in self.test_checkboxes.values():
            cb.blockSignals(True)
            cb.setChecked(False)
            cb.blockSignals(False)

        # Update file checkboxes
        for file_cb in self.file_checkboxes.values():
            file_cb.blockSignals(True)
            file_cb.setChecked(False)
            file_cb.blockSignals(False)

    def get_selected_tests(self) -> List[tuple]:
        """Gibt Liste der ausgewählten Test-Klassen zurück als (test_file, test_class) Tupel"""
        return [
            (test_file, test_class)
            for (test_file, test_class), cb in self.test_checkboxes.items()
            if cb.isChecked()
        ]

    def _toggle_file_tests(self, test_file: str, state: int):
        """Schaltet alle Tests in einer Datei an/aus"""
        is_checked = state == 2  # Qt.CheckState.Checked = 2

        # Blockiere Signale während des Batch-Updates um Rekursion zu vermeiden
        for (tf, tc), cb in self.test_checkboxes.items():
            if tf == test_file:
                cb.blockSignals(True)
                cb.setChecked(is_checked)
                cb.blockSignals(False)

    def _update_file_checkbox(self, test_file: str):
        """Aktualisiert den Datei-Checkbox basierend auf den Test-Checkboxen"""
        # Finde alle Tests für diese Datei
        file_tests = [
            cb for (tf, tc), cb in self.test_checkboxes.items() if tf == test_file
        ]

        if not file_tests:
            return

        # Zähle checked Tests
        checked_count = sum(1 for cb in file_tests if cb.isChecked())

        # Update file checkbox
        file_cb = self.file_checkboxes.get(test_file)
        if file_cb:
            file_cb.blockSignals(True)  # Verhindere Rekursion
            if checked_count == 0:
                file_cb.setChecked(False)
            elif checked_count == len(file_tests):
                file_cb.setChecked(True)
            else:
                # Teilweise ausgewählt - setze auf checked (PySide6 hat kein PartiallyChecked für normale Checkboxen)
                file_cb.setChecked(True)
            file_cb.blockSignals(False)

    def run_tests(self):
        """Startet Test-Ausführung"""
        selected = self.get_selected_tests()

        if not selected:
            self.progress_label.setText("⚠ Keine Tests ausgewählt")
            return

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
        pass  # Wird im Progress-Update angezeigt

    @Slot(str, str)
    def on_test_failed(self, test_name: str, error_msg: str):
        """Wird aufgerufen wenn ein Test fehlschlägt"""
        item = QListWidgetItem(f"[ERROR] {test_name}")
        item.setToolTip(error_msg)
        self.failed_tests_list.addItem(item)

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
        # Wir wissen die Gesamtzahl nicht vorher - berechne basierend auf aktuellen Werten
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
            self.progress_label.setText(f"[SUCCESS] Alle {passed} Tests erfolgreich!")
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


# ============================================================================
# HAUPTDIALOG MIT TABS
# ============================================================================


class SettingsDialog(QDialog):
    """
    Zentraler Einstellungen-Dialog mit mehreren Tabs:
    - Logging
    - Tests
    """

    settings_changed = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Einstellungen")
        self.setModal(True)
        self.setMinimumSize(700, 600)

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # === Tab Widget ===
        self.tab_widget = QTabWidget()

        # KAI Settings Tab
        self.kai_tab = KaiSettingsTab()
        self.tab_widget.addTab(self.kai_tab, "⚙️ KAI-Einstellungen")

        # Neo4j Connection Tab
        self.neo4j_tab = Neo4jConnectionTab()
        self.tab_widget.addTab(self.neo4j_tab, "🗄️ Neo4j")

        # Confidence Thresholds Tab
        self.confidence_tab = ConfidenceThresholdsTab()
        self.tab_widget.addTab(self.confidence_tab, "🎯 Konfidenz")

        # Pattern Matching Tab
        self.pattern_tab = PatternMatchingTab()
        self.tab_widget.addTab(self.pattern_tab, "🔍 Muster")

        # Appearance Tab
        self.appearance_tab = AppearanceTab()
        self.tab_widget.addTab(self.appearance_tab, "🎨 Darstellung")

        # Logging Tab
        self.logging_tab = LoggingSettingsTab()
        self.tab_widget.addTab(self.logging_tab, "📝 Logging")

        # Test Runner Tab
        self.test_tab = TestRunnerTab()
        self.tab_widget.addTab(self.test_tab, "🧪 Tests")

        # === Buttons ===
        button_layout = QHBoxLayout()

        self.apply_button = QPushButton("Anwenden")
        self.apply_button.clicked.connect(self.apply_settings)
        self.apply_button.setStyleSheet("background-color: #27ae60; font-weight: bold;")

        self.close_button = QPushButton("Schließen")
        self.close_button.clicked.connect(self.accept)

        button_layout.addStretch()
        button_layout.addWidget(self.apply_button)
        button_layout.addWidget(self.close_button)

        # === Layout ===
        layout.addWidget(self.tab_widget)
        layout.addLayout(button_layout)

    def apply_settings(self):
        """Wendet Einstellungen aus dem aktiven Tab an"""
        current_tab = self.tab_widget.currentWidget()

        if current_tab == self.kai_tab:
            self.kai_tab.apply_settings()
            self.settings_changed.emit(self.kai_tab.get_settings())
        elif current_tab == self.neo4j_tab:
            self.neo4j_tab.apply_settings()
            self.settings_changed.emit(self.neo4j_tab.get_settings())
        elif current_tab == self.confidence_tab:
            self.confidence_tab.apply_settings()
            self.settings_changed.emit(self.confidence_tab.get_settings())
        elif current_tab == self.pattern_tab:
            self.pattern_tab.apply_settings()
            self.settings_changed.emit(self.pattern_tab.get_settings())
        elif current_tab == self.appearance_tab:
            self.appearance_tab.apply_settings()
            self.settings_changed.emit(self.appearance_tab.get_settings())
        elif current_tab == self.logging_tab:
            self.logging_tab.apply_settings()
            self.settings_changed.emit(self.logging_tab.get_settings())


# === Convenience-Funktion ===


def show_settings_dialog(parent=None):
    """Zeigt den Einstellungen-Dialog"""
    dialog = SettingsDialog(parent)
    return dialog.exec()


if __name__ == "__main__":
    """Test-Code"""
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)

    # Dark Theme (wie in main_ui_graphical.py)
    app.setStyleSheet(
        """
        QWidget {
            background-color: #34495e;
            color: #ecf0f1;
            font-size: 14px;
        }
        QGroupBox {
            font-weight: bold;
            border: 1px solid #7f8c8d;
            border-radius: 5px;
            margin-top: 10px;
            padding-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top center;
            padding: 0 5px;
        }
    """
    )

    show_settings_dialog()

    sys.exit(0)
