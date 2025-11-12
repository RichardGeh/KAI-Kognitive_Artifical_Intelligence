"""
settings_ui.py

Vereinheitlichter Einstellungen-Dialog für KAI mit mehreren Tabs:
- Logging-Einstellungen
- Test-Runner mit Progress-Anzeige

Ersetzt den alten "Logging"-Menü-Eintrag durch "Einstellungen".
"""

import logging
import sys

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from component_15_logging_config import LOG_DIR, PERFORMANCE_LOG_FILE, setup_logging
from kai_config import get_config
from test_runner_ui import TestRunnerTab


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
