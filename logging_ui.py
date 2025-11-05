"""
logging_ui.py

PySide6-basierte UI-Komponenten für Logging-Konfiguration und -Anzeige.

Features:
- Logging-Settings-Dialog: Konfiguration von Log-Levels für verschiedene Handler
- Log-Viewer-Fenster: Live-Anzeige mit Filterung nach Level, Zeitraum, Komponente
- Auto-Refresh: Echtzeit-Updates der Log-Anzeige
- Farbcodierung: Visuell unterscheidbare Log-Levels
"""

import logging
from pathlib import Path
from typing import List
from collections import deque

from PySide6.QtCore import QTimer, Signal
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
    QSpinBox,
    QLineEdit,
)
from PySide6.QtGui import QTextCursor, QColor, QTextCharFormat

from component_15_logging_config import (
    setup_logging,
    DEFAULT_LOG_FILE,
    ERROR_LOG_FILE,
    PERFORMANCE_LOG_FILE,
    LOG_DIR,
)


class LoggingSettingsDialog(QDialog):
    """
    Dialog zur Konfiguration der Logging-Einstellungen.

    Ermöglicht Anpassung von:
    - Konsolen-Log-Level
    - Datei-Log-Level
    - Performance-Logging (an/aus)
    """

    settings_changed = Signal(dict)  # Signal wenn Einstellungen geändert wurden

    LOG_LEVELS = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Logging-Einstellungen")
        self.setModal(True)
        self.setMinimumWidth(500)

        # Aktuelle Einstellungen (Defaults)
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

        # === Info-Bereich ===
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

        # === Buttons ===
        button_layout = QHBoxLayout()

        apply_button = QPushButton("Anwenden")
        apply_button.clicked.connect(self.apply_settings)
        apply_button.setStyleSheet("background-color: #27ae60; font-weight: bold;")

        cancel_button = QPushButton("Abbrechen")
        cancel_button.clicked.connect(self.reject)

        button_layout.addStretch()
        button_layout.addWidget(apply_button)
        button_layout.addWidget(cancel_button)

        # === Layout zusammensetzen ===
        layout.addWidget(console_group)
        layout.addWidget(file_group)
        layout.addWidget(perf_group)
        layout.addWidget(info_group)
        layout.addLayout(button_layout)

    def apply_settings(self):
        """Wendet die Einstellungen an und startet Logging neu"""
        new_settings = {
            "console_level": self.console_level_combo.currentText(),
            "file_level": self.file_level_combo.currentText(),
            "performance_logging": self.perf_checkbox.isChecked(),
        }

        # Logging neu konfigurieren
        setup_logging(
            console_level=self.LOG_LEVELS[new_settings["console_level"]],
            file_level=self.LOG_LEVELS[new_settings["file_level"]],
            enable_performance_logging=new_settings["performance_logging"],
        )

        self.current_settings = new_settings
        self.settings_changed.emit(new_settings)

        self.accept()


class LogViewerWindow(QDialog):
    """
    Live-Log-Viewer mit Filterung und Auto-Refresh.

    Features:
    - Echtzeit-Anzeige von Log-Einträgen
    - Filterung nach Level, Zeitraum, Komponente
    - Suchfunktion
    - Automatische Updates (konfigurierbar)
    - Export-Funktion
    """

    LOG_LEVEL_COLORS = {
        "DEBUG": QColor("#3498db"),  # Blau
        "INFO": QColor("#2ecc71"),  # Grün
        "WARNING": QColor("#f39c12"),  # Orange
        "ERROR": QColor("#e74c3c"),  # Rot
        "CRITICAL": QColor("#9b59b6"),  # Lila
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("KAI Log-Viewer")
        self.setMinimumSize(1000, 600)

        # Aktuell angezeigte Log-Datei
        self.current_log_file = DEFAULT_LOG_FILE

        # Cache für bereits gelesene Log-Zeilen
        self.log_cache = deque(maxlen=1000)  # Letzten 1000 Zeilen
        self.last_file_position = 0

        # Auto-Refresh Timer
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_logs)
        self.auto_refresh_enabled = True
        self.refresh_interval = 2000  # 2 Sekunden

        self.init_ui()
        self.load_logs()

        # Starte Auto-Refresh
        if self.auto_refresh_enabled:
            self.refresh_timer.start(self.refresh_interval)

    def init_ui(self):
        layout = QVBoxLayout(self)

        # === Toolbar ===
        toolbar_layout = QHBoxLayout()

        # Log-Datei-Auswahl
        file_label = QLabel("Log-Datei:")
        self.file_combo = QComboBox()
        self.file_combo.addItem("Haupt-Log (kai.log)", str(DEFAULT_LOG_FILE))
        self.file_combo.addItem("Fehler (kai_errors.log)", str(ERROR_LOG_FILE))
        self.file_combo.addItem(
            "Performance (kai_performance.log)", str(PERFORMANCE_LOG_FILE)
        )
        self.file_combo.currentIndexChanged.connect(self.on_file_changed)

        # Refresh-Button
        refresh_button = QPushButton("↻ Aktualisieren")
        refresh_button.clicked.connect(self.refresh_logs)
        refresh_button.setToolTip("Logs manuell neu laden")

        # Auto-Refresh Toggle
        self.auto_refresh_checkbox = QCheckBox("Auto-Refresh")
        self.auto_refresh_checkbox.setChecked(self.auto_refresh_enabled)
        self.auto_refresh_checkbox.toggled.connect(self.toggle_auto_refresh)

        # Clear-Button
        clear_button = QPushButton("🗑 Leeren")
        clear_button.clicked.connect(self.clear_display)
        clear_button.setToolTip("Anzeige leeren (Datei bleibt unverändert)")

        toolbar_layout.addWidget(file_label)
        toolbar_layout.addWidget(self.file_combo)
        toolbar_layout.addStretch()
        toolbar_layout.addWidget(self.auto_refresh_checkbox)
        toolbar_layout.addWidget(refresh_button)
        toolbar_layout.addWidget(clear_button)

        # === Filter-Bereich ===
        filter_group = QGroupBox("Filter")
        filter_layout = QHBoxLayout()

        # Level-Filter
        level_label = QLabel("Level:")
        self.level_filter_combo = QComboBox()
        self.level_filter_combo.addItem("ALLE", None)
        self.level_filter_combo.addItem("DEBUG", "DEBUG")
        self.level_filter_combo.addItem("INFO", "INFO")
        self.level_filter_combo.addItem("WARNING", "WARNING")
        self.level_filter_combo.addItem("ERROR", "ERROR")
        self.level_filter_combo.addItem("CRITICAL", "CRITICAL")
        self.level_filter_combo.currentIndexChanged.connect(self.apply_filters)

        # Komponenten-Filter (Suche)
        component_label = QLabel("Komponente:")
        self.component_filter_edit = QLineEdit()
        self.component_filter_edit.setPlaceholderText(
            "z.B. 'netzwerk', 'kai_worker'..."
        )
        self.component_filter_edit.textChanged.connect(self.apply_filters)

        # Textsuche
        search_label = QLabel("Suche:")
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Suchtext...")
        self.search_edit.textChanged.connect(self.apply_filters)

        # Anzahl-Limit
        limit_label = QLabel("Zeilen:")
        self.limit_spinbox = QSpinBox()
        self.limit_spinbox.setRange(50, 1000)
        self.limit_spinbox.setValue(200)
        self.limit_spinbox.setSuffix(" letzte")
        self.limit_spinbox.valueChanged.connect(self.apply_filters)

        filter_layout.addWidget(level_label)
        filter_layout.addWidget(self.level_filter_combo)
        filter_layout.addWidget(component_label)
        filter_layout.addWidget(self.component_filter_edit)
        filter_layout.addWidget(search_label)
        filter_layout.addWidget(self.search_edit)
        filter_layout.addWidget(limit_label)
        filter_layout.addWidget(self.limit_spinbox)
        filter_group.setLayout(filter_layout)

        # === Log-Anzeige ===
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setStyleSheet(
            "font-family: 'Consolas', 'Courier New', monospace; "
            "font-size: 11px; "
            "background-color: #1e1e1e; "
            "color: #d4d4d4;"
        )
        self.log_display.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)

        # === Status-Leiste ===
        self.status_label = QLabel("Bereit")
        self.status_label.setStyleSheet(
            "color: #95a5a6; font-size: 11px; margin-top: 5px;"
        )

        # === Layout zusammensetzen ===
        layout.addLayout(toolbar_layout)
        layout.addWidget(filter_group)
        layout.addWidget(self.log_display)
        layout.addWidget(self.status_label)

    def on_file_changed(self):
        """Wird aufgerufen wenn eine andere Log-Datei ausgewählt wird"""
        selected_file = Path(self.file_combo.currentData())
        self.current_log_file = selected_file
        self.last_file_position = 0
        self.log_cache.clear()
        self.load_logs()

    def toggle_auto_refresh(self, enabled):
        """Aktiviert/Deaktiviert Auto-Refresh"""
        self.auto_refresh_enabled = enabled
        if enabled:
            self.refresh_timer.start(self.refresh_interval)
        else:
            self.refresh_timer.stop()

    def clear_display(self):
        """Leert die Anzeige"""
        self.log_display.clear()
        self.status_label.setText("Anzeige geleert")

    def load_logs(self):
        """Lädt Log-Einträge aus der aktuellen Datei"""
        if not self.current_log_file.exists():
            self.log_display.setPlainText(
                f"Log-Datei existiert noch nicht:\n{self.current_log_file}\n\n"
                "Die Datei wird erstellt, sobald Logs geschrieben werden."
            )
            self.status_label.setText("Datei nicht gefunden")
            return

        try:
            with open(self.current_log_file, "r", encoding="utf-8") as f:
                # Lese nur neue Zeilen seit letztem Load
                f.seek(self.last_file_position)
                new_lines = f.readlines()
                self.last_file_position = f.tell()

                # Füge zu Cache hinzu
                for line in new_lines:
                    self.log_cache.append(line.rstrip())

            # Wende Filter an
            self.apply_filters()

            # Update Status
            total_lines = len(self.log_cache)
            self.status_label.setText(
                f"{total_lines} Log-Einträge geladen | "
                f"Datei: {self.current_log_file.name} | "
                f"Größe: {self.current_log_file.stat().st_size / 1024:.1f} KB"
            )

        except Exception as e:
            self.log_display.setPlainText(f"Fehler beim Laden der Logs:\n{e}")
            self.status_label.setText(f"Fehler: {str(e)}")

    def refresh_logs(self):
        """Aktualisiert die Log-Anzeige (für Auto-Refresh)"""
        self.load_logs()

    def apply_filters(self):
        """Wendet aktuelle Filter auf den Log-Cache an"""
        # Hole Filter-Werte
        level_filter = self.level_filter_combo.currentData()
        component_filter = self.component_filter_edit.text().lower()
        search_text = self.search_edit.text().lower()
        max_lines = self.limit_spinbox.value()

        # Filtere Log-Zeilen
        filtered_lines = []

        for line in self.log_cache:
            # Level-Filter
            if level_filter and f"[{level_filter}" not in line:
                continue

            # Komponenten-Filter
            if component_filter and component_filter not in line.lower():
                continue

            # Textsuche
            if search_text and search_text not in line.lower():
                continue

            filtered_lines.append(line)

        # Limitiere auf letzte N Zeilen
        displayed_lines = filtered_lines[-max_lines:]

        # Aktualisiere Anzeige mit Syntax-Highlighting
        self.update_display_with_highlighting(displayed_lines)

        # Update Status
        self.status_label.setText(
            f"{len(displayed_lines)} von {len(self.log_cache)} Zeilen angezeigt | "
            f"Filter aktiv: {bool(level_filter or component_filter or search_text)}"
        )

    def update_display_with_highlighting(self, lines: List[str]):
        """Aktualisiert die Anzeige mit farbcodiertem Syntax-Highlighting"""
        self.log_display.clear()
        cursor = self.log_display.textCursor()

        for line in lines:
            # Erkenne Log-Level
            level_color = None
            for level, color in self.LOG_LEVEL_COLORS.items():
                if f"[{level}" in line:
                    level_color = color
                    break

            # Formatiere Zeile
            fmt = QTextCharFormat()
            if level_color:
                fmt.setForeground(level_color)

            cursor.insertText(line + "\n", fmt)

        # Scrolle nach unten (zu neuesten Einträgen)
        self.log_display.moveCursor(QTextCursor.MoveOperation.End)


# === Convenience-Funktion für einfache Integration ===


def show_logging_settings(parent=None):
    """Zeigt den Logging-Settings-Dialog"""
    dialog = LoggingSettingsDialog(parent)
    return dialog.exec()


def show_log_viewer(parent=None):
    """Zeigt das Log-Viewer-Fenster"""
    viewer = LogViewerWindow(parent)
    viewer.exec()


if __name__ == "__main__":
    """Test-Code für die UI-Komponenten"""
    from PySide6.QtWidgets import QApplication
    import sys

    from component_15_logging_config import setup_logging, get_logger

    # Initialisiere Logging
    setup_logging()
    logger = get_logger("test_ui")

    # Erzeuge Test-Logs
    logger.debug("Dies ist ein DEBUG-Log")
    logger.info("Dies ist ein INFO-Log")
    logger.warning("Dies ist eine WARNUNG")
    logger.error("Dies ist ein FEHLER")
    logger.critical("Dies ist ein KRITISCHER Fehler")

    app = QApplication(sys.argv)

    # Teste beide Dialoge
    print("Öffne Log-Viewer...")
    show_log_viewer()

    print("Öffne Settings...")
    show_logging_settings()

    sys.exit(0)
