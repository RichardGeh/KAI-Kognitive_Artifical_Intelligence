"""
logging_settings_tab.py

Tab for logging configuration (console level, file level, performance logging).
"""

import logging

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from component_15_logging_config import LOG_DIR, PERFORMANCE_LOG_FILE, setup_logging


class LoggingSettingsTab(QWidget):
    """Tab fuer Logging-Einstellungen (aus logging_ui.py uebernommen)"""

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

        console_label = QLabel("Log-Level fuer Konsolen-Ausgabe:")
        self.console_level_combo = QComboBox()
        self.console_level_combo.addItems(self.LOG_LEVELS.keys())
        self.console_level_combo.setCurrentText(self.current_settings["console_level"])

        console_layout.addWidget(console_label)
        console_layout.addWidget(self.console_level_combo)
        console_group.setLayout(console_layout)

        # === Datei-Logging ===
        file_group = QGroupBox("Datei-Logging")
        file_layout = QVBoxLayout()

        file_label = QLabel("Log-Level fuer Datei-Ausgabe:")
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
            "Performance-Logs enthalten Zeitstempel fuer kritische Operationen.\n"
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
            "* Konsolen-Level: Ausgabe im Terminal waehrend KAI laeuft\n"
            "* Datei-Level: Persistente Logs fuer spaetere Analyse\n"
            "* Aenderungen werden beim Klick auf 'Anwenden' aktiv"
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
        """Gibt aktuelle Einstellungen zurueck"""
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
