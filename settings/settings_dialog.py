"""
settings_dialog.py

Main settings dialog that combines all settings tabs into a unified interface.
"""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
)

from settings.kai_settings_tab import KaiSettingsTab
from settings.logging_settings_tab import LoggingSettingsTab
from settings.neo4j_connection_tab import Neo4jConnectionTab
from settings.confidence_thresholds_tab import ConfidenceThresholdsTab
from settings.pattern_matching_tab import PatternMatchingTab
from settings.production_system_tab import ProductionSystemTab
from settings.appearance_tab import AppearanceTab
from test_runner_ui import TestRunnerTab


class SettingsDialog(QDialog):
    """
    Zentraler Einstellungen-Dialog mit mehreren Tabs:
    - KAI-Einstellungen
    - Neo4j
    - Konfidenz
    - Muster
    - Darstellung
    - Logging
    - Tests
    - Production System
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
        self.tab_widget.addTab(self.kai_tab, "KAI-Einstellungen")

        # Neo4j Connection Tab
        self.neo4j_tab = Neo4jConnectionTab()
        self.tab_widget.addTab(self.neo4j_tab, "Neo4j")

        # Confidence Thresholds Tab
        self.confidence_tab = ConfidenceThresholdsTab()
        self.tab_widget.addTab(self.confidence_tab, "Konfidenz")

        # Pattern Matching Tab
        self.pattern_tab = PatternMatchingTab()
        self.tab_widget.addTab(self.pattern_tab, "Muster")

        # Appearance Tab
        self.appearance_tab = AppearanceTab()
        self.tab_widget.addTab(self.appearance_tab, "Darstellung")

        # Logging Tab
        self.logging_tab = LoggingSettingsTab()
        self.tab_widget.addTab(self.logging_tab, "Logging")

        # Test Runner Tab
        self.test_tab = TestRunnerTab()
        self.tab_widget.addTab(self.test_tab, "Tests")

        # Production System Tab (PHASE 8.1)
        self.production_tab = ProductionSystemTab()
        self.tab_widget.addTab(self.production_tab, "Production System")

        # === Buttons ===
        button_layout = QHBoxLayout()

        self.apply_button = QPushButton("Anwenden")
        self.apply_button.clicked.connect(self.apply_settings)
        self.apply_button.setStyleSheet("background-color: #27ae60; font-weight: bold;")

        self.close_button = QPushButton("Schliessen")
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
