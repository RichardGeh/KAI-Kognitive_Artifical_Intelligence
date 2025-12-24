"""
settings_ui.py

Backward Compatibility Shim for Settings UI.

All components have been moved to the settings/ package for better modularity.
This file re-exports all components for backward compatibility.

Usage:
    # Both of these work:
    from settings_ui import SettingsDialog, show_settings_dialog
    from settings import SettingsDialog, show_settings_dialog

Structure:
    settings/
        __init__.py              - Package facade
        kai_settings_tab.py      - Word usage tracking settings
        logging_settings_tab.py  - Logging configuration
        neo4j_connection_tab.py  - Neo4j connection settings
        confidence_thresholds_tab.py - Confidence thresholds
        pattern_matching_tab.py  - Pattern matching settings
        production_system_tab.py - Production rules visualization
        appearance_tab.py        - UI appearance settings
        settings_dialog.py       - Main settings dialog
"""

import sys

from settings import (
    KaiSettingsTab,
    LoggingSettingsTab,
    Neo4jConnectionTab,
    ConfidenceThresholdsTab,
    PatternMatchingTab,
    ProductionSystemTab,
    AppearanceTab,
    SettingsDialog,
    show_settings_dialog,
)

__all__ = [
    "KaiSettingsTab",
    "LoggingSettingsTab",
    "Neo4jConnectionTab",
    "ConfidenceThresholdsTab",
    "PatternMatchingTab",
    "ProductionSystemTab",
    "AppearanceTab",
    "SettingsDialog",
    "show_settings_dialog",
]


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
