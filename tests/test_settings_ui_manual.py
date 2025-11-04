"""
Manual test script for enhanced Settings UI.

Tests:
1. Neo4j Connection Tab
2. Confidence Thresholds Tab
3. Pattern Matching Tab
4. Appearance Tab (Theme switching)
5. All existing tabs (KAI Settings, Logging, Tests)

Run this script to verify all tabs are working correctly.
"""

import sys
from PyQt6.QtWidgets import QApplication
from settings_ui import SettingsDialog
from kai_config import get_config


def test_settings_dialog():
    """Test the enhanced Settings Dialog"""
    app = QApplication(sys.argv)

    # Apply dark theme for testing
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

    # Show current config
    cfg = get_config()
    print("=== Current Configuration ===")
    print(f"Theme: {cfg.get('theme', 'dark')}")
    print(f"Neo4j URI: {cfg.get('neo4j_uri', 'N/A')}")
    print(f"Neo4j User: {cfg.get('neo4j_user', 'N/A')}")
    print(f"Confidence Low Threshold: {cfg.get('confidence_low_threshold', 'N/A')}")
    print(
        f"Confidence Medium Threshold: {cfg.get('confidence_medium_threshold', 'N/A')}"
    )
    print(
        f"Prototype Novelty Threshold: {cfg.get('prototype_novelty_threshold', 'N/A')}"
    )
    print(f"Typo Min Threshold: {cfg.get('typo_min_threshold', 'N/A')}")
    print(f"Typo Max Threshold: {cfg.get('typo_max_threshold', 'N/A')}")
    print()

    # Create and show dialog
    dialog = SettingsDialog()

    # Connect signals for testing
    def on_settings_changed(settings):
        print(f"\n[SIGNAL] Settings changed: {settings}")

    def on_theme_changed(theme):
        print(f"\n[SIGNAL] Theme changed to: {theme}")

    dialog.settings_changed.connect(on_settings_changed)
    if hasattr(dialog, "appearance_tab"):
        dialog.appearance_tab.theme_changed.connect(on_theme_changed)

    # Show dialog
    print("\n=== Testing Settings Dialog ===")
    print("Please test the following:")
    print("1. Navigate through all tabs")
    print("2. Change some settings")
    print("3. Click 'Anwenden' to save")
    print("4. Verify signals are emitted correctly")
    print("\nDialog opening...")

    result = dialog.exec()

    print(f"\nDialog closed with result: {result}")

    # Show updated config
    print("\n=== Updated Configuration ===")
    print(f"Theme: {cfg.get('theme', 'dark')}")
    print(f"Neo4j URI: {cfg.get('neo4j_uri', 'N/A')}")
    print(f"Neo4j User: {cfg.get('neo4j_user', 'N/A')}")
    print(f"Confidence Low Threshold: {cfg.get('confidence_low_threshold', 'N/A')}")
    print(
        f"Confidence Medium Threshold: {cfg.get('confidence_medium_threshold', 'N/A')}"
    )
    print(
        f"Prototype Novelty Threshold: {cfg.get('prototype_novelty_threshold', 'N/A')}"
    )
    print(f"Typo Min Threshold: {cfg.get('typo_min_threshold', 'N/A')}")
    print(f"Typo Max Threshold: {cfg.get('typo_max_threshold', 'N/A')}")

    sys.exit(0)


if __name__ == "__main__":
    test_settings_dialog()
