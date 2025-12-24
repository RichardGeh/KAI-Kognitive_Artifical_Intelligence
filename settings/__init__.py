"""
Settings UI Package - Modular settings dialog components.

This package provides a modular, maintainable structure for the KAI settings UI.
Each tab is implemented in its own module for better code organization.

Modules:
- kai_settings_tab: Word usage tracking settings
- logging_settings_tab: Logging configuration
- neo4j_connection_tab: Neo4j connection settings
- confidence_thresholds_tab: Confidence thresholds for GoalPlanner
- pattern_matching_tab: Pattern matching threshold settings
- production_system_tab: Production rules visualization (PHASE 8.1/9)
- appearance_tab: UI appearance/theme settings
- settings_dialog: Main settings dialog with all tabs
"""

from settings.kai_settings_tab import KaiSettingsTab
from settings.logging_settings_tab import LoggingSettingsTab
from settings.neo4j_connection_tab import Neo4jConnectionTab
from settings.confidence_thresholds_tab import ConfidenceThresholdsTab
from settings.pattern_matching_tab import PatternMatchingTab
from settings.production_system_tab import ProductionSystemTab
from settings.appearance_tab import AppearanceTab
from settings.settings_dialog import SettingsDialog, show_settings_dialog

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
