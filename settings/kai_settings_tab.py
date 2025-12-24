"""
kai_settings_tab.py

Tab for general KAI settings (word usage tracking, similarity thresholds, etc.)
"""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from kai_config import get_config


class KaiSettingsTab(QWidget):
    """Tab fuer allgemeine KAI-Einstellungen (Wortverwendungs-Tracking, etc.)"""

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
            "Speichert Kontext-Fragmente fuer jedes Wort, um spaeter Muster zu erkennen.\n"
            "Dies hilft KAI, typische Wortverbindungen zu lernen."
        )

        usage_info_label = QLabel(
            "Wenn aktiviert, speichert KAI bei jedem 'Lerne:' und 'Ingestiere Text:' Befehl "
            "typische Wortverbindungen mit Haeufigkeits-Zaehlern. Dies ermoeglicht spaetere Mustererkennung."
        )
        usage_info_label.setWordWrap(True)
        usage_info_label.setStyleSheet(
            "color: #95a5a6; font-size: 11px; margin-top: 5px;"
        )

        usage_layout.addWidget(self.usage_tracking_checkbox)
        usage_layout.addWidget(usage_info_label)
        usage_group.setLayout(usage_layout)

        # === Aehnlichkeits-Schwellenwert ===
        similarity_group = QGroupBox("Aehnlichkeits-Schwellenwert")
        similarity_layout = QVBoxLayout()

        similarity_label = QLabel("Schwellenwert fuer aehnliche Kontext-Fragmente (%):")
        self.similarity_combo = QComboBox()
        self.similarity_combo.addItems(["70", "75", "80", "85", "90", "95", "100"])
        self.similarity_combo.setCurrentText(
            str(self.current_settings["usage_similarity_threshold"])
        )
        self.similarity_combo.setToolTip(
            "Bei welcher Aehnlichkeit sollen beide Counter erhoeht werden?\n"
            "80% = 1-2 Woerter Unterschied OK, 100% = nur exakte Uebereinstimmung"
        )

        similarity_info_label = QLabel(
            "Bestimmt, ab wann zwei Kontext-Fragmente als 'aehnlich genug' gelten.\n"
            "Beispiel bei 80%: 'im grossen Haus' = 'im grossen alten Haus' (beide Counter werden erhoeht)"
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
        window_label = QLabel("Kontext-Fenster-Groesse (+/-N Woerter):")
        self.window_combo = QComboBox()
        self.window_combo.addItems(["1", "2", "3", "4", "5"])
        self.window_combo.setCurrentText(
            str(self.current_settings["context_window_size"])
        )
        self.window_combo.setToolTip(
            "Wie viele Woerter vor und nach dem Zielwort sollen gespeichert werden?"
        )

        # Max Words to Comma
        comma_label = QLabel("Max. Woerter bis Komma:")
        self.comma_combo = QComboBox()
        self.comma_combo.addItems(["2", "3", "4", "5", "6"])
        self.comma_combo.setCurrentText(
            str(self.current_settings["max_words_to_comma"])
        )
        self.comma_combo.setToolTip(
            "Wenn weniger als N Woerter bis zum Komma, wird bis Komma gespeichert.\n"
            "Sonst wird auf +/-N Fenster begrenzt."
        )

        context_info_label = QLabel(
            "Kontext-Fragmente werden entweder bis zum naechsten Komma ODER +/-N Woerter gespeichert.\n"
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
            "Beispiel-Satz: 'Das Haus steht im grossen Park.'\n\n"
            "Fuer 'grossen' wird gespeichert:\n"
            "* CONNECTION Edges: 'im' -> 'grossen' (distance=1, count++)\n"
            "                    'grossen' -> 'Park' (distance=1, count++)\n"
            "* UsageContext: 'im grossen Park' (+/-3 Fenster, word_position=1, count++)\n\n"
            "Bei erneutem Auftreten von 'im grossen Park' -> count wird erhoeht statt neuen Node zu erstellen."
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
        """Gibt aktuelle Einstellungen zurueck"""
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
