"""
appearance_tab.py

Tab for UI appearance settings (theme selection, etc.)
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from kai_config import get_config


class AppearanceTab(QWidget):
    """Tab fuer UI-Einstellungen (Theme, etc.)"""

    settings_changed = Signal(dict)
    theme_changed = Signal(str)  # Spezial-Signal fuer Theme-Aenderung

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

        theme_label = QLabel("Theme auswaehlen:")
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["dark", "light"])
        self.theme_combo.setCurrentText(self.current_settings["theme"])
        self.theme_combo.setToolTip("Waehle zwischen Dark Mode und Light Mode")

        theme_info = QLabel(
            "* Dark Mode: Dunkles Farbschema (Standard)\n"
            "* Light Mode: Helles Farbschema\n"
            "* Aenderungen werden sofort nach 'Anwenden' aktiv"
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
            "* Das Theme wird beim naechsten Start von KAI vollstaendig angewendet\n"
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
        """Aktualisiert Vorschau basierend auf ausgewaehltem Theme"""
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
        """Gibt aktuelle Einstellungen zurueck"""
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
