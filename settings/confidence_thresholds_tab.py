"""
confidence_thresholds_tab.py

Tab for confidence threshold settings (GoalPlanner thresholds).
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QGroupBox,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from kai_config import get_config


class ConfidenceThresholdsTab(QWidget):
    """Tab fuer Konfidenz-Schwellenwerte (GoalPlanner)"""

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
        low_group = QGroupBox("Niedrige Konfidenz (Rueckfrage)")
        low_layout = QVBoxLayout()

        low_label = QLabel("Schwellenwert fuer Klaerungsbedarf:")
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
        medium_group = QGroupBox("Mittlere Konfidenz (Bestaetigung)")
        medium_layout = QVBoxLayout()

        medium_label = QLabel("Schwellenwert fuer Bestaetigungsanfrage:")
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
            "Unter diesem Wert fragt KAI um Bestaetigung.\n"
            "Ueber diesem Wert fuehrt KAI die Aktion direkt aus.\n"
            "Beispiel: Bei 0.85 werden nur sehr sichere Interpretationen direkt ausgefuehrt."
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
            "* Konfidenz: 0.92 -> Direkte Ausfuehrung (>= Medium Threshold)\n"
            "* Konfidenz: 0.78 -> Bestaetigungsanfrage (< Medium Threshold)\n"
            "* Konfidenz: 0.35 -> Klaerungsfrage (< Low Threshold)"
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
        """Gibt aktuelle Einstellungen zurueck"""
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
