"""
pattern_matching_tab.py

Tab for pattern matching threshold settings (prototype novelty, typo detection, sequence prediction).
"""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QGroupBox,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from kai_config import get_config


class PatternMatchingTab(QWidget):
    """Tab fuer Pattern-Matching Thresholds"""

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
            "Schwellenwert fuer neue Prototypen (Euklidische Distanz):"
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
            ">= Threshold: Erstelle neuen Prototyp"
        )

        novelty_info = QLabel(
            "Bestimmt, wann ein Satz als 'neu genug' gilt, um einen eigenen Prototyp zu erhalten.\n"
            "Standard: 15.0 (ausgewogen zwischen Clustering und Granularitaet)"
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
            "Adaptive Thresholds basierend auf Vocabulary-Groesse.\n"
            "Formel: min(MAX, max(MIN, vocab_size^0.4))\n"
            "Beispiel: Bei 100 Woertern im Vocab -> Threshold = 4"
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
        """Gibt aktuelle Einstellungen zurueck"""
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
