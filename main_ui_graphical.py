# main_ui_graphical.py
import logging
import sys
from typing import Dict

from PySide6.QtCore import Qt, QThread, QTimer, Slot
from PySide6.QtGui import QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# WICHTIG: Encoding-Fix MUSS früh importiert werden
# Behebt Windows cp1252 -> UTF-8 Probleme für Unicode-Zeichen ([OK], [X], ->, etc.)
import kai_encoding_fix  # noqa: F401 (automatische Aktivierung beim Import)
from component_1_netzwerk import KonzeptNetzwerk
from component_11_embedding_service import EmbeddingService
from component_15_logging_config import setup_logging
from kai_config import get_config
from kai_exceptions import KAIException, get_user_friendly_message
from kai_worker import KaiWorker
from logging_ui import LogViewerWindow
from settings_ui import SettingsDialog
from test_runner_ui import TestRunnerWindow

logger = logging.getLogger(__name__)

# PHASE 3.4 (User Feedback Loop)
try:
    from component_51_feedback_handler import FeedbackHandler, FeedbackType

    FEEDBACK_HANDLER_AVAILABLE = True
except ImportError:
    FEEDBACK_HANDLER_AVAILABLE = False
    print("[UI] FeedbackHandler nicht verfügbar - Feedback-Loop deaktiviert")

# Import ProofTreeWidget (with fallback)
try:
    from component_18_proof_tree_widget import ProofTreeWidget

    PROOF_TREE_AVAILABLE = True
except ImportError:
    PROOF_TREE_AVAILABLE = False
    print("[UI] ProofTreeWidget nicht verfügbar - Beweisbaum-Tab wird nicht angezeigt")

# Import EpisodicMemoryWidget (with fallback)
try:
    from component_19_episodic_memory_widget import EpisodicMemoryWidget

    EPISODIC_MEMORY_WIDGET_AVAILABLE = True
except ImportError:
    EPISODIC_MEMORY_WIDGET_AVAILABLE = False
    print(
        "[UI] EpisodicMemoryWidget nicht verfügbar - Episodisches-Gedächtnis-Tab wird nicht angezeigt"
    )

# Import ContextVisualizationWidget (with fallback)
try:
    from context_visualization_widget import ContextVisualizationWidget

    CONTEXT_VISUALIZATION_AVAILABLE = True
except ImportError:
    CONTEXT_VISUALIZATION_AVAILABLE = False
    print(
        "[UI] ContextVisualizationWidget nicht verfügbar - Kontext-Tracker-Tab wird nicht angezeigt"
    )

# Import SpatialGridWidget (with fallback)
try:
    from component_43_spatial_grid_widget import SpatialGridWidget

    SPATIAL_GRID_WIDGET_AVAILABLE = True
except ImportError:
    SPATIAL_GRID_WIDGET_AVAILABLE = False
    print(
        "[UI] SpatialGridWidget nicht verfügbar - Räumliches-Grid-Tab wird nicht angezeigt"
    )

# Import ResonanceViewWidget (with fallback)
try:
    from component_45_resonance_view_widget import ResonanceViewWidget

    RESONANCE_VIEW_AVAILABLE = True
except ImportError:
    RESONANCE_VIEW_AVAILABLE = False
    print(
        "[UI] ResonanceViewWidget nicht verfügbar - Resonance-View-Tab wird nicht angezeigt"
    )


class PlanMonitor(QWidget):
    """Zeigt das Hauptziel und eine dynamische Liste von Unterzielen an."""

    STATUS_ICONS = {
        "PENDING": "[○]",
        "IN_PROGRESS": "[●]",
        "SUCCESS": "[[OK]]",
        "FAILED": "[[X]]",
    }
    STATUS_COLORS = {
        "PENDING": "#95a5a6",  # Grau
        "IN_PROGRESS": "#3498db",  # Blau
        "SUCCESS": "#2ecc71",  # Grün
        "FAILED": "#e74c3c",  # Rot
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.sub_goal_labels: Dict[str, QLabel] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.main_goal_label = QLabel("Warte auf Aufgabe...")
        self.main_goal_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_goal_label.setStyleSheet(
            """
            background-color: #2c3e50; color: #ecf0f1; padding: 10px;
            font-size: 16px; font-weight: bold; border-radius: 5px;
        """
        )

        sub_goal_widget = QWidget()
        self.sub_goal_layout = QVBoxLayout(sub_goal_widget)
        self.sub_goal_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(sub_goal_widget)
        self.scroll_area.setVisible(True)  # Immer sichtbar

        layout.addWidget(self.main_goal_label)
        layout.addWidget(self.scroll_area)

    @Slot()
    def clear_goals(self):
        """Entfernt alle alten Ziele von der Anzeige."""
        self.main_goal_label.setText("Warte auf Aufgabe...")
        for label in self.sub_goal_labels.values():
            label.deleteLater()
        self.sub_goal_labels.clear()

    @Slot(str)
    def set_main_goal(self, text: str):
        """Setzt den Text des Hauptziels."""
        self.main_goal_label.setText(text)

    @Slot(str, str)
    def add_sub_goal(self, sg_id: str, description: str):
        """Fügt ein neues Unterziel in der Anzeige hinzu."""
        if sg_id in self.sub_goal_labels:
            return

        label = QLabel(f"{self.STATUS_ICONS['PENDING']} {description}")
        label.setStyleSheet(
            f"color: {self.STATUS_COLORS['PENDING']}; "
            "padding: 5px; font-size: 12px; border-bottom: 1px solid #34495e;"
        )
        self.sub_goal_layout.addWidget(label)
        self.sub_goal_labels[sg_id] = label

    @Slot(str, str)
    def update_sub_goal_status(self, sg_id: str, status_name: str):
        """Aktualisiert den Status (Farbe, Icon) eines bestehenden Unterziels."""
        if sg_id not in self.sub_goal_labels:
            return

        label = self.sub_goal_labels[sg_id]
        # Bewahre die alte Beschreibung, ohne das alte Icon
        current_text = label.text().split(" ", 1)[-1]

        icon = self.STATUS_ICONS.get(status_name, "[?]")
        color = self.STATUS_COLORS.get(status_name, "#ecf0f1")

        label.setText(f"{icon} {current_text}")
        label.setStyleSheet(
            f"color: {color}; padding: 5px; font-size: 12px; "
            "border-bottom: 1px solid #34495e;"
        )


class TaskDisplay(QWidget):
    """Zeigt das Hauptziel und eine ausklappbare Liste von Unterzielen an."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.main_goal_label = QLabel("Warte auf Aufgabe...")
        self.main_goal_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_goal_label.setStyleSheet(
            """
            background-color: #2c3e50;
            color: #ecf0f1;
            padding: 10px;
            font-size: 16px;
            font-weight: bold;
            border-radius: 5px;
            cursor: pointer;
        """
        )

        self.sub_goal_container = QFrame()
        self.sub_goal_container.setObjectName("subGoalContainer")
        self.sub_goal_container.setStyleSheet(
            "#subGoalContainer { border: 1px solid #34495e; border-top: none; }"
        )
        self.sub_goal_layout = QVBoxLayout(self.sub_goal_container)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.sub_goal_container)
        self.scroll_area.setMaximumHeight(120)
        self.scroll_area.setVisible(False)

        layout.addWidget(self.main_goal_label)
        layout.addWidget(self.scroll_area)

        self.main_goal_label.mousePressEvent = self.toggle_sub_goals

    def toggle_sub_goals(self, event):
        self.scroll_area.setVisible(not self.scroll_area.isVisible())

    def update_main_goal(self, text):
        self.main_goal_label.setText(text)

    def add_sub_goal(self, text):
        MAX_SUB_GOALS = 20
        if self.sub_goal_layout.count() >= MAX_SUB_GOALS:
            # Entferne ältestes Goal
            child = self.sub_goal_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        """ANGEPASST: Formatiert die Anzeige der Ziele basierend auf ihrer Priorität."""
        label = QLabel(text)
        style = "padding: 5px; font-size: 12px; border-bottom: 1px solid #34495e;"

        if "[MUST]" in text:
            style += "font-weight: bold; color: #f1c40f;"  # Gelb für MUST
        elif "[SHOULD]" in text:
            style += "color: #ecf0f1;"  # Standardfarbe für SHOULD
        elif "[COULD]" in text:
            style += "color: #95a5a6;"  # Grau für COULD

        label.setStyleSheet(style)
        self.sub_goal_layout.addWidget(label)

    def clear_goals(self):
        self.update_main_goal("Warte auf Aufgabe...")
        while self.sub_goal_layout.count():
            child = self.sub_goal_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.scroll_area.setVisible(False)


class ChatInterface(QWidget):
    """Der Chat-Bereich am unteren Rand des Fensters."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        self.history = QTextEdit()
        self.history.setReadOnly(True)

        # PHASE 8 (Extended Features): Progress Bar für Datei-Ingestion
        self.file_progress_bar = QProgressBar()
        self.file_progress_bar.setVisible(False)
        self.file_progress_bar.setStyleSheet(
            """
            QProgressBar {
                border: 2px solid #34495e;
                border-radius: 5px;
                text-align: center;
                background-color: #2c3e50;
                color: #ecf0f1;
            }
            QProgressBar::chunk {
                background-color: #3498db;
                border-radius: 3px;
            }
        """
        )
        self.file_progress_label = QLabel("")
        self.file_progress_label.setVisible(False)
        self.file_progress_label.setStyleSheet(
            "color: #3498db; font-size: 11px; padding: 2px;"
        )

        # PHASE 2 (Multi-Turn): Context-Anzeige oberhalb der Eingabezeile
        self.context_label = QLabel("")
        self.context_label.setStyleSheet(
            """
            background-color: #2c3e50;
            color: #f39c12;
            padding: 5px;
            border-radius: 3px;
            font-size: 11px;
            font-style: italic;
        """
        )
        self.context_label.setVisible(False)  # Initial versteckt
        self.context_label.setWordWrap(True)

        # Input-Bereich: Mehrzeiliges Textfeld statt einzeilig
        self.input_text = QTextEdit()
        self.input_text.setPlaceholderText(
            "Stelle eine Frage an KAI... (Strg+Enter zum Senden)"
        )
        self.input_text.setMaximumHeight(100)  # Begrenzt auf ~4 Zeilen
        self.input_text.setMinimumHeight(60)  # Mindestens ~2 Zeilen
        self.input_text.setStyleSheet(
            """
            QTextEdit {
                background-color: #2c3e50;
                border: 2px solid #7f8c8d;
                border-radius: 4px;
                padding: 8px;
                font-size: 14px;
            }
            QTextEdit:focus {
                border: 2px solid #3498db;
            }
        """
        )

        # Bottom controls layout (Datei-Button + Checkbox)
        controls_layout = QHBoxLayout()

        # File-Picker-Button (kompakt, nur Icon)
        self.file_picker_button = QPushButton("📁")
        self.file_picker_button.setToolTip(
            "Datei einlesen (DOCX, PDF, TXT, MD, HTML)\nUnterstützt auch mehrere Dateien"
        )
        self.file_picker_button.setFixedSize(40, 40)
        self.file_picker_button.setStyleSheet(
            """
            QPushButton {
                background-color: #34495e;
                color: #ecf0f1;
                border: 2px solid #7f8c8d;
                border-radius: 6px;
                font-size: 18px;
            }
            QPushButton:hover {
                background-color: #3498db;
                border: 2px solid #2980b9;
            }
            QPushButton:pressed {
                background-color: #2980b9;
            }
        """
        )
        self.file_picker_button.clicked.connect(self._on_file_picker_clicked)

        self.curiosity_checkbox = QCheckBox("Aktives Nachfragen")
        self.curiosity_checkbox.setChecked(True)
        self.curiosity_checkbox.setToolTip(
            "Wenn aktiviert, fragt KAI von sich aus nach, wenn es Wörter nicht kennt."
        )

        controls_layout.addWidget(self.file_picker_button)
        controls_layout.addWidget(self.curiosity_checkbox)
        controls_layout.addStretch()  # Push alles nach links

        # PHASE 3.4 (User Feedback Loop): Feedback-Buttons
        self.feedback_widget = self._create_feedback_widget()
        self.current_answer_id = None  # Track aktuellen Answer für Feedback

        layout.addWidget(self.history)
        layout.addWidget(self.file_progress_label)  # Progress Label
        layout.addWidget(self.file_progress_bar)  # Progress Bar
        layout.addWidget(self.feedback_widget)  # Feedback-Buttons (dynamisch ein/aus)
        layout.addWidget(self.context_label)  # Context-Label zwischen History und Input
        layout.addWidget(self.input_text)  # Mehrzeiliges Eingabefeld
        layout.addLayout(controls_layout)  # Datei-Button + Checkbox

    def _create_feedback_widget(self):
        """
        PHASE 3.4: Erstellt Feedback-Widget mit Buttons

        Returns:
            QWidget mit Feedback-Buttons
        """
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)

        # Label
        label = QLabel("War diese Antwort hilfreich?")
        label.setStyleSheet("color: #95a5a6; font-size: 12px; font-weight: bold;")

        # Buttons
        button_style = """
            QPushButton {
                background-color: #34495e;
                color: #ecf0f1;
                border: 2px solid #7f8c8d;
                border-radius: 6px;
                padding: 6px 12px;
                font-size: 13px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #3498db;
                border: 2px solid #2980b9;
            }
            QPushButton:pressed {
                background-color: #2980b9;
            }
        """

        self.feedback_btn_correct = QPushButton("👍 Richtig")
        self.feedback_btn_correct.setStyleSheet(button_style)
        self.feedback_btn_correct.setToolTip("Diese Antwort war korrekt")
        self.feedback_btn_correct.clicked.connect(
            lambda: self._on_feedback_clicked("correct")
        )

        self.feedback_btn_incorrect = QPushButton("👎 Falsch")
        self.feedback_btn_incorrect.setStyleSheet(button_style)
        self.feedback_btn_incorrect.setToolTip("Diese Antwort war falsch")
        self.feedback_btn_incorrect.clicked.connect(
            lambda: self._on_feedback_clicked("incorrect")
        )

        self.feedback_btn_unsure = QPushButton("❓ Unsicher")
        self.feedback_btn_unsure.setStyleSheet(button_style)
        self.feedback_btn_unsure.setToolTip("Ich bin mir bei dieser Antwort unsicher")
        self.feedback_btn_unsure.clicked.connect(
            lambda: self._on_feedback_clicked("unsure")
        )

        self.feedback_btn_partial = QPushButton("⚖️ Teilweise")
        self.feedback_btn_partial.setStyleSheet(button_style)
        self.feedback_btn_partial.setToolTip("Diese Antwort war teilweise korrekt")
        self.feedback_btn_partial.clicked.connect(
            lambda: self._on_feedback_clicked("partially_correct")
        )

        layout.addWidget(label)
        layout.addWidget(self.feedback_btn_correct)
        layout.addWidget(self.feedback_btn_incorrect)
        layout.addWidget(self.feedback_btn_unsure)
        layout.addWidget(self.feedback_btn_partial)
        layout.addStretch()

        # Widget-Stil
        widget.setStyleSheet(
            """
            QWidget {
                background-color: #2c3e50;
                border: 2px solid #7f8c8d;
                border-radius: 8px;
                padding: 5px;
            }
        """
        )

        # Initial versteckt
        widget.setVisible(False)

        return widget

    def _on_feedback_clicked(self, feedback_type: str):
        """
        PHASE 3.4: Handler für Feedback-Button Klicks

        Args:
            feedback_type: 'correct', 'incorrect', 'unsure', 'partially_correct'
        """
        if not self.current_answer_id:
            logger.warning("Kein current_answer_id für Feedback vorhanden")
            return

        # Emit Signal an MainWindow
        # Das MainWindow sollte ein Signal haben: feedback_given(answer_id, feedback_type)
        # Wir nutzen parent() um zum MainWindow zu kommen
        main_window = self.parent()
        while main_window and not isinstance(main_window, QMainWindow):
            main_window = main_window.parent()

        if main_window and hasattr(main_window, "process_feedback"):
            main_window.process_feedback(self.current_answer_id, feedback_type)

        # Zeige Danke-Nachricht und blende Buttons aus
        self._show_feedback_thanks(feedback_type)

    def _show_feedback_thanks(self, feedback_type: str):
        """Zeigt Danke-Nachricht und blendet Feedback-Widget aus"""
        # Map feedback_type zu Emoji
        emoji_map = {
            "correct": "👍",
            "incorrect": "👎",
            "unsure": "❓",
            "partially_correct": "⚖️",
        }
        emoji = emoji_map.get(feedback_type, "✓")

        # Zeige Danke-Nachricht im Chat
        self.history.append(
            f'<i style="color:#95a5a6; font-size:12px;">{emoji} '
            f"Danke für dein Feedback! Ich lerne daraus.</i>"
        )

        # Blende Feedback-Widget aus
        self.feedback_widget.setVisible(False)
        self.current_answer_id = None

    def show_feedback_buttons(self, answer_id: str):
        """
        PHASE 3.4: Zeigt Feedback-Buttons für eine trackbare Antwort

        Args:
            answer_id: ID der Antwort, für die Feedback gegeben werden kann
        """
        if answer_id:
            self.current_answer_id = answer_id
            self.feedback_widget.setVisible(True)
            logger.debug(f"Feedback-Buttons aktiviert für answer_id={answer_id[:8]}")
        else:
            self.feedback_widget.setVisible(False)

    def add_message(self, sender, message):
        color = "#3498db" if sender == "Du" else "#e74c3c"
        self.history.append(f'<b style="color:{color};">{sender}:</b> {message}')

    def get_input(self):
        return self.input_text.toPlainText().strip()

    def clear_input(self):
        self.input_text.clear()

    @Slot(str)
    def update_context(self, context_summary: str):
        """
        PHASE 2 (Multi-Turn): Aktualisiert die Context-Anzeige.

        Args:
            context_summary: Formatierte Zusammenfassung des aktuellen Kontexts
        """
        if context_summary:
            self.context_label.setText(f"📝 Kontext: {context_summary}")
            self.context_label.setVisible(True)
        else:
            self.context_label.setText("")
            self.context_label.setVisible(False)

    @Slot(int, int, str)
    def update_file_progress(self, current: int, total: int, message: str):
        """
        PHASE 8 (Extended Features): Aktualisiert die Datei-Ingestion Progress Bar.

        Args:
            current: Aktueller Fortschritt
            total: Gesamtanzahl
            message: Status-Nachricht
        """
        if total > 0:
            percent = int((current / total) * 100)
            self.file_progress_bar.setMaximum(total)
            self.file_progress_bar.setValue(current)
            self.file_progress_bar.setFormat(f"{percent}%")
            self.file_progress_label.setText(message)

            # WICHTIG: Stelle sicher dass beide Widgets sichtbar sind
            if not self.file_progress_bar.isVisible():
                self.file_progress_bar.setVisible(True)
                self.file_progress_label.setVisible(True)

            # Process events to keep UI responsive during file processing
            QApplication.processEvents()

        # Verstecke Progress-Bar wenn fertig
        if current >= total and total > 0:
            # Kurz anzeigen, dann ausblenden (2 Sekunden Delay)
            QTimer.singleShot(2000, lambda: self.file_progress_bar.setVisible(False))
            QTimer.singleShot(2000, lambda: self.file_progress_label.setVisible(False))

    @Slot()
    def _on_file_picker_clicked(self):
        """
        Öffnet File-Picker für Dokument-Auswahl (einzeln oder mehrere).

        Workflow:
            1. Öffnet QFileDialog mit Mehrfachauswahl
            2. Fügt Command-String in Eingabefeld ein
            3. User kann Command prüfen/anpassen vor Enter
        """
        # Öffne File-Dialog mit Mehrfachauswahl
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Wähle eine oder mehrere Dateien zum Einlesen",
            "",  # Startverzeichnis (leer = letztes verwendetes)
            "Alle unterstützten Formate (*.docx *.pdf *.txt *.md *.markdown *.html *.htm);;"
            "Word-Dokumente (*.docx);;"
            "PDF-Dokumente (*.pdf);;"
            "Text-Dateien (*.txt);;"
            "Markdown-Dateien (*.md *.markdown);;"
            "HTML-Dateien (*.html *.htm);;"
            "Alle Dateien (*.*)",
        )

        # Wenn User Datei(en) gewählt hat
        if file_paths:
            # Erstelle Command (einzelne Datei oder Batch)
            if len(file_paths) == 1:
                command = f"Lese Datei: {file_paths[0]}"
            else:
                files_str = ";".join(file_paths)
                command = f"Lese Dateien: {files_str}"

            self.input_text.setPlainText(command)
            self.input_text.setFocus()

            # Cursor ans Ende bewegen
            cursor = self.input_text.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            self.input_text.setTextCursor(cursor)


class InnerPictureDisplay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        self.label = QLabel("Inneres Bild (Gedankengang)")
        self.label.setStyleSheet(
            "font-weight: bold; font-size: 16px; margin-bottom: 5px;"
        )

        self.trace_view = QTextEdit()
        self.trace_view.setReadOnly(True)
        self.trace_view.setStyleSheet(
            "font-family: Consolas, monospace; color: #2ecc71;"
        )

        layout.addWidget(self.label)
        layout.addWidget(self.trace_view)

    def update_trace(self, trace_steps: list):
        if not trace_steps:
            self.trace_view.setText("Kein spezifischer Gedankengang für diese Antwort.")
        else:
            formatted_trace = "\n".join(f"-> {step}" for step in trace_steps)
            self.trace_view.setText(formatted_trace)

    @Slot(str)
    def update_trace_from_string(self, trace_str: str):
        """PHASE 2: Update trace from formatted string (Working Memory)"""
        self.trace_view.setText(trace_str)


# --- Separates Analyse-Fenster (ausgelagert) ---


class AnalysisWindow(QMainWindow):
    """Separates Fenster für Inneres Bild, Beweisbaum, Episodisches Gedächtnis und Kontext-Tracker."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("KAI - Analyse & Visualisierungen")
        self.setGeometry(150, 150, 1000, 700)

        # Central Widget mit Tab-Widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout(self.central_widget)

        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

        # Tab 1: Inneres Bild
        self.inner_picture = InnerPictureDisplay()
        self.tab_widget.addTab(self.inner_picture, "Inneres Bild")

        # Tab 2: Beweisbaum (only if available)
        if PROOF_TREE_AVAILABLE:
            self.proof_tree_widget = ProofTreeWidget()
            self.tab_widget.addTab(self.proof_tree_widget, "Beweisbaum")
        else:
            self.proof_tree_widget = None  # type: ignore[assignment]

        # Tab 3: Episodisches Gedächtnis (only if available)
        if EPISODIC_MEMORY_WIDGET_AVAILABLE:
            self.episodic_memory_widget = EpisodicMemoryWidget()
            self.tab_widget.addTab(
                self.episodic_memory_widget, "Episodisches Gedächtnis"
            )
        else:
            self.episodic_memory_widget = None  # type: ignore[assignment]

        # Tab 4: Kontext-Tracker (only if available)
        if CONTEXT_VISUALIZATION_AVAILABLE:
            self.context_visualization_widget = ContextVisualizationWidget()
            self.tab_widget.addTab(self.context_visualization_widget, "Kontext-Tracker")
        else:
            self.context_visualization_widget = None  # type: ignore[assignment]

        # Tab 5: Räumliches Grid (only if available)
        if SPATIAL_GRID_WIDGET_AVAILABLE:
            self.spatial_grid_widget = SpatialGridWidget()
            self.tab_widget.addTab(self.spatial_grid_widget, "Räumliches Grid")
        else:
            self.spatial_grid_widget = None  # type: ignore[assignment]

        # Tab 6: Resonance View (only if available)
        if RESONANCE_VIEW_AVAILABLE:
            self.resonance_view_widget = ResonanceViewWidget()
            self.tab_widget.addTab(self.resonance_view_widget, "Resonance View")
        else:
            self.resonance_view_widget = None  # type: ignore[assignment]


# --- Hauptfenster ---


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("KAI - Konzeptueller AI Prototyp (v3.0 Fokus-UI)")
        self.setGeometry(100, 100, 1200, 800)

        self.setup_kai_backend()
        self.create_menu_bar()
        self.create_status_bar()

        # Separates Analysis-Fenster (initial geschlossen)
        self.analysis_window = None

        # Central Widget mit vertikalem Splitter
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        main_layout = QVBoxLayout(self.central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)

        # Vertikaler Splitter: PlanMonitor (oben) + Chat (unten)
        self.splitter = QSplitter(Qt.Orientation.Vertical)

        # PlanMonitor (kollabierbar)
        self.plan_monitor_container = QFrame()
        self.plan_monitor_container.setFrameShape(QFrame.Shape.StyledPanel)
        plan_monitor_layout = QVBoxLayout(self.plan_monitor_container)
        plan_monitor_layout.setContentsMargins(0, 0, 0, 0)

        # Header mit Toggle-Button
        plan_monitor_header = QWidget()
        header_layout = QHBoxLayout(plan_monitor_header)
        header_layout.setContentsMargins(5, 5, 5, 5)

        self.plan_monitor_toggle_btn = QPushButton("▼ Plan Monitor")
        self.plan_monitor_toggle_btn.setCheckable(True)
        self.plan_monitor_toggle_btn.setChecked(True)
        self.plan_monitor_toggle_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #34495e;
                color: #ecf0f1;
                border: 1px solid #7f8c8d;
                border-radius: 4px;
                padding: 5px;
                font-weight: bold;
                text-align: left;
            }
            QPushButton:hover {
                background-color: #3498db;
            }
            QPushButton:checked {
                background-color: #2c3e50;
            }
        """
        )
        self.plan_monitor_toggle_btn.clicked.connect(self.toggle_plan_monitor)
        header_layout.addWidget(self.plan_monitor_toggle_btn)
        header_layout.addStretch()

        self.plan_monitor = PlanMonitor()
        plan_monitor_layout.addWidget(plan_monitor_header)
        plan_monitor_layout.addWidget(self.plan_monitor)

        # Chat Interface
        self.chat_interface = ChatInterface()

        # Splitter konfigurieren
        self.splitter.addWidget(self.plan_monitor_container)
        self.splitter.addWidget(self.chat_interface)
        self.splitter.setStretchFactor(0, 1)  # PlanMonitor: weniger Platz
        self.splitter.setStretchFactor(1, 3)  # Chat: mehr Platz
        self.splitter.setSizes([150, 650])  # Initial: 150px PlanMonitor, 650px Chat

        main_layout.addWidget(self.splitter)

        self.connect_signals()
        self.chat_interface.add_message(
            "KAI",
            "Hallo. Ich bin bereit. 💬 Nutze das Analyse-Fenster (Menü) für detaillierte Visualisierungen.",
        )

    def setup_kai_backend(self):
        try:
            self.netzwerk = KonzeptNetzwerk()
            embedding_service = EmbeddingService()
            self.kai_thread = QThread()
            self.kai_worker = KaiWorker(self.netzwerk, embedding_service)
            self.kai_worker.moveToThread(self.kai_thread)
            self.kai_thread.start()

            # PHASE 3.4 (User Feedback Loop): Initialize FeedbackHandler
            if FEEDBACK_HANDLER_AVAILABLE:
                try:
                    meta_learning = (
                        self.kai_worker.meta_learning
                        if hasattr(self.kai_worker, "meta_learning")
                        else None
                    )
                    self.feedback_handler = FeedbackHandler(
                        netzwerk=self.netzwerk, meta_learning=meta_learning
                    )
                    # Connect FeedbackHandler to ResponseFormatter
                    if hasattr(self.kai_worker, "response_formatter"):
                        self.kai_worker.response_formatter.feedback_handler = (
                            self.feedback_handler
                        )
                        print("[UI] FeedbackHandler mit ResponseFormatter verbunden")
                    print("[UI] FeedbackHandler initialisiert")
                except Exception as e:
                    print(
                        f"[UI] [WARNING] FeedbackHandler konnte nicht initialisiert werden: {e}"
                    )
                    self.feedback_handler = None  # type: ignore[assignment]
            else:
                self.feedback_handler = None  # type: ignore[assignment]

            # Prüfe ob Worker erfolgreich initialisiert wurde
            if not self.kai_worker.is_initialized_successfully:
                error_msg = (
                    self.kai_worker.initialization_error_message
                    or "[ERROR] Unbekannter Initialisierungsfehler"
                )
                print(f"[UI] {error_msg}")
            else:
                print("[UI] KAI Backend erfolgreich initialisiert.")
        except Exception as e:
            # Nutzerfreundliche Fehlermeldung generieren
            if isinstance(e, KAIException):
                user_friendly_msg = get_user_friendly_message(e, include_details=False)
                print(f"[UI] {user_friendly_msg}")
            else:
                print(
                    f"[UI] [ERROR] KRITISCHER FEHLER beim Starten des KAI Backends: {e}"
                )

    def create_menu_bar(self):
        """Erstellt die Menu-Leiste mit Einstellungen und Logging-Optionen"""
        menubar = self.menuBar()

        # === Ansicht-Menu (NEU: für UI-Optionen) ===
        view_menu = menubar.addMenu("&Ansicht")

        # Analyse-Fenster öffnen
        analysis_window_action = QAction("&Analyse-Fenster öffnen", self)
        analysis_window_action.setShortcut("Ctrl+A")
        analysis_window_action.setStatusTip(
            "Öffnet das Analyse-Fenster (Inneres Bild, Beweisbaum, etc.)"
        )
        analysis_window_action.triggered.connect(self.open_analysis_window)
        view_menu.addAction(analysis_window_action)

        view_menu.addSeparator()

        # Plan Monitor ein-/ausblenden
        toggle_plan_action = QAction("Plan Monitor &umschalten", self)
        toggle_plan_action.setShortcut("Ctrl+P")
        toggle_plan_action.setStatusTip("Plan Monitor ein-/ausblenden")
        toggle_plan_action.triggered.connect(self.toggle_plan_monitor)
        view_menu.addAction(toggle_plan_action)

        # === Einstellungen-Menu ===
        settings_menu = menubar.addMenu("&Einstellungen")

        # Einstellungen-Dialog (Logging + Tests)
        settings_dialog_action = QAction("&Einstellungen öffnen...", self)
        settings_dialog_action.setShortcut("Ctrl+Shift+S")
        settings_dialog_action.setStatusTip(
            "Öffnet den Einstellungen-Dialog (Logging, Tests)"
        )
        settings_dialog_action.triggered.connect(self.open_settings_dialog)
        settings_menu.addAction(settings_dialog_action)

        # Test-Runner öffnen (direkter Zugriff)
        test_runner_action = QAction("&Test-Runner öffnen...", self)
        test_runner_action.setShortcut("Ctrl+T")
        test_runner_action.setStatusTip("Öffnet den Test-Runner direkt")
        test_runner_action.triggered.connect(self.open_test_runner)
        settings_menu.addAction(test_runner_action)

        settings_menu.addSeparator()

        # Log-Viewer öffnen
        viewer_action = QAction("Log-&Viewer öffnen", self)
        viewer_action.setShortcut("Ctrl+L")
        viewer_action.setStatusTip("Öffnet das Log-Viewer-Fenster")
        viewer_action.triggered.connect(self.open_log_viewer)
        settings_menu.addAction(viewer_action)

        # Quick-Access zu Log-Dateien
        open_logs_folder_action = QAction("Log-Ordner öffnen", self)
        open_logs_folder_action.setStatusTip("Öffnet den Ordner mit allen Log-Dateien")
        open_logs_folder_action.triggered.connect(self.open_logs_folder)
        settings_menu.addAction(open_logs_folder_action)

    def open_settings_dialog(self):
        """Öffnet den neuen Einstellungen-Dialog mit Tabs (Logging + Tests)"""
        settings_dialog = SettingsDialog(self)

        # Callback wenn Einstellungen geändert wurden
        def on_settings_changed(new_settings):
            print(f"[UI] Einstellungen aktualisiert: {new_settings}")
            self.chat_interface.add_message(
                "System", f"Einstellungen geändert: {new_settings}"
            )

        # Callback für Theme-Änderungen
        def on_theme_changed(theme: str):
            print(f"[UI] Theme geändert zu: {theme}")
            # Apply new theme to application
            app = QApplication.instance()
            if app:
                set_stylesheet(app, theme)
            self.chat_interface.add_message(
                "System",
                f"Theme geändert zu: {theme}. Einige Elemente aktualisieren sich sofort, andere nach Neustart.",
            )

        settings_dialog.settings_changed.connect(on_settings_changed)
        # Connect theme_changed signal from AppearanceTab
        if hasattr(settings_dialog, "appearance_tab"):
            settings_dialog.appearance_tab.theme_changed.connect(on_theme_changed)
        settings_dialog.exec()

    def open_test_runner(self):
        """Öffnet den Test-Runner in einem separaten Fenster"""
        test_runner_window = TestRunnerWindow(self)
        test_runner_window.exec()

    def open_log_viewer(self):
        """Öffnet das Log-Viewer-Fenster"""
        viewer = LogViewerWindow(self)
        viewer.exec()

    def open_logs_folder(self):
        """Öffnet den Log-Ordner im Datei-Explorer"""
        import os
        import subprocess

        from component_15_logging_config import LOG_DIR

        try:
            if os.name == "nt":  # Windows
                os.startfile(LOG_DIR)
            elif os.name == "posix":  # macOS, Linux
                subprocess.call(
                    ["open" if sys.platform == "darwin" else "xdg-open", str(LOG_DIR)]
                )

            self.chat_interface.add_message("System", f"Log-Ordner geöffnet: {LOG_DIR}")
        except Exception as e:
            self.chat_interface.add_message(
                "System", f"Fehler beim Öffnen des Log-Ordners: {e}"
            )

    def toggle_plan_monitor(self):
        """Blendet den Plan Monitor ein oder aus"""
        is_visible = self.plan_monitor.isVisible()
        self.plan_monitor.setVisible(not is_visible)

        # Update Button-Text
        if is_visible:
            self.plan_monitor_toggle_btn.setText("▶ Plan Monitor")
            self.plan_monitor_toggle_btn.setChecked(False)
        else:
            self.plan_monitor_toggle_btn.setText("▼ Plan Monitor")
            self.plan_monitor_toggle_btn.setChecked(True)

    def open_analysis_window(self):
        """Öffnet das separate Analyse-Fenster"""
        if self.analysis_window is None:
            self.analysis_window = AnalysisWindow(self)
            # Verbinde Signale mit Analysis-Fenster
            self.connect_analysis_window_signals()

        self.analysis_window.show()
        self.analysis_window.raise_()
        self.analysis_window.activateWindow()

    def connect_analysis_window_signals(self):
        """Verbindet Worker-Signale mit dem Analysis-Fenster"""
        if self.analysis_window is None:
            return

        signals = self.kai_worker.signals

        # Inner Picture
        signals.inner_picture_update.connect(
            self.analysis_window.inner_picture.update_trace_from_string
        )

        # Proof Tree
        if PROOF_TREE_AVAILABLE and self.analysis_window.proof_tree_widget:
            signals.proof_tree_update.connect(
                lambda proof_tree: self.update_analysis_proof_tree(proof_tree)
            )

        # Episodic Memory
        if (
            EPISODIC_MEMORY_WIDGET_AVAILABLE
            and self.analysis_window.episodic_memory_widget
        ):
            signals.episodic_data_update.connect(
                self.analysis_window.episodic_memory_widget.update_episodes
            )

    def update_analysis_proof_tree(self, proof_tree):
        """Updates proof tree in analysis window and switches to that tab"""
        if (
            self.analysis_window
            and PROOF_TREE_AVAILABLE
            and self.analysis_window.proof_tree_widget
        ):
            self.analysis_window.proof_tree_widget.set_proof_tree(proof_tree)
            # Switch to Beweisbaum tab
            self.analysis_window.tab_widget.setCurrentWidget(
                self.analysis_window.proof_tree_widget
            )

    def create_status_bar(self):
        """Erstellt die Statusleiste mit Neo4j-Verbindungsindikator"""
        status_bar = self.statusBar()

        # Neo4j-Status-Label
        self.neo4j_status_label = QLabel("Neo4j: Prüfe Verbindung...")
        self.neo4j_status_label.setStyleSheet("padding: 2px 8px; font-size: 11px;")
        status_bar.addPermanentWidget(self.neo4j_status_label)

        # Timer für regelmäßige Status-Updates (alle 30 Sekunden)
        self.status_check_timer = QTimer(self)
        self.status_check_timer.timeout.connect(self.check_neo4j_status)
        self.status_check_timer.start(30000)  # 30 Sekunden

        # Initiale Überprüfung
        QTimer.singleShot(1000, self.check_neo4j_status)  # Nach 1 Sekunde

    def check_neo4j_status(self):
        """Prüft den Neo4j-Verbindungsstatus"""
        try:
            # Prüfe ob netzwerk verfügbar ist
            if (
                hasattr(self, "netzwerk")
                and self.netzwerk
                and hasattr(self.netzwerk, "driver")
            ):
                # Versuche eine einfache Cypher-Query
                try:
                    with self.netzwerk.driver.session(database="neo4j") as session:
                        result = session.run("RETURN 1 AS test")
                        result.single()

                    # Verbindung erfolgreich
                    self.neo4j_status_label.setText("🟢 Neo4j: Verbunden")
                    self.neo4j_status_label.setStyleSheet(
                        "background-color: #2ecc71; color: white; padding: 2px 8px; "
                        "border-radius: 3px; font-size: 11px;"
                    )
                except Exception as e:
                    # Verbindung fehlgeschlagen
                    self.neo4j_status_label.setText("🔴 Neo4j: Getrennt")
                    self.neo4j_status_label.setStyleSheet(
                        "background-color: #e74c3c; color: white; padding: 2px 8px; "
                        "border-radius: 3px; font-size: 11px;"
                    )
                    print(f"[UI] Neo4j-Verbindungsfehler: {e}")
            else:
                # Netzwerk nicht initialisiert
                self.neo4j_status_label.setText("[WARNING] Neo4j: Nicht initialisiert")
                self.neo4j_status_label.setStyleSheet(
                    "background-color: #f39c12; color: white; padding: 2px 8px; "
                    "border-radius: 3px; font-size: 11px;"
                )
        except Exception as e:
            # Fehler beim Status-Check
            self.neo4j_status_label.setText("[ERROR] Neo4j: Fehler")
            self.neo4j_status_label.setStyleSheet(
                "background-color: #95a5a6; color: white; padding: 2px 8px; "
                "border-radius: 3px; font-size: 11px;"
            )
            print(f"[UI] Fehler beim Neo4j-Status-Check: {e}")

    def connect_signals(self):
        # Strg+Enter zum Senden (statt Enter)
        send_shortcut = QShortcut(
            QKeySequence("Ctrl+Return"), self.chat_interface.input_text
        )
        send_shortcut.activated.connect(self.send_query_to_kai)

        signals = self.kai_worker.signals
        signals.clear_goals.connect(self.plan_monitor.clear_goals)
        signals.set_main_goal.connect(self.plan_monitor.set_main_goal)
        signals.add_sub_goal.connect(self.plan_monitor.add_sub_goal)
        signals.update_sub_goal_status.connect(self.plan_monitor.update_sub_goal_status)
        signals.finished.connect(self.on_kai_finished)
        # Context-Update für Chat-Interface
        signals.context_update.connect(self.chat_interface.update_context)
        # File-Progress für Chat-Interface
        signals.file_progress_update.connect(self.chat_interface.update_file_progress)
        # Preview-Confirmation für File-Upload
        signals.preview_confirmation_needed.connect(
            self.show_preview_confirmation_dialog
        )

    def closeEvent(self, event):
        print("UI wird geschlossen. Speichere finales Netzwerk.")
        if self.netzwerk:
            self.netzwerk.close()
        if self.kai_thread:
            self.kai_thread.quit()
            self.kai_thread.wait()
        event.accept()

    def send_query_to_kai(self):
        query = self.chat_interface.get_input()
        if query:
            self.chat_interface.add_message("Du", query)
            self.chat_interface.clear_input()
            self.chat_interface.input_text.setEnabled(False)
            # Emit signal to process query in worker thread (non-blocking)
            self.kai_worker.signals.query_submitted.emit(query)

    @Slot(object)
    def on_kai_finished(self, response_obj):
        # Check for reprocessing request (from context manager)
        if response_obj.text == "__REPROCESS_QUERY__":
            # Extract corrected query from trace
            if response_obj.trace and len(response_obj.trace) > 0:
                corrected_query = response_obj.trace[0]
                # Reprocess the corrected query automatically via signal (non-blocking)
                self.kai_worker.signals.query_submitted.emit(corrected_query)
                return  # Don't display "__REPROCESS_QUERY__" or enable input yet
            else:
                # Fallback: Enable input if no corrected query found
                self.chat_interface.add_message(
                    "KAI", "Es gab einen Fehler bei der Verarbeitung."
                )
                self.chat_interface.input_text.setEnabled(True)
                self.chat_interface.input_text.setFocus()
                return

        # Normal response handling
        self.chat_interface.add_message("KAI", response_obj.text)

        # PHASE 3.4 (User Feedback Loop): Show feedback buttons if answer_id available
        if hasattr(response_obj, "answer_id") and response_obj.answer_id:
            self.chat_interface.show_feedback_buttons(response_obj.answer_id)

        self.chat_interface.input_text.setEnabled(True)
        self.chat_interface.input_text.setFocus()

    def process_feedback(self, answer_id: str, feedback_type: str):
        """
        PHASE 3.4 (User Feedback Loop): Verarbeitet User-Feedback für eine Antwort.

        Args:
            answer_id: Eindeutige ID der Antwort
            feedback_type: Art des Feedbacks ('correct', 'incorrect', 'unsure', 'partially_correct')
        """
        if not FEEDBACK_HANDLER_AVAILABLE or not self.feedback_handler:
            print("[UI] [WARNING] FeedbackHandler nicht verfügbar")
            return

        # Map string to FeedbackType enum
        feedback_type_map = {
            "correct": FeedbackType.CORRECT,
            "incorrect": FeedbackType.INCORRECT,
            "unsure": FeedbackType.UNSURE,
            "partially_correct": FeedbackType.PARTIALLY_CORRECT,
        }

        fb_type = feedback_type_map.get(feedback_type)
        if not fb_type:
            print(f"[UI] [WARNING] Unbekannter Feedback-Type: {feedback_type}")
            return

        try:
            result = self.feedback_handler.process_user_feedback(
                answer_id=answer_id, feedback_type=fb_type
            )

            if result["success"]:
                print(f"[UI] Feedback verarbeitet: {feedback_type} für {answer_id[:8]}")
                # Optional: Stats anzeigen
                stats = self.feedback_handler.get_feedback_stats()
                print(
                    f"[UI] Feedback-Stats: Accuracy={stats['accuracy']:.2%}, "
                    f"Total={stats['total_feedbacks']}"
                )
            else:
                print(
                    f"[UI] [WARNING] Feedback-Verarbeitung fehlgeschlagen: "
                    f"{result.get('message', 'Unknown')}"
                )

        except Exception as e:
            print(f"[UI] [ERROR] Fehler beim Feedback-Processing: {e}")

    @Slot(str, str, int)
    def show_preview_confirmation_dialog(
        self, preview: str, file_name: str, char_count: int
    ):
        """
        PHASE 8 (Extended Features): Zeigt Preview-Dialog und fordert User-Bestätigung an.

        Args:
            preview: Text-Preview (erste 500 Zeichen)
            file_name: Name der Datei
            char_count: Gesamtanzahl Zeichen in der Datei
        """
        # Erstelle Message Box
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Datei-Ingestion bestätigen")
        msg_box.setIcon(QMessageBox.Icon.Question)

        # Formatiere Text
        msg_box.setText(
            f"Möchtest du die Datei '{file_name}' ({char_count:,} Zeichen) einlesen und lernen?"
        )

        # Zeige Preview als detaillierten Text
        msg_box.setDetailedText(f"Vorschau:\n\n{preview}")

        # Buttons
        msg_box.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        msg_box.setDefaultButton(QMessageBox.StandardButton.Yes)

        # Zeige Dialog (modal, blockiert UI-Thread bis Antwort)
        result = msg_box.exec()

        # Emittiere Antwort an Worker
        confirmed = result == QMessageBox.StandardButton.Yes
        self.kai_worker.signals.preview_confirmation_response.emit(confirmed)

        # Zeige Feedback in Chat
        if confirmed:
            self.chat_interface.add_message(
                "System", f"Starte Ingestion von '{file_name}'..."
            )
        else:
            self.chat_interface.add_message(
                "System", f"Ingestion von '{file_name}' abgebrochen."
            )


def get_dark_theme():
    """Dark Theme Stylesheet"""
    return """
        QWidget {
            background-color: #34495e;
            color: #ecf0f1;
            font-size: 14px;
        }
        QMainWindow {
            background-color: #2c3e50;
        }
        QGroupBox {
            font-weight: bold;
            font-size: 14px;
            border: 1px solid #7f8c8d;
            border-radius: 5px;
            margin-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top center;
            padding: 0 5px;
        }
        QTextEdit, QLineEdit {
            background-color: #2c3e50;
            border: 1px solid #7f8c8d;
            border-radius: 4px;
            padding: 5px;
        }
        QLineEdit:focus {
            border: 1px solid #3498db;
        }
    """


def get_light_theme():
    """Light Theme Stylesheet"""
    return """
        QWidget {
            background-color: #ecf0f1;
            color: #2c3e50;
            font-size: 14px;
        }
        QMainWindow {
            background-color: #bdc3c7;
        }
        QGroupBox {
            font-weight: bold;
            font-size: 14px;
            border: 1px solid #7f8c8d;
            border-radius: 5px;
            margin-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top center;
            padding: 0 5px;
        }
        QTextEdit, QLineEdit {
            background-color: #ffffff;
            border: 1px solid #95a5a6;
            border-radius: 4px;
            padding: 5px;
            color: #2c3e50;
        }
        QLineEdit:focus {
            border: 1px solid #3498db;
        }
    """


def set_stylesheet(app, theme="dark"):
    """
    Setzt das Stylesheet der Anwendung basierend auf dem gewählten Theme.

    Args:
        app: QApplication Instanz
        theme: "dark" oder "light"
    """
    if theme == "light":
        app.setStyleSheet(get_light_theme())
    else:  # default to dark
        app.setStyleSheet(get_dark_theme())


if __name__ == "__main__":
    setup_logging()
    app = QApplication(sys.argv)

    # Load theme from config
    cfg = get_config()
    theme = cfg.get("theme", "dark")
    set_stylesheet(app, theme)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())
