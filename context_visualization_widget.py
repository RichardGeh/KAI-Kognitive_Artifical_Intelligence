"""
context_visualization_widget.py

PySide6 Widget zur Visualisierung des Working Memory Context-Stacks.
Zeigt hierarchische Tree-View der verschachtelten Kontexte mit Details.
"""

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTreeWidget,
    QTreeWidgetItem,
    QPushButton,
    QLabel,
    QFileDialog,
    QTextEdit,
    QSplitter,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QBrush
from typing import Optional
from datetime import datetime

from component_13_working_memory import WorkingMemory, ContextFrame, ContextType


# Farb-Mapping für Context-Types
CONTEXT_TYPE_COLORS = {
    ContextType.QUESTION: QColor(100, 149, 237),  # Cornflower Blue
    ContextType.DEFINITION: QColor(60, 179, 113),  # Medium Sea Green
    ContextType.PATTERN_LEARNING: QColor(255, 140, 0),  # Dark Orange
    ContextType.INFERENCE: QColor(147, 112, 219),  # Medium Purple
    ContextType.TEXT_INGESTION: QColor(70, 130, 180),  # Steel Blue
    ContextType.CLARIFICATION: QColor(255, 165, 0),  # Orange
}


class ContextVisualizationWidget(QWidget):
    """
    Widget zur Visualisierung des Working Memory Context-Stacks.

    Features:
    - Hierarchische Tree-View der Context-Frames
    - Farbcodierung nach Context-Type
    - Detailansicht für ausgewählte Frames
    - Export-Funktionalität (JSON, Debug-Report)
    - Live-Update via Signal
    """

    # Signal um Working Memory von außen zu setzen
    memory_updated = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.working_memory: Optional[WorkingMemory] = None

        self.init_ui()

    def init_ui(self):
        """Initialisiere UI-Komponenten"""
        layout = QVBoxLayout(self)

        # Header mit Informationen und Actions
        header_layout = QHBoxLayout()

        self.info_label = QLabel("Working Memory: Nicht initialisiert")
        self.info_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        header_layout.addWidget(self.info_label)

        header_layout.addStretch()

        # Export-Buttons
        self.export_json_button = QPushButton("Export JSON")
        self.export_json_button.clicked.connect(self.export_json)
        self.export_json_button.setEnabled(False)
        header_layout.addWidget(self.export_json_button)

        self.export_report_button = QPushButton("Export Debug-Report")
        self.export_report_button.clicked.connect(self.export_debug_report)
        self.export_report_button.setEnabled(False)
        header_layout.addWidget(self.export_report_button)

        layout.addLayout(header_layout)

        # Splitter für Tree-View und Detail-View
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Tree-View für Context-Stack
        self.tree_widget = QTreeWidget()
        self.tree_widget.setHeaderLabels(["Frame", "Query", "Type", "Idle Duration"])
        self.tree_widget.setColumnWidth(0, 200)
        self.tree_widget.setColumnWidth(1, 300)
        self.tree_widget.setColumnWidth(2, 150)
        self.tree_widget.itemSelectionChanged.connect(self.on_frame_selected)
        splitter.addWidget(self.tree_widget)

        # Detail-View für ausgewählten Frame
        self.detail_view = QTextEdit()
        self.detail_view.setReadOnly(True)
        self.detail_view.setPlaceholderText(
            "Wähle einen Frame aus, um Details anzuzeigen..."
        )
        splitter.addWidget(self.detail_view)

        splitter.setStretchFactor(0, 2)  # Tree View größer
        splitter.setStretchFactor(1, 1)  # Detail View kleiner

        layout.addWidget(splitter)

        # Legende
        legend_layout = QHBoxLayout()
        legend_layout.addWidget(QLabel("Legende:"))

        for context_type, color in CONTEXT_TYPE_COLORS.items():
            label = QLabel(f"  {context_type.value}  ")
            label.setAutoFillBackground(True)
            palette = label.palette()
            palette.setColor(label.backgroundRole(), color)
            label.setPalette(palette)
            legend_layout.addWidget(label)

        legend_layout.addStretch()
        layout.addLayout(legend_layout)

    def set_working_memory(self, memory: WorkingMemory):
        """
        Setze das Working Memory Objekt für Visualisierung.

        Args:
            memory: WorkingMemory-Instanz
        """
        self.working_memory = memory
        self.update_visualization()

    def update_visualization(self):
        """
        Aktualisiere die Visualisierung des Context-Stacks.

        Wird aufgerufen wenn sich das Working Memory geändert hat.
        """
        if not self.working_memory:
            return

        # Clear bisherige Darstellung
        self.tree_widget.clear()

        stack_depth = self.working_memory.get_stack_depth()

        # Update Info-Label
        self.info_label.setText(
            f"Working Memory: {stack_depth} Frame(s) | "
            f"Max: {self.working_memory.max_stack_depth}"
        )

        # Enable/Disable Export-Buttons
        has_frames = stack_depth > 0
        self.export_json_button.setEnabled(has_frames)
        self.export_report_button.setEnabled(has_frames)

        if stack_depth == 0:
            self.detail_view.setPlainText("Working Memory ist leer.")
            return

        # Erstelle Tree-Items für alle Frames
        parent_items = {}

        for i, frame in enumerate(self.working_memory.context_stack):
            # Erstelle Tree-Item
            item = QTreeWidgetItem()

            # Frame ID
            item.setText(0, f"Frame {i+1}")
            item.setData(0, Qt.ItemDataRole.UserRole, frame)  # Speichere Frame-Referenz

            # Query (erste 50 Zeichen)
            query_preview = (
                frame.query[:50] + "..." if len(frame.query) > 50 else frame.query
            )
            item.setText(1, query_preview)

            # Context-Type
            item.setText(2, frame.context_type.value)

            # Idle Duration
            idle_duration = frame.get_idle_duration()
            item.setText(3, f"{idle_duration:.1f}s")

            # Farbcodierung nach Context-Type
            color = CONTEXT_TYPE_COLORS.get(frame.context_type, QColor(200, 200, 200))
            for col in range(4):
                item.setBackground(col, QBrush(color))
                item.setForeground(col, QBrush(QColor(0, 0, 0)))

            # Parent-Child-Beziehung herstellen
            if frame.parent_frame_id and frame.parent_frame_id in parent_items:
                parent_items[frame.parent_frame_id].addChild(item)
            else:
                self.tree_widget.addTopLevelItem(item)

            # Speichere Item für spätere Parent-Child-Referenz
            parent_items[frame.frame_id] = item

            # Expandiere alle Frames
            item.setExpanded(True)

    def on_frame_selected(self):
        """
        Callback wenn ein Frame im Tree ausgewählt wurde.

        Zeigt Details des Frames in der Detail-View.
        """
        selected_items = self.tree_widget.selectedItems()

        if not selected_items:
            self.detail_view.clear()
            return

        # Hole Frame-Daten
        frame: ContextFrame = selected_items[0].data(0, Qt.ItemDataRole.UserRole)

        if not frame:
            return

        # Formatiere Detail-Anzeige
        details = []
        details.append("=" * 60)
        details.append(f"FRAME DETAILS: {frame.frame_id}")
        details.append("=" * 60)
        details.append("")

        details.append(f"Context-Type: {frame.context_type.value}")
        details.append(f"Query: {frame.query}")
        details.append("")

        details.append(f"Created: {frame.created_at.isoformat()}")
        details.append(f"Last Access: {frame.last_access_time.isoformat()}")
        details.append(f"Idle Duration: {frame.get_idle_duration():.2f}s")
        details.append(f"Parent Frame ID: {frame.parent_frame_id or 'None'}")
        details.append("")

        # Entities
        if frame.entities:
            details.append(f"Entities ({len(frame.entities)}):")
            for entity in frame.entities:
                details.append(f"  - {entity}")
        else:
            details.append("Entities: None")
        details.append("")

        # Relations
        if frame.relations:
            details.append(f"Relations ({len(frame.relations)}):")
            for rel_type, rel_values in frame.relations.items():
                details.append(f"  {rel_type}: {rel_values}")
        else:
            details.append("Relations: None")
        details.append("")

        # Variables
        if frame.variables:
            details.append(f"Variables ({len(frame.variables)}):")
            for key, value in frame.variables.items():
                value_str = str(value)[:200]  # Limit zu 200 Zeichen
                details.append(f"  {key} = {value_str}")
        else:
            details.append("Variables: None")
        details.append("")

        # Reasoning-States
        if frame.reasoning_states:
            details.append(f"Reasoning-States ({len(frame.reasoning_states)}):")
            for i, rs in enumerate(frame.reasoning_states, 1):
                details.append(f"  {i}. [{rs.step_type}] {rs.description}")
                details.append(f"     Confidence: {rs.confidence:.2f}")
                details.append(f"     Timestamp: {rs.timestamp.isoformat()}")
                if rs.data:
                    details.append(f"     Data: {rs.data}")
                details.append("")
        else:
            details.append("Reasoning-States: None")
        details.append("")

        # Metadata
        if frame.metadata:
            details.append("Metadata:")
            for key, value in frame.metadata.items():
                details.append(f"  {key}: {value}")
        else:
            details.append("Metadata: None")

        details.append("")
        details.append("=" * 60)

        # Setze Detail-Text
        self.detail_view.setPlainText("\n".join(details))

    def export_json(self):
        """Export Working Memory als JSON-Datei"""
        if not self.working_memory:
            return

        # Datei-Dialog
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Working Memory als JSON",
            f"working_memory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "JSON Files (*.json)",
        )

        if filename:
            success = self.working_memory.export_to_json(
                filename, include_timestamps=True
            )
            if success:
                self.detail_view.setPlainText(
                    f"[OK] Export erfolgreich nach: {filename}"
                )
            else:
                self.detail_view.setPlainText(f"[X] Export fehlgeschlagen!")

    def export_debug_report(self):
        """Export Working Memory als Debug-Report"""
        if not self.working_memory:
            return

        # Datei-Dialog
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Debug-Report",
            f"working_memory_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "Text Files (*.txt)",
        )

        if filename:
            success = self.working_memory.export_debug_report(filename)
            if success:
                self.detail_view.setPlainText(
                    f"[OK] Debug-Report erfolgreich exportiert nach: {filename}"
                )
            else:
                self.detail_view.setPlainText(f"[X] Export fehlgeschlagen!")

    def update_from_context_summary(self, context_summary: str):
        """
        DEPRECATED: Diese Methode wird nicht mehr benötigt.
        Verwende stattdessen set_working_memory() mit direkter WorkingMemory-Instanz.

        Args:
            context_summary: Context-Summary-String (nicht mehr verwendet)
        """
        # Placeholder für Backward-Compatibility
        # In der Praxis sollte direkt set_working_memory() aufgerufen werden


if __name__ == "__main__":
    """Demo/Test der Context-Visualisierung"""
    from PySide6.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)

    # Erstelle Demo-Working-Memory
    memory = WorkingMemory()

    # Füge Demo-Frames hinzu
    memory.push_context(
        ContextType.QUESTION,
        "Was ist ein Apfel?",
        entities=["apfel"],
        metadata={"user": "demo"},
    )
    memory.add_reasoning_state("fact_retrieval", "Suche Fakten über Apfel")
    memory.set_variable("topic", "apfel")

    memory.push_context(
        ContextType.CLARIFICATION,
        "Meinst du die Frucht oder das Unternehmen?",
        entities=["apfel", "frucht", "unternehmen"],
    )
    memory.add_reasoning_state("clarification_request", "Frage nach Klärung")

    memory.push_context(
        ContextType.DEFINITION,
        "Ein Apfel ist eine Frucht",
        entities=["apfel", "frucht"],
    )
    memory.add_reasoning_state("learning", "Speichere IS_A Relation")

    # Erstelle Widget
    widget = ContextVisualizationWidget()
    widget.set_working_memory(memory)
    widget.resize(1000, 700)
    widget.setWindowTitle("Context Visualization - Demo")
    widget.show()

    sys.exit(app.exec())
