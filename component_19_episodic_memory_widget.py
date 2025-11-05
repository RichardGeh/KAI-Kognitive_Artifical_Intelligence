"""
component_19_episodic_memory_widget.py

Interactive Episodic Memory Visualization Widget for KAI

Provides a PySide6-based timeline visualization for episodic memory queries
with support for filtering, detailed inspection, and export functionality.

Features:
- Timeline view showing learning and inference episodes
- Filter by episode type (Learning/Inference)
- Filter by topic/content search
- Detailed episode information panel
- Export to JSON
- Timeline visualization export
"""

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QLabel,
    QLineEdit,
    QComboBox,
    QTextEdit,
    QSplitter,
    QFileDialog,
    QHeaderView,
    QGroupBox,
)
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QColor, QFont
from typing import List, Dict, Any
import json
from datetime import datetime


class EpisodicMemoryWidget(QWidget):
    """
    Widget zur Visualisierung und Exploration des episodischen Gedächtnisses.

    Zeigt eine chronologische Timeline aller Lern- und Inferenz-Episoden mit
    Filter- und Export-Funktionen.
    """

    # Signal emitted when user requests episodes
    episodes_requested = Signal(str, str)  # (topic, episode_type)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.episodes: List[Dict[str, Any]] = []
        self.filtered_episodes: List[Dict[str, Any]] = []
        self._init_ui()

    def _init_ui(self):
        """Initialisiert die UI-Komponenten"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # === Filter-Panel (oben) ===
        filter_group = QGroupBox("Filter & Suche")
        filter_layout = QHBoxLayout()

        # Themen-Filter
        filter_layout.addWidget(QLabel("Thema:"))
        self.topic_filter = QLineEdit()
        self.topic_filter.setPlaceholderText("Thema filtern (leer = alle)")
        self.topic_filter.textChanged.connect(self._apply_filters)
        filter_layout.addWidget(self.topic_filter, stretch=2)

        # Typ-Filter
        filter_layout.addWidget(QLabel("Typ:"))
        self.type_filter = QComboBox()
        self.type_filter.addItems(
            [
                "Alle Typen",
                "Learning (Lernen)",
                "Inference (Schlussfolgerung)",
                "Ingestion (Text-Verarbeitung)",
                "Definition (Fakten-Speicherung)",
                "Pattern Learning (Muster-Lernen)",
            ]
        )
        self.type_filter.currentTextChanged.connect(self._apply_filters)
        filter_layout.addWidget(self.type_filter, stretch=1)

        # Reset-Button
        reset_btn = QPushButton("Filter zurücksetzen")
        reset_btn.clicked.connect(self._reset_filters)
        filter_layout.addWidget(reset_btn)

        filter_group.setLayout(filter_layout)
        main_layout.addWidget(filter_group)

        # === Splitter für Timeline + Detail-View ===
        splitter = QSplitter(Qt.Orientation.Vertical)

        # === Timeline-Tabelle ===
        self.timeline_table = QTableWidget()
        self.timeline_table.setColumnCount(5)
        self.timeline_table.setHorizontalHeaderLabels(
            ["Zeitstempel", "Typ", "Thema/Inhalt", "Fakten", "Konfidenz"]
        )

        # Tabellen-Einstellungen
        header = self.timeline_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)

        self.timeline_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.timeline_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.timeline_table.setAlternatingRowColors(True)
        self.timeline_table.itemSelectionChanged.connect(self._on_episode_selected)

        splitter.addWidget(self.timeline_table)

        # === Detail-View ===
        detail_group = QGroupBox("Episoden-Details")
        detail_layout = QVBoxLayout()

        self.detail_view = QTextEdit()
        self.detail_view.setReadOnly(True)
        self.detail_view.setPlaceholderText(
            "Wähle eine Episode aus der Timeline, um Details anzuzeigen."
        )
        detail_layout.addWidget(self.detail_view)

        detail_group.setLayout(detail_layout)
        splitter.addWidget(detail_group)

        # Splitter-Verhältnis: 60% Timeline, 40% Details
        splitter.setStretchFactor(0, 6)
        splitter.setStretchFactor(1, 4)

        main_layout.addWidget(splitter)

        # === Export-Buttons (unten) ===
        export_layout = QHBoxLayout()

        self.stats_label = QLabel("Keine Episoden geladen")
        self.stats_label.setStyleSheet("color: #7f8c8d; font-style: italic;")
        export_layout.addWidget(self.stats_label, stretch=1)

        export_json_btn = QPushButton("📄 Als JSON exportieren")
        export_json_btn.clicked.connect(self._export_json)
        export_layout.addWidget(export_json_btn)

        export_timeline_btn = QPushButton("📊 Timeline exportieren")
        export_timeline_btn.clicked.connect(self._export_timeline)
        export_timeline_btn.setEnabled(
            False
        )  # Placeholder für zukünftige Implementierung
        export_layout.addWidget(export_timeline_btn)

        main_layout.addLayout(export_layout)

        # Styling
        self.setStyleSheet(
            """
            QGroupBox {
                font-weight: bold;
                border: 1px solid #bdc3c7;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QTableWidget {
                border: 1px solid #bdc3c7;
                border-radius: 5px;
                background-color: #ecf0f1;
            }
            QTableWidget::item:selected {
                background-color: #3498db;
                color: white;
            }
            QTableWidget::item:hover {
                background-color: #d5dbdb;
            }
        """
        )

    @Slot(list)
    def update_episodes(self, episodes: List[Dict[str, Any]]):
        """
        Aktualisiert die angezeigten Episoden.

        Args:
            episodes: Liste von Episode-Dictionaries aus KonzeptNetzwerkMemory
        """
        self.episodes = episodes
        self.filtered_episodes = episodes.copy()
        self._apply_filters()
        self._update_stats()

    def _apply_filters(self):
        """Wendet aktuelle Filter auf die Episoden-Liste an"""
        topic_query = self.topic_filter.text().strip().lower()
        type_selection = self.type_filter.currentText()

        # Typ-Mapping
        type_mapping = {
            "Alle Typen": None,
            "Learning (Lernen)": "learning",
            "Inference (Schlussfolgerung)": "inference",
            "Ingestion (Text-Verarbeitung)": "ingestion",
            "Definition (Fakten-Speicherung)": "definition",
            "Pattern Learning (Muster-Lernen)": "pattern_learning",
        }

        target_type = type_mapping.get(type_selection)

        # Filtere Episoden
        self.filtered_episodes = []
        for episode in self.episodes:
            # Typ-Filter
            if target_type and episode.get("type", "").lower() != target_type:
                continue

            # Themen-Filter (sucht in content und learned_facts)
            if topic_query:
                content = episode.get("content", "").lower()
                facts = str(episode.get("learned_facts", [])).lower()
                query_text = episode.get("query", "").lower()

                if (
                    topic_query not in content
                    and topic_query not in facts
                    and topic_query not in query_text
                ):
                    continue

            self.filtered_episodes.append(episode)

        self._populate_table()
        self._update_stats()

    def _populate_table(self):
        """Füllt die Timeline-Tabelle mit gefilterten Episoden"""
        self.timeline_table.setRowCount(0)

        for episode in self.filtered_episodes:
            row = self.timeline_table.rowCount()
            self.timeline_table.insertRow(row)

            # Zeitstempel formatieren
            timestamp = episode.get("timestamp")
            if timestamp:
                # Neo4j timestamp ist in Millisekunden seit Epoch
                try:
                    dt = datetime.fromtimestamp(timestamp / 1000.0)
                    time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                except:
                    time_str = str(timestamp)
            else:
                time_str = "Unbekannt"

            # Typ
            episode_type = episode.get("type", "unknown")
            type_display = self._format_type(episode_type)

            # Inhalt/Thema
            if "content" in episode:
                content = episode["content"][:100]
                if len(episode["content"]) > 100:
                    content += "..."
            elif "query" in episode:
                content = episode["query"][:100]
                if len(episode["query"]) > 100:
                    content += "..."
            else:
                content = "Keine Beschreibung"

            # Fakten-Anzahl
            learned_facts = episode.get("learned_facts", [])
            # Filtere None-Werte aus der Liste
            learned_facts = [f for f in learned_facts if f is not None]
            facts_count = len(learned_facts)

            # Konfidenz (falls vorhanden)
            confidence = episode.get("confidence", episode.get("final_confidence"))
            if confidence is not None:
                conf_str = f"{confidence:.0%}"
            else:
                conf_str = "—"

            # Zeilen-Items erstellen
            items = [
                QTableWidgetItem(time_str),
                QTableWidgetItem(type_display),
                QTableWidgetItem(content),
                QTableWidgetItem(str(facts_count)),
                QTableWidgetItem(conf_str),
            ]

            # Styling basierend auf Typ
            color = self._get_type_color(episode_type)
            for i, item in enumerate(items):
                if i == 1:  # Typ-Spalte einfärben
                    item.setBackground(color)
                    item.setForeground(QColor("#ffffff"))
                    font = QFont()
                    font.setBold(True)
                    item.setFont(font)
                self.timeline_table.setItem(row, i, item)

        # Sortiere nach Zeitstempel (neueste zuerst)
        self.timeline_table.sortItems(0, Qt.SortOrder.DescendingOrder)

    def _format_type(self, episode_type: str) -> str:
        """Formatiert Episode-Typ für Anzeige"""
        type_names = {
            "learning": "Lernen",
            "inference": "Inferenz",
            "ingestion": "Ingestion",
            "definition": "Definition",
            "pattern_learning": "Muster",
            "forward_chaining": "Vorwärts",
            "backward_chaining": "Rückwärts",
            "graph_traversal": "Graph",
            "abductive": "Abduktiv",
            "hybrid": "Hybrid",
        }
        return type_names.get(episode_type.lower(), episode_type.capitalize())

    def _get_type_color(self, episode_type: str) -> QColor:
        """Gibt Farbe für Episode-Typ zurück"""
        colors = {
            "learning": QColor("#27ae60"),  # Grün
            "inference": QColor("#3498db"),  # Blau
            "ingestion": QColor("#9b59b6"),  # Lila
            "definition": QColor("#1abc9c"),  # Türkis
            "pattern_learning": QColor("#f39c12"),  # Orange
            "forward_chaining": QColor("#3498db"),
            "backward_chaining": QColor("#2980b9"),
            "graph_traversal": QColor("#8e44ad"),
            "abductive": QColor("#e74c3c"),
            "hybrid": QColor("#95a5a6"),
        }
        return colors.get(episode_type.lower(), QColor("#7f8c8d"))

    def _on_episode_selected(self):
        """Handler für Episode-Auswahl in der Tabelle"""
        selected_rows = self.timeline_table.selectedItems()
        if not selected_rows:
            self.detail_view.clear()
            return

        # Hole die ausgewählte Zeile
        row = self.timeline_table.currentRow()
        if row < 0 or row >= len(self.filtered_episodes):
            return

        episode = self.filtered_episodes[row]
        self._display_episode_details(episode)

    def _display_episode_details(self, episode: Dict[str, Any]):
        """Zeigt detaillierte Informationen über eine Episode"""
        details = []

        # Header
        details.append(
            f"<h2 style='color: #2c3e50;'>[INFO] Episode: {episode.get('episode_id', 'N/A')[:8]}</h2>"
        )
        details.append("<hr>")

        # Typ
        episode_type = episode.get("type", episode.get("inference_type", "unknown"))
        details.append(f"<p><b>Typ:</b> {self._format_type(episode_type)}</p>")

        # Zeitstempel
        timestamp = episode.get("timestamp")
        if timestamp:
            try:
                dt = datetime.fromtimestamp(timestamp / 1000.0)
                time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                details.append(f"<p><b>Zeitpunkt:</b> {time_str}</p>")
            except:
                pass

        # Content oder Query
        if "content" in episode:
            content = episode["content"]
            details.append(f"<p><b>Inhalt:</b></p>")
            details.append(
                f"<blockquote style='background-color: #ecf0f1; padding: 10px; border-left: 4px solid #3498db;'>{content}</blockquote>"
            )
        elif "query" in episode:
            query = episode["query"]
            details.append(f"<p><b>Abfrage:</b></p>")
            details.append(
                f"<blockquote style='background-color: #ecf0f1; padding: 10px; border-left: 4px solid #3498db;'>{query}</blockquote>"
            )

        # Gelernte Fakten
        learned_facts = episode.get("learned_facts", [])
        learned_facts = [f for f in learned_facts if f is not None]
        if learned_facts:
            details.append(f"<p><b>Gelernte Fakten ({len(learned_facts)}):</b></p>")
            details.append("<ul style='background-color: #ecf0f1; padding: 10px;'>")
            for fact in learned_facts[:10]:  # Max 10 anzeigen
                if isinstance(fact, dict):
                    subj = fact.get("subject", "?")
                    rel = fact.get("relation", "?")
                    obj = fact.get("object", "?")
                    details.append(
                        f"<li><code>{subj}</code> -> <b>{rel}</b> -> <code>{obj}</code></li>"
                    )
                else:
                    details.append(f"<li>{fact}</li>")
            if len(learned_facts) > 10:
                details.append(
                    f"<li><i>... und {len(learned_facts) - 10} weitere</i></li>"
                )
            details.append("</ul>")

        # Verwendete Fakten (für Inferenz-Episoden)
        used_facts_count = episode.get("used_facts_count", 0)
        if used_facts_count > 0:
            details.append(f"<p><b>Verwendete Fakten:</b> {used_facts_count}</p>")

        # Angewendete Regeln (für Inferenz-Episoden)
        applied_rules_count = episode.get("applied_rules_count", 0)
        if applied_rules_count > 0:
            details.append(f"<p><b>Angewendete Regeln:</b> {applied_rules_count}</p>")

        # Konfidenz
        confidence = episode.get("confidence", episode.get("final_confidence"))
        if confidence is not None:
            details.append(f"<p><b>Konfidenz:</b> {confidence:.0%}</p>")

        # Metadaten
        metadata = episode.get("metadata")
        if metadata:
            try:
                if isinstance(metadata, str):
                    metadata_dict = json.loads(metadata)
                else:
                    metadata_dict = metadata

                if metadata_dict:
                    details.append("<p><b>Metadaten:</b></p>")
                    details.append(
                        "<pre style='background-color: #ecf0f1; padding: 10px; font-size: 10px;'>"
                    )
                    details.append(
                        json.dumps(metadata_dict, indent=2, ensure_ascii=False)
                    )
                    details.append("</pre>")
            except:
                pass

        # Anzeigen
        html = "\n".join(details)
        self.detail_view.setHtml(html)

    def _update_stats(self):
        """Aktualisiert die Statistik-Anzeige"""
        total = len(self.episodes)
        filtered = len(self.filtered_episodes)

        if total == 0:
            self.stats_label.setText("Keine Episoden geladen")
        elif total == filtered:
            self.stats_label.setText(f"📊 {total} Episode{'n' if total != 1 else ''}")
        else:
            self.stats_label.setText(
                f"📊 {filtered} von {total} Episode{'n' if total != 1 else ''} (gefiltert)"
            )

    def _reset_filters(self):
        """Setzt alle Filter zurück"""
        self.topic_filter.clear()
        self.type_filter.setCurrentIndex(0)
        self._apply_filters()

    def _export_json(self):
        """Exportiert gefilterte Episoden als JSON"""
        if not self.filtered_episodes:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Episoden als JSON exportieren",
            f"episoden_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "JSON Files (*.json)",
        )

        if file_path:
            try:
                # Konvertiere Episoden für JSON-Export
                export_data = {
                    "export_timestamp": datetime.now().isoformat(),
                    "total_episodes": len(self.filtered_episodes),
                    "episodes": self.filtered_episodes,
                }

                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)

                self.stats_label.setText(
                    f"[SUCCESS] {len(self.filtered_episodes)} Episoden exportiert"
                )
            except Exception as e:
                self.stats_label.setText(f"[ERROR] Export fehlgeschlagen: {e}")

    def _export_timeline(self):
        """
        Exportiert Timeline-Visualisierung (Placeholder).

        TODO: Implementiere Timeline-Grafik mit matplotlib oder QGraphicsView
        """
        # Placeholder für zukünftige Implementierung

    @Slot()
    def clear(self):
        """Löscht alle angezeigten Episoden"""
        self.episodes.clear()
        self.filtered_episodes.clear()
        self.timeline_table.setRowCount(0)
        self.detail_view.clear()
        self._update_stats()
