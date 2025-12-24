"""
production_system_tab.py

Tab for Production System visualization and statistics (PHASE 8.1 + PHASE 9).
Shows all production rules, their categories, statistics, and Neo4j sync status.
"""

import logging

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class ProductionSystemTab(QWidget):
    """Tab fuer Production System Visualisierung und Statistiken (PHASE 8.1 + PHASE 9)"""

    settings_changed = Signal(dict)

    def __init__(self, parent=None, netzwerk=None):
        """
        Initialisiert den Production System Tab.

        Args:
            parent: Parent Widget
            netzwerk: Optional KonzeptNetzwerk fuer Neo4j-Stats (PHASE 9)
        """
        super().__init__(parent)

        self.netzwerk = netzwerk  # PHASE 9: Neo4j Integration
        self.rules = []
        self.neo4j_stats = {}  # PHASE 9: Stats aus Neo4j

        self.load_production_rules()

        self.init_ui()

    def load_production_rules(self):
        """Laedt alle verfuegbaren Produktionsregeln."""
        try:
            from component_54_production_system import (
                create_all_content_selection_rules,
                create_all_lexicalization_rules,
                create_all_discourse_management_rules,
                create_all_syntactic_realization_rules,
            )

            # Sammle alle Regeln
            self.rules.extend(create_all_content_selection_rules())
            self.rules.extend(create_all_lexicalization_rules())
            self.rules.extend(create_all_discourse_management_rules())
            self.rules.extend(create_all_syntactic_realization_rules())

            logger.info(f"Loaded {len(self.rules)} production rules for visualization")

            # PHASE 9: Synchronisiere Stats aus Neo4j
            if self.netzwerk:
                self.sync_neo4j_stats()

        except ImportError as e:
            logger.warning(f"Could not load production rules: {e}")
            self.rules = []

    def sync_neo4j_stats(self):
        """
        Synchronisiert Stats aus Neo4j mit in-memory Regeln (PHASE 9).
        """
        if not self.netzwerk:
            return

        try:
            all_rules_data = self.netzwerk.get_all_production_rules()

            # Erstelle Dict fuer schnellen Lookup
            self.neo4j_stats = {rd["name"]: rd for rd in all_rules_data}

            # Synchronisiere Stats zu in-memory Regeln
            for rule in self.rules:
                if rule.name in self.neo4j_stats:
                    stats = self.neo4j_stats[rule.name]
                    rule.application_count = stats["application_count"]
                    rule.success_count = stats["success_count"]
                    if stats["last_applied"]:
                        from datetime import datetime
                        rule.last_applied = datetime.fromtimestamp(
                            stats["last_applied"] / 1000
                        )

            logger.info(
                f"Synchronized Neo4j stats for {len(self.neo4j_stats)} rules"
            )
        except Exception as e:
            logger.error(f"Failed to sync Neo4j stats: {e}", exc_info=True)

    def init_ui(self):
        layout = QVBoxLayout(self)

        # === Header: Statistiken ===
        stats_group = QGroupBox("Statistiken")
        stats_layout = QVBoxLayout()

        # PHASE 9: Horizontales Layout fuer Stats + Refresh Button
        stats_header_layout = QHBoxLayout()

        # Kategorien zaehlen
        category_counts = {}
        total_applications = 0
        total_successes = 0
        for rule in self.rules:
            cat = rule.category.value
            category_counts[cat] = category_counts.get(cat, 0) + 1
            total_applications += rule.application_count
            total_successes += rule.success_count

        stats_text = f"Geladene Regeln: {len(self.rules)}\n"
        stats_text += f"Gesamt-Anwendungen: {total_applications}\n"
        stats_text += f"Gesamt-Erfolge: {total_successes}\n\n"
        stats_text += "Nach Kategorie:\n"
        for cat, count in sorted(category_counts.items()):
            stats_text += f"  * {cat}: {count} Regeln\n"

        self.stats_label = QLabel(stats_text)
        self.stats_label.setStyleSheet(
            "color: #3498db; font-size: 12px; font-family: 'Courier New', monospace;"
        )

        # PHASE 9: Refresh Button
        self.refresh_button = QPushButton("Refresh Stats")
        self.refresh_button.setToolTip("Laedt Statistiken aus Neo4j neu")
        self.refresh_button.setMaximumWidth(150)
        self.refresh_button.clicked.connect(self.refresh_stats)

        # Neo4j Status Indicator
        self.neo4j_status_label = QLabel()
        if self.netzwerk:
            self.neo4j_status_label.setText("[OK] Neo4j verbunden")
            self.neo4j_status_label.setStyleSheet("color: #2ecc71; font-size: 11px;")
        else:
            self.neo4j_status_label.setText("[!] Neo4j nicht verbunden (Stats nicht persistiert)")
            self.neo4j_status_label.setStyleSheet("color: #e67e22; font-size: 11px;")
            self.refresh_button.setEnabled(False)

        stats_header_layout.addWidget(self.stats_label)
        stats_header_layout.addStretch()
        stats_header_layout.addWidget(self.refresh_button)

        stats_layout.addLayout(stats_header_layout)
        stats_layout.addWidget(self.neo4j_status_label)
        stats_group.setLayout(stats_layout)

        # === Splitter: Liste (links) + Details (rechts) ===
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Liste aller Regeln
        list_widget_container = QGroupBox("Regeln")
        list_layout = QVBoxLayout()

        self.rule_list = QListWidget()
        self.rule_list.setAlternatingRowColors(True)
        self.rule_list.itemClicked.connect(self.show_rule_details)

        # Regeln nach Kategorie sortieren und hinzufuegen
        sorted_rules = sorted(self.rules, key=lambda r: (r.category.value, r.name))
        for rule in sorted_rules:
            # Format: "[CATEGORY] RuleName (util=0.9, spec=1.0)"
            item_text = (
                f"[{rule.category.value.upper()}] {rule.name} "
                f"(util={rule.utility:.2f}, spec={rule.specificity:.2f})"
            )
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, rule)  # Store rule object
            self.rule_list.addItem(item)

        list_layout.addWidget(self.rule_list)
        list_widget_container.setLayout(list_layout)

        # Details-Anzeige
        details_widget_container = QGroupBox("Details")
        details_layout = QVBoxLayout()

        self.details_browser = QTextBrowser()
        self.details_browser.setOpenExternalLinks(False)
        self.details_browser.setHtml(
            "<i>Klicke auf eine Regel, um Details zu sehen.</i>"
        )

        details_layout.addWidget(self.details_browser)
        details_widget_container.setLayout(details_layout)

        # Splitter konfigurieren
        splitter.addWidget(list_widget_container)
        splitter.addWidget(details_widget_container)
        splitter.setStretchFactor(0, 1)  # Liste: 40%
        splitter.setStretchFactor(1, 1)  # Details: 60%

        # === Hinweis ===
        info_group = QGroupBox("Hinweise")
        info_layout = QVBoxLayout()

        info_text = QLabel(
            "* Diese Regeln werden zur Laufzeit fuer Response-Generierung verwendet\n"
            "* Statistiken zeigen, wie oft jede Regel angewendet wurde\n"
            "* Utility x Specificity = Prioritaet fuer Conflict Resolution"
        )
        info_text.setWordWrap(True)
        info_text.setStyleSheet("color: #95a5a6; font-size: 11px;")

        info_layout.addWidget(info_text)
        info_group.setLayout(info_layout)

        # Layout zusammensetzen
        layout.addWidget(stats_group)
        layout.addWidget(splitter)
        layout.addWidget(info_group)

    def show_rule_details(self, item):
        """Zeigt Details fuer ausgewaehlte Regel."""
        rule = item.data(Qt.ItemDataRole.UserRole)

        if rule is None:
            return

        # HTML-formatierte Details
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: 'Segoe UI', sans-serif; font-size: 13px; }}
                h2 {{ color: #3498db; margin-top: 5px; }}
                h3 {{ color: #2ecc71; margin-top: 15px; }}
                .field {{ margin: 5px 0; }}
                .label {{ font-weight: bold; color: #7f8c8d; }}
                .value {{ color: #ecf0f1; }}
                .category {{ color: #e74c3c; font-weight: bold; }}
                .stats {{ background-color: #2c3e50; padding: 10px; border-radius: 5px; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <h2>{rule.name}</h2>

            <div class="field">
                <span class="label">Kategorie:</span>
                <span class="category">{rule.category.value.upper()}</span>
            </div>

            <div class="field">
                <span class="label">Utility:</span>
                <span class="value">{rule.utility:.2f}</span>
            </div>

            <div class="field">
                <span class="label">Specificity:</span>
                <span class="value">{rule.specificity:.2f}</span>
            </div>

            <div class="field">
                <span class="label">Prioritaet (Utility x Specificity):</span>
                <span class="value">{rule.get_priority():.2f}</span>
            </div>

            <h3>Statistiken</h3>
            <div class="stats">
                <div class="field">
                    <span class="label">Anwendungen:</span>
                    <span class="value">{rule.application_count}</span>
                </div>
                <div class="field">
                    <span class="label">Erfolge:</span>
                    <span class="value">{rule.success_count}</span>
                </div>
                <div class="field">
                    <span class="label">Zuletzt angewendet:</span>
                    <span class="value">{rule.last_applied.strftime('%Y-%m-%d %H:%M:%S') if rule.last_applied else 'Nie'}</span>
                </div>
            </div>

            <h3>Metadaten</h3>
            <div class="field">
                <span class="label">Beschreibung:</span>
                <span class="value">{rule.metadata.get('description', 'Keine Beschreibung verfuegbar')}</span>
            </div>
        </body>
        </html>
        """

        self.details_browser.setHtml(html)

    def refresh_stats(self):
        """
        Laedt Stats aus Neo4j neu und aktualisiert die UI (PHASE 9).
        """
        if not self.netzwerk:
            return

        # Synchronisiere Stats
        self.sync_neo4j_stats()

        # Aktualisiere Statistiken-Label
        category_counts = {}
        total_applications = 0
        total_successes = 0
        for rule in self.rules:
            cat = rule.category.value
            category_counts[cat] = category_counts.get(cat, 0) + 1
            total_applications += rule.application_count
            total_successes += rule.success_count

        stats_text = f"Geladene Regeln: {len(self.rules)}\n"
        stats_text += f"Gesamt-Anwendungen: {total_applications}\n"
        stats_text += f"Gesamt-Erfolge: {total_successes}\n\n"
        stats_text += "Nach Kategorie:\n"
        for cat, count in sorted(category_counts.items()):
            stats_text += f"  * {cat}: {count} Regeln\n"

        self.stats_label.setText(stats_text)

        # Aktualisiere Rule List (neues Label mit neuen Stats)
        self.rule_list.clear()
        sorted_rules = sorted(self.rules, key=lambda r: (r.category.value, r.name))
        for rule in sorted_rules:
            item_text = (
                f"[{rule.category.value.upper()}] {rule.name} "
                f"(util={rule.utility:.2f}, spec={rule.specificity:.2f}, "
                f"apps={rule.application_count})"
            )
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, rule)
            self.rule_list.addItem(item)

        logger.info("Production System Tab: Stats refreshed")

    def get_settings(self) -> dict:
        """Gibt aktuelle Einstellungen zurueck"""
        return {}  # Keine persistenten Settings fuer Production System Tab

    def apply_settings(self):
        """Speichert Einstellungen"""
        pass  # Keine Settings zu speichern
