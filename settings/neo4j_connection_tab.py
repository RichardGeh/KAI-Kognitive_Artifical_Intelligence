"""
neo4j_connection_tab.py

Tab for Neo4j database connection settings.
"""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from kai_config import get_config


class Neo4jConnectionTab(QWidget):
    """Tab fuer Neo4j-Verbindungseinstellungen"""

    settings_changed = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)

        # Lade aktuelle Einstellungen
        cfg = get_config()
        self.current_settings = {
            "neo4j_uri": cfg.get("neo4j_uri", "bolt://127.0.0.1:7687"),
            "neo4j_user": cfg.get("neo4j_user", "neo4j"),
            "neo4j_password": cfg.get("neo4j_password", "password"),
        }

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # === Connection Parameters ===
        connection_group = QGroupBox("Verbindungsparameter")
        connection_layout = QVBoxLayout()

        # URI
        uri_label = QLabel("Neo4j URI:")
        self.uri_edit = QLineEdit()
        self.uri_edit.setText(self.current_settings["neo4j_uri"])
        self.uri_edit.setPlaceholderText("bolt://127.0.0.1:7687")
        self.uri_edit.setToolTip("Format: bolt://host:port oder neo4j://host:port")

        # User
        user_label = QLabel("Benutzername:")
        self.user_edit = QLineEdit()
        self.user_edit.setText(self.current_settings["neo4j_user"])
        self.user_edit.setPlaceholderText("neo4j")

        # Password
        password_label = QLabel("Passwort:")
        self.password_edit = QLineEdit()
        self.password_edit.setText(self.current_settings["neo4j_password"])
        self.password_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.password_edit.setPlaceholderText("password")

        # Show Password Checkbox
        self.show_password_checkbox = QCheckBox("Passwort anzeigen")
        self.show_password_checkbox.stateChanged.connect(
            self.toggle_password_visibility
        )

        connection_layout.addWidget(uri_label)
        connection_layout.addWidget(self.uri_edit)
        connection_layout.addWidget(user_label)
        connection_layout.addWidget(self.user_edit)
        connection_layout.addWidget(password_label)
        connection_layout.addWidget(self.password_edit)
        connection_layout.addWidget(self.show_password_checkbox)
        connection_group.setLayout(connection_layout)

        # === Hinweis ===
        info_group = QGroupBox("Hinweise")
        info_layout = QVBoxLayout()

        info_text = QLabel(
            "* Aenderungen werden erst nach Neustart von KAI aktiv\n"
            "* Stelle sicher, dass Neo4j laeuft und die Zugangsdaten korrekt sind\n"
            "* Standard-Port: 7687 (bolt protocol)\n"
            "* Die Verbindung wird beim Start von KAI getestet"
        )
        info_text.setWordWrap(True)
        info_text.setStyleSheet("color: #3498db; font-size: 11px;")

        info_layout.addWidget(info_text)
        info_group.setLayout(info_layout)

        # Layout zusammensetzen
        layout.addWidget(connection_group)
        layout.addWidget(info_group)
        layout.addStretch()

    def toggle_password_visibility(self, state):
        """Schaltet Passwort-Sichtbarkeit um"""
        if state == 2:  # Qt.CheckState.Checked
            self.password_edit.setEchoMode(QLineEdit.EchoMode.Normal)
        else:
            self.password_edit.setEchoMode(QLineEdit.EchoMode.Password)

    def get_settings(self) -> dict:
        """Gibt aktuelle Einstellungen zurueck"""
        return {
            "neo4j_uri": self.uri_edit.text().strip(),
            "neo4j_user": self.user_edit.text().strip(),
            "neo4j_password": self.password_edit.text(),
        }

    def apply_settings(self):
        """Speichert Einstellungen"""
        new_settings = self.get_settings()
        self.current_settings = new_settings

        cfg = get_config()
        cfg.update(new_settings)

        self.settings_changed.emit(new_settings)
