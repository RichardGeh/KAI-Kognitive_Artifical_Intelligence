"""
kai_config.py

Zentrales Konfigurations-Management für KAI.
Speichert und lädt Einstellungen persistent in einer JSON-Datei.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from component_15_logging_config import get_logger

logger = get_logger(__name__)


# ============================================================================
# DEFAULT CONFIGURATION
# ============================================================================

DEFAULT_CONFIG = {
    # Wortverwendungs-Tracking
    "word_usage_tracking": True,
    "usage_similarity_threshold": 80,  # Prozent
    "context_window_size": 3,  # ±N Wörter
    "max_words_to_comma": 3,  # Max. Wörter bis Komma
    # Pattern Recognition
    "pattern_recognition_enabled": True,
    "min_word_occurrences_for_typo": 10,
    "min_sequence_count_for_prediction": 5,
    "typo_auto_correct_threshold": 0.85,
    "typo_ask_user_threshold": 0.60,
    "sequence_suggest_threshold": 0.70,
    "implication_auto_add_threshold": 0.75,
    "implication_ask_user_threshold": 0.50,
    # Pattern Matching Thresholds
    "prototype_novelty_threshold": 15.0,  # Euclidean distance for 384D embeddings
    "typo_min_threshold": 3,  # Adaptive: Minimum occurrences
    "typo_max_threshold": 10,  # Adaptive: Maximum occurrences
    "sequence_min_threshold": 2,  # Adaptive: Minimum sequence count
    "sequence_max_threshold": 5,  # Adaptive: Maximum sequence count
    # Confidence Thresholds (GoalPlanner)
    "confidence_low_threshold": 0.40,  # Below this: Ask for clarification
    "confidence_medium_threshold": 0.85,  # Below this: Request confirmation
    # Above this: Direct execution
    # Performance / Parallel Processing
    "parallel_processing_enabled": True,  # Enable/Disable parallel chunk processing
    "parallel_processing_max_workers": None,  # None = auto-detect (CPU cores * 2), or set specific number
    # Neo4j Connection
    "neo4j_uri": "bolt://127.0.0.1:7687",
    "neo4j_user": "neo4j",
    "neo4j_password": "password",
    # Logging (wird von settings_ui.py gesetzt)
    "console_log_level": "INFO",
    "file_log_level": "DEBUG",
    "performance_logging": True,
    # UI Settings
    "theme": "dark",  # "dark" or "light"
}


# ============================================================================
# CONFIGURATION MANAGER
# ============================================================================


class KaiConfig:
    """
    Singleton-Klasse für KAI-Konfiguration.

    Speichert Einstellungen persistent in 'kai_config.json' im Projektverzeichnis.
    """

    _instance: Optional["KaiConfig"] = None
    _config: Dict[str, Any] = {}
    _config_file: Path = Path(__file__).parent / "kai_config.json"

    def __new__(cls):
        """Singleton-Pattern: Nur eine Instanz erlauben"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        """Lädt Konfiguration aus JSON-Datei oder erstellt Default-Config"""
        if self._config_file.exists():
            try:
                with open(self._config_file, "r", encoding="utf-8") as f:
                    self._config = json.load(f)
                logger.info(
                    "Konfiguration geladen", extra={"file": str(self._config_file)}
                )

                # Merge mit Default-Config (falls neue Einstellungen hinzugefügt wurden)
                for key, value in DEFAULT_CONFIG.items():
                    if key not in self._config:
                        self._config[key] = value
                        logger.debug(
                            "Default-Wert ergänzt", extra={"key": key, "value": value}
                        )

            except Exception as e:
                logger.error(
                    "Fehler beim Laden der Konfiguration", extra={"error": str(e)}
                )
                self._config = DEFAULT_CONFIG.copy()
        else:
            logger.info("Keine Konfigurationsdatei gefunden, verwende Defaults")
            self._config = DEFAULT_CONFIG.copy()
            self._save_config()  # Erstelle initiale Config-Datei

    def _save_config(self):
        """Speichert aktuelle Konfiguration in JSON-Datei"""
        try:
            with open(self._config_file, "w", encoding="utf-8") as f:
                json.dump(self._config, f, indent=4, ensure_ascii=False)
            logger.debug(
                "Konfiguration gespeichert", extra={"file": str(self._config_file)}
            )
        except Exception as e:
            logger.error(
                "Fehler beim Speichern der Konfiguration", extra={"error": str(e)}
            )

    def get(self, key: str, default: Any = None) -> Any:
        """Gibt Konfigurations-Wert zurück"""
        return self._config.get(key, default)

    def set(self, key: str, value: Any):
        """Setzt Konfigurations-Wert und speichert"""
        self._config[key] = value
        self._save_config()
        logger.info("Konfiguration aktualisiert", extra={"key": key, "value": value})

    def update(self, settings: Dict[str, Any]):
        """Aktualisiert mehrere Konfigurations-Werte auf einmal"""
        self._config.update(settings)
        self._save_config()
        logger.info(
            "Konfiguration aktualisiert", extra={"updated_keys": list(settings.keys())}
        )

    def get_all(self) -> Dict[str, Any]:
        """Gibt gesamte Konfiguration zurück"""
        return self._config.copy()

    def reset_to_defaults(self):
        """Setzt Konfiguration auf Default-Werte zurück"""
        self._config = DEFAULT_CONFIG.copy()
        self._save_config()
        logger.warning("Konfiguration auf Defaults zurückgesetzt")

    # ========================================================================
    # CONVENIENCE PROPERTIES für häufig verwendete Einstellungen
    # ========================================================================

    @property
    def word_usage_tracking_enabled(self) -> bool:
        """Ist Wortverwendungs-Tracking aktiviert?"""
        return bool(self.get("word_usage_tracking", True))

    @property
    def usage_similarity_threshold(self) -> int:
        """Schwellenwert für Ähnlichkeit (Prozent)"""
        return int(self.get("usage_similarity_threshold", 80))

    @property
    def context_window_size(self) -> int:
        """Kontext-Fenster-Größe (±N Wörter)"""
        return int(self.get("context_window_size", 3))

    @property
    def max_words_to_comma(self) -> int:
        """Max. Wörter bis Komma für Kontext"""
        return int(self.get("max_words_to_comma", 3))

    @property
    def parallel_processing_enabled(self) -> bool:
        """Ist parallele Batch-Verarbeitung aktiviert?"""
        return bool(self.get("parallel_processing_enabled", True))

    @property
    def parallel_processing_max_workers(self) -> Optional[int]:
        """Maximale Anzahl paralleler Worker (None = auto-detect)"""
        return self.get("parallel_processing_max_workers", None)


# ============================================================================
# GLOBALE INSTANZ (für einfachen Import)
# ============================================================================

# Globale Singleton-Instanz
config = KaiConfig()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def get_config() -> KaiConfig:
    """Gibt die globale Config-Instanz zurück"""
    return config


def is_word_usage_tracking_enabled() -> bool:
    """Convenience-Funktion: Ist Wortverwendungs-Tracking aktiviert?"""
    return config.word_usage_tracking_enabled


if __name__ == "__main__":
    # Test-Code
    print("=== KAI Configuration Test ===\n")

    cfg = get_config()

    print("Aktuelle Konfiguration:")
    for key, value in cfg.get_all().items():
        print(f"  {key}: {value}")

    print("\nWortverwendungs-Tracking aktiviert?", cfg.word_usage_tracking_enabled)
    print("Ähnlichkeits-Schwellenwert:", cfg.usage_similarity_threshold, "%")
    print("Kontext-Fenster-Größe: ±", cfg.context_window_size, "Wörter")
    print("Max. Wörter bis Komma:", cfg.max_words_to_comma)

    # Test: Einstellung ändern
    print("\n=== Test: Einstellung ändern ===")
    cfg.set("usage_similarity_threshold", 90)
    print("Neuer Schwellenwert:", cfg.usage_similarity_threshold, "%")

    # Zurücksetzen
    print("\n=== Test: Auf Defaults zurücksetzen ===")
    cfg.reset_to_defaults()
    print("Schwellenwert nach Reset:", cfg.usage_similarity_threshold, "%")
