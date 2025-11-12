# component_5_linguistik_strukturen.py
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

# --- Enums für feste Kategorien ---


class MeaningPointCategory(Enum):
    GREETING = "Greeting"
    TASK = "Task"
    QUESTION = "Question"
    DEFINITION = "Definition"
    CONSTRAINT = "Constraint"
    FORMAT = "Format"
    EXAMPLE = "Example"
    META = "Meta"
    LANGUAGE = "Language"
    DOMAIN = "Domain"
    CONFIRMATION = "Confirmation"
    UNKNOWN = "Unknown"
    COMMAND = "Command"
    ARITHMETIC_QUESTION = (
        "ArithmeticQuestion"  # Arithmetische Fragen wie "Was ist 3 + 5?"
    )


class Modality(Enum):
    IMPERATIVE = "imperative"
    INTERROGATIVE = "interrogative"
    DECLARATIVE = "declarative"
    DESIDERATIVE = "desiderative"  # Wunsch


class Polarity(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


class Priority(Enum):
    """Veraltet, Priorität wird durch die Reihenfolge der SubGoals im Plan abgebildet."""

    MUST = "Must"
    SHOULD = "Should"
    COULD = "Could"


# --- Goal Type Enum (muss vor Goal-Klasse definiert sein) ---


class GoalType(Enum):
    """Definiert die übergeordnete Absicht eines Plans."""

    ANSWER_QUESTION = "Frage beantworten"
    LEARN_KNOWLEDGE = "Wissen lernen"
    PERFORM_TASK = "Aufgabe ausführen"
    PERFORM_CALCULATION = (
        "Berechnung durchführen"  # Phase Mathematik: Arithmetische Berechnungen
    )
    GENERIC_RESPONSE = "Allgemein antworten"
    CLARIFY_INTENT = "Absicht klären"  # Neu: Für niedrige Confidence
    READ_DOCUMENT = "Dokument lesen"  # Phase 3: Datei-Ingestion
    UNKNOWN = "Unbekannt"


# --- Datenklassen ---


@dataclass
class MeaningPoint:
    """Repräsentiert einen einzelnen, extrahierten Bedeutungspunkt aus dem Input."""

    id: str
    category: MeaningPointCategory
    cue: str  # Das auslösende Wort/Lemma
    text_span: str
    modality: Modality
    polarity: Polarity
    confidence: float
    span_offsets: Optional[List[int]] = field(default_factory=list)
    arguments: Dict[str, Any] = field(default_factory=dict)
    source_rules: List[str] = field(default_factory=list)


@dataclass
class Goal:
    """Veraltet, wird durch MainGoal und SubGoal ersetzt."""

    id: str
    type: GoalType
    description: str
    must_should_could: Priority
    confidence: float
    priority_score: float
    sources: List[str] = field(
        default_factory=list
    )  # IDs der verursachenden MeaningPoints
    dependencies: List[str] = field(default_factory=list)  # IDs anderer Goals
    constraints: List[str] = field(default_factory=list)
    trace_rules: List[str] = field(default_factory=list)


@dataclass
class Trace:
    """Ein einzelner Eintrag im Trace-Log zur Nachverfolgung."""

    rule_id: str
    description: str
    contribution: str


@dataclass
class OutputEnvelope:
    """Der finale Container für das Ergebnis der gesamten linguistischen Analyse."""

    language: str = "de"
    input_hash: Optional[str] = None
    meaning_points: List[MeaningPoint] = field(default_factory=list)
    goals: List[Goal] = field(default_factory=list)
    goal_order: List[str] = field(default_factory=list)  # Geordnete Liste von Goal-IDs
    trace_summary: List[Trace] = field(default_factory=list)


# --- NEUE DATENSTRUKTUREN FÜR DIE PLANUNGS-ARCHITEKTUR ---


class GoalStatus(Enum):
    """Repräsentiert den Ausführungsstatus eines Ziels oder Unterziels."""

    PENDING = "Ausstehend"
    IN_PROGRESS = "In Bearbeitung"
    SUCCESS = "Erfolgreich"
    FAILED = "Fehlgeschlagen"


@dataclass
class SubGoal:
    """Repräsentiert einen einzelnen, atomaren Schritt in einem Ausführungsplan."""

    description: str
    id: str = field(default_factory=lambda: f"sg-{uuid.uuid4().hex[:8]}")
    status: GoalStatus = GoalStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(
        default_factory=dict
    )  # Für orchestrierte Sub-Goals


@dataclass
class MainGoal:
    """Kapselt einen vollständigen Plan zur Bearbeitung einer Nutzeranfrage."""

    type: GoalType
    description: str
    id: str = field(default_factory=lambda: f"mg-{uuid.uuid4().hex[:8]}")
    status: GoalStatus = GoalStatus.PENDING
    sub_goals: List[SubGoal] = field(default_factory=list)


# --- PHASE 4: KONTEXT-MANAGEMENT ---


class ContextAction(Enum):
    """Definiert die erwartete Aktion für die nächste Benutzereingabe."""

    NONE = "none"  # Kein spezieller Kontext
    ERWARTE_BEISPIELSATZ = (
        "erwarte_beispielsatz"  # Erwartet Definition nach Wissenslücke
    )
    ERWARTE_BESTAETIGUNG = "erwarte_bestaetigung"  # Erwartet Ja/Nein nach Confirmation
    ERWARTE_FEEDBACK_ZU_CLARIFICATION = "erwarte_feedback_zu_clarification"  # PHASE 5.1: Erwartet Lern-Feedback nach Clarification
    ERWARTE_TYPO_KLARSTELLUNG = "erwarte_typo_klarstellung"  # Pattern Recognition: Erwartet Korrektur bei Tippfehler
    ERWARTE_BEFEHL_BESTAETIGUNG = "erwarte_befehl_bestaetigung"  # Command Suggestions: Erwartet Bestätigung des Befehlsvorschlags


@dataclass
class ContextSnapshot:
    """
    Ein Snapshot eines Konversationskontexts zu einem bestimmten Zeitpunkt.

    PHASE 2 (Multi-Turn): Ermöglicht das Speichern und Wiederherstellen
    von Kontext-Zuständen für verschachtelte Dialoge.

    Attributes:
        snapshot_id: Eindeutige ID für diesen Snapshot
        aktion: Die erwartete Aktion zum Zeitpunkt des Snapshots
        thema: Das Hauptthema der Konversation
        plan_zur_ausfuehrung: Geplanter Plan (falls vorhanden)
        original_intent: Ursprünglicher MeaningPoint
        metadata: Zusätzliche Metadaten
        timestamp: Zeitstempel der Snapshot-Erstellung
        entities: Liste erwähnter Entitäten
        parent_snapshot_id: ID des vorherigen Snapshots (für verschachtelte Kontexte)
    """

    snapshot_id: str
    aktion: ContextAction
    thema: Optional[str] = None
    plan_zur_ausfuehrung: Optional[MainGoal] = None
    original_intent: Optional[MeaningPoint] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: str(uuid.uuid4()))
    entities: List[str] = field(default_factory=list)
    parent_snapshot_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialisiert den Snapshot zu einem Dictionary."""
        return {
            "snapshot_id": self.snapshot_id,
            "aktion": self.aktion.value,
            "thema": self.thema,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "entities": self.entities,
            "parent_snapshot_id": self.parent_snapshot_id,
        }


@dataclass
class KaiContext:
    """
    Formalisiert den Konversationskontext für Multi-Turn-Interaktionen.

    Phase 4: Ersetzt das Dictionary-basierte `kontext_fuer_naechste_eingabe`
    mit einer typsicheren, strukturierten Datenklasse.

    PHASE 2 (Multi-Turn): Erweitert um History-Stack für Session-basierte
    Kontext-Persistenz über mehrere Fragen hinweg.

    Attributes:
        aktion: Die erwartete Aktion (Beispielsatz, Bestätigung, etc.)
        thema: Das Hauptthema der Konversation (z.B. ein unbekanntes Wort)
        plan_zur_ausfuehrung: Der Plan, der nach Bestätigung ausgeführt werden soll
        original_intent: Der ursprüngliche MeaningPoint vor Bestätigung
        metadata: Zusätzliche flexible Metadaten
        history: Stack von Context-Snapshots (für Multi-Turn)
        max_history_size: Maximale Anzahl an Snapshots
        entities_in_session: Alle erwähnten Entitäten in dieser Session
    """

    aktion: ContextAction = ContextAction.NONE
    thema: Optional[str] = None
    plan_zur_ausfuehrung: Optional[MainGoal] = None
    original_intent: Optional[MeaningPoint] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # PHASE 2: Multi-Turn-Dialog-Erweiterungen
    history: List[ContextSnapshot] = field(default_factory=list)
    max_history_size: int = 10
    entities_in_session: List[str] = field(default_factory=list)

    def is_active(self) -> bool:
        """Prüft, ob ein aktiver Kontext gesetzt ist."""
        return self.aktion != ContextAction.NONE

    def has_history(self) -> bool:
        """PHASE 2: Prüft, ob Kontext-History vorhanden ist."""
        return len(self.history) > 0

    def save_snapshot(self) -> str:
        """
        PHASE 2: Speichert den aktuellen Kontext-Zustand als Snapshot.

        Returns:
            Die ID des erstellten Snapshots
        """
        snapshot_id = f"ctx-{uuid.uuid4().hex[:8]}"
        parent_id = self.history[-1].snapshot_id if self.history else None

        snapshot = ContextSnapshot(
            snapshot_id=snapshot_id,
            aktion=self.aktion,
            thema=self.thema,
            plan_zur_ausfuehrung=self.plan_zur_ausfuehrung,
            original_intent=self.original_intent,
            metadata=self.metadata.copy(),
            entities=self.entities_in_session.copy(),
            parent_snapshot_id=parent_id,
        )

        self.history.append(snapshot)

        # Begrenze History-Größe
        if len(self.history) > self.max_history_size:
            self.history.pop(0)

        return snapshot_id

    def restore_snapshot(self, snapshot_id: str) -> bool:
        """
        PHASE 2: Stellt einen früheren Kontext-Zustand wieder her.

        Args:
            snapshot_id: ID des wiederherzustellenden Snapshots

        Returns:
            True wenn erfolgreich, False wenn Snapshot nicht gefunden
        """
        for snapshot in reversed(self.history):
            if snapshot.snapshot_id == snapshot_id:
                self.aktion = snapshot.aktion
                self.thema = snapshot.thema
                self.plan_zur_ausfuehrung = snapshot.plan_zur_ausfuehrung
                self.original_intent = snapshot.original_intent
                self.metadata = snapshot.metadata.copy()
                self.entities_in_session = snapshot.entities.copy()
                return True
        return False

    def get_last_snapshot(self) -> Optional[ContextSnapshot]:
        """PHASE 2: Holt den letzten Snapshot aus der History."""
        return self.history[-1] if self.history else None

    def add_entity(self, entity: str) -> None:
        """PHASE 2: Fügt eine Entität zur Session hinzu."""
        if entity and entity not in self.entities_in_session:
            self.entities_in_session.append(entity)

    def set_action(self, action: ContextAction) -> None:
        """Setzt die erwartete Aktion für die nächste Eingabe."""
        self.aktion = action

    def set_data(self, key: str, value: Any) -> None:
        """Speichert einen Wert in den Context-Metadaten."""
        self.metadata[key] = value

    def get_data(self, key: str, default: Any = None) -> Any:
        """Holt einen Wert aus den Context-Metadaten."""
        return self.metadata.get(key, default)

    def get_context_summary(self) -> str:
        """
        PHASE 2: Erstellt eine lesbare Zusammenfassung des Kontexts.

        Returns:
            Formatierte String-Darstellung für UI
        """
        if not self.is_active() and not self.has_history():
            return ""

        parts = []

        if self.thema:
            parts.append(f"Thema: {self.thema}")

        if self.aktion != ContextAction.NONE:
            action_desc = {
                ContextAction.ERWARTE_BEISPIELSATZ: "Erwarte Definition",
                ContextAction.ERWARTE_BESTAETIGUNG: "Erwarte Bestätigung",
                ContextAction.ERWARTE_FEEDBACK_ZU_CLARIFICATION: "Erwarte Feedback",
                ContextAction.ERWARTE_TYPO_KLARSTELLUNG: "Erwarte Tippfehler-Korrektur",
            }
            parts.append(action_desc.get(self.aktion, self.aktion.value))

        if self.entities_in_session:
            entities_str = ", ".join(self.entities_in_session[:3])
            if len(self.entities_in_session) > 3:
                entities_str += f" (+{len(self.entities_in_session) - 3} mehr)"
            parts.append(f"Entitäten: {entities_str}")

        return " | ".join(parts) if parts else ""

    def clear(self) -> None:
        """Setzt den Kontext zurück auf Standardwerte (behält History)."""
        # Speichere Snapshot vor dem Löschen
        if self.is_active():
            self.save_snapshot()

        self.aktion = ContextAction.NONE
        self.thema = None
        self.plan_zur_ausfuehrung = None
        self.original_intent = None
        self.metadata.clear()

    def clear_all(self) -> None:
        """PHASE 2: Löscht Kontext UND History komplett."""
        self.clear()
        self.history.clear()
        self.entities_in_session.clear()
