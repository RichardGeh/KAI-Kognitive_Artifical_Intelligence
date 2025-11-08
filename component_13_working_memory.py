"""
component_13_working_memory.py

Working Memory System für KAI
Verwaltet kurzfristige Kontext-Informationen und Reasoning-States während der Dialogverarbeitung.

Funktionen:
- Stack-basiertes Kontext-Management für verschachtelte Dialoge
- Speicherung von Zwischenschritten im Reasoning-Prozess
- Verwaltung von Variablen-Bindungen und temporären Fakten
- Kontext-Wiederherstellung bei Rückfragen und Follow-ups
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class ContextType(Enum):
    """Typen von Kontext-Einträgen"""

    QUESTION = "question"  # Beantwortung einer Frage
    DEFINITION = "definition"  # Lernen einer Definition
    PATTERN_LEARNING = "pattern_learning"  # Muster-Training
    INFERENCE = "inference"  # Logisches Schließen
    TEXT_INGESTION = "text_ingestion"  # Text-Verarbeitung
    CLARIFICATION = "clarification"  # Rückfrage/Klärung


@dataclass
class ReasoningState:
    """
    Repräsentiert einen Zwischenschritt im Reasoning-Prozess.
    Speichert alle relevanten Informationen für einen Reasoning-Schritt.
    """

    step_id: str
    step_type: str  # z.B. "fact_retrieval", "pattern_match", "inference"
    description: str
    data: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Konvertiere zu Dictionary für Serialisierung"""
        return {
            "step_id": self.step_id,
            "step_type": self.step_type,
            "description": self.description,
            "data": self.data,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ContextFrame:
    """
    Ein Kontext-Frame auf dem Stack.
    Repräsentiert einen Dialog-Kontext mit allen relevanten Informationen.
    """

    frame_id: str
    context_type: ContextType
    query: str
    entities: List[str] = field(default_factory=list)
    relations: Dict[str, Any] = field(default_factory=dict)
    reasoning_states: List[ReasoningState] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_access_time: datetime = field(default_factory=datetime.now)
    parent_frame_id: Optional[str] = None

    def add_reasoning_state(self, state: ReasoningState):
        """Füge einen Reasoning-Schritt hinzu"""
        self.reasoning_states.append(state)
        self.touch()  # Aktualisiere Access-Time

    def get_last_reasoning_state(self) -> Optional[ReasoningState]:
        """Hole den letzten Reasoning-Schritt"""
        return self.reasoning_states[-1] if self.reasoning_states else None

    def touch(self):
        """Aktualisiere last_access_time auf jetzt"""
        self.last_access_time = datetime.now()

    def is_idle(self, timeout_seconds: int) -> bool:
        """
        Prüfe ob dieser Frame länger als timeout_seconds inaktiv ist.

        Args:
            timeout_seconds: Timeout in Sekunden

        Returns:
            True wenn Frame idle ist, sonst False
        """
        idle_duration = (datetime.now() - self.last_access_time).total_seconds()
        return idle_duration > timeout_seconds

    def get_idle_duration(self) -> float:
        """
        Hole die Idle-Dauer in Sekunden.

        Returns:
            Idle-Dauer in Sekunden
        """
        return (datetime.now() - self.last_access_time).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        """Konvertiere zu Dictionary für Serialisierung"""
        return {
            "frame_id": self.frame_id,
            "context_type": self.context_type.value,
            "query": self.query,
            "entities": self.entities,
            "relations": self.relations,
            "reasoning_states": [rs.to_dict() for rs in self.reasoning_states],
            "variables": self.variables,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "last_access_time": self.last_access_time.isoformat(),
            "parent_frame_id": self.parent_frame_id,
        }


class WorkingMemory:
    """
    Stack-basiertes Working Memory System.

    Verwaltet einen Stack von Kontext-Frames für verschachtelte Dialoge
    und ermöglicht Zugriff auf Reasoning-States und temporäre Informationen.

    Beispiel-Verwendung:
        memory = WorkingMemory()

        # Neuen Kontext erstellen
        frame_id = memory.push_context(
            context_type=ContextType.QUESTION,
            query="Was ist ein Apfel?",
            entities=["apfel"]
        )

        # Reasoning-Schritte hinzufügen
        memory.add_reasoning_state(
            step_type="fact_retrieval",
            description="Suche Fakten über 'apfel'",
            data={"facts": {"IS_A": ["frucht"]}}
        )

        # Kontext abrufen
        current = memory.get_current_context()

        # Kontext entfernen
        memory.pop_context()
    """

    def __init__(self, max_stack_depth: int = 10):
        """
        Initialisiere Working Memory.

        Args:
            max_stack_depth: Maximale Tiefe des Kontext-Stacks
        """
        self.context_stack: List[ContextFrame] = []
        self.max_stack_depth = max_stack_depth
        self._frame_counter = 0

    def _generate_frame_id(self) -> str:
        """Generiere eine eindeutige Frame-ID"""
        self._frame_counter += 1
        return (
            f"frame_{self._frame_counter}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        )

    def push_context(
        self,
        context_type: ContextType,
        query: str,
        entities: Optional[List[str]] = None,
        relations: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Erstelle einen neuen Kontext-Frame und lege ihn auf den Stack.

        Args:
            context_type: Typ des Kontexts
            query: Die aktuelle Anfrage/Eingabe
            entities: Relevante Entitäten
            relations: Relevante Relationen
            metadata: Zusätzliche Metadaten

        Returns:
            Die Frame-ID des erstellten Kontexts

        Raises:
            ValueError: Wenn Stack-Limit erreicht ist
        """
        if len(self.context_stack) >= self.max_stack_depth:
            raise ValueError(f"Kontext-Stack Limit erreicht ({self.max_stack_depth})")

        frame_id = self._generate_frame_id()
        parent_id = self.context_stack[-1].frame_id if self.context_stack else None

        frame = ContextFrame(
            frame_id=frame_id,
            context_type=context_type,
            query=query,
            entities=entities or [],
            relations=relations or {},
            metadata=metadata or {},
            parent_frame_id=parent_id,
        )

        self.context_stack.append(frame)
        return frame_id

    def pop_context(self) -> Optional[ContextFrame]:
        """
        Entferne den obersten Kontext-Frame vom Stack.

        Returns:
            Der entfernte Frame, oder None wenn Stack leer ist
        """
        if self.context_stack:
            return self.context_stack.pop()
        return None

    def get_current_context(self) -> Optional[ContextFrame]:
        """
        Hole den aktuellen (obersten) Kontext-Frame.

        Returns:
            Der aktuelle Frame, oder None wenn Stack leer ist
        """
        return self.context_stack[-1] if self.context_stack else None

    def get_context_by_id(self, frame_id: str) -> Optional[ContextFrame]:
        """
        Hole einen spezifischen Kontext-Frame anhand seiner ID.

        Args:
            frame_id: Die Frame-ID

        Returns:
            Der Frame, oder None wenn nicht gefunden
        """
        for frame in self.context_stack:
            if frame.frame_id == frame_id:
                return frame
        return None

    def get_parent_context(self) -> Optional[ContextFrame]:
        """
        Hole den Parent-Frame des aktuellen Kontexts.

        Returns:
            Der Parent-Frame, oder None wenn nicht vorhanden
        """
        current = self.get_current_context()
        if not current or not current.parent_frame_id:
            return None
        return self.get_context_by_id(current.parent_frame_id)

    def add_reasoning_state(
        self,
        step_type: str,
        description: str,
        data: Optional[Dict[str, Any]] = None,
        confidence: float = 1.0,
    ) -> Optional[str]:
        """
        Füge einen Reasoning-Schritt zum aktuellen Kontext hinzu.

        Args:
            step_type: Typ des Reasoning-Schritts
            description: Beschreibung des Schritts
            data: Zusätzliche Daten
            confidence: Konfidenz-Wert (0.0 - 1.0)

        Returns:
            Die Step-ID, oder None wenn kein aktueller Kontext existiert
        """
        current = self.get_current_context()
        if not current:
            return None

        step_id = f"step_{len(current.reasoning_states) + 1}"
        state = ReasoningState(
            step_id=step_id,
            step_type=step_type,
            description=description,
            data=data or {},
            confidence=confidence,
        )

        current.add_reasoning_state(state)
        return step_id

    def get_reasoning_trace(self) -> List[ReasoningState]:
        """
        Hole alle Reasoning-Schritte des aktuellen Kontexts.

        Returns:
            Liste aller Reasoning-States
        """
        current = self.get_current_context()
        return current.reasoning_states if current else []

    def get_full_reasoning_trace(self) -> List[ReasoningState]:
        """
        Hole alle Reasoning-Schritte über den gesamten Stack hinweg.

        Returns:
            Liste aller Reasoning-States aller Frames
        """
        all_states = []
        for frame in self.context_stack:
            all_states.extend(frame.reasoning_states)
        return all_states

    def set_variable(self, key: str, value: Any):
        """
        Setze eine Variable im aktuellen Kontext.

        Args:
            key: Variablen-Name
            value: Variablen-Wert
        """
        current = self.get_current_context()
        if current:
            current.variables[key] = value
            current.touch()  # Aktualisiere Access-Time

    def get_variable(self, key: str, search_parent: bool = True) -> Optional[Any]:
        """
        Hole eine Variable aus dem aktuellen oder Parent-Kontext.

        Args:
            key: Variablen-Name
            search_parent: Auch in Parent-Frames suchen

        Returns:
            Der Variablen-Wert, oder None wenn nicht gefunden
        """
        # Suche im aktuellen Frame
        current = self.get_current_context()
        if current:
            if key in current.variables:
                current.touch()  # Aktualisiere Access-Time
                return current.variables[key]

        # Suche in Parent-Frames
        if search_parent:
            for frame in reversed(self.context_stack[:-1]):
                if key in frame.variables:
                    frame.touch()  # Aktualisiere Access-Time
                    return frame.variables[key]

        return None

    def get_all_entities(self) -> List[str]:
        """
        Hole alle Entitäten aus dem gesamten Stack.

        Returns:
            Liste aller Entitäten (dedupliziert)
        """
        all_entities = set()
        for frame in self.context_stack:
            all_entities.update(frame.entities)
        return list(all_entities)

    def add_entity(self, entity: str):
        """
        Füge eine Entität zum aktuellen Kontext hinzu.

        Args:
            entity: Die Entität
        """
        current = self.get_current_context()
        if current and entity not in current.entities:
            current.entities.append(entity)
            current.touch()  # Aktualisiere Access-Time

    def touch_current_context(self):
        """
        Aktualisiere last_access_time des aktuellen Kontext-Frames.

        Sollte aufgerufen werden bei jeder Interaktion mit dem Frame.
        """
        current = self.get_current_context()
        if current:
            current.touch()

    def get_idle_frames(self, timeout_seconds: int) -> List[ContextFrame]:
        """
        Finde alle Frames, die länger als timeout_seconds inaktiv sind.

        Args:
            timeout_seconds: Idle-Timeout in Sekunden

        Returns:
            Liste von idle Frames
        """
        idle_frames = []
        for frame in self.context_stack:
            if frame.is_idle(timeout_seconds):
                idle_frames.append(frame)
        return idle_frames

    def cleanup_idle_contexts(
        self, timeout_seconds: int, preserve_root: bool = True
    ) -> List[str]:
        """
        Entferne inaktive Kontext-Frames vom Stack.

        WICHTIG: Diese Methode entfernt Frames bottom-up (von unten nach oben),
        um Parent-Child-Beziehungen zu bewahren. Wenn ein Frame entfernt wird,
        werden auch alle seine Child-Frames entfernt.

        Args:
            timeout_seconds: Idle-Timeout in Sekunden
            preserve_root: Wenn True, wird der Root-Frame (unterster Frame) nie entfernt

        Returns:
            Liste der entfernten Frame-IDs
        """
        if not self.context_stack:
            return []

        removed_ids = []

        # Finde idle Frames
        idle_frames = self.get_idle_frames(timeout_seconds)

        if not idle_frames:
            return []

        # Wenn preserve_root=True, entferne Root-Frame aus Kandidaten
        if preserve_root and self.context_stack[0] in idle_frames:
            idle_frames.remove(self.context_stack[0])

        # Entferne Frames bottom-up (tiefster zuerst)
        # Sortiere nach Stack-Position (höchste zuerst = tiefer im Stack)
        frames_to_remove = sorted(
            idle_frames, key=lambda f: self.context_stack.index(f), reverse=True
        )

        for frame in frames_to_remove:
            # Prüfe ob Frame noch im Stack ist (könnte durch vorherigen Remove entfernt worden sein)
            if frame in self.context_stack:
                frame_index = self.context_stack.index(frame)

                # Entferne diesen Frame und alle darüber liegenden Frames
                # (um Parent-Child-Konsistenz zu bewahren)
                removed = self.context_stack[frame_index:]
                self.context_stack = self.context_stack[:frame_index]

                removed_ids.extend([f.frame_id for f in removed])

        return removed_ids

    def get_idle_status(self) -> Dict[str, Any]:
        """
        Hole Idle-Status aller Frames.

        Returns:
            Dictionary mit Frame-IDs und Idle-Dauer in Sekunden
        """
        status = {}
        for frame in self.context_stack:
            status[frame.frame_id] = {
                "query": frame.query[:50],  # Erste 50 Zeichen
                "context_type": frame.context_type.value,
                "idle_duration": frame.get_idle_duration(),
                "created_at": frame.created_at.isoformat(),
                "last_access_time": frame.last_access_time.isoformat(),
            }
        return status

    def clear(self):
        """Leere den gesamten Working Memory Stack"""
        self.context_stack.clear()
        self._frame_counter = 0

    def get_stack_depth(self) -> int:
        """Hole die aktuelle Stack-Tiefe"""
        return len(self.context_stack)

    def is_empty(self) -> bool:
        """Prüfe ob der Stack leer ist"""
        return len(self.context_stack) == 0

    def to_dict(self) -> Dict[str, Any]:
        """
        Exportiere den kompletten Working Memory State.

        Returns:
            Dictionary mit allen Frames
        """
        return {
            "stack_depth": len(self.context_stack),
            "max_stack_depth": self.max_stack_depth,
            "frames": [frame.to_dict() for frame in self.context_stack],
        }

    def get_context_summary(self) -> str:
        """
        Erstelle eine lesbare Zusammenfassung des aktuellen Kontexts.

        Returns:
            Formatierte String-Darstellung
        """
        if not self.context_stack:
            return "Working Memory: Leer"

        lines = [f"Working Memory: {len(self.context_stack)} Frames"]
        for i, frame in enumerate(self.context_stack):
            indent = "  " * i
            lines.append(f"{indent}Frame {i+1}: {frame.context_type.value}")
            lines.append(f"{indent}  Query: {frame.query}")
            lines.append(f"{indent}  Entities: {frame.entities}")
            lines.append(f"{indent}  Reasoning Steps: {len(frame.reasoning_states)}")

        return "\n".join(lines)

    def export_to_json(self, filepath: str, include_timestamps: bool = True) -> bool:
        """
        Exportiere den kompletten Working Memory State als JSON-Datei.

        Args:
            filepath: Pfad zur Export-Datei
            include_timestamps: Wenn True, werden Timestamps inkludiert

        Returns:
            True wenn erfolgreich, sonst False
        """
        try:
            export_data = self.to_dict()

            # Optional: Entferne Timestamps
            if not include_timestamps:
                for frame in export_data["frames"]:
                    frame.pop("created_at", None)
                    frame.pop("last_access_time", None)
                    for rs in frame.get("reasoning_states", []):
                        rs.pop("timestamp", None)

            # Schreibe JSON-Datei
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            return True

        except Exception as e:
            print(f"Fehler beim Export nach {filepath}: {e}")
            return False

    def export_debug_report(self, filepath: str) -> bool:
        """
        Exportiere einen human-readable Debug-Report.

        Der Report enthält:
        - ASCII-Tree-Visualisierung des Context-Stacks
        - Detaillierte Frame-Informationen
        - Reasoning-Traces
        - Idle-Status

        Args:
            filepath: Pfad zur Report-Datei

        Returns:
            True wenn erfolgreich, sonst False
        """
        try:
            lines = []
            lines.append("=" * 80)
            lines.append("WORKING MEMORY DEBUG REPORT")
            lines.append(f"Generiert: {datetime.now().isoformat()}")
            lines.append("=" * 80)
            lines.append("")

            # Stack-Übersicht
            lines.append(
                f"Stack-Tiefe: {len(self.context_stack)} / {self.max_stack_depth}"
            )
            lines.append(f"Frame-Counter: {self._frame_counter}")
            lines.append("")

            if not self.context_stack:
                lines.append("Working Memory ist leer.")
            else:
                # Idle-Status-Übersicht
                lines.append("=== IDLE-STATUS ===")
                idle_status = self.get_idle_status()
                for frame_id, status in idle_status.items():
                    lines.append(f"Frame: {frame_id}")
                    lines.append(f"  Query: {status['query']}")
                    lines.append(f"  Idle-Dauer: {status['idle_duration']:.2f}s")
                    lines.append("")

                # ASCII-Tree des Stacks
                lines.append("=== CONTEXT-STACK (Tree-View) ===")
                for i, frame in enumerate(self.context_stack):
                    indent = "  " * i
                    tree_prefix = "└─ " if i > 0 else ""
                    lines.append(f"{indent}{tree_prefix}Frame {i+1}: {frame.frame_id}")
                    lines.append(f"{indent}   Type: {frame.context_type.value}")
                    lines.append(f"{indent}   Query: {frame.query}")
                    lines.append(f"{indent}   Created: {frame.created_at.isoformat()}")
                    lines.append(
                        f"{indent}   Last Access: {frame.last_access_time.isoformat()}"
                    )
                    lines.append(
                        f"{indent}   Parent: {frame.parent_frame_id or 'None'}"
                    )
                    lines.append("")

                # Detaillierte Frame-Informationen
                lines.append("=== DETAILLIERTE FRAME-INFORMATIONEN ===")
                for i, frame in enumerate(self.context_stack):
                    lines.append(f"\n--- Frame {i+1}: {frame.frame_id} ---")
                    lines.append(f"Context-Type: {frame.context_type.value}")
                    lines.append(f"Query: {frame.query}")
                    lines.append(f"Created: {frame.created_at.isoformat()}")
                    lines.append(f"Last Access: {frame.last_access_time.isoformat()}")
                    lines.append(f"Idle Duration: {frame.get_idle_duration():.2f}s")

                    # Entities
                    if frame.entities:
                        lines.append(f"\nEntities ({len(frame.entities)}):")
                        for entity in frame.entities:
                            lines.append(f"  - {entity}")
                    else:
                        lines.append("\nEntities: None")

                    # Relations
                    if frame.relations:
                        lines.append(f"\nRelations ({len(frame.relations)}):")
                        for rel_type, rel_values in frame.relations.items():
                            lines.append(f"  {rel_type}: {rel_values}")
                    else:
                        lines.append("\nRelations: None")

                    # Variables
                    if frame.variables:
                        lines.append(f"\nVariables ({len(frame.variables)}):")
                        for key, value in frame.variables.items():
                            value_str = str(value)[:100]  # Limit zu 100 Zeichen
                            lines.append(f"  {key}: {value_str}")
                    else:
                        lines.append("\nVariables: None")

                    # Reasoning-States
                    if frame.reasoning_states:
                        lines.append(
                            f"\nReasoning-States ({len(frame.reasoning_states)}):"
                        )
                        for j, rs in enumerate(frame.reasoning_states, 1):
                            lines.append(f"  {j}. [{rs.step_type}] {rs.description}")
                            lines.append(f"     Confidence: {rs.confidence:.2f}")
                            if rs.data:
                                lines.append(f"     Data: {rs.data}")
                    else:
                        lines.append("\nReasoning-States: None")

                    # Metadata
                    if frame.metadata:
                        lines.append("\nMetadata:")
                        for key, value in frame.metadata.items():
                            lines.append(f"  {key}: {value}")

                    lines.append("")

            lines.append("=" * 80)
            lines.append("END OF REPORT")
            lines.append("=" * 80)

            # Schreibe Report-Datei
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

            return True

        except Exception as e:
            print(f"Fehler beim Export des Debug-Reports nach {filepath}: {e}")
            return False

    def import_from_json(self, filepath: str) -> bool:
        """
        Importiere Working Memory State aus JSON-Datei.

        WARNUNG: Diese Methode überschreibt den aktuellen State!
        Nützlich für Debugging und Testing mit gespeicherten States.

        Args:
            filepath: Pfad zur Import-Datei

        Returns:
            True wenn erfolgreich, sonst False
        """
        try:
            # Lade JSON-Datei
            with open(filepath, "r", encoding="utf-8") as f:
                import_data = json.load(f)

            # Validiere Struktur
            if "frames" not in import_data:
                print("Fehler: JSON-Datei hat keine 'frames'-Struktur")
                return False

            # Leere aktuellen Stack
            self.clear()

            # Importiere Frames
            for frame_dict in import_data["frames"]:
                # Rekonstruiere ContextFrame
                context_type = ContextType(frame_dict["context_type"])

                frame = ContextFrame(
                    frame_id=frame_dict["frame_id"],
                    context_type=context_type,
                    query=frame_dict["query"],
                    entities=frame_dict.get("entities", []),
                    relations=frame_dict.get("relations", {}),
                    metadata=frame_dict.get("metadata", {}),
                    parent_frame_id=frame_dict.get("parent_frame_id"),
                )

                # Setze Timestamps (falls vorhanden)
                if "created_at" in frame_dict:
                    frame.created_at = datetime.fromisoformat(frame_dict["created_at"])
                if "last_access_time" in frame_dict:
                    frame.last_access_time = datetime.fromisoformat(
                        frame_dict["last_access_time"]
                    )

                # Importiere Variables
                frame.variables = frame_dict.get("variables", {})

                # Importiere Reasoning-States
                for rs_dict in frame_dict.get("reasoning_states", []):
                    rs = ReasoningState(
                        step_id=rs_dict["step_id"],
                        step_type=rs_dict["step_type"],
                        description=rs_dict["description"],
                        data=rs_dict.get("data", {}),
                        confidence=rs_dict.get("confidence", 1.0),
                    )
                    if "timestamp" in rs_dict:
                        rs.timestamp = datetime.fromisoformat(rs_dict["timestamp"])

                    frame.reasoning_states.append(rs)

                # Füge Frame zum Stack hinzu
                self.context_stack.append(frame)

            # Aktualisiere Frame-Counter
            if "max_stack_depth" in import_data:
                self.max_stack_depth = import_data["max_stack_depth"]

            # Extrahiere höchsten Counter aus Frame-IDs
            if self.context_stack:
                max_counter = 0
                for frame in self.context_stack:
                    # Frame-IDs haben Format "frame_{counter}_{timestamp}"
                    parts = frame.frame_id.split("_")
                    if len(parts) >= 2:
                        try:
                            counter = int(parts[1])
                            max_counter = max(max_counter, counter)
                        except ValueError:
                            pass
                self._frame_counter = max_counter

            return True

        except Exception as e:
            print(f"Fehler beim Import von {filepath}: {e}")
            return False


# Hilfsfunktionen für Integration in kai_worker.py


def create_working_memory() -> WorkingMemory:
    """Factory-Funktion für Working Memory"""
    return WorkingMemory(max_stack_depth=10)


def format_reasoning_trace_for_ui(memory: WorkingMemory) -> str:
    """
    Formatiere Reasoning-Trace für UI-Anzeige.

    Args:
        memory: Das WorkingMemory-Objekt

    Returns:
        Formatierter String für "Inneres Bild"
    """
    trace = memory.get_full_reasoning_trace()
    if not trace:
        return "Keine Reasoning-Schritte vorhanden."

    lines = ["=== Reasoning-Verlauf ==="]
    for i, state in enumerate(trace, 1):
        lines.append(f"\nSchritt {i}: {state.step_type}")
        lines.append(f"  {state.description}")
        if state.data:
            lines.append(f"  Daten: {state.data}")
        lines.append(f"  Konfidenz: {state.confidence:.2f}")

    return "\n".join(lines)
