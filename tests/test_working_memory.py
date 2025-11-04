"""
test_working_memory.py

Test Suite für component_13_working_memory.py
Tests für Stack-basiertes Kontext-Management und Reasoning-State-Tracking
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from datetime import datetime
from component_13_working_memory import (
    WorkingMemory,
    ContextFrame,
    ReasoningState,
    ContextType,
    create_working_memory,
    format_reasoning_trace_for_ui,
)


class TestReasoningState:
    """Tests für ReasoningState Datenstruktur"""

    def test_reasoning_state_creation(self):
        """Test: Reasoning State kann erstellt werden"""
        state = ReasoningState(
            step_id="step_1",
            step_type="fact_retrieval",
            description="Suche Fakten über 'apfel'",
            data={"topic": "apfel"},
            confidence=0.95,
        )

        assert state.step_id == "step_1"
        assert state.step_type == "fact_retrieval"
        assert state.description == "Suche Fakten über 'apfel'"
        assert state.data["topic"] == "apfel"
        assert state.confidence == 0.95
        assert isinstance(state.timestamp, datetime)

    def test_reasoning_state_to_dict(self):
        """Test: Reasoning State kann zu Dictionary konvertiert werden"""
        state = ReasoningState(
            step_id="step_1",
            step_type="inference",
            description="Schließe von A auf B",
            data={"premise": "A", "conclusion": "B"},
        )

        state_dict = state.to_dict()

        assert state_dict["step_id"] == "step_1"
        assert state_dict["step_type"] == "inference"
        assert state_dict["description"] == "Schließe von A auf B"
        assert state_dict["data"]["premise"] == "A"
        assert "timestamp" in state_dict


class TestContextFrame:
    """Tests für ContextFrame Datenstruktur"""

    def test_context_frame_creation(self):
        """Test: Context Frame kann erstellt werden"""
        frame = ContextFrame(
            frame_id="frame_1",
            context_type=ContextType.QUESTION,
            query="Was ist ein Apfel?",
            entities=["apfel"],
            metadata={"user": "test"},
        )

        assert frame.frame_id == "frame_1"
        assert frame.context_type == ContextType.QUESTION
        assert frame.query == "Was ist ein Apfel?"
        assert "apfel" in frame.entities
        assert frame.metadata["user"] == "test"
        assert isinstance(frame.created_at, datetime)

    def test_add_reasoning_state_to_frame(self):
        """Test: Reasoning States können zu Frame hinzugefügt werden"""
        frame = ContextFrame(
            frame_id="frame_1", context_type=ContextType.QUESTION, query="Test Query"
        )

        state1 = ReasoningState(
            step_id="step_1", step_type="retrieval", description="Step 1"
        )

        state2 = ReasoningState(
            step_id="step_2", step_type="inference", description="Step 2"
        )

        frame.add_reasoning_state(state1)
        frame.add_reasoning_state(state2)

        assert len(frame.reasoning_states) == 2
        assert frame.reasoning_states[0].step_id == "step_1"
        assert frame.reasoning_states[1].step_id == "step_2"

    def test_get_last_reasoning_state(self):
        """Test: Letzter Reasoning State kann abgerufen werden"""
        frame = ContextFrame(
            frame_id="frame_1", context_type=ContextType.QUESTION, query="Test"
        )

        # Leerer Frame
        assert frame.get_last_reasoning_state() is None

        # Mit States
        state1 = ReasoningState(step_id="s1", step_type="t1", description="d1")
        state2 = ReasoningState(step_id="s2", step_type="t2", description="d2")

        frame.add_reasoning_state(state1)
        frame.add_reasoning_state(state2)

        last = frame.get_last_reasoning_state()
        assert last.step_id == "s2"

    def test_frame_to_dict(self):
        """Test: Frame kann zu Dictionary konvertiert werden"""
        frame = ContextFrame(
            frame_id="frame_1",
            context_type=ContextType.DEFINITION,
            query="Ein Apfel ist eine Frucht",
            entities=["apfel", "frucht"],
        )

        frame.add_reasoning_state(
            ReasoningState(step_id="s1", step_type="t1", description="d1")
        )

        frame_dict = frame.to_dict()

        assert frame_dict["frame_id"] == "frame_1"
        assert frame_dict["context_type"] == "definition"
        assert frame_dict["query"] == "Ein Apfel ist eine Frucht"
        assert "apfel" in frame_dict["entities"]
        assert len(frame_dict["reasoning_states"]) == 1


class TestWorkingMemory:
    """Tests für WorkingMemory Klasse"""

    def test_working_memory_creation(self):
        """Test: Working Memory kann erstellt werden"""
        memory = WorkingMemory(max_stack_depth=5)

        assert memory.max_stack_depth == 5
        assert len(memory.context_stack) == 0
        assert memory.is_empty()

    def test_push_context(self):
        """Test: Context kann auf Stack gepusht werden"""
        memory = WorkingMemory()

        frame_id = memory.push_context(
            context_type=ContextType.QUESTION,
            query="Was ist ein Hund?",
            entities=["hund"],
        )

        assert frame_id is not None
        assert len(memory.context_stack) == 1
        assert not memory.is_empty()
        assert memory.get_stack_depth() == 1

        current = memory.get_current_context()
        assert current.frame_id == frame_id
        assert current.query == "Was ist ein Hund?"
        assert "hund" in current.entities

    def test_push_multiple_contexts(self):
        """Test: Mehrere verschachtelte Contexts"""
        memory = WorkingMemory()

        frame1 = memory.push_context(
            context_type=ContextType.QUESTION, query="Query 1", entities=["entity1"]
        )

        frame2 = memory.push_context(
            context_type=ContextType.CLARIFICATION,
            query="Query 2",
            entities=["entity2"],
        )

        assert memory.get_stack_depth() == 2

        # Aktueller Context ist Frame 2
        current = memory.get_current_context()
        assert current.frame_id == frame2
        assert current.query == "Query 2"

        # Parent von Frame 2 ist Frame 1
        assert current.parent_frame_id == frame1

    def test_pop_context(self):
        """Test: Context kann vom Stack gepoppt werden"""
        memory = WorkingMemory()

        frame1 = memory.push_context(context_type=ContextType.QUESTION, query="Query 1")

        frame2 = memory.push_context(context_type=ContextType.QUESTION, query="Query 2")

        assert memory.get_stack_depth() == 2

        # Pop Frame 2
        popped = memory.pop_context()
        assert popped.frame_id == frame2
        assert memory.get_stack_depth() == 1

        # Current ist jetzt Frame 1
        current = memory.get_current_context()
        assert current.frame_id == frame1

    def test_pop_empty_stack(self):
        """Test: Pop auf leerem Stack gibt None zurück"""
        memory = WorkingMemory()

        popped = memory.pop_context()
        assert popped is None

    def test_max_stack_depth_limit(self):
        """Test: Stack-Limit wird eingehalten"""
        memory = WorkingMemory(max_stack_depth=3)

        memory.push_context(ContextType.QUESTION, "Q1")
        memory.push_context(ContextType.QUESTION, "Q2")
        memory.push_context(ContextType.QUESTION, "Q3")

        # Vierter Push sollte Exception werfen
        with pytest.raises(ValueError, match="Kontext-Stack Limit erreicht"):
            memory.push_context(ContextType.QUESTION, "Q4")

    def test_add_reasoning_state(self):
        """Test: Reasoning States können zum aktuellen Context hinzugefügt werden"""
        memory = WorkingMemory()

        memory.push_context(ContextType.QUESTION, "Test Query")

        step_id = memory.add_reasoning_state(
            step_type="fact_retrieval",
            description="Suche Fakten",
            data={"topic": "apfel"},
        )

        assert step_id is not None

        trace = memory.get_reasoning_trace()
        assert len(trace) == 1
        assert trace[0].step_type == "fact_retrieval"
        assert trace[0].data["topic"] == "apfel"

    def test_add_reasoning_state_without_context(self):
        """Test: Reasoning State ohne Context gibt None zurück"""
        memory = WorkingMemory()

        step_id = memory.add_reasoning_state(
            step_type="test", description="Should fail"
        )

        assert step_id is None

    def test_get_reasoning_trace(self):
        """Test: Reasoning Trace kann abgerufen werden"""
        memory = WorkingMemory()

        memory.push_context(ContextType.QUESTION, "Test")

        memory.add_reasoning_state("step1", "Description 1")
        memory.add_reasoning_state("step2", "Description 2")
        memory.add_reasoning_state("step3", "Description 3")

        trace = memory.get_reasoning_trace()
        assert len(trace) == 3
        assert trace[0].step_type == "step1"
        assert trace[1].step_type == "step2"
        assert trace[2].step_type == "step3"

    def test_get_full_reasoning_trace(self):
        """Test: Reasoning Trace über alle Frames"""
        memory = WorkingMemory()

        # Frame 1 mit 2 States
        memory.push_context(ContextType.QUESTION, "Query 1")
        memory.add_reasoning_state("step1", "Frame 1 Step 1")
        memory.add_reasoning_state("step2", "Frame 1 Step 2")

        # Frame 2 mit 1 State
        memory.push_context(ContextType.CLARIFICATION, "Query 2")
        memory.add_reasoning_state("step3", "Frame 2 Step 1")

        # Gesamter Trace sollte 3 States haben
        full_trace = memory.get_full_reasoning_trace()
        assert len(full_trace) == 3
        assert full_trace[0].description == "Frame 1 Step 1"
        assert full_trace[1].description == "Frame 1 Step 2"
        assert full_trace[2].description == "Frame 2 Step 1"

    def test_set_and_get_variable(self):
        """Test: Variablen können gesetzt und abgerufen werden"""
        memory = WorkingMemory()

        memory.push_context(ContextType.QUESTION, "Test")

        memory.set_variable("topic", "apfel")
        memory.set_variable("confidence", 0.95)

        assert memory.get_variable("topic") == "apfel"
        assert memory.get_variable("confidence") == 0.95

    def test_get_variable_from_parent(self):
        """Test: Variablen können von Parent-Frame vererbt werden"""
        memory = WorkingMemory()

        # Frame 1 mit Variable
        memory.push_context(ContextType.QUESTION, "Query 1")
        memory.set_variable("global_var", "value1")

        # Frame 2
        memory.push_context(ContextType.CLARIFICATION, "Query 2")

        # Variable sollte aus Parent-Frame gefunden werden
        assert memory.get_variable("global_var", search_parent=True) == "value1"

    def test_get_variable_no_parent_search(self):
        """Test: Variable nicht im Parent suchen wenn deaktiviert"""
        memory = WorkingMemory()

        memory.push_context(ContextType.QUESTION, "Query 1")
        memory.set_variable("var1", "value1")

        memory.push_context(ContextType.QUESTION, "Query 2")

        # Ohne Parent-Suche sollte None zurückkommen
        assert memory.get_variable("var1", search_parent=False) is None

    def test_add_entity(self):
        """Test: Entitäten können zum Context hinzugefügt werden"""
        memory = WorkingMemory()

        memory.push_context(ContextType.QUESTION, "Test", entities=["apfel"])

        memory.add_entity("birne")
        memory.add_entity("kirsche")

        current = memory.get_current_context()
        assert "apfel" in current.entities
        assert "birne" in current.entities
        assert "kirsche" in current.entities

    def test_add_duplicate_entity(self):
        """Test: Duplicate Entitäten werden nicht doppelt hinzugefügt"""
        memory = WorkingMemory()

        memory.push_context(ContextType.QUESTION, "Test", entities=["apfel"])

        memory.add_entity("apfel")  # Duplicate

        current = memory.get_current_context()
        assert current.entities.count("apfel") == 1

    def test_get_all_entities(self):
        """Test: Alle Entitäten über alle Frames"""
        memory = WorkingMemory()

        memory.push_context(ContextType.QUESTION, "Q1", entities=["apfel", "birne"])
        memory.push_context(ContextType.QUESTION, "Q2", entities=["kirsche", "apfel"])

        all_entities = memory.get_all_entities()

        # Sollte dedupliziert sein
        assert "apfel" in all_entities
        assert "birne" in all_entities
        assert "kirsche" in all_entities
        # Apfel sollte nur einmal vorkommen
        assert all_entities.count("apfel") == 1

    def test_clear(self):
        """Test: Working Memory kann geleert werden"""
        memory = WorkingMemory()

        memory.push_context(ContextType.QUESTION, "Q1")
        memory.push_context(ContextType.QUESTION, "Q2")
        memory.add_reasoning_state("test", "Test State")

        assert not memory.is_empty()

        memory.clear()

        assert memory.is_empty()
        assert memory.get_stack_depth() == 0
        assert memory.get_current_context() is None

    def test_get_context_by_id(self):
        """Test: Context kann anhand ID abgerufen werden"""
        memory = WorkingMemory()

        frame1_id = memory.push_context(ContextType.QUESTION, "Q1")
        frame2_id = memory.push_context(ContextType.QUESTION, "Q2")

        frame1 = memory.get_context_by_id(frame1_id)
        assert frame1 is not None
        assert frame1.query == "Q1"

        frame2 = memory.get_context_by_id(frame2_id)
        assert frame2 is not None
        assert frame2.query == "Q2"

        # Nicht existierende ID
        assert memory.get_context_by_id("nonexistent") is None

    def test_get_parent_context(self):
        """Test: Parent-Context kann abgerufen werden"""
        memory = WorkingMemory()

        frame1_id = memory.push_context(ContextType.QUESTION, "Parent Query")
        memory.push_context(ContextType.CLARIFICATION, "Child Query")

        parent = memory.get_parent_context()
        assert parent is not None
        assert parent.frame_id == frame1_id
        assert parent.query == "Parent Query"

    def test_get_parent_context_no_parent(self):
        """Test: Kein Parent bei Root-Context"""
        memory = WorkingMemory()

        memory.push_context(ContextType.QUESTION, "Root Query")

        parent = memory.get_parent_context()
        assert parent is None

    def test_to_dict(self):
        """Test: Working Memory kann zu Dictionary exportiert werden"""
        memory = WorkingMemory(max_stack_depth=5)

        memory.push_context(ContextType.QUESTION, "Q1", entities=["entity1"])
        memory.add_reasoning_state("step1", "Step 1")

        memory_dict = memory.to_dict()

        assert memory_dict["stack_depth"] == 1
        assert memory_dict["max_stack_depth"] == 5
        assert len(memory_dict["frames"]) == 1
        assert memory_dict["frames"][0]["query"] == "Q1"

    def test_get_context_summary(self):
        """Test: Context Summary kann generiert werden"""
        memory = WorkingMemory()

        # Leerer Stack
        summary = memory.get_context_summary()
        assert "Leer" in summary

        # Mit Frames
        memory.push_context(
            ContextType.QUESTION, "Was ist ein Apfel?", entities=["apfel"]
        )
        memory.add_reasoning_state("step1", "Step 1")
        memory.push_context(ContextType.CLARIFICATION, "Meinst du die Frucht?")

        summary = memory.get_context_summary()

        assert "2 Frames" in summary
        assert "question" in summary.lower()
        assert "clarification" in summary.lower()


class TestHelperFunctions:
    """Tests für Hilfsfunktionen"""

    def test_create_working_memory(self):
        """Test: Factory-Funktion erstellt Working Memory"""
        memory = create_working_memory()

        assert isinstance(memory, WorkingMemory)
        assert memory.max_stack_depth == 10

    def test_format_reasoning_trace_for_ui_empty(self):
        """Test: Formatting eines leeren Trace"""
        memory = WorkingMemory()

        trace_str = format_reasoning_trace_for_ui(memory)

        assert "Keine Reasoning-Schritte" in trace_str

    def test_format_reasoning_trace_for_ui_with_states(self):
        """Test: Formatting eines Trace mit States"""
        memory = WorkingMemory()

        memory.push_context(ContextType.QUESTION, "Test")
        memory.add_reasoning_state(
            step_type="fact_retrieval",
            description="Suche Fakten",
            data={"topic": "apfel"},
            confidence=0.95,
        )
        memory.add_reasoning_state(
            step_type="inference", description="Schließe Fakten ab", confidence=0.80
        )

        trace_str = format_reasoning_trace_for_ui(memory)

        assert "Reasoning-Verlauf" in trace_str
        assert "Schritt 1" in trace_str
        assert "fact_retrieval" in trace_str
        assert "Suche Fakten" in trace_str
        assert "0.95" in trace_str
        assert "Schritt 2" in trace_str
        assert "inference" in trace_str


class TestComplexScenarios:
    """Tests für komplexe Multi-Turn-Szenarien"""

    def test_nested_question_clarification_flow(self):
        """Test: Verschachtelter Frage-Klärung-Dialog"""
        memory = WorkingMemory()

        # Nutzer stellt Frage
        q_frame = memory.push_context(
            ContextType.QUESTION, "Was ist ein Apfel?", entities=["apfel"]
        )
        memory.add_reasoning_state("fact_retrieval", "Suche Fakten über Apfel")

        # System findet keine Fakten -> Rückfrage
        memory.push_context(
            ContextType.CLARIFICATION,
            "Meinst du die Frucht oder das Unternehmen?",
            entities=["apfel"],
        )
        memory.add_reasoning_state("clarification_request", "Frage nach Klärung")

        # Nutzer antwortet
        memory.set_variable("clarification_answer", "die Frucht")

        # Verifiziere Stack
        assert memory.get_stack_depth() == 2
        current = memory.get_current_context()
        assert current.context_type == ContextType.CLARIFICATION

        # Hole Parent (ursprüngliche Frage)
        parent = memory.get_parent_context()
        assert parent.frame_id == q_frame
        assert parent.context_type == ContextType.QUESTION

        # Verifiziere Reasoning-Trace über beide Frames
        full_trace = memory.get_full_reasoning_trace()
        assert len(full_trace) == 2

    def test_multi_hop_reasoning_chain(self):
        """Test: Multi-Hop Reasoning über mehrere Schritte"""
        memory = WorkingMemory()

        memory.push_context(
            ContextType.INFERENCE,
            "Ist ein Hund ein Säugetier?",
            entities=["hund", "säugetier"],
        )

        # Schritt 1: Hole Fakten über Hund
        memory.add_reasoning_state(
            step_type="fact_retrieval",
            description="Finde: Hund IS_A ?",
            data={"result": ["tier"]},
        )

        # Schritt 2: Hole Fakten über Tier
        memory.add_reasoning_state(
            step_type="fact_retrieval",
            description="Finde: Tier IS_A ?",
            data={"result": ["lebewesen"]},
        )

        # Schritt 3: Inferenz-Regel anwenden
        memory.add_reasoning_state(
            step_type="inference",
            description="Wende Transitivität an: Hund IS_A Tier IS_A Säugetier",
            data={"conclusion": "Hund IS_A Säugetier"},
            confidence=0.85,
        )

        trace = memory.get_reasoning_trace()
        assert len(trace) == 3
        assert trace[0].step_type == "fact_retrieval"
        assert trace[2].step_type == "inference"
        assert trace[2].confidence == 0.85

    def test_learning_from_failure_scenario(self):
        """Test: Lernen aus Fehlschlag (Wissenslücke -> Beispielsatz -> Retry)"""
        memory = WorkingMemory()

        # Versuch 1: Frage scheitert
        attempt1 = memory.push_context(
            ContextType.QUESTION, "Was ist eine Katze?", entities=["katze"]
        )
        memory.add_reasoning_state(
            "fact_retrieval", "Suche Fakten über Katze", data={"result": None}
        )
        memory.add_reasoning_state(
            "knowledge_gap_detected", "Keine Fakten gefunden", confidence=0.0
        )

        # System bittet um Beispiel
        learning_frame = memory.push_context(
            ContextType.PATTERN_LEARNING,
            "Eine Katze ist ein Tier",
            entities=["katze", "tier"],
        )
        memory.add_reasoning_state(
            "pattern_extraction", "Extrahiere Pattern: IS_A Relation"
        )

        # Speichere gelernte Info als Variable
        memory.set_variable("learned_fact", {"katze": {"IS_A": "tier"}})

        # Verifiziere verschachtelte Contexts
        assert memory.get_stack_depth() == 2

        # Variable sollte aus beiden Frames zugreifbar sein
        learned = memory.get_variable("learned_fact")
        assert learned["katze"]["IS_A"] == "tier"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
