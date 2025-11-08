"""
test_working_memory_stress.py

Stress-Tests für component_13_working_memory.py
Tests für verschachtelte Kontexte (5+ Ebenen), Performance, Memory-Leaks
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import time

import pytest

from component_13_working_memory import (
    ContextType,
    WorkingMemory,
)


class TestDeepNestedContexts:
    """Tests für tief verschachtelte Kontexte (5+ Ebenen)"""

    def test_5_level_nested_contexts(self):
        """Test: 5 verschachtelte Kontext-Ebenen"""
        memory = WorkingMemory(max_stack_depth=10)

        # Erstelle 5 verschachtelte Frames
        frame_ids = []
        for i in range(5):
            frame_id = memory.push_context(
                context_type=ContextType.QUESTION,
                query=f"Query Level {i+1}",
                entities=[f"entity_{i+1}"],
                metadata={"level": i + 1},
            )
            frame_ids.append(frame_id)

            # Füge Reasoning-State hinzu
            memory.add_reasoning_state(
                step_type=f"step_level_{i+1}",
                description=f"Reasoning at level {i+1}",
                data={"level": i + 1},
            )

        # Verifiziere Stack-Tiefe
        assert memory.get_stack_depth() == 5

        # Verifiziere aktuellen Kontext (Level 5)
        current = memory.get_current_context()
        assert current.query == "Query Level 5"
        assert current.metadata["level"] == 5

        # Verifiziere Parent-Chain
        parent = memory.get_parent_context()
        assert parent.query == "Query Level 4"

        # Verifiziere alle Entitäten
        all_entities = memory.get_all_entities()
        assert len(all_entities) == 5
        for i in range(5):
            assert f"entity_{i+1}" in all_entities

        # Verifiziere Reasoning-Trace über alle Ebenen
        full_trace = memory.get_full_reasoning_trace()
        assert len(full_trace) == 5
        for i, state in enumerate(full_trace):
            assert state.data["level"] == i + 1

    def test_10_level_nested_contexts_at_limit(self):
        """Test: 10 verschachtelte Ebenen (Stack-Limit)"""
        memory = WorkingMemory(max_stack_depth=10)

        # Erstelle 10 Frames (exakt am Limit)
        for i in range(10):
            memory.push_context(
                context_type=ContextType.QUESTION,
                query=f"Query {i+1}",
                entities=[f"entity_{i+1}"],
            )

        assert memory.get_stack_depth() == 10

        # 11. Frame sollte Exception werfen
        with pytest.raises(ValueError, match="Kontext-Stack Limit erreicht"):
            memory.push_context(context_type=ContextType.QUESTION, query="Query 11")

    def test_deep_nested_with_pop_operations(self):
        """Test: Tief verschachtelt mit Pop-Operationen"""
        memory = WorkingMemory(max_stack_depth=10)

        # Push 7 Frames
        for i in range(7):
            memory.push_context(context_type=ContextType.QUESTION, query=f"Query {i+1}")

        assert memory.get_stack_depth() == 7

        # Pop 3 Frames
        for _ in range(3):
            memory.pop_context()

        assert memory.get_stack_depth() == 4

        # Aktueller Frame sollte Query 4 sein
        current = memory.get_current_context()
        assert current.query == "Query 4"

        # Push wieder 2 Frames
        memory.push_context(ContextType.CLARIFICATION, "Clarification 1")
        memory.push_context(ContextType.CLARIFICATION, "Clarification 2")

        assert memory.get_stack_depth() == 6

    def test_nested_contexts_with_variable_inheritance(self):
        """Test: Variable-Vererbung über 5+ Ebenen"""
        memory = WorkingMemory(max_stack_depth=10)

        # Level 1: Setze globale Variable
        memory.push_context(ContextType.QUESTION, "Level 1")
        memory.set_variable("global_var", "global_value")
        memory.set_variable("level_1_var", "value_1")

        # Level 2-5: Setze lokale Variablen
        for i in range(2, 6):
            memory.push_context(ContextType.QUESTION, f"Level {i}")
            memory.set_variable(f"level_{i}_var", f"value_{i}")

        assert memory.get_stack_depth() == 5

        # Verifiziere Variable-Zugriff mit Parent-Suche
        assert memory.get_variable("global_var", search_parent=True) == "global_value"
        assert memory.get_variable("level_1_var", search_parent=True) == "value_1"
        assert memory.get_variable("level_3_var", search_parent=True) == "value_3"
        assert memory.get_variable("level_5_var", search_parent=False) == "value_5"

        # Variable aus tieferen Ebenen sollte nicht ohne Parent-Suche gefunden werden
        assert memory.get_variable("level_1_var", search_parent=False) is None


class TestLargeReasoningTraces:
    """Tests für große Reasoning-Traces"""

    def test_100_reasoning_states_single_context(self):
        """Test: 100 Reasoning-States in einem Kontext"""
        memory = WorkingMemory()

        memory.push_context(ContextType.INFERENCE, "Complex Reasoning Task")

        # Füge 100 Reasoning-States hinzu
        for i in range(100):
            memory.add_reasoning_state(
                step_type=f"step_type_{i % 10}",  # 10 verschiedene Typen
                description=f"Reasoning step {i+1}",
                data={"step_number": i + 1, "some_data": f"data_{i}"},
                confidence=0.5 + (i % 50) * 0.01,  # Variierende Confidence
            )

        trace = memory.get_reasoning_trace()
        assert len(trace) == 100

        # Verifiziere ersten und letzten State
        assert trace[0].description == "Reasoning step 1"
        assert trace[99].description == "Reasoning step 100"

    def test_reasoning_states_across_multiple_contexts(self):
        """Test: Reasoning-States über 5 verschachtelte Kontexte"""
        memory = WorkingMemory()

        # 5 Kontexte mit je 20 Reasoning-States
        for context_num in range(5):
            memory.push_context(ContextType.INFERENCE, f"Context {context_num + 1}")

            for state_num in range(20):
                memory.add_reasoning_state(
                    step_type="inference_step",
                    description=f"Context {context_num+1}, Step {state_num+1}",
                    data={"context": context_num + 1, "step": state_num + 1},
                )

        # Gesamter Trace sollte 100 States haben
        full_trace = memory.get_full_reasoning_trace()
        assert len(full_trace) == 100

        # Verifiziere Verteilung
        for i in range(5):
            context_states = [s for s in full_trace if s.data.get("context") == i + 1]
            assert len(context_states) == 20

    def test_performance_large_trace_retrieval(self):
        """Test: Performance bei großen Reasoning-Traces"""
        memory = WorkingMemory()

        memory.push_context(ContextType.INFERENCE, "Performance Test")

        # Füge 500 States hinzu
        for i in range(500):
            memory.add_reasoning_state(
                step_type="perf_test", description=f"Step {i}", data={"index": i}
            )

        # Messe Zeit für Trace-Abruf
        start_time = time.time()
        trace = memory.get_reasoning_trace()
        elapsed = time.time() - start_time

        assert len(trace) == 500
        # Sollte unter 10ms sein
        assert (
            elapsed < 0.01
        ), f"Trace retrieval took {elapsed*1000:.2f}ms (should be <10ms)"


class TestMemoryLeakPrevention:
    """Tests für Memory-Leak-Prevention"""

    def test_1000_push_pop_cycles(self):
        """Test: 1000 Push/Pop-Zyklen (Memory-Leak-Test)"""
        memory = WorkingMemory(max_stack_depth=10)

        # 1000 Push/Pop-Zyklen
        for cycle in range(1000):
            # Push 3 Frames
            for i in range(3):
                memory.push_context(
                    ContextType.QUESTION,
                    f"Cycle {cycle}, Frame {i}",
                    entities=[f"entity_{i}"],
                )
                memory.add_reasoning_state("test_step", f"Step in cycle {cycle}")

            # Pop alle 3 Frames
            for _ in range(3):
                memory.pop_context()

        # Stack sollte am Ende leer sein
        assert memory.is_empty()
        assert memory.get_stack_depth() == 0

    def test_clear_resets_counter(self):
        """Test: Clear() setzt Frame-Counter zurück"""
        memory = WorkingMemory()

        # Erstelle mehrere Frames
        for i in range(5):
            memory.push_context(ContextType.QUESTION, f"Query {i}")

        # Clear sollte Counter zurücksetzen
        memory.clear()

        assert memory._frame_counter == 0
        assert memory.is_empty()

        # Neue Frames sollten wieder bei 1 starten
        frame_id = memory.push_context(ContextType.QUESTION, "New Query")
        assert "frame_1_" in frame_id

    def test_massive_entities_in_single_context(self):
        """Test: Performance mit vielen Entitäten"""
        memory = WorkingMemory()

        memory.push_context(ContextType.QUESTION, "Test Query")

        # Füge 1000 Entitäten hinzu
        for i in range(1000):
            memory.add_entity(f"entity_{i}")

        current = memory.get_current_context()
        assert len(current.entities) == 1000

        # Test Deduplizierung
        memory.add_entity("entity_500")  # Duplicate
        assert len(current.entities) == 1000  # Sollte nicht wachsen

    def test_massive_variables_in_single_context(self):
        """Test: Performance mit vielen Variablen"""
        memory = WorkingMemory()

        memory.push_context(ContextType.QUESTION, "Test Query")

        # Setze 1000 Variablen
        for i in range(1000):
            memory.set_variable(f"var_{i}", f"value_{i}")

        current = memory.get_current_context()
        assert len(current.variables) == 1000

        # Verifiziere Random-Access
        assert memory.get_variable("var_500") == "value_500"
        assert memory.get_variable("var_999") == "value_999"


class TestComplexMultiTurnScenarios:
    """Tests für komplexe Multi-Turn-Szenarien mit tiefer Verschachtelung"""

    def test_complex_clarification_chain(self):
        """Test: Verschachtelte Klärungskette (6 Ebenen)"""
        memory = WorkingMemory(max_stack_depth=10)

        # Level 1: Ursprüngliche Frage
        memory.push_context(
            ContextType.QUESTION, "Was ist ein Apfel?", entities=["apfel"]
        )
        memory.add_reasoning_state("fact_retrieval", "Suche Fakten über Apfel")

        # Level 2: Erste Rückfrage
        memory.push_context(
            ContextType.CLARIFICATION,
            "Meinst du die Frucht oder das Unternehmen?",
            entities=["apfel"],
        )
        memory.add_reasoning_state("clarification_request", "Frage nach Typ")

        # Level 3: Nutzer-Antwort führt zu weiterer Frage
        memory.push_context(ContextType.QUESTION, "Die Frucht", entities=["frucht"])
        memory.set_variable("clarification_answer", "frucht")

        # Level 4: Weitere Wissenslücke
        memory.push_context(
            ContextType.CLARIFICATION,
            "Welche Sorte Apfel meinst du?",
            entities=["apfel", "sorte"],
        )

        # Level 5: Nutzer gibt Beispiel
        memory.push_context(
            ContextType.PATTERN_LEARNING,
            "Ein Granny Smith ist eine grüne Apfelsorte",
            entities=["granny smith", "apfel", "grün"],
        )
        memory.add_reasoning_state("pattern_extraction", "Extrahiere HAS_PROPERTY")

        # Level 6: System verarbeitet Lernergebnis
        memory.push_context(
            ContextType.DEFINITION,
            "Speichere: granny smith HAS_PROPERTY grün",
            entities=["granny smith", "grün"],
        )
        memory.set_variable("learned_fact", {"granny smith": {"HAS_PROPERTY": "grün"}})

        # Verifiziere Stack-Tiefe
        assert memory.get_stack_depth() == 6

        # Verifiziere Parent-Chain
        level_6 = memory.get_current_context()
        assert level_6.context_type == ContextType.DEFINITION

        # Verifiziere Variable-Zugriff aus Level 6
        learned = memory.get_variable("learned_fact", search_parent=False)
        assert learned is not None
        assert learned["granny smith"]["HAS_PROPERTY"] == "grün"

        # Verifiziere dass clarification_answer aus Level 3 erreichbar ist
        clarification = memory.get_variable("clarification_answer", search_parent=True)
        assert clarification == "frucht"

        # Pop zurück zu Level 5
        memory.pop_context()
        level_5 = memory.get_current_context()
        assert level_5.context_type == ContextType.PATTERN_LEARNING

    def test_multi_hop_reasoning_with_nested_inference(self):
        """Test: Multi-Hop Reasoning mit verschachtelten Inferenz-Schritten"""
        memory = WorkingMemory(max_stack_depth=10)

        # Level 1: Ursprüngliche Frage
        memory.push_context(
            ContextType.QUESTION,
            "Ist ein Pudel ein Säugetier?",
            entities=["pudel", "säugetier"],
        )

        # Level 2: Erste Inferenz (Pudel -> Hund)
        memory.push_context(
            ContextType.INFERENCE, "Prüfe: Pudel IS_A Hund", entities=["pudel", "hund"]
        )
        memory.add_reasoning_state(
            "inference_step",
            "Finde: Pudel IS_A Hund",
            data={"relation": "IS_A", "confidence": 0.95},
        )
        memory.set_variable("hop_1_result", "hund")

        # Level 3: Zweite Inferenz (Hund -> Tier)
        memory.push_context(
            ContextType.INFERENCE, "Prüfe: Hund IS_A Tier", entities=["hund", "tier"]
        )
        memory.add_reasoning_state(
            "inference_step",
            "Finde: Hund IS_A Tier",
            data={"relation": "IS_A", "confidence": 0.98},
        )
        memory.set_variable("hop_2_result", "tier")

        # Level 4: Dritte Inferenz (Tier -> Säugetier)
        memory.push_context(
            ContextType.INFERENCE,
            "Prüfe: Tier IS_A Säugetier",
            entities=["tier", "säugetier"],
        )
        memory.add_reasoning_state(
            "inference_step",
            "Finde: Tier IS_A Säugetier (via Transitivität)",
            data={"relation": "IS_A", "confidence": 0.90},
        )
        memory.set_variable("hop_3_result", "säugetier")

        # Level 5: Finale Konklusion
        memory.push_context(
            ContextType.INFERENCE,
            "Konklusion: Pudel IS_A Säugetier",
            entities=["pudel", "säugetier"],
        )
        memory.add_reasoning_state(
            "conclusion",
            "Transitiver Schluss: Pudel -> Hund -> Tier -> Säugetier",
            data={"hops": ["pudel", "hund", "tier", "säugetier"], "confidence": 0.85},
        )

        # Verifiziere Stack-Tiefe
        assert memory.get_stack_depth() == 5

        # Verifiziere Reasoning-Trace
        full_trace = memory.get_full_reasoning_trace()
        assert len(full_trace) == 4  # 3 Inferenz-Steps + 1 Konklusion

        # Verifiziere Variable-Vererbung über alle Hops
        assert memory.get_variable("hop_1_result", search_parent=True) == "hund"
        assert memory.get_variable("hop_2_result", search_parent=True) == "tier"
        assert memory.get_variable("hop_3_result", search_parent=True) == "säugetier"

        # Verifiziere finale Konklusion
        conclusion = memory.get_current_context()
        assert "Konklusion" in conclusion.query
        last_state = conclusion.get_last_reasoning_state()
        assert last_state.step_type == "conclusion"
        assert len(last_state.data["hops"]) == 4


class TestContextSummaryAndExport:
    """Tests für Context-Summary und Export-Funktionen"""

    def test_context_summary_deep_nested(self):
        """Test: Context-Summary für tief verschachtelte Kontexte"""
        memory = WorkingMemory()

        for i in range(5):
            memory.push_context(
                ContextType.QUESTION, f"Query Level {i+1}", entities=[f"entity_{i+1}"]
            )
            memory.add_reasoning_state(f"step_{i+1}", f"Reasoning at level {i+1}")

        summary = memory.get_context_summary()

        # Verifiziere Summary-Inhalt
        assert "5 Frames" in summary
        for i in range(5):
            assert f"Query Level {i+1}" in summary
            assert f"entity_{i+1}" in summary

    def test_to_dict_export_deep_nested(self):
        """Test: to_dict() Export für tief verschachtelte Kontexte"""
        memory = WorkingMemory(max_stack_depth=10)

        # Erstelle 5 verschachtelte Frames mit komplexen Daten
        for i in range(5):
            memory.push_context(
                ContextType.INFERENCE,
                f"Query {i+1}",
                entities=[f"entity_{i+1}"],
                relations={"IS_A": [f"relation_{i+1}"]},
                metadata={"level": i + 1, "timestamp": f"time_{i+1}"},
            )
            memory.add_reasoning_state(
                f"step_type_{i+1}",
                f"Description {i+1}",
                data={"custom_data": f"value_{i+1}"},
            )
            memory.set_variable(f"var_{i+1}", f"value_{i+1}")

        # Export zu Dictionary
        exported = memory.to_dict()

        # Verifiziere Export-Struktur
        assert exported["stack_depth"] == 5
        assert exported["max_stack_depth"] == 10
        assert len(exported["frames"]) == 5

        # Verifiziere Frame-Daten
        for i, frame_dict in enumerate(exported["frames"]):
            assert frame_dict["query"] == f"Query {i+1}"
            assert frame_dict["context_type"] == "inference"
            assert f"entity_{i+1}" in frame_dict["entities"]
            assert len(frame_dict["reasoning_states"]) == 1
            assert (
                frame_dict["reasoning_states"][0]["description"] == f"Description {i+1}"
            )
            assert f"var_{i+1}" in frame_dict["variables"]


class TestIdleCleanup:
    """Tests für automatisches Idle-Cleanup"""

    def test_frame_touch_updates_access_time(self):
        """Test: touch() aktualisiert last_access_time"""
        memory = WorkingMemory()

        memory.push_context(ContextType.QUESTION, "Test Query")
        current = memory.get_current_context()

        original_time = current.last_access_time
        time.sleep(0.1)  # Kurze Pause

        current.touch()
        new_time = current.last_access_time

        assert new_time > original_time

    def test_frame_is_idle_detection(self):
        """Test: is_idle() erkennt idle Frames korrekt"""
        memory = WorkingMemory()

        memory.push_context(ContextType.QUESTION, "Test Query")
        current = memory.get_current_context()

        # Frame ist frisch, sollte nicht idle sein
        assert not current.is_idle(timeout_seconds=1)

        # Warte 2 Sekunden
        time.sleep(2)

        # Jetzt sollte Frame idle sein (> 1 Sekunde)
        assert current.is_idle(timeout_seconds=1)

    def test_get_idle_frames(self):
        """Test: get_idle_frames() findet idle Frames"""
        memory = WorkingMemory()

        # Erstelle 3 Frames
        memory.push_context(ContextType.QUESTION, "Query 1")
        time.sleep(0.1)
        memory.push_context(ContextType.QUESTION, "Query 2")
        time.sleep(0.1)
        memory.push_context(ContextType.QUESTION, "Query 3")

        # Warte 1 Sekunde
        time.sleep(1)

        # Alle Frames sollten idle sein (> 0.5 Sekunden)
        idle_frames = memory.get_idle_frames(timeout_seconds=0.5)
        assert len(idle_frames) == 3

        # Touch den aktuellen Frame
        memory.touch_current_context()

        # Jetzt sollten nur 2 Frames idle sein
        idle_frames = memory.get_idle_frames(timeout_seconds=0.5)
        assert len(idle_frames) == 2

    def test_cleanup_idle_contexts_basic(self):
        """Test: cleanup_idle_contexts() entfernt idle Frames"""
        memory = WorkingMemory()

        # Erstelle 3 Frames
        memory.push_context(ContextType.QUESTION, "Query 1")
        time.sleep(0.1)
        frame2_id = memory.push_context(ContextType.QUESTION, "Query 2")
        time.sleep(0.1)
        frame3_id = memory.push_context(ContextType.QUESTION, "Query 3")

        assert memory.get_stack_depth() == 3

        # Warte 1 Sekunde
        time.sleep(1)

        # Cleanup mit 0.5 Sekunden Timeout
        removed_ids = memory.cleanup_idle_contexts(
            timeout_seconds=0.5, preserve_root=True
        )

        # Frame 1 sollte erhalten bleiben (preserve_root=True)
        # Frame 2 und 3 sollten entfernt werden
        assert len(removed_ids) == 2
        assert frame2_id in removed_ids
        assert frame3_id in removed_ids

        # Stack sollte jetzt nur noch Frame 1 enthalten
        assert memory.get_stack_depth() == 1
        current = memory.get_current_context()
        assert current.query == "Query 1"

    def test_cleanup_idle_contexts_preserve_root_false(self):
        """Test: cleanup_idle_contexts() mit preserve_root=False"""
        memory = WorkingMemory()

        # Erstelle 2 Frames
        memory.push_context(ContextType.QUESTION, "Query 1")
        memory.push_context(ContextType.QUESTION, "Query 2")

        # Warte 1 Sekunde
        time.sleep(1)

        # Cleanup ohne Root-Schutz
        removed_ids = memory.cleanup_idle_contexts(
            timeout_seconds=0.5, preserve_root=False
        )

        # Alle Frames sollten entfernt werden
        assert len(removed_ids) == 2
        assert memory.is_empty()

    def test_cleanup_idle_contexts_no_idle_frames(self):
        """Test: cleanup_idle_contexts() wenn keine Frames idle sind"""
        memory = WorkingMemory()

        memory.push_context(ContextType.QUESTION, "Query 1")
        memory.push_context(ContextType.QUESTION, "Query 2")

        # Sofortiges Cleanup (kein Frame sollte idle sein)
        removed_ids = memory.cleanup_idle_contexts(timeout_seconds=10)

        assert len(removed_ids) == 0
        assert memory.get_stack_depth() == 2

    def test_cleanup_idle_contexts_mixed_idle_active(self):
        """Test: cleanup_idle_contexts() mit gemischten idle/aktiven Frames"""
        memory = WorkingMemory()

        # Frame 1 (wird idle)
        memory.push_context(ContextType.QUESTION, "Query 1")
        time.sleep(1.1)

        # Frame 2 (wird idle)
        memory.push_context(ContextType.QUESTION, "Query 2")
        time.sleep(0.6)

        # Frame 3 (bleibt aktiv)
        memory.push_context(ContextType.QUESTION, "Query 3")

        # Touch Frame 3 um sicherzustellen dass es aktiv bleibt
        memory.touch_current_context()

        # Cleanup mit 0.5 Sekunden Timeout
        _ = memory.cleanup_idle_contexts(timeout_seconds=0.5, preserve_root=True)

        # Frame 2 sollte entfernt werden (und dadurch auch Frame 3 wegen Parent-Child-Konsistenz)
        # Frame 1 bleibt (preserve_root=True)
        assert memory.get_stack_depth() == 1

    def test_touch_current_context_updates_access_time(self):
        """Test: touch_current_context() aktualisiert Access-Time"""
        memory = WorkingMemory()

        memory.push_context(ContextType.QUESTION, "Test Query")
        current = memory.get_current_context()
        original_time = current.last_access_time

        time.sleep(0.1)
        memory.touch_current_context()

        new_time = current.last_access_time
        assert new_time > original_time

    def test_automatic_touch_on_operations(self):
        """Test: Operationen aktualisieren automatisch Access-Time"""
        memory = WorkingMemory()

        memory.push_context(ContextType.QUESTION, "Test Query")
        current = memory.get_current_context()
        original_time = current.last_access_time

        time.sleep(0.1)

        # Set Variable sollte touch() aufrufen
        memory.set_variable("test_var", "value")
        assert current.last_access_time > original_time

        original_time = current.last_access_time
        time.sleep(0.1)

        # Add Entity sollte touch() aufrufen
        memory.add_entity("test_entity")
        assert current.last_access_time > original_time

        original_time = current.last_access_time
        time.sleep(0.1)

        # Add Reasoning State sollte touch() aufrufen
        memory.add_reasoning_state("test_step", "Test Description")
        assert current.last_access_time > original_time

    def test_get_variable_touches_frame(self):
        """Test: get_variable() aktualisiert Access-Time"""
        memory = WorkingMemory()

        memory.push_context(ContextType.QUESTION, "Test Query")
        memory.set_variable("test_var", "value")

        current = memory.get_current_context()
        original_time = current.last_access_time

        time.sleep(0.1)

        # Get Variable sollte touch() aufrufen
        value = memory.get_variable("test_var")
        assert value == "value"
        assert current.last_access_time > original_time

    def test_get_idle_status(self):
        """Test: get_idle_status() liefert korrekten Status"""
        memory = WorkingMemory()

        # Erstelle 2 Frames
        memory.push_context(ContextType.QUESTION, "Query 1")
        time.sleep(0.5)
        memory.push_context(ContextType.QUESTION, "Query 2")

        status = memory.get_idle_status()

        # Sollte 2 Einträge haben
        assert len(status) == 2

        # Jeder Eintrag sollte die erwarteten Felder haben
        for frame_id, frame_status in status.items():
            assert "query" in frame_status
            assert "context_type" in frame_status
            assert "idle_duration" in frame_status
            assert "created_at" in frame_status
            assert "last_access_time" in frame_status
            assert isinstance(frame_status["idle_duration"], float)

    def test_idle_cleanup_with_reasoning_states(self):
        """Test: Cleanup funktioniert mit Frames die Reasoning-States haben"""
        memory = WorkingMemory()

        # Frame 1 mit Reasoning-States
        memory.push_context(ContextType.QUESTION, "Query 1")
        memory.add_reasoning_state("step1", "Step 1")
        memory.add_reasoning_state("step2", "Step 2")

        time.sleep(0.6)

        # Frame 2
        memory.push_context(ContextType.QUESTION, "Query 2")

        time.sleep(0.6)  # Warte damit Frame 2 auch idle wird

        # Cleanup
        removed_ids = memory.cleanup_idle_contexts(
            timeout_seconds=0.5, preserve_root=True
        )

        # Frame 2 sollte entfernt werden
        assert len(removed_ids) == 1
        assert memory.get_stack_depth() == 1

        # Frame 1 mit seinen Reasoning-States sollte noch da sein
        current = memory.get_current_context()
        assert len(current.reasoning_states) == 2


class TestExportImport:
    """Tests für Export/Import-Funktionalität"""

    def test_export_to_json_basic(self, tmp_path):
        """Test: JSON-Export grundlegende Funktionalität"""
        memory = WorkingMemory()

        # Erstelle Test-Frames
        memory.push_context(ContextType.QUESTION, "Test Query 1", entities=["entity1"])
        memory.add_reasoning_state("step1", "Reasoning Step 1")
        memory.set_variable("var1", "value1")

        memory.push_context(ContextType.CLARIFICATION, "Test Query 2")

        # Export
        export_file = tmp_path / "memory_export.json"
        success = memory.export_to_json(str(export_file))

        assert success
        assert export_file.exists()

        # Verifiziere JSON-Inhalt
        with open(export_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["stack_depth"] == 2
        assert len(data["frames"]) == 2
        assert data["frames"][0]["query"] == "Test Query 1"
        assert "entity1" in data["frames"][0]["entities"]

    def test_export_to_json_without_timestamps(self, tmp_path):
        """Test: JSON-Export ohne Timestamps"""
        memory = WorkingMemory()

        memory.push_context(ContextType.QUESTION, "Test Query")
        memory.add_reasoning_state("step1", "Step 1")

        # Export ohne Timestamps
        export_file = tmp_path / "memory_export_no_timestamps.json"
        success = memory.export_to_json(str(export_file), include_timestamps=False)

        assert success

        # Verifiziere dass keine Timestamps im Export sind
        with open(export_file, "r", encoding="utf-8") as f:
            import json as json_module

            data = json_module.load(f)

        assert "created_at" not in data["frames"][0]
        assert "last_access_time" not in data["frames"][0]
        assert "timestamp" not in data["frames"][0]["reasoning_states"][0]

    def test_import_from_json_basic(self, tmp_path):
        """Test: JSON-Import grundlegende Funktionalität"""
        # Erstelle Original-Memory
        memory1 = WorkingMemory()
        memory1.push_context(
            ContextType.QUESTION, "Original Query", entities=["entity1"]
        )
        memory1.add_reasoning_state("step1", "Step 1", data={"key": "value"})
        memory1.set_variable("test_var", "test_value")

        # Export
        export_file = tmp_path / "memory_import_test.json"
        memory1.export_to_json(str(export_file))

        # Importiere in neues Memory
        memory2 = WorkingMemory()
        success = memory2.import_from_json(str(export_file))

        assert success
        assert memory2.get_stack_depth() == 1

        # Verifiziere importierte Daten
        current = memory2.get_current_context()
        assert current.query == "Original Query"
        assert "entity1" in current.entities
        assert current.variables["test_var"] == "test_value"
        assert len(current.reasoning_states) == 1
        assert current.reasoning_states[0].description == "Step 1"

    def test_import_from_json_complex_stack(self, tmp_path):
        """Test: Import eines komplexen verschachtelten Stacks"""
        # Erstelle komplexen Stack
        memory1 = WorkingMemory()

        for i in range(5):
            memory1.push_context(
                ContextType.QUESTION,
                f"Query {i+1}",
                entities=[f"entity_{i+1}"],
                relations={"IS_A": [f"relation_{i+1}"]},
                metadata={"level": i + 1},
            )
            memory1.add_reasoning_state(f"step_{i+1}", f"Description {i+1}")
            memory1.set_variable(f"var_{i+1}", f"value_{i+1}")

        # Export
        export_file = tmp_path / "complex_stack.json"
        memory1.export_to_json(str(export_file))

        # Import
        memory2 = WorkingMemory()
        success = memory2.import_from_json(str(export_file))

        assert success
        assert memory2.get_stack_depth() == 5

        # Verifiziere alle Frames
        for i, frame in enumerate(memory2.context_stack):
            assert frame.query == f"Query {i+1}"
            assert f"entity_{i+1}" in frame.entities
            assert frame.metadata["level"] == i + 1
            assert len(frame.reasoning_states) == 1
            assert frame.variables[f"var_{i+1}"] == f"value_{i+1}"

    def test_import_overwrites_existing_memory(self, tmp_path):
        """Test: Import überschreibt existierendes Memory"""
        # Erstelle Memory 1
        memory1 = WorkingMemory()
        memory1.push_context(ContextType.QUESTION, "Original Query")

        # Export
        export_file = tmp_path / "overwrite_test.json"
        memory1.export_to_json(str(export_file))

        # Erstelle Memory 2 mit anderem Inhalt
        memory2 = WorkingMemory()
        memory2.push_context(ContextType.CLARIFICATION, "Different Query 1")
        memory2.push_context(ContextType.CLARIFICATION, "Different Query 2")
        assert memory2.get_stack_depth() == 2

        # Import sollte Memory 2 überschreiben
        success = memory2.import_from_json(str(export_file))
        assert success
        assert memory2.get_stack_depth() == 1
        assert memory2.get_current_context().query == "Original Query"

    def test_export_debug_report_basic(self, tmp_path):
        """Test: Debug-Report-Export grundlegende Funktionalität"""
        memory = WorkingMemory()

        memory.push_context(
            ContextType.QUESTION, "Test Query", entities=["entity1", "entity2"]
        )
        memory.add_reasoning_state("step1", "Step 1", data={"key": "value"})
        memory.set_variable("test_var", "test_value")

        # Export Debug-Report
        report_file = tmp_path / "debug_report.txt"
        success = memory.export_debug_report(str(report_file))

        assert success
        assert report_file.exists()

        # Verifiziere Report-Inhalt
        with open(report_file, "r", encoding="utf-8") as f:
            content = f.read()

        assert "WORKING MEMORY DEBUG REPORT" in content
        assert "Test Query" in content
        assert "entity1" in content
        assert "entity2" in content
        assert "test_var" in content
        assert "Step 1" in content

    def test_export_debug_report_complex(self, tmp_path):
        """Test: Debug-Report für komplexen Stack"""
        memory = WorkingMemory()

        # Erstelle mehrere verschachtelte Frames
        for i in range(3):
            memory.push_context(
                ContextType.QUESTION, f"Query Level {i+1}", entities=[f"entity_{i+1}"]
            )
            memory.add_reasoning_state(f"step_{i+1}", f"Description {i+1}")

        # Export
        report_file = tmp_path / "debug_report_complex.txt"
        success = memory.export_debug_report(str(report_file))

        assert success

        # Verifiziere Report
        with open(report_file, "r", encoding="utf-8") as f:
            content = f.read()

        assert "Stack-Tiefe: 3" in content
        assert "CONTEXT-STACK (Tree-View)" in content
        assert "Query Level 1" in content
        assert "Query Level 2" in content
        assert "Query Level 3" in content

    def test_export_debug_report_empty_memory(self, tmp_path):
        """Test: Debug-Report für leeres Memory"""
        memory = WorkingMemory()

        # Export leeres Memory
        report_file = tmp_path / "debug_report_empty.txt"
        success = memory.export_debug_report(str(report_file))

        assert success

        with open(report_file, "r", encoding="utf-8") as f:
            content = f.read()

        assert "Working Memory ist leer" in content
        assert "Stack-Tiefe: 0" in content

    def test_roundtrip_export_import(self, tmp_path):
        """Test: Roundtrip Export -> Import -> Export sollte identisch sein"""
        # Erstelle komplexes Memory
        memory1 = WorkingMemory()

        for i in range(3):
            memory1.push_context(
                ContextType.QUESTION,
                f"Query {i+1}",
                entities=[f"entity_{i+1}"],
                relations={"IS_A": [f"rel_{i+1}"]},
                metadata={"level": i + 1},
            )
            memory1.add_reasoning_state(
                f"step_{i+1}", f"Desc {i+1}", data={"num": i + 1}
            )
            memory1.set_variable(f"var_{i+1}", f"value_{i+1}")

        # Export 1
        export1_file = tmp_path / "roundtrip1.json"
        memory1.export_to_json(str(export1_file))

        # Import
        memory2 = WorkingMemory()
        memory2.import_from_json(str(export1_file))

        # Export 2
        export2_file = tmp_path / "roundtrip2.json"
        memory2.export_to_json(str(export2_file))

        # Vergleiche beide JSON-Dateien (ohne Timestamps wegen Timing-Unterschieden)
        memory1.export_to_json(
            str(tmp_path / "roundtrip1_no_ts.json"), include_timestamps=False
        )
        memory2.export_to_json(
            str(tmp_path / "roundtrip2_no_ts.json"), include_timestamps=False
        )

        with (
            open(tmp_path / "roundtrip1_no_ts.json", "r") as f1,
            open(tmp_path / "roundtrip2_no_ts.json", "r") as f2,
        ):
            data1 = json.load(f1)
            data2 = json.load(f2)

        # Sollten identisch sein
        assert data1 == data2

    def test_import_invalid_json(self, tmp_path):
        """Test: Import einer ungültigen JSON-Datei"""
        memory = WorkingMemory()

        # Erstelle ungültige JSON-Datei
        invalid_file = tmp_path / "invalid.json"
        with open(invalid_file, "w") as f:
            f.write('{"invalid": "json", "no_frames": true}')

        # Import sollte fehlschlagen
        success = memory.import_from_json(str(invalid_file))
        assert not success

    def test_import_nonexistent_file(self):
        """Test: Import einer nicht existierenden Datei"""
        memory = WorkingMemory()

        success = memory.import_from_json("/nonexistent/path/file.json")
        assert not success


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
