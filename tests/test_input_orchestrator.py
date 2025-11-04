# tests/test_input_orchestrator.py
"""
Tests für Input Orchestrator - Segmentierung komplexer Eingaben

Testet:
- Segmentierung von Erklärungen und Fragen
- Klassifikation von Segmenten
- Erstellung von orchestrierten Plänen
- Integration in KAI Worker
"""

import pytest
from component_41_input_orchestrator import InputOrchestrator, SegmentType
from component_5_linguistik_strukturen import GoalType


class TestInputSegmentation:
    """Tests für die Segmentierung von Eingaben."""

    @pytest.fixture
    def orchestrator(self):
        """Erstellt einen InputOrchestrator für Tests."""
        return InputOrchestrator()

    def test_simple_question_no_orchestration(self, orchestrator):
        """Einfache Frage sollte NICHT orchestriert werden."""
        text = "Was ist ein Apfel?"

        result = orchestrator.orchestrate_input(text)

        assert result is None  # Keine Orchestrierung für einfache Fragen

    def test_simple_statement_no_orchestration(self, orchestrator):
        """Einfache Aussage sollte NICHT orchestriert werden."""
        text = "Ein Apfel ist eine Frucht."

        result = orchestrator.orchestrate_input(text)

        assert result is None  # Keine Orchestrierung für einfache Aussagen

    def test_explanation_plus_question_orchestration(self, orchestrator):
        """Erklärung + Frage sollte orchestriert werden."""
        text = "Ein Apfel ist eine Frucht. Was ist ein Apfel?"

        result = orchestrator.orchestrate_input(text)

        assert result is not None
        assert "segments" in result
        assert "plan" in result
        assert len(result["segments"]) == 2

        # Prüfe Segment-Typen
        segments = result["segments"]
        assert segments[0].segment_type == SegmentType.EXPLANATION
        assert segments[1].segment_type == SegmentType.QUESTION

    def test_logic_puzzle_format(self, orchestrator):
        """Logik-Rätsel-Format sollte korrekt erkannt werden."""
        text = """
        Ein Hund ist ein Säugetier. Ein Hund kann bellen.
        Ein Hund hat vier Beine. Was ist ein Hund?
        """

        result = orchestrator.orchestrate_input(text)

        assert result is not None
        segments = result["segments"]

        # Prüfe dass Erklärungen vor Frage kommen
        explanation_count = sum(1 for s in segments if s.is_explanation())
        question_count = sum(1 for s in segments if s.is_question())

        assert explanation_count >= 3
        assert question_count >= 1

    def test_multiple_questions_after_explanation(self, orchestrator):
        """Mehrere Fragen nach Erklärungen sollten orchestriert werden."""
        text = """
        Ein Vogel ist ein Tier. Ein Vogel kann fliegen.
        Was ist ein Vogel? Kann ein Vogel fliegen?
        """

        result = orchestrator.orchestrate_input(text)

        assert result is not None
        segments = result["segments"]

        # Prüfe dass mehrere Fragen erkannt werden
        questions = [s for s in segments if s.is_question()]
        assert len(questions) >= 2


class TestSegmentClassification:
    """Tests für die Klassifikation von Segmenten."""

    @pytest.fixture
    def orchestrator(self):
        """Erstellt einen InputOrchestrator für Tests."""
        return InputOrchestrator()

    def test_classify_question_with_mark(self, orchestrator):
        """Frage mit Fragezeichen sollte als QUESTION klassifiziert werden."""
        text = "Was ist ein Apfel?"

        segment = orchestrator.classify_segment(text)

        assert segment.segment_type == SegmentType.QUESTION
        assert segment.confidence >= 0.9

    def test_classify_question_with_wh_word(self, orchestrator):
        """Frage mit Fragewort sollte als QUESTION klassifiziert werden."""
        text = "Wer ist das"

        segment = orchestrator.classify_segment(text)

        assert segment.segment_type == SegmentType.QUESTION
        assert segment.confidence >= 0.85

    def test_classify_explanation_with_pattern(self, orchestrator):
        """Erklärung mit deklarativem Muster sollte als EXPLANATION klassifiziert werden."""
        text = "Ein Apfel ist eine Frucht."

        segment = orchestrator.classify_segment(text)

        assert segment.segment_type == SegmentType.EXPLANATION
        assert segment.confidence >= 0.7

    def test_classify_command(self, orchestrator):
        """Expliziter Befehl sollte als COMMAND klassifiziert werden."""
        text = "Lerne: Ein Apfel ist rot."

        segment = orchestrator.classify_segment(text)

        assert segment.segment_type == SegmentType.COMMAND
        assert segment.confidence == 1.0


class TestOrchestratedPlan:
    """Tests für die Erstellung von orchestrierten Plänen."""

    @pytest.fixture
    def orchestrator(self):
        """Erstellt einen InputOrchestrator für Tests."""
        return InputOrchestrator()

    def test_orchestrated_plan_structure(self, orchestrator):
        """Orchestrierter Plan sollte korrekte Struktur haben."""
        text = "Ein Apfel ist rot. Was ist ein Apfel?"

        result = orchestrator.orchestrate_input(text)

        assert result is not None
        plan = result["plan"]

        # Plan sollte GoalType.PERFORM_TASK haben
        assert plan.type == GoalType.PERFORM_TASK

        # Sub-Goals sollten in korrekter Reihenfolge sein
        assert len(plan.sub_goals) >= 2

        # Erstes Sub-Goal sollte Batch-Learning sein
        first_goal = plan.sub_goals[0]
        assert "Lerne Kontext:" in first_goal.description
        assert first_goal.metadata["orchestrated_type"] == "batch_learning"

        # Letztes Sub-Goal sollte Frage sein
        last_goal = plan.sub_goals[-1]
        assert "Beantworte Frage:" in last_goal.description
        assert last_goal.metadata["orchestrated_type"] == "question_answering"

    def test_orchestrated_plan_metadata(self, orchestrator):
        """Orchestrierter Plan sollte korrekte Metadata haben."""
        text = "Ein Hund ist ein Tier. Ein Hund kann bellen. Was ist ein Hund?"

        result = orchestrator.orchestrate_input(text)

        assert result is not None
        metadata = result["metadata"]

        assert metadata["explanation_count"] == 2
        assert metadata["question_count"] == 1
        assert metadata["total_segments"] == 3


class TestEdgeCases:
    """Tests für Edge Cases."""

    @pytest.fixture
    def orchestrator(self):
        """Erstellt einen InputOrchestrator für Tests."""
        return InputOrchestrator()

    def test_empty_input(self, orchestrator):
        """Leere Eingabe sollte nicht orchestriert werden."""
        text = ""

        result = orchestrator.orchestrate_input(text)

        assert result is None

    def test_whitespace_only(self, orchestrator):
        """Whitespace-only Eingabe sollte nicht orchestriert werden."""
        text = "   \n   \t   "

        result = orchestrator.orchestrate_input(text)

        assert result is None

    def test_explicit_command_no_orchestration(self, orchestrator):
        """Explizite Befehle sollten NICHT orchestriert werden."""
        text = "Lerne: Ein Apfel ist rot. Ein Apfel ist süß."

        result = orchestrator.orchestrate_input(text)

        # Sollte nicht orchestriert werden, da expliziter Befehl
        assert result is None

    def test_only_questions_no_orchestration(self, orchestrator):
        """Nur Fragen (ohne Erklärungen) sollten nicht orchestriert werden."""
        text = "Was ist ein Apfel? Ist ein Apfel rot?"

        result = orchestrator.orchestrate_input(text)

        # Sollte nicht orchestriert werden, da keine Erklärungen
        assert result is None

    def test_only_explanations_no_orchestration(self, orchestrator):
        """Nur Erklärungen (ohne Fragen) sollten nicht orchestriert werden."""
        text = "Ein Apfel ist rot. Ein Apfel ist süß."

        result = orchestrator.orchestrate_input(text)

        # Sollte nicht orchestriert werden, da keine Fragen
        assert result is None


class TestRealWorldScenarios:
    """Tests mit realistischen Szenarien."""

    @pytest.fixture
    def orchestrator(self):
        """Erstellt einen InputOrchestrator für Tests."""
        return InputOrchestrator()

    def test_simple_logic_puzzle(self, orchestrator):
        """Einfaches Logik-Rätsel."""
        text = """
        Ein Vogel ist ein Tier. Ein Vogel kann fliegen.
        Ein Pinguin ist ein Vogel. Ein Pinguin kann nicht fliegen.
        Kann ein Pinguin fliegen?
        """

        result = orchestrator.orchestrate_input(text)

        assert result is not None
        segments = result["segments"]

        # 4 Erklärungen + 1 Frage
        explanations = [s for s in segments if s.is_explanation()]
        questions = [s for s in segments if s.is_question()]

        assert len(explanations) == 4
        assert len(questions) == 1

        # Plan sollte Batch-Learning + Question-Answering haben
        plan = result["plan"]
        assert len(plan.sub_goals) == 2  # Batch-Learning + Frage

    def test_complex_logic_puzzle(self, orchestrator):
        """Komplexes Logik-Rätsel mit mehreren Fragen."""
        text = """
        Alice ist eine Person. Bob ist eine Person.
        Alice hat einen Hut. Bob hat keinen Hut.
        Wer hat einen Hut? Ist Alice eine Person?
        """

        result = orchestrator.orchestrate_input(text)

        assert result is not None
        segments = result["segments"]

        # 4 Erklärungen + 2 Fragen
        explanations = [s for s in segments if s.is_explanation()]
        questions = [s for s in segments if s.is_question()]

        assert len(explanations) == 4
        assert len(questions) == 2

        # Plan sollte Batch-Learning + 2x Question-Answering haben
        plan = result["plan"]
        assert len(plan.sub_goals) == 3  # Batch-Learning + 2 Fragen

    def test_german_logic_puzzle(self, orchestrator):
        """Deutsches Logik-Rätsel (typisches Format)."""
        text = """
        Eine Katze ist ein Säugetier. Eine Katze kann miauen.
        Ein Hund ist ein Säugetier. Ein Hund kann bellen.
        Was ist eine Katze? Was ist ein Hund?
        """

        result = orchestrator.orchestrate_input(text)

        assert result is not None

        # Plan sollte korrekt strukturiert sein
        plan = result["plan"]

        # Erstes Sub-Goal: Batch-Learning
        assert "Lerne Kontext:" in plan.sub_goals[0].description

        # Letzte Sub-Goals: Fragen
        assert "Beantworte Frage:" in plan.sub_goals[1].description
        assert "Beantworte Frage:" in plan.sub_goals[2].description


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
