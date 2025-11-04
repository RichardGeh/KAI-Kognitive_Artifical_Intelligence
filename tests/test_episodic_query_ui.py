"""
test_episodic_query_ui.py

Tests für die Episodic Query UI-Funktionalität.

Testet:
- Query-Erkennung in meaning_extractor.py
- Plan-Erstellung in goal_planner.py
- Episodic-Memory-Strategy Execution
- Response-Formatierung
"""

import pytest
from unittest.mock import Mock

from component_5_linguistik_strukturen import (
    MeaningPoint,
    MeaningPointCategory,
    Modality,
    Polarity,
)
from component_4_goal_planner import GoalPlanner, GoalType
from component_7_meaning_extractor import MeaningPointExtractor
from component_6_linguistik_engine import LinguisticPreprocessor
from component_11_embedding_service import EmbeddingService
from kai_response_formatter import KaiResponseFormatter


class TestEpisodicQueryRecognition:
    """Tests für die Erkennung episodischer Queries"""

    @pytest.fixture
    def preprocessor(self):
        """Mock Preprocessor"""
        return Mock(spec=LinguisticPreprocessor)

    @pytest.fixture
    def embedding_service(self):
        """Mock Embedding Service"""
        service = Mock(spec=EmbeddingService)
        service.is_available.return_value = (
            False  # Deaktiviere Vektor-Matching für Tests
        )
        return service

    @pytest.fixture
    def extractor(self, preprocessor, embedding_service):
        """Meaning Extractor"""
        return MeaningPointExtractor(
            embedding_service=embedding_service,
            preprocessor=preprocessor,
            prototyping_engine=None,
        )

    @pytest.fixture
    def planner(self):
        """Goal Planner"""
        return GoalPlanner()

    def test_recognize_when_learned_query(self, extractor, preprocessor):
        """Test: 'Wann habe ich X gelernt?' wird erkannt"""
        # Mock spaCy Doc
        mock_doc = Mock()
        mock_doc.text = "Wann habe ich über Hunde gelernt?"
        mock_doc.__len__ = lambda self: 1
        mock_doc.__bool__ = lambda self: True

        preprocessor.preprocess.return_value = mock_doc

        # Extract meaning
        mps = extractor.extract(mock_doc)

        assert len(mps) == 1
        mp = mps[0]
        assert mp.category == MeaningPointCategory.QUESTION
        assert mp.arguments.get("query_type") == "episodic_memory"
        assert mp.arguments.get("episodic_query_type") == "when_learned"
        assert mp.arguments.get("topic") == "hunde"
        assert mp.confidence >= 0.90

    def test_recognize_show_episodes_query(self, extractor, preprocessor):
        """Test: 'Zeige mir Episoden' wird erkannt"""
        mock_doc = Mock()
        mock_doc.text = "Zeige mir alle Episoden über Katzen"
        mock_doc.__len__ = lambda self: 1
        mock_doc.__bool__ = lambda self: True

        preprocessor.preprocess.return_value = mock_doc

        mps = extractor.extract(mock_doc)

        assert len(mps) == 1
        mp = mps[0]
        assert mp.category == MeaningPointCategory.QUESTION
        assert mp.arguments.get("query_type") == "episodic_memory"
        assert mp.arguments.get("episodic_query_type") == "show_episodes"
        assert mp.arguments.get("topic") == "katzen"
        assert mp.confidence >= 0.90

    def test_recognize_learning_history_query(self, extractor, preprocessor):
        """Test: 'Zeige Lernverlauf' wird erkannt"""
        mock_doc = Mock()
        mock_doc.text = "Zeige mir den Lernverlauf von Vögeln"
        mock_doc.__len__ = lambda self: 1
        mock_doc.__bool__ = lambda self: True

        preprocessor.preprocess.return_value = mock_doc

        mps = extractor.extract(mock_doc)

        assert len(mps) == 1
        mp = mps[0]
        assert mp.category == MeaningPointCategory.QUESTION
        assert mp.arguments.get("query_type") == "episodic_memory"
        assert mp.arguments.get("episodic_query_type") == "learning_history"
        assert mp.arguments.get("topic") == "vögeln"
        assert mp.confidence >= 0.90

    def test_recognize_what_learned_query(self, extractor, preprocessor):
        """Test: 'Was habe ich über X gelernt?' wird erkannt"""
        mock_doc = Mock()
        mock_doc.text = "Was habe ich über Fische gelernt?"
        mock_doc.__len__ = lambda self: 1
        mock_doc.__bool__ = lambda self: True

        preprocessor.preprocess.return_value = mock_doc

        mps = extractor.extract(mock_doc)

        assert len(mps) == 1
        mp = mps[0]
        assert mp.category == MeaningPointCategory.QUESTION
        assert mp.arguments.get("query_type") == "episodic_memory"
        assert mp.arguments.get("episodic_query_type") == "what_learned"
        assert mp.arguments.get("topic") == "fische"
        assert mp.confidence >= 0.90

    def test_episodic_query_without_topic(self, extractor, preprocessor):
        """Test: Episodische Query ohne Thema (alle Episoden)"""
        mock_doc = Mock()
        mock_doc.text = "Zeige mir alle Episoden"
        mock_doc.__len__ = lambda self: 1
        mock_doc.__bool__ = lambda self: True

        preprocessor.preprocess.return_value = mock_doc

        mps = extractor.extract(mock_doc)

        assert len(mps) == 1
        mp = mps[0]
        assert mp.category == MeaningPointCategory.QUESTION
        assert mp.arguments.get("query_type") == "episodic_memory"
        assert mp.arguments.get("topic") is None  # Kein Thema


class TestEpisodicQueryPlanning:
    """Tests für die Plan-Erstellung für episodische Queries"""

    @pytest.fixture
    def planner(self):
        """Goal Planner"""
        return GoalPlanner()

    def test_plan_for_episodic_query(self, planner):
        """Test: Plan-Erstellung für episodische Query"""
        mp = MeaningPoint(
            id="mp-test-001",
            category=MeaningPointCategory.QUESTION,
            cue="heuristic_episodic_when_learned",
            text_span="Wann habe ich über Hunde gelernt?",
            modality=Modality.INTERROGATIVE,
            polarity=Polarity.POSITIVE,
            confidence=0.92,
            arguments={
                "query_type": "episodic_memory",
                "episodic_query_type": "when_learned",
                "topic": "hunde",
            },
        )

        plan = planner.create_plan(mp)

        assert plan is not None
        assert plan.type == GoalType.ANSWER_QUESTION
        assert "episodische Abfrage" in plan.description.lower()
        assert len(plan.sub_goals) == 2
        assert "Frage episodisches Gedächtnis ab" in plan.sub_goals[0].description
        assert (
            "Formuliere eine Antwort mit Episoden-Zusammenfassung"
            in plan.sub_goals[1].description
        )

    def test_plan_confidence_gate(self, planner):
        """Test: Confidence Gate funktioniert für episodische Queries"""
        # High confidence -> Direct execution
        mp_high = MeaningPoint(
            id="mp-test-002",
            category=MeaningPointCategory.QUESTION,
            cue="heuristic_episodic_show_episodes",
            text_span="Zeige mir Episoden",
            modality=Modality.IMPERATIVE,
            polarity=Polarity.POSITIVE,
            confidence=0.93,
            arguments={
                "query_type": "episodic_memory",
                "episodic_query_type": "show_episodes",
            },
        )

        plan = planner.create_plan(mp_high)
        assert plan is not None
        assert plan.type == GoalType.ANSWER_QUESTION


class TestEpisodicResponseFormatting:
    """Tests für die Antwort-Formatierung"""

    @pytest.fixture
    def formatter(self):
        """Response Formatter"""
        return KaiResponseFormatter(confidence_manager=None)

    def test_format_episodic_answer_with_episodes(self, formatter):
        """Test: Formatierung mit Episoden"""
        episodes = [
            {
                "episode_id": "123",
                "type": "learning",
                "content": "Ein Hund ist ein Tier",
                "timestamp": 1700000000000,
                "learned_facts": [
                    {"subject": "hund", "relation": "IS_A", "object": "tier"}
                ],
            },
            {
                "episode_id": "456",
                "type": "ingestion",
                "content": "Hunde sind Säugetiere",
                "timestamp": 1700010000000,
                "learned_facts": [],
            },
        ]

        response = formatter.format_episodic_answer(
            episodes=episodes, query_type="when_learned", topic="hund"
        )

        assert "2 Mal über 'hund' gelernt" in response
        assert "2023-11-14" in response  # Zeitstempel formatiert
        assert "learning" in response.lower()
        assert "Episodisches Gedächtnis" in response

    def test_format_episodic_answer_empty(self, formatter):
        """Test: Formatierung ohne Episoden"""
        response = formatter.format_episodic_answer(
            episodes=[], query_type="when_learned", topic="drache"
        )

        assert "noch nichts über 'drache' gelernt" in response

    def test_format_episodic_answer_no_topic(self, formatter):
        """Test: Formatierung ohne Thema (alle Episoden)"""
        episodes = [
            {"episode_id": "1", "type": "learning", "content": "Test", "timestamp": 0}
        ]

        response = formatter.format_episodic_answer(
            episodes=episodes, query_type="show_episodes", topic=None
        )

        assert "1" in response
        assert "gesamt" in response.lower()

    def test_format_many_episodes_truncation(self, formatter):
        """Test: Viele Episoden werden abgeschnitten (nur 5 in Text)"""
        episodes = [
            {
                "episode_id": str(i),
                "type": "learning",
                "content": f"Test {i}",
                "timestamp": 0,
            }
            for i in range(10)
        ]

        response = formatter.format_episodic_answer(
            episodes=episodes, query_type="show_episodes", topic="test"
        )

        assert "10 gesamt" in response
        assert "... und 5 weitere Episoden" in response


class TestEpisodicMemoryStrategy:
    """Tests für die EpisodicMemoryStrategy (Integration)"""

    def test_strategy_can_handle(self):
        """Test: Strategy erkennt episodische SubGoals"""
        from kai_sub_goal_executor import EpisodicMemoryStrategy

        worker = Mock()
        strategy = EpisodicMemoryStrategy(worker)

        assert strategy.can_handle("Frage episodisches Gedächtnis ab über 'test'.")
        assert strategy.can_handle("Zeige Episoden")
        assert strategy.can_handle("Zeige Lernverlauf")
        assert not strategy.can_handle("Frage den Wissensgraphen nach Fakten ab")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
