"""
KAI Test Suite - Goal Planner Tests
Extrahiert aus test_kai_worker.py fuer bessere Wartbarkeit.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

logger = logging.getLogger(__name__)
TEST_VECTOR_DIM = 384


class TestGoalPlanner:
    """Tests für den Goal Planner - Konvertierung von MeaningPoints zu MainGoals."""

    def test_high_confidence_direct_execution(self):
        """Testet direkten Ausführungsplan bei hoher Konfidenz (>= 0.8)."""
        from component_4_goal_planner import GoalPlanner
        from component_5_linguistik_strukturen import (
            MeaningPoint,
            MeaningPointCategory,
            Modality,
            Polarity,
            GoalType,
        )

        planner = GoalPlanner()

        # MeaningPoint mit hoher Konfidenz
        mp = MeaningPoint(
            id="test-high-conf",
            category=MeaningPointCategory.QUESTION,
            cue="was",
            text_span="Was ist ein Apfel?",
            modality=Modality.INTERROGATIVE,
            polarity=Polarity.POSITIVE,
            confidence=0.9,  # Hohe Konfidenz
            arguments={"topic": "apfel"},
        )

        plan = planner.create_plan(mp)

        assert plan is not None
        assert plan.type == GoalType.ANSWER_QUESTION
        assert (
            "[Bestätigung erforderlich]" not in plan.description
        )  # Keine Bestätigung nötig
        assert len(plan.sub_goals) == 4  # Standard-Frage-Plan

        logger.info("[SUCCESS] Hohe Konfidenz führt zu direktem Ausführungsplan")

    def test_medium_confidence_confirmation_plan(self):
        """Testet Bestätigungsplan bei mittlerer Konfidenz (0.4 <= conf < 0.8)."""
        from component_4_goal_planner import GoalPlanner
        from component_5_linguistik_strukturen import (
            MeaningPoint,
            MeaningPointCategory,
            Modality,
            Polarity,
            GoalType,
        )

        planner = GoalPlanner()

        # MeaningPoint mit mittlerer Konfidenz
        mp = MeaningPoint(
            id="test-med-conf",
            category=MeaningPointCategory.QUESTION,
            cue="was",
            text_span="Was ist ein unklares Ding?",
            modality=Modality.INTERROGATIVE,
            polarity=Polarity.POSITIVE,
            confidence=0.6,  # Mittlere Konfidenz
            arguments={"topic": "ding"},
        )

        plan = planner.create_plan(mp)

        assert plan is not None
        assert plan.type == GoalType.ANSWER_QUESTION
        assert "[Bestätigung erforderlich]" in plan.description
        assert len(plan.sub_goals) == 5  # +1 Bestätigungs-SubGoal
        assert "Bestätige die erkannte Absicht" in plan.sub_goals[0].description

        logger.info("[SUCCESS] Mittlere Konfidenz fügt Bestätigungsschritt hinzu")

    def test_low_confidence_clarification_plan(self):
        """Testet Klarstellungsplan bei niedriger Konfidenz (< 0.4)."""
        from component_4_goal_planner import GoalPlanner
        from component_5_linguistik_strukturen import (
            MeaningPoint,
            MeaningPointCategory,
            Modality,
            Polarity,
            GoalType,
        )

        planner = GoalPlanner()

        # MeaningPoint mit niedriger Konfidenz
        mp = MeaningPoint(
            id="test-low-conf",
            category=MeaningPointCategory.QUESTION,
            cue="was",
            text_span="Völlig unklarer Text xyz",
            modality=Modality.INTERROGATIVE,
            polarity=Polarity.POSITIVE,
            confidence=0.2,  # Niedrige Konfidenz
            arguments={"topic": "xyz"},
        )

        plan = planner.create_plan(mp)

        assert plan is not None
        assert plan.type == GoalType.CLARIFY_INTENT
        assert "Kläre die Absicht" in plan.description
        assert len(plan.sub_goals) == 1  # Nur Klarstellungsfrage

        logger.info("[SUCCESS] Niedrige Konfidenz führt zu Klarstellungsplan")

    def test_unknown_category_clarification_plan(self):
        """Testet dass UNKNOWN category zu Clarification führt."""
        from component_4_goal_planner import GoalPlanner
        from component_5_linguistik_strukturen import (
            MeaningPoint,
            MeaningPointCategory,
            Modality,
            Polarity,
            GoalType,
        )

        planner = GoalPlanner()

        # MeaningPoint mit UNKNOWN category
        mp = MeaningPoint(
            id="test-unknown",
            category=MeaningPointCategory.UNKNOWN,
            cue="",
            text_span="Irgendein unbekannter Input",
            modality=Modality.DECLARATIVE,
            polarity=Polarity.POSITIVE,
            confidence=0.5,  # Konfidenz ist egal bei UNKNOWN
            arguments={},
        )

        plan = planner.create_plan(mp)

        assert plan is not None
        assert plan.type == GoalType.CLARIFY_INTENT

        logger.info("[SUCCESS] UNKNOWN category führt zu Clarification Plan")

    def test_plan_for_question_structure(self):
        """Testet die Struktur eines Frage-Plans."""
        from component_4_goal_planner import GoalPlanner
        from component_5_linguistik_strukturen import (
            MeaningPoint,
            MeaningPointCategory,
            Modality,
            Polarity,
            GoalType,
        )

        planner = GoalPlanner()

        mp = MeaningPoint(
            id="test-q",
            category=MeaningPointCategory.QUESTION,
            cue="was",
            text_span="Was ist ein Test?",
            modality=Modality.INTERROGATIVE,
            polarity=Polarity.POSITIVE,
            confidence=0.9,
            arguments={"topic": "test"},
        )

        plan = planner.create_plan(mp)

        assert plan.type == GoalType.ANSWER_QUESTION
        assert "Beantworte die Frage" in plan.description

        # Prüfe SubGoal-Struktur
        expected_steps = [
            "Identifiziere das Thema",
            "Frage den Wissensgraphen",
            "Prüfe auf Wissenslücken",
            "Formuliere eine Antwort",
        ]

        for i, expected in enumerate(expected_steps):
            assert (
                expected in plan.sub_goals[i].description
            ), f"SubGoal {i} sollte '{expected}' enthalten"

        logger.info("[SUCCESS] Frage-Plan hat korrekte Struktur")

    def test_plan_for_define_command_structure(self):
        """Testet die Struktur eines Definiere-Befehls-Plans."""
        from component_4_goal_planner import GoalPlanner
        from component_5_linguistik_strukturen import (
            MeaningPoint,
            MeaningPointCategory,
            Modality,
            Polarity,
            GoalType,
        )

        planner = GoalPlanner()

        mp = MeaningPoint(
            id="test-def",
            category=MeaningPointCategory.COMMAND,
            cue="definiere",
            text_span="Definiere: apfel / bedeutung = frucht",
            modality=Modality.IMPERATIVE,
            polarity=Polarity.POSITIVE,
            confidence=0.95,
            arguments={
                "command": "definiere",
                "topic": "apfel",
                "key_path": "bedeutung",
                "value": "frucht",
            },
        )

        plan = planner.create_plan(mp)

        assert plan.type == GoalType.LEARN_KNOWLEDGE
        assert "Lerne Wissen aus Befehl" in plan.description
        assert len(plan.sub_goals) == 3

        logger.info("[SUCCESS] Definiere-Plan hat korrekte Struktur")

    def test_plan_for_learn_pattern_command_structure(self):
        """Testet die Struktur eines Lerne-Muster-Plans."""
        from component_4_goal_planner import GoalPlanner
        from component_5_linguistik_strukturen import (
            MeaningPoint,
            MeaningPointCategory,
            Modality,
            Polarity,
            GoalType,
        )

        planner = GoalPlanner()

        mp = MeaningPoint(
            id="test-learn",
            category=MeaningPointCategory.COMMAND,
            cue="lerne",
            text_span='Lerne Muster: "X ist Y" bedeutet IS_A',
            modality=Modality.IMPERATIVE,
            polarity=Polarity.POSITIVE,
            confidence=0.98,
            arguments={
                "command": "learn_pattern",
                "example_sentence": "X ist Y",
                "relation_type": "IS_A",
            },
        )

        plan = planner.create_plan(mp)

        assert plan.type == GoalType.LEARN_KNOWLEDGE
        assert "Lehre KAI die Bedeutung des Musters" in plan.description

        # Prüfe Meta-Learning-Struktur
        expected_steps = [
            "Verarbeite Beispielsatz zu Vektor",
            "Finde oder erstelle zugehörigen Muster-Prototypen",
            "Verknüpfe Prototyp mit Extraktionsregel",
            "Formuliere eine Lernbestätigung",
        ]

        assert len(plan.sub_goals) == 4
        for i, expected in enumerate(expected_steps):
            assert expected in plan.sub_goals[i].description

        logger.info("[SUCCESS] Lerne-Muster-Plan hat korrekte Meta-Learning-Struktur")

    def test_plan_for_ingest_text_command_structure(self):
        """Testet die Struktur eines Text-Ingestion-Plans."""
        from component_4_goal_planner import GoalPlanner
        from component_5_linguistik_strukturen import (
            MeaningPoint,
            MeaningPointCategory,
            Modality,
            Polarity,
            GoalType,
        )

        planner = GoalPlanner()

        mp = MeaningPoint(
            id="test-ingest",
            category=MeaningPointCategory.COMMAND,
            cue="ingestiere",
            text_span='Ingestiere Text: "Ein Hund ist ein Tier"',
            modality=Modality.IMPERATIVE,
            polarity=Polarity.POSITIVE,
            confidence=1.0,
            arguments={
                "command": "ingest_text",
                "text_to_ingest": "Ein Hund ist ein Tier",
            },
        )

        plan = planner.create_plan(mp)

        assert plan.type == GoalType.PERFORM_TASK
        assert "Ingestiere Text" in plan.description
        assert len(plan.sub_goals) == 3

        logger.info("[SUCCESS] Ingestion-Plan hat korrekte Struktur")

    def test_plan_for_auto_detected_definition(self):
        """Testet Plan für automatisch erkannte Definitionen."""
        from component_4_goal_planner import GoalPlanner
        from component_5_linguistik_strukturen import (
            MeaningPoint,
            MeaningPointCategory,
            Modality,
            Polarity,
            GoalType,
        )

        planner = GoalPlanner()

        mp = MeaningPoint(
            id="test-auto-def",
            category=MeaningPointCategory.DEFINITION,
            cue="ist",
            text_span="Ein Apfel ist eine Frucht",
            modality=Modality.DECLARATIVE,
            polarity=Polarity.POSITIVE,
            confidence=0.85,
            arguments={"subject": "apfel", "relation_type": "IS_A", "object": "frucht"},
        )

        plan = planner.create_plan(mp)

        assert plan.type == GoalType.LEARN_KNOWLEDGE
        assert "automatisch erkannte Relation" in plan.description
        assert "apfel" in plan.description
        assert "IS_A" in plan.description
        assert "frucht" in plan.description

        logger.info("[SUCCESS] Auto-detected Definition Plan korrekt erstellt")

    def test_edge_case_invalid_meaning_point(self):
        """Testet Verhalten bei ungültigem MeaningPoint."""
        from component_4_goal_planner import GoalPlanner
        from component_5_linguistik_strukturen import (
            MeaningPoint,
            MeaningPointCategory,
            Modality,
            Polarity,
        )

        planner = GoalPlanner()

        # MeaningPoint mit unbekanntem Befehl
        mp = MeaningPoint(
            id="test-invalid",
            category=MeaningPointCategory.COMMAND,
            cue="unbekannt",
            text_span="Unbekannter Befehl xyz",
            modality=Modality.IMPERATIVE,
            polarity=Polarity.POSITIVE,
            confidence=0.9,
            arguments={"command": "unbekannter_befehl"},
        )

        plan = planner.create_plan(mp)

        # Sollte None oder Fallback-Plan zurückgeben
        assert plan is None or plan.type == GoalType.CLARIFY_INTENT

        logger.info("[SUCCESS] Ungültiger MeaningPoint wird korrekt behandelt")

    def test_confidence_threshold_boundaries(self):
        """Testet exakte Grenzen der Confidence Thresholds."""
        from component_4_goal_planner import GoalPlanner
        from component_5_linguistik_strukturen import (
            MeaningPoint,
            MeaningPointCategory,
            Modality,
            Polarity,
            GoalType,
        )

        planner = GoalPlanner()

        # Edge Case 1: Exakt 0.4 (sollte Confirmation sein, nicht Clarification)
        mp_04 = MeaningPoint(
            id="test-04",
            category=MeaningPointCategory.QUESTION,
            cue="was",
            text_span="Test",
            modality=Modality.INTERROGATIVE,
            polarity=Polarity.POSITIVE,
            confidence=0.4,
            arguments={"topic": "test"},
        )
        plan_04 = planner.create_plan(mp_04)
        assert plan_04.type == GoalType.ANSWER_QUESTION  # Basis-Plan
        assert "[Bestätigung erforderlich]" in plan_04.description  # Mit Confirmation

        # Edge Case 2: Exakt 0.8 (sollte Direct Execution sein)
        mp_08 = MeaningPoint(
            id="test-08",
            category=MeaningPointCategory.QUESTION,
            cue="was",
            text_span="Test",
            modality=Modality.INTERROGATIVE,
            polarity=Polarity.POSITIVE,
            confidence=0.8,
            arguments={"topic": "test"},
        )
        plan_08 = planner.create_plan(mp_08)
        assert plan_08.type == GoalType.ANSWER_QUESTION
        assert "[Bestätigung erforderlich]" not in plan_08.description  # Direkt

        logger.info("[SUCCESS] Confidence Threshold Grenzen korrekt: 0.4 und 0.8")


# ============================================================================
# TESTS FÜR MEANING EXTRACTOR (component_7_meaning_extractor.py)
# ============================================================================
