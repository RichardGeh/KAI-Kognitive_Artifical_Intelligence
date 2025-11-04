"""
KAI Test Suite - Goal Planner File Reading Tests
Phase 3: Tests fuer Datei-Ingestion Pipeline
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import pytest
from component_4_goal_planner import GoalPlanner
from component_5_linguistik_strukturen import (
    MeaningPoint,
    MeaningPointCategory,
    Modality,
    Polarity,
    GoalType,
)

logger = logging.getLogger(__name__)


class TestGoalPlannerFileReading:
    """Tests fuer Goal Planner Datei-Ingestion (Phase 3)."""

    def test_plan_for_file_reading_structure(self):
        """Testet die Struktur eines Datei-Lese-Plans."""
        planner = GoalPlanner()

        mp = MeaningPoint(
            id="test-file",
            category=MeaningPointCategory.COMMAND,
            cue="lese",
            text_span="Lese Datei: test.txt",
            modality=Modality.IMPERATIVE,
            polarity=Polarity.POSITIVE,
            confidence=0.95,
            arguments={"command": "read_file", "file_path": "test.txt"},
        )

        plan = planner.create_plan(mp)

        assert plan is not None
        assert plan.type == GoalType.READ_DOCUMENT
        assert "Lese und verarbeite Datei" in plan.description
        assert "test.txt" in plan.description

        logger.info("[SUCCESS] Datei-Lese-Plan hat korrekten Type und Description")

    def test_plan_for_file_reading_subgoals(self):
        """Testet dass alle erwarteten SubGoals vorhanden sind."""
        planner = GoalPlanner()

        mp = MeaningPoint(
            id="test-file-subgoals",
            category=MeaningPointCategory.COMMAND,
            cue="lese",
            text_span="Lese Datei: dokument.pdf",
            modality=Modality.IMPERATIVE,
            polarity=Polarity.POSITIVE,
            confidence=0.9,
            arguments={"command": "read_file", "file_path": "dokument.pdf"},
        )

        plan = planner.create_plan(mp)

        # Pruefe SubGoal-Struktur: VALIDATE_FILE -> EXTRACT_TEXT -> LEARN_KNOWLEDGE -> REPORT
        expected_steps = [
            "Validiere Dateipfad",
            "Extrahiere Text",
            "Verarbeite extrahierten Text",
            "Formuliere Ingestion-Bericht",
        ]

        assert (
            len(plan.sub_goals) == 4
        ), f"Erwarte 4 SubGoals, erhalten {len(plan.sub_goals)}"

        for i, expected in enumerate(expected_steps):
            assert (
                expected in plan.sub_goals[i].description
            ), f"SubGoal {i} sollte '{expected}' enthalten, ist aber: '{plan.sub_goals[i].description}'"

        logger.info("[SUCCESS] Datei-Lese-Plan hat korrekte SubGoal-Pipeline")

    def test_plan_for_file_reading_with_different_paths(self):
        """Testet dass verschiedene Dateipfade korrekt im Plan erscheinen."""
        planner = GoalPlanner()

        test_paths = [
            "C:/Users/test/dokument.txt",
            "./relative/path/file.md",
            "/absolute/path/data.json",
        ]

        for file_path in test_paths:
            mp = MeaningPoint(
                id=f"test-{file_path}",
                category=MeaningPointCategory.COMMAND,
                cue="lese",
                text_span=f"Lese Datei: {file_path}",
                modality=Modality.IMPERATIVE,
                polarity=Polarity.POSITIVE,
                confidence=0.9,
                arguments={"command": "read_file", "file_path": file_path},
            )

            plan = planner.create_plan(mp)

            assert plan is not None
            assert plan.type == GoalType.READ_DOCUMENT
            assert file_path in plan.description

        logger.info(
            "[SUCCESS] Verschiedene Dateipfade werden korrekt im Plan gespeichert"
        )

    def test_plan_for_file_reading_high_confidence_direct_execution(self):
        """Testet dass hohe Konfidenz zu direkter Ausfuehrung fuehrt (keine Bestaetigung)."""
        planner = GoalPlanner()

        mp = MeaningPoint(
            id="test-file-high-conf",
            category=MeaningPointCategory.COMMAND,
            cue="lese",
            text_span="Lese Datei: wichtig.txt",
            modality=Modality.IMPERATIVE,
            polarity=Polarity.POSITIVE,
            confidence=0.95,  # Hohe Konfidenz
            arguments={"command": "read_file", "file_path": "wichtig.txt"},
        )

        plan = planner.create_plan(mp)

        assert plan is not None
        assert plan.type == GoalType.READ_DOCUMENT
        assert "[Bestätigung erforderlich]" not in plan.description
        assert len(plan.sub_goals) == 4  # Keine zusaetzliche Bestaetigung

        logger.info("[SUCCESS] Hohe Konfidenz fuehrt zu direkter Datei-Verarbeitung")

    def test_plan_for_file_reading_medium_confidence_confirmation(self):
        """Testet dass mittlere Konfidenz zu Bestaetigung fuehrt."""
        planner = GoalPlanner()

        mp = MeaningPoint(
            id="test-file-med-conf",
            category=MeaningPointCategory.COMMAND,
            cue="lese",
            text_span="Lese Datei: unsicher.txt",
            modality=Modality.IMPERATIVE,
            polarity=Polarity.POSITIVE,
            confidence=0.6,  # Mittlere Konfidenz
            arguments={"command": "read_file", "file_path": "unsicher.txt"},
        )

        plan = planner.create_plan(mp)

        assert plan is not None
        assert plan.type == GoalType.READ_DOCUMENT
        assert "[Bestätigung erforderlich]" in plan.description
        assert len(plan.sub_goals) == 5  # +1 Bestaetigungs-SubGoal
        assert "Bestätige die erkannte Absicht" in plan.sub_goals[0].description

        logger.info(
            "[SUCCESS] Mittlere Konfidenz fuegt Bestaetigungsschritt fuer Datei hinzu"
        )

    def test_plan_for_file_reading_low_confidence_clarification(self):
        """Testet dass niedrige Konfidenz zu Klarstellung fuehrt."""
        planner = GoalPlanner()

        mp = MeaningPoint(
            id="test-file-low-conf",
            category=MeaningPointCategory.COMMAND,
            cue="lese",
            text_span="Lese vielleicht eine Datei?",
            modality=Modality.IMPERATIVE,
            polarity=Polarity.POSITIVE,
            confidence=0.3,  # Niedrige Konfidenz
            arguments={"command": "read_file", "file_path": "unbekannt.txt"},
        )

        plan = planner.create_plan(mp)

        assert plan is not None
        assert plan.type == GoalType.CLARIFY_INTENT  # Nicht READ_DOCUMENT!
        assert "Kläre die Absicht" in plan.description

        logger.info(
            "[SUCCESS] Niedrige Konfidenz fuehrt zu Klarstellung statt Datei-Verarbeitung"
        )

    def test_plan_for_file_reading_missing_file_path(self):
        """Testet Verhalten wenn Dateipfad fehlt."""
        planner = GoalPlanner()

        mp = MeaningPoint(
            id="test-file-missing",
            category=MeaningPointCategory.COMMAND,
            cue="lese",
            text_span="Lese Datei",
            modality=Modality.IMPERATIVE,
            polarity=Polarity.POSITIVE,
            confidence=0.9,
            arguments={
                "command": "read_file"
                # file_path fehlt!
            },
        )

        plan = planner.create_plan(mp)

        assert plan is not None
        assert plan.type == GoalType.READ_DOCUMENT
        # Sollte leeren String verwenden
        assert plan.description is not None

        logger.info("[SUCCESS] Fehlender Dateipfad wird graceful behandelt")

    def test_plan_for_file_reading_edge_case_empty_path(self):
        """Testet explizit leeren Dateipfad."""
        planner = GoalPlanner()

        mp = MeaningPoint(
            id="test-file-empty",
            category=MeaningPointCategory.COMMAND,
            cue="lese",
            text_span="Lese Datei: ",
            modality=Modality.IMPERATIVE,
            polarity=Polarity.POSITIVE,
            confidence=0.9,
            arguments={"command": "read_file", "file_path": ""},
        )

        plan = planner.create_plan(mp)

        assert plan is not None
        assert plan.type == GoalType.READ_DOCUMENT
        # Plan sollte erstellt werden, Validierung erfolgt im Executor
        assert len(plan.sub_goals) == 4

        logger.info(
            "[SUCCESS] Leerer Dateipfad wird akzeptiert (Validierung in Executor)"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
