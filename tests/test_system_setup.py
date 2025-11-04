"""
KAI Test Suite - System Setup Tests
Extrahiert aus test_kai_worker.py fuer bessere Wartbarkeit.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

logger = logging.getLogger(__name__)
TEST_VECTOR_DIM = 384


class TestSystemSetup:
    """Tests für die Systeminitialisierung."""

    def test_initial_knowledge_rules_exist(self, netzwerk_session):
        """Prüft, ob die initialen Regeln aus setup_initial_knowledge.py existieren."""
        rules = netzwerk_session.get_all_extraction_rules()

        # Erwartete Regeln
        expected_rules = ["IS_A", "HAS_PROPERTY", "CAPABLE_OF", "PART_OF", "LOCATED_IN"]
        existing_rule_types = {r["relation_type"] for r in rules}

        for expected in expected_rules:
            assert (
                expected in existing_rule_types
            ), f"Regel '{expected}' fehlt. Bitte setup_initial_knowledge.py ausführen."

    def test_initial_triggers_exist(self, netzwerk_session):
        """Prüft, ob initiale Trigger existieren."""
        triggers = netzwerk_session.get_lexical_triggers()

        # Mindestens einige Trigger sollten vorhanden sein
        expected_triggers = ["ist", "hat", "kann"]
        for trigger in expected_triggers:
            assert (
                trigger in triggers
            ), f"Trigger '{trigger}' fehlt. Bitte setup_initial_knowledge.py ausführen."


# ============================================================================
# PERFORMANCE UND EDGE CASES
# ============================================================================
