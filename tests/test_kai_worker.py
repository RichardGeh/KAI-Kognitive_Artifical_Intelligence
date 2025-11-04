"""
KAI Test Suite - Index und Uebersicht

HINWEIS: Diese Datei wurde aufgeteilt in spezialisierte Test-Dateien fuer bessere Wartbarkeit.

=============================================================================
TEST-ORGANISATION
=============================================================================

Die Tests sind nun in folgende Dateien aufgeteilt:

KOMPONENTEN-TESTS (Basic):
---------------------------
- test_netzwerk_basic.py              - Netzwerk/Neo4j Grundfunktionen
- test_meaning_extractor_basic.py     - Intent-Erkennung Basis-Tests
- test_prototype_matcher_basic.py     - Pattern Learning Basis-Tests
- test_episodic_memory_basic.py       - Episodisches Gedaechtnis Basis-Tests
- test_embedding_service.py           - Embedding Service Tests
- test_enhanced_nlp.py                - NLP-Features Tests

KOMPONENTEN-TESTS (Additional):
--------------------------------
- test_netzwerk_additional.py         - Erweiterte Netzwerk-Tests
- test_meaning_extractor_additional.py - Erweiterte Intent-Tests
- test_prototype_matcher_additional.py - Erweiterte Pattern-Tests
- test_episodic_reasoning.py          - Episodisches Reasoning
- test_linguistik_engine.py           - Linguistische Verarbeitung
- test_autonomous_learning.py         - Autonomes Lernen
- test_lerne_command.py               - "Lerne:"-Befehl Tests

SYSTEM-TESTS:
--------------
- test_system_setup.py                - Initiales System-Setup
- test_kai_integration.py             - End-to-End Integration Tests
                                        (inkl. Edge Cases, DB-Konsistenz, Performance)

FEATURE-TESTS:
--------------
- test_goal_planner.py                - Goal Planning System
- test_dialog_system.py               - Multi-Turn Dialog System
- test_interactive_learning.py        - Interaktives Lernen
- test_intelligent_ingestion.py       - Intelligente Text-Verarbeitung
- test_w_fragen.py                    - W-Fragen Verarbeitung
- test_all_relation_types.py          - Alle Relationstypen

REASONING-TESTS:
----------------
- test_backward_chaining.py           - Backward Chaining
- test_graph_traversal.py             - Multi-Hop Graph Reasoning
- test_working_memory.py              - Working Memory System
- test_abductive_reasoning.py         - Abductive Reasoning
- test_probabilistic_engine.py        - Probabilistisches Reasoning

VISUALISIERUNG & UI:
--------------------
- test_proof_explanation.py           - Proof Tree Datenstrukturen
- test_proof_tree_widget.py           - Proof Tree Visualisierung

UTILITIES & INFRASTRUCTURE:
---------------------------
- test_logging_and_exceptions.py     - Logging System
- test_kai_exceptions.py              - Exception Handling
- test_exception_handling.py          - Exception-Szenarien
- test_text_normalization.py         - Text-Normalisierung
- test_plural_and_definitions.py      - Plural & Definitionen
- test_auto_detect_definitions.py     - Auto-Erkennung von Definitionen

FIXTURES:
---------
- conftest.py                         - Gemeinsame Pytest-Fixtures

=============================================================================
VERWENDUNG
=============================================================================

Alle Tests ausfuehren:
    pytest tests/ -v

Spezifische Kategorie:
    pytest tests/test_kai_integration.py -v
    pytest tests/test_netzwerk_basic.py -v
    pytest tests/test_backward_chaining.py -v

Einzelner Test:
    pytest tests/test_kai_integration.py::TestKaiWorkerIntegration -v

=============================================================================
BACKUP
=============================================================================

Das Original test_kai_worker.py wurde gesichert als:
    tests/test_kai_worker_BACKUP_*.py

=============================================================================
"""


def test_readme_exists():
    """Dummy-Test um sicherzustellen, dass diese README-Datei erkannt wird."""
    assert True, "README fuer Test-Organisation wurde geladen"


if __name__ == "__main__":
    print(__doc__)
