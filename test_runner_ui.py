"""
test_runner_ui.py

Test-Runner UI fuer KAI - Hierarchische Test-Ausfuehrung mit Live-Updates.

FACADE MODULE - Backward Compatibility Layer.

Dieses Modul wurde in folgende Module aufgeteilt:
- test_runner_data.py: Test-Definitionen (get_all_test_data)
- test_runner_thread.py: TestRunnerThread Klasse
- test_runner_widget.py: TestRunnerTab und TestRunnerWindow Widgets

Alle oeffentlichen Klassen und Funktionen werden hier re-exportiert
fuer vollstaendige Backward-Compatibility.
"""

# Re-export all public classes and functions
from test_runner_data import get_all_test_data
from test_runner_thread import TestRunnerThread
from test_runner_widget import TestRunnerTab, TestRunnerWindow

# Backward compatibility alias
get_test_data = get_all_test_data

__all__ = [
    "TestRunnerThread",
    "TestRunnerTab",
    "TestRunnerWindow",
    "get_all_test_data",
    "get_test_data",  # Backward compatibility alias
]
