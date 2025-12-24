"""
test_runner_thread.py

Thread-Klasse fuer die Ausfuehrung von pytest-Tests mit Live-Updates.

Extrahiert aus test_runner_ui.py fuer bessere Modularitaet.
"""

import subprocess
import sys
import traceback
from typing import Dict, List

from PySide6.QtCore import QThread, Signal


class TestRunnerThread(QThread):
    """
    Thread zur Ausfuehrung von pytest-Tests mit Live-Updates.

    Signals:
        test_started: Emitted wenn ein Test startet (test_name)
        test_passed: Emitted wenn ein Test erfolgreich ist (test_name)
        test_failed: Emitted wenn ein Test fehlschlaegt (test_name, error_msg)
        progress_updated: Emitted bei Fortschritt (current, total, percentage)
        finished_all: Emitted wenn alle Tests abgeschlossen sind (passed, failed, total)
        output_line: Emitted fuer jede Ausgabezeile (line)
    """

    test_started = Signal(str)
    test_passed = Signal(str)
    test_failed = Signal(str, str)  # test_name, error_message
    progress_updated = Signal(int, int, float)  # current, total, percentage
    finished_all = Signal(int, int, int)  # passed, failed, total
    output_line = Signal(str)  # Ausgabezeile

    def __init__(self, selected_tests: List[tuple]):
        super().__init__()
        self.selected_tests = selected_tests  # List of (test_file, test_class) tuples
        self.should_stop = False

    def run(self):
        """Fuehrt ausgewaehlte Tests aus und parst die Ausgabe"""
        if not self.selected_tests:
            # Keine Tests ausgewaehlt
            self.output_line.emit("[WARNING] Keine Tests ausgewaehlt")
            self.finished_all.emit(0, 0, 0)
            return

        # Gruppiere Tests nach Datei
        tests_by_file: Dict[str, List[str]] = {}
        for test_file, test_class in self.selected_tests:
            if test_file not in tests_by_file:
                tests_by_file[test_file] = []
            tests_by_file[test_file].append(test_class)

        # Baue Test-Spezifikationen
        # Fuer jede Datei: tests/test_file.py -k "TestClass1 or TestClass2"
        test_specs = []
        for test_file, test_classes in tests_by_file.items():
            test_filter = " or ".join(test_classes)
            test_specs.append((f"tests/{test_file}", test_filter))

        # Fuehre pytest mit verbose und kurzem traceback aus
        # Run all test files in a single pytest call for efficiency
        cmd = [sys.executable, "-m", "pytest"]

        # Add all test files
        for test_file_path, _ in test_specs:
            cmd.append(test_file_path)

        # Add combined filter for all test classes
        all_test_classes = [
            tc for _, test_classes in tests_by_file.items() for tc in test_classes
        ]
        test_filter = " or ".join(all_test_classes)

        cmd.extend(
            [
                "-k",
                test_filter,
                "-v",
                "--tb=short",
                "--no-header",
                "-p",
                "no:warnings",  # Unterdruecke Warning-Summary fuer saubere Ausgabe
            ]
        )

        self.output_line.emit(f"Kommando: {' '.join(cmd)}\n")
        self.output_line.emit("=" * 80)

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )

            passed_count = 0
            failed_count = 0
            total_count = 0

            # Parse Output Zeile fuer Zeile
            for line in process.stdout:
                if self.should_stop:
                    process.terminate()
                    self.output_line.emit("\n[STOPPED] Test-Ausfuehrung gestoppt")
                    break

                # Emit jede Zeile fuer Konsolen-Ausgabe
                self.output_line.emit(line.rstrip())

                line = line.strip()

                # Erkenne Test-Start (pytest -v Format)
                if "::" in line and (" PASSED" in line or " FAILED" in line):
                    # Beispiel: test_kai_worker.py::TestGoalPlanner::test_high_confidence_direct_execution PASSED
                    parts = line.split("::")
                    if len(parts) >= 3:
                        test_class = parts[1]
                        test_name_with_status = parts[2]
                        test_name = test_name_with_status.split()[0]
                        full_test_name = f"{test_class}::{test_name}"

                        total_count += 1
                        self.test_started.emit(full_test_name)

                        if "PASSED" in line:
                            passed_count += 1
                            self.test_passed.emit(full_test_name)
                        elif "FAILED" in line:
                            failed_count += 1
                            # Fehlermeldung kommt in naechsten Zeilen - hier erstmal kurz
                            self.test_failed.emit(
                                full_test_name, "Siehe Test-Output fuer Details"
                            )

                        # Update Progress
                        percentage = (
                            total_count / max(1, total_count)
                        ) * 100  # Wird spaeter korrigiert
                        self.progress_updated.emit(total_count, total_count, percentage)

            return_code = process.wait()

            # Final Update
            self.output_line.emit("=" * 80)
            self.output_line.emit(f"Tests abgeschlossen. Return Code: {return_code}")
            self.output_line.emit(
                f"Bestanden: {passed_count}, Fehlgeschlagen: {failed_count}, Gesamt: {total_count}"
            )
            self.finished_all.emit(passed_count, failed_count, total_count)

        except FileNotFoundError as e:
            error_msg = f"Fehler: pytest nicht gefunden. Bitte installieren Sie pytest: pip install pytest\n\nDetails: {e}"
            self.output_line.emit(f"\n[ERROR] {error_msg}")
            self.test_failed.emit("System", error_msg)
            self.finished_all.emit(0, 1, 1)
        except Exception as e:
            # Bei Fehler: Alle als fehlgeschlagen markieren
            error_msg = f"Fehler beim Ausfuehren der Tests:\n{e}\n\nTraceback:\n{traceback.format_exc()}"
            self.output_line.emit(f"\n[ERROR] {error_msg}")
            self.test_failed.emit("System", error_msg)
            self.finished_all.emit(0, 1, 1)

    def stop(self):
        """Stoppt die Test-Ausfuehrung"""
        self.should_stop = True
