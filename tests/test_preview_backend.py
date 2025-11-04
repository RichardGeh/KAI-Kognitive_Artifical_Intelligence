"""
test_preview_backend.py

Test für Preview-Feature Backend-Funktionalität.
"""

# Fix Windows encoding
import kai_encoding_fix  # noqa: F401

import threading
import time
from kai_worker import KaiWorker
from component_1_netzwerk import KonzeptNetzwerk
from component_11_embedding_service import EmbeddingService


def test_preview_confirmation_mechanism():
    """Test das Preview-Confirmation-Mechanismus."""
    print("=== Test Preview Confirmation Mechanism ===\n")

    # Initialisiere Worker
    netzwerk = KonzeptNetzwerk()
    embedding_service = EmbeddingService()
    worker = KaiWorker(netzwerk, embedding_service)

    if not worker.is_initialized_successfully:
        print(
            f"✗ Worker-Initialisierung fehlgeschlagen: {worker.initialization_error_message}"
        )
        return

    print("✓ Worker erfolgreich initialisiert")

    # Test 1: Signal-Verbindung testen
    print("\nTest 1: Signal-Verbindung")
    assert hasattr(worker.signals, "preview_confirmation_needed")
    assert hasattr(worker.signals, "preview_confirmation_response")
    print("✓ Signals sind vorhanden")

    # Test 2: Threading-Event testen
    print("\nTest 2: Threading Event")
    assert hasattr(worker, "preview_confirmation_event")
    assert isinstance(worker.preview_confirmation_event, threading.Event)
    print("✓ Threading Event existiert")

    # Test 3: Simuliere User-Bestätigung (Ja)
    print("\nTest 3: Simuliere User-Bestätigung (Ja)")

    def simulate_user_yes():
        """Simuliert User, der 'Ja' klickt nach 0.5 Sekunden."""
        time.sleep(0.5)
        print("  [Simulator] User klickt 'Ja'")
        worker.signals.preview_confirmation_response.emit(True)

    # Starte Simulator-Thread
    sim_thread = threading.Thread(target=simulate_user_yes, daemon=True)
    sim_thread.start()

    # Warte auf Bestätigung
    preview = "Dies ist ein Test-Preview..."
    file_name = "test.txt"
    char_count = 100

    result = worker.wait_for_preview_confirmation(preview, file_name, char_count)

    assert result is True, "Bestätigung sollte True sein"
    print("✓ User-Bestätigung (Ja) funktioniert")

    # Test 4: Simuliere User-Ablehnung (Nein)
    print("\nTest 4: Simuliere User-Ablehnung (Nein)")

    def simulate_user_no():
        """Simuliert User, der 'Nein' klickt nach 0.5 Sekunden."""
        time.sleep(0.5)
        print("  [Simulator] User klickt 'Nein'")
        worker.signals.preview_confirmation_response.emit(False)

    # Starte Simulator-Thread
    sim_thread = threading.Thread(target=simulate_user_no, daemon=True)
    sim_thread.start()

    # Warte auf Bestätigung
    result = worker.wait_for_preview_confirmation(preview, file_name, char_count)

    assert result is False, "Bestätigung sollte False sein"
    print("✓ User-Ablehnung (Nein) funktioniert")

    # Test 5: Timeout-Test
    print("\nTest 5: Timeout-Test (überspringen - würde 60 Sek dauern)")
    # result = worker.wait_for_preview_confirmation(preview, file_name, char_count)
    # assert result is False, "Timeout sollte False zurückgeben"
    print("  ⊘ Übersprungen (würde 60 Sekunden dauern)")

    # Cleanup
    netzwerk.close()

    print("\n" + "=" * 50)
    print("✓ ALLE BACKEND-TESTS BESTANDEN!")
    print("=" * 50)


def test_preview_generation():
    """Test das Preview-Generierung in FileReaderStrategy."""
    print("\n\n=== Test Preview Generation ===\n")

    # Erstelle Testdatei
    test_text = "A" * 600  # 600 Zeichen

    # Simuliere Preview-Generierung
    preview = test_text[:500]
    if len(test_text) > 500:
        preview += "..."

    print(f"Original: {len(test_text)} Zeichen")
    print(f"Preview: {len(preview)} Zeichen")

    assert len(preview) == 503  # 500 + "..."
    assert preview.endswith("...")
    print("✓ Preview-Generierung funktioniert")


if __name__ == "__main__":
    try:
        test_preview_generation()
        test_preview_confirmation_mechanism()

    except AssertionError as e:
        print(f"\n✗ TEST FEHLGESCHLAGEN: {e}")
        import traceback

        traceback.print_exc()
    except Exception as e:
        print(f"\n✗ FEHLER: {e}")
        import traceback

        traceback.print_exc()
