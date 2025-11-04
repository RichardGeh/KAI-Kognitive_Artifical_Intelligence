"""
KAI Test Suite - Multi-Turn Dialog System Tests
Extrahiert aus test_kai_worker.py fuer bessere Wartbarkeit.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from component_utils_text_normalization import clean_entity

logger = logging.getLogger(__name__)
TEST_VECTOR_DIM = 384


class TestMultiTurnDialogSystem:
    """Tests für die Multi-Turn-Dialog-Fähigkeiten (Context-Persistenz)."""

    def test_context_snapshot_creation(self, kai_worker_with_mocks):
        """Testet das Erstellen von Context-Snapshots."""
        worker = kai_worker_with_mocks

        # Setze Kontext
        worker.context.aktion = worker.context.aktion.ERWARTE_BEISPIELSATZ
        worker.context.thema = "test_thema"
        worker.context.add_entity("test_entity1")
        worker.context.add_entity("test_entity2")

        # Erstelle Snapshot
        snapshot_id = worker.context.save_snapshot()

        assert snapshot_id.startswith("ctx-"), "Snapshot-ID sollte mit 'ctx-' beginnen"
        assert worker.context.has_history(), "Context sollte History haben"
        assert len(worker.context.history) == 1, "Genau ein Snapshot sollte existieren"

        snapshot = worker.context.get_last_snapshot()
        assert snapshot is not None, "Letzter Snapshot sollte existieren"
        assert snapshot.thema == "test_thema", "Snapshot sollte Thema speichern"
        assert (
            "test_entity1" in snapshot.entities
        ), "Snapshot sollte Entitäten speichern"
        assert (
            "test_entity2" in snapshot.entities
        ), "Snapshot sollte alle Entitäten speichern"

        logger.info(f"[SUCCESS] Context-Snapshot erfolgreich erstellt: {snapshot_id}")

    def test_context_snapshot_restore(self, kai_worker_with_mocks):
        """Testet das Wiederherstellen von Context-Snapshots."""
        worker = kai_worker_with_mocks

        # Erstelle ersten Snapshot
        worker.context.aktion = worker.context.aktion.ERWARTE_BEISPIELSATZ
        worker.context.thema = "thema1"
        worker.context.add_entity("entity1")
        snapshot_id1 = worker.context.save_snapshot()

        # Ändere Kontext und erstelle zweiten Snapshot
        worker.context.aktion = worker.context.aktion.ERWARTE_BESTAETIGUNG
        worker.context.thema = "thema2"
        worker.context.add_entity("entity2")
        worker.context.save_snapshot()

        # Stelle ersten Snapshot wieder her
        restored = worker.context.restore_snapshot(snapshot_id1)

        assert restored, "Snapshot sollte wiederhergestellt werden können"
        assert worker.context.thema == "thema1", "Thema sollte wiederhergestellt sein"
        assert (
            "entity1" in worker.context.entities_in_session
        ), "Entitäten sollten wiederhergestellt sein"

        logger.info(
            f"[SUCCESS] Context-Snapshot erfolgreich wiederhergestellt: {snapshot_id1}"
        )

    def test_context_history_limit(self, kai_worker_with_mocks):
        """Testet, dass die Context-History auf max_history_size begrenzt ist."""
        worker = kai_worker_with_mocks
        max_history = worker.context.max_history_size

        # Erstelle mehr Snapshots als das Limit
        for i in range(max_history + 5):
            worker.context.aktion = worker.context.aktion.ERWARTE_BEISPIELSATZ
            worker.context.thema = f"thema_{i}"
            worker.context.save_snapshot()

        # Prüfe, dass History auf max_history_size begrenzt ist
        assert (
            len(worker.context.history) == max_history
        ), f"History sollte auf {max_history} Einträge begrenzt sein"

        # Prüfe, dass die ältesten Einträge entfernt wurden
        last_snapshot = worker.context.get_last_snapshot()
        assert (
            last_snapshot.thema == f"thema_{max_history + 4}"
        ), "Letzter Snapshot sollte der neueste sein"

        logger.info(
            f"[SUCCESS] Context-History korrekt auf {max_history} Einträge begrenzt"
        )

    def test_entity_tracking_across_queries(
        self, kai_worker_with_mocks, clean_test_concepts
    ):
        """Testet, dass Entitäten über mehrere Queries hinweg getrackt werden."""
        worker = kai_worker_with_mocks

        # Verwende normale deutsche Wörter (clean_test_concepts für Cleanup)
        test_entity1 = "katze"
        test_entity2 = "hund"
        test_tier = "tier"

        worker.netzwerk.ensure_wort_und_konzept(test_entity1)
        worker.netzwerk.ensure_wort_und_konzept(test_entity2)
        worker.netzwerk.ensure_wort_und_konzept(test_tier)
        # Beide Entities brauchen Fakten UND Bedeutungen, damit kein Kontext aktiviert wird
        worker.netzwerk.assert_relation(test_entity1, "IS_A", test_tier, "Test")
        worker.netzwerk.assert_relation(test_entity2, "IS_A", test_tier, "Test")
        worker.netzwerk.add_information_zu_wort(
            test_entity1, "bedeutung", f"Eine {test_entity1} ist ein {test_tier}."
        )
        worker.netzwerk.add_information_zu_wort(
            test_entity2, "bedeutung", f"Ein {test_entity2} ist ein {test_tier}."
        )

        # Query 1: Frage über Entität 1
        worker.process_query(f"Was ist eine {test_entity1}?")

        # Prüfe, dass Entität getrackt wurde
        normalized_entity1 = clean_entity(test_entity1)
        assert (
            normalized_entity1 in worker.context.entities_in_session
        ), f"'{normalized_entity1}' sollte in Session-Entitäten sein"

        # Query 2: Frage über Entität 2
        worker.process_query(f"Was ist ein {test_entity2}?")

        # Prüfe, dass beide Entitäten getrackt werden
        normalized_entity2 = clean_entity(test_entity2)
        assert (
            normalized_entity1 in worker.context.entities_in_session
        ), f"'{normalized_entity1}' sollte weiterhin getrackt werden"
        assert (
            normalized_entity2 in worker.context.entities_in_session
        ), f"'{normalized_entity2}' sollte nun auch getrackt werden"

        logger.info(
            f"[SUCCESS] Entities getrackt: {worker.context.entities_in_session}"
        )

    def test_context_summary_generation(self, kai_worker_with_mocks):
        """Testet die Generierung von Context-Zusammenfassungen für UI."""
        worker = kai_worker_with_mocks

        # Leerer Kontext -> Leere Summary
        summary = worker.context.get_context_summary()
        assert summary == "", "Leerer Kontext sollte leere Summary erzeugen"

        # Setze Kontext mit Thema
        worker.context.aktion = worker.context.aktion.ERWARTE_BEISPIELSATZ
        worker.context.thema = "test_thema"
        worker.context.add_entity("entity1")
        worker.context.add_entity("entity2")

        summary = worker.context.get_context_summary()

        assert "test_thema" in summary, "Summary sollte Thema enthalten"
        assert "Erwarte Definition" in summary, "Summary sollte Aktion beschreiben"
        assert (
            "entity1" in summary or "entity2" in summary
        ), "Summary sollte Entitäten enthalten"

        logger.info(f"[SUCCESS] Context-Summary: '{summary}'")

    def test_context_update_signal_emission(
        self, kai_worker_with_mocks, clean_test_concepts
    ):
        """Testet, dass Context-Update-Signale korrekt gesendet werden."""
        worker = kai_worker_with_mocks

        # Erstelle Test-Daten
        test_entity = f"{clean_test_concepts}unknown_word"

        # Query, die eine Wissenslücke auslöst
        worker.process_query(f"Was ist ein {test_entity}?")

        # Prüfe, dass context_update Signal gesendet wurde
        assert (
            worker.signals.context_update.emit.called
        ), "context_update Signal sollte gesendet worden sein"

        # Hole den gesendeten Context-String
        calls = worker.signals.context_update.emit.call_args_list
        assert (
            len(calls) > 0
        ), "Mindestens ein context_update Signal sollte gesendet worden sein"

        # Prüfe den Inhalt der letzten Context-Update
        last_context_update = calls[-1][0][0]  # Erstes Argument des letzten Aufrufs

        logger.info(
            f"[SUCCESS] Context-Update Signal gesendet: '{last_context_update}'"
        )

    def test_context_clear_preserves_history(self, kai_worker_with_mocks):
        """Testet, dass clear() einen Snapshot speichert, aber History behält."""
        worker = kai_worker_with_mocks

        # Setze Kontext
        worker.context.aktion = worker.context.aktion.ERWARTE_BEISPIELSATZ
        worker.context.thema = "test_thema"
        worker.context.add_entity("entity1")

        # Rufe clear() auf
        worker.context.clear()

        # Prüfe, dass Kontext gelöscht wurde
        assert not worker.context.is_active(), "Kontext sollte nicht mehr aktiv sein"
        assert worker.context.thema is None, "Thema sollte gelöscht sein"

        # Prüfe, dass History erhalten bleibt
        assert worker.context.has_history(), "History sollte erhalten bleiben"
        assert len(worker.context.history) == 1, "Ein Snapshot sollte in History sein"

        # Prüfe Snapshot-Inhalt
        snapshot = worker.context.get_last_snapshot()
        assert snapshot.thema == "test_thema", "Snapshot sollte alten Kontext enthalten"
        assert "entity1" in snapshot.entities, "Snapshot sollte Entitäten enthalten"

        logger.info(f"[SUCCESS] clear() speichert Snapshot und behält History")

    def test_context_clear_all(self, kai_worker_with_mocks):
        """Testet, dass clear_all() Kontext UND History komplett löscht."""
        worker = kai_worker_with_mocks

        # Setze Kontext und erstelle Snapshots
        worker.context.aktion = worker.context.aktion.ERWARTE_BEISPIELSATZ
        worker.context.thema = "test_thema"
        worker.context.add_entity("entity1")
        worker.context.save_snapshot()

        # Rufe clear_all() auf
        worker.context.clear_all()

        # Prüfe, dass alles gelöscht wurde
        assert not worker.context.is_active(), "Kontext sollte nicht aktiv sein"
        assert not worker.context.has_history(), "History sollte leer sein"
        assert (
            len(worker.context.entities_in_session) == 0
        ), "Entitäten sollten gelöscht sein"

        logger.info(f"[SUCCESS] clear_all() löscht Kontext und History komplett")


# ============================================================================
# PHASE 3: EPISODISCHES GEDÄCHTNIS TESTS
# ============================================================================
