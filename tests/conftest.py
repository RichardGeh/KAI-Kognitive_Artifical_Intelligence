# conftest.py
"""
Gemeinsame Pytest-Fixtures für alle KAI Tests.
Diese Fixtures werden automatisch von pytest geladen und sind in allen Test-Dateien verfügbar.
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import pytest
from unittest.mock import MagicMock
from component_1_netzwerk import KonzeptNetzwerk
from component_11_embedding_service import EmbeddingService
from kai_worker import KaiWorker

# Semantische Embeddings haben 384 Dimensionen (vorher 8D Featurizer)
TEST_VECTOR_DIM = 384
logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def netzwerk_session():
    """Stellt eine DB-Verbindung für die gesamte Test-Session bereit."""
    nw = KonzeptNetzwerk()
    assert nw.driver is not None, "DB-Verbindung fehlgeschlagen"

    # Verifiziere Constraints
    with nw.driver.session(database="neo4j") as session:
        result = session.run("SHOW CONSTRAINTS")
        constraints = [record["name"] for record in result]
        logger.info(f"Vorhandene Constraints: {constraints}")

    yield nw
    nw.close()


@pytest.fixture(scope="session")
def embedding_service_session():
    """Stellt einen echten Embedding-Service bereit."""
    service = EmbeddingService()
    assert service.is_available(), "Embedding-Service nicht verfügbar"
    return service


@pytest.fixture(scope="function")
def clean_test_concepts(netzwerk_session):
    """
    Stellt sicher, dass Test-Konzepte vor und nach jedem Test gelöscht werden.
    Verwendet ein Präfix für Testdaten.
    """
    test_prefix = "test_"

    def cleanup():
        with netzwerk_session.driver.session(database="neo4j") as session:
            # Lösche alle Knoten, die mit test_ beginnen
            session.run(
                """
                MATCH (n)
                WHERE n.name STARTS WITH $prefix OR n.lemma STARTS WITH $prefix
                DETACH DELETE n
            """,
                prefix=test_prefix,
            )
            logger.info(
                f"[CLEANUP] Alle Test-Konzepte mit Praefix '{test_prefix}' geloescht."
            )

    cleanup()  # Vor dem Test
    yield test_prefix
    cleanup()  # Nach dem Test


@pytest.fixture(scope="function")
def kai_worker_with_mocks(netzwerk_session):
    """Erstellt einen KaiWorker mit Mock-Signals für Tests."""
    embedding_service = EmbeddingService()
    worker = KaiWorker(netzwerk_session, embedding_service)
    worker.signals = MagicMock()
    assert (
        worker.is_initialized_successfully
    ), "KAI Worker Initialisierung fehlgeschlagen"
    return worker
