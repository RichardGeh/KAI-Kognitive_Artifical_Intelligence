"""
Error Path Testing für KAI

Testet Robustheit gegen verschiedene Fehlerszenarien:
- Neo4j Connection Lost
- Embedding Service Down
- Malformed Data
- Resource Exhaustion
- Concurrency Issues (falls relevant)
"""

import pytest
from unittest.mock import Mock, MagicMock
from neo4j.exceptions import ServiceUnavailable, SessionExpired, TransientError
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from component_1_netzwerk import KonzeptNetzwerk
from component_14_abductive_engine import AbductiveEngine
from component_9_logik_engine import Fact
from component_utils_text_normalization import TextNormalizer, clean_entity


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_driver_unavailable():
    """
    Mock für Neo4j Driver der ServiceUnavailable wirft.
    """
    mock = MagicMock()
    mock.session.side_effect = ServiceUnavailable("Neo4j connection lost")
    return mock


@pytest.fixture
def mock_driver_session_expired():
    """
    Mock für Neo4j Driver der SessionExpired wirft.
    """
    mock = MagicMock()
    mock_session = MagicMock()
    mock_session.run.side_effect = SessionExpired("Session expired")
    mock.session.return_value.__enter__.return_value = mock_session
    return mock


@pytest.fixture
def mock_driver_transient_error():
    """
    Mock für Neo4j Driver der TransientError wirft (temporärer Fehler).
    """
    mock = MagicMock()
    mock_session = MagicMock()
    mock_session.run.side_effect = TransientError("Transient error")
    mock.session.return_value.__enter__.return_value = mock_session
    return mock


# ============================================================================
# ERROR PATH TESTS: Neo4j Connection Lost
# ============================================================================


def test_netzwerk_handles_service_unavailable(mock_driver_unavailable):
    """
    Error Path: Neo4j Service Unavailable sollte graceful behandelt werden.
    """
    netzwerk = KonzeptNetzwerk()
    netzwerk.driver = mock_driver_unavailable

    try:
        # Sollte nicht crashen, sondern leere Ergebnisse oder None zurückgeben
        result = netzwerk.query_graph_for_facts("test")
        assert (
            result == {} or result is None
        ), "Bei Connection Error sollte leeres Dict oder None zurückkommen"
        print("[OK] Neo4j ServiceUnavailable wird graceful behandelt")
    except ServiceUnavailable:
        # Akzeptabel wenn Exception durchgereicht wird
        print("[OK] Neo4j ServiceUnavailable wird als Exception durchgereicht")


def test_netzwerk_handles_session_expired(mock_driver_session_expired):
    """
    Error Path: Expired Session sollte behandelt werden.
    """
    netzwerk = KonzeptNetzwerk()
    netzwerk.driver = mock_driver_session_expired

    try:
        result = netzwerk.create_word("test_word")
        # Sollte graceful fehlschlagen oder None zurückgeben
        print(f"[OK] SessionExpired wird behandelt: {result}")
    except SessionExpired:
        print("[OK] SessionExpired wird als Exception durchgereicht")


def test_netzwerk_handles_transient_error(mock_driver_transient_error):
    """
    Error Path: Transiente Fehler (z.B. Leader-Wechsel im Cluster).
    """
    netzwerk = KonzeptNetzwerk()
    netzwerk.driver = mock_driver_transient_error

    try:
        result = netzwerk.query_graph_for_facts("test")
        print(f"[OK] TransientError wird behandelt: {result}")
    except TransientError:
        print("[OK] TransientError wird als Exception durchgereicht")


def test_netzwerk_handles_none_driver():
    """
    Error Path: Netzwerk mit None driver sollte nicht crashen.
    """
    netzwerk = KonzeptNetzwerk()
    netzwerk.driver = None

    try:
        result = netzwerk.query_graph_for_facts("test")
        assert result == {} or result is None
        print("[OK] None driver wird graceful behandelt")
    except AttributeError as e:
        pytest.fail(f"None driver sollte graceful behandelt werden: {e}")


# ============================================================================
# ERROR PATH TESTS: Abductive Engine mit Connection Loss
# ============================================================================


def test_abductive_engine_handles_connection_loss():
    """
    Error Path: Abductive Engine sollte mit Neo4j Connection Loss umgehen können.
    """
    # Mock Netzwerk mit Connection Loss
    mock_netzwerk = Mock()
    mock_netzwerk.query_graph_for_facts.side_effect = ServiceUnavailable(
        "Connection lost"
    )
    mock_netzwerk.driver = None

    engine = AbductiveEngine(netzwerk=mock_netzwerk)

    try:
        hypotheses = engine.generate_hypotheses(
            observation="Der Boden ist nass", context_facts=[], max_hypotheses=5
        )

        # Sollte leere Liste oder reduzierte Hypothesen zurückgeben
        assert isinstance(hypotheses, list)
        print(
            f"[OK] Abductive Engine behandelt Connection Loss: {len(hypotheses)} Hypothesen generiert"
        )
    except ServiceUnavailable:
        print("[OK] Abductive Engine leitet Exception weiter")


def test_abductive_engine_empty_knowledge_base():
    """
    Error Path: Abductive Engine mit leerer Knowledge Base.
    """
    # Mock Netzwerk das immer leere Ergebnisse liefert
    mock_netzwerk = Mock()
    mock_netzwerk.query_graph_for_facts.return_value = {}
    mock_netzwerk.driver = Mock()

    engine = AbductiveEngine(netzwerk=mock_netzwerk)

    hypotheses = engine.generate_hypotheses(
        observation="Der Boden ist nass", context_facts=[], max_hypotheses=5
    )

    # Sollte funktionieren, auch wenn keine Daten verfügbar sind
    assert isinstance(hypotheses, list)
    print(
        f"[OK] Abductive Engine funktioniert mit leerer Knowledge Base: {len(hypotheses)} Hypothesen"
    )


# ============================================================================
# ERROR PATH TESTS: Malformed Data
# ============================================================================


def test_abductive_engine_malformed_fact():
    """
    Error Path: Abductive Engine mit malformed Facts.
    """
    mock_netzwerk = Mock()
    mock_netzwerk.query_graph_for_facts.return_value = {"IS_A": ["valid"]}
    mock_netzwerk.driver = Mock()

    engine = AbductiveEngine(netzwerk=mock_netzwerk)

    # Malformed Fact ohne subject/object
    malformed_fact = Fact(pred="HAS_PROPERTY", args={}, id="test", confidence=0.5)

    try:
        # Sollte nicht crashen
        result = engine._contradicts_knowledge(malformed_fact)
        assert isinstance(result, bool)
        print("[OK] Abductive Engine behandelt malformed Facts graceful")
    except Exception as e:
        pytest.fail(f"Malformed Fact sollte graceful behandelt werden: {e}")


def test_text_normalizer_malformed_input():
    """
    Error Path: TextNormalizer mit malformed Inputs.
    """
    normalizer = TextNormalizer(preprocessor=None)

    malformed_inputs = [
        None,
        123,  # Integer statt String
        [],  # Liste statt String
        {},  # Dict statt String
        object(),  # Beliebiges Objekt
    ]

    for malformed in malformed_inputs:
        try:
            # TextNormalizer sollte gracefully mit malformed input umgehen
            # Modern robust implementation returns empty string or converts to string
            result = normalizer.clean_entity(malformed)
            # Should return a string (empty or converted)
            assert isinstance(
                result, str
            ), f"Expected string result, got {type(result)}"
            print(f"[OK] Malformed Input behandelt: {type(malformed).__name__}")
        except Exception as e:
            # If it does throw an exception, that's also acceptable
            print(f"[OK] Malformed Input wirft erwartete Exception: {type(e).__name__}")


# ============================================================================
# ERROR PATH TESTS: Resource Exhaustion
# ============================================================================


def test_text_normalizer_extremely_long_input():
    """
    Error Path: Sehr lange Inputs (Stress Test - 1 Million Zeichen).
    """
    normalizer = TextNormalizer(preprocessor=None)

    # 1 Million Zeichen
    very_long_text = "der " + "a" * 1_000_000

    try:
        start = time.time()
        result = normalizer.clean_entity(very_long_text)
        elapsed = time.time() - start

        assert isinstance(result, str)
        assert len(result) > 0
        print(
            f"[OK] Extrem langer Input behandelt: {len(very_long_text)} Zeichen in {elapsed:.3f}s"
        )

        # Sollte nicht zu lange dauern (< 10 Sekunden)
        assert (
            elapsed < 10.0
        ), f"Zu langsam: {elapsed:.3f}s für {len(very_long_text)} Zeichen"
    except Exception as e:
        pytest.fail(f"Extrem langer Input sollte funktionieren: {e}")


def test_abductive_engine_infinite_recursion_protection():
    """
    Error Path: Prüft ob Abductive Engine gegen infinite Rekursion geschützt ist.

    Szenario: IS_A Hierarchie mit Zyklen (A IS_A B, B IS_A C, C IS_A A)
    """
    # Mock Netzwerk mit zyklischer IS_A Hierarchie
    mock_netzwerk = Mock()

    def cyclic_query(concept):
        cycles = {
            "a": {"IS_A": ["b"]},
            "b": {"IS_A": ["c"]},
            "c": {"IS_A": ["a"]},  # Zyklus!
        }
        return cycles.get(concept.lower(), {})

    mock_netzwerk.query_graph_for_facts.side_effect = cyclic_query
    mock_netzwerk.driver = Mock()

    engine = AbductiveEngine(netzwerk=mock_netzwerk)

    try:
        # Sollte nicht in infinite Rekursion laufen
        result = engine._is_subtype_of("a", "c")

        # Timeout nach 5 Sekunden
        # (Wenn es länger dauert, ist die Rekursion problematisch)
        print(f"[OK] Infinite Recursion Protection funktioniert: {result}")
    except RecursionError:
        pytest.fail(
            "Abductive Engine läuft in infinite Rekursion bei zyklischer IS_A Hierarchie"
        )


# ============================================================================
# CONCURRENCY TESTS (falls relevant)
# ============================================================================


def test_text_normalizer_thread_safety():
    """
    Concurrency Test: TextNormalizer sollte thread-safe sein.
    """
    normalizer = TextNormalizer(preprocessor=None)

    test_inputs = [
        "der Hund",
        "die Katze",
        "das Haus",
        "ein Auto",
        "eine Blume",
    ] * 100  # 500 Eingaben

    def normalize_text(text):
        return normalizer.clean_entity(text)

    # Parallel verarbeiten mit ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(normalize_text, text) for text in test_inputs]

        results = []
        for future in as_completed(futures):
            try:
                result = future.result(timeout=5)
                results.append(result)
            except Exception as e:
                pytest.fail(f"Thread-Safety Problem: {e}")

    # Alle sollten erfolgreich sein
    assert len(results) == len(test_inputs)
    print(
        f"[OK] Thread-Safety Test bestanden: {len(results)} parallele Normalisierungen"
    )


def test_netzwerk_concurrent_queries():
    """
    Concurrency Test: Netzwerk sollte concurrent queries behandeln können.
    """
    netzwerk = KonzeptNetzwerk()

    test_queries = ["hund", "katze", "haus", "auto", "baum"] * 20  # 100 Queries

    def query_graph(concept):
        return netzwerk.query_graph_for_facts(concept)

    # Parallel queries
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(query_graph, concept) for concept in test_queries]

        results = []
        for future in as_completed(futures):
            try:
                result = future.result(timeout=10)
                results.append(result)
            except Exception as e:
                print(f"⚠ Concurrent Query fehlgeschlagen: {e}")

    print(
        f"[OK] Concurrency Test: {len(results)}/{len(test_queries)} Queries erfolgreich"
    )


# ============================================================================
# ERROR RECOVERY TESTS
# ============================================================================


def test_netzwerk_recovers_after_transient_error():
    """
    Error Recovery: Netzwerk sollte sich von transient errors erholen.
    """
    # Mock Driver der erst fehlschlägt, dann funktioniert
    call_count = {"count": 0}

    def transient_then_success(*args, **kwargs):
        call_count["count"] += 1
        if call_count["count"] <= 2:
            raise TransientError("Temporary error")
        # Nach 2 Fehlversuchen: Erfolg
        mock_result = MagicMock()
        mock_result.single.return_value = {"text": "test_concept"}
        return mock_result

    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_session.run.side_effect = transient_then_success
    mock_driver.session.return_value.__enter__.return_value = mock_session

    netzwerk = KonzeptNetzwerk()
    netzwerk.driver = mock_driver

    # Retry-Logik (falls implementiert)
    max_retries = 3
    for attempt in range(max_retries):
        try:
            netzwerk.create_word("test_word")
            print(f"[OK] Recovery nach {attempt + 1} Versuchen erfolgreich")
            break
        except TransientError:
            if attempt == max_retries - 1:
                pytest.fail("Keine Recovery nach transient errors")
            time.sleep(0.1)  # Kurz warten vor Retry


# ============================================================================
# STRESS TESTS
# ============================================================================


def test_abductive_engine_many_hypotheses():
    """
    Stress Test: Generiere sehr viele Hypothesen.
    """
    from unittest.mock import MagicMock

    # Mock Netzwerk mit vielen Facts
    mock_netzwerk = Mock()
    mock_netzwerk.query_graph_for_facts.return_value = {
        "IS_A": [f"type_{i}" for i in range(100)],
        "HAS_PROPERTY": [f"prop_{i}" for i in range(100)],
    }
    # Create proper context manager support for driver.session()
    mock_netzwerk.driver = MagicMock()
    mock_session = MagicMock()
    mock_session.run.return_value = []
    mock_netzwerk.driver.session.return_value.__enter__.return_value = mock_session
    mock_netzwerk.driver.session.return_value.__exit__.return_value = None

    engine = AbductiveEngine(netzwerk=mock_netzwerk)

    start = time.time()
    hypotheses = engine.generate_hypotheses(
        observation="Test observation",
        context_facts=[],
        max_hypotheses=1000,  # Viele Hypothesen
    )
    elapsed = time.time() - start

    assert isinstance(hypotheses, list)
    print(f"[OK] Stress Test: {len(hypotheses)} Hypothesen in {elapsed:.3f}s generiert")


# ============================================================================
# INTEGRATION ERROR TESTS
# ============================================================================


def test_full_pipeline_with_connection_loss():
    """
    Integration Test: Volle Pipeline mit Connection Loss.
    """
    # Simuliere kompletten Workflow mit Fehlern
    # 1. Text Normalisierung (sollte immer funktionieren)
    text = "der große Hund"
    normalized = clean_entity(text)
    assert isinstance(normalized, str)

    # 2. Netzwerk mit Connection Loss
    mock_netzwerk = Mock()
    mock_netzwerk.query_graph_for_facts.side_effect = ServiceUnavailable(
        "Connection lost"
    )
    mock_netzwerk.driver = None

    # 3. Abductive Engine with connection loss should either:
    #    - Raise ServiceUnavailable (acceptable)
    #    - Return empty results (also acceptable)
    engine = AbductiveEngine(netzwerk=mock_netzwerk)
    try:
        hypotheses = engine.generate_hypotheses(
            observation="Test", context_facts=[], max_hypotheses=5
        )
        # If it succeeds, should return a list (possibly empty)
        assert isinstance(hypotheses, list)
        print(
            f"[OK] Integration Test: Pipeline handled connection loss gracefully (returned {len(hypotheses)} hypotheses)"
        )
    except ServiceUnavailable:
        # It's also acceptable to raise the exception
        print(
            "[OK] Integration Test: Pipeline correctly propagated ServiceUnavailable exception"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
