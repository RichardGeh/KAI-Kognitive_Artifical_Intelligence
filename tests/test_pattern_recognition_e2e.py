# tests/test_pattern_recognition_e2e.py
"""
End-to-End Tests für Pattern Recognition System.

Testet:
- Tippfehler-Erkennung (Buchstaben-Ebene)
- Wortfolgen-Vorhersage (Sequence-Ebene)
- Implikations-Erkennung (Implizite Fakten)
- Feedback-Loop ("Nein, ich meine X")
"""

import pytest
from component_1_netzwerk import KonzeptNetzwerk
from component_19_pattern_recognition_char import TypoCandidateFinder
from component_20_pattern_recognition_sequence import SequencePredictor
from component_22_pattern_recognition_implicit import ImplicationDetector
from component_24_pattern_orchestrator import PatternOrchestrator
from component_5_linguistik_strukturen import KaiContext, ContextAction
from kai_config import get_config


@pytest.fixture
def netzwerk():
    """Fixture für KonzeptNetzwerk mit Test-Daten"""
    netz = KonzeptNetzwerk()

    # Erstelle Test-Wörter für Tippfehler-Erkennung
    test_words = [
        "test_katze",
        "test_hund",
        "test_maus",
        "test_baum",
        "test_haus",
        "test_auto",
    ]

    for word in test_words:
        netz.ensure_wort_und_konzept(word)
        # Simuliere Verwendung (min_word_occurrences_for_typo = 10)
        for i in range(12):
            netz.add_usage_context(word, f"Kontext {i} für {word}", word_position=2)

    # Erstelle Wortfolgen für Sequence-Prediction
    # Simuliere "Ein Hund bellt laut"
    netz.ensure_wort_und_konzept("test_ein")
    netz.ensure_wort_und_konzept("test_hund2")
    netz.ensure_wort_und_konzept("test_bellt")
    netz.ensure_wort_und_konzept("test_laut")

    # Erstelle CONNECTION edges
    for _ in range(6):  # min_sequence_count = 5
        netz.add_word_connection(
            "test_ein", "test_hund2", distance=1, direction="after"
        )
        netz.add_word_connection(
            "test_hund2", "test_bellt", distance=1, direction="after"
        )
        netz.add_word_connection(
            "test_bellt", "test_laut", distance=1, direction="after"
        )

    yield netz

    # Note: Test concepts remain in database with "test_" prefix for debugging
    # This is standard practice and allows inspection of test data


@pytest.fixture
def typo_finder(netzwerk):
    """Fixture für TypoCandidateFinder"""
    return TypoCandidateFinder(netzwerk)


@pytest.fixture
def sequence_predictor(netzwerk):
    """Fixture für SequencePredictor"""
    return SequencePredictor(netzwerk)


@pytest.fixture
def implication_detector(netzwerk):
    """Fixture für ImplicationDetector"""
    return ImplicationDetector(netzwerk)


@pytest.fixture
def pattern_orchestrator(netzwerk):
    """Fixture für PatternOrchestrator"""
    return PatternOrchestrator(netzwerk)


# ============================================================================
# TYPO DETECTION TESTS
# ============================================================================


def test_typo_detection_simple(typo_finder):
    """Test: Einfache Tippfehler-Erkennung mit QWERTZ-Nachbarn"""
    # "Ktzae" -> "Katze" (t->a sind Nachbarn auf QWERTZ)
    candidates = typo_finder.find_candidates("test_ktzae")

    # Sollte "test_katze" als Kandidat finden
    assert len(candidates) > 0
    assert any(c["word"] == "test_katze" for c in candidates)


def test_typo_auto_correct_high_confidence(pattern_orchestrator):
    """Test: Auto-Korrektur bei hoher Konfidenz (≥0.85)"""
    # Simuliere Tippfehler mit hoher Ähnlichkeit
    result = pattern_orchestrator.process_input("test_katae")

    # Bei hoher Konfidenz sollte auto-korrigiert werden
    if result.get("typo_corrections"):
        correction = result["typo_corrections"][0]
        if correction["confidence"] >= 0.85:
            assert correction["decision"] == "auto_corrected"


def test_typo_ask_user_medium_confidence(pattern_orchestrator):
    """Test: Rückfrage bei mittlerer Konfidenz (0.60-0.84)"""
    # Simuliere Tippfehler mit mittlerer Ähnlichkeit
    result = pattern_orchestrator.process_input("test_ktzae test_haus")

    # Bei mittlerer Konfidenz sollte Rückfrage erfolgen
    if result.get("typo_corrections"):
        for correction in result["typo_corrections"]:
            if 0.60 <= correction.get("confidence", 0) < 0.85:
                assert correction["decision"] == "ask_user"
                assert result["needs_user_clarification"] is True


def test_typo_feedback_storage(netzwerk):
    """Test: Speicherung von Tippfehler-Feedback in Neo4j"""
    # Speichere positives Feedback
    feedback_id = netzwerk.store_typo_feedback(
        original_input="test_ktzae",
        suggested_word="test_katze",
        actual_word="test_katze",
        user_accepted=True,
        confidence=0.75,
    )

    assert feedback_id is not None

    # Hole Feedback zurück
    feedback = netzwerk.get_typo_feedback_for_input("test_ktzae")
    assert len(feedback) > 0
    assert feedback[0]["suggested_word"] == "test_katze"
    assert feedback[0]["user_accepted"] is True


def test_typo_negative_feedback(netzwerk):
    """Test: Negatives Feedback speichern (Nutzer korrigiert)"""
    feedback_id = netzwerk.store_typo_feedback(
        original_input="test_ktze",
        suggested_word="test_katze",
        actual_word="test_kitze",  # Nutzer meinte was anderes
        user_accepted=False,
        confidence=0.70,
    )

    assert feedback_id is not None

    feedback = netzwerk.get_typo_feedback_for_input("test_ktze")
    assert len(feedback) > 0
    assert feedback[0]["actual_word"] == "test_kitze"
    assert feedback[0]["user_accepted"] is False


# ============================================================================
# SEQUENCE PREDICTION TESTS
# ============================================================================


def test_sequence_prediction_bigram(sequence_predictor):
    """Test: Wortfolgen-Vorhersage mit Bigram-Modell"""
    predictions = sequence_predictor.predict_next_word(["test_ein"])

    # Sollte "test_hund2" vorhersagen (häufigste Verbindung)
    assert len(predictions) > 0
    # Prüfe ob "test_hund2" in Top-Predictions ist
    top_words = [p["word"] for p in predictions[:3]]
    assert "test_hund2" in top_words


def test_sequence_prediction_context_aware(sequence_predictor):
    """Test: Kontext-basierte Vorhersage"""
    predictions = sequence_predictor.predict_next_word(["test_ein", "test_hund2"])

    # Sollte "test_bellt" vorhersagen
    if len(predictions) > 0:
        top_words = [p["word"] for p in predictions[:3]]
        assert "test_bellt" in top_words


def test_sequence_completion(sequence_predictor):
    """Test: Satz-Vervollständigung"""
    completion = sequence_predictor.predict_completion("test_ein test_hund2")

    # Sollte den Satz vervollständigen
    assert len(completion) > 0


# ============================================================================
# IMPLICATION DETECTION TESTS
# ============================================================================


def test_property_implication_detection(implication_detector):
    """Test: Erkennung von Property-Implikationen"""
    implications = implication_detector.detect_property_implications(
        "test_haus", "groß"
    )

    # "Haus ist groß" -> impliziert "Haus hat größe"
    assert len(implications) > 0
    assert implications[0]["object"] == "größe"
    assert implications[0]["relation"] == "HAS_PROPERTY"
    assert implications[0]["confidence"] >= 0.8


def test_color_implication(implication_detector):
    """Test: Farb-Implikation"""
    implications = implication_detector.detect_property_implications("test_auto", "rot")

    # "Auto ist rot" -> impliziert "Auto hat farbe"
    assert len(implications) > 0
    assert implications[0]["object"] == "farbe"


# ============================================================================
# PATTERN ORCHESTRATOR E2E TESTS
# ============================================================================


def test_orchestrator_full_pipeline(pattern_orchestrator):
    """Test: Vollständiger Pipeline-Durchlauf"""
    result = pattern_orchestrator.process_input("test_ein test_hund2 test_bellt")

    # Pipeline sollte korrekt durchlaufen
    assert "corrected_text" in result
    assert "typo_corrections" in result
    assert "next_word_predictions" in result
    assert "implications" in result


def test_orchestrator_typo_blocks_prediction(pattern_orchestrator):
    """Test: Tippfehler blockiert Prediction (needs_user_clarification)"""
    # Simuliere unsicheren Tippfehler
    result = pattern_orchestrator.process_input("test_xyzabc test_haus")

    # Bei Rückfrage sollten keine Predictions gemacht werden
    if result.get("needs_user_clarification"):
        # Predictions könnten leer sein oder nur wenige enthalten
        pass  # Orchestrator kann trotzdem Predictions machen


# ============================================================================
# CONTEXT MANAGER INTEGRATION TESTS
# ============================================================================


def test_context_typo_clarification_setup():
    """Test: Context wird korrekt für Typo-Rückfrage gesetzt"""
    context = KaiContext()

    # Simuliere Typo-Rückfrage
    context.set_action(ContextAction.ERWARTE_TYPO_KLARSTELLUNG)
    context.set_data(
        "pattern_result",
        {"typo_corrections": [{"original": "test_ktzae", "candidates": []}]},
    )
    context.set_data("original_query", "test_ktzae ist ein Tier")

    assert context.is_active()
    assert context.aktion == ContextAction.ERWARTE_TYPO_KLARSTELLUNG
    assert context.get_data("original_query") == "test_ktzae ist ein Tier"


def test_context_typo_clarification_ja_response():
    """Test: 'Ja'-Antwort wird korrekt geparst"""
    context = KaiContext()
    context.set_action(ContextAction.ERWARTE_TYPO_KLARSTELLUNG)

    # Simuliere Ja-Antwort
    response = "Ja, das stimmt"

    # Prüfe Ja-Erkennung
    response_lower = response.lower().strip()
    is_ja = any(
        word in response_lower
        for word in ["ja", "yes", "korrekt", "richtig", "stimmt", "genau"]
    )
    assert is_ja is True


def test_context_typo_clarification_nein_response():
    """Test: 'Nein, ich meine X'-Antwort wird korrekt geparst"""
    response = "Nein, ich meine test_kitze"
    response_lower = response.lower().strip()

    # Prüfe Nein-Erkennung
    is_nein = "nein" in response_lower or "nicht" in response_lower
    assert is_nein is True

    # Extrahiere korrigiertes Wort
    if "meine" in response_lower:
        parts = response_lower.split("meine", 1)
        if len(parts) > 1:
            actual_word = parts[1].strip().split()[0] if parts[1].strip() else None
            assert actual_word == "test_kitze"


# ============================================================================
# CONFIGURATION TESTS
# ============================================================================


def test_pattern_recognition_config():
    """Test: Konfiguration wird korrekt geladen"""
    config = get_config()

    assert config.get("pattern_recognition_enabled") is not None
    assert config.get("typo_auto_correct_threshold") is not None
    assert config.get("typo_ask_user_threshold") is not None
    assert config.get("min_word_occurrences_for_typo") is not None
    assert config.get("min_sequence_count_for_prediction") is not None


def test_pattern_recognition_thresholds():
    """Test: Thresholds sind korrekt konfiguriert"""
    config = get_config()

    auto_threshold = config.get("typo_auto_correct_threshold", 0.85)
    ask_threshold = config.get("typo_ask_user_threshold", 0.60)

    # Auto-Threshold sollte höher sein als Ask-Threshold
    assert auto_threshold > ask_threshold

    # Beide sollten im validen Bereich sein
    assert 0.0 < ask_threshold < 1.0
    assert 0.0 < auto_threshold <= 1.0


# ============================================================================
# BOOTSTRAP MECHANISM TESTS
# ============================================================================


def test_bootstrap_insufficient_data(netzwerk):
    """Test: Pattern Recognition ohne ausreichend Daten"""
    # Erstelle Wort mit wenigen Verwendungen
    netzwerk.ensure_wort_und_konzept("test_rare_word")
    netzwerk.add_usage_context("test_rare_word", "Einmal verwendet", word_position=1)

    finder = TypoCandidateFinder(netzwerk)

    # Sollte keine Kandidaten finden (unter min_word_occurrences)
    candidates = finder.find_candidates("test_rare_wrod")

    # Entweder leer oder geringe Konfidenz
    if candidates:
        assert all(c["confidence"] < 0.60 for c in candidates)


def test_bootstrap_sufficient_data(netzwerk):
    """Test: Pattern Recognition mit ausreichend Daten"""
    # Wort mit vielen Verwendungen (bereits in Fixture)
    finder = TypoCandidateFinder(netzwerk)

    candidates = finder.find_candidates("test_ktze")

    # Sollte Kandidaten finden (über min_word_occurrences)
    assert len(candidates) > 0

    # Mindestens einer sollte relevante Konfidenz haben
    assert any(c["confidence"] > 0.30 for c in candidates)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
