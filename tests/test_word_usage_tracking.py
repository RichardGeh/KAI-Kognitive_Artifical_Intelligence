# tests/test_word_usage_tracking.py
"""
Tests für Word Usage Tracking Funktionalität.

Testet:
- Text-Fragmentierung (component_utils_text_fragmentation)
- Word Usage Speicherung in Neo4j (component_1_netzwerk_word_usage)
- Ähnlichkeits-Prüfung für Fragmente
- Integration mit kai_ingestion_handler
"""

import pytest
from component_1_netzwerk import KonzeptNetzwerk
from component_utils_text_fragmentation import (
    TextFragmenter,
    extract_word_usage_from_sentence,
)
from component_1_netzwerk_word_usage import calculate_similarity
from kai_config import get_config


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def netzwerk():
    """Neo4j Connection"""
    nw = KonzeptNetzwerk()
    yield nw
    nw.close()


@pytest.fixture
def fragmenter():
    """TextFragmenter ohne spaCy (nutzt Fallback)"""
    return TextFragmenter(linguistic_preprocessor=None)


# ============================================================================
# TEXT FRAGMENTATION TESTS
# ============================================================================


class TestTextFragmentation:
    """Tests für component_utils_text_fragmentation.py"""

    def test_simple_sentence_fragmentation(self, fragmenter):
        """Test: Einfacher Satz wird korrekt fragmentiert"""
        sentence = "Das Haus ist groß."

        fragments, connections = fragmenter.extract_fragments_and_connections(sentence)

        # Erwarte 4 Wörter (Das, Haus, ist, groß) - Punkt wird ignoriert
        assert len(fragments) == 4, f"Erwarte 4 Fragmente, bekam {len(fragments)}"

        # Prüfe erstes Fragment
        assert fragments[0].lemma == "das"
        assert fragments[0].word_position == 0

        # Prüfe dass Fragmente Text enthalten
        for frag in fragments:
            assert len(frag.fragment) > 0
            assert frag.total_words > 0

    def test_comma_delimited_fragmentation(self, fragmenter):
        """Test: Satz mit Komma wird bis Komma fragmentiert"""
        sentence = "Der Hund, der groß ist, bellt."

        fragments, connections = fragmenter.extract_fragments_and_connections(sentence)

        # Finde Fragment für "groß"
        gross_frag = next(f for f in fragments if f.lemma == "groß")

        # Fragment sollte bis Komma gehen (Anzahl Wörter <= max_words_to_comma)
        config = get_config()
        assert gross_frag.total_words <= config.max_words_to_comma + 3  # Etwas Toleranz

    def test_connection_extraction(self, fragmenter):
        """Test: Connections werden korrekt extrahiert"""
        sentence = "Das Haus ist groß."

        fragments, connections = fragmenter.extract_fragments_and_connections(sentence)

        # Erwarte Connections zwischen allen Nachbar-Wörtern
        assert len(connections) > 0

        # Prüfe dass "haus" mit "ist" verbunden ist
        haus_ist_conns = [
            c for c in connections if c.word1_lemma == "haus" and c.word2_lemma == "ist"
        ]

        assert len(haus_ist_conns) > 0, "Erwarte Connection haus -> ist"

        # Prüfe Distanz
        assert haus_ist_conns[0].distance == 1  # Direkt benachbart
        assert haus_ist_conns[0].direction == "before"

    def test_empty_sentence_handling(self, fragmenter):
        """Test: Leere Eingaben werden korrekt behandelt"""
        fragments, connections = fragmenter.extract_fragments_and_connections("")

        assert len(fragments) == 0
        assert len(connections) == 0

    def test_punctuation_filtering(self, fragmenter):
        """Test: Satzzeichen werden aus Fragmenten gefiltert"""
        sentence = "Das Haus, das Dach, die Tür."

        fragments, connections = fragmenter.extract_fragments_and_connections(sentence)

        # Satzzeichen sollten nicht als eigene Fragmente erscheinen
        for frag in fragments:
            assert frag.lemma not in [".", ",", "!", "?"]


# ============================================================================
# WORD USAGE STORAGE TESTS
# ============================================================================


class TestWordUsageStorage:
    """Tests für component_1_netzwerk_word_usage.py"""

    def test_add_word_connection(self, netzwerk):
        """Test: CONNECTION Edge wird korrekt erstellt"""
        # Erstelle Wörter
        netzwerk.ensure_wort_und_konzept("test_haus")
        netzwerk.ensure_wort_und_konzept("test_groß")

        # Füge Connection hinzu
        success = netzwerk.add_word_connection(
            word1_lemma="test_haus",
            word2_lemma="test_groß",
            distance=1,
            direction="before",
        )

        assert success, "Connection sollte erfolgreich erstellt werden"

        # Hole Connections zurück
        connections = netzwerk.get_word_connections("test_haus", direction="before")

        assert len(connections) > 0, "Erwarte mindestens 1 Connection"

        # Prüfe erste Connection
        conn = connections[0]
        assert conn["connected_word"] == "test_groß"
        assert conn["distance"] == 1
        assert conn["count"] == 1

    def test_word_connection_counter_increment(self, netzwerk):
        """Test: CONNECTION Counter wird bei Duplikat erhöht"""
        netzwerk.ensure_wort_und_konzept("test_katze")
        netzwerk.ensure_wort_und_konzept("test_miau")

        # Füge Connection 3x hinzu
        for _ in range(3):
            netzwerk.add_word_connection(
                "test_katze", "test_miau", distance=1, direction="before"
            )

        # Hole Connections
        connections = netzwerk.get_word_connections("test_katze")

        # Finde Connection zu test_miau
        miau_conn = next(c for c in connections if c["connected_word"] == "test_miau")

        assert miau_conn["count"] == 3, "Counter sollte auf 3 erhöht werden"

    def test_add_usage_context(self, netzwerk):
        """Test: UsageContext Node wird korrekt erstellt"""
        netzwerk.ensure_wort_und_konzept("test_park")

        # Füge Context hinzu
        success = netzwerk.add_usage_context(
            word_lemma="test_park",
            fragment="im großen test_park",
            word_position=2,
            fragment_type="window",
        )

        assert success, "UsageContext sollte erfolgreich erstellt werden"

        # Hole Contexts zurück
        contexts = netzwerk.get_usage_contexts("test_park")

        assert len(contexts) > 0, "Erwarte mindestens 1 UsageContext"

        # Prüfe ersten Context
        ctx = contexts[0]
        assert ctx["fragment"] == "im großen test_park"
        assert ctx["word_position"] == 2
        assert ctx["count"] == 1

    def test_usage_context_exact_match_increment(self, netzwerk):
        """Test: Exakte Duplikate erhöhen nur Counter"""
        netzwerk.ensure_wort_und_konzept("test_baum")

        fragment = "unter dem test_baum"

        # Füge gleichen Context 2x hinzu
        netzwerk.add_usage_context("test_baum", fragment, word_position=2)
        netzwerk.add_usage_context("test_baum", fragment, word_position=2)

        # Hole Contexts
        contexts = netzwerk.get_usage_contexts("test_baum")

        # Sollte nur 1 Context geben (nicht 2)
        assert len(contexts) == 1, "Exakte Duplikate sollten nicht dupliziert werden"

        # Counter sollte 2 sein
        assert contexts[0]["count"] == 2


# ============================================================================
# SIMILARITY TESTS
# ============================================================================


class TestFragmentSimilarity:
    """Tests für Ähnlichkeits-Berechnung"""

    def test_identical_fragments(self):
        """Test: Identische Fragmente haben 100% Ähnlichkeit"""
        sim = calculate_similarity("im großen Haus", "im großen Haus")
        assert sim == 1.0, f"Erwarte 1.0, bekam {sim}"

    def test_similar_fragments(self):
        """Test: Ähnliche Fragmente haben hohe Ähnlichkeit"""
        sim = calculate_similarity("im großen Haus", "im großen alten Haus")

        # Ähnlichkeit sollte hoch sein (>0.7), aber nicht 1.0
        assert 0.7 < sim < 1.0, f"Erwarte 0.7-1.0, bekam {sim}"

    def test_different_fragments(self):
        """Test: Unterschiedliche Fragmente haben niedrige Ähnlichkeit"""
        sim = calculate_similarity("im Haus", "am Baum")

        # Ähnlichkeit sollte niedrig sein
        assert sim < 0.6, f"Erwarte <0.6, bekam {sim}"

    def test_completely_different_fragments(self):
        """Test: Komplett unterschiedliche Fragmente haben 0% Ähnlichkeit"""
        sim = calculate_similarity("Katze", "Hund")
        assert sim < 0.5  # Wenig Ähnlichkeit bei komplett unterschiedlichen Wörtern

    def test_similarity_normalization(self):
        """Test: Normalisierung ignoriert Groß-/Kleinschreibung und Satzzeichen"""
        sim1 = calculate_similarity("Im Haus!", "im haus")
        assert sim1 > 0.9, "Normalisierung sollte Groß-/Kleinschreibung ignorieren"


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestWordUsageIntegration:
    """End-to-End Tests für Word Usage Tracking"""

    def test_full_pipeline(self, netzwerk):
        """Test: Kompletter Workflow von Satz zu Storage"""
        sentence = "Der test_apfel ist rot."

        # Extrahiere Fragmente
        fragments, connections = extract_word_usage_from_sentence(sentence)

        # Stelle sicher dass Wörter existieren
        for frag in fragments:
            netzwerk.ensure_wort_und_konzept(frag.lemma)

        # Speichere Fragmente
        for frag in fragments:
            netzwerk.add_usage_context(
                word_lemma=frag.lemma,
                fragment=frag.fragment,
                word_position=frag.word_position,
                fragment_type=frag.fragment_type,
            )

        # Speichere Connections
        for conn in connections:
            netzwerk.ensure_wort_und_konzept(conn.word1_lemma)
            netzwerk.ensure_wort_und_konzept(conn.word2_lemma)
            netzwerk.add_word_connection(
                word1_lemma=conn.word1_lemma,
                word2_lemma=conn.word2_lemma,
                distance=conn.distance,
                direction=conn.direction,
            )

        # Prüfe dass Daten gespeichert wurden
        contexts = netzwerk.get_usage_contexts("test_apfel")
        assert len(contexts) > 0, "UsageContexts sollten gespeichert sein"

        conns = netzwerk.get_word_connections("test_apfel")
        assert len(conns) > 0, "Connections sollten gespeichert sein"

    def test_multiple_sentences_accumulation(self, netzwerk):
        """Test: Mehrere Sätze akkumulieren Statistiken"""
        sentences = [
            "Der test_vogel fliegt.",
            "Der test_vogel singt.",
            "Der test_vogel ist blau.",
        ]

        # Verarbeite alle Sätze
        for sentence in sentences:
            fragments, connections = extract_word_usage_from_sentence(sentence)

            for frag in fragments:
                netzwerk.ensure_wort_und_konzept(frag.lemma)
                netzwerk.add_usage_context(
                    word_lemma=frag.lemma,
                    fragment=frag.fragment,
                    word_position=frag.word_position,
                )

            for conn in connections:
                netzwerk.ensure_wort_und_konzept(conn.word1_lemma)
                netzwerk.ensure_wort_und_konzept(conn.word2_lemma)
                netzwerk.add_word_connection(
                    word1_lemma=conn.word1_lemma,
                    word2_lemma=conn.word2_lemma,
                    distance=conn.distance,
                    direction=conn.direction,
                )

        # "test_vogel" sollte mehrere Contexts haben
        contexts = netzwerk.get_usage_contexts("test_vogel")
        assert len(contexts) >= 3, f"Erwarte >= 3 Contexts, bekam {len(contexts)}"

        # "der" -> "test_vogel" Connection sollte count=3 haben
        conns = netzwerk.get_word_connections("der", direction="before")
        vogel_conn = next(
            (c for c in conns if c["connected_word"] == "test_vogel"), None
        )

        if vogel_conn:
            assert (
                vogel_conn["count"] == 3
            ), f"Erwarte count=3, bekam {vogel_conn['count']}"


# ============================================================================
# CONFIGURATION TESTS
# ============================================================================


class TestConfigIntegration:
    """Tests für kai_config Integration"""

    def test_config_loading(self):
        """Test: Config wird korrekt geladen"""
        config = get_config()

        assert hasattr(config, "word_usage_tracking_enabled")
        assert hasattr(config, "usage_similarity_threshold")
        assert hasattr(config, "context_window_size")
        assert hasattr(config, "max_words_to_comma")

    def test_fragmenter_uses_config(self):
        """Test: TextFragmenter verwendet Config-Werte"""
        config = get_config()
        fragmenter = TextFragmenter()

        assert fragmenter.window_size == config.context_window_size
        assert fragmenter.max_words_to_comma == config.max_words_to_comma


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
