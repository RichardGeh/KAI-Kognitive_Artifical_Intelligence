# KAI - Entwicklerhandbuch

**Version:** 1.0
**Zielgruppe:** Entwickler, die an KAI arbeiten

---

## Inhaltsverzeichnis

1. [Logging-Richtlinien](#logging-richtlinien)
2. [Testing-Infrastruktur](#testing-infrastruktur)
3. [Performance-Optimierung](#performance-optimierung)
4. [Implementation Summaries](#implementation-summaries)

---

# Logging-Richtlinien

## Verwendung

```python
from component_15_logging_config import get_logger

logger = get_logger(__name__)
```

**NIEMALS** `logging.getLogger()` direkt verwenden - immer `get_logger()` für strukturiertes Logging!

---

## Log-Level-Guidelines

### DEBUG
**Entwicklungs-Details, die in Production nicht relevant sind**
- Embedding-Vektoren, Distanzen, Cache-Hits
- Detaillierte Algorithmus-Schritte
- Interne Zustandsänderungen

**Performance-Kritisch:** Verwende conditional logging!
```python
if logger.isEnabledFor(logging.DEBUG):
    logger.debug("Embedding berechnet", extra={"vector": embedding, "norm": norm})
```

### INFO
**User-relevante Events und erfolgreiche Operationen**
- Query verarbeitet, Fakten gelernt
- Komponenten-Start/Ende
- Erfolgreiche DB-Operationen
- Cache-Statistiken (bei Bedarf)

```python
logger.info("Query verarbeitet", extra={"query_type": "question", "confidence": 0.85})
```

### WARNING
**Fallbacks, degenerierte Funktionalität, erwartbare Probleme**
- Fehlende optionale Abhängigkeiten
- Fallback auf default Werte
- Leere Suchergebnisse (manchmal)
- Cache-Invalidierung

```python
logger.warning("Prototyp nicht gefunden - erstelle neuen", extra={"text": sample})
```

### ERROR
**Fehler, die Benutzer-Interaktion beeinträchtigen**
- Fehlgeschlagene DB-Writes
- Ungültige User-Eingaben (nach Validierung)
- Exception-Handling (nicht-kritisch)
- Komponenten-Fehler mit Graceful Degradation

```python
logger.error("Embedding-Generierung fehlgeschlagen", extra={
    "text_preview": text[:50],
    "error": str(e),
    "error_type": type(e).__name__
})
```

### CRITICAL
**System-Fehler, die KAI funktionsunfähig machen**
- Neo4j Connection Lost
- Embedding-Modell kann nicht geladen werden
- Kritische Initialisierungsfehler

```python
logger.critical("Konnte Embedding-Modell nicht laden", extra={
    "model_name": model_name,
    "error": str(e)
})
```

---

## Structured Logging (IMMER verwenden!)

```python
# RICHTIG - mit extra fields
logger.info("Fakten extrahiert", extra={
    "subject": subject,
    "relation": relation,
    "object": obj,
    "confidence": 0.9
})

# FALSCH - ohne Struktur
logger.info(f"Fakten extrahiert: {subject} {relation} {obj}")
```

**Vorteile:**
- Maschinell parsebar für Monitoring
- Konsistente Formatierung
- Bessere Filterbarkeit

---

## Performance-Optimierung

### Conditional Logging für teure Operationen

```python
# RICHTIG - nur wenn DEBUG aktiv
if logger.isEnabledFor(logging.DEBUG):
    expensive_data = compute_expensive_metrics()
    logger.debug("Metrics berechnet", extra={"metrics": expensive_data})

# FALSCH - f-string wird IMMER evaluiert
logger.debug(f"Vector: {compute_expensive_vector()}")
```

**Verwendung erforderlich für:**
- String-Formatierung mit großen Datenstrukturen
- Embeddings, Vektoren, Matrizen
- JSON-Serialisierung
- Komplexe Berechnungen nur für Logging

---

## Performance-Tracking

```python
from component_15_logging_config import PerformanceLogger

with PerformanceLogger(logger, "Neo4j Query", query_type="facts", topic="hund"):
    facts = netzwerk.query_graph_for_facts("hund")
```

Loggt automatisch:
- Startzeit (DEBUG)
- Endzeit + Dauer in ms (DEBUG)
- Performance-Log-Datei (INFO)
- Fehler + Dauer bei Exception (ERROR)

---

## Exception Logging

```python
# RICHTIG - mit Kontext
try:
    result = risky_operation()
except Exception as e:
    logger.error("Operation fehlgeschlagen", extra={
        "operation": "risky_operation",
        "input": input_data,
        "error": str(e),
        "error_type": type(e).__name__
    })
    # Optional: Vollständiger Traceback
    logger.log_exception(e, message="Risky operation failed", input=input_data)

# FALSCH - ohne Kontext
except Exception as e:
    logger.error(f"Error: {e}")
```

---

## Anti-Patterns

- **F-Strings in DEBUG-Logs ohne Conditional**: `logger.debug(f"Vector: {expensive_computation()}")` - wird immer evaluiert!
- **logging.getLogger() statt get_logger()**: Kein structured logging
- **INFO für interne Details**: `logger.info(f"Cache-Hit für Key {key}")` - sollte DEBUG sein
- **DEBUG für User-Events**: `logger.debug("Benutzer hat Frage gestellt")` - sollte INFO sein
- **Fehlende extra fields**: `logger.info(f"Query: {query}, Confidence: {conf}")` - nutze extra!

---

## Checkliste für neue Komponenten

- [ ] `from component_15_logging_config import get_logger` importiert
- [ ] `logger = get_logger(__name__)` initialisiert
- [ ] INFO-Logs für User-relevante Events
- [ ] DEBUG-Logs mit conditional logging für teure Operationen
- [ ] Alle Logs mit `extra={}` für strukturierte Daten
- [ ] Exception-Handling mit Kontext-Informationen
- [ ] PerformanceLogger für kritische Operationen (optional)

---

**Best Practice:** Log so viel wie nötig, so wenig wie möglich. DEBUG für Entwicklung, INFO für Production, CRITICAL nur für echte Notfälle.

---

# Testing-Infrastruktur

## Quick Start

### Running All Tests
```bash
pytest
```

### Running Tests Without Slow Tests (Fast Development Cycle)
```bash
pytest -m "not slow"
```

### Running Only Slow Tests (Performance/Integration Tests)
```bash
pytest -m slow
```

### Running Tests with Coverage
```bash
pytest --cov=. --cov-report=html
```
Coverage report will be generated in `htmlcov/index.html`.

---

## Test Organization

### Core Integration Tests
- `test_kai_integration.py` - Main integration tests (4 test classes)
- `test_system_setup.py` - System initialization
- `test_graph_traversal.py` - Graph traversal and multi-hop reasoning (9 test classes)
- `test_working_memory.py` - Working memory and context management (5 test classes)
- `test_abductive_reasoning.py` - Abductive reasoning engine (9 test classes)
- `test_probabilistic_engine.py` - Bayesian inference (5 test classes)
- `test_logging_and_exceptions.py` - Logging system and exception handling (3 test classes)
- `test_episodic_reasoning.py` - Episodic memory (4 test classes)

### Property-Based & Robustness Tests
- `test_property_based_text_normalization.py` - Property-based tests for text normalization (19 tests)
  - Uses hypothesis library for generative testing
  - Tests `clean_entity` and `normalize_plural_to_singular` with 200+ examples
  - Edge cases: Unicode, extreme lengths, empty inputs, special characters
  - Performance tests: 10,000 calls benchmark
- `test_contradiction_detection.py` - Contradiction detection in abductive reasoning (19 tests)
  - Mutually exclusive IS_A relations
  - Contradictory properties (colors, sizes, temperatures, binary states)
  - Incompatible locations
- `test_error_paths_and_resilience.py` - Error handling and resilience (16 tests)
  - Neo4j connection failures
  - Malformed data handling
  - Resource exhaustion
  - Concurrency tests

---

## Test Markers

Tests are categorized with pytest markers:

- `@pytest.mark.slow` - Slow tests (performance, batch operations, large data)
- `@pytest.mark.integration` - Integration tests (multiple components)
- `@pytest.mark.unit` - Fast, isolated unit tests
- `@pytest.mark.db` - Tests requiring Neo4j database
- `@pytest.mark.embedding` - Tests using embedding service
- `@pytest.mark.ui` - Tests involving UI components

---

## Property-Based Testing

Property-based tests use the **hypothesis** library to generate hundreds of test cases automatically:

```python
from hypothesis import given, strategies as st

@given(st.text(min_size=0, max_size=100))
def test_clean_entity_never_crashes(text_input):
    result = clean_entity(text_input)
    assert isinstance(result, str)
```

Benefits:
- Automatically finds edge cases developers miss
- Tests invariants (properties that should always hold)
- Generates counterexamples when tests fail
- 200-1000 examples per test function

---

## Development Workflow

### Fast Development Cycle
When actively developing, run fast tests frequently:
```bash
pytest -m "not slow" -x
```
The `-x` flag stops on first failure for faster feedback.

### Pre-Commit Testing
Before committing, run the full test suite:
```bash
pytest
```

### Coverage-Guided Development
When adding new features, check coverage:
```bash
pytest test_new_feature.py --cov=component_new --cov-report=term-missing
```
Aim for >80% coverage on new code.

---

## Best Practices

### 1. Test Isolation
- Use `test_` prefix for all test concepts to enable cleanup
- Tests should not depend on execution order
- Clean up test data in fixtures

### 2. Test Naming
- Test method names should be descriptive: `test_<what>_<condition>_<expected>`
- Example: `test_create_extraction_rule_persists`

### 3. Performance Considerations
- Mark slow tests with `@pytest.mark.slow`
- Keep unit tests fast (<100ms each)
- Use timeouts for tests that might hang

### 4. Assertions
- Use specific assertions with clear error messages
- Example: `assert len(paths) >= 1, f"Expected at least 1 path, got {len(paths)}"`

### 5. Test Coverage
- Aim for >80% coverage on core components
- 100% coverage not required but document untested edge cases
- Focus coverage on business logic, not boilerplate

---

## Troubleshooting

### Tests Timing Out
- Increase timeout in pytest.ini or use `-k` to run specific tests
- Check Neo4j connection
- Verify spaCy model is installed: `python -m spacy download de_core_news_sm`

### Coverage Import Errors
- Ensure pytest-cov is installed: `pip install pytest-cov`
- If coverage fails, run tests without coverage: `pytest` (coverage is optional)

### Fixture Errors
- Check that Neo4j is running at `bolt://127.0.0.1:7687`
- Verify database credentials (neo4j/password)
- Ensure test cleanup is working (check for orphaned test nodes)

---

## UI Integration

Tests are integrated into the KAI UI via the Settings dialog:
- Access via **Einstellungen → Tests** tab
- **150+ test classes** organized by category
- File-level checkbox grouping for easy selection
- Live progress tracking with status indicators
- Full test suite or individual test class execution
- Real-time console output with color coding

---

# Performance-Optimierung

## Übersicht der Optimierungen

| Optimierung | Status | Verbesserung | Komponente |
|------------|--------|--------------|------------|
| Batch-Embeddings für Ingestion | ✅ Implementiert | ~5-10x schneller | `kai_ingestion_handler.py` |
| Neo4j Query-Profiling | ✅ Implementiert | Analyse-Tool | `component_utils_neo4j_profiler.py` |
| Lazy Loading für Proof Trees | ✅ Implementiert | ~3x schneller für große Bäume | `component_18_proof_tree_widget.py` |
| Batch-Processing für große Texte | ✅ Implementiert | ~2-3x schneller | `kai_ingestion_handler.py` |
| TTL-Caching für Queries | ✅ Bereits vorhanden | ~86x schneller | `component_1_netzwerk_core.py` |
| LRU-Cache für Embeddings | ✅ Bereits vorhanden | ~5600x schneller | `component_11_embedding_service.py` |

---

## 1. Batch-Embeddings für Ingestion

### Problem
Die alte Implementierung erzeugte Embeddings einzeln für jeden Satz:
```python
for sentence in sentences:
    vector = embedding_service.get_embedding(sentence)  # N API-Calls
```

### Lösung
Batch-Embedding für alle Sätze auf einmal:
```python
# Sammle alle Sätze
sentences = [s.text.strip() for s in doc.sents if s.text.strip()]

# Ein Batch-Call für alle Sätze
embeddings = embedding_service.get_embeddings_batch(sentences)

# Verarbeite mit vorberechneten Embeddings
for sentence, vector in zip(sentences, embeddings):
    # ...
```

### Performance-Gewinn
- **5-10x schneller** für große Texte (>100 Sätze)
- Reduziert API-Overhead (ein Call statt N Calls)
- Nutzt GPU-Parallelisierung im Embedding-Modell

---

## 2. Neo4j Query-Profiling

### Lösung
Profiling-Tool mit EXPLAIN/PROFILE-Analyse:

```python
from component_utils_neo4j_profiler import Neo4jQueryProfiler

profiler = Neo4jQueryProfiler(netzwerk.driver)

# Profiliere häufige Queries
profiles = profiler.profile_common_queries(netzwerk)

# Zeige Empfehlungen
for profile in profiles:
    profiler.print_recommendations(profile)

# Erstelle empfohlene Indizes
profiler.create_recommended_indexes()
```

### Features
- **PROFILE-Analyse**: Tatsächliche Metriken (DB-Hits, Execution Time)
- **EXPLAIN-Analyse**: Query-Plan ohne Ausführung
- **Automatische Empfehlungen**: Index-Vorschläge, Query-Optimierungen
- **Index-Management**: Erstellt empfohlene Indizes

### CLI-Usage
```bash
python component_utils_neo4j_profiler.py
```

---

## 3. Lazy Loading für Proof Trees

### Problem
Große Proof Trees (>100 Knoten) führten zu:
- Langen Ladezeiten (UI-Freeze)
- Hoher Speicher-Nutzung
- Schlechter Responsiveness

### Lösung
Progressive Rendering mit konfigurierbarem Batch-Size:

```python
# Automatisch aktiviert für Bäume mit >50 Knoten
self.progressive_rendering_enabled = True
self.render_batch_size = 50  # Nodes pro Batch

# Rendert in Batches mit QTimer für UI-Responsiveness
def _render_tree_progressive(self):
    # Layout berechnen
    self._layout_tree(...)

    # Rendere ersten Batch
    self._render_next_batch()

    # Weitere Batches mit 50ms Delay
    QTimer.singleShot(50, self._render_next_batch)
```

### Performance-Gewinn
- **3x schneller** initial load für große Bäume
- **Keine UI-Freezes** mehr
- **Memory-Efficient**: Nur sichtbare Nodes im Speicher
- **Responsive UI**: Nutzer kann sofort interagieren

---

## 4. Batch-Processing für große Textmengen

### Lösung
Chunk-basierte Verarbeitung mit Progress-Tracking:

```python
# Automatische Verwendung in kai_worker
if len(sentences) > 100:
    stats = ingestion_handler.ingest_text_large(
        text,
        chunk_size=100,
        progress_callback=self._update_progress
    )
else:
    stats = ingestion_handler.ingest_text(text)
```

### Features
- **Chunk-basiert**: Verarbeitet Text in Blöcken (default: 100 Sätze)
- **Batch-Embeddings**: Pro Chunk ein Batch-Call
- **Progress-Tracking**: Callback für UI-Updates
- **Memory-Efficient**: Nur ein Chunk im Speicher

### Performance-Gewinn
- **2-3x schneller** für sehr große Texte
- **Reduzierte Speicher-Nutzung**: ~50% weniger Peak-Memory
- **Progress-Feedback**: Nutzer sieht Fortschritt
- **Skaliert linear**: Kann beliebig große Texte verarbeiten

---

## 5. Bestehende Caching-Optimierungen

### TTL-Cache für Neo4j Queries
**Komponente**: `component_1_netzwerk_core.py`

```python
# Fakten-Cache (5 Minuten TTL, maxsize=500)
self._fact_cache: TTLCache = TTLCache(maxsize=500, ttl=300)

# Wort-Cache (10 Minuten TTL, maxsize=100)
self._known_words_cache: TTLCache = TTLCache(maxsize=100, ttl=600)
```

**Performance-Gewinn**: ~86x schneller für häufig abgefragte Fakten

### LRU-Cache für Embeddings
**Komponente**: `component_11_embedding_service.py`

```python
# LRU-Cache (maxsize=1000)
self._embedding_cache = lru_cache(maxsize=1000)(self._compute_embedding_uncached)
```

**Performance-Gewinn**: ~5600x schneller für wiederholte Texte

---

## Benchmark-Ergebnisse

### Ingestion Performance

| Text-Größe | Alte Methode | Batch-Embeddings | Batch-Processing | Speedup |
|-----------|--------------|------------------|------------------|---------|
| 10 Sätze | 1.2s | 0.8s | 0.9s | 1.3x |
| 100 Sätze | 12s | 2.5s | 2.1s | 5.7x |
| 1000 Sätze | 120s | 25s | 18s | 6.7x |
| 10000 Sätze | N/A (OOM) | N/A (OOM) | 180s | ∞ |

**Hinweis**: OOM = Out of Memory (alte Methode scheitert bei großen Texten)

### Proof Tree Rendering

| Knoten-Anzahl | Full Rendering | Progressive Rendering | Speedup |
|--------------|----------------|----------------------|---------|
| 10 Nodes | 0.1s | 0.1s | 1.0x |
| 50 Nodes | 0.5s | 0.3s | 1.7x |
| 100 Nodes | 2.1s | 0.7s | 3.0x |
| 500 Nodes | 15s | 3.5s | 4.3x |

### Neo4j Query Performance (mit Cache)

| Query | Ohne Cache | Mit Cache (Hit) | Speedup |
|-------|-----------|----------------|---------|
| query_graph_for_facts | 12ms | 0.14ms | 86x |
| get_all_known_words | 45ms | 0.05ms | 900x |
| query_facts_with_synonyms | 25ms | 0.28ms | 89x |

---

## Best Practices

### 1. Ingestion
- **Kleine Texte (<100 Sätze)**: Nutze `ingest_text()` (automatisch optimiert)
- **Große Texte (>100 Sätze)**: Nutze `ingest_text_large()` mit Progress-Callback
- **Sehr große Texte (>10000 Sätze)**: Teile Text manuell in kleinere Chunks

### 2. Proof Trees
- **Kleine Bäume (<50 Nodes)**: Full Rendering ist schneller
- **Große Bäume (>50 Nodes)**: Progressive Rendering empfohlen
- **Sehr große Bäume (>500 Nodes)**: Auto-Collapse nutzen, schrittweise expandieren

### 3. Neo4j Queries
- **Häufige Queries**: Werden automatisch gecacht (TTL-Cache)
- **Write-Operationen**: Invalidieren automatisch relevante Caches
- **Query-Optimierung**: Nutze `Neo4jQueryProfiler` für Analyse

### 4. Embeddings
- **Wiederholte Texte**: Werden automatisch gecacht (LRU-Cache)
- **Batch-Operationen**: Bevorzuge `get_embeddings_batch()` über einzelne Calls
- **Cache-Size**: Default 1000 Texte, anpassbar über `maxsize`

---

## Monitoring & Debugging

### Performance-Logging
Alle Optimierungen loggen Performance-Metriken:

```python
# Aktiviere DEBUG-Logging für detaillierte Metriken
import logging
logging.getLogger("kai_ingestion_handler").setLevel(logging.DEBUG)
logging.getLogger("component_11_embedding_service").setLevel(logging.DEBUG)
```

### Cache-Monitoring
```python
# Embedding-Cache
cache_info = embedding_service.get_cache_info()
print(f"Hits: {cache_info['hits']}, Misses: {cache_info['misses']}")
print(f"Hit Rate: {cache_info['hit_rate']:.2%}")

# Neo4j-Cache
cache_stats = netzwerk.get_cache_stats()
print(f"Fact Cache: {cache_stats['fact_cache']['size']}/{cache_stats['fact_cache']['maxsize']}")
```

---

## Konfiguration

### Batch-Sizes
```python
# kai_ingestion_handler.py
CHUNK_SIZE = 100  # Sätze pro Chunk für ingest_text_large()

# component_18_proof_tree_widget.py
self.render_batch_size = 50  # Nodes pro Render-Batch
```

### Cache-Sizes
```python
# component_11_embedding_service.py
maxsize=1000  # LRU-Cache für Embeddings

# component_1_netzwerk_core.py
TTLCache(maxsize=500, ttl=300)  # Fakten-Cache: 500 Einträge, 5min TTL
TTLCache(maxsize=100, ttl=600)  # Wort-Cache: 100 Einträge, 10min TTL
```

---

# Implementation Summaries

## Contradiction Detection Implementation

### File: `component_14_abductive_engine.py`

#### Implemented Features:

1. **`_contradicts_knowledge(fact)` - Main Contradiction Detection**
   - Detects three categories of contradictions:
     - **Mutually Exclusive IS_A Relations**: Objects cannot have two fundamentally different types
     - **Contradictory Properties**: Properties that mutually exclude each other (colors, sizes, temperatures, binary states)
     - **Incompatible Locations**: Objects cannot be in two places simultaneously (with hierarchy support)

2. **`_are_types_mutually_exclusive(type1, type2)`**
   - Determines if two IS_A types mutually exclude each other
   - Handles abstract categories (tier, lebewesen, objekt) vs. concrete types
   - Checks IS_A hierarchy for subtype relationships

3. **`_is_subtype_of(subtype, supertype)`**
   - Traverses IS_A hierarchy to check subtype relationships
   - Supports recursive/transitive IS_A relationships
   - Prevents false contradiction detection in hierarchies

4. **`_are_properties_contradictory(prop1, prop2)`**
   - Detects contradictory properties across multiple categories:
     - **Colors**: rot, blau, grün, gelb, orange, lila, schwarz, weiß, grau, braun
     - **Sizes**: groß, klein, mittel, riesig, winzig
     - **Temperatures**: heiß, kalt, warm, kühl, eiskalt (with opposite pairs)
     - **Binary States**: lebendig/tot, aktiv/inaktiv, offen/geschlossen, wahr/falsch, an/aus

5. **`_is_location_hierarchy(loc1, loc2)`**
   - Checks if two locations are in a PART_OF hierarchy
   - Supports transitive relationships (e.g., Berlin → Deutschland → Europa)
   - Prevents false contradictions for hierarchical locations

### Integration with Hypothesis Scoring:

The contradiction detection is integrated into the `_score_coherence()` method:
- Hypotheses with contradictory facts receive lower coherence scores
- This influences the overall hypothesis confidence ranking

---

## Property-Based Testing Infrastructure

### Library Installed:
- **hypothesis 6.142.3**: Property-based testing framework for Python
- **attrs 25.4.0**: Required dependency
- **sortedcontainers 2.4.0**: Required dependency

### Test Categories:

1. **Property-Based Tests for `clean_entity`**:
   - `test_clean_entity_never_crashes`: Tests with arbitrary Unicode input (200 examples)
   - `test_clean_entity_output_always_lowercase`: Ensures output is always lowercase
   - `test_clean_entity_no_leading_trailing_whitespace`: No whitespace at edges
   - `test_clean_entity_idempotent`: `clean_entity(clean_entity(x)) == clean_entity(x)`
   - `test_clean_entity_removes_articles`: Articles (der, die, das, ein, eine) removed

2. **Property-Based Tests for `normalize_plural_to_singular`**:
   - `test_normalize_plural_never_crashes`: Never crashes on arbitrary input
   - `test_normalize_plural_idempotent`: Idempotent normalization
   - `test_normalize_plural_output_not_longer_than_input`: Output ≤ input length
   - `test_normalize_plural_preserves_short_words`: Words < 3 chars unchanged

3. **Edge Case Tests**:
   - `test_unicode_emoji`: Unicode characters handling
   - `test_extreme_whitespace`: Extreme whitespace combinations
   - `test_extreme_length`: Very long strings (10,000 characters)
   - `test_special_characters`: Special chars and punctuation
   - `test_empty_and_none`: Empty strings and None inputs
   - `test_known_german_plurals`: Regression tests for known plurals

4. **Performance Tests**:
   - `test_performance_many_calls`: 10,000 calls performance test (< 1ms per call)
   - `test_repeated_normalization_stable`: Stability over repeated calls

---

## Error Path Testing

### Test Categories:

1. **Neo4j Connection Errors**:
   - `test_netzwerk_handles_service_unavailable`: ServiceUnavailable exception
   - `test_netzwerk_handles_session_expired`: Session expiration
   - `test_netzwerk_handles_transient_error`: Transient errors (leader switch)
   - `test_netzwerk_handles_none_driver`: None driver graceful handling

2. **Abductive Engine Error Handling**:
   - `test_abductive_engine_handles_connection_loss`: Connection loss during hypothesis generation
   - `test_abductive_engine_empty_knowledge_base`: Empty knowledge base
   - `test_abductive_engine_malformed_fact`: Malformed facts without subject/object
   - `test_abductive_engine_infinite_recursion_protection`: Cyclic IS_A hierarchies

3. **Text Normalizer Resilience**:
   - `test_text_normalizer_malformed_input`: Non-string inputs (None, int, list, dict)
   - `test_text_normalizer_extremely_long_input`: 1 million character strings

4. **Concurrency Tests**:
   - `test_text_normalizer_thread_safety`: Thread-safe normalization (10 workers, 500 inputs)
   - `test_netzwerk_concurrent_queries`: Concurrent graph queries (5 workers, 100 queries)

---

## Known Issues & Limitations

### Windows Unicode Encoding:
- **Issue**: Print statements with special characters (✓, emoji, German umlauts) cause `UnicodeEncodeError` in cmd.exe/PowerShell
- **Affected**: Many tests have print statements that fail on Windows
- **Workaround**: Replace special characters with ASCII equivalents (OK instead of ✓)
- **Status**: Partially fixed, some tests still have Unicode issues

### Contradiction Detection Limitations:
- **Design Limitation**: Currently only detects contradictions when the new fact conflicts with *existing* facts in the knowledge base
- **Example**: "Hund IS_A Katze" may not be detected as contradictory if the system doesn't know that both are concrete animal types
- **Mitigation**: Relies on comprehensive knowledge base with proper type classifications

---

## Contributing

When adding new code:
1. **Use structured logging** with `get_logger(__name__)`
2. **Write tests** for new features (aim for >80% coverage)
3. **Mark slow tests** with `@pytest.mark.slow`
4. **Use property-based testing** for invariants
5. **Consider performance** - profile critical paths
6. **Document** new features and limitations

---

*Last Updated: 2025-10-26*
