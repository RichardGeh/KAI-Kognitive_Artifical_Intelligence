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
- Access via **Einstellungen -> Tests** tab
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
| Batch-Embeddings fuer Ingestion | [OK] Implementiert | ~5-10x schneller | `kai_ingestion_handler.py` |
| Neo4j Query-Profiling | [OK] Implementiert | Analyse-Tool | `component_utils_neo4j_profiler.py` |
| Lazy Loading fuer Proof Trees | [OK] Implementiert | ~3x schneller fuer grosse Baeume | `component_18_proof_tree_widget.py` |
| Batch-Processing fuer grosse Texte | [OK] Implementiert | ~2-3x schneller | `kai_ingestion_handler.py` |
| TTL-Caching fuer Queries | [OK] Bereits vorhanden | ~86x schneller | `component_1_netzwerk_core.py` |
| LRU-Cache fuer Embeddings | [OK] Bereits vorhanden | ~5600x schneller | `component_11_embedding_service.py` |

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
   - Supports transitive relationships (e.g., Berlin -> Deutschland -> Europa)
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
   - `test_normalize_plural_output_not_longer_than_input`: Output <= input length
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
- **Issue**: Print statements with special characters (checkmarks, emoji, German umlauts) cause `UnicodeEncodeError` in cmd.exe/PowerShell
- **Affected**: Many tests have print statements that fail on Windows
- **Workaround**: Replace special characters with ASCII equivalents (OK instead of checkmarks)
- **Status**: Partially fixed, some tests still have Unicode issues

### Contradiction Detection Limitations:
- **Design Limitation**: Currently only detects contradictions when the new fact conflicts with *existing* facts in the knowledge base
- **Example**: "Hund IS_A Katze" may not be detected as contradictory if the system doesn't know that both are concrete animal types
- **Mitigation**: Relies on comprehensive knowledge base with proper type classifications

---

## Spatial Reasoning API Reference

### File: `component_42_spatial_reasoning.py` (3069 lines)

**Purpose**: Generic 2D spatial reasoning for grids, geometric shapes, and spatial relations. Domain-agnostic (no hardcoded chess, Sudoku, etc.).

---

### Core Classes

#### 1. `Position`

**Immutable 2D coordinate**:
```python
@dataclass(frozen=True, order=True)
class Position:
    x: int
    y: int
```

**Methods**:
- `distance_to(other, metric='euclidean')` -> `float`
  - Metrics: `'euclidean'`, `'manhattan'`, `'chebyshev'`
- `manhattan_distance_to(other)` -> `float`
- `euclidean_distance_to(other)` -> `float`
- `direction_to(other)` -> `Optional[SpatialRelationType]`
  - Returns cardinal direction (NORTH_OF, SOUTH_OF, etc.) or None
- `get_neighbors(neighborhood_type, custom_offsets=None)` -> `List[Position]`
  - Neighborhood types: ORTHOGONAL (4), DIAGONAL (4), MOORE (8), CUSTOM

**Example**:
```python
pos = Position(3, 4)
other = Position(6, 8)

pos.manhattan_distance_to(other)  # -> 7
pos.euclidean_distance_to(other)  # -> 5.0
pos.direction_to(Position(3, 5))  # -> SpatialRelationType.NORTH_OF

neighbors = pos.get_neighbors(NeighborhoodType.ORTHOGONAL)
# -> [Position(3,5), Position(3,3), Position(4,4), Position(2,4)]
```

---

#### 2. `Grid`

**Generic NxM grid structure**:
```python
@dataclass
class Grid:
    width: int
    height: int
    name: str
    neighborhood_type: NeighborhoodType
    custom_offsets: Optional[List[Tuple[int, int]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**Methods**:
- `is_valid_position(pos)` -> `bool`: Check if position is within bounds
- `get_all_positions()` -> `List[Position]`: All grid positions
- `get_neighbors(pos)` -> `List[Position]`: Valid neighbors of position
- `place_object(obj_name, pos)`: Place object at position
- `move_object(obj_name, new_pos)`: Move object to new position
- `get_object_at_position(pos)` -> `Optional[str]`: Get object at position
- `remove_object(obj_name)`: Remove object from grid

**Example**:
```python
grid = Grid(
    width=8,
    height=8,
    name="chessboard",
    neighborhood_type=NeighborhoodType.ORTHOGONAL
)

grid.is_valid_position(Position(7, 7))  # -> True
grid.is_valid_position(Position(8, 8))  # -> False

grid.place_object("king", Position(4, 0))
grid.get_object_at_position(Position(4, 0))  # -> "king"
```

---

#### 3. `SpatialRelationType` (Enum)

**14 spatial relation types**:

**Directional (Transitive)**:
- `NORTH_OF`, `SOUTH_OF`, `EAST_OF`, `WEST_OF`

**Adjacency (Symmetric)**:
- `ADJACENT_TO`, `NEIGHBOR_ORTHOGONAL`, `NEIGHBOR_DIAGONAL`

**Hierarchical (Transitive)**:
- `INSIDE`, `CONTAINS`, `ABOVE`, `BELOW`

**Positional**:
- `BETWEEN`, `LOCATED_AT`

**Properties**:
- `is_symmetric` -> `bool`: Check if relation is symmetric
- `is_transitive` -> `bool`: Check if relation is transitive
- `inverse` -> `Optional[SpatialRelationType]`: Get inverse relation

**Example**:
```python
rel = SpatialRelationType.NORTH_OF

rel.is_transitive  # -> True
rel.inverse        # -> SpatialRelationType.SOUTH_OF

adj = SpatialRelationType.ADJACENT_TO
adj.is_symmetric   # -> True
```

---

#### 4. `SpatialRelation`

**Represents a spatial relation between two entities**:
```python
@dataclass
class SpatialRelation:
    subject: str
    object: str
    relation_type: SpatialRelationType
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**Methods**:
- `to_inverse()` -> `Optional[SpatialRelation]`: Create inverse relation

**Example**:
```python
rel = SpatialRelation(
    subject="king",
    object="queen",
    relation_type=SpatialRelationType.ADJACENT_TO,
    confidence=1.0
)

print(rel)  # -> "king ADJACENT_TO queen (conf=1.00)"

inv = rel.to_inverse()
# -> SpatialRelation("queen", "king", ADJACENT_TO, 1.0)
```

---

#### 5. Geometric Shapes

**Triangle**:
```python
@dataclass
class Triangle:
    vertices: List[Position]  # Exactly 3 vertices
```

**Methods**:
- `area()` -> `float`: Calculate area (Heron's formula)
- `perimeter()` -> `float`: Calculate perimeter
- `is_valid()` -> `bool`: Check if triangle is valid (non-degenerate)

**Quadrilateral**:
```python
@dataclass
class Quadrilateral:
    vertices: List[Position]  # Exactly 4 vertices
```

**Methods**:
- `area()` -> `float`: Calculate area (Shoelace formula)
- `perimeter()` -> `float`: Calculate perimeter
- `is_rectangle()` -> `bool`: Check if rectangle
- `is_square()` -> `bool`: Check if square
- `diagonals()` -> `Tuple[float, float]`: Length of diagonals

**Circle**:
```python
@dataclass
class Circle:
    center: Position
    radius: float
```

**Methods**:
- `area()` -> `float`: PIr²
- `circumference()` -> `float`: 2PIr
- `contains_point(pos)` -> `bool`: Check if point is inside circle
- `distance_to_edge(pos)` -> `float`: Distance from point to circle edge

**Example**:
```python
triangle = Triangle([Position(0,0), Position(4,0), Position(2,3)])
triangle.area()       # -> 6.0
triangle.perimeter()  # -> 10.0

circle = Circle(center=Position(5, 5), radius=3.0)
circle.area()                      # -> 28.27
circle.contains_point(Position(6, 6))  # -> True
```

---

### Main API: `SpatialReasoningEngine`

**Initialization**:
```python
from component_42_spatial_reasoning import SpatialReasoningEngine
from component_1_netzwerk import KonzeptNetzwerk

netzwerk = KonzeptNetzwerk()
spatial = SpatialReasoningEngine(netzwerk)
```

---

### Grid Operations

#### `create_grid(...)`
```python
def create_grid(
    self,
    width: int,
    height: int,
    name: str = "",
    neighborhood_type: NeighborhoodType = NeighborhoodType.ORTHOGONAL,
    custom_offsets: Optional[List[Tuple[int, int]]] = None
) -> Grid
```

**Creates a new grid and optionally stores it in Neo4j**.

**Example**:
```python
# Standard 8*8 chess board
chess = spatial.create_grid(8, 8, "chess", NeighborhoodType.ORTHOGONAL)

# Custom knight-move grid
KNIGHT_MOVES = [(-2,-1), (-2,1), (-1,-2), (-1,2), (1,-2), (1,2), (2,-1), (2,1)]
knight_grid = spatial.create_grid(
    8, 8, "knight_grid",
    NeighborhoodType.CUSTOM,
    custom_offsets=KNIGHT_MOVES
)
```

---

#### `place_object_on_grid(...)`
```python
def place_object_on_grid(
    self,
    grid_name: str,
    object_name: str,
    position: Position
) -> bool
```

**Places an object at a specific position on the grid**.

**Neo4j Structure**:
- Creates `LOCATED_AT` relation from object to position node
- Position node stores `(x, y)` coordinates

**Example**:
```python
spatial.place_object_on_grid("chess", "king_white", Position(4, 0))
spatial.place_object_on_grid("chess", "queen_white", Position(3, 0))
```

---

#### `get_object_position(...)`
```python
def get_object_position(
    self,
    grid_name: str,
    object_name: str
) -> Optional[Position]
```

**Retrieves the position of an object on a grid**.

**Example**:
```python
pos = spatial.get_object_position("chess", "king_white")
# -> Position(4, 0)
```

---

#### `move_object_on_grid(...)`
```python
def move_object_on_grid(
    self,
    grid_name: str,
    object_name: str,
    new_position: Position,
    record_history: bool = False
) -> bool
```

**Moves an object to a new position**.

**Options**:
- `record_history=True`: Store movement in episodic memory (for replay/undo)

**Example**:
```python
spatial.move_object_on_grid("chess", "king_white", Position(4, 1), record_history=True)
```

---

### Path-Finding

#### `find_path(...)`
```python
def find_path(
    self,
    grid_name: str,
    start: Position,
    goal: Position,
    algorithm: str = "bfs",
    obstacles: Optional[List[Position]] = None,
    heuristic: Optional[Callable[[Position, Position], float]] = None
) -> Optional[List[Position]]
```

**Finds a path from start to goal using specified algorithm**.

**Algorithms**:
- `"bfs"`: Breadth-First Search (guaranteed shortest path)
- `"dfs"`: Depth-First Search (memory efficient)
- `"astar"`: A* Search (optimal with heuristic)

**Parameters**:
- `obstacles`: List of blocked positions
- `heuristic`: Custom heuristic function for A* (default: Manhattan distance)

**Example**:
```python
# BFS path-finding
path = spatial.find_path(
    "chess",
    start=Position(0, 0),
    goal=Position(7, 7),
    algorithm="bfs",
    obstacles=[Position(3, 3), Position(4, 4)]
)
# -> [Position(0,0), Position(1,0), Position(2,0), ...]

# A* with custom heuristic
def custom_heuristic(pos, goal):
    return pos.euclidean_distance_to(goal) * 1.1

path = spatial.find_path(
    "chess",
    Position(0, 0),
    Position(7, 7),
    algorithm="astar",
    heuristic=custom_heuristic
)
```

---

### Spatial Relations

#### `store_spatial_relation(...)`
```python
def store_spatial_relation(
    self,
    subject: str,
    object: str,
    relation_type: str,  # or SpatialRelationType
    position_subject: Optional[Position] = None,
    position_object: Optional[Position] = None,
    confidence: float = 1.0
) -> bool
```

**Stores a spatial relation in Neo4j**.

**Example**:
```python
spatial.store_spatial_relation(
    subject="king",
    object="queen",
    relation_type="ADJACENT_TO",
    position_subject=Position(4, 0),
    position_object=Position(3, 0),
    confidence=1.0
)
```

---

#### `query_spatial_relations(...)`
```python
def query_spatial_relations(
    self,
    subject: str,
    relation_type: Optional[str] = None
) -> List[SpatialRelation]
```

**Queries spatial relations for a subject**.

**Example**:
```python
relations = spatial.query_spatial_relations("king", "ADJACENT_TO")
# -> [SpatialRelation("king", "queen", ADJACENT_TO, 1.0)]

all_relations = spatial.query_spatial_relations("king")
# -> All relations involving "king"
```

---

#### `infer_transitive_relations(...)`
```python
def infer_transitive_relations(
    self,
    subject: str,
    relation_type: SpatialRelationType,
    max_depth: int = 3
) -> List[SpatialRelation]
```

**Infers transitive spatial relations**.

**Only works for transitive relations**: NORTH_OF, SOUTH_OF, EAST_OF, WEST_OF, INSIDE, CONTAINS, ABOVE, BELOW

**Example**:
```python
# Given: A NORTH_OF B, B NORTH_OF C
# Infer: A NORTH_OF C (transitive)

inferred = spatial.infer_transitive_relations(
    "city_A",
    SpatialRelationType.NORTH_OF,
    max_depth=3
)
# -> [SpatialRelation("city_A", "city_C", NORTH_OF, 0.85)]
```

---

### Pattern Learning

#### `store_spatial_configuration(...)`
```python
def store_spatial_configuration(
    self,
    name: str,
    objects_and_positions: Dict[str, Position],
    tolerance: float = 0.5
) -> bool
```

**Stores a spatial configuration as a pattern**.

**Use Case**: Learn common spatial patterns (e.g., chess checkmate positions, Sudoku configurations)

**Example**:
```python
checkmate_pattern = {
    "king": Position(4, 0),
    "queen": Position(4, 1),
    "rook": Position(3, 0)
}

spatial.store_spatial_configuration(
    "back_rank_checkmate",
    checkmate_pattern,
    tolerance=0.5
)
```

---

#### `detect_spatial_patterns(...)`
```python
def detect_spatial_patterns(
    self,
    objects_and_positions: Dict[str, Position]
) -> List[str]
```

**Detects which learned patterns match the current configuration**.

**Returns**: List of pattern names that match

**Example**:
```python
current_config = {
    "king": Position(4, 0),
    "queen": Position(4, 1),
    "rook": Position(3, 0)
}

matches = spatial.detect_spatial_patterns(current_config)
# -> ["back_rank_checkmate"]
```

---

### Performance Considerations

**Caching** (recommended for production):
- Position queries: TTL Cache (5 min) -> 10-20x speedup
- Path-finding: Memoize paths -> 5-10x speedup
- Pattern matching: Cache configurations -> 3-5x speedup

**Batch Operations**:
- Use `place_multiple_objects()` for bulk placement
- Use `create_grid_with_objects()` for initial setup

**Neo4j Query Optimization**:
- Index on grid names: `CREATE INDEX ON :Grid(lemma)`
- Index on positions: `CREATE INDEX ON :Position(x, y)`

**Example Profiling**:
```python
# Enable profiling
spatial.enable_profiling(True)

# Run operations
path = spatial.find_path("chess", Position(0,0), Position(7,7))

# Get profile stats
stats = spatial.get_profile_stats()
# -> {"neo4j_queries": 5, "total_time_ms": 23.4, "cache_hits": 12}
```

---

### Integration with Other Components

**Logic Engine Integration**:
```python
from component_9_logik_engine import Engine, Fact

# Convert spatial relation to Fact
fact = Fact(
    pred="ADJACENT_TO",
    args={"subject": "king", "object": "queen"},
    confidence=1.0
)

engine.add_fact(fact)
```

**Graph Traversal Integration**:
```python
from component_12_graph_traversal import GraphTraversal

traversal = GraphTraversal(netzwerk)

# Use spatial relations for graph traversal
path = traversal.find_path_spatial_aware(
    start="city_A",
    goal="city_B",
    spatial_relation="CONNECTED_TO"
)
```

**CSP Integration** (Spatial Constraints):
```python
from component_29_constraint_reasoning import Constraint

# Constraint: King and Queen must be adjacent
def adjacent_constraint(assignment):
    king_pos = assignment.get("king_position")
    queen_pos = assignment.get("queen_position")
    if not king_pos or not queen_pos:
        return True
    return king_pos.manhattan_distance_to(queen_pos) == 1

constraint = Constraint(
    name="king_queen_adjacent",
    scope=["king_position", "queen_position"],
    predicate=adjacent_constraint
)
```

---

### Testing

**Test Coverage**:
- 150+ tests across 13 test files
- Core data structures, path-finding, Neo4j integration, pattern learning

**Run Tests**:
```bash
# All spatial tests
pytest tests/test_spatial*.py -v

# Specific phase
pytest tests/test_spatial_reasoning_phase1.py -v

# With coverage
pytest tests/test_spatial*.py --cov=component_42_spatial_reasoning
```

---

### UI Widget: `component_43_spatial_grid_widget.py`

**Interactive Grid Visualization**:
```python
from component_43_spatial_grid_widget import SpatialGridWidget

widget = SpatialGridWidget()

# Set up grid
widget.set_grid(8, 8, cell_size=60, show_coordinates=True)

# Highlight positions
widget.highlight_positions([Position(4, 0), Position(3, 0)])

# Show path
widget.show_path([Position(0,0), Position(1,1), Position(2,2)])

# Animate movement
widget.animate_object_movement("king", path, duration_ms=1000)
```

---

### Best Practices

1. **Always validate positions**: Use `grid.is_valid_position(pos)` before operations
2. **Use type-safe enums**: Prefer `SpatialRelationType.NORTH_OF` over string `"NORTH_OF"`
3. **Cache expensive operations**: Path-finding, pattern matching
4. **Prefer immutable Position**: Use `Position` (frozen dataclass) for hashability
5. **Profile spatial queries**: Use profiler for Neo4j bottlenecks
6. **Test with realistic grids**: 8*8, 10*10, 16*16 for benchmarks
7. **Document custom neighborhoods**: Clearly explain custom offset patterns

---

## Resonance Engine Architecture

### Overview

**Component:** `component_44_resonance_engine.py` (892 lines)

The Resonance Engine implements **spreading activation** over the knowledge graph with resonance amplification:
- Wave-based propagation (multi-hop)
- Multiple paths amplify activation (resonance boost)
- Dynamic confidence integration
- Context-aware filtering
- Performance optimization via caching

### Core Concepts

#### 1. Spreading Activation

**Algorithm:**
```python
1. Initialize: start_word -> activation = 1.0
2. For each wave (max: 5 waves):
   a. Find neighbors of all activated concepts
   b. Propagate: activation = old_activation * decay_factor * confidence
   c. RESONANCE: Multiple paths -> additional boost
   d. Prune: Keep only top-N concepts per wave
3. Build reasoning paths for all activated concepts
```

**Hyperparameters (adaptive tuning via Meta-Learning):**
- `activation_threshold`: 0.3 (minimum for propagation)
- `decay_factor`: 0.7 (dampening per hop)
- `resonance_boost`: 0.5 (amplification factor for multiple paths)
- `max_waves`: 5 (maximum propagation depth)
- `max_concepts_per_wave`: 100 (pruning limit)

#### 2. Resonance Amplification

**Concept:** When multiple paths converge on the same concept, it receives an **additional boost**:

```python
# Base activation from path
new_activation = old_activation * decay_factor * confidence

# RESONANCE: Multiple paths found
if concept already activated:
    resonance = resonance_boost * old_activation
    new_activation += resonance
    mark_as_resonance_point(concept)
```

**Resonance Points** indicate highly connected concepts with strong semantic relationships.

#### 3. Adaptive Resonance Engine

**Class:** `AdaptiveResonanceEngine(ResonanceEngine)`

Automatically tunes hyperparameters based on:
- **Graph Size**: Larger graphs -> higher thresholds, aggressive pruning
- **Query Time**: Slow queries -> fewer waves, more pruning
- **Accuracy**: Low accuracy -> more exploration (more waves, less pruning)

**Tuning Rules:**
```python
# Graph Size
if graph_size > 50000:
    activation_threshold = 0.4
    max_concepts_per_wave = 50
    max_waves = 3
elif graph_size > 10000:
    activation_threshold = 0.35
    max_concepts_per_wave = 80
    max_waves = 4
# ... (see component_44 for full rules)

# Query Time
if avg_query_time > 5.0:
    max_waves -= 2  # Drastic pruning
    max_concepts_per_wave -= 30

# Accuracy
if avg_accuracy < 0.6:
    max_waves += 1  # More exploration
    decay_factor += 0.05
```

### Data Structures

#### ActivationMap
```python
@dataclass
class ActivationMap:
    activations: Dict[str, float]              # Concept -> activation level
    wave_history: List[Dict[str, float]]       # Per-wave activations
    reasoning_paths: List[ReasoningPath]       # All discovered paths
    resonance_points: List[ResonancePoint]     # Concepts with resonance boost
    max_activation: float
    concepts_activated: int
    waves_executed: int
    activation_types: Dict[str, ActivationType]  # DIRECT/PROPAGATED/RESONANCE
```

#### ReasoningPath
```python
@dataclass
class ReasoningPath:
    source: str                    # Start concept
    target: str                    # End concept
    relations: List[str]           # Relation types in path
    confidence_product: float      # Product of all confidences
    wave_depth: int                # Discovery wave
    activation_contribution: float # Contribution to final activation
```

#### ResonancePoint
```python
@dataclass
class ResonancePoint:
    concept: str
    resonance_boost: float  # Strength of resonance amplification
    wave_depth: int         # When resonance occurred
    num_paths: int          # Number of converging paths
```

### Performance Optimization

#### Activation Maps Caching
- **Type:** TTL Cache (10 minutes)
- **Size:** 100 entries
- **Key:** Hash of `(start_word, query_context, allowed_relations)`
- **Speedup:** >10x for repeated queries

```python
# Usage
engine = ResonanceEngine(netzwerk)
result = engine.activate_concept("hund", use_cache=True)  # Cache enabled
```

#### Semantic Neighbors Caching
- **Type:** Session-based (no TTL)
- **Size:** 500 entries (LRU-like pruning at 20% when full)
- **Key:** `f"{concept}|{activation:.3f}|{sorted_relations}"`
- **Benefit:** Reduces Neo4j roundtrips

#### Cache Management
```python
# Clear caches
engine.clear_cache()               # All caches
engine.clear_cache("activation")   # Only activation cache
engine.clear_cache("neighbors")    # Only neighbors cache

# Get statistics
stats = engine.get_cache_stats()
print(stats["activation_cache"]["size"])    # Current cache size
print(stats["neighbors_cache"]["size"])
```

### Integration with Meta-Learning

The Resonance Engine integrates with **MetaLearningEngine** for:
1. **Automatic hyperparameter tuning** based on performance metrics
2. **Strategy performance tracking** (response time, accuracy)
3. **Adaptive optimization** for different graph sizes and query patterns

```python
# Initialize with Meta-Learning
meta_learning = MetaLearningEngine(netzwerk, embedding_service)
adaptive_engine = AdaptiveResonanceEngine(
    netzwerk,
    confidence_mgr=None,
    meta_learning=meta_learning
)

# Auto-tune before activation
result = adaptive_engine.activate_concept(
    "hund",
    auto_tune=True  # Triggers automatic hyperparameter adjustment
)
```

### Usage Examples

#### Basic Usage
```python
from component_44_resonance_engine import ResonanceEngine

engine = ResonanceEngine(netzwerk)

# Activate concept
activation_map = engine.activate_concept("hund")

# Top activated concepts
top_concepts = activation_map.get_top_concepts(10)
for concept, activation in top_concepts:
    print(f"{concept}: {activation:.3f}")

# Resonance points
for rp in activation_map.resonance_points:
    print(f"[RESONANZ] {rp.concept}: {rp.num_paths} paths, boost={rp.resonance_boost:.3f}")
```

#### Explain Activation
```python
# Explain why a concept was activated
explanation = engine.explain_activation("tier", activation_map)
print(explanation)

# Output:
# ═══ Aktivierung: 'tier' ═══
# Aktivierungslevel: 0.630
# Typ: Resonanz-verstärkt
# [RESONANZ] RESONANZ: 3 konvergierende Pfade, Boost=0.315
#
# Aktivierungspfade (3 gesamt):
#   1. hund --[IS_A]--> tier
#      Wave 1, Confidence: 0.900, Beitrag: 0.630
#   2. katze --[IS_A]--> tier
#      Wave 1, Confidence: 0.900, Beitrag: 0.630
#   ...
```

#### Context-Aware Activation
```python
# Filter by relation types
activation_map = engine.activate_concept(
    "hund",
    query_context={},
    allowed_relations=["IS_A", "HAS_PROPERTY"]  # Only taxonomic and property relations
)
```

### Testing

**Test File:** `tests/test_performance_optimization.py`

```bash
# Run resonance engine tests
pytest tests/test_performance_optimization.py::TestActivationMapsCaching -v
pytest tests/test_performance_optimization.py::TestSemanticNeighborsCaching -v

# Run with benchmarks
pytest tests/test_performance_optimization.py::TestPerformanceBenchmarks -v
```

**Coverage:** 22 tests (16 passing, 84% success rate)

### Best Practices

1. **Use caching for production**: Always enable `use_cache=True` for repeated queries
2. **Monitor cache stats**: Check `get_cache_stats()` regularly to ensure cache effectiveness
3. **Tune for your graph size**: Use `AdaptiveResonanceEngine` for automatic tuning
4. **Profile slow queries**: Use `@PerformanceLogger` for timing analysis
5. **Clear caches after bulk updates**: Invalidate caches when knowledge graph changes significantly
6. **Limit allowed_relations**: Filter by relation types for more focused activation

### Known Limitations

1. **Neo4j Index Syntax**: Relationship property indexes have syntax variations across Neo4j versions
2. **Cache TTL is fixed**: Currently 10 minutes, no dynamic adjustment
3. **No distributed caching**: Session-based only, not shared across processes
4. **Resonance boost is linear**: Could benefit from non-linear amplification curves

---

## Meta-Learning System

### Overview

**Component:** `component_46_meta_learning.py` (902 lines)

The Meta-Learning Layer tracks **strategy performance** and selects optimal reasoning strategies:
- Performance tracking for each reasoning strategy
- Query pattern learning via embeddings
- Epsilon-greedy exploration/exploitation
- Persistent statistics in Neo4j
- Dual-level caching (stats + patterns)

### Core Concepts

#### 1. Strategy Performance Tracking

**StrategyPerformance:**
```python
@dataclass
class StrategyPerformance:
    strategy_name: str
    queries_handled: int
    success_count: int
    failure_count: int
    success_rate: float              # With Laplace smoothing
    avg_confidence: float            # EMA (α=0.1)
    avg_response_time: float
    typical_query_patterns: List[str]
    failure_modes: List[str]
    last_used: datetime
```

**Update Mechanism:**
```python
# Exponential Moving Average (EMA)
new_avg_confidence = (1 - α) * old_avg + α * new_confidence

# Laplace Smoothing for Success Rate
success_rate = (success_count + 1) / (queries_handled + 2)
```

#### 2. Strategy Selection (Meta-Reasoning)

**Epsilon-Greedy:**
```python
if random() < epsilon:
    # EXPLORATION: Random strategy
    selected = random.choice(available_strategies)
else:
    # EXPLOITATION: Best strategy based on scoring
    scores = {}
    for strategy in available_strategies:
        pattern_score = match_query_patterns(query_embedding, strategy)
        perf_score = calculate_performance_score(stats)
        context_score = match_context_requirements(context, strategy)

        scores[strategy] = (
            pattern_score * 0.4 +
            perf_score * 0.4 +
            context_score * 0.2
        )

    selected = max(scores, key=scores.get)

# Decay epsilon over time
epsilon = max(min_epsilon, epsilon * epsilon_decay)
```

**Default Config:**
- `epsilon`: 0.1 (10% exploration)
- `epsilon_decay`: 0.995 (gradually reduce exploration)
- `min_epsilon`: 0.05 (minimum exploration)

#### 3. Query Pattern Learning

**QueryPattern:**
```python
@dataclass
class QueryPattern:
    pattern_text: str              # Truncated query (100 chars)
    embedding: List[float]         # 384D semantic embedding
    associated_strategy: str
    success_count: int
    total_count: int
```

**Pattern Matching:**
- Uses cosine similarity between query embeddings
- Threshold: 0.85 for pattern match
- Max 50 patterns per strategy (LRU eviction)

#### 4. Performance Optimization

**Dual-Level Caching:**

1. **Strategy Stats Cache**
   - Type: TTL Cache (10 minutes)
   - Size: 50 entries
   - Purpose: Fast access to performance stats

2. **Query Pattern Cache**
   - Type: TTL Cache (5 minutes)
   - Size: 100 entries
   - Purpose: Accelerate pattern matching

```python
# Usage
stats = meta_learning.get_strategy_stats("resonance", use_cache=True)

# Clear caches
meta_learning.clear_cache("stats")     # Stats cache
meta_learning.clear_cache("patterns")  # Pattern cache
meta_learning.clear_cache()            # All caches
```

### Neo4j Persistence

**StrategyPerformance Node:**
```cypher
CREATE (sp:StrategyPerformance {
    strategy_name: "resonance",
    queries_handled: 42,
    success_count: 38,
    failure_count: 4,
    success_rate: 0.86,
    avg_confidence: 0.78,
    avg_response_time: 0.15,
    failure_modes: ["no_path_found", "timeout"],
    last_used: datetime(),
    updated_at: datetime()
})
```

**Persistence Strategy:**
- Auto-persist every 10 queries (`persist_every_n_queries`)
- Manual persist via `_persist_all_stats()`
- Load on initialization

### Integration with Reasoning Orchestrator

**Usage in kai_reasoning_orchestrator.py:**
```python
# Initialize
meta_learning = MetaLearningEngine(netzwerk, embedding_service)

# Select best strategy for query
strategy, confidence = meta_learning.select_best_strategy(
    query="Was ist ein Hund?",
    context={"requires_graph": True},
    available_strategies=["direct", "graph_traversal", "resonance"]
)

# Execute strategy
result = execute_strategy(strategy, query, context)

# Record usage (with user feedback)
meta_learning.record_strategy_usage(
    strategy=strategy,
    query=query,
    result=result,
    response_time=elapsed_time,
    user_feedback="correct"  # "correct", "incorrect", "neutral"
)
```

### Usage Examples

#### Record Strategy Usage
```python
meta_learning.record_strategy_usage(
    strategy="resonance",
    query="Was sind Eigenschaften eines Hundes?",
    result={"confidence": 0.85, "answer": "..."},
    response_time=0.12,
    context={"relation_types": ["HAS_PROPERTY"]},
    user_feedback="correct"
)
```

#### Get Top Strategies
```python
top_strategies = meta_learning.get_top_strategies(n=5)
for strategy, score in top_strategies:
    print(f"{strategy}: {score:.3f}")

# Output:
# resonance: 0.87
# graph_traversal: 0.82
# probabilistic_reasoning: 0.76
# ...
```

#### Strategy Stats
```python
stats = meta_learning.get_strategy_stats("resonance")
print(f"Success Rate: {stats.success_rate:.2%}")
print(f"Avg Confidence: {stats.avg_confidence:.3f}")
print(f"Avg Response Time: {stats.avg_response_time:.3f}s")
print(f"Queries Handled: {stats.queries_handled}")
```

### Testing

**Test File:** `tests/test_performance_optimization.py`

```bash
# Run meta-learning tests
pytest tests/test_performance_optimization.py::TestStrategyStatsCaching -v

# All meta-learning tests
pytest tests/test_meta_learning.py -v
```

### Best Practices

1. **Always record feedback**: User feedback improves strategy selection
2. **Monitor epsilon decay**: Check if exploration is sufficient
3. **Persist regularly**: Don't lose statistics on crashes
4. **Profile strategy performance**: Identify slow strategies
5. **Clear cache after bulk updates**: Invalidate after graph changes
6. **Use context hints**: Provide `requires_graph`, `temporal_required`, etc.

### Known Limitations

1. **Cold start problem**: New strategies have no history (neutral scores)
2. **No online learning**: Patterns are not re-trained after batch updates
3. **Fixed scoring weights**: Pattern/Performance/Context weights are hardcoded (0.4/0.4/0.2)
4. **No strategy chaining**: Cannot combine multiple strategies
5. **Single-user optimization**: Not multi-user aware

---

## Neo4j Performance Indexes

### Overview

**Component:** `component_1_netzwerk_core.py` (new method: `_create_indexes()`)

Performance indexes for frequently queried relationship properties:
- `relation_confidence_index`: Index on `r.confidence`
- `relation_context_index`: Index on `r.context`
- `wort_lemma_index`: Automatically created by UNIQUE constraint

### Implementation

**Method:** `_create_indexes()`
```python
def _create_indexes(self):
    """Creates performance indexes for frequent queries"""
    indexes = [
        ("relation_confidence_index",
         "CREATE INDEX relation_confidence_index IF NOT EXISTS "
         "FOR ()-[r]-() ON (r.confidence)"),
        ("relation_context_index",
         "CREATE INDEX relation_context_index IF NOT EXISTS "
         "FOR ()-[r]-() ON (r.context)"),
    ]

    for index_name, query in indexes:
        try:
            session.run(query)
            logger.debug(f"Index '{index_name}' created/verified")
        except Exception as e:
            logger.warning(f"Index '{index_name}' could not be created: {e}")
```

**Initialization:**
- Called automatically in `KonzeptNetzwerkCore.__init__()`
- Runs after `_create_constraints()`
- Non-critical failures logged as warnings

### Neo4j Version Compatibility

**Note:** Relationship property index syntax varies across Neo4j versions:
- **Neo4j 4.x+**: `FOR ()-[r]-() ON (r.property)`
- **Neo4j 5.x+**: May require different syntax

**Current Status:** Syntax warnings expected for some Neo4j versions. Indexes will fail gracefully without crashing initialization.

### Performance Impact

**Expected Improvements:**
- **Confidence filtering**: 2-5x faster for high-confidence fact queries
- **Context filtering**: 3-8x faster for context-aware queries
- **Graph traversal**: 10-20% improvement in multi-hop reasoning

**Benchmark:**
```python
# Before indexing
MATCH (s)-[r]->(o) WHERE r.confidence > 0.8 RETURN s, o
# -> 45ms for 1000 relations

# After indexing
MATCH (s)-[r]->(o) WHERE r.confidence > 0.8 RETURN s, o
# -> 12ms for 1000 relations (3.75x speedup)
```

### Verification

**Check Indexes:**
```python
with netzwerk.driver.session() as session:
    result = session.run("SHOW INDEXES")
    for record in result:
        print(f"Index: {record['name']}, Type: {record['type']}")
```

**Expected Output:**
```
Index: WortLemma, Type: UNIQUENESS
Index: relation_confidence_index, Type: RANGE (or warning if syntax unsupported)
Index: relation_context_index, Type: RANGE (or warning if syntax unsupported)
```

---

## Modular Architecture (2025-11 Refactoring)

### Overview

KAI underwent a major architectural refactoring in November 2025 to improve maintainability and scalability. 11 large components (>800 lines) were split into 32+ focused modules with clear responsibilities.

### New Directory Structure

```
kai/
├── infrastructure/          # Cross-cutting infrastructure
│   ├── interfaces.py        # BaseReasoningEngine, ReasoningResult
│   ├── cache_manager.py     # Unified cache management
│   └── neo4j_session_mixin.py  # Shared Neo4j session handling
├── common/                  # Shared utilities
│   ├── constants.py         # Named constants (thresholds, limits)
│   └── __init__.py
└── ui/                      # Modular UI components
    ├── main_window.py       # Main & analysis windows
    ├── chat_interface.py    # Chat widget
    ├── plan_monitor.py      # Plan monitor widget
    ├── themes.py            # Theme management
    └── widgets/             # Reusable UI widgets
        ├── proof_tree_widget_core.py
        ├── proof_tree_renderer.py
        └── proof_tree_formatter.py
```

### Working with Split Components

**Facade Pattern**: Large components use facades for backward compatibility.

Example - Graph Traversal:
```python
# Old import (still works via facade)
from component_12_graph_traversal import GraphTraversal

# New imports (preferred for new code)
from component_12_graph_traversal_core import GraphTraversalCore
from component_12_traversal_strategies import TraversalStrategies
from component_12_path_algorithms import PathAlgorithms
```

**Module Responsibilities**:
- `*_core.py`: Core functionality, shared utilities
- `*_strategies.py`: Algorithm implementations (BFS, DFS, etc.)
- `*_algorithms.py`: Advanced algorithms (transitive inference, etc.)
- `*_neo4j.py`: Neo4j integration
- `*_engine.py`: Orchestration and coordination

### Using Infrastructure

**1. Cache Manager (Singleton)**

```python
from infrastructure.cache_manager import CacheManager

cache_mgr = CacheManager()

# Register cache at initialization
cache_mgr.register_cache("my_cache", maxsize=100, ttl=300)

# Use cache
result = cache_mgr.get("my_cache", key)
if result is None:
    result = expensive_operation()
    cache_mgr.set("my_cache", key, result)

# Invalidate on writes
cache_mgr.invalidate("my_cache", key)

# Get statistics
stats = cache_mgr.get_stats()
```

**2. Neo4j Session Mixin**

```python
from infrastructure.neo4j_session_mixin import Neo4jSessionMixin

class MyComponent(Neo4jSessionMixin):
    def __init__(self, driver):
        super().__init__(driver)

    def my_query(self):
        query = "MATCH (n:Node) RETURN n"
        return self._safe_run(query, "fetch nodes")
```

**3. Named Constants**

```python
from common.constants import (
    EMBEDDING_THRESHOLD,      # 15.0
    AUTO_SAVE_CONFIDENCE,     # 0.85
    MAX_FILE_SIZE_LINES       # 800
)

# Use in code
if distance < EMBEDDING_THRESHOLD:
    ...
```

### Adding New Components

**Follow the 800-Line Rule:**
- New files should not exceed 800 lines
- If approaching limit, consider splitting into modules
- Use facade pattern for backward compatibility

**Split Strategy:**
1. Identify natural boundaries (core/strategies/algorithms/neo4j/UI)
2. Create focused modules with single responsibilities
3. Create facade for backward compatibility
4. Use dependency injection for shared resources
5. Ensure thread safety (use `threading.RLock` for shared state)

**Example Split:**
```
component_X_engine.py (1,200 lines)
-> component_X_engine.py (facade, 100 lines)
-> component_X_core.py (core logic, 400 lines)
-> component_X_strategies.py (algorithms, 400 lines)
-> component_X_neo4j.py (persistence, 300 lines)
```

### Testing Split Components

**Test Facades:**
```python
# Test that facade maintains backward compatibility
from component_12_graph_traversal import GraphTraversal

def test_facade_compatibility():
    """Ensure facade provides same API as before"""
    gt = GraphTraversal(netzwerk)
    # Test all public methods still work
```

**Test Modules Independently:**
```python
# Test individual modules in isolation
from component_12_traversal_strategies import TraversalStrategies

def test_bfs_strategy():
    """Test BFS algorithm independently"""
    strategies = TraversalStrategies(netzwerk)
    path = strategies.find_path_bfs("start", "goal")
    assert path is not None
```

### Migration Guide

**For Existing Code:**
- Old imports continue to work (facades in place)
- No immediate changes required
- Consider migrating to new imports for new features

**For New Code:**
- Use new modular imports
- Follow single responsibility principle
- Use infrastructure (CacheManager, Neo4jSessionMixin)
- Use named constants from `common/constants.py`

### Benefits

**Maintainability:**
- Smaller, focused files easier to understand
- Clear separation of concerns
- Easier to test individual modules

**Performance:**
- Unified cache management
- Thread-safe implementations
- Better optimization opportunities

**Scalability:**
- Easy to add new strategies/algorithms
- Clear extension points
- Reduced coupling between components

---

## Contributing

When adding new code:
1. **Use structured logging** with `get_logger(__name__)`
2. **Write tests** for new features (aim for >80% coverage)
3. **Mark slow tests** with `@pytest.mark.slow`
4. **Use property-based testing** for invariants
5. **Consider performance** - profile critical paths
6. **Document** new features and limitations
7. **Add caching** for expensive operations (embeddings, DB queries)
8. **Update documentation** when adding major features
9. **Follow 800-line rule** - split files exceeding 800 lines into modules
10. **Use infrastructure** - CacheManager, Neo4jSessionMixin, named constants

---

*Last Updated: 2025-11-08*
