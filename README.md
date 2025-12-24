# KAI - Konzeptueller AI Prototyp

**Selbstlernende KI, die autonom Wissen aus Text erwirbt und transparent begruendet.**

---

## Vision

KAI ist eine deutschsprachige KI, die Sprache wie Menschen lernt: durch Mustererkennung, logisches Schlussfolgern und autonome Wissensextraktion. Das System kombiniert symbolische und statistische KI-Methoden mit dem Fokus auf **Erklaerbarkeit** und **Meta-Learning**.

### Kernprinzipien
- **Explainability First**: Jede Schlussfolgerung wird transparent mit Beweisbaeumen dargestellt
- **Bootstrapping**: Von manuellen Regeln ueber Mustererkennung zur autonomen Wissensextraktion
- **Kognitiv inspiriert**: Episodisches Gedaechtnis, Arbeitsspeicher, mehrstufiges Reasoning
- **Hybrid**: Symbolische Logik + statistische Embeddings

---

## Hauptfunktionen

### Reasoning-Engines
- **Multi-Hop Reasoning**: Transitive Relationen ueber Graph-Traversierung
- **Abductive Reasoning**: Hypothesengenerierung mit Template-/Analogie-/Kausal-Strategien
- **Probabilistisches Reasoning**: Bayessche Inferenz, Noisy-OR, Konfidenzpropagierung
- **Constraint-Reasoning**: CSP-Solver mit Backtracking, AC-3, MRV/LCV
- **SAT-Solver**: Boolesche Erfuellbarkeit mit DPLL-Algorithmus
- **Epistemisches Reasoning**: Multi-Agenten-Wissen, partielle Beobachtungen
- **Kombinatorisches Reasoning**: Permutationen, Zyklen, Strategiebewertung
- **Cognitive Resonance**: Spreading Activation mit Resonanz-Boost, Adaptive Hyperparameter-Tuning
- **Spatial Reasoning**: 2D-Grids, Path-Finding (BFS/DFS/A*), geometrische Formen
- **Arithmetic Reasoning**: Grundoperationen, Vergleiche, Primzahlen, Potenzen, Wurzeln, Brueche

### Lern-Systeme
- **Autonome Definitions-Erkennung**: Lernt deklarative Aussagen automatisch (Konfidenz >= 0.85)
- **Pattern Recognition**: 3-Ebenen-System (Tippfehler, Sequenzen, implizite Fakten)
- **Adaptive Thresholds**: Cold/Warming/Mature-Phasen fuer dynamische Lernschwellen
- **Prototype Matching**: Clustering mit 384D-Embeddings (Schwellwert 15.0)
- **Meta-Learning**: Automatische Strategy-Selection via Epsilon-Greedy, Performance-Tracking
- **Feedback Loop**: User-Feedback verbessert Strategie-Auswahl (Correct/Incorrect/Unsure)

### Wissensverarbeitung
- **Neo4j Knowledge Graph**: Knoten (Wort/Konzept, Episode, Hypothese), Relationen (IS_A, HAS_PROPERTY, CAPABLE_OF, PART_OF, LOCATED_IN)
- **Input Orchestration**: Intelligente Segmentierung fuer Logik-Raetsel (Erklaerungen -> Lernen, dann Fragen -> Reasoning)
- **Document Parsing**: PDF/DOCX-Unterstuetzung mit automatischer Faktenextraktion
- **Episodisches Gedaechtnis**: Zeitstempel, Provenance, Kontextverwaltung
- **Production System**: 54 Regeln (4 Kategorien), A/B Testing, ProofTree-Integration

### Visualisierung
- **Interactive Proof Trees**: Aufklappbare Beweisbaeume mit Reasoning-Steps (PySide6 UI)
- **Plan Monitor**: Live-Tracking von Sub-Goals und Strategien
- **Inner Picture Display**: Visualisierung interner Repraesentationen
- **Spatial Grid Widget**: Interaktive 2D-Grid-Visualisierung, Path-Display, Object Animation
- **Feedback Buttons**: UI-Buttons fuer sofortiges Feedback (OK/Falsch/Unsicher/Kommentar)
- **A/B Dashboard**: Vergleich von Production System vs. Pipeline-Antworten

---

## Voraussetzungen

- **Python**: 3.13.2 oder hoeher
- **Neo4j**: Graph-Datenbank (`bolt://127.0.0.1:7687`, Credentials: `neo4j/password`)
- **spaCy Modell**: `de_core_news_sm` (Deutsches NLP-Modell)

---

## Installation

### 1. Repository klonen
```bash
git clone https://github.com/RichardGeh/KAI.git
cd kai
```

### 2. Dependencies installieren
```bash
pip install -r requirements.txt
```

**Hauptabhaengigkeiten:**
- `neo4j >= 5.0.0` - Graph-Datenbank-Treiber
- `spacy >= 3.7.0` - NLP-Framework
- `PySide6 >= 6.6.0` - GUI (LGPL 3.0, siehe [LICENSES/LGPL-3.0.txt](LICENSES/LGPL-3.0.txt))
- `sentence-transformers >= 2.2.0` - 384D-Embeddings
- `pdfplumber`, `python-docx` - Dokumenten-Parser

### 3. spaCy-Modell herunterladen
```bash
python -m spacy download de_core_news_sm
```

### 4. Neo4j einrichten
- Neo4j installieren und starten
- Datenbank mit Default-Credentials konfigurieren: `neo4j/password`
- Verbindung pruefen: `bolt://127.0.0.1:7687`

### 5. Initiales Wissen laden (optional)
```bash
python setup_initial_knowledge.py
```

---

## Schnellstart

### GUI starten
```bash
python main_ui_graphical.py
```

### Erste Schritte

#### 1. Einfaches Lernen
```
Lerne: Ein Apfel ist eine Frucht
```

#### 2. Autonomes Lernen (ohne "Lerne:")
```
Ein Vogel ist ein Tier. Ein Vogel kann fliegen.
```
-> KAI erkennt automatisch Definitionen mit Konfidenz >= 0.85

#### 3. Fragen stellen
```
Was ist ein Apfel?
Kann ein Vogel fliegen?
```

#### 4. Musterlernen
```
Lerne Muster: "X schmeckt Y" bedeutet HAS_TASTE
Ein Apfel schmeckt suess
```
-> KAI lernt das Muster und wendet es auf neue Aussagen an

#### 5. Dokumente verarbeiten
```
Lese Datei: /pfad/zur/datei.pdf
Lese Datei: /pfad/zum/dokument.docx
```

#### 6. Logik-Raetsel (mit Input Orchestration)
```
Ein Pinguin ist ein Vogel. Ein Vogel kann fliegen. Ein Pinguin kann nicht fliegen.
Kann ein Pinguin fliegen?
```
-> KAI lernt zuerst die Erklaerungen, dann beantwortet es die Frage mit gelerntem Kontext

---

## Dokumentation

Die vollstaendige Dokumentation finden Sie im **[docs/](docs/)** Verzeichnis:

- **[USER_GUIDE.md](docs/USER_GUIDE.md)**: Benutzerhandbuch (Befehle, Einstellungen, Tipps, Troubleshooting)
- **[DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md)**: Entwicklerhandbuch (Logging, Testing, Performance, Implementierung)
- **[FEATURES_LEARNING.md](docs/FEATURES_LEARNING.md)**: Pattern Recognition, Adaptive Thresholds, Autonome Erkennung
- **[FEATURES_REASONING.md](docs/FEATURES_REASONING.md)**: Hybrid Reasoning, Multi-Strategie-Aggregation, Proof Trees
- **[PRODUCTION_SYSTEM.md](docs/PRODUCTION_SYSTEM.md)**: Production Rules, A/B Testing, ProofTree

### Fuer Entwickler
- **Code-Stil**: Black, isort, flake8, mypy mit pre-commit hooks (siehe `.pre-commit-config.yaml`)
- **Tests**: `pytest tests/ -v` (38+ Testdateien, >500 Tests)

---

## Architektur (Ueberblick)

```
User Input (PySide6)
  |
  v
Pattern Recognition -> Input Orchestrator (optional)
  |
  v
Linguistic Engine (spaCy) -> Meaning Extractor -> Goal Planner
  |
  v
KAI Worker -> Context/Sub-Goal/Inference/Ingestion Handlers
  |
  v
Knowledge Graph (Neo4j)
  |
  v
Response Formatter -> Proof Tree Generator -> Production System (A/B)
  |
  v
UI Update
```

---

## Tests ausfuehren

```bash
# Alle Tests
pytest tests/ -v

# Spezifische Test-Datei
pytest tests/test_kai_worker.py -v

# Einzelner Test
pytest tests/test_kai_worker.py::TestClass::test_method -v
```

**Hinweis**: Neue Testdateien in `settings_ui.py` fuer GUI-Discoverability hinzufuegen.

---

## Status

- **Version**: 0.0.01 (Alpha)
- **Python**: 3.13.2+
- **Letzte Updates (2025-12-23)**: Production System (54 Rules), A/B Testing, Arithmetic Reasoning, Pattern Discovery
- **Aktive Entwicklung**: [OK] Multi-Strategy Reasoning (Logic/Graph/Abductive/Probabilistic/Combinatorial/Spatial/Arithmetic), Memory Systems, Proof Trees, Pattern Recognition (3 Ebenen), Production System (A/B Testing), Performance Optimizations
- **In Entwicklung**: Temporal/Causal Reasoning, Dynamic Rule Generation

---

## Lizenz

Dieses Projekt ist unter der **Apache License 2.0** lizenziert - siehe [LICENSE](LICENSE) fuer Details.

### Third-Party Lizenzen

- **PySide6** (GUI-Framework): LGPL 3.0 - siehe [LICENSES/LGPL-3.0.txt](LICENSES/LGPL-3.0.txt) und [NOTICE](NOTICE)
- Weitere Dependencies: Apache 2.0, MIT - siehe [NOTICE](NOTICE)

**Wichtig**: PySide6 wird als Dependency verwendet (dynamische Verlinkung via pip). Sie koennen PySide6 unabhaengig ersetzen oder aktualisieren.

---

## Troubleshooting

**Haeufige Probleme:**

1. **Neo4j-Verbindung fehlgeschlagen**
   - Pruefen Sie, ob Neo4j laeuft: `bolt://127.0.0.1:7687`
   - Credentials: `neo4j/password`

2. **spaCy-Modell nicht gefunden**
   - `python -m spacy download de_core_news_sm`

3. **Extraction Rule funktioniert nicht**
   - Regel in Neo4j ueberpruefen
   - 2 Capture Groups im Regex
   - Text-Normalisierung beachten

Weitere Hilfe: [docs/USER_GUIDE.md - Troubleshooting](docs/USER_GUIDE.md)

---

## Neue Features

### Production System (Components 54_*)
- **54 Regeln**: 4 Kategorien (Content, Lexical, Discourse, Syntax)
- **Conflict Resolution**: Utility * Specificity Ranking
- **ProofTree Integration**: Vollstaendige Begruendungsketten
- **A/B Testing**: Vergleich Production System vs. Pipeline-Antworten

### Arithmetic Reasoning (Components 52/53)
- **Grundoperationen**: Addition, Subtraktion, Multiplikation, Division
- **Erweitert**: Brueche, Potenzen, Wurzeln, Logarithmen, Konstanten (pi, e, phi)
- **Vergleiche**: Groesser/Kleiner/Gleich mit Proof Trees
- **Zahlensprache**: Deutsche Zahlen (0-999) bidirektional mit Neo4j

### Cognitive Resonance (Component 44)
- **Spreading Activation**: Wellenfoermige Aktivierung ueber Knowledge Graph
- **Resonance Boost**: Multiple Pfade -> Verstaerkung zentraler Konzepte
- **Adaptive Tuning**: Automatische Hyperparameter-Anpassung basierend auf Graph-Groesse
- **Performance**: TTL Cache 10min, >10x Speedup fuer wiederholte Queries

### Meta-Learning (Component 46)
- **Strategy Performance Tracking**: Automatisches Tracking von Success Rate, Confidence, Response Time
- **Epsilon-Greedy Selection**: 10% Exploration, 90% Exploitation
- **Query Pattern Learning**: 384D-Embeddings fuer Pattern Matching
- **Neo4j Persistence**: Auto-Persist alle 10 Queries

### Performance Optimization
- **Activation Maps Cache**: TTL 10min, maxsize 100
- **Semantic Neighbors Cache**: Session-based, maxsize 500
- **Strategy Stats Cache**: Dual-Cache (Stats + Patterns)
- **Neo4j Indexes**: relation_confidence, relation_context (auto-created)

---

*Letzte Aktualisierung: 2025-12-23* 
