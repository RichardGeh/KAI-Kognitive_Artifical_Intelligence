# KAI - Benutzerhandbuch

**Version:** 1.0
**Zielgruppe:** End-User, die KAI nutzen möchten

---

## Inhaltsverzeichnis

1. [Was ist KAI?](#was-ist-kai)
2. [Grundlegende Befehle](#grundlegende-befehle)
3. [Hauptfähigkeiten](#hauptfähigkeiten)
4. [Relationtypen](#relationtypen)
5. [System starten](#system-starten)
6. [Einstellungen & Konfiguration](#einstellungen--konfiguration)
7. [Performance & Caching](#performance--caching)
8. [Tipps & Best Practices](#tipps--best-practices)
9. [Technische Details](#technische-details)

---

## Was ist KAI?

KAI (Konzeptueller AI Prototyp) ist ein lernfähiges KI-System, das Wissen in einem Wissensgraphen speichert und auf Fragen antworten kann. KAI lernt durch Beispiele und kann neue Sprachmuster erkennen.

---

## Grundlegende Befehle

### 1. Einfach lernen (NEU)
```
Lerne: <Wort oder Satz>
```
**Die einfachste Art, KAI etwas beizubringen!**

**Beispiele:**
```
Lerne: Ein Apfel ist eine Frucht
Lerne: Katzen können miauen
Lerne: Berlin liegt in Deutschland
Lerne: Elefant
```

**Was passiert:**
- Bei vollständigen Sätzen: KAI extrahiert automatisch Fakten und Relationen
- Bei einzelnen Wörtern: KAI speichert das Wort als neues Konzept
- Kein spezielles Format nötig - einfach natürlich formulieren!

### 2. Wissen definieren
```
Definiere: <Wort> / <Typ> = <Information>
```
**Beispiel:**
```
Definiere: Apfel / bedeutung = Eine runde Frucht
```

### 3. Muster lehren
```
Lerne Muster: "<Beispielsatz>" bedeutet <RELATION_TYP>
```
**Beispiel:**
```
Lerne Muster: "Ein Hund ist ein Tier" bedeutet IS_A
```

### 4. Text ingestieren
```
Ingestiere Text: "<Text mit mehreren Sätzen>"
```
**Beispiel:**
```
Ingestiere Text: "Ein Vogel kann fliegen. Eine Katze ist ein Haustier."
```

### 5. Dateien einlesen (NEU)
```
Lese Datei: <Pfad zur Datei>
```
**KAI kann jetzt Wissen direkt aus Dokumenten lernen!**

**Beispiele:**
```
Lese Datei: C:\Dokumente\bericht.pdf
Lese Datei: .\notizen.docx
Lese Datei: /home/user/dokumente/artikel.pdf
```

**Was passiert:**
- KAI extrahiert automatisch den Text aus der Datei
- Der Text wird analysiert und verarbeitet (wie "Ingestiere Text:")
- Fakten und Relationen werden in den Wissensgraphen übernommen
- Fortschrittsanzeige mit Zeichenanzahl und Status

**Unterstützte Formate:**
- **PDF** (.pdf): Mehrseitige Dokumente, Layout-aware Extraktion
- **Microsoft Word** (.docx): Absatzbasierte Extraktion

**Hinweise:**
- Absolute oder relative Pfade werden unterstützt
- Dateigröße unbegrenzt (große Dateien werden in Chunks verarbeitet)
- Bei geschützten PDFs: Passwortschutz wird nicht unterstützt
- OCR für gescannte Dokumente: Nicht integriert (Text muss bereits im Dokument vorhanden sein)

### 6. Fragen stellen
Einfache Fragen in natürlicher Sprache:
```
Was ist ein Apfel?
Wo liegt Berlin?
```

---

## Hauptfähigkeiten

### Wissensspeicherung
- Speichert Fakten in einem Neo4j-Graphen
- Erkennt Relationen: IS_A, HAS_PROPERTY, CAPABLE_OF, PART_OF, LOCATED_IN
- Entfernt automatisch Artikel und normalisiert Text

### Mustererkennung
- Lernt Satzmuster aus Beispielen
- Verwendet Vektoren für ähnliche Sätze (384D-Embeddings)
- Clustert ähnliche Muster automatisch

### Fragenbeantwortung
- Durchsucht Wissensgraph nach relevantem Wissen
- Unterstützt Multi-Hop-Reasoning (Verkettung mehrerer Fakten)
- Fragt bei Wissenslücken nach Beispielen
- Gibt strukturierte Antworten mit Begründung

### Kontextverständnis
- Working Memory für Gesprächskontext
- Merkt sich, wenn Wissen fehlt
- Erwartet Beispielsätze als Folge-Eingabe
- Lernt aus dem Dialog

### Beweisbaum-Visualisierung
- **Interaktive Visualisierung**: Zeigt den kompletten Denkprozess von KAI als Baumstruktur
- **Drei Reasoning-Typen**:
  - Graph-Traversal: Mehrschrittige Pfade durch den Wissensgraphen
  - Logic Engine: Regelbasierte Schlussfolgerungen mit Beweisketten
  - Abductive Reasoning: Hypothesen-Generierung bei unvollständigem Wissen
- **Visuelle Features**:
  - Farbkodierung nach Konfidenz: Grün (hoch), Gelb (mittel), Rot (niedrig)
  - Verschiedene Formen: Rechtecke (Fakten), Rauten (Regeln), Kreise (Hypothesen)
  - Interaktive Navigation: Expandieren/Kollabieren, Pfad-Highlighting, Tooltips
- **Export-Funktionen**: Speichern als JSON oder Bild (PNG/PDF)
- **Zugriff**: Automatischer Wechsel zum "Beweisbaum"-Tab bei komplexen Schlussfolgerungen

### Dateiverarbeitung (NEU)
- **Unterstützte Formate**: PDF (.pdf), Microsoft Word (.docx)
- **Automatische Parser-Auswahl**: Format wird anhand der Dateierweiterung erkannt
- **Robuste Extraktion**:
  - PDF: Layout-aware Textextraktion Seite für Seite (pdfplumber)
  - DOCX: Absatzbasierte Extraktion mit Formatierung (python-docx)
- **Intelligente Verarbeitung**:
  - Großer Text wird automatisch in Chunks verarbeitet (siehe Performance-Optimierungen)
  - Batch-Embeddings für schnellere Verarbeitung
  - Progress-Tracking in UI
- **Fehlerbehandlung**:
  - Klare Fehlermeldungen bei nicht unterstützten Formaten
  - Validierung von Dateipfaden und Zugriffsrechten
  - Graceful Degradation bei partiell beschädigten Dokumenten
- **Workflow**: `Lese Datei: path/to/file.pdf` → Text-Extraktion → Automatische Ingestion → Wissensgraph-Update

---

### Räumliches Reasoning (NEU - Spatial Reasoning)

KAI kann jetzt räumliche Beziehungen verstehen, lernen und darauf schließen! Das System unterstützt 2D-Grids, geometrische Formen und räumliche Relationen.

**Was KAI kann:**
- Grids erstellen und verwalten (Schachbretter, Sudoku, beliebige N×M Grids)
- Objekte auf Grids platzieren und bewegen
- Räumliche Beziehungen lernen (nördlich von, benachbart zu, enthalten in, etc.)
- Pfade finden (kürzeste Wege, Hindernisse umgehen)
- Räumliche Muster erkennen (z.B. Schachmatt-Positionen)
- Transitive räumliche Schlussfolgerungen ziehen

**Wichtig:** KAI hat **keine vorgefertigten Regeln** für Schach, Sudoku oder andere Spiele! Diese Anwendungen werden via Lernen und Regeln gelehrt.

#### Räumliche Befehle

**1. Grid erstellen und Objekte platzieren**
```
Lerne: Erstelle ein 8×8 Grid namens "schachbrett"
Lerne: Der weiße König steht auf Position (4, 0) auf dem schachbrett
Lerne: Die weiße Dame steht auf Position (3, 0) auf dem schachbrett
```

**2. Räumliche Relationen lernen**
```
Lerne: Der König ist benachbart zur Dame
Lerne: Berlin liegt nördlich von München
Lerne: Der Turm ist in der Ecke
```

**3. Fragen über räumliche Beziehungen**
```
Wo steht der König?
Ist der König benachbart zur Dame?
Was liegt nördlich von München?
Welche Objekte sind auf dem schachbrett?
```

**4. Pfadsuche (für fortgeschrittene Nutzer)**
```
Finde den kürzesten Weg von (0,0) nach (7,7) auf dem schachbrett
Gibt es einen Pfad um die Hindernisse herum?
```

**5. Räumliche Muster lernen und erkennen**
```
Lerne: Diese Position ist ein "Schachmatt" (speichert Konfiguration)
Erkenne Muster auf dem aktuellen schachbrett
```

#### Räumliche Relationen

**Richtungsbezogen** (Transitiv - A nördlich von B, B nördlich von C → A nördlich von C):
- `NORTH_OF` (nördlich von)
- `SOUTH_OF` (südlich von)
- `EAST_OF` (östlich von)
- `WEST_OF` (westlich von)

**Nachbarschaft** (Symmetrisch - A benachbart zu B ⇔ B benachbart zu A):
- `ADJACENT_TO` (benachbart zu)
- `NEIGHBOR_ORTHOGONAL` (orthogonaler Nachbar, 4-Richtungen)
- `NEIGHBOR_DIAGONAL` (diagonaler Nachbar)

**Hierarchisch** (Transitiv):
- `INSIDE` (innerhalb von)
- `CONTAINS` (enthält)
- `ABOVE` (oberhalb von)
- `BELOW` (unterhalb von)

**Positional**:
- `BETWEEN` (zwischen)
- `LOCATED_AT` (befindet sich an)

#### Beispiele

**Beispiel 1: Schach-ähnliches Grid**
```
Benutzer: Lerne: Erstelle ein 8×8 Grid "chess"
KAI: Grid "chess" erstellt (8×8, orthogonale Nachbarschaft)

Benutzer: Lerne: Der König steht auf (4, 0) auf chess
KAI: König platziert auf Position (4, 0)

Benutzer: Lerne: Die Dame steht auf (3, 0) auf chess
KAI: Dame platziert auf Position (3, 0)

Benutzer: Wo steht der König?
KAI: Der König steht auf Position (4, 0) auf Grid "chess"

Benutzer: Ist der König benachbart zur Dame?
KAI: Ja, der König (4,0) ist benachbart zur Dame (3,0) - Distanz: 1
```

**Beispiel 2: Geografische Relationen**
```
Benutzer: Lerne: Hamburg liegt nördlich von München
KAI: Räumliche Relation gespeichert: Hamburg NORTH_OF München

Benutzer: Lerne: München liegt nördlich von Zürich
KAI: Räumliche Relation gespeichert: München NORTH_OF Zürich

Benutzer: Liegt Hamburg nördlich von Zürich?
KAI: Ja (transitive Schlussfolgerung):
    Hamburg NORTH_OF München (direkt)
    München NORTH_OF Zürich (direkt)
    → Hamburg NORTH_OF Zürich (transitiv, Konfidenz: 0.85)
```

**Beispiel 3: Räumliche Muster**
```
Benutzer: Lerne: Speichere aktuelle chess-Position als "backrank_checkmate"
KAI: Räumliche Konfiguration "backrank_checkmate" gespeichert
     - König: (4, 0)
     - Dame: (4, 1)
     - Turm: (3, 0)

[Später, mit ähnlicher Position]
Benutzer: Welche Muster erkenne ich auf chess?
KAI: Erkannte Muster: "backrank_checkmate" (Übereinstimmung: 95%)
```

**Beispiel 4: Pfadsuche mit Hindernissen**
```
Benutzer: Lerne: Erstelle ein 10×10 Grid "labyrinth"
KAI: Grid "labyrinth" erstellt

Benutzer: Lerne: Hindernisse auf (3,3), (3,4), (3,5), (3,6)
KAI: Hindernisse platziert

Benutzer: Finde Weg von (0,0) nach (9,9) auf labyrinth
KAI: Pfad gefunden (12 Schritte, Algorithmus: BFS):
     (0,0) → (1,0) → (2,0) → (2,1) → ... → (9,9)
     [Visualisierung im Spatial Grid Widget]
```

#### Visualisierung

KAI bietet ein **interaktives Grid-Visualisierungs-Widget** im "Spatial Grid" Tab:

**Features:**
- Grafische Darstellung von Grids (8×8, 10×10, beliebige Größen)
- Objekte werden als farbige Zellen angezeigt
- Pfade werden als Linien visualisiert
- Animierte Objektbewegungen
- Zoom und Pan für große Grids
- Export zu Bild-Datei

**Nutzung:**
1. Öffne den "Spatial Grid" Tab in der UI
2. Grid wird automatisch aktualisiert, wenn du räumliche Befehle verwendest
3. Markierte Positionen werden farbig hervorgehoben
4. Pfade werden als durchgezogene Linie angezeigt

**Farben:**
- **Grün**: Start-Position
- **Rot**: Ziel-Position
- **Grau**: Hindernisse
- **Blau**: Pfad
- **Gelb**: Markierte Objekte

#### Geometrische Formen

KAI kann auch mit geometrischen Formen arbeiten:

**Dreiecke:**
```
Lerne: Dreieck mit Eckpunkten (0,0), (4,0), (2,3)
Berechne Fläche des Dreiecks → 6.0 (Heron's Formel)
```

**Vierecke:**
```
Lerne: Rechteck mit Eckpunkten (0,0), (4,0), (4,3), (0,3)
Ist das ein Rechteck? → Ja
Berechne Fläche → 12.0 (Shoelace-Formel)
```

**Kreise:**
```
Lerne: Kreis mit Zentrum (5,5) und Radius 3
Berechne Fläche → 28.27 (πr²)
Ist Punkt (6,6) im Kreis? → Ja
```

#### Tipps für räumliches Reasoning

1. **Verwende klare Positionsangaben**: `(x, y)` Format ist eindeutig
2. **Benenne Grids eindeutig**: "schachbrett_1", "sudoku_easy", etc.
3. **Nutze transitives Reasoning**: KAI schlussfolgert automatisch aus Ketten
4. **Speichere Muster**: Häufige Konfigurationen können gespeichert werden
5. **Visualisiere**: Nutze das Spatial Grid Widget für visuelle Überprüfung
6. **Experimentiere**: Keine Hardcoding - alles wird gelernt!

#### Performance

**Grid-Operationen:**
- Grid erstellen (8×8): <1ms
- Objekt platzieren: 2-5ms (Neo4j)
- Position abfragen: <1ms (mit Cache: 10-20x schneller)

**Pfadsuche:**
- BFS (8×8 Grid, leer): 2-5ms
- A* (8×8 Grid, mit Hindernissen): 8-15ms
- Große Grids (16×16): 20-50ms

**Muster-Erkennung:**
- Muster speichern: 10-30ms
- Muster erkennen: 20-50ms (abhängig von Anzahl gespeicherter Muster)

**Optimierungen:**
- Caching für Position-Queries (5 Min TTL)
- Batch-Operationen für mehrere Objekte
- Lazy Loading für große Grids

---

## Relationtypen

| Typ | Bedeutung | Beispiel |
|-----|-----------|----------|
| IS_A | "ist ein/eine" | Hund IS_A Tier |
| HAS_PROPERTY | "hat Eigenschaft" | Apfel HAS_PROPERTY rot |
| CAPABLE_OF | "kann" | Vogel CAPABLE_OF fliegen |
| PART_OF | "ist Teil von" / Synonym | Rad PART_OF Auto |
| LOCATED_IN | "befindet sich in" | Berlin LOCATED_IN Deutschland |

---

## System starten

```bash
# Hauptanwendung mit grafischer Oberfläche
python main_ui_graphical.py

# Initiales Setup (einmalig)
python setup_initial_knowledge.py

# Tests ausführen
pytest test_kai_worker.py -v
```

---

## Einstellungen & Konfiguration

KAI verfügt über ein umfassendes Settings-System mit folgenden Features:
- **GUI-basierte Konfiguration** über Settings-Dialog (Menü → Einstellungen)
- **Persistente Speicherung** in `kai_config.json`
- **7 Settings-Tabs** für verschiedene Aspekte
- **Echtzeit-Aktualisierung** für Theme-Änderungen
- **Automatische Migration** neuer Einstellungen

### Settings-Tabs

#### 1. KAI-Einstellungen
**Wortverwendungs-Tracking & Kontext-Extraktion**

Einstellungen:
- `word_usage_tracking` (bool): Automatisch Wortverwendungen speichern
- `usage_similarity_threshold` (0-100%): Ähnlichkeits-Schwellenwert für Kontext-Fragmente
- `context_window_size` (1-5): ±N Wörter für Kontext-Fenster
- `max_words_to_comma` (2-6): Max. Wörter bis Komma

**Verwendung:**
- Speichert Kontext-Fragmente für jedes Wort
- Hilft KAI, typische Wortverbindungen zu lernen
- Ermöglicht spätere Mustererkennung

#### 2. Neo4j
**Datenbank-Verbindungsparameter**

Einstellungen:
- `neo4j_uri` (string): Verbindungs-URI (z.B. `bolt://127.0.0.1:7687`)
- `neo4j_user` (string): Benutzername (Standard: `neo4j`)
- `neo4j_password` (string): Passwort (Standard: `password`)

**Wichtig:**
- Änderungen werden erst nach **Neustart von KAI** aktiv
- Neo4j muss laufen und erreichbar sein
- Standard-Port: 7687 (bolt protocol)

#### 3. Konfidenz
**Konfidenz-Schwellenwerte für automatische Entscheidungen**

Einstellungen:
- `confidence_low_threshold` (0.0-1.0): Schwellenwert für Klärungsbedarf (Standard: 0.40)
- `confidence_medium_threshold` (0.0-1.0): Schwellenwert für Bestätigung (Standard: 0.85)

**Verhalten:**
- **< Low Threshold**: KAI fragt nach, was gemeint ist (Klärungsfrage)
- **< Medium Threshold**: KAI bittet um Bestätigung
- **≥ Medium Threshold**: KAI führt Aktion direkt aus

**Beispiel:**
```
Eingabe: "Ein Hund ist ein Tier"

Konfidenz: 0.92 → Direkte Ausführung (≥ 0.85)
Konfidenz: 0.78 → Bestätigungsanfrage (< 0.85)
Konfidenz: 0.35 → Klärungsfrage (< 0.40)
```

#### 4. Muster
**Pattern-Matching Schwellenwerte**

**Prototype Novelty Threshold:**
- `prototype_novelty_threshold` (5.0-30.0): Euklidische Distanz in 384D semantischem Raum
- Standard: 15.0
- **< Threshold**: Update existierenden Prototyp
- **≥ Threshold**: Erstelle neuen Prototyp

**Typo Detection (Adaptive):**
- `typo_min_threshold` (1-10): Minimum Wort-Vorkommen (Standard: 3)
- `typo_max_threshold` (5-20): Maximum Wort-Vorkommen (Standard: 10)
- Formel: `min(MAX, max(MIN, vocab_size^0.4))`

**Sequence Prediction (Adaptive):**
- `sequence_min_threshold` (1-5): Minimum Sequenz-Vorkommen (Standard: 2)
- `sequence_max_threshold` (3-10): Maximum Sequenz-Vorkommen (Standard: 5)
- Formel: `min(MAX, max(MIN, connection_count^0.35))`

#### 5. Darstellung
**Theme & UI-Einstellungen**

Einstellungen:
- `theme` (string): "dark" oder "light"

**Features:**
- **Echtzeit-Wechsel**: Theme wird sofort nach "Anwenden" aktiv
- **Vorschau**: Live-Vorschau im Settings-Dialog
- **Persistenz**: Gespeichert für nächsten Start

**Themes:**

**Dark Mode** (Standard):
```
Background: #34495e
Text: #ecf0f1
Main Window: #2c3e50
Inputs: #2c3e50
```

**Light Mode:**
```
Background: #ecf0f1
Text: #2c3e50
Main Window: #bdc3c7
Inputs: #ffffff
```

#### 6. Logging
**Log-Level & Performance-Logging**

Einstellungen:
- `console_log_level`: DEBUG, INFO, WARNING, ERROR, CRITICAL
- `file_log_level`: DEBUG, INFO, WARNING, ERROR, CRITICAL
- `performance_logging` (bool): Performance-Metriken loggen

**Log-Dateien:**
- `logs/kai.log`: Haupt-Log
- `logs/error.log`: Nur Fehler
- `logs/performance.log`: Performance-Metriken

#### 7. Tests
**Test-Runner mit Live-Progress**

Features:
- Auswahl einzelner Test-Klassen oder Dateien
- Live-Ausgabe mit Fortschrittsbalken
- Anzeige fehlgeschlagener Tests
- Start/Stop-Steuerung

---

## Performance & Caching

KAI nutzt intelligentes Caching für deutlich schnellere Response-Zeiten:

### Automatische Caching-Strategien

1. **Embedding-Cache**
   - Speichert berechnete Text-Vektoren (bis zu 1000 Einträge)
   - **Speedup**: Bis zu 5600x schneller für wiederholte Texte
   - Nutzt LRU-Strategie (Least Recently Used)

2. **Query-Cache**
   - Cached Datenbank-Abfragen (5-10 Minuten TTL)
   - **Speedup**: Bis zu 86x schneller für häufige Queries
   - Automatische Invalidierung bei Datenänderungen

3. **Prototyp-Match-Cache**
   - Session-Cache für Pattern-Matching
   - **Speedup**: Bis zu 52x schneller
   - Verhindert redundante Datenbank-Zugriffe

4. **Extraktionsregeln-Cache**
   - Cached Extraktionsregeln (10 Minuten TTL)
   - **Speedup**: Bis zu 39x schneller
   - Beschleunigt Text-Ingestion erheblich

### Messbarer Nutzen

**30-50% schnellere Response-Zeiten** für:
- Wiederkehrende Fragen
- Text-Verarbeitung mit bekannten Mustern
- Fakten-Abfragen über bereits gecachte Konzepte
- Pattern-Matching bei Musterlernen

### Cache-Verhalten

- **Transparent**: Caching erfolgt automatisch im Hintergrund
- **Intelligent**: TTL-basierte Invalidierung (Time-To-Live)
- **Konsistent**: Automatische Cache-Invalidierung bei Datenänderungen
- **Sicher**: Graceful Degradation bei Cache-Fehlern

**Keine Konfiguration nötig** - KAI optimiert sich selbst!

### Neue Performance-Optimierungen (2025-10-26)

**Batch-Embeddings:**
- Text-Verarbeitung 5-10x schneller für große Texte (>100 Sätze)
- Automatisch aktiv in Text-Ingestion

**Lazy Loading:**
- Proof Tree Widget rendert große Bäume progressiv
- 3x schneller, keine UI-Freezes bei großen Beweisbäumen

**Batch-Processing:**
- Große Texte (>10000 Sätze) werden in Chunks verarbeitet
- Skaliert zu sehr großen Texten ohne Out-of-Memory

---

## Tipps & Best Practices

1. **Einfach anfangen**: Nutze `Lerne: ...` für schnelles, unkompliziertes Lernen
2. **Explizite Befehle**: Alle Befehle (z.B. "Lerne:", "Definiere:", "Lerne Muster:") haben höchste Priorität
3. **Normalisierung**: KAI entfernt Artikel automatisch ("der Hund" → "hund")
4. **Groß-/Kleinschreibung**: KAI speichert alles in Kleinbuchstaben
5. **Wissenslücken**: Bei unbekannten Begriffen fragt KAI automatisch nach
6. **Setup**: Vor der ersten Nutzung `setup_initial_knowledge.py` ausführen
7. **Beweisbaum**: Nutze den "Beweisbaum"-Tab für Einblicke in KAIs Denkprozess

### Beweisbaum-Visualisierung nutzen

**Wann erscheint der Beweisbaum?**
Der Beweisbaum wird automatisch angezeigt, wenn KAI komplexe Schlussfolgerungen durchführt:
- **Multi-Hop-Reasoning**: "Was ist ein Hund?" (wenn mehrere Schritte nötig sind)
- **Regelbasierte Inferenz**: Wenn logische Regeln angewendet werden
- **Hypothesen-Generierung**: Wenn KAI Vermutungen über unbekannte Begriffe anstellt

**Interaktion mit dem Beweisbaum:**
1. **Navigation**: Klicke auf Knoten mit Kindern, um Unterbäume ein-/auszublenden
2. **Pfad-Highlighting**: Klicke auf einen Knoten, um den Pfad zur Wurzel zu markieren
3. **Details ansehen**: Fahre mit der Maus über Knoten für vollständige Erklärungen
4. **Zoom**: Nutze +/- Buttons oder "Fit" für automatische Skalierung
5. **Export**: Speichere den Beweisbaum als JSON (Datenstruktur) oder Bild (Visualisierung)

**Symbole und Farben:**

**Knotenformen:**
- Rechteck: Fakten, Inferenzen
- Raute: Regeln, Regelanwendungen
- Kreis: Hypothesen, probabilistische Schlüsse

**Farben (Konfidenz):**
- Grün: Hohe Sicherheit (≥80%)
- Gelb: Mittlere Sicherheit (50-80%)
- Rot: Niedrige Sicherheit (<50%)

---

## Technische Details

- **Datenbank**: Neo4j (bolt://127.0.0.1:7687)
- **NLP-Modell**: spaCy `de_core_news_sm`
- **Embedding-Modell**: `paraphrase-multilingual-MiniLM-L12-v2`
- **Python-Version**: 3.13.2
- **UI-Framework**: PySide6
- **Caching**: LRU-Cache + TTL-Cache (cachetools)

---

## Troubleshooting

### Problem: Config-Datei wird nicht erstellt
**Lösung:** Stelle sicher, dass das Projektverzeichnis schreibbar ist.

### Problem: Änderungen werden nicht übernommen
**Lösung:**
1. Überprüfe, ob "Anwenden" geklickt wurde
2. Für Neo4j: Neustart erforderlich
3. Prüfe `kai_config.json` manuell

### Problem: Falsche Defaults
**Lösung:**
1. Lösche `kai_config.json`
2. Starte KAI neu (erzeugt neue Config mit aktuellen Defaults)

### Problem: Settings-Dialog öffnet sich nicht
**Lösung:**
```bash
# Test standalone
python settings_ui.py

# Check logs
cat logs/error.log
```

### Problem: Langsame Performance
**Lösung:**
- Prüfe Neo4j-Verbindung
- Cache-Statistiken überprüfen
- Bei großen Texten: Nutze Text-Ingestion in Chunks
- Für große Beweisbäume: Progressive Rendering aktivieren

### Problem: "Lese Datei:" funktioniert nicht
**Lösung:**
1. Überprüfe den Dateipfad (absolute oder relative Pfade unterstützt)
2. Stelle sicher, dass die Datei existiert und lesbar ist
3. Prüfe die Dateierweiterung (.pdf oder .docx)
4. Für PDFs: pdfplumber muss installiert sein (`pip install pdfplumber`)
5. Für DOCX: python-docx muss installiert sein (`pip install python-docx`)
6. Bei Fehlern: Prüfe `logs/kai.log` für detaillierte Fehlermeldungen

### Problem: PDF wird nicht korrekt geparst
**Lösung:**
- **Gescannte PDFs**: Text muss bereits extrahierbar sein (OCR nicht integriert)
- **Geschützte PDFs**: Passwortschutz wird nicht unterstützt
- **Komplexe Layouts**: pdfplumber versucht Layout-Awareness, aber komplexe Tabellen/Spalten können Probleme bereiten
- **Bilder im PDF**: Werden ignoriert, nur Text wird extrahiert

### Problem: DOCX-Datei leer oder unvollständig
**Lösung:**
- Überprüfe, ob die Datei wirklich Text enthält (nicht nur Bilder)
- Textboxen und eingebettete Objekte werden möglicherweise nicht extrahiert
- Fußnoten und Kopfzeilen werden aktuell nicht extrahiert
- Tabellen-Inhalte werden zeilenweise extrahiert

---

## Neue UI-Features (v2.1)

### Feedback-Buttons

**Location**: Hauptfenster, unter Antworttext

KAI lernt jetzt aus deinem Feedback! Nach jeder Antwort kannst du bewerten:

**Buttons:**
- ✅ **Correct**: Antwort war richtig
- ❌ **Incorrect**: Antwort war falsch
- ❓ **Unsure**: Nicht sicher / Teilweise richtig
- 💬 **Custom Feedback**: Detailliertes Textfeedback

**Was passiert:**
1. Feedback wird mit Query und verwendeter Strategy gespeichert
2. **Meta-Learning** lernt, welche Strategy für welche Fragen am besten funktioniert
3. **Success Rate** der Strategien wird aktualisiert
4. Zukünftige Queries verwenden bessere Strategien

**Integration:**
- Feedback-Daten persistent in Neo4j gespeichert
- Real-time Strategy Performance Updates
- Epsilon-Greedy Exploration (10% testing neuer Strategien)

**Beispiel:**
```
User: "Was ist ein Hund?"
KAI: "Ein Hund ist ein Tier..." (Strategy: resonance)
User: [Klickt ✅ Correct]

→ Strategy "resonance" erhält +1 success_count
→ Success Rate steigt von 85% → 86%
→ Zukünftige ähnliche Fragen bevorzugen "resonance"
```

### Resonance View (geplant)

**Status**: 🔜 Kommend in v2.2

Visualisierung der Cognitive Resonance Aktivierung:

**Geplante Features:**
- **Activation Wave Animation**: Zeigt, wie Aktivierung sich ausbreitet
- **Resonance Points Highlighting**: Markiert zentrale Konzepte (⭐)
- **Reasoning Path Display**: Visualisiert multiple Pfade
- **Interactive Exploration**: Klicke auf Konzepte für Details
- **Wave-by-Wave Replay**: Schrittweise Ansicht der Ausbreitung

**Mockup:**
```
┌─────────────────────────────────────┐
│ Resonance View                      │
├─────────────────────────────────────┤
│  Wave 0:  hund [1.000] (START)      │
│  Wave 1:  tier [0.630] ⭐ (3 paths) │
│           säugetier [0.490]         │
│           haustier [0.441] ⭐        │
│  Wave 2:  lebewesen [0.315]         │
│           fleischfresser [0.343]    │
│                                     │
│  [Play] [Pause] [Step] [Reset]     │
└─────────────────────────────────────┘
```

---

## Performance-Tipps (v2.1)

### Caching nutzen

KAI verwendet jetzt **automatisches Caching** für bessere Performance:

**Activation Maps Cache**:
- TTL: 10 Minuten
- Speedup: >10x für wiederholte Queries
- Automatisch aktiviert bei Resonance Reasoning

**Strategy Stats Cache**:
- TTL: 10 Minuten
- Speedup: ~2x für Strategy-Selection
- Automatisch aktiviert bei Meta-Learning

**Tipp**: WiederkehrendeQueries profitieren massiv vom Caching!

### Hyperparameter-Tuning

**AdaptiveResonanceEngine** passt sich automatisch an:
- **Kleine Graphen** (<1k Knoten): Liberal (mehr Exploration)
- **Mittlere Graphen** (1k-10k): Balanced
- **Große Graphen** (>10k): Conservative (mehr Pruning)

**Manuelles Tuning** (fortgeschritten):
```python
from component_44_resonance_engine import ResonanceEngine

engine = ResonanceEngine(netzwerk)
engine.set_hyperparameters(
    activation_threshold=0.35,  # Höher = weniger Konzepte
    decay_factor=0.75,          # Höher = stärkere Ausbreitung
    resonance_boost=0.6,        # Höher = stärkere Resonanz
    max_waves=6,                # Mehr = tiefere Exploration
    max_concepts_per_wave=150   # Mehr = breite Exploration
)
```

### Neo4j Performance

**Neue Indexes** (automatisch erstellt):
- `wort_lemma_index`: Schnellere Wortsuche
- `relation_confidence_index`: Schnellere Confidence-Filter
- `relation_context_index`: Schnellere Context-Filter

**Speedup**: 2-5x für relation-basierte Queries

---

Für detaillierte technische Informationen siehe `CLAUDE.md` und `DEVELOPER_GUIDE.md`

---

*Last Updated: 2025-11-08*
