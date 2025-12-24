# Production System Evaluation Guide

**Purpose**: Anleitung zur Evaluation des Production Systems gegen die Pipeline

---

## Quick Start

### 1. Evaluation ausführen

```bash
# Standard-Evaluation (1000 Queries)
python evaluate_production_system.py

# Custom Anzahl Queries
python evaluate_production_system.py --queries 5000

# Custom Output-Dateien
python evaluate_production_system.py --output my_results.json --report my_report.txt
```

### 2. Ergebnisse ansehen

**Text-Report:**
```bash
cat evaluation_report.txt
```

**JSON-Daten:**
```bash
python -m json.tool evaluation_results.json
```

---

## Was wird evaluiert?

### Metriken

**1. Confidence** (Vertrauenswürdigkeit):
- Durchschnittliche Confidence pro System
- Höher = bessere Antwortqualität

**2. Response Time** (Geschwindigkeit):
- Durchschnitt, P50, P95, P99
- Niedriger = schneller

**3. Success Rate** (Erfolgsquote):
- Prozent erfolgreicher Queries
- Höher = stabiler

**4. Cycles** (nur Production System):
- Durchschnittliche Anzahl Regelzyklen
- Zeigt Komplexität des Generierungsprozesses

**5. Rule Usage** (Regelverwendung):
- Welche Regeln werden am häufigsten verwendet?
- Identifiziert wichtige vs. selten genutzte Regeln

### Test-Queries

**5 Kategorien** (je 10 Queries):

1. **Taxonomy** (Was ist X?):
   - "Was ist ein Hund?"
   - "Was ist eine Katze?"
   - ...

2. **Properties** (Eigenschaften):
   - "Welche Eigenschaften hat ein Hund?"
   - "Welche Farbe hat ein Apfel?"
   - ...

3. **Capabilities** (Fähigkeiten):
   - "Was kann ein Vogel?"
   - "Was kann ein Computer?"
   - ...

4. **Multi-Hop** (Mehrstufiges Reasoning):
   - "Ist ein Hund ein Lebewesen?"
   - "Kann ein Vogel ein Tier sein?"
   - ...

5. **Complex** (Komplexe Fragen):
   - "Was ist der Unterschied zwischen einem Hund und einer Katze?"
   - "Warum können Vögel fliegen?"
   - ...

**Total**: 50 unique Queries, rotiert bis zur gewünschten Anzahl

---

## Report-Struktur

### 1. System-Metriken

```
PIPELINE METRICS
────────────────────────────────────────
Queries Handled: 500
Avg Confidence: 0.847
Avg Response Time: 0.234s
P95 Response Time: 0.452s
Success Rate: 97.2%

PRODUCTION SYSTEM METRICS
────────────────────────────────────────
Queries Handled: 500
Avg Confidence: 0.863
Avg Response Time: 0.189s
P95 Response Time: 0.321s
Success Rate: 98.4%
Avg Cycles: 4.7
Total Rules Applied: 2341
```

### 2. Comparison

```
COMPARISON
────────────────────────────────────────
Winner: Production System
Confidence Difference: +0.016 (Production better)
Speed Difference: +0.045s (Production faster)
```

### 3. Bottlenecks

Identifiziert Performance-Probleme:

```
BOTTLENECKS
────────────────────────────────────────
• PERFORMANCE: 52 queries (10.4%) took >500ms
• CYCLES: 23 queries required >10 cycles (avg: 12.3)
• CONFIDENCE: 31 queries had low confidence (<0.6)
```

**Keine Bottlenecks:**
```
• No significant bottlenecks detected
```

### 4. Tuning Recommendations

Empfehlungen basierend auf Regel-Usage:

```
TUNING RECOMMENDATIONS
────────────────────────────────────────
• UTILITY UP: Rule 'select_highest_confidence_fact' used in 87.3% of queries.
  Consider increasing utility to prioritize further.

• UTILITY DOWN: Rule 'format_complex_conjunction' rarely used (3 times).
  Consider lowering utility or removing if not critical.

• CYCLES: Average 6.8 cycles per query.
  Consider adding terminal rules or reducing max_cycles.
```

### 5. Rollout Recommendation

Entscheidung über Rollout basierend auf Metriken:

**Szenario A: Production System klar besser**
```
ROLLOUT: RECOMMENDED (High Priority)
- Confidence improvement: +5.2%
- Speed improvement: +19.2%
- Success rate: 98.4%

Recommended schedule:
  Week 1: 50% Production Weight (A/B testing, current)
  Week 2: 75% Production Weight (if no issues)
  Week 3: 100% Production Weight (full rollout)
  Week 4+: Monitor, keep Pipeline as fallback
```

**Szenario B: Pipeline besser**
```
ROLLOUT: NOT RECOMMENDED
- Pipeline outperforms Production System
- Confidence decline: -3.1%

Actions:
  1. Investigate why Production System underperforms
  2. Review rule utilities and specificities
  3. Re-evaluate after tuning
  4. Keep Production Weight at 50% or lower
```

**Szenario C: Tie**
```
ROLLOUT: CAUTIOUS APPROACH
- Both systems perform similarly
- No clear winner

Recommended schedule:
  Week 1-3: 50% Production Weight (monitor closely)
  Week 4: Decide based on user feedback and edge case analysis
```

---

## Interpretation der Ergebnisse

### Confidence Difference

**Interpretation:**
- **+0.05 oder mehr**: Signifikante Verbesserung -> Production System klar besser
- **+0.02 bis +0.05**: Moderate Verbesserung -> Production System besser
- **-0.02 bis +0.02**: Neutral -> Kein klarer Winner
- **-0.05 oder weniger**: Signifikante Verschlechterung -> Pipeline besser

**Beispiel:**
```
Confidence Difference: +0.016
-> Production System hat 1.6% höhere Confidence
-> Moderate Verbesserung, aber nicht dramatisch
```

### Speed Difference

**Interpretation:**
- **Positiv** (z.B. +0.045s): Production System ist schneller
- **Negativ** (z.B. -0.032s): Pipeline ist schneller

**Beispiel:**
```
Speed Difference: +0.045s
-> Production System ist 45ms schneller
-> 19.2% Verbesserung (bei 234ms Baseline)
```

### Success Rate

**Interpretation:**
- **>95%**: Exzellent, sehr stabil
- **90-95%**: Gut, akzeptabel
- **85-90%**: Akzeptabel, aber Verbesserungspotenzial
- **<85%**: Problematisch, Debugging nötig

### Rule Usage Distribution

**Top-Regeln** (>50% Usage):
- Sehr wichtig, Utility erhöhen wenn noch nicht hoch
- Sicherstellen dass Condition/Action optimal sind

**Bottom-Regeln** (<1% Usage):
- Selten genutzt -> Utility senken oder entfernen?
- ODER: Sehr spezifische Regel für Edge Cases (dann OK)

---

## Tuning-Workflow

### Schritt 1: Baseline-Evaluation

```bash
# Initiale Evaluation
python evaluate_production_system.py --queries 1000 --report baseline_report.txt
```

### Schritt 2: Analyse

```bash
# Analysiere Report
cat baseline_report.txt

# Identifiziere:
# - Bottlenecks
# - Selten genutzte Regeln
# - Häufig genutzte Regeln
```

### Schritt 3: Tuning

**Beispiel: Utility einer häufig genutzten Regel erhöhen**

```python
# In component_54_production_system.py oder deiner Custom-Regel-Datei
rule = create_production_rule(
    name="select_highest_confidence_fact",
    category="content_selection",
    utility=0.95,  # War 0.90, jetzt erhöht
    specificity=0.7,
    # ...
)
```

**Beispiel: Selten genutzte Regel entfernen**

```python
# Kommentiere Regel aus oder lösche
# rule_rarely_used = create_production_rule(...)
# engine.add_rule(rule_rarely_used)  # Nicht mehr hinzufügen
```

### Schritt 4: Re-Evaluation

```bash
# Nach Tuning erneut evaluieren
python evaluate_production_system.py --queries 1000 --report tuned_report.txt
```

### Schritt 5: Vergleich

```bash
# Vergleiche Berichte
diff baseline_report.txt tuned_report.txt

# Oder manuell:
# - Confidence verbessert?
# - Response Time verbessert?
# - Bottlenecks reduziert?
```

---

## Erweiterte Usage

### Custom Test-Queries

**Edit `evaluate_production_system.py`:**

```python
TEST_QUERIES = {
    "my_custom_category": [
        "Meine Test-Frage 1?",
        "Meine Test-Frage 2?",
        # ...
    ],
    # ... existing categories
}
```

### Production Weight anpassen

**Während Evaluation:**

```python
# In evaluate_production_system.py, __init__()
self.router = ResponseGenerationRouter(
    # ...
    production_weight=0.75  # 75% Production System, 25% Pipeline
)
```

### Nur Production System testen

```python
self.router = ResponseGenerationRouter(
    # ...
    production_weight=1.0  # 100% Production System
)
```

### Nur Pipeline testen

```python
self.router = ResponseGenerationRouter(
    # ...
    production_weight=0.0  # 100% Pipeline
)
```

---

## Troubleshooting

### Problem: Evaluation dauert zu lange

**Lösung**: Reduziere Anzahl Queries

```bash
python evaluate_production_system.py --queries 100
```

### Problem: Hohe Fehlerrate

**Diagnose**: Check `evaluation_results.json`

```bash
python -c "import json; data = json.load(open('evaluation_results.json')); print([r for r in data['results'] if not r['success']])"
```

**Lösung**:
- Prüfe Neo4j-Verbindung
- Prüfe Logs (`logs/kai.log`)
- Prüfe ob Test-Queries valide sind

### Problem: Alle Queries nutzen Pipeline

**Ursache**: Production Weight ist 0 oder Production System crasht

**Lösung**:
1. Check Production Weight in `evaluate_production_system.py`
2. Check Logs für Exceptions
3. Test Production System manuell:
   ```bash
   python -c "from component_54_production_system import ProductionSystemEngine; from component_1_netzwerk import KonzeptNetzwerk; engine = ProductionSystemEngine(KonzeptNetzwerk()); print('OK')"
   ```

### Problem: JSON-Parsing-Fehler

**Ursache**: Encoding-Problem oder abgebrochene Evaluation

**Lösung**:
```bash
# Check JSON-Validität
python -m json.tool evaluation_results.json > /dev/null

# Wenn invalide: Lösche und re-run
rm evaluation_results.json
python evaluate_production_system.py
```

---

## Best Practices

### 1. Baseline vor Tuning

Immer eine Baseline-Evaluation durchführen, bevor du Änderungen machst.

### 2. Schrittweise Tuning

Ändere jeweils nur 1-2 Regeln, dann re-evaluieren. Nicht alles auf einmal!

### 3. Statistisch signifikant

Mindestens 1000 Queries für aussagekräftige Ergebnisse. Besser: 5000+.

### 4. Reproduzierbarkeit

Gleiche Test-Queries für Vergleichbarkeit. Nutze Seeds für Randomness falls nötig.

### 5. Dokumentiere Änderungen

Halte fest, welche Tuning-Änderungen du gemacht hast und warum.

**Template:**
```markdown
## Tuning Log

### 2025-11-14: Baseline
- Evaluation: 1000 queries
- Winner: Production System
- Confidence: +0.016, Speed: +0.045s

### 2025-11-15: Utility-Tuning
- Changed: select_highest_confidence_fact utility 0.90 -> 0.95
- Re-Evaluation: 1000 queries
- Result: Confidence +0.021, Speed +0.048s (improved!)
```

---

## Nächste Schritte

Nach erfolgreicher Evaluation:

1. **Review Rollout Recommendation** im Report
2. **Implementiere Tuning-Empfehlungen** falls nötig
3. **Setze Production Weight** gemäß Empfehlung
4. **Monitor in Production** über 1-2 Wochen
5. **Sammle User-Feedback** (Feedback-Buttons!)
6. **Re-Evaluiere** nach 1 Monat

---

**Last Updated**: 2025-11-14
