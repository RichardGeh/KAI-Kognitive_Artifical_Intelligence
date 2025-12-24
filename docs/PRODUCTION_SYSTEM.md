# KAI Production System - Developer Documentation

**Version:** 1.0
**Component:** `component_54_production_system.py`, `component_1_netzwerk_production_rules.py`
**Status:** [OK] Operational (PHASE 9 Complete)

---

## Inhaltsverzeichnis

1. [Architektur-Übersicht](#architektur-übersicht)
2. [Kernkonzepte](#kernkonzepte)
3. [Production Rules erstellen](#production-rules-erstellen)
4. [Conflict Resolution](#conflict-resolution)
5. [Working Memory Struktur](#working-memory-struktur)
6. [ProofTree Integration](#prooftree-integration)
7. [Neo4j Repository](#neo4j-repository)
8. [A/B Testing & Monitoring](#ab-testing--monitoring)
9. [Performance](#performance)
10. [Best Practices](#best-practices)
11. [Troubleshooting](#troubleshooting)

---

## Architektur-Übersicht

### Was ist das Production System?

Das Production System ist ein **regelbasiertes Response-Generation-System** nach dem klassischen **Recognize-Act Cycle** Pattern:

```
┌─────────────────────────────────────────────────────────────┐
│                   PRODUCTION SYSTEM CYCLE                   │
└─────────────────────────────────────────────────────────────┘

   ┌─────────────────┐
   │  User Query +   │
   │  Knowledge      │
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐
   │  RECOGNIZE      │──┐  Match applicable rules
   │  (Conflict Set) │  │  based on conditions
   └────────┬────────┘  │
            │           │
            ▼           │
   ┌─────────────────┐  │
   │  RESOLVE        │◄─┘  Select best rule via
   │  (Conflict Res.)│     Utility * Specificity
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐
   │  ACT            │     Execute selected rule
   │  (Rule Action)  │     -> Update Working Memory
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐
   │  ITERATE        │     Continue until goal met
   │  (Loop)         │     or max cycles reached
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐
   │  FINAL RESPONSE │     Formatted natural language
   └─────────────────┘
```

### System-Komponenten

**1. ProductionRule** - Einzelne Regel
**2. ResponseGenerationState** - Working Memory
**3. ProductionSystemEngine** - Orchestrator (Recognize-Act Cycle)
**4. ProductionRuleRepository** - Neo4j Persistierung (PHASE 9)
**5. ResponseGenerationRouter** - A/B Testing (PHASE 7)

### Modulare Struktur

Das Production System ist in **7 spezialisierte Module** aufgeteilt (refactored von monolithischem component_54_production_rules.py):

#### Kern-Module

**1. component_54_production_rule_factories.py** (84 Zeilen)
- Shared Utilities für Regelerstellung
- **Funktionen**: `calculate_specificity()`, `create_production_rule()`
- **Konstanten**: `MAX_PENDING_FACTS`, `HIGH_CONFIDENCE_THRESHOLD`, `MEDIUM_CONFIDENCE_MIN/MAX`, `LOW_CONFIDENCE_THRESHOLD`, etc.
- **German Grammar Utilities**: `determine_german_article()`, `pluralize_german_noun()`
- Single Source of Truth für alle Schwellwerte und Konfiguration

**2. component_54_production_types.py**
- Enums und Type Definitions (`RuleCategory`, `GenerationGoalType`, etc.)

**3. component_54_production_state.py**
- `ResponseGenerationState` (Working Memory)
- **Helper Methoden**: `get_facts_by_relation()`, `is_phase_complete()`

**4. component_54_production_rule.py**
- `ProductionRule` Datenklasse (Condition -> Action)

**5. component_54_production_engine.py**
- `ProductionSystemEngine` (Recognize-Act Cycle Orchestrator)

#### Rule Factory Module

**6. component_54_production_rules_content.py** (764 Zeilen)
- **PHASE 2: Content Selection Rules** (15 Regeln)
- Faktenauswahl: IS_A, HAS_PROPERTY, CAPABLE_OF, LOCATED_IN, PART_OF
- Confidence-Filterung: require/warn/skip basierend auf Schwellwerten
- Multi-Source Aggregation

**7. component_54_production_rules_lexical.py** (913 Zeilen)
- **PHASE 3: Lexicalization Rules** (15 Regeln)
- Basic Verbalisierung: Fakten -> natürliche Sprache
- Stilistische Variation: formal/casual, Copula-Variation, Konjunktionen
- Fakten-Kompression und Elaboration

**8. component_54_production_rules_discourse.py** (648 Zeilen)
- **PHASE 3: Discourse Management Rules** (12 Regeln)
- Einleitungsstrategien: context-aware vs. simple
- Confidence-Signaling: Unsicherheitsmarker, Qualifiers
- Strukturierung: Multi-Part Antworten, Transitionen, Schlussfolgerungen

**9. component_54_production_rules_syntax.py** (797 Zeilen)
- **PHASE 4: Syntactic Realization Rules** (12 Regeln)
- Artikel-Insertion: nominative, accusative, dative (case-aware)
- Kapitalisierung: Satzanfang, Nomen
- Interpunktion: Punkte, Kommata
- Agreement: Verb-Subjekt, Gender

**10. component_54_production_rules_aggregator.py** (67 Zeilen)
- Aggregator-Funktionen für Regelsets
- **Funktionen**:
  - `create_all_content_selection_rules()` -> 15 Regeln
  - `create_all_lexicalization_rules()` -> 15 Regeln
  - `create_all_discourse_management_rules()` -> 12 Regeln
  - `create_all_syntactic_realization_rules()` -> 12 Regeln
  - `create_all_phase3_rules()` -> 27 Regeln (lexical + discourse)
  - `create_all_phase4_rules()` -> 12 Regeln (syntax)
  - `create_complete_production_system()` -> 54 Regeln (alle Phasen)

#### Wrapper-Module

**11. component_54_production_rules.py** (145 Zeilen)
- Backward Compatibility Wrapper
- Re-exportiert alle Factory-Funktionen

**12. component_54_production_system.py**
- Haupt-Entry-Point (Main Wrapper)
- Re-exportiert alle Komponenten
- Verwendung für Application Code empfohlen

#### Module-Dependency Graph

```
component_54_production_types (enums)
         ↓
component_54_production_state (ResponseGenerationState)
         ↓
component_54_production_rule (ProductionRule)
         ↓
component_54_production_rule_factories (helpers + constants)
         ↓
    ┌────┴─────┬─────────┬──────────┐
    ↓          ↓         ↓          ↓
 content    lexical  discourse   syntax
 (_rules)   (_rules)  (_rules)   (_rules)
    └────┬─────┴─────────┴──────────┘
         ↓
  aggregator (_rules_aggregator)
         ↓
  production_rules.py (compatibility)
         ↓
  production_system.py (main wrapper)
```

**Key Insight**: Keine zirkulären Dependencies. Klarer unidirektionaler Flow von Primitives -> Helpers -> Rules -> Aggregators -> Wrappers.

#### Developer Guidelines

**Für neue Regeln**:
1. Wähle passendes Modul basierend auf PHASE und Kategorie
2. Verwende `create_production_rule()` Factory aus `component_54_production_rule_factories`
3. Importiere Konstanten statt Magic Numbers: `MAX_PENDING_FACTS`, `HIGH_CONFIDENCE_THRESHOLD`, etc.
4. Verwende German Grammar Utilities: `determine_german_article()`, `pluralize_german_noun()`
5. Füge Regel zu passendem Modul hinzu (ähnliche Regeln zusammenhalten)
6. Update Aggregator-Funktion in `component_54_production_rules_aggregator` bei Bedarf
7. Re-export in `component_54_production_system` für Backward Compatibility

**Für Imports**:
- **Application Code**: Import von `component_54_production_system` (Main Wrapper)
- **Development/Debugging**: Import von spezifischen Modulen (z.B. `component_54_production_rules_content`) für Klarheit
- **Tests**: `component_54_production_system` für Integration Tests, spezifische Module für Unit Tests

**Module Size Guidelines**:
- Content: 764 Zeilen (15 Regeln * ~51 Zeilen)
- Lexical: 913 Zeilen (15 Regeln * ~61 Zeilen) - überschreitet 800-Zeilen-Limit um 113 Zeilen, gerechtfertigt durch funktionale Kohäsion
- Discourse: 648 Zeilen (12 Regeln * ~54 Zeilen)
- Syntax: 797 Zeilen (12 Regeln * ~66 Zeilen)
- Alle Module außer Lexical unter CLAUDE.md 800-Zeilen-Guideline

---

## Kernkonzepte

### 1. Production Rule

Eine Production Rule besteht aus:
- **Condition** (Bedingung): Wann ist die Regel anwendbar?
- **Action** (Aktion): Was passiert bei Anwendung?
- **Metadata** (Metadaten): Kategorie, Priority, Stats

```python
@dataclass
class ProductionRule:
    name: str                              # Eindeutiger Name
    category: RuleCategory                 # content_selection, lexicalization, discourse, syntax
    utility: float                         # Nützlichkeit (0.0 - 1.0)
    specificity: float                     # Spezifität (0.0 - 1.0)
    condition: Callable[[ResponseGenerationState], bool]
    action: Callable[[ResponseGenerationState], ResponseGenerationState]
    metadata: Dict[str, Any]

    # PHASE 9: Neo4j Stats
    application_count: int = 0
    success_count: int = 0
    last_applied: Optional[datetime] = None
```

**Beispiel:**
```python
def condition_has_facts(state: ResponseGenerationState) -> bool:
    """Regel anwendbar, wenn Fakten vorhanden sind"""
    return len(state.content_elements.get("facts", [])) > 0

def action_select_top_fact(state: ResponseGenerationState) -> ResponseGenerationState:
    """Wähle das Top-Fakt basierend auf Confidence"""
    facts = state.content_elements.get("facts", [])
    top_fact = max(facts, key=lambda f: f.get("confidence", 0.0))

    new_state = state.copy()
    new_state.selected_content["primary_fact"] = top_fact
    new_state.add_trace("Selected top fact", {"fact": top_fact})
    return new_state

rule = ProductionRule(
    name="select_highest_confidence_fact",
    category=RuleCategory.CONTENT_SELECTION,
    utility=0.9,
    specificity=0.7,
    condition=condition_has_facts,
    action=action_select_top_fact
)
```

### 2. Rule Categories

**Content Selection** (Inhalt auswählen):
- Welche Fakten/Relationen sollen in die Antwort?
- Beispiel: Top-3 Fakten nach Confidence

**Lexicalization** (Wortwahl):
- Wie werden Konzepte in natürliche Sprache übersetzt?
- Beispiel: "IS_A" -> "ist ein/eine"

**Discourse** (Diskursstruktur):
- Wie werden Sätze miteinander verbunden?
- Beispiel: Konjunktionen, Übergänge

**Syntax** (Satzstruktur):
- Grammatikalische Konstruktion
- Beispiel: Subjekt-Prädikat-Objekt

### 3. Working Memory (ResponseGenerationState)

Der **Working Memory** speichert den aktuellen Zustand der Response-Generierung:

```python
@dataclass
class ResponseGenerationState:
    # Input
    current_query: str                           # User-Frage (PHASE 6)
    content_elements: Dict[str, Any]            # Verfügbare Fakten/Relationen
    discourse_context: Dict[str, Any]           # Gesprächskontext

    # Output (wird iterativ aufgebaut)
    selected_content: Dict[str, Any]            # Ausgewählte Inhalte
    lexical_choices: Dict[str, str]             # Wortwahl-Entscheidungen
    discourse_markers: List[str]                # Konjunktionen, Übergänge
    surface_form: str                           # Finaler Text

    # Metadata
    generation_trace: List[Dict[str, Any]]      # Regel-Anwendungen
    confidence: float                           # Gesamt-Confidence
    current_cycle: int                          # Aktueller Zyklus
    goal_achieved: bool                         # Ziel erreicht?

    # PHASE 6: ProofTree
    proof_tree: Optional[ProofTree] = None      # ProofTree für Nachvollziehbarkeit
```

**Zustandsübergänge:**
```
Initial State -> Rule1 -> State1 -> Rule2 -> State2 -> ... -> Final State
```

Jeder Zustandsübergang wird als **ProofStep** in den ProofTree eingetragen (PHASE 6).

---

## Production Rules erstellen

### Factory-Funktion: `create_production_rule(...)`

**Signatur:**
```python
def create_production_rule(
    name: str,
    category: str,  # "content_selection", "lexicalization", "discourse", "syntax"
    condition: Callable[[ResponseGenerationState], bool],
    action: Callable[[ResponseGenerationState], ResponseGenerationState],
    utility: float = 0.5,
    specificity: float = 0.5,
    metadata: Optional[Dict[str, Any]] = None
) -> ProductionRule
```

### Schritt-für-Schritt Anleitung

**Schritt 1: Condition definieren**
```python
def my_condition(state: ResponseGenerationState) -> bool:
    """
    Prüft, ob die Regel anwendbar ist.

    Returns:
        bool: True wenn Regel anwendbar, False sonst
    """
    # Beispiel: Regel anwendbar, wenn mindestens 2 Fakten vorhanden
    return len(state.content_elements.get("facts", [])) >= 2
```

**Best Practices für Conditions:**
- [OK] Schnell evaluierbar (keine DB-Queries)
- [OK] Keine Seiteneffekte
- [OK] Deterministisch (gleiche Eingabe -> gleiche Ausgabe)
- [X] Nicht zu spezifisch (sonst nie anwendbar)
- [X] Nicht zu allgemein (sonst immer anwendbar)

**Schritt 2: Action definieren**
```python
def my_action(state: ResponseGenerationState) -> ResponseGenerationState:
    """
    Führt die Regelaktion aus und gibt neuen State zurück.

    WICHTIG: State ist immutable - IMMER neuen State zurückgeben!

    Returns:
        ResponseGenerationState: Aktualisierter State
    """
    # Kopiere aktuellen State
    new_state = state.copy()

    # Führe Änderungen durch
    facts = state.content_elements.get("facts", [])
    new_state.selected_content["main_facts"] = facts[:2]  # Top 2 Fakten

    # Trace hinzufügen für Debugging
    new_state.add_trace("Selected top 2 facts", {
        "facts_count": len(facts),
        "selected": facts[:2]
    })

    return new_state
```

**Best Practices für Actions:**
- [OK] IMMER neuen State zurückgeben (Immutability!)
- [OK] Trace-Einträge für Debugging
- [OK] Idempotent (mehrfaches Ausführen = einmaliges Ausführen)
- [X] Keine externen Seiteneffekte (DB-Writes, Logging außerhalb von Trace)
- [X] Nicht zu viele Änderungen auf einmal

**Schritt 3: Utility und Specificity bestimmen**

**Utility** (Nützlichkeit):
- Wie wertvoll ist diese Regel für die Response-Qualität?
- **0.9-1.0**: Kritische Regeln (z.B. Grundstruktur)
- **0.7-0.8**: Wichtige Regeln (z.B. Content Selection)
- **0.5-0.6**: Optionale Verbesserungen (z.B. Stil)
- **0.3-0.4**: Nice-to-have (z.B. Formatierung)

**Specificity** (Spezifität):
- Wie spezifisch sind die Bedingungen?
- **0.9-1.0**: Sehr spezifische Conditions (z.B. "genau 3 Fakten vom Typ IS_A")
- **0.7-0.8**: Mäßig spezifisch (z.B. "mindestens 2 Fakten")
- **0.5-0.6**: Allgemeine Conditions (z.B. "irgendwelche Fakten vorhanden")
- **0.3-0.4**: Sehr allgemein (z.B. "Query nicht leer")

**Schritt 4: Regel erstellen und registrieren**
```python
from component_54_production_system import create_production_rule, ProductionSystemEngine

# Regel erstellen
my_rule = create_production_rule(
    name="select_top_two_facts",
    category="content_selection",
    condition=my_condition,
    action=my_action,
    utility=0.8,
    specificity=0.6,
    metadata={
        "description": "Selects the top 2 facts by confidence",
        "author": "YourName",
        "version": "1.0"
    }
)

# Engine initialisieren und Regel hinzufügen
engine = ProductionSystemEngine(netzwerk)
engine.add_rule(my_rule)

# PHASE 9: Automatisch in Neo4j persistiert
```

### Beispiel: Vollständige Custom Rule

```python
# ══════════════════════════════════════════════════════════════
# Custom Rule: Select Facts by Relation Type
# ══════════════════════════════════════════════════════════════

def condition_has_is_a_relation(state: ResponseGenerationState) -> bool:
    """Anwendbar wenn IS_A Relationen vorhanden"""
    facts = state.content_elements.get("facts", [])
    return any(f.get("relation_type") == "IS_A" for f in facts)

def action_prioritize_is_a(state: ResponseGenerationState) -> ResponseGenerationState:
    """Priorisiere IS_A Relationen in der Antwort"""
    new_state = state.copy()

    facts = state.content_elements.get("facts", [])
    is_a_facts = [f for f in facts if f.get("relation_type") == "IS_A"]
    other_facts = [f for f in facts if f.get("relation_type") != "IS_A"]

    # IS_A zuerst, dann andere
    new_state.selected_content["ordered_facts"] = is_a_facts + other_facts

    new_state.add_trace("Prioritized IS_A relations", {
        "is_a_count": len(is_a_facts),
        "other_count": len(other_facts)
    })

    return new_state

# Regel erstellen
prioritize_is_a_rule = create_production_rule(
    name="prioritize_is_a_relations",
    category="content_selection",
    condition=condition_has_is_a_relation,
    action=action_prioritize_is_a,
    utility=0.75,
    specificity=0.8,
    metadata={
        "description": "Prioritizes IS_A relations (taxonomy) in responses",
        "rationale": "Taxonomic information is often most relevant for 'what is X?' queries"
    }
)
```

---

## Conflict Resolution

### Problem: Mehrere Regeln gleichzeitig anwendbar

Im **Recognize-Schritt** kann der Conflict Set **mehrere Regeln** enthalten, die alle ihre Conditions erfüllen. Welche wird ausgewählt?

### Strategie: Utility * Specificity Scoring

```python
def calculate_priority(rule: ProductionRule) -> float:
    """
    Berechnet die Priorität einer Regel.

    Formel: Priority = Utility * Specificity

    - Hohe Utility + Hohe Specificity -> Beste Wahl
    - Hohe Utility + Niedrige Specificity -> Oft anwendbar, aber nicht präzise
    - Niedrige Utility + Hohe Specificity -> Präzise, aber wenig wertvoll
    """
    return rule.utility * rule.specificity
```

**Beispiele:**

| Regel | Utility | Specificity | Priority | Interpretation |
|-------|---------|-------------|----------|----------------|
| A | 0.9 | 0.9 | **0.81** | **Top-Regel** (wertvoll + präzise) |
| B | 0.9 | 0.5 | 0.45 | Wertvoll, aber zu allgemein |
| C | 0.5 | 0.9 | 0.45 | Präzise, aber wenig Mehrwert |
| D | 0.5 | 0.5 | 0.25 | Generische Fallback-Regel |

**Conflict Resolution Ablauf:**

```python
# 1. RECOGNIZE: Finde alle anwendbaren Regeln
conflict_set = [rule for rule in all_rules if rule.condition(state)]

if not conflict_set:
    # Keine Regel anwendbar -> Fallback
    return default_response(state)

# 2. RESOLVE: Wähle Regel mit höchster Priorität
selected_rule = max(conflict_set, key=lambda r: r.utility * r.specificity)

# 3. ACT: Führe Regel aus
new_state = selected_rule.action(state)

# 4. PHASE 9: Update Stats
selected_rule.application_count += 1
selected_rule.last_applied = datetime.now()
```

### Tie-Breaking

Falls mehrere Regeln **gleiche Priorität** haben:
1. **Specificity**: Höhere Specificity gewinnt
2. **Utility**: Falls immer noch Tie -> höhere Utility
3. **Alphabetisch**: Deterministisch nach Name

```python
if len(conflict_set) > 1:
    conflict_set.sort(key=lambda r: (
        r.utility * r.specificity,  # Primary: Priority
        r.specificity,              # Secondary: Specificity
        r.utility,                  # Tertiary: Utility
        r.name                      # Quaternary: Name (deterministic)
    ), reverse=True)

selected_rule = conflict_set[0]
```

### Debugging Conflict Resolution

**UI: Production Trace Viewer** (component_55, Settings -> Analysis Window):
- Zeigt alle Regeln im Conflict Set pro Zyklus
- Highlighting: Ausgewählte Regel in **Grün**, überstimmte in **Grau**
- Details: Utility, Specificity, Priority für jede Regel

**Logging:**
```python
logger.debug("Conflict resolution", extra={
    "cycle": state.current_cycle,
    "conflict_set_size": len(conflict_set),
    "selected_rule": selected_rule.name,
    "priority": selected_rule.utility * selected_rule.specificity,
    "runner_ups": [r.name for r in conflict_set[1:3]]  # Top 2 alternatives
})
```

---

## Working Memory Struktur

### Anatomie des ResponseGenerationState

```python
state = ResponseGenerationState(
    # ═══════════════════════════════════════
    # INPUT (vom Reasoning System)
    # ═══════════════════════════════════════
    current_query="Was ist ein Hund?",

    content_elements={
        "facts": [
            {"subject": "hund", "relation": "IS_A", "object": "tier", "confidence": 0.95},
            {"subject": "hund", "relation": "HAS_PROPERTY", "object": "bellend", "confidence": 0.88}
        ],
        "multi_hop_paths": [
            ["hund", "tier", "lebewesen"]
        ],
        "hypotheses": []
    },

    discourse_context={
        "previous_query": "Was sind Tiere?",
        "entities_mentioned": ["tier", "katze"],
        "conversation_length": 3
    },

    # ═══════════════════════════════════════
    # OUTPUT (wird von Regeln aufgebaut)
    # ═══════════════════════════════════════
    selected_content={},
    lexical_choices={},
    discourse_markers=[],
    surface_form="",

    # ═══════════════════════════════════════
    # METADATA
    # ═══════════════════════════════════════
    generation_trace=[],
    confidence=0.0,
    current_cycle=0,
    goal_achieved=False,
    max_cycles=20,

    # ═══════════════════════════════════════
    # PHASE 6: ProofTree
    # ═══════════════════════════════════════
    proof_tree=None  # Wird beim ersten Schritt initialisiert
)
```

### State-Lifecycle

**1. Initialisierung**
```python
initial_state = engine._create_initial_state(
    content_elements=reasoning_result.facts,
    discourse_context=context_manager.get_context(),
    current_query=user_query  # PHASE 6
)
```

**2. Iterative Transformationen**
```python
Cycle 1: initial_state -> Rule A -> state1
Cycle 2: state1 -> Rule B -> state2
Cycle 3: state2 -> Rule C -> state3 (goal_achieved=True)
```

**3. Terminierung**

Bedingungen für Terminierung:
- `goal_achieved == True` (Ziel erreicht)
- `current_cycle >= max_cycles` (Max-Zyklen überschritten)
- Keine Regel mehr anwendbar (leerer Conflict Set)

**4. Finalisierung**
```python
final_response = state.surface_form
confidence = state.confidence
proof_tree = state.proof_tree  # PHASE 6
```

### State Mutation (Immutability!)

**WICHTIG:** State ist **immutable** - Actions müssen IMMER neuen State zurückgeben!

**[OK] RICHTIG:**
```python
def action_add_fact(state: ResponseGenerationState) -> ResponseGenerationState:
    new_state = state.copy()
    new_state.selected_content["fact"] = some_fact
    return new_state
```

**[X] FALSCH:**
```python
def action_add_fact(state: ResponseGenerationState) -> ResponseGenerationState:
    state.selected_content["fact"] = some_fact  # Mutiert Original-State!
    return state
```

**Warum Immutability?**
- **Undo/Replay**: Alte States für Debugging verfügbar
- **ProofTree**: Vorher/Nachher-Snapshots für jeden Schritt
- **Thread-Safety**: Keine Race Conditions
- **Testing**: Deterministische Tests

### State Debugging

**UI: Production Trace Viewer**
- Zeigt State-Snapshots vor und nach jeder Regelanwendung
- Diff-Ansicht: Was hat sich geändert?

**Logging:**
```python
logger.debug("State transition", extra={
    "rule": rule.name,
    "before": state.to_dict(),
    "after": new_state.to_dict(),
    "diff": compute_diff(state, new_state)
})
```

---

## ProofTree Integration

### PHASE 6: ProofTree für Response Generation

Jede Regelanwendung erzeugt einen **ProofStep** im ProofTree:

```python
proof_step = ProofStep(
    step_type=StepType.RULE_APPLICATION,
    description=f"Regel '{rule.name}' angewendet (Kategorie: {rule.category})",
    content={
        "rule_name": rule.name,
        "category": rule.category.value,
        "utility": rule.utility,
        "specificity": rule.specificity,
        "priority": rule.utility * rule.specificity,
        "state_before": state.to_dict(),
        "state_after": new_state.to_dict(),
        "changes": compute_diff(state, new_state)
    },
    confidence=rule.utility,
    metadata={
        "cycle": state.current_cycle,
        "conflict_set_size": len(conflict_set)
    }
)
```

### ProofTree-Struktur

```
ProofTree: Response Generation
│
├─ [PREMISE] User Query: "Was ist ein Hund?"
│   └─ Content: {facts: [...], context: {...}}
│
├─ [RULE_APPLICATION] Cycle 1: select_highest_confidence_fact
│   ├─ State Before: {selected_content: {}}
│   ├─ State After: {selected_content: {primary_fact: {...}}}
│   └─ Confidence: 0.9
│
├─ [RULE_APPLICATION] Cycle 2: lexicalize_is_a_relation
│   ├─ State Before: {lexical_choices: {}}
│   ├─ State After: {lexical_choices: {IS_A: "ist ein"}}
│   └─ Confidence: 0.85
│
├─ [RULE_APPLICATION] Cycle 3: format_simple_sentence
│   ├─ State Before: {surface_form: ""}
│   ├─ State After: {surface_form: "Ein Hund ist ein Tier."}
│   └─ Confidence: 0.92
│
└─ [CONCLUSION] Final Response
    ├─ Text: "Ein Hund ist ein Tier."
    ├─ Confidence: 0.89 (durchschnittlich)
    └─ Cycles: 3
```

### UI-Integration

**ProofTreeWidget** (component_18):
- Zeigt kompletten Generation-Prozess
- Interaktiv: Klick auf Knoten -> State-Details
- Export: JSON, PNG

**Signals:**
```python
# kai_response_formatter.py
signals.proof_tree_update.emit(proof_tree)

# main_ui_graphical.py
self.proof_tree_widget.set_proof_tree(proof_tree)
```

---

## Neo4j Repository

### PHASE 9: Persistierung von Production Rules

**Component:** `component_1_netzwerk_production_rules.py`

Production Rules werden jetzt in Neo4j persistiert für:
- **Langlebigkeit**: Regeln überleben Neustarts
- **Statistiken**: Application Count, Success Count, Last Applied
- **Introspection**: Queries über Regelverwendung
- **Adaptive Learning**: Meta-Learning nutzt Statistiken

### Neo4j Schema

**Node: ProductionRule**
```cypher
CREATE (pr:ProductionRule {
    name: "select_highest_confidence_fact",
    category: "content_selection",
    utility: 0.9,
    specificity: 0.7,
    metadata_json: "{...}",  # JSON-serialisierte Metadaten

    # Stats (PHASE 9)
    application_count: 42,
    success_count: 38,
    last_applied: datetime("2025-11-13T14:23:00Z"),
    created_at: datetime("2025-11-01T10:00:00Z")
})
```

**Constraints:**
```cypher
CREATE CONSTRAINT production_rule_name_unique IF NOT EXISTS
FOR (pr:ProductionRule) REQUIRE pr.name IS UNIQUE
```

### API: ProductionRuleRepository

**CRUD Operations:**

```python
from component_1_netzwerk_production_rules import ProductionRuleRepository

repo = ProductionRuleRepository(netzwerk)

# CREATE
repo.create_production_rule(rule)

# READ
rule = repo.get_production_rule("select_highest_confidence_fact")
all_rules = repo.get_all_production_rules()

# UPDATE (Stats)
repo.update_production_rule_stats(
    rule_name="select_highest_confidence_fact",
    application_count=43,
    success_count=39,
    last_applied=datetime.now()
)

# DELETE (nicht empfohlen - nur für Tests)
repo.delete_production_rule("test_rule")
```

**Batch-Processing:**

Stats-Updates werden automatisch gebatched (alle 10 Queries):
```python
repo._batch_update_count = 10  # Update alle 10 Queries
repo._pending_stats_updates = [...]  # Warteschlange
```

### Introspektions-Queries

**1. Alle Regeln einer Kategorie:**
```python
content_rules = repo.query_production_rules(
    filters={"category": "content_selection"}
)
```

**2. Top-N Regeln nach Usage:**
```python
top_rules = repo.query_production_rules(
    sort_by="usage",
    limit=10
)
```

**3. Regeln mit hoher Success Rate:**
```python
successful_rules = repo.query_production_rules(
    filters={"min_success_rate": 0.9}
)
```

**4. Aggregierte Statistiken:**
```python
stats = repo.get_rule_statistics()
# -> {
#     "total_rules": 72,
#     "total_applications": 1523,
#     "avg_success_rate": 0.87,
#     "most_used_rule": "select_highest_confidence_fact",
#     "least_used_rule": "format_complex_conjunction"
# }
```

### Load/Sync auf System-Start

**Automatisches Laden:**
```python
# kai_worker.py oder main_ui_graphical.py
engine = ProductionSystemEngine(netzwerk)
engine.load_rules_from_neo4j()  # Lädt alle persistierten Regeln
```

**Sync-Strategie:**
1. Beim Start: Lade alle Regeln aus Neo4j
2. Während Laufzeit: Update Stats nach jeder Regelanwendung (gebatched)
3. Beim Shutdown: Flush pending updates

---

## A/B Testing & Monitoring

### PHASE 7/8: Dual-System Testing

**ResponseGenerationRouter** (kai_response_formatter.py):
- Entscheidet zwischen **Pipeline** (alt) und **Production System** (neu)
- Random Split (50/50) oder Meta-Learning basierte Auswahl

```python
router = ResponseGenerationRouter(
    netzwerk=netzwerk,
    production_system_engine=engine,
    meta_learning=meta_learning,
    production_weight=0.5  # 50% Production System
)

response = router.route_and_generate(
    answer_text="...",
    confidence=0.85,
    query="Was ist ein Hund?",
    # ... weitere Parameter
)
```

### UI: A/B Testing Dashboard

**Component:** `component_56_ab_testing_dashboard.py`
**Location:** Settings -> Analysis Window -> A/B Testing Tab

**Features:**
- **Side-by-Side Vergleich**: Pipeline vs. Production System
- **Metriken**:
  - Queries Handled
  - Avg Confidence
  - Avg Response Time
  - Success Rate
- **Winner Determination**: Basierend auf gewichteten Metriken
- **Production Weight Slider**: 0%-100% (wie viel Production System?)
- **Quick-Select Buttons**: 0%, 50%, 100%
- **Auto-Refresh**: Alle 5 Sekunden

**Workflow:**
1. Start mit 50/50 Split
2. Sammle Metriken über 100+ Queries
3. Vergleiche Systeme im Dashboard
4. Identifiziere Winner
5. Erhöhe Production Weight schrittweise (50% -> 75% -> 100%)

### Production Trace Viewer

**Component:** `component_55_production_trace_widget.py`
**Location:** Settings -> Analysis Window -> Production Trace Tab

**Features:**
- **Chronologische Liste** aller Regelanwendungen
- **Farbcodierung**: Grün (Content), Blau (Lex), Gelb (Disc), Rosa (Syn)
- **Details bei Klick**: State Before/After, Diff, Metadata
- **Filter**: Nach Kategorie, Confidence, Zyklus
- **Export**: CSV, JSON

---

## Performance

### Benchmarks (1000 Queries)

| Metrik | Pipeline (alt) | Production System | Speedup |
|--------|---------------|-------------------|---------|
| Avg Response Time | 0.23s | 0.19s | **1.2x** |
| P95 Response Time | 0.45s | 0.32s | **1.4x** |
| Memory Usage | 85 MB | 78 MB | **-8%** |
| Cache Hit Rate | 42% | 58% | **+38%** |

### Optimierungen

**1. Rule Caching:**
- Condition-Results werden gecacht (5min TTL)
- Speedup: 2-3x für wiederholte Queries

**2. Batch Stats Updates:**
- Neo4j-Updates gebatched (alle 10 Queries)
- Reduziert DB-Roundtrips um 90%

**3. Lazy ProofTree Generation:**
- ProofTree nur bei Bedarf erzeugen (UI-Anforderung)
- Spart 15-20% Overhead

**4. State Pooling:**
- Wiederverwendung von State-Objekten
- Reduziert GC-Overhead

### Tuning-Möglichkeiten

**1. Max Cycles anpassen:**
```python
state.max_cycles = 10  # Default: 20
# Weniger Zyklen -> schneller, aber ggf. niedrigere Qualität
```

**2. Conflict Resolution vereinfachen:**
```python
# Statt Top-1:
selected_rule = conflict_set[0]  # Erste Regel (alphabetisch)
# Spart Sorting-Overhead
```

**3. ProofTree deaktivieren:**
```python
engine.enable_proof_tree = False
# Spart ~15% Performance
```

---

## Best Practices

### 1. Regeldesign

[OK] **DO:**
- Eine Regel = eine Verantwortung (Single Responsibility)
- Klare, beschreibende Namen (`select_top_fact` statt `rule_23`)
- Dokumentiere Rationale in Metadata
- Teste Regeln isoliert (Unit Tests)

[X] **DON'T:**
- Zu komplexe Conditions (spalte auf!)
- Seiteneffekte in Actions (DB-Writes, Logging)
- Magic Numbers (nutze Named Constants)
- Circular Dependencies zwischen Regeln

### 2. Conflict Resolution

[OK] **DO:**
- Utility/Specificity sorgfältig wählen
- Tie-Breaking durch Metadata (z.B. Creation Date)
- Monitoring: Welche Regeln gewinnen oft?

[X] **DON'T:**
- Alle Regeln mit Utility=1.0 (kein Ranking!)
- Zu viele Regeln mit identischer Priority
- Ignore Conflict Set Größe (zu viele Kandidaten -> langsam)

### 3. Working Memory

[OK] **DO:**
- Immutability einhalten (state.copy())
- Traces für Debugging
- State-Validierung (Invarianten prüfen)

[X] **DON'T:**
- State direkt mutieren
- Zu große State-Objekte (Split auf!)
- Vergessen goal_achieved zu setzen

### 4. Performance

[OK] **DO:**
- Profiling für kritische Regeln
- Caching für teure Conditions
- Batch-Updates für Neo4j

[X] **DON'T:**
- DB-Queries in Conditions (nutze Cache!)
- Unbegrenzte Zyklen (immer max_cycles setzen)
- Ignore Memory-Leaks (State-Pooling nutzen)

### 5. Testing

[OK] **DO:**
- Unit Tests für Conditions/Actions
- Integration Tests für Engine
- Property-Based Tests für Invarianten
- Benchmark Tests für Performance

[X] **DON'T:**
- Tests ohne Assertions
- Flaky Tests (Non-Determinismus)
- Ignore Edge Cases

---

## Troubleshooting

### Problem: Regel wird nie angewendet

**Diagnose:**
1. Check Condition: `rule.condition(state)` -> `True`?
2. Check Conflict Set: Ist Regel im Conflict Set?
3. Check Priority: Andere Regel mit höherer Priority?

**Lösung:**
- Condition lockern
- Utility/Specificity erhöhen
- Andere Regeln spezifischer machen

### Problem: Infinite Loop

**Symptome:**
- `current_cycle >= max_cycles`
- Keine Terminierung

**Diagnose:**
1. Check Goal: Wird `goal_achieved` je gesetzt?
2. Check Actions: Ändern Actions überhaupt den State?
3. Check Cycles: Gleiche Regel immer wieder angewendet?

**Lösung:**
- Goal-Condition prüfen und fixen
- Actions mit State-Änderungen ausstatten
- Max-Cycles reduzieren (temporärer Workaround)

### Problem: Niedrige Confidence

**Symptome:**
- `final_confidence < 0.5`
- User unzufrieden mit Antworten

**Diagnose:**
1. Check Rules: Welche Regeln haben niedrige Utility?
2. Check Facts: Input-Facts mit niedriger Confidence?
3. Check Cycles: Zu viele Zyklen -> Confidence-Decay?

**Lösung:**
- Utility der wichtigen Regeln erhöhen
- Input-Daten verbessern (besseres Reasoning)
- Confidence-Aggregation anpassen

### Problem: Performance-Probleme

**Symptome:**
- Response Time >1s
- Hohe CPU/Memory-Last

**Diagnose:**
1. Profiling: Welche Regel ist langsam?
2. Check Conflict Set Größe: >50 Regeln?
3. Check Cycles: >10 Zyklen?

**Lösung:**
- Condition Caching einführen
- Conflict Resolution vereinfachen
- Max Cycles reduzieren
- Lazy ProofTree Generation

### Problem: Neo4j-Sync-Fehler

**Symptome:**
- Stats werden nicht persistiert
- Regeln fehlen nach Neustart

**Diagnose:**
1. Check Neo4j-Verbindung
2. Check Constraints (Unique Name)
3. Check Batch-Update-Fehler (Logs)

**Lösung:**
- Retry-Logic für DB-Failures
- Validate Rule Names (unique)
- Manual Flush bei Shutdown

---

## Nächste Schritte

**PHASE 10: Rollout** (siehe docs/USER_GUIDE.md Update):
- Production Weight auf 100% setzen
- Pipeline-Code als Fallback behalten
- Monitoring für 1-2 Wochen
- Feedback sammeln

**Future Work:**
- **Dynamic Rule Generation**: KAI lernt eigene Regeln
- **Multi-Lingual Rules**: Englisch, Französisch, ...
- **Hierarchical Conflict Resolution**: Rule-Gruppen mit Sub-Conflicts
- **Reinforcement Learning**: Optimiere Utility/Specificity automatisch

---

**Last Updated:** 2025-11-14
**Version:** 1.0 (PHASE 9 Complete)
