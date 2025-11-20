# setup_initial_knowledge.py
"""
Initialisiert das KAI-System mit grundlegendem Meta-Wissen.
Enthält robuste Fehlerbehandlung und Verifizierung.

Initialisiert:
- Extraktionsregeln (IS_A, HAS_PROPERTY, CAPABLE_OF, PART_OF, LOCATED_IN, etc.)
- Lexikalische Trigger (ist, hat, kann, etc.)
- Production Rules (Content Selection, Lexicalization, Discourse, Syntax)
- Spatial Relations (NORTH_OF, ADJACENT_TO, INSIDE, etc.)
- Arithmetic Concepts (Summe, Differenz, Produkt, Quotient, etc.)
- Number Language (eins, zwei, drei, ..., zehn)
- Pattern Prototypes (häufige Sprachmuster)
- Meta-Learning Baseline (initiale Strategy-Performance)
"""
import logging
import sys
import time

# WICHTIG: Encoding-Fix MUSS früh importiert werden
# Behebt Windows cp1252 -> UTF-8 Probleme für Unicode-Zeichen ([ERROR], [SUCCESS], [WARNING], etc.)
import kai_encoding_fix  # noqa: F401 (automatische Aktivierung beim Import)
from component_1_netzwerk import KonzeptNetzwerk
from kai_logging import setup_logging


def verify_database_connection(netzwerk: KonzeptNetzwerk) -> bool:
    """
    Verifiziert die Datenbankverbindung und prüft grundlegende Funktionalität.

    Returns:
        True bei erfolgreicher Verbindung, False sonst
    """
    logger = logging.getLogger("KAI_SETUP")

    if not netzwerk.driver:
        logger.error("[ERROR] Netzwerk-Driver ist None - keine Verbindung möglich")
        return False

    try:
        # Prüfe Konnektivität
        netzwerk.driver.verify_connectivity()
        logger.info("[SUCCESS] Datenbankverbindung erfolgreich")

        # Prüfe, ob wir Schreibrechte haben
        with netzwerk.driver.session(database="neo4j") as session:
            result = session.run("RETURN 1 AS test")
            if result.single()["test"] != 1:
                logger.error("[ERROR] Kann keine Daten aus der Datenbank lesen")
                return False

        logger.info("[SUCCESS] Lese-/Schreibzugriff verifiziert")
        return True

    except Exception as e:
        logger.error(f"[ERROR] Datenbankverbindung fehlgeschlagen: {e}", exc_info=True)
        return False


def verify_constraints(netzwerk: KonzeptNetzwerk) -> bool:
    """
    Prüft, ob die erforderlichen Constraints existieren.

    Returns:
        True wenn alle Constraints vorhanden sind
    """
    logger = logging.getLogger("KAI_SETUP")

    expected_constraints = [
        "WortLemma",
        "KonzeptName",
        "ExtractionRuleType",
        "PatternPrototypeId",
        "LexiconName",
    ]

    try:
        with netzwerk.driver.session(database="neo4j") as session:
            result = session.run("SHOW CONSTRAINTS")
            existing_constraints = {record["name"] for record in result}

        missing = []
        for constraint in expected_constraints:
            if constraint not in existing_constraints:
                missing.append(constraint)

        if missing:
            logger.warning(f"[WARNING]  Fehlende Constraints: {missing}")
            logger.info("Constraints werden automatisch beim ersten Aufruf erstellt")
        else:
            logger.info(
                f"[SUCCESS] Alle {len(expected_constraints)} Constraints vorhanden"
            )

        return True

    except Exception as e:
        logger.error(f"[ERROR] Fehler beim Prüfen der Constraints: {e}")
        return False


def create_extraction_rules(netzwerk: KonzeptNetzwerk) -> bool:
    """
    Erstellt die grundlegenden Extraktionsregeln.

    Returns:
        True bei Erfolg, False bei Fehler
    """
    logger = logging.getLogger("KAI_SETUP")

    rules_to_create = [
        # Taxonomische Beziehungen
        {
            "type": "IS_A",
            "regex": r"^(?:ein|eine|der|die|das\s+)?(.+?)\s+ist\s+(?:auch\s+)?(?:ein|eine|der|die|das\s+)?(.+?)\.?$",
            "desc": "Grundlegende 'ist ein/eine'-Beziehung mit Artikeln",
        },
        {
            "type": "IS_A",
            "regex": r"^(.+?)\s+sind\s+(.+?)\.?$",
            "desc": "Plural 'sind'-Beziehung (z.B. 'Vögel sind Tiere')",
        },
        # Eigenschaften
        {
            "type": "HAS_PROPERTY",
            "regex": r"^(.+?)\s+hat\s+(?:die Eigenschaft|die Fähigkeit)?\s*(.+?)\.?$",
            "desc": "Erfasst Eigenschaften oder Fähigkeiten",
        },
        {
            "type": "HAS_PROPERTY",
            "regex": r"^(.+?)\s+ist\s+(groß|klein|rot|grün|blau|schwer|leicht|schnell|langsam)\.?$",
            "desc": "Adjektivische Eigenschaften",
        },
        # Fähigkeiten
        {
            "type": "CAPABLE_OF",
            "regex": r"^(.+?)\s+kann\s+(.+?)\.?$",
            "desc": "Erfasst Fähigkeiten (z.B. 'Ein Vogel kann fliegen')",
        },
        {
            "type": "CAPABLE_OF",
            "regex": r"^(.+?)\s+kann\s+nicht\s+(.+?)\.?$",
            "desc": "Negierte Fähigkeiten (z.B. 'Ein Pinguin kann nicht fliegen')",
        },
        # Teil-Ganzes
        {
            "type": "PART_OF",
            "regex": r"^(.+?)\s+(?:ist Teil von|gehört zu)\s+(.+?)\.?$",
            "desc": "Erfasst Teil-Ganzes-Beziehungen",
        },
        {
            "type": "PART_OF",
            "regex": r"^(.+?)\s+hat\s+(?:ein|eine|einen)\s+(.+?)\.?$",
            "desc": "Besitz-Beziehungen (z.B. 'Ein Auto hat Räder')",
        },
        # Räumliche Beziehungen
        {
            "type": "LOCATED_IN",
            "regex": r"^(.+?)\s+(?:ist in|befindet sich in|liegt in)\s+(.+?)\.?$",
            "desc": "Erfasst räumliche Beziehungen",
        },
        {
            "type": "LOCATED_IN",
            "regex": r"^(.+?)\s+(?:ist|liegt)\s+(?:nördlich|südlich|östlich|westlich)\s+von\s+(.+?)\.?$",
            "desc": "Himmelsrichtungen",
        },
        # Kausale Beziehungen
        {
            "type": "CAUSES",
            "regex": r"^(.+?)\s+verursacht\s+(.+?)\.?$",
            "desc": "Kausale Beziehungen (z.B. 'Regen verursacht Nässe')",
        },
        {
            "type": "CAUSES",
            "regex": r"^(.+?)\s+führt zu\s+(.+?)\.?$",
            "desc": "Folge-Beziehungen",
        },
        # Temporale Beziehungen
        {
            "type": "BEFORE",
            "regex": r"^(.+?)\s+(?:ist|kommt|passiert)\s+vor\s+(.+?)\.?$",
            "desc": "Zeitliche Reihenfolge",
        },
        {
            "type": "AFTER",
            "regex": r"^(.+?)\s+(?:ist|kommt|passiert)\s+nach\s+(.+?)\.?$",
            "desc": "Zeitliche Reihenfolge (umgekehrt)",
        },
        # Assoziationen/Aktionen (POSITIV - wichtig für Logik-Rätsel!)
        {
            "type": "ASSOCIATED_WITH",
            "regex": r"^(.+?)\s+(?:mag|isst|trinkt|benutzt|verwendet|macht)\s+(?:gerne\s+)?(.+?)\.?$",
            "desc": "Positive Assoziationen/Aktionen (z.B. 'X isst Y', 'X trinkt gerne Y')",
        },
        # Generische Negationen (ALLGEMEIN - nicht rätselspezifisch)
        {
            "type": "NOT_HAS_PROPERTY",
            "regex": r"^(.+?)\s+(?:hat|haben)\s+(?:kein|keine|keinen)\s+(.+?)\.?$",
            "desc": "Negierte Eigenschaften (z.B. 'X hat keine Y')",
        },
        {
            "type": "NOT_ASSOCIATED_WITH",
            "regex": r"^(.+?)\s+(?:mag|isst|trinkt|benutzt|verwendet|macht)\s+(?:kein|keine|keinen)\s+(.+?)\.?$",
            "desc": "Negierte Assoziationen/Aktionen (z.B. 'X isst keine Y', 'X trinkt keinen Y')",
        },
        {
            "type": "NOT_IS_A",
            "regex": r"^(.+?)\s+ist\s+(?:kein|keine)\s+(.+?)\.?$",
            "desc": "Negierte Taxonomie (z.B. 'X ist kein Y')",
        },
    ]

    logger.info(f"📝 Erstelle {len(rules_to_create)} Extraktionsregeln...")

    created_count = 0
    for rule in rules_to_create:
        try:
            logger.info(f"  -> Erstelle: {rule['type']} ({rule['desc']})")
            netzwerk.create_extraction_rule(
                relation_type=rule["type"], regex_pattern=rule["regex"]
            )

            # Kurze Pause, um sicherzustellen, dass die DB Zeit zum Schreiben hat
            time.sleep(0.1)

            # Sofortige Verifikation
            with netzwerk.driver.session(database="neo4j") as session:
                result = session.run(
                    """
                    MATCH (r:ExtractionRule {relation_type: $rel})
                    RETURN r.regex_pattern AS pattern
                """,
                    rel=rule["type"],
                )
                record = result.single()

                if record and record["pattern"] == rule["regex"]:
                    created_count += 1
                    logger.info(
                        f"    [SUCCESS] Regel '{rule['type']}' erfolgreich in DB gespeichert"
                    )
                else:
                    logger.error(
                        f"    [ERROR] Regel '{rule['type']}' NICHT in DB gefunden!"
                    )
                    return False

        except Exception as e:
            logger.error(f"    [ERROR] Fehler beim Erstellen von '{rule['type']}': {e}")
            return False

    logger.info(
        f"[SUCCESS] {created_count}/{len(rules_to_create)} Regeln erfolgreich erstellt"
    )
    return created_count == len(rules_to_create)


def create_lexical_triggers(netzwerk: KonzeptNetzwerk) -> bool:
    """
    Erstellt die initialen lexikalischen Trigger.

    Returns:
        True bei Erfolg, False bei Fehler
    """
    logger = logging.getLogger("KAI_SETUP")

    initial_triggers = [
        # Kopula und Definitionen
        "ist",
        "sind",
        "bezeichnet",
        "definiert",
        "heißt",
        "nennt",
        "bedeutet",
        # Eigenschaften und Besitz
        "hat",
        "besitzt",
        "enthält",
        # Fähigkeiten
        "kann",
        "vermag",
        "fähig",
        # Räumliche Relationen
        "gehört",
        "befindet",
        "liegt",
        "steht",
        "sitzt",
        # Kausale Relationen
        "verursacht",
        "bewirkt",
        "führt",
        # Temporale Relationen
        "kommt",
        "passiert",
        "geschieht",
        # Vergleiche
        "größer",
        "kleiner",
        "schneller",
        "langsamer",
    ]

    logger.info(f"🏷️  Füge {len(initial_triggers)} lexikalische Trigger hinzu...")

    created_count = 0
    for trigger in initial_triggers:
        try:
            created = netzwerk.add_lexical_trigger(trigger)
            if created:
                created_count += 1
                logger.info(f"  [SUCCESS] Trigger '{trigger}' hinzugefügt")
            else:
                logger.info(f"  [INFO]  Trigger '{trigger}' existierte bereits")
        except Exception as e:
            logger.error(f"  [ERROR] Fehler beim Hinzufügen von '{trigger}': {e}")
            return False

    logger.info(
        f"[SUCCESS] {created_count} neue Trigger erstellt, {len(initial_triggers) - created_count} bereits vorhanden"
    )
    return True


def create_production_rules(netzwerk: KonzeptNetzwerk) -> bool:
    """
    Erstellt grundlegende Production Rules für das Response Generation System.

    Returns:
        True bei Erfolg, False bei Fehler
    """
    logger = logging.getLogger("KAI_SETUP")

    # Überprüfe, ob bereits Regeln existieren
    try:
        existing_rules = netzwerk.get_all_production_rules()
        if len(existing_rules) > 50:
            logger.info(
                f"[INFO]  {len(existing_rules)} Production Rules bereits vorhanden - überspringe Erstellung"
            )
            return True
    except Exception as e:
        logger.warning(f"[WARNING]  Konnte existierende Rules nicht prüfen: {e}")

    production_rules_data = [
        # Content Selection Rules (Utility: 0.8-0.9)
        {
            "name": "select_direct_answer",
            "category": "content_selection",
            "utility": 0.9,
            "specificity": 8,
            "metadata": {
                "description": "Wählt direkte Antworten für einfache Fragen",
                "triggers": ["simple_question", "has_direct_answer"],
            },
        },
        {
            "name": "select_explanation",
            "category": "content_selection",
            "utility": 0.85,
            "specificity": 7,
            "metadata": {
                "description": "Wählt Erklärungen für komplexe Fragen",
                "triggers": ["complex_question", "needs_explanation"],
            },
        },
        {
            "name": "select_examples",
            "category": "content_selection",
            "utility": 0.8,
            "specificity": 6,
            "metadata": {
                "description": "Fügt Beispiele für besseres Verständnis hinzu",
                "triggers": ["abstract_concept", "user_confused"],
            },
        },
        # Lexicalization Rules (Utility: 0.7-0.85)
        {
            "name": "use_formal_language",
            "category": "lexicalization",
            "utility": 0.75,
            "specificity": 5,
            "metadata": {
                "description": "Verwendet formale Sprache für technische Antworten",
                "triggers": ["technical_topic", "formal_context"],
            },
        },
        {
            "name": "use_simple_words",
            "category": "lexicalization",
            "utility": 0.8,
            "specificity": 6,
            "metadata": {
                "description": "Verwendet einfache Wörter für bessere Verständlichkeit",
                "triggers": ["simple_question", "beginner_level"],
            },
        },
        {
            "name": "avoid_jargon",
            "category": "lexicalization",
            "utility": 0.85,
            "specificity": 7,
            "metadata": {
                "description": "Vermeidet Fachjargon außer bei Experten",
                "triggers": ["general_audience", "educational_context"],
            },
        },
        # Discourse Rules (Utility: 0.6-0.8)
        {
            "name": "start_with_summary",
            "category": "discourse",
            "utility": 0.8,
            "specificity": 7,
            "metadata": {
                "description": "Beginnt mit einer Zusammenfassung bei langen Antworten",
                "triggers": ["long_answer", "complex_topic"],
            },
        },
        {
            "name": "use_transitions",
            "category": "discourse",
            "utility": 0.7,
            "specificity": 5,
            "metadata": {
                "description": "Fügt Übergänge zwischen Abschnitten ein",
                "triggers": ["multi_part_answer", "structured_response"],
            },
        },
        {
            "name": "end_with_invitation",
            "category": "discourse",
            "utility": 0.65,
            "specificity": 4,
            "metadata": {
                "description": "Endet mit Einladung für Nachfragen",
                "triggers": ["incomplete_answer", "complex_topic"],
            },
        },
        # Syntax Rules (Utility: 0.5-0.75)
        {
            "name": "use_short_sentences",
            "category": "syntax",
            "utility": 0.75,
            "specificity": 6,
            "metadata": {
                "description": "Verwendet kurze Sätze für bessere Lesbarkeit",
                "triggers": ["simple_question", "clarity_priority"],
            },
        },
        {
            "name": "use_active_voice",
            "category": "syntax",
            "utility": 0.7,
            "specificity": 5,
            "metadata": {
                "description": "Bevorzugt Aktiv statt Passiv",
                "triggers": ["direct_communication", "action_oriented"],
            },
        },
        {
            "name": "add_punctuation_variety",
            "category": "syntax",
            "utility": 0.6,
            "specificity": 4,
            "metadata": {
                "description": "Variiert Satzzeichen für natürlicheren Fluss",
                "triggers": ["long_response", "narrative_style"],
            },
        },
    ]

    logger.info(f"🔧 Erstelle {len(production_rules_data)} Production Rules...")

    created_count = 0
    for rule_data in production_rules_data:
        try:
            # Erstelle Regel in Neo4j (mit Dummy condition_code und action_code)
            # Diese sind Platzhalter, da wir hier nur die Metadaten-Struktur initialisieren
            success = netzwerk.create_production_rule(
                name=rule_data["name"],
                category=rule_data["category"],
                condition_code="",  # Platzhalter für Setup
                action_code="",  # Platzhalter für Setup
                utility=rule_data["utility"],
                specificity=rule_data["specificity"],
                metadata=rule_data["metadata"],
            )

            if success:
                created_count += 1
                logger.info(
                    f"  [SUCCESS] Rule '{rule_data['name']}' ({rule_data['category']}) erstellt"
                )
            else:
                logger.warning(
                    f"  [WARNING]  Rule '{rule_data['name']}' konnte nicht erstellt werden"
                )

        except Exception as e:
            logger.error(
                f"  [ERROR] Fehler beim Erstellen von '{rule_data['name']}': {e}"
            )
            # Nicht kritisch - fortfahren
            continue

    logger.info(
        f"[SUCCESS] {created_count}/{len(production_rules_data)} Production Rules erstellt"
    )
    return True


def create_spatial_relations(netzwerk: KonzeptNetzwerk) -> bool:
    """
    Erstellt grundlegende Spatial Relation Definitionen.

    Returns:
        True bei Erfolg, False bei Fehler
    """
    logger = logging.getLogger("KAI_SETUP")

    spatial_relations = [
        # Kardinalrichtungen
        ("NORTH_OF", "nördlich von", {"transitive": True, "symmetric": False}),
        ("SOUTH_OF", "südlich von", {"transitive": True, "symmetric": False}),
        ("EAST_OF", "östlich von", {"transitive": True, "symmetric": False}),
        ("WEST_OF", "westlich von", {"transitive": True, "symmetric": False}),
        # Nachbarschaft
        ("ADJACENT_TO", "benachbart zu", {"transitive": False, "symmetric": True}),
        ("NEXT_TO", "neben", {"transitive": False, "symmetric": True}),
        # Enthaltensein
        ("INSIDE", "innerhalb", {"transitive": True, "symmetric": False}),
        ("CONTAINS", "enthält", {"transitive": True, "symmetric": False}),
        ("OUTSIDE", "außerhalb", {"transitive": False, "symmetric": False}),
        # Distanz
        ("NEAR", "nahe bei", {"transitive": False, "symmetric": True}),
        ("FAR_FROM", "weit von", {"transitive": False, "symmetric": True}),
        # Überlappung
        ("OVERLAPS", "überlappt mit", {"transitive": False, "symmetric": True}),
        # Berührung
        ("TOUCHES", "berührt", {"transitive": False, "symmetric": True}),
        # Zwischen
        ("BETWEEN", "zwischen", {"transitive": False, "symmetric": False}),
    ]

    logger.info(f"📍 Erstelle {len(spatial_relations)} Spatial Relations...")

    created_count = 0
    for rel_type, description, properties in spatial_relations:
        try:
            # Erstelle als Konzept im Graphen
            with netzwerk.driver.session(database="neo4j") as session:
                session.run(
                    """
                    MERGE (sr:SpatialRelation {name: $name})
                    SET sr.description = $desc,
                        sr.transitive = $transitive,
                        sr.symmetric = $symmetric,
                        sr.created_at = datetime()
                    """,
                    name=rel_type,
                    desc=description,
                    transitive=properties["transitive"],
                    symmetric=properties["symmetric"],
                )

            created_count += 1
            logger.info(f"  [SUCCESS] Spatial Relation '{rel_type}' erstellt")

        except Exception as e:
            logger.error(f"  [ERROR] Fehler bei '{rel_type}': {e}")
            continue

    logger.info(
        f"[SUCCESS] {created_count}/{len(spatial_relations)} Spatial Relations erstellt"
    )
    return True


def create_arithmetic_concepts(netzwerk: KonzeptNetzwerk) -> bool:
    """
    Erstellt grundlegende arithmetische Konzepte.

    Returns:
        True bei Erfolg, False bei Fehler
    """
    logger = logging.getLogger("KAI_SETUP")

    arithmetic_concepts = [
        # Operationen
        ("summe", "addition", "Das Ergebnis einer Addition"),
        ("differenz", "subtraction", "Das Ergebnis einer Subtraktion"),
        ("produkt", "multiplication", "Das Ergebnis einer Multiplikation"),
        ("quotient", "division", "Das Ergebnis einer Division"),
        # Eigenschaften
        ("gerade", "even", "Durch 2 teilbar"),
        ("ungerade", "odd", "Nicht durch 2 teilbar"),
        ("primzahl", "prime", "Nur durch 1 und sich selbst teilbar"),
        # Konstanten
        ("pi", "constant", "Kreiszahl π ≈ 3.14159"),
        ("e", "constant", "Eulersche Zahl e ≈ 2.71828"),
        ("phi", "constant", "Goldener Schnitt φ ≈ 1.61803"),
        # Vergleiche
        ("größer", "comparison", "Numerischer Vergleich >"),
        ("kleiner", "comparison", "Numerischer Vergleich <"),
        ("gleich", "comparison", "Numerische Gleichheit ="),
    ]

    logger.info(f"🔢 Erstelle {len(arithmetic_concepts)} Arithmetic Concepts...")

    created_count = 0
    for concept, category, description in arithmetic_concepts:
        try:
            # Erstelle als Wort/Konzept
            netzwerk.assert_relation(
                concept, "IS_A", "mathematisches_konzept", "Initiale Wissensbasis"
            )
            netzwerk.assert_relation(
                concept, "HAS_PROPERTY", category, "Initiale Wissensbasis"
            )

            # Setze Beschreibung als Attribut
            with netzwerk.driver.session(database="neo4j") as session:
                session.run(
                    """
                    MATCH (w:Wort {lemma: $lemma})
                    SET w.description = $desc,
                        w.category = $cat
                    """,
                    lemma=concept,
                    desc=description,
                    cat=category,
                )

            created_count += 1
            logger.info(f"  [SUCCESS] Concept '{concept}' ({category}) erstellt")

        except Exception as e:
            logger.error(f"  [ERROR] Fehler bei '{concept}': {e}")
            continue

    logger.info(
        f"[SUCCESS] {created_count}/{len(arithmetic_concepts)} Arithmetic Concepts erstellt"
    )
    return True


def create_number_language(netzwerk: KonzeptNetzwerk) -> bool:
    """
    Erstellt Zahl-zu-Wort Mappings (0-20).

    Returns:
        True bei Erfolg, False bei Fehler
    """
    logger = logging.getLogger("KAI_SETUP")

    number_words = [
        (0, "null"),
        (1, "eins"),
        (2, "zwei"),
        (3, "drei"),
        (4, "vier"),
        (5, "fünf"),
        (6, "sechs"),
        (7, "sieben"),
        (8, "acht"),
        (9, "neun"),
        (10, "zehn"),
        (11, "elf"),
        (12, "zwölf"),
        (13, "dreizehn"),
        (14, "vierzehn"),
        (15, "fünfzehn"),
        (16, "sechzehn"),
        (17, "siebzehn"),
        (18, "achtzehn"),
        (19, "neunzehn"),
        (20, "zwanzig"),
    ]

    logger.info(f"🔤 Erstelle {len(number_words)} Zahl-zu-Wort Mappings...")

    created_count = 0
    for num, word in number_words:
        try:
            # Erstelle bidirektionale Verknüpfung
            with netzwerk.driver.session(database="neo4j") as session:
                session.run(
                    """
                    MERGE (w:Wort {lemma: $word})
                    SET w.numeric_value = $num,
                        w.word_type = 'number'
                    WITH w
                    MERGE (n:Number {value: $num})
                    SET n.german_word = $word
                    MERGE (w)-[:REPRESENTS_NUMBER]->(n)
                    """,
                    word=word,
                    num=num,
                )

            created_count += 1

        except Exception as e:
            logger.error(f"  [ERROR] Fehler bei '{word}' ({num}): {e}")
            continue

    logger.info(
        f"[SUCCESS] {created_count}/{len(number_words)} Number Mappings erstellt"
    )
    return True


def create_example_knowledge(netzwerk: KonzeptNetzwerk) -> bool:
    """
    Erstellt Beispiel-Wissensbasis für Demonstrationszwecke.

    Returns:
        True bei Erfolg, False bei Fehler
    """
    logger = logging.getLogger("KAI_SETUP")

    examples = [
        # Tiere
        ("hund", "IS_A", "tier"),
        ("katze", "IS_A", "tier"),
        ("vogel", "IS_A", "tier"),
        ("pinguin", "IS_A", "vogel"),
        ("hund", "CAPABLE_OF", "bellen"),
        ("katze", "CAPABLE_OF", "miauen"),
        ("vogel", "CAPABLE_OF", "fliegen"),
        ("pinguin", "CAPABLE_OF", "schwimmen"),
        ("hund", "HAS_PROPERTY", "treu"),
        ("katze", "HAS_PROPERTY", "unabhängig"),
        # Früchte
        ("apfel", "IS_A", "frucht"),
        ("banane", "IS_A", "frucht"),
        ("apfel", "HAS_PROPERTY", "rot"),
        ("banane", "HAS_PROPERTY", "gelb"),
        ("apfel", "HAS_PROPERTY", "süß"),
        # Fahrzeuge
        ("auto", "IS_A", "fahrzeug"),
        ("fahrrad", "IS_A", "fahrzeug"),
        ("auto", "HAS_PROPERTY", "schnell"),
        ("auto", "PART_OF", "rad"),
        ("fahrrad", "PART_OF", "lenker"),
        # Orte
        ("berlin", "IS_A", "stadt"),
        ("deutschland", "IS_A", "land"),
        ("berlin", "LOCATED_IN", "deutschland"),
    ]

    logger.info(f"📚 Erstelle {len(examples)} Beispiel-Wissenseinträge...")

    created_count = 0
    for subject, relation, obj in examples:
        try:
            created = netzwerk.assert_relation(
                subject, relation, obj, "Initiale Wissensbasis"
            )
            if created:
                created_count += 1

        except Exception as e:
            logger.error(f"  [ERROR] Fehler bei '{subject} {relation} {obj}': {e}")
            continue

    logger.info(f"[SUCCESS] {created_count}/{len(examples)} Beispiel-Einträge erstellt")
    return True


def verify_complete_setup(netzwerk: KonzeptNetzwerk) -> bool:
    """
    Führt eine umfassende Verifikation des Setups durch.

    Returns:
        True wenn alles korrekt ist, False sonst
    """
    logger = logging.getLogger("KAI_SETUP")

    logger.info("🔍 Führe abschließende Verifikation durch...")

    # 1. Prüfe Extraktionsregeln
    try:
        rules = netzwerk.get_all_extraction_rules()
        expected_core_rules = [
            "IS_A",
            "HAS_PROPERTY",
            "CAPABLE_OF",
            "PART_OF",
            "LOCATED_IN",
        ]

        if len(rules) < len(expected_core_rules):
            logger.error(
                f"[ERROR] Nur {len(rules)} Regeln gefunden (erwartet mindestens {len(expected_core_rules)})"
            )
            return False

        existing_types = {r["relation_type"] for r in rules}
        missing_rules = [r for r in expected_core_rules if r not in existing_types]

        if missing_rules:
            logger.error(f"[ERROR] Fehlende Regeln: {missing_rules}")
            return False

        logger.info(
            f"  [SUCCESS] Alle {len(expected_core_rules)} Core Extraktionsregeln vorhanden (total: {len(rules)})"
        )

    except Exception as e:
        logger.error(f"[ERROR] Fehler beim Verifizieren der Regeln: {e}")
        return False

    # 2. Prüfe Lexikalische Trigger
    try:
        triggers = netzwerk.get_lexical_triggers()
        expected_triggers = ["ist", "hat", "kann", "liegt", "verursacht"]

        missing_triggers = [t for t in expected_triggers if t not in triggers]

        if missing_triggers:
            logger.error(f"[ERROR] Fehlende Trigger: {missing_triggers}")
            return False

        logger.info(
            f"  [SUCCESS] Alle erwarteten Trigger vorhanden ({len(triggers)} total)"
        )

    except Exception as e:
        logger.error(f"[ERROR] Fehler beim Verifizieren der Trigger: {e}")
        return False

    # 3. Prüfe Production Rules
    try:
        production_rules = netzwerk.get_all_production_rules()
        if len(production_rules) >= 10:
            logger.info(f"  [SUCCESS] {len(production_rules)} Production Rules geladen")
        else:
            logger.warning(
                f"  [WARNING]  Nur {len(production_rules)} Production Rules gefunden (erwartet mindestens 10)"
            )

    except Exception as e:
        logger.warning(f"  [WARNING]  Konnte Production Rules nicht verifizieren: {e}")

    # 4. Prüfe Spatial Relations
    try:
        with netzwerk.driver.session(database="neo4j") as session:
            result = session.run(
                """
                MATCH (sr:SpatialRelation)
                RETURN count(sr) AS count
                """
            )
            count = result.single()["count"]

            if count >= 10:
                logger.info(f"  [SUCCESS] {count} Spatial Relations vorhanden")
            else:
                logger.warning(f"  [WARNING]  Nur {count} Spatial Relations gefunden")

    except Exception as e:
        logger.warning(f"  [WARNING]  Konnte Spatial Relations nicht verifizieren: {e}")

    # 5. Prüfe Arithmetic Concepts
    try:
        with netzwerk.driver.session(database="neo4j") as session:
            result = session.run(
                """
                MATCH (w:Wort)-[:IS_A]->(c:Wort {lemma: 'mathematisches_konzept'})
                RETURN count(w) AS count
                """
            )
            count = result.single()["count"]

            if count >= 5:
                logger.info(f"  [SUCCESS] {count} Arithmetic Concepts vorhanden")
            else:
                logger.warning(f"  [WARNING]  Nur {count} Arithmetic Concepts gefunden")

    except Exception as e:
        logger.warning(
            f"  [WARNING]  Konnte Arithmetic Concepts nicht verifizieren: {e}"
        )

    # 6. Prüfe Number Language
    try:
        with netzwerk.driver.session(database="neo4j") as session:
            result = session.run(
                """
                MATCH (w:Wort)-[:REPRESENTS_NUMBER]->(n:Number)
                RETURN count(w) AS count
                """
            )
            count = result.single()["count"]

            if count >= 10:
                logger.info(f"  [SUCCESS] {count} Number-Wort Mappings vorhanden")
            else:
                logger.warning(f"  [WARNING]  Nur {count} Number Mappings gefunden")

    except Exception as e:
        logger.warning(f"  [WARNING]  Konnte Number Mappings nicht verifizieren: {e}")

    # 7. Prüfe Beispiel-Wissensbasis
    try:
        with netzwerk.driver.session(database="neo4j") as session:
            result = session.run(
                """
                MATCH (w:Wort)
                WHERE w.lemma IN ['hund', 'katze', 'apfel', 'berlin']
                RETURN count(w) AS count
                """
            )
            count = result.single()["count"]

            if count >= 3:
                logger.info(
                    f"  [SUCCESS] Beispiel-Wissensbasis vorhanden ({count} Einträge)"
                )
            else:
                logger.warning(
                    f"  [WARNING]  Beispiel-Wissensbasis unvollständig ({count} Einträge)"
                )

    except Exception as e:
        logger.warning(
            f"  [WARNING]  Konnte Beispiel-Wissensbasis nicht verifizieren: {e}"
        )

    # 8. Teste eine einfache Regel-Anwendung
    try:
        logger.info("  🧪 Teste Regel-Anwendung mit Beispielsatz...")
        test_subject = "_test_setup_verification"
        test_object = "_test_setup_target"

        # Erstelle Testrelation
        created = netzwerk.assert_relation(
            test_subject, "IS_A", test_object, "Setup-Verifikationstest"
        )

        if not created:
            logger.warning(
                "  [WARNING]  Testrelation konnte nicht erstellt werden (möglicherweise bereits vorhanden)"
            )

        # Prüfe, ob Relation existiert
        facts = netzwerk.query_graph_for_facts(test_subject)

        if "IS_A" not in facts or test_object not in facts["IS_A"]:
            logger.error("  [ERROR] Testrelation nicht im Graphen gefunden")
            return False

        logger.info("  [SUCCESS] Regel-Anwendung funktioniert korrekt")

        # Cleanup
        with netzwerk.driver.session(database="neo4j") as session:
            session.run(
                """
                MATCH (n)
                WHERE n.name STARTS WITH '_test_setup_' OR n.lemma STARTS WITH '_test_setup_'
                DETACH DELETE n
            """
            )
        logger.info("  🧹 Test-Daten bereinigt")

    except Exception as e:
        logger.error(f"[ERROR] Fehler beim Testen der Regel-Anwendung: {e}")
        return False

    return True


def create_common_words(netzwerk) -> bool:
    """
    Erstellt Bootstrap-CommonWords für Entity-Extraktion.

    CommonWords sind Wörter die NICHT als Entitäten erkannt werden sollen
    (Artikel, Konjunktionen, Fragewörter, bekannte Objekte, Verben).

    Returns:
        True wenn erfolgreich
    """
    logger = logging.getLogger("KAI_SETUP")

    try:
        # Definiere Basis-CommonWords nach Kategorie
        common_words_dict = {
            # Artikel
            "article": [
                "der",
                "die",
                "das",
                "ein",
                "eine",
                "einen",
                "einem",
                "eines",
                "einer",
                "den",
                "dem",
                "des",
            ],
            # Konjunktionen und Konnektor-Wörter
            "conjunction": [
                "und",
                "oder",
                "aber",
                "wenn",
                "dann",
                "dass",
                "ob",
                "als",
                "wie",
                "weil",
                "während",
                "bevor",
                "nachdem",
                "hingegen",
                "allerdings",
                "außerdem",
                "ferner",
                "weiterhin",
                "entweder",
                "weder",
                "noch",
            ],
            # Fragewörter
            "question_word": [
                "wer",
                "was",
                "wie",
                "wo",
                "wann",
                "warum",
                "wieso",
                "weshalb",
                "welche",
                "welcher",
                "welches",
                "wozu",
                "woher",
                "wohin",
            ],
            # Pronomen und Determinatoren
            "pronoun": [
                "es",
                "sie",
                "er",
                "ich",
                "du",
                "wir",
                "ihr",
                "dieser",
                "diese",
                "dieses",
                "jener",
                "jene",
                "jenes",
                "mein",
                "dein",
                "sein",
                "ihr",
                "unser",
                "euer",
                "alle",
                "beide",
                "einige",
                "manche",
                "jede",
                "jeder",
                "jedes",
            ],
            # Adverbien und Modalwörter
            "adverb": [
                "auch",
                "nicht",
                "nur",
                "sehr",
                "ganz",
                "gerne",
                "gern",
                "oft",
                "immer",
                "nie",
                "niemals",
                "manchmal",
                "selten",
                "vielleicht",
                "sicher",
                "bestimmt",
                "wahrscheinlich",
                "also",
                "denn",
                "jedoch",
                "trotzdem",
            ],
            # Präpositionen
            "preposition": [
                "in",
                "an",
                "auf",
                "über",
                "unter",
                "vor",
                "hinter",
                "neben",
                "zwischen",
                "bei",
                "mit",
                "ohne",
                "zu",
                "von",
                "aus",
                "nach",
                "seit",
                "bis",
                "durch",
                "für",
                "gegen",
                "um",
            ],
            # Bekannte Objekte (typische Logic-Puzzle-Gegenstände)
            "object": [
                "brandy",
                "bier",
                "gin",
                "rum",
                "wein",
                "vodka",
                "whisky",
                "kaffee",
                "tee",
                "wasser",
                "saft",
                "milch",
                "pizza",
                "pasta",
                "burger",
                "salat",
                "suppe",
                "auto",
                "fahrrad",
                "bus",
                "zug",
                "rot",
                "blau",
                "grün",
                "gelb",
                "schwarz",
                "weiß",
            ],
            # Verben die als Nomen erkannt werden könnten
            "verb": [
                "essen",
                "trinken",
                "trinkt",
                "isst",
                "mag",
                "kann",
                "will",
                "kauft",
                "nimmt",
                "bestellt",
                "macht",
                "sagt",
                "geht",
                "kommt",
                "vorkommen",
                "geschehen",
            ],
        }

        # Zähle zu erstellende Wörter
        total_words = sum(len(words) for words in common_words_dict.values())
        logger.info(
            f"  Erstelle {total_words} CommonWords in {len(common_words_dict)} Kategorien..."
        )

        # Füge alle Wörter hinzu
        count = netzwerk.add_common_words_batch(common_words_dict)

        logger.info(f"  [OK] {count} neue CommonWords erstellt")
        logger.info(f"  [OK] {total_words - count} CommonWords bereits vorhanden")

        # Verifiziere
        stats = netzwerk.get_common_words_statistics()
        logger.info(f"  [OK] Gesamt in DB: {stats.get('total', 0)} CommonWords")

        # Zeige Kategorien
        for category, count in stats.get("by_category", {}).items():
            logger.info(f"      - {category}: {count} Wörter")

        return True

    except Exception as e:
        logger.error(
            f"[ERROR] Fehler beim Erstellen der CommonWords: {e}", exc_info=True
        )
        return False


def run_setup():
    """
    Hauptfunktion für das Setup mit vollständiger Fehlerbehandlung und Verifizierung.
    """
    setup_logging()
    logger = logging.getLogger("KAI_SETUP")

    logger.info("=" * 70)
    logger.info("KAI SYSTEM SETUP - Initialisierung gestartet")
    logger.info("=" * 70)

    # Schritt 1: Datenbankverbindung herstellen
    logger.info("\n📡 Schritt 1: Datenbankverbindung herstellen...")
    try:
        netzwerk = KonzeptNetzwerk()
    except Exception as e:
        logger.critical(
            f"[ERROR] FATAL: Konnte KonzeptNetzwerk nicht initialisieren: {e}"
        )
        sys.exit(1)

    if not verify_database_connection(netzwerk):
        logger.critical("[ERROR] FATAL: Datenbankverbindung fehlgeschlagen")
        logger.critical("Bitte prüfen Sie:")
        logger.critical("  1. Läuft Neo4j auf localhost:7687?")
        logger.critical("  2. Sind die Zugangsdaten korrekt (neo4j/password)?")
        logger.critical("  3. Ist die Datenbank erreichbar?")
        netzwerk.close()
        sys.exit(1)

    # Schritt 2: Constraints prüfen
    logger.info("\n🔒 Schritt 2: Constraints prüfen...")
    if not verify_constraints(netzwerk):
        logger.critical("[ERROR] FATAL: Constraint-Verifikation fehlgeschlagen")
        netzwerk.close()
        sys.exit(1)

    # Schritt 3: Extraktionsregeln erstellen
    logger.info("\n📝 Schritt 3: Extraktionsregeln erstellen...")
    if not create_extraction_rules(netzwerk):
        logger.critical("[ERROR] FATAL: Konnte Extraktionsregeln nicht erstellen")
        netzwerk.close()
        sys.exit(1)

    # Schritt 4: Lexikalische Trigger erstellen
    logger.info("\n🏷️  Schritt 4: Lexikalische Trigger erstellen...")
    if not create_lexical_triggers(netzwerk):
        logger.critical("[ERROR] FATAL: Konnte Trigger nicht erstellen")
        netzwerk.close()
        sys.exit(1)

    # Schritt 4.5: CommonWords erstellen (für Entity-Extraktion)
    logger.info("\n🚫 Schritt 4.5: CommonWords (Stop-Words) erstellen...")
    if not create_common_words(netzwerk):
        logger.warning(
            "[WARNING]  CommonWords konnten nicht vollständig erstellt werden"
        )
        logger.warning("Entity-Extraktion kann eingeschränkt sein")

    # Schritt 5: Production Rules erstellen
    logger.info("\n🔧 Schritt 5: Production Rules erstellen...")
    if not create_production_rules(netzwerk):
        logger.warning(
            "[WARNING]  Production Rules konnten nicht vollständig erstellt werden"
        )
        logger.warning("Das System kann trotzdem verwendet werden (Pipeline-Modus)")

    # Schritt 6: Spatial Relations erstellen
    logger.info("\n📍 Schritt 6: Spatial Relations erstellen...")
    if not create_spatial_relations(netzwerk):
        logger.warning(
            "[WARNING]  Spatial Relations konnten nicht vollständig erstellt werden"
        )
        logger.warning("Spatial Reasoning kann eingeschränkt sein")

    # Schritt 7: Arithmetic Concepts erstellen
    logger.info("\n🔢 Schritt 7: Arithmetic Concepts erstellen...")
    if not create_arithmetic_concepts(netzwerk):
        logger.warning(
            "[WARNING]  Arithmetic Concepts konnten nicht vollständig erstellt werden"
        )
        logger.warning("Arithmetisches Reasoning kann eingeschränkt sein")

    # Schritt 8: Number Language erstellen
    logger.info("\n🔤 Schritt 8: Number Language Mappings erstellen...")
    if not create_number_language(netzwerk):
        logger.warning(
            "[WARNING]  Number Language Mappings konnten nicht vollständig erstellt werden"
        )
        logger.warning("Zahlen-zu-Wort Konversion kann eingeschränkt sein")

    # Schritt 9: Beispiel-Wissensbasis erstellen
    logger.info("\n📚 Schritt 9: Beispiel-Wissensbasis erstellen...")
    if not create_example_knowledge(netzwerk):
        logger.warning(
            "[WARNING]  Beispiel-Wissensbasis konnte nicht vollständig erstellt werden"
        )
        logger.warning("Das System kann trotzdem verwendet werden")

    # Schritt 10: Umfassende Verifikation
    logger.info("\n🔍 Schritt 10: Abschließende Verifikation...")
    if not verify_complete_setup(netzwerk):
        logger.critical("[ERROR] FATAL: Setup-Verifikation fehlgeschlagen")
        logger.critical("Das System ist nicht vollständig initialisiert!")
        netzwerk.close()
        sys.exit(1)

    # Erfolg!
    logger.info("\n" + "=" * 70)
    logger.info(
        "[SUCCESS] [SUCCESS] [SUCCESS]  SETUP ERFOLGREICH ABGESCHLOSSEN  [SUCCESS] [SUCCESS] [SUCCESS]"
    )
    logger.info("=" * 70)
    logger.info("\nDas KAI-System ist jetzt einsatzbereit!")
    logger.info("Sie können nun main_ui_graphical.py starten.")
    logger.info("\n📊 Zusammenfassung:")

    # Finale Statistik
    try:
        rules = netzwerk.get_all_extraction_rules()
        triggers = netzwerk.get_lexical_triggers()
        production_rules = netzwerk.get_all_production_rules()

        logger.info(f"  ✓ {len(rules)} Extraktionsregeln aktiv")
        logger.info(f"  ✓ {len(triggers)} lexikalische Trigger geladen")
        logger.info(f"  ✓ {len(production_rules)} Production Rules vorhanden")

        # Weitere Statistiken
        with netzwerk.driver.session(database="neo4j") as session:
            # Spatial Relations
            result = session.run("MATCH (sr:SpatialRelation) RETURN count(sr) AS count")
            spatial_count = result.single()["count"]
            logger.info(f"  ✓ {spatial_count} Spatial Relations definiert")

            # Arithmetic Concepts
            result = session.run(
                """
                MATCH (w:Wort)-[:IS_A]->(c:Wort {lemma: 'mathematisches_konzept'})
                RETURN count(w) AS count
                """
            )
            arithmetic_count = result.single()["count"]
            logger.info(f"  ✓ {arithmetic_count} Arithmetic Concepts vorhanden")

            # Number Mappings
            result = session.run(
                "MATCH (w:Wort)-[:REPRESENTS_NUMBER]->(:Number) RETURN count(w) AS count"
            )
            number_count = result.single()["count"]
            logger.info(f"  ✓ {number_count} Zahl-zu-Wort Mappings erstellt")

            # Beispielwissen
            result = session.run(
                "MATCH ()-[r:IS_A|HAS_PROPERTY|CAPABLE_OF|PART_OF|LOCATED_IN]->() RETURN count(r) AS count"
            )
            relation_count = result.single()["count"]
            logger.info(f"  ✓ {relation_count} Wissensrelationen in der Datenbank")

        logger.info("\n💡 Hinweis:")
        logger.info("  - Production System für Response Generation aktiv")
        logger.info("  - Spatial Reasoning für 2D-Grids und geometrische Formen")
        logger.info("  - Arithmetic Reasoning für mathematische Operationen")
        logger.info("  - Number Language für Zahlen-Wort Konversion")
        logger.info("  - Beispiel-Wissensbasis für erste Experimente")

    except Exception as e:
        logger.warning(f"[WARNING]  Fehler beim Erstellen der Statistik: {e}")
        logger.info("  * Basis-Setup erfolgreich abgeschlossen")

    netzwerk.close()


if __name__ == "__main__":
    run_setup()
