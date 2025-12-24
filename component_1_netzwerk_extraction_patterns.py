# component_1_netzwerk_patterns.py
"""
Pattern learning and extraction rule management.

This module handles:
- Extraction rule creation and retrieval
- Pattern prototype management (CRUD operations)
- Lexical trigger management
- Linking prototypes to rules
"""

from typing import Any, Dict, List, Optional

from neo4j import Driver

from component_15_logging_config import get_logger
from component_27_regex_validator import get_regex_validator
from infrastructure.cache_manager import cache_manager

logger = get_logger(__name__)


class KonzeptNetzwerkPatterns:
    """
    Pattern learning and extraction rule operations.

    Performance-Optimierung:
    - TTL-Cache für Extraktionsregeln (10 Minuten TTL, da sich diese selten ändern)
    """

    def __init__(self, driver: Driver):
        """
        Initialize with an existing Neo4j driver.

        Args:
            driver: Neo4j driver instance from KonzeptNetzwerkCore
        """
        self.driver = driver

        # Cache für Extraktionsregeln (10 Minuten TTL, da sich diese selten ändern) via CacheManager
        cache_manager.register_cache("extraction_rules", maxsize=50, ttl=600)

    def create_extraction_rule(self, relation_type: str, regex_pattern: str) -> bool:
        """
        Erstellt oder aktualisiert eine Extraktionsregel im Graphen.

        KORRIGIERT: Robuste Fehlerbehandlung und explizite Verifikation der Persistierung.
        NEU: Inline-Validierung des Regex-Musters vor dem Speichern.

        Args:
            relation_type: Der Typ der Relation (z.B. 'IS_A', 'HAS_PROPERTY')
            regex_pattern: Das Regex-Pattern zur Extraktion

        Returns:
            bool: True wenn erfolgreich erstellt/aktualisiert, False bei Fehler

        Raises:
            Exception: Bei kritischen Datenbankfehlern
        """
        if not self.driver:
            logger.error("create_extraction_rule: Kein DB-Driver verfügbar")
            return False

        # INLINE-VALIDIERUNG: Prüfe Regex-Muster vor dem Speichern
        validator = get_regex_validator()
        is_valid, error_msg, details = validator.validate_pattern(regex_pattern)

        if not is_valid:
            logger.error(
                f"create_extraction_rule: Regex-Validierung fehlgeschlagen für '{relation_type}'",
                extra={"error": error_msg, "pattern": regex_pattern},
            )
            # Gebe nutzerfreundliche Fehlermeldung zurück
            # (Der Aufrufer sollte diese loggen/anzeigen)
            return False

        # Log Warnungen (falls vorhanden)
        warnings = details.get("warnings", [])
        if warnings:
            logger.warning(
                f"create_extraction_rule: Regex-Warnungen für '{relation_type}'",
                extra={"warnings": warnings, "pattern": regex_pattern},
            )

        try:
            # FIX: Explizite Transaktion für atomare CREATE + VERIFY Operation (Code Review 2025-11-21, Issue 3)
            with self.driver.session(database="neo4j") as session:
                with session.begin_transaction() as tx:
                    # SCHRITT 1: Erstelle/Aktualisiere die Regel mit explizitem Return
                    result = tx.run(
                        """
                        MERGE (r:ExtractionRule {relation_type: $rel_type})
                        ON CREATE SET
                            r.regex_pattern = $regex,
                            r.created_at = timestamp(),
                            r.updated_at = timestamp()
                        ON MATCH SET
                            r.regex_pattern = $regex,
                            r.updated_at = timestamp()
                        RETURN r.relation_type AS type,
                            r.regex_pattern AS pattern,
                            r.created_at AS created,
                            r.updated_at AS updated
                        """,
                        rel_type=relation_type,
                        regex=regex_pattern,
                    )

                    record = result.single()

                    # SCHRITT 2: Verifikation der Rückgabe
                    if not record:
                        logger.error(
                            f"create_extraction_rule: Keine Rückgabe für Regel '{relation_type}'. "
                            "Transaktion möglicherweise fehlgeschlagen."
                        )
                        return False

                    # SCHRITT 3: Validiere die gespeicherten Daten
                    if record["type"] != relation_type:
                        logger.error(
                            f"create_extraction_rule: Relation-Type Mismatch. "
                            f"Erwartet: '{relation_type}', Gespeichert: '{record['type']}'"
                        )
                        return False

                    if record["pattern"] != regex_pattern:
                        logger.error(
                            f"create_extraction_rule: Pattern Mismatch für '{relation_type}'. "
                            f"Erwartet: '{regex_pattern}', Gespeichert: '{record['pattern']}'"
                        )
                        return False

                    # SCHRITT 4: Explizite Commit-Verifikation durch zweiten Read
                    # Dies stellt sicher, dass die Daten wirklich persistiert wurden
                    verify_result = tx.run(
                        """
                        MATCH (r:ExtractionRule {relation_type: $rel_type})
                        RETURN count(r) AS count, r.regex_pattern AS pattern
                        """,
                        rel_type=relation_type,
                    )

                    verify_record = verify_result.single()

                    if not verify_record or verify_record["count"] != 1:
                        logger.error(
                            f"create_extraction_rule: Verifikation fehlgeschlagen für '{relation_type}'. "
                            f"Regel nicht in DB gefunden nach Commit."
                        )
                        return False

                    if verify_record["pattern"] != regex_pattern:
                        logger.error(
                            f"create_extraction_rule: Verifikation zeigt falsches Pattern für '{relation_type}'"
                        )
                        return False

                    # ERFOLG - Commit Transaktion
                    tx.commit()

                    action = (
                        "aktualisiert"
                        if record["created"] != record["updated"]
                        else "erstellt"
                    )
                    logger.info(
                        f"Extraktionsregel '{relation_type}' erfolgreich {action} und verifiziert. "
                        f"Pattern: {regex_pattern[:50]}..."
                    )

                    # Invalidiere Cache nach Änderung
                    cache_manager.invalidate("extraction_rules")
                    logger.debug(
                        "Extraktionsregeln-Cache invalidiert nach Regel-Änderung"
                    )

                    return True

        except Exception as e:
            logger.error(
                f"create_extraction_rule: Kritischer Fehler bei '{relation_type}': {e}",
                exc_info=True,
            )
            return False

    def get_all_pattern_prototypes(
        self, category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Holt alle Pattern-Prototypen aus dem Graphen.

        Args:
            category: Optional - filtert nach dieser Kategorie

        Returns:
            Liste von Prototyp-Dictionaries
        """
        if not self.driver:
            return []

        with self.driver.session(database="neo4j") as session:
            if category:
                result = session.run(
                    """
                    MATCH (p:PatternPrototype {category: $cat})
                    RETURN p.id AS id, p.centroid AS centroid, p.variance AS variance,
                        p.count AS count, p.category AS category
                    """,
                    cat=category.upper(),
                )
            else:
                result = session.run(
                    """
                    MATCH (p:PatternPrototype)
                    RETURN p.id AS id, p.centroid AS centroid, p.variance AS variance,
                        p.count AS count, p.category AS category
                    """
                )
            return [record.data() for record in result]

    def create_pattern_prototype(
        self, initial_vector: List[float], category: str
    ) -> Optional[str]:
        if not self.driver:
            return None
        initial_variance = [0.0] * len(initial_vector)
        with self.driver.session(database="neo4j") as session:
            result = session.run(
                """
                CREATE (p:PatternPrototype {
                    id: randomUUID(),
                    centroid: $vector,
                    variance: $variance,
                    count: 1,
                    created_at: timestamp(),
                    category: $category
                })
                RETURN p.id AS id
                """,
                vector=initial_vector,
                variance=initial_variance,
                category=category,
            )
            record = result.single()
            return str(record["id"]) if record else None

    def update_pattern_prototype(
        self,
        prototype_id: str,
        new_centroid: List[float],
        new_variance: List[float],
        new_count: int,
    ) -> bool:
        """
        Aktualisiert einen bestehenden Pattern-Prototyp.

        FIX: Return-Type und Verifikation hinzugefügt (Code Review 2025-11-21, Concern 6)

        Returns:
            True wenn erfolgreich aktualisiert, False bei Fehler
        """
        if not self.driver:
            logger.warning("update_pattern_prototype: Kein Driver verfügbar")
            return False

        with self.driver.session(database="neo4j") as session:
            # SCHRITT 1: Update mit RETURN für Verifikation
            result = session.run(
                """
                MATCH (p:PatternPrototype {id: $id})
                SET p.centroid = $centroid,
                    p.variance = $variance,
                    p.count = $count,
                    p.updated_at = timestamp()
                RETURN p.id AS id, p.count AS updated_count
                """,
                id=prototype_id,
                centroid=new_centroid,
                variance=new_variance,
                count=new_count,
            )

            # SCHRITT 2: Verifikation
            record = result.single()
            if not record:
                logger.error(
                    f"update_pattern_prototype: Prototype '{prototype_id}' nicht gefunden oder Update fehlgeschlagen"
                )
                return False

            if record["updated_count"] != new_count:
                logger.warning(
                    f"update_pattern_prototype: Count-Mismatch für '{prototype_id}'. "
                    f"Erwartet: {new_count}, Gesetzt: {record['updated_count']}"
                )

            logger.debug(
                f"update_pattern_prototype: Prototype '{prototype_id}' erfolgreich aktualisiert"
            )
            return True

    def link_prototype_to_rule(self, prototype_id: str, relation_type: str) -> bool:
        if not self.driver:
            return False
        with self.driver.session(database="neo4j") as session:
            result = session.run(
                """
                MATCH (p:PatternPrototype {id: $p_id})
                MATCH (r:ExtractionRule {relation_type: $rel_type})
                MERGE (p)-[t:TRIGGERS]->(r)
                ON CREATE SET t.created_at = timestamp()
                RETURN t
                """,
                p_id=prototype_id,
                rel_type=relation_type,
            )
            # result.single() gibt den Trigger 't' zurück, wenn er erstellt wurde,
            # oder None, wenn die MATCH-Klauseln scheiterten.
            record = result.single()
            if record is None:
                logger.warning(
                    f"Konnte Prototyp '{prototype_id}' nicht mit Regel '{relation_type}' verknüpfen, "
                    "da einer der beiden Knoten nicht existiert."
                )
                return False
            return True

    def add_lexical_trigger(self, lemma: str, ensure_wort_callback) -> bool:
        """
        Markiert ein Wort als lexikalischen Trigger und gibt an, ob eine neue Verknüpfung erstellt wurde.

        Args:
            lemma: Das Lemma, das als Trigger markiert werden soll
            ensure_wort_callback: Callback-Funktion um ensure_wort_und_konzept aufzurufen
        """
        if not self.driver:
            return False
        if not ensure_wort_callback(lemma):
            return False
        try:
            with self.driver.session(database="neo4j") as session:
                result = session.run(
                    """
                    MATCH (w:Wort {lemma: $lemma})
                    MERGE (lex:Lexicon {name: 'triggers'})
                    MERGE (lex)-[r:CONTAINS]->(w)
                    RETURN r IS NOT NULL as success
                    """,
                    lemma=lemma.lower(),
                )
                record = result.single()
                return record["success"] if record else False
        except Exception as e:
            logger.error(f"Fehler in add_lexical_trigger für '{lemma}': {e}")
            return False

    def get_lexical_triggers(self) -> List[str]:
        if not self.driver:
            return []
        with self.driver.session(database="neo4j") as session:
            result = session.run(
                """
                MATCH (:Lexicon {name: 'triggers'})-[:CONTAINS]->(w:Wort)
                RETURN w.lemma AS trigger
            """
            )
            return [record["trigger"] for record in result]

    def get_rule_for_prototype(self, prototype_id: str) -> Optional[Dict[str, str]]:
        """
        NEU: Findet die Extraktionsregel, die von einem bestimmten Prototypen
        ausgelöst wird. Dies ist die entscheidende Brücke für die Ingestion.
        """
        if not self.driver:
            return None
        with self.driver.session(database="neo4j") as session:
            result = session.run(
                """
                MATCH (:PatternPrototype {id: $p_id})-[:TRIGGERS]->(r:ExtractionRule)
                RETURN r.relation_type as relation_type, r.regex_pattern as regex_pattern
                LIMIT 1
                """,
                p_id=prototype_id,
            )
            record = result.single()
            return record.data() if record else None

    def get_all_extraction_rules(self) -> List[Dict[str, str]]:
        """
        Holt alle Extraktionsregeln aus dem Graphen.

        Performance-Optimierung: Nutzt TTL-Cache (10 Minuten) da Extraktionsregeln
        selten ändern, aber häufig abgerufen werden (bei jedem Text-Ingestion).

        Returns:
            Liste von Regel-Dictionaries mit relation_type und regex_pattern
        """
        if not self.driver:
            return []

        # Cache-Key
        cache_key = "all_extraction_rules"

        # Prüfe Cache
        cached_rules = cache_manager.get("extraction_rules", cache_key)
        if cached_rules is not None:
            logger.debug("Cache-Hit für get_all_extraction_rules")
            return cached_rules

        with self.driver.session(database="neo4j") as session:
            result = session.run(
                "MATCH (r:ExtractionRule) RETURN r.relation_type AS relation_type, r.regex_pattern AS regex_pattern"
            )
            rules = [record.data() for record in result]

            # In Cache speichern
            cache_manager.set("extraction_rules", cache_key, rules)
            logger.debug(
                "Cache-Miss für get_all_extraction_rules",
                extra={"rule_count": len(rules)},
            )

            return rules
