#!/usr/bin/env python3
# migrate_to_enhanced_schema.py
"""
Migration Script: Enhanced Graph Schema für KAI

Migriert bestehende Knowledge-Graph-Daten zu Enhanced Schema (Phase 1.2).

Was wird migriert:
1. Wort/Konzept Nodes: Neue Properties (pos, definitions, semantic_field, etc.)
2. Relations: Neue Properties (context, bidirectional, usage_count, etc.)
3. Aggregation: typical_relations aus bestehenden Relations
4. Contexts: Aus Episodes extrahieren

Features:
- Batch-Processing für Performance
- Progress-Tracking
- Dry-Run Mode für Testing
- Rollback-fähig (via Backup-Empfehlung)
"""

import argparse
import sys
from datetime import datetime
from typing import Any, Dict

# Import KAI modules
from component_1_netzwerk import KonzeptNetzwerk
from component_15_logging_config import get_logger
from component_48_enhanced_schema import (
    EnhancedRelationProperties,
    EnhancedSchemaManager,
)

logger = get_logger(__name__)


class SchemaMigration:
    """
    Manager für Schema-Migration.

    Koordiniert alle Migrations-Schritte und tracked Progress.
    """

    def __init__(self, netzwerk: KonzeptNetzwerk, dry_run: bool = False):
        """
        Initialisiert die Migration.

        Args:
            netzwerk: KonzeptNetzwerk-Instanz
            dry_run: Wenn True, werden keine Änderungen vorgenommen
        """
        self.netzwerk = netzwerk
        self.schema_manager = EnhancedSchemaManager(netzwerk)
        self.dry_run = dry_run

        self.stats = {
            "nodes_migrated": 0,
            "relations_migrated": 0,
            "typical_relations_updated": 0,
            "contexts_extracted": 0,
            "errors": 0,
        }

        logger.info(f"SchemaMigration initialisiert (dry_run={dry_run})")

    # ==================== STEP 1: MIGRATE NODES ====================

    def migrate_nodes(self, batch_size: int = 100) -> Dict[str, int]:
        """
        Migriert alle Wort/Konzept Nodes zu Enhanced Schema.

        Setzt Defaults für alle neuen Properties und inferiert intelligente Werte
        wo möglich (z.B. abstraction_level aus POS).

        Args:
            batch_size: Anzahl Nodes pro Batch

        Returns:
            Dict mit Statistiken
        """
        logger.info("=== STEP 1: Migrate Nodes ===")

        if self.dry_run:
            logger.info("[DRY RUN] Würde Nodes migrieren")
            return {"dry_run": True}

        try:
            with self.netzwerk.driver.session(database="neo4j") as session:
                # Zähle Nodes ohne enhanced properties
                count_result = session.run(
                    """
                    MATCH (k:Konzept)
                    WHERE k.abstraction_level IS NULL
                    RETURN count(k) AS total
                    """
                )
                total = count_result.single()["total"]

                logger.info(f"Migriere {total} Nodes...")

                # Batch-Migration
                offset = 0
                while offset < total:
                    result = session.run(
                        """
                        MATCH (k:Konzept)
                        WHERE k.abstraction_level IS NULL
                        WITH k LIMIT $batch_size
                        SET k.pos = NULL,
                            k.definitions = [],
                            k.semantic_field = NULL,
                            k.abstraction_level = 3,
                            k.contexts = [],
                            k.typical_relations = '{}',
                            k.usage_frequency = 0,
                            k.first_seen = COALESCE(k.first_seen, datetime({timezone: 'UTC'})),
                            k.last_used = NULL
                        RETURN count(k) AS migrated
                        """,
                        batch_size=batch_size,
                    )

                    migrated = result.single()["migrated"]
                    offset += migrated
                    self.stats["nodes_migrated"] += migrated

                    logger.info(
                        f"  Progress: {offset}/{total} nodes migrated "
                        f"({offset*100//total if total > 0 else 0}%)"
                    )

                    if migrated == 0:
                        break

            logger.info(
                f"✓ Nodes Migration abgeschlossen: {self.stats['nodes_migrated']} nodes"
            )
            return {"nodes_migrated": self.stats["nodes_migrated"]}

        except Exception as e:
            logger.error(f"Fehler bei Node-Migration: {e}", exc_info=True)
            self.stats["errors"] += 1
            return {"error": str(e)}

    # ==================== STEP 2: MIGRATE RELATIONS ====================

    def migrate_relations(self, batch_size: int = 100) -> Dict[str, int]:
        """
        Migriert alle Relations zu Enhanced Schema.

        Setzt Defaults für neue Properties und prüft ob bidirectional.

        Args:
            batch_size: Anzahl Relations pro Batch

        Returns:
            Dict mit Statistiken
        """
        logger.info("=== STEP 2: Migrate Relations ===")

        if self.dry_run:
            logger.info("[DRY RUN] Würde Relations migrieren")
            return {"dry_run": True}

        try:
            # Bidirectional Relations
            bidirectional_rels = EnhancedRelationProperties.BIDIRECTIONAL_RELATIONS

            with self.netzwerk.driver.session(database="neo4j") as session:
                # Zähle Relations ohne enhanced properties
                count_result = session.run(
                    """
                    MATCH ()-[r]->()
                    WHERE r.usage_count IS NULL
                    RETURN count(r) AS total
                    """
                )
                total = count_result.single()["total"]

                logger.info(f"Migriere {total} Relations...")

                # Batch-Migration
                offset = 0
                while offset < total:
                    result = session.run(
                        """
                        MATCH ()-[r]->()
                        WHERE r.usage_count IS NULL
                        WITH r LIMIT $batch_size
                        SET r.context = [],
                            r.bidirectional = CASE
                                WHEN type(r) IN $bidirectional_types THEN true
                                ELSE false
                            END,
                            r.inference_rule = NULL,
                            r.usage_count = 0,
                            r.last_reinforced = NULL
                        RETURN count(r) AS migrated
                        """,
                        batch_size=batch_size,
                        bidirectional_types=list(bidirectional_rels),
                    )

                    migrated = result.single()["migrated"]
                    offset += migrated
                    self.stats["relations_migrated"] += migrated

                    logger.info(
                        f"  Progress: {offset}/{total} relations migrated "
                        f"({offset*100//total if total > 0 else 0}%)"
                    )

                    if migrated == 0:
                        break

            logger.info(
                f"✓ Relations Migration abgeschlossen: {self.stats['relations_migrated']} relations"
            )
            return {"relations_migrated": self.stats["relations_migrated"]}

        except Exception as e:
            logger.error(f"Fehler bei Relation-Migration: {e}", exc_info=True)
            self.stats["errors"] += 1
            return {"error": str(e)}

    # ==================== STEP 3: AGGREGATE TYPICAL RELATIONS ====================

    def aggregate_typical_relations(self, batch_size: int = 50) -> Dict[str, int]:
        """
        Aggregiert typical_relations für alle Nodes aus bestehenden Relations.

        Zählt Häufigkeit jedes Relationstyps pro Node.

        Args:
            batch_size: Anzahl Nodes pro Batch

        Returns:
            Dict mit Statistiken
        """
        logger.info("=== STEP 3: Aggregate Typical Relations ===")

        if self.dry_run:
            logger.info("[DRY RUN] Würde typical_relations aggregieren")
            return {"dry_run": True}

        try:
            with self.netzwerk.driver.session(database="neo4j") as session:
                # Zähle Nodes
                count_result = session.run(
                    """
                    MATCH (k:Konzept)
                    RETURN count(k) AS total
                    """
                )
                total = count_result.single()["total"]

                logger.info(f"Aggregiere typical_relations für {total} Nodes...")

                # Batch-Aggregation
                offset = 0
                while offset < total:
                    result = session.run(
                        """
                        MATCH (k:Konzept)
                        WITH k SKIP $offset LIMIT $batch_size
                        OPTIONAL MATCH (k)-[r]->()
                        WITH k, type(r) AS rel_type, count(r) AS count
                        WHERE rel_type IS NOT NULL
                        WITH k, collect({type: rel_type, count: count}) AS relations
                        SET k.typical_relations = apoc.convert.toJson(
                            apoc.map.fromPairs([rel IN relations | [rel.type, rel.count]])
                        )
                        RETURN count(k) AS updated
                        """,
                        offset=offset,
                        batch_size=batch_size,
                    )

                    updated = result.single()["updated"] if result.single() else 0
                    offset += batch_size
                    self.stats["typical_relations_updated"] += updated

                    logger.info(
                        f"  Progress: {offset}/{total} nodes processed "
                        f"({offset*100//total if total > 0 else 0}%)"
                    )

            logger.info(
                f"✓ Typical Relations Aggregation abgeschlossen: "
                f"{self.stats['typical_relations_updated']} nodes"
            )
            return {
                "typical_relations_updated": self.stats["typical_relations_updated"]
            }

        except Exception as e:
            logger.error(
                f"Fehler bei Typical Relations Aggregation: {e}", exc_info=True
            )
            self.stats["errors"] += 1

            # Fallback: Ohne apoc
            logger.info("Versuche Aggregation ohne apoc...")
            return self._aggregate_typical_relations_fallback(batch_size)

    def _aggregate_typical_relations_fallback(
        self, batch_size: int = 50
    ) -> Dict[str, int]:
        """Fallback für Aggregation ohne APOC."""
        try:
            import json

            with self.netzwerk.driver.session(database="neo4j") as session:
                # Hole alle Nodes
                nodes_result = session.run("MATCH (k:Konzept) RETURN k.name AS name")
                nodes = [record["name"] for record in nodes_result]

                for i, node_name in enumerate(nodes):
                    # Zähle Relations für diesen Node
                    rel_result = session.run(
                        """
                        MATCH (k:Konzept {name: $name})-[r]->()
                        RETURN type(r) AS rel_type, count(r) AS count
                        """,
                        name=node_name,
                    )

                    distribution = {}
                    for record in rel_result:
                        distribution[record["rel_type"]] = record["count"]

                    # Update Node
                    if distribution:
                        session.run(
                            """
                            MATCH (k:Konzept {name: $name})
                            SET k.typical_relations = $distribution_json
                            """,
                            name=node_name,
                            distribution_json=json.dumps(distribution),
                        )
                        self.stats["typical_relations_updated"] += 1

                    if (i + 1) % batch_size == 0:
                        logger.info(
                            f"  Progress: {i+1}/{len(nodes)} nodes processed "
                            f"({(i+1)*100//len(nodes)}%)"
                        )

            logger.info(
                f"✓ Typical Relations Aggregation (Fallback) abgeschlossen: "
                f"{self.stats['typical_relations_updated']} nodes"
            )
            return {
                "typical_relations_updated": self.stats["typical_relations_updated"]
            }

        except Exception as e:
            logger.error(f"Fehler bei Fallback-Aggregation: {e}", exc_info=True)
            return {"error": str(e)}

    # ==================== STEP 4: EXTRACT CONTEXTS FROM EPISODES ====================

    def extract_contexts_from_episodes(self, batch_size: int = 100) -> Dict[str, int]:
        """
        Extrahiert Contexts aus Episodes und fügt sie zu Nodes hinzu.

        Aggregiert Episode-Inhalte als Kontexte für verwendete Konzepte.

        Args:
            batch_size: Anzahl Episodes pro Batch

        Returns:
            Dict mit Statistiken
        """
        logger.info("=== STEP 4: Extract Contexts from Episodes ===")

        if self.dry_run:
            logger.info("[DRY RUN] Würde Contexts aus Episodes extrahieren")
            return {"dry_run": True}

        try:
            with self.netzwerk.driver.session(database="neo4j") as session:
                # Zähle Episodes
                count_result = session.run("MATCH (e:Episode) RETURN count(e) AS total")
                total = count_result.single()["total"]

                if total == 0:
                    logger.info("Keine Episodes gefunden, überspringe Schritt")
                    return {"contexts_extracted": 0}

                logger.info(f"Extrahiere Contexts aus {total} Episodes...")

                # Batch-Extraktion
                offset = 0
                while offset < total:
                    result = session.run(
                        """
                        MATCH (e:Episode)-[:LEARNED_FACT]->(f:Fact)
                        WITH e, f SKIP $offset LIMIT $batch_size
                        MATCH (k:Konzept)
                        WHERE k.name = f.subject OR k.name = f.object
                        WITH k, e.content AS context
                        WHERE context IS NOT NULL
                        SET k.contexts = CASE
                            WHEN k.contexts IS NULL THEN [context]
                            WHEN NOT context IN k.contexts AND size(k.contexts) < 10
                                THEN k.contexts + context
                            ELSE k.contexts
                        END
                        RETURN count(DISTINCT k) AS updated
                        """,
                        offset=offset,
                        batch_size=batch_size,
                    )

                    updated = result.single()["updated"] if result.single() else 0
                    offset += batch_size
                    self.stats["contexts_extracted"] += updated

                    logger.info(
                        f"  Progress: {offset}/{total} episodes processed "
                        f"({offset*100//total if total > 0 else 0}%)"
                    )

            logger.info(
                f"✓ Context Extraction abgeschlossen: "
                f"{self.stats['contexts_extracted']} contexts extracted"
            )
            return {"contexts_extracted": self.stats["contexts_extracted"]}

        except Exception as e:
            logger.error(f"Fehler bei Context-Extraktion: {e}", exc_info=True)
            self.stats["errors"] += 1
            return {"error": str(e)}

    # ==================== MAIN MIGRATION ====================

    def run_migration(self) -> Dict[str, Any]:
        """
        Führt vollständige Migration aus.

        Returns:
            Dict mit Statistiken und Erfolg/Fehler
        """
        start_time = datetime.now()

        logger.info("=" * 60)
        logger.info("STARTE ENHANCED SCHEMA MIGRATION")
        logger.info(f"Dry Run: {self.dry_run}")
        logger.info("=" * 60)

        # Step 1: Migrate Nodes
        self.migrate_nodes()

        # Step 2: Migrate Relations
        self.migrate_relations()

        # Step 3: Aggregate Typical Relations
        self.aggregate_typical_relations()

        # Step 4: Extract Contexts from Episodes
        self.extract_contexts_from_episodes()

        # Summary
        duration = (datetime.now() - start_time).total_seconds()

        logger.info("=" * 60)
        logger.info("MIGRATION ABGESCHLOSSEN")
        logger.info("=" * 60)
        logger.info(f"Dauer: {duration:.2f} Sekunden")
        logger.info(f"Nodes migriert: {self.stats['nodes_migrated']}")
        logger.info(f"Relations migriert: {self.stats['relations_migrated']}")
        logger.info(
            f"Typical Relations aktualisiert: {self.stats['typical_relations_updated']}"
        )
        logger.info(f"Contexts extrahiert: {self.stats['contexts_extracted']}")
        logger.info(f"Fehler: {self.stats['errors']}")

        return {
            "success": self.stats["errors"] == 0,
            "duration_seconds": duration,
            "stats": self.stats,
        }


# ==================== CLI ====================


def main():
    """CLI Entry Point."""
    parser = argparse.ArgumentParser(
        description="Migriere KAI Knowledge-Graph zu Enhanced Schema"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry-Run Mode: Keine Änderungen vornehmen",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch-Größe für Verarbeitung (Default: 100)",
    )

    args = parser.parse_args()

    # Banner
    print("=" * 60)
    print("KAI Enhanced Schema Migration")
    print("Phase 1.2: Cognitive Resonance")
    print("=" * 60)
    print()

    if args.dry_run:
        print("⚠️  DRY RUN MODE - Keine Änderungen werden vorgenommen")
        print()

    # Warnung
    if not args.dry_run:
        print("⚠️  WARNUNG: Diese Migration ändert die Datenbankstruktur!")
        print("   Empfehlung: Erstelle zuerst ein Backup der Neo4j-Datenbank.")
        print()
        response = input("Fortfahren? (yes/no): ")
        if response.lower() not in ["yes", "y"]:
            print("Migration abgebrochen.")
            sys.exit(0)

    # Initialisiere Netzwerk
    print("\nVerbinde mit Neo4j...")
    netzwerk = KonzeptNetzwerk()

    # Erstelle Migration
    migration = SchemaMigration(netzwerk, dry_run=args.dry_run)

    # Run Migration
    result = migration.run_migration()

    # Exit Code
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
