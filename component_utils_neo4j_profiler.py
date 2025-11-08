# component_utils_neo4j_profiler.py
"""
Neo4j Query Profiling und Optimierungs-Utilities

Dieses Modul bietet Tools zur Analyse und Optimierung von Neo4j Queries:
- EXPLAIN PLAN Analyse für teure Queries
- Index-Empfehlungen
- Performance-Metriken
- Query-Optimierungsvorschläge
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from neo4j import Driver

from component_15_logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class QueryProfile:
    """Profiling-Ergebnisse für eine Query"""

    query_name: str
    query_text: str
    execution_time_ms: float
    db_hits: int
    rows_returned: int
    plan_summary: str
    recommendations: List[str]
    has_index_scan: bool
    has_label_scan: bool
    estimated_rows: int


class Neo4jQueryProfiler:
    """
    Profiler für Neo4j Queries mit EXPLAIN/PROFILE Analyse.

    Verwendung:
        profiler = Neo4jQueryProfiler(driver)
        profile = profiler.profile_query("query_name", "MATCH (n:Wort) RETURN n")
        profiler.print_recommendations(profile)
    """

    def __init__(self, driver: Driver):
        self.driver = driver

    def profile_query(
        self, query_name: str, query: str, parameters: Optional[Dict[str, Any]] = None
    ) -> QueryProfile:
        """
        Führt PROFILE-Analyse für eine Query durch.

        Args:
            query_name: Name der Query (für Logging)
            query: Die zu analysierende Cypher-Query
            parameters: Optional - Parameter für die Query

        Returns:
            QueryProfile mit detaillierten Metriken
        """
        if parameters is None:
            parameters = {}

        logger.info(f"Profiliere Query: {query_name}")

        with self.driver.session(database="neo4j") as session:
            # PROFILE gibt tatsächliche Metriken zurück (führt Query aus)
            profiled_query = f"PROFILE {query}"
            result = session.run(profiled_query, parameters)

            # Extrahiere Plan und Metriken
            summary = result.consume()
            plan = summary.profile

            # Analysiere Plan
            db_hits = self._extract_db_hits(plan)
            has_index_scan = self._has_operation(plan, "NodeIndexSeek")
            has_label_scan = self._has_operation(plan, "NodeByLabelScan")
            estimated_rows = plan.get("rows", 0) if plan else 0

            # Generiere Empfehlungen
            recommendations = self._generate_recommendations(
                query, has_index_scan, has_label_scan, db_hits
            )

            # Execution Time (in ms)
            exec_time = summary.result_available_after + summary.result_consumed_after

            profile = QueryProfile(
                query_name=query_name,
                query_text=query,
                execution_time_ms=exec_time,
                db_hits=db_hits,
                rows_returned=summary.counters.nodes_created
                + summary.counters.relationships_created,
                plan_summary=str(plan),
                recommendations=recommendations,
                has_index_scan=has_index_scan,
                has_label_scan=has_label_scan,
                estimated_rows=estimated_rows,
            )

            logger.info(
                "Query-Profiling abgeschlossen",
                extra={
                    "query_name": query_name,
                    "execution_time_ms": exec_time,
                    "db_hits": db_hits,
                    "has_index": has_index_scan,
                },
            )

            return profile

    def explain_query(
        self, query_name: str, query: str, parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Führt EXPLAIN-Analyse durch (ohne Query auszuführen).

        Args:
            query_name: Name der Query
            query: Die zu analysierende Cypher-Query
            parameters: Optional - Parameter für die Query

        Returns:
            Dict mit Plan-Informationen
        """
        if parameters is None:
            parameters = {}

        logger.info(f"Erkläre Query: {query_name}")

        with self.driver.session(database="neo4j") as session:
            explained_query = f"EXPLAIN {query}"
            result = session.run(explained_query, parameters)

            summary = result.consume()
            plan = summary.plan

            return {
                "query_name": query_name,
                "plan": plan,
                "has_index_scan": self._has_operation(plan, "NodeIndexSeek"),
                "has_label_scan": self._has_operation(plan, "NodeByLabelScan"),
            }

    def _extract_db_hits(self, plan: Optional[Dict[str, Any]]) -> int:
        """Extrahiert DB-Hits aus Plan"""
        if not plan:
            return 0

        db_hits = plan.get("dbHits", 0)

        # Rekursiv für Kinder
        children = plan.get("children", [])
        for child in children:
            db_hits += self._extract_db_hits(child)

        return db_hits

    def _has_operation(self, plan: Optional[Dict[str, Any]], operation: str) -> bool:
        """Prüft ob Plan eine bestimmte Operation enthält"""
        if not plan:
            return False

        if plan.get("operatorType") == operation:
            return True

        # Rekursiv für Kinder
        children = plan.get("children", [])
        for child in children:
            if self._has_operation(child, operation):
                return True

        return False

    def _generate_recommendations(
        self, query: str, has_index_scan: bool, has_label_scan: bool, db_hits: int
    ) -> List[str]:
        """Generiert Optimierungsempfehlungen"""
        recommendations = []

        # Empfehlung 1: Fehlende Indizes
        if has_label_scan and not has_index_scan:
            recommendations.append(
                "[WARNING] Label Scan erkannt - Erwäge Index auf häufig abgefragten Properties"
            )

        # Empfehlung 2: Hohe DB-Hits
        if db_hits > 10000:
            recommendations.append(
                f"[WARNING] Hohe DB-Hits ({db_hits}) - Query könnte optimiert werden"
            )

        # Empfehlung 3: MATCH ohne WHERE
        if "MATCH" in query and "WHERE" not in query and "{" not in query:
            recommendations.append(
                "💡 MATCH ohne Filter - Erwäge WHERE-Clause oder Property-Match"
            )

        # Empfehlung 4: OPTIONAL MATCH Overuse
        if query.count("OPTIONAL MATCH") > 2:
            recommendations.append(
                "💡 Viele OPTIONAL MATCH - Könnte Performance beeinträchtigen"
            )

        # Empfehlung 5: Cartesian Product
        if query.count("MATCH") > 1 and "WHERE" not in query:
            recommendations.append(
                "[WARNING] Mehrere MATCH ohne WHERE - Risiko für Cartesian Product"
            )

        if not recommendations:
            recommendations.append(
                "[SUCCESS] Keine offensichtlichen Optimierungen nötig"
            )

        return recommendations

    def print_recommendations(self, profile: QueryProfile):
        """Gibt Profiling-Ergebnisse formatiert aus"""
        print(f"\n{'=' * 70}")
        print(f"Query Profile: {profile.query_name}")
        print(f"{'=' * 70}")
        print(f"Execution Time: {profile.execution_time_ms:.2f} ms")
        print(f"DB Hits: {profile.db_hits}")
        print(f"Rows Returned: {profile.rows_returned}")
        print(f"Has Index Scan: {profile.has_index_scan}")
        print(f"Has Label Scan: {profile.has_label_scan}")
        print("\nRecommendations:")
        for rec in profile.recommendations:
            print(f"  {rec}")
        print(f"{'=' * 70}\n")

    def profile_common_queries(self, netzwerk) -> List[QueryProfile]:
        """
        Profiliert häufig verwendete KAI-Queries.

        Args:
            netzwerk: KonzeptNetzwerk-Instanz (für Kontext)

        Returns:
            Liste von QueryProfile-Objekten
        """
        profiles = []

        # Query 1: Fakten-Suche (query_graph_for_facts)
        profiles.append(
            self.profile_query(
                "query_graph_for_facts",
                """
            MATCH (s:Konzept {name: $topic})-[r]->(o:Konzept)
            RETURN type(r) AS relation, o.name AS object
            """,
                {"topic": "hund"},  # Beispiel-Topic
            )
        )

        # Query 2: Inverse Relationen
        profiles.append(
            self.profile_query(
                "query_inverse_relations",
                """
            MATCH (s:Konzept)-[r]->(o:Konzept {name: $topic})
            RETURN type(r) AS relation, s.name AS subject
            """,
                {"topic": "säugetier"},
            )
        )

        # Query 3: Wort-Details
        profiles.append(
            self.profile_query(
                "get_details_fuer_wort",
                """
            MATCH (w:Wort {lemma: $lemma})
            OPTIONAL MATCH (w)-[:HAT_BEDEUTUNG]->(b:Bedeutung)
            WITH w, collect(b.text) as bedeutungen
            OPTIONAL MATCH (w)-[:TEIL_VON]->(sg:Synonymgruppe)<-[:TEIL_VON]-(syn:Wort) WHERE syn <> w
            WITH w, bedeutungen, collect(DISTINCT syn.lemma) as synonyme
            OPTIONAL MATCH (w)-[:BEDEUTET]->(k:Konzept)
            OPTIONAL MATCH (k)-[:IST_EINE*]->(superclass:Konzept) WHERE superclass <> k
            WITH w, bedeutungen, synonyme, k, collect(DISTINCT superclass.name) as superclasses
            RETURN
                w.lemma as lemma,
                bedeutungen,
                synonyme,
                { name: k.name, superclasses: superclasses } as konzept
            """,
                {"lemma": "hund"},
            )
        )

        # Query 4: Alle Extraktionsregeln
        profiles.append(
            self.profile_query(
                "get_all_extraction_rules",
                """
            MATCH (r:ExtractionRule)
            RETURN r.relation_type AS relation_type, r.regex_pattern AS regex_pattern
            """,
            )
        )

        # Query 5: Pattern Prototypen
        profiles.append(
            self.profile_query(
                "get_all_prototypes",
                """
            MATCH (p:PatternPrototype)
            RETURN p.id AS id, p.vector AS vector, p.example_count AS example_count
            """,
            )
        )

        # Query 6: Word Frequency
        profiles.append(
            self.profile_query(
                "get_word_frequency",
                """
            OPTIONAL MATCH (w:Wort {lemma: $word})
            OPTIONAL MATCH (k:Konzept {name: $word})

            OPTIONAL MATCH (w)-[r_w_out]->()
            OPTIONAL MATCH ()-[r_w_in]->(w)
            OPTIONAL MATCH (k)-[r_k_out]->()
            OPTIONAL MATCH ()-[r_k_in]->(k)

            WITH
                count(DISTINCT r_w_out) + count(DISTINCT r_k_out) AS out_count,
                count(DISTINCT r_w_in) + count(DISTINCT r_k_in) AS in_count

            RETURN out_count, in_count, out_count + in_count AS total_count
            """,
                {"word": "hund"},
            )
        )

        return profiles

    def suggest_indexes(self) -> List[str]:
        """
        Schlägt nützliche Indizes für KAI vor.

        Returns:
            Liste von CREATE INDEX Statements
        """
        index_suggestions = [
            # Index auf Konzept.name für schnellere Fakten-Queries
            "CREATE INDEX konzept_name_index IF NOT EXISTS FOR (k:Konzept) ON (k.name)",
            # Index auf Wort.lemma (sollte bereits durch Constraint existieren)
            # "CREATE INDEX wort_lemma_index IF NOT EXISTS FOR (w:Wort) ON (w.lemma)",
            # Index auf Relation-Timestamps für zeitbasierte Queries
            # Hinweis: Neo4j unterstützt keine Property-Indizes auf Relationen in Community Edition
            # Alternative: Timestamp als Property auf Konzept-Knoten
            # Index auf Episode.episode_type für schnellere Episode-Queries
            "CREATE INDEX episode_type_index IF NOT EXISTS FOR (e:Episode) ON (e.episode_type)",
            # Index auf Hypothesis.status für schnellere Hypothesen-Queries
            "CREATE INDEX hypothesis_status_index IF NOT EXISTS FOR (h:Hypothesis) ON (h.status)",
        ]

        return index_suggestions

    def create_recommended_indexes(self):
        """Erstellt empfohlene Indizes"""
        suggestions = self.suggest_indexes()

        logger.info(f"Erstelle {len(suggestions)} empfohlene Indizes")

        with self.driver.session(database="neo4j") as session:
            for index_stmt in suggestions:
                try:
                    session.run(index_stmt)
                    logger.info(f"Index erstellt: {index_stmt[:50]}...")
                except Exception as e:
                    logger.warning(f"Index-Erstellung fehlgeschlagen: {e}")

        logger.info("Index-Optimierung abgeschlossen")


# CLI-Interface für direktes Ausführen
if __name__ == "__main__":
    from component_1_netzwerk import KonzeptNetzwerk

    print("Neo4j Query Profiler für KAI")
    print("=" * 70)

    # Initialisiere Netzwerk
    netzwerk = KonzeptNetzwerk()

    # Erstelle Profiler
    profiler = Neo4jQueryProfiler(netzwerk.driver)

    # Profiliere häufige Queries
    print("\n📊 Profiliere häufige KAI-Queries...\n")
    profiles = profiler.profile_common_queries(netzwerk)

    # Zeige Ergebnisse
    for profile in profiles:
        profiler.print_recommendations(profile)

    # Index-Empfehlungen
    print("\n💡 Index-Empfehlungen:")
    print("=" * 70)
    suggestions = profiler.suggest_indexes()
    for i, suggestion in enumerate(suggestions, 1):
        print(f"{i}. {suggestion}")

    print(
        "\n❓ Möchten Sie die empfohlenen Indizes erstellen? (Warnung: Kann Datenbank ändern)"
    )
    user_input = input("Eingabe (ja/nein): ").strip().lower()

    if user_input == "ja":
        profiler.create_recommended_indexes()
        print("[SUCCESS] Indizes erstellt")
    else:
        print("⏭️ Übersprungen")

    netzwerk.close()
    print("\n[SUCCESS] Profiling abgeschlossen")
