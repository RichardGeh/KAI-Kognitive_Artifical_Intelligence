# component_1_netzwerk_core.py
"""
Core database operations and basic word/concept management.

This module contains the foundational functionality for KAI's knowledge graph:
- Neo4j connection management
- Word and concept creation
- Relations and fact assertions
- Fact queries and semantic similarity
"""

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

from cachetools import TTLCache
from neo4j import Driver, GraphDatabase

from component_15_logging_config import PerformanceLogger, get_logger
from kai_exceptions import Neo4jConnectionError, wrap_exception

logger = get_logger(__name__)

INFO_TYPE_ALIASES: Dict[str, str] = {
    "bedeutung": "bedeutung",
    "definition": "bedeutung",
    "synonym": "synonym",
}


class KonzeptNetzwerkCore:
    """
    Core functionality for Neo4j knowledge graph management.

    Performance-Optimierung:
    - TTL-Cache für Fakten-Queries (5 Minuten TTL, maxsize=500)
    - Cache für bekannte Wörter (10 Minuten TTL, maxsize=100)
    """

    def __init__(
        self,
        uri: str = "bolt://127.0.0.1:7687",
        user: str = "neo4j",
        password: str = "password",
    ):
        # Cache für Fakten-Queries (5 Minuten TTL, maxsize=500)
        self._fact_cache: TTLCache = TTLCache(
            maxsize=500, ttl=300
        )  # 300 Sekunden = 5 Minuten

        # Cache für bekannte Wörter (10 Minuten TTL, da sich diese seltener ändern)
        self._known_words_cache: TTLCache = TTLCache(
            maxsize=100, ttl=600
        )  # 600 Sekunden = 10 Minuten

        try:
            logger.info(
                "Initialisiere Neo4j-Verbindung", extra={"uri": uri, "user": user}
            )
            self.driver: Optional[Driver] = GraphDatabase.driver(
                uri, auth=(user, password)
            )
            self.driver.verify_connectivity()
            logger.info("Neo4j-Verbindung erfolgreich hergestellt", extra={"uri": uri})
            self._create_constraints()
            self._create_indexes()
        except Exception as e:
            raise wrap_exception(
                e,
                Neo4jConnectionError,
                "Konnte keine Verbindung zur Neo4j-DB herstellen",
                uri=uri,
                user=user,
            )

    def close(self):
        if self.driver:
            self.driver.close()

    def clear_cache(self, cache_type: Optional[str] = None):
        """
        Invalidiert Caches.

        Args:
            cache_type: Optional - 'facts', 'words', oder None für alle Caches
        """
        if cache_type == "facts" or cache_type is None:
            self._fact_cache.clear()
            logger.debug("Fakten-Cache geleert")

        if cache_type == "words" or cache_type is None:
            self._known_words_cache.clear()
            logger.debug("Wort-Cache geleert")

    def get_cache_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Gibt Statistiken über die Caches zurück.

        Returns:
            Dict mit Cache-Statistiken für verschiedene Cache-Typen
        """
        return {
            "fact_cache": {
                "size": len(self._fact_cache),
                "maxsize": self._fact_cache.maxsize,
                "ttl": self._fact_cache.ttl,
            },
            "known_words_cache": {
                "size": len(self._known_words_cache),
                "maxsize": self._known_words_cache.maxsize,
                "ttl": self._known_words_cache.ttl,
            },
        }

    def _create_constraints(self):
        if not self.driver:
            return

        logger.debug("Erstelle Neo4j Constraints")

        try:
            with self.driver.session(database="neo4j") as session:
                constraints: List[tuple[str, str]] = [
                    (
                        "WortLemma",
                        "CREATE CONSTRAINT WortLemma IF NOT EXISTS FOR (w:Wort) REQUIRE w.lemma IS UNIQUE",
                    ),
                    (
                        "KonzeptName",
                        "CREATE CONSTRAINT KonzeptName IF NOT EXISTS FOR (k:Konzept) REQUIRE k.name IS UNIQUE",
                    ),
                    (
                        "ExtractionRuleType",
                        "CREATE CONSTRAINT ExtractionRuleType IF NOT EXISTS FOR (r:ExtractionRule) REQUIRE r.relation_type IS UNIQUE",
                    ),
                    (
                        "PatternPrototypeId",
                        "CREATE CONSTRAINT PatternPrototypeId IF NOT EXISTS FOR (p:PatternPrototype) REQUIRE p.id IS UNIQUE",
                    ),
                    (
                        "LexiconName",
                        "CREATE CONSTRAINT LexiconName IF NOT EXISTS FOR (l:Lexicon) REQUIRE l.name IS UNIQUE",
                    ),
                    (
                        "EpisodeId",
                        "CREATE CONSTRAINT EpisodeId IF NOT EXISTS FOR (e:Episode) REQUIRE e.id IS UNIQUE",
                    ),
                    (
                        "InferenceEpisodeId",
                        "CREATE CONSTRAINT InferenceEpisodeId IF NOT EXISTS FOR (ie:InferenceEpisode) REQUIRE ie.id IS UNIQUE",
                    ),
                    (
                        "ProofStepId",
                        "CREATE CONSTRAINT ProofStepId IF NOT EXISTS FOR (ps:ProofStep) REQUIRE ps.id IS UNIQUE",
                    ),
                    (
                        "HypothesisId",
                        "CREATE CONSTRAINT HypothesisId IF NOT EXISTS FOR (h:Hypothesis) REQUIRE h.id IS UNIQUE",
                    ),
                    (
                        "AgentId",
                        "CREATE CONSTRAINT AgentId IF NOT EXISTS FOR (a:Agent) REQUIRE a.id IS UNIQUE",
                    ),
                    (
                        "BeliefId",
                        "CREATE CONSTRAINT BeliefId IF NOT EXISTS FOR (b:Belief) REQUIRE b.id IS UNIQUE",
                    ),
                    (
                        "MetaBeliefId",
                        "CREATE CONSTRAINT MetaBeliefId IF NOT EXISTS FOR (mb:MetaBelief) REQUIRE mb.id IS UNIQUE",
                    ),
                ]

                for constraint_name, query in constraints:
                    try:
                        session.run(query)
                        logger.debug(
                            f"Constraint '{constraint_name}' erstellt/verifiziert"
                        )
                    except Exception as e:
                        # Einzelner Constraint-Fehler ist nicht kritisch
                        logger.warning(
                            f"Constraint '{constraint_name}' konnte nicht erstellt werden",
                            extra={"error": str(e)},
                        )

            logger.info("Neo4j Constraints erfolgreich konfiguriert")
        except Exception:
            # Constraint-Fehler sind nicht kritisch (können bereits existieren)
            # Werfe keine Exception, nur loggen
            logger.error("Fehler beim Konfigurieren der Constraints", exc_info=True)

    def _create_indexes(self):
        """
        Erstellt Performance-Indizes für häufig verwendete Queries

        Indizes:
        - wort_lemma_index: Index auf Wort.lemma (bereits durch Constraint abgedeckt)
        - relation_confidence: Index auf relationship.confidence
        - relation_context: Index auf relationship.context
        """
        if not self.driver:
            return

        logger.debug("Erstelle Neo4j Performance-Indizes")

        try:
            with self.driver.session(database="neo4j") as session:
                # Note: wort_lemma_index wird automatisch durch UNIQUE Constraint erstellt

                # Indizes für Relationship Properties
                # Neo4j syntax variiert je nach Version, wir verwenden die moderne Syntax
                indexes: List[tuple[str, str]] = [
                    # Index für Confidence-Filter auf allen Relationships
                    (
                        "relation_confidence_index",
                        "CREATE INDEX relation_confidence_index IF NOT EXISTS "
                        "FOR ()-[r]-() ON (r.confidence)",
                    ),
                    # Index für Context-Filter auf allen Relationships
                    (
                        "relation_context_index",
                        "CREATE INDEX relation_context_index IF NOT EXISTS "
                        "FOR ()-[r]-() ON (r.context)",
                    ),
                ]

                for index_name, query in indexes:
                    try:
                        session.run(query)
                        logger.debug(f"Index '{index_name}' erstellt/verifiziert")
                    except Exception as e:
                        # Index-Fehler sind nicht kritisch
                        logger.warning(
                            f"Index '{index_name}' konnte nicht erstellt werden: {e}"
                        )

            logger.info("Neo4j Performance-Indizes erfolgreich konfiguriert")

        except Exception:
            # Index-Fehler sind nicht kritisch
            logger.error("Fehler beim Konfigurieren der Indizes", exc_info=True)

    def ensure_wort_und_konzept(self, lemma: str) -> bool:
        """
        Returns:
            bool: True bei Erfolg, False bei Fehler
        """
        if not self.driver:
            logger.error(
                "ensure_wort_und_konzept: Kein DB-Driver verfügbar",
                extra={"lemma": lemma},
            )
            return False

        lemma = lemma.lower()

        try:
            with PerformanceLogger(
                logger.logger, "ensure_wort_und_konzept", lemma=lemma
            ):
                with self.driver.session(database="neo4j") as session:
                    # Verwende explizite Transaktion
                    with session.begin_transaction() as tx:
                        result = tx.run(
                            """
                            MERGE (w:Wort {lemma: $l})
                            MERGE (k:Konzept {name: $l})
                            MERGE (w)-[:BEDEUTET]->(k)
                            RETURN w.lemma AS lemma
                            """,
                            l=lemma,
                        )
                        record = result.single()

                        # Expliziter Commit
                        tx.commit()

                        # Verifikation
                        if not record or record["lemma"] != lemma:
                            logger.error(
                                "ensure_wort_und_konzept: Verifikation fehlgeschlagen",
                                extra={"lemma": lemma},
                            )
                            return False

                        logger.debug(
                            "Wort und Konzept erfolgreich sichergestellt",
                            extra={"lemma": lemma},
                        )
                        return True

        except Exception as e:
            # Spezifische Exception für Write-Fehler
            logger.log_exception(e, "ensure_wort_und_konzept: Fehler", lemma=lemma)
            # Graceful Degradation: False zurückgeben
            return False

    def set_wort_attribut(
        self, lemma: str, attribut_name: str, attribut_wert: Any
    ) -> None:
        """Setzt ein Attribut direkt am :Wort-Knoten."""
        if not self.driver:
            return
        self.ensure_wort_und_konzept(lemma)
        safe_attribut_name: str = re.sub(r"[^a-zA-Z0-9_]", "", attribut_name.lower())
        if not safe_attribut_name:
            logger.warning(f"Ungültiger Attributname '{attribut_name}'.")
            return
        with self.driver.session(database="neo4j") as session:
            query = (
                f"MATCH (w:Wort {{lemma: $lemma}}) SET w.{safe_attribut_name} = $wert"
            )
            session.run(query, lemma=lemma.lower(), wert=attribut_wert)

    def add_information_zu_wort(
        self, lemma: str, info_typ: str, info_inhalt: str
    ) -> Dict[str, Any]:
        # Diese Methode orchestriert nur, die Fehlerbehandlung liegt in den _add-Methoden.
        # Wir stellen aber sicher, dass der erste Schritt erfolgreich ist.
        if not self.ensure_wort_und_konzept(lemma):
            return {"created": False, "error": "base_concept_creation_failed"}

        kanonischer_typ: Optional[str] = INFO_TYPE_ALIASES.get(info_typ.lower())

        if kanonischer_typ == "bedeutung":
            return self._add_bedeutung(lemma, info_inhalt)
        if kanonischer_typ == "synonym":
            return self._add_synonym(lemma, info_inhalt)

        logger.warning(f"Unbekannter oder nicht-kanonischer info_typ '{info_typ}'.")
        return {"created": False, "error": "unknown_type"}

    def _add_bedeutung(self, lemma: str, bedeutung_text: str) -> Dict[str, bool]:
        if not self.driver:
            return {"created": False}
        with self.driver.session(database="neo4j") as session:
            result = session.run(
                """
                WITH timestamp() as now
                MATCH (w:Wort {lemma: $lemma})
                MERGE (w)-[r:HAT_BEDEUTUNG]->(b:Bedeutung {text: $text})
                ON CREATE SET r.created_at = now
                RETURN r.created_at = now AS created
                """,
                lemma=lemma,
                text=bedeutung_text,
            )
            created = result.single(strict=True)["created"]
        return {"created": created}

    def _add_synonym(self, lemma1: str, lemma2: str) -> Dict[str, bool]:
        if not self.driver:
            return {"created": False}
        self.ensure_wort_und_konzept(lemma2)
        with self.driver.session(database="neo4j") as session:
            result = session.run(
                """
                WITH timestamp() as now
                MATCH (w1:Wort {lemma: $l1}), (w2:Wort {lemma: $l2})
                OPTIONAL MATCH (w1)-[:TEIL_VON]->(sg:Synonymgruppe)
                WITH w1, w2, head(collect(sg)) as existing_sg, now
                MERGE (sg_final:Synonymgruppe {id: coalesce(existing_sg.id, randomUUID())})
                MERGE (w1)-[r1:TEIL_VON]->(sg_final)
                MERGE (w2)-[r2:TEIL_VON]->(sg_final)
                ON CREATE SET r1.created_at = now, r2.created_at = now
                RETURN r1.created_at = now OR r2.created_at = now AS created
                """,
                l1=lemma1.lower(),
                l2=lemma2.lower(),
            )
            created = result.single(strict=True)["created"]
        return {"created": created}

    def get_details_fuer_wort(self, lemma: str) -> Optional[Dict[str, Any]]:
        if not self.driver:
            return None
        with self.driver.session(database="neo4j") as session:
            result = session.run(
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
                lemma=lemma.lower(),
            )
            record = result.single()
            return record.data() if record else None

    def assert_relation(
        self,
        subject: str,
        relation: str,
        object: str,
        source_sentence: Optional[str] = None,
    ) -> bool:
        """Erstellt eine "behauptete" Beziehung. Gibt True zurück, wenn neu erstellt."""
        if not self.driver:
            logger.error(
                "assert_relation: Kein DB-Driver verfügbar",
                extra={"subject": subject, "relation": relation, "object": object},
            )
            return False

        safe_relation: str = re.sub(r"[^a-zA-Z0-9_]", "", relation.upper())
        if not safe_relation:
            logger.error(
                "Ungültiger Relationstyp",
                extra={"relation": relation, "subject": subject, "object": object},
            )
            return False

        if not self.ensure_wort_und_konzept(
            subject
        ) or not self.ensure_wort_und_konzept(object):
            logger.warning(
                "Konnte Subject oder Object nicht sicherstellen",
                extra={"subject": subject, "object": object},
            )
            return False

        try:
            with PerformanceLogger(
                logger.logger,
                "assert_relation",
                subject=subject,
                relation=safe_relation,
                object=object,
            ):
                with self.driver.session(database="neo4j") as session:
                    result = session.run(
                        f"""
                        MATCH (s:Konzept {{name: $subject}})
                        MATCH (o:Konzept {{name: $object}})
                        MERGE (s)-[rel:{safe_relation}]->(o)
                        ON CREATE SET
                            rel.source_text = $source,
                            rel.asserted_at = timestamp(),
                            rel.timestamp = datetime({{timezone: 'UTC'}}),
                            rel.confidence = 0.85
                        // 'was_created' ist true, wenn das 'created_at' jetzt gesetzt wurde
                        RETURN rel.asserted_at = timestamp() AS was_created
                        """,
                        subject=subject.lower(),
                        object=object.lower(),
                        source=source_sentence,
                    )
                    record = result.single()
                    was_created: bool = record["was_created"] if record else False

                    if was_created:
                        # Invalidiere Cache für betroffenes Subject (neue Fakten hinzugefügt)
                        cache_key = f"facts:{subject.lower()}"
                        if cache_key in self._fact_cache:
                            del self._fact_cache[cache_key]
                            logger.debug(
                                "Cache invalidiert", extra={"subject": subject}
                            )

                        logger.info(
                            "Neue Relation erstellt",
                            extra={
                                "subject": subject,
                                "relation": safe_relation,
                                "object": object,
                                "source": source_sentence,
                            },
                        )
                    else:
                        logger.debug(
                            "Relation bereits vorhanden",
                            extra={
                                "subject": subject,
                                "relation": safe_relation,
                                "object": object,
                            },
                        )

                    return was_created

        except Exception as e:
            # Spezifische Exception für Write-Fehler
            logger.log_exception(
                e,
                "Fehler in assert_relation",
                subject=subject,
                relation=relation,
                object=object,
            )
            # Optional: raise wrap_exception(e, Neo4jWriteError, ...)
            # Aktuell: Graceful Degradation - gebe False zurück
            return False

    def query_graph_for_facts(self, topic: str) -> Dict[str, List[str]]:
        """
        Fragt den Graphen nach allen bekannten, ausgehenden Fakten (Beziehungen)
        für ein bestimmtes Thema ab und gruppiert sie.

        PHASE 5.3: Erweitert um bidirektionale Synonym-Suche via Synonymgruppen.

        Performance-Optimierung: Nutzt TTL-Cache (5 Minuten) für häufig abgefragte Topics.

        Hinweis: Diese Methode sucht nur Fakten über das direkt angefragte Thema.
        Für erweiterte Suche über alle Synonyme, verwende query_facts_with_synonyms().
        """
        if not self.driver:
            logger.error(
                "query_graph_for_facts: Kein DB-Driver verfügbar",
                extra={"topic": topic},
            )
            return {}

        # Cache-Key
        cache_key = f"facts:{topic.lower()}"

        # Prüfe Cache
        if cache_key in self._fact_cache:
            logger.debug("Cache-Hit für query_graph_for_facts", extra={"topic": topic})
            return self._fact_cache[cache_key]

        try:
            with PerformanceLogger(logger.logger, "query_graph_for_facts", topic=topic):
                with self.driver.session(database="neo4j") as session:
                    self.ensure_wort_und_konzept(topic)

                    # Haupt-Query: Alle ausgehenden Relationen
                    result = session.run(
                        """
                        MATCH (s:Konzept {name: $topic})-[r]->(o:Konzept)
                        RETURN type(r) AS relation, o.name AS object
                        """,
                        topic=topic.lower(),
                    )

                    facts: Dict[str, List[str]] = defaultdict(list)
                    for record in result:
                        facts[record["relation"]].append(record["object"])

                    # PHASE 5.3: Zusätzliche Synonym-Query (bidirektional)
                    # Findet alle Wörter in derselben Synonymgruppe
                    synonym_result = session.run(
                        """
                        MATCH (w:Wort {lemma: $topic})-[:TEIL_VON]->(sg:Synonymgruppe)<-[:TEIL_VON]-(syn:Wort)
                        WHERE syn.lemma <> $topic
                        RETURN DISTINCT syn.lemma AS synonym
                        """,
                        topic=topic.lower(),
                    )

                    synonyms: List[str] = [
                        record["synonym"] for record in synonym_result
                    ]
                    if synonyms:
                        # Überschreibe oder merge mit existierenden TEIL_VON Fakten
                        facts["TEIL_VON"] = list(
                            set(synonyms + facts.get("TEIL_VON", []))
                        )

                    fact_count: int = sum(len(v) for v in facts.values())
                    logger.debug(
                        "Fakten abgerufen (Cache-Miss)",
                        extra={
                            "topic": topic,
                            "fact_count": fact_count,
                            "relation_types": list(facts.keys()),
                        },
                    )

                    # In Cache speichern
                    result_dict = dict(facts)
                    self._fact_cache[cache_key] = result_dict

                    return result_dict

        except Exception as e:
            # Spezifische Exception für Query-Fehler
            logger.log_exception(e, "Fehler in query_graph_for_facts", topic=topic)
            # Graceful Degradation: Leeres Dictionary zurückgeben
            return {}

    def query_inverse_relations(
        self, topic: str, relation_type: Optional[str] = None
    ) -> Dict[str, List[str]]:
        """
        Fragt den Graphen nach allen eingehenden Relationen für ein Konzept ab.

        Im Gegensatz zu query_graph_for_facts(), das ausgehende Relationen findet
        (z.B. "hund" -> IS_A -> "säugetier"), findet diese Methode eingehende
        Relationen (z.B. "säugetier" <- IS_A <- "hund").

        Nützlich für:
        - Finden von Nachfahren in Hierarchien (descendants)
        - "Warum"-Fragen (z.B. "Warum ist X ein Y?")
        - Rückwärts-Reasoning

        Args:
            topic: Das Zielkonzept
            relation_type: Optional - nur Relationen dieses Typs (z.B. "IS_A")

        Returns:
            Dict mit {relation_type: [source_concepts]}
            Beispiel: {"IS_A": ["hund", "katze", "elefant"]} für topic="säugetier"
        """
        if not self.driver:
            logger.error(
                "query_inverse_relations: Kein DB-Driver verfügbar",
                extra={"topic": topic},
            )
            return {}

        try:
            with PerformanceLogger(
                logger.logger,
                "query_inverse_relations",
                topic=topic,
                relation_type=relation_type,
            ):
                with self.driver.session(database="neo4j") as session:
                    self.ensure_wort_und_konzept(topic)

                    # Query: Alle eingehenden Relationen
                    if relation_type:
                        # Nur spezifische Relation
                        result = session.run(
                            """
                            MATCH (s:Konzept)-[r]->(o:Konzept {name: $topic})
                            WHERE type(r) = $relation_type
                            RETURN type(r) AS relation, s.name AS subject
                            """,
                            topic=topic.lower(),
                            relation_type=relation_type,
                        )
                    else:
                        # Alle Relationen
                        result = session.run(
                            """
                            MATCH (s:Konzept)-[r]->(o:Konzept {name: $topic})
                            RETURN type(r) AS relation, s.name AS subject
                            """,
                            topic=topic.lower(),
                        )

                    inverse_facts: Dict[str, List[str]] = defaultdict(list)
                    for record in result:
                        inverse_facts[record["relation"]].append(record["subject"])

                    fact_count: int = sum(len(v) for v in inverse_facts.values())
                    logger.debug(
                        "Inverse Relationen abgerufen",
                        extra={
                            "topic": topic,
                            "fact_count": fact_count,
                            "relation_types": list(inverse_facts.keys()),
                        },
                    )

                    return dict(inverse_facts)

        except Exception as e:
            logger.log_exception(e, "Fehler in query_inverse_relations", topic=topic)
            return {}

    def query_graph_for_facts_with_confidence(
        self, topic: str
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Fragt den Graphen nach allen ausgehenden Fakten UND deren Confidence-Werten und Timestamps.

        Im Gegensatz zu query_graph_for_facts() gibt diese Methode nicht nur die
        Ziel-Konzepte zurück, sondern auch die Confidence-Werte und Timestamps aus den Relationen.

        UPDATED: Erweitert um Timestamp-Unterstützung für Confidence-Decay

        Args:
            topic: Das Konzept, für das Fakten gesucht werden

        Returns:
            Dict mit {relation_type: [{"target": str, "confidence": float, "timestamp": str}]}
            Beispiel: {
                "IS_A": [
                    {"target": "säugetier", "confidence": 0.85, "timestamp": "2025-01-15T10:30:00"},
                    {"target": "tier", "confidence": 0.9, "timestamp": "2025-01-20T14:00:00"}
                ],
                "HAS_PROPERTY": [
                    {"target": "vierbeinig", "confidence": 1.0, "timestamp": None}
                ]
            }
        """
        if not self.driver:
            logger.error(
                "query_graph_for_facts_with_confidence: Kein DB-Driver verfügbar",
                extra={"topic": topic},
            )
            return {}

        try:
            with PerformanceLogger(
                logger.logger, "query_graph_for_facts_with_confidence", topic=topic
            ):
                with self.driver.session(database="neo4j") as session:
                    self.ensure_wort_und_konzept(topic)

                    # Query: Alle ausgehenden Relationen mit Confidence UND Timestamp
                    result = session.run(
                        """
                        MATCH (s:Konzept {name: $topic})-[r]->(o:Konzept)
                        RETURN type(r) AS relation,
                               o.name AS object,
                               COALESCE(r.confidence, 1.0) AS confidence,
                               toString(r.timestamp) AS timestamp
                        """,
                        topic=topic.lower(),
                    )

                    facts: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
                    for record in result:
                        facts[record["relation"]].append(
                            {
                                "target": record["object"],
                                "confidence": float(record["confidence"]),
                                "timestamp": record[
                                    "timestamp"
                                ],  # Kann None sein, sonst ISO-String
                            }
                        )

                    fact_count: int = sum(len(v) for v in facts.values())
                    logger.debug(
                        "Fakten mit Confidence und Timestamps abgerufen",
                        extra={
                            "topic": topic,
                            "fact_count": fact_count,
                            "relation_types": list(facts.keys()),
                        },
                    )

                    return dict(facts)

        except Exception as e:
            logger.log_exception(
                e, "Fehler in query_graph_for_facts_with_confidence", topic=topic
            )
            return {}

    def query_inverse_relations_with_confidence(
        self, topic: str, relation_type: Optional[str] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Fragt den Graphen nach eingehenden Relationen UND deren Confidence-Werten.

        Im Gegensatz zu query_inverse_relations() gibt diese Methode nicht nur die
        Quell-Konzepte zurück, sondern auch die Confidence-Werte aus den Relationen.

        Args:
            topic: Das Zielkonzept
            relation_type: Optional - nur Relationen dieses Typs (z.B. "IS_A")

        Returns:
            Dict mit {relation_type: [{"source": str, "confidence": float}]}
            Beispiel: {
                "IS_A": [
                    {"source": "hund", "confidence": 0.85},
                    {"source": "katze", "confidence": 0.9}
                ]
            }
        """
        if not self.driver:
            logger.error(
                "query_inverse_relations_with_confidence: Kein DB-Driver verfügbar",
                extra={"topic": topic},
            )
            return {}

        try:
            with PerformanceLogger(
                logger.logger,
                "query_inverse_relations_with_confidence",
                topic=topic,
                relation_type=relation_type,
            ):
                with self.driver.session(database="neo4j") as session:
                    self.ensure_wort_und_konzept(topic)

                    # Query: Alle eingehenden Relationen mit Confidence
                    if relation_type:
                        result = session.run(
                            """
                            MATCH (s:Konzept)-[r]->(o:Konzept {name: $topic})
                            WHERE type(r) = $relation_type
                            RETURN type(r) AS relation,
                                   s.name AS subject,
                                   COALESCE(r.confidence, 1.0) AS confidence
                            """,
                            topic=topic.lower(),
                            relation_type=relation_type,
                        )
                    else:
                        result = session.run(
                            """
                            MATCH (s:Konzept)-[r]->(o:Konzept {name: $topic})
                            RETURN type(r) AS relation,
                                   s.name AS subject,
                                   COALESCE(r.confidence, 1.0) AS confidence
                            """,
                            topic=topic.lower(),
                        )

                    inverse_facts: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
                    for record in result:
                        inverse_facts[record["relation"]].append(
                            {
                                "source": record["subject"],
                                "confidence": float(record["confidence"]),
                            }
                        )

                    fact_count: int = sum(len(v) for v in inverse_facts.values())
                    logger.debug(
                        "Inverse Relationen mit Confidence abgerufen",
                        extra={
                            "topic": topic,
                            "fact_count": fact_count,
                            "relation_types": list(inverse_facts.keys()),
                        },
                    )

                    return dict(inverse_facts)

        except Exception as e:
            logger.log_exception(
                e, "Fehler in query_inverse_relations_with_confidence", topic=topic
            )
            return {}

    def query_facts_with_synonyms(self, topic: str) -> Dict[str, Any]:
        """
        PHASE 5.3 (Aktion 3): Robuste Synonym-erweiterte Faktensuche.

        Diese Methode findet ALLE Fakten über ein Thema UND alle seine Synonyme.
        Dies ermöglicht es KAI, Wissen über verschiedene Namen desselben Konzepts
        zusammenzuführen und intelligent zu antworten.

        Beispiel:
        - "Auto" ist Synonym zu "PKW"
        - Fakt: "Auto IS_A Fahrzeug"
        - Fakt: "PKW HAS_PROPERTY schnell"
        - Anfrage "Was ist ein Auto?" -> Gibt beide Fakten zurück

        Args:
            topic: Das angefragte Thema

        Returns:
            Dictionary mit:
            - "primary_topic": Der angefragte Begriff
            - "synonyms": Liste aller Synonyme
            - "facts": Alle Fakten (über topic + Synonyme zusammengeführt)
            - "sources": Zuordnung welcher Fakt von welchem Begriff kommt
            - "bedeutungen": Liste aller Bedeutungen/Definitionen
        """
        if not self.driver:
            return {
                "primary_topic": topic,
                "synonyms": [],
                "facts": {},
                "sources": {},
                "bedeutungen": [],
            }

        with self.driver.session(database="neo4j") as session:
            self.ensure_wort_und_konzept(topic)
            topic_lower: str = topic.lower()

            # SCHRITT 1: Finde alle Synonyme
            synonym_result = session.run(
                """
                MATCH (w:Wort {lemma: $topic})-[:TEIL_VON]->(sg:Synonymgruppe)<-[:TEIL_VON]-(syn:Wort)
                WHERE syn.lemma <> $topic
                RETURN DISTINCT syn.lemma AS synonym
                """,
                topic=topic_lower,
            )
            synonyms: List[str] = [record["synonym"] for record in synonym_result]

            # SCHRITT 2: Sammle Fakten über das Hauptthema UND alle Synonyme (Konzept-Ebene)
            all_topics: List[str] = [topic_lower] + synonyms
            combined_facts: Dict[str, List[str]] = defaultdict(list)
            fact_sources: Dict[str, List[str]] = (
                {}
            )  # Tracking welcher Fakt von welchem Begriff kommt

            for search_topic in all_topics:
                result = session.run(
                    """
                    MATCH (s:Konzept {name: $topic})-[r]->(o:Konzept)
                    RETURN type(r) AS relation, o.name AS object
                    """,
                    topic=search_topic,
                )

                for record in result:
                    relation: str = record["relation"]
                    obj: str = record["object"]

                    # Verhindere Duplikate
                    if obj not in combined_facts[relation]:
                        combined_facts[relation].append(obj)

                        # Tracking für Transparenz
                        fact_key: str = f"{relation}:{obj}"
                        if fact_key not in fact_sources:
                            fact_sources[fact_key] = []
                        fact_sources[fact_key].append(search_topic)

            # SCHRITT 3: Hole Bedeutungen/Definitionen (Wort-Ebene)
            # Suche HAT_BEDEUTUNG Relationen für alle Themen inkl. Synonyme
            bedeutungen_list: List[str] = []
            for search_topic in all_topics:
                bedeutung_result = session.run(
                    """
                    MATCH (w:Wort {lemma: $topic})-[:HAT_BEDEUTUNG]->(b:Bedeutung)
                    RETURN DISTINCT b.text AS bedeutung
                    """,
                    topic=search_topic,
                )

                for record in bedeutung_result:
                    bedeutung_text: str = record["bedeutung"]
                    if bedeutung_text and bedeutung_text not in bedeutungen_list:
                        bedeutungen_list.append(bedeutung_text)

            # SCHRITT 4: Füge Synonyme als TEIL_VON Fakten hinzu (falls vorhanden)
            if synonyms:
                combined_facts["TEIL_VON"] = list(
                    set(synonyms + combined_facts.get("TEIL_VON", []))
                )

            return {
                "primary_topic": topic_lower,
                "synonyms": synonyms,
                "facts": dict(combined_facts),
                "sources": fact_sources,
                "bedeutungen": bedeutungen_list,
            }

    def get_all_known_words(self) -> List[str]:
        """
        Holt alle bekannten Wörter (Lemmas) aus dem Graphen.
        Wird für Fuzzy-Matching bei Tippfehlern verwendet.

        Performance-Optimierung: Nutzt TTL-Cache (10 Minuten) da sich die Wortliste
        selten ändert und die Query bei großen Graphen teuer sein kann.

        Returns:
            Liste aller Lemmas (lowercase)
        """
        if not self.driver:
            return []

        # Cache-Key
        cache_key = "all_known_words"

        # Prüfe Cache
        if cache_key in self._known_words_cache:
            logger.debug("Cache-Hit für get_all_known_words")
            return self._known_words_cache[cache_key]

        with self.driver.session(database="neo4j") as session:
            result = session.run(
                """
                MATCH (w:Wort)
                RETURN w.lemma AS lemma
                ORDER BY w.lemma
                """
            )
            words = [record["lemma"] for record in result]

            # In Cache speichern
            self._known_words_cache[cache_key] = words
            logger.debug(
                "Cache-Miss für get_all_known_words", extra={"word_count": len(words)}
            )

            return words

    def find_similar_words(
        self,
        query_word: str,
        embedding_service=None,
        similarity_threshold: float = 0.75,
        max_results: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Findet ähnliche Wörter für Tippfehlertoleranz mittels semantischer Embeddings.

        Algorithmus:
        1. Erzeugt Embedding für query_word
        2. Erzeugt Embeddings für alle bekannten Wörter
        3. Berechnet Kosinus-Ähnlichkeit
        4. Gibt Top-N Kandidaten zurück

        Args:
            query_word: Das zu suchende Wort (ggf. mit Tippfehler)
            embedding_service: EmbeddingService-Instanz
            similarity_threshold: Minimale Ähnlichkeit (0.0 bis 1.0)
            max_results: Maximale Anzahl Ergebnisse

        Returns:
            Liste von Dicts: [{"word": str, "similarity": float}, ...]
            Sortiert nach Ähnlichkeit (höchste zuerst)
        """
        if (
            not self.driver
            or not embedding_service
            or not embedding_service.is_available()
        ):
            logger.warning(
                "find_similar_words: Driver oder Embedding-Service nicht verfügbar"
            )
            return []

        try:
            # Hole alle bekannten Wörter
            known_words: List[str] = self.get_all_known_words()

            if not known_words:
                logger.debug("find_similar_words: Keine bekannten Wörter gefunden")
                return []

            # Erzeuge Embedding für query_word (nicht verwendet, nur für Validierung)
            try:
                _ = embedding_service.get_embedding(query_word.lower())
            except Exception as e:
                logger.error(f"find_similar_words: Fehler beim Query-Embedding: {e}")
                return []

            # Berechne Ähnlichkeiten
            similarities: List[Dict[str, Any]] = []

            for known_word in known_words:
                # Exakte Übereinstimmung überspringen
                if known_word.lower() == query_word.lower():
                    continue

                try:
                    similarity: float = embedding_service.get_similarity(
                        query_word.lower(), known_word
                    )

                    if similarity >= similarity_threshold:
                        similarities.append(
                            {"word": known_word, "similarity": similarity}
                        )
                except Exception as e:
                    logger.debug(f"find_similar_words: Fehler bei '{known_word}': {e}")
                    continue

            # Sortiere nach Ähnlichkeit (höchste zuerst) und limitiere
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            result: List[Dict[str, Any]] = similarities[:max_results]

            logger.debug(
                f"find_similar_words: '{query_word}' -> {len(result)} Kandidaten "
                f"(von {len(known_words)} bekannten Wörtern)"
            )

            return result

        except Exception as e:
            logger.error(f"find_similar_words: Unerwarteter Fehler: {e}", exc_info=True)
            return []

    def get_word_frequency(self, word: str) -> Dict[str, int]:
        """
        Berechnet Häufigkeits-Metriken für ein Wort.

        Frequency wird gemessen als:
        - Anzahl ausgehender Relations (out_degree)
        - Anzahl eingehender Relations (in_degree)
        - Gesamt-Degree (total_degree)

        Je häufiger ein Wort verwendet wird, desto mehr Fakten sind damit verbunden.
        Nutzt diese Info für Typo-Detection: Häufige Wörter sind wahrscheinlichere Kandidaten.

        Args:
            word: Das Wort (wird normalisiert)

        Returns:
            Dict mit "out_degree", "in_degree", "total_degree"
            Bei Fehler: {"out_degree": 0, "in_degree": 0, "total_degree": 0}
        """
        if not self.driver:
            return {"out_degree": 0, "in_degree": 0, "total_degree": 0}

        word_lower = word.lower()

        try:
            with self.driver.session(database="neo4j") as session:
                # Query zählt Relations sowohl von Wort als auch vom zugehörigen Konzept
                query = """
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
                """

                result = session.run(query, word=word_lower)
                record = result.single()

                if record:
                    out_degree = int(record["out_count"])
                    in_degree = int(record["in_count"])
                    total_degree = int(record["total_count"])

                    logger.debug(
                        "Word frequency berechnet",
                        extra={
                            "word": word_lower,
                            "out_degree": out_degree,
                            "in_degree": in_degree,
                            "total_degree": total_degree,
                        },
                    )

                    return {
                        "out_degree": out_degree,
                        "in_degree": in_degree,
                        "total_degree": total_degree,
                    }
                else:
                    # Wort existiert nicht
                    logger.debug(
                        "Word frequency: Wort nicht gefunden",
                        extra={"word": word_lower},
                    )
                    return {"out_degree": 0, "in_degree": 0, "total_degree": 0}

        except Exception as e:
            logger.warning(
                "Fehler beim Berechnen von Word Frequency",
                extra={"word": word_lower, "error": str(e)},
            )
            return {"out_degree": 0, "in_degree": 0, "total_degree": 0}

    def get_normalized_word_frequency(self, word: str) -> float:
        """
        Gibt normalisierte Word Frequency zurück (0.0 - 1.0).

        Verwendet Sigmoid-Funktion für sanfte Normalisierung:
        - 0 relations -> 0.0
        - 5 relations -> ~0.5
        - 20+ relations -> ~1.0

        Args:
            word: Das Wort

        Returns:
            Normalisierte Frequency (0.0 - 1.0)
        """
        freq = self.get_word_frequency(word)
        total = freq["total_degree"]

        # Sigmoid-Normalisierung mit Midpoint bei 5 Relations
        import math

        if total == 0:
            return 0.0

        # Sigmoid: 1 / (1 + e^(-(x-5)/3))
        # Midpoint bei x=5, Steigung kontrolliert durch 3
        sigmoid = 1.0 / (1.0 + math.exp(-(total - 5.0) / 3.0))

        return min(1.0, sigmoid)

    def word_exists(self, word: str) -> bool:
        """
        Prüft ob ein Wort im Wissensgraph existiert.

        Diese Methode ist optimiert für schnelle Existenzprüfungen und nutzt den
        _known_words_cache für häufige Abfragen.

        Args:
            word: Das zu prüfende Wort (wird normalisiert zu lowercase)

        Returns:
            True wenn das Wort existiert, sonst False
        """
        if not self.driver:
            return False

        word_lower = word.lower()

        # Cache-Key für diese Prüfung
        cache_key = f"word_exists:{word_lower}"

        # Prüfe Cache
        if cache_key in self._known_words_cache:
            result = self._known_words_cache[cache_key]
            logger.debug(f"word_exists: Cache-Hit für '{word_lower}' -> {result}")
            return result

        try:
            with self.driver.session(database="neo4j") as session:
                result = session.run(
                    """
                    MATCH (w:Wort {lemma: $lemma})
                    RETURN count(w) > 0 AS exists
                    """,
                    lemma=word_lower,
                )
                record = result.single()
                exists = record["exists"] if record else False

                # In Cache speichern
                self._known_words_cache[cache_key] = exists

                logger.debug(f"word_exists: '{word_lower}' -> {exists}")
                return exists

        except Exception as e:
            logger.error(
                f"word_exists: Fehler bei Prüfung für '{word_lower}': {e}",
                exc_info=True,
            )
            return False

    def create_agent(
        self, agent_id: str, name: str, reasoning_capacity: int = 5
    ) -> bool:
        """
        Erstellt einen Agent Node in Neo4j für Theory of Mind.

        Args:
            agent_id: Eindeutige ID für den Agent
            name: Name des Agents (z.B. "Alice", "Bob")
            reasoning_capacity: Max. Meta-Level für Reasoning (default: 5)

        Returns:
            True bei Erfolg, False bei Fehler
        """
        if not self.driver:
            logger.error(
                "create_agent: Kein DB-Driver verfügbar", extra={"agent_id": agent_id}
            )
            return False

        try:
            with PerformanceLogger(
                logger.logger, "create_agent", agent_id=agent_id, name=name
            ):
                with self.driver.session(database="neo4j") as session:
                    with session.begin_transaction() as tx:
                        result = tx.run(
                            """
                            MERGE (a:Agent {id: $agent_id})
                            ON CREATE SET
                                a.name = $name,
                                a.reasoning_capacity = $reasoning_capacity,
                                a.created_at = timestamp()
                            RETURN a.id AS id
                            """,
                            agent_id=agent_id,
                            name=name,
                            reasoning_capacity=reasoning_capacity,
                        )
                        record = result.single()
                        tx.commit()

                        if not record or record["id"] != agent_id:
                            logger.error(
                                "create_agent: Verifikation fehlgeschlagen",
                                extra={"agent_id": agent_id},
                            )
                            return False

                        logger.info(
                            "Agent erfolgreich erstellt",
                            extra={"agent_id": agent_id, "name": name},
                        )
                        return True

        except Exception as e:
            logger.log_exception(
                e, "create_agent: Fehler", agent_id=agent_id, name=name
            )
            return False

    def add_belief(
        self, agent_id: str, proposition: str, certainty: float = 1.0
    ) -> bool:
        """
        Erstellt einen Belief Node und verbindet ihn mit einem Agent via KNOWS Relation.

        Args:
            agent_id: ID des Agents, der den Belief hat
            proposition: Die Proposition/das Faktum (z.B. "hund IS_A tier")
            certainty: Gewissheit/Confidence (0.0 - 1.0, default: 1.0)

        Returns:
            True bei Erfolg, False bei Fehler
        """
        if not self.driver:
            logger.error(
                "add_belief: Kein DB-Driver verfügbar", extra={"agent_id": agent_id}
            )
            return False

        try:
            with PerformanceLogger(
                logger.logger, "add_belief", agent_id=agent_id, proposition=proposition
            ):
                with self.driver.session(database="neo4j") as session:
                    with session.begin_transaction() as tx:
                        result = tx.run(
                            """
                            MATCH (a:Agent {id: $agent_id})
                            CREATE (b:Belief {
                                id: randomUUID(),
                                proposition: $proposition,
                                certainty: $certainty,
                                created_at: timestamp()
                            })
                            CREATE (a)-[:KNOWS]->(b)
                            RETURN b.id AS belief_id
                            """,
                            agent_id=agent_id,
                            proposition=proposition,
                            certainty=certainty,
                        )
                        record = result.single()
                        tx.commit()

                        if not record:
                            logger.error(
                                "add_belief: Belief konnte nicht erstellt werden",
                                extra={"agent_id": agent_id},
                            )
                            return False

                        logger.info(
                            "Belief erfolgreich erstellt",
                            extra={
                                "agent_id": agent_id,
                                "proposition": proposition,
                                "belief_id": record["belief_id"],
                            },
                        )
                        return True

        except Exception as e:
            logger.log_exception(
                e, "add_belief: Fehler", agent_id=agent_id, proposition=proposition
            )
            return False

    def add_meta_belief(
        self, observer_id: str, subject_id: str, proposition: str, meta_level: int
    ) -> bool:
        """
        Erstellt einen MetaBelief Node für verschachtelte Beliefs ("A knows that B knows P").

        Args:
            observer_id: ID des Agents, der den Meta-Belief hat (z.B. "Alice")
            subject_id: ID des Agents, über dessen Belief gesprochen wird (z.B. "Bob")
            proposition: Die Proposition (z.B. "hund IS_A tier")
            meta_level: Verschachtelungs-Level (1 = "A knows B knows", 2 = "A knows B knows C knows", etc.)

        Returns:
            True bei Erfolg, False bei Fehler
        """
        if not self.driver:
            logger.error(
                "add_meta_belief: Kein DB-Driver verfügbar",
                extra={"observer_id": observer_id, "subject_id": subject_id},
            )
            return False

        try:
            with PerformanceLogger(
                logger.logger,
                "add_meta_belief",
                observer_id=observer_id,
                subject_id=subject_id,
                proposition=proposition,
            ):
                with self.driver.session(database="neo4j") as session:
                    with session.begin_transaction() as tx:
                        result = tx.run(
                            """
                            MATCH (observer:Agent {id: $observer_id})
                            MATCH (subject:Agent {id: $subject_id})
                            CREATE (mb:MetaBelief {
                                id: randomUUID(),
                                proposition: $proposition,
                                meta_level: $meta_level,
                                created_at: timestamp()
                            })
                            CREATE (observer)-[:KNOWS_THAT]->(mb)
                            CREATE (mb)-[:ABOUT_AGENT]->(subject)
                            RETURN mb.id AS meta_belief_id
                            """,
                            observer_id=observer_id,
                            subject_id=subject_id,
                            proposition=proposition,
                            meta_level=meta_level,
                        )
                        record = result.single()
                        tx.commit()

                        if not record:
                            logger.error(
                                "add_meta_belief: MetaBelief konnte nicht erstellt werden",
                                extra={
                                    "observer_id": observer_id,
                                    "subject_id": subject_id,
                                },
                            )
                            return False

                        logger.info(
                            "MetaBelief erfolgreich erstellt",
                            extra={
                                "observer_id": observer_id,
                                "subject_id": subject_id,
                                "proposition": proposition,
                                "meta_level": meta_level,
                                "meta_belief_id": record["meta_belief_id"],
                            },
                        )
                        return True

        except Exception as e:
            logger.log_exception(
                e,
                "add_meta_belief: Fehler",
                observer_id=observer_id,
                subject_id=subject_id,
                proposition=proposition,
            )

    def get_node_count(self) -> int:
        """
        Ermittelt die Gesamtanzahl der Knoten im Graph.

        Zählt alle Wort- und Konzept-Nodes für adaptive Hyperparameter-Tuning.

        Returns:
            Anzahl der Nodes im Graph
        """
        if not self.driver:
            logger.error("get_node_count: Kein DB-Driver verfügbar")
            return 0

        try:
            with self.driver.session(database="neo4j") as session:
                result = session.run(
                    """
                    MATCH (n)
                    WHERE n:Wort OR n:Konzept
                    RETURN count(n) AS node_count
                    """
                )
                record = result.single()

                if not record:
                    logger.warning("get_node_count: Keine Nodes gefunden")
                    return 0

                count = record["node_count"]
                logger.debug(f"Graph enthält {count} Nodes")
                return count

        except Exception as e:
            logger.log_exception(e, "get_node_count: Fehler")
            return 0
