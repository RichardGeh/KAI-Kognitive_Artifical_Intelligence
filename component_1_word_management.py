# component_1_word_management.py
"""
Word and concept CRUD operations for Neo4j knowledge graph.

This module handles all operations related to creating and managing words and
concepts, including:
- Word and concept node creation (ensure_wort_und_konzept)
- Word attribute management (set_wort_attribut, get_details_fuer_wort)
- Information association (meanings, synonyms)
- Word existence checks with caching

Extracted from monolithic component_1_netzwerk_core.py as part of architecture
refactoring (Task 5).
"""

import re
import threading
from typing import Any, Dict, List, Optional

from neo4j import Driver

from component_15_logging_config import PerformanceLogger, get_logger
from infrastructure import get_cache_manager

logger = get_logger(__name__)

INFO_TYPE_ALIASES: Dict[str, str] = {
    "bedeutung": "bedeutung",
    "definition": "bedeutung",
    "synonym": "synonym",
}

# Whitelist of allowed word attributes (security - prevent Cypher injection)
ALLOWED_ATTRIBUTES = {
    "pos",
    "frequency",
    "custom_meta",
    "importance",
    "lemma",
    "created_at",
    "updated_at",
    "confidence",
    "source",
}


class WordManager:
    """
    Manages word and concept CRUD operations in Neo4j.

    Responsibilities:
    - Create/read word and concept nodes
    - Set word attributes
    - Associate information (meanings, synonyms)
    - Word existence checks with caching

    Thread Safety:
        This class is thread-safe. All cache operations are protected by the
        CacheManager's internal locking.

    Attributes:
        driver: Neo4j driver instance
        cache_mgr: CacheManager for caching word operations
        _lock: Thread lock for critical operations
    """

    def __init__(self, driver: Driver):
        """
        Initialize word manager.

        Args:
            driver: Neo4j driver instance

        Raises:
            ValueError: If driver is None
        """
        if not driver:
            raise ValueError("Driver cannot be None")

        self.driver = driver
        self._lock = threading.RLock()

        # Register caches with CacheManager
        self.cache_mgr = get_cache_manager()

        # Cache for word existence checks (10 minutes TTL)
        self.cache_mgr.register_cache(
            "netzwerk_word_exists", maxsize=1000, ttl=600, overwrite=True
        )

        # Cache for known words list (10 minutes TTL)
        self.cache_mgr.register_cache(
            "netzwerk_all_words", maxsize=10, ttl=600, overwrite=True
        )

        logger.debug("WordManager initialisiert mit CacheManager")

    def ensure_wort_und_konzept(self, lemma: str) -> bool:
        """
        Ensure a word and its concept exist in the graph.

        Creates Wort node, Konzept node, and BEDEUTET relationship if they
        don't exist.

        Args:
            lemma: The word lemma (will be normalized to lowercase)

        Returns:
            True on success, False on error
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
                    # Use explicit transaction
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

                        # Explicit commit
                        tx.commit()

                        # Verification
                        if not record or record["lemma"] != lemma:
                            logger.error(
                                "ensure_wort_und_konzept: Verifikation fehlgeschlagen",
                                extra={"lemma": lemma},
                            )
                            return False

                        # Invalidate caches after successful creation
                        self._invalidate_word_caches(lemma)

                        logger.debug(
                            "Wort und Konzept erfolgreich sichergestellt",
                            extra={"lemma": lemma},
                        )
                        return True

        except Exception as e:
            # Specific exception for write errors
            logger.log_exception(e, "ensure_wort_und_konzept: Fehler", lemma=lemma)
            # Graceful degradation: return False
            return False

    def set_wort_attribut(
        self, lemma: str, attribut_name: str, attribut_wert: Any
    ) -> bool:
        """
        Set an attribute directly on the :Wort node.

        Args:
            lemma: The word lemma
            attribut_name: Attribute name (will be sanitized and validated)
            attribut_wert: Attribute value

        Returns:
            True if successfully set, False on error

        Raises:
            ValueError: If attribute name is not in whitelist (security)
        """
        if not self.driver:
            logger.warning("set_wort_attribut: Kein Driver verfügbar")
            return False

        if not self.ensure_wort_und_konzept(lemma):
            logger.error(
                f"set_wort_attribut: Wort '{lemma}' konnte nicht erstellt werden"
            )
            return False

        # Sanitize attribute name (only alphanumeric and underscore)
        safe_attribut_name: str = re.sub(r"[^a-zA-Z0-9_]", "", attribut_name.lower())
        if not safe_attribut_name:
            logger.warning(f"Ungültiger Attributname '{attribut_name}'.")
            return False

        # Validate against whitelist (security - prevent Cypher injection)
        if safe_attribut_name not in ALLOWED_ATTRIBUTES:
            error_msg = (
                f"Attribute '{safe_attribut_name}' not in whitelist. "
                f"Allowed: {', '.join(sorted(ALLOWED_ATTRIBUTES))}"
            )
            logger.warning(
                "Attribute not in whitelist",
                extra={"attribute": safe_attribut_name, "lemma": lemma},
            )
            raise ValueError(error_msg)

        lemma_lower = lemma.lower()

        with self.driver.session(database="neo4j") as session:
            # STEP 1: Set attribute with RETURN for verification
            query = f"""
                MATCH (w:Wort {{lemma: $lemma}})
                SET w.{safe_attribut_name} = $wert
                RETURN w.{safe_attribut_name} AS set_value
            """
            result = session.run(query, lemma=lemma_lower, wert=attribut_wert)
            record = result.single()

            # STEP 2: Verification
            if not record:
                logger.error(
                    f"set_wort_attribut: Wort '{lemma_lower}' nicht gefunden oder Update fehlgeschlagen"
                )
                return False

            # Compare set value (comparison may be difficult for complex types)
            set_value = record["set_value"]
            if set_value != attribut_wert:
                logger.warning(
                    f"set_wort_attribut: Gesetzter Wert weicht ab. "
                    f"Erwartet: {attribut_wert}, Gesetzt: {set_value}"
                )

            logger.debug(
                f"set_wort_attribut: Attribut '{safe_attribut_name}' "
                f"für '{lemma_lower}' erfolgreich gesetzt"
            )

        # Invalidate caches after modification
        self._invalidate_word_caches(lemma_lower)

        return True

    def add_information_zu_wort(
        self, lemma: str, info_typ: str, info_inhalt: str
    ) -> Dict[str, Any]:
        """
        Add information to a word (meaning, synonym, etc.).

        Args:
            lemma: The word lemma
            info_typ: Information type (bedeutung, definition, synonym)
            info_inhalt: Information content

        Returns:
            Dict with "created" boolean and optional "error" field
        """
        # This method orchestrates only, error handling is in _add methods
        # But we ensure the first step succeeds
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
        """Add a meaning (Bedeutung) to a word."""
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

        # Invalidate caches after relation addition
        self._invalidate_word_caches(lemma)

        return {"created": created}

    def _add_synonym(self, lemma1: str, lemma2: str) -> Dict[str, bool]:
        """Add a synonym relationship between two words."""
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

        # Invalidate caches after synonym addition
        for lemma in [lemma1, lemma2]:
            self._invalidate_word_caches(lemma)

        return {"created": created}

    def get_details_fuer_wort(self, lemma: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a word.

        Returns:
            Dict with lemma, bedeutungen, synonyme, konzept or None if not found
        """
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

    def get_all_known_words(self) -> List[str]:
        """
        Get all known words (lemmas) from the graph.

        Used for fuzzy matching and typo detection.

        Returns:
            List of all lemmas (lowercase)
        """
        if not self.driver:
            return []

        # Cache key
        cache_key = "all_words_list"

        # Check cache
        cached = self.cache_mgr.get("netzwerk_all_words", cache_key)
        if cached is not None:
            logger.debug("Cache-Hit für get_all_known_words")
            return cached

        with self.driver.session(database="neo4j") as session:
            result = session.run(
                """
                MATCH (w:Wort)
                RETURN w.lemma AS lemma
                ORDER BY w.lemma
                """
            )
            words = [record["lemma"] for record in result]

            # Store in cache
            self.cache_mgr.set("netzwerk_all_words", cache_key, words)
            logger.debug(
                "Cache-Miss für get_all_known_words", extra={"word_count": len(words)}
            )

            return words

    def word_exists(self, word: str) -> bool:
        """
        Check if a word exists in the knowledge graph.

        This method is optimized for fast existence checks and uses caching
        for frequent queries.

        Args:
            word: The word to check (will be normalized to lowercase)

        Returns:
            True if the word exists, False otherwise
        """
        if not self.driver:
            return False

        word_lower = word.lower()

        # Cache key for this check
        cache_key = f"exists:{word_lower}"

        # Check cache
        cached = self.cache_mgr.get("netzwerk_word_exists", cache_key)
        if cached is not None:
            logger.debug(f"word_exists: Cache-Hit für '{word_lower}' -> {cached}")
            return cached

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

                # Store in cache
                self.cache_mgr.set("netzwerk_word_exists", cache_key, exists)

                logger.debug(f"word_exists: '{word_lower}' -> {exists}")
                return exists

        except Exception as e:
            logger.error(
                f"word_exists: Fehler bei Prüfung für '{word_lower}': {e}",
                exc_info=True,
            )
            return False

    def get_word_frequency(self, word: str) -> Dict[str, int]:
        """
        Calculate frequency metrics for a word.

        Frequency is measured as:
        - Number of outgoing relations (out_degree)
        - Number of incoming relations (in_degree)
        - Total degree (total_degree)

        Args:
            word: The word (will be normalized)

        Returns:
            Dict with "out_degree", "in_degree", "total_degree"
            On error: {"out_degree": 0, "in_degree": 0, "total_degree": 0}
        """
        if not self.driver:
            return {"out_degree": 0, "in_degree": 0, "total_degree": 0}

        word_lower = word.lower()

        try:
            with self.driver.session(database="neo4j") as session:
                # Query counts relations from both Wort and associated Konzept
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
                    # Word does not exist
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
        Get normalized word frequency (0.0 - 1.0).

        Uses sigmoid function for smooth normalization:
        - 0 relations -> 0.0
        - 5 relations -> ~0.5
        - 20+ relations -> ~1.0

        Args:
            word: The word

        Returns:
            Normalized frequency (0.0 - 1.0)
        """
        freq = self.get_word_frequency(word)
        total = freq["total_degree"]

        # Sigmoid normalization with midpoint at 5 relations
        import math

        if total == 0:
            return 0.0

        # Sigmoid: 1 / (1 + e^(-(x-5)/3))
        # Midpoint at x=5, slope controlled by 3
        sigmoid = 1.0 / (1.0 + math.exp(-(total - 5.0) / 3.0))

        return min(1.0, sigmoid)

    def _invalidate_word_caches(self, lemma: str):
        """
        Invalidate all caches related to a word after modifications.

        Args:
            lemma: The word lemma (lowercase)
        """
        lemma_lower = lemma.lower()

        # Invalidate word existence cache
        cache_key = f"exists:{lemma_lower}"
        self.cache_mgr.invalidate("netzwerk_word_exists", cache_key)

        # Invalidate all words list cache (word was added/modified)
        self.cache_mgr.invalidate("netzwerk_all_words", "all_words_list")

        logger.debug(f"Caches invalidiert für '{lemma_lower}'")
