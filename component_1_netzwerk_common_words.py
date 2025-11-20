# component_1_netzwerk_common_words.py
"""
Neo4j CommonWords Management - Stop-Words für Entity-Extraktion

Verantwortlichkeiten:
- Speicherung von Common Words (Artikel, Konjunktionen, etc.)
- Dynamische Erweiterung zur Laufzeit
- Kategorisierung (article, conjunction, question_word, object, verb)
- Integration mit Entity-Extraktion (Logic-Puzzles)

Design:
- Node-Typ: CommonWord
- Properties: word (lowercase, unique), category, confidence
- CRUD-Operationen für flexible Verwaltung
"""

from typing import Optional, Set

from component_15_logging_config import get_logger

logger = get_logger(__name__)


class CommonWordsManager:
    """
    Manager für CommonWords in Neo4j.

    CommonWords sind Wörter die NICHT als Entitäten erkannt werden sollen
    (z.B. Artikel, Konjunktionen, Fragewörter).
    """

    def __init__(self, driver):
        """
        Args:
            driver: Neo4j-Driver-Instanz
        """
        self.driver = driver
        self._ensure_constraints()

    def _ensure_constraints(self):
        """Erstellt Unique-Constraint für CommonWord.word"""
        with self.driver.session() as session:
            try:
                session.run(
                    "CREATE CONSTRAINT common_word_unique IF NOT EXISTS "
                    "FOR (cw:CommonWord) REQUIRE cw.word IS UNIQUE"
                )
                logger.debug("CommonWord constraint erstellt/verifiziert")
            except Exception as e:
                logger.debug(f"Constraint bereits vorhanden oder Fehler: {e}")

    def add_common_word(
        self, word: str, category: str, confidence: float = 1.0
    ) -> bool:
        """
        Fügt ein CommonWord zur Datenbank hinzu.

        Args:
            word: Das Wort (wird lowercase gespeichert)
            category: Kategorie (article, conjunction, question_word, object, verb)
            confidence: Konfidenz 0.0-1.0 (für späteres Auto-Learning)

        Returns:
            True wenn erfolgreich erstellt, False wenn bereits vorhanden
        """
        word_lower = word.lower().strip()

        with self.driver.session() as session:
            result = session.run(
                """
                MERGE (cw:CommonWord {word: $word})
                ON CREATE SET
                    cw.category = $category,
                    cw.confidence = $confidence,
                    cw.created_at = datetime()
                ON MATCH SET
                    cw.category = $category,
                    cw.confidence = $confidence
                RETURN cw.word AS word,
                       CASE WHEN cw.created_at = datetime() THEN true ELSE false END AS created
                """,
                word=word_lower,
                category=category,
                confidence=confidence,
            )

            record = result.single()
            if record:
                created = record.get("created", False)
                if created:
                    logger.debug(f"CommonWord erstellt: {word_lower} ({category})")
                return created

        return False

    def add_common_words_batch(self, words_dict: dict) -> int:
        """
        Fügt mehrere CommonWords in einem Batch hinzu.

        Args:
            words_dict: {category: [word1, word2, ...]}
                       z.B. {"article": ["der", "die", "das"]}

        Returns:
            Anzahl der neu erstellten Wörter
        """
        count = 0
        for category, words in words_dict.items():
            for word in words:
                if self.add_common_word(word, category):
                    count += 1

        logger.info(f"CommonWords Batch: {count} neue Wörter erstellt")
        return count

    def get_common_words(self, category: Optional[str] = None) -> Set[str]:
        """
        Holt alle CommonWords (optional gefiltert nach Kategorie).

        Args:
            category: Optional - nur Wörter dieser Kategorie

        Returns:
            Set von Wörtern (lowercase)
        """
        with self.driver.session() as session:
            if category:
                result = session.run(
                    """
                    MATCH (cw:CommonWord {category: $category})
                    RETURN cw.word AS word
                    """,
                    category=category,
                )
            else:
                result = session.run(
                    """
                    MATCH (cw:CommonWord)
                    RETURN cw.word AS word
                    """
                )

            words = {record["word"] for record in result}
            logger.debug(
                f"Geladene CommonWords: {len(words)}"
                + (f" (category={category})" if category else "")
            )
            return words

    def is_common_word(self, word: str) -> bool:
        """
        Prüft ob ein Wort ein CommonWord ist.

        Args:
            word: Zu prüfendes Wort

        Returns:
            True wenn CommonWord, False sonst
        """
        word_lower = word.lower().strip()

        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (cw:CommonWord {word: $word})
                RETURN count(cw) > 0 AS is_common
                """,
                word=word_lower,
            )

            record = result.single()
            return record["is_common"] if record else False

    def remove_common_word(self, word: str) -> bool:
        """
        Entfernt ein CommonWord aus der Datenbank.

        Args:
            word: Zu entfernendes Wort

        Returns:
            True wenn erfolgreich entfernt
        """
        word_lower = word.lower().strip()

        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (cw:CommonWord {word: $word})
                DELETE cw
                RETURN count(cw) AS deleted
                """,
                word=word_lower,
            )

            record = result.single()
            deleted = record["deleted"] if record else 0

            if deleted > 0:
                logger.info(f"CommonWord entfernt: {word_lower}")
                return True

        return False

    def get_statistics(self) -> dict:
        """
        Holt Statistiken über CommonWords.

        Returns:
            Dictionary mit total, by_category
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (cw:CommonWord)
                RETURN
                    count(cw) AS total,
                    collect(DISTINCT cw.category) AS categories,
                    cw.category AS category,
                    count(cw) AS count
                """
            )

            stats = {"total": 0, "by_category": {}}

            for record in result:
                if record.get("total"):
                    stats["total"] = record["total"]
                if record.get("category"):
                    stats["by_category"][record["category"]] = record["count"]

            return stats
