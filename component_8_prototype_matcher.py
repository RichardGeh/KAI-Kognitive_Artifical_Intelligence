# component_8_prototype_matcher.py
import logging
from typing import Any, Optional

import numpy as np

from component_1_netzwerk import KonzeptNetzwerk
from component_11_embedding_service import EmbeddingService
from component_15_logging_config import get_logger

logger = get_logger(__name__)

# Ein fester Schwellenwert, um zu entscheiden, ob ein Vektor neu ist.
# Dies kann später durch eine dynamische Methode (z.B. basierend auf Varianz)
# ersetzt werden. Ein höherer Wert bedeutet, dass Vektoren eher bestehenden
# Clustern zugeordnet werden.
# HINWEIS: Angepasst für 384-dimensionale semantische Vektoren (vorher: 8D Featurizer)
NOVELTY_THRESHOLD = 15.0


class PrototypingEngine:
    """
    Verwaltet das dynamische Lernen von Satzmustern (PatternPrototypes).
    Vergleicht neue Sätze mit existierenden Mustern und erstellt neue,
    wenn eine signifikante Neuheit erkannt wird.

    Verwendet semantische Embeddings (384D) statt manueller Featurizer (8D).

    Performance-Optimierung:
    - Session-Cache für Prototypen (verhindert wiederholte DB-Abfragen)
    - Cache wird invalidiert bei Prototyp-Updates/Erstellung
    """

    def __init__(self, netzwerk: KonzeptNetzwerk, embedding_service: EmbeddingService):
        self.netzwerk = netzwerk
        self.embedding_service = embedding_service

        # Session-Cache für Prototypen (wird bei Änderungen invalidiert)
        self._prototype_cache: Optional[list[dict[str, Any]]] = None
        self._cache_hits: int = 0
        self._cache_misses: int = 0

    def _get_prototypes_cached(self) -> list[dict[str, Any]]:
        """
        Holt alle Prototypen mit Session-Caching.

        Performance-Optimierung: Prototypen werden für die Session gecacht,
        da get_all_pattern_prototypes() bei jedem process_vector() Aufruf
        teuer ist (DB-Query + Deserialisierung).

        Returns:
            Liste aller Prototypen
        """
        if self._prototype_cache is not None:
            self._cache_hits += 1
            logger.debug(
                "Prototyp-Cache-Hit",
                extra={
                    "cache_size": len(self._prototype_cache),
                    "hits": self._cache_hits,
                },
            )
            return self._prototype_cache

        # Cache-Miss: Lade von DB
        self._cache_misses += 1
        prototypes = self.netzwerk.get_all_pattern_prototypes()
        self._prototype_cache = prototypes

        logger.debug(
            "Prototyp-Cache-Miss",
            extra={
                "prototype_count": len(prototypes),
                "misses": self._cache_misses,
                "hits": self._cache_hits,
            },
        )

        return prototypes

    def _invalidate_cache(self):
        """
        Invalidiert den Prototyp-Cache.

        Wird aufgerufen wenn:
        - Neue Prototypen erstellt werden
        - Bestehende Prototypen aktualisiert werden
        """
        if self._prototype_cache is not None:
            logger.debug(
                "Invalidiere Prototyp-Cache",
                extra={"cached_count": len(self._prototype_cache)},
            )
            self._prototype_cache = None

    def get_cache_stats(self) -> dict:
        """
        Gibt Statistiken über den Prototyp-Cache zurück.

        Returns:
            Dict mit Cache-Statistiken
        """
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": (
                self._cache_hits / (self._cache_hits + self._cache_misses)
                if (self._cache_hits + self._cache_misses) > 0
                else 0.0
            ),
            "cached": self._prototype_cache is not None,
            "cached_count": (
                len(self._prototype_cache) if self._prototype_cache is not None else 0
            ),
        }

    def _calculate_euclidean_distance(
        self, vec1: np.ndarray, vec2: np.ndarray
    ) -> float:
        """Berechnet den euklidischen Abstand zwischen zwei Numpy-Arrays."""
        return np.linalg.norm(vec1 - vec2)

    def get_embedding_for_text(self, text: str) -> Optional[list[float]]:
        """
        Konvertiert Text in einen semantischen Embedding-Vektor.

        Args:
            text: Der zu vektorisierende Text

        Returns:
            384-dimensionaler Embedding-Vektor oder None bei Fehler
        """
        try:
            embedding = self.embedding_service.get_embedding(text)
            if embedding is None or len(embedding) == 0:
                logger.error(
                    "Kein gültiger Embedding für Text",
                    extra={"text_preview": text[:50], "text_length": len(text)},
                )
                return None
            return embedding
        except Exception as e:
            logger.error(
                "Fehler beim Erstellen des Embeddings",
                extra={
                    "text_preview": text[:50],
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            return None

    def _update_prototype(
        self, prototype: dict[str, Any], new_vector: np.ndarray
    ) -> bool:
        """
        Aktualisiert einen existierenden Prototypen mit einem neuen Vektor.
        Verwendet eine Online-Variante zur Berechnung von Mittelwert und Varianz
        (angelehnt an Welford's Algorithm) für numerische Stabilität.

        Args:
            prototype: Dict mit Prototypen-Daten (id, centroid, variance, count)
            new_vector: Numpy-Array mit neuem Vektor

        Returns:
            True bei Erfolg, False bei Fehler
        """
        try:
            # Conditional logging für DEBUG (Performance-kritisch)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Aktualisiere Prototyp",
                    extra={
                        "prototype_id": prototype["id"][:8],
                        "old_count": prototype["count"],
                    },
                )

            old_centroid = np.array(prototype["centroid"])
            old_m2 = (
                np.array(prototype.get("variance", [0.0] * len(old_centroid)))
                * prototype["count"]
            )
            old_count = prototype["count"]

            new_count = old_count + 1
            delta = new_vector - old_centroid
            new_centroid = old_centroid + delta / new_count

            delta2 = new_vector - new_centroid
            new_m2 = old_m2 + delta * delta2
            new_variance = new_m2 / new_count

            success = self.netzwerk.update_pattern_prototype(
                prototype_id=prototype["id"],
                new_centroid=new_centroid.tolist(),
                new_variance=new_variance.tolist(),
                new_count=new_count,
            )

            if success:
                # Invalidiere Cache nach Update
                self._invalidate_cache()
                logger.info(
                    "Prototyp aktualisiert",
                    extra={
                        "prototype_id": prototype["id"][:8],
                        "new_count": new_count,
                        "old_count": old_count,
                    },
                )
                return True
            else:
                logger.error(
                    "Prototyp-Update fehlgeschlagen",
                    extra={"prototype_id": prototype["id"][:8]},
                )
                return False

        except Exception as e:
            logger.error(
                "Fehler beim Prototyp-Update",
                extra={
                    "prototype_id": prototype.get("id", "unknown")[:8],
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            return False

    def process_vector(self, vector: list[float], category: str) -> Optional[str]:
        """
        Verarbeitet einen neuen Satzvektor. Findet den besten Match,
        aktualisiert ihn oder erstellt einen neuen Prototypen.

        Performance-Optimierung: Nutzt gecachte Prototypen statt wiederholter DB-Abfragen.

        Args:
            vector: Der zu verarbeitende Vektor.
            category: Die semantische Kategorie, die dem Vektor zugeordnet ist.
                      Wird verwendet, wenn ein neuer Prototyp erstellt wird.

        Returns:
            Die ID des gematchten oder neu erstellten Prototypen, oder None bei Fehler.
        """
        try:
            # Validierung
            if not vector:
                logger.error("Leerer Vektor übergeben")
                return None

            if not self.netzwerk or not self.netzwerk.driver:
                logger.error(
                    "Netzwerkverbindung nicht verfügbar",
                    extra={"component": "PrototypingEngine"},
                )
                return None

            input_vector = np.array(vector)
            # Nutze gecachte Prototypen (Performance-Optimierung)
            prototypes = self._get_prototypes_cached()

            # Fall 1: Keine existierenden Prototypen -> Erstelle ersten Prototyp
            if not prototypes:
                logger.info("Erstelle ersten Prototyp", extra={"category": category})
                new_id = self.netzwerk.create_pattern_prototype(
                    input_vector.tolist(), category
                )
                if new_id:
                    # Invalidiere Cache nach Erstellung
                    self._invalidate_cache()
                    logger.info(
                        "Erster Prototyp erstellt",
                        extra={"prototype_id": new_id[:8], "category": category},
                    )
                else:
                    logger.error(
                        "Erstellung des ersten Prototyps fehlgeschlagen",
                        extra={"category": category},
                    )
                return new_id

            # Fall 2: Suche besten Match in gleicher Kategorie
            best_match: Optional[dict[str, Any]] = None
            min_distance = float("inf")
            normalized_category = category.upper()

            for p in prototypes:
                # Kategorie-Filter (robust gegenüber Groß-/Kleinschreibung)
                prototype_category = p.get("category")
                if (
                    not prototype_category
                    or prototype_category.upper() != normalized_category
                ):
                    continue

                # Validiere Centroid vor Verarbeitung
                centroid = p.get("centroid")
                if centroid is None or not isinstance(centroid, (list, np.ndarray)):
                    logger.warning(
                        "Überspringe Prototyp mit ungültigem Centroid",
                        extra={
                            "prototype_id": p.get("id", "unknown")[:8],
                            "centroid_type": type(centroid).__name__,
                        },
                    )
                    continue

                try:
                    p_centroid = np.array(centroid)

                    # Validiere Dimensionen
                    if p_centroid.shape != input_vector.shape:
                        logger.warning(
                            "Überspringe Prototyp mit falscher Dimensionalität",
                            extra={
                                "prototype_id": p.get("id", "unknown")[:8],
                                "expected_dim": input_vector.shape,
                                "actual_dim": p_centroid.shape,
                            },
                        )
                        continue

                    distance = self._calculate_euclidean_distance(
                        input_vector, p_centroid
                    )

                    if distance < min_distance:
                        min_distance = distance
                        best_match = p
                except Exception as e:
                    logger.warning(
                        "Fehler beim Berechnen der Distanz für Prototyp",
                        extra={
                            "prototype_id": p.get("id", "unknown")[:8],
                            "error": str(e),
                        },
                    )
                    continue

            # Fall 3: Match gefunden und nah genug -> Update
            if best_match and min_distance < NOVELTY_THRESHOLD:
                logger.info(
                    "Found match for prototype {} with distance {:.4f}".format(
                        best_match["id"], min_distance
                    )
                )
                update_success = self._update_prototype(best_match, input_vector)
                if not update_success:
                    logger.warning("Update failed, but returning prototype ID anyway")
                return best_match["id"]

            # Fall 4: Kein passender Match -> Neuer Prototyp
            else:
                logger.info(
                    "Novelty detected. Min distance {:.4f} >= threshold. "
                    "Creating new prototype for category '{}'.".format(
                        min_distance, normalized_category
                    )
                )
                new_id = self.netzwerk.create_pattern_prototype(
                    input_vector.tolist(), normalized_category
                )
                if new_id:
                    # Invalidiere Cache nach Erstellung
                    self._invalidate_cache()
                    logger.info("Created new prototype with ID: {}".format(new_id))
                else:
                    logger.error("Failed to create new prototype")
                return new_id

        except Exception as e:
            logger.error(f"Error in process_vector: {e}", exc_info=True)
            return None

    def find_best_match(
        self, vector: list[float], category_filter: str = None
    ) -> Optional[tuple[dict[str, Any], float]]:
        """
        Findet den Prototyp mit der geringsten Distanz zum gegebenen Vektor.

        Performance-Optimierung: Nutzt gecachte Prototypen statt wiederholter DB-Abfragen.

        Args:
            vector: Der zu matchende Vektor
            category_filter: Optional - filtert nach dieser Kategorie

        Returns:
            Tuple von (Prototyp-Dict, Distanz) oder None wenn keine Prototypen gefunden
        """
        try:
            # Validierung
            if not vector:
                logger.error("find_best_match: Leerer Vektor übergeben")
                return None

            if not self.netzwerk or not self.netzwerk.driver:
                logger.error("find_best_match: Netzwerkverbindung nicht verfügbar")
                return None

            input_vector = np.array(vector)
            # Nutze gecachte Prototypen (Performance-Optimierung)
            prototypes = self._get_prototypes_cached()

            if not prototypes:
                logger.debug("find_best_match: Keine Prototypen in der Datenbank")
                return None

            # Optional: Nach Kategorie filtern
            if category_filter:
                original_count = len(prototypes)
                prototypes = [
                    p
                    for p in prototypes
                    if p.get("category", "").upper() == category_filter.upper()
                ]

                if not prototypes:
                    logger.debug(
                        f"find_best_match: Keine Prototypen für Kategorie '{category_filter}' "
                        f"(von {original_count} Prototypen)"
                    )
                    return None

            # Finde besten Match
            best_match = None
            min_distance = float("inf")

            for prototype in prototypes:
                try:
                    # Validiere Centroid vor Verarbeitung
                    centroid = prototype.get("centroid")
                    if centroid is None or not isinstance(centroid, (list, np.ndarray)):
                        logger.warning(
                            "Überspringe Prototyp mit ungültigem Centroid",
                            extra={
                                "prototype_id": prototype.get("id", "unknown")[:8],
                                "centroid_type": type(centroid).__name__,
                            },
                        )
                        continue

                    p_centroid = np.array(centroid)

                    # Validiere Dimensionen
                    if p_centroid.shape != input_vector.shape:
                        logger.warning(
                            "Überspringe Prototyp mit falscher Dimensionalität",
                            extra={
                                "prototype_id": prototype.get("id", "unknown")[:8],
                                "expected_dim": input_vector.shape,
                                "actual_dim": p_centroid.shape,
                            },
                        )
                        continue

                    distance = self._calculate_euclidean_distance(
                        input_vector, p_centroid
                    )

                    if distance < min_distance:
                        min_distance = distance
                        best_match = prototype
                except Exception as e:
                    logger.warning(
                        f"Fehler beim Verarbeiten von Prototyp {prototype.get('id', 'unknown')}: {e}"
                    )
                    continue

            if best_match:
                logger.debug(
                    f"find_best_match: Bester Match ist {best_match['id']} "
                    f"mit Distanz {min_distance:.4f}"
                )
                return (best_match, min_distance)

            logger.debug("find_best_match: Kein Match gefunden")
            return None

        except Exception as e:
            logger.error(f"Error in find_best_match: {e}", exc_info=True)
            return None
