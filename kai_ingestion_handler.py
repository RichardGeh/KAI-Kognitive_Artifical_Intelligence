# kai_ingestion_handler.py
"""
Text Ingestion Handler Module für KAI

Verantwortlichkeiten:
- Intelligente, vektorgesteuerte Text-Ingestion
- Anwendung von Extraktionsregeln
- Episode-Tracking für Wissensherkunft
- Pattern-Matching mit Prototypen
- Word Usage Tracking (Kontext-Fragmente und Wortverbindungen)
- Parallele Batch-Verarbeitung für große Textmengen
"""
import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from component_1_netzwerk import KonzeptNetzwerk
from component_8_prototype_matcher import NOVELTY_THRESHOLD, PrototypingEngine
from component_utils_text_fragmentation import TextFragmenter
from kai_config import get_config

# Import exception utilities for user-friendly error messages
from kai_exceptions import EmbeddingError, Neo4jWriteError, get_user_friendly_message
from kai_response_formatter import KaiResponseFormatter

logger = logging.getLogger(__name__)
config = get_config()


class KaiIngestionHandler:
    """
    Handler für Text-Ingestion und Faktenextraktion.

    Diese Klasse verwaltet:
    - Intelligente, vektorbasierte Ingestion mit Prototypen
    - Legacy Brute-Force Ingestion als Fallback
    - Episode-Tracking für Wissensherkunft
    - Anwendung von Extraktionsregeln auf Sätze
    """

    def __init__(
        self,
        netzwerk: KonzeptNetzwerk,
        preprocessor,  # LinguisticPreprocessor
        prototyping_engine: PrototypingEngine,
        embedding_service,  # EmbeddingService
    ):
        """
        Initialisiert den Ingestion Handler.

        Args:
            netzwerk: KonzeptNetzwerk für Datenspeicherung
            preprocessor: LinguisticPreprocessor für Text-Analyse
            prototyping_engine: PrototypingEngine für Pattern-Matching
            embedding_service: EmbeddingService für Vektor-Embeddings
        """
        self.netzwerk = netzwerk
        self.preprocessor = preprocessor
        self.prototyping_engine = prototyping_engine
        self.embedding_service = embedding_service
        self.formatter = KaiResponseFormatter()

        # Word Usage Tracking
        self.text_fragmenter = TextFragmenter(linguistic_preprocessor=preprocessor)

    def ingest_text(self, text: str) -> Dict[str, int]:
        """
        PHASE 6.2: Intelligente, Vektor-gesteuerte Ingestion-Pipeline.
        PHASE 3: Erweitert mit Episodischem Gedächtnis - tracked alle gelernten Fakten.

        Algorithmus:
        1. Erstelle Episode-Knoten für dieses Lernereignis
        2. Für jeden Satz: Erzeuge Embedding-Vektor
        3. Finde besten Prototyp-Match
        4. Wenn Match gefunden (Distanz < NOVELTY_THRESHOLD):
           - Hole verknüpfte Regel via get_rule_for_prototype()
           - Wende nur diese eine spezifische Regel an
        5. Sonst: Fallback auf Brute-Force über alle Regeln
        6. Verknüpfe alle erstellten Fakten mit der Episode

        Args:
            text: Der zu ingestierende Text

        Returns:
            Dictionary mit Statistiken: {
                "facts_created": Anzahl neuer Fakten,
                "learned_patterns": Anzahl via Prototypen verarbeiteter Sätze,
                "fallback_patterns": Anzahl via Brute-Force verarbeiteter Sätze
            }
        """
        # PHASE 3: Erstelle Episode für diesen Ingestion-Vorgang
        episode_id = self.netzwerk.create_episode(
            episode_type="ingestion",
            content=text,
            metadata={"method": "intelligent_vectorized"},
        )

        if not episode_id:
            logger.warning("Konnte keine Episode erstellen, fahre ohne Tracking fort")

        doc = self.preprocessor.process(text)
        stats = {
            "facts_created": 0,
            "learned_patterns": 0,
            "fallback_patterns": 0,
        }

        # Prüfe ob Embedding-Service verfügbar ist
        if not self.embedding_service or not self.embedding_service.is_available():
            logger.warning("Embedding-Service nicht verfügbar, verwende Legacy-Methode")
            # Fallback auf Legacy
            facts = self.ingest_text_legacy(text)
            return {
                "facts_created": facts,
                "learned_patterns": 0,
                "fallback_patterns": facts,
            }

        # PERFORMANCE-OPTIMIERUNG: Batch-Embedding für alle Sätze
        # Sammle alle Sätze zuerst
        sentences = []
        for sentence in doc.sents:
            sentence_text = sentence.text.strip()
            if sentence_text:
                sentences.append(sentence_text)

        if not sentences:
            logger.debug("Keine Sätze zum Verarbeiten gefunden")
            return stats

        # BATCH-EMBEDDING: Erzeuge Embeddings für alle Sätze auf einmal
        logger.debug(f"Erzeuge Batch-Embeddings für {len(sentences)} Sätze")
        try:
            sentence_embeddings = self.embedding_service.get_embeddings_batch(sentences)
        except Exception as e:
            logger.warning(
                f"Batch-Embedding fehlgeschlagen, Fallback auf Einzelverarbeitung: {e}"
            )
            sentence_embeddings = [None] * len(sentences)  # Alle None -> Fallback

        # Verarbeite Sätze mit vorberechneten Embeddings
        for sentence_text, vector in zip(sentences, sentence_embeddings):
            logger.debug(f"Verarbeite Satz: '{sentence_text}'")

            # WORD USAGE TRACKING: VOR der Satz-Verarbeitung (läuft immer, unabhängig von Erfolg)
            if config.word_usage_tracking_enabled:
                try:
                    usage_stats = self._track_word_usage(sentence_text)
                    # Initialisiere Statistiken beim ersten Mal
                    if "fragments_stored" not in stats:
                        stats["fragments_stored"] = 0
                        stats["connections_stored"] = 0
                    stats["fragments_stored"] += usage_stats["fragments_stored"]
                    stats["connections_stored"] += usage_stats["connections_stored"]
                except Exception as e:
                    logger.error(f"Fehler beim Word Usage Tracking (ignoriert): {e}")

            try:
                # PHASE 1: Vektor-basiertes Routing (mit vorberechnetem Embedding)
                if not vector:
                    logger.warning(
                        f"Kein Embedding für Satz: '{sentence_text[:50]}...'"
                    )
                    # Fallback (mit Episode-Tracking)
                    facts = self.apply_all_rules(sentence_text, episode_id)
                    stats["facts_created"] += facts
                    stats["fallback_patterns"] += 1
                    continue

                match_result = self.prototyping_engine.find_best_match(vector)

                if match_result:
                    prototype, distance = match_result

                    if distance < NOVELTY_THRESHOLD:
                        # Intelligentes Routing: Hole verknüpfte Regel
                        rule = self.netzwerk.get_rule_for_prototype(prototype["id"])

                        if rule:
                            logger.debug(
                                f"  -> Prototype-Match: {prototype['id'][:8]} "
                                f"(distance={distance:.2f}, rule={rule['relation_type']})"
                            )
                            # PHASE 3: Übergebe episode_id für Tracking
                            facts = self.apply_single_rule(
                                sentence_text, rule, episode_id
                            )
                            stats["facts_created"] += facts
                            stats["learned_patterns"] += 1
                            continue  # Erfolgreich verarbeitet
                        else:
                            logger.debug(
                                f"  -> Prototype {prototype['id'][:8]} hat keine TRIGGERS-Verknüpfung"
                            )

                # PHASE 2: Fallback (Brute-Force für unbekannte Muster)
                # PHASE 3: Mit Episode-Tracking
                logger.debug("  -> Fallback: Brute-Force über alle Regeln")
                facts = self.apply_all_rules(sentence_text, episode_id)
                stats["facts_created"] += facts
                stats["fallback_patterns"] += 1

            except EmbeddingError as e:
                # Spezifischer Fehler beim Embedding-Erstellen
                logger.warning(
                    f"EmbeddingError bei Satz-Verarbeitung: {e}", exc_info=True
                )
                user_msg = get_user_friendly_message(e)
                logger.info(f"User-Message: {user_msg}")
                # Graceful Degradation: Fallback auf Brute-Force
                facts = self.apply_all_rules(sentence_text, episode_id)
                stats["facts_created"] += facts
                stats["fallback_patterns"] += 1
            except Neo4jWriteError as e:
                # Spezifischer Fehler beim Schreiben in Neo4j
                logger.error(
                    f"Neo4jWriteError bei Satz-Verarbeitung: {e}", exc_info=True
                )
                user_msg = get_user_friendly_message(e)
                logger.info(f"User-Message: {user_msg}")
                # Keine Fallback-Verarbeitung bei Write-Fehler
            except Exception as e:
                # Unerwarteter Fehler
                logger.error(
                    f"Unerwarteter Fehler bei Satz-Verarbeitung: {e}", exc_info=True
                )
                # Fallback bei Fehler (mit Episode-Tracking)
                facts = self.apply_all_rules(sentence_text, episode_id)
                stats["facts_created"] += facts
                stats["fallback_patterns"] += 1

        # Logging der Statistiken
        logger.info(
            f"Ingestion abgeschlossen: {stats['facts_created']} Fakten "
            f"({stats['learned_patterns']} via Prototypen, "
            f"{stats['fallback_patterns']} via Brute-Force)"
        )

        return stats

    def ingest_text_large(
        self,
        text: str,
        chunk_size: int = 100,
        progress_callback=None,
        max_workers: int = None,
    ) -> Dict[str, int]:
        """
        PERFORMANCE-OPTIMIERUNG: Batch-Processing für große Textmengen mit paralleler Verarbeitung.

        Diese Methode ist speziell für große Texte (>100 Sätze) optimiert:
        - Teilt Text in Chunks auf
        - Verarbeitet Chunks PARALLEL mit ThreadPoolExecutor
        - Verarbeitet jeden Chunk mit Batch-Embeddings
        - Gibt Progress-Updates via Callback (thread-safe)
        - Optimiert Speicher-Nutzung und CPU-Auslastung

        Args:
            text: Der zu ingestierende Text
            chunk_size: Anzahl Sätze pro Chunk (default: 100)
            progress_callback: Optional callback(current, total, stats) für Progress-Updates
            max_workers: Maximale Anzahl paralleler Worker (default: CPU-Cores * 2)

        Returns:
            Dictionary mit Statistiken wie ingest_text()
        """
        # PHASE 3: Erstelle Episode für diesen Ingestion-Vorgang
        episode_id = self.netzwerk.create_episode(
            episode_type="ingestion_large",
            content=(
                text[:1000] + "..." if len(text) > 1000 else text
            ),  # Truncate für Speicher
            metadata={"method": "batch_processing", "chunk_size": chunk_size},
        )

        if not episode_id:
            logger.warning("Konnte keine Episode erstellen, fahre ohne Tracking fort")

        doc = self.preprocessor.process(text)
        total_stats = {
            "facts_created": 0,
            "learned_patterns": 0,
            "fallback_patterns": 0,
            "chunks_processed": 0,
            "fragments_stored": 0,
            "connections_stored": 0,
        }

        # Prüfe ob Embedding-Service verfügbar ist
        if not self.embedding_service or not self.embedding_service.is_available():
            logger.warning("Embedding-Service nicht verfügbar, verwende Legacy-Methode")
            # Fallback auf Legacy
            facts = self.ingest_text_legacy(text)
            return {
                "facts_created": facts,
                "learned_patterns": 0,
                "fallback_patterns": facts,
                "chunks_processed": 1,
            }

        # Sammle alle Sätze
        all_sentences = []
        for sentence in doc.sents:
            sentence_text = sentence.text.strip()
            if sentence_text:
                all_sentences.append(sentence_text)

        if not all_sentences:
            logger.debug("Keine Sätze zum Verarbeiten gefunden")
            return total_stats

        total_sentences = len(all_sentences)

        # Prüfe ob parallele Verarbeitung aktiviert ist
        parallel_enabled = config.parallel_processing_enabled

        # Bestimme Worker-Anzahl aus Config oder Parameter
        if max_workers is None:
            max_workers = config.parallel_processing_max_workers

        # Auto-detect wenn immer noch None
        if max_workers is None:
            max_workers = min(
                os.cpu_count() * 2, 8
            )  # Max 8 Worker um Overhead zu vermeiden

        # Log Processing-Mode
        if parallel_enabled and max_workers > 1:
            logger.info(
                f"Starte Batch-Processing: {total_sentences} Sätze in Chunks zu {chunk_size} "
                f"(PARALLELE Verarbeitung mit {max_workers} Workers)"
            )
        else:
            logger.info(
                f"Starte Batch-Processing: {total_sentences} Sätze in Chunks zu {chunk_size} "
                f"(SEQUENZIELLE Verarbeitung)"
            )
            parallel_enabled = False  # Force sequential if only 1 worker

        # Thread-safe Lock für Statistik-Updates
        stats_lock = threading.Lock()
        sentences_processed = 0

        # Erstelle Chunk-Liste mit (start, end) Tupeln
        chunk_ranges = [
            (i, min(i + chunk_size, total_sentences))
            for i in range(0, total_sentences, chunk_size)
        ]

        def process_chunk_wrapper(chunk_range):
            """
            Wrapper-Funktion für parallele Chunk-Verarbeitung.

            Args:
                chunk_range: Tuple (start, end) für Satz-Indizes

            Returns:
                Tuple (chunk_end, chunk_stats)
            """
            chunk_start, chunk_end = chunk_range
            chunk_sentences = all_sentences[chunk_start:chunk_end]
            chunk_num = chunk_start // chunk_size + 1

            logger.debug(
                f"[Worker] Verarbeite Chunk {chunk_num}: Sätze {chunk_start}-{chunk_end}"
            )

            # BATCH-EMBEDDING für Chunk
            try:
                sentence_embeddings = self.embedding_service.get_embeddings_batch(
                    chunk_sentences
                )
            except Exception as e:
                logger.warning(
                    f"Batch-Embedding für Chunk {chunk_num} fehlgeschlagen: {e}"
                )
                sentence_embeddings = [None] * len(chunk_sentences)

            # Verarbeite Chunk
            chunk_stats = self._process_chunk(
                chunk_sentences, sentence_embeddings, episode_id
            )

            return (chunk_end, chunk_stats)

        # VERARBEITUNG: Parallel oder Sequenziell
        if parallel_enabled and max_workers > 1:
            # PARALLEL-VERARBEITUNG mit ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit alle Chunks
                future_to_chunk = {
                    executor.submit(process_chunk_wrapper, chunk_range): chunk_range
                    for chunk_range in chunk_ranges
                }

                # Verarbeite Ergebnisse sobald sie fertig sind
                for future in as_completed(future_to_chunk):
                    chunk_range = future_to_chunk[future]
                    try:
                        chunk_end, chunk_stats = future.result()

                        # Thread-safe Update der Gesamtstatistiken
                        with stats_lock:
                            total_stats["facts_created"] += chunk_stats["facts_created"]
                            total_stats["learned_patterns"] += chunk_stats[
                                "learned_patterns"
                            ]
                            total_stats["fallback_patterns"] += chunk_stats[
                                "fallback_patterns"
                            ]
                            total_stats["fragments_stored"] += chunk_stats.get(
                                "fragments_stored", 0
                            )
                            total_stats["connections_stored"] += chunk_stats.get(
                                "connections_stored", 0
                            )
                            total_stats["chunks_processed"] += 1
                            sentences_processed = chunk_end

                            # Progress Callback (thread-safe)
                            if progress_callback:
                                progress_callback(
                                    sentences_processed, total_sentences, total_stats
                                )

                    except Exception as e:
                        chunk_start, chunk_end = chunk_range
                        logger.error(
                            f"Fehler bei Chunk-Verarbeitung (Sätze {chunk_start}-{chunk_end}): {e}",
                            exc_info=True,
                        )
        else:
            # SEQUENZIELLE VERARBEITUNG (Fallback)
            for chunk_range in chunk_ranges:
                try:
                    chunk_end, chunk_stats = process_chunk_wrapper(chunk_range)

                    # Update Gesamtstatistiken
                    total_stats["facts_created"] += chunk_stats["facts_created"]
                    total_stats["learned_patterns"] += chunk_stats["learned_patterns"]
                    total_stats["fallback_patterns"] += chunk_stats["fallback_patterns"]
                    total_stats["fragments_stored"] += chunk_stats.get(
                        "fragments_stored", 0
                    )
                    total_stats["connections_stored"] += chunk_stats.get(
                        "connections_stored", 0
                    )
                    total_stats["chunks_processed"] += 1

                    # Progress Callback
                    if progress_callback:
                        progress_callback(chunk_end, total_sentences, total_stats)

                except Exception as e:
                    chunk_start, chunk_end = chunk_range
                    logger.error(
                        f"Fehler bei Chunk-Verarbeitung (Sätze {chunk_start}-{chunk_end}): {e}",
                        exc_info=True,
                    )

        # Logging der Statistiken
        logger.info(
            f"Batch-Processing abgeschlossen: {total_stats['facts_created']} Fakten aus {total_sentences} Sätzen "
            f"({total_stats['chunks_processed']} Chunks, "
            f"{total_stats['learned_patterns']} via Prototypen, "
            f"{total_stats['fallback_patterns']} via Brute-Force)"
        )

        return total_stats

    def _process_chunk(
        self,
        sentences: List[str],
        embeddings: List[Optional[list]],
        episode_id: Optional[str] = None,
    ) -> Dict[str, int]:
        """
        Verarbeitet einen Chunk von Sätzen mit vorberechneten Embeddings.

        Args:
            sentences: Liste von Sätzen
            embeddings: Liste von vorberechneten Embeddings (parallel zu sentences)
            episode_id: Optional - Episode-ID für Tracking

        Returns:
            Statistik-Dictionary
        """
        stats = {
            "facts_created": 0,
            "learned_patterns": 0,
            "fallback_patterns": 0,
            "fragments_stored": 0,
            "connections_stored": 0,
        }

        for sentence_text, vector in zip(sentences, embeddings):
            # WORD USAGE TRACKING
            if config.word_usage_tracking_enabled:
                try:
                    usage_stats = self._track_word_usage(sentence_text)
                    stats["fragments_stored"] += usage_stats["fragments_stored"]
                    stats["connections_stored"] += usage_stats["connections_stored"]
                except Exception as e:
                    logger.error(f"Fehler beim Word Usage Tracking (ignoriert): {e}")

            try:
                # Vektor-basiertes Routing
                if not vector:
                    logger.warning(
                        f"Kein Embedding für Satz: '{sentence_text[:50]}...'"
                    )
                    facts = self.apply_all_rules(sentence_text, episode_id)
                    stats["facts_created"] += facts
                    stats["fallback_patterns"] += 1
                    continue

                match_result = self.prototyping_engine.find_best_match(vector)

                if match_result:
                    prototype, distance = match_result

                    if distance < NOVELTY_THRESHOLD:
                        rule = self.netzwerk.get_rule_for_prototype(prototype["id"])

                        if rule:
                            facts = self.apply_single_rule(
                                sentence_text, rule, episode_id
                            )
                            stats["facts_created"] += facts
                            stats["learned_patterns"] += 1
                            continue

                # Fallback
                facts = self.apply_all_rules(sentence_text, episode_id)
                stats["facts_created"] += facts
                stats["fallback_patterns"] += 1

            except Exception as e:
                logger.error(f"Fehler bei Satz-Verarbeitung: {e}", exc_info=True)
                # Graceful Degradation
                facts = self.apply_all_rules(sentence_text, episode_id)
                stats["facts_created"] += facts
                stats["fallback_patterns"] += 1

        return stats

    def ingest_text_legacy(self, text: str) -> int:
        """
        PHASE 6.5: Legacy-Implementierung der Ingestion (Brute-Force).
        Behält für Fallback und Tests. Wendet alle Regeln auf alle Sätze an.

        Args:
            text: Der zu ingestierende Text

        Returns:
            Anzahl der erstellten Fakten
        """
        doc = self.preprocessor.process(text)
        facts_created = 0

        all_rules = self.netzwerk.get_all_extraction_rules()
        if not all_rules:
            logger.warning("Keine Extraktionsregeln gefunden.")
            return 0

        for sentence in doc.sents:
            sentence_text = sentence.text.strip()
            if not sentence_text:
                continue

            logger.debug(f"Verarbeite Satz: '{sentence_text}'")
            facts_created += self.apply_all_rules(sentence_text)

        logger.info(f"Ingestion abgeschlossen: {facts_created} neue Fakten erstellt.")
        return facts_created

    def apply_single_rule(
        self, sentence_text: str, rule: Dict[str, str], episode_id: Optional[str] = None
    ) -> int:
        """
        PHASE 6.1: Wendet eine einzelne Extraktionsregel auf einen Satz an.
        PHASE 3: Erweitert mit Episode-Tracking für Wissensherkunft.

        Args:
            sentence_text: Der zu verarbeitende Satz
            rule: Dictionary mit 'relation_type' und 'regex_pattern'
            episode_id: Optional - ID der Episode für Source-Tracking

        Returns:
            Anzahl der erstellten Fakten (0 oder 1)
        """
        try:
            match = re.match(rule["regex_pattern"], sentence_text, re.IGNORECASE)

            if match and len(match.groups()) >= 2:
                subject_raw = match.group(1).strip()
                object_raw = match.group(2).strip()

                # Normalisierung
                subject = self.formatter.clean_entity(subject_raw)
                obj = self.formatter.clean_entity(object_raw)

                if not subject or not obj or len(subject) < 2 or len(obj) < 2:
                    logger.debug(
                        f"  -> Überspringe: subject='{subject}', object='{obj}'"
                    )
                    return 0

                logger.debug(
                    f"  -> Extrahiert: ({subject})-[{rule['relation_type']}]->({obj})"
                )

                created = self.netzwerk.assert_relation(
                    subject, rule["relation_type"], obj, sentence_text
                )

                # PHASE 3: Verknüpfe Fakt mit Episode (falls Episode-ID vorhanden)
                if created and episode_id:
                    link_success = self.netzwerk.link_fact_to_episode(
                        subject, rule["relation_type"], obj, episode_id
                    )
                    if link_success:
                        logger.debug(f"  -> Mit Episode {episode_id[:8]} verknüpft")

                if created:
                    logger.info(
                        f"  -> GESPEICHERT: ({subject})-[{rule['relation_type']}]->({obj})"
                    )
                    return 1
                else:
                    logger.debug(
                        f"  -> Bereits bekannt: ({subject})-[{rule['relation_type']}]->({obj})"
                    )
                    return 0

        except Neo4jWriteError as e:
            # Spezifischer Fehler beim Schreiben in Neo4j
            logger.error(
                f"Neo4jWriteError beim Verarbeiten von '{sentence_text}' mit Regel {rule['relation_type']}: {e}"
            )
            user_msg = get_user_friendly_message(e)
            logger.info(f"User-Message: {user_msg}")
            return 0
        except Exception as e:
            # Unerwarteter Fehler
            logger.error(
                f"Unerwarteter Fehler beim Verarbeiten von '{sentence_text}' mit Regel {rule['relation_type']}: {e}"
            )
            return 0

        return 0

    def apply_all_rules(
        self, sentence_text: str, episode_id: Optional[str] = None
    ) -> int:
        """
        PHASE 6.1: Wendet alle verfügbaren Extraktionsregeln auf einen Satz an (Brute-Force).
        PHASE 3: Erweitert mit Episode-Tracking.
        Dies ist der Fallback für unbekannte Muster.

        Args:
            sentence_text: Der zu verarbeitende Satz
            episode_id: Optional - ID der Episode für Source-Tracking

        Returns:
            Anzahl der erstellten Fakten
        """
        all_rules = self.netzwerk.get_all_extraction_rules()
        if not all_rules:
            logger.warning("Keine Extraktionsregeln gefunden.")
            return 0

        facts_created = 0
        for rule in all_rules:
            # Versuche Regel anzuwenden
            created = self.apply_single_rule(sentence_text, rule, episode_id)
            facts_created += created

            # WICHTIG: Break nach erstem Match (wie im Original)
            if created > 0:
                break

        return facts_created

    def _track_word_usage(self, sentence_text: str) -> Dict[str, int]:
        """
        Tracked Wortverwendungen für einen Satz (Fragmente + Connections).

        Wird nur aufgerufen wenn word_usage_tracking in Config aktiviert ist.

        Args:
            sentence_text: Der Satz

        Returns:
            Dictionary mit Statistiken: {
                "fragments_stored": Anzahl gespeicherter Fragmente,
                "connections_stored": Anzahl gespeicherter Connections
            }
        """
        stats = {"fragments_stored": 0, "connections_stored": 0}

        try:
            # Extrahiere Fragmente und Connections
            fragments, connections = (
                self.text_fragmenter.extract_fragments_and_connections(sentence_text)
            )

            # Speichere Fragmente
            for fragment in fragments:
                # Stelle sicher, dass Wort existiert
                self.netzwerk.ensure_wort_und_konzept(fragment.lemma)

                # Füge Usage Context hinzu
                success = self.netzwerk.add_usage_context(
                    word_lemma=fragment.lemma,
                    fragment=fragment.fragment,
                    word_position=fragment.word_position,
                    fragment_type=fragment.fragment_type,
                )

                if success:
                    stats["fragments_stored"] += 1

            # Speichere Connections
            for connection in connections:
                # Stelle sicher, dass beide Wörter existieren
                self.netzwerk.ensure_wort_und_konzept(connection.word1_lemma)
                self.netzwerk.ensure_wort_und_konzept(connection.word2_lemma)

                # Füge Connection hinzu
                success = self.netzwerk.add_word_connection(
                    word1_lemma=connection.word1_lemma,
                    word2_lemma=connection.word2_lemma,
                    distance=connection.distance,
                    direction=connection.direction,
                )

                if success:
                    stats["connections_stored"] += 1

            logger.debug(
                "Word Usage getrackt",
                extra={
                    "sentence_preview": sentence_text[:50],
                    "fragments": stats["fragments_stored"],
                    "connections": stats["connections_stored"],
                },
            )

        except Exception as e:
            logger.error(f"Fehler beim Word Usage Tracking: {e}", exc_info=True)

        return stats
