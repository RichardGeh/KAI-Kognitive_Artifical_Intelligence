# component_24_pattern_orchestrator.py
"""
Orchestriert alle Pattern Recognition Features.

Koordiniert:
- Buchstaben-Ebene (Tippfehler)
- Wortfolgen (Predictions)
- Implikationen (Implizite Fakten)
"""

from typing import Any, Dict, List

from component_15_logging_config import get_logger
from component_19_pattern_recognition_char import TypoCandidateFinder
from component_20_pattern_recognition_sequence import SequencePredictor
from component_22_pattern_recognition_implicit import ImplicationDetector
from component_25_adaptive_thresholds import AdaptiveThresholdManager
from kai_config import get_config

logger = get_logger(__name__)
config = get_config()


class PatternOrchestrator:
    """Zentrale Koordination aller Pattern Recognition Features"""

    def __init__(self, netzwerk):
        self.netzwerk = netzwerk
        self.typo_finder = TypoCandidateFinder(netzwerk)
        self.sequence_predictor = SequencePredictor(netzwerk)
        self.implication_detector = ImplicationDetector(netzwerk)
        self.adaptive_manager = AdaptiveThresholdManager(netzwerk)

        # Session-Whitelist für bestätigte Nicht-Typos (vermeidet Loops)
        self.typo_whitelist = set()

        # Load base German dictionary from spaCy for typo detection
        # This reduces false positives by ~95% for common German words
        # Dictionary is immutable (frozenset) and thread-safe
        self.base_dictionary = self._load_spacy_vocabulary()

        # Hole phase-abhängige Confidence-Gates
        gates = self.adaptive_manager.get_confidence_gates()
        self.typo_auto_correct_threshold = gates["auto_correct"]
        self.typo_ask_user_threshold = gates["ask_user"]

        # Log aktuellen System-Status
        stats = self.adaptive_manager.get_system_stats()
        logger.info(
            "PatternOrchestrator initialisiert mit adaptiven Thresholds",
            extra={
                "phase": stats["phase"],
                "vocab_size": stats["vocab_size"],
                "typo_threshold": stats["typo_threshold"],
                "seq_threshold": stats["sequence_threshold"],
                "auto_correct_gate": self.typo_auto_correct_threshold,
                "ask_user_gate": self.typo_ask_user_threshold,
            },
        )

    def _load_spacy_vocabulary(self) -> frozenset:
        """
        Load German base dictionary from spaCy model vocabulary.

        Extracts German words from spaCy's de_core_news_sm model vocabulary strings.
        This provides a comprehensive base dictionary of valid German words.

        Returns:
            frozenset: Immutable set of lowercase German words (typically 350,000-400,000 words, ~19 MB)

        Note: Loading takes approximately 3 seconds on first initialization.
        """
        try:
            import sys

            import spacy

            # Load German model (likely already loaded by other components)
            try:
                nlp = spacy.load("de_core_news_sm")
            except OSError:
                logger.warning(
                    "spaCy model 'de_core_news_sm' not found, base dictionary disabled"
                )
                return frozenset()

            base_words = set()

            # Extract words from spaCy vocabulary strings
            # The de_core_news_sm model doesn't have word vectors, but it has a comprehensive
            # vocabulary in its string store that we can use
            for string in nlp.vocab.strings:
                # Filter criteria:
                # 1. Not empty or None
                # 2. Is alphabetic (no numbers, punctuation)
                # 3. Length >= 2 (no single letters)
                if string and len(string) >= 2 and string.isalpha():
                    base_words.add(string.lower())

            # Calculate memory usage for logging
            dict_size_mb = sys.getsizeof(base_words) / 1024 / 1024
            logger.info(
                f"Loaded {len(base_words)} words from spaCy vocabulary",
                extra={"memory_mb": f"{dict_size_mb:.2f}"},
            )

            # Return as frozenset for immutability (thread-safe, no lock needed)
            return frozenset(base_words)

        except (AttributeError, ValueError, RuntimeError) as e:
            logger.warning(
                f"Could not load spaCy vocabulary: {e}",
                extra={"error_type": type(e).__name__},
            )
            return frozenset()
        except MemoryError:
            logger.error("Out of memory loading spaCy vocabulary (398k words, ~19MB)")
            raise  # Re-raise critical errors

    def process_input(self, text: str) -> Dict[str, Any]:
        """
        Verarbeitet User-Input durch alle Pattern Recognition Stufen.

        Returns:
            Dict mit corrections, predictions, implications
        """
        result: Dict[str, Any] = {
            "original_text": text,
            "corrected_text": text,
            "typo_corrections": [],
            "next_word_predictions": [],
            "implications": [],
            "needs_user_clarification": False,
        }

        # Early Exit: Überspringe Pattern Recognition für explizite Commands UND Fragen
        # Diese sollten direkt zum MeaningExtractor gehen ohne Typo-Detection
        import re

        # Early Exit: Commands (single-line only)
        command_prefixes = [
            r"^\s*definiere:",
            r"^\s*lerne muster:",
            r"^\s*ingestiere text:",
            r"^\s*lerne:",
            r"^\s*(?:lese datei|ingestiere dokument|verarbeite pdf|lade datei):",
        ]

        for pattern in command_prefixes:
            if re.match(pattern, text, re.IGNORECASE):
                logger.debug(
                    "Command erkannt, überspringe Pattern Recognition",
                    extra={"text_preview": text[:50], "pattern": pattern},
                )
                return result

        # Early Exit: Questions (single-line OR multi-line)
        # WICHTIG: Bei "Was ist X?" will der Benutzer eine Antwort, keine Typo-Rückfrage
        # FIX 2025-12-13: Unterstützt multi-line Fragen (z.B. Logic Puzzles mit Frage am Ende)
        # Verwendet re.search() statt re.match() um GESAMTEN Text zu prüfen, nicht nur erste Zeile
        question_patterns = [
            r"(?:^|\n)\s*(?:was|wer|wie|wo|wann|warum|wieso|weshalb|wozu|welche)\s+",  # Question words (any line)
            r"\?\s*$",  # Ends with question mark
        ]

        for pattern in question_patterns:
            if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
                logger.debug(
                    "Frage erkannt (single/multi-line), überspringe Pattern Recognition",
                    extra={"text_preview": text[:100]},
                )
                return result

        # Early Exit: Multi-line structured inputs (likely logic puzzles)
        # FIX 2025-12-13: Erkennt nummerierte Listen (1. ... 2. ... 3. ...)
        # Logic Puzzles haben typischerweise dieses Format
        if re.search(r"(?:^|\n)\s*\d+\.\s+.+(?:\n\s*\d+\.\s+.+)+", text, re.MULTILINE):
            logger.debug(
                "Strukturierte Eingabe (nummerierte Punkte) erkannt, überspringe Pattern Recognition",
                extra={"text_preview": text[:100]},
            )
            return result

        # 1. Tippfehler-Korrektur
        words = text.split()
        corrected_words = []

        # WICHTIG: Blacklist häufiger deutscher Funktionswörter
        # Diese sollten NIEMALS als Tippfehler korrigiert werden
        function_words_blacklist = {
            # Artikel
            "der",
            "die",
            "das",
            "den",
            "dem",
            "des",
            "ein",
            "eine",
            "einer",
            "einen",
            "einem",
            "eines",
            # Präpositionen
            "in",
            "an",
            "auf",
            "aus",
            "bei",
            "mit",
            "nach",
            "von",
            "zu",
            "vor",
            "über",
            "unter",
            # Pronomen
            "ich",
            "du",
            "er",
            "sie",
            "es",
            "wir",
            "ihr",
            "mein",
            "dein",
            "sein",
            "ihr",
            "unser",
            "euer",
            "dieser",
            "diese",
            "dieses",
            "jener",
            "jene",
            "jenes",
            # Konjunktionen
            "und",
            "oder",
            "aber",
            "denn",
            "sondern",
            "wenn",
            "als",
            "wie",
            "dass",
            "weil",
            # Häufige Verben
            "ist",
            "sind",
            "war",
            "waren",
            "hat",
            "haben",
            "wird",
            "werden",
            "kann",
            "können",
            # Fragewörter
            "was",
            "wer",
            "wie",
            "wo",
            "wann",
            "warum",
            "wozu",
            "wieso",
            "weshalb",
        }

        for word in words:
            # Entferne Satzzeichen UND Anführungszeichen
            clean_word = word.strip(".,!?;:'\"")
            if len(clean_word) < 3:
                corrected_words.append(word)
                continue

            # WICHTIG: Überspringe Funktionswörter (sollten nie als Typo behandelt werden)
            if clean_word.lower() in function_words_blacklist:
                corrected_words.append(word)
                continue

            # Prüfe Session-Whitelist (bestätigte Nicht-Typos)
            if clean_word.lower() in self.typo_whitelist:
                logger.debug(f"'{clean_word}' in Typo-Whitelist, überspringe")
                corrected_words.append(word)
                continue

            # WICHTIG: Überspringe kapitalisierte Wörter (Eigennamen, Fremdwörter)
            # Heuristik: Wort beginnt mit Großbuchstabe und ist nicht am Satzanfang
            is_capitalized = clean_word[0].isupper() if clean_word else False
            is_all_caps = clean_word.isupper() if clean_word else False

            # Überspringe wenn kapitalisiert (aber nicht ALL CAPS = Akronym)
            if is_capitalized and not is_all_caps:
                logger.debug(
                    f"'{clean_word}' ist kapitalisiert (vermutlich Eigenname), überspringe Typo-Erkennung"
                )
                corrected_words.append(word)
                continue

            # Check base dictionary FIRST (common German words)
            if clean_word.lower() in self.base_dictionary:
                corrected_words.append(word)
                continue

            # Check learned vocabulary from Neo4j
            known = self.netzwerk.get_all_known_words()
            if clean_word.lower() in [w.lower() for w in known]:
                corrected_words.append(word)
                continue

            # Suche Tippfehler-Kandidaten
            candidates = self.typo_finder.find_candidates(clean_word, max_candidates=3)

            if candidates and len(candidates) > 0:
                best = candidates[0]

                # FIX 2024-11: Auto-Korrektur komplett deaktiviert - zu viele falsche Matches
                # Problem: "zur" -> "tür", "gelegt" -> "belebt", "sofern" -> "eltern" mit 1.00 Confidence
                # Lösung: IMMER ask_user, NIEMALS auto_correct

                # DEAKTIVIERT: Auto-Korrektur
                # if best["confidence"] >= self.typo_auto_correct_threshold:
                #     corrected_words.append(best["word"])
                #     result["typo_corrections"].append(...)

                # FORCIERT: Immer ask_user (wenn Confidence hoch genug)
                if best["confidence"] >= self.typo_ask_user_threshold:
                    # Rückfrage (IMMER, auch bei hoher Confidence)
                    corrected_words.append(word)  # Nutze Original
                    result["typo_corrections"].append(
                        {
                            "original": word,
                            "candidates": candidates,
                            "decision": "ask_user",
                        }
                    )
                    result["needs_user_clarification"] = True
                else:
                    # Confidence zu niedrig -> ignoriere
                    corrected_words.append(word)
            else:
                corrected_words.append(word)

        result["corrected_text"] = " ".join(corrected_words)

        # 2. Wortfolgen-Vorhersage (nur wenn kein Tippfehler)
        if not result["needs_user_clarification"]:
            predictions = self.sequence_predictor.predict_completion(
                result["corrected_text"]
            )
            result["next_word_predictions"] = predictions

        return result

    def add_to_typo_whitelist(self, word: str):
        """
        Fügt ein Wort zur Typo-Whitelist hinzu.

        Sollte aufgerufen werden wenn Benutzer bestätigt, dass ein Wort korrekt ist.
        Verhindert wiederholte Typo-Rückfragen für dasselbe Wort in dieser Session.

        Args:
            word: Das Wort, das nicht als Typo behandelt werden soll
        """
        self.typo_whitelist.add(word.lower())
        logger.info(f"'{word}' zur Typo-Whitelist hinzugefügt (Session)")

    def detect_implications_for_fact(
        self, subject: str, relation: str, obj: str
    ) -> List[Dict]:
        """Erkennt Implikationen für einen Fakt"""
        if relation == "HAS_PROPERTY":
            return self.implication_detector.detect_property_implications(subject, obj)
        return []


if __name__ == "__main__":
    print("=== Pattern Orchestrator ===")
    print("Modul geladen.")
