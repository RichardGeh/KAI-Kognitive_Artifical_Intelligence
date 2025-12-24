# component_41_input_orchestrator.py
"""
Input Orchestrator - Intelligente Segmentierung und Verarbeitung komplexer Eingaben

Verantwortlichkeiten:
- Segmentierung von Eingaben in Erklärungen und Fragen
- Semantische Klassifikation von Segmenten
- Erstellung von Multi-Step-Plänen für Logik-Rätsel
- Dynamische Anpassung an verschiedene Eingabetypen

Entwickelt für:
- Logik-Rätsel (Erklärung → Frage)
- Mehrere Erklärungen gefolgt von Fragen
- Natürlichsprachliche Verarbeitung ohne starre Muster
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from component_5_linguistik_strukturen import GoalType, MainGoal, SubGoal
from component_15_logging_config import get_logger
from kai_exceptions import ParsingError

logger = get_logger(__name__)

# Constants for segment classification
MAX_ABBREVIATION_LENGTH = 5  # Maximum characters for abbreviation detection
MIN_SEGMENTS_FOR_ORCHESTRATION = 2  # Minimum segments needed for orchestration
MIN_LOGIC_PATTERNS_FOR_PUZZLE = 2  # Minimum logic patterns to classify as puzzle

# Confidence thresholds for segment classification
CONFIDENCE_QUESTION_MARK = 0.95  # High confidence when question mark present
CONFIDENCE_FRAGE_PREFIX = 0.90  # Confidence for "Frage:" prefix detection
CONFIDENCE_QUESTION_WORD = 0.90  # Confidence for question word detection
CONFIDENCE_COMMAND = 1.0  # Maximum confidence for explicit commands
CONFIDENCE_DECLARATIVE = 0.85  # Confidence for declarative patterns
CONFIDENCE_FALLBACK = 0.70  # Lower confidence for fallback classification


class SegmentType(Enum):
    """Typ eines Eingabe-Segments."""

    EXPLANATION = "explanation"  # Erklärung, Kontext, deklarative Aussage
    QUESTION = "question"  # Frage, Anfrage
    COMMAND = "command"  # Befehl (explizit)
    UNKNOWN = "unknown"  # Unklarer Typ


class PuzzleType(Enum):
    """Typ eines Logic Puzzles (für Solver-Routing)."""

    ENTITY_SAT = "entity_sat"  # Entitäten-basierte Rätsel (Leo/Mark/Nick)
    NUMERICAL_CSP = "numerical_csp"  # Zahlen-basierte Constraint-Rätsel
    HYBRID = "hybrid"  # Kombination aus beiden
    UNKNOWN = "unknown"  # Unbestimmt


@dataclass
class InputSegment:
    """
    Repräsentiert ein Segment der Benutzereingabe.

    Attributes:
        text: Der Text des Segments
        segment_type: Typ des Segments (EXPLANATION, QUESTION, etc.)
        confidence: Konfidenz der Klassifikation (0.0-1.0)
        metadata: Zusätzliche Metadaten (z.B. erkannte Entitäten)
    """

    text: str
    segment_type: SegmentType
    confidence: float = 0.8
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_explanation(self) -> bool:
        """Prüft ob Segment eine Erklärung ist."""
        return self.segment_type == SegmentType.EXPLANATION

    def is_question(self) -> bool:
        """Prüft ob Segment eine Frage ist."""
        return self.segment_type == SegmentType.QUESTION

    def is_command(self) -> bool:
        """Prüft ob Segment ein Befehl ist."""
        return self.segment_type == SegmentType.COMMAND


class InputOrchestrator:
    """
    Orchestriert die Verarbeitung komplexer Eingaben.

    Segmentiert Eingaben in Erklärungen und Fragen, klassifiziert diese
    und erstellt optimierte Multi-Step-Pläne für die Verarbeitung.
    """

    def __init__(self, preprocessor=None):
        """
        Initialisiert den Orchestrator.

        Args:
            preprocessor: Optional - LinguisticPreprocessor für erweiterte Analyse
        """
        self.preprocessor = preprocessor

        # Fragewörter für deutsche Sprache
        self.question_words = [
            "was",
            "wer",
            "wie",
            "wo",
            "wann",
            "warum",
            "welche",
            "welcher",
            "welches",
            "wozu",
            "wieso",
            "weshalb",
        ]

        # Explizite Befehls-Präfixe
        self.command_prefixes = [
            "lerne:",
            "definiere:",
            "ingestiere text:",
            "lerne muster:",
            "lese datei:",
            "ingestiere dokument:",
            "verarbeite pdf:",
            "lade datei:",
        ]

        logger.info("InputOrchestrator initialisiert")

    def should_orchestrate(self, text: str) -> bool:
        """
        Prüft ob eine Eingabe orchestriert werden sollte.

        Kriterien:
        - Mehr als ein Satz
        - Enthält eine MISCHUNG aus Segmenttypen (z.B. COMMAND + QUESTION,
          EXPLANATION + QUESTION)
        - Nur homogene Befehle werden NICHT orchestriert

        Args:
            text: Die zu prüfende Eingabe

        Returns:
            True wenn Orchestrierung sinnvoll ist
        """
        # Segmentiere ZUERST (vor Command-Prefix-Pruefung)
        segments = self._segment_text(text)

        # Orchestrierung nur bei mehreren Segmenten
        if len(segments) < MIN_SEGMENTS_FOR_ORCHESTRATION:
            logger.debug(
                f"Nur {len(segments)} Segment(e) - ueberspringe Orchestrierung"
            )
            return False

        # Klassifiziere alle Segmente
        classified_segments = [self.classify_segment(seg) for seg in segments]

        # Zaehle Segmenttypen
        has_explanation = any(s.is_explanation() for s in classified_segments)
        has_question = any(s.is_question() for s in classified_segments)
        has_command = any(s.is_command() for s in classified_segments)

        # Orchestrierung wenn Mischung aus verschiedenen Typen
        # (z.B. COMMAND + QUESTION, EXPLANATION + QUESTION)
        if (has_command or has_explanation) and has_question:
            logger.info(
                f"Orchestrierung aktiviert: {len(classified_segments)} Segmente, "
                f"commands={sum(s.is_command() for s in classified_segments)}, "
                f"explanations={sum(s.is_explanation() for s in classified_segments)}, "
                f"questions={sum(s.is_question() for s in classified_segments)}"
            )
            return True

        # Nur homogene Befehle oder Erklaerungen - keine Orchestrierung noetig
        if has_command and not has_question and not has_explanation:
            logger.debug("Nur Befehle erkannt - ueberspringe Orchestrierung")
            return False

        logger.debug("Keine Mischung erkannt - ueberspringe Orchestrierung")
        return False

    def _is_simple_entity_puzzle(self, segments: List[InputSegment]) -> bool:
        """
        Erkennt einfache Entitäten-basierte Rätsel mit repetitiven Fakten.

        Pattern: "A trinkt X. B trinkt Y. C trinkt Z. Wer trinkt X?"

        Args:
            segments: Liste von klassifizierten Segmenten

        Returns:
            True wenn es ein einfaches Entitäten-Rätsel ist
        """
        # Zähle entity-verb-object Patterns
        entity_fact_count = 0
        for seg in segments:
            if seg.is_explanation():
                # Pattern: [Großgeschriebener Name] [Verb] [Objekt]
                if re.match(
                    r"^[A-Z]\w+\s+(?:trinkt|isst|mag|hat|ist|kauft|nimmt|bestellt)\s+\w+",
                    seg.text.strip(),
                ):
                    entity_fact_count += 1

        # Hat Frage?
        has_question = any(seg.is_question() for seg in segments)

        # Einfaches Entitäten-Rätsel wenn:
        # - Mindestens 3 Entity-Fakten (3+ Personen)
        # - Hat eine Frage
        is_simple_puzzle = entity_fact_count >= 3 and has_question

        if is_simple_puzzle:
            logger.info(
                f"Einfaches Entitäten-Rätsel erkannt: {entity_fact_count} Entity-Fakten"
            )

        return is_simple_puzzle

    def is_logic_puzzle(self, text: str, segments: List[InputSegment]) -> bool:
        """
        Erkennt ob die Eingabe ein Logik-Rätsel ist.

        Kriterien:
        - Enthält logische Bedingungen (wenn...dann, oder, und, nie beide)
        - Endet mit einer Frage
        - Mehrere Erklärungen vor der Frage
        - ODER: Einfache repetitive Entity-Fakten + Frage
        - ODER: Numerisches Constraint-Rätsel
        - ODER: Constraint-basiertes Rätsel (negative constraints, assignments)

        Args:
            text: Der vollständige Eingabetext
            segments: Liste von klassifizierten Segmenten

        Returns:
            True wenn es ein Logik-Rätsel ist
        """
        # Prüfe zunächst auf einfache Entitäten-Rätsel
        if self._is_simple_entity_puzzle(segments):
            return True

        # Prüfe auf numerisches Rätsel
        if self._is_numerical_puzzle(text, segments):
            return True

        # [NEW] Prüfe auf Constraint-basiertes Rätsel
        if self._is_constraint_puzzle(text, segments):
            return True

        # Logik-Puzzle-Patterns (komplexe Rätsel)
        logic_patterns = [
            r"\bwenn\s+.+?,?\s+.+?\b",  # "wenn X, dann Y"
            r"\boder\s+.+?,?\s*aber\s+(?:nie|nicht|niemals)\s+beide",  # XOR
            r"\beinzeln\s+oder\s+gleichzeitig",  # Individual or simultaneous
            r"\bentweder\s+.+?\s+oder\s+",  # "entweder X oder Y"
            r"\bgenau\s+(?:einer|eine)",  # "genau einer"
        ]

        # Zähle Pattern-Matches
        pattern_matches = sum(
            1 for pattern in logic_patterns if re.search(pattern, text, re.IGNORECASE)
        )

        # Prüfe Segment-Struktur
        has_multiple_explanations = sum(s.is_explanation() for s in segments) >= 2
        has_question = any(s.is_question() for s in segments)

        # Logik-Rätsel wenn:
        # - Mindestens MIN_LOGIC_PATTERNS_FOR_PUZZLE logische Patterns erkannt
        # - Mehrere Erklärungen + Frage
        is_puzzle = (
            pattern_matches >= MIN_LOGIC_PATTERNS_FOR_PUZZLE
            and has_multiple_explanations
            and has_question
        )

        if is_puzzle:
            logger.info(
                f"Logik-Rätsel erkannt: {pattern_matches} Pattern-Matches, "
                f"{sum(s.is_explanation() for s in segments)} Erklärungen"
            )

        return is_puzzle

    def _is_numerical_puzzle(self, text: str, segments: List[InputSegment]) -> bool:
        """
        Erkennt numerische Constraint-Rätsel.

        Pattern: "Gesucht ist eine Zahl..." + mathematische Keywords

        Args:
            text: Der vollständige Eingabetext
            segments: Liste von klassifizierten Segmenten

        Returns:
            True wenn es ein numerisches Rätsel ist
        """
        text_lower = text.lower()

        # Pattern: "Gesucht ist/wird eine/die [adj]* Zahl"
        has_number_search = bool(
            re.search(
                r"gesucht\s+(?:ist|wird)\s+(?:eine?|die)\s+(?:\w+\s+)*zahl", text_lower
            )
        )

        # Mathematische Keywords (erweitert)
        has_numerical_keywords = bool(
            re.search(
                r"(?:teilbar|summe|ziffer|größer|kleiner|prim|gerade|ungerade|vielfaches|differenz|produkt|quotient|teiler|beträgt|eigenschaften)",
                text_lower,
            )
        )

        # Hat Frage oder "Welche Zahl"-Pattern?
        has_question = any(seg.is_question() for seg in segments)
        has_welche_zahl = bool(re.search(r"\bwelche\s+zahl\b", text_lower))

        is_numerical = (
            has_number_search
            and has_numerical_keywords
            and (has_question or has_welche_zahl)
        )

        if is_numerical:
            logger.info("Numerisches Constraint-Rätsel erkannt")

        return is_numerical

    def _is_constraint_puzzle(self, text: str, segments: List[InputSegment]) -> bool:
        """
        Prüft ob der Text ein Constraint-Satisfaction-Problem darstellt.

        Verwendet mehrere Heuristiken:
        - Numbered lists (1. 2. 3.) mit 3+ Items + Frage
        - Negative Constraints: "nicht", "kein", "weder...noch"
        - All-Different: "unterschiedliche", "verschiedene"
        - Assignments: "Name verb Value"
        - Uniqueness: "genau eine/einer"

        Args:
            text: Der zu prüfende Text
            segments: Liste von klassifizierten Segmenten

        Returns:
            True wenn Constraint-Problem erkannt wurde, sonst False
        """
        import re

        try:
            has_question = any(seg.is_question() for seg in segments)
            if not has_question:
                return False

            text_lower = text.lower()

            # [NEW] Strong signal: Numbered list with 3+ items
            has_numbered_list = bool(
                re.search(r"(?:\n|^)\s*\d+\.\s+", text, re.MULTILINE)
            )
            numbered_item_count = len(
                re.findall(r"(?:\n|^)\s*\d+\.\s+", text, re.MULTILINE)
            )

            if has_numbered_list and numbered_item_count >= 3 and has_question:
                logger.info(
                    f"Constraint-Rätsel erkannt (Numbered List) | "
                    f"items={numbered_item_count}, has_question={has_question}"
                )
                return True

            # [IMPROVED] Negative Constraint Patterns (generalized for any verb)
            negative_patterns = [
                r"\b\w+\s+kein(?:e|er|en)?\s+",  # "traegt keine", "ist kein"
                r"\b\w+\s+nicht\s+(?:[A-Z]|die|der|das|den|dem|des|ein|eine|einen)",  # "traegt nicht Rot"
                r"\bweder\s+.+?\s+noch\s+",  # "weder Gruen noch Gelb"
                r"\bnicht\s+(?:die|der|das|den)\s+(?:gleiche|selbe)",  # "nicht die gleiche Farbe"
            ]

            # All-Different Constraint Patterns
            all_different_patterns = [
                r"\bunterschiedlich(?:e|en)?\s+",
                r"\bverschieden(?:e|en)?\s+",
                r"\bjeweils\s+(?:eine|ein)\s+",  # "jeweils eine Farbe" (each one different)
            ]

            # [IMPROVED] Positive assignment pattern (generalized for multiple verbs)
            assignment_pattern = r"\b[A-Z][a-z]+\s+(?:ist|sind|traegt|tragen|hat|haben|mag|moegen|trinkt|trinken|isst|essen|kauft|kaufen)\s+[A-Z][a-z]+"

            # [NEW] Uniqueness constraint patterns
            uniqueness_patterns = [
                r"\bgenau\s+(?:eine|einer|ein)\s+",  # "Genau eine Person"
                r"\bnur\s+(?:eine|einer|ein)\s+",  # "Nur einer"
                r"\beinzige\s+",  # "Die einzige"
            ]

            # Count pattern matches
            negative_count = sum(
                len(re.findall(p, text, re.IGNORECASE)) for p in negative_patterns
            )

            all_different_count = sum(
                len(re.findall(p, text_lower, re.IGNORECASE))
                for p in all_different_patterns
            )

            assignment_count = len(re.findall(assignment_pattern, text))

            uniqueness_count = sum(
                len(re.findall(p, text_lower, re.IGNORECASE))
                for p in uniqueness_patterns
            )

            # Real constraints (not just assignments)
            real_constraint_count = (
                negative_count + all_different_count + uniqueness_count
            )

            total_constraint_patterns = real_constraint_count + assignment_count

            # Need at least 1 real constraint (negative, all-different, uniqueness)
            # Simple assignments like "Anna mag Aepfel" alone are NOT constraint puzzles
            # They are just facts to be learned, not puzzles to be solved
            is_real_constraint_puzzle = real_constraint_count >= 1 and has_question

            if is_real_constraint_puzzle:
                logger.info(
                    f"Constraint-Rätsel erkannt (Pattern Matching) | "
                    f"negative={negative_count}, all_different={all_different_count}, "
                    f"assignment={assignment_count}, uniqueness={uniqueness_count}, "
                    f"total={total_constraint_patterns}, has_question={has_question}"
                )
                return True

            return False

        except re.error as e:
            logger.error(f"Regex compilation error in constraint detection: {e}")
            raise ParsingError(
                "Invalid regex pattern in constraint detector",
                context={"text_preview": text[:100]},
                original_exception=e,
            )
        except (AttributeError, TypeError) as e:
            logger.error(
                f"Unexpected error in constraint detection - possible API change: {e}",
                exc_info=True,
            )
            raise ParsingError(
                f"Constraint detection failed due to {type(e).__name__}",
                context={"text_length": len(text), "num_segments": len(segments)},
                original_exception=e,
            )
        # Remove generic catch-all - let real errors propagate

    def classify_logic_puzzle_type(
        self, text: str, segments: List[InputSegment]
    ) -> PuzzleType:
        """
        Klassifiziert den Typ eines Logik-Rätsels.

        Unterscheidet zwischen:
        - ENTITY_SAT: Entitäten-basiert (Leo/Mark/Nick trinkt was)
        - NUMERICAL_CSP: Zahlen-basiert (gesuchte Zahl, teilbar durch X)
        - HYBRID: Kombination aus beiden
        - UNKNOWN: Kein Rätsel oder unbekannt

        Args:
            text: Der vollständige Eingabetext
            segments: Liste von klassifizierten Segmenten

        Returns:
            PuzzleType enum
        """
        # CRITICAL: Check for entity-based puzzles FIRST before numerical puzzles
        # Reason: Entity puzzles often have numbered constraints (1. 2. 3.)
        # which would cause misclassification as numerical puzzles

        # Prüfe auf einfaches Entity-Rätsel
        if self._is_simple_entity_puzzle(segments):
            logger.info("Puzzle-Typ: ENTITY_SAT (simple entity puzzle)")
            return PuzzleType.ENTITY_SAT

        text_lower = text.lower()

        # Patterns für Entitäten-basierte Rätsel
        entity_patterns = [
            r"\b[A-Z][a-z]+\s+(?:trinkt|mag|isst|bestellt|traegt|hat)\b",  # "Leo trinkt", "Anna traegt"
            r"\b(?:wer|was|welche[rs]?)\s+(?:trinkt|mag|isst|bestellt|traegt)\b",  # "Wer/Was trinkt"
            r"\b(?:einer|eine)\s+von\s+(?:ihnen|den\s+dreien)\b",  # "einer von ihnen"
        ]

        # Patterns für numerische/Constraint-basierte Rätsel
        numerical_patterns = [
            r"\bzahl\b",  # "gesuchte Zahl", "die Zahl"
            r"\bteilbar\s+durch\b",  # "teilbar durch X"
            r"\bvielfaches\s+von\b",  # "Vielfaches von"
            r"\bsumme\s+der\b",  # "Summe der Nummern"
            r"\bdifferenz\b",  # "Differenz"
            r"\bprodukt\b",  # "Produkt"
            r"\bquotient\b",  # "Quotient"
            r"\banzahl\s+der\b",  # "Anzahl der Teiler"
            r"\bteiler\b",  # "Teiler"
            r"\brichtig(?:e|en)?\s+behauptung(?:en)?\b",  # "richtige Behauptungen"
            r"\bfalsch(?:e|en)?\s+behauptung(?:en)?\b",  # "falsche Behauptungen"
            r"\b(?:erste|letzte|n-te)\s+(?:richtig|falsch)\b",  # meta-constraints
            # NOTE: Removed r"\b\d+\.\s" (numbered lists) - not reliable for numerical classification
        ]

        # Zähle Matches
        entity_count = sum(
            len(re.findall(p, text, re.IGNORECASE))
            for p in entity_patterns  # Use text (not text_lower) for case-sensitive entity patterns
        )

        numerical_count = sum(
            len(re.findall(p, text_lower, re.IGNORECASE)) for p in numerical_patterns
        )

        # IMPROVED: Prioritize entity classification if entity patterns found
        # Klassifikation basierend auf Matches (entity-biased)
        if entity_count >= 2:
            # Entity patterns dominate
            if numerical_count == 0:
                logger.info(
                    f"Puzzle-Typ: ENTITY_SAT ({entity_count} Entitäts-Patterns)"
                )
                return PuzzleType.ENTITY_SAT
            else:
                logger.info(
                    f"Puzzle-Typ: ENTITY_SAT (dominant) ({entity_count} Entitäts-Patterns, {numerical_count} numerisch)"
                )
                return PuzzleType.ENTITY_SAT  # Entity patterns take precedence
        elif numerical_count >= 2:
            logger.info(
                f"Puzzle-Typ: NUMERICAL_CSP ({numerical_count} numerische Patterns)"
            )
            return PuzzleType.NUMERICAL_CSP
        elif entity_count >= 1 and numerical_count >= 1:
            # Ambiguous - default to entity if any entity patterns
            logger.info(
                f"Puzzle-Typ: ENTITY_SAT (fallback) ({entity_count} Entitäts-Patterns, {numerical_count} numerisch)"
            )
            return PuzzleType.ENTITY_SAT
        else:
            logger.debug("Puzzle-Typ: UNKNOWN (nicht genug Patterns)")
            return PuzzleType.UNKNOWN

    def orchestrate_input(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Orchestriert eine komplexe Eingabe.

        Returns:
            Dictionary mit:
            - segments: Liste von InputSegment
            - plan: MainGoal für orchestrierte Verarbeitung
            - metadata: Zusätzliche Informationen
            - is_logic_puzzle: Bool ob es ein Logik-Rätsel ist

            None wenn keine Orchestrierung notwendig
        """
        if not self.should_orchestrate(text):
            return None

        # Segmentiere und klassifiziere
        raw_segments = self._segment_text(text)
        segments = [self.classify_segment(seg) for seg in raw_segments]

        # Prüfe ob es ein Logik-Rätsel ist
        is_puzzle = self.is_logic_puzzle(text, segments)

        # Klassifiziere Puzzle-Typ (falls Rätsel erkannt)
        puzzle_type = PuzzleType.UNKNOWN
        if is_puzzle:
            puzzle_type = self.classify_logic_puzzle_type(text, segments)

        # Erstelle orchestrierten Plan
        plan = self._create_orchestrated_plan(segments)

        logger.info(
            f"Orchestrierung abgeschlossen: {len(segments)} Segmente verarbeitet"
            + (f", Logik-Rätsel erkannt ({puzzle_type.value})" if is_puzzle else "")
        )

        return {
            "segments": segments,
            "plan": plan,
            "is_logic_puzzle": is_puzzle,
            "puzzle_type": puzzle_type,
            "metadata": {
                "explanation_count": sum(s.is_explanation() for s in segments),
                "question_count": sum(s.is_question() for s in segments),
                "total_segments": len(segments),
                "puzzle_classification": puzzle_type.value if is_puzzle else None,
            },
        }

    def _merge_abbreviations(self, segments: List[str]) -> List[str]:
        """
        Merge abbreviations (e.g., "Dr.") with the following segment.

        SpaCy sometimes splits abbreviations as separate sentences.
        This post-processing step merges them back.

        Args:
            segments: List of text segments

        Returns:
            Merged segments with abbreviations combined
        """
        # Common German abbreviations that should be merged
        abbreviations = [
            "dr.",
            "prof.",
            "mr.",
            "mrs.",
            "ms.",
            "st.",
            "ca.",
            "z.b.",
            "d.h.",
            "bzw.",
            "usw.",
            "etc.",
        ]

        merged = []
        i = 0
        while i < len(segments):
            segment = segments[i]

            # Check if this is a short segment ending with "." (likely abbreviation)
            is_abbrev = (
                len(segment) <= MAX_ABBREVIATION_LENGTH
                and segment.endswith(".")
                and segment.lower() in abbreviations
            )

            # If abbreviation and there's a next segment, merge them
            if is_abbrev and i + 1 < len(segments):
                merged_segment = segment + " " + segments[i + 1]
                merged.append(merged_segment)
                logger.debug(
                    f"Merged abbreviation: '{segment}' + '{segments[i + 1]}' -> '{merged_segment}'"
                )
                i += 2  # Skip next segment as it was merged
            else:
                merged.append(segment)
                i += 1

        return merged

    def _merge_question_words(self, segments: List[str]) -> List[str]:
        """
        Merge standalone question words with the following segment.

        Sometimes spaCy splits question words (wer, was, wie, etc.) as separate sentences.
        This post-processing step merges them with the following segment to keep
        questions intact.

        Args:
            segments: List of text segments

        Returns:
            Merged segments with question words combined
        """
        merged = []
        i = 0
        while i < len(segments):
            segment = segments[i]
            segment_lower = segment.lower().strip()

            # Check if this segment is ONLY a question word (or question word + punctuation)
            is_lone_question_word = (
                segment_lower.rstrip("?!.,;: ") in self.question_words
            )

            # If lone question word and there's a next segment, merge them
            if is_lone_question_word and i + 1 < len(segments):
                merged_segment = segment.rstrip() + " " + segments[i + 1]
                merged.append(merged_segment)
                logger.debug(
                    f"Merged question word: '{segment}' + '{segments[i + 1]}' -> '{merged_segment[:50]}...'"
                )
                i += 2  # Skip next segment as it was merged
            else:
                merged.append(segment)
                i += 1

        return merged

    def _merge_numbered_items(self, segments: List[str]) -> List[str]:
        """
        Merge standalone numbered items (1., 2., 3., etc.) with following segment.

        spaCy sometimes splits numbered list items as separate sentences.
        Pattern: "1." (alone) + "Anna traegt..." -> "1. Anna traegt..."

        Args:
            segments: List of text segments

        Returns:
            Merged segments with numbered items combined
        """
        merged = []
        i = 0
        while i < len(segments):
            segment = segments[i].strip()

            # Check if segment is ONLY a number followed by period: "1.", "2.", etc.
            is_numbered_item = bool(re.match(r"^\d+\.$", segment))

            # If numbered item and there's a next segment, merge them
            if is_numbered_item and i + 1 < len(segments):
                merged_segment = segment + " " + segments[i + 1]
                merged.append(merged_segment)
                logger.debug(
                    f"Merged numbered item: '{segment}' + '{segments[i + 1][:30]}...' "
                    f"-> '{merged_segment[:50]}...'"
                )
                i += 2  # Skip next segment as it was merged
            else:
                merged.append(segment)
                i += 1

        return merged

    def _strip_turn_labels(self, text: str) -> str:
        """
        Strip dialogue turn labels like 'Turn 1:', 'Schritt 2:' before segmentation.

        These labels are metadata, not content, and should not be parsed as entities.

        Args:
            text: The text to clean

        Returns:
            Text with turn labels removed
        """
        # Pattern: "Turn 1:", "Schritt 2:", "Runde 3:", "Step 4:", etc.
        pattern = r"^(Turn|Schritt|Runde|Step)\s*\d+\s*:\s*"
        lines = text.split("\n")
        cleaned_lines = []
        for line in lines:
            cleaned = re.sub(pattern, "", line.strip(), flags=re.IGNORECASE)
            if cleaned:
                cleaned_lines.append(cleaned)
        logger.debug(
            f"Stripped turn labels from {len(lines)} lines -> {len(cleaned_lines)} cleaned lines"
        )
        return "\n".join(cleaned_lines)

    def _normalize_sentence_boundaries(self, text: str) -> str:
        """
        Normalize sentence boundaries so spaCy can properly segment them.

        spaCy's sentence tokenizer does not recognize single newlines as sentence
        boundaries. This method converts ".\n" (period + single newline) to ".\n\n"
        (period + double newline) to ensure proper segmentation.

        Args:
            text: The text to normalize

        Returns:
            Text with normalized sentence boundaries
        """
        # Replace period followed by single newline with period + double newline
        # This ensures spaCy treats them as separate sentences
        normalized = re.sub(r"\.(\n)(?!\n)", r".\n\n", text)
        # Also handle question marks and exclamation marks
        normalized = re.sub(r"\?(\n)(?!\n)", r"?\n\n", normalized)
        normalized = re.sub(r"!(\n)(?!\n)", r"!\n\n", normalized)
        logger.debug("Normalized sentence boundaries for spaCy segmentation")
        return normalized

    def _segment_text(self, text: str) -> List[str]:
        """
        Segmentiert Text in Sätze mit spaCy's Sentence Tokenizer.

        Strategie:
        1. Primär: spaCy Sentence Tokenizer (präzise, handled Dr., Zahlen, etc.)
        2. Fallback: Regex-basierte Segmentierung wenn spaCy nicht verfügbar

        Args:
            text: Der zu segmentierende Text

        Returns:
            Liste von Text-Segmenten
        """
        # Pre-processing: Strip turn labels first (e.g., "Turn 1:", "Schritt 2:")
        text = self._strip_turn_labels(text)

        # Pre-processing: Normalize sentence boundaries for spaCy
        # spaCy does not recognize single newlines as sentence boundaries
        text = self._normalize_sentence_boundaries(text)

        # Strategie 1: spaCy Sentence Tokenizer (bevorzugt)
        if self.preprocessor and hasattr(self.preprocessor, "nlp"):
            try:
                doc = self.preprocessor.process(text)
                segments = [
                    sent.text.strip() for sent in doc.sents if sent.text.strip()
                ]

                # Post-processing: Merge abbreviations (like "Dr.") with next segment
                segments = self._merge_abbreviations(segments)

                # Post-processing: Merge question words with following segment
                segments = self._merge_question_words(segments)

                # Post-processing: Merge numbered items with following segment
                segments = self._merge_numbered_items(segments)

                logger.debug(f"Text mit spaCy segmentiert: {len(segments)} Segmente")
                return segments
            except Exception as e:
                logger.warning(
                    f"spaCy Segmentierung fehlgeschlagen, nutze Regex-Fallback: {e}"
                )
                # Falle durch zu Fallback

        # Strategie 2: Regex-Fallback (wenn spaCy nicht verfügbar)
        return self._segment_text_regex(text)

    def _segment_text_regex(self, text: str) -> List[str]:
        """
        Fallback: Regex-basierte Segmentierung (wenn spaCy nicht verfügbar).

        WARNUNG: Diese Methode hat Schwächen bei:
        - Abkürzungen (Dr., Prof., etc.)
        - Dezimalzahlen (3.14)
        - Mehrfachen Satzzeichen (!!!, ???)

        Args:
            text: Der zu segmentierende Text

        Returns:
            Liste von Text-Segmenten
        """
        # Normalisiere Whitespace
        text = re.sub(r"\s+", " ", text.strip())

        # Strategie 1: Splitte an doppelten Zeilenumbrüchen (Absätze)
        paragraphs = re.split(r"\n\s*\n", text)

        segments = []
        for paragraph in paragraphs:
            # Strategie 2: Splitte an Satzgrenzen
            # WICHTIG: Behalte Interpunktion (?, !, .) für Klassifikation
            sentence_pattern = r"([^.!?]+[.!?]+)"
            sentences = re.findall(sentence_pattern, paragraph)

            # Fallback: Wenn keine Satzgrenzen gefunden, nimm gesamten Absatz
            if not sentences:
                sentences = [paragraph]

            # Bereinige und füge hinzu
            for sentence in sentences:
                cleaned = sentence.strip()
                if cleaned:
                    segments.append(cleaned)

        logger.debug(f"Text mit Regex segmentiert: {len(segments)} Segmente")
        return segments

    def classify_segment(self, text: str) -> InputSegment:
        """
        Klassifiziert ein Text-Segment.

        Nutzt mehrere Heuristiken:
        1. Fragezeichen → QUESTION
        1b. "Frage:" Prefix → QUESTION (common in puzzle formulations)
        2. Fragewörter am Anfang → QUESTION
        3. Explizite Befehle → COMMAND
        4. Deklarative Muster → EXPLANATION
        5. Fallback → EXPLANATION (konservativ)

        Args:
            text: Das zu klassifizierende Segment

        Returns:
            InputSegment mit Klassifikation
        """
        text_lower = text.lower().strip()

        # Heuristik 1: Fragezeichen → QUESTION
        if text_lower.endswith("?"):
            logger.debug(
                f"Segment klassifiziert als QUESTION (Fragezeichen): '{text[:50]}...'"
            )
            return InputSegment(
                text=text,
                segment_type=SegmentType.QUESTION,
                confidence=CONFIDENCE_QUESTION_MARK,
                metadata={"heuristic": "question_mark"},
            )

        # Heuristik 1b: "Frage:" prefix (German for "Question:")
        # Common in puzzle formulations where the question doesn't end with "?"
        if re.match(r"^frage:\s*", text_lower):
            logger.debug(
                f"Segment klassifiziert als QUESTION (Frage-Prefix): '{text[:50]}...'"
            )
            return InputSegment(
                text=text,
                segment_type=SegmentType.QUESTION,
                confidence=CONFIDENCE_FRAGE_PREFIX,
                metadata={"heuristic": "frage_prefix"},
            )

        # Heuristik 2: Fragewörter am Anfang → QUESTION
        first_word = text_lower.split()[0] if text_lower.split() else ""
        if first_word in self.question_words:
            logger.debug(
                f"Segment klassifiziert als QUESTION (Fragewort '{first_word}'): '{text[:50]}...'"
            )
            return InputSegment(
                text=text,
                segment_type=SegmentType.QUESTION,
                confidence=CONFIDENCE_QUESTION_WORD,
                metadata={"heuristic": "question_word", "question_word": first_word},
            )

        # Heuristik 3: Explizite Befehle → COMMAND
        for prefix in self.command_prefixes:
            if text_lower.startswith(prefix):
                logger.debug(
                    f"Segment klassifiziert als COMMAND ('{prefix}'): '{text[:50]}...'"
                )
                return InputSegment(
                    text=text,
                    segment_type=SegmentType.COMMAND,
                    confidence=CONFIDENCE_COMMAND,
                    metadata={"heuristic": "command_prefix", "command": prefix},
                )

        # Heuristik 4: Deklarative Muster → EXPLANATION
        # Prüfe auf typische deklarative Konstruktionen
        declarative_patterns = [
            r"\b(ist ein|ist eine|sind)\b",  # IS_A
            r"\b(kann|können)\b",  # CAPABLE_OF
            r"\b(hat|haben)\b",  # HAS_PROPERTY/PART_OF
            r"\b(liegt in|ist in)\b",  # LOCATED_IN
            r"\b(bedeutet|heißt|meint)\b",  # Definition
        ]

        has_declarative_pattern = any(
            re.search(pattern, text_lower) for pattern in declarative_patterns
        )

        if has_declarative_pattern:
            logger.debug(
                f"Segment klassifiziert als EXPLANATION (deklaratives Muster): '{text[:50]}...'"
            )
            return InputSegment(
                text=text,
                segment_type=SegmentType.EXPLANATION,
                confidence=CONFIDENCE_DECLARATIVE,
                metadata={"heuristic": "declarative_pattern"},
            )

        # Heuristik 5: Fallback → EXPLANATION (konservativ)
        # Wenn nichts anderes greift, gehe von Erklärung aus
        logger.debug(
            f"Segment klassifiziert als EXPLANATION (Fallback): '{text[:50]}...'"
        )
        return InputSegment(
            text=text,
            segment_type=SegmentType.EXPLANATION,
            confidence=CONFIDENCE_FALLBACK,
            metadata={"heuristic": "fallback"},
        )

    def _create_orchestrated_plan(self, segments: List[InputSegment]) -> MainGoal:
        """
        Erstellt einen orchestrierten Plan für segmentierte Eingaben.

        Strategie:
        1. Gruppiere zusammenhängende Erklärungen
        2. Verarbeite alle Erklärungen ZUERST (Lernen)
        3. Verarbeite dann Fragen (Reasoning mit gelerntem Wissen)

        Args:
            segments: Liste klassifizierter Segmente

        Returns:
            MainGoal mit orchestriertem Plan
        """
        # Trenne Erklärungen und Fragen
        explanations = [s for s in segments if s.is_explanation()]
        questions = [s for s in segments if s.is_question()]
        commands = [s for s in segments if s.is_command()]

        # Erstelle Haupt-Ziel
        plan = MainGoal(
            type=GoalType.PERFORM_TASK,
            description=f"Verarbeite {len(explanations)} Erklärung(en) und {len(questions)} Frage(n)",
        )

        # Sub-Goal 1: Verarbeite alle Erklärungen (Lernen)
        if explanations:
            # Kombiniere alle Erklärungen zu einem Text für Batch-Learning
            combined_explanations = ". ".join(e.text.rstrip(".") for e in explanations)

            plan.sub_goals.append(
                SubGoal(
                    description=f"Lerne Kontext: '{combined_explanations[:60]}...'",
                    metadata={
                        "orchestrated_type": "batch_learning",
                        "segment_texts": [e.text for e in explanations],
                        "segment_count": len(explanations),
                    },
                )
            )

        # Sub-Goal 2: Verarbeite alle Befehle
        if commands:
            for cmd in commands:
                plan.sub_goals.append(
                    SubGoal(
                        description=f"Führe Befehl aus: '{cmd.text[:60]}...'",
                        metadata={
                            "orchestrated_type": "command_execution",
                            "segment_text": cmd.text,
                        },
                    )
                )

        # Sub-Goal 3: Beantworte alle Fragen (mit gelerntem Kontext)
        if questions:
            for q in questions:
                plan.sub_goals.append(
                    SubGoal(
                        description=f"Beantworte Frage: '{q.text[:60]}...'",
                        metadata={
                            "orchestrated_type": "question_answering",
                            "segment_text": q.text,
                            "has_learned_context": len(explanations) > 0,
                        },
                    )
                )

        logger.info(
            f"Orchestrierter Plan erstellt: "
            f"{len(explanations)} Erklärungen, "
            f"{len(commands)} Befehle, "
            f"{len(questions)} Fragen"
        )

        return plan

    def get_segment_text(
        self, segment: InputSegment, strip_punctuation: bool = False
    ) -> str:
        """
        Holt den Text eines Segments.

        Args:
            segment: Das Segment
            strip_punctuation: Ob Interpunktion entfernt werden soll

        Returns:
            Text des Segments
        """
        text = segment.text
        if strip_punctuation:
            text = text.rstrip(".!?")
        return text.strip()
