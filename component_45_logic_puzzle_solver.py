"""
component_45_logic_puzzle_solver.py
====================================
Allgemeingültiger Logik-Rätsel-Löser mit Natürlichsprache-Parser.

Funktionen:
- Parst logische Bedingungen aus Natürlichsprache
- Übersetzt in aussagenlogische Formeln (CNF)
- Nutzt SAT-Solver für Lösungsfindung
- Generiert verständliche Antworten mit Proof Trees

Unterstützte logische Konstrukte:
- Implikation: "Wenn X, dann Y" → X → Y
- Konjunktion: "X und Y" → X ∧ Y
- Disjunktion: "X oder Y" → X ∨ Y
- Negation: "nicht X", "nie X" → ¬X
- Exklusiv-Oder: "X oder Y, aber nicht beide" → X ⊕ Y
- Bikonditional: "X genau dann wenn Y" → X ↔ Y

Author: KAI Development Team
Date: 2025-11-19
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

import spacy

from component_15_logging_config import get_logger
from component_30_sat_solver import Clause, CNFFormula, Literal, SATSolver

logger = get_logger(__name__)

# Lade spaCy-Modell für dynamische Objekt-Extraktion
nlp = spacy.load("de_core_news_sm")


@dataclass
class LogicVariable:
    """
    Variable für Logik-Rätsel (Person + Eigenschaft).

    Beispiel: "Leo trinkt Brandy" → LogicVariable("Leo", "trinkt_brandy")
    """

    entity: str  # z.B. "Leo", "Mark", "Nick"
    property: str  # z.B. "trinkt_brandy", "isst_pizza"

    def to_var_name(self) -> str:
        """Konvertiert zu Variable-Name für SAT-Solver."""
        return f"{self.entity}_{self.property}"

    def __str__(self):
        return f"{self.entity}:{self.property}"


@dataclass
class LogicCondition:
    """
    Eine geparste logische Bedingung.

    Typen:
    - IMPLICATION: "Wenn X, dann Y" → (X → Y)
    - CONJUNCTION: "X und Y" → (X ∧ Y)
    - DISJUNCTION: "X oder Y" → (X ∨ Y)
    - XOR: "X oder Y, aber nicht beide" → (X ⊕ Y)
    - NEGATION: "nicht X" → (¬X)
    - NEVER_BOTH: "nie X und Y" → ¬(X ∧ Y)
    """

    condition_type: str
    operands: List[str]  # Liste von Variable-Namen
    text: str  # Original-Text für Proof Tree


class LogicConditionParser:
    """
    Parst natürlichsprachliche logische Bedingungen.

    Generischer Parser, der NICHT rätselspezifisch ist!
    """

    def __init__(self):
        self.entities: Set[str] = set()
        self.properties: Set[str] = set()
        self.variables: Dict[str, LogicVariable] = {}
        self._context_object: Optional[str] = (
            None  # Haupt-Objekt aus dem Puzzle-Kontext
        )
        self._detected_objects: Set[str] = set()  # Dynamisch erkannte Objekte (Nomen)

    def _extract_objects_from_text(self, text: str) -> None:
        """
        Extrahiert dynamisch alle Objekte (Nomen) aus dem Text mit spaCy.

        Nutzt POS-Tagging, um Nomen zu erkennen, die NICHT in der Entity-Liste sind.
        Filtert aggressive, um nur relevante Objekte zu erkennen.

        Args:
            text: Vollständiger Text des Rätsels
        """
        doc = nlp(text)

        # Sammle potenzielle Objekte mit Kontext
        potential_objects = {}

        for i, token in enumerate(doc):
            # Finde Nomen (NOUN), die keine Entitäten sind
            if token.pos_ == "NOUN":
                lemma = token.lemma_.lower()

                # Ignoriere Entitäten, generische Wörter und Verben-als-Nomen
                # WICHTIG: Verben in Nomen-Form ausschließen (z.B. "das Essen", "das Trinken")
                verb_nouns = {
                    "essen",
                    "trinken",
                    "laufen",
                    "gehen",
                    "kommen",
                    "sein",
                    "haben",
                    "werden",
                }
                if (
                    lemma not in self.entities
                    and lemma
                    not in {
                        "person",
                        "leute",
                        "menschen",
                        "sache",
                        "ding",
                        "folgende",
                        "auto",
                    }
                    and lemma not in verb_nouns
                ):

                    # ERWEITERTE KONTEXT-PRÜFUNG
                    is_object = False

                    # Fall 1: Nach Artikel/Pronomen ("einen Brandy", "ein Hund")
                    if i > 0:
                        prev_token = doc[i - 1]
                        if prev_token.text.lower() in {
                            "einen",
                            "eine",
                            "ein",
                            "das",
                            "den",
                            "dem",
                            "der",
                            "die",
                        }:
                            is_object = True

                    # Fall 2: Nach Adjektiv ("rotes Auto" → prüfe 2 Tokens zurück)
                    if not is_object and i > 1:
                        prev_token = doc[i - 1]
                        prev_prev_token = doc[i - 2]
                        # "ein rotes Auto" → prev=ADJ, prev_prev=Artikel
                        if prev_token.pos_ == "ADJ":
                            if prev_prev_token.text.lower() in {
                                "ein",
                                "eine",
                                "einen",
                                "das",
                                "den",
                                "der",
                                "die",
                            }:
                                is_object = True

                    # Fall 3: Direkt nach Verb ("Kaffee bestellt", "Wasser trinkt")
                    if not is_object and i > 0:
                        prev_token = doc[i - 1]
                        if prev_token.pos_ == "VERB":
                            is_object = True

                    # Fall 4: Nomen gefolgt von Verb ("Kaffee bestellt" in umgekehrter Reihenfolge)
                    if not is_object and i < len(doc) - 1:
                        next_token = doc[i + 1]
                        if next_token.pos_ == "VERB":
                            is_object = True

                    if is_object:
                        # Zähle Häufigkeit
                        if lemma not in potential_objects:
                            potential_objects[lemma] = 0
                        potential_objects[lemma] += 1

        # Filtere: Nur Objekte, die mindestens 1x vorkommen
        # (Keine Häufigkeits-Filterung mehr, um alle Objekte zu erfassen)
        for obj, count in potential_objects.items():
            if count >= 1:
                self._detected_objects.add(obj)

        logger.info(f"Dynamisch erkannte Objekte: {self._detected_objects}")

    def parse_conditions(self, text: str, entities: List[str]) -> List[LogicCondition]:
        """
        Parst alle logischen Bedingungen aus einem Text.

        Args:
            text: Text mit Bedingungen (mehrere Sätze)
            entities: Liste bekannter Entitäten (z.B. ["Leo", "Mark", "Nick"])

        Returns:
            Liste von LogicCondition Objekten
        """
        self.entities = set(e.lower() for e in entities)
        conditions = []

        # SCHRITT 1: Extrahiere dynamisch alle Objekte aus dem Text
        self._extract_objects_from_text(text)

        # SCHRITT 2: Erkenne Haupt-Objekt (häufigstes/erstes Objekt)
        if self._detected_objects:
            # Nutze das erste erkannte Objekt als Kontext (kann später verfeinert werden)
            self._context_object = list(self._detected_objects)[0]
            logger.debug(f"Kontext-Objekt erkannt: {self._context_object}")

        # VERBESSERTE SEGMENTIERUNG:
        # Splitte bei:
        # - Satzende (.!?)
        # - Zeilenumbrüchen (wichtig für Listen-Format!)
        # - Aufzählungen mit Nummerierung (1., 2., etc.)
        # - Semantische Marker (Allerdings, Hingegen, Wenn ... dann als neue Zeile)

        # Schritt 1: Normalisiere Zeilenumbrüche zu künstlichen Breaks
        # WICHTIG: Zeilenumbrüche NICHT einfach entfernen, sondern als Trenner nutzen!
        text = text.replace("\r\n", "|").replace("\n", "|").replace("\r", "|")

        # Schritt 2: Füge künstliche Breaks bei semantischen Markern ein
        # WICHTIG: Auch bei fehlenden Punkten splitten!
        semantic_markers = [
            "Allerdings",
            "Hingegen",
            "Außerdem",
            "Ferner",
            "Weiterhin",
            "Es kann vorkommen",
            "Es geschieht",
            "Wenn",  # Neue Bedingung beginnt
        ]

        for marker in semantic_markers:
            # Füge | VOR dem Marker ein
            # Pattern 1: Nach Punkt/Fragezeichen/Ausrufezeichen
            text = re.sub(r"(?<=[.!?]\s)" + re.escape(marker), "|" + marker, text)
            # Pattern 2: Nach Doppelpunkt
            text = re.sub(r":\s*" + re.escape(marker), ":" + "|" + marker, text)
            # Pattern 3: OHNE Punkt (wichtig für zusammengeklebte Sätze!)
            # Ersetze "einen Es kann" mit "einen | Es kann"
            # Aber NUR wenn das Wort großgeschrieben ist (Satzanfang)
            text = re.sub(
                r"([a-z])\s+" + re.escape(marker) + r"\b", r"\1 |" + marker, text
            )

        # Schritt 3: Splitte bei Satzende UND bei künstlichen Breaks
        sentences = re.split(r"[.!?]|\|", text)

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence) < 10:  # Ignoriere zu kurze Fragmente
                continue

            # FILTER: Ignoriere Kontext-Sätze (keine echten logischen Bedingungen)
            # Heuristiken für Kontext-Sätze:
            # 1. Häufigkeits-Adverbien (oft, manchmal, gelegentlich)
            # 2. Negative Wissens-Aussagen (wissen nicht, nicht wissen)
            # 3. Sätze ohne logische Operatoren
            if (
                re.search(
                    r"\b(oft|manchmal|gelegentlich|normalerweise|üblicherweise)\b",
                    sentence,
                    re.IGNORECASE,
                )
                or re.search(
                    r"\b(wissen\s+nicht|nicht\s+wissen|weiß\s+nicht|nicht\s+weiß)\b",
                    sentence,
                    re.IGNORECASE,
                )
                or sentence.lower().startswith("allerdings wissen wir")
                or sentence.lower() == "allerdings wissen wir folgendes:"
            ):
                logger.debug(f"Kontext-Satz ignoriert: '{sentence}'")
                continue

            logger.debug(f"Parse Bedingung: '{sentence}'")

            # Versuche verschiedene Condition-Types zu matchen
            condition = (
                self._parse_implication(sentence)
                or self._parse_xor(sentence)
                or self._parse_never_both(sentence)
                or self._parse_conjunction(sentence)
                or self._parse_disjunction(sentence)
                or self._parse_negation(sentence)
            )

            if condition:
                conditions.append(condition)
                logger.info(f"[OK] Bedingung geparst: {condition.condition_type}")
            else:
                logger.debug(f"Keine logische Bedingung erkannt in: '{sentence}'")

        return conditions

    def _parse_implication(self, sentence: str) -> Optional[LogicCondition]:
        """
        Parst Implikationen: "Wenn X, dann Y" → (X → Y)

        Patterns:
        - "Wenn X, (dann) Y"
        - "Falls X, Y"
        """
        # Pattern: "Wenn X, (dann) Y"
        match = re.match(
            r"(?:wenn|falls)\s+(.+?),\s*(?:dann\s+)?(.+)", sentence, re.IGNORECASE
        )

        if match:
            antecedent_text = match.group(1).strip()
            consequent_text = match.group(2).strip()

            # Extrahiere Variablen aus beiden Teilen
            antecedent_vars = self._extract_variables(antecedent_text)
            consequent_vars = self._extract_variables(consequent_text)

            if antecedent_vars and consequent_vars:
                # Für Implikation: X → Y = ¬X ∨ Y (CNF)
                return LogicCondition(
                    condition_type="IMPLICATION",
                    operands=[antecedent_vars[0], consequent_vars[0]],
                    text=sentence,
                )

        return None

    def _parse_xor(self, sentence: str) -> Optional[LogicCondition]:
        """
        Parst Exklusiv-Oder: "X oder Y, aber nicht beide" → (X ⊕ Y)

        Patterns:
        - "X oder Y, aber nicht beide"
        - "X oder Y, aber nie beide"
        - "entweder X oder Y"
        - "dass X oder Y ... aber nie beide zusammen" (mit dass-Präfix)
        """
        # Pattern 1: "... dass X oder Y ... aber (nie/nicht) beide (zusammen)"
        # Beispiel: "Es kann vorkommen, dass Mark oder Nick ... aber nie beide zusammen"

        # Strategie: Entferne zuerst das Präfix, dann parse die eigentliche Bedingung
        cleaned_sentence = sentence

        # Entferne Präfixe wie "Es kann vorkommen, dass", "Es geschieht, dass", etc.
        prefix_pattern = r"^.+?(?:dass|wenn|falls)\s+"
        cleaned_sentence = re.sub(
            prefix_pattern, "", cleaned_sentence, flags=re.IGNORECASE
        )

        # Jetzt parse: "Mark oder Nick einen Brandy bestellen, aber nie beide zusammen"
        match = re.search(
            r"(.+?)\s+oder\s+(.+?),?\s*aber\s+(?:nie|nicht|niemals)\s+beide(?:\s+zusammen)?",
            cleaned_sentence,
            re.IGNORECASE,
        )

        if match:
            left_text = match.group(1).strip()
            right_text = match.group(2).strip()

            left_vars = self._extract_variables(left_text)
            right_vars = self._extract_variables(right_text)

            if left_vars and right_vars:
                logger.info(f"[OK] XOR erkannt: {left_vars[0]} ⊕ {right_vars[0]}")
                # XOR: (X ∨ Y) ∧ ¬(X ∧ Y)
                return LogicCondition(
                    condition_type="XOR",
                    operands=[left_vars[0], right_vars[0]],
                    text=sentence,
                )

        # Pattern 2: "entweder X oder Y"
        match = re.match(r"entweder\s+(.+?)\s+oder\s+(.+)", sentence, re.IGNORECASE)

        if match:
            left_text = match.group(1).strip()
            right_text = match.group(2).strip()

            left_vars = self._extract_variables(left_text)
            right_vars = self._extract_variables(right_text)

            if left_vars and right_vars:
                return LogicCondition(
                    condition_type="XOR",
                    operands=[left_vars[0], right_vars[0]],
                    text=sentence,
                )

        return None

    def _parse_never_both(self, sentence: str) -> Optional[LogicCondition]:
        """
        Parst "nie beide": "nie X und Y" → ¬(X ∧ Y)

        Patterns:
        - "nie X und Y"
        - "niemals X und Y zusammen"
        - "nicht X und Y gleichzeitig"
        """
        # Pattern: "nie(mals) X (und) Y (zusammen/gleichzeitig)"
        match = re.match(
            r"nie(?:mals)?\s+(.+?)\s+(?:und\s+)?(.+?)(?:\s+(?:zusammen|gleichzeitig))?",
            sentence,
            re.IGNORECASE,
        )

        if match:
            left_text = match.group(1).strip()
            right_text = match.group(2).strip()

            left_vars = self._extract_variables(left_text)
            right_vars = self._extract_variables(right_text)

            if left_vars and right_vars:
                return LogicCondition(
                    condition_type="NEVER_BOTH",
                    operands=[left_vars[0], right_vars[0]],
                    text=sentence,
                )

        return None

    def _parse_conjunction(self, sentence: str) -> Optional[LogicCondition]:
        """
        Parst Konjunktion: "X und Y" → (X ∧ Y)

        ABER: "X und Y einzeln oder gleichzeitig" → (X ∨ Y) [DISJUNCTION!]
        """
        # SPEZIALFALL: "X und Y einzeln oder gleichzeitig/zusammen"
        # Bedeutung: "individually or simultaneously" = at least one = DISJUNCTION
        if re.search(
            r"einzeln\s+oder\s+(gleichzeitig|zusammen)", sentence, re.IGNORECASE
        ):
            # Parse als DISJUNCTION statt CONJUNCTION
            match = re.search(
                r"(.+?)\s+und\s+(.+?)\s+einzeln\s+oder", sentence, re.IGNORECASE
            )
            if match:
                left_text = match.group(1).strip()
                right_text = match.group(2).strip()

                left_vars = self._extract_variables(left_text)
                right_vars = self._extract_variables(right_text)

                if left_vars and right_vars:
                    logger.info(
                        f"[OK] DISJUNCTION (einzeln/gleichzeitig): {left_vars[0]} ∨ {right_vars[0]}"
                    )
                    return LogicCondition(
                        condition_type="DISJUNCTION",
                        operands=[left_vars[0], right_vars[0]],
                        text=sentence,
                    )

        # Standard-Pattern: "X und Y"
        match = re.match(r"(.+?)\s+und\s+(.+)", sentence, re.IGNORECASE)

        if match:
            left_text = match.group(1).strip()
            right_text = match.group(2).strip()

            left_vars = self._extract_variables(left_text)
            right_vars = self._extract_variables(right_text)

            if left_vars and right_vars:
                return LogicCondition(
                    condition_type="CONJUNCTION",
                    operands=[left_vars[0], right_vars[0]],
                    text=sentence,
                )

        return None

    def _parse_disjunction(self, sentence: str) -> Optional[LogicCondition]:
        """Parst Disjunktion: "X oder Y" → (X ∨ Y)"""
        # Pattern: "X oder Y"
        match = re.match(r"(.+?)\s+oder\s+(.+)", sentence, re.IGNORECASE)

        if match:
            left_text = match.group(1).strip()
            right_text = match.group(2).strip()

            left_vars = self._extract_variables(left_text)
            right_vars = self._extract_variables(right_text)

            if left_vars and right_vars:
                return LogicCondition(
                    condition_type="DISJUNCTION",
                    operands=[left_vars[0], right_vars[0]],
                    text=sentence,
                )

        return None

    def _parse_negation(self, sentence: str) -> Optional[LogicCondition]:
        """Parst Negation: "nicht X" → (¬X)"""
        # Pattern: "nicht X"
        match = re.match(r"nicht\s+(.+)", sentence, re.IGNORECASE)

        if match:
            text = match.group(1).strip()
            vars = self._extract_variables(text)

            if vars:
                return LogicCondition(
                    condition_type="NEGATION", operands=[vars[0]], text=sentence
                )

        return None

    def _extract_variables(self, text: str) -> List[str]:
        """
        Extrahiert Variablen aus einem Text-Fragment.

        Sucht nach Mustern wie:
        - "Leo einen Brandy bestellt" → Variable("Leo", "bestellt_brandy")
        - "Mark trinkt" → Variable("Mark", "trinkt_brandy")

        Returns:
            Liste von Variable-Namen (z.B. ["Leo_bestellt_brandy"])
        """
        variables = []

        # WICHTIG: Entferne Präfix-Text wie "Es kann vorkommen, dass"
        # Damit "dass Mark oder Nick..." → "Mark oder Nick..."
        text = re.sub(
            r"^.+?(?:dass|wenn|falls)\s+", "", text, flags=re.IGNORECASE
        ).strip()

        # Finde Entität (Person) im Text
        for entity in self.entities:
            if entity in text.lower():
                # Extrahiere Eigenschaft (Verb + Objekt)
                # Pattern: "[Person] [Verb] [Objekt]"
                # Beispiel: "Leo einen Brandy bestellt" → "bestellt_brandy"

                # STRATEGIE: Suche nach Verb + Hauptobjekt (Nomen)
                # Verben: bestellt, trinkt, mag, isst, will
                # Objekte: Brandy, Bier, Pizza, etc.

                property_name = self._extract_action_object(text.lower(), entity)

                # SPEZIALFALL: Nur Entitäts-Name ohne Aktion
                # Beispiel: "Mark" (aus "Mark oder Nick einen Brandy bestellen")
                # → Nutze Kontext-Objekt
                if (
                    not property_name
                    and hasattr(self, "_context_object")
                    and self._context_object
                ):
                    # Standard-Aktion mit Kontext-Objekt (einheitlich: "hat")
                    property_name = f"hat_{self._context_object}"
                    logger.debug(
                        f"Entity ohne Aktion, nutze Kontext: {entity} → {property_name}"
                    )

                if property_name:
                    var = LogicVariable(entity.capitalize(), property_name)
                    var_name = var.to_var_name()
                    self.variables[var_name] = var
                    variables.append(var_name)

                    logger.debug(f"Extrahierte Variable: {var}")

                break  # Nur eine Entität pro Fragment

        return variables

    def _extract_action_object(self, text: str, entity: str) -> str:
        """
        Extrahiert Aktion + Objekt aus Text (z.B. "bestellt_brandy").

        Strategie:
        1. Finde Verb (bestellt, trinkt, mag, will, etc.)
        2. Finde Objekt (Brandy, Bier, Pizza, etc.)
        3. Kombiniere zu "verb_objekt"

        Args:
            text: Text-Fragment (lowercase)
            entity: Entität (Person), die bereits gefunden wurde

        Returns:
            Property-Name (z.B. "bestellt_brandy")
        """
        # Verb-Normalisierungs-Map: Konjugierte Form -> Kanonische Form
        verb_normalization = {
            "bestellt": "hat",
            "bestellen": "hat",
            "bestellst": "hat",
            "trinkt": "hat",
            "trinken": "hat",
            "trinkst": "hat",
            "mag": "hat",
            "mögen": "hat",
            "magst": "hat",
            "isst": "hat",
            "essen": "hat",
            "kauft": "hat",
            "kaufen": "hat",
            "nimmt": "hat",
            "nehmen": "hat",
            "hat": "hat",
            "haben": "hat",
            "habe": "hat",
            "hast": "hat",
            "besitzt": "hat",
            "besitzen": "hat",
            "will": "hat",  # Modalverb: "will X" = "hat X"
            "wollen": "hat",
            "möchte": "hat",
            "möchten": "hat",
        }

        # Finde Verb im Text
        found_verb = None
        canonical_verb = None
        for verb in verb_normalization.keys():
            if verb in text:
                found_verb = verb
                canonical_verb = verb_normalization[verb]
                break

        if not found_verb:
            # Fallback: Nutze normalisierten ganzen Text
            return self._normalize_property(text.replace(entity, "").strip())

        # Suche das Objekt im GESAMTEN Text (vor oder nach dem Verb)
        # Deutsches SOV: "einen Brandy bestellt" (Objekt VOR Verb)
        # Deutsches SVO: "bestellt einen Brandy" (Objekt NACH Verb)

        # DYNAMISCH: Nutze die erkannten Objekte statt hardcodierter Liste
        obj = None
        for detected_obj in self._detected_objects:
            if detected_obj in text:
                obj = detected_obj
                break

        if obj:
            # Alle Verben werden auf "hat" normalisiert → einheitliche Variable-Namen
            return f"hat_{obj}"
        else:
            # Kein spezifisches Objekt gefunden
            # SPEZIALFALL: "einen/eine/ein/eins" als implizite Referenz auf Kontext-Objekt
            # "bestellt auch Mark einen" → "einen" = implizite Referenz auf "brandy"
            # "will auch einen" → "einen" = implizite Referenz auf "brandy"
            # "kauft auch Hanna eins" → "eins" = implizite Referenz auf "fahrrad"
            if re.search(r"\b(einen|eine|ein|eins)\b", text):
                if hasattr(self, "_context_object") and self._context_object:
                    # Alle Verben → "hat_[kontext_objekt]" (einheitlich)
                    return f"hat_{self._context_object}"
                else:
                    # Kein Kontext-Objekt vorhanden, nutze "unbekannt"
                    return f"hat_unbekannt"

            return canonical_verb if canonical_verb else "hat"

    def _normalize_property(self, text: str) -> str:
        """
        Normalisiert Eigenschafts-Text zu einem Property-Namen.

        Beispiel:
        - "einen brandy bestellt" → "bestellt_brandy"
        - "trinkt gerne bier" → "trinkt_bier"
        """
        # Entferne Füllwörter
        text = re.sub(
            r"\b(einen?|eine?|der|die|das|den|dem|gerne?|auch|will)\b",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = text.strip()

        # Normalisiere zu snake_case
        text = re.sub(r"\s+", "_", text)
        text = re.sub(r"[^\w_]", "", text)

        return text

    def get_variable(self, var_name: str) -> Optional[LogicVariable]:
        """Holt LogicVariable für einen Variable-Namen."""
        return self.variables.get(var_name)


class LogicPuzzleSolver:
    """
    Löst Logik-Rätsel mit SAT-Solver.

    Workflow:
    1. Parse Bedingungen → LogicCondition Liste
    2. Konvertiere zu CNF → CNFFormula
    3. Löse mit SAT-Solver → Model
    4. Formatiere Antwort mit Proof Tree
    """

    def __init__(self):
        self.parser = LogicConditionParser()
        self.solver = SATSolver()

    def solve(
        self, conditions_text: str, entities: List[str], question: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Löst ein Logik-Rätsel.

        Args:
            conditions_text: Text mit logischen Bedingungen
            entities: Liste der Entitäten (z.B. ["Leo", "Mark", "Nick"])
            question: Optionale Frage (z.B. "Wer trinkt Brandy?")

        Returns:
            Dictionary mit:
            - solution: Dict[var_name, bool] - Variable-Assignments
            - proof_tree: ProofTree - Lösungsweg
            - answer: str - Formatierte Antwort
        """
        logger.info(f"Löse Logik-Rätsel mit {len(entities)} Entitäten")

        # SCHRITT 1: Parse Bedingungen
        conditions = self.parser.parse_conditions(conditions_text, entities)
        logger.info(f"Geparst: {len(conditions)} Bedingungen")

        if not conditions:
            return {
                "solution": {},
                "proof_tree": None,
                "answer": "Keine logischen Bedingungen gefunden.",
            }

        # SCHRITT 2: Konvertiere zu CNF
        cnf = self._build_cnf(conditions)
        logger.info(f"CNF erstellt: {len(cnf.clauses)} Klauseln")

        # SCHRITT 3: Löse mit SAT-Solver
        # SATSolver.solve() gibt None zurück wenn UNSAT, sonst model
        model = self.solver.solve(cnf)

        if model is not None:
            logger.info(f"[OK] Lösung gefunden: {model}")

            # SCHRITT 4: Formatiere Antwort
            answer = self._format_answer(model, question)

            return {
                "solution": model,
                "proof_tree": None,  # TODO: Extract from solver if needed
                "answer": answer,
                "result": "SATISFIABLE",
            }
        else:
            logger.warning("Rätsel ist unlösbar (UNSAT)")
            return {
                "solution": {},
                "proof_tree": None,
                "answer": "Das Rätsel hat keine Lösung (Widerspruch in den Bedingungen).",
                "result": "UNSATISFIABLE",
            }

    def _build_cnf(self, conditions: List[LogicCondition]) -> CNFFormula:
        """
        Konvertiert LogicCondition Liste zu CNF-Formel.

        Logik-Umwandlungen:
        - IMPLICATION (X → Y):  ¬X ∨ Y
        - XOR (X ⊕ Y):          (X ∨ Y) ∧ (¬X ∨ ¬Y)
        - NEVER_BOTH ¬(X ∧ Y):  ¬X ∨ ¬Y
        - CONJUNCTION (X ∧ Y):  X, Y (separate Klauseln)
        - DISJUNCTION (X ∨ Y):  X ∨ Y
        - NEGATION (¬X):        ¬X
        """
        clauses: List[Clause] = []

        for cond in conditions:
            if cond.condition_type == "IMPLICATION":
                # X → Y = ¬X ∨ Y
                x, y = cond.operands
                clauses.append(
                    Clause({Literal(x, negated=True), Literal(y, negated=False)})
                )

            elif cond.condition_type == "XOR":
                # X ⊕ Y = (X ∨ Y) ∧ (¬X ∨ ¬Y)
                x, y = cond.operands
                clauses.append(Clause({Literal(x), Literal(y)}))  # X ∨ Y
                clauses.append(
                    Clause({Literal(x, negated=True), Literal(y, negated=True)})
                )  # ¬X ∨ ¬Y

            elif cond.condition_type == "NEVER_BOTH":
                # ¬(X ∧ Y) = ¬X ∨ ¬Y
                x, y = cond.operands
                clauses.append(
                    Clause({Literal(x, negated=True), Literal(y, negated=True)})
                )

            elif cond.condition_type == "CONJUNCTION":
                # X ∧ Y = X, Y (zwei separate Klauseln)
                x, y = cond.operands
                clauses.append(Clause({Literal(x)}))
                clauses.append(Clause({Literal(y)}))

            elif cond.condition_type == "DISJUNCTION":
                # X ∨ Y
                x, y = cond.operands
                clauses.append(Clause({Literal(x), Literal(y)}))

            elif cond.condition_type == "NEGATION":
                # ¬X
                x = cond.operands[0]
                clauses.append(Clause({Literal(x, negated=True)}))

        return CNFFormula(clauses)

    def _format_answer(self, model: Dict[str, bool], question: Optional[str]) -> str:
        """
        Formatiert die Lösung als natürlichsprachliche Antwort.

        Args:
            model: Variable-Assignments (var_name → bool)
            question: Optionale Frage (für kontextuelle Antwort)

        Returns:
            Formatierte Antwort
        """
        # Finde alle TRUE Variablen
        true_vars = [var for var, value in model.items() if value]

        if not true_vars:
            return "Keine der Bedingungen ist erfüllt."

        # Formatiere Variablen als Sätze
        # Beispiel: "Leo_bestellt_brandy" → "Leo bestellt Brandy"
        statements = []
        for var_name in true_vars:
            var = self.parser.get_variable(var_name)
            if var:
                # Konvertiere Property zurück zu Natürlichsprache
                property_text = var.property.replace("_", " ")
                statements.append(f"{var.entity} {property_text}")

        if len(statements) == 1:
            return statements[0].capitalize()
        else:
            return ", ".join(statements[:-1]) + " und " + statements[-1]
