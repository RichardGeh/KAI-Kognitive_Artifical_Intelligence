"""
component_45_logic_puzzle_parser.py
===================================
Natural language parser for logic puzzles with spaCy integration.

This module handles:
- Parsing natural language conditions into structured logic
- Variable extraction (entity + property)
- Pattern matching for logical constructs (implication, XOR, negation, etc.)
- Dynamic object detection using spaCy POS tagging

Supported logical constructs:
- Implication: "Wenn X, dann Y" -> X -> Y
- Conjunction: "X und Y" -> X AND Y
- Disjunction: "X oder Y" -> X OR Y
- Negation: "nicht X", "nie X" -> NOT X
- Exclusive-OR: "X oder Y, aber nicht beide" -> X XOR Y
- Biconditional: "X genau dann wenn Y" -> X <-> Y

Author: KAI Development Team
Date: 2025-11-29 (Split from component_45)
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Set

import spacy

from component_15_logging_config import get_logger
from kai_exceptions import ParsingError, SpaCyModelError

logger = get_logger(__name__)

# Lazy loading for spaCy model (only loaded when needed)
_nlp_model = None


def _get_nlp_model():
    """
    Lazy loading for spaCy model with error handling.

    Returns:
        spaCy Language model

    Raises:
        SpaCyModelError: If model cannot be loaded
    """
    global _nlp_model
    if _nlp_model is None:
        try:
            _nlp_model = spacy.load("de_core_news_sm")
            logger.info("spaCy model 'de_core_news_sm' successfully loaded")
        except OSError as e:
            raise SpaCyModelError(
                "spaCy model 'de_core_news_sm' could not be loaded. "
                "Please install with: python -m spacy download de_core_news_sm",
                context={"model_name": "de_core_news_sm"},
                original_exception=e,
            )
        except Exception as e:
            raise SpaCyModelError(
                "Unexpected error loading spaCy model",
                context={"model_name": "de_core_news_sm"},
                original_exception=e,
            )
    return _nlp_model


@dataclass
class LogicVariable:
    """
    Variable for logic puzzles (entity + property).

    Example: "Leo trinkt Brandy" -> LogicVariable("Leo", "trinkt_brandy")
    """

    entity: str  # e.g. "Leo", "Mark", "Nick"
    property: str  # e.g. "trinkt_brandy", "isst_pizza"

    def to_var_name(self) -> str:
        """Converts to variable name for SAT solver (normalized to lowercase)."""
        return f"{self.entity.lower()}_{self.property}"

    def __str__(self):
        return f"{self.entity}:{self.property}"


@dataclass
class LogicCondition:
    """
    A parsed logical condition.

    Types:
    - IMPLICATION: "Wenn X, dann Y" -> (X -> Y)
    - CONJUNCTION: "X und Y" -> (X AND Y)
    - DISJUNCTION: "X oder Y" -> (X OR Y)
    - XOR: "X oder Y, aber nicht beide" -> (X XOR Y)
    - NEGATION: "nicht X" -> (NOT X)
    - NEVER_BOTH: "nie X und Y" -> NOT (X AND Y)
    """

    condition_type: str
    operands: List[str]  # List of variable names
    text: str  # Original text for proof tree


class LogicConditionParser:
    """
    Parses natural language logical conditions.

    Generic parser that is NOT puzzle-specific!
    """

    def __init__(self):
        self.entities: Set[str] = set()
        self.properties: Set[str] = set()
        self.variables: dict[str, LogicVariable] = {}
        self._context_object: Optional[str] = None  # Main object from puzzle context
        self._detected_objects: Set[str] = set()  # Dynamically detected objects (nouns)

    def _extract_objects_from_text(self, text: str) -> None:
        """
        Dynamically extracts objects (nouns) from text using spaCy POS tagging.

        NEW: Detects colon-enumerated objects (e.g., "Berufe: Lehrer, Arzt, Ingenieur")

        Args:
            text: Full puzzle text

        Raises:
            ParsingError: If spaCy processing fails
        """
        try:
            nlp = _get_nlp_model()
            doc = nlp(text)
        except SpaCyModelError:
            raise
        except Exception as e:
            raise ParsingError(
                "Error during spaCy text processing",
                context={"text_length": len(text)},
                original_exception=e,
            )

        potential_objects = {}
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
        articles = {"einen", "eine", "ein", "das", "den", "dem", "der", "die"}
        generic = {
            "person",
            "leute",
            "menschen",
            "sache",
            "ding",
            "folgende",
            "auto",
            "beruf",
            "berufe",
        }

        # ===== NEW: PATTERN 1 - Colon-separated enumerations =====
        # Pattern: "category_noun: Object1, Object2 und Object3"
        # Example: "Berufe: Lehrer, Arzt und Ingenieur"
        colon_pattern = r"(\w+):\s*([A-Z]\w+(?:,\s*[A-Z]\w+)*(?:\s+und\s+[A-Z]\w+)?)"
        for match in re.finditer(colon_pattern, text):
            category = match.group(1).lower()
            object_list = match.group(2)

            # Extract individual objects from enumeration
            # Split by comma and "und"
            objects = re.split(r",\s*|\s+und\s+", object_list)
            for obj in objects:
                obj_clean = obj.strip()
                if obj_clean:
                    obj_lemma = obj_clean.lower()
                    # Don't filter by entities here - these are clearly objects!
                    if obj_lemma not in verb_nouns and obj_lemma not in generic:
                        potential_objects[obj_lemma] = (
                            potential_objects.get(obj_lemma, 0) + 3
                        )  # High confidence
                        logger.debug(
                            f"Colon enumeration detected: '{category}' -> '{obj_lemma}'"
                        )
        # =========================================================

        for i, token in enumerate(doc):
            if token.pos_ == "NOUN":
                lemma = token.lemma_.lower()
                # MODIFIED: Only filter out true generic nouns (not entities - they might be objects!)
                if lemma in verb_nouns or lemma in generic:
                    continue

                is_object = False
                # After article
                if i > 0 and doc[i - 1].text.lower() in articles:
                    is_object = True
                # After adjective + article
                elif (
                    i > 1
                    and doc[i - 1].pos_ == "ADJ"
                    and doc[i - 2].text.lower() in articles
                ):
                    is_object = True
                # After/before verb
                elif i > 0 and doc[i - 1].pos_ == "VERB":
                    is_object = True
                elif i < len(doc) - 1 and doc[i + 1].pos_ == "VERB":
                    is_object = True

                if is_object:
                    potential_objects[lemma] = potential_objects.get(lemma, 0) + 1

        self._detected_objects = {
            obj for obj, count in potential_objects.items() if count >= 1
        }
        logger.info(f"Dynamically detected objects: {self._detected_objects}")

    def parse_conditions(self, text: str, entities: List[str]) -> List[LogicCondition]:
        """
        Parses all logical conditions from text.

        Args:
            text: Text with conditions (multiple sentences)
            entities: List of known entities (e.g., ["Leo", "Mark", "Nick"])

        Returns:
            List of LogicCondition objects

        Raises:
            ParsingError: If text parsing fails
            SpaCyModelError: If spaCy model is unavailable
        """
        try:
            self.entities = set(e.lower() for e in entities)
            conditions = []

            # STEP 1: Dynamically extract all objects from text
            try:
                self._extract_objects_from_text(text)
            except (ParsingError, SpaCyModelError):
                raise  # Re-raise known errors
            except Exception as e:
                raise ParsingError(
                    "Error extracting objects from text",
                    context={"text_length": len(text)},
                    original_exception=e,
                )

            # STEP 2: Detect main object (context)
            if self._detected_objects:
                self._context_object = list(self._detected_objects)[0]
                logger.debug(f"Context object detected: {self._context_object}")

            # STEP 3: Improved segmentation at sentence/line breaks and semantic markers
            text = text.replace("\r\n", "|").replace("\n", "|").replace("\r", "|")

            semantic_markers = [
                "Allerdings",
                "Hingegen",
                "Außerdem",
                "Ferner",
                "Weiterhin",
                "Es kann vorkommen",
                "Es geschieht",
                "Wenn",
            ]

            for marker in semantic_markers:
                text = re.sub(r"(?<=[.!?]\s)" + re.escape(marker), "|" + marker, text)
                text = re.sub(r":\s*" + re.escape(marker), ":" + "|" + marker, text)
                text = re.sub(
                    r"([a-z])\s+" + re.escape(marker) + r"\b", r"\1 |" + marker, text
                )

            sentences = re.split(r"[.!?]|\|", text)

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence or len(sentence) < 10:
                    continue

                # Filter context sentences (frequency adverbs, knowledge negations)
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
                ):
                    logger.debug(f"Context sentence ignored: '{sentence}'")
                    continue

                # NEW: Filter entity/object enumeration sentences (introductions)
                # Pattern: "X, Y und Z haben unterschiedliche A: B, C und D"
                # These don't contain actual constraints, just list entities and objects
                if re.search(
                    r"haben\s+unterschiedliche\s+\w+:\s*[A-Z]", sentence, re.IGNORECASE
                ):
                    logger.debug(
                        f"Entity/object enumeration sentence ignored: '{sentence}'"
                    )
                    continue

                logger.debug(f"Parse condition: '{sentence}'")

                condition = (
                    self._parse_implication(sentence)
                    or self._parse_xor(sentence)
                    or self._parse_never_both(sentence)
                    or self._parse_conjunction(sentence)
                    or self._parse_disjunction(sentence)
                    or self._parse_negation(sentence)
                    or self._parse_simple_fact(sentence)
                )

                if condition:
                    conditions.append(condition)
                    logger.info(f"[OK] Condition parsed: {condition.condition_type}")
                else:
                    logger.debug(f"No logical condition recognized in: '{sentence}'")

            # STEP 4: Detect assignment puzzle and add uniqueness constraints
            if self._is_assignment_puzzle(text, entities):
                logger.info(
                    "Assignment puzzle detected - adding uniqueness constraints"
                )
                uniqueness_conditions = self._generate_uniqueness_constraints(entities)
                conditions.extend(uniqueness_conditions)
                logger.info(
                    f"Added {len(uniqueness_conditions)} uniqueness constraints"
                )

            return conditions

        except (ParsingError, SpaCyModelError):
            raise  # Re-raise known errors
        except Exception as e:
            raise ParsingError(
                "Error parsing conditions",
                context={"text_length": len(text), "num_entities": len(entities)},
                original_exception=e,
            )

    def _parse_implication(self, sentence: str) -> Optional[LogicCondition]:
        """
        Parses implications: "Wenn X, dann Y" -> (X -> Y)

        Patterns:
        - "Wenn X, (dann) Y"
        - "Falls X, Y"

        Raises:
            ParsingError: If variable extraction fails
        """
        try:
            # Pattern: "Wenn X, (dann) Y"
            match = re.match(
                r"(?:wenn|falls)\s+(.+?),\s*(?:dann\s+)?(.+)", sentence, re.IGNORECASE
            )

            if match:
                antecedent_text = match.group(1).strip()
                consequent_text = match.group(2).strip()

                # Extract variables from both parts
                antecedent_vars = self._extract_variables(antecedent_text)
                consequent_vars = self._extract_variables(consequent_text)

                if antecedent_vars and consequent_vars:
                    # For implication: X -> Y = NOT X OR Y (CNF)
                    return LogicCondition(
                        condition_type="IMPLICATION",
                        operands=[antecedent_vars[0], consequent_vars[0]],
                        text=sentence,
                    )

            return None
        except Exception as e:
            raise ParsingError(
                "Error parsing implication",
                context={"sentence": sentence},
                original_exception=e,
            )

    def _parse_xor(self, sentence: str) -> Optional[LogicCondition]:
        """
        Parses exclusive-OR: "X oder Y, aber nicht beide" -> (X XOR Y)

        Patterns: "aber (nie|nicht) beide", "entweder X oder Y"

        Raises:
            ParsingError: If variable extraction fails
        """
        try:
            cleaned = re.sub(
                r"^.+?(?:dass|wenn|falls)\s+", "", sentence, flags=re.IGNORECASE
            )

            # Pattern 1: "X oder Y, aber (nie|nicht|niemals) beide (zusammen)"
            match = re.search(
                r"(.+?)\s+oder\s+(.+?),?\s*aber\s+(?:nie|nicht|niemals)\s+beide(?:\s+zusammen)?",
                cleaned,
                re.IGNORECASE,
            )
            if not match:
                # Pattern 2: "entweder X oder Y"
                match = re.match(
                    r"entweder\s+(.+?)\s+oder\s+(.+)", sentence, re.IGNORECASE
                )

            if match:
                left_vars = self._extract_variables(match.group(1).strip())
                right_vars = self._extract_variables(match.group(2).strip())

                if left_vars and right_vars:
                    logger.info(
                        f"[OK] XOR detected: {left_vars[0]} XOR {right_vars[0]}"
                    )
                    return LogicCondition(
                        condition_type="XOR",
                        operands=[left_vars[0], right_vars[0]],
                        text=sentence,
                    )

            return None
        except Exception as e:
            raise ParsingError(
                "Error parsing XOR condition",
                context={"sentence": sentence},
                original_exception=e,
            )

    def _parse_never_both(self, sentence: str) -> Optional[LogicCondition]:
        """
        Parses "nie beide": "nie X und Y" -> NOT (X AND Y)

        Patterns:
        - "nie X und Y"
        - "niemals X und Y zusammen"
        - "nicht X und Y gleichzeitig"

        Raises:
            ParsingError: If variable extraction fails
        """
        try:
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
        except Exception as e:
            raise ParsingError(
                "Error parsing NEVER_BOTH condition",
                context={"sentence": sentence},
                original_exception=e,
            )

    def _parse_conjunction(self, sentence: str) -> Optional[LogicCondition]:
        """
        Parses conjunction: "X und Y" -> (X AND Y)
        Special case: "einzeln oder gleichzeitig" -> DISJUNCTION
        """
        try:
            # Special: "X und Y einzeln oder gleichzeitig" = DISJUNCTION
            if re.search(
                r"einzeln\s+oder\s+(gleichzeitig|zusammen)", sentence, re.IGNORECASE
            ):
                match = re.search(
                    r"(.+?)\s+und\s+(.+?)\s+einzeln\s+oder", sentence, re.IGNORECASE
                )
                if match:
                    left_vars = self._extract_variables(match.group(1).strip())
                    right_vars = self._extract_variables(match.group(2).strip())
                    if left_vars and right_vars:
                        logger.info(
                            f"[OK] DISJUNCTION (einzeln/gleichzeitig): {left_vars[0]} OR {right_vars[0]}"
                        )
                        return LogicCondition(
                            condition_type="DISJUNCTION",
                            operands=[left_vars[0], right_vars[0]],
                            text=sentence,
                        )

            match = re.match(r"(.+?)\s+und\s+(.+)", sentence, re.IGNORECASE)
            if match:
                left_vars = self._extract_variables(match.group(1).strip())
                right_vars = self._extract_variables(match.group(2).strip())
                if left_vars and right_vars:
                    return LogicCondition(
                        condition_type="CONJUNCTION",
                        operands=[left_vars[0], right_vars[0]],
                        text=sentence,
                    )
            return None
        except Exception as e:
            raise ParsingError(
                "Error parsing conjunction",
                context={"sentence": sentence},
                original_exception=e,
            )

    def _parse_disjunction(self, sentence: str) -> Optional[LogicCondition]:
        """Parses disjunction: "X oder Y" -> (X OR Y)"""
        try:
            match = re.match(r"(.+?)\s+oder\s+(.+)", sentence, re.IGNORECASE)
            if match:
                left_vars = self._extract_variables(match.group(1).strip())
                right_vars = self._extract_variables(match.group(2).strip())
                if left_vars and right_vars:
                    return LogicCondition(
                        condition_type="DISJUNCTION",
                        operands=[left_vars[0], right_vars[0]],
                        text=sentence,
                    )
            return None
        except Exception as e:
            raise ParsingError(
                "Error parsing disjunction",
                context={"sentence": sentence},
                original_exception=e,
            )

    def _parse_negation(self, sentence: str) -> Optional[LogicCondition]:
        """
        Parses negation with multiple German patterns.

        Supports:
        - "nicht X" (simple negation)
        - "X ist nicht Y" (copula negation)
        - "X ist kein/keine/keinen Y" (copula + negated article)
        - "X hat kein/keine Y" (possession negation)
        - "X weder Y noch Z" (neither...nor)
        """
        try:
            # Pattern 0: "weder...noch" (neither...nor) -> CONJUNCTION of two negations
            # Example: "Clara traegt weder Gruen noch Gelb" -> NOT(Clara_gruen) AND NOT(Clara_gelb)
            match = re.match(
                r"(.+?)\s+weder\s+(.+?)\s+noch\s+(.+)", sentence, re.IGNORECASE
            )
            if match:
                entity_text = match.group(1).strip()
                object1_text = match.group(2).strip()
                object2_text = match.group(3).strip()

                # Extract variables for both negations
                var1 = self._extract_variables(f"{entity_text} {object1_text}")
                var2 = self._extract_variables(f"{entity_text} {object2_text}")

                if var1 and var2:
                    logger.info(
                        f"[OK] WEDER...NOCH detected: NOT({var1[0]}) AND NOT({var2[0]})"
                    )
                    # Return CONJUNCTION of two NEGATION conditions
                    return LogicCondition(
                        condition_type="CONJUNCTION",
                        operands=[
                            LogicCondition(
                                condition_type="NEGATION",
                                operands=[var1[0]],
                                text=f"{entity_text} nicht {object1_text}",
                            ),
                            LogicCondition(
                                condition_type="NEGATION",
                                operands=[var2[0]],
                                text=f"{entity_text} nicht {object2_text}",
                            ),
                        ],
                        text=sentence,
                    )

            # Pattern 1: Simple negation starting with "nicht"
            match = re.match(r"nicht\s+(.+)", sentence, re.IGNORECASE)
            if match:
                vars = self._extract_variables(match.group(1).strip())
                if vars:
                    return LogicCondition(
                        condition_type="NEGATION", operands=[vars[0]], text=sentence
                    )

            # Pattern 2: "X ist nicht Y" (copula negation)
            match = re.match(r"(.+?)\s+ist\s+nicht\s+(.+)", sentence, re.IGNORECASE)
            if match:
                entity_text = match.group(1).strip()
                object_text = match.group(2).strip()
                # Construct "entity object" to extract variable
                combined = f"{entity_text} {object_text}"
                vars = self._extract_variables(combined)
                if vars:
                    return LogicCondition(
                        condition_type="NEGATION", operands=[vars[0]], text=sentence
                    )

            # Pattern 3: "X ist kein/keine/keinen Y" (copula + negated article)
            match = re.match(
                r"(.+?)\s+ist\s+kein(?:e[nrs]?)?\s+(.+)", sentence, re.IGNORECASE
            )
            if match:
                entity_text = match.group(1).strip()
                object_text = match.group(2).strip()
                # Construct "entity object" to extract variable
                combined = f"{entity_text} {object_text}"
                vars = self._extract_variables(combined)
                if vars:
                    return LogicCondition(
                        condition_type="NEGATION", operands=[vars[0]], text=sentence
                    )

            # Pattern 4: "X hat kein/keine Y" (possession negation)
            match = re.match(
                r"(.+?)\s+hat\s+kein(?:e[nrs]?)?\s+(.+)", sentence, re.IGNORECASE
            )
            if match:
                entity_text = match.group(1).strip()
                object_text = match.group(2).strip()
                combined = f"{entity_text} {object_text}"
                vars = self._extract_variables(combined)
                if vars:
                    return LogicCondition(
                        condition_type="NEGATION", operands=[vars[0]], text=sentence
                    )

            return None
        except Exception as e:
            raise ParsingError(
                "Error parsing negation",
                context={"sentence": sentence},
                original_exception=e,
            )

    def _parse_simple_fact(self, sentence: str) -> Optional[LogicCondition]:
        """
        Parse simple entity-property facts.

        Patterns:
        - "Leo trinkt Tee" -> Variable("Leo", "trinkt_tee") = True
        - "Mark ist aktiv" -> Variable("Mark", "ist_aktiv") = True

        Returns LogicCondition if exactly 1 variable found, None otherwise.
        """
        try:
            # Extract variables from sentence
            variables = self._extract_variables(sentence)

            if len(variables) == 1:
                # Create a simple fact condition with positive variable
                # SIMPLE_FACT means this variable is TRUE
                return LogicCondition(
                    condition_type="SIMPLE_FACT",
                    operands=[variables[0]],  # Use operands, not variables
                    text=sentence,
                )

            return None
        except Exception as e:
            logger.warning(f"Error parsing simple fact '{sentence}': {e}")
            return None

    def _extract_variables(self, text: str) -> List[str]:
        """
        Extracts variables from a text fragment.

        Searches for patterns like:
        - "Leo einen Brandy bestellt" -> Variable("Leo", "bestellt_brandy")
        - "Mark trinkt" -> Variable("Mark", "trinkt_brandy")
        - "Bob ist Lehrer" -> Variable("Bob", "hat_lehrer")

        Returns:
            List of variable names (e.g., ["Leo_bestellt_brandy"])
        """
        variables = []

        # IMPORTANT: Remove prefix text like "Es kann vorkommen, dass"
        # So "dass Mark oder Nick..." -> "Mark oder Nick..."
        text = re.sub(
            r"^.+?(?:dass|wenn|falls)\s+", "", text, flags=re.IGNORECASE
        ).strip()

        # NEW: Prioritize entities that are NOT in detected_objects
        # For "Bob ist Lehrer", Bob (person) should be entity, Lehrer (job) should be object
        # Sort entities: persons (not in objects) first, then objects
        prioritized_entities = sorted(
            self.entities, key=lambda e: e.lower() in self._detected_objects
        )

        # Find entity (person) in text
        for entity in prioritized_entities:
            if entity in text.lower():
                # Extract property (verb + object)
                # Pattern: "[Person] [Verb] [Object]"
                # Example: "Leo einen Brandy bestellt" -> "bestellt_brandy"

                # STRATEGY: Search for verb + main object (noun)
                # Verbs: bestellt, trinkt, mag, isst, will
                # Objects: Brandy, Bier, Pizza, etc.

                property_name = self._extract_action_object(text.lower(), entity)

                # SPECIAL CASE: Only entity name without action
                # Example: "Mark" (from "Mark oder Nick einen Brandy bestellen")
                # -> Use context object
                if (
                    not property_name
                    and hasattr(self, "_context_object")
                    and self._context_object
                ):
                    # Standard action with context object (uniform: "hat")
                    property_name = f"hat_{self._context_object}"
                    logger.debug(
                        f"Entity without action, using context: {entity} -> {property_name}"
                    )

                if property_name:
                    var = LogicVariable(entity.capitalize(), property_name)
                    var_name = var.to_var_name()
                    self.variables[var_name] = var
                    variables.append(var_name)

                    logger.debug(f"Extracted variable: {var}")

                break  # Only one entity per fragment

        return variables

    def _extract_action_object(self, text: str, entity: str) -> str:
        """
        Extracts action + object from text (e.g., "bestellt_brandy").
        All verbs normalized to "hat" for uniform variable names.
        """
        # All verbs map to "hat" (uniform canonical form)
        verb_forms = {
            "bestellt",
            "bestellen",
            "bestellst",
            "trinkt",
            "trinken",
            "trinkst",
            "mag",
            "mögen",
            "magst",
            "isst",
            "essen",
            "kauft",
            "kaufen",
            "nimmt",
            "nehmen",
            "hat",
            "haben",
            "habe",
            "hast",
            "besitzt",
            "besitzen",
            "will",
            "wollen",
            "möchte",
            "möchten",
            "ist",  # NEW: Copula for "X ist Y" patterns
            "sind",
            "war",
            "waren",
            "wird",
            "werden",
        }

        found_verb = any(verb in text for verb in verb_forms)

        if not found_verb:
            return self._normalize_property(text.replace(entity, "").strip())

        # Find object in detected objects
        for detected_obj in self._detected_objects:
            if detected_obj in text:
                return f"hat_{detected_obj}"

        # Implicit object reference ("einen/eine/ein/eins")
        if re.search(r"\b(einen|eine|ein|eins)\b", text):
            if hasattr(self, "_context_object") and self._context_object:
                return f"hat_{self._context_object}"
            return "hat_unbekannt"

        return "hat"

    def _normalize_property(self, text: str) -> str:
        """
        Normalizes property text to a property name.

        Example:
        - "einen brandy bestellt" -> "bestellt_brandy"
        - "trinkt gerne bier" -> "trinkt_bier"
        - "arzt" (if arzt in detected_objects) -> "hat_arzt"
        """
        # Remove filler words
        text = re.sub(
            r"\b(einen?|eine?|der|die|das|den|dem|gerne?|auch|will|ist|sind)\b",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = text.strip()

        # NEW: If text matches a detected object, use "hat_" prefix
        if text.lower() in self._detected_objects:
            return f"hat_{text.lower()}"

        # Normalize to snake_case
        text = re.sub(r"\s+", "_", text)
        text = re.sub(r"[^\w_]", "", text)

        return text

    def get_variable(self, var_name: str) -> Optional[LogicVariable]:
        """Gets LogicVariable for a variable name."""
        return self.variables.get(var_name)

    def _is_assignment_puzzle(self, text: str, entities: List[str]) -> bool:
        """
        Detect if this is an assignment puzzle (bijection problem).

        Heuristics:
        - Multiple entities (>=2)
        - Multiple detected objects (>=2)
        - Text contains "unterschiedliche" or enumeration pattern

        Returns:
            True if assignment puzzle detected
        """
        # Need at least 2 entities and 2 objects
        if len(entities) < 2 or len(self._detected_objects) < 2:
            return False

        # Check for assignment indicators
        text_lower = text.lower()
        assignment_indicators = [
            "unterschiedliche",  # "different jobs"
            "verschiedene",  # "various"
            "jeweils",  # "each"
            "hat welchen",  # "who has which"
            "wer hat",  # "who has"
        ]

        for indicator in assignment_indicators:
            if indicator in text_lower:
                logger.debug(f"Assignment indicator found: '{indicator}'")
                return True

        # Check for colon enumeration (strong signal)
        # "Berufe: Lehrer, Arzt und Ingenieur"
        if re.search(r"\w+:\s*[A-Z]\w+(?:,\s*[A-Z]\w+)+", text):
            logger.debug("Colon enumeration found - assignment puzzle")
            return True

        return False

    def _generate_uniqueness_constraints(
        self, entities: List[str]
    ) -> List[LogicCondition]:
        """
        Generate uniqueness constraints for assignment puzzles.

        For 3 entities (A, B, C) and 3 objects (x, y, z), generates:

        1. At-least-one (each entity has at least one object):
           - (A_x OR A_y OR A_z)
           - (B_x OR B_y OR B_z)
           - (C_x OR C_y OR C_z)

        2. At-most-one (each entity has at most one object):
           - NOT(A_x AND A_y)  [for all pairs]
           - NOT(A_x AND A_z)
           - NOT(A_y AND A_z)
           - [same for B and C]

        3. Each object assigned to at least one entity:
           - (A_x OR B_x OR C_x)
           - (A_y OR B_y OR C_y)
           - (A_z OR B_z OR C_z)

        4. Each object assigned to at most one entity:
           - NOT(A_x AND B_x)  [for all pairs]
           - NOT(A_x AND C_x)
           - NOT(B_x AND C_x)
           - [same for y and z]

        Args:
            entities: List of entity names

        Returns:
            List of LogicCondition objects
        """
        conditions = []

        # Get detected objects
        objects = list(self._detected_objects)

        # Normalize entity names
        entities_lower = [e.lower() for e in entities]

        # CRITICAL FIX: Filter out entities that are actually objects!
        # If "Lehrer" appears in both entities and objects, it's an object, not an entity
        actual_entities = [e for e in entities_lower if e not in self._detected_objects]

        if not actual_entities:
            logger.warning(
                f"No actual entities found after filtering objects. "
                f"Entities: {entities_lower}, Objects: {objects}"
            )
            return []

        if len(actual_entities) != len(objects):
            logger.warning(
                f"Actual entity count ({len(actual_entities)}) != object count ({len(objects)}) - "
                "uniqueness constraints may be incomplete. "
                f"Actual entities: {actual_entities}, Objects: {objects}"
            )

        # 1. At-least-one: Each entity has at least one object
        for entity in actual_entities:
            # Create variables: entity_hat_obj1, entity_hat_obj2, ...
            variables = [f"{entity}_hat_{obj}" for obj in objects]

            # Register variables
            for var in variables:
                if var not in self.variables:
                    # Extract property from variable name
                    property_name = var.split("_", 1)[1] if "_" in var else var
                    self.variables[var] = LogicVariable(
                        entity=entity, property=property_name
                    )

            # (entity_obj1 OR entity_obj2 OR ...)
            # Create single DISJUNCTION with all variables
            conditions.append(
                LogicCondition(
                    condition_type="DISJUNCTION",
                    operands=variables,
                    text=f"{entity} has at least one of {objects}",
                )
            )

        # 2. At-most-one: Each entity has at most one object
        for entity in actual_entities:
            variables = [f"{entity}_hat_{obj}" for obj in objects]

            # For all pairs (i, j) where i < j:
            # NOT(var_i AND var_j) = (NOT var_i OR NOT var_j) [NEVER_BOTH]
            for i in range(len(variables)):
                for j in range(i + 1, len(variables)):
                    conditions.append(
                        LogicCondition(
                            condition_type="NEVER_BOTH",
                            operands=[variables[i], variables[j]],
                            text=f"{entity} cannot have both {objects[i]} and {objects[j]}",
                        )
                    )

        # 3. At-least-one: Each object assigned to at least one entity
        for obj in objects:
            variables = [f"{entity}_hat_{obj}" for entity in actual_entities]

            conditions.append(
                LogicCondition(
                    condition_type="DISJUNCTION",
                    operands=variables,
                    text=f"{obj} assigned to at least one entity",
                )
            )

        # 4. At-most-one: Each object assigned to at most one entity
        for obj in objects:
            variables = [f"{entity}_hat_{obj}" for entity in actual_entities]

            # For all pairs (i, j) where i < j:
            # NOT(var_i AND var_j)
            for i in range(len(variables)):
                for j in range(i + 1, len(variables)):
                    conditions.append(
                        LogicCondition(
                            condition_type="NEVER_BOTH",
                            operands=[variables[i], variables[j]],
                            text=f"{obj} cannot be assigned to both {actual_entities[i]} and {actual_entities[j]}",
                        )
                    )

        logger.debug(f"Generated {len(conditions)} uniqueness constraints")
        return conditions
