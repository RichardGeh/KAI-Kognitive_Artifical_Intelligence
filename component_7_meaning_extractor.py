# component_7_meaning_extractor.py (Fundamentally Overhauled Version with Vector-Based Matching)
import logging
import re
import uuid
from typing import Any

from spacy.tokens import Doc

from component_1_netzwerk import INFO_TYPE_ALIASES
from component_5_linguistik_strukturen import (
    MeaningPoint,
    MeaningPointCategory,
    Modality,
    Polarity,
)
from component_6_linguistik_engine import LinguisticPreprocessor
from component_11_embedding_service import EmbeddingService, ModelNotLoadedError
from component_15_logging_config import get_logger
from component_utils_text_normalization import TextNormalizer

logger = get_logger(__name__)

# Threshold for determining if a vector match is novel or known
# Matches with distance > this value are considered too dissimilar
NOVELTY_THRESHOLD = 5.0


class MeaningPointExtractor:
    """
    Extrahiert MeaningPoints aus Text durch ein zweiphasiges Verfahren:
    1. Phase: Explizite Befehle (regex-basiert) mit confidence=1.0
    2. Phase: Vektor-basierte Prototypen-Erkennung mit distance-basierter confidence
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        preprocessor: LinguisticPreprocessor,
        prototyping_engine=None,
    ):
        """
        Args:
            embedding_service: Service für Vektor-Embeddings
            preprocessor: Linguistischer Vorverarbeitungs-Service
            prototyping_engine: Optional - für Prototypen-Matching (Phase 2)
        """
        self.embedding_service = embedding_service
        self.preprocessor = preprocessor
        self.prototyping_engine = prototyping_engine

        # Initialisiere zentralen TextNormalizer mit spaCy-Integration
        self.text_normalizer = TextNormalizer(preprocessor=preprocessor)

        logger.info("MeaningPointExtractor initialisiert mit zweiphasiger Erkennung")

    def extract(self, doc: Doc) -> list[MeaningPoint]:
        """
        Haupt-Orchestrierungs-Methode für die Meaning-Extraktion.

        Flow:
        1. Prüfe auf explizite Befehle (regex) -> confidence=1.0
        2. Falls kein Befehl: Vektor-basierte Erkennung -> distance-basierte confidence
        3. Falls kein Vektor-Match: Fallback auf Heuristiken
        4. Falls nichts gefunden: UNKNOWN MeaningPoint mit confidence=0.0

        Args:
            doc: spaCy Doc-Objekt mit vorverarbeitetem Text

        Returns:
            Liste mit genau einem MeaningPoint (niemals leer)
        """
        try:
            # Input-Validierung
            if not doc or not doc.text.strip():
                logger.debug("Leere Eingabe erhalten, keine Extraktion möglich")
                return [
                    self._create_meaning_point(
                        category=MeaningPointCategory.UNKNOWN,
                        cue="empty_input",
                        text_span="",
                        confidence=0.0,
                        arguments={},
                    )
                ]

            text = doc.text.strip()

            # Conditional logging für DEBUG
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Starte Meaning-Extraktion",
                    extra={"text_preview": text[:50], "text_length": len(text)},
                )

            # PHASE 1: Explizite Befehle (höchste Priorität, confidence=1.0)
            command_mps = self._parse_explicit_commands(text)
            if command_mps:
                logger.info(
                    "Expliziter Befehl erkannt",
                    extra={
                        "cue": command_mps[0].cue,
                        "category": command_mps[0].category.name,
                    },
                )
                return command_mps

            # PHASE 1.5: Auto-Erkennung von arithmetischen Fragen
            # Muss VOR Definitions-Erkennung erfolgen (wegen "sind" in "10 durch 2 sind...")
            arithmetic_mps = self._detect_arithmetic_question(text, doc)
            if arithmetic_mps:
                logger.info(
                    "Arithmetische Frage erkannt",
                    extra={
                        "cue": arithmetic_mps[0].cue,
                        "category": arithmetic_mps[0].category.name,
                        "confidence": arithmetic_mps[0].confidence,
                    },
                )
                return arithmetic_mps

            # PHASE 1.6: Auto-Erkennung von Definitionen (deklarative Aussagen)
            # Dies muss vor dem Vektor-Matching erfolgen, da es spezifischer ist
            definition_mps = self._detect_declarative_statements(text, doc)
            if definition_mps:
                logger.info(
                    "Definition erkannt",
                    extra={
                        "cue": definition_mps[0].cue,
                        "category": definition_mps[0].category.name,
                    },
                )
                return definition_mps

            # PHASE 2: Vektor-basierte Prototypen-Erkennung
            if self.prototyping_engine and self.embedding_service.is_available():
                vector_mps = self._extract_with_vector_matching(text, doc)
                if vector_mps:
                    logger.info(
                        "Vektor-Match gefunden",
                        extra={
                            "category": vector_mps[0].category.name,
                            "confidence": vector_mps[0].confidence,
                        },
                    )
                    return vector_mps

            # PHASE 3: Fallback auf Heuristiken (für Rückwärtskompatibilität)
            heuristic_mps = self._extract_with_heuristics(doc)
            if heuristic_mps:
                logger.info(
                    "Heuristik-Match gefunden",
                    extra={
                        "cue": heuristic_mps[0].cue,
                        "category": heuristic_mps[0].category.name,
                    },
                )
                return heuristic_mps

            # PHASE 4: Nichts gefunden -> Gib UNKNOWN mit confidence=0.0 zurück
            logger.warning(
                "Keine Bedeutung extrahiert",
                extra={"text_preview": text[:50], "text_length": len(text)},
            )
            unknown_mp = self._create_meaning_point(
                category=MeaningPointCategory.UNKNOWN,
                cue="no_match_found",
                text_span=text,
                confidence=0.0,  # Maximale Unsicherheit
                arguments={"original_text": text},
            )
            return [unknown_mp]

        except Exception as e:
            logger.error(
                "Fehler bei Meaning-Extraktion",
                extra={"error": str(e), "error_type": type(e).__name__},
            )
            # Auch bei Fehler: UNKNOWN zurückgeben statt leere Liste
            return [
                self._create_meaning_point(
                    category=MeaningPointCategory.UNKNOWN,
                    cue="extraction_error",
                    text_span=doc.text if doc else "",
                    confidence=0.0,
                    arguments={"error": str(e)},
                )
            ]

    def _extract_with_vector_matching(self, text: str, doc: Doc) -> list[MeaningPoint]:
        """
        Vektor-basierte Meaning-Extraktion via Prototypen-Matching.

        Args:
            text: Der Eingabetext
            doc: spaCy Doc für zusätzliche linguistische Features

        Returns:
            Liste mit einem MeaningPoint oder leere Liste
        """
        try:
            # Erzeuge Embedding-Vektor für den Eingabesatz
            vector = self.embedding_service.get_embedding(text)

            # Conditional logging für DEBUG (Performance-kritisch)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Embedding für Meaning-Extraktion erzeugt",
                    extra={"dimensions": len(vector), "text_preview": text[:30]},
                )

            # Finde besten Match unter allen Prototypen
            match_result = self.prototyping_engine.find_best_match(vector)

            if not match_result:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Kein Prototyp-Match gefunden")
                return []

            prototype, distance = match_result

            # Conditional logging für DEBUG
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Prototyp-Match gefunden",
                    extra={"prototype_id": prototype["id"][:8], "distance": distance},
                )

            # Berechne Confidence aus Distanz (näher = höhere Confidence)
            # confidence = max(0, 1 - (distance / NOVELTY_THRESHOLD))
            # Sicherstellen, dass Confidence zwischen 0 und 1 bleibt
            confidence = max(0.0, min(1.0, 1.0 - (distance / NOVELTY_THRESHOLD)))

            # Nur Matches mit Confidence > 0.3 akzeptieren (empirischer Schwellwert)
            if confidence < 0.3:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "Confidence zu niedrig, verwerfe Match",
                        extra={"confidence": confidence, "threshold": 0.3},
                    )
                return []

            # Hole Kategorie aus Prototyp
            category_str = prototype.get("category", "UNKNOWN")
            try:
                category = MeaningPointCategory[category_str.upper()]
            except KeyError:
                logger.warning(
                    "Unbekannte Prototyp-Kategorie",
                    extra={
                        "category": category_str,
                        "prototype_id": prototype["id"][:8],
                    },
                )
                category = MeaningPointCategory.UNKNOWN

            # Extrahiere Thema/Argumente aus dem Satz
            arguments = self._extract_arguments_from_text(text, doc, category)

            # Erstelle MeaningPoint mit berechneter Confidence
            mp = self._create_meaning_point(
                category=category,
                cue=f"vector_match_{prototype['id'][:8]}",
                text_span=text,
                confidence=confidence,
                arguments=arguments,
                source_rules=[f"prototype:{prototype['id']}"],
            )

            logger.info(
                "Vektor-Match erstellt",
                extra={
                    "category": category.name,
                    "confidence": confidence,
                    "distance": distance,
                    "prototype_id": prototype["id"][:8],
                },
            )
            return [mp]

        except ModelNotLoadedError as e:
            logger.warning(f"Embedding-Service nicht verfügbar: {e}")
            return []
        except Exception as e:
            logger.error(f"Fehler bei Vektor-Matching: {e}", exc_info=True)
            return []

    def _extract_arguments_from_text(
        self, text: str, doc: Doc, category: MeaningPointCategory
    ) -> dict[str, Any]:
        """
        Extrahiert Argumente aus dem Text basierend auf der erkannten Kategorie.

        Args:
            text: Der Eingabetext
            doc: spaCy Doc
            category: Die erkannte Kategorie

        Returns:
            Dictionary mit extrahierten Argumenten
        """
        arguments = {}

        try:
            # Kategorie-spezifische Extraktion
            if category == MeaningPointCategory.QUESTION:
                # Vollständige W-Wort-Liste (konsistent mit Heuristiken)
                wh_words = [
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
                words = text.lower().split()

                for wh in wh_words:
                    if wh in words:
                        arguments["question_word"] = wh
                        # Thema ist oft nach dem Fragewort und "ist"
                        if "ist" in words:
                            ist_idx = words.index("ist")
                            if ist_idx + 1 < len(words):
                                topic_words = words[ist_idx + 1 :]
                                topic = " ".join(topic_words).rstrip("?")
                                arguments["topic"] = self.text_normalizer.clean_entity(
                                    topic
                                )
                        break

            elif category == MeaningPointCategory.COMMAND:
                # Bei Befehlen das Hauptverb extrahieren
                for token in doc:
                    if token.pos_ == "VERB":
                        arguments["action"] = token.lemma_
                        break

            elif category == MeaningPointCategory.DEFINITION:
                # Bei Definitionen Subjekt und Prädikat
                for token in doc:
                    if token.dep_ == "sb":  # Subjekt
                        arguments["subject"] = token.text.lower()
                    elif token.dep_ == "pd":  # Prädikativ
                        arguments["predicate"] = token.text.lower()

            # Fallback: Speichere den gesamten Text
            if not arguments:
                arguments["text"] = text

        except Exception as e:
            logger.warning(f"Fehler bei Argument-Extraktion: {e}")
            arguments["text"] = text

        return arguments

    def _parse_explicit_commands(self, text: str) -> list[MeaningPoint]:
        """
        Erkennt explizite, unmissverständliche Befehle via Regex.
        Diese haben immer confidence=1.0, da sie eindeutig sind.

        Args:
            text: Der zu parsende Text

        Returns:
            Liste mit einem MeaningPoint (bei Match) oder leere Liste
        """
        try:
            # BEFEHL: definiere:
            define_match = re.match(
                r"^\s*definiere:\s*(\S+)\s*/\s*(.*?)\s*=\s*(.*)$", text, re.IGNORECASE
            )
            if define_match:
                topic, key_path_str, value = define_match.groups()
                return [
                    self._create_meaning_point(
                        category=MeaningPointCategory.COMMAND,
                        cue="definiere:",
                        text_span=text,
                        modality=Modality.IMPERATIVE,
                        confidence=1.0,  # Expliziter Befehl = maximale Confidence
                        arguments={
                            "command": "definiere",
                            "topic": topic.lower(),
                            "key_path": [
                                p.strip().lower() for p in key_path_str.split("/")
                            ],
                            "value": value.strip(),
                        },
                    )
                ]

            # BEFEHL: lerne muster:
            learn_pattern_match = re.match(
                r'^\s*lerne muster:\s*"(.*)"(?:\s*bedeutet\s*(\S+))?\s*$',
                text,
                re.IGNORECASE,
            )
            if learn_pattern_match:
                example_sentence, relation_type = learn_pattern_match.groups()
                if relation_type is None:
                    relation_type = "IS_A"

                return [
                    self._create_meaning_point(
                        category=MeaningPointCategory.COMMAND,
                        cue="lerne muster:",
                        text_span=text,
                        modality=Modality.IMPERATIVE,
                        confidence=1.0,  # Expliziter Befehl = maximale Confidence
                        arguments={
                            "command": "learn_pattern",
                            "example_sentence": example_sentence.strip(),
                            "relation_type": relation_type.strip().upper(),
                        },
                    )
                ]

            # BEFEHL: ingestiere text:
            ingest_match = re.match(
                r'^\s*ingestiere text:\s*"(.*)"\s*$', text, re.IGNORECASE | re.DOTALL
            )
            if ingest_match:
                text_to_ingest = ingest_match.group(1)
                return [
                    self._create_meaning_point(
                        category=MeaningPointCategory.COMMAND,
                        cue="ingestiere text:",
                        text_span=text,
                        modality=Modality.IMPERATIVE,
                        confidence=1.0,  # Expliziter Befehl = maximale Confidence
                        arguments={
                            "command": "ingest_text",
                            "text_to_ingest": text_to_ingest.strip(),
                        },
                    )
                ]

            # BEFEHL: lerne: (einfache Lernform für Wörter/Sätze)
            learn_simple_match = re.match(r"^\s*lerne:\s*(.+)\s*$", text, re.IGNORECASE)
            if learn_simple_match:
                text_to_learn = learn_simple_match.group(1).strip()
                return [
                    self._create_meaning_point(
                        category=MeaningPointCategory.COMMAND,
                        cue="lerne:",
                        text_span=text,
                        modality=Modality.IMPERATIVE,
                        confidence=1.0,  # Expliziter Befehl = maximale Confidence
                        arguments={
                            "command": "learn_simple",
                            "text_to_learn": text_to_learn,
                        },
                    )
                ]

            # BEFEHL: Datei-Commands (lese datei:, ingestiere dokument:, verarbeite pdf:, lade datei:)
            file_command_match = re.match(
                r"^\s*(?:lese datei|ingestiere dokument|verarbeite pdf|lade datei):\s*(.+)\s*$",
                text,
                re.IGNORECASE,
            )
            if file_command_match:
                file_path = file_command_match.group(1).strip()

                # Erkenne den spezifischen Command-Typ aus dem Text
                command_type = None
                command_cue = None
                if re.match(r"^\s*lese datei:", text, re.IGNORECASE):
                    command_type = "read_file"
                    command_cue = "lese datei:"
                elif re.match(r"^\s*ingestiere dokument:", text, re.IGNORECASE):
                    command_type = "ingest_document"
                    command_cue = "ingestiere dokument:"
                elif re.match(r"^\s*verarbeite pdf:", text, re.IGNORECASE):
                    command_type = "process_pdf"
                    command_cue = "verarbeite pdf:"
                elif re.match(r"^\s*lade datei:", text, re.IGNORECASE):
                    command_type = "load_file"
                    command_cue = "lade datei:"

                return [
                    self._create_meaning_point(
                        category=MeaningPointCategory.COMMAND,
                        cue=command_cue,
                        text_span=text,
                        modality=Modality.IMPERATIVE,
                        confidence=0.95,  # Sehr hohe Confidence für klare Datei-Commands
                        arguments={
                            "command": command_type,
                            "file_path": file_path,
                        },
                    )
                ]

            return []

        except Exception as e:
            logger.error(f"Fehler beim Parsen expliziter Befehle: {e}", exc_info=True)
            return []

    def _detect_declarative_statements(self, text: str, doc: Doc) -> list[MeaningPoint]:
        """
        Erkennt deklarative Aussagen automatisch und wandelt sie in DEFINITION MeaningPoints um.
        Entfernt die Notwendigkeit für explizite "Ingestiere Text:"-Befehle.

        Unterstützte Muster:
        - IS_A: "X ist ein/eine Y" -> (X, IS_A, Y)
        - HAS_PROPERTY: "X ist Y" (Adjektiv) -> (X, HAS_PROPERTY, Y)
        - CAPABLE_OF: "X kann Y" -> (X, CAPABLE_OF, Y)
        - PART_OF: "X hat Y" / "X gehört zu Y" -> (X, PART_OF, Y)
        - LOCATED_IN: "X liegt in Y" / "X ist in Y" -> (X, LOCATED_IN, Y)

        Args:
            text: Der zu analysierende Text
            doc: spaCy Doc für linguistische Analyse

        Returns:
            Liste mit einem DEFINITION MeaningPoint oder leere Liste
        """
        try:
            text_lower = text.lower().strip()

            # Filter: Ignoriere Fragen (beginnen mit Fragewörtern)
            question_words = [
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
            first_word = text_lower.split()[0] if text_lower.split() else ""
            if first_word in question_words:
                return []

            # Pattern 1: IS_A - "X ist ein/eine Y"
            is_a_match = re.match(
                r"^\s*(.+?)\s+ist\s+(?:ein|eine|der|die|das)\s+(.+?)\s*\.?\s*$",
                text_lower,
                re.IGNORECASE,
            )
            if is_a_match:
                subject_raw = is_a_match.group(1).strip()
                object_raw = is_a_match.group(2).strip()

                # Bereinige Entities
                subject = self.text_normalizer.clean_entity(subject_raw)
                object_entity = self.text_normalizer.clean_entity(object_raw)

                logger.debug(f"IS_A erkannt: '{subject}' ist ein '{object_entity}'")

                # PHASE 3 (Schritt 3): Hohe Confidence für eindeutige IS_A-Muster mit Artikel
                # confidence >= 0.85 -> Auto-Save ohne Rückfrage
                return [
                    self._create_meaning_point(
                        category=MeaningPointCategory.DEFINITION,
                        cue="auto_detect_is_a",
                        text_span=text,
                        confidence=0.92,  # Sehr hohe Confidence für klare IS_A Muster
                        arguments={
                            "subject": subject,
                            "relation_type": "IS_A",
                            "object": object_entity,
                            "auto_detected": True,
                        },
                    )
                ]

            # Pattern 2: IS_A (Plural ohne Artikel) - "X sind Y"
            # Behandelt Fälle wie "Katzen sind Tiere", "Hunde sind Säugetiere"
            is_a_plural_match = re.match(
                r"^\s*(.+?)\s+sind\s+(?!ein|eine|der|die|das)(.+?)\s*\.?\s*$",
                text_lower,
                re.IGNORECASE,
            )
            if is_a_plural_match:
                subject_raw = is_a_plural_match.group(1).strip()
                object_raw = is_a_plural_match.group(2).strip()

                # Prüfe ob das Objekt wahrscheinlich ein Nomen ist (und nicht Adjektiv)
                # Heuristik: Nomen sind länger als 3 Zeichen und enden nicht auf typische Adjektiv-Endungen
                adjective_endings = ["bar", "lich", "ig", "isch", "los", "voll", "sam"]
                object_clean = self.text_normalizer.clean_entity(object_raw)

                is_likely_noun = len(object_clean) > 2 and not any(
                    object_clean.endswith(ending) for ending in adjective_endings
                )

                # Zusätzliche Heuristik: Prüfe mit spaCy ob das Objekt ein Nomen ist
                try:
                    # Nutze spaCy für POS-Tagging
                    object_tokens = [
                        token
                        for token in doc
                        if token.text.lower() in object_raw.lower()
                    ]
                    if object_tokens:
                        # Wenn eines der Tokens ein NOUN ist, ist es wahrscheinlich IS_A
                        has_noun = any(token.pos_ == "NOUN" for token in object_tokens)
                        if has_noun:
                            is_likely_noun = True
                except Exception:
                    pass  # Fallback auf Heuristik oben

                if is_likely_noun:
                    subject = self.text_normalizer.clean_entity(subject_raw)
                    object_entity = self.text_normalizer.clean_entity(object_raw)

                    logger.debug(
                        f"IS_A (Plural) erkannt: '{subject}' sind '{object_entity}'"
                    )

                    # PHASE 3 (Schritt 3): Hohe Confidence für Plural IS_A ohne Artikel
                    # confidence >= 0.85 -> Auto-Save
                    return [
                        self._create_meaning_point(
                            category=MeaningPointCategory.DEFINITION,
                            cue="auto_detect_is_a_plural",
                            text_span=text,
                            confidence=0.87,  # Hohe Confidence (>= 0.85), auto-save
                            arguments={
                                "subject": subject,
                                "relation_type": "IS_A",
                                "object": object_entity,
                                "auto_detected": True,
                            },
                        )
                    ]
                # Sonst: Falle durch zu HAS_PROPERTY (siehe unten)

            # Pattern 3: HAS_PROPERTY - "X ist Y" (ohne Artikel -> Eigenschaft)
            has_property_match = re.match(
                r"^\s*(.+?)\s+(?:ist|sind)\s+(?!ein|eine|der|die|das)(.+?)\s*\.?\s*$",
                text_lower,
                re.IGNORECASE,
            )
            if has_property_match:
                subject_raw = has_property_match.group(1).strip()
                property_raw = has_property_match.group(2).strip()

                # Prüfe, ob es eine Eigenschaft ist (kein weiteres Nomen mit Artikel)
                # Einfache Heuristik: Eigenschaft hat kein "in", "von", "aus"
                if not any(
                    prep in property_raw.split()
                    for prep in ["in", "von", "aus", "bei", "zu"]
                ):
                    subject = self.text_normalizer.clean_entity(subject_raw)
                    property_value = self.text_normalizer.clean_entity(property_raw)

                    logger.debug(
                        f"HAS_PROPERTY erkannt: '{subject}' ist '{property_value}'"
                    )

                    # PHASE 3 (Schritt 3): Mittlere Confidence für Eigenschaften (mehrdeutig)
                    # 0.70 <= confidence < 0.85 -> Confirmation erforderlich
                    return [
                        self._create_meaning_point(
                            category=MeaningPointCategory.DEFINITION,
                            cue="auto_detect_has_property",
                            text_span=text,
                            confidence=0.78,  # Mittlere Confidence -> triggert Confirmation Gate
                            arguments={
                                "subject": subject,
                                "relation_type": "HAS_PROPERTY",
                                "object": property_value,
                                "auto_detected": True,
                            },
                        )
                    ]

            # Pattern 4: CAPABLE_OF - "X kann Y"
            capable_of_match = re.match(
                r"^\s*(.+?)\s+(?:kann|können)\s+(.+?)\s*\.?\s*$",
                text_lower,
                re.IGNORECASE,
            )
            if capable_of_match:
                subject_raw = capable_of_match.group(1).strip()
                ability_raw = capable_of_match.group(2).strip()

                subject = self.text_normalizer.clean_entity(subject_raw)
                ability = self.text_normalizer.clean_entity(ability_raw)

                logger.debug(f"CAPABLE_OF erkannt: '{subject}' kann '{ability}'")

                # PHASE 3 (Schritt 3): Sehr hohe Confidence für "kann"-Konstruktionen
                # confidence >= 0.85 -> Auto-Save
                return [
                    self._create_meaning_point(
                        category=MeaningPointCategory.DEFINITION,
                        cue="auto_detect_capable_of",
                        text_span=text,
                        confidence=0.91,  # Sehr hohe Confidence -> Auto-Save
                        arguments={
                            "subject": subject,
                            "relation_type": "CAPABLE_OF",
                            "object": ability,
                            "auto_detected": True,
                        },
                    )
                ]

            # Pattern 5: PART_OF - "X hat Y" / "X gehört zu Y"
            part_of_match = re.match(
                r"^\s*(.+?)\s+(?:hat|haben|gehört zu|besitzt)\s+(.+?)\s*\.?\s*$",
                text_lower,
                re.IGNORECASE,
            )
            if part_of_match:
                subject_raw = part_of_match.group(1).strip()
                object_raw = part_of_match.group(2).strip()

                subject = self.text_normalizer.clean_entity(subject_raw)
                part = self.text_normalizer.clean_entity(object_raw)

                logger.debug(f"PART_OF erkannt: '{subject}' hat/gehört zu '{part}'")

                # PHASE 3 (Schritt 3): Hohe Confidence für PART_OF Relationen
                # confidence >= 0.85 -> Auto-Save
                return [
                    self._create_meaning_point(
                        category=MeaningPointCategory.DEFINITION,
                        cue="auto_detect_part_of",
                        text_span=text,
                        confidence=0.88,  # Hohe Confidence -> Auto-Save
                        arguments={
                            "subject": subject,
                            "relation_type": "PART_OF",
                            "object": part,
                            "auto_detected": True,
                        },
                    )
                ]

            # Pattern 6: LOCATED_IN - "X liegt in Y" / "X ist in Y" / "X befindet sich in Y"
            located_in_match = re.match(
                r"^\s*(.+?)\s+(?:liegt in|ist in|befindet sich in)\s+(.+?)\s*\.?\s*$",
                text_lower,
                re.IGNORECASE,
            )
            if located_in_match:
                subject_raw = located_in_match.group(1).strip()
                location_raw = located_in_match.group(2).strip()

                subject = self.text_normalizer.clean_entity(subject_raw)
                location = self.text_normalizer.clean_entity(location_raw)

                logger.debug(f"LOCATED_IN erkannt: '{subject}' liegt in '{location}'")

                # PHASE 3 (Schritt 3): Sehr hohe Confidence für klare Lokations-Muster
                # confidence >= 0.85 -> Auto-Save
                return [
                    self._create_meaning_point(
                        category=MeaningPointCategory.DEFINITION,
                        cue="auto_detect_located_in",
                        text_span=text,
                        confidence=0.93,  # Sehr hohe Confidence -> Auto-Save
                        arguments={
                            "subject": subject,
                            "relation_type": "LOCATED_IN",
                            "object": location,
                            "auto_detected": True,
                        },
                    )
                ]

            # Keine deklarative Aussage erkannt
            return []

        except Exception as e:
            logger.error(f"Fehler bei Deklaration-Erkennung: {e}", exc_info=True)
            return []

    def _detect_arithmetic_question(self, text: str, doc: Doc) -> list[MeaningPoint]:
        """
        Erkennt arithmetische Fragen automatisch.

        Unterstützte Muster:
        - "Was ist 3 + 5?" -> 0.95
        - "Wie viel ist 7 mal 8?" -> 0.93
        - "Wieviel sind 10 durch 2?" -> 0.92
        - "Berechne 15 minus 6" -> 0.90
        - "Was ist drei plus fünf?" -> 0.95 (Zahlwörter)

        Args:
            text: Der zu analysierende Text
            doc: spaCy Doc für linguistische Analyse

        Returns:
            Liste mit einem ARITHMETIC_QUESTION MeaningPoint oder leere Liste
        """
        try:
            text_lower = text.lower().strip()

            # Arithmetische Trigger-Wörter (Operatoren)
            arithmetic_operators = [
                "plus",
                "minus",
                "mal",
                "geteilt",
                "durch",
                "+",
                "-",
                "*",
                "/",
                "×",
                "÷",
                "multipliziert",
                "addiert",
                "subtrahiert",
                "dividiert",
            ]

            # Frage-Trigger
            question_triggers = [
                "was ist",
                "wie viel",
                "wieviel",
                "wie viele",
                "berechne",
                "rechne",
                "errechne",
                "berechnen",
            ]

            # Prüfe auf arithmetische Operatoren
            has_arithmetic = any(op in text_lower for op in arithmetic_operators)

            # Prüfe auf Frage-Trigger
            has_question = any(trigger in text_lower for trigger in question_triggers)

            # Wenn beides vorhanden: Hohe Confidence
            if has_arithmetic and has_question:
                confidence = 0.95
            elif has_arithmetic and text.endswith("?"):
                confidence = 0.90
            elif has_arithmetic:
                confidence = 0.80
            else:
                # Keine arithmetische Frage erkannt
                return []

            logger.debug(
                f"Arithmetische Frage erkannt: '{text}' (confidence={confidence})"
            )

            return [
                self._create_meaning_point(
                    category=MeaningPointCategory.ARITHMETIC_QUESTION,
                    cue="auto_detect_arithmetic",
                    text_span=text,
                    confidence=confidence,
                    arguments={
                        "query_text": text,
                        "auto_detected": True,
                    },
                )
            ]

        except Exception as e:
            logger.error(f"Fehler bei Arithmetik-Erkennung: {e}", exc_info=True)
            return []

    def _extract_with_heuristics(self, doc: Doc) -> list[MeaningPoint]:
        """
        Fallback-Heuristiken für häufige Frage-Muster.
        Verwendet wenn weder explizite Befehle noch Vektor-Matching greifen.

        Args:
            doc: spaCy Doc-Objekt

        Returns:
            Liste mit einem MeaningPoint oder leere Liste
        """
        try:
            text = doc.text.strip()

            # Heuristik 1: Spezifische Info-Frage (z.B. "Was ist die Farbe von Apfel?")
            match_specific_info = re.match(
                r"^\s*was ist (?:ein|eine|der|die|das)\s+(.+?)\s+(?:von|für)\s+(.+?)\??\s*$",
                text,
                re.IGNORECASE,
            )
            if match_specific_info:
                info_type_str, topic_str = match_specific_info.groups()
                kanonischer_typ = INFO_TYPE_ALIASES.get(info_type_str.lower().strip())
                if kanonischer_typ:
                    return [
                        self._create_meaning_point(
                            category=MeaningPointCategory.QUESTION,
                            cue="heuristic_specific_info_question",
                            text_span=text,
                            confidence=0.85,  # Heuristiken haben niedrigere Confidence als explizite Befehle
                            arguments={
                                "info_type": kanonischer_typ,
                                "topic": self.text_normalizer.clean_entity(topic_str),
                            },
                        )
                    ]

            # Heuristik 2: Property-Frage (z.B. "Was ist die Farbe von Apfel?")
            match_prop = re.match(
                r"^\s*(?:was ist|welche|welches)\s+(?:der|die|das)\s+(.+?)\s+von\s+(.+?)\??\s*$",
                text,
                re.IGNORECASE,
            )
            if match_prop:
                prop_name, topic_str = match_prop.groups()
                return [
                    self._create_meaning_point(
                        category=MeaningPointCategory.QUESTION,
                        cue="heuristic_conceptual_property_question",
                        text_span=text,
                        confidence=0.85,
                        arguments={
                            "property_name": prop_name.strip().lower(),
                            "topic": self.text_normalizer.clean_entity(topic_str),
                        },
                    )
                ]

            # Heuristik 3: "Was weißt du über X alles?" / "Zeige alles über X" (erweiterte Info-Frage)
            match_show_all = re.match(
                r"^\s*(?:was weißt du (?:über|von|zu)\s+(.+?)\s+(?:alles|gesamt)|zeige?\s+alles\s+über\s+(.+?))\??\s*$",
                text,
                re.IGNORECASE,
            )
            if match_show_all:
                # Eine der beiden Gruppen ist nicht None
                topic_str_raw = match_show_all.group(1) or match_show_all.group(2)
                cleaned_topic = self.text_normalizer.clean_entity(topic_str_raw)
                return [
                    self._create_meaning_point(
                        category=MeaningPointCategory.QUESTION,
                        cue="heuristic_show_all_knowledge",
                        text_span=text,
                        confidence=0.90,  # Sehr spezifische Frageform, hohe Confidence
                        arguments={
                            "topic": cleaned_topic,
                            "query_type": "show_all_knowledge",  # Marker für umfassende Antwort
                        },
                    )
                ]

            # Heuristik 3.1: Episodisches Gedächtnis - "Wann habe ich X gelernt?"
            match_episodic_when_learned = re.match(
                r"^\s*wann\s+(?:habe ich|hab ich|wurde|haben wir)\s+(?:über\s+)?(.+?)\s+gelernt\??\s*$",
                text,
                re.IGNORECASE,
            )
            if match_episodic_when_learned:
                topic_str_raw = match_episodic_when_learned.group(1).strip()
                cleaned_topic = (
                    self.text_normalizer.clean_entity(topic_str_raw)
                    if topic_str_raw
                    else None
                )
                return [
                    self._create_meaning_point(
                        category=MeaningPointCategory.QUESTION,
                        cue="heuristic_episodic_when_learned",
                        text_span=text,
                        confidence=0.92,  # Sehr spezifische Frageform
                        arguments={
                            "query_type": "episodic_memory",
                            "episodic_query_type": "when_learned",
                            "topic": cleaned_topic,
                        },
                    )
                ]

            # Heuristik 3.2: Episodisches Gedächtnis - "Zeige (mir) Episoden"
            match_episodic_show_episodes = re.match(
                r"^\s*zeige?\s+(?:mir\s+)?(?:alle\s+)?episoden\s*(?:über|von|zu|für)?\s*(.*?)\??\s*$",
                text,
                re.IGNORECASE,
            )
            if match_episodic_show_episodes:
                topic_str_raw = match_episodic_show_episodes.group(1).strip()
                cleaned_topic = (
                    self.text_normalizer.clean_entity(topic_str_raw)
                    if topic_str_raw
                    else None
                )
                return [
                    self._create_meaning_point(
                        category=MeaningPointCategory.QUESTION,
                        cue="heuristic_episodic_show_episodes",
                        text_span=text,
                        confidence=0.93,  # Sehr spezifische Frageform
                        arguments={
                            "query_type": "episodic_memory",
                            "episodic_query_type": "show_episodes",
                            "topic": cleaned_topic,
                        },
                    )
                ]

            # Heuristik 3.3: Episodisches Gedächtnis - "Zeige Lernverlauf"
            match_episodic_learning_history = re.match(
                r"^\s*zeige?\s+(?:mir\s+)?(?:den\s+)?lernverlauf\s*(?:von|für|über)?\s*(.*?)\??\s*$",
                text,
                re.IGNORECASE,
            )
            if match_episodic_learning_history:
                topic_str_raw = match_episodic_learning_history.group(1).strip()
                cleaned_topic = (
                    self.text_normalizer.clean_entity(topic_str_raw)
                    if topic_str_raw
                    else None
                )
                return [
                    self._create_meaning_point(
                        category=MeaningPointCategory.QUESTION,
                        cue="heuristic_episodic_learning_history",
                        text_span=text,
                        confidence=0.91,
                        arguments={
                            "query_type": "episodic_memory",
                            "episodic_query_type": "learning_history",
                            "topic": cleaned_topic,
                        },
                    )
                ]

            # Heuristik 3.4: Episodisches Gedächtnis - "Was habe ich über X gelernt?"
            match_episodic_what_learned = re.match(
                r"^\s*was\s+(?:habe ich|hab ich|haben wir)\s+über\s+(.+?)\s+gelernt\??\s*$",
                text,
                re.IGNORECASE,
            )
            if match_episodic_what_learned:
                topic_str_raw = match_episodic_what_learned.group(1).strip()
                cleaned_topic = self.text_normalizer.clean_entity(topic_str_raw)
                return [
                    self._create_meaning_point(
                        category=MeaningPointCategory.QUESTION,
                        cue="heuristic_episodic_what_learned",
                        text_span=text,
                        confidence=0.90,
                        arguments={
                            "query_type": "episodic_memory",
                            "episodic_query_type": "what_learned",
                            "topic": cleaned_topic,
                        },
                    )
                ]

            # Heuristik 3.5: Räumliche Queries - Grid-basiert
            match_grid_query = re.match(
                r"^\s*(?:wo ist|zeige mir)\s+(?:das\s+)?(?:feld|position|stelle)\s+(.+?)\s+(?:auf|in|im)\s+(?:einem?\s+)?(.+?)\??\s*$",
                text,
                re.IGNORECASE,
            )
            if match_grid_query:
                position_str, grid_type = match_grid_query.groups()
                return [
                    self._create_meaning_point(
                        category=MeaningPointCategory.QUESTION,
                        cue="heuristic_spatial_grid_query",
                        text_span=text,
                        confidence=0.88,
                        arguments={
                            "query_type": "spatial_reasoning",
                            "spatial_query_type": "grid_query",
                            "query_position": position_str.strip(),
                            "grid_type": grid_type.strip().lower(),
                        },
                    )
                ]

            # Heuristik 3.6: Räumliche Queries - Relationen
            match_spatial_relation = re.match(
                r"^\s*(?:ist|liegt)\s+(.+?)\s+(nördlich von|südlich von|östlich von|westlich von|neben|zwischen|über|unter)\s+(.+?)\??\s*$",
                text,
                re.IGNORECASE,
            )
            if match_spatial_relation:
                subject, relation, target = match_spatial_relation.groups()
                relation_map = {
                    "nördlich von": "NORTH_OF",
                    "südlich von": "SOUTH_OF",
                    "östlich von": "EAST_OF",
                    "westlich von": "WEST_OF",
                    "neben": "ADJACENT_TO",
                    "zwischen": "BETWEEN",
                    "über": "ABOVE",
                    "unter": "BELOW",
                }
                relation_type = relation_map.get(relation.lower(), "ADJACENT_TO")

                return [
                    self._create_meaning_point(
                        category=MeaningPointCategory.QUESTION,
                        cue="heuristic_spatial_relation_query",
                        text_span=text,
                        confidence=0.90,
                        arguments={
                            "query_type": "spatial_reasoning",
                            "spatial_query_type": "relation_query",
                            "subject": subject.strip().lower(),
                            "relation": relation_type,
                            "target": target.strip().lower(),
                        },
                    )
                ]

            # Heuristik 3.7: Räumliche Queries - Path-Finding
            match_path_finding = re.match(
                r"^\s*wie\s+(?:komme ich|kommt man|gelange ich)\s+von\s+(.+?)\s+(?:nach|zu)\s+(.+?)\??\s*$",
                text,
                re.IGNORECASE,
            )
            if match_path_finding:
                start, goal = match_path_finding.groups()
                return [
                    self._create_meaning_point(
                        category=MeaningPointCategory.QUESTION,
                        cue="heuristic_spatial_path_finding",
                        text_span=text,
                        confidence=0.89,
                        arguments={
                            "query_type": "spatial_reasoning",
                            "spatial_query_type": "path_finding",
                            "start": start.strip().lower(),
                            "goal": goal.strip().lower(),
                        },
                    )
                ]

            # Heuristik 3.8: Räumliche Queries - Position-basiert
            match_position_query = re.match(
                r"^\s*(?:wo liegt|wo befindet sich|wo ist)\s+(.+?)\??\s*$",
                text,
                re.IGNORECASE,
            )
            if match_position_query:
                object_name = match_position_query.group(1).strip()
                cleaned_object = self.text_normalizer.clean_entity(object_name)
                return [
                    self._create_meaning_point(
                        category=MeaningPointCategory.QUESTION,
                        cue="heuristic_spatial_position_query",
                        text_span=text,
                        confidence=0.87,
                        arguments={
                            "query_type": "spatial_reasoning",
                            "spatial_query_type": "position_query",
                            "object": cleaned_object,
                        },
                    )
                ]

            # Heuristik 4: "Was ist X?" Frage (häufigste Form)
            match_what_is = re.match(
                r"^\s*was\s+ist\s+(.+?)\??\s*$",
                text,
                re.IGNORECASE,
            )
            if match_what_is:
                topic_str_raw = match_what_is.group(1).strip()
                cleaned_topic = self.text_normalizer.clean_entity(topic_str_raw)
                return [
                    self._create_meaning_point(
                        category=MeaningPointCategory.QUESTION,
                        cue="heuristic_question_wh",
                        text_span=text,
                        confidence=0.80,
                        arguments={"topic": cleaned_topic},
                    )
                ]

            # Heuristik 5: "Wo ist/liegt X?" Frage
            match_where_is = re.match(
                r"^\s*wo\s+(?:ist|liegt)\s+(.+?)\??\s*$",
                text,
                re.IGNORECASE,
            )
            if match_where_is:
                topic_str = match_where_is.group(1).strip()
                cleaned_topic = self.text_normalizer.clean_entity(topic_str)
                return [
                    self._create_meaning_point(
                        category=MeaningPointCategory.QUESTION,
                        cue="heuristic_question_where",
                        text_span=text,
                        confidence=0.85,
                        arguments={
                            "topic": cleaned_topic,
                            "property_name": "LOCATED_IN",  # Implizite Ortsfrage
                        },
                    )
                ]

            # Heuristik 6: Wer-Fragen (nach Personen/Akteuren)
            match_wer = re.match(
                r"^\s*wer\s+(.+?)\??\s*$",
                text,
                re.IGNORECASE,
            )
            if match_wer:
                rest_of_question = match_wer.group(1).strip()

                # Versuche Thema zu extrahieren (erweiterte Verb-Liste)
                topic_match = re.search(
                    r"(?:ist|sind|hat|haben|war|waren|wird|werden|kann|können|kennt|kennen|macht|machen|sagt|sagen|weiß|wissen)\s+(.+?)(?:\s|$)",
                    rest_of_question,
                    re.IGNORECASE,
                )
                if topic_match:
                    topic_str = topic_match.group(1)
                else:
                    topic_str = rest_of_question

                cleaned_topic = self.text_normalizer.clean_entity(topic_str)

                return [
                    self._create_meaning_point(
                        category=MeaningPointCategory.QUESTION,
                        cue="heuristic_question_wer",
                        text_span=text,
                        confidence=0.80,
                        arguments={
                            "topic": cleaned_topic,
                            "question_word": "wer",
                            "question_type": "person_query",  # Marker für Personen-Frage
                            "full_question": rest_of_question,
                        },
                    )
                ]

            # Heuristik 7: Wie-Fragen (nach Prozessen/Methoden)
            match_wie = re.match(
                r"^\s*wie\s+(.+?)\??\s*$",
                text,
                re.IGNORECASE,
            )
            if match_wie:
                rest_of_question = match_wie.group(1).strip()

                # Versuche Thema zu extrahieren
                topic_match = re.search(
                    r"(?:funktioniert|geht|macht man|ist|sind)\s+(.+?)(?:\s|$)",
                    rest_of_question,
                    re.IGNORECASE,
                )
                if topic_match:
                    topic_str = topic_match.group(1)
                else:
                    topic_str = rest_of_question

                cleaned_topic = self.text_normalizer.clean_entity(topic_str)

                return [
                    self._create_meaning_point(
                        category=MeaningPointCategory.QUESTION,
                        cue="heuristic_question_wie",
                        text_span=text,
                        confidence=0.80,
                        arguments={
                            "topic": cleaned_topic,
                            "question_word": "wie",
                            "question_type": "process_query",  # Marker für Prozess-Frage
                            "full_question": rest_of_question,
                        },
                    )
                ]

            # Heuristik 8: Warum/Wieso/Weshalb-Fragen (nach Gründen/Ursachen)
            match_warum = re.match(
                r"^\s*(warum|wieso|weshalb)\s+(.+?)\??\s*$",
                text,
                re.IGNORECASE,
            )
            if match_warum:
                question_word = match_warum.group(1).lower()
                rest_of_question = match_warum.group(2).strip()

                # Versuche Thema zu extrahieren
                topic_match = re.search(
                    r"(?:ist|sind|hat|haben|kann|können)\s+(.+?)(?:\s|$)",
                    rest_of_question,
                    re.IGNORECASE,
                )
                if topic_match:
                    topic_str = topic_match.group(1)
                else:
                    topic_str = rest_of_question

                cleaned_topic = self.text_normalizer.clean_entity(topic_str)

                return [
                    self._create_meaning_point(
                        category=MeaningPointCategory.QUESTION,
                        cue=f"heuristic_question_{question_word}",
                        text_span=text,
                        confidence=0.80,
                        arguments={
                            "topic": cleaned_topic,
                            "question_word": question_word,
                            "question_type": "reason_query",  # Marker für Grund-Frage
                            "full_question": rest_of_question,
                        },
                    )
                ]

            # Heuristik 9: Wann-Fragen (nach Zeit/Zeitpunkten)
            match_wann = re.match(
                r"^\s*wann\s+(.+?)\??\s*$",
                text,
                re.IGNORECASE,
            )
            if match_wann:
                rest_of_question = match_wann.group(1).strip()

                # Versuche Thema zu extrahieren
                topic_match = re.search(
                    r"(?:ist|war|wird|findet statt|passiert)\s+(.+?)(?:\s|$)",
                    rest_of_question,
                    re.IGNORECASE,
                )
                if topic_match:
                    topic_str = topic_match.group(1)
                else:
                    topic_str = rest_of_question

                cleaned_topic = self.text_normalizer.clean_entity(topic_str)

                return [
                    self._create_meaning_point(
                        category=MeaningPointCategory.QUESTION,
                        cue="heuristic_question_wann",
                        text_span=text,
                        confidence=0.80,
                        arguments={
                            "topic": cleaned_topic,
                            "question_word": "wann",
                            "question_type": "time_query",  # Marker für Zeit-Frage
                            "full_question": rest_of_question,
                        },
                    )
                ]

            # Heuristik 10: Flexible Fragewort-Erkennung (Fallback für andere Fragetypen)
            # Erkennt: Sonstige W-Fragen die nicht von den spezifischen Heuristiken erfasst wurden
            match_flexible_wh = re.match(
                r"^\s*(welche[rs]?|wozu|wohin|woher|womit)\s+(.+?)\??\s*$",
                text,
                re.IGNORECASE,
            )
            if match_flexible_wh:
                question_word = match_flexible_wh.group(1).lower()
                rest_of_question = match_flexible_wh.group(2).strip()

                # Versuche Thema zu extrahieren (nach "ist", "sind", "hat", etc.)
                topic_match = re.search(
                    r"(?:ist|sind|hat|haben|kann|können)\s+(.+?)(?:\s|$)",
                    rest_of_question,
                    re.IGNORECASE,
                )
                if topic_match:
                    topic_str = topic_match.group(1)
                else:
                    # Fallback: Nimm den Rest
                    topic_str = rest_of_question

                cleaned_topic = self.text_normalizer.clean_entity(topic_str)

                return [
                    self._create_meaning_point(
                        category=MeaningPointCategory.QUESTION,
                        cue=f"heuristic_question_{question_word}",
                        text_span=text,
                        confidence=0.75,  # Etwas niedrigere Confidence bei Fallback
                        arguments={
                            "topic": cleaned_topic,
                            "question_word": question_word,
                            "full_question": rest_of_question,
                        },
                    )
                ]

            return []

        except Exception as e:
            logger.error(f"Fehler bei heuristischer Extraktion: {e}", exc_info=True)
            return []

    def _create_meaning_point(self, **kwargs) -> MeaningPoint:
        """
        Factory-Methode zum Erstellen von MeaningPoint-Objekten mit sinnvollen Defaults.

        Args:
            **kwargs: Beliebige MeaningPoint-Attribute (überschreiben Defaults)

        Returns:
            Ein vollständig initialisiertes MeaningPoint-Objekt
        """
        try:
            # Sinnvolle Defaults
            defaults = {
                "id": f"mp-{uuid.uuid4().hex[:6]}",
                "modality": Modality.DECLARATIVE,
                "polarity": Polarity.POSITIVE,
                "confidence": 0.7,  # Konservativ, wird oft überschrieben
                "arguments": {},
                "span_offsets": [],
                "source_rules": [],
            }

            # Kategorie-spezifische Modality
            category = kwargs.get("category")
            if category == MeaningPointCategory.QUESTION:
                defaults["modality"] = Modality.INTERROGATIVE
            elif category == MeaningPointCategory.COMMAND:
                defaults["modality"] = Modality.IMPERATIVE

            # Merge mit übergebenen Parametern
            defaults.update(kwargs)

            return MeaningPoint(**defaults)

        except Exception as e:
            logger.error(f"Fehler beim Erstellen des MeaningPoints: {e}", exc_info=True)
            # Rethrow, da ein MeaningPoint essentiell ist
            raise
