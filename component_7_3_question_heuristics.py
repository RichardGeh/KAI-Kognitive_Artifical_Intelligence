# component_7_3_question_heuristics.py
"""
Heuristik-basierte Frage-Erkennung.
Fallback-Heuristiken für häufige Frage-Muster wenn weder explizite
Befehle noch Vektor-Matching greifen.

WICHTIG: KEINE Unicode-Zeichen verwenden, die Windows cp1252 Encoding-Probleme verursachen.
Verboten: OK FEHLER -> x / != <= >= etc.
Erlaubt: [OK] [FEHLER] -> * / != <= >= AND OR NOT

Unterstützt:
- W-Fragen (was, wer, wie, warum, wann, wo, welche, wozu)
- Episodische Queries (Lernverlauf, Episoden)
- Räumliche Queries (Grid, Relationen, Path-Finding, Positionen)
- Spezifische Info-Fragen
"""
import re

from spacy.tokens import Doc

from component_1_netzwerk import INFO_TYPE_ALIASES
from component_5_linguistik_strukturen import MeaningPoint, MeaningPointCategory
from component_6_linguistik_engine import LinguisticPreprocessor
from component_7_meaning_point_factory import create_meaning_point
from component_15_logging_config import get_logger
from component_utils_text_normalization import TextNormalizer

logger = get_logger(__name__)


class QuestionHeuristicsExtractor:
    """
    Heuristik-basierte Erkennung von Frage-Mustern.
    Verwendet wenn weder explizite Befehle noch Vektor-Matching greifen.
    """

    def __init__(self, preprocessor: LinguisticPreprocessor):
        """
        Args:
            preprocessor: Linguistischer Vorverarbeitungs-Service
        """
        self.preprocessor = preprocessor
        self.text_normalizer = TextNormalizer(preprocessor=preprocessor)

    def extract(self, doc: Doc) -> list[MeaningPoint] | None:
        """
        Extrahiert MeaningPoints via Heuristiken.

        Args:
            doc: spaCy Doc-Objekt

        Returns:
            Liste mit einem MeaningPoint oder None
        """
        try:
            text = doc.text.strip()

            # Heuristik 1: Spezifische Info-Frage (z.B. "Was ist die Farbe von Apfel?")
            if mp := self._match_specific_info_question(text):
                return mp

            # Heuristik 2: Property-Frage (z.B. "Was ist die Farbe von Apfel?")
            if mp := self._match_property_question(text):
                return mp

            # Heuristik 3: "Was weißt du über X alles?" / "Zeige alles über X"
            if mp := self._match_show_all_knowledge(text):
                return mp

            # Heuristik 3.1-3.4: Episodisches Gedächtnis
            if mp := self._match_episodic_queries(text):
                return mp

            # Heuristik 3.5-3.8: Räumliche Queries
            if mp := self._match_spatial_queries(text):
                return mp

            # Heuristik 4: "Was ist X?" Frage (häufigste Form)
            if mp := self._match_what_is_question(text):
                return mp

            # Heuristik 4.5: "Was macht X?" Frage (action-based questions)
            if mp := self._match_was_macht_question(text):
                return mp

            # Heuristik 5: "Wo ist/liegt X?" Frage
            if mp := self._match_where_is_question(text):
                return mp

            # Heuristik 6: Wer-Fragen (nach Personen/Akteuren)
            if mp := self._match_wer_question(text):
                return mp

            # Heuristik 7: Wie-Fragen (nach Prozessen/Methoden)
            if mp := self._match_wie_question(text):
                return mp

            # Heuristik 8: Warum/Wieso/Weshalb-Fragen (nach Gründen/Ursachen)
            if mp := self._match_warum_question(text):
                return mp

            # Heuristik 9: Wann-Fragen (nach Zeit/Zeitpunkten)
            if mp := self._match_wann_question(text):
                return mp

            # Heuristik 10: Flexible Fragewort-Erkennung (Fallback)
            if mp := self._match_flexible_wh_question(text):
                return mp

            return None

        except Exception as e:
            logger.error(f"Fehler bei heuristischer Extraktion: {e}", exc_info=True)
            return None

    def _match_specific_info_question(self, text: str) -> list[MeaningPoint] | None:
        """Erkennt spezifische Info-Fragen: "Was ist die Farbe von Apfel?"."""
        match_specific_info = re.match(
            r"^\s*was ist (?:ein|eine|der|die|das)\s+(.+?)\s+(?:von|fuer)\s+(.+?)\??\s*$",
            text,
            re.IGNORECASE,
        )
        if match_specific_info:
            info_type_str, topic_str = match_specific_info.groups()
            kanonischer_typ = INFO_TYPE_ALIASES.get(info_type_str.lower().strip())
            if kanonischer_typ:
                return [
                    create_meaning_point(
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
        return None

    def _match_property_question(self, text: str) -> list[MeaningPoint] | None:
        """Erkennt Property-Fragen: "Was ist die Farbe von Apfel?"."""
        match_prop = re.match(
            r"^\s*(?:was ist|welche|welches)\s+(?:der|die|das)\s+(.+?)\s+von\s+(.+?)\??\s*$",
            text,
            re.IGNORECASE,
        )
        if match_prop:
            prop_name, topic_str = match_prop.groups()
            return [
                create_meaning_point(
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
        return None

    def _match_show_all_knowledge(self, text: str) -> list[MeaningPoint] | None:
        """Erkennt "Was weißt du über X alles?"."""
        match_show_all = re.match(
            r"^\s*(?:was weisst du (?:ueber|von|zu)\s+(.+?)\s+(?:alles|gesamt)|zeige?\s+alles\s+ueber\s+(.+?))\??\s*$",
            text,
            re.IGNORECASE,
        )
        if match_show_all:
            # Eine der beiden Gruppen ist nicht None
            topic_str_raw = match_show_all.group(1) or match_show_all.group(2)
            cleaned_topic = self.text_normalizer.clean_entity(topic_str_raw)
            return [
                create_meaning_point(
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
        return None

    def _match_episodic_queries(self, text: str) -> list[MeaningPoint] | None:
        """Erkennt episodische Gedächtnis-Queries."""
        # 3.1: "Wann habe ich X gelernt?"
        match_episodic_when_learned = re.match(
            r"^\s*wann\s+(?:habe ich|hab ich|wurde|haben wir)\s+(?:ueber\s+)?(.+?)\s+gelernt\??\s*$",
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
                create_meaning_point(
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

        # 3.2: "Zeige (mir) Episoden"
        match_episodic_show_episodes = re.match(
            r"^\s*zeige?\s+(?:mir\s+)?(?:alle\s+)?episoden\s*(?:ueber|von|zu|fuer)?\s*(.*?)\??\s*$",
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
                create_meaning_point(
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

        # 3.3: "Zeige Lernverlauf"
        match_episodic_learning_history = re.match(
            r"^\s*zeige?\s+(?:mir\s+)?(?:den\s+)?lernverlauf\s*(?:von|fuer|ueber)?\s*(.*?)\??\s*$",
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
                create_meaning_point(
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

        # 3.4: "Was habe ich über X gelernt?"
        match_episodic_what_learned = re.match(
            r"^\s*was\s+(?:habe ich|hab ich|haben wir)\s+ueber\s+(.+?)\s+gelernt\??\s*$",
            text,
            re.IGNORECASE,
        )
        if match_episodic_what_learned:
            topic_str_raw = match_episodic_what_learned.group(1).strip()
            cleaned_topic = self.text_normalizer.clean_entity(topic_str_raw)
            return [
                create_meaning_point(
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

        return None

    def _match_spatial_queries(self, text: str) -> list[MeaningPoint] | None:
        """Erkennt räumliche Queries (Grid, Relationen, Pfade, Positionen)."""
        # 3.5: Grid-basiert
        match_grid_query = re.match(
            r"^\s*(?:wo ist|zeige mir)\s+(?:das\s+)?(?:feld|position|stelle)\s+(.+?)\s+(?:auf|in|im)\s+(?:einem?\s+)?(.+?)\??\s*$",
            text,
            re.IGNORECASE,
        )
        if match_grid_query:
            position_str, grid_type = match_grid_query.groups()
            return [
                create_meaning_point(
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

        # 3.6: Relationen
        match_spatial_relation = re.match(
            r"^\s*(?:ist|liegt)\s+(.+?)\s+(noerdlich von|suedlich von|oestlich von|westlich von|neben|zwischen|ueber|unter)\s+(.+?)\??\s*$",
            text,
            re.IGNORECASE,
        )
        if match_spatial_relation:
            subject, relation, target = match_spatial_relation.groups()
            relation_map = {
                "noerdlich von": "NORTH_OF",
                "suedlich von": "SOUTH_OF",
                "oestlich von": "EAST_OF",
                "westlich von": "WEST_OF",
                "neben": "ADJACENT_TO",
                "zwischen": "BETWEEN",
                "ueber": "ABOVE",
                "unter": "BELOW",
            }
            relation_type = relation_map.get(relation.lower(), "ADJACENT_TO")

            return [
                create_meaning_point(
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

        # 3.7: Path-Finding
        match_path_finding = re.match(
            r"^\s*wie\s+(?:komme ich|kommt man|gelange ich)\s+von\s+(.+?)\s+(?:nach|zu)\s+(.+?)\??\s*$",
            text,
            re.IGNORECASE,
        )
        if match_path_finding:
            start, goal = match_path_finding.groups()
            return [
                create_meaning_point(
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

        # 3.8: Position-basiert
        match_position_query = re.match(
            r"^\s*(?:wo liegt|wo befindet sich|wo ist)\s+(.+?)\??\s*$",
            text,
            re.IGNORECASE,
        )
        if match_position_query:
            object_name = match_position_query.group(1).strip()
            cleaned_object = self.text_normalizer.clean_entity(object_name)
            return [
                create_meaning_point(
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

        return None

    def _match_what_is_question(self, text: str) -> list[MeaningPoint] | None:
        """Erkennt "Was ist X?" Fragen (häufigste Form)."""
        match_what_is = re.match(
            r"^\s*was\s+ist\s+(.+?)\??\s*$",
            text,
            re.IGNORECASE,
        )
        if match_what_is:
            topic_str_raw = match_what_is.group(1).strip()
            cleaned_topic = self.text_normalizer.clean_entity(topic_str_raw)
            # FIX: Höhere Confidence wenn Fragezeichen vorhanden
            confidence = 0.90 if text.rstrip().endswith("?") else 0.80
            return [
                create_meaning_point(
                    category=MeaningPointCategory.QUESTION,
                    cue="heuristic_question_wh",
                    text_span=text,
                    confidence=confidence,
                    arguments={"topic": cleaned_topic},
                )
            ]
        return None

    def _match_was_macht_question(self, text: str) -> list[MeaningPoint] | None:
        """
        Erkennt 'Was macht X?' Fragen (action-based questions).

        Muster: 'Was macht [NOUN]?' oder 'Was tut [NOUN]?'
        Beispiele:
          - 'Was macht ein Hund?' -> (0.90, 'hund', 'action')
          - 'Was tut der Vogel?' -> (0.90, 'vogel', 'action')
          - 'Was kann X machen?' -> (0.92, 'X', 'capability')

        Returns:
            Liste mit MeaningPoint oder None
        """
        # Pattern 1: "Was macht X?"
        match_macht = re.match(
            r"^\s*was\s+macht\s+(?:ein|eine|der|die|das|den|dem|des)?\s*(.+?)\??\s*$",
            text,
            re.IGNORECASE,
        )
        if match_macht:
            topic_str_raw = match_macht.group(1).strip()
            cleaned_topic = self.text_normalizer.clean_entity(topic_str_raw)
            confidence = 0.90 if text.rstrip().endswith("?") else 0.85
            return [
                create_meaning_point(
                    category=MeaningPointCategory.QUESTION,
                    cue="heuristic_was_macht_question",
                    text_span=text,
                    confidence=confidence,
                    arguments={
                        "topic": cleaned_topic,
                        "question_type": "action",
                        "relation_hint": "CAPABLE_OF",
                    },
                )
            ]

        # Pattern 2: "Was tut X?"
        match_tut = re.match(
            r"^\s*was\s+tut\s+(?:ein|eine|der|die|das|den|dem|des)?\s*(.+?)\??\s*$",
            text,
            re.IGNORECASE,
        )
        if match_tut:
            topic_str_raw = match_tut.group(1).strip()
            cleaned_topic = self.text_normalizer.clean_entity(topic_str_raw)
            confidence = 0.90 if text.rstrip().endswith("?") else 0.85
            return [
                create_meaning_point(
                    category=MeaningPointCategory.QUESTION,
                    cue="heuristic_was_tut_question",
                    text_span=text,
                    confidence=confidence,
                    arguments={
                        "topic": cleaned_topic,
                        "question_type": "action",
                        "relation_hint": "CAPABLE_OF",
                    },
                )
            ]

        # Pattern 3: "Was mag X?" / "Was liebt X?" / "Was moechte X?" (preference questions)
        match_preference = re.match(
            r"^\s*was\s+(?:mag|liebt|moechte|bevorzugt)\s+(?:ein|eine|der|die|das|den|dem|des)?\s*(.+?)\??\s*$",
            text,
            re.IGNORECASE,
        )
        if match_preference:
            topic_str_raw = match_preference.group(1).strip()
            cleaned_topic = self.text_normalizer.clean_entity(topic_str_raw)
            confidence = 0.90 if text.rstrip().endswith("?") else 0.85
            return [
                create_meaning_point(
                    category=MeaningPointCategory.QUESTION,
                    cue="heuristic_was_mag_question",
                    text_span=text,
                    confidence=confidence,
                    arguments={
                        "topic": cleaned_topic,
                        "question_type": "preference",
                        "relation_hint": "ASSOCIATED_WITH",
                    },
                )
            ]

        # Pattern 3: "Was kann X machen?" / "Was kann X tun?"
        match_kann_machen = re.match(
            r"^\s*was\s+kann\s+(?:ein|eine|der|die|das|den|dem|des)?\s*(.+?)\s+(?:machen|tun)\??\s*$",
            text,
            re.IGNORECASE,
        )
        if match_kann_machen:
            topic_str_raw = match_kann_machen.group(1).strip()
            cleaned_topic = self.text_normalizer.clean_entity(topic_str_raw)
            return [
                create_meaning_point(
                    category=MeaningPointCategory.QUESTION,
                    cue="heuristic_was_kann_machen_question",
                    text_span=text,
                    confidence=0.92,  # Very explicit pattern
                    arguments={
                        "topic": cleaned_topic,
                        "question_type": "capability",
                        "relation_hint": "CAPABLE_OF",
                    },
                )
            ]

        return None

    def _match_where_is_question(self, text: str) -> list[MeaningPoint] | None:
        """Erkennt "Wo ist/liegt X?" Fragen."""
        match_where_is = re.match(
            r"^\s*wo\s+(?:ist|liegt)\s+(.+?)\??\s*$",
            text,
            re.IGNORECASE,
        )
        if match_where_is:
            topic_str = match_where_is.group(1).strip()
            cleaned_topic = self.text_normalizer.clean_entity(topic_str)
            return [
                create_meaning_point(
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
        return None

    def _match_wer_question(self, text: str) -> list[MeaningPoint] | None:
        """Erkennt Wer-Fragen (nach Personen/Akteuren)."""
        match_wer = re.match(
            r"^\s*wer\s+(.+?)\??\s*$",
            text,
            re.IGNORECASE,
        )
        if match_wer:
            rest_of_question = match_wer.group(1).strip()

            # ALLGEMEINE LÖSUNG: Extrahiere das Objekt als Topic (NICHT Verb + Objekt!)
            # Beispiel: "Wer von den dreien trinkt gerne einen Brandy?"
            # -> Extrahiere: "brandy" als Topic (das gesuchte Objekt)
            # Für WER-Fragen suchen wir: Welche Person ist mit diesem Objekt assoziiert?

            # Strategie: Nimm das letzte substantivische Nomen nach Filterung
            # Das ist typischerweise das Objekt der Frage
            words = rest_of_question.split()

            # Filtere Stopp-Wörter (Artikel, Präpositionen, Adverbien, Füllwörter, häufige Verben)
            filtered_words = [
                w
                for w in words
                if w.lower()
                not in [
                    # Artikel & Pronomen
                    "von",
                    "den",
                    "der",
                    "die",
                    "das",
                    "dem",
                    "einen",
                    "eine",
                    "ein",
                    "dreien",
                    "beiden",
                    "allen",
                    "diesen",
                    "jenen",
                    # Adverbien & Füllwörter
                    "gerne",
                    "gern",
                    "oft",
                    "immer",
                    "manchmal",
                    "also",
                    "denn",
                    # Präpositionen
                    "aus",
                    "bei",
                    "zu",
                    "in",
                    "an",
                    "auf",
                    "über",
                    "unter",
                    "mit",
                    # Häufige Verben (sollten ignoriert werden, da wir das Objekt suchen)
                    "ist",
                    "sind",
                    "hat",
                    "haben",
                    "kann",
                    "können",
                    "trinkt",
                    "trinken",
                    "isst",
                    "essen",
                    "mag",
                    "mögen",
                    "benutzt",
                    "benutzen",
                    "verwendet",
                    "verwenden",
                    "macht",
                    "machen",
                ]
            ]

            if filtered_words:
                # Nimm das LETZTE Wort als Objekt (typisch für deutsche Fragen)
                extracted_topic = filtered_words[-1].lower()
            else:
                # Fallback: Ganzer Rest
                extracted_topic = rest_of_question

            cleaned_topic = self.text_normalizer.clean_entity(extracted_topic)

            # FIX: Höhere Confidence wenn Fragezeichen vorhanden
            confidence = 0.90 if text.rstrip().endswith("?") else 0.80

            return [
                create_meaning_point(
                    category=MeaningPointCategory.QUESTION,
                    cue="heuristic_question_wer",
                    text_span=text,
                    confidence=confidence,
                    arguments={
                        "topic": cleaned_topic,
                        "question_word": "wer",
                        "question_type": "person_query",  # Marker für Personen-Frage
                        "full_question": rest_of_question,
                    },
                )
            ]
        return None

    def _match_wie_question(self, text: str) -> list[MeaningPoint] | None:
        """Erkennt Wie-Fragen (nach Prozessen/Methoden)."""
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

            # FIX: Höhere Confidence wenn Fragezeichen vorhanden
            confidence = 0.90 if text.rstrip().endswith("?") else 0.80

            return [
                create_meaning_point(
                    category=MeaningPointCategory.QUESTION,
                    cue="heuristic_question_wie",
                    text_span=text,
                    confidence=confidence,
                    arguments={
                        "topic": cleaned_topic,
                        "question_word": "wie",
                        "question_type": "process_query",  # Marker für Prozess-Frage
                        "full_question": rest_of_question,
                    },
                )
            ]
        return None

    def _match_warum_question(self, text: str) -> list[MeaningPoint] | None:
        """Erkennt Warum/Wieso/Weshalb-Fragen (nach Gründen/Ursachen)."""
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
                r"(?:ist|sind|hat|haben|kann|koennen)\s+(.+?)(?:\s|$)",
                rest_of_question,
                re.IGNORECASE,
            )
            if topic_match:
                topic_str = topic_match.group(1)
            else:
                topic_str = rest_of_question

            cleaned_topic = self.text_normalizer.clean_entity(topic_str)

            # FIX: Höhere Confidence wenn Fragezeichen vorhanden
            confidence = 0.90 if text.rstrip().endswith("?") else 0.80

            return [
                create_meaning_point(
                    category=MeaningPointCategory.QUESTION,
                    cue=f"heuristic_question_{question_word}",
                    text_span=text,
                    confidence=confidence,
                    arguments={
                        "topic": cleaned_topic,
                        "question_word": question_word,
                        "question_type": "reason_query",  # Marker für Grund-Frage
                        "full_question": rest_of_question,
                    },
                )
            ]
        return None

    def _match_wann_question(self, text: str) -> list[MeaningPoint] | None:
        """Erkennt Wann-Fragen (nach Zeit/Zeitpunkten)."""
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

            # FIX: Höhere Confidence wenn Fragezeichen vorhanden
            confidence = 0.90 if text.rstrip().endswith("?") else 0.80

            return [
                create_meaning_point(
                    category=MeaningPointCategory.QUESTION,
                    cue="heuristic_question_wann",
                    text_span=text,
                    confidence=confidence,
                    arguments={
                        "topic": cleaned_topic,
                        "question_word": "wann",
                        "question_type": "time_query",  # Marker für Zeit-Frage
                        "full_question": rest_of_question,
                    },
                )
            ]
        return None

    def _match_flexible_wh_question(self, text: str) -> list[MeaningPoint] | None:
        """Erkennt flexible W-Fragen (Fallback)."""
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
                r"(?:ist|sind|hat|haben|kann|koennen)\s+(.+?)(?:\s|$)",
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
                create_meaning_point(
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
        return None
