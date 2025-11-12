"""
Zahl-Wort-Konvertierung für KAI (Deutsch)
Unterstützt bidirektionale Konvertierung: Zahl ↔ Wort
Bereich: 0-999 (erweiterbar auf größere Zahlen)
"""

import re
from typing import Dict, List, Optional

from component_1_netzwerk_core import KonzeptNetzwerkCore


class NumberParser:
    """Konvertiert deutsche Zahlwörter zu Zahlen"""

    # Basis-Mapping (0-20)
    BASIC_NUMBERS = {
        "null": 0,
        "eins": 1,
        "ein": 1,
        "zwei": 2,
        "drei": 3,
        "vier": 4,
        "fünf": 5,
        "sechs": 6,
        "sieben": 7,
        "acht": 8,
        "neun": 9,
        "zehn": 10,
        "elf": 11,
        "zwölf": 12,
        "dreizehn": 13,
        "vierzehn": 14,
        "fünfzehn": 15,
        "sechzehn": 16,
        "siebzehn": 17,
        "achtzehn": 18,
        "neunzehn": 19,
        "zwanzig": 20,
    }

    # Zehner (20-90)
    TENS = {
        "zwanzig": 20,
        "dreißig": 30,
        "vierzig": 40,
        "fünfzig": 50,
        "sechzig": 60,
        "siebzig": 70,
        "achtzig": 80,
        "neunzig": 90,
    }

    # Größere Einheiten
    MAGNITUDES = {
        "hundert": 100,
        "tausend": 1000,
        "million": 1000000,
        "millionen": 1000000,
        "milliarde": 1000000000,
        "milliarden": 1000000000,
    }

    def __init__(self, netzwerk: Optional[KonzeptNetzwerkCore] = None):
        self.netzwerk = netzwerk
        self.learned_numbers: Dict[str, int] = {}
        self._load_learned_numbers()

    def _load_learned_numbers(self):
        """Lädt gelernte Zahlen aus Neo4j"""
        if not self.netzwerk:
            return

        try:
            # Query für EQUIVALENT_TO Relationen mit Zahlen
            query = """
            MATCH (w:Wort)-[:EQUIVALENT_TO]->(n:NumberNode)
            WHERE n.value IS NOT NULL
            RETURN w.lemma AS word, n.value AS value
            """
            with self.netzwerk.driver.session() as session:
                result = session.run(query)
                for record in result:
                    word = record["word"]
                    value = record["value"]
                    if isinstance(value, (int, float)):
                        self.learned_numbers[word] = int(value)
        except Exception:
            # Fehler beim Laden ignorieren (DB kann leer sein)
            pass

    def parse(self, word: str) -> Optional[int]:
        """
        Konvertiert deutsches Zahlwort zu Zahl

        Beispiele:
            "drei" → 3
            "einundzwanzig" → 21
            "zweihundertfünfundvierzig" → 245
            "dreitausendzweihundert" → 3200

        Args:
            word: Deutsches Zahlwort (lowercase)

        Returns:
            Integer-Zahl oder None bei Fehler
        """
        word = word.lower().strip()

        # Leerzeichen entfernen (für Eingaben wie "drei hundert")
        word = word.replace(" ", "")

        # Direkt-Lookup in Basis-Zahlen
        if word in self.BASIC_NUMBERS:
            return self.BASIC_NUMBERS[word]

        # Lookup in gelernten Zahlen
        if word in self.learned_numbers:
            return self.learned_numbers[word]

        # Komplexe Zahlen parsen
        return self._parse_complex(word)

    def _parse_complex(self, word: str) -> Optional[int]:
        """Parst zusammengesetzte Zahlen"""
        # Negative Zahlen
        if word.startswith("minus"):
            rest = word[5:].strip()
            parsed = self.parse(rest)
            return -parsed if parsed is not None else None

        # Strategie: Von groß nach klein parsen (tausend → hundert → zehner)
        total = 0

        # 1. Tausender extrahieren (dreitausendzweihundert)
        if "tausend" in word:
            parts = word.split("tausend", 1)
            thousands_word = parts[0]
            rest = parts[1] if len(parts) > 1 else ""

            # Parse Tausender-Teil (kann leer sein für "eintausend")
            if thousands_word == "":
                thousands = 1
            else:
                thousands = self._parse_hundreds(thousands_word)
                if thousands is None:
                    return None

            total += thousands * 1000

            # Parse Rest
            if rest:
                rest_value = self._parse_hundreds(rest)
                if rest_value is None:
                    return None
                total += rest_value

            return total

        # 2. Nur Hunderter (ohne Tausender)
        return self._parse_hundreds(word)

    def _parse_hundreds(self, word: str) -> Optional[int]:
        """Parst Zahlen 0-999 (ohne Tausender)"""
        if not word:
            return None

        # Direkt-Lookup
        if word in self.BASIC_NUMBERS:
            return self.BASIC_NUMBERS[word]

        total = 0

        # Hunderter extrahieren (zweihundertfünfundvierzig)
        if "hundert" in word:
            parts = word.split("hundert", 1)
            hundreds_word = parts[0]
            rest = parts[1] if len(parts) > 1 else ""

            # Parse Hunderter-Teil
            if hundreds_word == "":
                hundreds = 1  # "einhundert"
            else:
                hundreds = self.BASIC_NUMBERS.get(hundreds_word)
                if hundreds is None:
                    return None

            total += hundreds * 100

            # Parse Rest (0-99)
            if rest:
                rest_value = self._parse_tens(rest)
                if rest_value is None:
                    return None
                total += rest_value

            return total

        # Nur Zehner (21-99)
        return self._parse_tens(word)

    def _parse_tens(self, word: str) -> Optional[int]:
        """Parst Zahlen 0-99 (ohne Hunderter)"""
        if not word:
            return None

        # Direkt-Lookup (0-20)
        if word in self.BASIC_NUMBERS:
            return self.BASIC_NUMBERS[word]

        # Reine Zehner (20, 30, ..., 90)
        if word in self.TENS:
            return self.TENS[word]

        # Zusammengesetzte Zehner (einundzwanzig, zweiunddreißig)
        if "und" in word:
            parts = word.split("und")
            if len(parts) == 2:
                ones_word = parts[0]
                tens_word = parts[1]

                ones = self.BASIC_NUMBERS.get(ones_word)
                tens = self.TENS.get(tens_word)

                if ones is not None and tens is not None:
                    return tens + ones

        return None


class NumberFormatter:
    """Konvertiert Zahlen zu deutschen Wörtern"""

    # Basis-Wörter (0-20)
    BASIC_WORDS = {
        0: "null",
        1: "eins",
        2: "zwei",
        3: "drei",
        4: "vier",
        5: "fünf",
        6: "sechs",
        7: "sieben",
        8: "acht",
        9: "neun",
        10: "zehn",
        11: "elf",
        12: "zwölf",
        13: "dreizehn",
        14: "vierzehn",
        15: "fünfzehn",
        16: "sechzehn",
        17: "siebzehn",
        18: "achtzehn",
        19: "neunzehn",
        20: "zwanzig",
    }

    # Zehner-Wörter
    TENS_WORDS = {
        20: "zwanzig",
        30: "dreißig",
        40: "vierzig",
        50: "fünfzig",
        60: "sechzig",
        70: "siebzig",
        80: "achtzig",
        90: "neunzig",
    }

    def format(self, number: int) -> str:
        """
        Konvertiert Zahl zu deutschem Wort

        Beispiele:
            3 → "drei"
            21 → "einundzwanzig"
            245 → "zweihundertfünfundvierzig"
            3200 → "dreitausendzweihundert"

        Args:
            number: Integer-Zahl

        Returns:
            Deutsches Zahlwort
        """
        if number == 0:
            return "null"

        if number < 0:
            return "minus" + self.format(abs(number))

        if number <= 20:
            return self.BASIC_WORDS[number]

        if number < 100:
            return self._format_tens(number)

        if number < 1000:
            return self._format_hundreds(number)

        if number < 1000000:
            return self._format_thousands(number)

        # Fallback für sehr große Zahlen
        return str(number)

    def _format_tens(self, number: int) -> str:
        """Formatiert Zehner (21-99)"""
        tens = (number // 10) * 10
        ones = number % 10

        tens_word = self.TENS_WORDS[tens]

        if ones == 0:
            return tens_word
        else:
            # "eins" wird zu "ein" in Komposita
            ones_word = "ein" if ones == 1 else self.BASIC_WORDS[ones]
            return f"{ones_word}und{tens_word}"

    def _format_hundreds(self, number: int) -> str:
        """Formatiert Hunderter (100-999)"""
        hundreds = number // 100
        rest = number % 100

        # "eins" wird zu "ein" bei "einhundert"
        if hundreds == 1:
            result = "einhundert"
        else:
            result = self.BASIC_WORDS[hundreds] + "hundert"

        if rest > 0:
            if rest <= 20:
                # Bei standalone 1 am Ende bleibt "eins", sonst normal
                result += self.BASIC_WORDS[rest]
            else:
                result += self._format_tens(rest)

        return result

    def _format_thousands(self, number: int) -> str:
        """Formatiert Tausender (1000-999999)"""
        thousands = number // 1000
        rest = number % 1000

        # Tausender-Teil formatieren
        if thousands == 1:
            result = "eintausend"
        elif thousands < 100:
            if thousands <= 20:
                thousands_word = (
                    "ein" if thousands == 1 else self.BASIC_WORDS[thousands]
                )
            else:
                thousands_word = self._format_tens(thousands)
            result = thousands_word + "tausend"
        else:
            # 100-999 Tausender
            result = self._format_hundreds(thousands) + "tausend"

        # Rest formatieren
        if rest > 0:
            result += self._format_hundreds(rest)

        return result


class ArithmeticConceptConnector:
    """
    Verbindet arithmetische Konzepte (Summe, Produkt, etc.) mit Operationen
    Erstellt EQUIVALENT_TO Relationen zwischen Konzept-Wörtern und Operationen
    """

    # Mapping: Konzeptwort → (Operationsname, Symbol)
    ARITHMETIC_CONCEPTS = {
        "summe": ("addition", "+"),
        "differenz": ("subtraction", "-"),
        "produkt": ("multiplication", "*"),
        "quotient": ("division", "/"),
        # Erweiterte Synonyme
        "ergebnis der addition": ("addition", "+"),
        "ergebnis der subtraktion": ("subtraction", "-"),
        "ergebnis der multiplikation": ("multiplication", "*"),
        "ergebnis der division": ("division", "/"),
    }

    # Reverse Mapping: Operation → Konzeptwort
    OPERATION_TO_CONCEPT = {
        "addition": "summe",
        "subtraction": "differenz",
        "multiplication": "produkt",
        "division": "quotient",
    }

    def __init__(self, netzwerk: KonzeptNetzwerkCore):
        self.netzwerk = netzwerk

    def learn_concept(self, concept_word: str, operation: str, symbol: str) -> bool:
        """
        Lernt arithmetisches Konzept und speichert in Neo4j

        Erstellt:
        (:Konzept {name: "Summe"})-[:EQUIVALENT_TO]->(:Operation {name: "Addition", symbol: "+"})

        Args:
            concept_word: Konzeptwort (z.B. "summe")
            operation: Operationsname (z.B. "addition")
            symbol: Operations-Symbol (z.B. "+")

        Returns:
            True bei Erfolg
        """
        try:
            concept_lemma = concept_word.lower().strip()
            operation_name = operation.lower().strip()

            query = """
            MERGE (c:Konzept {name: $concept})
            ON CREATE SET c.type = 'arithmetic_concept'
            MERGE (o:Operation {name: $operation, symbol: $symbol})
            ON CREATE SET o.type = 'arithmetic_operation'
            MERGE (c)-[r:EQUIVALENT_TO]->(o)
            ON CREATE SET r.confidence = 1.0, r.source = 'arithmetic_concepts'
            ON MATCH SET r.confidence = 1.0, r.source = 'arithmetic_concepts'
            RETURN c, o
            """

            with self.netzwerk.driver.session() as session:
                result = session.run(
                    query,
                    concept=concept_lemma,
                    operation=operation_name,
                    symbol=symbol,
                )
                record = result.single()
                return record is not None

        except Exception as e:
            print(f"Fehler beim Lernen des Konzepts: {e}")
            return False

    def query_operation(self, concept_word: str) -> Optional[Dict[str, str]]:
        """
        Fragt Operation für Konzeptwort aus Neo4j ab

        Args:
            concept_word: Konzeptwort (z.B. "summe")

        Returns:
            Dict mit "operation" und "symbol" oder None
        """
        concept_lemma = concept_word.lower().strip()

        try:
            query = """
            MATCH (c:Konzept {name: $concept})-[:EQUIVALENT_TO]->(o:Operation)
            RETURN o.name AS operation, o.symbol AS symbol
            """

            with self.netzwerk.driver.session() as session:
                result = session.run(query, concept=concept_lemma)
                record = result.single()
                if record:
                    return {
                        "operation": record["operation"],
                        "symbol": record["symbol"],
                    }

        except Exception:
            pass

        return None

    def query_concept(self, operation: str) -> Optional[str]:
        """
        Fragt Konzeptwort für Operation aus Neo4j ab

        Args:
            operation: Operationsname (z.B. "addition")

        Returns:
            Konzeptwort oder None
        """
        operation_name = operation.lower().strip()

        try:
            query = """
            MATCH (o:Operation {name: $operation})<-[:EQUIVALENT_TO]-(c:Konzept)
            RETURN c.name AS concept
            """

            with self.netzwerk.driver.session() as session:
                result = session.run(query, operation=operation_name)
                record = result.single()
                if record:
                    return record["concept"]

        except Exception:
            pass

        return None

    def extract_concept_from_text(self, text: str) -> Optional[Dict[str, str]]:
        """
        Extrahiert arithmetisches Konzept aus Text

        Args:
            text: Eingabetext (z.B. "Was ist die Summe von drei und fünf?")

        Returns:
            Dict mit "operation" und "symbol" oder None
        """
        text_lower = text.lower()

        # Prüfe alle bekannten Konzepte
        for concept, (operation, symbol) in self.ARITHMETIC_CONCEPTS.items():
            if concept in text_lower:
                return {"operation": operation, "symbol": symbol, "concept": concept}

        # Fallback: Query aus Neo4j (für gelernte Konzepte)
        words = text_lower.split()
        for word in words:
            result = self.query_operation(word)
            if result:
                result["concept"] = word
                return result

        return None

    def initialize_basic_concepts(self) -> int:
        """
        Initialisiert Basis-Arithmetik-Konzepte in Neo4j

        Returns:
            Anzahl der erfolgreich gespeicherten Konzepte
        """
        success_count = 0

        for concept, (operation, symbol) in self.ARITHMETIC_CONCEPTS.items():
            if self.learn_concept(concept, operation, symbol):
                success_count += 1

        return success_count


class NumberLanguageConnector:
    """
    Verbindet Zahl-Wort-System mit Neo4j
    Erstellt EQUIVALENT_TO Relationen zwischen Wörtern und Zahlen
    """

    def __init__(self, netzwerk: KonzeptNetzwerkCore):
        self.netzwerk = netzwerk
        self.parser = NumberParser(netzwerk)
        self.formatter = NumberFormatter()
        self.arithmetic_concepts = ArithmeticConceptConnector(netzwerk)

    def learn_number(self, word: str, value: int) -> bool:
        """
        Lernt Zahl-Wort-Zuordnung und speichert in Neo4j

        Args:
            word: Deutsches Zahlwort (z.B. "drei")
            value: Zahlenwert (z.B. 3)

        Returns:
            True bei Erfolg
        """
        try:
            # Erstelle Wort-Node und Number-Node mit EQUIVALENT_TO Relation
            word_lemma = word.lower().strip()

            query = """
            MERGE (w:Wort {lemma: $word})
            ON CREATE SET w.pos = 'NUM'
            ON MATCH SET w.pos = 'NUM'
            MERGE (n:NumberNode {value: $value})
            ON CREATE SET n.word = $word
            MERGE (w)-[r:EQUIVALENT_TO]->(n)
            ON CREATE SET r.confidence = 1.0, r.source = 'number_language'
            ON MATCH SET r.confidence = 1.0, r.source = 'number_language'
            RETURN w, n
            """

            with self.netzwerk.driver.session() as session:
                result = session.run(query, word=word_lemma, value=value)
                record = result.single()
                return record is not None

        except Exception:
            # Fehler beim Lernen loggen
            return False

    def query_number(self, word: str) -> Optional[int]:
        """
        Fragt Zahlenwert für Wort aus Neo4j ab

        Args:
            word: Deutsches Zahlwort

        Returns:
            Zahlenwert oder None
        """
        word_lemma = word.lower().strip()

        try:
            query = """
            MATCH (w:Wort {lemma: $word})-[:EQUIVALENT_TO]->(n:NumberNode)
            RETURN n.value AS value
            """

            with self.netzwerk.driver.session() as session:
                result = session.run(query, word=word_lemma)
                record = result.single()
                if record:
                    return int(record["value"])

        except Exception:
            pass

        return None

    def query_word(self, value: int) -> Optional[str]:
        """
        Fragt Wort für Zahlenwert aus Neo4j ab

        Args:
            value: Zahlenwert

        Returns:
            Deutsches Zahlwort oder None
        """
        try:
            query = """
            MATCH (n:NumberNode {value: $value})<-[:EQUIVALENT_TO]-(w:Wort)
            RETURN w.lemma AS word
            """

            with self.netzwerk.driver.session() as session:
                result = session.run(query, value=value)
                record = result.single()
                if record:
                    return record["word"]

        except Exception:
            pass

        return None

    def initialize_basic_numbers(self, count: int = 100) -> int:
        """
        Initialisiert Basis-Zahlen in Neo4j (0 bis count)

        Args:
            count: Anzahl der zu initialisierenden Zahlen

        Returns:
            Anzahl der erfolgreich gespeicherten Zahlen
        """
        success_count = 0

        for i in range(count + 1):
            word = self.formatter.format(i)
            if self.learn_number(word, i):
                success_count += 1

        return success_count


class ArithmeticQuestionParser:
    """
    Parst arithmetische Fragen und extrahiert Operationen und Operanden

    Beispiele:
    - "Was ist die Summe von drei und fünf?" → ("addition", [3, 5])
    - "Was ist das Produkt von vier und sieben?" → ("multiplication", [4, 7])
    - "Was ist die Differenz zwischen zehn und drei?" → ("subtraction", [10, 3])
    """

    def __init__(
        self,
        number_parser: NumberParser,
        arithmetic_concepts: ArithmeticConceptConnector,
    ):
        self.number_parser = number_parser
        self.arithmetic_concepts = arithmetic_concepts

    def parse_question(self, text: str) -> Optional[Dict[str, any]]:
        """
        Parst arithmetische Frage und extrahiert alle Komponenten

        Args:
            text: Frage-Text (z.B. "Was ist die Summe von drei und fünf?")

        Returns:
            Dict mit "operation", "symbol", "operands", "concept" oder None
        """
        # Extrahiere Konzept (Summe, Produkt, etc.)
        concept_info = self.arithmetic_concepts.extract_concept_from_text(text)
        if not concept_info:
            return None

        # Extrahiere Operanden (Zahlen)
        operands = self._extract_operands(text)
        if not operands or len(operands) < 2:
            return None

        return {
            "operation": concept_info["operation"],
            "symbol": concept_info["symbol"],
            "concept": concept_info["concept"],
            "operands": operands,
        }

    def _extract_operands(self, text: str) -> List[int]:
        """
        Extrahiert Zahlen aus Text

        Args:
            text: Text mit Zahlwörtern (z.B. "drei und fünf")

        Returns:
            Liste von Integer-Zahlen
        """
        operands = []
        text_lower = text.lower()

        # Entferne Fragewörter und Konzeptwörter
        noise_words = [
            "was",
            "ist",
            "die",
            "der",
            "das",
            "von",
            "und",
            "zwischen",
            "aus",
            "?",
            "summe",
            "differenz",
            "produkt",
            "quotient",
        ]

        # Tokenisiere
        words = re.findall(r"\w+", text_lower)

        for word in words:
            if word in noise_words:
                continue

            # Versuche als Zahl zu parsen
            number = self.number_parser.parse(word)
            if number is not None:
                operands.append(number)

        return operands

    def format_answer(
        self, concept: str, operands: List[int], result: any, symbol: str
    ) -> str:
        """
        Formatiert Antwort für arithmetische Frage

        Args:
            concept: Konzeptwort (z.B. "summe")
            operands: Liste der Operanden
            result: Ergebnis der Operation
            symbol: Operations-Symbol

        Returns:
            Formatierte Antwort
        """
        # Konvertiere Zahlen zu Wörtern
        formatter = NumberFormatter()
        operands_words = [formatter.format(op) for op in operands]
        result_word = (
            formatter.format(result) if isinstance(result, int) else str(result)
        )

        # Formuliere Antwort
        operands_str = " und ".join(operands_words)
        concept_capitalized = concept.capitalize()

        # Operation als Zeichenkette
        operation_str = " ".join([str(op) for op in operands]).replace(
            " ", f" {symbol} "
        )

        return (
            f"Die {concept_capitalized} von {operands_str} ist {result_word}. "
            f"({operation_str} = {result})"
        )

    def is_arithmetic_question(self, text: str) -> bool:
        """
        Prüft ob Text eine arithmetische Frage ist

        Args:
            text: Eingabetext

        Returns:
            True wenn arithmetische Frage
        """
        return self.arithmetic_concepts.extract_concept_from_text(text) is not None
