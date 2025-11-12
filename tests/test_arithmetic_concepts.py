"""
Tests für arithmetische Konzepte (Summe, Produkt, Differenz, Quotient)
"""

import pytest

from component_1_netzwerk_core import KonzeptNetzwerkCore
from component_53_number_language import (
    ArithmeticConceptConnector,
    ArithmeticQuestionParser,
    NumberParser,
)


@pytest.fixture
def netzwerk():
    """Test-Netzwerk"""
    return KonzeptNetzwerkCore(
        uri="bolt://127.0.0.1:7687", user="neo4j", password="password"
    )


@pytest.fixture
def concept_connector(netzwerk):
    """ArithmeticConceptConnector Fixture"""
    connector = ArithmeticConceptConnector(netzwerk)
    # Initialisiere Basis-Konzepte
    connector.initialize_basic_concepts()
    return connector


@pytest.fixture
def number_parser(netzwerk):
    """NumberParser Fixture"""
    return NumberParser(netzwerk)


@pytest.fixture
def question_parser(number_parser, concept_connector):
    """ArithmeticQuestionParser Fixture"""
    return ArithmeticQuestionParser(number_parser, concept_connector)


class TestArithmeticConceptConnector:
    """Tests für ArithmeticConceptConnector"""

    def test_learn_concept_summe(self, concept_connector):
        """Test: Lerne 'Summe' als Konzept"""
        result = concept_connector.learn_concept("summe", "addition", "+")
        assert result is True

    def test_learn_concept_differenz(self, concept_connector):
        """Test: Lerne 'Differenz' als Konzept"""
        result = concept_connector.learn_concept("differenz", "subtraction", "-")
        assert result is True

    def test_learn_concept_produkt(self, concept_connector):
        """Test: Lerne 'Produkt' als Konzept"""
        result = concept_connector.learn_concept("produkt", "multiplication", "*")
        assert result is True

    def test_learn_concept_quotient(self, concept_connector):
        """Test: Lerne 'Quotient' als Konzept"""
        result = concept_connector.learn_concept("quotient", "division", "/")
        assert result is True

    def test_query_operation_summe(self, concept_connector):
        """Test: Query Operation für 'Summe'"""
        result = concept_connector.query_operation("summe")
        assert result is not None
        assert result["operation"] == "addition"
        assert result["symbol"] == "+"

    def test_query_operation_produkt(self, concept_connector):
        """Test: Query Operation für 'Produkt'"""
        result = concept_connector.query_operation("produkt")
        assert result is not None
        assert result["operation"] == "multiplication"
        assert result["symbol"] == "*"

    def test_query_concept_addition(self, concept_connector):
        """Test: Query Konzept für 'addition'"""
        result = concept_connector.query_concept("addition")
        assert result == "summe"

    def test_query_concept_multiplication(self, concept_connector):
        """Test: Query Konzept für 'multiplication'"""
        result = concept_connector.query_concept("multiplication")
        assert result == "produkt"

    def test_extract_concept_from_text_summe(self, concept_connector):
        """Test: Extrahiere 'Summe' aus Text"""
        result = concept_connector.extract_concept_from_text(
            "Was ist die Summe von drei und fünf?"
        )
        assert result is not None
        assert result["operation"] == "addition"
        assert result["symbol"] == "+"
        assert result["concept"] == "summe"

    def test_extract_concept_from_text_produkt(self, concept_connector):
        """Test: Extrahiere 'Produkt' aus Text"""
        result = concept_connector.extract_concept_from_text(
            "Was ist das Produkt von vier und sieben?"
        )
        assert result is not None
        assert result["operation"] == "multiplication"
        assert result["symbol"] == "*"

    def test_extract_concept_from_text_differenz(self, concept_connector):
        """Test: Extrahiere 'Differenz' aus Text"""
        result = concept_connector.extract_concept_from_text(
            "Was ist die Differenz zwischen zehn und drei?"
        )
        assert result is not None
        assert result["operation"] == "subtraction"
        assert result["symbol"] == "-"

    def test_extract_concept_from_text_quotient(self, concept_connector):
        """Test: Extrahiere 'Quotient' aus Text"""
        result = concept_connector.extract_concept_from_text(
            "Was ist der Quotient von zwölf und drei?"
        )
        assert result is not None
        assert result["operation"] == "division"
        assert result["symbol"] == "/"

    def test_extract_concept_from_text_no_concept(self, concept_connector):
        """Test: Kein Konzept im Text"""
        result = concept_connector.extract_concept_from_text("Wie heißt du?")
        assert result is None

    def test_initialize_basic_concepts(self, concept_connector):
        """Test: Initialisiere alle Basis-Konzepte"""
        count = concept_connector.initialize_basic_concepts()
        assert count >= 4  # Mindestens summe, differenz, produkt, quotient


class TestArithmeticQuestionParser:
    """Tests für ArithmeticQuestionParser"""

    def test_parse_question_summe(self, question_parser):
        """Test: Parse 'Was ist die Summe von drei und fünf?'"""
        result = question_parser.parse_question("Was ist die Summe von drei und fünf?")
        assert result is not None
        assert result["operation"] == "addition"
        assert result["symbol"] == "+"
        assert result["concept"] == "summe"
        assert result["operands"] == [3, 5]

    def test_parse_question_produkt(self, question_parser):
        """Test: Parse 'Was ist das Produkt von vier und sieben?'"""
        result = question_parser.parse_question(
            "Was ist das Produkt von vier und sieben?"
        )
        assert result is not None
        assert result["operation"] == "multiplication"
        assert result["operands"] == [4, 7]

    def test_parse_question_differenz(self, question_parser):
        """Test: Parse 'Was ist die Differenz zwischen zehn und drei?'"""
        result = question_parser.parse_question(
            "Was ist die Differenz zwischen zehn und drei?"
        )
        assert result is not None
        assert result["operation"] == "subtraction"
        assert result["operands"] == [10, 3]

    def test_parse_question_quotient(self, question_parser):
        """Test: Parse 'Was ist der Quotient von zwölf und drei?'"""
        result = question_parser.parse_question(
            "Was ist der Quotient von zwölf und drei?"
        )
        assert result is not None
        assert result["operation"] == "division"
        assert result["operands"] == [12, 3]

    def test_parse_question_no_concept(self, question_parser):
        """Test: Keine arithmetische Frage"""
        result = question_parser.parse_question("Wie heißt du?")
        assert result is None

    def test_parse_question_missing_operands(self, question_parser):
        """Test: Konzept aber fehlende Operanden"""
        result = question_parser.parse_question("Was ist die Summe?")
        assert result is None

    def test_parse_question_single_operand(self, question_parser):
        """Test: Nur ein Operand (zu wenige)"""
        result = question_parser.parse_question("Was ist die Summe von drei?")
        assert result is None

    def test_parse_question_multiple_operands(self, question_parser):
        """Test: Mehr als zwei Operanden"""
        result = question_parser.parse_question(
            "Was ist die Summe von drei und fünf und sieben?"
        )
        assert result is not None
        assert len(result["operands"]) == 3
        assert result["operands"] == [3, 5, 7]

    def test_extract_operands_simple(self, question_parser):
        """Test: Extrahiere Operanden aus einfachem Text"""
        operands = question_parser._extract_operands("drei und fünf")
        assert operands == [3, 5]

    def test_extract_operands_with_noise(self, question_parser):
        """Test: Extrahiere Operanden mit Störwörtern"""
        operands = question_parser._extract_operands("Was ist von drei und fünf?")
        assert operands == [3, 5]

    def test_extract_operands_larger_numbers(self, question_parser):
        """Test: Extrahiere größere Zahlen"""
        operands = question_parser._extract_operands("einundzwanzig und siebzehn")
        assert operands == [21, 17]

    def test_format_answer_summe(self, question_parser):
        """Test: Formatiere Antwort für Summe"""
        answer = question_parser.format_answer("summe", [3, 5], 8, "+")
        assert "Summe" in answer
        assert "drei" in answer
        assert "fünf" in answer
        assert "acht" in answer
        assert "3 + 5 = 8" in answer

    def test_format_answer_produkt(self, question_parser):
        """Test: Formatiere Antwort für Produkt"""
        answer = question_parser.format_answer("produkt", [4, 7], 28, "*")
        assert "Produkt" in answer
        assert "vier" in answer
        assert "sieben" in answer
        assert "4 * 7 = 28" in answer

    def test_format_answer_differenz(self, question_parser):
        """Test: Formatiere Antwort für Differenz"""
        answer = question_parser.format_answer("differenz", [10, 3], 7, "-")
        assert "Differenz" in answer
        assert "zehn" in answer
        assert "drei" in answer
        assert "sieben" in answer
        assert "10 - 3 = 7" in answer

    def test_format_answer_quotient(self, question_parser):
        """Test: Formatiere Antwort für Quotient"""
        answer = question_parser.format_answer("quotient", [12, 3], 4, "/")
        assert "Quotient" in answer
        assert "zwölf" in answer
        assert "drei" in answer
        assert "vier" in answer
        assert "12 / 3 = 4" in answer

    def test_is_arithmetic_question_true(self, question_parser):
        """Test: Erkennung als arithmetische Frage"""
        assert (
            question_parser.is_arithmetic_question(
                "Was ist die Summe von drei und fünf?"
            )
            is True
        )
        assert (
            question_parser.is_arithmetic_question(
                "Was ist das Produkt von zwei und vier?"
            )
            is True
        )
        assert (
            question_parser.is_arithmetic_question(
                "Was ist die Differenz zwischen zehn und drei?"
            )
            is True
        )
        assert (
            question_parser.is_arithmetic_question(
                "Was ist der Quotient von acht und zwei?"
            )
            is True
        )

    def test_is_arithmetic_question_false(self, question_parser):
        """Test: Keine arithmetische Frage"""
        assert question_parser.is_arithmetic_question("Wie heißt du?") is False
        assert (
            question_parser.is_arithmetic_question("Was ist dein Lieblingsessen?")
            is False
        )
        assert (
            question_parser.is_arithmetic_question("Erzähle mir eine Geschichte")
            is False
        )


class TestIntegration:
    """Integrationstests für arithmetische Konzepte"""

    def test_full_pipeline_summe(self, question_parser, netzwerk):
        """Test: Vollständige Pipeline für Summe"""
        from component_52_arithmetic_reasoning import ArithmeticEngine

        # Parse Frage
        parsed = question_parser.parse_question("Was ist die Summe von drei und fünf?")
        assert parsed is not None

        # Führe Operation aus
        engine = ArithmeticEngine(netzwerk)
        result = engine.calculate(parsed["symbol"], *parsed["operands"])

        assert result.value == 8
        assert result.confidence == 1.0

        # Formatiere Antwort
        answer = question_parser.format_answer(
            parsed["concept"], parsed["operands"], result.value, parsed["symbol"]
        )
        assert "acht" in answer

    def test_full_pipeline_produkt(self, question_parser, netzwerk):
        """Test: Vollständige Pipeline für Produkt"""
        from component_52_arithmetic_reasoning import ArithmeticEngine

        # Parse Frage
        parsed = question_parser.parse_question(
            "Was ist das Produkt von vier und sieben?"
        )
        assert parsed is not None

        # Führe Operation aus
        engine = ArithmeticEngine(netzwerk)
        result = engine.calculate(parsed["symbol"], *parsed["operands"])

        assert result.value == 28

        # Formatiere Antwort
        answer = question_parser.format_answer(
            parsed["concept"], parsed["operands"], result.value, parsed["symbol"]
        )
        assert "Produkt" in answer
        assert "28" in answer

    def test_full_pipeline_differenz(self, question_parser, netzwerk):
        """Test: Vollständige Pipeline für Differenz"""
        from component_52_arithmetic_reasoning import ArithmeticEngine

        # Parse Frage
        parsed = question_parser.parse_question(
            "Was ist die Differenz zwischen zehn und drei?"
        )
        assert parsed is not None

        # Führe Operation aus
        engine = ArithmeticEngine(netzwerk)
        result = engine.calculate(parsed["symbol"], *parsed["operands"])

        assert result.value == 7

    def test_full_pipeline_quotient(self, question_parser, netzwerk):
        """Test: Vollständige Pipeline für Quotient"""
        from fractions import Fraction

        from component_52_arithmetic_reasoning import ArithmeticEngine

        # Parse Frage
        parsed = question_parser.parse_question(
            "Was ist der Quotient von zwölf und drei?"
        )
        assert parsed is not None

        # Führe Operation aus
        engine = ArithmeticEngine(netzwerk)
        result = engine.calculate(parsed["symbol"], *parsed["operands"])

        # Division von Integers gibt Fraction zurück
        assert result.value == 4 or result.value == Fraction(4, 1)


class TestEdgeCases:
    """Tests für Edge Cases"""

    def test_concept_case_insensitive(self, concept_connector):
        """Test: Konzepterkennung case-insensitive"""
        result1 = concept_connector.extract_concept_from_text(
            "Was ist die SUMME von drei und fünf?"
        )
        result2 = concept_connector.extract_concept_from_text(
            "Was ist die Summe von drei und fünf?"
        )
        result3 = concept_connector.extract_concept_from_text(
            "Was ist die summe von drei und fünf?"
        )

        assert result1 is not None
        assert result2 is not None
        assert result3 is not None
        assert result1["operation"] == result2["operation"] == result3["operation"]

    def test_multiple_concepts_in_text(self, concept_connector):
        """Test: Mehrere Konzepte im Text (nur erstes wird erkannt)"""
        result = concept_connector.extract_concept_from_text(
            "Was ist die Summe von drei und fünf und das Produkt von zwei und vier?"
        )
        # Sollte nur "summe" erkennen (erstes Konzept)
        assert result is not None
        assert result["concept"] == "summe"

    def test_parse_question_with_extra_words(self, question_parser):
        """Test: Parse Frage mit zusätzlichen Wörtern"""
        result = question_parser.parse_question(
            "Kannst du mir bitte sagen was die Summe von drei und fünf ist?"
        )
        assert result is not None
        assert result["operands"] == [3, 5]

    def test_format_answer_large_numbers(self, question_parser):
        """Test: Formatiere Antwort mit großen Zahlen"""
        answer = question_parser.format_answer("summe", [100, 200], 300, "+")
        assert "300" in answer

    def test_concept_with_synonyms(self, concept_connector):
        """Test: Konzepterkennung mit Synonymen"""
        result = concept_connector.extract_concept_from_text(
            "Was ist das ergebnis der addition von drei und fünf?"
        )
        assert result is not None
        assert result["operation"] == "addition"
