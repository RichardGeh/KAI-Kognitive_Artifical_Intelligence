"""
test_exception_handling.py

Tests für die verbesserte Exception-Hierarchie und user-freundliche Fehlermeldungen.
"""

import pytest
from kai_exceptions import (
    KAIException,
    Neo4jConnectionError,
    Neo4jQueryError,
    EmbeddingError,
    ConceptNotFoundError,
    GraphTraversalError,
    ProbabilisticReasoningError,
    WorkingMemoryError,
    InferenceError,
    wrap_exception,
    get_user_friendly_message,
)


class TestExceptionHierarchy:
    """Tests für die Exception-Hierarchie"""

    def test_base_exception(self):
        """Test der Basis-Exception mit Context"""
        exc = KAIException(
            "Test Fehler",
            context={"key": "value"},
            original_exception=ValueError("original"),
        )

        assert exc.message == "Test Fehler"
        assert exc.context == {"key": "value"}
        assert isinstance(exc.original_exception, ValueError)
        assert "key=value" in str(exc)
        assert "ValueError" in str(exc)

    def test_neo4j_connection_error(self):
        """Test Neo4j Connection Error"""
        exc = Neo4jConnectionError(
            "Verbindung fehlgeschlagen", context={"uri": "bolt://localhost:7687"}
        )

        assert "Verbindung fehlgeschlagen" in str(exc)
        assert "uri=bolt://localhost:7687" in str(exc)

    def test_neo4j_query_error_with_query_context(self):
        """Test Neo4j Query Error mit Query-Kontext"""
        exc = Neo4jQueryError(
            "Query fehlgeschlagen", query="MATCH (n) RETURN n", parameters={"limit": 10}
        )

        assert exc.context["query"] == "MATCH (n) RETURN n"
        assert exc.context["parameters"] == {"limit": 10}

    def test_concept_not_found_error(self):
        """Test ConceptNotFoundError mit Konzept-Name"""
        exc = ConceptNotFoundError(
            "Konzept nicht gefunden", concept_name="unbekanntes_wort"
        )

        assert exc.context["concept_name"] == "unbekanntes_wort"

    def test_graph_traversal_error(self):
        """Test GraphTraversalError mit Source/Target"""
        exc = GraphTraversalError(
            "Kein Pfad gefunden", source_concept="hund", target_concept="fahrzeug"
        )

        assert exc.context["source_concept"] == "hund"
        assert exc.context["target_concept"] == "fahrzeug"

    def test_probabilistic_reasoning_error(self):
        """Test ProbabilisticReasoningError mit Wahrscheinlichkeit"""
        exc = ProbabilisticReasoningError(
            "Ungültige Wahrscheinlichkeit", probability_value=1.5
        )

        assert exc.context["probability_value"] == 1.5


class TestWrapException:
    """Tests für wrap_exception Utility"""

    def test_wrap_generic_exception(self):
        """Test wrapping einer generischen Exception"""
        try:
            int("invalid")
        except ValueError as e:
            wrapped = wrap_exception(
                e,
                EmbeddingError,
                "Embedding konnte nicht erstellt werden",
                input_text="invalid",
            )

            assert isinstance(wrapped, EmbeddingError)
            assert wrapped.message == "Embedding konnte nicht erstellt werden"
            assert wrapped.context["input_text"] == "invalid"
            assert isinstance(wrapped.original_exception, ValueError)


class TestUserFriendlyMessages:
    """Tests für user-freundliche Fehlermeldungen"""

    def test_neo4j_connection_error_message(self):
        """Test user-freundliche Nachricht für Neo4j Connection Error"""
        exc = Neo4jConnectionError("Connection failed")
        message = get_user_friendly_message(exc)

        assert "Wissensdatenbank" in message
        assert "Neo4j" in message
        assert "[ERROR]" in message

    def test_concept_not_found_error_message(self):
        """Test user-freundliche Nachricht für ConceptNotFoundError"""
        exc = ConceptNotFoundError(
            "Konzept nicht gefunden", concept_name="unbekanntes_konzept"
        )
        message = get_user_friendly_message(exc)

        assert "unbekanntes_konzept" in message
        assert "beibringen" in message
        assert "🤔" in message

    def test_graph_traversal_error_message(self):
        """Test user-freundliche Nachricht für GraphTraversalError"""
        exc = GraphTraversalError(
            "Kein Pfad", source_concept="hund", target_concept="auto"
        )
        message = get_user_friendly_message(exc)

        assert "hund" in message
        assert "auto" in message
        assert "keine Verbindung" in message

    def test_probabilistic_reasoning_error_message(self):
        """Test user-freundliche Nachricht für ProbabilisticReasoningError"""
        exc = ProbabilisticReasoningError(
            "Ungültige Probability", probability_value=1.5
        )
        message = get_user_friendly_message(exc)

        assert "1.5" in message
        assert "0 und 1" in message

    def test_embedding_error_message(self):
        """Test user-freundliche Nachricht für EmbeddingError"""
        exc = EmbeddingError("Model not loaded")
        message = get_user_friendly_message(exc)

        assert "Bedeutungsanalyse" in message or "Embedding" in message
        assert "[ERROR]" in message

    def test_inference_error_message(self):
        """Test user-freundliche Nachricht für InferenceError"""
        exc = InferenceError("Keine Schlussfolgerung möglich")
        message = get_user_friendly_message(exc)

        assert "Schlussfolgerung" in message
        assert "🤔" in message

    def test_working_memory_error_message(self):
        """Test user-freundliche Nachricht für WorkingMemoryError"""
        exc = WorkingMemoryError("Stack underflow")
        message = get_user_friendly_message(exc)

        assert "Arbeitsgedächtnis" in message or "Kontext" in message
        assert "[ERROR]" in message

    def test_unknown_exception_fallback(self):
        """Test Fallback für unbekannte Exception-Typen"""
        exc = ValueError("Some error")
        message = get_user_friendly_message(exc)

        assert "unerwarteter Fehler" in message
        assert "[ERROR]" in message

    def test_user_friendly_message_with_details(self):
        """Test user-freundliche Nachricht mit technischen Details"""
        exc = Neo4jQueryError("Query failed", query="MATCH (n) RETURN n")
        message = get_user_friendly_message(exc, include_details=True)

        assert "💡 Technische Details" in message
        assert "Query failed" in message
        assert "query" in message.lower()


class TestExceptionInheritance:
    """Tests für Exception-Vererbung"""

    def test_all_exceptions_inherit_from_kai_exception(self):
        """Teste dass alle Exceptions von KAIException erben"""
        exceptions = [
            Neo4jConnectionError("test"),
            EmbeddingError("test"),
            ConceptNotFoundError("test"),
            GraphTraversalError("test"),
            InferenceError("test"),
            WorkingMemoryError("test"),
        ]

        for exc in exceptions:
            assert isinstance(exc, KAIException)

    def test_exception_catching(self):
        """Test dass spezifische Exceptions richtig gefangen werden können"""
        try:
            raise ConceptNotFoundError("Test", concept_name="test_konzept")
        except KAIException as e:
            assert e.context["concept_name"] == "test_konzept"
            assert isinstance(e, ConceptNotFoundError)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
