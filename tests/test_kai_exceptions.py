"""
Test suite for kai_exceptions.py

Tests all exception classes, string representations, context handling,
and exception wrapping functionality.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from kai_exceptions import (
    # Base
    KAIException,
    # Database
    DatabaseException,
    Neo4jConnectionError,
    Neo4jQueryError,
    Neo4jWriteError,
    # Linguistic
    LinguisticException,
    ParsingError,
    SpaCyModelError,
    EmbeddingError,
    # Knowledge
    KnowledgeException,
    ExtractionRuleError,
    PatternMatchingError,
    ConceptNotFoundError,
    RelationValidationError,
    # Reasoning
    ReasoningException,
    LogicEngineError,
    UnificationError,
    InferenceError,
    # Planning
    PlanningException,
    GoalPlanningError,
    InvalidMeaningPointError,
    PlanExecutionError,
    # Configuration
    ConfigurationException,
    InvalidConfigError,
    MissingDependencyError,
    # Utility
    wrap_exception,
)


class TestKAIExceptionBase:
    """Tests for the base KAIException class."""

    def test_basic_exception(self):
        """Test basic exception creation."""
        exc = KAIException("Test error")
        assert str(exc) == "Test error"
        assert exc.message == "Test error"
        assert exc.context == {}
        assert exc.original_exception is None

    def test_exception_with_context(self):
        """Test exception with context dictionary."""
        context = {"key1": "value1", "key2": 123}
        exc = KAIException("Test error", context=context)
        assert exc.context == context
        assert "key1=value1" in str(exc)
        assert "key2=123" in str(exc)

    def test_exception_with_original(self):
        """Test exception with original exception."""
        original = ValueError("Original error")
        exc = KAIException("Wrapped error", original_exception=original)
        assert exc.original_exception is original
        assert "ValueError" in str(exc)
        assert "Original error" in str(exc)

    def test_exception_str_with_all_info(self):
        """Test string representation with all information."""
        original = ValueError("Original")
        context = {"test": "data"}
        exc = KAIException("Main error", context=context, original_exception=original)

        exc_str = str(exc)
        assert "Main error" in exc_str
        assert "test=data" in exc_str
        assert "ValueError" in exc_str
        assert "Original" in exc_str

    def test_exception_repr(self):
        """Test repr representation."""
        exc = KAIException("Test", context={"a": 1})
        repr_str = repr(exc)
        assert "KAIException" in repr_str
        assert "message='Test'" in repr_str
        assert "context={'a': 1}" in repr_str


class TestDatabaseExceptions:
    """Tests for database-related exceptions."""

    def test_database_exception(self):
        """Test DatabaseException base class."""
        exc = DatabaseException("DB error")
        assert isinstance(exc, KAIException)
        assert str(exc) == "DB error"

    def test_neo4j_connection_error(self):
        """Test Neo4jConnectionError."""
        exc = Neo4jConnectionError(
            "Connection failed", context={"uri": "bolt://localhost"}
        )
        assert isinstance(exc, DatabaseException)
        assert "Connection failed" in str(exc)
        assert "uri=bolt://localhost" in str(exc)

    def test_neo4j_query_error(self):
        """Test Neo4jQueryError with query context."""
        exc = Neo4jQueryError(
            "Query failed", query="MATCH (n) RETURN n", parameters={"limit": 10}
        )
        assert isinstance(exc, DatabaseException)
        assert "Query failed" in str(exc)
        assert exc.context["query"] == "MATCH (n) RETURN n"
        assert exc.context["parameters"] == {"limit": 10}

    def test_neo4j_query_error_without_query(self):
        """Test Neo4jQueryError without query info."""
        exc = Neo4jQueryError("Query failed")
        assert exc.context["query"] is None
        assert exc.context["parameters"] is None

    def test_neo4j_write_error(self):
        """Test Neo4jWriteError."""
        exc = Neo4jWriteError("Write failed")
        assert isinstance(exc, DatabaseException)
        assert "Write failed" in str(exc)


class TestLinguisticExceptions:
    """Tests for linguistic processing exceptions."""

    def test_linguistic_exception(self):
        """Test LinguisticException base class."""
        exc = LinguisticException("Linguistic error")
        assert isinstance(exc, KAIException)

    def test_parsing_error(self):
        """Test ParsingError."""
        exc = ParsingError("Parse failed", context={"input": "test"})
        assert isinstance(exc, LinguisticException)
        assert "Parse failed" in str(exc)

    def test_spacy_model_error(self):
        """Test SpaCyModelError."""
        exc = SpaCyModelError("Model not found")
        assert isinstance(exc, LinguisticException)

    def test_embedding_error(self):
        """Test EmbeddingError."""
        exc = EmbeddingError("Embedding failed")
        assert isinstance(exc, LinguisticException)


class TestKnowledgeExceptions:
    """Tests for knowledge processing exceptions."""

    def test_knowledge_exception(self):
        """Test KnowledgeException base class."""
        exc = KnowledgeException("Knowledge error")
        assert isinstance(exc, KAIException)

    def test_extraction_rule_error(self):
        """Test ExtractionRuleError with rule context."""
        exc = ExtractionRuleError("Invalid rule", rule_id="rule_123", pattern=r"^(.+)$")
        assert isinstance(exc, KnowledgeException)
        assert exc.context["rule_id"] == "rule_123"
        assert exc.context["pattern"] == r"^(.+)$"

    def test_extraction_rule_error_without_context(self):
        """Test ExtractionRuleError without additional context."""
        exc = ExtractionRuleError("Invalid rule")
        assert exc.context["rule_id"] is None
        assert exc.context["pattern"] is None

    def test_pattern_matching_error(self):
        """Test PatternMatchingError."""
        exc = PatternMatchingError("Match failed")
        assert isinstance(exc, KnowledgeException)

    def test_concept_not_found_error(self):
        """Test ConceptNotFoundError with concept name."""
        exc = ConceptNotFoundError("Concept not found", concept_name="unbekanntes_wort")
        assert isinstance(exc, KnowledgeException)
        assert exc.context["concept_name"] == "unbekanntes_wort"

    def test_concept_not_found_error_without_name(self):
        """Test ConceptNotFoundError without concept name."""
        exc = ConceptNotFoundError("Concept not found")
        assert exc.context["concept_name"] is None

    def test_relation_validation_error(self):
        """Test RelationValidationError with relation type."""
        exc = RelationValidationError("Invalid relation", relation_type="UNKNOWN_TYPE")
        assert isinstance(exc, KnowledgeException)
        assert exc.context["relation_type"] == "UNKNOWN_TYPE"

    def test_relation_validation_error_without_type(self):
        """Test RelationValidationError without relation type."""
        exc = RelationValidationError("Invalid relation")
        assert exc.context["relation_type"] is None


class TestReasoningExceptions:
    """Tests for reasoning and logic exceptions."""

    def test_reasoning_exception(self):
        """Test ReasoningException base class."""
        exc = ReasoningException("Reasoning error")
        assert isinstance(exc, KAIException)

    def test_logic_engine_error(self):
        """Test LogicEngineError."""
        exc = LogicEngineError("Inference failed")
        assert isinstance(exc, ReasoningException)

    def test_unification_error(self):
        """Test UnificationError."""
        exc = UnificationError("Unification failed")
        assert isinstance(exc, ReasoningException)

    def test_inference_error(self):
        """Test InferenceError."""
        exc = InferenceError("Cannot infer")
        assert isinstance(exc, ReasoningException)


class TestPlanningExceptions:
    """Tests for planning and goal execution exceptions."""

    def test_planning_exception(self):
        """Test PlanningException base class."""
        exc = PlanningException("Planning error")
        assert isinstance(exc, KAIException)

    def test_goal_planning_error(self):
        """Test GoalPlanningError."""
        exc = GoalPlanningError("Cannot create plan")
        assert isinstance(exc, PlanningException)

    def test_invalid_meaning_point_error(self):
        """Test InvalidMeaningPointError with type."""
        exc = InvalidMeaningPointError(
            "Invalid meaning point", meaning_point_type="UNKNOWN"
        )
        assert isinstance(exc, PlanningException)
        assert exc.context["meaning_point_type"] == "UNKNOWN"

    def test_invalid_meaning_point_error_without_type(self):
        """Test InvalidMeaningPointError without type."""
        exc = InvalidMeaningPointError("Invalid meaning point")
        assert exc.context["meaning_point_type"] is None

    def test_plan_execution_error(self):
        """Test PlanExecutionError with goal context."""
        exc = PlanExecutionError(
            "Plan failed", goal_type="ANSWER_QUESTION", subgoal_index=2
        )
        assert isinstance(exc, PlanningException)
        assert exc.context["goal_type"] == "ANSWER_QUESTION"
        assert exc.context["subgoal_index"] == 2

    def test_plan_execution_error_without_context(self):
        """Test PlanExecutionError without context."""
        exc = PlanExecutionError("Plan failed")
        assert exc.context["goal_type"] is None
        assert exc.context["subgoal_index"] is None


class TestConfigurationExceptions:
    """Tests for configuration and setup exceptions."""

    def test_configuration_exception(self):
        """Test ConfigurationException base class."""
        exc = ConfigurationException("Config error")
        assert isinstance(exc, KAIException)

    def test_invalid_config_error(self):
        """Test InvalidConfigError."""
        exc = InvalidConfigError("Invalid config")
        assert isinstance(exc, ConfigurationException)

    def test_missing_dependency_error(self):
        """Test MissingDependencyError with dependency name."""
        exc = MissingDependencyError("Dependency missing", dependency_name="spacy")
        assert isinstance(exc, ConfigurationException)
        assert exc.context["dependency_name"] == "spacy"

    def test_missing_dependency_error_without_name(self):
        """Test MissingDependencyError without dependency name."""
        exc = MissingDependencyError("Dependency missing")
        assert exc.context["dependency_name"] is None


class TestExceptionWrapping:
    """Tests for wrap_exception utility function."""

    def test_wrap_exception_basic(self):
        """Test basic exception wrapping."""
        original = ValueError("Original error")
        wrapped = wrap_exception(original, ParsingError, "Wrapped message")

        assert isinstance(wrapped, ParsingError)
        assert wrapped.message == "Wrapped message"
        assert wrapped.original_exception is original

    def test_wrap_exception_with_context(self):
        """Test exception wrapping with context."""
        original = ConnectionError("Connection lost")
        wrapped = wrap_exception(
            original,
            Neo4jConnectionError,
            "DB connection failed",
            uri="bolt://localhost",
            timeout=30,
        )

        assert isinstance(wrapped, Neo4jConnectionError)
        assert wrapped.context["uri"] == "bolt://localhost"
        assert wrapped.context["timeout"] == 30
        assert wrapped.original_exception is original

    def test_wrap_exception_preserves_type(self):
        """Test that wrap_exception preserves exception type hierarchy."""
        original = Exception("Generic error")
        wrapped = wrap_exception(original, LogicEngineError, "Logic failed")

        assert isinstance(wrapped, LogicEngineError)
        assert isinstance(wrapped, ReasoningException)
        assert isinstance(wrapped, KAIException)


class TestExceptionHierarchy:
    """Tests for exception hierarchy and inheritance."""

    def test_all_exceptions_inherit_from_kai_exception(self):
        """Test that all custom exceptions inherit from KAIException."""
        exception_classes = [
            DatabaseException,
            Neo4jConnectionError,
            Neo4jQueryError,
            Neo4jWriteError,
            LinguisticException,
            ParsingError,
            SpaCyModelError,
            EmbeddingError,
            KnowledgeException,
            ExtractionRuleError,
            PatternMatchingError,
            ConceptNotFoundError,
            RelationValidationError,
            ReasoningException,
            LogicEngineError,
            UnificationError,
            InferenceError,
            PlanningException,
            GoalPlanningError,
            InvalidMeaningPointError,
            PlanExecutionError,
            ConfigurationException,
            InvalidConfigError,
            MissingDependencyError,
        ]

        for exc_class in exception_classes:
            exc = exc_class("Test")
            assert isinstance(exc, KAIException)
            assert isinstance(exc, Exception)

    def test_exception_categories(self):
        """Test that exceptions are properly categorized."""
        # Database
        assert issubclass(Neo4jConnectionError, DatabaseException)
        assert issubclass(Neo4jQueryError, DatabaseException)

        # Linguistic
        assert issubclass(ParsingError, LinguisticException)
        assert issubclass(SpaCyModelError, LinguisticException)

        # Knowledge
        assert issubclass(ExtractionRuleError, KnowledgeException)
        assert issubclass(ConceptNotFoundError, KnowledgeException)

        # Reasoning
        assert issubclass(LogicEngineError, ReasoningException)
        assert issubclass(UnificationError, ReasoningException)

        # Planning
        assert issubclass(GoalPlanningError, PlanningException)
        assert issubclass(InvalidMeaningPointError, PlanningException)

        # Configuration
        assert issubclass(InvalidConfigError, ConfigurationException)
        assert issubclass(MissingDependencyError, ConfigurationException)


class TestExceptionCatching:
    """Tests for exception catching and handling."""

    def test_catch_specific_exception(self):
        """Test catching specific exception types."""
        with pytest.raises(Neo4jConnectionError) as exc_info:
            raise Neo4jConnectionError("Connection failed")

        assert "Connection failed" in str(exc_info.value)

    def test_catch_by_base_class(self):
        """Test catching exceptions by base class."""
        with pytest.raises(DatabaseException):
            raise Neo4jQueryError("Query failed")

        with pytest.raises(KAIException):
            raise ConceptNotFoundError("Concept not found")

    def test_exception_chaining_with_from(self):
        """Test exception chaining using 'from' keyword."""
        with pytest.raises(ParsingError) as exc_info:
            try:
                int("invalid")
            except ValueError as e:
                raise ParsingError("Parse error") from e

        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, ValueError)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
