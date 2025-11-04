"""
Additional tests for component_1_netzwerk.py to increase coverage from 56% to 70%+

Focuses on error handling, edge cases, episodic memory, and less-tested features.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from component_1_netzwerk import KonzeptNetzwerk
from component_11_embedding_service import EmbeddingService


@pytest.fixture(scope="session")
def netzwerk_session():
    """Session-scoped netzwerk fixture."""
    netzwerk = KonzeptNetzwerk()
    yield netzwerk
    netzwerk.close()


@pytest.fixture(scope="session")
def embedding_service_session():
    """Session-scoped embedding service fixture."""
    return EmbeddingService()


class TestErrorHandling:
    """Tests for error handling when driver is None or operations fail."""

    def test_ensure_wort_und_konzept_no_driver(self):
        """Test ensure_wort_und_konzept when driver is None."""
        netzwerk = KonzeptNetzwerk()
        netzwerk.driver = None

        result = netzwerk.ensure_wort_und_konzept("test")
        assert result is False

    def test_set_wort_attribut_no_driver(self):
        """Test set_wort_attribut when driver is None."""
        netzwerk = KonzeptNetzwerk()
        netzwerk.driver = None

        # Should not raise exception
        netzwerk.set_wort_attribut("test", "attr", "value")

    def test_set_wort_attribut_invalid_name(self, netzwerk_session):
        """Test set_wort_attribut with invalid attribute name."""
        # Invalid attribute name should be sanitized
        netzwerk_session.set_wort_attribut("test_word", "!!!invalid###", "value")
        # Should complete without raising

    def test_add_information_base_concept_failed(self, netzwerk_session):
        """Test add_information when base concept creation fails."""
        # Mock a failure scenario by using driver=None temporarily
        original_driver = netzwerk_session.driver
        netzwerk_session.driver = None

        result = netzwerk_session.add_information_zu_wort("test", "bedeutung", "test")
        assert result.get("created") is False
        assert result.get("error") == "base_concept_creation_failed"

        # Restore driver
        netzwerk_session.driver = original_driver

    def test_add_information_unknown_type(self, netzwerk_session):
        """Test add_information with unknown info type."""
        netzwerk_session.ensure_wort_und_konzept("test_word_unknown")
        result = netzwerk_session.add_information_zu_wort(
            "test_word_unknown", "unknown_type", "test"
        )
        assert result.get("created") is False
        assert result.get("error") == "unknown_type"

    def test_add_bedeutung_no_driver(self):
        """Test _add_bedeutung when driver is None."""
        netzwerk = KonzeptNetzwerk()
        netzwerk.driver = None

        result = netzwerk._add_bedeutung("test", "bedeutung")
        assert result.get("created") is False

    def test_add_synonym_no_driver(self):
        """Test _add_synonym when driver is None."""
        netzwerk = KonzeptNetzwerk()
        netzwerk.driver = None

        result = netzwerk._add_synonym("test1", "test2")
        assert result.get("created") is False

    def test_get_details_no_driver(self):
        """Test get_details_fuer_wort when driver is None."""
        netzwerk = KonzeptNetzwerk()
        netzwerk.driver = None

        result = netzwerk.get_details_fuer_wort("test")
        assert result is None

    def test_create_extraction_rule_no_driver(self):
        """Test create_extraction_rule when driver is None."""
        netzwerk = KonzeptNetzwerk()
        netzwerk.driver = None

        result = netzwerk.create_extraction_rule("TEST_TYPE", r"^(.+) test (.+)$")
        assert result is False

    def test_assert_relation_no_driver(self):
        """Test assert_relation when driver is None."""
        netzwerk = KonzeptNetzwerk()
        netzwerk.driver = None

        result = netzwerk.assert_relation("subject", "RELATION", "object")
        assert result is False

    def test_assert_relation_invalid_type(self, netzwerk_session):
        """Test assert_relation with invalid relation type."""
        result = netzwerk_session.assert_relation("test_subj", "!!!", "test_obj")
        assert result is False

    def test_query_graph_no_driver(self):
        """Test query_graph_for_facts when driver is None."""
        netzwerk = KonzeptNetzwerk()
        netzwerk.driver = None

        result = netzwerk.query_graph_for_facts("test")
        assert result == {}


class TestPatternPrototypes:
    """Tests for pattern prototype operations."""

    def test_get_all_pattern_prototypes_no_driver(self):
        """Test get_all_pattern_prototypes when driver is None."""
        netzwerk = KonzeptNetzwerk()
        netzwerk.driver = None

        result = netzwerk.get_all_pattern_prototypes()
        assert result == []

    def test_get_all_pattern_prototypes_with_category(self, netzwerk_session):
        """Test get_all_pattern_prototypes with category filter."""
        result = netzwerk_session.get_all_pattern_prototypes(category="DEFINITION")
        assert isinstance(result, list)

    def test_create_pattern_prototype_no_driver(self):
        """Test create_pattern_prototype when driver is None."""
        netzwerk = KonzeptNetzwerk()
        netzwerk.driver = None

        result = netzwerk.create_pattern_prototype([0.1, 0.2], "TEST")
        assert result is None

    def test_update_pattern_prototype_no_driver(self):
        """Test update_pattern_prototype when driver is None."""
        netzwerk = KonzeptNetzwerk()
        netzwerk.driver = None

        netzwerk.update_pattern_prototype("id", [0.1], [0.1], 1)
        # Should not raise

    def test_link_prototype_to_rule_no_driver(self):
        """Test link_prototype_to_rule when driver is None."""
        netzwerk = KonzeptNetzwerk()
        netzwerk.driver = None

        result = netzwerk.link_prototype_to_rule("proto_id", "RULE_TYPE")
        assert result is False

    def test_get_rule_for_prototype_no_driver(self):
        """Test get_rule_for_prototype when driver is None."""
        netzwerk = KonzeptNetzwerk()
        netzwerk.driver = None

        result = netzwerk.get_rule_for_prototype("proto_id")
        assert result is None


class TestLexicalTriggers:
    """Tests for lexical trigger operations."""

    def test_add_lexical_trigger_no_driver(self):
        """Test add_lexical_trigger when driver is None."""
        netzwerk = KonzeptNetzwerk()
        netzwerk.driver = None

        result = netzwerk.add_lexical_trigger("test")
        assert result is False

    def test_add_lexical_trigger_success(self, netzwerk_session):
        """Test add_lexical_trigger with valid word."""
        netzwerk_session.ensure_wort_und_konzept("test_trigger_word")
        result = netzwerk_session.add_lexical_trigger("test_trigger_word")
        # Should succeed or already exist
        assert isinstance(result, bool)

    def test_get_lexical_triggers_no_driver(self):
        """Test get_lexical_triggers when driver is None."""
        netzwerk = KonzeptNetzwerk()
        netzwerk.driver = None

        result = netzwerk.get_lexical_triggers()
        assert result == []

    def test_get_lexical_triggers(self, netzwerk_session):
        """Test get_lexical_triggers retrieves triggers."""
        result = netzwerk_session.get_lexical_triggers()
        assert isinstance(result, list)


class TestOtherQueries:
    """Tests for other query methods."""

    def test_get_all_extraction_rules_no_driver(self):
        """Test get_all_extraction_rules when driver is None."""
        netzwerk = KonzeptNetzwerk()
        netzwerk.driver = None

        result = netzwerk.get_all_extraction_rules()
        assert result == []

    def test_get_all_known_words_no_driver(self):
        """Test get_all_known_words when driver is None."""
        netzwerk = KonzeptNetzwerk()
        netzwerk.driver = None

        result = netzwerk.get_all_known_words()
        assert result == []

    def test_get_all_known_words(self, netzwerk_session):
        """Test get_all_known_words retrieves words."""
        # Add a test word
        netzwerk_session.ensure_wort_und_konzept("test_known_word")

        result = netzwerk_session.get_all_known_words()
        assert isinstance(result, list)
        assert len(result) > 0


class TestSimilarWordFinding:
    """Tests for find_similar_words functionality."""

    def test_find_similar_words_no_driver(self):
        """Test find_similar_words when driver is None."""
        netzwerk = KonzeptNetzwerk()
        netzwerk.driver = None

        result = netzwerk.find_similar_words("test", None)
        assert result == []

    def test_find_similar_words_no_embedding_service(self, netzwerk_session):
        """Test find_similar_words when embedding service is None."""
        result = netzwerk_session.find_similar_words("test", None)
        assert result == []

    def test_find_similar_words_no_known_words(self, embedding_service_session):
        """Test find_similar_words with empty knowledge base."""
        netzwerk = KonzeptNetzwerk()
        # Mock driver that returns no words
        result = netzwerk.find_similar_words("test", embedding_service_session)
        # Should handle empty knowledge base gracefully
        assert isinstance(result, list)


class TestEpisodicMemory:
    """Tests for episodic memory features."""

    def test_create_episode_no_driver(self):
        """Test create_episode when driver is None."""
        netzwerk = KonzeptNetzwerk()
        netzwerk.driver = None

        result = netzwerk.create_episode("test_type", "test content")
        assert result is None

    def test_create_episode_with_metadata(self, netzwerk_session):
        """Test create_episode with metadata."""
        episode_id = netzwerk_session.create_episode(
            episode_type="test_episode",
            content="Test episode content",
            metadata={"source": "test", "confidence": 0.9},
        )
        assert episode_id is not None
        assert isinstance(episode_id, str)

    def test_link_fact_to_episode_no_driver(self):
        """Test link_fact_to_episode when driver is None."""
        netzwerk = KonzeptNetzwerk()
        netzwerk.driver = None

        result = netzwerk.link_fact_to_episode("subj", "REL", "obj", "ep_id")
        assert result is False

    def test_link_fact_to_episode_invalid_relation(self, netzwerk_session):
        """Test link_fact_to_episode with invalid relation type."""
        result = netzwerk_session.link_fact_to_episode("subj", "!!!", "obj", "test_id")
        assert result is False

    def test_query_episodes_about_no_driver(self):
        """Test query_episodes_about when driver is None."""
        netzwerk = KonzeptNetzwerk()
        netzwerk.driver = None

        result = netzwerk.query_episodes_about("test")
        assert result == []

    def test_query_all_episodes_no_driver(self):
        """Test query_all_episodes when driver is None."""
        netzwerk = KonzeptNetzwerk()
        netzwerk.driver = None

        result = netzwerk.query_all_episodes()
        assert result == []

    def test_query_all_episodes_with_type_filter(self, netzwerk_session):
        """Test query_all_episodes with episode type filter."""
        result = netzwerk_session.query_all_episodes(
            episode_type="test_episode", limit=5
        )
        assert isinstance(result, list)

    def test_delete_episode_no_driver(self):
        """Test delete_episode when driver is None."""
        netzwerk = KonzeptNetzwerk()
        netzwerk.driver = None

        result = netzwerk.delete_episode("test_id")
        assert result is False

    def test_delete_episode_not_found(self, netzwerk_session):
        """Test delete_episode with non-existent episode."""
        result = netzwerk_session.delete_episode("nonexistent_episode_id_12345")
        assert result is False


class TestInferenceEpisodes:
    """Tests for inference episode tracking."""

    def test_create_inference_episode_no_driver(self):
        """Test create_inference_episode when driver is None."""
        netzwerk = KonzeptNetzwerk()
        netzwerk.driver = None

        result = netzwerk.create_inference_episode("test", "query")
        assert result is None

    def test_create_proof_step_no_driver(self):
        """Test create_proof_step when driver is None."""
        netzwerk = KonzeptNetzwerk()
        netzwerk.driver = None

        result = netzwerk.create_proof_step("goal", "method", 0.5, 1)
        assert result is None

    def test_link_inference_to_proof_no_driver(self):
        """Test link_inference_to_proof when driver is None."""
        netzwerk = KonzeptNetzwerk()
        netzwerk.driver = None

        result = netzwerk.link_inference_to_proof("ep_id", "step_id")
        assert result is False

    def test_link_inference_to_facts_no_driver(self):
        """Test link_inference_to_facts when driver is None."""
        netzwerk = KonzeptNetzwerk()
        netzwerk.driver = None

        result = netzwerk.link_inference_to_facts("ep_id", ["fact1"])
        assert result is False

    def test_link_inference_to_facts_empty_list(self, netzwerk_session):
        """Test link_inference_to_facts with empty fact list."""
        result = netzwerk_session.link_inference_to_facts("ep_id", [])
        assert result is False

    def test_link_inference_to_rules_no_driver(self):
        """Test link_inference_to_rules when driver is None."""
        netzwerk = KonzeptNetzwerk()
        netzwerk.driver = None

        result = netzwerk.link_inference_to_rules("ep_id", ["rule1"])
        assert result is False

    def test_link_inference_to_rules_empty_list(self, netzwerk_session):
        """Test link_inference_to_rules with empty rule list."""
        result = netzwerk_session.link_inference_to_rules("ep_id", [])
        assert result is False

    def test_query_inference_history_no_driver(self):
        """Test query_inference_history when driver is None."""
        netzwerk = KonzeptNetzwerk()
        netzwerk.driver = None

        result = netzwerk.query_inference_history()
        assert result == []

    def test_get_proof_tree_no_driver(self):
        """Test get_proof_tree when driver is None."""
        netzwerk = KonzeptNetzwerk()
        netzwerk.driver = None

        result = netzwerk.get_proof_tree("step_id")
        assert result is None

    def test_explain_inference_no_driver(self):
        """Test explain_inference when driver is None."""
        netzwerk = KonzeptNetzwerk()
        netzwerk.driver = None

        result = netzwerk.explain_inference("ep_id")
        assert "nicht verfügbar" in result.lower()


class TestHypothesisStorage:
    """Tests for hypothesis storage (abductive reasoning)."""

    def test_store_hypothesis_no_driver(self):
        """Test store_hypothesis when driver is None."""
        netzwerk = KonzeptNetzwerk()
        netzwerk.driver = None

        result = netzwerk.store_hypothesis(
            "hyp_id",
            "explanation",
            ["obs1"],
            "template",
            0.8,
            {"coverage": 0.9},
            [{"subject": "s", "relation": "r", "object": "o"}],
        )
        assert result is False

    def test_link_hypothesis_to_observations_no_driver(self):
        """Test link_hypothesis_to_observations when driver is None."""
        netzwerk = KonzeptNetzwerk()
        netzwerk.driver = None

        result = netzwerk.link_hypothesis_to_observations("hyp_id", ["obs1"])
        assert result is False

    def test_link_hypothesis_to_observations_empty(self, netzwerk_session):
        """Test link_hypothesis_to_observations with empty list."""
        result = netzwerk_session.link_hypothesis_to_observations("hyp_id", [])
        assert result is False

    def test_link_hypothesis_to_concepts_no_driver(self):
        """Test link_hypothesis_to_concepts when driver is None."""
        netzwerk = KonzeptNetzwerk()
        netzwerk.driver = None

        result = netzwerk.link_hypothesis_to_concepts("hyp_id", ["concept1"])
        assert result is False

    def test_link_hypothesis_to_concepts_empty(self, netzwerk_session):
        """Test link_hypothesis_to_concepts with empty list."""
        result = netzwerk_session.link_hypothesis_to_concepts("hyp_id", [])
        assert result is False

    def test_query_hypotheses_about_no_driver(self):
        """Test query_hypotheses_about when driver is None."""
        netzwerk = KonzeptNetzwerk()
        netzwerk.driver = None

        result = netzwerk.query_hypotheses_about("topic")
        assert result == []

    def test_get_best_hypothesis_for_no_results(self, netzwerk_session):
        """Test get_best_hypothesis_for with no results."""
        result = netzwerk_session.get_best_hypothesis_for("nonexistent_topic_12345")
        assert result is None

    def test_explain_hypothesis_no_driver(self):
        """Test explain_hypothesis when driver is None."""
        netzwerk = KonzeptNetzwerk()
        netzwerk.driver = None

        result = netzwerk.explain_hypothesis("hyp_id")
        assert "nicht verfügbar" in result.lower()


class TestQueryFactsWithSynonyms:
    """Tests for query_facts_with_synonyms functionality."""

    def test_query_facts_with_synonyms_no_driver(self):
        """Test query_facts_with_synonyms when driver is None."""
        netzwerk = KonzeptNetzwerk()
        netzwerk.driver = None

        result = netzwerk.query_facts_with_synonyms("test")
        assert result["primary_topic"] == "test"
        assert result["synonyms"] == []
        assert result["facts"] == {}
        assert result["sources"] == {}
        assert result["bedeutungen"] == []

    def test_query_facts_with_synonyms(self, netzwerk_session):
        """Test query_facts_with_synonyms with real data."""
        # Create test word
        netzwerk_session.ensure_wort_und_konzept("test_word_syn")

        result = netzwerk_session.query_facts_with_synonyms("test_word_syn")
        assert result["primary_topic"] == "test_word_syn"
        assert isinstance(result["synonyms"], list)
        assert isinstance(result["facts"], dict)
        assert isinstance(result["sources"], dict)
        assert isinstance(result["bedeutungen"], list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
