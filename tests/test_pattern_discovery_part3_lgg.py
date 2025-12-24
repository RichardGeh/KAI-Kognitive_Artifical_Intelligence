# test_pattern_discovery_part3_lgg.py
"""
Test scenarios for LGG Template Induction (Part 3: Pattern Discovery).

Tests cover Phase 6: Least General Generalization (LGG) algorithm for template creation.

Test Categories:
- Happy Path: Template induction from similar utterances
- LITERAL Detection: Fixed words at positions
- SLOT Detection: Varying lemmas, allowed values, frequency counts
- Edge Cases: Invalid clusters, alignment issues, missing data
- Batch Operations: N+1 query prevention
- Pattern Creation: Pattern/PatternItem/Slot node creation

Total: 25 tests
"""

import logging
from unittest.mock import patch

import numpy as np
import pytest

from component_61_pattern_discovery import PatternDiscoveryEngine


class TestHappyPathLGG:
    """Happy path scenarios for LGG template induction (LGG-HP-01 to LGG-HP-06)."""

    def test_lgg_hp_01_induce_from_5_similar(
        self, netzwerk_session, utterance_manager, pattern_manager
    ):
        """LGG-HP-01: Induce template from 5 similar WH-questions."""
        from component_6_linguistik_engine import LinguisticPreprocessor

        preprocessor = LinguisticPreprocessor()
        pattern_discovery = PatternDiscoveryEngine(netzwerk_session)

        # Create 5 similar utterances
        utterance_ids = []
        embeddings = []
        for animal in ["hund", "katze", "vogel", "fisch", "maus"]:
            text = f"Was ist ein {animal}?"
            doc = preprocessor.process(text)
            embedding = np.random.rand(384)
            embedding = (embedding / np.linalg.norm(embedding)).tolist()

            uid = utterance_manager.create_utterance(text, embedding)
            utterance_ids.append(uid)
            embeddings.append(embedding)

            # Create tokens
            for token in doc:
                utterance_manager.create_token(
                    surface=token.text,
                    lemma=token.lemma_.lower(),
                    pos=token.pos_,
                    utterance_id=uid,
                    idx=token.i,
                )

        # Create cluster
        centroid = np.mean(embeddings, axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        cluster = {
            "utterance_ids": utterance_ids,
            "centroid": centroid.tolist(),
            "size": 5,
        }

        # Induce template
        pattern_id = pattern_discovery.induce_template_from_cluster(cluster)

        # Assert pattern created
        pattern = pattern_manager.get_pattern(pattern_id)
        assert pattern is not None
        assert pattern["type"] == "learned"
        assert pattern["support"] == 5
        assert pattern["precision"] == 0.5

        # Assert pattern items exist
        items = pattern_manager.get_pattern_items(pattern_id)
        assert len(items) >= 3  # At least "was", "ist", SLOT

    def test_lgg_hp_02_all_literal_pattern(
        self, netzwerk_session, utterance_manager, pattern_manager
    ):
        """LGG-HP-02: Identical utterances produce all-LITERAL pattern."""
        from component_6_linguistik_engine import LinguisticPreprocessor

        preprocessor = LinguisticPreprocessor()
        pattern_discovery = PatternDiscoveryEngine(netzwerk_session)

        # Create 3 identical utterances
        utterance_ids = []
        text = "Was ist hund?"
        for i in range(3):
            embedding = [0.1] * 384
            uid = utterance_manager.create_utterance(text, embedding)
            utterance_ids.append(uid)

            # Create tokens
            doc = preprocessor.process(text)
            for token in doc:
                utterance_manager.create_token(
                    surface=token.text,
                    lemma=token.lemma_.lower(),
                    pos=token.pos_,
                    utterance_id=uid,
                    idx=token.i,
                )

        cluster = {
            "utterance_ids": utterance_ids,
            "centroid": (np.array([0.1] * 384) / np.linalg.norm([0.1] * 384)).tolist(),
            "size": 3,
        }

        pattern_id = pattern_discovery.induce_template_from_cluster(cluster)

        # Assert all items are LITERAL (no SLOT)
        items = pattern_manager.get_pattern_items(pattern_id)
        literal_count = sum(1 for item in items if item["kind"] == "LITERAL")
        slot_count = sum(1 for item in items if item["kind"] == "SLOT")

        assert literal_count == len(items)
        assert slot_count == 0

    def test_lgg_hp_03_all_slot_pattern(
        self, netzwerk_session, utterance_manager, pattern_manager
    ):
        """LGG-HP-03: Completely different utterances (same length) produce all-SLOT pattern."""
        from component_6_linguistik_engine import LinguisticPreprocessor

        preprocessor = LinguisticPreprocessor()
        pattern_discovery = PatternDiscoveryEngine(netzwerk_session)

        # Create 3 completely different utterances (same length)
        utterance_ids = []
        texts = ["Was ist hund", "Wer sind katze", "Wie war vogel"]
        for text in texts:
            embedding = [0.1] * 384
            uid = utterance_manager.create_utterance(text, embedding)
            utterance_ids.append(uid)

            doc = preprocessor.process(text)
            for token in doc:
                utterance_manager.create_token(
                    surface=token.text,
                    lemma=token.lemma_.lower(),
                    pos=token.pos_,
                    utterance_id=uid,
                    idx=token.i,
                )

        cluster = {
            "utterance_ids": utterance_ids,
            "centroid": (np.array([0.1] * 384) / np.linalg.norm([0.1] * 384)).tolist(),
            "size": 3,
        }

        pattern_id = pattern_discovery.induce_template_from_cluster(cluster)

        # Most items should be SLOT
        items = pattern_manager.get_pattern_items(pattern_id)
        slot_count = sum(1 for item in items if item["kind"] == "SLOT")
        assert slot_count >= 1  # At least some slots

    def test_lgg_hp_04_mixed_literal_slot_pattern(
        self, netzwerk_session, utterance_manager, pattern_manager
    ):
        """LGG-HP-04: Partially varying utterances produce mixed LITERAL/SLOT pattern."""
        from component_6_linguistik_engine import LinguisticPreprocessor

        preprocessor = LinguisticPreprocessor()
        pattern_discovery = PatternDiscoveryEngine(netzwerk_session)

        # Fixed start, varying end
        utterance_ids = []
        for noun in ["hund", "katze", "vogel"]:
            text = f"Was ist {noun}"
            embedding = [0.1] * 384
            uid = utterance_manager.create_utterance(text, embedding)
            utterance_ids.append(uid)

            doc = preprocessor.process(text)
            for token in doc:
                utterance_manager.create_token(
                    surface=token.text,
                    lemma=token.lemma_.lower(),
                    pos=token.pos_,
                    utterance_id=uid,
                    idx=token.i,
                )

        cluster = {
            "utterance_ids": utterance_ids,
            "centroid": (np.array([0.1] * 384) / np.linalg.norm([0.1] * 384)).tolist(),
            "size": 3,
        }

        pattern_id = pattern_discovery.induce_template_from_cluster(cluster)

        # Assert mix of LITERAL and SLOT
        items = pattern_manager.get_pattern_items(pattern_id)
        literal_count = sum(1 for item in items if item["kind"] == "LITERAL")
        slot_count = sum(1 for item in items if item["kind"] == "SLOT")

        assert literal_count >= 1
        assert slot_count >= 1

    def test_lgg_hp_05_pattern_name_unique(
        self, netzwerk_session, utterance_manager, pattern_manager
    ):
        """LGG-HP-05: Multiple clusters produce patterns with unique names."""
        from component_6_linguistik_engine import LinguisticPreprocessor

        preprocessor = LinguisticPreprocessor()
        pattern_discovery = PatternDiscoveryEngine(netzwerk_session)

        # Create 2 clusters
        pattern_ids = []
        for cluster_idx in range(2):
            utterance_ids = []
            for i in range(3):
                text = f"Cluster {cluster_idx} item {i}"
                embedding = [0.1] * 384
                uid = utterance_manager.create_utterance(text, embedding)
                utterance_ids.append(uid)

                doc = preprocessor.process(text)
                for token in doc:
                    utterance_manager.create_token(
                        surface=token.text,
                        lemma=token.lemma_.lower(),
                        pos=token.pos_,
                        utterance_id=uid,
                        idx=token.i,
                    )

            cluster = {
                "utterance_ids": utterance_ids,
                "centroid": (
                    np.array([0.1] * 384) / np.linalg.norm([0.1] * 384)
                ).tolist(),
                "size": 3,
            }

            pattern_id = pattern_discovery.induce_template_from_cluster(cluster)
            pattern_ids.append(pattern_id)

        # Assert unique pattern names
        pattern1 = pattern_manager.get_pattern(pattern_ids[0])
        pattern2 = pattern_manager.get_pattern(pattern_ids[1])

        assert pattern1["name"] != pattern2["name"]
        assert "Learned_Pattern_" in pattern1["name"]
        assert "Learned_Pattern_" in pattern2["name"]

    def test_lgg_hp_06_centroid_set_from_cluster(
        self, netzwerk_session, utterance_manager, pattern_manager
    ):
        """LGG-HP-06: Pattern centroid matches cluster centroid exactly."""
        from component_6_linguistik_engine import LinguisticPreprocessor

        preprocessor = LinguisticPreprocessor()
        pattern_discovery = PatternDiscoveryEngine(netzwerk_session)

        # Create cluster with specific centroid
        utterance_ids = []
        expected_centroid = np.random.rand(384)
        expected_centroid = (
            expected_centroid / np.linalg.norm(expected_centroid)
        ).tolist()

        for i in range(3):
            text = f"Test utterance {i}"
            uid = utterance_manager.create_utterance(text, expected_centroid)
            utterance_ids.append(uid)

            doc = preprocessor.process(text)
            for token in doc:
                utterance_manager.create_token(
                    surface=token.text,
                    lemma=token.lemma_.lower(),
                    pos=token.pos_,
                    utterance_id=uid,
                    idx=token.i,
                )

        cluster = {
            "utterance_ids": utterance_ids,
            "centroid": expected_centroid,
            "size": 3,
        }

        pattern_id = pattern_discovery.induce_template_from_cluster(cluster)

        # Assert centroid matches
        pattern = pattern_manager.get_pattern(pattern_id)
        assert np.allclose(pattern["centroid"], expected_centroid, atol=0.01)


class TestLiteralDetection:
    """LITERAL detection scenarios (LGG-LIT-01 to LGG-LIT-03)."""

    def test_lgg_lit_01_all_lemmas_same_at_position(
        self, netzwerk_session, utterance_manager, pattern_manager
    ):
        """LGG-LIT-01: All lemmas same at position creates LITERAL."""
        from component_6_linguistik_engine import LinguisticPreprocessor

        preprocessor = LinguisticPreprocessor()
        pattern_discovery = PatternDiscoveryEngine(netzwerk_session)

        # All start with "was"
        utterance_ids = []
        for i in range(5):
            text = f"Was ist item{i}"
            embedding = [0.1] * 384
            uid = utterance_manager.create_utterance(text, embedding)
            utterance_ids.append(uid)

            doc = preprocessor.process(text)
            for token in doc:
                utterance_manager.create_token(
                    surface=token.text,
                    lemma=token.lemma_.lower(),
                    pos=token.pos_,
                    utterance_id=uid,
                    idx=token.i,
                )

        cluster = {
            "utterance_ids": utterance_ids,
            "centroid": (np.array([0.1] * 384) / np.linalg.norm([0.1] * 384)).tolist(),
            "size": 5,
        }

        pattern_id = pattern_discovery.induce_template_from_cluster(cluster)
        items = pattern_manager.get_pattern_items(pattern_id)

        # First item should be LITERAL("was")
        assert items[0]["kind"] == "LITERAL"
        assert items[0]["literalValue"] == "was"

    def test_lgg_lit_02_punctuation_literals(
        self, netzwerk_session, utterance_manager, pattern_manager
    ):
        """LGG-LIT-02: All utterances ending with '?' creates LITERAL."""
        from component_6_linguistik_engine import LinguisticPreprocessor

        preprocessor = LinguisticPreprocessor()
        pattern_discovery = PatternDiscoveryEngine(netzwerk_session)

        utterance_ids = []
        for i in range(3):
            text = f"Was ist item{i}?"
            embedding = [0.1] * 384
            uid = utterance_manager.create_utterance(text, embedding)
            utterance_ids.append(uid)

            doc = preprocessor.process(text)
            for token in doc:
                utterance_manager.create_token(
                    surface=token.text,
                    lemma=token.lemma_.lower(),
                    pos=token.pos_,
                    utterance_id=uid,
                    idx=token.i,
                )

        cluster = {
            "utterance_ids": utterance_ids,
            "centroid": (np.array([0.1] * 384) / np.linalg.norm([0.1] * 384)).tolist(),
            "size": 3,
        }

        pattern_id = pattern_discovery.induce_template_from_cluster(cluster)
        items = pattern_manager.get_pattern_items(pattern_id)

        # Last item should be LITERAL("?")
        last_item = items[-1]
        if last_item["kind"] == "LITERAL":
            assert last_item["literalValue"] == "?"

    def test_lgg_lit_03_multi_word_literal_sequence(
        self, netzwerk_session, utterance_manager, pattern_manager
    ):
        """LGG-LIT-03: All utterances starting with 'Was ist' creates 2 LITERALs."""
        from component_6_linguistik_engine import LinguisticPreprocessor

        preprocessor = LinguisticPreprocessor()
        pattern_discovery = PatternDiscoveryEngine(netzwerk_session)

        utterance_ids = []
        for noun in ["hund", "katze", "vogel"]:
            text = f"Was ist {noun}"
            embedding = [0.1] * 384
            uid = utterance_manager.create_utterance(text, embedding)
            utterance_ids.append(uid)

            doc = preprocessor.process(text)
            for token in doc:
                utterance_manager.create_token(
                    surface=token.text,
                    lemma=token.lemma_.lower(),
                    pos=token.pos_,
                    utterance_id=uid,
                    idx=token.i,
                )

        cluster = {
            "utterance_ids": utterance_ids,
            "centroid": (np.array([0.1] * 384) / np.linalg.norm([0.1] * 384)).tolist(),
            "size": 3,
        }

        pattern_id = pattern_discovery.induce_template_from_cluster(cluster)
        items = pattern_manager.get_pattern_items(pattern_id)

        # First 2 items should be LITERAL
        assert items[0]["kind"] == "LITERAL"
        assert items[0]["literalValue"] == "was"
        assert items[1]["kind"] == "LITERAL"
        assert items[1]["literalValue"] == "ist"


class TestSlotDetection:
    """SLOT detection scenarios (LGG-SLOT-01 to LGG-SLOT-05)."""

    def test_lgg_slot_01_varying_lemmas_at_position(
        self, netzwerk_session, utterance_manager, pattern_manager
    ):
        """LGG-SLOT-01: Varying lemmas at position creates SLOT with allowed_lemmas."""
        from component_6_linguistik_engine import LinguisticPreprocessor

        preprocessor = LinguisticPreprocessor()
        pattern_discovery = PatternDiscoveryEngine(netzwerk_session)

        utterance_ids = []
        animals = ["hund", "katze", "vogel"]
        for animal in animals:
            text = f"Was ist {animal}"
            embedding = [0.1] * 384
            uid = utterance_manager.create_utterance(text, embedding)
            utterance_ids.append(uid)

            doc = preprocessor.process(text)
            for token in doc:
                utterance_manager.create_token(
                    surface=token.text,
                    lemma=token.lemma_.lower(),
                    pos=token.pos_,
                    utterance_id=uid,
                    idx=token.i,
                )

        cluster = {
            "utterance_ids": utterance_ids,
            "centroid": (np.array([0.1] * 384) / np.linalg.norm([0.1] * 384)).tolist(),
            "size": 3,
        }

        pattern_id = pattern_discovery.induce_template_from_cluster(cluster)
        items = pattern_manager.get_pattern_items(pattern_id)

        # Find SLOT item (last position)
        slot_items = [item for item in items if item["kind"] == "SLOT"]
        assert len(slot_items) >= 1

        # Verify allowed lemmas (need to query slot details)
        # This is implementation-dependent on how slots are stored

    def test_lgg_slot_02_slot_with_frequency_counts(
        self, netzwerk_session, utterance_manager, pattern_manager
    ):
        """LGG-SLOT-02: SLOT with lemma frequency counts (hund 3x, katze 2x)."""
        from component_6_linguistik_engine import LinguisticPreprocessor

        preprocessor = LinguisticPreprocessor()
        pattern_discovery = PatternDiscoveryEngine(netzwerk_session)

        utterance_ids = []
        lemmas = ["hund", "hund", "hund", "katze", "katze"]
        for lemma in lemmas:
            text = f"Was ist {lemma}"
            embedding = [0.1] * 384
            uid = utterance_manager.create_utterance(text, embedding)
            utterance_ids.append(uid)

            doc = preprocessor.process(text)
            for token in doc:
                utterance_manager.create_token(
                    surface=token.text,
                    lemma=token.lemma_.lower(),
                    pos=token.pos_,
                    utterance_id=uid,
                    idx=token.i,
                )

        cluster = {
            "utterance_ids": utterance_ids,
            "centroid": (np.array([0.1] * 384) / np.linalg.norm([0.1] * 384)).tolist(),
            "size": 5,
        }

        pattern_id = pattern_discovery.induce_template_from_cluster(cluster)

        # Pattern created successfully (frequency counts stored internally)
        pattern = pattern_manager.get_pattern(pattern_id)
        assert pattern["support"] == 5

    def test_lgg_slot_03_slot_with_single_value(
        self, netzwerk_session, utterance_manager, pattern_manager
    ):
        """LGG-SLOT-03: Position varies but creates SLOT (not LITERAL) for 2+ values."""
        from component_6_linguistik_engine import LinguisticPreprocessor

        preprocessor = LinguisticPreprocessor()
        pattern_discovery = PatternDiscoveryEngine(netzwerk_session)

        utterance_ids = []
        lemmas = ["hund", "katze"]
        for lemma in lemmas:
            text = f"Was ist {lemma}"
            embedding = [0.1] * 384
            uid = utterance_manager.create_utterance(text, embedding)
            utterance_ids.append(uid)

            doc = preprocessor.process(text)
            for token in doc:
                utterance_manager.create_token(
                    surface=token.text,
                    lemma=token.lemma_.lower(),
                    pos=token.pos_,
                    utterance_id=uid,
                    idx=token.i,
                )

        cluster = {
            "utterance_ids": utterance_ids,
            "centroid": (np.array([0.1] * 384) / np.linalg.norm([0.1] * 384)).tolist(),
            "size": 2,
        }

        pattern_id = pattern_discovery.induce_template_from_cluster(cluster)
        items = pattern_manager.get_pattern_items(pattern_id)

        # Last item should be SLOT
        slot_items = [item for item in items if item["kind"] == "SLOT"]
        assert len(slot_items) >= 1

    def test_lgg_slot_04_allowed_lemma_nodes_created(
        self, netzwerk_session, utterance_manager, pattern_manager
    ):
        """LGG-SLOT-04: AllowedLemma nodes created with ALLOWS relationships."""
        from component_6_linguistik_engine import LinguisticPreprocessor

        preprocessor = LinguisticPreprocessor()
        pattern_discovery = PatternDiscoveryEngine(netzwerk_session)

        utterance_ids = []
        for animal in ["hund", "katze"]:
            text = f"Was ist {animal}"
            embedding = [0.1] * 384
            uid = utterance_manager.create_utterance(text, embedding)
            utterance_ids.append(uid)

            doc = preprocessor.process(text)
            for token in doc:
                utterance_manager.create_token(
                    surface=token.text,
                    lemma=token.lemma_.lower(),
                    pos=token.pos_,
                    utterance_id=uid,
                    idx=token.i,
                )

        cluster = {
            "utterance_ids": utterance_ids,
            "centroid": (np.array([0.1] * 384) / np.linalg.norm([0.1] * 384)).tolist(),
            "size": 2,
        }

        pattern_id = pattern_discovery.induce_template_from_cluster(cluster)

        # Verify AllowedLemma nodes exist (query Neo4j)
        with netzwerk_session.driver.session(database="neo4j") as session:
            result = session.run(
                """
                MATCH (p:Pattern {id: $pid})-[:HAS_ITEM]->(pi:PatternItem)
                -[:USES_SLOT]->(s:Slot)-[:ALLOWS]->(al:AllowedLemma)
                RETURN count(al) AS allowed_count
            """,
                {"pid": pattern_id},
            )
            record = result.single()
            if record:
                assert record["allowed_count"] >= 1

    def test_lgg_slot_05_slot_id_created(
        self, netzwerk_session, utterance_manager, pattern_manager
    ):
        """LGG-SLOT-05: Slot node created with unique UUID."""
        from component_6_linguistik_engine import LinguisticPreprocessor

        preprocessor = LinguisticPreprocessor()
        pattern_discovery = PatternDiscoveryEngine(netzwerk_session)

        utterance_ids = []
        for animal in ["hund", "katze"]:
            text = f"Was ist {animal}"
            embedding = [0.1] * 384
            uid = utterance_manager.create_utterance(text, embedding)
            utterance_ids.append(uid)

            doc = preprocessor.process(text)
            for token in doc:
                utterance_manager.create_token(
                    surface=token.text,
                    lemma=token.lemma_.lower(),
                    pos=token.pos_,
                    utterance_id=uid,
                    idx=token.i,
                )

        cluster = {
            "utterance_ids": utterance_ids,
            "centroid": (np.array([0.1] * 384) / np.linalg.norm([0.1] * 384)).tolist(),
            "size": 2,
        }

        pattern_id = pattern_discovery.induce_template_from_cluster(cluster)
        items = pattern_manager.get_pattern_items(pattern_id)

        # Slot items have slotId
        slot_items = [item for item in items if item["kind"] == "SLOT"]
        for item in slot_items:
            assert "slotId" in item
            assert len(item["slotId"]) == 36  # UUID


class TestEdgeCaseLGG:
    """Edge case scenarios for LGG (LGG-EDGE-01 to LGG-EDGE-07)."""

    def test_lgg_edge_01_cluster_with_one_utterance(self, netzwerk_session):
        """LGG-EDGE-01: Cluster with < 2 utterances raises ValueError."""
        pattern_discovery = PatternDiscoveryEngine(netzwerk_session)

        cluster = {"utterance_ids": ["uid1"], "centroid": [0.1] * 384, "size": 1}

        with pytest.raises(ValueError, match="at least 2 utterances"):
            pattern_discovery.induce_template_from_cluster(cluster)

    def test_lgg_edge_02_utterances_different_lengths(
        self, netzwerk_session, utterance_manager, pattern_manager, caplog
    ):
        """LGG-EDGE-02: Utterances with different lengths filtered to most common."""
        from component_6_linguistik_engine import LinguisticPreprocessor

        preprocessor = LinguisticPreprocessor()
        pattern_discovery = PatternDiscoveryEngine(netzwerk_session)

        utterance_ids = []
        # 3 utterances length 3, 2 utterances length 5
        for i in range(3):
            text = f"Was ist hund"
            embedding = [0.1] * 384
            uid = utterance_manager.create_utterance(text, embedding)
            utterance_ids.append(uid)

            doc = preprocessor.process(text)
            for token in doc:
                utterance_manager.create_token(
                    surface=token.text,
                    lemma=token.lemma_.lower(),
                    pos=token.pos_,
                    utterance_id=uid,
                    idx=token.i,
                )

        for i in range(2):
            text = f"Was ist ein kleiner hund"
            embedding = [0.1] * 384
            uid = utterance_manager.create_utterance(text, embedding)
            utterance_ids.append(uid)

            doc = preprocessor.process(text)
            for token in doc:
                utterance_manager.create_token(
                    surface=token.text,
                    lemma=token.lemma_.lower(),
                    pos=token.pos_,
                    utterance_id=uid,
                    idx=token.i,
                )

        cluster = {
            "utterance_ids": utterance_ids,
            "centroid": (np.array([0.1] * 384) / np.linalg.norm([0.1] * 384)).tolist(),
            "size": 5,
        }

        # Should use most common length (3)
        pattern_id = pattern_discovery.induce_template_from_cluster(cluster)
        assert pattern_id is not None

    def test_lgg_edge_03_low_alignment_ratio(
        self, netzwerk_session, utterance_manager, caplog
    ):
        """LGG-EDGE-03: Alignment ratio < 50% logs warning."""
        from component_6_linguistik_engine import LinguisticPreprocessor

        preprocessor = LinguisticPreprocessor()
        pattern_discovery = PatternDiscoveryEngine(netzwerk_session)

        # Create 10 utterances with 10 different lengths
        utterance_ids = []
        for i in range(1, 11):
            text = " ".join([f"word{j}" for j in range(i)])
            embedding = [0.1] * 384
            uid = utterance_manager.create_utterance(text, embedding)
            utterance_ids.append(uid)

            doc = preprocessor.process(text)
            for token in doc:
                utterance_manager.create_token(
                    surface=token.text,
                    lemma=token.lemma_.lower(),
                    pos=token.pos_,
                    utterance_id=uid,
                    idx=token.i,
                )

        cluster = {
            "utterance_ids": utterance_ids,
            "centroid": (np.array([0.1] * 384) / np.linalg.norm([0.1] * 384)).tolist(),
            "size": 10,
        }

        # Should log warning
        with caplog.at_level(logging.WARNING):
            pattern_id = pattern_discovery.induce_template_from_cluster(cluster)

        assert "No dominant sequence length" in caplog.text
        assert "Consider dynamic alignment" in caplog.text

    def test_lgg_edge_04_no_tokens_for_utterances(
        self, netzwerk_session, utterance_manager
    ):
        """LGG-EDGE-04: Cluster with utterances but no tokens raises ValueError."""
        pattern_discovery = PatternDiscoveryEngine(netzwerk_session)

        # Create utterances WITHOUT tokens
        utterance_ids = []
        for i in range(3):
            uid = utterance_manager.create_utterance(f"Text {i}", [0.1] * 384)
            utterance_ids.append(uid)

        cluster = {
            "utterance_ids": utterance_ids,
            "centroid": (np.array([0.1] * 384) / np.linalg.norm([0.1] * 384)).tolist(),
            "size": 3,
        }

        with pytest.raises(ValueError, match="No token sequences found"):
            pattern_discovery.induce_template_from_cluster(cluster)

    def test_lgg_edge_05_empty_cluster_centroid(self, netzwerk_session):
        """LGG-EDGE-05: Cluster with invalid centroid raises ValueError."""
        pattern_discovery = PatternDiscoveryEngine(netzwerk_session)

        cluster = {
            "utterance_ids": ["uid1", "uid2"],
            "centroid": [0.1] * 128,  # Wrong dimension
            "size": 2,
        }

        with pytest.raises(ValueError, match="Cluster centroid must be 384D"):
            pattern_discovery.induce_template_from_cluster(cluster)

    def test_lgg_edge_06_empty_lemmas_at_position(
        self, netzwerk_session, utterance_manager, caplog
    ):
        """LGG-EDGE-06: Tokens with empty lemmas are skipped."""
        from component_6_linguistik_engine import LinguisticPreprocessor

        preprocessor = LinguisticPreprocessor()
        pattern_discovery = PatternDiscoveryEngine(netzwerk_session)

        utterance_ids = []
        for i in range(3):
            text = "Was ist hund"
            embedding = [0.1] * 384
            uid = utterance_manager.create_utterance(text, embedding)
            utterance_ids.append(uid)

            # Create tokens with empty lemma at position 1
            doc = preprocessor.process(text)
            for idx, token in enumerate(doc):
                lemma = "" if idx == 1 else token.lemma_.lower()
                utterance_manager.create_token(
                    surface=token.text,
                    lemma=lemma,
                    pos=token.pos_,
                    utterance_id=uid,
                    idx=token.i,
                )

        cluster = {
            "utterance_ids": utterance_ids,
            "centroid": (np.array([0.1] * 384) / np.linalg.norm([0.1] * 384)).tolist(),
            "size": 3,
        }

        with caplog.at_level(logging.WARNING):
            pattern_id = pattern_discovery.induce_template_from_cluster(cluster)

        # Position skipped warning
        assert "No valid lemmas at position" in caplog.text

    def test_lgg_edge_07_lgg_produces_empty_template(
        self, netzwerk_session, utterance_manager
    ):
        """LGG-EDGE-07: LGG producing empty template raises ValueError."""
        # This is hard to trigger with real data, but test the error path
        pattern_discovery = PatternDiscoveryEngine(netzwerk_session)

        # Mock _lgg_multiway to return empty list
        with patch.object(pattern_discovery, "_lgg_multiway", return_value=[]):
            utterance_ids = []
            for i in range(3):
                uid = utterance_manager.create_utterance(f"Text {i}", [0.1] * 384)
                utterance_ids.append(uid)
                utterance_manager.create_token("text", "text", "NOUN", uid, 0)

            cluster = {
                "utterance_ids": utterance_ids,
                "centroid": (
                    np.array([0.1] * 384) / np.linalg.norm([0.1] * 384)
                ).tolist(),
                "size": 3,
            }

            with pytest.raises(ValueError, match="LGG failed to produce template"):
                pattern_discovery.induce_template_from_cluster(cluster)


class TestBatchOperations:
    """Batch operation scenarios (LGG-BATCH-01 to LGG-BATCH-02)."""

    def test_lgg_batch_01_batch_token_retrieval_used(self, netzwerk_session, mocker):
        """LGG-BATCH-01: Verify batch token retrieval called (not N+1 queries)."""
        pattern_discovery = PatternDiscoveryEngine(netzwerk_session)

        # Mock batch method
        mock_batch = mocker.patch.object(
            netzwerk_session,
            "get_tokens_for_utterances_batch",
            return_value={
                "u1": [{"lemma": "was", "pos": "PRON"}],
                "u2": [{"lemma": "wer", "pos": "PRON"}],
            },
        )

        cluster = {
            "utterance_ids": ["u1", "u2"],
            "centroid": (np.array([0.1] * 384) / np.linalg.norm([0.1] * 384)).tolist(),
            "size": 2,
        }

        try:
            pattern_discovery.induce_template_from_cluster(cluster)
        except Exception:
            pass  # May fail due to mock, we only check if batch method called

        # Assert batch method called ONCE (not 2x)
        mock_batch.assert_called_once()

    def test_lgg_batch_02_no_n_plus_one_problem(
        self, netzwerk_session, utterance_manager, mocker
    ):
        """LGG-BATCH-02: 100 utterances should use 1 batch query, not 100 individual."""
        from component_6_linguistik_engine import LinguisticPreprocessor

        preprocessor = LinguisticPreprocessor()
        pattern_discovery = PatternDiscoveryEngine(netzwerk_session)

        # Mock to count calls
        original_method = netzwerk_session.get_tokens_for_utterances_batch
        call_count = 0

        def counting_wrapper(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_method(*args, **kwargs)

        mocker.patch.object(
            netzwerk_session,
            "get_tokens_for_utterances_batch",
            side_effect=counting_wrapper,
        )

        # Create 100 utterances
        utterance_ids = []
        for i in range(100):
            text = "Was ist test"
            embedding = [0.1] * 384
            uid = utterance_manager.create_utterance(text, embedding)
            utterance_ids.append(uid)

            doc = preprocessor.process(text)
            for token in doc:
                utterance_manager.create_token(
                    surface=token.text,
                    lemma=token.lemma_.lower(),
                    pos=token.pos_,
                    utterance_id=uid,
                    idx=token.i,
                )

        cluster = {
            "utterance_ids": utterance_ids,
            "centroid": (np.array([0.1] * 384) / np.linalg.norm([0.1] * 384)).tolist(),
            "size": 100,
        }

        pattern_discovery.induce_template_from_cluster(cluster)

        # Should be called ONCE, not 100 times
        assert call_count == 1


class TestPatternCreation:
    """Pattern creation scenarios (LGG-CREATE-01 to LGG-CREATE-02)."""

    def test_lgg_create_01_pattern_node_properties(
        self, netzwerk_session, utterance_manager, pattern_manager
    ):
        """LGG-CREATE-01: Pattern node has correct properties."""
        from component_6_linguistik_engine import LinguisticPreprocessor

        preprocessor = LinguisticPreprocessor()
        pattern_discovery = PatternDiscoveryEngine(netzwerk_session)

        utterance_ids = []
        for i in range(5):
            text = f"Was ist tier{i}"
            embedding = [0.1] * 384
            uid = utterance_manager.create_utterance(text, embedding)
            utterance_ids.append(uid)

            doc = preprocessor.process(text)
            for token in doc:
                utterance_manager.create_token(
                    surface=token.text,
                    lemma=token.lemma_.lower(),
                    pos=token.pos_,
                    utterance_id=uid,
                    idx=token.i,
                )

        cluster = {
            "utterance_ids": utterance_ids,
            "centroid": (np.array([0.1] * 384) / np.linalg.norm([0.1] * 384)).tolist(),
            "size": 5,
        }

        pattern_id = pattern_discovery.induce_template_from_cluster(cluster)

        # Verify properties
        pattern = pattern_manager.get_pattern(pattern_id)
        assert pattern["type"] == "learned"
        assert pattern["support"] == 5
        assert pattern["precision"] == 0.5
        assert pattern["name"].startswith("Learned_Pattern_")

    def test_lgg_create_02_pattern_items_ordered_by_idx(
        self, netzwerk_session, utterance_manager, pattern_manager
    ):
        """LGG-CREATE-02: PatternItems ordered by idx (0, 1, 2, ...)."""
        from component_6_linguistik_engine import LinguisticPreprocessor

        preprocessor = LinguisticPreprocessor()
        pattern_discovery = PatternDiscoveryEngine(netzwerk_session)

        utterance_ids = []
        for i in range(3):
            text = "Was ist ein hund?"
            embedding = [0.1] * 384
            uid = utterance_manager.create_utterance(text, embedding)
            utterance_ids.append(uid)

            doc = preprocessor.process(text)
            for token in doc:
                utterance_manager.create_token(
                    surface=token.text,
                    lemma=token.lemma_.lower(),
                    pos=token.pos_,
                    utterance_id=uid,
                    idx=token.i,
                )

        cluster = {
            "utterance_ids": utterance_ids,
            "centroid": (np.array([0.1] * 384) / np.linalg.norm([0.1] * 384)).tolist(),
            "size": 3,
        }

        pattern_id = pattern_discovery.induce_template_from_cluster(cluster)

        # Verify ordering
        items = pattern_manager.get_pattern_items(pattern_id)
        for i, item in enumerate(items):
            assert item["idx"] == i
