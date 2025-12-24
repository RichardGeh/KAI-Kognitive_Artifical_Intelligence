# test_pattern_discovery_part3_clustering.py
"""
Test scenarios for Clustering (Part 3: Pattern Discovery).

Tests cover Phase 5: Embedding-based clustering of utterances.

Test Categories:
- Happy Path: Clear groups, moderate similarity, single cluster, empty results
- Edge Cases: Insufficient utterances, identical/different embeddings, missing embeddings
- Parameter Validation: min_cluster_size, similarity_threshold validation
- Performance: 1000 utterances benchmark

Total: 20 tests
"""

import logging
import time

import numpy as np
import pytest

from component_61_pattern_discovery import PatternDiscoveryEngine


class TestHappyPathClustering:
    """Happy path scenarios for clustering (C-HP-01 to C-HP-05)."""

    def test_c_hp_01_cluster_clear_groups(
        self, netzwerk_session, utterance_manager, embedding_service_session
    ):
        """C-HP-01: Cluster 10 utterances forming 2 clear groups (WH-questions vs declaratives)."""
        pattern_discovery = PatternDiscoveryEngine(netzwerk_session)

        # Create WH-question utterances with similar embeddings
        wh_utterances = []
        for query in [
            "Was ist ein Hund?",
            "Wer ist Peter?",
            "Wie ist das Wetter?",
            "Was ist eine Katze?",
            "Wer ist Maria?",
        ]:
            embedding = embedding_service_session.get_embedding(query)
            uid = utterance_manager.create_utterance(query, embedding)
            wh_utterances.append(uid)

        # Create declarative utterances with similar embeddings
        decl_utterances = []
        for stmt in [
            "Hund ist Tier.",
            "Peter ist Mensch.",
            "Wetter ist gut.",
            "Katze ist Tier.",
            "Maria ist Mensch.",
        ]:
            embedding = embedding_service_session.get_embedding(stmt)
            uid = utterance_manager.create_utterance(stmt, embedding)
            decl_utterances.append(uid)

        # Cluster
        clusters = pattern_discovery.cluster_utterances(
            min_cluster_size=3,
            similarity_threshold=0.75,  # Lower threshold for real embeddings
        )

        # Assert
        assert len(clusters) >= 1  # At least 1 cluster
        for cluster in clusters:
            assert len(cluster["utterance_ids"]) >= 3
            assert len(cluster["centroid"]) == 384
            # Verify normalized
            centroid_norm = np.linalg.norm(cluster["centroid"])
            assert 0.95 <= centroid_norm <= 1.05

    def test_c_hp_02_cluster_moderate_similarity(
        self, netzwerk_session, utterance_manager, embedding_384d
    ):
        """C-HP-02: Cluster 15 utterances with moderate similarity."""
        pattern_discovery = PatternDiscoveryEngine(netzwerk_session)

        # Create 15 utterances with slight variations in embeddings
        base_embedding = np.array(embedding_384d)
        base_embedding = base_embedding / np.linalg.norm(base_embedding)

        for i in range(15):
            # Add small noise to create moderate similarity
            noise = np.random.normal(0, 0.1, 384)
            embedding = base_embedding + noise
            embedding = (embedding / np.linalg.norm(embedding)).tolist()

            utterance_manager.create_utterance(f"Utterance {i}", embedding)

        # Cluster with moderate threshold
        clusters = pattern_discovery.cluster_utterances(
            min_cluster_size=3, similarity_threshold=0.80
        )

        # Expect 1-5 clusters
        assert 0 <= len(clusters) <= 5
        for cluster in clusters:
            assert 3 <= len(cluster["utterance_ids"]) <= 15

    def test_c_hp_03_single_large_cluster(
        self, netzwerk_session, utterance_manager, embedding_384d
    ):
        """C-HP-03: 20 nearly identical utterances form single cluster."""
        pattern_discovery = PatternDiscoveryEngine(netzwerk_session)

        # Create 20 utterances with identical embeddings
        for i in range(20):
            utterance_manager.create_utterance(f"Was ist X{i}?", embedding_384d)

        # Cluster
        clusters = pattern_discovery.cluster_utterances(
            min_cluster_size=3, similarity_threshold=0.85
        )

        # Assert single cluster with all 20
        assert len(clusters) == 1
        assert clusters[0]["size"] == 20

    def test_c_hp_04_empty_result_high_threshold(
        self, netzwerk_session, utterance_manager
    ):
        """C-HP-04: High threshold (0.95) with diverse utterances yields no clusters."""
        pattern_discovery = PatternDiscoveryEngine(netzwerk_session)

        # Create 10 diverse utterances with orthogonal embeddings
        for i in range(10):
            # Create orthogonal vectors
            embedding = np.zeros(384)
            embedding[i * 38] = 1.0  # Sparse, orthogonal vectors
            utterance_manager.create_utterance(f"Diverse {i}", embedding.tolist())

        # Cluster with very high threshold
        clusters = pattern_discovery.cluster_utterances(
            min_cluster_size=3, similarity_threshold=0.95
        )

        # No clusters meet threshold
        assert len(clusters) == 0

    def test_c_hp_05_multiple_small_clusters_filtered(
        self, netzwerk_session, utterance_manager
    ):
        """C-HP-05: 20 utterances forming 10 pairs, min_size=3 filters all."""
        pattern_discovery = PatternDiscoveryEngine(netzwerk_session)

        # Create 10 pairs of identical utterances
        for pair_idx in range(10):
            embedding = np.zeros(384)
            embedding[pair_idx * 38] = 1.0
            embedding = (embedding / np.linalg.norm(embedding)).tolist()

            for i in range(2):
                utterance_manager.create_utterance(
                    f"Pair {pair_idx} item {i}", embedding
                )

        # Cluster
        clusters = pattern_discovery.cluster_utterances(
            min_cluster_size=3, similarity_threshold=0.85
        )

        # All clusters filtered (size < 3)
        assert len(clusters) == 0


class TestEdgeCaseClustering:
    """Edge case scenarios for clustering (C-EDGE-01 to C-EDGE-08)."""

    def test_c_edge_01_insufficient_utterances(
        self, netzwerk_session, utterance_manager, embedding_384d, caplog
    ):
        """C-EDGE-01: Less than min_cluster_size utterances returns empty list."""
        pattern_discovery = PatternDiscoveryEngine(netzwerk_session)

        # Create only 2 utterances
        utterance_manager.create_utterance("Utterance 1", embedding_384d)
        utterance_manager.create_utterance("Utterance 2", embedding_384d)

        with caplog.at_level(logging.INFO):
            clusters = pattern_discovery.cluster_utterances(min_cluster_size=3)

        # Assert empty result
        assert clusters == []
        assert "Insufficient utterances for clustering: 2 < 3" in caplog.text

    def test_c_edge_02_exactly_min_cluster_size(
        self, netzwerk_session, utterance_manager, embedding_384d
    ):
        """C-EDGE-02: Exactly min_cluster_size similar utterances form 1 cluster."""
        pattern_discovery = PatternDiscoveryEngine(netzwerk_session)

        # Create exactly 3 identical utterances
        for i in range(3):
            utterance_manager.create_utterance(f"Similar {i}", embedding_384d)

        clusters = pattern_discovery.cluster_utterances(min_cluster_size=3)

        # Expect 1 cluster with size 3
        assert len(clusters) == 1
        assert clusters[0]["size"] == 3

    def test_c_edge_03_all_utterances_identical(
        self, netzwerk_session, utterance_manager, embedding_384d
    ):
        """C-EDGE-03: 10 utterances with identical embeddings form 1 cluster."""
        pattern_discovery = PatternDiscoveryEngine(netzwerk_session)

        # Create 10 identical embeddings
        for i in range(10):
            utterance_manager.create_utterance(f"Identical {i}", embedding_384d)

        clusters = pattern_discovery.cluster_utterances(min_cluster_size=3)

        # Single cluster
        assert len(clusters) == 1
        assert clusters[0]["size"] == 10

        # Centroid equals individual embeddings (normalized)
        centroid = np.array(clusters[0]["centroid"])
        expected = np.array(embedding_384d)
        expected = expected / np.linalg.norm(expected)
        assert np.allclose(centroid, expected, atol=0.01)

    def test_c_edge_04_all_utterances_different(
        self, netzwerk_session, utterance_manager
    ):
        """C-EDGE-04: 10 utterances with orthogonal embeddings yield no clusters."""
        pattern_discovery = PatternDiscoveryEngine(netzwerk_session)

        # Create orthogonal embeddings (cosine similarity < 0.5)
        for i in range(10):
            embedding = np.zeros(384)
            embedding[i * 30 : i * 30 + 30] = 1.0  # Non-overlapping segments
            embedding = (embedding / np.linalg.norm(embedding)).tolist()
            utterance_manager.create_utterance(f"Different {i}", embedding)

        clusters = pattern_discovery.cluster_utterances(
            min_cluster_size=3, similarity_threshold=0.85
        )

        # No clusters (all below threshold)
        assert len(clusters) == 0

    def test_c_edge_05_missing_embeddings(
        self, netzwerk_session, utterance_manager, embedding_384d, caplog
    ):
        """C-EDGE-05: Utterances with missing embeddings are skipped."""
        pattern_discovery = PatternDiscoveryEngine(netzwerk_session)

        # Create 5 utterances: 3 valid, 2 with invalid embeddings
        for i in range(3):
            utterance_manager.create_utterance(f"Valid {i}", embedding_384d)

        # Invalid embeddings (wrong dimension)
        utterance_manager.create_utterance("Invalid 1", [0.1] * 128)
        utterance_manager.create_utterance("Invalid 2", None)

        with caplog.at_level(logging.WARNING):
            clusters = pattern_discovery.cluster_utterances(min_cluster_size=3)

        # 3 valid utterances should cluster
        assert len(clusters) == 1
        assert clusters[0]["size"] == 3

        # Warning logged for invalid embeddings
        assert "missing valid embedding" in caplog.text

    def test_c_edge_06_empty_utterance_list(self, netzwerk_session, caplog):
        """C-EDGE-06: No utterances returns empty list."""
        pattern_discovery = PatternDiscoveryEngine(netzwerk_session)

        with caplog.at_level(logging.INFO):
            clusters = pattern_discovery.cluster_utterances(min_cluster_size=3)

        assert clusters == []
        assert "Insufficient utterances" in caplog.text

    def test_c_edge_07_normalized_centroids_always(
        self, netzwerk_session, utterance_manager
    ):
        """C-EDGE-07: Cluster centroids always normalized even with unnormalized embeddings."""
        pattern_discovery = PatternDiscoveryEngine(netzwerk_session)

        # Create embeddings with unnormalized vectors (norm = 2.0)
        for i in range(10):
            embedding = np.array([0.2] * 384)  # Norm = sqrt(384 * 0.04) ~= 3.92
            utterance_manager.create_utterance(f"Unnormalized {i}", embedding.tolist())

        clusters = pattern_discovery.cluster_utterances(min_cluster_size=3)

        # Centroids should still be normalized
        assert len(clusters) >= 1
        for cluster in clusters:
            centroid_norm = np.linalg.norm(cluster["centroid"])
            assert 0.95 <= centroid_norm <= 1.05

    def test_c_edge_08_zero_norm_centroid_skipped(
        self, netzwerk_session, utterance_manager, caplog
    ):
        """C-EDGE-08: Cluster with zero-norm centroid is skipped."""
        pattern_discovery = PatternDiscoveryEngine(netzwerk_session)

        # Create utterances with all-zero embeddings (pathological)
        for i in range(10):
            utterance_manager.create_utterance(f"Zero {i}", [0.0] * 384)

        with caplog.at_level(logging.WARNING):
            clusters = pattern_discovery.cluster_utterances(min_cluster_size=3)

        # Cluster skipped
        assert len(clusters) == 0
        assert "zero-norm centroid" in caplog.text


class TestParameterValidation:
    """Parameter validation scenarios (C-VALID-01 to C-VALID-04)."""

    def test_c_valid_01_min_cluster_size_below_two(self, netzwerk_session):
        """C-VALID-01: min_cluster_size < 2 raises ValueError."""
        pattern_discovery = PatternDiscoveryEngine(netzwerk_session)

        with pytest.raises(ValueError, match="Minimum cluster size must be >= 2"):
            pattern_discovery.cluster_utterances(min_cluster_size=1)

    def test_c_valid_02_min_cluster_size_zero(self, netzwerk_session):
        """C-VALID-02: min_cluster_size = 0 raises ValueError."""
        pattern_discovery = PatternDiscoveryEngine(netzwerk_session)

        with pytest.raises(ValueError, match="Minimum cluster size must be >= 2"):
            pattern_discovery.cluster_utterances(min_cluster_size=0)

    def test_c_valid_03_similarity_threshold_zero(self, netzwerk_session):
        """C-VALID-03: similarity_threshold <= 0 raises ValueError."""
        pattern_discovery = PatternDiscoveryEngine(netzwerk_session)

        with pytest.raises(ValueError, match="Similarity threshold must be in"):
            pattern_discovery.cluster_utterances(similarity_threshold=0.0)

    def test_c_valid_04_similarity_threshold_above_one(self, netzwerk_session):
        """C-VALID-04: similarity_threshold > 1.0 raises ValueError."""
        pattern_discovery = PatternDiscoveryEngine(netzwerk_session)

        with pytest.raises(ValueError, match="Similarity threshold must be in"):
            pattern_discovery.cluster_utterances(similarity_threshold=1.5)


class TestPerformanceClustering:
    """Performance tests for clustering (C-PERF-01 to C-PERF-03)."""

    @pytest.mark.slow
    def test_c_perf_01_cluster_1000_utterances(
        self, netzwerk_session, utterance_manager
    ):
        """C-PERF-01: Cluster 1000 utterances completes within 10 seconds."""
        pattern_discovery = PatternDiscoveryEngine(netzwerk_session)

        # Create 1000 utterances with random embeddings
        for i in range(1000):
            embedding = np.random.rand(384)
            embedding = (embedding / np.linalg.norm(embedding)).tolist()
            utterance_manager.create_utterance(f"Utterance {i}", embedding)

        # Cluster with timeout
        start = time.time()
        clusters = pattern_discovery.cluster_utterances(
            min_cluster_size=5, similarity_threshold=0.85
        )
        elapsed = time.time() - start

        # Assert performance
        assert elapsed < 10.0, f"Clustering took {elapsed:.2f}s (limit: 10s)"
        assert isinstance(clusters, list)

    def test_c_perf_02_cluster_100_utterances_baseline(
        self, netzwerk_session, utterance_manager
    ):
        """C-PERF-02: Cluster 100 utterances completes within 2 seconds."""
        pattern_discovery = PatternDiscoveryEngine(netzwerk_session)

        # Create 100 utterances
        for i in range(100):
            embedding = np.random.rand(384)
            embedding = (embedding / np.linalg.norm(embedding)).tolist()
            utterance_manager.create_utterance(f"Utterance {i}", embedding)

        start = time.time()
        clusters = pattern_discovery.cluster_utterances(
            min_cluster_size=3, similarity_threshold=0.85
        )
        elapsed = time.time() - start

        assert elapsed < 2.0, f"Clustering took {elapsed:.2f}s (limit: 2s)"

    def test_c_perf_03_centroid_normalization_efficient(
        self, netzwerk_session, utterance_manager, embedding_384d
    ):
        """C-PERF-03: Centroid normalization adds no performance regression."""
        pattern_discovery = PatternDiscoveryEngine(netzwerk_session)

        # Create 50 utterances
        for i in range(50):
            utterance_manager.create_utterance(f"Utterance {i}", embedding_384d)

        start = time.time()
        clusters = pattern_discovery.cluster_utterances(min_cluster_size=3)
        elapsed = time.time() - start

        # All centroids normalized
        for cluster in clusters:
            centroid_norm = np.linalg.norm(cluster["centroid"])
            assert 0.95 <= centroid_norm <= 1.05

        # No performance regression
        assert elapsed < 1.0, f"Clustering took {elapsed:.2f}s (limit: 1s)"
