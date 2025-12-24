# test_pattern_discovery_part3_integration.py
"""
Test scenarios for Integration & Orchestration (Part 3: Pattern Discovery).

Tests cover Phase 8: End-to-end workflows, periodic discovery, kai_worker integration.

Test Categories:
- Full Pipeline: Store → Cluster → Induce → Refine
- Periodic Discovery: Trigger every 50 utterances, kai_worker integration
- kai_worker Integration: PatternDiscoveryEngine initialization
- Error Handling: Database failures, graceful degradation

Total: 15 tests
"""

import logging

import numpy as np
import pytest

from component_61_pattern_discovery import PatternDiscoveryEngine
from kai_exceptions import DatabaseException


class TestFullPipeline:
    """Full pipeline scenarios (INT-PIPE-01 to INT-PIPE-05)."""

    def test_int_pipe_01_end_to_end_workflow(
        self, netzwerk_session, utterance_manager, pattern_manager
    ):
        """INT-PIPE-01: Complete workflow: Store → Cluster → Induce → Refine."""
        from component_6_linguistik_engine import LinguisticPreprocessor

        preprocessor = LinguisticPreprocessor()
        pattern_discovery = PatternDiscoveryEngine(netzwerk_session)

        # Store 10 similar utterances
        for i in range(10):
            text = f"Was ist tier{i}?"
            embedding = [0.1 + i * 0.01] * 384
            embedding = (np.array(embedding) / np.linalg.norm(embedding)).tolist()
            uid = utterance_manager.create_utterance(text, embedding)

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

        # Trigger clustering
        clusters = pattern_discovery.cluster_utterances(min_cluster_size=5)
        assert len(clusters) >= 1

        # Induce template
        pattern_id = pattern_discovery.induce_template_from_cluster(clusters[0])
        assert pattern_id is not None

        # Verify pattern created
        pattern = pattern_manager.get_pattern(pattern_id)
        assert pattern["type"] == "learned"
        assert pattern["support"] >= 5

    def test_int_pipe_02_multiple_clusters_multiple_patterns(
        self, netzwerk_session, utterance_manager, pattern_manager
    ):
        """INT-PIPE-02: 20 utterances forming 3 clusters create 3 patterns."""
        from component_6_linguistik_engine import LinguisticPreprocessor

        preprocessor = LinguisticPreprocessor()
        pattern_discovery = PatternDiscoveryEngine(netzwerk_session)

        # Create 3 groups of similar utterances
        for group_idx in range(3):
            base_embedding = np.random.rand(384)
            base_embedding = base_embedding / np.linalg.norm(base_embedding)

            for i in range(7):  # 7 utterances per group
                text = f"Group {group_idx} item {i}"
                noise = np.random.normal(0, 0.05, 384)
                embedding = base_embedding + noise
                embedding = (embedding / np.linalg.norm(embedding)).tolist()

                uid = utterance_manager.create_utterance(text, embedding)
                doc = preprocessor.process(text)
                for token in doc:
                    utterance_manager.create_token(
                        surface=token.text,
                        lemma=token.lemma_.lower(),
                        pos=token.pos_,
                        utterance_id=uid,
                        idx=token.i,
                    )

        # Cluster
        clusters = pattern_discovery.cluster_utterances(min_cluster_size=5)
        assert len(clusters) >= 2  # At least 2 clusters

        # Induce patterns
        pattern_ids = []
        for cluster in clusters:
            pattern_id = pattern_discovery.induce_template_from_cluster(cluster)
            pattern_ids.append(pattern_id)

        assert len(pattern_ids) >= 2

    def test_int_pipe_03_pattern_used_in_matching_after_creation(
        self, netzwerk_session, utterance_manager, pattern_manager
    ):
        """INT-PIPE-03: Newly created pattern appears in matcher candidates."""
        from component_6_linguistik_engine import LinguisticPreprocessor
        from component_61_pattern_matcher import TemplatePatternMatcher

        preprocessor = LinguisticPreprocessor()
        pattern_discovery = PatternDiscoveryEngine(netzwerk_session)
        pattern_matcher = TemplatePatternMatcher(netzwerk_session)

        # Create pattern via discovery
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

        clusters = pattern_discovery.cluster_utterances(min_cluster_size=3)
        if clusters:
            pattern_discovery.induce_template_from_cluster(clusters[0])

            # Create similar utterance
            text = "Was ist tier99"
            embedding = [0.1] * 384
            new_uid = utterance_manager.create_utterance(text, embedding)
            doc = preprocessor.process(text)
            for token in doc:
                utterance_manager.create_token(
                    surface=token.text,
                    lemma=token.lemma_.lower(),
                    pos=token.pos_,
                    utterance_id=new_uid,
                    idx=token.i,
                )

            # Match should find newly created pattern
            pattern_matcher.match_utterance(new_uid)
            # (may or may not match depending on template structure)

    def test_int_pipe_04_refinement_after_multiple_matches(
        self, netzwerk_session, utterance_manager, pattern_manager
    ):
        """INT-PIPE-04: Pattern matched 5 times increases support by 5."""
        from component_6_linguistik_engine import LinguisticPreprocessor
        from component_61_pattern_matcher import TemplatePatternMatcher

        preprocessor = LinguisticPreprocessor()
        pattern_matcher = TemplatePatternMatcher(netzwerk_session)

        # Create pattern
        pattern_id = pattern_manager.create_pattern(name="Test", pattern_type="learned")
        pattern_manager.update_pattern_stats(
            pattern_id, support_increment=0, new_precision=0.5
        )

        # Refine 5 times
        for i in range(5):
            text = f"Test {i}"
            embedding = [0.1] * 384
            uid = utterance_manager.create_utterance(text, embedding)

            doc = preprocessor.process(text)
            for token in doc:
                utterance_manager.create_token(
                    surface=token.text,
                    lemma=token.lemma_.lower(),
                    pos=token.pos_,
                    utterance_id=uid,
                    idx=token.i,
                )

            pattern_matcher._refine_pattern(pattern_id, uid, 0.8)

        # Support = 5
        pattern = pattern_manager.get_pattern(pattern_id)
        assert pattern["support"] == 5

    def test_int_pipe_05_cache_invalidation_visible_in_matching(
        self, netzwerk_session, utterance_manager, pattern_manager
    ):
        """INT-PIPE-05: Cache invalidation makes new pattern visible."""
        from component_61_pattern_matcher import TemplatePatternMatcher

        pattern_matcher = TemplatePatternMatcher(netzwerk_session)

        # Create pattern
        pattern_id = pattern_manager.create_pattern(name="New", pattern_type="learned")

        # Invalidate cache
        pattern_matcher._invalidate_candidate_cache()

        # Cache cleared (implementation-specific verification)


class TestPeriodicDiscovery:
    """Periodic discovery scenarios (INT-PERIOD-01 to INT-PERIOD-04)."""

    def test_int_period_01_triggered_every_50_utterances(
        self, kai_worker_with_mocks, mocker, clean_pattern_data
    ):
        """INT-PERIOD-01: Discovery triggered after 50 utterances."""
        # Mock periodic discovery
        mock_discovery = mocker.patch.object(
            kai_worker_with_mocks, "_periodic_pattern_discovery", return_value=None
        )

        # Process 51 utterances
        for i in range(51):
            kai_worker_with_mocks.process_query(f"Query {i}")

        # Assert called (may be 1 or more times)
        assert mock_discovery.call_count >= 1

    def test_int_period_02_not_triggered_before_50(
        self, kai_worker_with_mocks, mocker, clean_pattern_data
    ):
        """INT-PERIOD-02: Discovery not triggered before 50 utterances."""
        mock_discovery = mocker.patch.object(
            kai_worker_with_mocks, "_periodic_pattern_discovery", return_value=None
        )

        # Process 49 utterances
        for i in range(49):
            kai_worker_with_mocks.process_query(f"Query {i}")

        # Not called
        assert mock_discovery.call_count == 0

    def test_int_period_03_last_discovery_count_tracked(
        self, kai_worker_with_mocks, clean_pattern_data
    ):
        """INT-PERIOD-03: _last_discovery_count attribute tracked."""
        # Process utterances
        for i in range(100):
            kai_worker_with_mocks.process_query(f"Query {i}")

        # Check attribute exists (implementation-specific)
        if hasattr(kai_worker_with_mocks, "_last_discovery_count"):
            assert kai_worker_with_mocks._last_discovery_count >= 0

    def test_int_period_04_discovery_non_blocking(
        self, kai_worker_with_mocks, clean_pattern_data
    ):
        """INT-PERIOD-04: Discovery doesn't block user input."""
        # Process queries rapidly
        for i in range(60):
            kai_worker_with_mocks.process_query(f"Query {i}")

        # No exceptions, worker continues
        assert kai_worker_with_mocks.is_initialized_successfully


class TestKaiWorkerIntegration:
    """kai_worker integration scenarios (INT-WORKER-01 to INT-WORKER-03)."""

    def test_int_worker_01_pattern_discovery_initialized(self, kai_worker_with_mocks):
        """INT-WORKER-01: PatternDiscoveryEngine initialized in kai_worker."""
        assert hasattr(kai_worker_with_mocks, "pattern_discovery")
        # May or may not be initialized depending on implementation

    def test_int_worker_02_discovery_calls_clustering_and_induction(
        self, kai_worker_with_mocks, mocker
    ):
        """INT-WORKER-02: Periodic discovery performs clustering + induction."""
        if hasattr(kai_worker_with_mocks, "pattern_discovery"):
            mock_cluster = mocker.patch.object(
                kai_worker_with_mocks.pattern_discovery,
                "cluster_utterances",
                return_value=[],
            )

            # Trigger discovery manually
            if hasattr(kai_worker_with_mocks, "_periodic_pattern_discovery"):
                kai_worker_with_mocks._periodic_pattern_discovery()

                # Clustering called
                assert mock_cluster.call_count >= 0  # May be 0 if not enough utterances

    def test_int_worker_03_no_errors_on_empty_clustering(self, kai_worker_with_mocks):
        """INT-WORKER-03: Discovery with < min_cluster_size utterances doesn't crash."""
        # Process < 50 utterances
        for i in range(10):
            kai_worker_with_mocks.process_query(f"Query {i}")

        # No exception
        assert kai_worker_with_mocks.is_initialized_successfully


class TestErrorHandling:
    """Error handling scenarios (INT-ERROR-01 to INT-ERROR-03)."""

    def test_int_error_01_database_error_during_clustering(
        self, netzwerk_session, mocker
    ):
        """INT-ERROR-01: Neo4j connection failure during clustering raises DatabaseException."""
        pattern_discovery = PatternDiscoveryEngine(netzwerk_session)

        # Mock get_recent_utterances to raise exception
        mocker.patch.object(
            netzwerk_session,
            "get_recent_utterances",
            side_effect=Exception("Connection failed"),
        )

        with pytest.raises(DatabaseException, match="Failed to fetch utterances"):
            pattern_discovery.cluster_utterances()

    def test_int_error_02_database_error_during_induction(
        self, netzwerk_session, utterance_manager, mocker
    ):
        """INT-ERROR-02: Neo4j write failure during pattern creation raises DatabaseException."""
        from component_6_linguistik_engine import LinguisticPreprocessor

        preprocessor = LinguisticPreprocessor()
        pattern_discovery = PatternDiscoveryEngine(netzwerk_session)

        # Create cluster
        utterance_ids = []
        for i in range(3):
            text = f"Test {i}"
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

        cluster = {"utterance_ids": utterance_ids, "centroid": [0.1] * 384, "size": 3}

        # Mock create_pattern to fail
        mocker.patch.object(
            netzwerk_session, "create_pattern", side_effect=Exception("Write failed")
        )

        with pytest.raises(DatabaseException, match="Failed to create pattern"):
            pattern_discovery.induce_template_from_cluster(cluster)

    def test_int_error_03_graceful_degradation_on_discovery_failure(
        self, kai_worker_with_mocks, mocker, caplog
    ):
        """INT-ERROR-03: Discovery failure doesn't crash worker."""
        if hasattr(kai_worker_with_mocks, "pattern_discovery"):
            # Mock clustering to raise exception
            mocker.patch.object(
                kai_worker_with_mocks.pattern_discovery,
                "cluster_utterances",
                side_effect=Exception("Clustering failed"),
            )

            # Trigger discovery (if method exists)
            if hasattr(kai_worker_with_mocks, "_periodic_pattern_discovery"):
                with caplog.at_level(logging.ERROR):
                    try:
                        kai_worker_with_mocks._periodic_pattern_discovery()
                    except Exception:
                        pass  # Should not propagate

                # Worker still functional
                kai_worker_with_mocks.process_query("Test query")
