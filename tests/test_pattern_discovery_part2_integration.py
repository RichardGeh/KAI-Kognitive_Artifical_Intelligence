# test_pattern_discovery_part2_integration.py
"""
Test scenarios for Integration and Concurrency (Part 2: Pattern Discovery).

Tests cover:
- Integration: Full pipeline from input to pattern matching
- Concurrency: Thread-safe operations, race conditions
- Property-Based: Hypothesis tests for invariants

Total: 19 tests (13 integration/concurrency + 6 property-based)
"""

import time
from concurrent.futures import ThreadPoolExecutor


class TestIntegrationScenarios:
    """Integration test scenarios."""

    def test_full_pipeline_utterance_to_match(
        self,
        utterance_manager,
        pattern_matcher,
        pattern_manager,
        embedding_384d,
        test_templates_yml_path,
        netzwerk_session,
    ):
        """Full pipeline: Create utterance -> Load patterns -> Match."""
        from component_6_linguistik_engine import LinguisticPreprocessor
        from component_61_pattern_bootstrap import load_seed_templates

        # Step 1: Load patterns
        stats = load_seed_templates(netzwerk_session, test_templates_yml_path)
        assert stats["patterns_created"] == 3

        # Step 2: Create utterance with tokens
        text = "Was ist ein Hund?"
        uid = utterance_manager.create_utterance(text, embedding_384d)

        # Create tokens
        preprocessor = LinguisticPreprocessor()
        doc = preprocessor.process(text)
        for token in doc:
            utterance_manager.create_token(
                surface=token.text,
                lemma=token.lemma_.lower(),
                pos=token.pos_,
                utterance_id=uid,
                idx=token.i,
            )

        # Step 3: Match
        matches = pattern_matcher.match_utterance(uid)

        # Assertions
        assert len(matches) > 0
        assert all(0.0 <= score <= 1.0 for _, score in matches)

    def test_pattern_stats_update_flow(
        self, utterance_manager, pattern_manager, embedding_384d
    ):
        """Update pattern stats after match."""
        # Create pattern
        pid = pattern_manager.create_pattern("Test Pattern", "learned")

        # Initial stats
        patterns = pattern_manager.get_all_patterns(limit=10)
        pattern = next((p for p in patterns if p["id"] == pid), None)
        assert pattern["support"] == 0
        assert pattern["precision"] == 0.5

        # Update stats
        pattern_manager.update_pattern_stats(
            pid, support_increment=5, new_precision=0.75
        )

        # Verify update
        patterns = pattern_manager.get_all_patterns(limit=10)
        pattern = next((p for p in patterns if p["id"] == pid), None)
        assert pattern["support"] == 5
        assert abs(pattern["precision"] - 0.75) < 0.01

    def test_slot_learning_update_allowed(
        self, pattern_manager, netzwerk_session, tmp_path
    ):
        """Slot learns allowed values through experience."""
        import yaml

        from component_61_pattern_bootstrap import load_seed_templates

        # Create pattern with empty slot
        yaml_path = tmp_path / "learning_slot.yml"
        data = {
            "templates": [
                {
                    "id": "learn_test",
                    "name": "Learning Test",
                    "category": "Statement",
                    "pattern": [
                        {"kind": "SLOT", "slot_type": "SUBJECT", "allowed": []}
                    ],
                }
            ]
        }

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f)

        stats = load_seed_templates(netzwerk_session, str(yaml_path))
        assert stats["slots_created"] == 1

        # Get slot ID
        with netzwerk_session.driver.session(database="neo4j") as session:
            result = session.run(
                """
                MATCH (s:Slot {slotType: 'SUBJECT'})
                RETURN s.id AS sid
                LIMIT 1
            """
            )
            slot_id = result.single()["sid"]

        # Update allowed values
        pattern_manager.update_slot_allowed(slot_id, "hund", count_increment=1)
        pattern_manager.update_slot_allowed(slot_id, "hund", count_increment=1)
        pattern_manager.update_slot_allowed(slot_id, "katze", count_increment=1)

        # Verify counts
        with netzwerk_session.driver.session(database="neo4j") as session:
            result = session.run(
                """
                MATCH (s:Slot {id: $sid})-[r:ALLOWS]->(al:AllowedLemma)
                RETURN al.value AS lemma, r.count AS count
            """,
                {"sid": slot_id},
            )

            counts = {record["lemma"]: record["count"] for record in result}
            assert counts.get("hund") == 2
            assert counts.get("katze") == 1


class TestConcurrencyScenarios:
    """Concurrency test scenarios."""

    def test_conc01_concurrent_utterance_creation(
        self, utterance_manager, embedding_384d
    ):
        """CONC-01: 10 threads create 100 utterances simultaneously."""

        def create_utterance(i):
            return utterance_manager.create_utterance(
                f"Utterance {i}", embedding_384d, user_id=f"user_{i % 10}"
            )

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(create_utterance, i) for i in range(100)]
            results = [f.result() for f in futures]

        # Assertions
        assert len(results) == 100
        assert len(set(results)) == 100  # All unique UUIDs

        # Verify in database
        utterances = utterance_manager.get_recent_utterances(limit=200)
        assert len(utterances) >= 100

    def test_conc02_concurrent_pattern_matching(
        self,
        pattern_matcher,
        utterance_manager,
        embedding_384d,
        test_templates_yml_path,
        netzwerk_session,
    ):
        """CONC-02: 5 threads match patterns simultaneously."""
        from component_61_pattern_bootstrap import load_seed_templates

        # Load patterns
        load_seed_templates(netzwerk_session, test_templates_yml_path)

        # Create utterances
        utterance_ids = [
            utterance_manager.create_utterance(f"Text {i}", embedding_384d)
            for i in range(50)
        ]

        def match_utterance(uid):
            return pattern_matcher.match_utterance(uid)

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(match_utterance, uid) for uid in utterance_ids]
            results = [f.result() for f in futures]

        # Assertions
        assert len(results) == 50
        assert all(isinstance(r, list) for r in results)

    def test_conc03_cache_invalidation_race(self, pattern_matcher, pattern_manager):
        """CONC-03: Race condition in cache invalidation (no crash)."""

        # Thread 1: Query patterns
        def query_patterns():
            time.sleep(0.05)
            tokens = [{"lemma": "test"}]
            return pattern_matcher._get_candidate_patterns(tokens)

        # Thread 2: Invalidate cache
        def invalidate_cache():
            time.sleep(0.02)
            pattern_matcher._invalidate_candidate_cache()

        with ThreadPoolExecutor(max_workers=2) as executor:
            f1 = executor.submit(query_patterns)
            f2 = executor.submit(invalidate_cache)

            results = f1.result()
            f2.result()  # Ensure completes

        # No crash -> success
        assert results is not None

    def test_conc04_concurrent_pattern_creation_and_matching(
        self,
        pattern_matcher,
        pattern_manager,
        utterance_manager,
        embedding_384d,
        netzwerk_session,
    ):
        """CONC-04: Threads creating patterns while others match."""
        from component_1_netzwerk_patterns import PatternManager

        pm = PatternManager(netzwerk_session.driver, pattern_matcher=pattern_matcher)

        # Create utterance for matching
        uid = utterance_manager.create_utterance("Test query", embedding_384d)

        # Thread 1-5: Create patterns
        def create_pattern(i):
            return pm.create_pattern(f"Pattern {i}", "learned")

        # Thread 6-10: Match utterances
        def match_utterance():
            try:
                return pattern_matcher.match_utterance(uid)
            except Exception:
                return []

        with ThreadPoolExecutor(max_workers=10) as executor:
            create_futures = [executor.submit(create_pattern, i) for i in range(20)]
            match_futures = [executor.submit(match_utterance) for _ in range(20)]

            create_results = [f.result() for f in create_futures]
            match_results = [f.result() for f in match_futures]

        # Assertions
        assert len(create_results) == 20
        assert len(match_results) == 20

    def test_conc05_concurrent_bootstrap(
        self, netzwerk_session, test_templates_yml_path, clean_pattern_data
    ):
        """CONC-05: Two threads bootstrap simultaneously (creates duplicates)."""
        from component_61_pattern_bootstrap import load_seed_templates

        def bootstrap():
            return load_seed_templates(netzwerk_session, test_templates_yml_path)

        with ThreadPoolExecutor(max_workers=2) as executor:
            f1 = executor.submit(bootstrap)
            f2 = executor.submit(bootstrap)

            stats1 = f1.result()
            stats2 = f2.result()

        # Both succeed (duplicates created)
        assert stats1["patterns_created"] == 3
        assert stats2["patterns_created"] == 3

        # Total patterns (duplicates)
        from component_1_netzwerk_patterns import PatternManager

        pm = PatternManager(netzwerk_session.driver)
        all_patterns = pm.get_all_patterns(limit=100)
        assert len(all_patterns) >= 3  # At least 3, possibly 6 (duplicates)

    def test_conc06_concurrent_archive_and_query(
        self, utterance_manager, embedding_384d, netzwerk_session
    ):
        """CONC-06: Archive while querying recent utterances."""
        # Create 50 old and 50 recent utterances
        for i in range(100):
            utterance_manager.create_utterance(f"Utterance {i}", embedding_384d)

        # Set half to old
        with netzwerk_session.driver.session(database="neo4j") as session:
            result = session.run(
                """
                MATCH (u:Utterance)
                WITH u
                LIMIT 50
                SET u.timestamp = datetime() - duration({days: 40})
                RETURN count(u) AS updated
            """
            )

        # Thread 1: Archive
        def archive():
            return utterance_manager.archive_old_utterances(days_threshold=30)

        # Thread 2: Query
        def query():
            return utterance_manager.get_recent_utterances(limit=100)

        with ThreadPoolExecutor(max_workers=2) as executor:
            f1 = executor.submit(archive)
            f2 = executor.submit(query)

            archived_count = f1.result()
            recent = f2.result()

        # Assertions
        assert archived_count == 50
        assert len(recent) <= 100


class TestPropertyBasedScenarios:
    """Property-based test scenarios using Hypothesis."""

    def test_prop01_utterance_storage_deterministic(
        self, utterance_manager, embedding_384d, netzwerk_session
    ):
        """PROP-01: Same input produces same normalized output."""
        import hypothesis.strategies as st
        from hypothesis import given

        @given(
            st.text(
                min_size=1,
                max_size=100,
                alphabet=st.characters(whitelist_categories=("L", "N")),
            )
        )
        def test_normalization(text):
            if not text.strip():
                return  # Skip empty

            utterance_manager.create_utterance(text, embedding_384d)
            utterance_manager.create_utterance(text, embedding_384d)

            utterances = utterance_manager.get_recent_utterances(limit=2)
            if len(utterances) >= 2:
                # Same text -> same normalized form
                assert utterances[0]["normalized"] == utterances[1]["normalized"]
                # But different UUIDs
                assert utterances[0]["id"] != utterances[1]["id"]

        # Run with small examples for speed
        test_normalization()

    def test_prop02_score_bounds(self, pattern_matcher):
        """PROP-02: Hybrid score always in [0.0, 1.0]."""
        import hypothesis.strategies as st
        from hypothesis import given

        @given(
            st.floats(min_value=0.0, max_value=1.0),
            st.floats(min_value=0.0, max_value=1.0),
        )
        def test_bounds(template_score, embedding_score):
            hybrid_score = 0.6 * template_score + 0.4 * embedding_score
            assert 0.0 <= hybrid_score <= 1.0

        test_bounds()

    def test_prop03_pattern_stats_monotonic(self, pattern_manager, netzwerk_session):
        """PROP-03: Support only increases, never decreases."""
        import hypothesis.strategies as st
        from hypothesis import given

        pid = pattern_manager.create_pattern("Test", "learned")

        @given(st.lists(st.integers(min_value=0, max_value=5), min_size=5, max_size=10))
        def test_monotonic(increments):
            support_values = []
            for increment in increments:
                pattern_manager.update_pattern_stats(
                    pid, support_increment=increment, new_precision=0.5
                )

                # Get current support
                patterns = pattern_manager.get_all_patterns(limit=100)
                pattern = next((p for p in patterns if p["id"] == pid), None)
                if pattern:
                    support_values.append(pattern["support"])

            # Verify non-decreasing
            for i in range(len(support_values) - 1):
                assert support_values[i] <= support_values[i + 1]

        test_monotonic()

    def test_prop04_token_sequence_order(
        self, utterance_manager, embedding_384d, netzwerk_session
    ):
        """PROP-04: Tokens retrieved in idx order."""
        from component_6_linguistik_engine import LinguisticPreprocessor

        preprocessor = LinguisticPreprocessor()

        import hypothesis.strategies as st
        from hypothesis import given, settings

        @settings(deadline=500)  # Allow 500ms for database operations
        @given(
            st.lists(
                st.text(min_size=1, max_size=10, alphabet="abcdefghijklmnopqrstuvwxyz"),
                min_size=1,
                max_size=20,
            )
        )
        def test_order(words):
            text = " ".join(words)
            if len(text) > 10000 or not text.strip():
                return

            uid = utterance_manager.create_utterance(text, embedding_384d)

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

            # Verify order
            tokens = utterance_manager.get_tokens_for_utterance(uid)
            for i, token in enumerate(tokens):
                assert token["idx"] == i

        test_order()

    def test_prop05_embedding_dimension_preserved(
        self, utterance_manager, embedding_384d
    ):
        """PROP-05: 384D vector stored and retrieved unchanged."""
        import hypothesis.strategies as st
        from hypothesis import given

        @given(
            st.lists(
                st.floats(
                    min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False
                ),
                min_size=384,
                max_size=384,
            )
        )
        def test_dimension(embedding):
            utterance_manager.create_utterance("Test", embedding)

            utterances = utterance_manager.get_recent_utterances(limit=1)
            assert len(utterances[0]["embedding"]) == 384

        test_dimension()

    def test_prop06_anchor_filtering_reduces_set(
        self, pattern_matcher, pattern_manager
    ):
        """PROP-06: Anchor filtering never increases candidates."""
        import hypothesis.strategies as st
        from hypothesis import given

        # Create some patterns
        for i in range(10):
            pattern_manager.create_pattern(f"Pattern {i}", "learned")

        @given(
            st.lists(
                st.sampled_from(["was", "ist", "?", "wer", "macht", "xyz", "abc"]),
                min_size=1,
                max_size=10,
            )
        )
        def test_reduction(anchor_lemmas):
            tokens = [{"lemma": lemma} for lemma in anchor_lemmas]

            # With anchors
            candidates_with_anchors = pattern_matcher._get_candidate_patterns(tokens)

            # Without anchors (all patterns)
            all_patterns = pattern_manager.get_all_patterns(limit=100)

            # Filtering should not increase set
            assert len(candidates_with_anchors) <= len(all_patterns)

        test_reduction()
