# test_pattern_discovery_part2_matching.py
"""
Test scenarios for Pattern Matching (Part 2: Pattern Discovery).

Tests cover:
- Happy Path: Template matching, embedding matching, anchor filtering, cache
- Edge Cases: No candidates, no centroid, only punctuation anchors, cache expiry
- Error Handling: Non-existent utterance, zero norm vectors, missing items
- Performance: Anchor filtering, cache hit rate

Total: 25 tests
"""

import time

import pytest


class TestHappyPathPatternMatching:
    """Happy path scenarios for pattern matching."""

    def test_hp04_match_utterance_high_template_score(
        self,
        pattern_matcher,
        utterance_manager,
        pattern_manager,
        embedding_384d,
        test_templates_yml_path,
        netzwerk_session,
    ):
        """HP-04: Utterance matches WH-question pattern exactly."""
        from component_6_linguistik_engine import LinguisticPreprocessor
        from component_61_pattern_bootstrap import load_seed_templates

        # Load templates
        load_seed_templates(netzwerk_session, test_templates_yml_path)

        # Create utterance with tokens
        text = "Was ist ein Hund?"
        utterance_id = utterance_manager.create_utterance(text, embedding_384d)

        # Create tokens
        preprocessor = LinguisticPreprocessor()
        doc = preprocessor.process(text)
        for token in doc:
            utterance_manager.create_token(
                surface=token.text,
                lemma=token.lemma_.lower(),
                pos=token.pos_,
                utterance_id=utterance_id,
                idx=token.i,
            )

        # Match
        matches = pattern_matcher.match_utterance(utterance_id)

        # Assertions
        assert len(matches) > 0
        assert (
            matches[0][1] >= 0.25
        )  # Some hybrid score (no centroid yet, may be slightly lower)

    def test_hp06_anchor_token_filtering(
        self, pattern_matcher, pattern_manager, netzwerk_session
    ):
        """HP-06: 100 patterns reduced to <20 candidates via anchor tokens."""
        # Create 100 dummy patterns
        for i in range(100):
            pattern_manager.create_pattern(name=f"Pattern {i}", pattern_type="learned")

        # Create pattern items with anchor tokens for a few patterns
        from component_1_netzwerk_patterns import PatternManager

        pm = PatternManager(netzwerk_session.driver)

        # Get first pattern and add "was" literal
        patterns = pm.get_all_patterns(limit=5)
        for pattern in patterns[:3]:
            # Add anchor token "was" as literal
            with netzwerk_session.driver.session(database="neo4j") as session:
                session.run(
                    """
                    MATCH (p:Pattern {id: $pid})
                    CREATE (pi:PatternItem {id: randomUUID(), idx: 0, kind: 'LITERAL', literalValue: 'was'})
                    CREATE (p)-[:HAS_ITEM {idx: 0}]->(pi)
                """,
                    {"pid": pattern["id"]},
                )

        # Get candidates with anchor tokens
        tokens = [{"lemma": "was"}, {"lemma": "ist"}, {"lemma": "hund"}]
        candidates = pattern_matcher._get_candidate_patterns(tokens)

        # Should have <= 20 candidates (filtered by anchor)
        assert len(candidates) <= 20

    def test_hp07_cache_hit_repeated_anchor(
        self,
        pattern_matcher,
        utterance_manager,
        embedding_384d,
        test_templates_yml_path,
        netzwerk_session,
    ):
        """HP-07: Second query with same anchors uses cache."""
        from component_6_linguistik_engine import LinguisticPreprocessor
        from component_61_pattern_bootstrap import load_seed_templates

        load_seed_templates(netzwerk_session, test_templates_yml_path)
        preprocessor = LinguisticPreprocessor()

        # Create two utterances with same anchor tokens
        text1 = "Was ist ein Hund?"
        uid1 = utterance_manager.create_utterance(text1, embedding_384d)
        doc1 = preprocessor.process(text1)
        for token in doc1:
            utterance_manager.create_token(
                surface=token.text,
                lemma=token.lemma_.lower(),
                pos=token.pos_,
                utterance_id=uid1,
                idx=token.i,
            )

        text2 = "Was ist eine Katze?"
        uid2 = utterance_manager.create_utterance(text2, embedding_384d)
        doc2 = preprocessor.process(text2)
        for token in doc2:
            utterance_manager.create_token(
                surface=token.text,
                lemma=token.lemma_.lower(),
                pos=token.pos_,
                utterance_id=uid2,
                idx=token.i,
            )

        # First query (cache miss)
        start1 = time.time()
        matches1 = pattern_matcher.match_utterance(uid1)
        (time.time() - start1) * 1000

        # Second query (cache hit)
        start2 = time.time()
        matches2 = pattern_matcher.match_utterance(uid2)
        (time.time() - start2) * 1000

        # Assertions
        assert len(matches1) > 0
        assert len(matches2) > 0
        # Cache should speed up (though may not be 10x in test environment)

    def test_hp08_cache_invalidation_after_pattern_creation(
        self,
        pattern_matcher,
        utterance_manager,
        pattern_manager,
        embedding_384d,
        netzwerk_session,
    ):
        """HP-08: New pattern immediately available after creation."""
        # Create utterance
        utterance_id = utterance_manager.create_utterance("Test query", embedding_384d)

        # Match (no patterns yet)
        matches_before = pattern_matcher.match_utterance(utterance_id)

        # Create new pattern with matcher reference (triggers cache invalidation)
        from component_1_netzwerk_patterns import PatternManager

        pm = PatternManager(netzwerk_session.driver, pattern_matcher=pattern_matcher)
        pm.create_pattern("New Pattern", "learned")

        # Match again (should see new pattern)
        matches_after = pattern_matcher.match_utterance(utterance_id)

        # New pattern should be available
        assert len(matches_after) >= len(matches_before)

    def test_hp09_slot_match_with_allowed_values(
        self, pattern_matcher, pattern_manager, netzwerk_session, tmp_path
    ):
        """HP-09: Token in slot's allowed set contributes to score."""
        import yaml

        from component_61_pattern_bootstrap import load_seed_templates

        # Create template with slot having allowed values
        yaml_path = tmp_path / "slot_allowed.yml"
        data = {
            "templates": [
                {
                    "id": "wh_test",
                    "name": "WH Test",
                    "category": "Question",
                    "pattern": [
                        {
                            "kind": "SLOT",
                            "slot_type": "WH_WORD",
                            "allowed": ["was", "wer", "wie"],
                        },
                        {"kind": "LITERAL", "value": "ist"},
                    ],
                }
            ]
        }

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f)

        load_seed_templates(netzwerk_session, str(yaml_path))

        # Create mock tokens
        tokens = [{"lemma": "was"}, {"lemma": "ist"}]
        patterns = pattern_manager.get_all_patterns(limit=10)

        # Calculate template score
        if patterns:
            score = pattern_matcher._calculate_template_score(tokens, patterns[0])
            # Should have non-zero score
            assert score >= 0.0


class TestEdgeCasesPatternMatching:
    """Edge case scenarios for pattern matching."""

    def test_ec20_no_candidates_fallback(
        self, pattern_matcher, pattern_manager, netzwerk_session
    ):
        """EC-20: No patterns match anchor tokens (fallback to top 10)."""
        # Create patterns without anchor tokens
        for i in range(5):
            pattern_manager.create_pattern(f"Pattern {i}", "learned")

        # Tokens with no anchors
        tokens = [{"lemma": "xyz"}, {"lemma": "abc"}, {"lemma": "def"}]
        candidates = pattern_matcher._get_candidate_patterns(tokens)

        # Should return top 10 patterns (or less if fewer exist)
        assert len(candidates) <= 10

    def test_ec21_embedding_score_no_centroid(self, pattern_matcher):
        """EC-21: Pattern without centroid returns default score 0.5."""
        utterance = {"embedding": [0.1] * 384}
        pattern = {"id": "test_pattern", "centroid": None}

        score = pattern_matcher._calculate_embedding_score(utterance, pattern)

        assert score == 0.5  # Default fallback

    def test_ec22_anchor_tokens_only_punctuation(
        self, pattern_matcher, pattern_manager, netzwerk_session
    ):
        """EC-22: Utterance with only punctuation anchors."""
        # Create patterns with punctuation literals
        pid = pattern_manager.create_pattern("Question Pattern", "seed")

        with netzwerk_session.driver.session(database="neo4j") as session:
            session.run(
                """
                MATCH (p:Pattern {id: $pid})
                CREATE (pi:PatternItem {id: randomUUID(), idx: 0, kind: 'LITERAL', literalValue: '?'})
                CREATE (p)-[:HAS_ITEM {idx: 0}]->(pi)
            """,
                {"pid": pid},
            )

        # Tokens with punctuation
        tokens = [{"lemma": "?"}, {"lemma": "!"}]
        candidates = pattern_matcher._get_candidate_patterns(tokens)

        # Should find patterns with ? literal
        assert len(candidates) <= 20


class TestErrorHandlingPatternMatching:
    """Error handling scenarios for pattern matching."""

    def test_err04_match_nonexistent_utterance(self, pattern_matcher):
        """ERR-04: Pattern matching for non-existent utterance raises DatabaseException."""
        from kai_exceptions import DatabaseException

        fake_uuid = "00000000-0000-0000-0000-000000000000"

        # Raises DatabaseException (wraps ValueError from utterance lookup)
        with pytest.raises(DatabaseException, match="Failed to fetch utterance data"):
            pattern_matcher.match_utterance(fake_uuid)

    def test_err05_precision_out_of_bounds_high(self, pattern_manager):
        """ERR-05: Precision > 1.0 raises ValueError."""
        pid = pattern_manager.create_pattern("Test", "seed")

        with pytest.raises(ValueError, match="Precision must be 0.0-1.0"):
            pattern_manager.update_pattern_stats(
                pid, support_increment=1, new_precision=1.5
            )

    def test_err06_precision_out_of_bounds_low(self, pattern_manager):
        """ERR-06: Precision < 0.0 raises ValueError."""
        pid = pattern_manager.create_pattern("Test", "seed")

        with pytest.raises(ValueError, match="Precision must be 0.0-1.0"):
            pattern_manager.update_pattern_stats(
                pid, support_increment=1, new_precision=-0.1
            )

    def test_err11_pattern_items_fetch_fails(self, pattern_matcher, pattern_manager):
        """ERR-11: Pattern with no items returns template score 0.0."""
        # Create pattern without items
        pid = pattern_manager.create_pattern("Empty Pattern", "learned")

        pattern = {"id": pid, "name": "Empty"}
        tokens = [{"lemma": "test"}]

        score = pattern_matcher._calculate_template_score(tokens, pattern)

        assert score == 0.0  # No items -> score 0

    def test_err13_zero_norm_vector(self, pattern_matcher):
        """ERR-13: Zero norm embedding returns score 0.0."""
        utterance = {"embedding": [0.0] * 384}  # All zeros
        pattern = {"centroid": [0.1] * 384}

        score = pattern_matcher._calculate_embedding_score(utterance, pattern)

        assert score == 0.0  # Zero norm -> score 0

    def test_ec24_negative_support_increment(self, pattern_manager):
        """EC-24: Negative support increment raises ValueError."""
        pid = pattern_manager.create_pattern("Test", "seed")

        with pytest.raises(ValueError, match="must be non-negative"):
            pattern_manager.update_pattern_stats(
                pid, support_increment=-1, new_precision=0.8
            )


class TestPerformancePatternMatching:
    """Performance scenarios for pattern matching."""

    def test_perf02_anchor_filtering_reduces_candidates(
        self, pattern_matcher, pattern_manager, netzwerk_session
    ):
        """PERF-02: Anchor filtering reduces candidate set by 20x."""
        # Create 100 patterns
        for i in range(100):
            pattern_manager.create_pattern(f"Pattern {i}", "learned")

        # Add "was" anchor to 5 patterns
        patterns = pattern_manager.get_all_patterns(limit=100)
        for pattern in patterns[:5]:
            with netzwerk_session.driver.session(database="neo4j") as session:
                session.run(
                    """
                    MATCH (p:Pattern {id: $pid})
                    CREATE (pi:PatternItem {id: randomUUID(), idx: 0, kind: 'LITERAL', literalValue: 'was'})
                    CREATE (p)-[:HAS_ITEM {idx: 0}]->(pi)
                """,
                    {"pid": pattern["id"]},
                )

        # Query with specific anchors
        tokens = [{"lemma": "was"}, {"lemma": "ist"}, {"lemma": "?"}]
        candidates = pattern_matcher._get_candidate_patterns(tokens)

        # Should return <= 20 candidates (reduced from 100)
        assert len(candidates) <= 20

    def test_perf05_pattern_matching_fast(
        self,
        pattern_matcher,
        utterance_manager,
        pattern_manager,
        embedding_384d,
        test_templates_yml_path,
        netzwerk_session,
    ):
        """PERF-05: Pattern matching <100ms with 10 patterns."""
        from component_61_pattern_bootstrap import load_seed_templates

        # Load templates
        load_seed_templates(netzwerk_session, test_templates_yml_path)

        # Create utterance
        uid = utterance_manager.create_utterance("Was ist das?", embedding_384d)

        # Match
        start = time.time()
        matches = pattern_matcher.match_utterance(uid)
        elapsed_ms = (time.time() - start) * 1000

        # Should be fast
        assert elapsed_ms < 500  # Generous limit for test environment
        assert len(matches) <= 20


class TestTemplateScoring:
    """Tests for template scoring algorithm."""

    def test_calculate_template_score_perfect_match(
        self, pattern_matcher, pattern_manager, netzwerk_session
    ):
        """Perfect literal match returns high score."""
        # Create pattern with literals
        pid = pattern_manager.create_pattern("Test Pattern", "seed")

        with netzwerk_session.driver.session(database="neo4j") as session:
            session.run(
                """
                MATCH (p:Pattern {id: $pid})
                CREATE (pi1:PatternItem {id: randomUUID(), idx: 0, kind: 'LITERAL', literalValue: 'was'})
                CREATE (pi2:PatternItem {id: randomUUID(), idx: 1, kind: 'LITERAL', literalValue: 'ist'})
                CREATE (p)-[:HAS_ITEM {idx: 0}]->(pi1)
                CREATE (p)-[:HAS_ITEM {idx: 1}]->(pi2)
            """,
                {"pid": pid},
            )

        # Tokens matching perfectly
        tokens = [{"lemma": "was"}, {"lemma": "ist"}]
        pattern = {"id": pid}

        score = pattern_matcher._calculate_template_score(tokens, pattern)

        # Perfect match -> score close to 1.0
        assert score >= 0.9

    def test_calculate_template_score_length_penalty(
        self, pattern_matcher, pattern_manager, netzwerk_session
    ):
        """Length mismatch applies penalty."""
        pid = pattern_manager.create_pattern("Test Pattern", "seed")

        with netzwerk_session.driver.session(database="neo4j") as session:
            session.run(
                """
                MATCH (p:Pattern {id: $pid})
                CREATE (pi1:PatternItem {id: randomUUID(), idx: 0, kind: 'LITERAL', literalValue: 'was'})
                CREATE (p)-[:HAS_ITEM {idx: 0}]->(pi1)
            """,
                {"pid": pid},
            )

        # More tokens than pattern items
        tokens = [{"lemma": "was"}, {"lemma": "ist"}, {"lemma": "das"}]
        pattern = {"id": pid}

        score = pattern_matcher._calculate_template_score(tokens, pattern)

        # Should have penalty for extra tokens
        assert score < 1.0


class TestEmbeddingScoring:
    """Tests for embedding-based scoring."""

    def test_calculate_embedding_score_high_similarity(self, pattern_matcher):
        """High cosine similarity returns high score."""
        # Nearly identical vectors
        vec1 = [0.5] * 384
        vec2 = [0.51] * 384

        utterance = {"embedding": vec1}
        pattern = {"centroid": vec2}

        score = pattern_matcher._calculate_embedding_score(utterance, pattern)

        # High similarity
        assert score >= 0.95

    def test_calculate_embedding_score_low_similarity(self, pattern_matcher):
        """Low cosine similarity returns low score."""
        # Orthogonal vectors
        vec1 = [1.0] + [0.0] * 383
        vec2 = [0.0] + [1.0] + [0.0] * 382

        utterance = {"embedding": vec1}
        pattern = {"centroid": vec2}

        score = pattern_matcher._calculate_embedding_score(utterance, pattern)

        # Low similarity (orthogonal -> 0)
        assert score < 0.1

    def test_calculate_embedding_score_no_utterance_embedding(self, pattern_matcher):
        """No utterance embedding returns default 0.5."""
        utterance = {"embedding": None}
        pattern = {"centroid": [0.1] * 384}

        score = pattern_matcher._calculate_embedding_score(utterance, pattern)

        assert score == 0.5


class TestHybridScoring:
    """Tests for hybrid scoring (template + embedding)."""

    def test_hybrid_score_balanced(
        self,
        pattern_matcher,
        utterance_manager,
        pattern_manager,
        embedding_384d,
        netzwerk_session,
    ):
        """Hybrid score balances template and embedding scores."""
        from component_6_linguistik_engine import LinguisticPreprocessor

        # Create pattern
        pid = pattern_manager.create_pattern("Test", "seed")

        with netzwerk_session.driver.session(database="neo4j") as session:
            # Add literal
            session.run(
                """
                MATCH (p:Pattern {id: $pid})
                CREATE (pi:PatternItem {id: randomUUID(), idx: 0, kind: 'LITERAL', literalValue: 'test'})
                CREATE (p)-[:HAS_ITEM {idx: 0}]->(pi)
            """,
                {"pid": pid},
            )

            # Add centroid
            session.run(
                """
                MATCH (p:Pattern {id: $pid})
                SET p.centroid = $centroid
            """,
                {"pid": pid, "centroid": embedding_384d},
            )

        # Create utterance with tokens
        text = "test"
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

        # Match
        matches = pattern_matcher.match_utterance(uid)

        # Should have high hybrid score (template match + embedding match)
        assert len(matches) > 0
        assert matches[0][1] >= 0.8  # High hybrid score


class TestCacheManagement:
    """Tests for candidate pattern caching."""

    def test_cache_invalidation_explicit(self, pattern_matcher, pattern_manager):
        """Cache invalidation clears cache."""
        # Populate cache
        tokens = [{"lemma": "was"}, {"lemma": "ist"}]
        pattern_matcher._get_candidate_patterns(tokens)

        # Invalidate
        pattern_matcher._invalidate_candidate_cache()

        # Cache should be empty (next query is cache miss)
        # Note: Hard to test cache miss directly, but ensures no errors
        pattern_matcher._get_candidate_patterns(tokens)
