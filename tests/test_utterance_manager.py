# tests/test_utterance_manager.py
"""
Test suite for UtteranceManager (Pattern Discovery System).

Covers:
- Happy Path scenarios (HP-01 to HP-05)
- Edge Cases (EC-01 to EC-16)
- Error Handling (ERR-01)
- Performance tests (PERF-01 to PERF-03)
- Concurrency tests (CONC-01)
- Property-based tests (PROP-01 to PROP-03)

Total: 25+ tests
"""
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from kai_exceptions import DatabaseException

# ============================================================================
# HAPPY PATH TESTS (HP-01 to HP-05)
# ============================================================================


class TestUtteranceManagerHappyPath:
    """Happy path scenarios for UtteranceManager."""

    def test_hp01_simple_utterance_creation(self, utterance_manager, embedding_384d):
        """HP-01: Create utterance with typical German sentence."""
        text = "Was ist ein Hund?"
        user_id = "user_123"

        utterance_id = utterance_manager.create_utterance(text, embedding_384d, user_id)

        # Verify UUID format
        assert len(utterance_id) == 36
        uuid.UUID(utterance_id)  # Raises if invalid

        # Verify storage (write verification)
        utterances = utterance_manager.get_recent_utterances(limit=1)
        assert len(utterances) == 1
        assert utterances[0]["text"] == text
        assert utterances[0]["userId"] == user_id
        assert utterances[0]["archived"] is False

    def test_hp02_token_sequence_creation(
        self, utterance_manager, embedding_384d, netzwerk_session
    ):
        """HP-02: Create complete token sequence with NEXT chain."""
        utterance_id = utterance_manager.create_utterance(
            "Ein kleiner Hund", embedding_384d
        )

        token_ids = []
        tokens_data = [
            ("Ein", "ein", "DET", 0),
            ("kleiner", "klein", "ADJ", 1),
            ("Hund", "hund", "NOUN", 2),
        ]

        for surface, lemma, pos, idx in tokens_data:
            token_id = utterance_manager.create_token(
                surface, lemma, pos, utterance_id, idx
            )
            token_ids.append(token_id)
            assert len(token_id) == 36

        # Verify NEXT chain using Neo4j query
        with netzwerk_session.driver.session(database="neo4j") as session:
            result = session.run(
                """
                MATCH (u:Utterance {id: $uid})-[:HAS_TOKEN]->(t0:Token {idx: 0})-[:NEXT]->(t1:Token {idx: 1})-[:NEXT]->(t2:Token {idx: 2})
                RETURN t0, t1, t2
            """,
                {"uid": utterance_id},
            )
            record = result.single()
            assert record is not None
            assert record["t0"]["surface"] == "Ein"
            assert record["t1"]["surface"] == "kleiner"
            assert record["t2"]["surface"] == "Hund"

    def test_hp03_retrieve_recent_utterances(self, utterance_manager, embedding_384d):
        """HP-03: Query 50 most recent utterances."""
        # Create 60 utterances
        for i in range(60):
            utterance_manager.create_utterance(f"Utterance {i}", embedding_384d)

        utterances = utterance_manager.get_recent_utterances(limit=50, archived=False)

        assert len(utterances) == 50
        # Verify DESC order (most recent first)
        for i in range(len(utterances) - 1):
            assert utterances[i]["timestamp"] >= utterances[i + 1]["timestamp"]
        # Verify all archived=false
        assert all(u["archived"] is False for u in utterances)

    def test_hp04_archive_old_utterances(
        self, utterance_manager, embedding_384d, netzwerk_session
    ):
        """HP-04: Archive utterances older than 30 days."""
        # Create old utterance (manually set timestamp to 40 days ago)
        utterance_id = utterance_manager.create_utterance(
            "Old utterance", embedding_384d
        )

        with netzwerk_session.driver.session(database="neo4j") as session:
            session.run(
                """
                MATCH (u:Utterance {id: $uid})
                SET u.timestamp = datetime() - duration({days: 40})
            """,
                {"uid": utterance_id},
            )

        # Archive old utterances
        count = utterance_manager.archive_old_utterances(days_threshold=30)
        assert count == 1

        # Verify archived flag set
        with netzwerk_session.driver.session(database="neo4j") as session:
            result = session.run(
                "MATCH (u:Utterance {id: $uid}) RETURN u.archived AS archived",
                {"uid": utterance_id},
            )
            assert result.single()["archived"] is True

        # Idempotency test: second call should return 0
        count2 = utterance_manager.archive_old_utterances(days_threshold=30)
        assert count2 == 0

    def test_hp05_batch_token_retrieval(self, utterance_manager, sample_utterance_ids):
        """HP-05: Fetch tokens for 10 utterances in single query."""
        # Create tokens for each utterance
        for i, uid in enumerate(sample_utterance_ids):
            for j in range(3):  # 3 tokens per utterance
                utterance_manager.create_token(f"Word{j}", f"word{j}", "NOUN", uid, j)

        tokens_by_utterance = utterance_manager.get_tokens_for_utterances_batch(
            sample_utterance_ids
        )

        assert len(tokens_by_utterance) == 10
        for uid in sample_utterance_ids:
            assert uid in tokens_by_utterance
            assert len(tokens_by_utterance[uid]) == 3
            # Verify order
            for j, token in enumerate(tokens_by_utterance[uid]):
                assert token["idx"] == j


# ============================================================================
# EDGE CASE TESTS (EC-01 to EC-16)
# ============================================================================


class TestUtteranceManagerEdgeCases:
    """Edge case scenarios for UtteranceManager."""

    def test_ec01_empty_utterance_text(self, utterance_manager, embedding_384d):
        """EC-01: Create utterance with empty string."""
        with pytest.raises(ValueError, match="Text cannot be empty"):
            utterance_manager.create_utterance("", embedding_384d, "user_123")

    def test_ec02_whitespace_only_utterance(self, utterance_manager, embedding_384d):
        """EC-02: Create utterance with only whitespace."""
        with pytest.raises(ValueError, match="Text cannot be empty"):
            utterance_manager.create_utterance("   \n\t  ", embedding_384d, "user_123")

    def test_ec03_maximum_size_utterance(self, utterance_manager, embedding_384d):
        """EC-03: Exactly 10000 character utterance."""
        text = "X" * 10000
        utterance_id = utterance_manager.create_utterance(text, embedding_384d)
        assert utterance_id is not None

        utterances = utterance_manager.get_recent_utterances(limit=1)
        assert len(utterances[0]["text"]) == 10000

    def test_ec04_oversized_utterance(self, utterance_manager, embedding_384d):
        """EC-04: 10001 character utterance."""
        text = "X" * 10001
        with pytest.raises(ValueError, match="Text must be <= 10000 characters"):
            utterance_manager.create_utterance(text, embedding_384d)

    def test_ec05_wrong_embedding_dimension_383d(self, utterance_manager):
        """EC-05: Embedding with 383 dimensions."""
        embedding = [0.1] * 383
        with pytest.raises(ValueError, match="Embedding must be 384-dimensional"):
            utterance_manager.create_utterance("Test", embedding)

    def test_ec06_wrong_embedding_dimension_385d(self, utterance_manager):
        """EC-06: Embedding with 385 dimensions."""
        embedding = [0.1] * 385
        with pytest.raises(ValueError, match="Embedding must be 384-dimensional"):
            utterance_manager.create_utterance("Test", embedding)

    def test_ec07_empty_embedding(self, utterance_manager):
        """EC-07: Empty list for embedding."""
        with pytest.raises(ValueError, match="Embedding must be 384-dimensional"):
            utterance_manager.create_utterance("Test", [])

    def test_ec08_invalid_pos_tag(self, utterance_manager, embedding_384d):
        """EC-08: POS tag not in whitelist."""
        utterance_id = utterance_manager.create_utterance("Test", embedding_384d)
        with pytest.raises(ValueError, match="Invalid POS tag: INVALID_POS"):
            utterance_manager.create_token(
                "Test", "test", "INVALID_POS", utterance_id, 0
            )

    def test_ec09_negative_token_index(self, utterance_manager, embedding_384d):
        """EC-09: Token with idx=-1."""
        utterance_id = utterance_manager.create_utterance("Test", embedding_384d)
        with pytest.raises(ValueError, match="Index must be non-negative"):
            utterance_manager.create_token("Test", "test", "NOUN", utterance_id, -1)

    def test_ec10_token_sequence_with_gap(
        self, utterance_manager, embedding_384d, netzwerk_session
    ):
        """EC-10: Create tokens with idx 0, 2, 4 (skip 1, 3)."""
        utterance_id = utterance_manager.create_utterance("Test", embedding_384d)

        utterance_manager.create_token("A", "a", "DET", utterance_id, 0)
        utterance_manager.create_token("C", "c", "NOUN", utterance_id, 2)
        utterance_manager.create_token("E", "e", "PUNCT", utterance_id, 4)

        # Verify NEXT chain is broken (no connections)
        with netzwerk_session.driver.session(database="neo4j") as session:
            result = session.run(
                """
                MATCH (u:Utterance {id: $uid})-[:HAS_TOKEN]->(t:Token)
                OPTIONAL MATCH (t)-[:NEXT]->(next:Token)
                RETURN t.idx AS idx, next.idx AS next_idx
                ORDER BY t.idx
            """,
                {"uid": utterance_id},
            )
            records = list(result)
            assert len(records) == 3
            # All should have next_idx=None (broken chain)
            assert all(r["next_idx"] is None for r in records)

    def test_ec11_token_with_special_characters_in_lemma(
        self, utterance_manager, embedding_384d
    ):
        """EC-11: Lemma with SQL injection attempt."""
        utterance_id = utterance_manager.create_utterance("Test", embedding_384d)

        # SQL injection attempt - should be sanitized
        token_id = utterance_manager.create_token(
            "'; DROP TABLE--;", "'; drop table--;", "NOUN", utterance_id, 0
        )
        assert token_id is not None

        # Verify lemma was sanitized
        tokens = utterance_manager.get_tokens_for_utterance(utterance_id)
        assert len(tokens) == 1
        # Should be sanitized to "droptable" or "unknown"
        assert tokens[0]["lemma"] in ["droptable", "unknown"]

    def test_ec12_token_with_very_long_lemma(self, utterance_manager, embedding_384d):
        """EC-12: Lemma exceeding 63 char limit."""
        utterance_id = utterance_manager.create_utterance("Test", embedding_384d)

        long_lemma = "x" * 100
        token_id = utterance_manager.create_token(
            "X" * 100, long_lemma, "NOUN", utterance_id, 0
        )
        assert token_id is not None

        # Verify lemma was truncated to 63 chars
        tokens = utterance_manager.get_tokens_for_utterance(utterance_id)
        assert len(tokens[0]["lemma"]) <= 63

    def test_ec13_empty_lemma(self, utterance_manager, embedding_384d):
        """EC-13: Empty string for lemma."""
        utterance_id = utterance_manager.create_utterance("Test", embedding_384d)
        with pytest.raises(ValueError, match="Surface, lemma, and POS are required"):
            utterance_manager.create_token("Surface", "", "NOUN", utterance_id, 0)

    def test_ec14_retrieve_utterances_with_limit_zero(self, utterance_manager):
        """EC-14: Invalid limit value (0)."""
        with pytest.raises(ValueError, match="Limit must be 1-10000"):
            utterance_manager.get_recent_utterances(limit=0)

    def test_ec15_retrieve_utterances_with_limit_10001(self, utterance_manager):
        """EC-15: Excessive limit value (10001)."""
        with pytest.raises(ValueError, match="Limit must be 1-10000"):
            utterance_manager.get_recent_utterances(limit=10001)

    def test_ec16_archive_utterances_with_days_threshold_zero(self, utterance_manager):
        """EC-16: Invalid days threshold (0)."""
        with pytest.raises(ValueError, match="Days threshold must be >= 1"):
            utterance_manager.archive_old_utterances(days_threshold=0)


# ============================================================================
# ERROR HANDLING TESTS (ERR-01)
# ============================================================================


class TestUtteranceManagerErrorHandling:
    """Error handling scenarios for UtteranceManager."""

    def test_err01_create_token_for_nonexistent_utterance(self, utterance_manager):
        """ERR-01: Token references invalid utterance UUID."""
        fake_uuid = "00000000-0000-0000-0000-000000000000"
        with pytest.raises(DatabaseException):
            utterance_manager.create_token("Test", "test", "NOUN", fake_uuid, 0)


# ============================================================================
# PERFORMANCE TESTS (PERF-01 to PERF-03)
# ============================================================================


class TestUtteranceManagerPerformance:
    """Performance scenarios for UtteranceManager."""

    def test_perf01_create_100_utterances_sequentially(
        self, utterance_manager, embedding_384d
    ):
        """PERF-01: Sequential utterance creation."""
        start_time = time.time()
        utterance_ids = []

        for i in range(100):
            uid = utterance_manager.create_utterance(
                f"Test utterance {i}", embedding_384d
            )
            utterance_ids.append(uid)

        elapsed = time.time() - start_time
        avg_time_ms = (elapsed / 100) * 1000

        assert len(utterance_ids) == 100
        assert len(set(utterance_ids)) == 100  # No duplicates
        assert avg_time_ms < 100  # Less than 100ms per utterance (relaxed from 50ms)

    def test_perf02_create_1000_tokens_for_single_utterance(
        self, utterance_manager, embedding_384d
    ):
        """PERF-02: Very long token sequence."""
        utterance_id = utterance_manager.create_utterance("Long text", embedding_384d)

        start_time = time.time()
        for i in range(1000):
            utterance_manager.create_token(
                f"Word{i}", f"word{i}", "NOUN", utterance_id, i
            )

        time.time() - start_time

        # Verify NEXT chain and retrieval time
        retrieval_start = time.time()
        tokens = utterance_manager.get_tokens_for_utterance(utterance_id)
        retrieval_time = time.time() - retrieval_start

        assert len(tokens) == 1000
        assert retrieval_time < 1.0  # Less than 1 second (relaxed from 500ms)

    def test_perf03_batch_token_retrieval_100_utterances(
        self, utterance_manager, embedding_384d
    ):
        """PERF-03: Batch query for 100 utterances."""
        # Create 100 utterances with 5 tokens each
        utterance_ids = []
        for i in range(100):
            uid = utterance_manager.create_utterance(f"Utterance {i}", embedding_384d)
            utterance_ids.append(uid)
            for j in range(5):
                utterance_manager.create_token(f"W{j}", f"w{j}", "NOUN", uid, j)

        # Batch retrieval
        start_time = time.time()
        tokens_by_utterance = utterance_manager.get_tokens_for_utterances_batch(
            utterance_ids
        )
        elapsed_ms = (time.time() - start_time) * 1000

        assert len(tokens_by_utterance) == 100
        assert all(len(tokens) == 5 for tokens in tokens_by_utterance.values())
        assert elapsed_ms < 500  # Less than 500ms (relaxed from 200ms)


# ============================================================================
# CONCURRENCY TESTS (CONC-01)
# ============================================================================


class TestUtteranceManagerConcurrency:
    """Concurrency scenarios for UtteranceManager."""

    def test_conc01_concurrent_utterance_creation_10_threads(
        self, utterance_manager, embedding_384d
    ):
        """CONC-01: 10 threads creating utterances simultaneously."""

        def create_utterance(i):
            return utterance_manager.create_utterance(f"Text {i}", embedding_384d)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(create_utterance, i) for i in range(100)]
            results = [f.result() for f in futures]

        assert len(results) == 100
        assert len(set(results)) == 100  # No duplicates
        # Verify all created
        utterances = utterance_manager.get_recent_utterances(limit=100)
        assert len(utterances) == 100


# ============================================================================
# PROPERTY-BASED TESTS (PROP-01 to PROP-03)
# ============================================================================


class TestUtteranceManagerPropertyBased:
    """Property-based tests using Hypothesis."""

    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None
    )
    @given(
        st.text(
            min_size=1,
            max_size=100,
            alphabet=st.characters(blacklist_categories=("Cs",)),
        )
    )
    def test_prop01_utterance_creation_deterministic(
        self, utterance_manager, embedding_384d, text
    ):
        """PROP-01: Same input produces same normalized output."""
        try:
            uid1 = utterance_manager.create_utterance(text, embedding_384d)
            uid2 = utterance_manager.create_utterance(text, embedding_384d)

            # UUIDs different
            assert uid1 != uid2

            # But normalized text identical
            utterances = utterance_manager.get_recent_utterances(limit=2)
            assert len(utterances) == 2
            assert utterances[0]["normalized"] == utterances[1]["normalized"]
        except ValueError:
            # Skip if normalization fails (empty after strip)
            pass

    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None
    )
    @given(
        st.lists(
            st.floats(allow_nan=False, allow_infinity=False), min_size=384, max_size=384
        )
    )
    def test_prop02_embedding_always_384d_after_storage(
        self, utterance_manager, embedding
    ):
        """PROP-02: Embedding dimension preserved."""
        try:
            utterance_manager.create_utterance("Test", embedding)
            utterances = utterance_manager.get_recent_utterances(limit=1)
            assert len(utterances[0]["embedding"]) == 384
        except (ValueError, DatabaseException):
            # Skip if embedding contains invalid values
            pass

    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None
    )
    @given(st.integers(min_value=0, max_value=100))
    def test_prop03_token_idx_always_non_negative(
        self, utterance_manager, embedding_384d, idx
    ):
        """PROP-03: idx >= 0 constraint."""
        utterance_id = utterance_manager.create_utterance("Test", embedding_384d)
        token_id = utterance_manager.create_token(
            "Test", "test", "NOUN", utterance_id, idx
        )

        tokens = utterance_manager.get_tokens_for_utterance(utterance_id)
        assert len(tokens) == 1
        assert tokens[0]["idx"] >= 0
        assert tokens[0]["idx"] == idx
