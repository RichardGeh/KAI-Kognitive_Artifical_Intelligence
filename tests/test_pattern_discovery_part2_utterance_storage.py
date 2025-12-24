# test_pattern_discovery_part2_utterance_storage.py
"""
Test scenarios for Utterance Storage (Part 2: Pattern Discovery).

Tests cover:
- Happy Path: Simple and batch utterance storage
- Edge Cases: Empty, whitespace, size limits, punctuation
- Error Handling: Validation, embedding failures, transaction rollback
- Performance: Sequential vs batch token creation

Total: 25 tests
"""

import time

import pytest

from kai_exceptions import DatabaseException


class TestHappyPathUtteranceStorage:
    """Happy path scenarios for utterance storage (HP-01, HP-02)."""

    def test_hp01_store_simple_utterance_sequential(
        self, utterance_manager, embedding_384d, preprocessor_mock
    ):
        """HP-01: Store 5-word German question with sequential token creation."""
        text = "Was ist ein kleiner Hund"

        # Create utterance
        utterance_id = utterance_manager.create_utterance(
            text=text, embedding=embedding_384d, user_id="test_user"
        )

        # Assertions
        assert utterance_id is not None
        assert len(utterance_id) == 36  # UUID format

        # Verify utterance created
        utterances = utterance_manager.get_recent_utterances(limit=1)
        assert len(utterances) == 1
        assert utterances[0]["text"] == text
        assert utterances[0]["userId"] == "test_user"
        assert len(utterances[0]["embedding"]) == 384
        assert utterances[0]["archived"] is False

    def test_hp01_store_with_tokens(
        self, utterance_manager, embedding_384d, netzwerk_session
    ):
        """HP-01: Verify token creation and NEXT chain."""
        # Import here to avoid circular dependency
        from component_6_linguistik_engine import LinguisticPreprocessor

        preprocessor = LinguisticPreprocessor()
        text = "Was ist ein Hund?"

        # Create utterance
        utterance_id = utterance_manager.create_utterance(
            text=text, embedding=embedding_384d
        )

        # Create tokens manually (simulating kai_worker._store_utterance)
        doc = preprocessor.process(text)
        for token in doc:
            utterance_manager.create_token(
                surface=token.text,
                lemma=token.lemma_.lower(),
                pos=token.pos_,
                utterance_id=utterance_id,
                idx=token.i,
            )

        # Verify tokens
        tokens = utterance_manager.get_tokens_for_utterance(utterance_id)
        assert len(tokens) >= 4  # At least "was", "ist", "ein", "hund"

        # Verify token order
        for i, token in enumerate(tokens):
            assert token["idx"] == i

        # Verify NEXT chain exists (query Neo4j directly)
        with netzwerk_session.driver.session(database="neo4j") as session:
            result = session.run(
                """
                MATCH (u:Utterance {id: $uid})-[:HAS_TOKEN]->(t1:Token)
                -[:NEXT]->(t2:Token)
                RETURN count(*) AS next_count
            """,
                {"uid": utterance_id},
            )
            next_count = result.single()["next_count"]
            assert next_count == len(tokens) - 1  # N-1 NEXT relations


class TestEdgeCasesUtteranceStorage:
    """Edge case scenarios for utterance storage."""

    def test_ec01_empty_utterance_text(self, utterance_manager, embedding_384d):
        """EC-01: Empty string raises ValueError."""
        with pytest.raises(ValueError, match="Text cannot be empty"):
            utterance_manager.create_utterance("", embedding_384d)

    def test_ec02_whitespace_only_utterance(self, utterance_manager, embedding_384d):
        """EC-02: Whitespace-only string raises ValueError."""
        with pytest.raises(ValueError, match="Text cannot be empty"):
            utterance_manager.create_utterance("   \n\t  ", embedding_384d)

    def test_ec03_single_token_utterance(self, utterance_manager, embedding_384d):
        """EC-03: Single token utterance creates 1 token, 0 NEXT relations."""
        from component_6_linguistik_engine import LinguisticPreprocessor

        preprocessor = LinguisticPreprocessor()
        text = "Hund"

        utterance_id = utterance_manager.create_utterance(text, embedding_384d)

        # Create token
        doc = preprocessor.process(text)
        for token in doc:
            utterance_manager.create_token(
                surface=token.text,
                lemma=token.lemma_.lower(),
                pos=token.pos_,
                utterance_id=utterance_id,
                idx=token.i,
            )

        tokens = utterance_manager.get_tokens_for_utterance(utterance_id)
        assert len(tokens) == 1
        assert tokens[0]["lemma"] == "hund"

    def test_ec06_maximum_size_utterance(self, utterance_manager, embedding_384d):
        """EC-06: 10,000 character utterance at limit."""
        text = "X" * 10000

        utterance_manager.create_utterance(text, embedding_384d)

        utterances = utterance_manager.get_recent_utterances(limit=1)
        assert len(utterances[0]["text"]) == 10000

    def test_ec07_exceeds_maximum_size(self, utterance_manager, embedding_384d):
        """EC-07: 10,001 character utterance raises ValueError."""
        text = "X" * 10001

        with pytest.raises(ValueError, match="must be <= 10000 characters"):
            utterance_manager.create_utterance(text, embedding_384d)

    def test_ec09_utterance_only_punctuation(
        self, utterance_manager, embedding_384d, netzwerk_session
    ):
        """EC-09: Utterance with only punctuation creates PUNCT tokens."""
        from component_6_linguistik_engine import LinguisticPreprocessor

        preprocessor = LinguisticPreprocessor()
        text = "...!!???"

        utterance_id = utterance_manager.create_utterance(text, embedding_384d)

        # Create tokens
        doc = preprocessor.process(text)
        for token in doc:
            utterance_manager.create_token(
                surface=token.text,
                lemma=token.lemma_.lower() if token.lemma_ else token.text.lower(),
                pos=token.pos_,
                utterance_id=utterance_id,
                idx=token.i,
            )

        tokens = utterance_manager.get_tokens_for_utterance(utterance_id)
        assert all(t["pos"] == "PUNCT" for t in tokens)


class TestErrorHandlingUtteranceStorage:
    """Error handling scenarios for utterance storage."""

    def test_err01_embedding_service_unavailable(
        self, utterance_manager, embedding_384d, netzwerk_session
    ):
        """ERR-01: Storage continues even if embedding fails (non-critical)."""
        # Note: In actual implementation, utterance creation requires embedding
        # This tests the scenario where we validate embedding presence

        with pytest.raises(ValueError, match="Embedding must be"):
            utterance_manager.create_utterance("Test", None)

    def test_err01_invalid_embedding_dimension(self, utterance_manager):
        """ERR-01: Invalid embedding dimension raises ValueError."""
        with pytest.raises(ValueError, match="384-dimensional"):
            utterance_manager.create_utterance("Test", [0.1] * 100)

    def test_err07_create_token_invalid_utterance(
        self, utterance_manager, netzwerk_session
    ):
        """ERR-07: Token creation for non-existent utterance fails."""
        fake_uuid = "00000000-0000-0000-0000-000000000000"

        with pytest.raises(DatabaseException):
            utterance_manager.create_token(
                surface="Test", lemma="test", pos="NOUN", utterance_id=fake_uuid, idx=0
            )

    def test_err_invalid_pos_tag(self, utterance_manager, embedding_384d):
        """ERR: Invalid POS tag raises ValueError."""
        utterance_id = utterance_manager.create_utterance("Test", embedding_384d)

        with pytest.raises(ValueError, match="Invalid POS tag"):
            utterance_manager.create_token(
                surface="Test",
                lemma="test",
                pos="INVALID_POS",
                utterance_id=utterance_id,
                idx=0,
            )

    def test_err_negative_token_index(self, utterance_manager, embedding_384d):
        """ERR: Negative token index raises ValueError."""
        utterance_id = utterance_manager.create_utterance("Test", embedding_384d)

        with pytest.raises(ValueError, match="must be non-negative"):
            utterance_manager.create_token(
                surface="Test",
                lemma="test",
                pos="NOUN",
                utterance_id=utterance_id,
                idx=-1,
            )


class TestPerformanceUtteranceStorage:
    """Performance scenarios for utterance storage."""

    def test_perf06_archive_old_utterances(
        self, utterance_manager, embedding_384d, netzwerk_session
    ):
        """PERF-06: Archive 100 old utterances efficiently."""
        # Create 100 utterances and manually set old timestamps
        utterance_ids = []
        for i in range(100):
            uid = utterance_manager.create_utterance(
                text=f"Old utterance {i}", embedding=embedding_384d
            )
            utterance_ids.append(uid)

        # Manually set timestamps to 40 days ago
        with netzwerk_session.driver.session(database="neo4j") as session:
            session.run(
                """
                UNWIND $uids AS uid
                MATCH (u:Utterance {id: uid})
                SET u.timestamp = datetime() - duration({days: 40})
            """,
                {"uids": utterance_ids},
            )

        # Archive old utterances
        start = time.time()
        archived_count = utterance_manager.archive_old_utterances(days_threshold=30)
        elapsed_ms = (time.time() - start) * 1000

        # Assertions
        assert archived_count == 100
        assert elapsed_ms < 2000  # Less than 2 seconds

        # Verify archived
        active = utterance_manager.get_recent_utterances(limit=200, archived=False)
        assert len(active) == 0

        archived = utterance_manager.get_recent_utterances(limit=200, archived=True)
        assert len(archived) == 100

    def test_perf07_batch_token_retrieval(
        self, utterance_manager, embedding_384d, netzwerk_session
    ):
        """PERF-07: Batch token retrieval for 50 utterances."""
        from component_6_linguistik_engine import LinguisticPreprocessor

        preprocessor = LinguisticPreprocessor()

        # Create 50 utterances with 5 tokens each
        utterance_ids = []
        for i in range(50):
            text = f"Test utterance number {i} here"
            uid = utterance_manager.create_utterance(text, embedding_384d)
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

        # Batch retrieval
        start = time.time()
        tokens_by_utterance = utterance_manager.get_tokens_for_utterances_batch(
            utterance_ids
        )
        batch_ms = (time.time() - start) * 1000

        # Assertions
        assert len(tokens_by_utterance) == 50
        assert batch_ms < 1000  # Less than 1 second

        for uid, tokens in tokens_by_utterance.items():
            assert len(tokens) >= 5  # At least 5 tokens per utterance


class TestArchivalMechanism:
    """Tests for utterance archival mechanism."""

    def test_hp16_cleanup_old_utterances(
        self, utterance_manager, embedding_384d, netzwerk_session
    ):
        """HP-16: Archive old utterances while keeping recent ones."""
        # Create 5 old utterances (40 days)
        old_uids = []
        for i in range(5):
            uid = utterance_manager.create_utterance(
                f"Old utterance {i}", embedding_384d
            )
            old_uids.append(uid)

        # Create 5 recent utterances (10 days)
        recent_uids = []
        for i in range(5):
            uid = utterance_manager.create_utterance(
                f"Recent utterance {i}", embedding_384d
            )
            recent_uids.append(uid)

        # Set timestamps
        with netzwerk_session.driver.session(database="neo4j") as session:
            session.run(
                """
                UNWIND $uids AS uid
                MATCH (u:Utterance {id: uid})
                SET u.timestamp = datetime() - duration({days: 40})
            """,
                {"uids": old_uids},
            )

            session.run(
                """
                UNWIND $uids AS uid
                MATCH (u:Utterance {id: uid})
                SET u.timestamp = datetime() - duration({days: 10})
            """,
                {"uids": recent_uids},
            )

        # Archive
        archived_count = utterance_manager.archive_old_utterances(days_threshold=30)

        # Assertions
        assert archived_count == 5

        active = utterance_manager.get_recent_utterances(limit=100, archived=False)
        assert len(active) == 5

        archived = utterance_manager.get_recent_utterances(limit=100, archived=True)
        assert len(archived) == 5

    def test_archival_idempotency(
        self, utterance_manager, embedding_384d, netzwerk_session
    ):
        """Archive operation is idempotent."""
        # Create old utterances
        old_uids = []
        for i in range(3):
            uid = utterance_manager.create_utterance(f"Old {i}", embedding_384d)
            old_uids.append(uid)

        # Set old timestamps
        with netzwerk_session.driver.session(database="neo4j") as session:
            session.run(
                """
                UNWIND $uids AS uid
                MATCH (u:Utterance {id: uid})
                SET u.timestamp = datetime() - duration({days: 40})
            """,
                {"uids": old_uids},
            )

        # Archive twice
        count1 = utterance_manager.archive_old_utterances(days_threshold=30)
        count2 = utterance_manager.archive_old_utterances(days_threshold=30)

        # Second call should archive 0 (already archived)
        assert count1 == 3
        assert count2 == 0


class TestUtteranceRetrieval:
    """Tests for utterance retrieval operations."""

    def test_get_recent_utterances_ordering(
        self, utterance_manager, embedding_384d, netzwerk_session
    ):
        """Recent utterances returned in reverse chronological order."""
        # Create 3 utterances with deliberate timestamps
        uids = []
        for i in range(3):
            uid = utterance_manager.create_utterance(f"Utterance {i}", embedding_384d)
            uids.append(uid)
            time.sleep(0.1)  # Ensure different timestamps

        # Retrieve
        utterances = utterance_manager.get_recent_utterances(limit=3)

        # Most recent first
        assert utterances[0]["text"] == "Utterance 2"
        assert utterances[1]["text"] == "Utterance 1"
        assert utterances[2]["text"] == "Utterance 0"

    def test_get_recent_utterances_limit(self, utterance_manager, embedding_384d):
        """Limit parameter controls number of utterances returned."""
        # Create 10 utterances
        for i in range(10):
            utterance_manager.create_utterance(f"Utterance {i}", embedding_384d)

        # Retrieve with limit
        utterances = utterance_manager.get_recent_utterances(limit=5)
        assert len(utterances) == 5

    def test_get_recent_utterances_invalid_limit(self, utterance_manager):
        """Invalid limit raises ValueError."""
        with pytest.raises(ValueError, match="Limit must be"):
            utterance_manager.get_recent_utterances(limit=0)

        with pytest.raises(ValueError, match="Limit must be"):
            utterance_manager.get_recent_utterances(limit=20000)
