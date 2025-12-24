# test_utterance_manager_part3.py
"""
Test scenarios for UtteranceManager new methods (Part 3: Pattern Discovery).

Tests cover:
- count_utterances(): Non-archived, archived, empty database
- get_utterance(): Found, not found

Total: 5 tests
"""


class TestCountUtterances:
    """Tests for count_utterances method (UM-01 to UM-03)."""

    def test_um_01_count_non_archived(
        self, netzwerk_session, utterance_manager, embedding_384d
    ):
        """UM-01: count_utterances returns only non-archived utterances."""
        # Create 10 non-archived utterances
        for i in range(10):
            utterance_manager.create_utterance(f"Active {i}", embedding_384d)

        # Create 5 archived utterances
        for i in range(5):
            uid = utterance_manager.create_utterance(f"Archived {i}", embedding_384d)
            # Archive manually
            with netzwerk_session.driver.session(database="neo4j") as session:
                session.run(
                    """
                    MATCH (u:Utterance {id: $uid})
                    SET u.archived = true
                """,
                    {"uid": uid},
                )

        # Count non-archived
        count = utterance_manager.count_utterances(archived=False)
        assert count == 10

    def test_um_02_count_archived(
        self, netzwerk_session, utterance_manager, embedding_384d
    ):
        """UM-02: count_utterances(archived=True) returns only archived."""
        # Create 10 non-archived
        for i in range(10):
            utterance_manager.create_utterance(f"Active {i}", embedding_384d)

        # Create 5 archived
        for i in range(5):
            uid = utterance_manager.create_utterance(f"Archived {i}", embedding_384d)
            with netzwerk_session.driver.session(database="neo4j") as session:
                session.run(
                    """
                    MATCH (u:Utterance {id: $uid})
                    SET u.archived = true
                """,
                    {"uid": uid},
                )

        # Count archived
        count = utterance_manager.count_utterances(archived=True)
        assert count == 5

    def test_um_03_count_empty_database(self, netzwerk_session, utterance_manager):
        """UM-03: Empty database returns 0."""
        count = utterance_manager.count_utterances()
        assert count == 0


class TestGetUtterance:
    """Tests for get_utterance method (UM-04 to UM-05)."""

    def test_um_04_get_utterance_found(
        self, netzwerk_session, utterance_manager, embedding_384d
    ):
        """UM-04: get_utterance returns dict with all properties."""
        text = "Test utterance"
        uid = utterance_manager.create_utterance(
            text, embedding_384d, user_id="test_user"
        )

        # Get utterance
        utterance = utterance_manager.get_utterance(uid)

        # Assert properties
        assert utterance is not None
        assert utterance["id"] == uid
        assert utterance["text"] == text
        assert utterance["userId"] == "test_user"
        assert len(utterance["embedding"]) == 384
        assert "timestamp" in utterance

    def test_um_05_get_utterance_not_found(self, netzwerk_session, utterance_manager):
        """UM-05: Non-existent utterance returns None."""
        utterance = utterance_manager.get_utterance("nonexistent_uuid")
        assert utterance is None
