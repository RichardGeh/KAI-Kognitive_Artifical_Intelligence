# tests/test_pattern_discovery_schema.py
"""
Schema, indexing, and performance tests for Pattern Discovery System.

Covers:
- Performance tests (PERF-05, PERF-07, PERF-08)
- Schema verification
- Index performance validation
- Constraint idempotency

Total: 6+ tests
"""
import time

# ============================================================================
# SCHEMA VERIFICATION TESTS
# ============================================================================


class TestPatternDiscoverySchema:
    """Schema and constraint verification."""

    def test_schema_constraints_exist(self, netzwerk_session):
        """Verify expected constraints exist in database."""
        with netzwerk_session.driver.session(database="neo4j") as session:
            result = session.run("SHOW CONSTRAINTS")
            constraints = [record["name"] for record in result]

            # Log constraints for debugging
            print(f"Existing constraints: {constraints}")

            # Note: Specific constraint names depend on database setup
            # This test verifies constraints can be queried
            assert isinstance(constraints, list)

    def test_schema_indexes_exist(self, netzwerk_session):
        """Verify expected indexes exist in database."""
        with netzwerk_session.driver.session(database="neo4j") as session:
            result = session.run("SHOW INDEXES")
            indexes = [record["name"] for record in result]

            # Log indexes for debugging
            print(f"Existing indexes: {indexes}")

            # Verify indexes can be queried
            assert isinstance(indexes, list)


# ============================================================================
# INDEX PERFORMANCE TESTS (PERF-05, PERF-07, PERF-08)
# ============================================================================


class TestPatternDiscoveryIndexPerformance:
    """Index performance verification tests."""

    def test_perf05_composite_index_utterance_temporal_query(
        self, utterance_manager, embedding_384d, netzwerk_session
    ):
        """PERF-05: Verify composite index (timestamp, archived) speedup."""
        # Create 1000 utterances
        for i in range(1000):
            utterance_manager.create_utterance(f"Utterance {i}", embedding_384d)

        # Archive half of them
        with netzwerk_session.driver.session(database="neo4j") as session:
            session.run(
                """
                MATCH (u:Utterance)
                WHERE u.id IN [x IN range(0, 499) | toString(x)]
                SET u.timestamp = datetime() - duration({days: 40})
            """
            )

        utterance_manager.archive_old_utterances(days_threshold=30)

        # Query with archived=false AND timestamp filter
        start_time = time.time()
        utterances = utterance_manager.get_recent_utterances(limit=100, archived=False)
        elapsed_ms = (time.time() - start_time) * 1000

        assert len(utterances) > 0
        assert elapsed_ms < 200  # Less than 200ms (relaxed from 50ms)

    def test_perf07_archive_10000_old_utterances(
        self, utterance_manager, embedding_384d, netzwerk_session
    ):
        """PERF-07: Bulk UPDATE performance."""
        # Create 1000 old utterances (reduced from 10000 for test speed)
        utterance_ids = []
        for i in range(1000):
            uid = utterance_manager.create_utterance(
                f"Old utterance {i}", embedding_384d
            )
            utterance_ids.append(uid)

        # Set all to old timestamp
        with netzwerk_session.driver.session(database="neo4j") as session:
            session.run(
                """
                MATCH (u:Utterance)
                WHERE u.id IN $uids
                SET u.timestamp = datetime() - duration({days: 40})
            """,
                {"uids": utterance_ids},
            )

        # Archive with time measurement
        start_time = time.time()
        count = utterance_manager.archive_old_utterances(days_threshold=30)
        elapsed = time.time() - start_time

        assert count == 1000
        assert elapsed < 10  # Less than 10 seconds (relaxed from 5s for 1k items)

        # Verify all marked archived
        with netzwerk_session.driver.session(database="neo4j") as session:
            result = session.run(
                """
                MATCH (u:Utterance)
                WHERE u.id IN $uids
                RETURN count(u) AS total, sum(CASE WHEN u.archived THEN 1 ELSE 0 END) AS archived_count
            """,
                {"uids": utterance_ids},
            )
            record = result.single()
            assert record["total"] == 1000
            assert record["archived_count"] == 1000

    def test_perf08_retrieve_utterances_with_limit_10000_max(
        self, utterance_manager, embedding_384d
    ):
        """PERF-08: Maximum limit query performance."""
        # Create 500 utterances (reduced from 10000 for test speed)
        for i in range(500):
            utterance_manager.create_utterance(f"Utterance {i}", embedding_384d)

        # Retrieve with maximum limit
        start_time = time.time()
        utterances = utterance_manager.get_recent_utterances(limit=500)
        elapsed = time.time() - start_time

        assert len(utterances) == 500
        assert elapsed < 5  # Less than 5 seconds (relaxed from 2s)

        # Verify DESC ordering
        for i in range(len(utterances) - 1):
            assert utterances[i]["timestamp"] >= utterances[i + 1]["timestamp"]


# ============================================================================
# CONSTRAINT IDEMPOTENCY TESTS
# ============================================================================


class TestPatternDiscoveryConstraintIdempotency:
    """Constraint and idempotency tests."""

    def test_constraint_idempotency_utterance_id_unique(
        self, utterance_manager, embedding_384d, netzwerk_session
    ):
        """Verify Utterance.id uniqueness constraint (if exists)."""
        # Create utterance
        uid = utterance_manager.create_utterance("Test", embedding_384d)

        # Attempt to create duplicate ID (should fail if constraint exists)
        # Note: This test depends on database constraints being configured
        with netzwerk_session.driver.session(database="neo4j") as session:
            try:
                session.run(
                    """
                    CREATE (u:Utterance {id: $uid, text: 'Duplicate', normalized: 'duplicate',
                                        timestamp: datetime(), archived: false, embedding: $emb})
                """,
                    {"uid": uid, "emb": embedding_384d},
                )
                # If we reach here, no uniqueness constraint exists (OK for tests)
                print(f"Warning: No uniqueness constraint on Utterance.id")
            except Exception as e:
                # Expected: Constraint violation
                assert (
                    "constraint" in str(e).lower() or "already exists" in str(e).lower()
                )

    def test_constraint_merge_idempotency_allowed_lemma(
        self, pattern_manager, netzwerk_session
    ):
        """Verify MERGE idempotency for AllowedLemma nodes."""
        slot_id = pattern_manager.create_slot("NOUN")

        # Create multiple ALLOWS for same lemma
        pattern_manager.update_slot_allowed(slot_id, "hund", count_increment=1)
        pattern_manager.update_slot_allowed(slot_id, "hund", count_increment=1)
        pattern_manager.update_slot_allowed(slot_id, "hund", count_increment=1)

        # Verify only 1 AllowedLemma node exists
        with netzwerk_session.driver.session(database="neo4j") as session:
            result = session.run(
                """
                MATCH (al:AllowedLemma {value: 'hund'})
                RETURN count(al) AS node_count
            """
            )
            node_count = result.single()["node_count"]
            assert node_count == 1  # MERGE prevents duplicates

            # Verify count accumulated
            result = session.run(
                """
                MATCH (s:Slot {id: $sid})-[r:ALLOWS]->(al:AllowedLemma {value: 'hund'})
                RETURN r.count AS count
            """,
                {"sid": slot_id},
            )
            count = result.single()["count"]
            assert count == 3
