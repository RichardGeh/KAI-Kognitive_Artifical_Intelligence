# tests/test_pattern_discovery_integration.py
"""
Integration tests for Pattern Discovery System.

Covers:
- Logic/Contradiction scenarios (LOGIC-01 to LOGIC-05)
- Concurrency edge case (CONC-04)
- Facade integration tests
- Property-based integration (PROP-06, PROP-07)

Total: 8+ tests
"""
import uuid
from concurrent.futures import ThreadPoolExecutor

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

# ============================================================================
# LOGIC/CONTRADICTION TESTS (LOGIC-01 to LOGIC-05)
# ============================================================================


class TestPatternDiscoveryLogic:
    """Logic and contradiction scenarios."""

    def test_logic01_double_archive_same_utterances(
        self, utterance_manager, embedding_384d, netzwerk_session
    ):
        """LOGIC-01: Archive twice, second should return 0 (idempotent)."""
        # Create old utterances
        for i in range(5):
            uid = utterance_manager.create_utterance(f"Old {i}", embedding_384d)
            with netzwerk_session.driver.session(database="neo4j") as session:
                session.run(
                    """
                    MATCH (u:Utterance {id: $uid})
                    SET u.timestamp = datetime() - duration({days: 40})
                """,
                    {"uid": uid},
                )

        # First archive
        count1 = utterance_manager.archive_old_utterances(days_threshold=30)
        assert count1 == 5

        # Second archive (idempotent)
        count2 = utterance_manager.archive_old_utterances(days_threshold=30)
        assert count2 == 0

    def test_logic02_retrieve_archived_vs_non_archived(
        self, utterance_manager, embedding_384d, netzwerk_session
    ):
        """LOGIC-02: Verify archived filter works (disjoint sets)."""
        # Create mix of archived and non-archived
        for i in range(10):
            uid = utterance_manager.create_utterance(f"Utterance {i}", embedding_384d)
            if i < 5:  # Archive first 5
                with netzwerk_session.driver.session(database="neo4j") as session:
                    session.run(
                        """
                        MATCH (u:Utterance {id: $uid})
                        SET u.timestamp = datetime() - duration({days: 40})
                    """,
                        {"uid": uid},
                    )

        utterance_manager.archive_old_utterances(days_threshold=30)

        # Retrieve archived and non-archived
        non_archived = utterance_manager.get_recent_utterances(
            limit=100, archived=False
        )
        archived = utterance_manager.get_recent_utterances(limit=100, archived=True)

        # Verify disjoint sets
        non_archived_ids = {u["id"] for u in non_archived}
        archived_ids = {u["id"] for u in archived}

        assert len(non_archived_ids & archived_ids) == 0  # No overlap
        assert len(non_archived) == 5
        assert len(archived) == 5

    def test_logic03_token_next_chain_verification(
        self, utterance_manager, embedding_384d, netzwerk_session
    ):
        """LOGIC-03: Verify NEXT chain is correct sequence (no extra relations)."""
        utterance_id = utterance_manager.create_utterance(
            "Five token sentence here now", embedding_384d
        )

        # Create 5 tokens in sequence
        for i in range(5):
            utterance_manager.create_token(
                f"Word{i}", f"word{i}", "NOUN", utterance_id, i
            )

        # Verify NEXT chain
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
            assert len(records) == 5

            # Verify sequential NEXT chain
            for i in range(4):
                assert records[i]["idx"] == i
                assert records[i]["next_idx"] == i + 1

            # Last token has no NEXT
            assert records[4]["idx"] == 4
            assert records[4]["next_idx"] is None

    def test_logic04_pattern_precision_monotonicity_not_enforced(
        self, pattern_manager, netzwerk_session
    ):
        """LOGIC-04: Precision can decrease with new evidence."""
        pattern_id = pattern_manager.create_pattern("test", "learned")

        # First update: high precision
        pattern_manager.update_pattern_stats(
            pattern_id, support_increment=1, new_precision=0.9
        )

        # Second update: lower precision (allowed)
        pattern_manager.update_pattern_stats(
            pattern_id, support_increment=1, new_precision=0.7
        )

        # Verify both updates succeeded
        with netzwerk_session.driver.session(database="neo4j") as session:
            result = session.run(
                "MATCH (p:Pattern {id: $pid}) RETURN p.precision AS precision, p.support AS support",
                {"pid": pattern_id},
            )
            record = result.single()
            assert abs(record["precision"] - 0.7) < 0.001  # Updated to lower precision
            assert record["support"] == 2  # Support increased

    def test_logic05_slot_allows_count_idempotency(
        self, pattern_manager, netzwerk_session
    ):
        """LOGIC-05: MERGE pattern ensures no duplicate ALLOWS relations."""
        slot_id = pattern_manager.create_slot("NOUN")

        # Update same lemma twice
        pattern_manager.update_slot_allowed(slot_id, "hund", count_increment=3)
        pattern_manager.update_slot_allowed(slot_id, "hund", count_increment=2)

        # Verify single ALLOWS relation with count=5
        with netzwerk_session.driver.session(database="neo4j") as session:
            result = session.run(
                """
                MATCH (s:Slot {id: $sid})-[r:ALLOWS]->(al:AllowedLemma {value: 'hund'})
                RETURN count(r) AS rel_count, r.count AS count
            """,
                {"sid": slot_id},
            )
            record = result.single()
            assert record["rel_count"] == 1  # Single relation (no duplicates)
            assert record["count"] == 5  # 3 + 2


# ============================================================================
# CONCURRENCY EDGE CASE (CONC-04)
# ============================================================================


class TestPatternDiscoveryConcurrencyEdgeCases:
    """Concurrency edge cases."""

    def test_conc04_concurrent_token_creation_same_utterance(
        self, utterance_manager, embedding_384d, netzwerk_session
    ):
        """CONC-04: Thread safety for NEXT chain construction (race allowed)."""
        utterance_id = utterance_manager.create_utterance("Test", embedding_384d)

        def create_token(idx):
            return utterance_manager.create_token(
                f"Word{idx}", f"word{idx}", "NOUN", utterance_id, idx
            )

        # 3 threads create tokens idx 0-2 concurrently
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(create_token, i) for i in range(3)]
            results = [f.result() for f in futures]

        # Verify all 3 tokens created (no duplicates)
        assert len(results) == 3
        assert len(set(results)) == 3

        # Verify tokens exist (NEXT chain may vary due to race)
        tokens = utterance_manager.get_tokens_for_utterance(utterance_id)
        assert len(tokens) == 3


# ============================================================================
# FACADE INTEGRATION TESTS
# ============================================================================


class TestPatternDiscoveryFacadeIntegration:
    """Integration tests for KonzeptNetzwerk facade."""

    def test_facade_integration_utterance_pattern_lifecycle(
        self, netzwerk_session, embedding_384d
    ):
        """Full lifecycle: Create utterance -> tokens -> pattern -> match."""
        from component_1_netzwerk_patterns import PatternManager
        from component_1_netzwerk_utterances import UtteranceManager

        utterance_mgr = UtteranceManager(netzwerk_session.driver)
        pattern_mgr = PatternManager(netzwerk_session.driver)

        # 1. Create utterance with tokens
        utterance_id = utterance_mgr.create_utterance(
            "Was ist ein Hund?", embedding_384d
        )
        utterance_mgr.create_token("Was", "was", "PRON", utterance_id, 0)
        utterance_mgr.create_token("ist", "sein", "AUX", utterance_id, 1)
        utterance_mgr.create_token("ein", "ein", "DET", utterance_id, 2)
        utterance_mgr.create_token("Hund", "hund", "NOUN", utterance_id, 3)

        # 2. Create pattern with items
        pattern_id = pattern_mgr.create_pattern("question_what_is", "seed")
        slot_id = pattern_mgr.create_slot("NOUN", allowed_values=["hund"])

        items = [
            {
                "id": str(uuid.uuid4()),
                "idx": 0,
                "kind": "LITERAL",
                "literalValue": "was",
            },
            {
                "id": str(uuid.uuid4()),
                "idx": 1,
                "kind": "LITERAL",
                "literalValue": "ist",
            },
            {
                "id": str(uuid.uuid4()),
                "idx": 2,
                "kind": "LITERAL",
                "literalValue": "ein",
            },
            {"id": str(uuid.uuid4()), "idx": 3, "kind": "SLOT", "slotId": slot_id},
        ]
        pattern_mgr.batch_create_pattern_items(pattern_id, items)

        # 3. Match utterance to pattern
        pattern_mgr.match_utterance_to_pattern(utterance_id, pattern_id, score=0.95)

        # 4. Update pattern stats
        pattern_mgr.update_pattern_stats(
            pattern_id, support_increment=1, new_precision=0.95
        )

        # 5. Verify full integration
        patterns = pattern_mgr.get_all_patterns(type_filter="seed")
        assert len(patterns) == 1
        assert patterns[0]["support"] == 1
        assert abs(patterns[0]["precision"] - 0.95) < 0.001


# ============================================================================
# PROPERTY-BASED INTEGRATION TESTS (PROP-06, PROP-07)
# ============================================================================


class TestPatternDiscoveryPropertyBasedIntegration:
    """Property-based integration tests."""

    def test_prop06_next_chain_is_acyclic(
        self, utterance_manager, embedding_384d, netzwerk_session
    ):
        """PROP-06: No cycles in NEXT relationships (DAG property)."""
        utterance_id = utterance_manager.create_utterance(
            "Token chain test", embedding_384d
        )

        # Create 10 token chain
        for i in range(10):
            utterance_manager.create_token(f"W{i}", f"w{i}", "NOUN", utterance_id, i)

        # Verify no token is reachable from itself via NEXT
        with netzwerk_session.driver.session(database="neo4j") as session:
            for i in range(10):
                result = session.run(
                    """
                    MATCH (u:Utterance {id: $uid})-[:HAS_TOKEN]->(t:Token {idx: $idx})
                    MATCH path = (t)-[:NEXT*]->(t)
                    RETURN count(path) AS cycle_count
                """,
                    {"uid": utterance_id, "idx": i},
                )
                cycle_count = result.single()["cycle_count"]
                assert cycle_count == 0  # No cycles

    @given(st.booleans())
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_prop07_archived_flag_is_boolean(
        self, utterance_manager, embedding_384d, netzwerk_session, should_archive
    ):
        """PROP-07: archived property is always true/false."""
        uid = utterance_manager.create_utterance("Test", embedding_384d)

        if should_archive:
            # Force archive by setting old timestamp
            with netzwerk_session.driver.session(database="neo4j") as session:
                session.run(
                    """
                    MATCH (u:Utterance {id: $uid})
                    SET u.timestamp = datetime() - duration({days: 40})
                """,
                    {"uid": uid},
                )
            utterance_manager.archive_old_utterances(days_threshold=30)

        # Verify archived is boolean
        utterances = utterance_manager.get_recent_utterances(
            limit=10, archived=should_archive
        )
        if utterances:
            assert isinstance(utterances[0]["archived"], bool)
