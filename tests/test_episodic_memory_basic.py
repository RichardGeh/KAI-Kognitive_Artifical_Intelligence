"""
KAI Test Suite - Episodic Memory Basic Tests
Basis-Tests aus test_kai_worker.py extrahiert.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from kai_worker import KaiWorker

logger = logging.getLogger(__name__)
TEST_VECTOR_DIM = 384


class TestEpisodicMemory:
    """
    Testet das Episodische Gedächtnis-System (PHASE 3).

    Funktionen:
    - Episode-Erstellung bei Text-Ingestion
    - Verknüpfung von Fakten mit Episoden
    - Query nach Episoden über ein Thema
    - Löschen von Episoden
    """

    def test_episode_creation_during_ingestion(
        self, netzwerk_session, embedding_service_session
    ):
        """
        Test: Beim Ingestieren eines Textes wird eine Episode erstellt
        und alle extrahierten Fakten werden mit der Episode verknüpft.
        """
        netzwerk = netzwerk_session
        embedding_service = embedding_service_session
        worker = KaiWorker(netzwerk, embedding_service)

        # Ingestiere Text mit bekanntem Fakt
        text = "Ein test_katze_episodic ist ein test_tier_episodic."
        stats = worker.ingestion_handler.ingest_text(text)

        # Prüfe dass ein Fakt erstellt wurde
        assert (
            stats["facts_created"] >= 1
        ), "Mindestens ein Fakt sollte erstellt worden sein"

        # Prüfe dass eine Episode existiert
        episodes = netzwerk.query_all_episodes(episode_type="ingestion", limit=5)
        assert len(episodes) > 0, "Mindestens eine Episode sollte existiert"

        # Finde die Episode für unseren Text
        matching_episode = None
        for ep in episodes:
            if "test_katze_episodic" in ep["content"].lower():
                matching_episode = ep
                break

        assert (
            matching_episode is not None
        ), "Episode für unseren Text sollte gefunden werden"

        # Prüfe dass Fakten mit der Episode verknüpft sind
        learned_facts = matching_episode.get("learned_facts", [])
        assert len(learned_facts) > 0, "Episode sollte verknüpfte Fakten haben"

        logger.info(
            f"[SUCCESS] Episode bei Ingestion erstellt: {matching_episode['episode_id'][:8]}"
        )
        logger.info(f"[SUCCESS] {len(learned_facts)} Fakten mit Episode verknüpft")

        # Cleanup
        netzwerk.delete_episode(matching_episode["episode_id"], cascade=True)

    def test_query_episodes_about_topic(
        self, netzwerk_session, embedding_service_session
    ):
        """
        Test: Query nach allen Episoden, in denen über ein bestimmtes Thema gelernt wurde.
        """
        netzwerk = netzwerk_session

        # Erstelle Episode manuell
        test_text = "Ein test_hund_ep ist ein test_tier_ep."
        episode_id = netzwerk.create_episode(
            episode_type="ingestion",
            content=test_text,
            metadata={"test": "episodic_query"},
        )

        assert episode_id is not None, "Episode sollte erstellt worden sein"

        # Erstelle Fakt und verknüpfe mit Episode
        created = netzwerk.assert_relation(
            "test_hund_ep", "IS_A", "test_tier_ep", test_text
        )
        assert created, "Fakt sollte erstellt worden sein"

        link_success = netzwerk.link_fact_to_episode(
            "test_hund_ep", "IS_A", "test_tier_ep", episode_id
        )
        assert link_success, "Fakt sollte mit Episode verknüpft worden sein"

        # Query nach Episoden über "test_hund_ep"
        episodes = netzwerk.query_episodes_about("test_hund_ep", limit=10)

        assert len(episodes) > 0, "Mindestens eine Episode sollte gefunden werden"

        # Finde unsere Episode
        our_episode = None
        for ep in episodes:
            if ep["episode_id"] == episode_id:
                our_episode = ep
                break

        assert our_episode is not None, "Unsere Episode sollte gefunden werden"
        assert (
            len(our_episode["learned_facts"]) > 0
        ), "Episode sollte verknüpfte Fakten haben"

        # Prüfe dass der Fakt korrekt verknüpft ist
        fact = our_episode["learned_facts"][0]
        assert fact["subject"] == "test_hund_ep"
        assert fact["relation"] == "IS_A"
        assert fact["object"] == "test_tier_ep"

        logger.info(f"[SUCCESS] Query nach Episoden über 'test_hund_ep' erfolgreich")
        logger.info(
            f"[SUCCESS] Episode {episode_id[:8]} gefunden mit {len(our_episode['learned_facts'])} Fakten"
        )

        # Cleanup
        netzwerk.delete_episode(episode_id, cascade=True)

    def test_episode_deletion_with_cascade(
        self, netzwerk_session, embedding_service_session
    ):
        """
        Test: Löschen einer Episode mit cascade=True löscht auch die verknüpften Fakten.
        """
        netzwerk = netzwerk_session

        # Erstelle Episode und Fakt
        test_text = "Ein test_delete_ep ist ein test_delete_ep_tier."
        episode_id = netzwerk.create_episode(
            episode_type="test", content=test_text, metadata={"test": "cascade_delete"}
        )

        # Erstelle Fakt
        netzwerk.assert_relation(
            "test_delete_ep", "IS_A", "test_delete_ep_tier", test_text
        )
        netzwerk.link_fact_to_episode(
            "test_delete_ep", "IS_A", "test_delete_ep_tier", episode_id
        )

        # Prüfe dass Fakt existiert
        facts = netzwerk.query_graph_for_facts("test_delete_ep")
        assert "IS_A" in facts, "Fakt sollte existieren"

        # Lösche Episode MIT cascade
        success = netzwerk.delete_episode(episode_id, cascade=True)
        assert success, "Episode sollte gelöscht worden sein"

        # Prüfe dass Fakt AUCH gelöscht wurde
        facts_after = netzwerk.query_graph_for_facts("test_delete_ep")
        assert (
            "IS_A" not in facts_after
        ), "Fakt sollte auch gelöscht worden sein (cascade=True)"

        logger.info(f"[SUCCESS] Episode gelöscht mit cascade=True")
        logger.info(f"[SUCCESS] Verknüpfte Fakten wurden ebenfalls gelöscht")

    def test_episode_deletion_without_cascade(
        self, netzwerk_session, embedding_service_session
    ):
        """
        Test: Löschen einer Episode ohne cascade behält die Fakten.
        """
        netzwerk = netzwerk_session

        # Erstelle Episode und Fakt
        test_text = "Ein test_keep_ep ist ein test_keep_ep_tier."
        episode_id = netzwerk.create_episode(
            episode_type="test", content=test_text, metadata={"test": "keep_facts"}
        )

        # Erstelle Fakt
        netzwerk.assert_relation("test_keep_ep", "IS_A", "test_keep_ep_tier", test_text)
        netzwerk.link_fact_to_episode(
            "test_keep_ep", "IS_A", "test_keep_ep_tier", episode_id
        )

        # Prüfe dass Fakt existiert
        facts = netzwerk.query_graph_for_facts("test_keep_ep")
        assert "IS_A" in facts, "Fakt sollte existieren"

        # Lösche Episode OHNE cascade
        success = netzwerk.delete_episode(episode_id, cascade=False)
        assert success, "Episode sollte gelöscht worden sein"

        # Prüfe dass Fakt NOCH existiert
        facts_after = netzwerk.query_graph_for_facts("test_keep_ep")
        assert "IS_A" in facts_after, "Fakt sollte noch existieren (cascade=False)"

        logger.info(f"[SUCCESS] Episode gelöscht ohne cascade")
        logger.info(f"[SUCCESS] Fakten wurden beibehalten")

        # Cleanup
        # Lösche Fakt manuell (da cascade=False)
        # (Neo4j hat keine direkte "DELETE RELATIONSHIP" ohne Cascade, daher löschen wir Konzepte)


# ============================================================================
# TESTS FÜR BACKWARD-CHAINING UND MULTI-HOP REASONING (Phase 3)
# ============================================================================
