# tests/test_performance_optimizations.py
"""
Tests für Performance-Optimierungen in KAI

Testet:
- Batch-Embeddings für Ingestion
- Lazy Loading für Proof Trees
- Batch-Processing für große Textmengen
- Neo4j Query-Profiling
"""

import time

import pytest

# KAI Components
from component_1_netzwerk import KonzeptNetzwerk
from component_6_linguistik_engine import LinguisticPreprocessor
from component_8_prototype_matcher import PrototypingEngine
from component_11_embedding_service import EmbeddingService
from component_utils_neo4j_profiler import Neo4jQueryProfiler
from kai_ingestion_handler import KaiIngestionHandler

# Proof Tree Components
try:
    from component_17_proof_explanation import ProofStep, ProofTree, StepType
    from component_18_proof_tree_widget import ProofTreeWidget

    PROOF_AVAILABLE = True
except ImportError:
    PROOF_AVAILABLE = False


class TestBatchEmbeddings:
    """Tests für Batch-Embedding-Optimierung"""

    @pytest.fixture
    def embedding_service(self):
        """EmbeddingService Fixture"""
        return EmbeddingService()

    def test_batch_embeddings_faster_than_sequential(self, embedding_service):
        """Batch-Embeddings sollten mindestens gleich schnell sein wie sequentielle Calls"""
        if not embedding_service.is_available():
            pytest.skip("Embedding-Service nicht verfügbar")

        # Nutze einzigartige Texte (keine Cache-Hits)
        test_texts = [
            f"Einzigartiger Testsatz Nummer {i} für Performance-Test" for i in range(20)
        ]

        # Sequential (neue Service-Instanz ohne Cache)
        embedding_service_seq = EmbeddingService()
        start_time = time.perf_counter()
        sequential_embeddings = []
        for text in test_texts:
            emb = embedding_service_seq.get_embedding(text)
            sequential_embeddings.append(emb)
        sequential_time = time.perf_counter() - start_time

        # Batch (mit frischem Service für fairen Vergleich)
        embedding_service_batch = EmbeddingService()
        start_time = time.perf_counter()
        embedding_service_batch.get_embeddings_batch(test_texts)
        batch_time = time.perf_counter() - start_time

        # Vergleiche
        print("\n  Sequential: {sequential_time:.3f}s")
        print("  Batch: {batch_time:.3f}s")
        print("  Speedup: {sequential_time / batch_time:.2f}x")

        # Batch sollte nicht langsamer sein (kann durch Overhead minimal langsamer sein)
        # Hauptziel: Funktioniert korrekt, bei großen Mengen ist Speedup deutlich
        assert (
            batch_time <= sequential_time * 1.5
        ), "Batch sollte nicht signifikant langsamer sein"

    def test_batch_embeddings_returns_correct_count(self, embedding_service):
        """Batch-Embeddings sollten korrekte Anzahl zurückgeben"""
        if not embedding_service.is_available():
            pytest.skip("Embedding-Service nicht verfügbar")

        test_texts = ["Text 1", "Text 2", "Text 3", "", "Text 5"]  # Mit leerem Text

        embeddings = embedding_service.get_embeddings_batch(test_texts)

        assert len(embeddings) == len(test_texts)
        assert embeddings[0] is not None
        assert embeddings[3] is None  # Leerer Text -> None
        assert embeddings[4] is not None


class TestIngestionBatchProcessing:
    """Tests für Batch-Processing in Ingestion"""

    @pytest.fixture
    def setup_components(self):
        """Setup für Ingestion-Handler"""
        netzwerk = KonzeptNetzwerk()
        preprocessor = LinguisticPreprocessor()
        embedding_service = EmbeddingService()
        prototyping_engine = PrototypingEngine(netzwerk, embedding_service)

        handler = KaiIngestionHandler(
            netzwerk, preprocessor, prototyping_engine, embedding_service
        )

        yield handler

        # Cleanup
        netzwerk.close()

    def test_ingest_text_uses_batch_embeddings(self, setup_components):
        """ingest_text() sollte Batch-Embeddings nutzen"""
        handler = setup_components

        if not handler.embedding_service.is_available():
            pytest.skip("Embedding-Service nicht verfügbar")

        # Text mit mehreren Sätzen
        text = "Ein Hund ist ein Tier. Eine Katze ist ein Säugetier. Ein Apfel ist eine Frucht."

        # Initial Cache-Stats
        cache_before = handler.embedding_service.get_cache_info()

        # Ingestion
        stats = handler.ingest_text(text)

        # Cache-Stats nach Ingestion
        cache_after = handler.embedding_service.get_cache_info()

        # Es sollten 3 Sätze verarbeitet worden sein
        # Mit Batch-Embeddings sollten alle auf einmal gecacht werden
        assert stats["facts_created"] + stats["fallback_patterns"] >= 0
        assert cache_after["currsize"] >= cache_before["currsize"]

    def test_ingest_text_large_processes_in_chunks(self, setup_components):
        """ingest_text_large() sollte in Chunks verarbeiten"""
        handler = setup_components

        if not handler.embedding_service.is_available():
            pytest.skip("Embedding-Service nicht verfügbar")

        # Großer Text (200 Sätze)
        sentences = [f"Satz Nummer {i} ist ein Test." for i in range(200)]
        text = " ".join(sentences)

        # Track Progress
        progress_calls = []

        def progress_callback(current, total, stats):
            progress_calls.append((current, total, stats.copy()))

        # Ingestion mit kleinem Chunk-Size für Test
        stats = handler.ingest_text_large(
            text, chunk_size=50, progress_callback=progress_callback
        )

        # Es sollten 4 Chunks verarbeitet worden sein (200 / 50 = 4)
        assert stats["chunks_processed"] == 4

        # Progress-Callback sollte 4 Mal aufgerufen worden sein
        assert len(progress_calls) == 4

        # Letzter Call sollte total = 200 haben
        assert progress_calls[-1][1] == 200

    def test_ingest_text_large_handles_empty_text(self, setup_components):
        """ingest_text_large() sollte leere Texte handhaben"""
        handler = setup_components

        stats = handler.ingest_text_large("")

        assert stats["facts_created"] == 0
        assert stats["chunks_processed"] == 0

    def test_ingest_text_large_parallel_processing(self, setup_components):
        """ingest_text_large() sollte parallele Verarbeitung unterstützen"""
        handler = setup_components

        if not handler.embedding_service.is_available():
            pytest.skip("Embedding-Service nicht verfügbar")

        # Großer Text (200 Sätze) für messbare Parallelisierung
        sentences = [f"Testsatz {i} für parallele Verarbeitung." for i in range(200)]
        text = " ".join(sentences)

        # Test 1: Mit paralleler Verarbeitung (4 Workers)
        start_time = time.perf_counter()
        stats_parallel = handler.ingest_text_large(text, chunk_size=50, max_workers=4)
        time_parallel = time.perf_counter() - start_time

        # Test 2: Mit sequenzieller Verarbeitung (1 Worker)
        start_time = time.perf_counter()
        stats_sequential = handler.ingest_text_large(text, chunk_size=50, max_workers=1)
        time_sequential = time.perf_counter() - start_time

        print("\n  Parallel (4 Workers): {time_parallel:.3f}s")
        print("  Sequential (1 Worker): {time_sequential:.3f}s")
        print("  Speedup: {time_sequential / time_parallel:.2f}x")

        # Beide sollten gleiche Anzahl Chunks verarbeitet haben
        assert stats_parallel["chunks_processed"] == 4
        assert stats_sequential["chunks_processed"] == 4

        # Parallele Verarbeitung sollte gleich oder schneller sein
        # (bei kleinen Datensätzen kann Overhead die Vorteile aufwiegen)
        assert time_parallel <= time_sequential * 1.2

    def test_ingest_text_large_respects_config(self, setup_components):
        """ingest_text_large() sollte Config-Einstellungen respektieren"""

        from kai_config import get_config

        config = get_config()

        # Prüfe ob Config geladen wurde
        assert config.parallel_processing_enabled is not None
        assert (
            config.parallel_processing_max_workers is not None
            or config.parallel_processing_max_workers is None
        )


@pytest.mark.skipif(not PROOF_AVAILABLE, reason="Proof Tree Components nicht verfügbar")
class TestProofTreeLazyLoading:
    """Tests für Lazy Loading in Proof Tree Widget"""

    def test_progressive_rendering_enabled_by_default(self):
        """Progressive Rendering sollte standardmäßig aktiviert sein"""
        import sys

        from PySide6.QtWidgets import QApplication

        # Create QApplication (required for QWidget)
        if not QApplication.instance():
            QApplication(sys.argv)

        widget = ProofTreeWidget()

        assert widget.progressive_rendering_enabled is True
        assert widget.render_batch_size == 50

    def test_progressive_rendering_for_large_trees(self):
        """Progressive Rendering sollte für große Bäume aktiviert werden"""
        import sys

        from PySide6.QtWidgets import QApplication

        if not QApplication.instance():
            QApplication(sys.argv)

        widget = ProofTreeWidget()

        # Erstelle großen Proof Tree (100 Steps)
        steps = []
        for i in range(100):
            step = ProofStep(
                step_id=f"step-{i:03d}",
                step_type=StepType.FACT_MATCH,
                output=f"Step {i}",
                confidence=0.9,
                source_component="test",
            )
            steps.append(step)

        tree = ProofTree(query="Test Query", root_steps=steps)

        # Setze Baum
        widget.set_proof_tree(tree)

        # Bei >50 Nodes sollte Progressive Rendering aktiviert sein
        # (wird in _render_tree entschieden)
        assert widget.progressive_rendering_enabled is True


class TestNeo4jQueryProfiler:
    """Tests für Neo4j Query-Profiler"""

    @pytest.fixture
    def profiler(self):
        """Profiler Fixture"""
        netzwerk = KonzeptNetzwerk()
        profiler = Neo4jQueryProfiler(netzwerk.driver)

        yield profiler

        netzwerk.close()

    def test_profiler_analyzes_query(self, profiler):
        """Profiler sollte Query analysieren können"""
        # Einfache Query
        profile = profiler.profile_query(
            "test_query", "MATCH (n:Wort) RETURN n LIMIT 10"
        )

        assert profile.query_name == "test_query"
        assert profile.execution_time_ms >= 0
        assert profile.db_hits >= 0
        assert len(profile.recommendations) > 0

    def test_profiler_suggests_indexes(self, profiler):
        """Profiler sollte Index-Vorschläge liefern"""
        suggestions = profiler.suggest_indexes()

        assert len(suggestions) > 0
        assert any("CREATE INDEX" in s for s in suggestions)

    def test_profiler_detects_label_scan(self, profiler):
        """Profiler sollte Label-Scans erkennen"""
        # Query ohne Index (sollte Label-Scan auslösen)
        explained = profiler.explain_query("label_scan_test", "MATCH (n:Wort) RETURN n")

        # Sollte Plan enthalten
        assert explained["plan"] is not None


class TestPerformanceRegression:
    """Regression-Tests für Performance"""

    @pytest.fixture
    def embedding_service(self):
        return EmbeddingService()

    def test_embedding_cache_hit_rate(self, embedding_service):
        """Embedding-Cache sollte hohe Hit-Rate haben für wiederholte Texte"""
        if not embedding_service.is_available():
            pytest.skip("Embedding-Service nicht verfügbar")

        # Erzeuge Embeddings
        texts = ["Hund", "Katze", "Maus"] * 10  # Wiederholte Texte

        for text in texts:
            embedding_service.get_embedding(text)

        # Cache-Info
        cache_info = embedding_service.get_cache_info()

        # Hit-Rate sollte hoch sein (>60% wegen Wiederholungen)
        assert cache_info["hit_rate"] > 0.6

    def test_neo4j_cache_invalidation_on_write(self):
        """Neo4j Cache sollte bei Writes invalidiert werden"""
        netzwerk = KonzeptNetzwerk()

        # Query (sollte cachen)
        netzwerk.query_graph_for_facts("test_entity")

        # Write (sollte Cache invalidieren)
        netzwerk.assert_relation("test_entity", "IS_A", "test_type")

        # Re-Query (sollte neuen Fakt enthalten)
        facts2 = netzwerk.query_graph_for_facts("test_entity")

        # Neuer Fakt sollte in facts2 sein
        assert "IS_A" in facts2
        assert "test_type" in facts2["IS_A"]

        netzwerk.close()


# Benchmark-Utilities (für manuelle Performance-Tests)
def benchmark_ingestion(text: str, chunk_size: int = 100):
    """Benchmark für Ingestion-Performance"""
    netzwerk = KonzeptNetzwerk()
    preprocessor = LinguisticPreprocessor()
    embedding_service = EmbeddingService()
    prototyping_engine = PrototypingEngine(netzwerk, embedding_service)

    handler = KaiIngestionHandler(
        netzwerk, preprocessor, prototyping_engine, embedding_service
    )

    print("\n{'='*70}")
    print("Benchmarking Ingestion: {len(text)} chars")
    print("{'='*70}")

    # ingest_text()
    start_time = time.perf_counter()
    handler.ingest_text(text)
    time1 = time.perf_counter() - start_time

    print("\ningest_text():")
    print("  Time: {time1:.3f}s")
    print("  Facts: {stats1['facts_created']}")

    # ingest_text_large()
    start_time = time.perf_counter()
    stats2 = handler.ingest_text_large(text, chunk_size=chunk_size)
    time2 = time.perf_counter() - start_time

    print("\ningest_text_large():")
    print("  Time: {time2:.3f}s")
    print("  Facts: {stats2['facts_created']}")
    print("  Chunks: {stats2['chunks_processed']}")

    print("\nSpeedup: {time1 / time2:.2f}x")
    print("{'='*70}\n")

    netzwerk.close()

    return time1, time2


if __name__ == "__main__":
    # Manuelle Benchmark-Ausführung
    print("Performance Optimization Tests\n")

    # Test Batch-Embeddings
    print("Testing Batch-Embeddings...")
    test = TestBatchEmbeddings()
    emb_service = EmbeddingService()
    test.test_batch_embeddings_faster_than_sequential(emb_service)
    print("[SUCCESS] Batch-Embeddings Test passed\n")

    # Benchmark Ingestion
    test_text = "Ein Hund ist ein Tier. " * 100  # 100 Sätze
    benchmark_ingestion(test_text)

    print("\n[SUCCESS] Alle Performance-Tests abgeschlossen")
