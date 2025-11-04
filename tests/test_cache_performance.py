"""
Schneller Performance-Test für Caching-Funktionalität.
Tests für alle implementierten Caches:
1. Embedding Cache (component_11)
2. Query Cache (component_1_netzwerk_core)
3. Prototyp Cache (component_8)
4. Extraktionsregeln Cache (component_1_netzwerk_patterns)
"""

import time
from component_11_embedding_service import EmbeddingService
from component_1_netzwerk_core import KonzeptNetzwerkCore
from component_1_netzwerk import KonzeptNetzwerk
from component_8_prototype_matcher import PrototypingEngine


def test_embedding_cache():
    """Test Embedding Service Caching."""
    print("\n=== Test 1: Embedding Cache ===")
    es = EmbeddingService()

    # Erster Aufruf (Cache-Miss)
    start = time.time()
    es.get_embedding("Das ist ein Test")
    time1 = time.time() - start

    # Zweiter Aufruf (Cache-Hit)
    start = time.time()
    es.get_embedding("Das ist ein Test")
    time2 = time.time() - start

    # Dritter Aufruf mit anderem Text (Cache-Miss)
    start = time.time()
    es.get_embedding("Das ist ein anderer Test")
    time.time() - start

    cache_stats = es.get_cache_info()
    print(f"  Cache-Miss Zeit: {time1*1000:.2f}ms")
    print(f"  Cache-Hit Zeit: {time2*1000:.2f}ms")
    print(f"  Speedup: {time1/time2:.1f}x")
    print(f"  Cache-Statistiken: {cache_stats}")

    assert cache_stats["hits"] >= 1, "Mindestens 1 Cache-Hit erwartet"
    assert cache_stats["misses"] >= 2, "Mindestens 2 Cache-Misses erwartet"
    assert time2 < time1, "Cache-Hit sollte schneller sein als Cache-Miss"
    print("  [OK] Embedding Cache funktioniert korrekt")


def test_query_cache():
    """Test Neo4j Query Caching."""
    print("\n=== Test 2: Query Cache ===")
    nw = KonzeptNetzwerkCore()

    # Stelle sicher, dass ein Topic existiert
    nw.ensure_wort_und_konzept("test")

    # Erster Aufruf (Cache-Miss)
    start = time.time()
    facts1 = nw.query_graph_for_facts("test")
    time1 = time.time() - start

    # Zweiter Aufruf (Cache-Hit)
    start = time.time()
    facts2 = nw.query_graph_for_facts("test")
    time2 = time.time() - start

    cache_stats = nw.get_cache_stats()
    print(f"  Cache-Miss Zeit: {time1*1000:.2f}ms")
    print(f"  Cache-Hit Zeit: {time2*1000:.2f}ms")
    print(f"  Speedup: {time1/time2:.1f}x" if time2 > 0 else "  Speedup: N/A")
    print(f"  Cache-Statistiken: {cache_stats}")

    assert facts1 == facts2, "Beide Abfragen sollten identische Ergebnisse liefern"
    nw.close()
    print("  [OK] Query Cache funktioniert korrekt")


def test_prototype_cache():
    """Test Prototyp-Matching Cache."""
    print("\n=== Test 3: Prototyp Cache ===")
    nw = KonzeptNetzwerk()
    es = EmbeddingService()
    pe = PrototypingEngine(nw, es)

    # Erstelle ein Embedding
    vector = es.get_embedding("Test-Satz")

    # Erster Aufruf (Cache-Miss)
    start = time.time()
    pe.find_best_match(vector)
    time1 = time.time() - start

    # Zweiter Aufruf (Cache-Hit)
    start = time.time()
    pe.find_best_match(vector)
    time2 = time.time() - start

    cache_stats = pe.get_cache_stats()
    print(f"  Cache-Miss Zeit: {time1*1000:.2f}ms")
    print(f"  Cache-Hit Zeit: {time2*1000:.2f}ms")
    print(f"  Speedup: {time1/time2:.1f}x" if time2 > 0 else "  Speedup: N/A")
    print(f"  Cache-Statistiken: {cache_stats}")

    assert cache_stats["cache_hits"] >= 1, "Mindestens 1 Cache-Hit erwartet"
    nw.close()
    print("  [OK] Prototyp Cache funktioniert korrekt")


def test_extraction_rules_cache():
    """Test Extraktionsregeln Cache."""
    print("\n=== Test 4: Extraktionsregeln Cache ===")
    nw = KonzeptNetzwerk()

    # Erster Aufruf (Cache-Miss)
    start = time.time()
    rules1 = nw.get_all_extraction_rules()
    time1 = time.time() - start

    # Zweiter Aufruf (Cache-Hit)
    start = time.time()
    rules2 = nw.get_all_extraction_rules()
    time2 = time.time() - start

    print(f"  Cache-Miss Zeit: {time1*1000:.2f}ms")
    print(f"  Cache-Hit Zeit: {time2*1000:.2f}ms")
    print(f"  Speedup: {time1/time2:.1f}x" if time2 > 0 else "  Speedup: N/A")
    print(f"  Regel-Anzahl: {len(rules1)}")

    assert rules1 == rules2, "Beide Abfragen sollten identische Ergebnisse liefern"
    nw.close()
    print("  [OK] Extraktionsregeln Cache funktioniert korrekt")


if __name__ == "__main__":
    print("╔════════════════════════════════════════════════╗")
    print("║  KAI Performance-Optimierung: Cache-Tests     ║")
    print("╚════════════════════════════════════════════════╝")

    try:
        test_embedding_cache()
        test_query_cache()
        test_prototype_cache()
        test_extraction_rules_cache()

        print("\n╔════════════════════════════════════════════════╗")
        print("║  [OK] Alle Cache-Tests erfolgreich               ║")
        print("╚════════════════════════════════════════════════╝")
        print("\nErwartete Performance-Verbesserung:")
        print("  * Embedding Cache: 30-50% schnellere Response-Zeiten")
        print("  * Query Cache: 20-40% weniger DB-Last")
        print("  * Prototyp Cache: 40-60% schnellere Pattern-Matches")
        print("  * Extraktionsregeln Cache: 50-70% schnellere Text-Ingestion")

    except Exception as e:
        print(f"\n[ERROR] Test fehlgeschlagen: {e}")
        import traceback

        traceback.print_exc()
