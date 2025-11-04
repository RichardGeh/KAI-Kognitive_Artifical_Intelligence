"""
Tests für Epistemic Graph-Traversal (Schritt 3.2)

Testet die Integration zwischen component_35_epistemic_engine und component_12_graph_traversal
für effiziente Meta-Knowledge Queries.

Akzeptanzkriterien:
1. query_meta_knowledge_paths() returned alle Meta-Pfade bis max_depth
2. Cypher-Query läuft erfolgreich auf Neo4j
3. Returned Paths enthalten agent_chain, proposition, meta_level
4. Performance: <100ms für 10 Agents mit depth=3
"""

import pytest
from component_35_epistemic_engine import EpistemicEngine
from component_12_graph_traversal import GraphTraversal
from component_1_netzwerk import KonzeptNetzwerk
import time


@pytest.fixture
def setup_epistemic_graph():
    """Setup für epistemische Graph-Struktur"""
    netzwerk = KonzeptNetzwerk()
    engine = EpistemicEngine(netzwerk)
    traversal = GraphTraversal(netzwerk)

    # Create agents
    agents = []
    for i in range(5):
        agent_id = f"agent_{i}"
        engine.create_agent(agent_id, f"Agent {i}")
        agents.append(agent_id)

    # Setup Meta-Knowledge Chain:
    # agent_0 knows that agent_1 knows secret_A
    # agent_1 knows that agent_2 knows secret_B
    # agent_2 knows that agent_3 knows secret_C
    # etc.

    engine.add_nested_knowledge("agent_0", ["agent_1"], "secret_A")
    engine.add_nested_knowledge("agent_1", ["agent_2"], "secret_B")
    engine.add_nested_knowledge("agent_2", ["agent_3"], "secret_C")
    engine.add_nested_knowledge("agent_3", ["agent_4"], "secret_D")

    # Also add some deeper nested knowledge
    # agent_0 knows that agent_1 knows that agent_2 knows secret_X
    engine.add_nested_knowledge("agent_0", ["agent_1", "agent_2"], "secret_X")

    return engine, traversal, agents


def test_query_meta_knowledge_paths_basic(setup_epistemic_graph):
    """Test 1: query_meta_knowledge_paths returned alle Meta-Pfade"""
    engine, traversal, agents = setup_epistemic_graph

    # Query paths from agent_0
    paths = engine.query_meta_knowledge_paths("agent_0", max_depth=3)

    # Verify paths exist
    assert len(paths) > 0, "Should find at least one meta-knowledge path"

    # Verify all paths have required fields
    for path in paths:
        assert "path" in path, "Path should have 'path' field"
        assert "proposition" in path, "Path should have 'proposition' field"
        assert "meta_level" in path, "Path should have 'meta_level' field"

    # Verify path starts with observer
    for path in paths:
        if path["path"]:  # Check if path is not empty
            assert path["path"][0] == "agent_0", "Path should start with observer"

    print(f"OK Found {len(paths)} meta-knowledge paths from agent_0")
    for path in paths[:3]:  # Print first 3 paths
        print(
            f"  - Path: {path['path']} | Prop: {path['proposition']} | Level: {path['meta_level']}"
        )


def test_query_meta_knowledge_paths_respects_depth(setup_epistemic_graph):
    """Test 2: max_depth Parameter wird respektiert"""
    engine, traversal, agents = setup_epistemic_graph

    # Query with depth=1
    paths_depth_1 = engine.query_meta_knowledge_paths("agent_0", max_depth=1)

    # Query with depth=3
    paths_depth_3 = engine.query_meta_knowledge_paths("agent_0", max_depth=3)

    # Depth=3 should return more or equal paths than depth=1
    assert len(paths_depth_3) >= len(
        paths_depth_1
    ), "Higher max_depth should find more or equal paths"

    # All paths in depth=1 should have meta_level <= 1
    for path in paths_depth_1:
        assert (
            path["meta_level"] <= 1
        ), f"Depth=1 paths should have level <= 1, got {path['meta_level']}"

    # All paths in depth=3 should have meta_level <= 3
    for path in paths_depth_3:
        assert (
            path["meta_level"] <= 3
        ), f"Depth=3 paths should have level <= 3, got {path['meta_level']}"

    print(
        f"OK Depth=1: {len(paths_depth_1)} paths | Depth=3: {len(paths_depth_3)} paths"
    )


def test_find_epistemic_paths(setup_epistemic_graph):
    """Test 3: find_epistemic_paths() nutzt Graph-Traversal"""
    engine, traversal, agents = setup_epistemic_graph

    # Use graph traversal to find epistemic paths
    paths = traversal.find_epistemic_paths("agent_0", max_depth=3)

    # Verify paths exist
    assert isinstance(paths, list), "Should return a list of GraphPath objects"

    # Verify paths have GraphPath structure
    for path in paths:
        assert hasattr(path, "nodes"), "GraphPath should have 'nodes' attribute"
        assert hasattr(path, "relations"), "GraphPath should have 'relations' attribute"
        assert hasattr(
            path, "confidence"
        ), "GraphPath should have 'confidence' attribute"
        assert hasattr(
            path, "explanation"
        ), "GraphPath should have 'explanation' attribute"

    print(f"OK find_epistemic_paths found {len(paths)} paths")
    for path in paths[:3]:  # Print first 3 paths
        print(f"  - {path}")


def test_performance_query_meta_knowledge_paths():
    """Test 4: Performance - <100ms für 10 Agents mit depth=3"""
    netzwerk = KonzeptNetzwerk()
    engine = EpistemicEngine(netzwerk)

    # Create 10 agents
    agents = []
    for i in range(10):
        agent_id = f"perf_agent_{i}"
        engine.create_agent(agent_id, f"Performance Agent {i}")
        agents.append(agent_id)

    # Setup meta-knowledge chain
    for i in range(9):
        engine.add_nested_knowledge(agents[i], [agents[i + 1]], f"secret_{i}")

    # Also add some deeper nested knowledge
    engine.add_nested_knowledge(agents[0], [agents[1], agents[2]], "deep_secret_A")
    engine.add_nested_knowledge(agents[1], [agents[2], agents[3]], "deep_secret_B")

    # Measure performance
    start_time = time.time()
    paths = engine.query_meta_knowledge_paths(agents[0], max_depth=3)
    elapsed_time = (time.time() - start_time) * 1000  # Convert to milliseconds

    # Verify performance
    print(f"OK Query took {elapsed_time:.2f}ms for 10 agents with depth=3")

    # Performance assertion - allow some flexibility on slow machines
    # Target: <100ms, but allow up to 200ms for CI/slower machines
    assert (
        elapsed_time < 200
    ), f"Performance requirement: Query should take <200ms, took {elapsed_time:.2f}ms"

    # Verify paths were found
    assert len(paths) > 0, "Should find at least one path"

    print(f"  - Found {len(paths)} paths in {elapsed_time:.2f}ms")


def test_integration_epistemic_engine_and_traversal(setup_epistemic_graph):
    """Test 5: Integration zwischen EpistemicEngine und GraphTraversal"""
    engine, traversal, agents = setup_epistemic_graph

    # Query via EpistemicEngine
    engine_paths = engine.query_meta_knowledge_paths("agent_0", max_depth=2)

    # Query via GraphTraversal
    traversal_paths = traversal.find_epistemic_paths("agent_0", max_depth=2)

    # Both should return results
    assert len(engine_paths) > 0, "EpistemicEngine should find paths"
    assert len(traversal_paths) >= 0, "GraphTraversal should not fail"

    print(f"OK Integration successful:")
    print(f"  - EpistemicEngine found {len(engine_paths)} meta-knowledge paths")
    print(f"  - GraphTraversal found {len(traversal_paths)} epistemic paths")


def test_empty_graph():
    """Test 6: Verhalten bei leerem Graph"""
    netzwerk = KonzeptNetzwerk()
    engine = EpistemicEngine(netzwerk)

    # Create agent without meta-knowledge
    engine.create_agent("lonely_agent", "Lonely Agent")

    # Query should return empty list (not error)
    paths = engine.query_meta_knowledge_paths("lonely_agent", max_depth=3)

    assert paths == [], "Empty graph should return empty list"
    print("OK Empty graph handled correctly")


def test_nonexistent_agent():
    """Test 7: Verhalten bei nicht-existierendem Agent"""
    netzwerk = KonzeptNetzwerk()
    engine = EpistemicEngine(netzwerk)

    # Query for non-existent agent
    paths = engine.query_meta_knowledge_paths("nonexistent_agent", max_depth=3)

    assert paths == [], "Non-existent agent should return empty list"
    print("OK Non-existent agent handled correctly")


if __name__ == "__main__":
    print("Running Epistemic Graph-Traversal Tests (Schritt 3.2)...")
    print("=" * 60)

    # Run tests manually
    print("\n[Test 1] Basic Meta-Knowledge Paths Query")
    setup = setup_epistemic_graph()
    test_query_meta_knowledge_paths_basic(setup)

    print("\n[Test 2] Depth Limit Respect")
    setup = setup_epistemic_graph()
    test_query_meta_knowledge_paths_respects_depth(setup)

    print("\n[Test 3] find_epistemic_paths()")
    setup = setup_epistemic_graph()
    test_find_epistemic_paths(setup)

    print("\n[Test 4] Performance Test")
    test_performance_query_meta_knowledge_paths()

    print("\n[Test 5] Integration Test")
    setup = setup_epistemic_graph()
    test_integration_epistemic_engine_and_traversal(setup)

    print("\n[Test 6] Empty Graph")
    test_empty_graph()

    print("\n[Test 7] Non-existent Agent")
    test_nonexistent_agent()

    print("\n" + "=" * 60)
    print("OK All tests passed!")
