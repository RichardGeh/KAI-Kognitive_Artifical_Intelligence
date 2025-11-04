"""
test_graph_traversal.py

Tests für component_12_graph_traversal.py
Testet Multi-Hop Reasoning, Graph-Traversierung und Pfad-Findung.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from component_1_netzwerk import KonzeptNetzwerk
from component_12_graph_traversal import GraphTraversal, TraversalStrategy


@pytest.fixture
def netzwerk():
    """Erstellt KonzeptNetzwerk-Instanz"""
    return KonzeptNetzwerk()


@pytest.fixture
def traversal(netzwerk):
    """Erstellt GraphTraversal-Instanz"""
    return GraphTraversal(netzwerk)


@pytest.fixture
def setup_hierarchy(netzwerk):
    """
    Erstellt Test-Hierarchie:
    test_pudel -> test_hund -> test_säugetier -> test_tier -> test_lebewesen
    """
    # Erstelle Hierarchie
    netzwerk.assert_relation("test_pudel", "IS_A", "test_hund")
    netzwerk.assert_relation("test_hund", "IS_A", "test_säugetier")
    netzwerk.assert_relation("test_säugetier", "IS_A", "test_tier")
    netzwerk.assert_relation("test_tier", "IS_A", "test_lebewesen")

    yield netzwerk

    # Cleanup (done manually - test_ prefix allows manual cleanup if needed)


@pytest.fixture
def setup_complex_graph(netzwerk):
    """
    Erstellt komplexeren Test-Graph mit mehreren Pfaden:

    test_a -> test_b -> test_d
       ↓        ↓
    test_c -> test_e -> test_d
    """
    netzwerk.assert_relation("test_a", "IS_A", "test_b")
    netzwerk.assert_relation("test_a", "IS_A", "test_c")
    netzwerk.assert_relation("test_b", "IS_A", "test_d")
    netzwerk.assert_relation("test_b", "PART_OF", "test_e")
    netzwerk.assert_relation("test_c", "IS_A", "test_e")
    netzwerk.assert_relation("test_e", "IS_A", "test_d")

    yield netzwerk

    # Cleanup (done manually - test_ prefix allows manual cleanup if needed)


class TestTransitiveRelations:
    """Tests für transitive Relationen (IS_A-Hierarchien)"""

    def test_find_single_hop_relation(self, traversal, setup_hierarchy):
        """Test: Einstufige Relation finden"""
        paths = traversal.find_transitive_relations("test_hund", "IS_A", max_depth=1)

        assert len(paths) >= 1
        assert paths[0].nodes == ["test_hund", "test_säugetier"]
        assert paths[0].relations == ["IS_A"]
        # Confidence ist 0.85 (default aus component_1_netzwerk_core.py:276)
        assert abs(paths[0].confidence - 0.85) < 0.01

    def test_find_multi_hop_relation(self, traversal, setup_hierarchy):
        """Test: Mehrstufige Relation finden"""
        paths = traversal.find_transitive_relations("test_hund", "IS_A")

        # Sollte mindestens: hund->säugetier, hund->säugetier->tier, etc. finden
        assert len(paths) >= 3

        # Kürzester Pfad zuerst
        assert len(paths[0].relations) == 1  # hund -> säugetier
        assert len(paths[1].relations) == 2  # hund -> säugetier -> tier

        # Längster Pfad
        longest = paths[-1]
        assert "test_lebewesen" in longest.nodes

    def test_transitive_relation_explanation(self, traversal, setup_hierarchy):
        """Test: Erklärungsgenerierung für transitive Relation"""
        paths = traversal.find_transitive_relations("test_pudel", "IS_A")

        # Finde längsten Pfad
        longest_path = max(paths, key=lambda p: len(p.relations))

        # Erklärung sollte alle Knoten enthalten
        assert "test_pudel" in longest_path.explanation
        assert "test_lebewesen" in longest_path.explanation
        assert (
            "über" in longest_path.explanation or "ist ein" in longest_path.explanation
        )

    def test_max_depth_limit(self, traversal, setup_hierarchy):
        """Test: Maximale Traversierungstiefe wird respektiert"""
        paths = traversal.find_transitive_relations("test_pudel", "IS_A", max_depth=2)

        # Kein Pfad sollte länger als 2 Hops sein
        for path in paths:
            assert len(path.relations) <= 2

    def test_no_transitive_relations(self, traversal, netzwerk):
        """Test: Konzept ohne transitive Relationen"""
        netzwerk.ensure_wort_und_konzept("test_isolated")

        paths = traversal.find_transitive_relations("test_isolated", "IS_A")

        assert len(paths) == 0


class TestPathFinding:
    """Tests für Pfad-Findung zwischen Konzepten"""

    def test_bfs_shortest_path(self, traversal, setup_hierarchy):
        """Test: BFS findet kürzesten Pfad"""
        path = traversal.find_path_between_concepts(
            "test_pudel", "test_lebewesen", strategy=TraversalStrategy.BREADTH_FIRST
        )

        assert path is not None
        assert path.nodes[0] == "test_pudel"
        assert path.nodes[-1] == "test_lebewesen"
        assert len(path.relations) == 4  # pudel->hund->säugetier->tier->lebewesen

    def test_dfs_finds_path(self, traversal, setup_hierarchy):
        """Test: DFS findet einen Pfad"""
        path = traversal.find_path_between_concepts(
            "test_hund", "test_tier", strategy=TraversalStrategy.DEPTH_FIRST
        )

        assert path is not None
        assert path.nodes[0] == "test_hund"
        assert path.nodes[-1] == "test_tier"

    def test_path_with_relation_filter(self, traversal, setup_complex_graph):
        """Test: Pfad-Findung mit Relationsfilter"""
        # Nur IS_A-Relationen erlauben
        path = traversal.find_path_between_concepts(
            "test_a", "test_d", allowed_relations=["IS_A"]
        )

        assert path is not None
        # Sollte nur IS_A-Relationen verwenden
        assert all(rel == "IS_A" for rel in path.relations)

    def test_no_path_exists(self, traversal, netzwerk):
        """Test: Kein Pfad zwischen unverbundenen Konzepten"""
        netzwerk.ensure_wort_und_konzept("test_isolated1")
        netzwerk.ensure_wort_und_konzept("test_isolated2")

        path = traversal.find_path_between_concepts("test_isolated1", "test_isolated2")

        assert path is None

    def test_path_explanation(self, traversal, setup_hierarchy):
        """Test: Pfad-Erklärung ist verständlich"""
        path = traversal.find_path_between_concepts("test_pudel", "test_tier")

        assert path is not None
        assert "test_pudel" in path.explanation
        assert "test_tier" in path.explanation
        assert "->" in path.explanation  # Pfeil-Separator


class TestMultiplePaths:
    """Tests für Findung mehrerer Pfade"""

    def test_find_all_paths(self, traversal, setup_complex_graph):
        """Test: Alle Pfade zwischen Konzepten finden"""
        paths = traversal.find_all_paths_between_concepts(
            "test_a", "test_d", max_paths=10
        )

        # Es sollte mehrere Pfade geben (verschiedene Routen)
        assert len(paths) >= 2

        # Alle Pfade sollten von a nach d führen
        for path in paths:
            assert path.nodes[0] == "test_a"
            assert path.nodes[-1] == "test_d"

    def test_paths_sorted_by_length(self, traversal, setup_complex_graph):
        """Test: Pfade sind nach Länge sortiert"""
        paths = traversal.find_all_paths_between_concepts(
            "test_a", "test_d", max_paths=10
        )

        # Pfade sollten nach Länge aufsteigend sortiert sein
        path_lengths = [len(p.relations) for p in paths]
        assert path_lengths == sorted(path_lengths)

    def test_max_paths_limit(self, traversal, setup_complex_graph):
        """Test: Maximale Anzahl Pfade wird respektiert"""
        paths = traversal.find_all_paths_between_concepts(
            "test_a", "test_d", max_paths=2
        )

        assert len(paths) <= 2


class TestInferenceExplanation:
    """Tests für Inferenz-Ketten-Erklärungen"""

    def test_explain_single_hop_inference(self, traversal, setup_hierarchy):
        """Test: Erklärung für einstufige Inferenz"""
        explanation = traversal.explain_inference_chain(
            "test_hund", "test_säugetier", "IS_A"
        )

        assert explanation is not None
        assert "test_hund" in explanation
        assert "test_säugetier" in explanation
        assert "ist ein" in explanation

    def test_explain_multi_hop_inference(self, traversal, setup_hierarchy):
        """Test: Erklärung für mehrstufige Inferenz"""
        explanation = traversal.explain_inference_chain(
            "test_pudel", "test_tier", "IS_A"
        )

        assert explanation is not None
        assert "test_pudel" in explanation
        assert "test_tier" in explanation
        assert "weil" in explanation.lower()

        # Sollte intermediate Knoten erwähnen
        assert "test_hund" in explanation or "test_säugetier" in explanation

    def test_explain_nonexistent_inference(self, traversal, netzwerk):
        """Test: Keine Erklärung für nicht existierende Inferenz"""
        # Verwende eindeutige Namen, um Test-Isolation sicherzustellen
        import uuid

        unique_x = f"test_nonexistent_{uuid.uuid4().hex[:8]}_x"
        unique_y = f"test_nonexistent_{uuid.uuid4().hex[:8]}_y"

        netzwerk.ensure_wort_und_konzept(unique_x)
        netzwerk.ensure_wort_und_konzept(unique_y)

        explanation = traversal.explain_inference_chain(unique_x, unique_y, "IS_A")

        assert explanation is None


class TestConceptHierarchy:
    """Tests für Konzept-Hierarchien"""

    def test_get_ancestors(self, traversal, setup_hierarchy):
        """Test: Vorfahren eines Konzepts finden"""
        hierarchy = traversal.get_concept_hierarchy("test_pudel", "IS_A")

        ancestors = hierarchy["ancestors"]

        # Pudel sollte alle höheren Ebenen als Vorfahren haben
        assert "test_hund" in ancestors
        assert "test_säugetier" in ancestors
        assert "test_tier" in ancestors
        assert "test_lebewesen" in ancestors

    def test_hierarchy_deduplication(self, traversal, netzwerk):
        """Test: Duplikate in Hierarchie werden entfernt"""
        # Erstelle Graph mit mehreren Pfaden zum gleichen Ziel
        netzwerk.assert_relation("test_x", "IS_A", "test_y")
        netzwerk.assert_relation("test_x", "IS_A", "test_z")
        netzwerk.assert_relation("test_y", "IS_A", "test_w")
        netzwerk.assert_relation("test_z", "IS_A", "test_w")

        hierarchy = traversal.get_concept_hierarchy("test_x", "IS_A")

        ancestors = hierarchy["ancestors"]

        # "test_w" sollte nur einmal vorkommen
        assert ancestors.count("test_w") == 1


class TestInverseTraversal:
    """Tests für inverse Graph-Traversierung (Rückwärts-Reasoning)"""

    def test_find_single_hop_inverse_relation(self, traversal, setup_hierarchy):
        """Test: Einstufige inverse Relation finden"""
        # Finde alle Konzepte, die IS_A "test_säugetier" haben
        paths = traversal.find_inverse_transitive_relations(
            "test_säugetier", "IS_A", max_depth=1
        )

        # Sollte mindestens test_hund finden
        assert len(paths) >= 1
        # Prüfe ob test_hund in den Ergebnissen ist
        found_hund = any("test_hund" in path.nodes for path in paths)
        assert found_hund

        # Prüfe Struktur eines Pfads
        hund_path = [p for p in paths if "test_hund" in p.nodes][0]
        assert hund_path.nodes == ["test_säugetier", "test_hund"]
        assert hund_path.relations == ["IS_A"]
        # Confidence ist 0.85 (default aus component_1_netzwerk_core.py:276)
        assert abs(hund_path.confidence - 0.85) < 0.01

    def test_find_multi_hop_inverse_relation(self, traversal, setup_hierarchy):
        """Test: Mehrstufige inverse Relation finden"""
        # Finde alle Konzepte, die transitiv IS_A "test_tier" haben
        paths = traversal.find_inverse_transitive_relations("test_tier", "IS_A")

        # Sollte mindestens: tier<-säugetier, tier<-säugetier<-hund, tier<-säugetier<-hund<-pudel finden
        assert len(paths) >= 3

        # Kürzester Pfad zuerst
        assert len(paths[0].relations) == 1  # tier <- säugetier

        # Prüfe ob längerer Pfad existiert, der pudel enthält
        pudel_paths = [p for p in paths if "test_pudel" in p.nodes]
        assert len(pudel_paths) > 0

        # Längster Pfad mit pudel sollte: tier <- säugetier <- hund <- pudel sein
        longest_pudel = max(pudel_paths, key=lambda p: len(p.relations))
        assert len(longest_pudel.relations) == 3

    def test_inverse_relation_explanation(self, traversal, setup_hierarchy):
        """Test: Erklärungsgenerierung für inverse Relation"""
        paths = traversal.find_inverse_transitive_relations("test_tier", "IS_A")

        # Finde Pfad mit test_pudel
        pudel_paths = [p for p in paths if "test_pudel" in p.nodes]
        assert len(pudel_paths) > 0

        pudel_path = pudel_paths[0]
        # Erklärung sollte beide Endpunkte enthalten
        assert (
            "test_tier" in pudel_path.explanation
            or "test_pudel" in pudel_path.explanation
        )

    def test_get_descendants(self, traversal, setup_hierarchy):
        """Test: Nachfahren (descendants) eines Konzepts finden"""
        hierarchy = traversal.get_concept_hierarchy("test_tier", "IS_A")

        descendants = hierarchy["descendants"]

        # Tier sollte säugetier, hund und pudel als Nachfahren haben
        assert "test_säugetier" in descendants
        assert "test_hund" in descendants
        assert "test_pudel" in descendants

    def test_get_hierarchy_complete(self, traversal, setup_hierarchy):
        """Test: Vollständige Hierarchie mit Vorfahren UND Nachfahren"""
        # Für säugetier: Vorfahren nach oben, Nachfahren nach unten
        hierarchy = traversal.get_concept_hierarchy("test_säugetier", "IS_A")

        ancestors = hierarchy["ancestors"]
        descendants = hierarchy["descendants"]

        # Vorfahren (aufwärts)
        assert "test_tier" in ancestors
        assert "test_lebewesen" in ancestors

        # Nachfahren (abwärts)
        assert "test_hund" in descendants
        assert "test_pudel" in descendants

        # säugetier selbst sollte nicht in Listen sein
        assert "test_säugetier" not in ancestors
        assert "test_säugetier" not in descendants

    def test_inverse_with_no_descendants(self, traversal, setup_hierarchy):
        """Test: Konzept ohne Nachfahren (Blattknoten)"""
        # test_pudel ist ein Blattknoten (keine Konzepte haben IS_A test_pudel)
        hierarchy = traversal.get_concept_hierarchy("test_pudel", "IS_A")

        descendants = hierarchy["descendants"]

        # Sollte keine Nachfahren haben
        assert len(descendants) == 0

        # Sollte aber Vorfahren haben
        ancestors = hierarchy["ancestors"]
        assert len(ancestors) > 0

    def test_inverse_traversal_max_depth(self, traversal, setup_hierarchy):
        """Test: Maximale Tiefe für inverse Traversierung"""
        paths = traversal.find_inverse_transitive_relations(
            "test_tier", "IS_A", max_depth=2
        )

        # Kein Pfad sollte länger als 2 Hops sein
        for path in paths:
            assert len(path.relations) <= 2

    def test_query_inverse_relations_direct(self, netzwerk, setup_hierarchy):
        """Test: Direkte Abfrage eingehender Relationen"""
        # Teste die neue query_inverse_relations Methode direkt
        inverse_facts = netzwerk.query_inverse_relations("test_säugetier", "IS_A")

        # Sollte test_hund enthalten (test_hund IS_A test_säugetier)
        assert "IS_A" in inverse_facts
        assert "test_hund" in inverse_facts["IS_A"]

    def test_query_inverse_relations_all_types(self, netzwerk):
        """Test: Eingehende Relationen aller Typen finden"""
        # Erstelle verschiedene eingehende Relationen
        netzwerk.assert_relation("test_apfel", "IS_A", "test_frucht")
        netzwerk.assert_relation("test_banane", "IS_A", "test_frucht")
        netzwerk.assert_relation(
            "test_apfel", "HAS_PROPERTY", "test_frucht"
        )  # Unterschiedlicher Typ

        # Finde alle eingehenden Relationen für test_frucht (ohne Filter)
        inverse_facts = netzwerk.query_inverse_relations("test_frucht")

        # Sollte beide Relationstypen finden
        assert "IS_A" in inverse_facts
        assert "test_apfel" in inverse_facts["IS_A"]
        assert "test_banane" in inverse_facts["IS_A"]
        assert "HAS_PROPERTY" in inverse_facts


class TestEdgeCases:
    """Tests für Grenzfälle und Robustheit"""

    def test_cycle_detection(self, traversal, netzwerk):
        """Test: Zyklen im Graph werden korrekt behandelt"""
        # Erstelle Zyklus: a -> b -> c -> a
        netzwerk.assert_relation("test_cycle_a", "IS_A", "test_cycle_b")
        netzwerk.assert_relation("test_cycle_b", "IS_A", "test_cycle_c")
        netzwerk.assert_relation("test_cycle_c", "IS_A", "test_cycle_a")

        # Sollte nicht in Endlosschleife laufen
        paths = traversal.find_transitive_relations(
            "test_cycle_a", "IS_A", max_depth=10
        )

        # Sollte begrenzte Anzahl Pfade zurückgeben
        assert len(paths) > 0
        assert len(paths) < 100  # Keine Explosion

    def test_self_loop(self, traversal, netzwerk):
        """Test: Self-Loop (Konzept zeigt auf sich selbst)"""
        netzwerk.assert_relation("test_self", "IS_A", "test_self")

        paths = traversal.find_transitive_relations("test_self", "IS_A")

        # Sollte Self-Loop ignorieren oder begrenzen
        assert len(paths) < 10

    def test_empty_graph(self, traversal, netzwerk):
        """Test: Leerer Graph"""
        netzwerk.ensure_wort_und_konzept("test_empty")

        paths = traversal.find_transitive_relations("test_empty", "IS_A")

        assert len(paths) == 0

    def test_very_deep_hierarchy(self, traversal, netzwerk):
        """Test: Sehr tiefe Hierarchie (Performance)"""
        # Erstelle Kette mit 20 Knoten
        for i in range(20):
            netzwerk.assert_relation(f"test_deep_{i}", "IS_A", f"test_deep_{i+1}")

        paths = traversal.find_transitive_relations("test_deep_0", "IS_A")

        # Sollte begrenzt werden durch max_depth
        max_path_length = max(len(p.relations) for p in paths) if paths else 0
        assert max_path_length <= traversal.max_depth


class TestGermanExplanations:
    """Tests für deutsche Erklärungsgenerierung"""

    def test_relation_translation(self, traversal):
        """Test: Relationstypen werden korrekt übersetzt"""
        assert "ist ein" in traversal._relation_to_german("IS_A")
        assert (
            "hat die eigenschaft"
            in traversal._relation_to_german("HAS_PROPERTY").lower()
        )
        assert "kann" in traversal._relation_to_german("CAPABLE_OF")

    def test_explanation_is_german(self, traversal, setup_hierarchy):
        """Test: Generierte Erklärungen sind auf Deutsch"""
        path = traversal.find_path_between_concepts("test_hund", "test_tier")

        assert path is not None

        # Prüfe auf deutsche Begriffe
        explanation_lower = path.explanation.lower()
        # Sollte keine englischen Begriffe enthalten
        assert "is_a" not in explanation_lower
        assert "has_property" not in explanation_lower


class TestConfidencePropagation:
    """Tests für Confidence-Berechnung und -Propagation"""

    @pytest.fixture
    def setup_confidence_hierarchy(self, netzwerk):
        """
        Erstellt Test-Hierarchie mit verschiedenen Confidence-Werten:
        test_conf_a --0.9--> test_conf_b --0.8--> test_conf_c --0.7--> test_conf_d
        """
        # Erstelle Relations mit verschiedenen Confidence-Werten
        # Wir müssen direkt in Neo4j schreiben, um custom Confidence zu setzen
        with netzwerk.driver.session(database="neo4j") as session:
            # Erstelle Konzepte
            for concept in ["test_conf_a", "test_conf_b", "test_conf_c", "test_conf_d"]:
                netzwerk.ensure_wort_und_konzept(concept)

            # Erstelle Relationen mit custom Confidence
            session.run(
                """
                MATCH (a:Konzept {name: $source}), (b:Konzept {name: $target})
                MERGE (a)-[r:IS_A]->(b)
                SET r.confidence = $conf, r.asserted_at = timestamp()
            """,
                source="test_conf_a",
                target="test_conf_b",
                conf=0.9,
            )

            session.run(
                """
                MATCH (a:Konzept {name: $source}), (b:Konzept {name: $target})
                MERGE (a)-[r:IS_A]->(b)
                SET r.confidence = $conf, r.asserted_at = timestamp()
            """,
                source="test_conf_b",
                target="test_conf_c",
                conf=0.8,
            )

            session.run(
                """
                MATCH (a:Konzept {name: $source}), (b:Konzept {name: $target})
                MERGE (a)-[r:IS_A]->(b)
                SET r.confidence = $conf, r.asserted_at = timestamp()
            """,
                source="test_conf_c",
                target="test_conf_d",
                conf=0.7,
            )

        yield netzwerk

    def test_query_with_confidence_single_hop(
        self, netzwerk, setup_confidence_hierarchy
    ):
        """Test: Einzelne Relation mit Confidence abrufen"""
        facts_with_conf = netzwerk.query_graph_for_facts_with_confidence("test_conf_a")

        assert "IS_A" in facts_with_conf
        targets = facts_with_conf["IS_A"]

        # Sollte test_conf_b mit Confidence 0.9 enthalten
        assert len(targets) == 1
        assert targets[0]["target"] == "test_conf_b"
        assert abs(targets[0]["confidence"] - 0.9) < 0.01

    def test_query_inverse_with_confidence(self, netzwerk, setup_confidence_hierarchy):
        """Test: Inverse Relationen mit Confidence abrufen"""
        inverse_facts = netzwerk.query_inverse_relations_with_confidence(
            "test_conf_c", "IS_A"
        )

        assert "IS_A" in inverse_facts
        sources = inverse_facts["IS_A"]

        # Sollte test_conf_b mit Confidence 0.8 enthalten
        assert len(sources) >= 1
        conf_b = [s for s in sources if s["source"] == "test_conf_b"]
        assert len(conf_b) == 1
        assert abs(conf_b[0]["confidence"] - 0.8) < 0.01

    def test_confidence_propagation_min_strategy(
        self, traversal, setup_confidence_hierarchy
    ):
        """Test: Confidence-Propagation mit min-Strategie"""
        # Finde Pfad: test_conf_a -> test_conf_b (0.9) -> test_conf_c (0.8) -> test_conf_d (0.7)
        paths = traversal.find_transitive_relations("test_conf_a", "IS_A")

        # Finde den längsten Pfad (sollte alle 3 Hops haben)
        longest_path = max(paths, key=lambda p: len(p.relations))

        assert longest_path.nodes == [
            "test_conf_a",
            "test_conf_b",
            "test_conf_c",
            "test_conf_d",
        ]

        # Gesamt-Confidence sollte min(0.9, 0.8, 0.7) = 0.7 sein
        assert abs(longest_path.confidence - 0.7) < 0.01

    def test_confidence_propagation_two_hops(
        self, traversal, setup_confidence_hierarchy
    ):
        """Test: Confidence bei 2 Hops"""
        paths = traversal.find_transitive_relations("test_conf_a", "IS_A", max_depth=2)

        # Finde Pfad mit 2 Hops: test_conf_a -> test_conf_b -> test_conf_c
        two_hop_path = [
            p for p in paths if len(p.relations) == 2 and "test_conf_c" in p.nodes
        ]
        assert len(two_hop_path) > 0

        path = two_hop_path[0]
        assert path.nodes == ["test_conf_a", "test_conf_b", "test_conf_c"]

        # Gesamt-Confidence sollte min(0.9, 0.8) = 0.8 sein
        assert abs(path.confidence - 0.8) < 0.01

    def test_confidence_in_path_finding(self, traversal, setup_confidence_hierarchy):
        """Test: Confidence bei find_path_between_concepts"""
        path = traversal.find_path_between_concepts(
            "test_conf_a", "test_conf_d", allowed_relations=["IS_A"]
        )

        assert path is not None
        assert path.nodes[0] == "test_conf_a"
        assert path.nodes[-1] == "test_conf_d"

        # Confidence sollte das Minimum aller Kanten sein
        assert abs(path.confidence - 0.7) < 0.01

    def test_default_confidence_for_old_relations(self, traversal, netzwerk):
        """Test: Standard-Confidence (1.0) für Relationen ohne explizite Confidence"""
        # Erstelle Relation mit assert_relation (setzt confidence=0.85 per default)
        netzwerk.assert_relation("test_default_a", "IS_A", "test_default_b")

        facts_with_conf = netzwerk.query_graph_for_facts_with_confidence(
            "test_default_a"
        )

        assert "IS_A" in facts_with_conf
        targets = facts_with_conf["IS_A"]

        # Standard-Confidence sollte 0.85 sein (siehe component_1_netzwerk_core.py:276)
        assert len(targets) == 1
        assert abs(targets[0]["confidence"] - 0.85) < 0.01

    def test_multiple_paths_different_confidence(self, traversal, netzwerk):
        """Test: Mehrere Pfade mit unterschiedlichen Confidences"""
        # Erstelle zwei Pfade zum gleichen Ziel mit verschiedenen Confidences
        with netzwerk.driver.session(database="neo4j") as session:
            for concept in [
                "test_multi_a",
                "test_multi_b1",
                "test_multi_b2",
                "test_multi_c",
            ]:
                netzwerk.ensure_wort_und_konzept(concept)

            # Pfad 1: a -> b1 (0.9) -> c (0.8) = min=0.8
            session.run(
                """
                MATCH (a:Konzept {name: $source}), (b:Konzept {name: $target})
                MERGE (a)-[r:IS_A]->(b)
                SET r.confidence = $conf
            """,
                source="test_multi_a",
                target="test_multi_b1",
                conf=0.9,
            )

            session.run(
                """
                MATCH (a:Konzept {name: $source}), (b:Konzept {name: $target})
                MERGE (a)-[r:IS_A]->(b)
                SET r.confidence = $conf
            """,
                source="test_multi_b1",
                target="test_multi_c",
                conf=0.8,
            )

            # Pfad 2: a -> b2 (0.95) -> c (0.95) = min=0.95
            session.run(
                """
                MATCH (a:Konzept {name: $source}), (b:Konzept {name: $target})
                MERGE (a)-[r:IS_A]->(b)
                SET r.confidence = $conf
            """,
                source="test_multi_a",
                target="test_multi_b2",
                conf=0.95,
            )

            session.run(
                """
                MATCH (a:Konzept {name: $source}), (b:Konzept {name: $target})
                MERGE (a)-[r:IS_A]->(b)
                SET r.confidence = $conf
            """,
                source="test_multi_b2",
                target="test_multi_c",
                conf=0.95,
            )

        # Finde alle Pfade
        all_paths = traversal.find_all_paths_between_concepts(
            "test_multi_a", "test_multi_c", allowed_relations=["IS_A"]
        )

        assert len(all_paths) == 2

        # Einer sollte Confidence 0.8, der andere 0.95 haben
        confidences = sorted([p.confidence for p in all_paths])
        assert abs(confidences[0] - 0.8) < 0.01
        assert abs(confidences[1] - 0.95) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
