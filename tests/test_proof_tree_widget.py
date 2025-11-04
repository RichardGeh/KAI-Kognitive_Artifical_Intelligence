"""
test_proof_tree_widget.py

Test Suite für component_18_proof_tree_widget.py
Tests für Interactive Proof Tree Visualization Widget
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import patch
import tempfile
import os

# PyQt6 imports
try:
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtGui import QColor

    PYQT6_AVAILABLE = True
except ImportError:
    PYQT6_AVAILABLE = False
    pytest.skip("PyQt6 not available", allow_module_level=True)

from component_18_proof_tree_widget import ProofNodeItem, ProofEdgeItem, ProofTreeWidget

from component_17_proof_explanation import ProofStep, ProofTreeNode, ProofTree, StepType


# Qt Application fixture (required for Qt tests)
@pytest.fixture(scope="session")
def qapp():
    """Create QApplication for tests"""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def proof_step_fact():
    """Create a sample FACT_MATCH ProofStep"""
    return ProofStep(
        step_id="fact_1",
        step_type=StepType.FACT_MATCH,
        inputs=["apfel"],
        output="apfel ist eine frucht",
        confidence=0.95,
        explanation_text="Fand Fakt in Wissensbasis",
    )


@pytest.fixture
def proof_step_rule():
    """Create a sample RULE_APPLICATION ProofStep"""
    return ProofStep(
        step_id="rule_1",
        step_type=StepType.RULE_APPLICATION,
        rule_name="transitivity",
        inputs=["premise1", "premise2"],
        output="hund ist tier",
        confidence=0.85,
        explanation_text="Wendete Transitivitätsregel an",
    )


@pytest.fixture
def proof_step_hypothesis():
    """Create a sample HYPOTHESIS ProofStep"""
    return ProofStep(
        step_id="hyp_1",
        step_type=StepType.HYPOTHESIS,
        output="Es hat geregnet",
        confidence=0.75,
        explanation_text="Generierte Hypothese",
        metadata={"strategy": "template-based", "score": 0.75},
    )


@pytest.fixture
def simple_proof_tree():
    """Create a simple proof tree with 3 steps"""
    tree = ProofTree(query="Was ist ein Apfel?")

    root = ProofStep(
        step_id="root",
        step_type=StepType.INFERENCE,
        output="apfel ist frucht",
        confidence=0.9,
        explanation_text="Inferenz",
    )

    child1 = ProofStep(
        step_id="child1",
        step_type=StepType.FACT_MATCH,
        output="apfel existiert",
        confidence=1.0,
        explanation_text="Fakt 1",
    )

    child2 = ProofStep(
        step_id="child2",
        step_type=StepType.FACT_MATCH,
        output="frucht ist kategorie",
        confidence=1.0,
        explanation_text="Fakt 2",
    )

    root.add_subgoal(child1)
    root.add_subgoal(child2)
    tree.add_root_step(root)

    return tree


@pytest.fixture
def complex_proof_tree():
    """Create a complex proof tree with multiple levels"""
    tree = ProofTree(query="Ist ein Hund ein Tier?")

    # Level 0: Root inference
    root = ProofStep(
        step_id="root",
        step_type=StepType.RULE_APPLICATION,
        rule_name="transitivity",
        output="hund ist tier",
        confidence=0.85,
        explanation_text="Transitive Inferenz",
    )

    # Level 1: Graph traversal steps
    hop1 = ProofStep(
        step_id="hop1",
        step_type=StepType.GRAPH_TRAVERSAL,
        output="hund ist säugetier",
        confidence=1.0,
        explanation_text="Hop 1",
    )

    hop2 = ProofStep(
        step_id="hop2",
        step_type=StepType.GRAPH_TRAVERSAL,
        output="säugetier ist tier",
        confidence=1.0,
        explanation_text="Hop 2",
    )

    # Level 2: Fact matches
    fact1 = ProofStep(
        step_id="fact1",
        step_type=StepType.FACT_MATCH,
        output="hund in database",
        confidence=1.0,
        explanation_text="Fakt gefunden",
    )

    fact2 = ProofStep(
        step_id="fact2",
        step_type=StepType.FACT_MATCH,
        output="säugetier in database",
        confidence=1.0,
        explanation_text="Fakt gefunden",
    )

    # Build tree
    hop1.add_subgoal(fact1)
    hop2.add_subgoal(fact2)
    root.add_subgoal(hop1)
    root.add_subgoal(hop2)
    tree.add_root_step(root)

    return tree


class TestProofNodeItem:
    """Tests für ProofNodeItem (Custom Graphics Item)"""

    def test_node_item_creation(self, qapp, proof_step_fact):
        """Test: ProofNodeItem kann erstellt werden"""
        tree_node = ProofTreeNode(step=proof_step_fact)
        node_item = ProofNodeItem(tree_node)

        assert node_item.tree_node == tree_node
        assert node_item.node_width == 150
        assert node_item.node_height == 60
        assert node_item.is_highlighted is False
        assert node_item.is_selected_item is False

    def test_node_item_has_tooltip(self, qapp, proof_step_fact):
        """Test: ProofNodeItem hat Tooltip"""
        tree_node = ProofTreeNode(step=proof_step_fact)
        node_item = ProofNodeItem(tree_node)

        tooltip = node_item.toolTip()
        assert "Schritt:" in tooltip
        assert "fact_match" in tooltip
        assert "Konfidenz:" in tooltip
        assert "0.95" in tooltip

    def test_node_item_bounding_rect(self, qapp, proof_step_fact):
        """Test: Bounding Rectangle hat korrekte Größe"""
        tree_node = ProofTreeNode(step=proof_step_fact)
        node_item = ProofNodeItem(tree_node)

        rect = node_item.boundingRect()
        assert rect.width() == 150
        assert rect.height() == 60
        # Centered at (0, 0)
        assert rect.center().x() == pytest.approx(0, abs=0.1)
        assert rect.center().y() == pytest.approx(0, abs=0.1)

    def test_node_shape_type_rectangle(self, qapp, proof_step_fact):
        """Test: FACT_MATCH und INFERENCE haben Rectangle-Shape"""
        # FACT_MATCH
        tree_node_fact = ProofTreeNode(step=proof_step_fact)
        node_item_fact = ProofNodeItem(tree_node_fact)
        assert node_item_fact._get_shape_type(StepType.FACT_MATCH) == "rectangle"

        # INFERENCE
        inference_step = ProofStep(
            step_id="inf", step_type=StepType.INFERENCE, output="test"
        )
        tree_node_inf = ProofTreeNode(step=inference_step)
        node_item_inf = ProofNodeItem(tree_node_inf)
        assert node_item_inf._get_shape_type(StepType.INFERENCE) == "rectangle"

    def test_node_shape_type_diamond(self, qapp, proof_step_rule):
        """Test: RULE_APPLICATION hat Diamond-Shape"""
        tree_node = ProofTreeNode(step=proof_step_rule)
        node_item = ProofNodeItem(tree_node)
        assert node_item._get_shape_type(StepType.RULE_APPLICATION) == "diamond"
        assert node_item._get_shape_type(StepType.DECOMPOSITION) == "diamond"

    def test_node_shape_type_circle(self, qapp, proof_step_hypothesis):
        """Test: HYPOTHESIS hat Circle-Shape"""
        tree_node = ProofTreeNode(step=proof_step_hypothesis)
        node_item = ProofNodeItem(tree_node)
        assert node_item._get_shape_type(StepType.HYPOTHESIS) == "circle"
        assert node_item._get_shape_type(StepType.PROBABILISTIC) == "circle"
        assert node_item._get_shape_type(StepType.GRAPH_TRAVERSAL) == "circle"

    def test_confidence_color_high(self, qapp):
        """Test: Hohe Konfidenz hat grüne Farbe"""
        step = ProofStep(
            step_id="high",
            step_type=StepType.FACT_MATCH,
            output="test",
            confidence=0.95,
        )
        tree_node = ProofTreeNode(step=step)
        node_item = ProofNodeItem(tree_node)

        color = node_item._get_confidence_color(0.95)
        assert color == QColor("#27ae60")  # Green

    def test_confidence_color_medium(self, qapp):
        """Test: Mittlere Konfidenz hat gelb/orange Farbe"""
        step = ProofStep(
            step_id="med", step_type=StepType.FACT_MATCH, output="test", confidence=0.65
        )
        tree_node = ProofTreeNode(step=step)
        node_item = ProofNodeItem(tree_node)

        color = node_item._get_confidence_color(0.65)
        assert color == QColor("#f39c12")  # Yellow/Orange

    def test_confidence_color_low(self, qapp):
        """Test: Niedrige Konfidenz hat rote Farbe"""
        step = ProofStep(
            step_id="low", step_type=StepType.FACT_MATCH, output="test", confidence=0.3
        )
        tree_node = ProofTreeNode(step=step)
        node_item = ProofNodeItem(tree_node)

        color = node_item._get_confidence_color(0.3)
        assert color == QColor("#e74c3c")  # Red

    def test_highlight_state(self, qapp, proof_step_fact):
        """Test: Highlight-State kann gesetzt werden"""
        tree_node = ProofTreeNode(step=proof_step_fact)
        node_item = ProofNodeItem(tree_node)

        assert node_item.is_highlighted is False

        node_item.set_highlighted(True)
        assert node_item.is_highlighted is True

        node_item.set_highlighted(False)
        assert node_item.is_highlighted is False

    def test_selected_state(self, qapp, proof_step_fact):
        """Test: Selected-State kann gesetzt werden"""
        tree_node = ProofTreeNode(step=proof_step_fact)
        node_item = ProofNodeItem(tree_node)

        assert node_item.is_selected_item is False

        node_item.set_selected_state(True)
        assert node_item.is_selected_item is True

        node_item.set_selected_state(False)
        assert node_item.is_selected_item is False


class TestProofEdgeItem:
    """Tests für ProofEdgeItem (Edge Graphics Item)"""

    def test_edge_item_creation(self, qapp, proof_step_fact, proof_step_rule):
        """Test: ProofEdgeItem kann erstellt werden"""
        parent_node = ProofTreeNode(step=proof_step_rule)
        child_node = ProofTreeNode(step=proof_step_fact)

        parent_item = ProofNodeItem(parent_node)
        child_item = ProofNodeItem(child_node)

        parent_item.setPos(0, 0)
        child_item.setPos(0, 100)

        edge_item = ProofEdgeItem(parent_item, child_item)

        assert edge_item.parent_item == parent_item
        assert edge_item.child_item == child_item
        assert edge_item.is_highlighted is False

    def test_edge_highlight(self, qapp, proof_step_fact, proof_step_rule):
        """Test: Edge kann highlighted werden"""
        parent_node = ProofTreeNode(step=proof_step_rule)
        child_node = ProofTreeNode(step=proof_step_fact)

        parent_item = ProofNodeItem(parent_node)
        child_item = ProofNodeItem(child_node)

        edge_item = ProofEdgeItem(parent_item, child_item)

        assert edge_item.is_highlighted is False

        edge_item.set_highlighted(True)
        assert edge_item.is_highlighted is True
        # Verify pen color changed
        assert edge_item.pen().color() == QColor("#f39c12")

        edge_item.set_highlighted(False)
        assert edge_item.is_highlighted is False
        assert edge_item.pen().color() == QColor("#7f8c8d")


class TestProofTreeWidget:
    """Tests für ProofTreeWidget (Main Widget)"""

    def test_widget_creation(self, qapp):
        """Test: ProofTreeWidget kann erstellt werden"""
        widget = ProofTreeWidget()

        assert widget.current_tree is None
        assert len(widget.tree_nodes) == 0
        assert len(widget.node_items) == 0
        assert len(widget.edge_items) == 0
        assert widget.selected_node is None

    def test_widget_ui_components(self, qapp):
        """Test: UI-Komponenten existieren"""
        widget = ProofTreeWidget()

        assert widget.scene is not None
        assert widget.view is not None
        assert widget.status_label is not None
        # Status label should show "Kein Beweisbaum geladen"
        assert "Kein Beweisbaum geladen" in widget.status_label.text()

    def test_set_proof_tree_simple(self, qapp, simple_proof_tree):
        """Test: Einfacher Proof Tree kann gesetzt werden"""
        widget = ProofTreeWidget()
        widget.set_proof_tree(simple_proof_tree)

        assert widget.current_tree == simple_proof_tree
        assert len(widget.tree_nodes) == 1
        # Should have created graphics items
        assert len(widget.node_items) > 0
        # Status should be updated
        assert "3 Schritte" in widget.status_label.text()

    def test_set_proof_tree_complex(self, qapp, complex_proof_tree):
        """Test: Komplexer Proof Tree kann gesetzt werden"""
        widget = ProofTreeWidget()
        widget.set_proof_tree(complex_proof_tree)

        assert widget.current_tree == complex_proof_tree
        assert len(widget.tree_nodes) == 1

        # Should have 5 total steps
        all_steps = complex_proof_tree.get_all_steps()
        assert len(all_steps) == 5

        # All steps should have node items
        assert len(widget.node_items) == 5

    def test_flatten_tree(self, qapp, complex_proof_tree):
        """Test: Baum wird korrekt abgeflacht"""
        widget = ProofTreeWidget()
        widget.set_proof_tree(complex_proof_tree)

        root_node = widget.tree_nodes[0]
        flattened = widget._flatten_tree(root_node)

        # Should include all 5 nodes
        assert len(flattened) == 5

    def test_layout_tree_positions(self, qapp, simple_proof_tree):
        """Test: Tree-Layout berechnet Positionen"""
        widget = ProofTreeWidget()
        widget.set_proof_tree(simple_proof_tree)

        root_node = widget.tree_nodes[0]

        # Root should have position
        assert root_node.position != (0, 0)

        # Children should have positions
        for child in root_node.children:
            assert child.position != (0, 0)
            # Children should be below parent (higher y value)
            assert child.position[1] > root_node.position[1]

    def test_expand_all(self, qapp, complex_proof_tree):
        """Test: Expand All funktioniert"""
        widget = ProofTreeWidget()
        widget.set_proof_tree(complex_proof_tree)

        # Collapse all first
        for root in widget.tree_nodes:
            root.collapse()

        # Now expand all
        widget.expand_all()

        # All nodes should be expanded
        for root in widget.tree_nodes:
            assert root.expanded is True
            for child in root.children:
                assert child.expanded is True

    def test_collapse_all(self, qapp, complex_proof_tree):
        """Test: Collapse All funktioniert"""
        widget = ProofTreeWidget()
        widget.set_proof_tree(complex_proof_tree)

        # All nodes should be expanded initially
        widget.collapse_all()

        # All nodes should be collapsed
        for root in widget.tree_nodes:
            assert root.expanded is False
            for child in root.children:
                assert child.expanded is False

    def test_clear_widget(self, qapp, simple_proof_tree):
        """Test: Widget kann geleert werden"""
        widget = ProofTreeWidget()
        widget.set_proof_tree(simple_proof_tree)

        assert widget.current_tree is not None
        assert len(widget.node_items) > 0

        widget.clear()

        assert widget.current_tree is None
        assert len(widget.tree_nodes) == 0
        assert len(widget.node_items) == 0
        assert len(widget.edge_items) == 0
        assert "Kein Beweisbaum geladen" in widget.status_label.text()

    def test_export_to_json(self, qapp, simple_proof_tree):
        """Test: Export zu JSON"""
        widget = ProofTreeWidget()
        widget.set_proof_tree(simple_proof_tree)

        # Mock QFileDialog to return temp file path
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        temp_path = temp_file.name
        temp_file.close()

        try:
            with patch(
                "component_18_proof_tree_widget.QFileDialog.getSaveFileName"
            ) as mock_dialog:
                mock_dialog.return_value = (temp_path, "JSON Files (*.json)")

                widget.export_to_json()

                # Verify file was created
                assert os.path.exists(temp_path)

                # Verify status was updated
                assert temp_path in widget.status_label.text()

        finally:
            # Cleanup
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_export_to_image(self, qapp, simple_proof_tree):
        """Test: Export zu Bild"""
        widget = ProofTreeWidget()
        widget.set_proof_tree(simple_proof_tree)

        # Mock QFileDialog to return temp file path
        temp_file = tempfile.NamedTemporaryFile(mode="wb", suffix=".png", delete=False)
        temp_path = temp_file.name
        temp_file.close()

        try:
            with patch(
                "component_18_proof_tree_widget.QFileDialog.getSaveFileName"
            ) as mock_dialog:
                mock_dialog.return_value = (temp_path, "PNG Files (*.png)")

                widget.export_to_image()

                # Verify file was created
                assert os.path.exists(temp_path)

                # Verify status was updated
                assert temp_path in widget.status_label.text()

        finally:
            # Cleanup
            if os.path.exists(temp_path):
                os.remove(temp_path)


class TestTreeLayoutAlgorithm:
    """Tests für hierarchischen Tree-Layout-Algorithmus"""

    def test_layout_single_node(self, qapp):
        """Test: Einzelner Knoten hat korrekte Position"""
        widget = ProofTreeWidget()

        tree = ProofTree(query="Single Node")
        step = ProofStep(
            step_id="single", step_type=StepType.FACT_MATCH, output="single node"
        )
        tree.add_root_step(step)

        widget.set_proof_tree(tree)

        root_node = widget.tree_nodes[0]
        assert root_node.position[1] == 0  # Depth 0

    def test_layout_two_levels(self, qapp, simple_proof_tree):
        """Test: Zwei Ebenen haben korrekte relative Positionen"""
        widget = ProofTreeWidget()
        widget.set_proof_tree(simple_proof_tree)

        root_node = widget.tree_nodes[0]
        child1 = root_node.children[0]
        child2 = root_node.children[1]

        # Root at depth 0
        assert root_node.position[1] == 0

        # Children at depth 1 (below root)
        assert child1.position[1] > root_node.position[1]
        assert child2.position[1] > root_node.position[1]

        # Both children at same depth
        assert child1.position[1] == child2.position[1]

    def test_layout_parent_centered_between_children(self, qapp, simple_proof_tree):
        """Test: Parent ist horizontal zwischen Kindern zentriert"""
        widget = ProofTreeWidget()
        widget.set_proof_tree(simple_proof_tree)

        root_node = widget.tree_nodes[0]
        child1 = root_node.children[0]
        child2 = root_node.children[1]

        # Parent X sollte zwischen oder nahe bei Kindern sein
        parent_x = root_node.position[0]
        child1_x = child1.position[0]
        child2_x = child2.position[0]

        min_child_x = min(child1_x, child2_x)
        max_child_x = max(child1_x, child2_x)

        # Parent sollte im Bereich der Kinder sein (mit Toleranz)
        assert (
            min_child_x <= parent_x <= max_child_x
            or abs(parent_x - (min_child_x + max_child_x) / 2) < 200
        )

    def test_layout_three_levels(self, qapp, complex_proof_tree):
        """Test: Drei Ebenen haben korrekte Tiefenverteilung"""
        widget = ProofTreeWidget()
        widget.set_proof_tree(complex_proof_tree)

        root_node = widget.tree_nodes[0]

        # Depth 0: root
        assert root_node.position[1] == 0

        # Depth 1: hop1, hop2
        for child in root_node.children:
            assert child.position[1] > root_node.position[1]

            # Depth 2: facts
            for grandchild in child.children:
                assert grandchild.position[1] > child.position[1]

    def test_layout_collapsed_nodes_dont_show_children(self, qapp, simple_proof_tree):
        """Test: Collapsed Nodes zeigen keine Kinder im Layout"""
        widget = ProofTreeWidget()
        widget.set_proof_tree(simple_proof_tree)

        root_node = widget.tree_nodes[0]
        root_node.collapse()

        # Re-render with collapsed root
        widget._render_tree()

        # Should only have root node item
        # (children should not be rendered when parent is collapsed)
        assert "root" in widget.node_items

        # Flattening should only return root
        flattened = widget._flatten_tree(root_node)
        assert len(flattened) == 1


class TestPerformance:
    """Performance-Tests mit großen Proof Trees"""

    def test_large_tree_rendering(self, qapp):
        """Test: Großer Baum (50 Knoten) kann gerendert werden"""
        tree = ProofTree(query="Large Tree Test")

        # Create root
        root = ProofStep(
            step_id="root", step_type=StepType.RULE_APPLICATION, output="root"
        )

        # Create 49 children (depth 1)
        for i in range(49):
            child = ProofStep(
                step_id=f"child_{i}", step_type=StepType.FACT_MATCH, output=f"child {i}"
            )
            root.add_subgoal(child)

        tree.add_root_step(root)

        # Widget should handle this without crashing
        widget = ProofTreeWidget()
        widget.set_proof_tree(tree)

        assert len(widget.node_items) == 50
        assert "50 Schritte" in widget.status_label.text()

    def test_deep_tree_rendering(self, qapp):
        """Test: Tiefer Baum (10 Ebenen) kann gerendert werden"""
        tree = ProofTree(query="Deep Tree Test")

        # Create deep chain
        current = ProofStep(
            step_id="level_0", step_type=StepType.RULE_APPLICATION, output="level 0"
        )
        tree.add_root_step(current)

        for i in range(1, 10):
            child = ProofStep(
                step_id=f"level_{i}", step_type=StepType.FACT_MATCH, output=f"level {i}"
            )
            current.add_subgoal(child)
            current = child

        # Widget should handle this
        widget = ProofTreeWidget()
        widget.set_proof_tree(tree)

        assert len(widget.node_items) == 10

        # Verify depth calculation
        root_node = widget.tree_nodes[0]
        deepest_node = root_node
        for _ in range(9):
            deepest_node = deepest_node.children[0]

        assert deepest_node.get_depth() == 9


class TestIntegration:
    """Integration-Tests mit vollständigen Reasoning-Szenarien"""

    def test_multi_hop_reasoning_visualization(self, qapp):
        """Test: Multi-Hop Reasoning wird korrekt visualisiert"""
        tree = ProofTree(query="Ist ein Hund ein Tier?")

        # Build multi-hop chain
        root = ProofStep(
            step_id="conclusion",
            step_type=StepType.INFERENCE,
            output="hund ist tier",
            confidence=0.85,
        )

        hop1 = ProofStep(
            step_id="hop1",
            step_type=StepType.GRAPH_TRAVERSAL,
            output="hund ist säugetier",
            confidence=1.0,
        )

        hop2 = ProofStep(
            step_id="hop2",
            step_type=StepType.GRAPH_TRAVERSAL,
            output="säugetier ist tier",
            confidence=1.0,
        )

        root.add_subgoal(hop1)
        root.add_subgoal(hop2)
        tree.add_root_step(root)

        # Render
        widget = ProofTreeWidget()
        widget.set_proof_tree(tree)

        # Verify structure
        assert len(widget.node_items) == 3

        # Verify node types (shapes)
        root_item = widget.node_items["conclusion"]
        assert (
            root_item._get_shape_type(root_item.tree_node.step.step_type) == "rectangle"
        )

        hop1_item = widget.node_items["hop1"]
        assert hop1_item._get_shape_type(hop1_item.tree_node.step.step_type) == "circle"

    def test_abductive_reasoning_visualization(self, qapp):
        """Test: Abductive Reasoning wird korrekt visualisiert"""
        tree = ProofTree(query="Warum ist der Boden nass?")

        hypothesis = ProofStep(
            step_id="hyp",
            step_type=StepType.HYPOTHESIS,
            output="Es hat geregnet",
            confidence=0.75,
        )

        evidence = ProofStep(
            step_id="evidence",
            step_type=StepType.FACT_MATCH,
            output="boden ist nass",
            confidence=1.0,
        )

        hypothesis.add_subgoal(evidence)
        tree.add_root_step(hypothesis)

        # Render
        widget = ProofTreeWidget()
        widget.set_proof_tree(tree)

        # Verify hypothesis has circle shape
        hyp_item = widget.node_items["hyp"]
        assert hyp_item._get_shape_type(StepType.HYPOTHESIS) == "circle"

        # Verify confidence color (0.75 = medium)
        color = hyp_item._get_confidence_color(0.75)
        assert color == QColor("#f39c12")


class TestMVPEnhancements:
    """Tests für MVP-Erweiterungen (Tooltips, Filter, Performance)"""

    def test_tooltip_includes_source_component(self, qapp):
        """Test: Tooltip enthält Source Component"""
        step = ProofStep(
            step_id="test",
            step_type=StepType.FACT_MATCH,
            output="test output",
            source_component="component_9_logik_engine",
        )
        tree_node = ProofTreeNode(step=step)
        node_item = ProofNodeItem(tree_node)

        tooltip = node_item.toolTip()
        assert "Quelle:" in tooltip
        assert "component_9_logik_engine" in tooltip

    def test_tooltip_includes_timestamp(self, qapp):
        """Test: Tooltip enthält formatierten Timestamp"""
        from datetime import datetime

        step = ProofStep(
            step_id="test",
            step_type=StepType.FACT_MATCH,
            output="test output",
            timestamp=datetime(2024, 1, 15, 14, 30, 45),
        )
        tree_node = ProofTreeNode(step=step)
        node_item = ProofNodeItem(tree_node)

        tooltip = node_item.toolTip()
        assert "Zeitstempel:" in tooltip
        assert "2024-01-15" in tooltip
        assert "14:30:45" in tooltip

    def test_tooltip_includes_metadata(self, qapp):
        """Test: Tooltip enthält Metadata wenn vorhanden"""
        step = ProofStep(
            step_id="test",
            step_type=StepType.HYPOTHESIS,
            output="test output",
            metadata={
                "strategy": "template-based",
                "score": 0.75,
                "source": "abductive_engine",
            },
        )
        tree_node = ProofTreeNode(step=step)
        node_item = ProofNodeItem(tree_node)

        tooltip = node_item.toolTip()
        assert "Metadata:" in tooltip
        assert "strategy" in tooltip
        assert "template-based" in tooltip

    def test_tooltip_metadata_truncation(self, qapp):
        """Test: Lange Metadata-Werte werden abgeschnitten"""
        long_value = "x" * 100
        step = ProofStep(
            step_id="test",
            step_type=StepType.FACT_MATCH,
            output="test",
            metadata={"long_field": long_value},
        )
        tree_node = ProofTreeNode(step=step)
        node_item = ProofNodeItem(tree_node)

        tooltip = node_item.toolTip()
        # Should be truncated to 50 chars + "..."
        assert "xxx...xxx" in tooltip or "..." in tooltip

    def test_confidence_filter_initialization(self, qapp):
        """Test: Confidence-Filter wird korrekt initialisiert"""
        widget = ProofTreeWidget()

        assert widget.min_confidence == 0.0
        assert widget.filter_enabled is False
        assert hasattr(widget, "filter_checkbox")
        assert hasattr(widget, "confidence_slider")
        assert hasattr(widget, "confidence_label")

    def test_confidence_filter_toggle(self, qapp, simple_proof_tree):
        """Test: Confidence-Filter kann aktiviert/deaktiviert werden"""
        widget = ProofTreeWidget()
        widget.set_proof_tree(simple_proof_tree)

        len(widget.node_items)

        # Activate filter
        widget._on_filter_toggled(True)
        assert widget.filter_enabled is True

        # Deactivate filter
        widget._on_filter_toggled(False)
        assert widget.filter_enabled is False

    def test_confidence_filter_slider_change(self, qapp):
        """Test: Confidence-Slider aktualisiert Threshold"""
        widget = ProofTreeWidget()

        # Slider value 0-100 maps to 0.0-1.0
        widget._on_confidence_changed(50)
        assert widget.min_confidence == 0.5
        assert widget.confidence_label.text() == "0.50"

        widget._on_confidence_changed(80)
        assert widget.min_confidence == 0.8
        assert widget.confidence_label.text() == "0.80"

    def test_confidence_filter_filters_low_confidence_nodes(self, qapp):
        """Test: Filter entfernt Knoten mit niedriger Konfidenz"""
        tree = ProofTree(query="Filter Test")

        high_conf = ProofStep(
            step_id="high",
            step_type=StepType.FACT_MATCH,
            output="high confidence",
            confidence=0.9,
        )

        low_conf = ProofStep(
            step_id="low",
            step_type=StepType.FACT_MATCH,
            output="low confidence",
            confidence=0.3,
        )

        high_conf.add_subgoal(low_conf)
        tree.add_root_step(high_conf)

        widget = ProofTreeWidget()
        widget.set_proof_tree(tree)

        # Initially both nodes visible
        assert len(widget.node_items) == 2

        # Enable filter with threshold 0.5
        widget.min_confidence = 0.5
        widget.filter_enabled = True
        widget._render_tree()

        # Only high_conf should be visible
        assert len(widget.node_items) == 1
        assert "high" in widget.node_items
        assert "low" not in widget.node_items

    def test_should_display_node_respects_filter(self, qapp):
        """Test: _should_display_node respektiert Filter-Einstellungen"""
        widget = ProofTreeWidget()

        high_step = ProofStep(
            step_id="high", step_type=StepType.FACT_MATCH, output="test", confidence=0.9
        )
        low_step = ProofStep(
            step_id="low", step_type=StepType.FACT_MATCH, output="test", confidence=0.3
        )

        high_node = ProofTreeNode(step=high_step)
        low_node = ProofTreeNode(step=low_step)

        # Filter disabled: all nodes pass
        widget.filter_enabled = False
        assert widget._should_display_node(high_node) is True
        assert widget._should_display_node(low_node) is True

        # Filter enabled with threshold 0.5
        widget.filter_enabled = True
        widget.min_confidence = 0.5
        assert widget._should_display_node(high_node) is True
        assert widget._should_display_node(low_node) is False

    def test_performance_node_count(self, qapp):
        """Test: _count_nodes zählt korrekt alle Knoten"""
        widget = ProofTreeWidget()

        # Create tree with 1 + 2 + 2 = 5 nodes
        root = ProofStep(step_id="root", step_type=StepType.INFERENCE, output="root")
        child1 = ProofStep(step_id="c1", step_type=StepType.FACT_MATCH, output="c1")
        child2 = ProofStep(step_id="c2", step_type=StepType.FACT_MATCH, output="c2")
        grandchild1 = ProofStep(
            step_id="gc1", step_type=StepType.FACT_MATCH, output="gc1"
        )
        grandchild2 = ProofStep(
            step_id="gc2", step_type=StepType.FACT_MATCH, output="gc2"
        )

        child1.add_subgoal(grandchild1)
        child2.add_subgoal(grandchild2)
        root.add_subgoal(child1)
        root.add_subgoal(child2)

        root_node = ProofTreeNode.from_proof_step(root)

        count = widget._count_nodes(root_node)
        assert count == 5

    def test_performance_auto_collapse_large_tree(self, qapp):
        """Test: Große Bäume werden automatisch kollabiert"""
        widget = ProofTreeWidget()
        widget.max_nodes_threshold = 10  # Lower threshold for testing

        tree = ProofTree(query="Large Tree")

        # Create tree with >10 nodes
        root = ProofStep(step_id="root", step_type=StepType.INFERENCE, output="root")
        for i in range(12):
            child = ProofStep(
                step_id=f"c{i}", step_type=StepType.FACT_MATCH, output=f"c{i}"
            )
            root.add_subgoal(child)

        tree.add_root_step(root)

        widget.set_proof_tree(tree)

        # Check that auto-collapse was triggered
        assert "Auto-Collapse" in widget.status_label.text()

        # Root should be expanded (depth 0-1)
        root_node = widget.tree_nodes[0]
        assert root_node.expanded is True

    def test_performance_collapse_beyond_depth(self, qapp):
        """Test: _collapse_beyond_depth kollabiert korrekt nach Tiefe"""
        widget = ProofTreeWidget()

        # Create 4-level tree
        level0 = ProofStep(step_id="l0", step_type=StepType.INFERENCE, output="l0")
        level1 = ProofStep(step_id="l1", step_type=StepType.FACT_MATCH, output="l1")
        level2 = ProofStep(step_id="l2", step_type=StepType.FACT_MATCH, output="l2")
        level3 = ProofStep(step_id="l3", step_type=StepType.FACT_MATCH, output="l3")

        level0.add_subgoal(level1)
        level1.add_subgoal(level2)
        level2.add_subgoal(level3)

        root_node = ProofTreeNode.from_proof_step(level0)

        # Collapse beyond depth 2
        widget._collapse_beyond_depth(root_node, current_depth=0, max_depth=2)

        # Levels 0, 1 should be expanded
        assert root_node.expanded is True
        assert root_node.children[0].expanded is True

        # Level 2 should be collapsed
        assert root_node.children[0].children[0].expanded is False

    def test_rendered_node_count_tracking(self, qapp, simple_proof_tree):
        """Test: Gerenderte Knoten werden korrekt gezählt"""
        widget = ProofTreeWidget()
        widget.set_proof_tree(simple_proof_tree)

        assert widget.rendered_node_count == 3

        # Enable filter to reduce count
        widget.min_confidence = 1.0  # Only show perfect confidence
        widget.filter_enabled = True
        widget._render_tree()

        # Should have fewer nodes now
        assert widget.rendered_node_count < 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
