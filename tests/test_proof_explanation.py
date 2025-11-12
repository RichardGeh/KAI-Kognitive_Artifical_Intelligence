"""
test_proof_explanation.py

Test Suite für component_17_proof_explanation.py
Tests für Unified Proof Explanation System
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import tempfile
from datetime import datetime

import pytest

# Import from component_9_logik_engine_proof (function was moved there)
from component_9_logik_engine_proof import create_proof_tree_from_logic_engine
from component_17_proof_explanation import (
    ProofStep,
    ProofTree,
    ProofTreeNode,
    StepType,
    _confidence_bar,
    _get_step_icon,
    export_proof_to_json,
    format_proof_chain,
    format_proof_step,
    format_proof_tree,
    generate_explanation_text,
    import_proof_from_json,
    merge_proof_trees,
)


class TestProofStep:
    """Tests für ProofStep Datenstruktur"""

    def test_proof_step_creation(self):
        """Test: ProofStep kann erstellt werden"""
        step = ProofStep(
            step_id="step_1",
            step_type=StepType.FACT_MATCH,
            inputs=["apfel"],
            output="apfel ist eine frucht",
            confidence=0.95,
            explanation_text="Fand Fakt in Wissensbasis",
        )

        assert step.step_id == "step_1"
        assert step.step_type == StepType.FACT_MATCH
        assert step.inputs == ["apfel"]
        assert step.output == "apfel ist eine frucht"
        assert step.confidence == 0.95
        assert step.explanation_text == "Fand Fakt in Wissensbasis"
        assert isinstance(step.timestamp, datetime)
        assert step.source_component == "unknown"

    def test_proof_step_with_subgoals(self):
        """Test: ProofStep mit Subgoals"""
        parent = ProofStep(
            step_id="parent_1",
            step_type=StepType.RULE_APPLICATION,
            rule_name="transitivity_rule",
            output="hund ist tier",
        )

        child1 = ProofStep(
            step_id="child_1",
            step_type=StepType.FACT_MATCH,
            output="hund ist säugetier",
        )

        child2 = ProofStep(
            step_id="child_2",
            step_type=StepType.FACT_MATCH,
            output="säugetier ist tier",
        )

        parent.add_subgoal(child1)
        parent.add_subgoal(child2)

        assert len(parent.subgoals) == 2
        assert parent.subgoals[0].step_id == "child_1"
        assert parent.subgoals[1].step_id == "child_2"

    def test_proof_step_to_dict(self):
        """Test: ProofStep kann zu Dictionary konvertiert werden"""
        step = ProofStep(
            step_id="step_1",
            step_type=StepType.INFERENCE,
            inputs=["input1", "input2"],
            rule_name="rule_1",
            output="conclusion",
            confidence=0.8,
            explanation_text="Inference explanation",
            bindings={"?x": "value"},
            metadata={"key": "value"},
            source_component="test_component",
        )

        step_dict = step.to_dict()

        assert step_dict["step_id"] == "step_1"
        assert step_dict["step_type"] == "inference"
        assert step_dict["inputs"] == ["input1", "input2"]
        assert step_dict["rule_name"] == "rule_1"
        assert step_dict["output"] == "conclusion"
        assert step_dict["confidence"] == 0.8
        assert step_dict["bindings"]["?x"] == "value"
        assert step_dict["metadata"]["key"] == "value"

    def test_proof_step_from_dict(self):
        """Test: ProofStep kann aus Dictionary erstellt werden"""
        data = {
            "step_id": "step_1",
            "step_type": "fact_match",
            "inputs": ["input1"],
            "rule_name": None,
            "output": "output",
            "confidence": 0.9,
            "explanation_text": "explanation",
            "parent_steps": [],
            "bindings": {},
            "metadata": {},
            "timestamp": datetime.now().isoformat(),
            "source_component": "test",
            "subgoals": [],
        }

        step = ProofStep.from_dict(data)

        assert step.step_id == "step_1"
        assert step.step_type == StepType.FACT_MATCH
        assert step.inputs == ["input1"]
        assert step.output == "output"
        assert step.confidence == 0.9

    def test_proof_step_get_all_dependencies(self):
        """Test: Alle Abhängigkeiten können abgerufen werden"""
        parent = ProofStep(
            step_id="parent", step_type=StepType.RULE_APPLICATION, output="conclusion"
        )

        child1 = ProofStep(
            step_id="child1",
            step_type=StepType.FACT_MATCH,
            parent_steps=["fact1", "fact2"],
            output="premise1",
        )

        child2 = ProofStep(
            step_id="child2",
            step_type=StepType.FACT_MATCH,
            parent_steps=["fact3"],
            output="premise2",
        )

        parent.add_subgoal(child1)
        parent.add_subgoal(child2)

        deps = parent.get_all_dependencies()
        assert "fact1" in deps
        assert "fact2" in deps
        assert "fact3" in deps
        assert len(deps) == 3


class TestProofTreeNode:
    """Tests für ProofTreeNode Datenstruktur"""

    def test_proof_tree_node_creation(self):
        """Test: ProofTreeNode kann erstellt werden"""
        step = ProofStep(
            step_id="step_1", step_type=StepType.FACT_MATCH, output="test output"
        )

        node = ProofTreeNode(step=step)

        assert node.step.step_id == "step_1"
        assert node.expanded is True
        assert node.parent is None
        assert len(node.children) == 0
        assert node.position == (0, 0)

    def test_add_child_sets_parent_reference(self):
        """Test: add_child setzt parent-Referenz korrekt"""
        parent_step = ProofStep(
            step_id="parent", step_type=StepType.RULE_APPLICATION, output="parent"
        )
        child_step = ProofStep(
            step_id="child", step_type=StepType.FACT_MATCH, output="child"
        )

        parent_node = ProofTreeNode(step=parent_step)
        child_node = ProofTreeNode(step=child_step)

        parent_node.add_child(child_node)

        assert len(parent_node.children) == 1
        assert child_node.parent == parent_node

    def test_get_depth(self):
        """Test: Tiefe des Knotens wird korrekt berechnet"""
        root_step = ProofStep(
            step_id="root", step_type=StepType.INFERENCE, output="root"
        )
        level1_step = ProofStep(
            step_id="level1", step_type=StepType.RULE_APPLICATION, output="level1"
        )
        level2_step = ProofStep(
            step_id="level2", step_type=StepType.FACT_MATCH, output="level2"
        )

        root = ProofTreeNode(step=root_step)
        level1 = ProofTreeNode(step=level1_step)
        level2 = ProofTreeNode(step=level2_step)

        root.add_child(level1)
        level1.add_child(level2)

        assert root.get_depth() == 0
        assert level1.get_depth() == 1
        assert level2.get_depth() == 2

    def test_get_path_to_root(self):
        """Test: Pfad zur Wurzel wird korrekt berechnet"""
        root_step = ProofStep(
            step_id="root", step_type=StepType.INFERENCE, output="root"
        )
        mid_step = ProofStep(
            step_id="mid", step_type=StepType.RULE_APPLICATION, output="mid"
        )
        leaf_step = ProofStep(
            step_id="leaf", step_type=StepType.FACT_MATCH, output="leaf"
        )

        root = ProofTreeNode(step=root_step)
        mid = ProofTreeNode(step=mid_step)
        leaf = ProofTreeNode(step=leaf_step)

        root.add_child(mid)
        mid.add_child(leaf)

        path = leaf.get_path_to_root()
        assert len(path) == 3
        assert path[0].step.step_id == "root"
        assert path[1].step.step_id == "mid"
        assert path[2].step.step_id == "leaf"

    def test_get_all_descendants(self):
        """Test: Alle Nachfahren werden korrekt gefunden"""
        root_step = ProofStep(
            step_id="root", step_type=StepType.INFERENCE, output="root"
        )
        child1_step = ProofStep(
            step_id="child1", step_type=StepType.FACT_MATCH, output="child1"
        )
        child2_step = ProofStep(
            step_id="child2", step_type=StepType.FACT_MATCH, output="child2"
        )
        grandchild_step = ProofStep(
            step_id="grandchild", step_type=StepType.FACT_MATCH, output="grandchild"
        )

        root = ProofTreeNode(step=root_step)
        child1 = ProofTreeNode(step=child1_step)
        child2 = ProofTreeNode(step=child2_step)
        grandchild = ProofTreeNode(step=grandchild_step)

        root.add_child(child1)
        root.add_child(child2)
        child1.add_child(grandchild)

        descendants = root.get_all_descendants()
        assert len(descendants) == 3
        assert grandchild in descendants

    def test_expand_collapse_toggle(self):
        """Test: Expand/Collapse/Toggle funktionieren korrekt"""
        step = ProofStep(step_id="test", step_type=StepType.FACT_MATCH, output="test")
        node = ProofTreeNode(step=step)

        # Standardmäßig expanded
        assert node.expanded is True

        node.collapse()
        assert node.expanded is False

        node.expand()
        assert node.expanded is True

        node.toggle_expansion()
        assert node.expanded is False

        node.toggle_expansion()
        assert node.expanded is True

    def test_from_proof_step_converts_subgoals(self):
        """Test: from_proof_step konvertiert Subgoals zu Children"""
        parent_step = ProofStep(
            step_id="parent", step_type=StepType.RULE_APPLICATION, output="parent"
        )

        child1_step = ProofStep(
            step_id="child1", step_type=StepType.FACT_MATCH, output="child1"
        )

        child2_step = ProofStep(
            step_id="child2", step_type=StepType.FACT_MATCH, output="child2"
        )

        parent_step.add_subgoal(child1_step)
        parent_step.add_subgoal(child2_step)

        tree_node = ProofTreeNode.from_proof_step(parent_step)

        assert len(tree_node.children) == 2
        assert tree_node.children[0].step.step_id == "child1"
        assert tree_node.children[1].step.step_id == "child2"
        assert tree_node.children[0].parent == tree_node


class TestProofTree:
    """Tests für ProofTree Struktur"""

    def test_proof_tree_creation(self):
        """Test: ProofTree kann erstellt werden"""
        tree = ProofTree(query="Was ist ein Apfel?")

        assert tree.query == "Was ist ein Apfel?"
        assert len(tree.root_steps) == 0
        assert isinstance(tree.created_at, datetime)

    def test_add_root_step(self):
        """Test: Root-Steps können hinzugefügt werden"""
        tree = ProofTree(query="Test Query")

        step1 = ProofStep(step_id="s1", step_type=StepType.FACT_MATCH, output="fact1")
        step2 = ProofStep(
            step_id="s2", step_type=StepType.INFERENCE, output="inference1"
        )

        tree.add_root_step(step1)
        tree.add_root_step(step2)

        assert len(tree.root_steps) == 2
        assert tree.root_steps[0].step_id == "s1"
        assert tree.root_steps[1].step_id == "s2"

    def test_get_all_steps_flattens_tree(self):
        """Test: get_all_steps gibt alle Schritte zurück (flach)"""
        tree = ProofTree(query="Test")

        root = ProofStep(
            step_id="root", step_type=StepType.RULE_APPLICATION, output="root"
        )
        child1 = ProofStep(
            step_id="child1", step_type=StepType.FACT_MATCH, output="child1"
        )
        child2 = ProofStep(
            step_id="child2", step_type=StepType.FACT_MATCH, output="child2"
        )

        root.add_subgoal(child1)
        root.add_subgoal(child2)

        tree.add_root_step(root)

        all_steps = tree.get_all_steps()
        assert len(all_steps) == 3
        assert root in all_steps
        assert child1 in all_steps
        assert child2 in all_steps

    def test_get_step_by_id(self):
        """Test: Schritte können anhand ID gefunden werden"""
        tree = ProofTree(query="Test")

        step1 = ProofStep(step_id="s1", step_type=StepType.FACT_MATCH, output="step1")
        step2 = ProofStep(step_id="s2", step_type=StepType.INFERENCE, output="step2")

        step1.add_subgoal(step2)
        tree.add_root_step(step1)

        found = tree.get_step_by_id("s2")
        assert found is not None
        assert found.step_id == "s2"
        assert found.output == "step2"

        not_found = tree.get_step_by_id("nonexistent")
        assert not_found is None

    def test_to_tree_nodes(self):
        """Test: ProofTree kann zu ProofTreeNode-Liste konvertiert werden"""
        tree = ProofTree(query="Test")

        step1 = ProofStep(step_id="s1", step_type=StepType.FACT_MATCH, output="step1")
        child = ProofStep(step_id="child", step_type=StepType.INFERENCE, output="child")
        step1.add_subgoal(child)

        tree.add_root_step(step1)

        tree_nodes = tree.to_tree_nodes()
        assert len(tree_nodes) == 1
        assert isinstance(tree_nodes[0], ProofTreeNode)
        assert tree_nodes[0].step.step_id == "s1"
        assert len(tree_nodes[0].children) == 1

    def test_to_dict_and_serialization(self):
        """Test: ProofTree kann zu Dictionary konvertiert werden"""
        tree = ProofTree(query="Test Query", metadata={"version": "1.0"})

        step = ProofStep(
            step_id="s1", step_type=StepType.FACT_MATCH, output="output", confidence=0.9
        )

        tree.add_root_step(step)

        tree_dict = tree.to_dict()

        assert tree_dict["query"] == "Test Query"
        assert tree_dict["metadata"]["version"] == "1.0"
        assert len(tree_dict["root_steps"]) == 1
        assert tree_dict["root_steps"][0]["step_id"] == "s1"


class TestExplanationGenerators:
    """Tests für Explanation Generation Functions"""

    def test_format_proof_step_basic(self):
        """Test: ProofStep wird korrekt formatiert"""
        step = ProofStep(
            step_id="s1",
            step_type=StepType.FACT_MATCH,
            inputs=["apfel"],
            output="apfel ist eine frucht",
            confidence=0.95,
            explanation_text="Fand Fakt in Wissensbasis",
        )

        formatted = format_proof_step(step, indent=0, show_details=True)

        assert "fact_match" in formatted
        assert "Fand Fakt in Wissensbasis" in formatted
        assert "apfel ist eine frucht" in formatted
        assert "0.95" in formatted
        assert "apfel" in formatted

    def test_format_proof_step_with_subgoals(self):
        """Test: ProofStep mit Subgoals wird rekursiv formatiert"""
        parent = ProofStep(
            step_id="parent",
            step_type=StepType.RULE_APPLICATION,
            rule_name="transitivity",
            output="conclusion",
            confidence=0.8,
            explanation_text="Wendete Transitivitätsregel an",
        )

        child = ProofStep(
            step_id="child",
            step_type=StepType.FACT_MATCH,
            output="premise",
            confidence=1.0,
            explanation_text="Fand Prämisse",
        )

        parent.add_subgoal(child)

        formatted = format_proof_step(parent, indent=0, show_details=True)

        assert "Unterbeweise" in formatted
        assert "transitivity" in formatted
        assert "premise" in formatted

    def test_format_proof_tree(self):
        """Test: Vollständiger ProofTree wird formatiert"""
        tree = ProofTree(query="Was ist ein Apfel?")

        step1 = ProofStep(
            step_id="s1",
            step_type=StepType.GRAPH_TRAVERSAL,
            output="apfel ist frucht",
            confidence=0.9,
            explanation_text="Pfad gefunden",
        )

        step2 = ProofStep(
            step_id="s2",
            step_type=StepType.FACT_MATCH,
            output="apfel ist rot",
            confidence=1.0,
            explanation_text="Direkter Fakt",
        )

        tree.add_root_step(step1)
        tree.add_root_step(step2)

        formatted = format_proof_tree(tree, show_details=True)

        assert "Beweisbaum für: Was ist ein Apfel?" in formatted
        assert "Beweiskette 1:" in formatted
        assert "Beweiskette 2:" in formatted
        assert "Gesamt: 2 Schritte" in formatted

    def test_format_proof_tree_empty(self):
        """Test: Leerer ProofTree wird korrekt formatiert"""
        tree = ProofTree(query="Empty Query")

        formatted = format_proof_tree(tree)

        assert "Empty Query" in formatted
        assert "Keine Beweisschritte vorhanden" in formatted

    def test_format_proof_chain(self):
        """Test: Lineare Proof-Chain wird formatiert"""
        steps = [
            ProofStep(
                step_id="s1",
                step_type=StepType.FACT_MATCH,
                output="fact1",
                explanation_text="Schritt 1",
            ),
            ProofStep(
                step_id="s2",
                step_type=StepType.RULE_APPLICATION,
                output="fact2",
                explanation_text="Schritt 2",
            ),
            ProofStep(
                step_id="s3",
                step_type=StepType.INFERENCE,
                output="conclusion",
                explanation_text="Schritt 3",
            ),
        ]

        formatted = format_proof_chain(steps)

        assert "Schritt 1:" in formatted
        assert "Schritt 2:" in formatted
        assert "Schritt 3:" in formatted
        assert "fact1" in formatted
        assert "conclusion" in formatted


class TestNaturalLanguageGeneration:
    """Tests für Natural Language Explanation Generation (German)"""

    def test_generate_explanation_fact_match(self):
        """Test: FACT_MATCH Explanation"""
        explanation = generate_explanation_text(
            step_type=StepType.FACT_MATCH,
            inputs=["apfel"],
            output="apfel ist eine frucht",
        )

        assert "Fand Fakt direkt in der Wissensbasis" in explanation
        assert "apfel ist eine frucht" in explanation

    def test_generate_explanation_rule_application(self):
        """Test: RULE_APPLICATION Explanation"""
        explanation = generate_explanation_text(
            step_type=StepType.RULE_APPLICATION,
            inputs=["premise1", "premise2"],
            output="conclusion",
            rule_name="transitivity_rule",
        )

        assert "Wendete Regel" in explanation
        assert "transitivity_rule" in explanation
        assert "2 Prämissen" in explanation

    def test_generate_explanation_graph_traversal(self):
        """Test: GRAPH_TRAVERSAL Explanation mit Pfad"""
        explanation = generate_explanation_text(
            step_type=StepType.GRAPH_TRAVERSAL,
            inputs=[],
            output="hund ist tier",
            metadata={"hops": 3, "path": ["hund", "säugetier", "tier"]},
        )

        assert "Fand Pfad über 3 Schritte" in explanation
        assert "hund -> säugetier -> tier" in explanation

    def test_generate_explanation_hypothesis(self):
        """Test: HYPOTHESIS Explanation"""
        explanation = generate_explanation_text(
            step_type=StepType.HYPOTHESIS,
            inputs=[],
            output="hypothese",
            metadata={"strategy": "template-based", "score": 0.75},
        )

        assert "Generierte Hypothese" in explanation
        assert "template-based" in explanation
        assert "0.75" in explanation

    def test_generate_explanation_probabilistic(self):
        """Test: PROBABILISTIC Explanation"""
        explanation = generate_explanation_text(
            step_type=StepType.PROBABILISTIC,
            inputs=[],
            output="probabilistic conclusion",
            metadata={"probability": 0.85},
        )

        assert "Bayesianische Inferenz" in explanation
        assert "0.85" in explanation

    def test_generate_explanation_decomposition(self):
        """Test: DECOMPOSITION Explanation"""
        explanation = generate_explanation_text(
            step_type=StepType.DECOMPOSITION,
            inputs=[],
            output="",
            metadata={"subgoals": ["sub1", "sub2", "sub3"]},
        )

        assert "Zerlegte Ziel in 3 Unterziele" in explanation

    def test_generate_explanation_unification(self):
        """Test: UNIFICATION Explanation"""
        explanation = generate_explanation_text(
            step_type=StepType.UNIFICATION,
            inputs=[],
            output="",
            bindings={"?x": "hund", "?y": "tier"},
        )

        assert "Unifizierte Variablen" in explanation
        assert "?x=hund" in explanation
        assert "?y=tier" in explanation


class TestHelperFunctions:
    """Tests für Helper Functions"""

    def test_get_step_icon(self):
        """Test: Icons für alle StepTypes"""
        assert _get_step_icon(StepType.FACT_MATCH) == "[INFO]"
        assert _get_step_icon(StepType.RULE_APPLICATION) == "⚙️"
        assert _get_step_icon(StepType.INFERENCE) == "💡"
        assert _get_step_icon(StepType.HYPOTHESIS) == "🔬"
        assert _get_step_icon(StepType.GRAPH_TRAVERSAL) == "🗺️"
        assert _get_step_icon(StepType.PROBABILISTIC) == "🎲"
        assert _get_step_icon(StepType.DECOMPOSITION) == "🔀"
        assert _get_step_icon(StepType.UNIFICATION) == "🔗"

    def test_confidence_bar(self):
        """Test: Confidence Bar wird korrekt generiert"""
        bar_high = _confidence_bar(1.0, width=10)
        assert bar_high == "[██████████]"

        bar_mid = _confidence_bar(0.5, width=10)
        assert bar_mid == "[█████░░░░░]"

        bar_low = _confidence_bar(0.0, width=10)
        assert bar_low == "[░░░░░░░░░░]"

        bar_75 = _confidence_bar(0.75, width=10)
        assert bar_75 == "[███████░░░]"


class TestIntegrationFunctions:
    """Tests für Integration mit anderen Komponenten"""

    def test_create_proof_tree_from_logic_engine(self):
        """Test: ProofTree aus Logic Engine Proof erstellen"""
        # Mock Logic Engine Proof (simplified)
        from dataclasses import dataclass
        from typing import Dict, List

        @dataclass
        class MockGoal:
            id: str
            pred: str
            args: Dict[str, str]
            depth: int = 0

        @dataclass
        class MockFact:
            pred: str
            args: str

        @dataclass
        class MockLogicProof:
            goal: MockGoal
            method: str
            supporting_facts: List[MockFact]
            rule_id: str
            bindings: Dict[str, str]
            confidence: float
            subgoals: List["MockLogicProof"]

        mock_proof = MockLogicProof(
            goal=MockGoal(
                id="g1",
                pred="is_a",
                args={"subject": "hund", "object": "tier"},
                depth=0,
            ),
            method="fact",
            supporting_facts=[MockFact(pred="is_a", args="hund, tier")],
            rule_id=None,
            bindings={},
            confidence=1.0,
            subgoals=[],
        )

        tree = create_proof_tree_from_logic_engine(mock_proof, "Was ist ein Hund?")

        assert tree.query == "Was ist ein Hund?"
        assert len(tree.root_steps) == 1
        assert tree.root_steps[0].step_type == StepType.FACT_MATCH
        assert tree.root_steps[0].source_component == "component_9_logik_engine"

    def test_merge_proof_trees(self):
        """Test: Mehrere ProofTrees zusammenführen"""
        tree1 = ProofTree(query="Test Query")
        tree1.add_root_step(
            ProofStep(step_id="s1", step_type=StepType.FACT_MATCH, output="fact1")
        )

        tree2 = ProofTree(query="Test Query")
        tree2.add_root_step(
            ProofStep(
                step_id="s2", step_type=StepType.GRAPH_TRAVERSAL, output="traversal1"
            )
        )

        merged = merge_proof_trees([tree1, tree2], "Test Query")

        assert merged.query == "Test Query"
        assert len(merged.root_steps) == 2
        assert merged.root_steps[0].step_id == "s1"
        assert merged.root_steps[1].step_id == "s2"


class TestExportImport:
    """Tests für JSON Export/Import"""

    def test_export_and_import_proof_tree(self):
        """Test: ProofTree kann exportiert und importiert werden"""
        # Create proof tree
        tree = ProofTree(query="Was ist ein Apfel?", metadata={"version": "1.0"})

        step1 = ProofStep(
            step_id="s1",
            step_type=StepType.FACT_MATCH,
            inputs=["apfel"],
            output="apfel ist eine frucht",
            confidence=0.95,
            explanation_text="Direkter Fakt",
            bindings={"?x": "apfel"},
            metadata={"source": "test"},
        )

        child = ProofStep(
            step_id="child",
            step_type=StepType.GRAPH_TRAVERSAL,
            output="frucht ist lebensmittel",
            confidence=0.9,
        )

        step1.add_subgoal(child)
        tree.add_root_step(step1)

        # Export to temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            temp_path = f.name

        try:
            export_proof_to_json(tree, temp_path)

            # Import from file
            imported_tree = import_proof_from_json(temp_path)

            # Verify
            assert imported_tree.query == "Was ist ein Apfel?"
            assert imported_tree.metadata["version"] == "1.0"
            assert len(imported_tree.root_steps) == 1
            assert imported_tree.root_steps[0].step_id == "s1"
            assert imported_tree.root_steps[0].confidence == 0.95
            assert imported_tree.root_steps[0].bindings["?x"] == "apfel"
            assert len(imported_tree.root_steps[0].subgoals) == 1
            assert imported_tree.root_steps[0].subgoals[0].step_id == "child"

        finally:
            # Cleanup
            import os

            if os.path.exists(temp_path):
                os.remove(temp_path)


class TestComplexScenarios:
    """Tests für komplexe Reasoning-Szenarien"""

    def test_multi_hop_reasoning_proof_tree(self):
        """Test: Multi-Hop Reasoning Proof Tree"""
        tree = ProofTree(query="Ist ein Hund ein Tier?")

        # Root: Final inference
        root = ProofStep(
            step_id="inference",
            step_type=StepType.RULE_APPLICATION,
            rule_name="transitivity",
            output="hund ist tier",
            confidence=0.85,
            explanation_text="Transitive Inferenz über Graphen-Pfad",
        )

        # Subgoal 1: hund -> säugetier
        hop1 = ProofStep(
            step_id="hop1",
            step_type=StepType.GRAPH_TRAVERSAL,
            inputs=["hund"],
            output="hund ist säugetier",
            confidence=1.0,
            explanation_text="Direkter Link in Graph",
        )

        # Subgoal 2: säugetier -> tier
        hop2 = ProofStep(
            step_id="hop2",
            step_type=StepType.GRAPH_TRAVERSAL,
            inputs=["säugetier"],
            output="säugetier ist tier",
            confidence=1.0,
            explanation_text="Direkter Link in Graph",
        )

        root.add_subgoal(hop1)
        root.add_subgoal(hop2)
        tree.add_root_step(root)

        # Verify structure
        all_steps = tree.get_all_steps()
        assert len(all_steps) == 3

        # Verify formatting
        formatted = format_proof_tree(tree)
        assert "hund ist tier" in formatted
        assert "transitivity" in formatted
        assert "Unterbeweise" in formatted

    def test_abductive_reasoning_proof_tree(self):
        """Test: Abductive Reasoning (Hypothesengenerierung)"""
        tree = ProofTree(query="Warum ist der Boden nass?")

        # Hypothese
        hypothesis = ProofStep(
            step_id="hyp1",
            step_type=StepType.HYPOTHESIS,
            output="Es hat geregnet",
            confidence=0.75,
            explanation_text="Generierte Hypothese (Strategie: template-based, Score: 0.75)",
            metadata={"strategy": "template-based", "score": 0.75},
        )

        # Supporting evidence
        evidence = ProofStep(
            step_id="evidence1",
            step_type=StepType.FACT_MATCH,
            output="boden ist nass",
            confidence=1.0,
            explanation_text="Beobachtung",
        )

        hypothesis.add_subgoal(evidence)
        tree.add_root_step(hypothesis)

        # Verify
        assert tree.root_steps[0].step_type == StepType.HYPOTHESIS
        assert tree.root_steps[0].confidence == 0.75
        assert len(tree.root_steps[0].subgoals) == 1

    def test_probabilistic_reasoning_proof_tree(self):
        """Test: Probabilistic Reasoning mit Konfidenzpropagation"""
        tree = ProofTree(query="Wahrscheinlichkeit: Vogel kann fliegen?")

        # Root: Probabilistic conclusion
        root = ProofStep(
            step_id="prob_conclusion",
            step_type=StepType.PROBABILISTIC,
            output="vogel kann fliegen",
            confidence=0.85,
            explanation_text="Bayesianische Inferenz ergab Wahrscheinlichkeit 0.85",
            metadata={"probability": 0.85},
        )

        # Prior
        prior = ProofStep(
            step_id="prior",
            step_type=StepType.FACT_MATCH,
            output="vogel ist tier",
            confidence=1.0,
        )

        # Evidence
        evidence = ProofStep(
            step_id="evidence",
            step_type=StepType.FACT_MATCH,
            output="hat flügel",
            confidence=0.9,
        )

        root.add_subgoal(prior)
        root.add_subgoal(evidence)
        tree.add_root_step(root)

        # Verify confidence propagation in formatting
        formatted = format_proof_tree(tree)
        assert "0.85" in formatted
        assert "Bayesianische Inferenz" in formatted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
