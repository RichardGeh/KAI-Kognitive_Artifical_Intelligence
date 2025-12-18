"""
tests/integration_scenarios/utils/test_reasoning_analyzer.py

Unit tests for reasoning_analyzer.py module.
Tests ReasoningAnalyzer class with synthetic ProofTrees.

NO Unicode characters - ASCII only (see CLAUDE.md encoding rules).
"""

from tests.integration_scenarios.utils.reasoning_analyzer import ReasoningAnalyzer


class TestReasoningAnalyzer:
    """Test ReasoningAnalyzer class"""

    def test_initialization(self):
        """Test analyzer initialization"""
        tree = {"step": "root"}
        traces = ["trace1", "trace2"]
        analyzer = ReasoningAnalyzer(tree, traces)

        assert analyzer.proof_tree == tree
        assert analyzer.trace_logs == traces
        assert analyzer.strategies_used == []
        assert analyzer.reasoning_paths == []

    def test_analyze_empty_tree(self):
        """Test analysis with empty proof tree"""
        analyzer = ReasoningAnalyzer({}, [])
        result = analyzer.analyze()

        assert result["strategy_diversity"] == 0.0
        assert result["depth_appropriateness"] < 1.0
        assert result["completeness"] == 0.0
        assert "observations" in result

    def test_analyze_simple_tree(self):
        """Test analysis with simple tree"""
        tree = {
            "step": "root",
            "strategy": "logic_engine",
            "children": [{"step": "conclusion", "type": "CONCLUSION"}],
        }
        analyzer = ReasoningAnalyzer(tree, [])
        result = analyzer.analyze()

        assert result["strategy_diversity"] > 0.0
        assert result["completeness"] > 0.0
        assert result["paths_explored"] == 1

    def test_analyze_complex_tree(self):
        """Test analysis with complex branching tree"""
        tree = {
            "step": "root",
            "strategy": "logic_engine",
            "children": [
                {
                    "step": "branch1",
                    "strategy": "graph_traversal",
                    "children": [{"step": "leaf1"}],
                },
                {
                    "step": "branch2",
                    "strategy": "abductive_reasoning",
                    "children": [{"step": "leaf2"}],
                },
            ],
        }
        analyzer = ReasoningAnalyzer(tree, [])
        result = analyzer.analyze()

        assert result["strategy_diversity"] > 0.2  # Multiple strategies
        assert result["depth_appropriateness"] > 0.5  # Good depth
        assert result["paths_explored"] == 2  # Two paths

    def test_extract_reasoning_strategies_from_tree(self):
        """Test strategy extraction from proof tree"""
        tree = {
            "strategy": "logic_engine",
            "children": [
                {"strategy": "graph_traversal"},
                {"engine": "abductive_reasoning"},
            ],
        }
        analyzer = ReasoningAnalyzer(tree, [])
        strategies = analyzer.extract_reasoning_strategies()

        assert "logic_engine" in strategies
        assert "graph_traversal" in strategies
        assert "abductive_reasoning" in strategies

    def test_extract_reasoning_strategies_from_traces(self):
        """Test strategy extraction from trace logs"""
        traces = [
            "Using logic engine for deduction",
            "Graph traversal to find path",
            "Probabilistic reasoning applied",
        ]
        analyzer = ReasoningAnalyzer({}, traces)
        strategies = analyzer.extract_reasoning_strategies()

        assert "logic_engine" in strategies
        assert "graph_traversal" in strategies
        assert "probabilistic_reasoning" in strategies

    def test_measure_reasoning_depth(self):
        """Test reasoning depth measurement"""
        # Create tree with depth 4
        tree = {
            "step": "root",
            "children": [
                {
                    "step": "level2",
                    "children": [{"step": "level3", "children": [{"step": "level4"}]}],
                }
            ],
        }
        analyzer = ReasoningAnalyzer(tree, [])
        depth_metrics = analyzer.measure_reasoning_depth()

        assert depth_metrics["max_depth"] == 4
        assert depth_metrics["leaf_nodes"] == 1
        assert depth_metrics["branch_points"] == 0  # No branching

    def test_measure_reasoning_depth_with_branching(self):
        """Test depth measurement with branching"""
        tree = {
            "step": "root",
            "children": [
                {"step": "child1"},
                {"step": "child2"},
                {"step": "child3"},
            ],
        }
        analyzer = ReasoningAnalyzer(tree, [])
        depth_metrics = analyzer.measure_reasoning_depth()

        assert depth_metrics["max_depth"] == 2
        assert depth_metrics["leaf_nodes"] == 3
        assert depth_metrics["branch_points"] == 1  # Root branches to 3 children

    def test_identify_reasoning_gaps_shallow(self):
        """Test identification of shallow reasoning"""
        tree = {"step": "root"}  # Depth 1
        analyzer = ReasoningAnalyzer(tree, [])
        gaps = analyzer.identify_reasoning_gaps()

        assert len(gaps) > 0
        assert any("shallow" in gap.lower() for gap in gaps)

    def test_identify_reasoning_gaps_no_conclusion(self):
        """Test identification of missing conclusions"""
        tree = {"step": "root", "children": [{"step": "intermediate"}]}
        analyzer = ReasoningAnalyzer(tree, [])
        gaps = analyzer.identify_reasoning_gaps()

        # Should identify lack of conclusions
        assert len(gaps) > 0

    def test_evaluate_strategy_selection_none(self):
        """Test strategy evaluation with no strategies"""
        analyzer = ReasoningAnalyzer({}, [])
        analyzer.strategies_used = []
        score, explanation = analyzer.evaluate_strategy_selection()

        assert score == 0.0
        assert "No strategies" in explanation

    def test_evaluate_strategy_selection_single(self):
        """Test strategy evaluation with single strategy"""
        analyzer = ReasoningAnalyzer({}, [])
        analyzer.strategies_used = ["logic_engine"]
        score, explanation = analyzer.evaluate_strategy_selection()

        assert score == 0.6
        assert "Only one strategy" in explanation

    def test_evaluate_strategy_selection_multiple(self):
        """Test strategy evaluation with multiple strategies"""
        analyzer = ReasoningAnalyzer({}, [])
        analyzer.strategies_used = [
            "logic_engine",
            "graph_traversal",
            "abductive_reasoning",
        ]
        score, explanation = analyzer.evaluate_strategy_selection()

        assert score >= 0.9
        assert "diversity" in explanation.lower()

    def test_detect_circular_reasoning_paths(self):
        """Test detection of circular reasoning in paths"""
        tree = {
            "step": "A",
            "children": [{"step": "B", "children": [{"step": "A"}]}],  # A -> B -> A
        }
        analyzer = ReasoningAnalyzer(tree, [])
        circular = analyzer.detect_circular_reasoning()

        # Should detect repeated node
        assert len(circular) > 0

    def test_detect_circular_reasoning_traces(self):
        """Test detection of circular reasoning in traces"""
        traces = ["Long reasoning step about X"] * 3  # Repeated 3 times
        analyzer = ReasoningAnalyzer({}, traces)
        circular = analyzer.detect_circular_reasoning()

        assert len(circular) > 0
        assert any("Identical trace" in c for c in circular)

    def test_depth_appropriateness_good(self):
        """Test depth appropriateness scoring - good depth"""
        tree = {
            "step": "root",
            "children": [
                {
                    "step": "level2",
                    "children": [{"step": "level3", "children": [{"step": "level4"}]}],
                }
            ],
        }
        analyzer = ReasoningAnalyzer(tree, [])
        result = analyzer.analyze()

        assert result["depth_appropriateness"] >= 0.9  # Depth 4 is good

    def test_depth_appropriateness_shallow(self):
        """Test depth appropriateness scoring - shallow"""
        tree = {"step": "root"}
        analyzer = ReasoningAnalyzer(tree, [])
        result = analyzer.analyze()

        assert result["depth_appropriateness"] < 0.5  # Depth 1 is shallow

    def test_efficiency_good(self):
        """Test efficiency scoring - optimal node count"""
        tree = {
            "step": "root",
            "children": [
                {"step": "child1"},
                {"step": "child2", "children": [{"step": "grandchild"}]},
            ],
        }
        analyzer = ReasoningAnalyzer(tree, [])
        result = analyzer.analyze()

        # 4 nodes total - within good range
        assert result["efficiency"] > 0.8

    def test_efficiency_too_many_nodes(self):
        """Test efficiency penalty for too many nodes"""
        # Create tree with 30 nodes
        tree = {"step": "root", "children": []}
        for i in range(29):
            tree["children"].append({"step": f"node{i}"})

        analyzer = ReasoningAnalyzer(tree, [])
        result = analyzer.analyze()

        # Should be penalized for excessive nodes
        assert result["efficiency"] < 0.8

    def test_creativity_basic_strategies(self):
        """Test creativity scoring with basic strategies"""
        analyzer = ReasoningAnalyzer({}, [])
        analyzer.strategies_used = ["logic_engine", "graph_traversal"]
        result = analyzer.analyze()

        # No creative strategies
        assert result["creativity"] < 0.7

    def test_creativity_advanced_strategies(self):
        """Test creativity scoring with advanced strategies"""
        analyzer = ReasoningAnalyzer({}, [])
        analyzer.strategies_used = [
            "abductive_reasoning",
            "combinatorial_reasoning",
            "meta_learning",
        ]
        result = analyzer.analyze()

        # Has creative strategies
        assert result["creativity"] > 0.6

    def test_completeness_with_conclusions(self):
        """Test completeness scoring with conclusions"""
        tree = {
            "step": "root",
            "children": [
                {"step": "leaf1", "type": "conclusion"},
                {"step": "leaf2", "type": "result"},
            ],
        }
        analyzer = ReasoningAnalyzer(tree, [])
        result = analyzer.analyze()

        assert result["completeness"] > 0.5  # Has conclusions

    def test_count_dead_ends(self):
        """Test dead end counting"""
        tree = {
            "step": "root",
            "children": [
                {"step": "good_leaf", "type": "conclusion"},
                {"step": "dead_end_leaf"},  # No conclusion type
            ],
        }
        analyzer = ReasoningAnalyzer(tree, [])
        result = analyzer.analyze()

        assert result["dead_ends_encountered"] >= 1

    def test_count_backtracking(self):
        """Test backtracking detection"""
        traces = ["Trying approach A", "Backtracking...", "Trying approach B"]
        analyzer = ReasoningAnalyzer({}, traces)
        result = analyzer.analyze()

        assert result["backtracking_count"] >= 1
