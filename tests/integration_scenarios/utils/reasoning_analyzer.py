"""
tests/integration_scenarios/utils/reasoning_analyzer.py

Deep analysis of KAI's reasoning process from ProofTree and traces.
Provides ReasoningAnalyzer class for comprehensive reasoning quality evaluation.

NO Unicode characters - ASCII only (see CLAUDE.md encoding rules).
"""

from typing import Any, Dict, List, Tuple


class ReasoningAnalyzer:
    """Analyzes KAI's reasoning process for quality and completeness"""

    def __init__(self, proof_tree: Dict, trace_logs: List[str]):
        """
        Initialize reasoning analyzer.

        Args:
            proof_tree: ProofTree dictionary from KAI
            trace_logs: List of trace log strings from reasoning process
        """
        self.proof_tree = proof_tree
        self.trace_logs = trace_logs
        self.strategies_used: List[str] = []
        self.reasoning_paths: List[List[str]] = []
        self.coherence_issues: List[str] = []

    def analyze(self) -> Dict[str, Any]:
        """
        Comprehensive analysis of reasoning quality.

        Returns:
            {
                "strategy_diversity": float,  # 0-1
                "depth_appropriateness": float,  # 0-1
                "logical_coherence": float,  # 0-1
                "completeness": float,  # 0-1
                "efficiency": float,  # 0-1 (no unnecessary steps)
                "creativity": float,  # 0-1 (novel approaches)
                "paths_explored": int,
                "dead_ends_encountered": int,
                "backtracking_count": int,
                "observations": List[str]
            }
        """
        observations = []

        # Extract strategies
        self.strategies_used = self.extract_reasoning_strategies()
        observations.append(
            f"Strategies used: {self.strategies_used if self.strategies_used else 'none'}"
        )

        # Calculate strategy diversity (0-1)
        unique_strategies = len(set(self.strategies_used))
        strategy_diversity = min(
            1.0, unique_strategies / 5.0
        )  # Normalize to max 5 strategies

        # Measure depth
        depth_metrics = self.measure_reasoning_depth()
        observations.append(
            f"Reasoning depth: max={depth_metrics['max_depth']}, avg={depth_metrics['avg_depth']:.1f}"
        )

        # Evaluate depth appropriateness (0-1)
        # Good depth is 3-10 for most scenarios
        max_depth = depth_metrics["max_depth"]
        if 3 <= max_depth <= 10:
            depth_appropriateness = 1.0
        elif max_depth < 3:
            depth_appropriateness = max_depth / 3.0
        else:
            depth_appropriateness = max(0.5, 1.0 - (max_depth - 10) * 0.05)

        # Evaluate logical coherence (0-1)
        circular_issues = self.detect_circular_reasoning()
        reasoning_gaps = self.identify_reasoning_gaps()
        coherence_issues_count = len(circular_issues) + len(reasoning_gaps)

        if coherence_issues_count == 0:
            logical_coherence = 1.0
        else:
            logical_coherence = max(0.0, 1.0 - (coherence_issues_count * 0.1))

        if circular_issues:
            observations.append(
                f"Circular reasoning detected: {len(circular_issues)} instances"
            )
        if reasoning_gaps:
            observations.append(f"Reasoning gaps identified: {len(reasoning_gaps)}")

        # Evaluate completeness (0-1)
        # Based on leaf nodes (conclusions reached)
        leaf_count = depth_metrics["leaf_nodes"]
        if leaf_count >= 1:
            completeness = min(1.0, leaf_count / 3.0)  # Ideal is 1-3 conclusions
        else:
            completeness = 0.0
            observations.append("No conclusions reached")

        # Evaluate efficiency (0-1)
        # Based on ratio of useful nodes to total nodes
        total_nodes = self._count_nodes(self.proof_tree)
        if total_nodes > 0:
            # Assume optimal is around 5-15 nodes
            if 5 <= total_nodes <= 15:
                efficiency = 1.0
            elif total_nodes < 5:
                efficiency = 0.8  # Too few nodes, but not a major issue
            else:
                # Too many nodes - penalize excess
                efficiency = max(0.3, 1.0 - (total_nodes - 15) * 0.02)
        else:
            efficiency = 0.0

        # Evaluate creativity (0-1)
        # Based on strategy diversity and non-standard approaches
        creativity = strategy_diversity * 0.6 + (
            0.4 if self._has_creative_strategies() else 0.0
        )

        # Count paths explored
        self.reasoning_paths = self._extract_all_paths(self.proof_tree)
        paths_explored = len(self.reasoning_paths)

        # Count dead ends (leaf nodes that are not conclusions)
        dead_ends = self._count_dead_ends(self.proof_tree)

        # Count backtracking (heuristic: look for retry patterns in trace logs)
        backtracking_count = self._count_backtracking_in_traces()

        observations.append(f"Explored {paths_explored} reasoning paths")
        if dead_ends > 0:
            observations.append(f"Encountered {dead_ends} dead ends")
        if backtracking_count > 0:
            observations.append(f"Backtracked {backtracking_count} times")

        return {
            "strategy_diversity": strategy_diversity,
            "depth_appropriateness": depth_appropriateness,
            "logical_coherence": logical_coherence,
            "completeness": completeness,
            "efficiency": efficiency,
            "creativity": creativity,
            "paths_explored": paths_explored,
            "dead_ends_encountered": dead_ends,
            "backtracking_count": backtracking_count,
            "observations": observations,
        }

    def extract_reasoning_strategies(self) -> List[str]:
        """
        Extract which reasoning engines were used.

        Supports both ProofTree format (root_steps/subgoals) and legacy format (children).
        Extracts strategies from source_component and rule_name fields.
        """
        strategies = []

        def traverse(node):
            # Check for strategy in node
            if isinstance(node, dict):
                if "strategy" in node:
                    strategies.append(node["strategy"])
                if "engine" in node:
                    strategies.append(node["engine"])
                if "type" in node and "engine" in str(node["type"]).lower():
                    strategies.append(node["type"])

                # NEW: Check source_component (ProofStep field)
                if "source_component" in node:
                    component = node["source_component"]
                    if "sat_solver" in component or "component_30" in component:
                        strategies.append("sat")
                    elif (
                        "constraint" in component
                        or "csp" in component
                        or "component_29" in component
                    ):
                        strategies.append("constraint_satisfaction")
                    elif "logic_engine" in component or "component_9" in component:
                        strategies.append("logic_engine")
                    elif "graph_traversal" in component or "component_12" in component:
                        strategies.append("graph_traversal")
                    elif "spatial" in component or "component_42" in component:
                        strategies.append("spatial_reasoning")
                    elif "arithmetic" in component or "component_52" in component:
                        strategies.append("arithmetic_reasoning")
                    elif "abductive" in component or "component_14" in component:
                        strategies.append("abductive_reasoning")
                    elif "probabilistic" in component or "component_16" in component:
                        strategies.append("probabilistic_reasoning")
                    elif "direct_fact_lookup" in component:
                        strategies.append("knowledge_retrieval")
                    elif "knowledge" in component or "netzwerk" in component:
                        strategies.append("knowledge_retrieval")

                # NEW: Check rule_name (ProofStep field)
                if "rule_name" in node and node["rule_name"]:
                    rule = node["rule_name"].lower()
                    if "sat" in rule:
                        strategies.append("sat")
                    elif "csp" in rule or "constraint" in rule:
                        strategies.append("constraint_satisfaction")
                    elif "unit propagation" in rule:
                        strategies.append("sat")
                    elif "logic" in rule:
                        strategies.append("logic_engine")

                # Traverse children (legacy format)
                if "children" in node:
                    for child in node["children"]:
                        traverse(child)

                # Traverse subgoals (ProofStep format)
                if "subgoals" in node:
                    for subgoal in node["subgoals"]:
                        traverse(subgoal)

                # Traverse root_steps (ProofTree format)
                if "root_steps" in node:
                    for step in node["root_steps"]:
                        traverse(step)

        if self.proof_tree:
            traverse(self.proof_tree)

        # Also extract from trace logs
        for trace in self.trace_logs:
            trace_lower = trace.lower()
            if "logic" in trace_lower and "engine" in trace_lower:
                strategies.append("logic_engine")
            if "graph" in trace_lower and "traversal" in trace_lower:
                strategies.append("graph_traversal")
            if "abductive" in trace_lower:
                strategies.append("abductive_reasoning")
            if "probabilistic" in trace_lower:
                strategies.append("probabilistic_reasoning")
            if "constraint" in trace_lower or "csp" in trace_lower:
                strategies.append("constraint_satisfaction")
            if "sat" in trace_lower and "solver" in trace_lower:
                strategies.append("sat")
            if "combinatorial" in trace_lower:
                strategies.append("combinatorial_reasoning")
            if "spatial" in trace_lower:
                strategies.append("spatial_reasoning")
            if "arithmetic" in trace_lower:
                strategies.append("arithmetic_reasoning")

        # Remove duplicates while preserving order
        seen = set()
        unique_strategies = []
        for s in strategies:
            if s not in seen:
                seen.add(s)
                unique_strategies.append(s)

        return unique_strategies

    def identify_reasoning_gaps(self) -> List[str]:
        """Identify missing reasoning steps or logical gaps"""
        gaps = []

        # Check if proof tree is too shallow
        if self.proof_tree:
            depth = self.measure_reasoning_depth()["max_depth"]
            if depth < 2:
                gaps.append("Proof tree too shallow - missing intermediate steps")

        # Check for missing conclusions
        leaf_count = self.measure_reasoning_depth()["leaf_nodes"]
        if leaf_count == 0:
            gaps.append("No conclusions reached - reasoning incomplete")

        # Check for disconnected nodes (nodes with no children and no conclusion)
        disconnected = self._find_disconnected_nodes(self.proof_tree)
        if disconnected > 0:
            gaps.append(
                f"{disconnected} disconnected nodes - incomplete reasoning paths"
            )

        return gaps

    def evaluate_strategy_selection(self) -> Tuple[float, str]:
        """
        Evaluate whether KAI chose appropriate strategies.

        Returns:
            (appropriateness_score 0-1, explanation)
        """
        if not self.strategies_used:
            return 0.0, "No strategies identified"

        # Heuristic evaluation based on strategy count and diversity
        unique_strategies = len(set(self.strategies_used))

        if unique_strategies == 0:
            return 0.0, "No strategies used"
        elif unique_strategies == 1:
            return 0.6, "Only one strategy used - limited approach"
        elif unique_strategies <= 3:
            return 0.9, "Multiple strategies used - good diversity"
        else:
            return 1.0, "Excellent strategy diversity"

    def detect_circular_reasoning(self) -> List[str]:
        """Detect any circular reasoning patterns"""
        circular_patterns = []

        # Check for repeated nodes in paths
        paths = self._extract_all_paths(self.proof_tree)
        for i, path in enumerate(paths):
            # Convert nodes to strings for comparison
            node_strs = [str(node) for node in path]

            # Check for repeated sequences
            for j in range(len(node_strs) - 1):
                for k in range(j + 1, len(node_strs)):
                    if node_strs[j] == node_strs[k]:
                        circular_patterns.append(
                            f"Path {i}: Node repeated at positions {j} and {k}"
                        )

        # Check trace logs for repeated reasoning steps
        for i, trace1 in enumerate(self.trace_logs):
            for j in range(i + 1, len(self.trace_logs)):
                trace2 = self.trace_logs[j]
                if trace1 == trace2 and len(trace1) > 20:  # Ignore short traces
                    circular_patterns.append(
                        f"Identical trace at positions {i} and {j}"
                    )

        return circular_patterns[:5]  # Return max 5 examples

    def measure_reasoning_depth(self) -> Dict[str, int]:
        """
        Measure reasoning depth metrics.

        Returns:
            {
                "max_depth": int,
                "avg_depth": float,
                "leaf_nodes": int,
                "branch_points": int
            }
        """
        if not self.proof_tree:
            return {
                "max_depth": 0,
                "avg_depth": 0.0,
                "leaf_nodes": 0,
                "branch_points": 0,
            }

        depths = []
        leaf_count = 0
        branch_count = 0

        def traverse(node, depth):
            nonlocal leaf_count, branch_count

            if not isinstance(node, dict):
                return

            # Check if leaf node
            children = node.get("children", [])
            if not children:
                leaf_count += 1
                depths.append(depth)
            else:
                # Check if branch point
                if len(children) > 1:
                    branch_count += 1

                # Traverse children
                for child in children:
                    traverse(child, depth + 1)

        traverse(self.proof_tree, 1)

        max_depth = max(depths) if depths else 0
        avg_depth = sum(depths) / len(depths) if depths else 0.0

        return {
            "max_depth": max_depth,
            "avg_depth": avg_depth,
            "leaf_nodes": leaf_count,
            "branch_points": branch_count,
        }

    # Helper methods

    def _count_nodes(self, tree: Dict) -> int:
        """Count total nodes in tree"""
        if not tree or not isinstance(tree, dict):
            return 0

        count = 1
        if "children" in tree:
            for child in tree["children"]:
                count += self._count_nodes(child)
        return count

    def _has_creative_strategies(self) -> bool:
        """Check if any creative/advanced strategies were used"""
        creative_strategies = {
            "abductive_reasoning",
            "combinatorial_reasoning",
            "resonance_reasoning",
            "meta_learning",
        }
        return bool(set(self.strategies_used) & creative_strategies)

    def _extract_all_paths(self, tree: Dict) -> List[List[Dict]]:
        """Extract all root-to-leaf paths from tree"""
        paths = []

        def traverse(node, current_path):
            if not isinstance(node, dict):
                return

            current_path = current_path + [node]

            children = node.get("children", [])
            if not children:
                # Leaf node - add path
                paths.append(current_path)
            else:
                # Traverse children
                for child in children:
                    traverse(child, current_path)

        if tree:
            traverse(tree, [])

        return paths

    def _count_dead_ends(self, tree: Dict) -> int:
        """Count leaf nodes that are not conclusions"""
        dead_ends = 0

        def traverse(node):
            nonlocal dead_ends

            if not isinstance(node, dict):
                return

            children = node.get("children", [])
            if not children:
                # Leaf node - check if it's a conclusion
                node_type = node.get("type", "").lower()
                if "conclusion" not in node_type and "result" not in node_type:
                    dead_ends += 1
            else:
                for child in children:
                    traverse(child)

        if tree:
            traverse(tree)

        return dead_ends

    def _count_backtracking_in_traces(self) -> int:
        """Count backtracking instances in trace logs"""
        backtrack_count = 0

        backtrack_keywords = [
            "backtrack",
            "retry",
            "revert",
            "undo",
            "alternative",
            "fallback",
        ]

        for trace in self.trace_logs:
            trace_lower = trace.lower()
            for keyword in backtrack_keywords:
                if keyword in trace_lower:
                    backtrack_count += 1
                    break  # Count once per trace

        return backtrack_count

    def _find_disconnected_nodes(self, tree: Dict) -> int:
        """Find nodes that have no children and no useful content"""
        disconnected = 0

        def traverse(node):
            nonlocal disconnected

            if not isinstance(node, dict):
                return

            children = node.get("children", [])
            if not children:
                # Leaf node - check if it has useful content
                has_content = bool(
                    node.get("conclusion")
                    or node.get("result")
                    or node.get("value")
                    or node.get("step")
                )
                if not has_content:
                    disconnected += 1
            else:
                for child in children:
                    traverse(child)

        if tree:
            traverse(tree)

        return disconnected
