"""
tests/integration_scenarios/utils/scoring_system.py

Multi-dimensional scoring algorithms for reasoning evaluation.
Provides functions for scoring proof tree quality, reasoning coherence,
confidence calibration, and partial correctness.

NO Unicode characters - ASCII only (see CLAUDE.md encoding rules).
"""

import math
from typing import Any, Dict, List, Tuple


def score_proof_tree_quality(
    proof_tree: Dict,
    expected_depth_range: Tuple[int, int] = (2, 10),
    required_strategies: List[str] = None,
    domain: str = "logic_puzzles",
) -> Tuple[float, List[str]]:
    """
    Analyze ProofTree structure and return quality score.

    Evaluation criteria:
    - Appropriate depth (not too shallow, not too deep)
    - Strategy diversity (used multiple reasoning approaches)
    - Logical coherence (steps follow logically)
    - Completeness (all necessary steps present)

    Args:
        proof_tree: ProofTree dictionary from KAI
        expected_depth_range: (min_depth, max_depth) - depth outside penalized
        required_strategies: List of strategies that should have been used
        domain: Domain name for domain-specific scoring rules

    Returns:
        (score 0-100, list of observations)
    """
    if required_strategies is None:
        required_strategies = []

    observations = []

    if not proof_tree:
        observations.append("No proof tree available")
        return 10.0, observations

    # Calculate depth
    depth = _calculate_tree_depth(proof_tree)
    observations.append(
        f"ProofTree depth: {depth} (expected {expected_depth_range[0]}-{expected_depth_range[1]})"
    )

    # Score depth appropriateness (0-40 points)
    min_depth, max_depth = expected_depth_range
    if min_depth <= depth <= max_depth:
        depth_score = 40.0
    elif depth < min_depth:
        # Too shallow - penalize proportionally
        depth_score = 40.0 * (depth / min_depth)
        observations.append(f"Proof tree too shallow (penalty applied)")
    else:
        # Too deep - smaller penalty
        excess = depth - max_depth
        depth_score = max(20.0, 40.0 - (excess * 2))
        observations.append(f"Proof tree deeper than expected (minor penalty)")

    # Extract strategies used
    strategies = _extract_strategies_from_tree(proof_tree)
    observations.append(f"Strategies used: {strategies if strategies else 'none'}")

    # Score strategy diversity (0-30 points)
    if strategies:
        strategy_count = len(set(strategies))
        strategy_score = min(30.0, strategy_count * 10.0)
    else:
        strategy_score = 5.0
        observations.append("No strategies identified in proof tree")

    # Check required strategies
    if required_strategies:
        missing = set(required_strategies) - set(strategies)
        if missing:
            observations.append(f"Missing expected strategies: {list(missing)}")
            strategy_score *= 0.7  # 30% penalty for missing required strategies

    # Score logical coherence (0-30 points)
    coherence_score, coherence_obs = _evaluate_tree_coherence(proof_tree)
    observations.extend(coherence_obs)

    # Total score
    total_score = depth_score + strategy_score + coherence_score

    return total_score, observations


def score_reasoning_coherence(
    reasoning_steps: List[str],
    domain_knowledge: Dict[str, Any],
) -> Tuple[float, List[str]]:
    """
    Evaluate logical coherence of reasoning steps.

    Checks:
    - Steps follow logically from previous steps
    - No circular reasoning
    - Appropriate use of domain knowledge
    - Valid inferences

    Args:
        reasoning_steps: List of reasoning step descriptions
        domain_knowledge: Domain-specific knowledge for validation

    Returns:
        (score 0-100, list of issues)
    """
    issues = []

    if not reasoning_steps:
        issues.append("No reasoning steps provided")
        return 20.0, issues

    # Base score
    score = 100.0

    # Check for circular reasoning (simple heuristic: repeated identical steps)
    step_counts = {}
    for step in reasoning_steps:
        normalized = step.strip().lower()
        step_counts[normalized] = step_counts.get(normalized, 0) + 1

    repeated_steps = [step for step, count in step_counts.items() if count > 2]
    if repeated_steps:
        issues.append(
            f"Potential circular reasoning: {len(repeated_steps)} steps repeated >2 times"
        )
        score -= 20.0

    # Check for step progression (each step should reference or build on previous)
    # Simplified: check if steps are not all identical
    unique_steps = len(set(step.strip().lower() for step in reasoning_steps))
    if unique_steps < len(reasoning_steps) * 0.5:
        issues.append("Many duplicate steps - low reasoning diversity")
        score -= 15.0

    # Check for minimal step count (at least 2 steps expected)
    if len(reasoning_steps) < 2:
        issues.append("Too few reasoning steps")
        score -= 20.0

    # Check for reasonable step count (not too many)
    if len(reasoning_steps) > 100:
        issues.append("Excessive reasoning steps - possible inefficiency")
        score -= 10.0

    # Check for empty or trivial steps
    empty_steps = sum(1 for step in reasoning_steps if len(step.strip()) < 10)
    if empty_steps > 0:
        issues.append(f"{empty_steps} trivial or empty steps")
        score -= empty_steps * 2

    # Ensure score is in valid range
    score = max(0.0, min(100.0, score))

    if not issues:
        issues.append("Reasoning appears coherent")

    return score, issues


def calculate_calibration_error(
    confidence_values: List[float],
    correctness_values: List[bool],
    num_bins: int = 10,
) -> Dict[str, float]:
    """
    Calculate Expected Calibration Error (ECE) and related metrics.

    Perfect calibration: When confidence=0.8, should be correct 80% of time.

    Args:
        confidence_values: List of confidence scores (0-1)
        correctness_values: List of boolean correctness values
        num_bins: Number of bins for calibration calculation

    Returns:
        {
            "ece": float,  # Expected Calibration Error (0-1, lower is better)
            "mce": float,  # Maximum Calibration Error (0-1, lower is better)
            "brier_score": float,  # Brier score (0-1, lower is better)
            "sharpness": float,  # Confidence distribution spread (0-1)
            "bin_accuracies": List[float],
            "bin_confidences": List[float]
        }
    """
    if not confidence_values or not correctness_values:
        return {
            "ece": 0.5,
            "mce": 0.5,
            "brier_score": 0.5,
            "sharpness": 0.0,
            "bin_accuracies": [],
            "bin_confidences": [],
        }

    n = len(confidence_values)
    if n != len(correctness_values):
        # Truncate to shorter length
        n = min(n, len(correctness_values))
        confidence_values = confidence_values[:n]
        correctness_values = correctness_values[:n]

    # Create bins
    bins = [[] for _ in range(num_bins)]
    [i / num_bins for i in range(num_bins + 1)]

    # Assign predictions to bins
    for conf, correct in zip(confidence_values, correctness_values):
        bin_idx = min(int(conf * num_bins), num_bins - 1)
        bins[bin_idx].append((conf, correct))

    # Calculate bin statistics
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for bin_data in bins:
        if bin_data:
            bin_conf = sum(conf for conf, _ in bin_data) / len(bin_data)
            bin_acc = sum(1 for _, correct in bin_data if correct) / len(bin_data)
            bin_accuracies.append(bin_acc)
            bin_confidences.append(bin_conf)
            bin_counts.append(len(bin_data))
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append(0.0)
            bin_counts.append(0)

    # Calculate ECE (Expected Calibration Error)
    ece = 0.0
    for acc, conf, count in zip(bin_accuracies, bin_confidences, bin_counts):
        if count > 0:
            ece += (count / n) * abs(acc - conf)

    # Calculate MCE (Maximum Calibration Error)
    mce = 0.0
    for acc, conf, count in zip(bin_accuracies, bin_confidences, bin_counts):
        if count > 0:
            mce = max(mce, abs(acc - conf))

    # Calculate Brier score
    brier_score = (
        sum(
            (conf - (1.0 if correct else 0.0)) ** 2
            for conf, correct in zip(confidence_values, correctness_values)
        )
        / n
    )

    # Calculate sharpness (variance of confidence values)
    mean_conf = sum(confidence_values) / n
    variance = sum((conf - mean_conf) ** 2 for conf in confidence_values) / n
    sharpness = math.sqrt(variance)

    return {
        "ece": ece,
        "mce": mce,
        "brier_score": brier_score,
        "sharpness": sharpness,
        "bin_accuracies": bin_accuracies,
        "bin_confidences": bin_confidences,
    }


def score_partial_correctness(
    actual: Any,
    expected: Any,
    domain: str,
) -> Tuple[float, str]:
    """
    Domain-aware partial correctness scoring.

    Logic puzzles: Count correct entity assignments
    Dynamic responses: Semantic similarity + key fact inclusion
    NLP intent: Intent classification accuracy + confidence

    Args:
        actual: Actual output from KAI
        expected: Expected output (can be dict, str, list, etc.)
        domain: Domain name for domain-specific scoring

    Returns:
        (score 0-100, explanation)
    """
    if expected is None or (isinstance(expected, dict) and not expected):
        return 50.0, "No expected output for comparison"

    # Convert actual to string if needed
    if not isinstance(actual, str):
        actual = str(actual)

    # Domain-specific scoring
    if domain == "logic_puzzles":
        return _score_logic_puzzle_correctness(actual, expected)
    elif domain == "dynamic_responses":
        return _score_dynamic_response_correctness(actual, expected)
    elif domain == "nlp_intent_recognition":
        return _score_nlp_intent_correctness(actual, expected)
    elif domain == "combined_scenarios":
        return _score_combined_correctness(actual, expected)
    else:
        # Generic scoring
        return _score_generic_correctness(actual, expected)


# Helper functions


def _calculate_tree_depth(tree: Dict) -> int:
    """
    Calculate maximum depth of tree.

    Handles both legacy format (children) and ProofTree format (root_steps/subgoals).
    """
    if not tree:
        return 0

    # Handle ProofTree format: {"root_steps": [...], "query": "...", ...}
    if "root_steps" in tree:
        root_steps = tree["root_steps"]
        if not root_steps:
            return 1
        # Calculate max depth across all root steps
        max_depth = 0
        for step in root_steps:
            step_depth = _calculate_step_depth(step, 1)
            max_depth = max(max_depth, step_depth)
        return max_depth

    # Handle legacy format: {"children": [...]}
    if "children" not in tree:
        return 1

    if not tree["children"]:
        return 1

    max_child_depth = max(_calculate_tree_depth(child) for child in tree["children"])
    return 1 + max_child_depth


def _calculate_step_depth(step: Dict, current_depth: int = 1) -> int:
    """
    Calculate depth of a ProofStep (recursive through subgoals).

    Args:
        step: ProofStep dictionary
        current_depth: Current depth level

    Returns:
        Maximum depth from this step
    """
    if not isinstance(step, dict):
        return current_depth

    # Check for subgoals (ProofStep format)
    subgoals = step.get("subgoals", [])
    if not subgoals:
        return current_depth

    # Recursively calculate depth for each subgoal
    max_subgoal_depth = current_depth
    for subgoal in subgoals:
        subgoal_depth = _calculate_step_depth(subgoal, current_depth + 1)
        max_subgoal_depth = max(max_subgoal_depth, subgoal_depth)

    return max_subgoal_depth


def _extract_strategies_from_tree(tree: Dict) -> List[str]:
    """
    Extract strategy names from proof tree.

    Handles both legacy format (children) and ProofTree format (root_steps/subgoals).
    Extracts strategies from source_component and rule_name fields.
    """
    strategies = []

    def traverse_step(step):
        """Traverse a ProofStep node"""
        if not isinstance(step, dict):
            return

        # Check legacy format fields
        if "strategy" in step:
            strategies.append(step["strategy"])
        if "type" in step and isinstance(step.get("type"), str):
            type_val = step["type"].lower()
            if "engine" in type_val:
                strategies.append(step["type"])

        # Check ProofTree format: source_component field
        if "source_component" in step:
            component = step["source_component"]
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
            elif "knowledge" in component or "netzwerk" in component:
                strategies.append("knowledge_retrieval")

        # Check ProofTree format: rule_name field
        if "rule_name" in step and step["rule_name"]:
            rule = step["rule_name"].lower()
            if "sat" in rule:
                strategies.append("sat")
            elif "csp" in rule or "constraint" in rule:
                strategies.append("constraint_satisfaction")
            elif "unit propagation" in rule:
                strategies.append("sat")
            elif "logic" in rule:
                strategies.append("logic_engine")

        # Traverse subgoals (ProofStep format)
        if "subgoals" in step:
            for subgoal in step["subgoals"]:
                traverse_step(subgoal)

        # Traverse children (legacy format)
        if "children" in step:
            for child in step["children"]:
                traverse_step(child)

    # Handle ProofTree format: root_steps
    if "root_steps" in tree:
        for step in tree["root_steps"]:
            traverse_step(step)
    else:
        # Legacy format
        traverse_step(tree)

    # Remove duplicates while preserving order
    seen = set()
    unique_strategies = []
    for s in strategies:
        if s not in seen:
            seen.add(s)
            unique_strategies.append(s)

    return unique_strategies


def _evaluate_tree_coherence(tree: Dict) -> Tuple[float, List[str]]:
    """
    Evaluate logical coherence of proof tree structure.

    Handles both legacy format (children) and ProofTree format (root_steps/subgoals).
    """
    observations = []
    score = 30.0  # Start with full coherence score

    # Count nodes
    node_count = _count_tree_nodes(tree)
    observations.append(f"Proof tree has {node_count} nodes")

    if node_count < 2:
        observations.append("Proof tree has too few nodes")
        score -= 15.0

    # Check for leaf nodes (conclusions)
    leaf_count = _count_leaf_nodes(tree)
    if leaf_count == 0:
        observations.append("No leaf nodes (conclusions) found")
        score -= 10.0

    # Check for branching (multiple reasoning paths)
    branch_count = _count_branch_nodes(tree)
    if branch_count > 0:
        observations.append(f"Proof tree has {branch_count} branch points")
    else:
        observations.append("Proof tree is linear (no branching)")
        score -= 5.0

    return max(0.0, score), observations


def _count_tree_nodes(tree: Dict) -> int:
    """
    Count total nodes in tree.

    Handles both legacy format (children) and ProofTree format (root_steps/subgoals).
    """
    if not tree:
        return 0

    # Handle ProofTree format: root_steps
    if "root_steps" in tree:
        count = 0
        for step in tree["root_steps"]:
            count += _count_step_nodes(step)
        return count

    # Legacy format
    count = 1
    if "children" in tree:
        for child in tree["children"]:
            count += _count_tree_nodes(child)
    return count


def _count_step_nodes(step: Dict) -> int:
    """Count nodes in a ProofStep (including subgoals)."""
    if not isinstance(step, dict):
        return 0

    count = 1
    subgoals = step.get("subgoals", [])
    for subgoal in subgoals:
        count += _count_step_nodes(subgoal)
    return count


def _count_leaf_nodes(tree: Dict) -> int:
    """
    Count leaf nodes in tree.

    Handles both legacy format (children) and ProofTree format (root_steps/subgoals).
    """
    if not tree:
        return 0

    # Handle ProofTree format: root_steps
    if "root_steps" in tree:
        count = 0
        for step in tree["root_steps"]:
            count += _count_step_leaves(step)
        return count

    # Legacy format
    if "children" not in tree or not tree["children"]:
        return 1

    return sum(_count_leaf_nodes(child) for child in tree["children"])


def _count_step_leaves(step: Dict) -> int:
    """Count leaf nodes in a ProofStep (nodes with no subgoals)."""
    if not isinstance(step, dict):
        return 0

    subgoals = step.get("subgoals", [])
    if not subgoals:
        return 1

    count = 0
    for subgoal in subgoals:
        count += _count_step_leaves(subgoal)
    return count


def _count_branch_nodes(tree: Dict) -> int:
    """
    Count nodes with multiple children (branch points).

    Handles both legacy format (children) and ProofTree format (root_steps/subgoals).
    """
    if not tree:
        return 0

    # Handle ProofTree format: root_steps
    if "root_steps" in tree:
        count = 0
        # Multiple root steps counts as a branch point
        if len(tree["root_steps"]) > 1:
            count = 1
        for step in tree["root_steps"]:
            count += _count_step_branches(step)
        return count

    # Legacy format
    count = 0
    if "children" in tree and len(tree["children"]) > 1:
        count = 1

    if "children" in tree:
        for child in tree["children"]:
            count += _count_branch_nodes(child)

    return count


def _count_step_branches(step: Dict) -> int:
    """Count branch points in a ProofStep (nodes with multiple subgoals)."""
    if not isinstance(step, dict):
        return 0

    subgoals = step.get("subgoals", [])
    count = 0

    if len(subgoals) > 1:
        count = 1

    for subgoal in subgoals:
        count += _count_step_branches(subgoal)

    return count


def _score_logic_puzzle_correctness(actual: str, expected: Any) -> Tuple[float, str]:
    """Score logic puzzle correctness with partial credit"""
    if isinstance(expected, dict):
        # Expected is dict of entity: value pairs
        correct_count = 0
        total_count = len(expected)

        for entity, expected_value in expected.items():
            if isinstance(expected_value, str):
                # Check if actual contains "entity: value" or "entity ist value"
                if (
                    f"{entity}: {expected_value}".lower() in actual.lower()
                    or f"{entity} ist {expected_value}".lower() in actual.lower()
                    or f"{entity} = {expected_value}".lower() in actual.lower()
                ):
                    correct_count += 1

        if total_count > 0:
            score = (correct_count / total_count) * 100.0
            explanation = f"{correct_count}/{total_count} entities correct"
        else:
            score = 50.0
            explanation = "No entities to check"

        return score, explanation

    # Expected is simple string or other type
    return _score_generic_correctness(actual, expected)


def _score_dynamic_response_correctness(
    actual: str, expected: Any
) -> Tuple[float, str]:
    """Score dynamic response correctness (semantic similarity)"""
    if isinstance(expected, dict):
        # Check for key facts inclusion
        key_facts = expected.get("key_facts", [])
        if key_facts:
            included_count = sum(
                1 for fact in key_facts if fact.lower() in actual.lower()
            )
            score = (included_count / len(key_facts)) * 100.0
            explanation = f"{included_count}/{len(key_facts)} key facts included"
            return score, explanation

        # Check for expected answer
        answer = expected.get("answer", "")
        if answer and answer.lower() in actual.lower():
            return 100.0, "Expected answer found in response"

    # Generic semantic check (simplified: keyword matching)
    return _score_generic_correctness(actual, expected)


def _score_nlp_intent_correctness(actual: str, expected: Any) -> Tuple[float, str]:
    """Score NLP intent recognition correctness"""
    if isinstance(expected, dict):
        expected_intent = expected.get("intent", "")
        if expected_intent:
            # Check if intent is mentioned in response
            if expected_intent.lower() in actual.lower():
                return 100.0, f"Expected intent '{expected_intent}' found"
            else:
                return 0.0, f"Expected intent '{expected_intent}' not found"

    return _score_generic_correctness(actual, expected)


def _score_combined_correctness(actual: str, expected: Any) -> Tuple[float, str]:
    """Score combined scenario correctness (multiple criteria)"""
    if isinstance(expected, dict):
        # Combine multiple scoring methods
        logic_score = 0.0
        response_score = 0.0
        weights = []

        if "entities" in expected:
            logic_score, _ = _score_logic_puzzle_correctness(
                actual, expected["entities"]
            )
            weights.append(0.5)
        else:
            weights.append(0.0)

        if "key_facts" in expected or "answer" in expected:
            response_score, _ = _score_dynamic_response_correctness(actual, expected)
            weights.append(0.5)
        else:
            weights.append(0.0)

        if sum(weights) > 0:
            total_weight = sum(weights)
            final_score = (
                logic_score * weights[0] + response_score * weights[1]
            ) / total_weight
            explanation = f"Combined score (logic={logic_score:.1f}, response={response_score:.1f})"
            return final_score, explanation

    return _score_generic_correctness(actual, expected)


def _score_generic_correctness(actual: str, expected: Any) -> Tuple[float, str]:
    """Generic correctness scoring (string matching)"""
    if isinstance(expected, str):
        if expected.lower() in actual.lower():
            return 100.0, "Expected string found in response"
        elif any(word in actual.lower() for word in expected.lower().split()):
            return 50.0, "Partial match: some expected words found"
        else:
            return 0.0, "Expected string not found"
    elif isinstance(expected, dict):
        answer = expected.get("answer", "")
        if answer and isinstance(answer, str):
            if answer.lower() in actual.lower():
                return 100.0, "Expected answer found"
            else:
                return 0.0, "Expected answer not found"

    return 50.0, "Unable to compare actual vs expected"
