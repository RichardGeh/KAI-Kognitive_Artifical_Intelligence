"""
tests/integration_scenarios/logic_puzzles/medium/test_simple_ordering.py

Medium-level logic puzzle: 5 people in a line with ordering constraints

Scenario:
Five people stand in a line with ordering constraints (A left of B, etc.).
Includes negative constraints ("not at end", "not directly next to").
Tests transitive reasoning and position constraints.

Expected Reasoning:
- Transitive relation reasoning (A < B < C)
- Position constraint handling (not at end, not adjacent)
- ProofTree shows ordering deductions
- Confidence medium-high (>0.75)

Success Criteria (Gradual Scoring):
- Correctness: 30% (correct ordering)
- Reasoning Quality: 50% (transitive reasoning, constraint handling)
- Confidence Calibration: 20% (confidence matches correctness)
"""

import re
from typing import Dict, List, Tuple

from tests.integration_scenarios.utils.scenario_base import ScenarioTestBase


class TestSimpleOrdering(ScenarioTestBase):
    """Test: 5-person ordering puzzle with transitive constraints"""

    DIFFICULTY = "medium"
    DOMAIN = "logic_puzzles"
    TIMEOUT_SECONDS = 600

    REASONING_QUALITY_WEIGHT = 0.5
    CONFIDENCE_CALIBRATION_WEIGHT = 0.2
    CORRECTNESS_WEIGHT = 0.3

    def test_five_person_ordering(
        self,
        kai_worker_scenario_mode,
        progress_reporter,
        scenario_logger,
        confidence_tracker,
    ):
        """
        Test: 5-person line ordering with transitive constraints

        Five people in a line with left/right constraints and negations.
        Expected: A, B, E, C, D (or similar valid ordering)
        """

        # Setup
        progress_reporter.total_steps = 5
        progress_reporter.start()

        puzzle_text = """
Fuenf Personen stehen in einer Reihe: A, B, C, D, E.
1. A steht links von B.
2. B steht links von C.
3. D steht rechts von C.
4. E steht nicht am Ende.
5. B steht nicht direkt neben D.

In welcher Reihenfolge stehen sie?
        """

        # Valid solution: A, B, E, C, D
        expected_solution = ["A", "B", "E", "C", "D"]

        # Execute scenario using base class
        result = self.run_scenario(
            input_text=puzzle_text,
            expected_outputs={"ordering": expected_solution},
            context={},
            kai_worker=kai_worker_scenario_mode,
            logger=scenario_logger,
            progress_reporter=progress_reporter,
            confidence_tracker=confidence_tracker,
        )

        # Assertions
        assert (
            result.overall_score >= 50
        ), f"Overall score too low: {result.overall_score:.1f}% (expected >= 50%)"

        assert (
            result.correctness_score >= 40
        ), f"Correctness too low: {result.correctness_score:.1f}%"

        assert (
            result.reasoning_quality_score >= 40
        ), f"Reasoning quality too low: {result.reasoning_quality_score:.1f}%"

        # Check for ordering/constraint-solving strategy
        # SAT solver is a valid approach for constraint satisfaction problems
        has_ordering_strategy = any(
            s
            in [
                "constraint",
                "transitive",
                "ordering",
                "graph_traversal",
                "constraint_satisfaction",
                "sat",  # SAT solver is valid for constraint problems
                "sat_solver",
                "logic",
            ]
            for s in result.strategies_used
        )
        assert (
            has_ordering_strategy
        ), f"Expected ordering/constraint strategy, got: {result.strategies_used}"

        # Verify appropriate depth
        # SAT solver may produce deeper trees due to branching decisions
        assert (
            3 <= result.proof_tree_depth <= 15
        ), f"ProofTree depth {result.proof_tree_depth} outside expected range [3-15]"

        # Performance
        assert (
            result.execution_time_ms < 45000
        ), f"Too slow: {result.execution_time_ms}ms"

        # Log summary
        print(f"\n[INFO] Logs: {scenario_logger.save_logs()}")
        print(f"[INFO] Score: {result.overall_score:.1f}/100")
        print(
            f"[INFO] Breakdown: Reasoning={result.reasoning_quality_score:.1f}, "
            f"Confidence={result.confidence_calibration_score:.1f}, "
            f"Correctness={result.correctness_score:.1f}"
        )

        if result.overall_score < 70:
            print("[WEAKNESS] Issues:")
            for w in result.identified_weaknesses:
                print(f"  - {w}")

        # Mark test as passed
        assert result.passed, f"Test failed: {result.error or 'Score below threshold'}"

    def _extract_ordering_from_response(
        self, response: str, persons: List[str]
    ) -> List[str]:
        """
        Extract ordering of persons from response text robustly.

        Uses multiple strategies to find standalone person identifiers:
        1. Look for explicit list patterns (comma/space separated)
        2. Look for position statements ("Position 1: A")
        3. Fall back to isolated capital letters

        Args:
            response: KAI's response text
            persons: List of expected person identifiers (e.g., ["A", "B", "C", "D", "E"])

        Returns:
            List of persons in the order they appear in the response
        """
        # Strategy 1: Look for explicit ordering patterns
        # Pattern: "A, B, C, E, D" or "A B C E D" or "A - B - C - E - D"
        # This captures sequences of the expected letters
        person_set = set(persons)

        # Try to find a clear sequence (comma or space separated, with optional delimiters)
        # Look for patterns like "A, B, C, E, D" or "A B C E D"
        sequence_patterns = [
            # "A, B, C, E, D" - comma separated
            r"\b([A-E])\s*,\s*([A-E])\s*,\s*([A-E])\s*,\s*([A-E])\s*,\s*([A-E])\b",
            # "A B C E D" - space separated (word boundaries)
            r"\b([A-E])\s+([A-E])\s+([A-E])\s+([A-E])\s+([A-E])\b",
            # After colon - "ist: A, B, C, E, D" or "ist: A B C E D"
            r":\s*([A-E])[,\s]+([A-E])[,\s]+([A-E])[,\s]+([A-E])[,\s]+([A-E])",
        ]

        for pattern in sequence_patterns:
            match = re.search(pattern, response)
            if match:
                extracted = list(match.groups())
                # Verify all are unique and from expected set
                if len(set(extracted)) == 5 and all(p in person_set for p in extracted):
                    return extracted

        # Strategy 2: Find standalone single letters (word boundaries on both sides)
        # Pattern matches isolated A, B, C, D, E
        standalone_pattern = r"(?<![A-Za-z])([A-E])(?![A-Za-z])"
        matches = re.finditer(standalone_pattern, response)

        found = []
        seen = set()
        for match in matches:
            letter = match.group(1)
            if letter in person_set and letter not in seen:
                found.append((match.start(), letter))
                seen.add(letter)

        # Sort by position and extract letters
        found.sort()
        result = [p[1] for p in found]

        # Verify we have all persons
        if len(result) == len(persons) and set(result) == person_set:
            return result

        # Strategy 3: Fallback - use simple find but prefer later occurrences
        # (later occurrences are more likely to be the actual answer, not preamble text)
        found_order = []
        for person in persons:
            # Find ALL occurrences
            positions = [m.start() for m in re.finditer(re.escape(person), response)]
            if positions:
                # Use the last occurrence that appears after common preamble words
                # or the first occurrence in a clear list context
                found_order.append((positions[-1], person))

        found_order.sort()
        return [p[1] for p in found_order]

    def _verify_ordering_constraints(self, ordering: List[str]) -> Tuple[bool, str]:
        """
        Verify if an ordering satisfies all puzzle constraints.

        Constraints for this puzzle:
        1. A steht links von B (A < B)
        2. B steht links von C (B < C)
        3. D steht rechts von C (D > C, i.e., C < D)
        4. E steht nicht am Ende (E not at position 0 or 4)
        5. B steht nicht direkt neben D (B and D not adjacent)

        Args:
            ordering: List of persons in order (left to right)

        Returns:
            Tuple of (is_valid: bool, error_message: str)
            If valid, error_message is empty string.
        """
        if len(ordering) != 5:
            return (False, f"Expected 5 persons, got {len(ordering)}")

        # Build position map
        try:
            pos = {person: idx for idx, person in enumerate(ordering)}
        except Exception as e:
            return (False, f"Invalid ordering format: {e}")

        # Check all required persons are present
        required = {"A", "B", "C", "D", "E"}
        present = set(ordering)
        if present != required:
            missing = required - present
            extra = present - required
            msg = []
            if missing:
                msg.append(f"missing: {missing}")
            if extra:
                msg.append(f"unexpected: {extra}")
            return (False, f"Person mismatch - {', '.join(msg)}")

        violations = []

        # Constraint 1: A left of B (A < B)
        if pos["A"] >= pos["B"]:
            violations.append(
                f"C1: A (pos {pos['A']}) must be left of B (pos {pos['B']})"
            )

        # Constraint 2: B left of C (B < C)
        if pos["B"] >= pos["C"]:
            violations.append(
                f"C2: B (pos {pos['B']}) must be left of C (pos {pos['C']})"
            )

        # Constraint 3: D right of C (C < D)
        if pos["C"] >= pos["D"]:
            violations.append(
                f"C3: D (pos {pos['D']}) must be right of C (pos {pos['C']})"
            )

        # Constraint 4: E not at end (not at position 0 or 4)
        if pos["E"] == 0 or pos["E"] == 4:
            violations.append(f"C4: E (pos {pos['E']}) must not be at end (0 or 4)")

        # Constraint 5: B not directly next to D
        if abs(pos["B"] - pos["D"]) == 1:
            violations.append(
                f"C5: B (pos {pos['B']}) must not be adjacent to D (pos {pos['D']})"
            )

        if violations:
            return (False, "; ".join(violations))

        return (True, "")

    def score_correctness(
        self, actual: str, expected: Dict, allow_partial: bool = True
    ) -> float:
        """
        Score correctness based on ordering accuracy.

        Uses constraint validation: ANY valid solution gets 100%.
        Falls back to partial-credit scoring if constraints not satisfied.

        Args:
            actual: Actual KAI response text
            expected: Dict with "ordering" key containing expected person list
            allow_partial: Whether to give partial credit

        Returns:
            Score 0-100
        """
        if not expected or "ordering" not in expected:
            return 50.0

        expected_order = expected["ordering"]

        # Extract ordering from response using robust extraction
        actual_order = self._extract_ordering_from_response(actual, expected_order)

        if len(actual_order) < len(expected_order):
            # Missing people - give minimal partial credit based on what's found
            if allow_partial and len(actual_order) > 0:
                return (len(actual_order) / len(expected_order)) * 20.0
            return 0.0

        # PRIMARY CHECK: Constraint validation
        # If the found ordering satisfies all constraints, it's a valid solution
        is_valid, error_msg = self._verify_ordering_constraints(actual_order)

        if is_valid:
            # Valid solution found - full score (100%)
            # Any solution satisfying all constraints gets full credit
            return 100.0

        # FALLBACK: Partial credit scoring when constraints not satisfied
        # Score based on correct positions (40%)
        correct_positions = 0
        for i, person in enumerate(expected_order):
            if i < len(actual_order) and actual_order[i] == person:
                correct_positions += 1

        # Partial credit for correct relative ordering (30%)
        correct_pairs = 0
        total_pairs = len(expected_order) - 1

        for i in range(len(expected_order) - 1):
            person1 = expected_order[i]
            person2 = expected_order[i + 1]
            if person1 in actual_order and person2 in actual_order:
                idx1 = actual_order.index(person1)
                idx2 = actual_order.index(person2)
                if idx1 < idx2:  # Correct relative order
                    correct_pairs += 1

        # Partial credit for satisfied constraints (30%)
        constraints_satisfied = 0
        total_constraints = 5

        pos = {person: idx for idx, person in enumerate(actual_order)}

        # Check each constraint individually
        if pos.get("A", 99) < pos.get("B", -1):
            constraints_satisfied += 1
        if pos.get("B", 99) < pos.get("C", -1):
            constraints_satisfied += 1
        if pos.get("C", 99) < pos.get("D", -1):
            constraints_satisfied += 1
        e_pos = pos.get("E", -1)
        if e_pos not in (0, 4) and e_pos != -1:
            constraints_satisfied += 1
        b_pos = pos.get("B", -1)
        d_pos = pos.get("D", -1)
        if b_pos != -1 and d_pos != -1 and abs(b_pos - d_pos) != 1:
            constraints_satisfied += 1

        # Combine scores: positions (40%) + relative order (30%) + constraints (30%)
        position_score = (correct_positions / len(expected_order)) * 40.0
        relative_score = (
            (correct_pairs / total_pairs) * 30.0 if total_pairs > 0 else 0.0
        )
        constraint_score = (constraints_satisfied / total_constraints) * 30.0

        return position_score + relative_score + constraint_score

    def score_reasoning_quality(
        self, proof_tree: Dict, strategies_used: List[str], reasoning_steps: List[str]
    ) -> float:
        """
        Score reasoning quality for ordering puzzle.

        Recognizes multiple valid strategies:
        - SAT solver (constraint solving via Boolean satisfiability)
        - Transitive reasoning (graph-based ordering)
        - Constraint satisfaction (CSP approach)

        Returns: 0-100
        """
        score = 0.0

        # Acceptable strategies for ordering puzzles
        # SAT is a valid constraint-solving approach for ordering problems
        ordering_strategies = [
            "transitive",
            "ordering",
            "constraint",
            "constraint_satisfaction",
            "sat",  # SAT solver is valid for constraint problems
            "sat_solver",
            "logic",  # Logic-based reasoning
        ]

        # Used appropriate strategy: +40%
        if any(s in ordering_strategies for s in strategies_used):
            score += 40

        # Appropriate depth [5-12]: +30%
        # SAT solver may produce deeper trees due to branching
        depth = self._calculate_proof_tree_depth(proof_tree) if proof_tree else 0
        if 5 <= depth <= 12:
            score += 30
        elif depth < 5:
            score += 10
        elif 12 < depth <= 15:
            score += 20

        # Evidence of systematic reasoning: +20%
        # SAT solver uses unit propagation, which is systematic constraint solving
        if "transitive" in strategies_used or "graph_traversal" in strategies_used:
            score += 20
        elif any(s.lower() in ordering_strategies for s in strategies_used):
            score += 15  # SAT and other constraint approaches get credit
        elif len(reasoning_steps) >= 5:
            score += 10

        # Multiple reasoning steps: +10%
        if len(reasoning_steps) >= 3:
            score += 10

        return min(score, 100.0)
