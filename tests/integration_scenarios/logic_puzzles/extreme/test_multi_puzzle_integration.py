"""
tests/integration_scenarios/logic_puzzles/extreme/test_multi_puzzle_integration.py

Extreme Logic Puzzle: Multi-Puzzle Integration with Cross-Domain Reasoning

Scenario: Two separate logic puzzles with shared constraints that require
cross-puzzle reasoning and constraint propagation. Tests KAI's ability to
maintain consistency across multiple problem domains simultaneously.

Expected Reasoning:
- Separate constraint solving for each puzzle
- Cross-domain constraint propagation
- Consistency checking across puzzle boundaries
- Multi-strategy coordination (SAT/CSP)

Success Criteria:
- Recognizes cross-puzzle constraints (weight: 30%)
- Solves both puzzles consistently (weight: 40%)
- Reasoning quality >= 20% (extreme threshold)
- Overall score >= 20%

NO Unicode characters - ASCII only (see CLAUDE.md encoding rules).
"""

import re
from typing import Dict

from tests.integration_scenarios.utils.scenario_base import ScenarioTestBase


class TestMultiPuzzleIntegration(ScenarioTestBase):
    """Test: Multi-puzzle integration with shared constraints"""

    # REQUIRED: Class attributes
    DIFFICULTY = "extreme"
    DOMAIN = "logic_puzzles"
    TIMEOUT_SECONDS = 7200  # 2 hours

    # REQUIRED: Scoring weights
    REASONING_QUALITY_WEIGHT = 0.5
    CONFIDENCE_CALIBRATION_WEIGHT = 0.2
    CORRECTNESS_WEIGHT = 0.3

    def test_multi_puzzle_integration(
        self,
        kai_worker_scenario_mode,
        progress_reporter,
        scenario_logger,
        confidence_tracker,
    ):
        """
        Test: Two separate puzzles with shared constraints requiring
        cross-puzzle reasoning.

        Puzzle 1: Entity assignment (3 people, 3 drinks)
        Puzzle 2: Numerical constraint (ages of same people)
        Shared Constraint: Person who drinks Coffee must be oldest

        This tests KAI's ability to:
        1. Solve each puzzle independently
        2. Recognize cross-puzzle constraints
        3. Propagate constraints between domains
        4. Maintain global consistency
        """

        # Setup progress tracking
        progress_reporter.total_steps = 5
        progress_reporter.start()

        # Define test input (German language, ASCII only)
        input_text = """
        Raetsel 1 (Getraenke-Zuordnung):
        Es gibt drei Personen: Anna, Bob, Clara.
        Es gibt drei Getraenke: Kaffee, Tee, Wasser.
        Jede Person trinkt genau ein Getraenk.

        Hinweise Raetsel 1:
        1. Anna trinkt nicht Tee.
        2. Bob trinkt nicht Kaffee.
        3. Clara trinkt nicht Wasser.

        Raetsel 2 (Alter):
        Die drei Personen sind 25, 30, und 35 Jahre alt.
        Jede Person hat ein unterschiedliches Alter.

        Hinweise Raetsel 2:
        1. Anna ist nicht 30 Jahre alt.
        2. Bob ist juenger als Clara.
        3. Die aelteste Person ist nicht 30.

        Verbindung zwischen beiden Raetseln:
        Die Person, die Kaffee trinkt, ist die aelteste Person.

        Frage: Wer trinkt was und wie alt ist jede Person?
        """

        # Define expected outputs
        # One valid solution: Anna=Wasser+25, Bob=Tee+30, Clara=Kaffee+35
        expected_outputs = {
            "multi_puzzle": True,
            "puzzle_count": 2,
            "cross_constraint": "coffee_oldest",
            "assignments": {
                "Anna": {"drink": "Wasser", "age": 25},
                "Bob": {"drink": "Tee", "age": 30},
                "Clara": {"drink": "Kaffee", "age": 35},
            },
        }

        # Execute using BASE CLASS method (NOT manual execution!)
        result = self.run_scenario(
            input_text=input_text,
            expected_outputs=expected_outputs,
            context={},
            kai_worker=kai_worker_scenario_mode,
            logger=scenario_logger,
            progress_reporter=progress_reporter,
            confidence_tracker=confidence_tracker,
        )

        # Assertions on ScenarioResult object
        # Extreme difficulty: target >= 20%
        assert (
            result.overall_score >= 20
        ), f"Overall score too low: {result.overall_score:.1f}% (expected >= 20%)"

        # Lower thresholds for extreme difficulty
        assert (
            result.reasoning_quality_score >= 15
        ), f"Reasoning quality too low: {result.reasoning_quality_score:.1f}%"

        # Domain-specific assertions
        # Check for multi-puzzle or constraint-related strategies
        strategy_keywords = [
            "constraint",
            "sat",
            "csp",
            "propagation",
            "multi",
            "cross",
        ]
        assert any(
            any(kw in s.lower() for kw in strategy_keywords)
            for s in result.strategies_used
        ), f"Expected multi-puzzle strategies, got: {result.strategies_used}"

        # ProofTree should be deep due to complexity
        assert (
            result.proof_tree_depth >= 5
        ), f"ProofTree depth {result.proof_tree_depth} too shallow for multi-puzzle"

        # Performance assertion (generous for extreme)
        assert (
            result.execution_time_ms < 7200000
        ), f"Exceeded timeout: {result.execution_time_ms}ms"

        # Check for cross-puzzle reasoning in response
        response_lower = result.kai_response.lower()
        cross_reasoning_indicators = [
            "kaffee" in response_lower and "aelteste" in response_lower,
            "verbindung" in response_lower,
            "beide raetsel" in response_lower,
        ]
        if not any(cross_reasoning_indicators):
            print("[WARNING] Response may not show cross-puzzle reasoning awareness")

        # Logging
        print(f"\n[INFO] Logs: {scenario_logger.save_logs()}")
        print(f"[INFO] Score: {result.overall_score:.1f}/100")
        print(
            f"[INFO] Breakdown: Reasoning={result.reasoning_quality_score:.1f}, "
            f"Confidence={result.confidence_calibration_score:.1f}, "
            f"Correctness={result.correctness_score:.1f}"
        )
        print(f"[INFO] ProofTree Depth: {result.proof_tree_depth}")
        print(f"[INFO] Strategies: {result.strategies_used}")

        if result.overall_score < 40:
            print("[EXPECTED] Low score is expected for extreme difficulty")
            print("[WEAKNESS] Issues identified:")
            for w in result.identified_weaknesses:
                print(f"  - {w}")

        # Final check - for extreme, any attempt >= 20% is success
        assert (
            result.passed or result.overall_score >= 20
        ), f"Test failed: {result.error or 'Score below extreme threshold'}"

    def score_correctness(
        self, actual: str, expected: Dict, allow_partial: bool = True
    ) -> float:
        """
        Custom correctness scoring for multi-puzzle scenario.

        Full credit requires:
        - Correct drink assignments (33%)
        - Correct age assignments (33%)
        - Recognition of cross-constraint (34%)

        Partial credit given for any correct elements.
        """
        if not actual or not expected:
            return 0.0

        actual_lower = actual.lower()
        score = 0.0

        # Check drink assignments (33 points)
        drink_score = 0.0
        if "assignments" in expected:
            assignments = expected["assignments"]
            for person, values in assignments.items():
                drink = values.get("drink", "").lower()
                # Negation-aware pattern
                pattern = rf"\b{person.lower()}\b(?!.*\b(?:nicht|kein)\b.*\b{drink}\b).*\b{drink}\b"
                if re.search(pattern, actual_lower):
                    drink_score += 33.0 / 3  # 11 points per person
        score += drink_score

        # Check age assignments (33 points)
        age_score = 0.0
        if "assignments" in expected:
            assignments = expected["assignments"]
            for person, values in assignments.items():
                age = str(values.get("age", ""))
                # Pattern: person + age number nearby
                pattern = rf"\b{person.lower()}\b.*\b{age}\b"
                if re.search(pattern, actual_lower):
                    age_score += 33.0 / 3  # 11 points per person

        score += age_score

        # Check cross-constraint recognition (34 points)
        cross_indicators = [
            "kaffee" in actual_lower and "aelteste" in actual_lower,
            "verbindung" in actual_lower,
            "beide" in actual_lower,
        ]
        if any(cross_indicators):
            score += 34.0

        # Normalize to 0-100
        return min(100.0, score)
