"""
tests/integration_scenarios/logic_puzzles/hard/test_transitive_relations.py

Hard-level logic puzzle: Multi-hop transitive reasoning (A > B > C > D > E)

Scenario:
Logic puzzle requiring transitive inference across multiple hops.
Relations like greater-than, taller-than, faster-than that are transitive.

Expected Reasoning:
- Graph traversal for transitive closure
- Multi-hop inference chains
- ProofTree should show step-by-step transitive reasoning
- Confidence should decay with inference distance (0.9 -> 0.7 over 4 hops)

Success Criteria (Gradual Scoring):
- Correctness: 30% (got right answer)
- Reasoning Quality: 50% (used graph traversal, correct transitive inference)
- Confidence Calibration: 20% (confidence reflects inference depth)
"""

import re
from typing import Dict, List

from tests.integration_scenarios.utils.scenario_base import ScenarioTestBase


class TestTransitiveRelations(ScenarioTestBase):
    """Test: Multi-hop transitive reasoning puzzle"""

    DIFFICULTY = "hard"
    DOMAIN = "logic_puzzles"
    TIMEOUT_SECONDS = 1800  # 30 minutes

    # Scoring weights for this scenario type
    REASONING_QUALITY_WEIGHT = 0.5
    CONFIDENCE_CALIBRATION_WEIGHT = 0.2
    CORRECTNESS_WEIGHT = 0.3

    def test_transitive_relations_puzzle(
        self,
        kai_worker_scenario_mode,
        progress_reporter,
        scenario_logger,
        confidence_tracker,
    ):
        """
        Test: Multi-hop transitive reasoning

        Use transitive inference to deduce indirect relationships.

        Expected: Korrekte transitive Schluesse ueber mehrere Schritte
        """

        # Setup progress reporter
        progress_reporter.total_steps = 5
        progress_reporter.start()

        # Puzzle input
        puzzle_text = """
Sechs Personen haben unterschiedliche Koerpergroessen.
Finde die Reihenfolge von der groessten zur kleinsten Person.

Gegeben sind folgende Vergleiche:
1. Anna ist groesser als Ben.
2. Ben ist groesser als Clara.
3. Clara ist groesser als David.
4. David ist groesser als Emma.
5. Emma ist groesser als Franz.
6. Franz ist groesser als Georg.

Zusaetzliche Hinweise:
7. Inge ist kleiner als Anna, aber groesser als Emma.
8. Julia ist groesser als Clara, aber kleiner als Ben.

Fragen:
a) Wer ist die groesste Person?
b) Wer ist die kleinste Person?
c) Ist Anna groesser als Emma? (Begruende mit transitiver Regel)
d) Ist Julia groesser als David?
e) Erstelle die vollstaendige Reihenfolge.
        """

        # Expected solution
        expected_solution = {
            "groesste": "Anna",
            "kleinste": "Georg",
            "anna_groesser_emma": "ja",
            "julia_groesser_david": "ja",
            "reihenfolge": "Anna > Ben > Julia > Clara > David > Inge > Emma > Franz > Georg",
        }

        # Execute scenario using base class
        result = self.run_scenario(
            input_text=puzzle_text,
            expected_outputs={"solution": expected_solution},
            context={},
            kai_worker=kai_worker_scenario_mode,
            logger=scenario_logger,
            progress_reporter=progress_reporter,
            confidence_tracker=confidence_tracker,
        )

        # Assertions
        assert (
            result.overall_score >= 40
        ), f"Overall score too low: {result.overall_score:.1f}% (expected >= 40%)"

        assert (
            result.correctness_score >= 30
        ), f"Expected at least 30% correctness, got {result.correctness_score:.1f}%"

        assert (
            result.reasoning_quality_score >= 35
        ), f"Reasoning quality too low: {result.reasoning_quality_score:.1f}%"

        # Check that appropriate strategies were used
        has_transitive_strategy = any(
            s in ["graph", "traversal", "transitive", "inference", "multi_hop"]
            for s in result.strategies_used
        )
        assert (
            has_transitive_strategy
        ), f"Expected graph/transitive strategy, got: {result.strategies_used}"

        # Verify reasoning depth is appropriate (multi-hop requires depth)
        assert (
            8 <= result.proof_tree_depth <= 20
        ), f"ProofTree depth {result.proof_tree_depth} outside expected range [8-20]"

        # Check performance
        assert (
            result.execution_time_ms < 100000
        ), f"Too slow: {result.execution_time_ms}ms (expected <100s)"

        # Log summary
        print(f"\n[INFO] Detailed logs saved to: {scenario_logger.save_logs()}")
        print(f"[INFO] Overall Score: {result.overall_score:.1f}/100")
        print(
            f"[INFO] Breakdown: "
            f"Reasoning={result.reasoning_quality_score:.1f}, "
            f"Confidence={result.confidence_calibration_score:.1f}, "
            f"Correctness={result.correctness_score:.1f}"
        )

        # Identify weaknesses if score is low
        if result.overall_score < 60:
            print("[WEAKNESS] Identified issues:")
            for weakness in result.identified_weaknesses:
                print(f"  - {weakness}")

        # Mark test as passed
        assert result.passed, f"Test failed: {result.error or 'Score below threshold'}"

    def score_correctness(
        self, actual: str, expected: Dict, allow_partial: bool = True
    ) -> float:
        """
        Score correctness based on how many questions were answered correctly.

        Args:
            actual: Actual KAI response text
            expected: Dict with "solution" key
            allow_partial: Whether to give partial credit

        Returns:
            Score 0-100
        """
        if not expected or "solution" not in expected:
            return 50.0

        solution = expected["solution"]
        correct_count = 0
        total = 5  # groesste, kleinste, 2 comparisons, order

        actual_lower = actual.lower()

        # Check groesste Person
        if "groesste" in solution:
            groesste = solution["groesste"].lower()
            if re.search(rf"\bgroesste\b.*\b{re.escape(groesste)}\b", actual_lower):
                correct_count += 1

        # Check kleinste Person
        if "kleinste" in solution:
            kleinste = solution["kleinste"].lower()
            if re.search(rf"\bkleinste\b.*\b{re.escape(kleinste)}\b", actual_lower):
                correct_count += 1

        # Check Anna > Emma
        if "anna_groesser_emma" in solution and solution["anna_groesser_emma"] == "ja":
            if re.search(r"\banna\b.*\bgroesser\b.*\bemma\b", actual_lower):
                correct_count += 1

        # Check Julia > David
        if (
            "julia_groesser_david" in solution
            and solution["julia_groesser_david"] == "ja"
        ):
            if re.search(r"\bjulia\b.*\bgroesser\b.*\bdavid\b", actual_lower):
                correct_count += 1

        # Check order (partial credit for correct subsequences)
        if "reihenfolge" in solution:
            # Check if some ordering is mentioned (simplified check)
            order_mentions = ["anna", "ben", "clara", "david", "emma", "franz", "georg"]
            mentions_found = sum(1 for name in order_mentions if name in actual_lower)
            if mentions_found >= 5:
                correct_count += 0.5  # Partial credit for attempting order

        # Calculate score with partial credit
        if allow_partial:
            score = (correct_count / total) * 100.0
        else:
            score = 100.0 if correct_count == total else 0.0

        return score

    def score_reasoning_quality(
        self, proof_tree: Dict, strategies_used: List[str], reasoning_steps: List[str]
    ) -> float:
        """
        Score reasoning quality based on strategy appropriateness and depth.

        Returns: 0-100 score
        """
        score = 0.0

        # Used appropriate strategy: +40%
        has_transitive = any(
            s in ["graph", "traversal", "transitive", "inference", "multi_hop"]
            for s in strategies_used
        )
        if has_transitive:
            score += 40

        # Appropriate ProofTree depth [8-20]: +30%
        depth = self._calculate_proof_tree_depth(proof_tree) if proof_tree else 0
        if 8 <= depth <= 20:
            score += 30
        elif 6 <= depth < 8:
            score += 20
        elif 20 < depth <= 25:
            score += 25

        # Multiple reasoning steps (multi-hop chains): +20%
        if len(reasoning_steps) >= 8:
            score += 20
        elif len(reasoning_steps) >= 5:
            score += 15
        elif len(reasoning_steps) >= 3:
            score += 10

        # Transitive inference visible: +10%
        transitive_keywords = ["transitiv", "folgt", "daher", "groesser.*als", "kette"]
        actual_text = " ".join(reasoning_steps).lower()
        if any(re.search(keyword, actual_text) for keyword in transitive_keywords):
            score += 10

        return min(score, 100.0)
