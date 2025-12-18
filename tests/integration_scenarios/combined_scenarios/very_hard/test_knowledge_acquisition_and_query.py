"""
tests/integration_scenarios/combined_scenarios/very_hard/test_knowledge_acquisition_and_query.py

Learn complex knowledge graph, then answer with 3-hop transitive reasoning.
Tests knowledge acquisition, graph building, and multi-hop query capabilities.

Expected Reasoning: learning, graph_building, multi_hop_query, transitive_inference
Success Criteria:
- Reasoning Quality >= 30% (graph construction + multi-hop)
- Confidence Calibration >= 30%
- Correctness >= 30%
- Overall >= 30%
"""

import re
from typing import Dict

from tests.integration_scenarios.utils.scenario_base import ScenarioTestBase


class TestKnowledgeAcquisitionAndQuery(ScenarioTestBase):
    """Test: Knowledge Acquisition + Multi-Hop Query - Very Hard Combined Scenario"""

    # REQUIRED: Class attributes
    DIFFICULTY = "very_hard"
    DOMAIN = "combined_scenarios"
    TIMEOUT_SECONDS = 3600  # 1 hour

    # REQUIRED: Scoring weights
    REASONING_QUALITY_WEIGHT = 0.5
    CONFIDENCE_CALIBRATION_WEIGHT = 0.2
    CORRECTNESS_WEIGHT = 0.3

    def test_knowledge_acquisition_and_query(
        self,
        kai_worker_scenario_mode,
        progress_reporter,
        scenario_logger,
        confidence_tracker,
    ):
        """
        Test: Learn complex knowledge, answer with multi-hop reasoning.

        Phase 1: Learn 15+ facts about taxonomy and properties
        Phase 2: Answer questions requiring 3-hop transitive inference

        Example: Dog -> Mammal -> Animal -> Living Being
        """

        # Setup progress tracking
        progress_reporter.total_steps = 5
        progress_reporter.start()

        # Define test input (German language, ASCII only)
        input_text = """
        Lerne folgende Fakten:

        Taxonomie (IS_A Hierarchie):
        1. Ein Hund ist ein Saeugetier.
        2. Eine Katze ist ein Saeugetier.
        3. Ein Saeugetier ist ein Tier.
        4. Ein Vogel ist ein Tier.
        5. Ein Tier ist ein Lebewesen.
        6. Eine Pflanze ist ein Lebewesen.
        7. Ein Baum ist eine Pflanze.
        8. Eine Blume ist eine Pflanze.

        Eigenschaften (HAS_PROPERTY):
        9. Ein Saeugetier hat Fell.
        10. Ein Saeugetier kann Milch geben.
        11. Ein Vogel hat Federn.
        12. Ein Vogel kann fliegen.
        13. Ein Lebewesen kann wachsen.
        14. Ein Lebewesen braucht Nahrung.
        15. Eine Pflanze kann Photosynthese betreiben.

        Faehigkeiten (CAPABLE_OF):
        16. Ein Hund kann bellen.
        17. Eine Katze kann miauen.
        18. Ein Baum kann Sauerstoff produzieren.

        Nach dem Lernen beantworte folgende Fragen:

        Frage 1: Kann ein Hund wachsen?
        (Erfordert: Hund -> Saeugetier -> Tier -> Lebewesen -> kann wachsen)

        Frage 2: Hat ein Hund Fell?
        (Erfordert: Hund -> Saeugetier -> hat Fell)

        Frage 3: Braucht ein Baum Nahrung?
        (Erfordert: Baum -> Pflanze -> Lebewesen -> braucht Nahrung)

        Frage 4: Kann eine Katze Milch geben?
        (Erfordert: Katze -> Saeugetier -> kann Milch geben)

        Frage 5: Welche Eigenschaften hat ein Hund?
        (Erfordert: Sammeln aller geerbten Eigenschaften)

        Beantworte alle Fragen mit Begruendung.
        Zeige die Ableitungskette.
        """

        # Define expected outputs
        expected_outputs = {
            "q1_answer": "Ja",  # Hund kann wachsen
            "q1_hops": 4,  # 4-hop chain
            "q2_answer": "Ja",  # Hund hat Fell
            "q2_hops": 2,  # 2-hop chain
            "q3_answer": "Ja",  # Baum braucht Nahrung
            "q3_hops": 3,  # 3-hop chain
            "q4_answer": "Ja",  # Katze kann Milch geben
            "q4_hops": 2,  # 2-hop chain
            "q5_properties": ["Fell", "Milch", "wachsen", "Nahrung", "bellen"],
        }

        # Execute using BASE CLASS method
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
        assert (
            result.overall_score >= 30
        ), f"Overall score too low: {result.overall_score:.1f}% (expected >= 30%)"

        assert (
            result.correctness_score >= 25
        ), f"Correctness too low: {result.correctness_score:.1f}%"

        assert (
            result.reasoning_quality_score >= 30
        ), f"Reasoning quality too low: {result.reasoning_quality_score:.1f}%"

        # Domain-specific assertions - check for learning + query strategies
        knowledge_strategies = [
            "learning",
            "graph",
            "multi_hop",
            "transitive",
            "inference",
            "query",
            "knowledge",
        ]
        found_strategy = any(
            any(ks in s.lower() for ks in knowledge_strategies)
            for s in result.strategies_used
        )

        assert (
            found_strategy
        ), f"Expected learning/graph/multi-hop strategy, got: {result.strategies_used}"

        # Check proof tree depth (multi-hop queries need depth)
        assert (
            6 <= result.proof_tree_depth <= 25
        ), f"ProofTree depth {result.proof_tree_depth} outside expected range 6-25"

        # Performance assertion
        assert (
            result.execution_time_ms < 3600000
        ), f"Too slow: {result.execution_time_ms}ms (expected <1 hour)"

        # Logging
        log_file = scenario_logger.save_logs()
        print(f"\n[INFO] Detailed logs saved to: {log_file}")
        print(f"[INFO] Overall Score: {result.overall_score:.1f}/100")
        print(
            f"[INFO] Breakdown: Reasoning={result.reasoning_quality_score:.1f}, "
            f"Confidence={result.confidence_calibration_score:.1f}, "
            f"Correctness={result.correctness_score:.1f}"
        )

        if result.overall_score < 50:
            print("[WEAKNESS] Identified issues:")
            for weakness in result.identified_weaknesses:
                print(f"  - {weakness}")

        if result.improvement_suggestions:
            print("[SUGGESTION] Improvements:")
            for suggestion in result.improvement_suggestions:
                print(f"  - {suggestion}")

        # Final check
        assert result.passed, f"Test failed: {result.error or 'Score below threshold'}"

    def score_correctness(
        self, actual: str, expected: Dict, allow_partial: bool = True
    ) -> float:
        """
        Custom correctness scoring for knowledge acquisition + query.

        Partial credit for:
        - Q1: Hund kann wachsen (20 points)
        - Q2: Hund hat Fell (20 points)
        - Q3: Baum braucht Nahrung (20 points)
        - Q4: Katze kann Milch geben (20 points)
        - Q5: Multiple properties listed (20 points)
        """
        score = 0.0

        # Question 1: Hund kann wachsen
        q1_patterns = [
            r"\bHund\b.*\bwachsen\b",
            r"\bwachsen\b.*\bHund\b",
            r"Frage 1.*\bJa\b",
        ]
        if any(re.search(pattern, actual, re.IGNORECASE) for pattern in q1_patterns):
            # Check no negation
            if not re.search(
                r"\bnicht.*\bwachsen\b|kein.*\bwachsen\b", actual, re.IGNORECASE
            ):
                score += 20.0

        # Question 2: Hund hat Fell
        q2_patterns = [
            r"\bHund\b.*\bFell\b",
            r"\bFell\b.*\bHund\b",
            r"Frage 2.*\bJa\b",
        ]
        if any(re.search(pattern, actual, re.IGNORECASE) for pattern in q2_patterns):
            if not re.search(
                r"\bnicht.*\bFell\b|kein.*\bFell\b", actual, re.IGNORECASE
            ):
                score += 20.0

        # Question 3: Baum braucht Nahrung
        q3_patterns = [
            r"\bBaum\b.*\bNahrung\b",
            r"\bNahrung\b.*\bBaum\b",
            r"Frage 3.*\bJa\b",
        ]
        if any(re.search(pattern, actual, re.IGNORECASE) for pattern in q3_patterns):
            if not re.search(
                r"\bnicht.*\bNahrung\b|kein.*\bNahrung\b", actual, re.IGNORECASE
            ):
                score += 20.0

        # Question 4: Katze kann Milch geben
        q4_patterns = [
            r"\bKatze\b.*\bMilch\b",
            r"\bMilch\b.*\bKatze\b",
            r"Frage 4.*\bJa\b",
        ]
        if any(re.search(pattern, actual, re.IGNORECASE) for pattern in q4_patterns):
            if not re.search(
                r"\bnicht.*\bMilch\b|kein.*\bMilch\b", actual, re.IGNORECASE
            ):
                score += 20.0

        # Question 5: Multiple properties
        properties = expected.get("q5_properties", [])
        properties_found = sum(
            1 for prop in properties if prop.lower() in actual.lower()
        )

        if properties_found >= 4:
            score += 20.0
        elif properties_found >= 3:
            score += 15.0
        elif properties_found >= 2:
            score += 10.0

        return score
