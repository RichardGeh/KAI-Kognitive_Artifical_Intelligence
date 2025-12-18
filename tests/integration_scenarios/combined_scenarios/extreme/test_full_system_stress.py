"""
tests/integration_scenarios/combined_scenarios/extreme/test_full_system_stress.py

Extreme Combined Scenario: Full System Stress Test

Scenario: Stress test with 100+ facts loaded, followed by complex multi-engine
query requiring constraint satisfaction, graph traversal, arithmetic reasoning,
and production system response generation. Tests KAI's scalability and ability
to coordinate multiple subsystems under heavy load.

Expected Reasoning:
- Graph query (large scale)
- Constraint reasoning
- Arithmetic reasoning
- Multi-strategy coordination
- Production system response generation

Success Criteria:
- Doesn't crash (weight: 50%)
- Handles scale gracefully (weight: 30%)
- Reasoning quality >= 20% (extreme threshold)
- Overall score >= 20%

NO Unicode characters - ASCII only (see CLAUDE.md encoding rules).
"""

from typing import Dict

from tests.integration_scenarios.utils.scenario_base import ScenarioTestBase


class TestFullSystemStress(ScenarioTestBase):
    """Test: Full system stress with 100+ facts and complex query"""

    # REQUIRED: Class attributes
    DIFFICULTY = "extreme"
    DOMAIN = "combined_scenarios"
    TIMEOUT_SECONDS = 7200  # 2 hours

    # REQUIRED: Scoring weights
    REASONING_QUALITY_WEIGHT = 0.5
    CONFIDENCE_CALIBRATION_WEIGHT = 0.2
    CORRECTNESS_WEIGHT = 0.3

    def test_full_system_stress(
        self,
        kai_worker_scenario_mode,
        progress_reporter,
        scenario_logger,
        confidence_tracker,
    ):
        """
        Test: Stress test with large knowledge base and complex query.

        Stress test components:
        1. Load 100+ facts into knowledge graph
        2. Query requiring multiple reasoning engines
        3. Constraint satisfaction + arithmetic + graph query
        4. Production system response generation

        This tests KAI's ability to:
        1. Handle large-scale knowledge base
        2. Coordinate multiple reasoning engines
        3. Maintain performance under load
        4. Generate coherent response despite complexity
        """

        # Setup progress tracking
        progress_reporter.total_steps = 5
        progress_reporter.start()

        # Build large context with 100+ facts
        context_facts = []

        # Animals (30 facts)
        animals = [
            "Hund",
            "Katze",
            "Vogel",
            "Fisch",
            "Pferd",
            "Kuh",
            "Schwein",
            "Schaf",
            "Ziege",
            "Huhn",
        ]
        for animal in animals:
            context_facts.append(f"Ein {animal} ist ein Tier.")
            context_facts.append(f"Ein {animal} hat vier Beine.")
            context_facts.append(f"Ein {animal} kann atmen.")

        # Numbers (30 facts)
        for i in range(1, 31):
            context_facts.append(f"Die Zahl {i} ist kleiner als {i+1}.")

        # Locations (40 facts)
        cities = [
            "Berlin",
            "Muenchen",
            "Hamburg",
            "Koeln",
            "Frankfurt",
            "Stuttgart",
            "Duesseldorf",
            "Dortmund",
            "Essen",
            "Leipzig",
        ]
        for city in cities:
            context_facts.append(f"{city} ist eine Stadt.")
            context_facts.append(f"{city} liegt in Deutschland.")
            context_facts.append(f"{city} hat eine Bevoelkerung.")
            context_facts.append(f"{city} hat Strassen.")

        # Total: 120 facts
        context = {"preload_facts": context_facts}

        # Define test input (German language, ASCII only)
        input_text = """
        Gegeben sind die vorher gelernten Fakten.

        Jetzt beantworte folgende komplexe Frage:

        Wie viele Tiere mit vier Beinen gibt es in der Wissensbasis?
        Welche Staedte liegen in Deutschland?
        Was ist die Summe der Zahlen von 1 bis 10?

        Fasse alles in einer strukturierten Antwort zusammen.
        """

        # Define expected outputs
        expected_outputs = {
            "multi_engine": True,
            "expected_components": {
                "animal_count": 10,  # 10 animals mentioned
                "city_list": cities,
                "sum_1_to_10": 55,
            },
        }

        # Execute using BASE CLASS method
        result = self.run_scenario(
            input_text=input_text,
            expected_outputs=expected_outputs,
            context=context,
            kai_worker=kai_worker_scenario_mode,
            logger=scenario_logger,
            progress_reporter=progress_reporter,
            confidence_tracker=confidence_tracker,
        )

        # Assertions on ScenarioResult object
        # Extreme difficulty: target >= 20%
        # Most important: didn't crash!
        assert (
            result.overall_score >= 20
        ), f"Overall score too low: {result.overall_score:.1f}% (expected >= 20%)"

        # Lower thresholds for extreme difficulty
        assert (
            result.reasoning_quality_score >= 15
        ), f"Reasoning quality too low: {result.reasoning_quality_score:.1f}%"

        # Domain-specific assertions
        # Check for multi-strategy usage
        strategy_keywords = [
            "graph",
            "query",
            "arithmetic",
            "constraint",
            "production",
        ]
        multi_strategy = len(result.strategies_used) >= 2

        if multi_strategy:
            print(f"[SUCCESS] Multiple strategies used: {result.strategies_used}")

        # Performance assertion - should complete despite scale
        assert (
            result.execution_time_ms < 7200000
        ), f"Exceeded timeout: {result.execution_time_ms}ms"

        print(f"[INFO] Performance: {result.execution_time_ms}ms for 120+ facts")

        # Check Neo4j queries (should be numerous for large dataset)
        if result.neo4j_query_count > 0:
            print(f"[INFO] Neo4j queries executed: {result.neo4j_query_count}")

        # Check cache hit rate (should benefit from caching)
        if result.cache_hit_rate > 0:
            print(f"[INFO] Cache hit rate: {result.cache_hit_rate:.1%}")

        # Check memory usage
        if result.memory_peak_mb > 0:
            print(f"[INFO] Peak memory: {result.memory_peak_mb:.1f} MB")

        # Response should exist (not empty)
        assert (
            len(result.kai_response) > 0
        ), "Response is empty (system may have crashed)"

        # Logging
        print(f"\n[INFO] Logs: {scenario_logger.save_logs()}")
        print(f"[INFO] Score: {result.overall_score:.1f}/100")
        print(
            f"[INFO] Breakdown: Reasoning={result.reasoning_quality_score:.1f}, "
            f"Confidence={result.confidence_calibration_score:.1f}, "
            f"Correctness={result.correctness_score:.1f}"
        )
        print(f"[INFO] Strategies: {result.strategies_used}")

        if result.overall_score < 40:
            print("[EXPECTED] Low score is expected for extreme difficulty")
            print("[SUCCESS] System handled stress test without crashing")
            print("[WEAKNESS] Issues identified:")
            for w in result.identified_weaknesses:
                print(f"  - {w}")

        # Final check - completing is success for stress test
        assert (
            result.passed or result.overall_score >= 20
        ), f"Test failed: {result.error or 'Score below extreme threshold'}"

    def score_correctness(
        self, actual: str, expected: Dict, allow_partial: bool = True
    ) -> float:
        """
        Custom correctness scoring for stress test.

        Full credit requires:
        - Animal count mentioned (30%)
        - City list mentioned (30%)
        - Arithmetic result mentioned (40%)
        """
        if not actual or not expected:
            return 0.0

        actual_lower = actual.lower()
        score = 0.0

        # Check animal count (30 points)
        if "10" in actual or "zehn" in actual_lower:
            if (
                "tier" in actual_lower
                or "hund" in actual_lower
                or "katze" in actual_lower
            ):
                score += 30.0

        # Check city list (30 points)
        cities = expected.get("expected_components", {}).get("city_list", [])
        city_count = 0
        for city in cities:
            if city.lower() in actual_lower:
                city_count += 1

        if city_count >= 7:
            score += 30.0
        elif city_count >= 5:
            score += 20.0
        elif city_count >= 3:
            score += 10.0

        # Check arithmetic result (40 points)
        if "55" in actual or "fuenf" in actual_lower and "fuenfzig" in actual_lower:
            if "summe" in actual_lower or "1 bis 10" in actual:
                score += 40.0

        return min(100.0, score)
