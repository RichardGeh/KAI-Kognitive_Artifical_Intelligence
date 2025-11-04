# tests/test_hybrid_reasoning_phase2.py
"""
Tests für Hybrid Reasoning Phase 2 Features

Testet:
- Additional Aggregation Methods (weighted_avg, max, dempster_shafer)
- YAML Configuration Loading
- Result Caching
- Parallel Execution
- Performance Optimizations
"""

import pytest
from unittest.mock import Mock
from pathlib import Path
import tempfile
import yaml

from kai_reasoning_orchestrator import (
    ReasoningOrchestrator,
    ReasoningStrategy,
    ReasoningResult,
)


class TestAggregationMethods:
    """Tests für verschiedene Aggregation Methods"""

    @pytest.fixture
    def orchestrator(self):
        """Basic orchestrator instance"""
        return ReasoningOrchestrator(
            netzwerk=Mock(),
            logic_engine=Mock(),
            graph_traversal=Mock(),
            working_memory=Mock(),
            signals=Mock(),
        )

    @pytest.fixture
    def sample_results(self):
        """Sample reasoning results for testing"""
        return [
            ReasoningResult(
                strategy=ReasoningStrategy.GRAPH_TRAVERSAL,
                success=True,
                confidence=0.8,
                inferred_facts={"IS_A": ["tier"]},
                proof_trace="Graph path found",
            ),
            ReasoningResult(
                strategy=ReasoningStrategy.LOGIC_ENGINE,
                success=True,
                confidence=0.7,
                inferred_facts={"IS_A": ["lebewesen"]},
                proof_trace="Rule applied",
            ),
            ReasoningResult(
                strategy=ReasoningStrategy.PROBABILISTIC,
                success=True,
                confidence=0.6,
                inferred_facts={},
                proof_trace="Probabilistic inference",
            ),
        ]

    def test_weighted_average_aggregation(self, orchestrator, sample_results):
        """Test Weighted Average Aggregation"""
        orchestrator.aggregation_method = "weighted_avg"

        # Custom weights for testing
        orchestrator.strategy_weights = {
            ReasoningStrategy.GRAPH_TRAVERSAL: 0.5,
            ReasoningStrategy.LOGIC_ENGINE: 0.3,
            ReasoningStrategy.PROBABILISTIC: 0.2,
        }

        aggregated = orchestrator._create_aggregated_result(sample_results)

        # Expected: (0.5*0.8 + 0.3*0.7 + 0.2*0.6) / (0.5+0.3+0.2)
        # = (0.4 + 0.21 + 0.12) / 1.0 = 0.73
        expected_confidence = 0.73
        assert aggregated.combined_confidence == pytest.approx(
            expected_confidence, rel=0.01
        )
        assert len(aggregated.strategies_used) == 3

    def test_maximum_aggregation(self, orchestrator, sample_results):
        """Test Maximum Confidence Aggregation"""
        orchestrator.aggregation_method = "max"

        aggregated = orchestrator._create_aggregated_result(sample_results)

        # Should take maximum confidence (0.8)
        assert aggregated.combined_confidence == 0.8
        assert len(aggregated.strategies_used) == 3

    def test_dempster_shafer_aggregation(self, orchestrator, sample_results):
        """Test Dempster-Shafer Aggregation"""
        orchestrator.aggregation_method = "dempster_shafer"

        aggregated = orchestrator._create_aggregated_result(sample_results)

        # Dempster-Shafer should combine beliefs accounting for conflict
        # Result should be between max and noisy-or
        assert 0.8 <= aggregated.combined_confidence <= 0.98
        assert len(aggregated.strategies_used) == 3

    def test_noisy_or_aggregation_default(self, orchestrator, sample_results):
        """Test Noisy-OR Aggregation (default)"""
        # Default is noisy_or
        assert orchestrator.aggregation_method == "noisy_or"

        aggregated = orchestrator._create_aggregated_result(sample_results)

        # Noisy-OR: 1 - (1-0.8)*(1-0.7)*(1-0.6)
        # = 1 - 0.2*0.3*0.4 = 1 - 0.024 = 0.976
        expected_confidence = 0.976
        assert aggregated.combined_confidence == pytest.approx(
            expected_confidence, rel=0.01
        )

    def test_unknown_aggregation_method_fallback(self, orchestrator, sample_results):
        """Test fallback to noisy_or for unknown aggregation method"""
        orchestrator.aggregation_method = "unknown_method"

        aggregated = orchestrator._create_aggregated_result(sample_results)

        # Should fall back to noisy_or
        expected_confidence = 0.976
        assert aggregated.combined_confidence == pytest.approx(
            expected_confidence, rel=0.01
        )

    def test_single_result_aggregation_all_methods(self, orchestrator):
        """Test aggregation with single result for all methods"""
        single_result = [
            ReasoningResult(
                strategy=ReasoningStrategy.DIRECT_FACT,
                success=True,
                confidence=0.95,
                inferred_facts={"IS_A": ["tier"]},
            )
        ]

        for method in ["noisy_or", "weighted_avg", "max", "dempster_shafer"]:
            orchestrator.aggregation_method = method
            aggregated = orchestrator._create_aggregated_result(single_result)

            # All methods should return ~0.95 for single result
            assert aggregated.combined_confidence == pytest.approx(0.95, abs=0.05)


class TestYAMLConfiguration:
    """Tests für YAML Configuration Loading"""

    def test_load_config_from_yaml(self):
        """Test loading configuration from YAML file"""
        # Create temporary YAML config
        config_data = {
            "orchestrator": {
                "enable_hybrid": False,
                "min_confidence_threshold": 0.6,
                "aggregation_method": "weighted_avg",
                "enable_parallel_execution": True,
                "enable_result_caching": False,
            },
            "strategy_weights": {
                "direct_fact": 0.5,
                "logic_engine": 0.3,
                "graph_traversal": 0.2,
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            orchestrator = ReasoningOrchestrator(
                netzwerk=Mock(),
                logic_engine=Mock(),
                graph_traversal=Mock(),
                working_memory=Mock(),
                signals=Mock(),
                config_path=config_path,
            )

            # Check that config was loaded
            assert orchestrator.enable_hybrid is False
            assert orchestrator.min_confidence_threshold == 0.6
            assert orchestrator.aggregation_method == "weighted_avg"
            assert orchestrator.enable_parallel_execution is True
            assert orchestrator.enable_result_caching is False

            # Check strategy weights
            assert orchestrator.strategy_weights[ReasoningStrategy.DIRECT_FACT] == 0.5
            assert orchestrator.strategy_weights[ReasoningStrategy.LOGIC_ENGINE] == 0.3
            assert (
                orchestrator.strategy_weights[ReasoningStrategy.GRAPH_TRAVERSAL] == 0.2
            )

        finally:
            # Cleanup
            Path(config_path).unlink()

    def test_load_config_file_not_found(self):
        """Test graceful handling when config file not found"""
        orchestrator = ReasoningOrchestrator(
            netzwerk=Mock(),
            logic_engine=Mock(),
            graph_traversal=Mock(),
            working_memory=Mock(),
            signals=Mock(),
            config_path="nonexistent_config.yaml",
        )

        # Should use defaults
        assert orchestrator.enable_hybrid is True
        assert orchestrator.aggregation_method == "noisy_or"

    def test_load_config_invalid_yaml(self):
        """Test handling of invalid YAML file"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")
            config_path = f.name

        try:
            # Should not crash, just use defaults
            orchestrator = ReasoningOrchestrator(
                netzwerk=Mock(),
                logic_engine=Mock(),
                graph_traversal=Mock(),
                working_memory=Mock(),
                signals=Mock(),
                config_path=config_path,
            )

            # Should have defaults
            assert orchestrator.enable_hybrid is True

        finally:
            Path(config_path).unlink()


class TestResultCaching:
    """Tests für Result Caching"""

    @pytest.fixture
    def mock_netzwerk(self):
        """Mock KonzeptNetzwerk"""
        netzwerk = Mock()
        netzwerk.query_graph_for_facts = Mock(return_value={"IS_A": ["säugetier"]})
        return netzwerk

    def test_caching_enabled_by_default(self):
        """Test that caching is enabled by default"""
        orchestrator = ReasoningOrchestrator(
            netzwerk=Mock(),
            logic_engine=Mock(),
            graph_traversal=Mock(),
            working_memory=Mock(),
            signals=Mock(),
        )

        assert orchestrator.enable_result_caching is True
        assert orchestrator._result_cache is not None

    def test_cache_hit_on_repeated_query(self, mock_netzwerk):
        """Test that repeated queries hit cache"""
        orchestrator = ReasoningOrchestrator(
            netzwerk=mock_netzwerk,
            logic_engine=Mock(),
            graph_traversal=Mock(),
            working_memory=Mock(),
            signals=Mock(),
        )

        # First query - should query netzwerk
        result1 = orchestrator.query_with_hybrid_reasoning(
            "hund", "IS_A", strategies=[ReasoningStrategy.DIRECT_FACT]
        )

        # Reset mock call count for clarity
        initial_call_count = mock_netzwerk.query_graph_for_facts.call_count

        # Second query - should hit cache
        result2 = orchestrator.query_with_hybrid_reasoning(
            "hund", "IS_A", strategies=[ReasoningStrategy.DIRECT_FACT]
        )

        # Should be same object from cache (same reference)
        assert result1 is result2

        # Netzwerk should NOT be called again (cache hit)
        assert mock_netzwerk.query_graph_for_facts.call_count == initial_call_count

    def test_cache_disabled(self, mock_netzwerk):
        """Test that caching can be disabled"""
        orchestrator = ReasoningOrchestrator(
            netzwerk=mock_netzwerk,
            logic_engine=Mock(),
            graph_traversal=Mock(),
            working_memory=Mock(),
            signals=Mock(),
        )

        # Disable caching
        orchestrator.enable_result_caching = False
        orchestrator._result_cache = None

        # Two queries
        result1 = orchestrator.query_with_hybrid_reasoning(
            "hund", "IS_A", strategies=[ReasoningStrategy.DIRECT_FACT]
        )

        result2 = orchestrator.query_with_hybrid_reasoning(
            "hund", "IS_A", strategies=[ReasoningStrategy.DIRECT_FACT]
        )

        # Netzwerk should be called twice (no caching)
        assert mock_netzwerk.query_graph_for_facts.call_count == 2

    def test_cache_different_strategies(self, mock_netzwerk):
        """Test that cache distinguishes different strategy combinations"""
        orchestrator = ReasoningOrchestrator(
            netzwerk=mock_netzwerk,
            logic_engine=Mock(),
            graph_traversal=Mock(),
            working_memory=Mock(),
            signals=Mock(),
        )

        # Query with different strategies
        result1 = orchestrator.query_with_hybrid_reasoning(
            "hund", "IS_A", strategies=[ReasoningStrategy.DIRECT_FACT]
        )

        result2 = orchestrator.query_with_hybrid_reasoning(
            "hund",
            "IS_A",
            strategies=[
                ReasoningStrategy.DIRECT_FACT,
                ReasoningStrategy.GRAPH_TRAVERSAL,
            ],
        )

        # Should be different results (different cache keys)
        assert result1 is not result2


class TestParallelExecution:
    """Tests für Parallel Execution"""

    @pytest.fixture
    def mock_graph_traversal(self):
        """Mock GraphTraversal"""
        from dataclasses import dataclass

        @dataclass
        class MockPath:
            nodes: list
            relations: list
            confidence: float

        graph = Mock()
        mock_path = MockPath(nodes=["hund", "tier"], relations=["IS_A"], confidence=0.9)
        graph.find_transitive_relations = Mock(return_value=[mock_path])
        graph.create_proof_step_from_path = Mock(return_value=None)
        return graph

    @pytest.fixture
    def mock_logic_engine(self):
        """Mock Logic Engine"""
        from component_9_logik_engine import Fact, Goal, ProofStep

        engine = Mock()
        engine.add_fact = Mock()

        # Mock proof
        proof = ProofStep(
            goal=Goal(pred="IS_A", args={"subject": "hund", "object": None}),
            method="rule",
            supporting_facts=[
                Fact(
                    pred="IS_A",
                    args={"subject": "hund", "object": "säugetier"},
                    confidence=1.0,
                )
            ],
            confidence=0.8,
        )
        engine.run_with_tracking = Mock(return_value=proof)
        engine.format_proof_trace = Mock(return_value="Rule applied")

        return engine

    def test_parallel_execution_disabled_by_default(self):
        """Test that parallel execution is disabled by default"""
        orchestrator = ReasoningOrchestrator(
            netzwerk=Mock(),
            logic_engine=Mock(),
            graph_traversal=Mock(),
            working_memory=Mock(),
            signals=Mock(),
        )

        assert orchestrator.enable_parallel_execution is False

    @pytest.mark.slow
    def test_parallel_execution_runs_strategies_concurrently(
        self, mock_graph_traversal, mock_logic_engine
    ):
        """Test that parallel execution actually runs strategies concurrently"""
        netzwerk = Mock()
        netzwerk.query_graph_for_facts = Mock(return_value={})

        orchestrator = ReasoningOrchestrator(
            netzwerk=netzwerk,
            logic_engine=mock_logic_engine,
            graph_traversal=mock_graph_traversal,
            working_memory=Mock(),
            signals=Mock(),
        )

        # Enable parallel execution
        orchestrator.enable_parallel_execution = True

        # Query with multiple strategies
        result = orchestrator.query_with_hybrid_reasoning(
            "hund",
            "IS_A",
            strategies=[
                ReasoningStrategy.GRAPH_TRAVERSAL,
                ReasoningStrategy.LOGIC_ENGINE,
            ],
        )

        # Both strategies should have been called
        mock_graph_traversal.find_transitive_relations.assert_called_once()
        mock_logic_engine.run_with_tracking.assert_called_once()

        # Result should combine both
        assert result is not None
        assert len(result.strategies_used) >= 2

    def test_parallel_execution_handles_exceptions(self, mock_graph_traversal):
        """Test that parallel execution handles strategy exceptions gracefully"""
        netzwerk = Mock()
        netzwerk.query_graph_for_facts = Mock(return_value={})

        # Mock logic engine that raises exception
        logic_engine = Mock()
        logic_engine.add_fact = Mock()
        logic_engine.run_with_tracking = Mock(side_effect=Exception("Test error"))

        orchestrator = ReasoningOrchestrator(
            netzwerk=netzwerk,
            logic_engine=logic_engine,
            graph_traversal=mock_graph_traversal,
            working_memory=Mock(),
            signals=Mock(),
        )

        orchestrator.enable_parallel_execution = True

        # Should not crash, just continue with working strategies
        result = orchestrator.query_with_hybrid_reasoning(
            "hund",
            "IS_A",
            strategies=[
                ReasoningStrategy.GRAPH_TRAVERSAL,
                ReasoningStrategy.LOGIC_ENGINE,
            ],
        )

        # Should still have graph traversal result
        assert result is not None
        assert ReasoningStrategy.GRAPH_TRAVERSAL in result.strategies_used


class TestPerformanceOptimizations:
    """Tests für Performance Optimizations"""

    def test_early_exit_on_high_confidence_direct_fact(self):
        """Test early exit when direct fact has high confidence"""
        netzwerk = Mock()
        netzwerk.query_graph_for_facts = Mock(
            return_value={"IS_A": ["säugetier", "tier"]}
        )

        graph_traversal = Mock()
        logic_engine = Mock()

        orchestrator = ReasoningOrchestrator(
            netzwerk=netzwerk,
            logic_engine=logic_engine,
            graph_traversal=graph_traversal,
            working_memory=Mock(),
            signals=Mock(),
        )

        result = orchestrator.query_with_hybrid_reasoning("hund", "IS_A")

        # Should have direct fact with confidence 1.0
        assert result.combined_confidence == 1.0

        # Should NOT have called graph traversal or logic engine (early exit)
        graph_traversal.find_transitive_relations.assert_not_called()
        logic_engine.run_with_tracking.assert_not_called()


class TestStrategyWeights:
    """Tests für Strategy Weights Configuration"""

    def test_default_strategy_weights(self):
        """Test that default strategy weights are reasonable"""
        orchestrator = ReasoningOrchestrator(
            netzwerk=Mock(),
            logic_engine=Mock(),
            graph_traversal=Mock(),
            working_memory=Mock(),
            signals=Mock(),
        )

        # Check defaults
        assert orchestrator.strategy_weights[ReasoningStrategy.DIRECT_FACT] == 0.40
        assert orchestrator.strategy_weights[ReasoningStrategy.LOGIC_ENGINE] == 0.30
        assert orchestrator.strategy_weights[ReasoningStrategy.GRAPH_TRAVERSAL] == 0.20
        assert orchestrator.strategy_weights[ReasoningStrategy.PROBABILISTIC] == 0.08
        assert orchestrator.strategy_weights[ReasoningStrategy.ABDUCTIVE] == 0.02

        # Should sum to ~1.0
        total_weight = sum(orchestrator.strategy_weights.values())
        assert total_weight == pytest.approx(1.0, rel=0.01)

    def test_custom_strategy_weights(self):
        """Test setting custom strategy weights"""
        orchestrator = ReasoningOrchestrator(
            netzwerk=Mock(),
            logic_engine=Mock(),
            graph_traversal=Mock(),
            working_memory=Mock(),
            signals=Mock(),
        )

        # Set custom weights
        orchestrator.strategy_weights[ReasoningStrategy.LOGIC_ENGINE] = 0.5
        orchestrator.strategy_weights[ReasoningStrategy.GRAPH_TRAVERSAL] = 0.5

        # Test with weighted_avg
        results = [
            ReasoningResult(
                strategy=ReasoningStrategy.LOGIC_ENGINE,
                success=True,
                confidence=0.8,
                inferred_facts={},
            ),
            ReasoningResult(
                strategy=ReasoningStrategy.GRAPH_TRAVERSAL,
                success=True,
                confidence=0.6,
                inferred_facts={},
            ),
        ]

        orchestrator.aggregation_method = "weighted_avg"
        confidence = orchestrator._weighted_average(results)

        # Expected: (0.5*0.8 + 0.5*0.6) / (0.5+0.5) = 0.7
        assert confidence == pytest.approx(0.7)


# ==================== Integration Tests ====================


@pytest.mark.integration
class TestPhase2Integration:
    """Integration tests for Phase 2 features"""

    def test_full_pipeline_with_custom_config(self):
        """Test complete pipeline with custom YAML configuration"""
        # Create config
        config_data = {
            "orchestrator": {
                "aggregation_method": "weighted_avg",
                "enable_parallel_execution": False,
                "min_confidence_threshold": 0.5,
            },
            "strategy_weights": {"direct_fact": 0.6, "graph_traversal": 0.4},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            # Setup mocks
            netzwerk = Mock()
            netzwerk.query_graph_for_facts = Mock(return_value={"IS_A": ["tier"]})

            orchestrator = ReasoningOrchestrator(
                netzwerk=netzwerk,
                logic_engine=Mock(),
                graph_traversal=Mock(),
                working_memory=Mock(),
                signals=Mock(),
                config_path=config_path,
            )

            # Query
            result = orchestrator.query_with_hybrid_reasoning(
                "hund", "IS_A", strategies=[ReasoningStrategy.DIRECT_FACT]
            )

            # Check that config was applied
            assert orchestrator.aggregation_method == "weighted_avg"
            assert orchestrator.min_confidence_threshold == 0.5
            assert result is not None

        finally:
            Path(config_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
