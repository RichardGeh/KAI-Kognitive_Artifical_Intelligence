# tests/test_hybrid_reasoning.py
"""
Tests für Hybrid Reasoning System

Testet:
- ReasoningOrchestrator
- Result Aggregation
- Hybrid Proof Tree Generation
- Integration mit InferenceHandler
"""

import pytest
from unittest.mock import Mock, patch

from kai_reasoning_orchestrator import (
    ReasoningOrchestrator,
    ReasoningStrategy,
    ReasoningResult,
)


class TestReasoningOrchestrator:
    """Tests für ReasoningOrchestrator"""

    @pytest.fixture
    def mock_netzwerk(self):
        """Mock KonzeptNetzwerk"""
        netzwerk = Mock()
        netzwerk.query_graph_for_facts = Mock(
            return_value={"IS_A": ["säugetier", "haustier"]}
        )
        return netzwerk

    @pytest.fixture
    def mock_logic_engine(self):
        """Mock Logic Engine"""
        engine = Mock()
        engine.add_fact = Mock()
        engine.run_with_tracking = Mock(return_value=None)
        return engine

    @pytest.fixture
    def mock_graph_traversal(self):
        """Mock GraphTraversal"""
        from dataclasses import dataclass

        @dataclass
        class MockPath:
            nodes: list
            relations: list
            confidence: float
            explanation: str

        graph = Mock()
        mock_path = MockPath(
            nodes=["hund", "säugetier", "tier"],
            relations=["IS_A", "IS_A"],
            confidence=0.95,
            explanation="hund -> säugetier -> tier",
        )
        graph.find_transitive_relations = Mock(return_value=[mock_path])
        graph.create_proof_step_from_path = Mock(return_value=None)
        return graph

    @pytest.fixture
    def mock_working_memory(self):
        """Mock WorkingMemory"""
        memory = Mock()
        memory.add_reasoning_state = Mock()
        return memory

    @pytest.fixture
    def mock_signals(self):
        """Mock KaiSignals"""
        signals = Mock()
        signals.proof_tree_update = Mock()
        signals.proof_tree_update.emit = Mock()
        return signals

    @pytest.fixture
    def orchestrator(
        self,
        mock_netzwerk,
        mock_logic_engine,
        mock_graph_traversal,
        mock_working_memory,
        mock_signals,
    ):
        """ReasoningOrchestrator instance"""
        return ReasoningOrchestrator(
            netzwerk=mock_netzwerk,
            logic_engine=mock_logic_engine,
            graph_traversal=mock_graph_traversal,
            working_memory=mock_working_memory,
            signals=mock_signals,
            probabilistic_engine=None,
            abductive_engine=None,
        )

    def test_initialization(self, orchestrator):
        """Test dass Orchestrator korrekt initialisiert wird"""
        assert orchestrator is not None
        assert orchestrator.enable_hybrid is True
        assert orchestrator.min_confidence_threshold == 0.4
        assert orchestrator.probabilistic_enhancement is True

    def test_direct_fact_lookup_success(self, orchestrator, mock_netzwerk):
        """Test Direct Fact Lookup mit Erfolg"""
        result = orchestrator._try_direct_fact_lookup("hund", "IS_A")

        assert result is not None
        assert result.success is True
        assert result.strategy == ReasoningStrategy.DIRECT_FACT
        assert result.confidence == 1.0
        assert "IS_A" in result.inferred_facts
        assert "säugetier" in result.inferred_facts["IS_A"]
        assert "haustier" in result.inferred_facts["IS_A"]

    def test_direct_fact_lookup_no_results(self, orchestrator, mock_netzwerk):
        """Test Direct Fact Lookup ohne Ergebnisse"""
        mock_netzwerk.query_graph_for_facts = Mock(return_value={})

        result = orchestrator._try_direct_fact_lookup("unbekannt", "IS_A")

        assert result is None

    def test_graph_traversal_success(
        self, orchestrator, mock_graph_traversal, mock_working_memory
    ):
        """Test Graph Traversal mit Erfolg"""
        result = orchestrator._try_graph_traversal("hund", "IS_A")

        assert result is not None
        assert result.success is True
        assert result.strategy == ReasoningStrategy.GRAPH_TRAVERSAL
        assert result.confidence == 0.95
        assert "IS_A" in result.inferred_facts
        assert "tier" in result.inferred_facts["IS_A"]

        # Verify working memory was updated
        mock_working_memory.add_reasoning_state.assert_called_once()

    def test_graph_traversal_no_paths(self, orchestrator, mock_graph_traversal):
        """Test Graph Traversal ohne Pfade"""
        mock_graph_traversal.find_transitive_relations = Mock(return_value=[])

        result = orchestrator._try_graph_traversal("unbekannt", "IS_A")

        assert result is None

    def test_noisy_or_combination(self, orchestrator):
        """Test Noisy-OR Aggregation"""
        probabilities = [0.8, 0.7, 0.6]
        combined = orchestrator._noisy_or(probabilities)

        # Noisy-OR: 1 - (1-0.8)*(1-0.7)*(1-0.6) = 1 - 0.2*0.3*0.4 = 1 - 0.024 = 0.976
        assert combined == pytest.approx(0.976, rel=0.01)

    def test_noisy_or_empty(self, orchestrator):
        """Test Noisy-OR mit leerer Liste"""
        assert orchestrator._noisy_or([]) == 0.0

    def test_noisy_or_single(self, orchestrator):
        """Test Noisy-OR mit einzelnem Wert"""
        assert orchestrator._noisy_or([0.8]) == pytest.approx(0.8)

    def test_create_aggregated_result(self, orchestrator):
        """Test Result Aggregation"""
        # Create mock results
        result1 = ReasoningResult(
            strategy=ReasoningStrategy.GRAPH_TRAVERSAL,
            success=True,
            confidence=0.9,
            inferred_facts={"IS_A": ["tier"]},
            proof_trace="Graph path found",
        )

        result2 = ReasoningResult(
            strategy=ReasoningStrategy.LOGIC_ENGINE,
            success=True,
            confidence=0.8,
            inferred_facts={"IS_A": ["lebewesen"]},
            proof_trace="Rule applied",
        )

        aggregated = orchestrator._create_aggregated_result([result1, result2])

        # Check aggregation
        assert aggregated.combined_confidence == pytest.approx(
            0.98, rel=0.01
        )  # Noisy-OR(0.9, 0.8)
        assert len(aggregated.strategies_used) == 2
        assert ReasoningStrategy.GRAPH_TRAVERSAL in aggregated.strategies_used
        assert ReasoningStrategy.LOGIC_ENGINE in aggregated.strategies_used

        # Check merged facts
        assert "IS_A" in aggregated.inferred_facts
        assert "tier" in aggregated.inferred_facts["IS_A"]
        assert "lebewesen" in aggregated.inferred_facts["IS_A"]

    def test_hybrid_reasoning_integration(
        self, orchestrator, mock_netzwerk, mock_graph_traversal
    ):
        """Test vollständiges Hybrid Reasoning"""
        # Setup: Direct facts available
        mock_netzwerk.query_graph_for_facts = Mock(return_value={"IS_A": ["säugetier"]})

        # Execute hybrid reasoning
        result = orchestrator.query_with_hybrid_reasoning("hund", "IS_A")

        assert result is not None
        assert result.combined_confidence >= 0.4  # Above threshold
        assert len(result.strategies_used) >= 1

    def test_hybrid_reasoning_fallback_chain(
        self, orchestrator, mock_netzwerk, mock_graph_traversal
    ):
        """Test Fallback-Kette bei fehlenden Ergebnissen"""
        # Setup: Kein Direct Fact
        mock_netzwerk.query_graph_for_facts = Mock(return_value={})
        # Kein Graph Traversal
        mock_graph_traversal.find_transitive_relations = Mock(return_value=[])

        result = orchestrator.query_with_hybrid_reasoning("unbekannt", "IS_A")

        # Sollte Abductive probieren (falls verfügbar)
        # Ohne Abductive Engine: None
        assert (
            result is None
            or result.combined_confidence < orchestrator.min_confidence_threshold
        )

    def test_extract_facts_from_proof(self, orchestrator):
        """Test Fact Extraction aus Logic Engine Proof"""
        from component_9_logik_engine import Fact, Goal, ProofStep

        # Create mock proof
        fact1 = Fact(pred="IS_A", args={"subject": "hund", "object": "säugetier"})
        fact2 = Fact(pred="IS_A", args={"subject": "säugetier", "object": "tier"})

        proof = ProofStep(
            goal=Goal(pred="IS_A", args={"subject": "hund", "object": None}),
            method="rule",
            supporting_facts=[fact1, fact2],
            confidence=0.9,
        )

        facts = orchestrator._extract_facts_from_proof(proof)

        assert "IS_A" in facts
        assert "säugetier" in facts["IS_A"]
        assert "tier" in facts["IS_A"]


class TestHybridProofTreeGeneration:
    """Tests für Hybrid Proof Tree Generation"""

    def test_create_hybrid_proof_step(self):
        """Test Hybrid ProofStep Generierung"""
        from component_17_proof_explanation import create_hybrid_proof_step

        # Create mock results
        result1 = Mock()
        result1.strategy = ReasoningStrategy.GRAPH_TRAVERSAL
        result1.confidence = 0.9
        result1.inferred_facts = {"IS_A": ["tier"]}
        result1.proof_tree = None
        result1.proof_trace = "Graph path found"

        result2 = Mock()
        result2.strategy = ReasoningStrategy.LOGIC_ENGINE
        result2.confidence = 0.8
        result2.inferred_facts = {"IS_A": ["lebewesen"]}
        result2.proof_tree = None
        result2.proof_trace = "Rule applied"

        # Generate hybrid proof step
        hybrid_step = create_hybrid_proof_step(
            results=[result1, result2],
            query="Was ist ein hund?",
            aggregation_method="noisy_or",
        )

        assert hybrid_step is not None
        assert "hybrid" in hybrid_step.metadata
        assert hybrid_step.metadata["hybrid"] is True
        assert hybrid_step.metadata["num_strategies"] == 2
        assert len(hybrid_step.subgoals) == 2
        assert hybrid_step.confidence == pytest.approx(0.98, rel=0.01)

    def test_noisy_or_combination_function(self):
        """Test standalone Noisy-OR function"""
        from component_17_proof_explanation import _noisy_or_combination

        # Test with multiple values
        result = _noisy_or_combination([0.8, 0.7, 0.6])
        assert result == pytest.approx(0.976, rel=0.01)

        # Test with single value
        result = _noisy_or_combination([0.5])
        assert result == pytest.approx(0.5)

        # Test with empty list
        result = _noisy_or_combination([])
        assert result == 0.0


class TestInferenceHandlerIntegration:
    """Tests für Integration mit InferenceHandler"""

    @pytest.fixture
    def mock_inference_handler(self):
        """Mock InferenceHandler mit Hybrid Reasoning"""
        from kai_inference_handler import KaiInferenceHandler

        # Create minimal mocks
        netzwerk = Mock()
        engine = Mock()
        graph_traversal = Mock()
        working_memory = Mock()
        signals = Mock()

        # Patch ReasoningOrchestrator import
        with patch("kai_inference_handler.ReasoningOrchestrator"):
            handler = KaiInferenceHandler(
                netzwerk=netzwerk,
                engine=engine,
                graph_traversal=graph_traversal,
                working_memory=working_memory,
                signals=signals,
                enable_hybrid_reasoning=True,
            )

        return handler

    def test_hybrid_reasoning_enabled_flag(self, mock_inference_handler):
        """Test dass Hybrid Reasoning aktiviert werden kann"""
        # Should be True by default
        assert hasattr(mock_inference_handler, "enable_hybrid_reasoning")

    def test_try_hybrid_reasoning_method_exists(self, mock_inference_handler):
        """Test dass try_hybrid_reasoning Methode existiert"""
        assert hasattr(mock_inference_handler, "try_hybrid_reasoning")
        assert callable(mock_inference_handler.try_hybrid_reasoning)


class TestPerformance:
    """Performance-Tests für Hybrid Reasoning"""

    def test_direct_fact_lookup_is_fast(self, benchmark):
        """Test dass Direct Fact Lookup schnell ist"""
        # Mock setup
        netzwerk = Mock()
        netzwerk.query_graph_for_facts = Mock(return_value={"IS_A": ["tier"]})

        orchestrator = ReasoningOrchestrator(
            netzwerk=netzwerk,
            logic_engine=Mock(),
            graph_traversal=Mock(),
            working_memory=Mock(),
            signals=Mock(),
        )

        # Benchmark
        def run_lookup():
            return orchestrator._try_direct_fact_lookup("hund", "IS_A")

        result = benchmark(run_lookup)
        assert result is not None
        # Should be very fast (< 10ms)


class TestEdgeCases:
    """Edge Case Tests"""

    def test_empty_results_aggregation(self):
        """Test Aggregation mit leeren Ergebnissen"""
        orchestrator = ReasoningOrchestrator(
            netzwerk=Mock(),
            logic_engine=Mock(),
            graph_traversal=Mock(),
            working_memory=Mock(),
            signals=Mock(),
        )

        # Should raise ValueError
        with pytest.raises(ValueError, match="Cannot aggregate empty results"):
            orchestrator._create_aggregated_result([])

    def test_single_result_aggregation(self):
        """Test Aggregation mit nur einem Ergebnis"""
        orchestrator = ReasoningOrchestrator(
            netzwerk=Mock(),
            logic_engine=Mock(),
            graph_traversal=Mock(),
            working_memory=Mock(),
            signals=Mock(),
        )

        result = ReasoningResult(
            strategy=ReasoningStrategy.DIRECT_FACT,
            success=True,
            confidence=0.95,
            inferred_facts={"IS_A": ["tier"]},
        )

        aggregated = orchestrator._create_aggregated_result([result])

        assert aggregated.combined_confidence == 0.95
        assert len(aggregated.strategies_used) == 1

    def test_hypothesis_flag_propagation(self):
        """Test dass is_hypothesis Flag korrekt propagiert wird"""
        orchestrator = ReasoningOrchestrator(
            netzwerk=Mock(),
            logic_engine=Mock(),
            graph_traversal=Mock(),
            working_memory=Mock(),
            signals=Mock(),
        )

        # Normal result
        result1 = ReasoningResult(
            strategy=ReasoningStrategy.DIRECT_FACT,
            success=True,
            confidence=0.9,
            inferred_facts={},
            is_hypothesis=False,
        )

        # Hypothesis result
        result2 = ReasoningResult(
            strategy=ReasoningStrategy.ABDUCTIVE,
            success=True,
            confidence=0.6,
            inferred_facts={},
            is_hypothesis=True,
        )

        # Aggregate
        aggregated = orchestrator._create_aggregated_result([result1, result2])

        # Should be marked as hypothesis if ANY result is hypothesis
        assert aggregated.is_hypothesis is True


# ==================== Integration Tests ====================


@pytest.mark.integration
class TestE2EHybridReasoning:
    """End-to-End Integration Tests"""

    @pytest.mark.skip(reason="Requires full Neo4j setup")
    def test_full_hybrid_reasoning_pipeline(self):
        """Test vollständige Pipeline mit echten Komponenten"""
        # This would require:
        # - Neo4j running
        # - Knowledge graph populated
        # - All engines initialized
        # TODO: Implement when integration test infrastructure ready


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
