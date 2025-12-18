"""
tests/integration_scenarios/utils/test_result_logger.py

Unit tests for result_logger.py module.
Tests ScenarioLogger class for comprehensive logging.

NO Unicode characters - ASCII only (see CLAUDE.md encoding rules).
"""

import json

import pytest

from tests.integration_scenarios.utils.result_logger import ScenarioLogger


class TestScenarioLogger:
    """Test ScenarioLogger class"""

    @pytest.fixture
    def temp_output_dir(self, tmp_path):
        """Create temporary output directory"""
        output_dir = tmp_path / "scenario_logs"
        output_dir.mkdir()
        return output_dir

    def test_initialization(self, temp_output_dir):
        """Test logger initialization"""
        logger = ScenarioLogger("test_scenario", temp_output_dir)

        assert logger.scenario_name == "test_scenario"
        assert logger.output_dir == temp_output_dir
        assert logger.logs == []
        assert "scenario_name" in logger.metadata
        assert "start_time" in logger.metadata

    def test_log_input(self, temp_output_dir):
        """Test logging input"""
        logger = ScenarioLogger("test_scenario", temp_output_dir)
        logger.log_input("Test query", {"key": "value"})

        assert len(logger.logs) == 1
        assert logger.logs[0]["type"] == "input"
        assert logger.logs[0]["input_text"] == "Test query"
        assert logger.logs[0]["context"] == {"key": "value"}

    def test_log_preprocessing(self, temp_output_dir):
        """Test logging preprocessing"""
        logger = ScenarioLogger("test_scenario", temp_output_dir)
        logger.log_preprocessing({"tokens": ["a", "b", "c"]})

        assert len(logger.logs) == 1
        assert logger.logs[0]["type"] == "preprocessing"
        assert logger.logs[0]["data"] == {"tokens": ["a", "b", "c"]}

    def test_log_strategy_selection(self, temp_output_dir):
        """Test logging strategy selection"""
        logger = ScenarioLogger("test_scenario", temp_output_dir)
        logger.log_strategy_selection(["logic", "graph"], "Best strategies")

        assert len(logger.logs) == 1
        assert logger.logs[0]["type"] == "strategy_selection"
        assert logger.logs[0]["strategies"] == ["logic", "graph"]
        assert logger.logs[0]["reason"] == "Best strategies"

    def test_log_reasoning_step(self, temp_output_dir):
        """Test logging reasoning step"""
        logger = ScenarioLogger("test_scenario", temp_output_dir)
        logger.log_reasoning_step(1, "logic_engine", "Deduce", {"result": "A"}, 0.85)

        assert len(logger.logs) == 1
        log = logger.logs[0]
        assert log["type"] == "reasoning_step"
        assert log["step_num"] == 1
        assert log["strategy"] == "logic_engine"
        assert log["action"] == "Deduce"
        assert log["confidence"] == 0.85

    def test_log_proof_tree_update(self, temp_output_dir):
        """Test logging proof tree update"""
        logger = ScenarioLogger("test_scenario", temp_output_dir)
        tree = {"step": "root", "children": []}
        logger.log_proof_tree_update(tree)

        assert len(logger.logs) == 1
        assert logger.logs[0]["type"] == "proof_tree_update"
        assert logger.logs[0]["proof_tree"] == tree

    def test_log_memory_update(self, temp_output_dir):
        """Test logging memory update"""
        logger = ScenarioLogger("test_scenario", temp_output_dir)
        logger.log_memory_update("episodic", {"episode": "data"})

        assert len(logger.logs) == 1
        assert logger.logs[0]["type"] == "memory_update"
        assert logger.logs[0]["memory_type"] == "episodic"
        assert logger.logs[0]["content"] == {"episode": "data"}

    def test_log_final_response(self, temp_output_dir):
        """Test logging final response"""
        logger = ScenarioLogger("test_scenario", temp_output_dir)
        logger.log_final_response("Final answer", 0.92)

        assert len(logger.logs) == 1
        assert logger.logs[0]["type"] == "final_response"
        assert logger.logs[0]["response"] == "Final answer"
        assert logger.logs[0]["confidence"] == 0.92

        # Should update metadata
        assert "end_time" in logger.metadata

    def test_save_logs(self, temp_output_dir):
        """Test saving logs to file"""
        logger = ScenarioLogger("test_scenario", temp_output_dir)
        logger.log_input("Test query", {})
        logger.log_reasoning_step(1, "logic", "action", "result", 0.8)
        logger.log_final_response("Answer", 0.9)

        filepath = logger.save_logs()

        # Check file was created
        assert filepath.exists()
        assert filepath.suffix == ".json"

        # Check file contents
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["scenario"] == "test_scenario"
        assert data["input"] is not None
        assert len(data["reasoning_steps"]) == 1
        assert data["final_response"] is not None

    def test_save_logs_organizes_by_type(self, temp_output_dir):
        """Test that save_logs organizes logs by type"""
        logger = ScenarioLogger("test_scenario", temp_output_dir)

        # Add various log types
        logger.log_input("Query", {})
        logger.log_preprocessing({"data": "preprocessed"})
        logger.log_strategy_selection(["strategy1"], "reason")
        logger.log_reasoning_step(1, "engine", "action", "result", 0.8)
        logger.log_reasoning_step(2, "engine", "action2", "result2", 0.9)
        logger.log_proof_tree_update({"tree": "data"})
        logger.log_memory_update("episodic", {"memory": "data"})
        logger.log_final_response("Answer", 0.95)

        filepath = logger.save_logs()

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Check organization
        assert data["input"]["type"] == "input"
        assert data["preprocessing"]["type"] == "preprocessing"
        assert data["strategy_selection"]["type"] == "strategy_selection"
        assert len(data["reasoning_steps"]) == 2
        assert len(data["proof_tree_evolution"]) == 1
        assert len(data["memory_updates"]) == 1
        assert data["final_response"]["type"] == "final_response"

    def test_generate_human_readable_summary(self, temp_output_dir):
        """Test human-readable summary generation"""
        logger = ScenarioLogger("test_scenario", temp_output_dir)

        logger.log_input("Test query", {})
        logger.log_strategy_selection(["logic", "graph"], "Best fit")
        logger.log_reasoning_step(1, "logic", "Deduce A", "result", 0.8)
        logger.log_reasoning_step(2, "graph", "Traverse", "result", 0.9)
        logger.log_memory_update("working", {"data": "memory"})
        logger.log_final_response("Final answer", 0.92)

        summary = logger.generate_human_readable_summary()

        # Check summary contains expected sections
        assert "# Scenario Test Summary" in summary
        assert "test_scenario" in summary
        assert "## Input" in summary
        assert "## Strategy Selection" in summary
        assert "## Reasoning Steps" in summary
        assert "## Memory Updates" in summary
        assert "## Final Response" in summary
        assert "## Statistics" in summary

        # Check content
        assert "Test query" in summary
        assert "logic" in summary
        assert "graph" in summary
        assert "Deduce A" in summary
        assert "Final answer" in summary

    def test_generate_summary_with_no_data(self, temp_output_dir):
        """Test summary generation with no logged data"""
        logger = ScenarioLogger("test_scenario", temp_output_dir)
        summary = logger.generate_human_readable_summary()

        assert "test_scenario" in summary
        assert "No strategy selection logged" in summary
        assert "No reasoning steps logged" in summary
        assert "No memory updates logged" in summary
        assert "No final response logged" in summary

    def test_add_metadata(self, temp_output_dir):
        """Test adding custom metadata"""
        logger = ScenarioLogger("test_scenario", temp_output_dir)
        logger.add_metadata("custom_key", "custom_value")
        logger.add_metadata("test_number", 123)

        assert logger.metadata["custom_key"] == "custom_value"
        assert logger.metadata["test_number"] == 123

    def test_get_logs_by_type(self, temp_output_dir):
        """Test retrieving logs by type"""
        logger = ScenarioLogger("test_scenario", temp_output_dir)

        logger.log_input("Query", {})
        logger.log_reasoning_step(1, "logic", "action", "result", 0.8)
        logger.log_reasoning_step(2, "graph", "action2", "result2", 0.9)
        logger.log_final_response("Answer", 0.9)

        reasoning_logs = logger.get_logs_by_type("reasoning_step")
        assert len(reasoning_logs) == 2

        input_logs = logger.get_logs_by_type("input")
        assert len(input_logs) == 1

        nonexistent_logs = logger.get_logs_by_type("nonexistent")
        assert len(nonexistent_logs) == 0

    def test_get_log_count(self, temp_output_dir):
        """Test getting log count"""
        logger = ScenarioLogger("test_scenario", temp_output_dir)

        assert logger.get_log_count() == 0

        logger.log_input("Query", {})
        logger.log_reasoning_step(1, "logic", "action", "result", 0.8)

        assert logger.get_log_count() == 2

    def test_clear_logs(self, temp_output_dir):
        """Test clearing logs"""
        logger = ScenarioLogger("test_scenario", temp_output_dir)

        logger.log_input("Query", {})
        logger.log_reasoning_step(1, "logic", "action", "result", 0.8)
        logger.add_metadata("custom", "value")

        assert len(logger.logs) > 0

        logger.clear_logs()

        assert len(logger.logs) == 0
        assert "scenario_name" in logger.metadata  # Basic metadata preserved
        assert "custom" not in logger.metadata  # Custom metadata cleared

    def test_multiple_scenarios_separate_files(self, temp_output_dir):
        """Test that multiple scenarios create separate files"""
        logger1 = ScenarioLogger("scenario_1", temp_output_dir)
        logger1.log_input("Query 1", {})
        filepath1 = logger1.save_logs()

        logger2 = ScenarioLogger("scenario_2", temp_output_dir)
        logger2.log_input("Query 2", {})
        filepath2 = logger2.save_logs()

        # Files should be different
        assert filepath1 != filepath2
        assert "scenario_1" in str(filepath1)
        assert "scenario_2" in str(filepath2)

    def test_complete_logging_workflow(self, temp_output_dir):
        """Test complete logging workflow"""
        logger = ScenarioLogger("complete_test", temp_output_dir)

        # Log entire scenario execution
        logger.log_input("What is the answer?", {"context": "test"})
        logger.log_preprocessing({"tokens": ["what", "answer"]})
        logger.log_strategy_selection(["logic", "graph"], "Multi-strategy approach")

        logger.log_reasoning_step(1, "logic", "Analyze query", "analyzed", 0.7)
        logger.log_proof_tree_update({"step": "root", "children": []})

        logger.log_reasoning_step(2, "graph", "Query graph", "found_facts", 0.85)
        logger.log_memory_update("working", {"facts": ["fact1", "fact2"]})

        logger.log_reasoning_step(3, "logic", "Deduce answer", "answer", 0.92)
        logger.log_final_response("The answer is 42", 0.92)

        # Add metadata
        logger.add_metadata("test_duration_ms", 1234)
        logger.add_metadata("neo4j_queries", 15)

        # Save and verify
        filepath = logger.save_logs()
        assert filepath.exists()

        # Load and verify structure
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["scenario"] == "complete_test"
        assert data["input"]["input_text"] == "What is the answer?"
        assert len(data["reasoning_steps"]) == 3
        assert data["final_response"]["confidence"] == 0.92
        assert data["metadata"]["test_duration_ms"] == 1234

        # Generate and verify summary
        summary = logger.generate_human_readable_summary()
        assert "complete_test" in summary
        assert "What is the answer?" in summary
        assert "The answer is 42" in summary
