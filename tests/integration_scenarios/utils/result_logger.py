"""
tests/integration_scenarios/utils/result_logger.py

Comprehensive logging of KAI's thought process for post-analysis.
Provides ScenarioLogger class for structured logging of scenario execution.

NO Unicode characters - ASCII only (see CLAUDE.md encoding rules).
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


class ScenarioLogger:
    """Comprehensive logger for scenario test execution"""

    def __init__(self, scenario_name: str, output_dir: Path):
        """
        Initialize scenario logger.

        Args:
            scenario_name: Name of the scenario being tested
            output_dir: Directory to save log files
        """
        self.scenario_name = scenario_name
        self.output_dir = Path(output_dir)
        self.logs: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {
            "scenario_name": scenario_name,
            "start_time": datetime.now().isoformat(),
        }

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def log_input(self, input_text: str, context: Dict):
        """
        Log the input query and context.

        Args:
            input_text: The input text/query
            context: Context dictionary (pre-learned facts, etc.)
        """
        self.logs.append(
            {
                "type": "input",
                "timestamp": datetime.now().isoformat(),
                "input_text": input_text,
                "context": context,
            }
        )

    def log_preprocessing(self, preprocessed: Dict):
        """
        Log linguistic preprocessing results.

        Args:
            preprocessed: Dictionary with preprocessing results
        """
        self.logs.append(
            {
                "type": "preprocessing",
                "timestamp": datetime.now().isoformat(),
                "data": preprocessed,
            }
        )

    def log_strategy_selection(self, strategies: List[str], reason: str):
        """
        Log which strategies were selected and why.

        Args:
            strategies: List of strategy names
            reason: Explanation for strategy selection
        """
        self.logs.append(
            {
                "type": "strategy_selection",
                "timestamp": datetime.now().isoformat(),
                "strategies": strategies,
                "reason": reason,
            }
        )

    def log_reasoning_step(
        self,
        step_num: int,
        strategy: str,
        action: str,
        result: Any,
        confidence: float,
    ):
        """
        Log a single reasoning step.

        Args:
            step_num: Step number
            strategy: Strategy being used
            action: Action description
            result: Result of the action
            confidence: Confidence value for this step
        """
        self.logs.append(
            {
                "type": "reasoning_step",
                "timestamp": datetime.now().isoformat(),
                "step_num": step_num,
                "strategy": strategy,
                "action": action,
                "result": str(result),  # Convert to string for JSON serialization
                "confidence": confidence,
            }
        )

    def log_proof_tree_update(self, proof_tree_snapshot: Dict):
        """
        Log ProofTree state at this point.

        Args:
            proof_tree_snapshot: Current state of ProofTree
        """
        self.logs.append(
            {
                "type": "proof_tree_update",
                "timestamp": datetime.now().isoformat(),
                "proof_tree": proof_tree_snapshot,
            }
        )

    def log_memory_update(self, memory_type: str, content: Dict):
        """
        Log episodic/working memory updates.

        Args:
            memory_type: Type of memory ("episodic", "working", etc.)
            content: Memory content
        """
        self.logs.append(
            {
                "type": "memory_update",
                "timestamp": datetime.now().isoformat(),
                "memory_type": memory_type,
                "content": content,
            }
        )

    def log_final_response(self, response: str, confidence: float):
        """
        Log KAI's final response.

        Args:
            response: Final response text
            confidence: Final confidence value
        """
        self.logs.append(
            {
                "type": "final_response",
                "timestamp": datetime.now().isoformat(),
                "response": response,
                "confidence": confidence,
            }
        )

        # Update metadata
        self.metadata["end_time"] = datetime.now().isoformat()

    def save_logs(self) -> Path:
        """
        Save all logs to structured JSON file.

        File: output_dir/scenario_name_TIMESTAMP.json

        Structure:
        {
            "scenario": str,
            "timestamp": str,
            "input": {...},
            "preprocessing": {...},
            "reasoning_steps": [...],
            "proof_tree_evolution": [...],
            "memory_updates": [...],
            "final_response": {...},
            "metadata": {...}
        }

        Returns:
            Path to saved log file
        """
        # Organize logs by type
        organized = {
            "scenario": self.scenario_name,
            "timestamp": self.metadata.get("start_time", "unknown"),
            "input": None,
            "preprocessing": None,
            "strategy_selection": None,
            "reasoning_steps": [],
            "proof_tree_evolution": [],
            "memory_updates": [],
            "final_response": None,
            "metadata": self.metadata,
        }

        for log in self.logs:
            log_type = log["type"]

            if log_type == "input":
                organized["input"] = log
            elif log_type == "preprocessing":
                organized["preprocessing"] = log
            elif log_type == "strategy_selection":
                organized["strategy_selection"] = log
            elif log_type == "reasoning_step":
                organized["reasoning_steps"].append(log)
            elif log_type == "proof_tree_update":
                organized["proof_tree_evolution"].append(log)
            elif log_type == "memory_update":
                organized["memory_updates"].append(log)
            elif log_type == "final_response":
                organized["final_response"] = log

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = self.scenario_name.replace(" ", "_").replace("/", "_")
        filename = f"{safe_name}_{timestamp}.json"
        filepath = self.output_dir / filename

        # Save to file
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(organized, f, indent=2, ensure_ascii=False)

        return filepath

    def generate_human_readable_summary(self) -> str:
        """
        Generate markdown summary of reasoning process.

        Returns:
            Markdown-formatted string
        """
        lines = []
        lines.append(f"# Scenario Test Summary: {self.scenario_name}")
        lines.append("")
        lines.append(
            f"**Date**: {self.metadata.get('start_time', 'unknown').split('T')[0]}"
        )
        lines.append("")

        # Input section
        lines.append("## Input")
        input_log = next((log for log in self.logs if log["type"] == "input"), None)
        if input_log:
            lines.append("```")
            lines.append(input_log["input_text"])
            lines.append("```")
        lines.append("")

        # Strategy selection
        lines.append("## Strategy Selection")
        strategy_log = next(
            (log for log in self.logs if log["type"] == "strategy_selection"), None
        )
        if strategy_log:
            lines.append(f"**Strategies**: {', '.join(strategy_log['strategies'])}")
            lines.append(f"**Reason**: {strategy_log['reason']}")
        else:
            lines.append("No strategy selection logged")
        lines.append("")

        # Reasoning steps
        lines.append("## Reasoning Steps")
        reasoning_steps = [log for log in self.logs if log["type"] == "reasoning_step"]
        if reasoning_steps:
            for step in reasoning_steps:
                step_num = step["step_num"]
                strategy = step["strategy"]
                action = step["action"]
                confidence = step["confidence"]
                lines.append(
                    f"{step_num}. **{strategy}**: {action} (confidence: {confidence:.2f})"
                )
        else:
            lines.append("No reasoning steps logged")
        lines.append("")

        # Memory updates
        lines.append("## Memory Updates")
        memory_updates = [log for log in self.logs if log["type"] == "memory_update"]
        if memory_updates:
            lines.append(f"Total memory updates: {len(memory_updates)}")
            for mem in memory_updates[:5]:  # Show first 5
                mem_type = mem["memory_type"]
                lines.append(f"- **{mem_type}** update at {mem['timestamp']}")
            if len(memory_updates) > 5:
                lines.append(f"- ... and {len(memory_updates) - 5} more")
        else:
            lines.append("No memory updates logged")
        lines.append("")

        # Final response
        lines.append("## Final Response")
        final_log = next(
            (log for log in self.logs if log["type"] == "final_response"), None
        )
        if final_log:
            lines.append(f"**Confidence**: {final_log['confidence']:.2f}")
            lines.append("```")
            lines.append(final_log["response"])
            lines.append("```")
        else:
            lines.append("No final response logged")
        lines.append("")

        # Summary stats
        lines.append("## Statistics")
        lines.append(f"- Total log entries: {len(self.logs)}")
        lines.append(f"- Reasoning steps: {len(reasoning_steps)}")
        lines.append(f"- Memory updates: {len(memory_updates)}")
        proof_tree_updates = [
            log for log in self.logs if log["type"] == "proof_tree_update"
        ]
        lines.append(f"- ProofTree updates: {len(proof_tree_updates)}")

        return "\n".join(lines)

    def add_metadata(self, key: str, value: Any):
        """
        Add metadata entry.

        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value

    def get_logs_by_type(self, log_type: str) -> List[Dict]:
        """
        Get all logs of a specific type.

        Args:
            log_type: Type of log to retrieve

        Returns:
            List of matching log entries
        """
        return [log for log in self.logs if log["type"] == log_type]

    def get_log_count(self) -> int:
        """
        Get total number of log entries.

        Returns:
            Count of log entries
        """
        return len(self.logs)

    def clear_logs(self):
        """Clear all logs (useful for resetting between scenarios)"""
        self.logs = []
        self.metadata = {
            "scenario_name": self.scenario_name,
            "start_time": datetime.now().isoformat(),
        }
