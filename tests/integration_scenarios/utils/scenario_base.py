"""
tests/integration_scenarios/utils/scenario_base.py

Base class for all scenario integration tests with gradual scoring infrastructure.
Provides ScenarioResult dataclass and ScenarioTestBase class for structured testing.

NO Unicode characters - ASCII only (see CLAUDE.md encoding rules).
"""

import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from tests.integration_scenarios.utils.scoring_system import (
    calculate_calibration_error,
    score_partial_correctness,
    score_proof_tree_quality,
    score_reasoning_coherence,
)


@dataclass
class ScenarioResult:
    """Result of a scenario test with multi-dimensional scoring"""

    scenario_name: str
    difficulty: str  # "medium", "hard", "very_hard", "extreme"
    domain: str  # "logic_puzzles", "dynamic_responses", etc.

    # Primary scores (0-100)
    reasoning_quality_score: float
    confidence_calibration_score: float
    correctness_score: float

    # Weighted overall score
    overall_score: float

    # Detailed metrics
    proof_tree_depth: int
    strategies_used: List[str]
    confidence_progression: List[Tuple[str, float]]
    execution_time_ms: int
    neo4j_query_count: int
    cache_hit_rate: float
    memory_peak_mb: float

    # Analysis
    reasoning_steps: List[str]
    identified_weaknesses: List[str]
    improvement_suggestions: List[str]

    # Raw data
    kai_response: str
    expected_response: Optional[str]
    proof_tree: Optional[Dict]
    full_trace: List[str]

    # Status
    passed: bool
    error: Optional[str]

    # Additional fields for convenience
    final_confidence: float = 0.0
    observations: List[str] = field(default_factory=list)


class ScenarioTestBase:
    """Base class for all scenario integration tests"""

    # Class attributes (override in subclasses)
    DIFFICULTY: str = "medium"
    DOMAIN: str = "unknown"
    TIMEOUT_SECONDS: int = 3600  # 1 hour default

    # Scoring weights (customize per domain)
    REASONING_QUALITY_WEIGHT: float = 0.5
    CONFIDENCE_CALIBRATION_WEIGHT: float = 0.3
    CORRECTNESS_WEIGHT: float = 0.2

    def setup_method(self):
        """Setup for each test - creates clean environment"""

    def teardown_method(self):
        """Cleanup after each test"""

    def run_scenario(
        self,
        input_text: str,
        expected_outputs: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        kai_worker=None,
        logger=None,
        progress_reporter=None,
        confidence_tracker=None,
    ) -> ScenarioResult:
        """
        Execute a scenario and return comprehensive results.

        Args:
            input_text: The query/puzzle/prompt to give to KAI
            expected_outputs: Optional dict with expected values for comparison
            context: Optional context (pre-learned facts, constraints, etc.)
            kai_worker: KaiWorker instance (uses fixture if not provided)
            logger: ScenarioLogger instance (uses fixture if not provided)
            progress_reporter: ProgressReporter instance (uses fixture if not provided)
            confidence_tracker: ConfidenceCalibrationTracker (uses fixture if not provided)

        Returns:
            ScenarioResult with scores, metrics, and analysis

        Raises:
            TimeoutError: If scenario exceeds TIMEOUT_SECONDS
            ValueError: If input_text is empty
        """
        if not input_text or not input_text.strip():
            raise ValueError("input_text cannot be empty")

        if expected_outputs is None:
            expected_outputs = {}
        if context is None:
            context = {}

        start_time = time.time()

        # Initialize result with default values
        result = ScenarioResult(
            scenario_name=self.__class__.__name__,
            difficulty=self.DIFFICULTY,
            domain=self.DOMAIN,
            reasoning_quality_score=0.0,
            confidence_calibration_score=0.0,
            correctness_score=0.0,
            overall_score=0.0,
            proof_tree_depth=0,
            strategies_used=[],
            confidence_progression=[],
            execution_time_ms=0,
            neo4j_query_count=0,
            cache_hit_rate=0.0,
            memory_peak_mb=0.0,
            reasoning_steps=[],
            identified_weaknesses=[],
            improvement_suggestions=[],
            kai_response="",
            expected_response=expected_outputs.get("answer", None),
            proof_tree=None,
            full_trace=[],
            passed=False,
            error=None,
        )

        try:
            # Log input if logger provided
            if logger:
                logger.log_input(input_text, context)

            # Report progress if progress_reporter provided
            if progress_reporter:
                progress_reporter.update("Executing scenario", 10)

            # Execute the scenario using kai_worker
            if kai_worker:
                # Get response from KAI
                response = self._execute_kai_worker(
                    kai_worker, input_text, context, logger, progress_reporter
                )
                result.kai_response = response.get("text", "")
                result.proof_tree = response.get("proof_tree", None)
                result.full_trace = response.get("trace", [])
                result.confidence_progression = response.get(
                    "confidence_progression", []
                )
                result.strategies_used = response.get("strategies", [])
                result.neo4j_query_count = response.get("neo4j_queries", 0)
                result.cache_hit_rate = response.get("cache_hit_rate", 0.0)
                result.memory_peak_mb = response.get("memory_peak_mb", 0.0)

                # Extract final confidence
                if result.confidence_progression:
                    result.final_confidence = result.confidence_progression[-1][1]

            # Report progress
            if progress_reporter:
                progress_reporter.update("Analyzing results", 50)

            # Convert ProofTree object to dict if needed (for compatibility with scoring methods)
            proof_tree_for_scoring = result.proof_tree or {}
            if hasattr(result.proof_tree, "to_dict"):
                proof_tree_for_scoring = result.proof_tree.to_dict()

            # Extract reasoning steps from proof tree FIRST (needed for scoring)
            if result.proof_tree:
                result.proof_tree_depth = self._calculate_proof_tree_depth(
                    proof_tree_for_scoring
                )
                result.reasoning_steps = self._extract_reasoning_steps(
                    proof_tree_for_scoring
                )

            # Score reasoning quality (use extracted reasoning_steps, not full_trace)
            result.reasoning_quality_score = self.score_reasoning_quality(
                proof_tree_for_scoring,
                result.strategies_used,
                result.reasoning_steps if result.reasoning_steps else result.full_trace,
            )

            # Score confidence calibration
            if result.confidence_progression:
                # For now, assume binary correctness based on expected outputs
                confidence_values = [conf for _, conf in result.confidence_progression]
                # Simplified: compare final answer to expected
                is_correct = self._is_answer_correct(
                    result.kai_response, expected_outputs
                )
                correctness_values = [is_correct] * len(confidence_values)
                result.confidence_calibration_score = self.score_confidence_calibration(
                    confidence_values, correctness_values
                )
            else:
                # No confidence data, neutral score
                result.confidence_calibration_score = 50.0

            # Score correctness
            result.correctness_score = self.score_correctness(
                result.kai_response,
                expected_outputs,
                allow_partial=True,
            )

            # Calculate overall score
            result.overall_score = (
                result.reasoning_quality_score * self.REASONING_QUALITY_WEIGHT
                + result.confidence_calibration_score
                * self.CONFIDENCE_CALIBRATION_WEIGHT
                + result.correctness_score * self.CORRECTNESS_WEIGHT
            )

            # Identify weaknesses and suggestions
            result.identified_weaknesses = self._identify_weaknesses(result)
            result.improvement_suggestions = self._generate_suggestions(result)

            # Mark as passed if overall score meets threshold
            threshold = self._get_pass_threshold()
            result.passed = result.overall_score >= threshold

            # Track confidence calibration if tracker provided
            if confidence_tracker and result.final_confidence > 0:
                is_correct = result.correctness_score >= 80
                confidence_tracker.add_prediction(result.final_confidence, is_correct)

            # Report progress
            if progress_reporter:
                progress_reporter.update("Complete", 100)
                progress_reporter.complete(success=result.passed)

        except TimeoutError as e:
            result.error = f"Timeout after {self.TIMEOUT_SECONDS}s: {str(e)}"
            result.passed = False
        except Exception as e:
            result.error = f"Error: {str(e)}\n{traceback.format_exc()}"
            result.passed = False

        # Calculate execution time
        end_time = time.time()
        result.execution_time_ms = int((end_time - start_time) * 1000)

        return result

    def score_reasoning_quality(
        self,
        proof_tree: Dict,
        strategies_used: List[str],
        reasoning_steps: List[str],
    ) -> float:
        """
        Score the quality of KAI's reasoning process (0-100).

        Args:
            proof_tree: ProofTree dictionary from KAI
            strategies_used: List of reasoning strategies used
            reasoning_steps: List of reasoning step descriptions

        Returns:
            Score 0-100
        """
        if not proof_tree:
            # No proof tree, minimal score
            return 20.0

        # Get expected depth range for this difficulty
        depth_range = self._get_expected_depth_range()

        # Score proof tree quality
        tree_score, observations = score_proof_tree_quality(
            proof_tree, depth_range, None, self.DOMAIN
        )

        # Score reasoning coherence
        coherence_score, issues = score_reasoning_coherence(reasoning_steps, {})

        # Combine scores (70% tree quality, 30% coherence)
        final_score = tree_score * 0.7 + coherence_score * 0.3

        return final_score

    def score_confidence_calibration(
        self,
        confidence_values: List[float],
        correctness_values: List[bool],
    ) -> float:
        """
        Score how well confidence aligns with correctness (0-100).

        Args:
            confidence_values: List of confidence scores (0-1)
            correctness_values: List of boolean correctness values

        Returns:
            Score 0-100 (higher = better calibration)
        """
        if not confidence_values or not correctness_values:
            return 50.0  # Neutral score if no data

        if len(confidence_values) != len(correctness_values):
            # Mismatch in lengths, use minimum
            min_len = min(len(confidence_values), len(correctness_values))
            confidence_values = confidence_values[:min_len]
            correctness_values = correctness_values[:min_len]

        # Calculate calibration metrics
        calibration_metrics = calculate_calibration_error(
            confidence_values, correctness_values
        )

        # Convert ECE (0-1, lower is better) to score (0-100, higher is better)
        ece = calibration_metrics.get("ece", 0.5)
        score = (1.0 - ece) * 100.0

        return score

    def score_correctness(
        self,
        actual: Any,
        expected: Any,
        allow_partial: bool = True,
    ) -> float:
        """
        Score correctness with partial credit (0-100).

        Args:
            actual: Actual output from KAI
            expected: Expected output dictionary
            allow_partial: Whether to give partial credit

        Returns:
            Score 0-100
        """
        if not expected:
            # No expected output to compare against
            return 50.0  # Neutral score

        score, explanation = score_partial_correctness(actual, expected, self.DOMAIN)
        return score

    def log_progress(self, message: str, percentage: float):
        """
        Log visible progress during long-running tests.

        Args:
            message: Progress message
            percentage: Progress percentage (0-100)
        """
        print(f"[PROGRESS] {percentage:.1f}% - {message}")

    # Helper methods

    def _execute_kai_worker(
        self, kai_worker, input_text: str, context: Dict, logger, progress_reporter
    ) -> Dict[str, Any]:
        """
        Execute KAI worker and extract response with instrumentation.

        Returns:
            Dict with keys: text, proof_tree, trace, confidence_progression,
                           strategies, neo4j_queries, cache_hit_rate, memory_peak_mb
        """

        from tests.integration_scenarios.utils.reasoning_analyzer import (
            ReasoningAnalyzer,
        )

        # Storage for captured data
        worker_result = {
            "text": "",
            "proof_tree": None,
            "trace": [],
            "confidence_progression": [],
            "strategies": [],
            "neo4j_queries": 0,
            "cache_hit_rate": 0.0,
            "memory_peak_mb": 0.0,
            "completed": False,
            "error": None,
        }

        # Signal handlers to capture KAI worker output
        def on_response_ready(response_data):
            """Capture response from KAI worker"""
            try:
                # response_data could be a KaiResponse object or dict
                if hasattr(response_data, "answer"):
                    worker_result["text"] = response_data.answer
                elif isinstance(response_data, dict):
                    worker_result["text"] = response_data.get("answer", "")
                else:
                    worker_result["text"] = str(response_data)

                # Extract confidence (default to 0.0 if None or missing)
                final_confidence = 0.0
                if (
                    hasattr(response_data, "confidence")
                    and response_data.confidence is not None
                ):
                    final_confidence = response_data.confidence
                    worker_result["confidence_progression"].append(
                        ("final", final_confidence)
                    )

                # Extract proof_tree if available
                if (
                    hasattr(response_data, "proof_tree")
                    and response_data.proof_tree is not None
                ):
                    worker_result["proof_tree"] = response_data.proof_tree

                # Extract strategy if available
                if hasattr(response_data, "strategy") and response_data.strategy:
                    strategy = response_data.strategy
                    if strategy not in worker_result["strategies"]:
                        worker_result["strategies"].append(strategy)

                # Extract trace if available
                if hasattr(response_data, "trace") and response_data.trace:
                    worker_result["trace"].extend(response_data.trace)

                if logger:
                    logger.log_final_response(
                        worker_result["text"],
                        final_confidence,
                    )
            except Exception as e:
                worker_result["error"] = f"Error capturing response: {e}"

        def on_proof_tree_updated(proof_tree):
            """Capture ProofTree updates"""
            try:
                worker_result["proof_tree"] = proof_tree

                if logger:
                    # Convert ProofTree to dict for logging
                    if hasattr(proof_tree, "to_dict"):
                        proof_dict = proof_tree.to_dict()
                    else:
                        proof_dict = proof_tree
                    logger.log_proof_tree_update(proof_dict)
            except Exception as e:
                worker_result["error"] = f"Error capturing proof tree: {e}"

        def on_reasoning_step(step_info):
            """Capture reasoning steps"""
            try:
                # Extract information from reasoning step
                if isinstance(step_info, dict):
                    strategy = step_info.get("strategy", "unknown")
                    action = step_info.get("action", "")
                    confidence = step_info.get("confidence", 0.0)

                    # Ensure confidence is not None
                    if confidence is None:
                        confidence = 0.0

                    worker_result["trace"].append(
                        f"[{strategy}] {action} (conf: {confidence:.2f})"
                    )

                    # Track strategies used
                    if strategy not in worker_result["strategies"]:
                        worker_result["strategies"].append(strategy)

                    # Track confidence progression (only if valid)
                    if confidence is not None:
                        worker_result["confidence_progression"].append(
                            (action, confidence)
                        )

                    if logger:
                        step_num = len(worker_result["trace"])
                        logger.log_reasoning_step(
                            step_num, strategy, action, "", confidence
                        )
            except Exception as e:
                worker_result["error"] = f"Error capturing reasoning step: {e}"

        def on_finished(response=None):
            """
            Mark execution as complete.

            Args:
                response: Optional KaiResponse object emitted by finished signal
            """
            # If response is provided, extract it (finished signal sends KaiResponse)
            if response:
                on_response_ready(response)
            worker_result["completed"] = True

        def on_error(error_msg):
            """
            Capture errors from KAI worker.

            Args:
                error_msg: Error message or exception
            """
            try:
                error_text = str(error_msg)
                worker_result["error"] = f"KAI worker error: {error_text}"
                worker_result["completed"] = True  # Mark as complete even on error
            except Exception as e:
                worker_result["error"] = f"Error capturing error: {e}"
                worker_result["completed"] = True

        # Connect signals if kai_worker has signals
        if hasattr(kai_worker, "signals"):
            signals = kai_worker.signals

            # Connect to signals
            if hasattr(signals, "response_ready"):
                signals.response_ready.connect(on_response_ready)
            if hasattr(signals, "proof_tree_updated"):
                signals.proof_tree_updated.connect(on_proof_tree_updated)
            if hasattr(signals, "reasoning_step"):
                signals.reasoning_step.connect(on_reasoning_step)
            if hasattr(signals, "finished"):
                signals.finished.connect(on_finished)
            if hasattr(signals, "error"):
                signals.error.connect(on_error)

        # Process input with KAI worker
        try:
            if progress_reporter:
                progress_reporter.update("Executing KAI worker", 30)

            # Execute the worker in a separate thread with timeout
            # This fixes the issue where process_query blocks and timeout is never checked
            def run_query():
                try:
                    kai_worker.process_query(input_text)
                except Exception as e:
                    worker_result["error"] = f"process_query exception: {e}"

            thread = threading.Thread(target=run_query, daemon=True)
            thread.start()
            thread.join(timeout=self.TIMEOUT_SECONDS)

            if thread.is_alive():
                # Thread didn't complete in time - the daemon thread will be
                # terminated when the main process exits
                raise TimeoutError(
                    f"KAI worker exceeded timeout of {self.TIMEOUT_SECONDS}s"
                )

            # Check if completed via signal
            if not worker_result["completed"]:
                # process_query returned but no finished signal - check for errors
                if worker_result["error"]:
                    raise RuntimeError(worker_result["error"])
                else:
                    raise RuntimeError(
                        "process_query completed but finished signal not emitted"
                    )

            if progress_reporter:
                progress_reporter.update("KAI worker completed", 70)

        except TimeoutError:
            raise
        except Exception as e:
            # Re-raise any errors that occurred
            raise RuntimeError(f"KAI worker execution failed: {e}")

        finally:
            # Disconnect signals
            signals = getattr(kai_worker, "signals", None)
            if signals is not None:

                def safe_disconnect(signal_name: str, slot) -> None:
                    sig = getattr(signals, signal_name, None)
                    if sig is None:
                        return
                    try:
                        sig.disconnect(slot)
                    except TypeError:
                        # Usually raised when `slot` isn't currently connected to `sig`
                        pass

                safe_disconnect("response_ready", on_response_ready)
                safe_disconnect("proof_tree_updated", on_proof_tree_updated)
                safe_disconnect("reasoning_step", on_reasoning_step)
                safe_disconnect("finished", on_finished)

        # Extract additional strategies from proof tree if available
        # ALWAYS extract from ProofTree and merge with existing strategies
        if worker_result["proof_tree"]:
            try:
                # Convert proof_tree to dict if needed
                if hasattr(worker_result["proof_tree"], "to_dict"):
                    proof_dict = worker_result["proof_tree"].to_dict()
                else:
                    proof_dict = worker_result["proof_tree"]

                # Use ReasoningAnalyzer to extract strategies from ProofTree
                analyzer = ReasoningAnalyzer(proof_dict, worker_result["trace"])
                proof_tree_strategies = analyzer.extract_reasoning_strategies()

                # Merge with existing strategies (remove duplicates, preserve order)
                existing = worker_result["strategies"]
                for strategy in proof_tree_strategies:
                    if strategy not in existing:
                        existing.append(strategy)
                worker_result["strategies"] = existing
            except Exception as e:
                # Non-critical - continue without strategy extraction
                pass

        # Extract performance metrics (if available from worker)
        get_metrics = getattr(kai_worker, "get_performance_metrics", None)
        if callable(get_metrics):
            try:
                metrics = get_metrics()
            except (TypeError, RuntimeError, AttributeError, ValueError):
                # Only swallow expected/benign failures; adjust this tuple to what you observe.
                pass
            else:
                if isinstance(metrics, dict):
                    worker_result["neo4j_queries"] = metrics.get("neo4j_queries", 0)
                    worker_result["cache_hit_rate"] = metrics.get("cache_hit_rate", 0.0)
                    worker_result["memory_peak_mb"] = metrics.get("memory_peak_mb", 0.0)

        return worker_result

    def _is_answer_correct(
        self, actual_response: str, expected_outputs: Dict[str, Any]
    ) -> bool:
        """Check if actual response matches expected outputs (binary)"""
        if not expected_outputs:
            return False

        # Simple substring check for now
        expected_answer = expected_outputs.get("answer", "")
        if isinstance(expected_answer, str):
            return expected_answer.lower() in actual_response.lower()

        return False

    def _calculate_proof_tree_depth(self, proof_tree: Dict) -> int:
        """
        Calculate maximum depth of proof tree.

        Handles both ProofTree format (root_steps + subgoals) and
        legacy format (children). Backward compatible.
        """
        if not proof_tree:
            return 1

        # Handle ProofTree format: {"root_steps": [...]}
        if "root_steps" in proof_tree:
            root_steps = proof_tree["root_steps"]
            if not root_steps:
                return 1

            # Calculate depth for each root step
            max_depth = 0
            for step in root_steps:
                step_depth = self._calculate_step_depth(step)
                max_depth = max(max_depth, step_depth)

            return max_depth

        # Handle legacy format: {"children": [...]}
        if "children" in proof_tree:
            if not proof_tree["children"]:
                return 1

            max_child_depth = max(
                self._calculate_proof_tree_depth(child)
                for child in proof_tree["children"]
            )
            return 1 + max_child_depth

        # Single node with no children/root_steps
        return 1

    def _calculate_step_depth(self, step: Dict, current_depth: int = 1) -> int:
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
            subgoal_depth = self._calculate_step_depth(subgoal, current_depth + 1)
            max_subgoal_depth = max(max_subgoal_depth, subgoal_depth)

        return max_subgoal_depth

    def _extract_reasoning_steps(self, proof_tree: Dict) -> List[str]:
        """
        Extract reasoning steps from proof tree.

        Handles both ProofTree format (root_steps) and legacy format (children).
        Backward compatible.
        """
        steps = []

        def traverse_proof_tree(tree_dict, depth=0):
            """Traverse ProofTree format"""
            if "root_steps" in tree_dict:
                for step in tree_dict["root_steps"]:
                    traverse_proof_step(step, depth)

        def traverse_proof_step(step_dict, depth=0):
            """Traverse ProofStep format"""
            if not isinstance(step_dict, dict):
                return

            # Extract step information
            rule_name = step_dict.get("rule_name", "")
            explanation = step_dict.get("explanation_text", "")
            step_type = step_dict.get("step_type", "")

            # Format step description
            if explanation:
                step_desc = f"{'  ' * depth}[{rule_name}] {explanation}"
            elif rule_name:
                step_desc = f"{'  ' * depth}{rule_name}"
            else:
                step_desc = f"{'  ' * depth}{step_type}"

            steps.append(step_desc)

            # Traverse subgoals
            subgoals = step_dict.get("subgoals", [])
            for subgoal in subgoals:
                traverse_proof_step(subgoal, depth + 1)

        def traverse_legacy(node, depth=0):
            """Traverse legacy format"""
            if "step" in node:
                steps.append(f"{'  ' * depth}{node['step']}")
            if "children" in node:
                for child in node["children"]:
                    traverse_legacy(child, depth + 1)

        # Try ProofTree format first
        if "root_steps" in proof_tree:
            traverse_proof_tree(proof_tree)
        else:
            # Fallback to legacy format
            traverse_legacy(proof_tree)

        return steps

    def _identify_weaknesses(self, result: ScenarioResult) -> List[str]:
        """Identify weaknesses from scenario result"""
        weaknesses = []

        if result.reasoning_quality_score < 40:
            weaknesses.append("Low reasoning quality score")

        if result.confidence_calibration_score < 50:
            weaknesses.append("Poor confidence calibration")

        if result.correctness_score < 30:
            weaknesses.append("Low correctness score")

        if not result.strategies_used:
            weaknesses.append("No reasoning strategies identified")

        if result.proof_tree_depth < 2:
            weaknesses.append("Proof tree too shallow")

        return weaknesses

    def _generate_suggestions(self, result: ScenarioResult) -> List[str]:
        """Generate improvement suggestions from result"""
        suggestions = []

        if result.reasoning_quality_score < 40:
            suggestions.append("Improve reasoning strategy selection")

        if result.confidence_calibration_score < 50:
            suggestions.append("Calibrate confidence scores better")

        if not result.strategies_used:
            suggestions.append("Enable more reasoning engines")

        if result.proof_tree_depth < 2:
            suggestions.append("Increase reasoning depth")

        return suggestions

    def _get_expected_depth_range(self) -> Tuple[int, int]:
        """Get expected proof tree depth range for this difficulty"""
        depth_ranges = {
            "medium": (2, 6),
            "hard": (4, 10),
            "very_hard": (6, 15),
            "extreme": (8, 20),
        }
        return depth_ranges.get(self.DIFFICULTY, (2, 10))

    def _get_pass_threshold(self) -> float:
        """Get pass threshold score for this difficulty"""
        thresholds = {
            "medium": 50.0,
            "hard": 40.0,
            "very_hard": 30.0,
            "extreme": 20.0,
        }
        return thresholds.get(self.DIFFICULTY, 50.0)
