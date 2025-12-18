"""
tests/integration_scenarios/utils/confidence_calibrator.py

Track and evaluate confidence calibration across scenarios.
Provides ConfidenceCalibrationTracker class for analyzing confidence vs. correctness.

NO Unicode characters - ASCII only (see CLAUDE.md encoding rules).
"""

import math
from typing import Dict, List, Tuple


class ConfidenceCalibrationTracker:
    """Tracks confidence values vs. actual correctness for calibration analysis"""

    def __init__(self):
        """Initialize confidence calibration tracker"""
        self.data_points: List[Tuple[float, bool]] = []  # (confidence, was_correct)

    def add_prediction(self, confidence: float, was_correct: bool):
        """
        Record a prediction with its confidence and actual correctness.

        Args:
            confidence: Confidence value (0-1)
            was_correct: Whether the prediction was correct
        """
        if not (0.0 <= confidence <= 1.0):
            # Clamp to valid range
            confidence = max(0.0, min(1.0, confidence))

        self.data_points.append((confidence, was_correct))

    def calculate_calibration_metrics(self) -> Dict[str, float]:
        """
        Calculate comprehensive calibration metrics.

        Returns:
            {
                "expected_calibration_error": float,  # 0-1, lower is better
                "maximum_calibration_error": float,  # 0-1, lower is better
                "brier_score": float,  # 0-1, lower is better
                "log_loss": float,  # 0-inf, lower is better
                "reliability": float,  # 0-1, higher is better
                "resolution": float,  # 0-1, higher is better
                "uncertainty": float,  # 0-1, lower is better
                "calibration_score": float  # 0-100 composite score
            }
        """
        if not self.data_points:
            return {
                "expected_calibration_error": 0.5,
                "maximum_calibration_error": 0.5,
                "brier_score": 0.5,
                "log_loss": 1.0,
                "reliability": 0.5,
                "resolution": 0.0,
                "uncertainty": 0.5,
                "calibration_score": 50.0,
            }

        n = len(self.data_points)
        confidences = [conf for conf, _ in self.data_points]
        correctness = [correct for _, correct in self.data_points]

        # Calculate ECE (Expected Calibration Error)
        ece = self._calculate_ece(confidences, correctness)

        # Calculate MCE (Maximum Calibration Error)
        mce = self._calculate_mce(confidences, correctness)

        # Calculate Brier score
        brier_score = self._calculate_brier_score(confidences, correctness)

        # Calculate log loss
        log_loss = self._calculate_log_loss(confidences, correctness)

        # Calculate reliability (how often high confidence = correct)
        reliability = self._calculate_reliability(confidences, correctness)

        # Calculate resolution (ability to distinguish correct from incorrect)
        resolution = self._calculate_resolution(confidences, correctness)

        # Calculate uncertainty (overall prediction difficulty)
        uncertainty = sum(correctness) / n  # Base rate

        # Calculate composite calibration score (0-100)
        # Lower ECE, lower Brier, higher reliability = better score
        calibration_score = (
            (1.0 - ece) * 40.0  # 40% weight on ECE
            + (1.0 - brier_score) * 30.0  # 30% weight on Brier
            + reliability * 30.0  # 30% weight on reliability
        )

        return {
            "expected_calibration_error": ece,
            "maximum_calibration_error": mce,
            "brier_score": brier_score,
            "log_loss": log_loss,
            "reliability": reliability,
            "resolution": resolution,
            "uncertainty": uncertainty,
            "calibration_score": calibration_score,
        }

    def get_calibration_plot_data(
        self,
        num_bins: int = 10,
    ) -> Dict[str, List[float]]:
        """
        Generate data for calibration plot.

        Args:
            num_bins: Number of bins for grouping predictions

        Returns:
            {
                "bin_centers": List[float],
                "bin_accuracies": List[float],
                "bin_confidences": List[float],
                "bin_counts": List[int]
            }
        """
        if not self.data_points:
            return {
                "bin_centers": [],
                "bin_accuracies": [],
                "bin_confidences": [],
                "bin_counts": [],
            }

        # Create bins
        bins = [[] for _ in range(num_bins)]

        # Assign data points to bins
        for conf, correct in self.data_points:
            bin_idx = min(int(conf * num_bins), num_bins - 1)
            bins[bin_idx].append((conf, correct))

        # Calculate bin statistics
        bin_centers = []
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []

        for i, bin_data in enumerate(bins):
            if bin_data:
                # Calculate bin center
                center = (i + 0.5) / num_bins
                bin_centers.append(center)

                # Calculate average confidence in bin
                avg_conf = sum(conf for conf, _ in bin_data) / len(bin_data)
                bin_confidences.append(avg_conf)

                # Calculate accuracy in bin
                accuracy = sum(1 for _, correct in bin_data if correct) / len(bin_data)
                bin_accuracies.append(accuracy)

                # Count
                bin_counts.append(len(bin_data))
            else:
                # Empty bin
                bin_centers.append((i + 0.5) / num_bins)
                bin_confidences.append(0.0)
                bin_accuracies.append(0.0)
                bin_counts.append(0)

        return {
            "bin_centers": bin_centers,
            "bin_accuracies": bin_accuracies,
            "bin_confidences": bin_confidences,
            "bin_counts": bin_counts,
        }

    def identify_overconfidence_underconfidence(self) -> Dict[str, any]:
        """
        Identify systematic over/under-confidence patterns.

        Returns:
            {
                "is_overconfident": bool,
                "is_underconfident": bool,
                "bias": float,  # positive = overconfident, negative = underconfident
                "problematic_confidence_ranges": List[Tuple[float, float]]
            }
        """
        if not self.data_points:
            return {
                "is_overconfident": False,
                "is_underconfident": False,
                "bias": 0.0,
                "problematic_confidence_ranges": [],
            }

        # Get calibration plot data
        plot_data = self.get_calibration_plot_data(num_bins=10)

        # Calculate overall bias
        total_bias = 0.0
        count = 0

        for conf, acc, cnt in zip(
            plot_data["bin_confidences"],
            plot_data["bin_accuracies"],
            plot_data["bin_counts"],
        ):
            if cnt > 0:
                bias = conf - acc  # Positive = overconfident, negative = underconfident
                total_bias += bias * cnt
                count += cnt

        avg_bias = total_bias / count if count > 0 else 0.0

        # Identify problematic ranges (where |bias| > 0.15)
        problematic_ranges = []
        for i, (conf, acc, cnt) in enumerate(
            zip(
                plot_data["bin_confidences"],
                plot_data["bin_accuracies"],
                plot_data["bin_counts"],
            )
        ):
            if cnt > 0:
                bias = conf - acc
                if abs(bias) > 0.15:
                    # This bin has significant miscalibration
                    bin_start = i / 10.0
                    bin_end = (i + 1) / 10.0
                    problematic_ranges.append((bin_start, bin_end))

        return {
            "is_overconfident": avg_bias > 0.05,
            "is_underconfident": avg_bias < -0.05,
            "bias": avg_bias,
            "problematic_confidence_ranges": problematic_ranges,
        }

    # Private helper methods

    def _calculate_ece(
        self, confidences: List[float], correctness: List[bool], num_bins: int = 10
    ) -> float:
        """Calculate Expected Calibration Error"""
        if not confidences:
            return 0.5

        n = len(confidences)
        bins = [[] for _ in range(num_bins)]

        # Assign to bins
        for conf, correct in zip(confidences, correctness):
            bin_idx = min(int(conf * num_bins), num_bins - 1)
            bins[bin_idx].append((conf, correct))

        # Calculate ECE
        ece = 0.0
        for bin_data in bins:
            if bin_data:
                bin_conf = sum(conf for conf, _ in bin_data) / len(bin_data)
                bin_acc = sum(1 for _, correct in bin_data if correct) / len(bin_data)
                ece += (len(bin_data) / n) * abs(bin_acc - bin_conf)

        return ece

    def _calculate_mce(
        self, confidences: List[float], correctness: List[bool], num_bins: int = 10
    ) -> float:
        """Calculate Maximum Calibration Error"""
        if not confidences:
            return 0.5

        bins = [[] for _ in range(num_bins)]

        # Assign to bins
        for conf, correct in zip(confidences, correctness):
            bin_idx = min(int(conf * num_bins), num_bins - 1)
            bins[bin_idx].append((conf, correct))

        # Calculate MCE
        mce = 0.0
        for bin_data in bins:
            if bin_data:
                bin_conf = sum(conf for conf, _ in bin_data) / len(bin_data)
                bin_acc = sum(1 for _, correct in bin_data if correct) / len(bin_data)
                mce = max(mce, abs(bin_acc - bin_conf))

        return mce

    def _calculate_brier_score(
        self, confidences: List[float], correctness: List[bool]
    ) -> float:
        """Calculate Brier score"""
        if not confidences:
            return 0.5

        n = len(confidences)
        brier = (
            sum(
                (conf - (1.0 if correct else 0.0)) ** 2
                for conf, correct in zip(confidences, correctness)
            )
            / n
        )

        return brier

    def _calculate_log_loss(
        self, confidences: List[float], correctness: List[bool]
    ) -> float:
        """Calculate log loss (cross-entropy loss)"""
        if not confidences:
            return 1.0

        n = len(confidences)
        epsilon = 1e-15  # To avoid log(0)

        log_loss = 0.0
        for conf, correct in zip(confidences, correctness):
            # Clip confidence to avoid log(0)
            conf = max(epsilon, min(1.0 - epsilon, conf))

            if correct:
                log_loss -= math.log(conf)
            else:
                log_loss -= math.log(1.0 - conf)

        return log_loss / n

    def _calculate_reliability(
        self, confidences: List[float], correctness: List[bool]
    ) -> float:
        """
        Calculate reliability (how often high confidence predictions are correct).
        Returns value 0-1 (higher is better).
        """
        if not confidences:
            return 0.5

        # Focus on high-confidence predictions (>0.7)
        high_conf_correct = sum(
            1
            for conf, correct in zip(confidences, correctness)
            if conf > 0.7 and correct
        )
        high_conf_total = sum(1 for conf in confidences if conf > 0.7)

        if high_conf_total > 0:
            reliability = high_conf_correct / high_conf_total
        else:
            # No high-confidence predictions, use overall accuracy
            reliability = sum(correctness) / len(correctness)

        return reliability

    def _calculate_resolution(
        self, confidences: List[float], correctness: List[bool]
    ) -> float:
        """
        Calculate resolution (ability to distinguish correct from incorrect).
        Returns value 0-1 (higher is better).
        """
        if not confidences:
            return 0.0

        # Calculate mean confidence for correct and incorrect predictions
        correct_confs = [
            conf for conf, correct in zip(confidences, correctness) if correct
        ]
        incorrect_confs = [
            conf for conf, correct in zip(confidences, correctness) if not correct
        ]

        if not correct_confs or not incorrect_confs:
            # Can't calculate resolution without both categories
            return 0.0

        mean_correct = sum(correct_confs) / len(correct_confs)
        mean_incorrect = sum(incorrect_confs) / len(incorrect_confs)

        # Resolution is the difference between mean confidences
        # Normalize to 0-1 range
        resolution = abs(mean_correct - mean_incorrect)

        return resolution
