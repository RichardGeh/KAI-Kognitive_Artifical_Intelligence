"""
tests/integration_scenarios/utils/test_confidence_calibrator.py

Unit tests for confidence_calibrator.py module.
Tests ConfidenceCalibrationTracker class with known confidence/correctness pairs.

NO Unicode characters - ASCII only (see CLAUDE.md encoding rules).
"""

from tests.integration_scenarios.utils.confidence_calibrator import (
    ConfidenceCalibrationTracker,
)


class TestConfidenceCalibrationTracker:
    """Test ConfidenceCalibrationTracker class"""

    def test_initialization(self):
        """Test tracker initialization"""
        tracker = ConfidenceCalibrationTracker()
        assert tracker.data_points == []

    def test_add_prediction(self):
        """Test adding prediction data"""
        tracker = ConfidenceCalibrationTracker()
        tracker.add_prediction(0.8, True)
        tracker.add_prediction(0.6, False)

        assert len(tracker.data_points) == 2
        assert tracker.data_points[0] == (0.8, True)
        assert tracker.data_points[1] == (0.6, False)

    def test_add_prediction_clamps_confidence(self):
        """Test that out-of-range confidence values are clamped"""
        tracker = ConfidenceCalibrationTracker()
        tracker.add_prediction(1.5, True)  # >1.0
        tracker.add_prediction(-0.3, False)  # <0.0

        assert tracker.data_points[0][0] == 1.0  # Clamped to 1.0
        assert tracker.data_points[1][0] == 0.0  # Clamped to 0.0

    def test_calculate_metrics_empty(self):
        """Test metrics calculation with no data"""
        tracker = ConfidenceCalibrationTracker()
        metrics = tracker.calculate_calibration_metrics()

        assert metrics["expected_calibration_error"] == 0.5
        assert metrics["calibration_score"] == 50.0

    def test_calculate_metrics_perfect_calibration(self):
        """Test metrics with perfect calibration"""
        tracker = ConfidenceCalibrationTracker()
        # Add perfectly calibrated predictions
        for _ in range(8):
            tracker.add_prediction(0.8, True)
        for _ in range(2):
            tracker.add_prediction(0.8, False)

        metrics = tracker.calculate_calibration_metrics()

        assert metrics["expected_calibration_error"] < 0.1  # Very low ECE
        assert metrics["calibration_score"] > 85.0  # High calibration score

    def test_calculate_metrics_overconfident(self):
        """Test metrics with overconfident predictions"""
        tracker = ConfidenceCalibrationTracker()
        # High confidence, low accuracy
        for _ in range(5):
            tracker.add_prediction(0.9, True)
        for _ in range(5):
            tracker.add_prediction(0.9, False)  # 50% correct with 90% confidence

        metrics = tracker.calculate_calibration_metrics()

        assert metrics["expected_calibration_error"] > 0.2  # High ECE
        assert metrics["calibration_score"] < 70.0  # Low calibration score

    def test_calculate_metrics_underconfident(self):
        """Test metrics with underconfident predictions"""
        tracker = ConfidenceCalibrationTracker()
        # Low confidence, high accuracy
        for _ in range(9):
            tracker.add_prediction(0.5, True)
        for _ in range(1):
            tracker.add_prediction(0.5, False)  # 90% correct with 50% confidence

        metrics = tracker.calculate_calibration_metrics()

        assert metrics["expected_calibration_error"] > 0.2  # High ECE
        assert metrics["calibration_score"] < 70.0  # Low calibration score

    def test_calculate_metrics_brier_score(self):
        """Test Brier score calculation"""
        tracker = ConfidenceCalibrationTracker()
        # Perfect predictions
        tracker.add_prediction(1.0, True)
        tracker.add_prediction(0.0, False)

        metrics = tracker.calculate_calibration_metrics()
        assert metrics["brier_score"] < 0.1  # Very low Brier score

        # Poor predictions
        tracker2 = ConfidenceCalibrationTracker()
        tracker2.add_prediction(1.0, False)
        tracker2.add_prediction(0.0, True)

        metrics2 = tracker2.calculate_calibration_metrics()
        assert metrics2["brier_score"] > 0.9  # Very high Brier score

    def test_calculate_metrics_log_loss(self):
        """Test log loss calculation"""
        tracker = ConfidenceCalibrationTracker()
        # Good predictions
        tracker.add_prediction(0.9, True)
        tracker.add_prediction(0.1, False)

        metrics = tracker.calculate_calibration_metrics()
        assert metrics["log_loss"] < 0.5  # Low log loss

    def test_calculate_metrics_reliability(self):
        """Test reliability calculation"""
        tracker = ConfidenceCalibrationTracker()
        # High confidence predictions that are mostly correct
        for _ in range(9):
            tracker.add_prediction(0.9, True)
        for _ in range(1):
            tracker.add_prediction(0.9, False)

        metrics = tracker.calculate_calibration_metrics()
        assert metrics["reliability"] > 0.85  # High reliability

    def test_calculate_metrics_resolution(self):
        """Test resolution calculation"""
        tracker = ConfidenceCalibrationTracker()
        # Clear distinction between correct and incorrect predictions
        for _ in range(5):
            tracker.add_prediction(0.9, True)  # High confidence, correct
        for _ in range(5):
            tracker.add_prediction(0.3, False)  # Low confidence, incorrect

        metrics = tracker.calculate_calibration_metrics()
        assert metrics["resolution"] > 0.5  # Good resolution

    def test_get_calibration_plot_data_empty(self):
        """Test plot data generation with no data"""
        tracker = ConfidenceCalibrationTracker()
        plot_data = tracker.get_calibration_plot_data()

        assert plot_data["bin_centers"] == []
        assert plot_data["bin_accuracies"] == []

    def test_get_calibration_plot_data_simple(self):
        """Test plot data generation"""
        tracker = ConfidenceCalibrationTracker()
        # Add data in bin 0.8-0.9
        for _ in range(8):
            tracker.add_prediction(0.85, True)
        for _ in range(2):
            tracker.add_prediction(0.85, False)

        plot_data = tracker.get_calibration_plot_data(num_bins=10)

        assert len(plot_data["bin_centers"]) == 10
        assert len(plot_data["bin_accuracies"]) == 10
        assert len(plot_data["bin_confidences"]) == 10
        assert len(plot_data["bin_counts"]) == 10

        # Bin 8 (0.8-0.9) should have data
        bin_8_count = plot_data["bin_counts"][8]
        assert bin_8_count == 10

        # Bin 8 accuracy should be 0.8 (8/10 correct)
        bin_8_acc = plot_data["bin_accuracies"][8]
        assert 0.75 < bin_8_acc < 0.85

    def test_get_calibration_plot_data_multiple_bins(self):
        """Test plot data with data in multiple bins"""
        tracker = ConfidenceCalibrationTracker()
        # Add data to different bins
        tracker.add_prediction(0.1, False)
        tracker.add_prediction(0.5, True)
        tracker.add_prediction(0.9, True)

        plot_data = tracker.get_calibration_plot_data(num_bins=10)

        # Should have data in bins 1, 5, and 9
        assert plot_data["bin_counts"][1] >= 1
        assert plot_data["bin_counts"][5] >= 1
        assert plot_data["bin_counts"][9] >= 1

    def test_identify_overconfidence_underconfidence_neutral(self):
        """Test over/underconfidence identification - neutral"""
        tracker = ConfidenceCalibrationTracker()
        # Perfectly calibrated
        for _ in range(7):
            tracker.add_prediction(0.7, True)
        for _ in range(3):
            tracker.add_prediction(0.7, False)

        result = tracker.identify_overconfidence_underconfidence()

        # Should be neither overconfident nor underconfident
        assert not result["is_overconfident"]
        assert not result["is_underconfident"]
        assert abs(result["bias"]) < 0.05

    def test_identify_overconfidence_underconfidence_overconfident(self):
        """Test over/underconfidence identification - overconfident"""
        tracker = ConfidenceCalibrationTracker()
        # High confidence, low accuracy
        for _ in range(5):
            tracker.add_prediction(0.9, True)
        for _ in range(5):
            tracker.add_prediction(0.9, False)  # Only 50% correct

        result = tracker.identify_overconfidence_underconfidence()

        assert result["is_overconfident"]
        assert not result["is_underconfident"]
        assert result["bias"] > 0.05  # Positive bias = overconfident

    def test_identify_overconfidence_underconfidence_underconfident(self):
        """Test over/underconfidence identification - underconfident"""
        tracker = ConfidenceCalibrationTracker()
        # Low confidence, high accuracy
        for _ in range(9):
            tracker.add_prediction(0.5, True)
        for _ in range(1):
            tracker.add_prediction(0.5, False)  # 90% correct

        result = tracker.identify_overconfidence_underconfidence()

        assert not result["is_overconfident"]
        assert result["is_underconfident"]
        assert result["bias"] < -0.05  # Negative bias = underconfident

    def test_identify_problematic_ranges(self):
        """Test identification of problematic confidence ranges"""
        tracker = ConfidenceCalibrationTracker()
        # Add well-calibrated data in low confidence
        for _ in range(3):
            tracker.add_prediction(0.3, True)
        for _ in range(7):
            tracker.add_prediction(0.3, False)

        # Add poorly-calibrated data in high confidence
        for _ in range(3):
            tracker.add_prediction(0.9, True)
        for _ in range(7):
            tracker.add_prediction(0.9, False)  # Should be mostly True

        result = tracker.identify_overconfidence_underconfidence()

        # Should identify 0.9 range as problematic
        assert len(result["problematic_confidence_ranges"]) > 0

    def test_identify_empty_data(self):
        """Test identification with empty data"""
        tracker = ConfidenceCalibrationTracker()
        result = tracker.identify_overconfidence_underconfidence()

        assert not result["is_overconfident"]
        assert not result["is_underconfident"]
        assert result["bias"] == 0.0
        assert result["problematic_confidence_ranges"] == []

    def test_multiple_predictions_workflow(self):
        """Test typical workflow with multiple predictions"""
        tracker = ConfidenceCalibrationTracker()

        # Add various predictions
        predictions = [
            (0.9, True),
            (0.8, True),
            (0.7, False),
            (0.6, True),
            (0.5, False),
            (0.4, False),
            (0.3, False),
            (0.2, False),
        ]

        for conf, correct in predictions:
            tracker.add_prediction(conf, correct)

        # Calculate metrics
        metrics = tracker.calculate_calibration_metrics()
        assert "expected_calibration_error" in metrics
        assert "calibration_score" in metrics

        # Get plot data
        plot_data = tracker.get_calibration_plot_data()
        assert len(plot_data["bin_centers"]) > 0

        # Identify biases
        biases = tracker.identify_overconfidence_underconfidence()
        assert "bias" in biases
