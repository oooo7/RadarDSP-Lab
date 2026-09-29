"""Unit tests for Validation Metrics & Target Matching Engine (src/validation/metrics.py)."""

import numpy as np
import pytest

from src.radar.cfar import CFARDetection
from src.radar.target import TargetState
from src.utils.config import RadarConfig, TargetConfig
from src.validation.metrics import (
    AggregateErrorStats,
    EstimationErrorMetrics,
    TargetMatchResult,
    calculate_absolute_error,
    calculate_aggregate_metrics,
    calculate_relative_error,
    compute_theoretical_limits,
    evaluate_estimation_errors,
    match_targets_to_detections,
)


def test_calculate_absolute_error():
    """Verify signed absolute error calculation."""
    assert calculate_absolute_error(105.0, 100.0) == pytest.approx(5.0)
    assert calculate_absolute_error(95.0, 100.0) == pytest.approx(-5.0)
    assert calculate_absolute_error(100.0, 100.0) == pytest.approx(0.0)


def test_calculate_relative_error_normal_and_zero_ref():
    """Verify relative error handling including true value zero reference handling."""
    # Normal case
    assert calculate_relative_error(105.0, 100.0) == pytest.approx(0.05)
    # Zero reference case (should return 0.0 to prevent division instability)
    assert calculate_relative_error(0.5, 0.0) == pytest.approx(0.0)
    assert calculate_relative_error(-1.0, 1e-6) == pytest.approx(0.0)


def test_calculate_aggregate_metrics():
    """Verify summary error statistics (Mean, MAE, RMSE, StdDev, Max)."""
    errors = np.array([2.0, -2.0, 4.0, -4.0], dtype=np.float64)
    stats = calculate_aggregate_metrics(errors)

    assert stats.num_trials == 4
    assert stats.mean_error == pytest.approx(0.0)
    assert stats.mean_absolute_error == pytest.approx(3.0)
    assert stats.rmse == pytest.approx(np.sqrt(10.0))  # sqrt((4+4+16+16)/4) = sqrt(10)
    assert stats.std_dev == pytest.approx(np.std(errors))
    assert stats.max_absolute_error == pytest.approx(4.0)


def test_calculate_aggregate_metrics_empty():
    """Verify empty error list returns zeroed aggregate stats."""
    stats = calculate_aggregate_metrics([])
    assert stats.num_trials == 0
    assert stats.mean_error == 0.0
    assert stats.mean_absolute_error == 0.0
    assert stats.rmse == 0.0


def test_compute_theoretical_limits_crlb():
    """Verify theoretical limits and CRLB calculation."""
    config = RadarConfig(
        carrier_frequency_hz=77e9,
        sweep_bandwidth_hz=150e6,
        chirp_duration_sec=100e-6,
        sampling_rate_hz=20e6,
        num_chirps=64
    )
    limits = compute_theoretical_limits(config, snr_db=20.0)

    # Range resolution Delta R = c / (2 * B) = 3e8 / (2 * 150e6) = 1.0 m
    assert limits.range_resolution_m == pytest.approx(0.9993, rel=1e-2)
    assert limits.range_crlb_m > 0.0
    assert limits.velocity_crlb_mps > 0.0
    assert limits.snr_linear == pytest.approx(100.0)


def test_evaluate_estimation_errors():
    """Verify full evaluation of estimation errors against TargetConfig ground truth."""
    target_cfg = TargetConfig(range_m=100.0, velocity_mps=10.0, rcs_sqm=1.0)
    metrics = evaluate_estimation_errors(
        ground_truth=target_cfg,
        estimated_range_m=102.0,
        estimated_velocity_mps=9.5
    )

    assert metrics.true_range_m == 100.0
    assert metrics.estimated_range_m == 102.0
    assert metrics.range_error_m == pytest.approx(2.0)
    assert metrics.abs_range_error_m == pytest.approx(2.0)
    assert metrics.rel_range_error == pytest.approx(0.02)

    assert metrics.true_velocity_mps == 10.0
    assert metrics.estimated_velocity_mps == 9.5
    assert metrics.velocity_error_mps == pytest.approx(-0.5)
    assert metrics.abs_velocity_error_mps == pytest.approx(0.5)
    assert metrics.rel_velocity_error == pytest.approx(0.05)


def test_match_targets_to_detections_one_to_one():
    """Verify one-to-one resolution-gated target-to-detection matching algorithm."""
    target1 = TargetState(
        target_id="T1", range_m=50.0, velocity_mps=10.0, rcs_sqm=1.0,
        amplitude_scaling=1.0, phase_offset_rad=0.0, propagation_delay_sec=3.33e-7,
        theoretical_beat_frequency_hz=500e3, theoretical_doppler_frequency_hz=5.13e3
    )
    target2 = TargetState(
        target_id="T2", range_m=120.0, velocity_mps=-5.0, rcs_sqm=0.7,
        amplitude_scaling=0.7, phase_offset_rad=0.0, propagation_delay_sec=8.0e-7,
        theoretical_beat_frequency_hz=1.2e6, theoretical_doppler_frequency_hz=-2.56e3
    )

    det1 = CFARDetection(
        range_m=50.5, velocity_mps=10.2, power=1.0, power_db=0.0,
        threshold_power=0.1, threshold_db=-10.0, snr_db=10.0, range_bin=50, doppler_bin=32
    )
    det2 = CFARDetection(
        range_m=120.2, velocity_mps=-4.8, power=0.7, power_db=-1.5,
        threshold_power=0.1, threshold_db=-10.0, snr_db=8.5, range_bin=120, doppler_bin=20
    )
    det_spurious = CFARDetection(
        range_m=300.0, velocity_mps=15.0, power=0.05, power_db=-13.0,
        threshold_power=0.01, threshold_db=-20.0, snr_db=7.0, range_bin=300, doppler_bin=50
    )

    res = match_targets_to_detections(
        targets=[target1, target2],
        detections=[det1, det2, det_spurious],
        range_gate_m=2.0,
        velocity_gate_mps=1.0
    )

    assert res.total_targets == 2
    assert res.total_detections == 3
    assert res.num_true_positives == 2
    assert res.num_false_negatives == 0
    assert res.num_false_positives == 1
    assert res.probability_of_detection == pytest.approx(1.0)
    assert len(res.matches) == 2
    assert res.unmatched_detections[0] == det_spurious
