"""Unit & Integration tests for Monte Carlo Engine & Experiment Suite (src/validation/monte_carlo.py & src/validation/experiments.py)."""

import pytest

from src.utils.config import RadarConfig, TargetConfig
from src.validation.experiments import (
    run_false_alarm_experiment,
    run_multi_target_monte_carlo_experiment,
    run_range_accuracy_experiment,
    run_snr_sweep_experiment,
    run_velocity_accuracy_experiment,
)
from src.validation.monte_carlo import (
    MonteCarloConfig,
    MonteCarloResult,
    run_monte_carlo_simulation,
)


def test_monte_carlo_deterministic_seeds_reproducibility():
    """Verify that Monte Carlo simulations produce exact reproducible results given the same seed."""
    config = RadarConfig(
        carrier_frequency_hz=77e9,
        sweep_bandwidth_hz=150e6,
        chirp_duration_sec=100e-6,
        sampling_rate_hz=20e6,
        num_chirps=32
    )
    targets = [TargetConfig(range_m=100.0, velocity_mps=5.0, rcs_sqm=1.0, target_id="T1")]
    mc_cfg = MonteCarloConfig(num_trials=5, master_seed=12345, snr_db=15.0)

    res1 = run_monte_carlo_simulation(config=config, targets=targets, mc_config=mc_cfg)
    res2 = run_monte_carlo_simulation(config=config, targets=targets, mc_config=mc_cfg)

    assert res1.overall_detection_probability == pytest.approx(res2.overall_detection_probability)
    assert res1.aggregate_range_stats["overall"].mean_absolute_error == pytest.approx(
        res2.aggregate_range_stats["overall"].mean_absolute_error
    )
    assert res1.aggregate_velocity_stats["overall"].mean_absolute_error == pytest.approx(
        res2.aggregate_velocity_stats["overall"].mean_absolute_error
    )


def test_range_accuracy_experiment():
    """Verify range accuracy experiment returns valid points across ranges."""
    res = run_range_accuracy_experiment()
    assert len(res.points) == 6
    assert res.aggregate_stats.mean_absolute_error < res.range_resolution_m
    assert res.points[0].true_range_m == 50.0
    assert abs(res.points[0].estimated_range_m - 50.0) <= res.range_resolution_m


def test_velocity_accuracy_experiment():
    """Verify velocity accuracy experiment returns valid points across velocities including 0 m/s."""
    res = run_velocity_accuracy_experiment()
    assert len(res.points) == 7
    # Zero velocity point
    zero_point = [p for p in res.points if p.true_velocity_mps == 0.0][0]
    assert zero_point.rel_error == 0.0  # Safe zero reference handling
    assert res.aggregate_stats.mean_absolute_error <= res.velocity_resolution_mps


def test_snr_sweep_experiment():
    """Verify SNR sweep experiment produces monotonic/valid detection probability curve."""
    res = run_snr_sweep_experiment(num_trials_per_snr=10, snr_levels_db=[30.0, 15.0, 0.0, -10.0])
    assert len(res.detection_probabilities) == 4
    # High SNR should have Pd >= Low SNR
    assert res.detection_probabilities[0] >= res.detection_probabilities[3]


def test_false_alarm_experiment():
    """Verify CFAR false alarm experiment returns empirical Pfa for noise and clutter."""
    res = run_false_alarm_experiment(pfa_levels=[1e-2, 1e-3], num_trials=10)
    assert len(res.points) == 2
    for pt in res.points:
        assert 0.0 <= pt.empirical_pfa_noise_only <= 1.0
        assert 0.0 <= pt.empirical_pfa_clutter <= 1.0


def test_multi_target_monte_carlo_experiment_integration():
    """Integration test: Monte Carlo runner on multi-target Phase 7/8 scenario."""
    res = run_multi_target_monte_carlo_experiment(num_trials=10, snr_db=20.0)
    assert res.num_trials == 10
    assert len(res.targets_theoretical) == 3
    assert res.overall_detection_probability > 0.8
