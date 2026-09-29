"""Unit tests for FMCW Radar Range Processing Engine (src/radar/processing.py)."""

import numpy as np
import pytest
from src.radar.chirp import generate_fmcw_chirp
from src.radar.processing import (
    BeatSamplingValidation,
    RangeEstimateResult,
    calculate_max_unambiguous_range,
    dechirp_signal,
    estimate_range,
    validate_beat_sampling_rate,
)
from src.radar.propagation import simulate_target_echo
from src.utils.config import RadarConfig, TargetConfig


def test_max_unambiguous_range_calculation() -> None:
    """Test 1: Verify R_max = c * Fs / (4 * S) calculation."""
    fs = 10e6
    s = 1.5e12  # 150 MHz / 100 us
    r_max = calculate_max_unambiguous_range(sampling_rate_hz=fs, chirp_slope_hz_per_sec=s, is_complex=False)
    # R_max = 299792458 * 10e6 / (4 * 1.5e12) = 499.654 m
    assert r_max == pytest.approx(499.654, abs=1e-3)


def test_range_estimation_accuracy_50m_100m_150m() -> None:
    """Test 2: Verify Range FFT estimation for targets at 50 m, 100 m, and 150 m."""
    cfg = RadarConfig(
        carrier_frequency_hz=77e9,
        sweep_bandwidth_hz=150e6,
        chirp_duration_sec=100e-6,
        sampling_rate_hz=10e6
    )
    chirp = generate_fmcw_chirp(cfg)

    for r_target in [50.0, 100.0, 150.0]:
        target = TargetConfig(range_m=r_target, velocity_mps=0.0, rcs_sqm=1.0)
        payload = simulate_target_echo(chirp, target)

        res = estimate_range(payload, chirp, window_name="hann")

        assert isinstance(res, RangeEstimateResult)
        assert res.theoretical_range_m == r_target
        assert res.estimated_range_m == pytest.approx(r_target, abs=0.1)  # Within 10 cm
        assert res.range_error_m < 0.1
        assert res.relative_range_error < 0.01  # < 1% error


def test_zero_padding_increases_grid_density_not_resolution() -> None:
    """Test 3: Verify zero-padding (N_fft = 4*N) refines grid spacing without altering physical resolution Delta R."""
    cfg = RadarConfig(
        carrier_frequency_hz=77e9,
        sweep_bandwidth_hz=150e6,
        chirp_duration_sec=100e-6,
        sampling_rate_hz=10e6
    )
    chirp = generate_fmcw_chirp(cfg)
    target = TargetConfig(range_m=100.0, velocity_mps=0.0, rcs_sqm=1.0)
    payload = simulate_target_echo(chirp, target)

    res_normal = estimate_range(payload, chirp, window_name="hann", n_fft=1000)
    res_zp = estimate_range(payload, chirp, window_name="hann", n_fft=4000)

    # Physical range resolution remains determined strictly by bandwidth c / (2*B)
    assert res_normal.range_resolution_m == pytest.approx(res_zp.range_resolution_m, abs=1e-6)

    # Range bin spacing is 4x denser for zero-padded spectrum
    assert res_zp.range_bin_spacing_m == pytest.approx(res_normal.range_bin_spacing_m / 4.0, abs=1e-5)
    assert len(res_zp.range_axis_m) == 2001  # rfftfreq(4000) = 2001 positive bins


def test_window_selection_options() -> None:
    """Test 4: Verify range estimation across different window functions (rect, hann, hamming, blackman)."""
    cfg = RadarConfig(
        carrier_frequency_hz=77e9,
        sweep_bandwidth_hz=150e6,
        chirp_duration_sec=100e-6,
        sampling_rate_hz=10e6
    )
    chirp = generate_fmcw_chirp(cfg)
    target = TargetConfig(range_m=100.0, velocity_mps=0.0, rcs_sqm=1.0)
    payload = simulate_target_echo(chirp, target)

    for w_name in ["rect", "hann", "hamming", "blackman"]:
        res = estimate_range(payload, chirp, window_name=w_name)
        assert res.estimated_range_m == pytest.approx(100.0, abs=0.1)


def test_dechirping_length_mismatch_error() -> None:
    """Test 5: Verify ValueError exception when dechirping mismatched signal lengths."""
    cfg = RadarConfig(
        carrier_frequency_hz=77e9,
        sweep_bandwidth_hz=150e6,
        chirp_duration_sec=100e-6,
        sampling_rate_hz=10e6
    )
    chirp = generate_fmcw_chirp(cfg)

    with pytest.raises(ValueError, match="must match"):
        dechirp_signal(chirp, np.ones(500))


def test_beat_signal_sampling_architecture_validation() -> None:
    """Test 6 (Audit): Verify Option B beat sampling validation and Nyquist bounds (Fs > 2*fb_max)."""
    cfg_20mhz = RadarConfig(
        carrier_frequency_hz=77e9,
        sweep_bandwidth_hz=150e6,
        chirp_duration_sec=100e-6,
        sampling_rate_hz=20e6
    )
    chirp_20mhz = generate_fmcw_chirp(cfg_20mhz, sampling_architecture="stretch_dechirp_analog")
    target = TargetConfig(range_m=100.0, velocity_mps=0.0, rcs_sqm=1.0)
    payload_20mhz = simulate_target_echo(chirp_20mhz, target)

    # Valid beat sampling for max range 500m at Fs=20MHz (requires min Fs > 10.0069 MHz)
    res = estimate_range(payload_20mhz, chirp_20mhz, max_expected_range_m=500.0)

    val = res.beat_sampling_validation
    assert val is not None
    assert val.is_valid is True
    assert val.configured_sampling_rate_hz == 20e6
    assert val.nyquist_frequency_hz == 10e6
    assert val.maximum_expected_range_m == 500.0
    assert val.minimum_required_beat_sampling_rate_hz == pytest.approx(10.0069e6, abs=1e3)
    assert val.sampling_margin_ratio > 1.9

    # Fs=10 MHz is REJECTED for R_max=500m because 2*fb_max = 10.0069 MHz > 10 MHz
    cfg_10mhz = RadarConfig(
        carrier_frequency_hz=77e9,
        sweep_bandwidth_hz=150e6,
        chirp_duration_sec=100e-6,
        sampling_rate_hz=10e6
    )
    chirp_10mhz = generate_fmcw_chirp(cfg_10mhz, sampling_architecture="stretch_dechirp_analog")
    payload_10mhz = simulate_target_echo(chirp_10mhz, target)

    with pytest.raises(ValueError, match="insufficient for beat frequency bandwidth"):
        estimate_range(payload_10mhz, chirp_10mhz, max_expected_range_m=500.0)


def test_beat_sampling_rmax_scaling_and_beat_bandwidth_basis() -> None:
    """Test 7 (Audit): Verify R_max scaling effects on beat Nyquist rate and beat vs bandwidth distinction."""
    cfg = RadarConfig(
        carrier_frequency_hz=77e9,
        sweep_bandwidth_hz=150e6,
        chirp_duration_sec=100e-6,
        sampling_rate_hz=20e6
    )
    chirp = generate_fmcw_chirp(cfg)

    # Increasing R_max increases required beat sampling rate
    val_200 = validate_beat_sampling_rate(sampling_rate_hz=20e6, chirp_slope_hz_per_sec=chirp.chirp_slope_hz_per_sec, max_expected_range_m=200.0)
    val_500 = validate_beat_sampling_rate(sampling_rate_hz=20e6, chirp_slope_hz_per_sec=chirp.chirp_slope_hz_per_sec, max_expected_range_m=500.0)

    assert val_500.minimum_required_beat_sampling_rate_hz > val_200.minimum_required_beat_sampling_rate_hz
    assert val_200.maximum_beat_frequency_hz < val_500.maximum_beat_frequency_hz

    # Decreasing R_max reduces required beat sampling rate
    val_50 = validate_beat_sampling_rate(sampling_rate_hz=20e6, chirp_slope_hz_per_sec=chirp.chirp_slope_hz_per_sec, max_expected_range_m=50.0)
    assert val_50.minimum_required_beat_sampling_rate_hz < val_200.minimum_required_beat_sampling_rate_hz

    # Validation is based on beat frequency (~5 MHz for R_max=500m), NOT directly on chirp bandwidth B (150 MHz)
    assert val_500.minimum_required_beat_sampling_rate_hz == pytest.approx(10.0069e6, abs=1e3)
    assert val_500.minimum_required_beat_sampling_rate_hz < chirp.bandwidth_hz


def test_500m_target_range_estimation() -> None:
    """Test 8 (Audit): Verify range processing for a target near maximum expected range R=500m with Fs=20MHz."""
    cfg = RadarConfig(
        carrier_frequency_hz=77e9,
        sweep_bandwidth_hz=150e6,
        chirp_duration_sec=100e-6,
        sampling_rate_hz=20e6
    )
    chirp = generate_fmcw_chirp(cfg)
    target = TargetConfig(range_m=500.0, velocity_mps=0.0, rcs_sqm=1.0)
    payload = simulate_target_echo(chirp, target)

    res = estimate_range(payload, chirp, window_name="hann", max_expected_range_m=500.0)

    assert isinstance(res, RangeEstimateResult)
    assert res.theoretical_range_m == 500.0
    assert res.estimated_range_m == pytest.approx(500.0, abs=0.1)
    # Beat frequency f_b ≈ 5.0035 MHz must remain below Nyquist frequency Fs/2 = 10 MHz
    assert res.beat_frequency_hz < 10e6
    assert res.beat_frequency_hz == pytest.approx(5003461.4, abs=1e4)


def test_direct_sampled_chirp_architecture_validation() -> None:
    """Test 9 (Audit): Verify Option A direct sampled chirp requires Fs >= B."""
    cfg_invalid = RadarConfig(
        carrier_frequency_hz=77e9,
        sweep_bandwidth_hz=150e6,
        chirp_duration_sec=100e-6,
        sampling_rate_hz=10e6  # 10 MHz < 150 MHz bandwidth B
    )
    # Option A raises explicit ValueError because 10 MHz violates Nyquist for 150 MHz raw chirp
    with pytest.raises(ValueError, match="requires Fs >= 150.0 MHz"):
        generate_fmcw_chirp(cfg_invalid, sampling_architecture="direct_sampled_chirp")

    cfg_valid = RadarConfig(
        carrier_frequency_hz=77e9,
        sweep_bandwidth_hz=150e6,
        chirp_duration_sec=100e-6,
        sampling_rate_hz=200e6  # 200 MHz > 150 MHz bandwidth B
    )
    chirp_valid = generate_fmcw_chirp(cfg_valid, sampling_architecture="direct_sampled_chirp")
    assert chirp_valid.sampling_architecture == "direct_sampled_chirp"


