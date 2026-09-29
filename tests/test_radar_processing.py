"""Unit tests for FMCW Radar Range Processing Engine (src/radar/processing.py)."""

import numpy as np
import pytest
from src.radar.chirp import generate_fmcw_chirp
from src.radar.processing import (
    RangeEstimateResult,
    calculate_max_unambiguous_range,
    dechirp_signal,
    estimate_range,
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
