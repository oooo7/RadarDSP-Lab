"""Unit tests for FMCW Radar Multi-Target & Doppler Processing Engine (src/radar/doppler.py)."""

import numpy as np
import pytest

from src.radar.doppler import (
    RadarDataCube,
    RangeDopplerPeak,
    RangeDopplerResult,
    compute_range_doppler_map,
    extract_range_doppler_peaks,
    generate_multi_chirp_data_cube,
    validate_range_estimate,
    validate_velocity_estimate,
)
from src.utils.config import RadarConfig, TargetConfig


def test_data_cube_generation_shape_and_properties() -> None:
    """Test 1: Verify data cube shape [num_chirps, samples_per_chirp] and physical metadata."""
    cfg = RadarConfig(
        carrier_frequency_hz=77e9,
        sweep_bandwidth_hz=150e6,
        chirp_duration_sec=100e-6,
        sampling_rate_hz=20e6,
        num_chirps=64
    )
    targets = [TargetConfig(range_m=50.0, velocity_mps=10.0, target_id="T1")]

    cube = generate_multi_chirp_data_cube(cfg, targets, num_chirps=64)

    assert isinstance(cube, RadarDataCube)
    assert cube.num_chirps == 64
    assert cube.samples_per_chirp == 2000  # 20 MHz * 100 us = 2000 samples
    assert cube.data.shape == (64, 2000)
    assert np.iscomplexobj(cube.data)
    assert cube.prf_hz == pytest.approx(10000.0, abs=1e-5)
    assert cube.wavelength_m == pytest.approx(0.0038934, abs=1e-5)


def test_single_stationary_target_zero_doppler() -> None:
    """Test 2: Verify single stationary target produces peak at 0 m/s velocity."""
    cfg = RadarConfig(
        carrier_frequency_hz=77e9,
        sweep_bandwidth_hz=150e6,
        chirp_duration_sec=50e-6,  # 50 us -> 20 kHz PRF
        sampling_rate_hz=20e6,
        num_chirps=128
    )
    targets = [TargetConfig(range_m=75.0, velocity_mps=0.0, target_id="T_stat")]

    cube = generate_multi_chirp_data_cube(cfg, targets)
    rd_map = compute_range_doppler_map(cube)

    peaks = extract_range_doppler_peaks(rd_map, threshold_db=-20.0)

    assert len(peaks) >= 1
    p = peaks[0]
    assert p.range_m == pytest.approx(75.0, abs=1.0)
    assert p.velocity_mps == pytest.approx(0.0, abs=0.5)


def test_positive_velocity_sign_convention() -> None:
    """Test 3: Verify positive velocity (receding target) maps to positive velocity peak."""
    cfg = RadarConfig(
        carrier_frequency_hz=77e9,
        sweep_bandwidth_hz=150e6,
        chirp_duration_sec=50e-6,
        sampling_rate_hz=20e6,
        num_chirps=128
    )
    targets = [TargetConfig(range_m=50.0, velocity_mps=10.0, target_id="T_rec")]

    cube = generate_multi_chirp_data_cube(cfg, targets)
    rd_map = compute_range_doppler_map(cube)

    peaks = extract_range_doppler_peaks(rd_map, threshold_db=-20.0)

    assert len(peaks) >= 1
    p = peaks[0]
    assert p.range_m == pytest.approx(50.0, abs=1.0)
    assert p.velocity_mps > 0.0  # Positive velocity
    assert p.velocity_mps == pytest.approx(10.0, abs=rd_map.velocity_resolution_mps * 2.0)


def test_negative_velocity_sign_convention() -> None:
    """Test 4: Verify negative velocity (approaching target) maps to negative velocity peak."""
    cfg = RadarConfig(
        carrier_frequency_hz=77e9,
        sweep_bandwidth_hz=150e6,
        chirp_duration_sec=50e-6,
        sampling_rate_hz=20e6,
        num_chirps=128
    )
    targets = [TargetConfig(range_m=120.0, velocity_mps=-5.0, target_id="T_app")]

    cube = generate_multi_chirp_data_cube(cfg, targets)
    rd_map = compute_range_doppler_map(cube)

    peaks = extract_range_doppler_peaks(rd_map, threshold_db=-20.0)

    assert len(peaks) >= 1
    p = peaks[0]
    assert p.range_m == pytest.approx(120.0, abs=1.0)
    assert p.velocity_mps < 0.0  # Negative velocity
    assert p.velocity_mps == pytest.approx(-5.0, abs=rd_map.velocity_resolution_mps * 2.0)


def test_multi_target_range_and_velocity_resolution() -> None:
    """Test 5 (Section 13): Verify detection of 3 simultaneous targets A (50m, +10m/s), B (120m, -5m/s), C (200m, 0m/s)."""
    cfg = RadarConfig(
        carrier_frequency_hz=77e9,
        sweep_bandwidth_hz=150e6,
        chirp_duration_sec=50e-6,
        sampling_rate_hz=20e6,
        num_chirps=128
    )
    targets = [
        TargetConfig(range_m=50.0, velocity_mps=10.0, amplitude=1.0, target_id="T1"),
        TargetConfig(range_m=120.0, velocity_mps=-5.0, amplitude=0.7, target_id="T2"),
        TargetConfig(range_m=200.0, velocity_mps=0.0, amplitude=0.5, target_id="T3"),
    ]

    cube = generate_multi_chirp_data_cube(cfg, targets)
    rd_map = compute_range_doppler_map(cube)

    assert rd_map.magnitude_matrix.shape == (128, rd_map.num_range_bins)

    peaks = extract_range_doppler_peaks(rd_map, threshold_db=-25.0, max_peaks=10)

    assert len(peaks) >= 3

    # Match each target
    matched_t1 = [p for p in peaks if abs(p.range_m - 50.0) < 2.0 and abs(p.velocity_mps - 10.0) < 1.0]
    matched_t2 = [p for p in peaks if abs(p.range_m - 120.0) < 2.0 and abs(p.velocity_mps - (-5.0)) < 1.0]
    matched_t3 = [p for p in peaks if abs(p.range_m - 200.0) < 2.0 and abs(p.velocity_mps - 0.0) < 1.0]

    assert len(matched_t1) == 1
    assert len(matched_t2) == 1
    assert len(matched_t3) == 1

    # Validate range & velocity accuracy
    v_t1 = validate_range_estimate(50.0, matched_t1[0].range_m, rd_map.range_resolution_m)
    v_t2 = validate_range_estimate(120.0, matched_t2[0].range_m, rd_map.range_resolution_m)
    v_t3 = validate_range_estimate(200.0, matched_t3[0].range_m, rd_map.range_resolution_m)

    assert v_t1.is_within_resolution_cell
    assert v_t2.is_within_resolution_cell
    assert v_t3.is_within_resolution_cell

    vel_t1 = validate_velocity_estimate(10.0, matched_t1[0].velocity_mps, rd_map.velocity_resolution_mps)
    vel_t2 = validate_velocity_estimate(-5.0, matched_t2[0].velocity_mps, rd_map.velocity_resolution_mps)
    vel_t3 = validate_velocity_estimate(0.0, matched_t3[0].velocity_mps, rd_map.velocity_resolution_mps)

    assert vel_t1.is_within_resolution_cell
    assert vel_t2.is_within_resolution_cell
    assert vel_t3.is_within_resolution_cell


def test_same_range_different_velocity_resolution() -> None:
    """Test 6: Verify two targets at the exact same range (100 m) but different velocities (+10 m/s vs -10 m/s) are resolved."""
    cfg = RadarConfig(
        carrier_frequency_hz=77e9,
        sweep_bandwidth_hz=150e6,
        chirp_duration_sec=50e-6,
        sampling_rate_hz=20e6,
        num_chirps=128
    )
    targets = [
        TargetConfig(range_m=100.0, velocity_mps=10.0, amplitude=1.0, target_id="T_rec"),
        TargetConfig(range_m=100.0, velocity_mps=-10.0, amplitude=1.0, target_id="T_app"),
    ]

    cube = generate_multi_chirp_data_cube(cfg, targets)
    rd_map = compute_range_doppler_map(cube)

    peaks = extract_range_doppler_peaks(rd_map, threshold_db=-20.0, max_peaks=10)

    assert len(peaks) >= 2
    velocities = [p.velocity_mps for p in peaks if abs(p.range_m - 100.0) < 2.0]
    assert len(velocities) >= 2
    assert any(v > 5.0 for v in velocities)
    assert any(v < -5.0 for v in velocities)


def test_increasing_chirp_count_improves_velocity_resolution() -> None:
    """Test 7: Verify increasing M (64 vs 256 chirps) refines velocity-bin resolution Delta v."""
    cfg = RadarConfig(
        carrier_frequency_hz=77e9,
        sweep_bandwidth_hz=150e6,
        chirp_duration_sec=100e-6,
        sampling_rate_hz=20e6
    )
    targets = [TargetConfig(range_m=50.0, velocity_mps=5.0)]

    cube_64 = generate_multi_chirp_data_cube(cfg, targets, num_chirps=64)
    cube_256 = generate_multi_chirp_data_cube(cfg, targets, num_chirps=256)

    rd_64 = compute_range_doppler_map(cube_64)
    rd_256 = compute_range_doppler_map(cube_256)

    assert rd_256.velocity_resolution_mps == pytest.approx(rd_64.velocity_resolution_mps / 4.0, abs=1e-5)


def test_invalid_parameters_raise_value_errors() -> None:
    """Test 8: Verify ValueError exceptions on invalid multi-target inputs."""
    cfg = RadarConfig(carrier_frequency_hz=77e9, sweep_bandwidth_hz=150e6, chirp_duration_sec=100e-6, sampling_rate_hz=20e6)

    # Empty target list
    with pytest.raises(ValueError, match="At least one target"):
        generate_multi_chirp_data_cube(cfg, [])

    # Zero chirps
    with pytest.raises(ValueError, match="Number of chirps must be at least 1"):
        generate_multi_chirp_data_cube(cfg, [TargetConfig(range_m=50.0)], num_chirps=0)

    # Invalid range <= 0 (Pydantic validation)
    with pytest.raises((ValueError, Exception)):
        TargetConfig(range_m=-50.0)
