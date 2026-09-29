"""Unit and Integration tests for 2D CA-CFAR Detection Engine (src/radar/cfar.py)."""

import numpy as np
import pytest

from src.radar.cfar import (
    CFARDetection,
    CFARResult,
    calculate_ca_cfar_alpha,
    extract_cfar_detections,
    run_ca_cfar,
)
from src.radar.doppler import RadarDataCube, compute_range_doppler_map, generate_multi_chirp_data_cube
from src.radar.noise import compose_radar_environment
from src.utils.config import CFARConfig, RadarConfig, TargetConfig


def test_alpha_multiplier_calculation() -> None:
    """Test 1: Verify alpha = N * (Pfa^(-1/N) - 1) formula for representative Pfa values."""
    n_train = 48
    # For Pfa = 1e-4 and N = 48: alpha = 48 * (10000^(1/48) - 1) ≈ 10.153
    alpha_1e4 = calculate_ca_cfar_alpha(pfa=1e-4, num_training_cells=n_train)
    assert alpha_1e4 == pytest.approx(10.153, abs=1e-2)

    alpha_1e2 = calculate_ca_cfar_alpha(pfa=1e-2, num_training_cells=n_train)
    assert alpha_1e2 < alpha_1e4  # Higher Pfa -> lower threshold alpha multiplier

    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        calculate_ca_cfar_alpha(pfa=0.0, num_training_cells=48)

    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        calculate_ca_cfar_alpha(pfa=1.5, num_training_cells=48)


def test_cfar_boundary_masking_no_wraparound() -> None:
    """Test 2: Verify boundary cells are marked as non-detection and NaN threshold (no np.roll wrap)."""
    rd_matrix = np.ones((64, 100), dtype=np.complex128)
    res = run_ca_cfar(
        rd_matrix,
        pfa=1e-3,
        num_guard_range=2,
        num_guard_doppler=2,
        num_train_range=4,
        num_train_doppler=4
    )

    border_r = 2 + 4  # 6
    border_d = 2 + 4  # 6

    # Border regions must be non-detection
    assert not np.any(res.detection_mask[:border_d, :])
    assert not np.any(res.detection_mask[-border_d:, :])
    assert not np.any(res.detection_mask[:, :border_r])
    assert not np.any(res.detection_mask[:, -border_r:])

    # Border threshold dB values should be NaN
    assert np.isnan(res.threshold_db[0, 0])
    assert np.isnan(res.threshold_db[31, 2])


def test_cfar_synthetic_strong_target_detection() -> None:
    """Test 3: Verify CA-CFAR detects a synthetic strong target peak above background noise."""
    # 2D RD matrix with noise power = 1.0 and single strong peak = 100.0 at (30, 40)
    rd_matrix = np.ones((64, 100), dtype=np.complex128)
    rd_matrix[30, 40] = 100.0  # Power = 10000.0

    res = run_ca_cfar(
        rd_matrix,
        pfa=1e-3,
        num_guard_range=2,
        num_guard_doppler=2,
        num_train_range=4,
        num_train_doppler=4
    )

    assert isinstance(res, CFARResult)
    assert bool(res.detection_mask[30, 40]) is True
    assert res.num_detections >= 1

    dets = extract_cfar_detections(res)
    assert len(dets) >= 1
    top_det = dets[0]
    assert top_det.doppler_bin == 30
    assert top_det.range_bin == 40


def test_cfar_noise_only_false_alarm_behavior() -> None:
    """Test 4: Verify false alarm count on pure noise matrix is small and bounded."""
    rng = np.random.default_rng(123)
    # 64 x 100 noise matrix
    noise_matrix = rng.normal(0, 1, (64, 100)) + 1j * rng.normal(0, 1, (64, 100))

    res = run_ca_cfar(noise_matrix, pfa=1e-3, num_guard_range=2, num_guard_doppler=2, num_train_range=4, num_train_doppler=4)

    # Valid interior cells = (64 - 12) * (100 - 12) = 52 * 88 = 4576 cells
    # Expected false alarms ≈ 4576 * 1e-3 = ~4.5 detections
    assert res.num_detections < 20  # Statistically bounded


def test_cfar_invalid_configurations() -> None:
    """Test 5: Verify ValueError exceptions on invalid CFAR window sizes and parameters."""
    rd_small = np.ones((10, 10), dtype=np.complex128)

    # Window larger than RD map dimensions
    with pytest.raises(ValueError, match="larger than Range-Doppler map"):
        run_ca_cfar(rd_small, num_guard_range=5, num_train_range=5)

    with pytest.raises(ValueError, match="Guard cells must be non-negative"):
        run_ca_cfar(rd_small, num_guard_range=-1)


def test_full_phase8_integration_pipeline() -> None:
    """Test 6 (Integration): Phase 7 multi-target -> compose AWGN + clutter -> Range-Doppler -> CA-CFAR."""
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

    clean_cube = generate_multi_chirp_data_cube(cfg, targets)

    # Compose environment with AWGN (SNR=25dB) and background clutter
    comp_res = compose_radar_environment(clean_cube.data, snr_db=25.0, clutter_power=0.1, seed=42)

    # Replace clean data cube matrix with noisy/cluttered composite matrix
    noisy_cube = RadarDataCube(
        data=comp_res.composite_signal,
        time_vector=clean_cube.time_vector,
        chirp_container=clean_cube.chirp_container,
        num_chirps=clean_cube.num_chirps,
        samples_per_chirp=clean_cube.samples_per_chirp,
        sampling_rate_hz=clean_cube.sampling_rate_hz,
        chirp_duration_sec=clean_cube.chirp_duration_sec,
        carrier_frequency_hz=clean_cube.carrier_frequency_hz,
        bandwidth_hz=clean_cube.bandwidth_hz,
        targets=clean_cube.targets
    )

    rd_map = compute_range_doppler_map(noisy_cube)

    # Run CA-CFAR
    cfar_res = run_ca_cfar(rd_map, pfa=1e-4, num_guard_range=2, num_guard_doppler=2, num_train_range=4, num_train_doppler=4)

    assert cfar_res.num_detections >= 3

    dets = cfar_res.detections
    matched_t1 = [d for d in dets if abs(d.range_m - 50.0) < 3.0 and abs(d.velocity_mps - 10.0) < 2.0]
    matched_t2 = [d for d in dets if abs(d.range_m - 120.0) < 3.0 and abs(d.velocity_mps - (-5.0)) < 2.0]
    matched_t3 = [d for d in dets if abs(d.range_m - 200.0) < 3.0 and abs(d.velocity_mps - 0.0) < 2.0]

    assert len(matched_t1) >= 1
    assert len(matched_t2) >= 1
    assert len(matched_t3) >= 1
