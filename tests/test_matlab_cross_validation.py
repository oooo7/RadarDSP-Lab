"""Tests for Phase 10 MATLAB Golden Reference & Cross-Validation infrastructure.

Validates the structure of exported python_reference_results.json, checks mathematical specifications
used for cross-validation, and verifies the static existence of MATLAB reference scripts.

Note:
    MATLAB runtime execution was NOT performed in this environment because neither MATLAB nor Octave CLI is installed.
    These tests validate Python reference data export and theoretical mathematical equations in PyTest.
"""

import json
from pathlib import Path
import numpy as np
import pytest
import scipy.constants

from src.dsp.sampling import calculate_alias_frequency
from src.radar.cfar import calculate_ca_cfar_alpha
from src.radar.processing import calculate_max_unambiguous_range, compute_range_fft
from src.radar.target import compute_target_state
from src.utils.config import RadarConfig, TargetConfig


PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
MATLAB_DIR = PROJECT_ROOT / "matlab"
SPEED_OF_LIGHT_M_PER_S = float(scipy.constants.c)


def test_python_reference_export_json_validity():
    """Verify that exported python_reference_results.json exists and contains complete reference sections."""
    json_path = RESULTS_DIR / "python_reference_results.json"
    assert json_path.exists(), f"Missing exported reference JSON at {json_path}"

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "dsp_aliasing" in data
    assert "dsp_fft" in data
    assert "fmcw_single_target" in data
    assert "fmcw_multi_target" in data
    assert "awgn_noise" in data
    assert "ca_cfar" in data


def test_dsp_alias_theoretical_specification():
    """Verify theoretical DSP aliasing formula (700 Hz @ 1000 Hz Fs -> 300 Hz alias foldover)."""
    f_alias = calculate_alias_frequency(700.0, 1000.0)
    assert abs(f_alias - 300.0) < 1e-6


def test_python_fmcw_range_reference_tolerances():
    """Verify that Python exported reference single-target ranges (50-500m) are within physical resolution Delta_R."""
    config = RadarConfig(
        carrier_frequency_hz=77e9,
        sweep_bandwidth_hz=150e6,
        chirp_duration_sec=100e-6,
        sampling_rate_hz=20e6,
    )
    delta_r = SPEED_OF_LIGHT_M_PER_S / (2.0 * config.sweep_bandwidth_hz)  # ~0.9993 m

    json_path = RESULTS_DIR / "python_reference_results.json"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    st_results = data["fmcw_single_target"]["targets"]
    assert len(st_results) == 7

    for item in st_results:
        target_r = item["true_range_m"]
        abs_err = item["abs_error_m"]

        assert abs_err <= delta_r, f"Range error {abs_err:.4f}m for target {target_r}m exceeds Delta_R={delta_r:.4f}m"


def test_cfar_alpha_theoretical_equation():
    """Verify 2D CA-CFAR alpha multiplier equation alpha = N * (Pfa^(-1/N) - 1)."""
    pfa = 1e-4
    n_train = 144
    alpha_py = calculate_ca_cfar_alpha(pfa, n_train)

    # Theoretical evaluation
    alpha_expected = 144.0 * (pfa**(-1.0 / 144.0) - 1.0)
    assert abs(alpha_py - alpha_expected) < 1e-9
    assert abs(alpha_py - 9.51127) < 1e-3


def test_matlab_source_files_exist():
    """Verify all Phase 10 MATLAB golden reference scripts exist statically in matlab/ directory."""
    expected_files = [
        MATLAB_DIR / "README.md",
        MATLAB_DIR / "dsp" / "generate_sine.m",
        MATLAB_DIR / "dsp" / "generate_multitone.m",
        MATLAB_DIR / "dsp" / "sampling_alias.m",
        MATLAB_DIR / "dsp" / "compute_spectrum.m",
        MATLAB_DIR / "radar" / "generate_fmcw_chirp.m",
        MATLAB_DIR / "radar" / "simulate_single_target.m",
        MATLAB_DIR / "radar" / "range_estimation.m",
        MATLAB_DIR / "radar" / "multi_target_demo.m",
        MATLAB_DIR / "radar" / "range_doppler_demo.m",
        MATLAB_DIR / "radar" / "add_awgn_noise.m",
        MATLAB_DIR / "radar" / "run_ca_cfar.m",
        MATLAB_DIR / "validation" / "cross_validate_python.m",
        MATLAB_DIR / "validation" / "export_reference_results.m",
    ]

    for filepath in expected_files:
        assert filepath.exists(), f"Missing required MATLAB reference file: {filepath}"
