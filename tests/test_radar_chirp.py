"""Unit tests for FMCW Chirp Generation Engine (src/radar/chirp.py)."""

import numpy as np
import pytest
from src.radar.chirp import FMCWChirpContainer, generate_fmcw_chirp
from src.utils.config import RadarConfig


def test_chirp_generation_basic_properties() -> None:
    """Test 1: Verify chirp sample count, time vector spacing, slope S = B/T_c, and range resolution."""
    cfg = RadarConfig(
        carrier_frequency_hz=77e9,
        sweep_bandwidth_hz=150e6,
        chirp_duration_sec=100e-6,
        sampling_rate_hz=10e6
    )

    chirp = generate_fmcw_chirp(cfg)

    assert isinstance(chirp, FMCWChirpContainer)
    assert chirp.num_samples == 1000  # 10 MHz * 100 us = 1000 samples
    assert len(chirp.time_vector) == 1000
    assert len(chirp.tx_signal) == 1000
    assert chirp.chirp_slope_hz_per_sec == pytest.approx(1.5e12, abs=1e-5)
    assert chirp.range_resolution_m == pytest.approx(0.9993, abs=1e-3)
    # Endpoint excluded time vector convention: t[last] < duration
    assert chirp.time_vector[-1] < 100e-6


def test_chirp_instantaneous_frequency_linear_sweep() -> None:
    """Test 2: Verify instantaneous frequency sweeps linearly from 0 to B (150 MHz)."""
    cfg = RadarConfig(
        carrier_frequency_hz=77e9,
        sweep_bandwidth_hz=150e6,
        chirp_duration_sec=100e-6,
        sampling_rate_hz=10e6
    )

    chirp = generate_fmcw_chirp(cfg)

    assert chirp.instantaneous_freq_hz[0] == 0.0
    # Last sample frequency: S * t[-1]
    expected_f_end = chirp.chirp_slope_hz_per_sec * chirp.time_vector[-1]
    assert chirp.instantaneous_freq_hz[-1] == pytest.approx(expected_f_end, abs=1e-3)


def test_chirp_complex_vs_real_baseband() -> None:
    """Test 3: Verify complex baseband vs real passband signal generation."""
    cfg = RadarConfig(
        carrier_frequency_hz=77e9,
        sweep_bandwidth_hz=150e6,
        chirp_duration_sec=100e-6,
        sampling_rate_hz=10e6
    )

    c_chirp = generate_fmcw_chirp(cfg, is_complex_baseband=True)
    r_chirp = generate_fmcw_chirp(cfg, is_complex_baseband=False)

    assert np.iscomplexobj(c_chirp.tx_signal)
    assert not np.iscomplexobj(r_chirp.tx_signal)
    assert np.max(np.abs(c_chirp.tx_signal)) == pytest.approx(1.0, abs=1e-5)
    assert np.max(np.abs(r_chirp.tx_signal)) == pytest.approx(1.0, abs=1e-5)


def test_chirp_invalid_parameters() -> None:
    """Test 4: Verify ValueError exceptions on invalid chirp configurations."""
    with pytest.raises((ValueError, TypeError)):
        RadarConfig(carrier_frequency_hz=-77e9)

    with pytest.raises((ValueError, TypeError)):
        RadarConfig(sweep_bandwidth_hz=-150e6)

    with pytest.raises((ValueError, TypeError)):
        RadarConfig(chirp_duration_sec=-100e-6)

    with pytest.raises((ValueError, TypeError)):
        RadarConfig(sampling_rate_hz=0.0)

