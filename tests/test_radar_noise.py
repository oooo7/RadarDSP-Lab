"""Unit tests for Radar Noise & SNR Engine (src/radar/noise.py)."""

import numpy as np
import pytest

from src.radar.noise import (
    ComplexNoiseResult,
    add_awgn_noise,
    calculate_noise_power,
    calculate_signal_power,
    calculate_snr_db,
    compose_radar_environment,
)


def test_power_calculation_real_and_complex() -> None:
    """Test 1: Verify signal and noise power calculations for known tone waveforms."""
    # Unit complex sinusoid e^(j*2pi*f*n): |e^(j*theta)| = 1 -> Power = 1.0
    t = np.linspace(0, 1, 1000)
    c_sig = np.exp(1j * 2 * np.pi * 100 * t)
    assert calculate_signal_power(c_sig) == pytest.approx(1.0, abs=1e-6)

    # Real sine wave sin(2pi*f*n): Average power = 0.5
    r_sig = np.sin(2 * np.pi * 100 * t)
    assert calculate_signal_power(r_sig) == pytest.approx(0.5, abs=1e-2)

    # Constant DC signal = 3.0 -> Power = 9.0
    dc_sig = np.full(500, 3.0)
    assert calculate_signal_power(dc_sig) == pytest.approx(9.0, abs=1e-6)


def test_snr_calculation_db_formula() -> None:
    """Test 2: Verify SNR calculation 10*log10(P_sig / P_noise)."""
    p_sig = 10.0
    p_noise = 1.0
    assert calculate_snr_db(p_sig, p_noise) == pytest.approx(10.0, abs=1e-5)

    p_sig2 = 100.0
    p_noise2 = 1.0
    assert calculate_snr_db(p_sig2, p_noise2) == pytest.approx(20.0, abs=1e-5)

    with pytest.raises(ValueError, match="strictly positive"):
        calculate_snr_db(0.0, 1.0)


def test_add_awgn_noise_snr_accuracy_and_shape() -> None:
    """Test 3: Verify add_awgn_noise preserves shape, complex type, and achieves requested SNR."""
    # 2D signal matrix shape (64, 1000)
    signal = np.ones((64, 1000), dtype=np.complex128)  # P_sig = 1.0
    requested_snr = 15.0

    res = add_awgn_noise(signal, snr_db=requested_snr, seed=42)

    assert isinstance(res, ComplexNoiseResult)
    assert res.noisy_signal.shape == (64, 1000)
    assert np.iscomplexobj(res.noisy_signal)
    assert res.signal_power == pytest.approx(1.0, abs=1e-5)

    # Measured SNR should match requested SNR within finite sample statistics (~0.5 dB for 64000 samples)
    assert res.snr_db == pytest.approx(requested_snr, abs=0.5)


def test_add_awgn_noise_deterministic_seed() -> None:
    """Test 4: Verify deterministic noise generation with fixed seed."""
    signal = np.exp(1j * np.linspace(0, 100, 500))

    res1 = add_awgn_noise(signal, snr_db=10.0, seed=123)
    res2 = add_awgn_noise(signal, snr_db=10.0, seed=123)
    res3 = add_awgn_noise(signal, snr_db=10.0, seed=999)

    np.testing.assert_array_equal(res1.noisy_signal, res2.noisy_signal)
    assert not np.array_equal(res1.noisy_signal, res3.noisy_signal)


def test_compose_radar_environment() -> None:
    """Test 5: Verify compose_radar_environment combines clean signal, clutter, and AWGN."""
    clean_sig = np.ones((32, 500), dtype=np.complex128)  # P_sig = 1.0

    comp_res = compose_radar_environment(
        clean_signal=clean_sig,
        snr_db=20.0,
        clutter_power=0.5,
        seed=42
    )

    assert comp_res.composite_signal.shape == (32, 500)
    assert comp_res.clean_signal.shape == (32, 500)
    assert comp_res.noise_added is not None
    assert comp_res.clutter_added is not None
    assert comp_res.signal_power == pytest.approx(1.0, abs=1e-5)
    assert comp_res.clutter_power == pytest.approx(0.5, abs=0.1)
    assert comp_res.total_interference_power > 0
