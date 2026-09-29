"""Unit tests for Radar Propagation Channel Engine (src/radar/propagation.py)."""

import numpy as np
import pytest
from src.radar.chirp import generate_fmcw_chirp
from src.radar.propagation import (
    RadarRxPayload,
    apply_propagation_delay,
    calculate_round_trip_delay,
    simulate_target_echo,
)
from src.utils.config import RadarConfig, TargetConfig


def test_calculate_round_trip_delay() -> None:
    """Test 1: Verify exact delay tau = 2*R/c calculation."""
    tau_100 = calculate_round_trip_delay(100.0)
    assert tau_100 == pytest.approx(6.671282e-7, abs=1e-10)

    with pytest.raises(ValueError, match="strictly positive"):
        calculate_round_trip_delay(-10.0)


def test_integer_and_fractional_delay_synthesis() -> None:
    """Test 2: Verify continuous analytical sub-sample fractional delay synthesis."""
    cfg = RadarConfig(
        carrier_frequency_hz=77e9,
        sweep_bandwidth_hz=150e6,
        chirp_duration_sec=100e-6,
        sampling_rate_hz=10e6
    )
    chirp = generate_fmcw_chirp(cfg)

    # 100m target delay (6.671282 fractional samples)
    tau = calculate_round_trip_delay(100.0)
    rx = apply_propagation_delay(chirp, delay_sec=tau, amplitude_scaling=0.8)

    assert len(rx) == len(chirp.time_vector)
    assert np.all(np.isfinite(rx))
    # Peak envelope amplitude is scaled by 0.8
    assert np.max(np.abs(rx)) == pytest.approx(0.8, abs=1e-5)


def test_simulate_target_echo_payload() -> None:
    """Test 3: Verify simulate_target_echo generates valid beat signal payload."""
    cfg = RadarConfig(
        carrier_frequency_hz=77e9,
        sweep_bandwidth_hz=150e6,
        chirp_duration_sec=100e-6,
        sampling_rate_hz=10e6
    )
    chirp = generate_fmcw_chirp(cfg)
    target = TargetConfig(range_m=75.0, velocity_mps=0.0, rcs_sqm=1.0)

    payload = simulate_target_echo(chirp, target, amplitude_scaling=1.0)

    assert isinstance(payload, RadarRxPayload)
    assert payload.target_state.range_m == 75.0
    assert len(payload.beat_signal) == len(chirp.time_vector)
    assert np.all(np.isfinite(payload.beat_signal))


def test_propagation_excessive_delay_error() -> None:
    """Test 4: Verify ValueError exception when target delay exceeds chirp duration."""
    cfg = RadarConfig(
        carrier_frequency_hz=77e9,
        sweep_bandwidth_hz=150e6,
        chirp_duration_sec=10e-6,  # 10 us
        sampling_rate_hz=10e6
    )
    chirp = generate_fmcw_chirp(cfg)
    # Target at 3000 m produces tau = 20 us > 10 us
    target = TargetConfig(range_m=3000.0, velocity_mps=0.0, rcs_sqm=1.0)

    with pytest.raises(ValueError, match="exceeds total chirp duration"):
        simulate_target_echo(chirp, target)
