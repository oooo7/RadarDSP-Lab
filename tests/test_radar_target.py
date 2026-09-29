"""Unit tests for Radar Target Kinematics Module (src/radar/target.py)."""

import pytest
from src.radar.target import TargetState, compute_target_state
from src.utils.config import TargetConfig


def test_target_state_computation() -> None:
    """Test 1: Verify propagation delay tau = 2*R/c and theoretical beat frequency fb = S * tau."""
    target = TargetConfig(range_m=100.0, velocity_mps=0.0, rcs_sqm=1.0)
    slope = 1.5e12  # 150 MHz / 100 us

    state = compute_target_state(target, chirp_slope_hz_per_sec=slope)

    assert isinstance(state, TargetState)
    assert state.range_m == 100.0
    # tau = 200 / 299792458 = 6.671282e-7 s
    assert state.propagation_delay_sec == pytest.approx(6.671282e-7, abs=1e-10)
    # fb = 1.5e12 * 6.671282e-7 = 1000692.29 Hz
    assert state.theoretical_beat_frequency_hz == pytest.approx(1000692.29, abs=1.0)


def test_target_validation_errors() -> None:
    """Test 2: Verify ValueError exceptions on invalid range or slope inputs."""
    with pytest.raises((ValueError, TypeError)):
        TargetConfig(range_m=-50.0)

    with pytest.raises(ValueError, match="strictly positive"):
        compute_target_state(TargetConfig(range_m=50.0), chirp_slope_hz_per_sec=-1.5e12)

