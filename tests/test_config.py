"""Unit tests for configuration models in src.utils.config."""

import pytest
from pydantic import ValidationError
from src.utils.config import (
    SignalGenConfig,
    FilterConfig,
    TargetConfig,
    CFARConfig,
    RadarConfig,
    LabConfig,
)


def test_signal_gen_config_defaults() -> None:
    """Test default initialization of SignalGenConfig."""
    cfg = SignalGenConfig()
    assert cfg.signal_type == "sine"
    assert cfg.sampling_rate_hz == 1000.0
    assert cfg.duration_sec == 1.0
    assert cfg.frequencies_hz == [100.0]


def test_signal_gen_config_validation() -> None:
    """Test parameter validation rules on SignalGenConfig."""
    with pytest.raises(ValidationError):
        # Sampling rate must be strictly > 0
        SignalGenConfig(sampling_rate_hz=0.0)

    with pytest.raises(ValidationError):
        # Duration must be strictly > 0
        SignalGenConfig(duration_sec=-1.0)


def test_target_config_validation() -> None:
    """Test target range and RCS bounds."""
    target = TargetConfig(range_m=50.0, velocity_mps=12.5, rcs_sqm=1.5)
    assert target.range_m == 50.0
    assert target.velocity_mps == 12.5

    with pytest.raises(ValidationError):
        # Negative range should fail validation
        TargetConfig(range_m=-10.0)


def test_radar_config_defaults() -> None:
    """Test default RadarConfig setup."""
    radar = RadarConfig()
    assert radar.carrier_frequency_hz == 77e9
    assert radar.sweep_bandwidth_hz == 150e6
    assert len(radar.targets) == 1
    assert radar.targets[0].range_m == 50.0


def test_lab_config_instantiation() -> None:
    """Test top-level LabConfig creation."""
    lab = LabConfig()
    assert isinstance(lab.dsp, SignalGenConfig)
    assert isinstance(lab.filter, FilterConfig)
    assert isinstance(lab.radar, RadarConfig)
