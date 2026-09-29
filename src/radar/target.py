"""Radar Target & Kinematics Module.

Defines target configuration, round-trip propagation delay calculation,
and theoretical beat frequency prediction for stationary FMCW radar targets.
"""

from dataclasses import dataclass, field
from typing import Any
import scipy.constants

from src.utils.config import TargetConfig

# Speed of light in m/s
SPEED_OF_LIGHT_M_PER_S = float(scipy.constants.c)


@dataclass(frozen=True)
class TargetState:
    """Immutable state payload for a stationary or moving radar target.

    Attributes:
        target_id: Unique string identifier for target (e.g. 'T1').
        range_m: Target range R in meters (R > 0).
        velocity_mps: Target radial velocity v in m/s (positive = receding, moving away).
        rcs_sqm: Radar Cross Section (RCS) in m^2.
        amplitude_scaling: Linear reflection amplitude scale factor alpha (default 1.0).
        phase_offset_rad: Reflection phase offset in radians.
        propagation_delay_sec: Two-way round-trip delay tau = 2*R/c in seconds.
        theoretical_beat_frequency_hz: Predicted beat frequency f_b = S * tau in Hz.
        theoretical_doppler_frequency_hz: Predicted Doppler frequency f_D = 2*v/lambda in Hz.
        metadata: Additional metadata dictionary.
    """
    target_id: str
    range_m: float
    velocity_mps: float
    rcs_sqm: float
    amplitude_scaling: float
    phase_offset_rad: float
    propagation_delay_sec: float
    theoretical_beat_frequency_hz: float
    theoretical_doppler_frequency_hz: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


def compute_target_state(
    target_cfg: TargetConfig,
    chirp_slope_hz_per_sec: float,
    amplitude_scaling: float = 1.0,
    phase_offset_rad: float = 0.0,
    carrier_frequency_hz: float = 77e9
) -> TargetState:
    """Compute exact physical propagation delay, beat frequency, and Doppler frequency for a target.

    Formulas & Conventions:
        Velocity Convention: Positive velocity (v > 0) means target is moving away (receding).
        Round-trip delay: tau = 2 * R / c
        Theoretical beat frequency: f_b = S * tau = 2 * S * R / c
        Theoretical Doppler frequency: f_D = 2 * v / lambda = 2 * v * fc / c

    Args:
        target_cfg: TargetConfig object containing range_m, velocity_mps, rcs_sqm, target_id.
        chirp_slope_hz_per_sec: FMCW chirp slope S = B / T_c in Hz/s.
        amplitude_scaling: Reflection amplitude scale factor alpha (> 0).
        phase_offset_rad: Target reflection phase offset in radians.
        carrier_frequency_hz: Radar carrier frequency fc in Hz (default 77 GHz).

    Returns:
        TargetState container holding physical delay, beat frequency, and Doppler frequency.

    Raises:
        ValueError: If range <= 0 or chirp slope <= 0 or parameters are invalid.
    """
    r = float(target_cfg.range_m)
    v = float(target_cfg.velocity_mps)
    rcs = float(target_cfg.rcs_sqm)
    slope = float(chirp_slope_hz_per_sec)
    alpha = float(target_cfg.amplitude) if hasattr(target_cfg, "amplitude") and target_cfg.amplitude is not None else float(amplitude_scaling)
    phi = float(target_cfg.phase_rad) if hasattr(target_cfg, "phase_rad") and target_cfg.phase_rad is not None else float(phase_offset_rad)
    fc = float(carrier_frequency_hz)
    t_id = str(target_cfg.target_id) if (hasattr(target_cfg, "target_id") and target_cfg.target_id is not None) else "T1"

    if r <= 0:
        raise ValueError(f"Target range must be strictly positive (> 0), got {r} m")
    if slope <= 0:
        raise ValueError(f"Chirp slope must be strictly positive (> 0), got {slope} Hz/s")
    if rcs <= 0:
        raise ValueError(f"Target RCS must be strictly positive (> 0), got {rcs} m^2")
    if alpha <= 0:
        raise ValueError(f"Amplitude scaling must be strictly positive (> 0), got {alpha}")
    if fc <= 0:
        raise ValueError(f"Carrier frequency must be strictly positive (> 0), got {fc} Hz")

    delay_sec = 2.0 * r / SPEED_OF_LIGHT_M_PER_S
    fb_hz = slope * delay_sec
    wavelength_m = SPEED_OF_LIGHT_M_PER_S / fc
    fd_hz = 2.0 * v / wavelength_m

    return TargetState(
        target_id=t_id,
        range_m=r,
        velocity_mps=v,
        rcs_sqm=rcs,
        amplitude_scaling=alpha,
        phase_offset_rad=phi,
        propagation_delay_sec=delay_sec,
        theoretical_beat_frequency_hz=fb_hz,
        theoretical_doppler_frequency_hz=fd_hz,
        metadata={
            "speed_of_light_m_per_s": SPEED_OF_LIGHT_M_PER_S,
            "carrier_frequency_hz": fc,
            "wavelength_m": wavelength_m,
        },
    )
