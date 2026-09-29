"""Radar Target & Kinematics Module.

Defines target kinematics, delay & Doppler calculations, and RCS models.
Actual calculations will be implemented in Phase 2.
"""

from dataclasses import dataclass
from src.utils.config import TargetConfig


@dataclass
class TargetState:
    """State vector of a radar target at a given timestamp."""
    range_m: float
    velocity_mps: float
    rcs_sqm: float
    propagation_delay_sec: float
    doppler_shift_hz: float


def compute_target_state(target_cfg: TargetConfig, carrier_freq_hz: float) -> TargetState:
    """Compute propagation delay and Doppler shift for a given target.

    Args:
        target_cfg: Target initial range, velocity, and RCS configuration.
        carrier_freq_hz: Radar carrier frequency fc in Hz.

    Returns:
        TargetState containing theoretical delay, Doppler shift, and range.

    Raises:
        NotImplementedError: Implementation scheduled for Phase 2.
    """
    raise NotImplementedError("Target kinematics and delay/Doppler calculation will be implemented in Phase 2.")
