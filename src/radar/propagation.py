"""Radar Propagation Channel & Signal Mixing Module.

Provides physical round-trip propagation delay calculation, fractional-delay echo synthesis,
and de-chirping (mixing transmit and receive signals to generate the beat signal).
"""

from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np
import scipy.constants

from src.radar.chirp import FMCWChirpContainer
from src.radar.target import TargetConfig, TargetState, compute_target_state
from src.utils.config import RadarConfig

# Speed of light in m/s
SPEED_OF_LIGHT_M_PER_S = float(scipy.constants.c)


@dataclass(frozen=True)
class RadarRxPayload:
    """Immutable payload holding received echo signal, de-chirped beat signal, and channel metadata.

    Attributes:
        rx_signal: 1D float64 or complex128 array of received echo signal s_rx(t).
        beat_signal: 1D float64 or complex128 array of de-chirped beat signal s_beat(t).
        time_vector: 1D float64 time vector t[n].
        target_state: TargetState object holding range, delay, and theoretical beat frequency.
        round_trip_delay_sec: Two-way delay tau = 2*R/c in seconds.
        snr_db: Optional Signal-to-Noise ratio in dB (None for noiseless channel).
        metadata: Additional channel metadata dictionary.
    """
    rx_signal: np.ndarray
    beat_signal: np.ndarray
    time_vector: np.ndarray
    target_state: TargetState
    round_trip_delay_sec: float
    snr_db: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)


def calculate_round_trip_delay(range_m: float) -> float:
    """Calculate exact two-way round-trip radar propagation delay tau = 2 * R / c.

    Args:
        range_m: Target range R in meters (> 0).

    Returns:
        Round-trip delay tau in seconds.

    Raises:
        ValueError: If range_m <= 0.
    """
    if range_m <= 0:
        raise ValueError(f"Target range must be strictly positive (> 0), got {range_m} m")
    return 2.0 * float(range_m) / SPEED_OF_LIGHT_M_PER_S


def apply_propagation_delay(
    tx_chirp: FMCWChirpContainer,
    delay_sec: float,
    amplitude_scaling: float = 1.0,
    phase_offset_rad: float = 0.0
) -> np.ndarray:
    """Synthesize delayed received echo signal with continuous sub-sample fractional delay accuracy.

    Analytical Fractional Delay Model:
        For FMCW chirp with slope S = B / T_c, the delayed received echo at time t is:
        s_rx(t) = alpha * A * exp( j * (pi * S * (t - tau)^2 + phi_0 + phi_target) )

        This analytical evaluation is mathematically exact for any fractional sample delay
        tau = 2*R/c without discretization or interpolation artifacts.

    Args:
        tx_chirp: Transmitted FMCWChirpContainer payload.
        delay_sec: Round-trip propagation delay tau in seconds (>= 0).
        amplitude_scaling: Reflection amplitude scale factor alpha (> 0).
        phase_offset_rad: Reflection phase offset in radians.

    Returns:
        1D float64 or complex128 numpy array of delayed received signal s_rx(t).

    Raises:
        ValueError: If delay_sec < 0 or amplitude_scaling <= 0.
    """
    if delay_sec < 0:
        raise ValueError(f"Propagation delay must be non-negative (>= 0), got {delay_sec} s")
    if amplitude_scaling <= 0:
        raise ValueError(f"Amplitude scaling must be strictly positive (> 0), got {amplitude_scaling}")

    tau = float(delay_sec)
    alpha = float(amplitude_scaling)
    phi_target = float(phase_offset_rad)

    t = tx_chirp.time_vector
    s = tx_chirp.chirp_slope_hz_per_sec
    a = tx_chirp.amplitude
    phi0 = tx_chirp.initial_phase_rad

    if tx_chirp.is_complex_baseband:
        phase_arg = np.pi * s * ((t - tau)**2) + phi0 + phi_target
        rx = alpha * a * np.exp(1j * phase_arg)
    else:
        phase_arg = np.pi * s * ((t - tau)**2) + phi0 + phi_target
        rx = alpha * a * np.cos(phase_arg)

    return rx


def simulate_target_echo(
    tx_chirp: FMCWChirpContainer,
    target_cfg: TargetConfig,
    amplitude_scaling: float = 1.0,
    phase_offset_rad: float = 0.0
) -> RadarRxPayload:
    """Simulate radar echo channel propagation and de-chirp mixing for a stationary target.

    De-chirping Operation:
        s_beat(t) = s_tx(t) * conj(s_rx(t))
        For baseband chirps s_tx = exp(j*pi*S*t^2) and s_rx = exp(j*pi*S*(t-tau)^2):
        s_beat(t) = exp( j * (2*pi*S*tau*t - pi*S*tau^2) )
        The beat signal frequency is f_b = S * tau = 2 * S * R / c.

    Args:
        tx_chirp: Transmitted FMCWChirpContainer payload.
        target_cfg: TargetConfig specifying target range R.
        amplitude_scaling: Reflection scale factor alpha.
        phase_offset_rad: Reflection phase offset.

    Returns:
        RadarRxPayload container holding received echo and de-chirped beat signal.

    Raises:
        ValueError: If target range <= 0 or delay exceeds chirp duration.
    """
    target_state = compute_target_state(
        target_cfg=target_cfg,
        chirp_slope_hz_per_sec=tx_chirp.chirp_slope_hz_per_sec,
        amplitude_scaling=amplitude_scaling,
        phase_offset_rad=phase_offset_rad
    )

    tau = target_state.propagation_delay_sec
    if tau >= tx_chirp.chirp_duration_sec:
        raise ValueError(
            f"Target range {target_cfg.range_m} m produces delay tau={tau*1e6:.2f} us "
            f"which exceeds total chirp duration T_c={tx_chirp.chirp_duration_sec*1e6:.2f} us"
        )

    # Synthesize received echo
    rx_signal = apply_propagation_delay(
        tx_chirp=tx_chirp,
        delay_sec=tau,
        amplitude_scaling=amplitude_scaling,
        phase_offset_rad=phase_offset_rad
    )

    # De-chirping / Mixing: s_beat(t) = s_tx(t) * conj(s_rx(t))
    if tx_chirp.is_complex_baseband:
        beat_signal = tx_chirp.tx_signal * np.conj(rx_signal)
    else:
        # Real beat signal mixing: s_tx(t) * s_rx(t)
        beat_signal = tx_chirp.tx_signal * rx_signal

    return RadarRxPayload(
        rx_signal=rx_signal,
        beat_signal=beat_signal,
        time_vector=tx_chirp.time_vector,
        target_state=target_state,
        round_trip_delay_sec=tau,
        snr_db=None,
        metadata={
            "theoretical_beat_frequency_hz": target_state.theoretical_beat_frequency_hz,
            "delay_fractional_samples": tau * tx_chirp.sampling_rate_hz,
        },
    )
