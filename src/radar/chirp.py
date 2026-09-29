"""FMCW Chirp Generation Module.

Provides generation of Linear Frequency Modulated (LFM) FMCW radar transmitted chirps
in complex baseband or real passband representation.
"""

from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np
import scipy.constants

from src.utils.config import RadarConfig

# Physical constant: Speed of light in m/s
SPEED_OF_LIGHT_M_PER_S = float(scipy.constants.c)


@dataclass(frozen=True)
class FMCWChirpContainer:
    """Immutable payload holding transmitted FMCW chirp time-domain signal, frequency trajectory, and metadata.

    Attributes:
        time_vector: 1D float64 time array t[n] = n / Fs (endpoint excluded).
        tx_signal: 1D float64 or complex128 array of transmitted chirp signal.
        instantaneous_freq_hz: 1D float64 array of instantaneous baseband sweep frequency.
        chirp_slope_hz_per_sec: Linear chirp sweep rate S = B / T_c in Hz/s.
        bandwidth_hz: Total frequency sweep bandwidth B in Hz.
        chirp_duration_sec: Chirp duration T_c in seconds.
        sampling_rate_hz: ADC sampling rate Fs in Hz.
        carrier_frequency_hz: Radar carrier frequency fc in Hz (e.g. 77 GHz).
        amplitude: Peak signal amplitude A.
        initial_phase_rad: Initial phase phi_0 in radians.
        is_complex_baseband: True if signal is complex baseband exp(j*pi*S*t^2).
        config: Original RadarConfig model.
        metadata: Additional metadata dictionary.
    """
    time_vector: np.ndarray
    tx_signal: np.ndarray
    instantaneous_freq_hz: np.ndarray
    chirp_slope_hz_per_sec: float
    bandwidth_hz: float
    chirp_duration_sec: float
    sampling_rate_hz: float
    carrier_frequency_hz: float
    amplitude: float
    initial_phase_rad: float
    is_complex_baseband: bool
    config: RadarConfig
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_samples(self) -> int:
        """Total number of discrete samples in chirp."""
        return len(self.time_vector)

    @property
    def range_resolution_m(self) -> float:
        """Theoretical FMCW radar range resolution Delta R = c / (2*B) in meters."""
        return SPEED_OF_LIGHT_M_PER_S / (2.0 * self.bandwidth_hz)

    @property
    def wavelength_m(self) -> float:
        """Radar operating wavelength lambda = c / fc in meters."""
        return SPEED_OF_LIGHT_M_PER_S / self.carrier_frequency_hz


def generate_fmcw_chirp(
    config: RadarConfig,
    amplitude: float = 1.0,
    initial_phase_rad: float = 0.0,
    is_complex_baseband: bool = True
) -> FMCWChirpContainer:
    """Generate deterministic transmitted Linear Frequency Modulated (LFM) FMCW chirp signal.

    Mathematical Model:
        Chirp Slope: S = B / T_c
        Complex Baseband: s_tx(t) = A * exp( j * (pi * S * t^2 + phi_0) )
        Real Baseband:    s_tx(t) = A * cos( pi * S * t^2 + phi_0 )

    Args:
        config: FMCW RadarConfig configuration containing fc, B, T_c, Fs.
        amplitude: Peak signal amplitude A (> 0).
        initial_phase_rad: Initial phase phi_0 in radians.
        is_complex_baseband: If True, returns complex analytic baseband signal.

    Returns:
        FMCWChirpContainer containing time vector, tx signal, sweep frequency, and metadata.

    Raises:
        ValueError: If configuration parameters are invalid or non-finite.
    """
    fc = float(config.carrier_frequency_hz)
    b = float(config.sweep_bandwidth_hz)
    tc = float(config.chirp_duration_sec)
    fs = float(config.sampling_rate_hz)
    a = float(amplitude)
    phi0 = float(initial_phase_rad)

    if fc <= 0:
        raise ValueError(f"Carrier frequency fc must be strictly positive (> 0), got {fc} Hz")
    if b <= 0:
        raise ValueError(f"Sweep bandwidth B must be strictly positive (> 0), got {b} Hz")
    if tc <= 0:
        raise ValueError(f"Chirp duration T_c must be strictly positive (> 0), got {tc} s")
    if fs <= 0:
        raise ValueError(f"Sampling rate Fs must be strictly positive (> 0), got {fs} Hz")
    if a <= 0 or not np.isfinite(a):
        raise ValueError(f"Amplitude must be strictly positive and finite (> 0), got {a}")
    if not np.isfinite(phi0):
        raise ValueError(f"Initial phase must be a finite number, got {phi0}")

    if fs < b and is_complex_baseband:
        # Note: In baseband FMCW processing, Fs should ideally meet or exceed sweep bandwidth B
        pass

    slope = b / tc
    num_samples = int(round(fs * tc))
    if num_samples < 2:
        raise ValueError(f"Chirp duration T_c produces too few samples ({num_samples}) at Fs={fs} Hz")

    # Time vector using endpoint-excluded convention
    t = np.arange(num_samples, dtype=np.float64) / fs
    f_inst = slope * t

    if is_complex_baseband:
        phase_arg = np.pi * slope * (t**2) + phi0
        tx = a * np.exp(1j * phase_arg)
    else:
        phase_arg = np.pi * slope * (t**2) + phi0
        tx = a * np.cos(phase_arg)

    return FMCWChirpContainer(
        time_vector=t,
        tx_signal=tx,
        instantaneous_freq_hz=f_inst,
        chirp_slope_hz_per_sec=slope,
        bandwidth_hz=b,
        chirp_duration_sec=tc,
        sampling_rate_hz=fs,
        carrier_frequency_hz=fc,
        amplitude=a,
        initial_phase_rad=phi0,
        is_complex_baseband=is_complex_baseband,
        config=config,
        metadata={
            "num_samples": num_samples,
            "range_resolution_m": SPEED_OF_LIGHT_M_PER_S / (2.0 * b),
            "wavelength_m": SPEED_OF_LIGHT_M_PER_S / fc,
        },
    )
