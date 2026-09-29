"""FMCW Chirp Generation & Sampling Architecture Module.

Provides mathematical representation of Linear Frequency Modulated (LFM) FMCW radar chirps
under two distinct sampling architectures:
- OPTION B (Default / Standard Stretch Processing): Continuous-time analytic chirp phase definition
  for analog RF mixer dechirp processing before digitizer ADC sampling (Fs is tied to beat frequency bandwidth).
- OPTION A (Directly Sampled Chirp Receiver): Digitally sampled raw chirp waveform (Fs >= Bandwidth B).
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
    """Immutable payload holding FMCW chirp parameters, continuous/discrete signals, and sampling architecture.

    Attributes:
        time_vector: 1D float64 time array t[n] = n / Fs (endpoint excluded).
        tx_signal: 1D float64 or complex128 array of chirp signal.
        instantaneous_freq_hz: 1D float64 array of instantaneous baseband sweep frequency.
        chirp_slope_hz_per_sec: Linear chirp sweep rate S = B / T_c in Hz/s.
        bandwidth_hz: Total frequency sweep bandwidth B in Hz.
        chirp_duration_sec: Chirp duration T_c in seconds.
        sampling_rate_hz: ADC sampling rate Fs in Hz.
        carrier_frequency_hz: Radar carrier frequency fc in Hz (e.g. 77 GHz).
        amplitude: Peak signal amplitude A.
        initial_phase_rad: Initial phase phi_0 in radians.
        is_complex_baseband: True if signal is complex baseband exp(j*pi*S*t^2).
        sampling_architecture: 'stretch_dechirp_analog' (Option B) or 'direct_sampled_chirp' (Option A).
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
    sampling_architecture: str
    config: RadarConfig
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_samples(self) -> int:
        """Total number of discrete samples in time vector."""
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
    is_complex_baseband: bool = True,
    sampling_architecture: str = "stretch_dechirp_analog"
) -> FMCWChirpContainer:
    """Generate deterministic LFM FMCW chirp container under specified sampling architecture.

    Sampling Architectures:
        - Option B ('stretch_dechirp_analog', Default):
          Defines continuous-time analytic chirp phase Phi_tx(t) = pi*S*t^2 + phi_0.
          Models hardware analog mixer dechirp processing before ADC sampling.
          The ADC sampling rate Fs is selected to satisfy Nyquist for the dechirped beat signal
          (Fs >= f_b_max = 2*S*R_max/c), avoiding artificial raw chirp undersampling.

        - Option A ('direct_sampled_chirp'):
          Digitally samples the raw chirp waveform directly before mixing.
          Requires Fs >= B (for complex baseband) or Fs >= 2*B (for real passband).

    Args:
        config: FMCW RadarConfig configuration containing fc, B, T_c, Fs.
        amplitude: Peak signal amplitude A (> 0).
        initial_phase_rad: Initial phase phi_0 in radians.
        is_complex_baseband: If True, uses complex baseband representation.
        sampling_architecture: 'stretch_dechirp_analog' or 'direct_sampled_chirp'.

    Returns:
        FMCWChirpContainer with time vector, tx signal, slope, and sampling metadata.

    Raises:
        ValueError: If configuration parameters or sampling requirements are violated.
    """
    fc = float(config.carrier_frequency_hz)
    b = float(config.sweep_bandwidth_hz)
    tc = float(config.chirp_duration_sec)
    fs = float(config.sampling_rate_hz)
    a = float(amplitude)
    phi0 = float(initial_phase_rad)
    arch = sampling_architecture.lower().strip()

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

    if arch not in ["stretch_dechirp_analog", "direct_sampled_chirp"]:
        raise ValueError(
            f"Unsupported sampling_architecture '{sampling_architecture}'. "
            "Supported: 'stretch_dechirp_analog', 'direct_sampled_chirp'."
        )

    # Validate Option A sampling requirements if raw chirp is directly sampled
    if arch == "direct_sampled_chirp":
        min_fs_req = b if is_complex_baseband else (2.0 * b)
        if fs < min_fs_req:
            raise ValueError(
                f"Directly sampled chirp architecture ('direct_sampled_chirp') requires Fs >= {min_fs_req/1e6:.1f} MHz "
                f"to satisfy Nyquist for chirp bandwidth B={b/1e6:.1f} MHz, but Fs={fs/1e6:.1f} MHz was provided."
            )

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
        sampling_architecture=arch,
        config=config,
        metadata={
            "num_samples": num_samples,
            "range_resolution_m": SPEED_OF_LIGHT_M_PER_S / (2.0 * b),
            "wavelength_m": SPEED_OF_LIGHT_M_PER_S / fc,
            "sampling_architecture_description": (
                "Option B: Analog Stretch Dechirp (Fs is tied to beat frequency bandwidth)"
                if arch == "stretch_dechirp_analog"
                else "Option A: Directly Sampled Chirp (Fs >= Bandwidth B)"
            ),
        },
    )
