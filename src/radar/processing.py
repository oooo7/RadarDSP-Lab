"""FMCW Radar Range & Doppler Signal Processing Module.

Performs 1D Range FFT on beat signal and 2D Range-Doppler processing across multi-chirp frames.
Actual algorithms will be implemented in Phase 2.
"""

from dataclasses import dataclass
import numpy as np
from src.radar.propagation import RadarRxPayload
from src.utils.config import RadarConfig


@dataclass
class RangeFFTResult:
    """Payload holding Range axis (m), beat frequency spectrum (dB), and magnitude spectrum."""
    range_axis_m: np.ndarray
    beat_freq_axis_hz: np.ndarray
    spectrum_db: np.ndarray


@dataclass
class RangeDopplerMap:
    """Payload holding 2D Range-Doppler matrix, Range axis, Velocity axis, and power magnitude in dB."""
    range_axis_m: np.ndarray
    velocity_axis_mps: np.ndarray
    rd_map_db: np.ndarray


def compute_range_fft(rx_payload: RadarRxPayload, config: RadarConfig) -> RangeFFTResult:
    """Compute 1D Range FFT on de-chirped beat signal.

    Args:
        rx_payload: Received signal payload containing beat signal.
        config: Radar configuration options.

    Returns:
        RangeFFTResult with range axis and spectrum.

    Raises:
        NotImplementedError: Implementation scheduled for Phase 2.
    """
    raise NotImplementedError("Range FFT processing will be implemented in Phase 2.")


def compute_range_doppler_map(
    multi_chirp_beat_signals: np.ndarray,
    config: RadarConfig
) -> RangeDopplerMap:
    """Compute 2D Range-Doppler FFT matrix across fast-time and slow-time.

    Args:
        multi_chirp_beat_signals: 2D array (num_chirps x num_samples) of de-chirped beat signals.
        config: Radar configuration options.

    Returns:
        RangeDopplerMap holding 2D range-doppler power distribution matrix.

    Raises:
        NotImplementedError: Implementation scheduled for Phase 2.
    """
    raise NotImplementedError("2D Range-Doppler map processing will be implemented in Phase 2.")
