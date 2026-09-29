"""FMCW Chirp Generation Module.

Generates linear frequency modulated (LFM) transmitted waveforms (chirps)
and multi-chirp frame sequences.
Actual algorithms will be implemented in Phase 2.
"""

from dataclasses import dataclass
import numpy as np
from src.utils.config import RadarConfig


@dataclass
class FMCWChirpContainer:
    """Payload holding transmitted chirp time-domain signal, frequency trajectory, and radar parameters."""
    time_vector: np.ndarray
    tx_signal: np.ndarray
    instantaneous_freq_hz: np.ndarray
    chirp_slope_hz_per_sec: float
    config: RadarConfig


def generate_fmcw_chirp(config: RadarConfig) -> FMCWChirpContainer:
    """Generate transmitted LFM FMCW chirp signal.

    Args:
        config: FMCW Radar configuration parameters.

    Returns:
        FMCWChirpContainer with time vector, tx signal, and slope metadata.

    Raises:
        NotImplementedError: Implementation scheduled for Phase 2.
    """
    raise NotImplementedError("FMCW chirp generation will be implemented in Phase 2.")
