"""Digital Filtering & Resampling Module.

FIR/IIR filter design, frequency/phase response analysis, interpolation,
decimation, and anti-aliasing/anti-imaging filtering.
Actual algorithms will be implemented in Phase 1.
"""

from dataclasses import dataclass
import numpy as np
from src.dsp.signals import SignalContainer
from src.utils.config import FilterConfig


@dataclass
class FilterResponse:
    """Payload holding frequency axis, magnitude response (dB), and phase response (rad)."""
    frequencies_hz: np.ndarray
    magnitude_db: np.ndarray
    phase_rad: np.ndarray
    group_delay_sec: np.ndarray
    b_coefficients: np.ndarray
    a_coefficients: np.ndarray


def design_filter(config: FilterConfig, fs_hz: float) -> FilterResponse:
    """Design FIR or IIR filter and compute frequency/phase response.

    Args:
        config: Filter configuration parameters.
        fs_hz: Sampling frequency in Hz.

    Returns:
        FilterResponse with coefficients and magnitude/phase frequency responses.

    Raises:
        NotImplementedError: Implementation scheduled for Phase 1.
    """
    raise NotImplementedError("Filter design logic will be implemented in Phase 1.")


def apply_filter(signal: SignalContainer, filter_resp: FilterResponse) -> SignalContainer:
    """Filter time-domain signal using designed filter coefficients.

    Args:
        signal: Input time-domain signal.
        filter_resp: Filter design container holding b, a coefficients.

    Returns:
        Filtered SignalContainer.

    Raises:
        NotImplementedError: Implementation scheduled for Phase 1.
    """
    raise NotImplementedError("Filter execution logic will be implemented in Phase 1.")


def resample_signal(
    signal: SignalContainer,
    factor_up: int = 1,
    factor_down: int = 1
) -> SignalContainer:
    """Interpolate (upsample) or decimate (downsample) signal with anti-aliasing filtering.

    Args:
        signal: Input signal container.
        factor_up: Interpolation factor L.
        factor_down: Decimation factor M.

    Returns:
        Resampled SignalContainer.

    Raises:
        NotImplementedError: Implementation scheduled for Phase 1.
    """
    raise NotImplementedError("Resampling logic will be implemented in Phase 1.")
