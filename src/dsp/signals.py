"""DSP Signal Generation Module.

Defines signal generation interfaces and data structures.
Actual mathematical generation algorithms will be implemented in Phase 1.
"""

from dataclasses import dataclass
import numpy as np
from src.utils.config import SignalGenConfig


@dataclass
class SignalContainer:
    """Immutable payload holding time vector, amplitude values, and metadata."""
    time_vector: np.ndarray
    amplitude: np.ndarray
    sampling_rate_hz: float
    description: str = ""


def generate_signal(config: SignalGenConfig) -> SignalContainer:
    """Generate time-domain signal according to SignalGenConfig.

    Args:
        config: Signal generation configuration options.

    Returns:
        SignalContainer with time vector and amplitude values.

    Raises:
        NotImplementedError: Implementation scheduled for Phase 1.
    """
    raise NotImplementedError("Signal generation logic will be implemented in Phase 1.")
