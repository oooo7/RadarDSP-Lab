"""Radar Statistical Background Clutter Module.

Provides complex-valued statistical clutter generation (Rayleigh magnitude envelope / complex Gaussian clutter)
for demonstrating CFAR adaptive thresholding in non-ideal background noise environments.

Note on Scope:
    This model serves as an educational and algorithmic statistical clutter baseline for CFAR validation.
    It does not simulate full physical urban, sea-surface, or terrain electromagnetic scattering models.
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple
import numpy as np

from src.radar.noise import calculate_noise_power


@dataclass(frozen=True)
class ClutterConfig:
    """Configuration container for statistical clutter generation.

    Attributes:
        clutter_power: Total clutter average power P_clutter = E[|c|^2] (> 0).
        clutter_type: Statistical model type ('gaussian' for Rayleigh magnitude / complex Gaussian).
        seed: Optional random seed for reproducible clutter generation.
    """
    clutter_power: float = 1.0
    clutter_type: str = "gaussian"
    seed: Optional[int] = None


@dataclass(frozen=True)
class ClutterResult:
    """Immutable payload holding generated clutter waveform and statistical metadata.

    Attributes:
        clutter_signal: 1D or 2D complex128 array of clutter waveform.
        clutter_power: Measured mean power P_clutter = E[|c|^2].
        target_clutter_power: Configured target clutter power.
        seed: Random seed used for generation.
        metadata: Additional clutter metadata dictionary.
    """
    clutter_signal: np.ndarray
    clutter_power: float
    target_clutter_power: float
    seed: Optional[int] = None
    metadata: dict[str, Any] = field(default_factory=dict)


def generate_clutter(
    shape: Tuple[int, ...],
    clutter_power: float = 1.0,
    seed: Optional[int] = None
) -> ClutterResult:
    """Generate complex-valued statistical background clutter array.

    Model Formulation:
        Complex Gaussian clutter background: c = c_I + j * c_Q
        c_I ~ N(0, P_clutter / 2), c_Q ~ N(0, P_clutter / 2)
        Envelope magnitude |c| follows a Rayleigh distribution with mean power E[|c|^2] = P_clutter.

    Args:
        shape: Tuple of dimensions for the clutter array (e.g., (128, 2000) or (2000,)).
        clutter_power: Total average clutter power P_clutter (> 0).
        seed: Random seed for deterministic NumPy default_rng generator.

    Returns:
        ClutterResult payload containing complex clutter array and measured power.

    Raises:
        ValueError: If clutter_power <= 0 or shape is empty.
    """
    p_clutter = float(clutter_power)
    if p_clutter <= 0 or not np.isfinite(p_clutter):
        raise ValueError(f"Clutter power must be strictly positive (> 0), got {p_clutter}")
    if not shape or any(dim <= 0 for dim in shape):
        raise ValueError(f"Shape dimensions must be positive integers, got {shape}")

    rng = np.random.default_rng(seed)
    std_comp = np.sqrt(p_clutter / 2.0)

    clutter_i = rng.normal(loc=0.0, scale=std_comp, size=shape)
    clutter_q = rng.normal(loc=0.0, scale=std_comp, size=shape)
    clutter_sig = clutter_i + 1j * clutter_q

    p_meas = calculate_noise_power(clutter_sig)

    return ClutterResult(
        clutter_signal=clutter_sig,
        clutter_power=p_meas,
        target_clutter_power=p_clutter,
        seed=seed,
        metadata={
            "clutter_type": "complex_gaussian_rayleigh",
            "shape": shape,
        }
    )


def add_clutter(
    signal: np.ndarray,
    clutter_power: float = 1.0,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, ClutterResult]:
    """Add complex statistical background clutter to an input signal array.

    Args:
        signal: 1D or 2D real or complex numpy array.
        clutter_power: Total clutter average power (> 0).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (cluttered_signal array, ClutterResult payload).

    Raises:
        ValueError: If signal or clutter_power is invalid.
    """
    arr = np.asarray(signal)
    if not np.all(np.isfinite(arr)):
        raise ValueError("Input signal contains non-finite NaN or Inf values.")

    clutter_res = generate_clutter(shape=arr.shape, clutter_power=clutter_power, seed=seed)
    cluttered_signal = arr + clutter_res.clutter_signal

    return cluttered_signal, clutter_res
