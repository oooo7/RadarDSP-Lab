"""Radar Complex Additive White Gaussian Noise (AWGN) & Signal-to-Noise Ratio (SNR) Engine.

Provides exact power measurement, SNR calculations, complex AWGN noise generation,
and environmental signal composition (clean signal + clutter + noise).
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple, Union
import numpy as np


@dataclass(frozen=True)
class ComplexNoiseResult:
    """Immutable payload holding noise generation metrics and noisy signal payload.

    Attributes:
        noisy_signal: 1D or 2D numpy array of signal plus added noise.
        noise_added: 1D or 2D numpy array of the generated noise waveform alone.
        signal_power: Average power P_signal = E[|x|^2] of clean input signal.
        noise_power: Measured average power P_noise = E[|w|^2] of generated noise.
        snr_db: Measured Signal-to-Noise Ratio in dB: 10 * log10(P_signal / P_noise).
        requested_snr_db: Requested SNR in dB (or None if noise power was specified directly).
        seed: Random seed used for deterministic generation.
        metadata: Additional metadata dictionary.
    """
    noisy_signal: np.ndarray
    noise_added: np.ndarray
    signal_power: float
    noise_power: float
    snr_db: float
    requested_snr_db: Optional[float] = None
    seed: Optional[int] = None
    metadata: dict[str, Any] = field(default_factory=dict)


def calculate_signal_power(signal: np.ndarray) -> float:
    """Calculate mean signal power P_signal = E[|x|^2].

    Args:
        signal: 1D or 2D real or complex numpy array.

    Returns:
        Mean signal power as a positive float64. Returns 0.0 if array is empty or zero.
    """
    arr = np.asarray(signal)
    if arr.size == 0:
        return 0.0
    return float(np.mean(np.abs(arr)**2))


def calculate_noise_power(noise: np.ndarray) -> float:
    """Calculate mean noise power P_noise = E[|w|^2].

    Args:
        noise: 1D or 2D real or complex numpy array.

    Returns:
        Mean noise power as a positive float64.
    """
    return calculate_signal_power(noise)


def calculate_snr_db(
    signal_or_power: Union[float, np.ndarray],
    noise_or_power: Union[float, np.ndarray]
) -> float:
    """Calculate Signal-to-Noise Ratio (SNR) in decibels (dB).

    Formula:
        SNR_dB = 10 * log10(P_signal / P_noise)

    Args:
        signal_or_power: Clean signal array or pre-calculated signal power (P_sig > 0).
        noise_or_power: Noise signal array or pre-calculated noise power (P_noise > 0).

    Returns:
        SNR in dB as a float64.

    Raises:
        ValueError: If signal power <= 0 or noise power <= 0.
    """
    p_sig = float(calculate_signal_power(signal_or_power)) if isinstance(signal_or_power, np.ndarray) else float(signal_or_power)
    p_noise = float(calculate_noise_power(noise_or_power)) if isinstance(noise_or_power, np.ndarray) else float(noise_or_power)

    if p_sig <= 0 or not np.isfinite(p_sig):
        raise ValueError(f"Signal power must be strictly positive (> 0), got {p_sig}")
    if p_noise <= 0 or not np.isfinite(p_noise):
        raise ValueError(f"Noise power must be strictly positive (> 0), got {p_noise}")

    return 10.0 * np.log10(p_sig / p_noise)


def add_awgn_noise(
    signal: np.ndarray,
    snr_db: Optional[float] = None,
    target_noise_power: Optional[float] = None,
    seed: Optional[int] = None
) -> ComplexNoiseResult:
    """Add Complex Additive White Gaussian Noise (AWGN) to input signal array.

    Noise Power & Component Scaling:
        For complex baseband noise w = w_I + j * w_Q:
        Total complex noise power P_noise = E[|w|^2] = E[w_I^2] + E[w_Q^2].
        Each quadrature component w_I, w_Q is independent Gaussian with variance:
        sigma_comp^2 = P_noise / 2.
        Therefore, w_I ~ N(0, P_noise / 2) and w_Q ~ N(0, P_noise / 2).

    Args:
        signal: 1D or 2D real or complex numpy array.
        snr_db: Requested Signal-to-Noise Ratio in dB.
        target_noise_power: Explicit noise power P_noise (alternative to snr_db).
        seed: Random seed for NumPy default_rng generator.

    Returns:
        ComplexNoiseResult dataclass container.

    Raises:
        ValueError: If neither snr_db nor target_noise_power is provided, or if values are invalid.
    """
    arr = np.asarray(signal)
    if not np.all(np.isfinite(arr)):
        raise ValueError("Input signal contains non-finite NaN or Inf values.")

    p_sig = calculate_signal_power(arr)

    if snr_db is not None:
        if p_sig <= 0:
            raise ValueError(f"Cannot scale noise by SNR for zero/empty signal (power={p_sig}).")
        snr_val = float(snr_db)
        p_noise_req = p_sig / (10.0**(snr_val / 10.0))
    elif target_noise_power is not None:
        p_noise_req = float(target_noise_power)
        if p_noise_req <= 0:
            raise ValueError(f"Target noise power must be strictly positive (> 0), got {p_noise_req}")
        snr_val = 10.0 * np.log10(p_sig / p_noise_req) if p_sig > 0 else 0.0
    else:
        raise ValueError("Must provide either 'snr_db' or 'target_noise_power'.")

    rng = np.random.default_rng(seed)

    if np.iscomplexobj(arr):
        std_comp = np.sqrt(p_noise_req / 2.0)
        noise_i = rng.normal(loc=0.0, scale=std_comp, size=arr.shape)
        noise_q = rng.normal(loc=0.0, scale=std_comp, size=arr.shape)
        noise = noise_i + 1j * noise_q
    else:
        std_real = np.sqrt(p_noise_req)
        noise = rng.normal(loc=0.0, scale=std_real, size=arr.shape)

    noisy_signal = arr + noise
    p_noise_measured = calculate_noise_power(noise)
    snr_measured = 10.0 * np.log10(p_sig / p_noise_measured) if p_sig > 0 else 0.0

    return ComplexNoiseResult(
        noisy_signal=noisy_signal,
        noise_added=noise,
        signal_power=p_sig,
        noise_power=p_noise_measured,
        snr_db=snr_measured,
        requested_snr_db=snr_db,
        seed=seed,
        metadata={
            "target_noise_power": p_noise_req,
            "measured_noise_power": p_noise_measured,
            "is_complex": np.iscomplexobj(arr),
            "shape": arr.shape,
        }
    )


@dataclass(frozen=True)
class CompositeEnvironmentResult:
    """Immutable payload holding clean signal, noise, clutter, and composite signal payload."""
    composite_signal: np.ndarray
    clean_signal: np.ndarray
    noise_added: Optional[np.ndarray]
    clutter_added: Optional[np.ndarray]
    signal_power: float
    noise_power: float
    clutter_power: float
    total_interference_power: float
    snr_db: Optional[float]
    seed: Optional[int] = None
    metadata: dict[str, Any] = field(default_factory=dict)


def compose_radar_environment(
    clean_signal: np.ndarray,
    snr_db: Optional[float] = None,
    clutter_power: Optional[float] = None,
    seed: Optional[int] = None
) -> CompositeEnvironmentResult:
    """Compose clean target signal with optional statistical background clutter and AWGN noise.

    Pipeline:
        composite = clean_target_signal + clutter + AWGN_noise

    Args:
        clean_signal: 1D or 2D numpy array of clean target signal.
        snr_db: Optional requested SNR in dB (adds AWGN if specified).
        clutter_power: Optional clutter power P_clutter (adds complex Gaussian clutter if specified).
        seed: Random seed for deterministic generation.

    Returns:
        CompositeEnvironmentResult container.
    """
    arr = np.asarray(clean_signal)
    if not np.all(np.isfinite(arr)):
        raise ValueError("Clean input signal contains non-finite NaN or Inf values.")

    p_sig = calculate_signal_power(arr)
    composite = arr.copy()

    # Import clutter generate_clutter locally to avoid circular import
    from src.radar.clutter import generate_clutter

    clutter_sig = None
    p_clutter = 0.0
    if clutter_power is not None and clutter_power > 0:
        c_res = generate_clutter(shape=arr.shape, clutter_power=clutter_power, seed=seed)
        clutter_sig = c_res.clutter_signal
        p_clutter = c_res.clutter_power
        composite = composite + clutter_sig

    noise_sig = None
    p_noise = 0.0
    measured_snr = None
    if snr_db is not None:
        # Generate AWGN with separate seed offset if seed is provided
        noise_seed = (seed + 1000) if seed is not None else None
        n_res = add_awgn_noise(signal=arr, snr_db=snr_db, seed=noise_seed)
        noise_sig = n_res.noise_added
        p_noise = n_res.noise_power
        composite = composite + noise_sig
        measured_snr = n_res.snr_db

    p_total_interf = p_noise + p_clutter

    return CompositeEnvironmentResult(
        composite_signal=composite,
        clean_signal=arr,
        noise_added=noise_sig,
        clutter_added=clutter_sig,
        signal_power=p_sig,
        noise_power=p_noise,
        clutter_power=p_clutter,
        total_interference_power=p_total_interf,
        snr_db=measured_snr,
        seed=seed,
        metadata={
            "has_clutter": clutter_sig is not None,
            "has_noise": noise_sig is not None,
        }
    )

