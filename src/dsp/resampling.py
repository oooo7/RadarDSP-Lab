"""Multirate Digital Signal Processing Engine.

Provides tools for discrete sampling-rate conversion, including:
- Decimation (anti-aliasing lowpass filtering + downsampling)
- Interpolation (upsampling zero-insertion + anti-imaging lowpass filtering)
- Rational Resampling (polyphase rate conversion Fs_out / Fs_in = L / M)
- Naive downsampling (x[::M]) & raw zero-insertion (x_up[n*L] = x[n]) for educational comparison
- Analytical decimation aliasing & interpolation spectral image analysis
"""

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, List, Optional, Tuple, Union
import numpy as np
import scipy.signal

from src.dsp.sampling import calculate_alias_frequency
from src.dsp.signals import SignalContainer


@dataclass(frozen=True)
class ResamplingResult:
    """Immutable data container holding results of a sampling-rate conversion operation.

    Attributes:
        input_sampling_rate_hz: Original sampling frequency Fs_in in Hz.
        output_sampling_rate_hz: Converted sampling frequency Fs_out in Hz.
        input_num_samples: Number of samples in original input signal.
        output_num_samples: Number of samples in resampled output signal.
        up_factor: Upsampling/interpolation factor L >= 1.
        down_factor: Downsampling/decimation factor M >= 1.
        original_signal: Input SignalContainer or 1D array.
        processed_signal: Resampled SignalContainer or 1D array.
        effective_duration_sec: Duration of processed signal in seconds.
        method: Name of resampling method used ('decimation', 'interpolation', 'rational_polyphase').
        metadata: Key-value dictionary of resampling configuration & metrics.
    """
    input_sampling_rate_hz: float
    output_sampling_rate_hz: float
    input_num_samples: int
    output_num_samples: int
    up_factor: int
    down_factor: int
    original_signal: Union[SignalContainer, np.ndarray]
    processed_signal: Union[SignalContainer, np.ndarray]
    effective_duration_sec: float
    method: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DecimationAliasingAnalysis:
    """Immutable container holding theoretical aliasing analysis for decimation by M.

    Attributes:
        input_frequencies_hz: Original signal component frequencies in Hz.
        input_sampling_rate_hz: Original sampling rate Fs_in in Hz.
        decimation_factor_M: Decimation factor M >= 1.
        output_sampling_rate_hz: New sampling rate Fs_out = Fs_in / M in Hz.
        output_nyquist_hz: New Nyquist limit Fs_out / 2 in Hz.
        aliasing_components_hz: Frequencies above Fs_out / 2 that will alias if un-filtered.
        theoretical_aliased_frequencies_hz: Predicted alias frequencies in [0, Fs_out / 2].
        surviving_passband_frequencies_hz: Frequencies at or below Fs_out / 2.
        metadata: Additional metadata dictionary.
    """
    input_frequencies_hz: List[float]
    input_sampling_rate_hz: float
    decimation_factor_M: int
    output_sampling_rate_hz: float
    output_nyquist_hz: float
    aliasing_components_hz: List[float]
    theoretical_aliased_frequencies_hz: List[float]
    surviving_passband_frequencies_hz: List[float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InterpolationImageAnalysis:
    """Immutable container holding theoretical spectral image analysis for interpolation by L.

    Attributes:
        signal_frequencies_hz: Fundamental frequencies present in input signal in Hz.
        input_sampling_rate_hz: Original sampling rate Fs_in in Hz.
        interpolation_factor_L: Interpolation factor L >= 1.
        output_sampling_rate_hz: New sampling rate Fs_out = Fs_in * L in Hz.
        output_nyquist_hz: New Nyquist limit Fs_out / 2 in Hz.
        replicated_image_frequencies_hz: Spectral image frequencies k*Fs_in +/- f within [0, Fs_out/2].
        metadata: Additional metadata dictionary.
    """
    signal_frequencies_hz: List[float]
    input_sampling_rate_hz: float
    interpolation_factor_L: int
    output_sampling_rate_hz: float
    output_nyquist_hz: float
    replicated_image_frequencies_hz: List[float]
    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# UTILITY & RATIO FUNCTIONS
# ============================================================================

def find_rational_resampling_ratio(
    fs_in_hz: float,
    fs_out_hz: float,
    max_denominator: int = 1000,
    max_error_percent: float = 1.0
) -> Tuple[int, int]:
    """Find exact or approximate integer rational ratio L/M = Fs_out / Fs_in.

    Args:
        fs_in_hz: Original sampling rate in Hz (> 0).
        fs_out_hz: Desired sampling rate in Hz (> 0).
        max_denominator: Maximum allowed denominator M for rational fraction.
        max_error_percent: Maximum allowed relative percentage error.

    Returns:
        Tuple of (interpolation factor L, decimation factor M) as positive integers.

    Raises:
        ValueError: If sampling rates <= 0 or ratio cannot be approximated within max_error_percent.
    """
    if fs_in_hz <= 0:
        raise ValueError(f"Input sampling rate Fs_in must be strictly positive (> 0), got {fs_in_hz}")
    if fs_out_hz <= 0:
        raise ValueError(f"Output sampling rate Fs_out must be strictly positive (> 0), got {fs_out_hz}")

    target_ratio = float(fs_out_hz) / float(fs_in_hz)
    frac = Fraction(target_ratio).limit_denominator(max_denominator)
    L, M = frac.numerator, frac.denominator

    approx_ratio = L / float(M)
    rel_error = abs(approx_ratio - target_ratio) / target_ratio * 100.0

    if rel_error > max_error_percent:
        raise ValueError(
            f"Cannot represent sampling rate ratio Fs_out/Fs_in ({fs_out_hz}/{fs_in_hz} = {target_ratio:.6f}) "
            f"with max_denominator={max_denominator}. Best fraction found: {L}/{M} ({approx_ratio:.6f}) "
            f"has error {rel_error:.2f}% which exceeds tolerance {max_error_percent:.2f}%."
        )

    return L, M


def _unpack_signal(
    signal_or_array: Union[SignalContainer, np.ndarray],
    expected_fs: Optional[float] = None
) -> Tuple[np.ndarray, float, bool]:
    """Extract 1D float64 array and sampling frequency from input signal or array.

    Args:
        signal_or_array: Input SignalContainer or 1D array.
        expected_fs: Optional default sampling rate if array is passed.

    Returns:
        Tuple of (1D float64 numpy array x, sampling rate Fs in Hz, is_container boolean).

    Raises:
        ValueError: If array is empty, non-finite, or sampling rate <= 0.
    """
    if isinstance(signal_or_array, SignalContainer):
        x = np.asarray(signal_or_array.amplitude, dtype=np.float64)
        fs = float(signal_or_array.sampling_rate_hz)
        is_container = True
    else:
        x = np.asarray(signal_or_array, dtype=np.float64)
        fs = float(expected_fs) if expected_fs is not None else 1.0
        is_container = False

    if len(x) < 2:
        raise ValueError(f"Input signal must contain at least 2 samples, got {len(x)}")

    if not np.all(np.isfinite(x)):
        raise ValueError("Input signal contains non-finite NaN or Inf values.")

    if fs <= 0:
        raise ValueError(f"Sampling frequency must be strictly positive (> 0), got {fs}")

    return x, fs, is_container


def _wrap_resampled_output(
    y: np.ndarray,
    fs_out_hz: float,
    original_input: Union[SignalContainer, np.ndarray],
    is_container: bool,
    method_desc: str
) -> Union[SignalContainer, np.ndarray]:
    """Wrap output numpy array back into SignalContainer if original input was a SignalContainer.

    Time Vector Endpoint Convention:
        For N samples at rate Fs_out, time vector is t[n] = n / Fs_out for n = 0..N-1 (endpoint excluded).

    Args:
        y: Resampled 1D float64 numpy array.
        fs_out_hz: New sampling rate Fs_out in Hz.
        original_input: Original input SignalContainer or np.ndarray.
        is_container: True if input was a SignalContainer.
        method_desc: Description string for metadata.

    Returns:
        New SignalContainer or 1D float64 numpy array matching input type.
    """
    if is_container and isinstance(original_input, SignalContainer):
        n_out = len(y)
        duration_sec = n_out / float(fs_out_hz)
        time_vector = np.arange(n_out, dtype=np.float64) / float(fs_out_hz)
        return SignalContainer(
            time_vector=time_vector,
            amplitude=y,
            sampling_rate_hz=fs_out_hz,
            duration_sec=duration_sec,
            signal_type=original_input.signal_type,
            metadata={
                **original_input.metadata,
                "resampling_method": method_desc,
                "original_fs_hz": original_input.sampling_rate_hz,
            },
        )
    return y


# ============================================================================
# NAIVE DOWN/UPSAMPLING (EDUCATIONAL COMPARISON)
# ============================================================================

def downsample_naive(
    signal_or_array: Union[SignalContainer, np.ndarray],
    factor_M: int,
    sampling_rate_hz: Optional[float] = None
) -> Union[SignalContainer, np.ndarray]:
    """Perform naive downsampling by factor M without anti-aliasing filtering (x[::M]).

    Warning:
        This operation retains every M-th sample without lowpass filtering. Frequency components
        above Fs_out / 2 will fold over and alias into the output spectrum!

    Args:
        signal_or_array: Input SignalContainer or 1D array.
        factor_M: Downsampling factor M (must be integer >= 1).
        sampling_rate_hz: Sampling frequency in Hz if array is passed.

    Returns:
        Downsampled SignalContainer or 1D float64 numpy array.

    Raises:
        ValueError/TypeError: If factor_M is invalid or input signal is non-finite.
    """
    if not isinstance(factor_M, (int, np.integer)):
        raise TypeError(f"Decimation factor M must be an integer, got {type(factor_M).__name__}")
    if factor_M < 1:
        raise ValueError(f"Decimation factor M must be at least 1, got {factor_M}")

    x, fs_in, is_container = _unpack_signal(signal_or_array, expected_fs=sampling_rate_hz)
    y = x[::factor_M].copy()
    fs_out = fs_in / float(factor_M)

    return _wrap_resampled_output(
        y=y,
        fs_out_hz=fs_out,
        original_input=signal_or_array,
        is_container=is_container,
        method_desc=f"naive_downsample_M{factor_M}"
    )


def upsample_zero_insertion(
    signal_or_array: Union[SignalContainer, np.ndarray],
    factor_L: int,
    sampling_rate_hz: Optional[float] = None
) -> Union[SignalContainer, np.ndarray]:
    """Perform raw zero-insertion upsampling by factor L (x_up[n*L] = x[n], others 0).

    Warning:
        Zero insertion expands the sampling rate but introduces un-filtered spectral images
        at multiples of the original sampling rate k * Fs_in. Anti-imaging filtering is required
        to complete true interpolation.

    Args:
        signal_or_array: Input SignalContainer or 1D array.
        factor_L: Interpolation factor L (must be integer >= 1).
        sampling_rate_hz: Sampling frequency in Hz if array is passed.

    Returns:
        Zero-inserted SignalContainer or 1D float64 numpy array.

    Raises:
        ValueError/TypeError: If factor_L is invalid or input signal is non-finite.
    """
    if not isinstance(factor_L, (int, np.integer)):
        raise TypeError(f"Interpolation factor L must be an integer, got {type(factor_L).__name__}")
    if factor_L < 1:
        raise ValueError(f"Interpolation factor L must be at least 1, got {factor_L}")

    x, fs_in, is_container = _unpack_signal(signal_or_array, expected_fs=sampling_rate_hz)
    y = np.zeros(len(x) * factor_L, dtype=np.float64)
    y[::factor_L] = x
    fs_out = fs_in * float(factor_L)

    return _wrap_resampled_output(
        y=y,
        fs_out_hz=fs_out,
        original_input=signal_or_array,
        is_container=is_container,
        method_desc=f"zero_insertion_L{factor_L}"
    )


# ============================================================================
# SCIENTIFIC MULTIRATE DSP OPERATIONS
# ============================================================================

def decimate_signal(
    signal_or_array: Union[SignalContainer, np.ndarray],
    factor_M: int,
    sampling_rate_hz: Optional[float] = None,
    method: str = "iir",
    filter_order: Optional[int] = None,
    zero_phase: bool = True
) -> ResamplingResult:
    """Decimate a signal by integer factor M with appropriate anti-aliasing filtering.

    Decimation Pipeline:
        1. Lowpass anti-aliasing filter to suppress frequencies above Fs_out / 2 = Fs_in / (2*M).
        2. Downsample filtered signal by factor M.

    Args:
        signal_or_array: Input SignalContainer or 1D numpy array.
        factor_M: Decimation factor M >= 1.
        sampling_rate_hz: Original sampling frequency Fs in Hz if array is passed.
        method: Anti-aliasing filter method ('iir' Chebyshev/Butterworth, 'fir', or 'polyphase').
        filter_order: Filter order (None uses SciPy defaults).
        zero_phase: If True, applies zero-phase filtering (`filtfilt`/`sosfiltfilt`).

    Returns:
        ResamplingResult immutable container holding resampled signal and metadata.

    Raises:
        ValueError/TypeError: If parameters are invalid or signal is non-finite.
    """
    if not isinstance(factor_M, (int, np.integer)):
        raise TypeError(f"Decimation factor M must be an integer, got {type(factor_M).__name__}")
    if factor_M < 1:
        raise ValueError(f"Decimation factor M must be at least 1, got {factor_M}")

    x, fs_in, is_container = _unpack_signal(signal_or_array, expected_fs=sampling_rate_hz)
    fs_out = fs_in / float(factor_M)

    if factor_M == 1:
        processed = signal_or_array
        y_array = x
    else:
        m_lower = method.lower().strip()
        if m_lower in ["polyphase", "poly"]:
            y_array = scipy.signal.resample_poly(x, up=1, down=factor_M)
        elif m_lower in ["iir", "fir"]:
            # Use scipy.signal.decimate
            y_array = scipy.signal.decimate(
                x,
                q=factor_M,
                n=filter_order,
                ftype=m_lower,
                zero_phase=zero_phase
            )
        else:
            raise ValueError(f"Unsupported decimation method '{method}'. Supported: iir, fir, polyphase.")

        processed = _wrap_resampled_output(
            y=y_array,
            fs_out_hz=fs_out,
            original_input=signal_or_array,
            is_container=is_container,
            method_desc=f"decimate_M{factor_M}_{m_lower}"
        )

    n_in = len(x)
    n_out = len(y_array)
    duration_sec = n_out / float(fs_out)

    return ResamplingResult(
        input_sampling_rate_hz=fs_in,
        output_sampling_rate_hz=fs_out,
        input_num_samples=n_in,
        output_num_samples=n_out,
        up_factor=1,
        down_factor=factor_M,
        original_signal=signal_or_array,
        processed_signal=processed,
        effective_duration_sec=duration_sec,
        method=f"decimation_{method}",
        metadata={
            "anti_aliasing_filter_cutoff_hz": fs_out / 2.0,
            "zero_phase": zero_phase,
            "filter_type": method,
        },
    )


def interpolate_signal(
    signal_or_array: Union[SignalContainer, np.ndarray],
    factor_L: int,
    sampling_rate_hz: Optional[float] = None
) -> ResamplingResult:
    """Interpolate a signal by integer factor L using polyphase upsampling & anti-imaging filtering.

    Interpolation Pipeline:
        1. Insert L-1 zeros between input samples.
        2. Lowpass anti-imaging filter (cutoff at Fs_in / 2) to eliminate spectral images.
        3. Scale by factor L to maintain correct peak physical signal amplitude.

    Args:
        signal_or_array: Input SignalContainer or 1D numpy array.
        factor_L: Interpolation factor L >= 1.
        sampling_rate_hz: Original sampling frequency Fs in Hz if array is passed.

    Returns:
        ResamplingResult immutable container holding resampled signal and metadata.

    Raises:
        ValueError/TypeError: If factor_L is invalid or signal is non-finite.
    """
    if not isinstance(factor_L, (int, np.integer)):
        raise TypeError(f"Interpolation factor L must be an integer, got {type(factor_L).__name__}")
    if factor_L < 1:
        raise ValueError(f"Interpolation factor L must be at least 1, got {factor_L}")

    x, fs_in, is_container = _unpack_signal(signal_or_array, expected_fs=sampling_rate_hz)
    fs_out = fs_in * float(factor_L)

    if factor_L == 1:
        processed = signal_or_array
        y_array = x
    else:
        # scipy.signal.resample_poly handles zero-insertion, anti-imaging FIR filtering, and gain scaling L
        y_array = scipy.signal.resample_poly(x, up=factor_L, down=1)
        processed = _wrap_resampled_output(
            y=y_array,
            fs_out_hz=fs_out,
            original_input=signal_or_array,
            is_container=is_container,
            method_desc=f"interpolate_L{factor_L}"
        )

    n_in = len(x)
    n_out = len(y_array)
    duration_sec = n_out / float(fs_out)

    return ResamplingResult(
        input_sampling_rate_hz=fs_in,
        output_sampling_rate_hz=fs_out,
        input_num_samples=n_in,
        output_num_samples=n_out,
        up_factor=factor_L,
        down_factor=1,
        original_signal=signal_or_array,
        processed_signal=processed,
        effective_duration_sec=duration_sec,
        method="interpolation_polyphase",
        metadata={
            "anti_imaging_filter_cutoff_hz": fs_in / 2.0,
            "gain_scaling_factor": float(factor_L),
        },
    )


def resample_rational(
    signal_or_array: Union[SignalContainer, np.ndarray],
    target_sampling_rate_hz: float,
    sampling_rate_hz: Optional[float] = None,
    max_denominator: int = 1000,
    max_error_percent: float = 1.0
) -> ResamplingResult:
    """Resample a signal to a target sampling rate using rational ratio polyphase resampling (L / M).

    Args:
        signal_or_array: Input SignalContainer or 1D numpy array.
        target_sampling_rate_hz: Desired output sampling frequency Fs_out in Hz (> 0).
        sampling_rate_hz: Original sampling frequency Fs_in in Hz if array is passed.
        max_denominator: Maximum denominator M allowed for rational approximation.
        max_error_percent: Maximum allowed percentage error in sampling rate ratio.

    Returns:
        ResamplingResult immutable container.

    Raises:
        ValueError: If sampling rates <= 0 or ratio cannot be approximated cleanly.
    """
    x, fs_in, is_container = _unpack_signal(signal_or_array, expected_fs=sampling_rate_hz)
    fs_out = float(target_sampling_rate_hz)

    L, M = find_rational_resampling_ratio(
        fs_in_hz=fs_in,
        fs_out_hz=fs_out,
        max_denominator=max_denominator,
        max_error_percent=max_error_percent
    )

    if L == 1 and M == 1:
        processed = signal_or_array
        y_array = x
    else:
        y_array = scipy.signal.resample_poly(x, up=L, down=M)
        processed = _wrap_resampled_output(
            y=y_array,
            fs_out_hz=fs_out,
            original_input=signal_or_array,
            is_container=is_container,
            method_desc=f"rational_polyphase_L{L}_M{M}"
        )

    n_in = len(x)
    n_out = len(y_array)
    duration_sec = n_out / float(fs_out)

    return ResamplingResult(
        input_sampling_rate_hz=fs_in,
        output_sampling_rate_hz=fs_out,
        input_num_samples=n_in,
        output_num_samples=n_out,
        up_factor=L,
        down_factor=M,
        original_signal=signal_or_array,
        processed_signal=processed,
        effective_duration_sec=duration_sec,
        method="rational_polyphase",
        metadata={
            "rational_up_factor_L": L,
            "rational_down_factor_M": M,
            "effective_ratio": L / float(M),
        },
    )


# ============================================================================
# EDUCATIONAL ANALYSIS TOOLS
# ============================================================================

def analyze_decimation_aliasing(
    signal_frequencies_hz: List[float],
    fs_in_hz: float,
    decimation_factor_M: int
) -> DecimationAliasingAnalysis:
    """Perform educational aliasing analysis for decimation by factor M.

    Reuses canonical calculate_alias_frequency from src.dsp.sampling.

    Args:
        signal_frequencies_hz: List of tone frequencies present in signal in Hz.
        fs_in_hz: Original sampling frequency Fs_in in Hz (> 0).
        decimation_factor_M: Integer decimation factor M >= 1.

    Returns:
        DecimationAliasingAnalysis dataclass payload.
    """
    if fs_in_hz <= 0:
        raise ValueError(f"Input sampling frequency Fs_in must be strictly positive, got {fs_in_hz}")
    if decimation_factor_M < 1:
        raise ValueError(f"Decimation factor M must be at least 1, got {decimation_factor_M}")

    fs_out = fs_in_hz / float(decimation_factor_M)
    nyquist_out = fs_out / 2.0

    aliased_components = []
    theoretical_alias_freqs = []
    surviving_passband = []

    for f in signal_frequencies_hz:
        if f < 0:
            raise ValueError(f"Signal frequency must be non-negative (>= 0), got {f}")
        if f > nyquist_out:
            aliased_components.append(float(f))
            f_alias = calculate_alias_frequency(f, fs_out)
            theoretical_alias_freqs.append(f_alias)
        else:
            surviving_passband.append(float(f))

    return DecimationAliasingAnalysis(
        input_frequencies_hz=[float(f) for f in signal_frequencies_hz],
        input_sampling_rate_hz=fs_in_hz,
        decimation_factor_M=decimation_factor_M,
        output_sampling_rate_hz=fs_out,
        output_nyquist_hz=nyquist_out,
        aliasing_components_hz=aliased_components,
        theoretical_aliased_frequencies_hz=theoretical_alias_freqs,
        surviving_passband_frequencies_hz=surviving_passband,
    )


def analyze_interpolation_images(
    signal_frequencies_hz: List[float],
    fs_in_hz: float,
    interpolation_factor_L: int
) -> InterpolationImageAnalysis:
    """Perform educational spectral image analysis for interpolation by factor L.

    Identifies replicated spectral images at frequencies k * Fs_in +/- f within [0, Fs_out / 2].

    Args:
        signal_frequencies_hz: List of signal fundamental frequencies in Hz.
        fs_in_hz: Original sampling frequency Fs_in in Hz (> 0).
        interpolation_factor_L: Integer interpolation factor L >= 1.

    Returns:
        InterpolationImageAnalysis dataclass payload.
    """
    if fs_in_hz <= 0:
        raise ValueError(f"Input sampling frequency Fs_in must be strictly positive, got {fs_in_hz}")
    if interpolation_factor_L < 1:
        raise ValueError(f"Interpolation factor L must be at least 1, got {interpolation_factor_L}")

    fs_out = fs_in_hz * float(interpolation_factor_L)
    nyquist_out = fs_out / 2.0

    images = set()

    for f in signal_frequencies_hz:
        if f < 0:
            raise ValueError(f"Signal frequency must be non-negative, got {f}")
        # Replicated images around harmonics k * Fs_in for k = 1..L
        for k in range(1, interpolation_factor_L + 1):
            f_img1 = k * fs_in_hz - f
            f_img2 = k * fs_in_hz + f
            if 0.0 <= f_img1 <= nyquist_out and not np.isclose(f_img1, f, atol=1e-5):
                images.add(round(f_img1, 6))
            if 0.0 <= f_img2 <= nyquist_out and not np.isclose(f_img2, f, atol=1e-5):
                images.add(round(f_img2, 6))

    sorted_images = sorted(list(images))

    return InterpolationImageAnalysis(
        signal_frequencies_hz=[float(f) for f in signal_frequencies_hz],
        input_sampling_rate_hz=fs_in_hz,
        interpolation_factor_L=interpolation_factor_L,
        output_sampling_rate_hz=fs_out,
        output_nyquist_hz=nyquist_out,
        replicated_image_frequencies_hz=sorted_images,
    )
