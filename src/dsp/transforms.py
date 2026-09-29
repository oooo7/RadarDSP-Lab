"""FFT, Spectrum Analysis, Windowing, and STFT Engine.

Provides reusable, numerically accurate frequency-domain processing including:
- FFT, rFFT, iFFT, FFT shifting
- Physically normalized magnitude and power spectra (one-sided & two-sided)
- Window functions (Rectangular, Hann, Hamming, Blackman, Kaiser) with coherent gain & ENBW
- Parabolic sub-bin dominant frequency estimation
- Zero-padding & resolution analysis (distinguishing physical resolution from bin density)
- Multi-window spectral leakage analysis
- Short-Time Fourier Transform (STFT) / Spectrogram engine
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import scipy.signal

from src.dsp.signals import SignalContainer


@dataclass(frozen=True)
class WindowMetrics:
    """Metadata and normalization metrics for a tapering window function."""
    name: str
    coherent_gain: float
    enbw_bins: float
    sum_w: float
    sum_w2: float


@dataclass(frozen=True)
class SpectrumResult:
    """Immutable data payload holding full spectral transform results."""
    frequency_hz: np.ndarray
    magnitude: np.ndarray
    power: np.ndarray
    phase_rad: np.ndarray
    complex_spectrum: np.ndarray
    sampling_frequency_hz: float
    n_signal_samples: int
    n_fft_samples: int
    physical_resolution_hz: float
    bin_resolution_hz: float
    window_name: str
    coherent_gain: float
    enbw_bins: float
    is_one_sided: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    # Backward-compatible property aliases
    @property
    def frequencies_hz(self) -> np.ndarray:
        return self.frequency_hz

    @property
    def spectrum(self) -> np.ndarray:
        return self.complex_spectrum

    @property
    def magnitude_db(self) -> np.ndarray:
        return 20.0 * np.log10(np.maximum(self.magnitude, 1e-12))

    @property
    def frequency_resolution_hz(self) -> float:
        return self.physical_resolution_hz


# Alias for Phase 0 compatibility
SpectrumContainer = SpectrumResult


@dataclass(frozen=True)
class DominantFrequencyResult:
    """Result container for peak spectral frequency estimation."""
    frequency_hz: float
    magnitude: float
    interpolated_bin: float
    raw_bin_index: int
    phase_rad: float


@dataclass(frozen=True)
class SpectralLeakageResult:
    """Result container for multi-window spectral leakage analysis."""
    signal_frequency_hz: float
    sampling_rate_hz: float
    n_samples: int
    is_bin_centered: bool
    window_spectra: Dict[str, SpectrumResult]
    peak_magnitudes: Dict[str, float]
    sidelobe_suppression_db: Dict[str, float]


@dataclass(frozen=True)
class SpectrogramResult:
    """Result container for Short-Time Fourier Transform (STFT) spectrogram."""
    time_vector: np.ndarray
    frequency_vector: np.ndarray
    stft_matrix: np.ndarray
    magnitude_matrix: np.ndarray
    power_db_matrix: np.ndarray
    sampling_frequency_hz: float
    n_per_seg: int
    n_overlap: int
    window_name: str

    @property
    def times_sec(self) -> np.ndarray:
        return self.time_vector

    @property
    def frequencies_hz(self) -> np.ndarray:
        return self.frequency_vector

    @property
    def spectrogram_db(self) -> np.ndarray:
        return self.power_db_matrix


# Alias for Phase 0 compatibility
SpectrogramContainer = SpectrogramResult


# ============================================================================
# WINDOW FUNCTIONS & METRICS
# ============================================================================

def get_window(
    window_name: str,
    n_samples: int,
    kaiser_beta: float = 8.6
) -> Tuple[np.ndarray, WindowMetrics]:
    """Generate normalized window coefficients and calculate coherent gain and ENBW metrics.

    Coherent Gain (C_gain):
        C_gain = (1 / N) * sum(w[n])

    Equivalent Noise Bandwidth (ENBW in bins):
        ENBW = N * sum(w[n]^2) / (sum(w[n]))^2

    Args:
        window_name: Tapering window name ('rect', 'rectangular', 'hann', 'hanning', 'hamming', 'blackman', 'kaiser').
        n_samples: Number of samples N (must be >= 1).
        kaiser_beta: Shape parameter beta for Kaiser window (default 8.6).

    Returns:
        Tuple of (1D float64 window array, WindowMetrics container).

    Raises:
        ValueError: If n_samples < 1 or unknown window_name.
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be at least 1, got {n_samples}")

    name = window_name.lower().strip()

    if name in ["rect", "rectangular", "boxcar"]:
        w = np.ones(n_samples, dtype=np.float64)
        canonical_name = "rectangular"
    elif name in ["hann", "hanning"]:
        w = scipy.signal.windows.hann(n_samples, sym=False)
        canonical_name = "hann"
    elif name in ["hamming"]:
        w = scipy.signal.windows.hamming(n_samples, sym=False)
        canonical_name = "hamming"
    elif name in ["blackman"]:
        w = scipy.signal.windows.blackman(n_samples, sym=False)
        canonical_name = "blackman"
    elif name in ["kaiser"]:
        w = scipy.signal.windows.kaiser(n_samples, beta=kaiser_beta, sym=False)
        canonical_name = f"kaiser(beta={kaiser_beta})"
    else:
        raise ValueError(
            f"Unsupported window_name '{window_name}'. Supported windows: "
            "rectangular (rect), hann, hamming, blackman, kaiser."
        )

    sum_w = float(np.sum(w))
    sum_w2 = float(np.sum(w ** 2))

    coherent_gain = sum_w / float(n_samples)
    enbw_bins = (float(n_samples) * sum_w2) / (sum_w ** 2) if sum_w > 0 else 1.0

    metrics = WindowMetrics(
        name=canonical_name,
        coherent_gain=coherent_gain,
        enbw_bins=enbw_bins,
        sum_w=sum_w,
        sum_w2=sum_w2,
    )
    return w, metrics


# ============================================================================
# FFT, IFFT, AND SPECTRUM COMPUTATIONS
# ============================================================================

def compute_dft_educational(x: np.ndarray) -> np.ndarray:
    """Educational O(N^2) Discrete Fourier Transform reference implementation for testing.

    X[k] = sum_{n=0}^{N-1} x[n] * exp(-j * 2 * pi * k * n / N)

    Args:
        x: Input 1D numerical array.

    Returns:
        1D complex128 array of DFT coefficients.
    """
    n = len(x)
    if n == 0:
        return np.array([], dtype=np.complex128)
    n_idx = np.arange(n)
    k_idx = n_idx.reshape((n, 1))
    e = np.exp(-2j * np.pi * k_idx * n_idx / n)
    return np.dot(e, x)


def compute_fft(
    signal_or_array: SignalContainer | np.ndarray,
    sampling_rate_hz: float = 1.0,
    window_type: str = "rect",
    n_fft: Optional[int] = None,
    one_sided: bool = True,
    kaiser_beta: float = 8.6
) -> SpectrumResult:
    """Compute normalized Discrete Fourier Transform (FFT) with windowing and zero-padding.

    Magnitude Normalization:
        - For a real sinusoid x[n] = A * cos(2*pi*f*t + phase), the one-sided magnitude
          spectrum peak correctly recovers amplitude A at the fundamental frequency bin.
        - DC bin (bin 0) and Nyquist bin (bin N/2 for even N) are scaled by 1 / sum(w[n]).
        - Interior positive frequency bins are scaled by 2 / sum(w[n]).

    Resolution Distinction:
        - Physical resolution: Delta f_phys = Fs / N_signal (determined by observation time T_obs).
        - Bin grid spacing: Delta f_bin = Fs / N_fft (density of displayed spectral points).

    Args:
        signal_or_array: Input SignalContainer or 1D float/complex numpy array.
        sampling_rate_hz: Sampling frequency Fs in Hz (> 0) if array is passed.
        window_type: Window name ('rect', 'hann', 'hamming', 'blackman', 'kaiser').
        n_fft: Number of FFT points (zero-padding if n_fft > N_signal).
        one_sided: If True, returns one-sided spectrum [0, Fs/2]; if False, returns full spectrum.
        kaiser_beta: Shape parameter for Kaiser window.

    Returns:
        SpectrumResult container holding frequencies, magnitude, power, phase, and metrics.

    Raises:
        ValueError: If inputs are invalid or non-finite.
    """
    if isinstance(signal_or_array, SignalContainer):
        x = np.asarray(signal_or_array.amplitude, dtype=np.float64)
        fs = float(signal_or_array.sampling_rate_hz)
    else:
        x = np.asarray(signal_or_array)
        fs = float(sampling_rate_hz)

    if fs <= 0:
        raise ValueError(f"Sampling frequency Fs must be strictly positive (> 0), got {fs}")

    if len(x) == 0:
        raise ValueError("Input signal array is empty.")

    if not np.all(np.isfinite(x)):
        raise ValueError("Input signal contains non-finite NaN or Inf values.")

    n_signal = len(x)
    n_points = n_fft if (n_fft is not None and n_fft > n_signal) else n_signal

    # Apply window
    w, win_metrics = get_window(window_type, n_signal, kaiser_beta=kaiser_beta)
    x_windowed = x * w

    # Complex FFT with zero-padding
    complex_fft = np.fft.fft(x_windowed, n=n_points)

    # Resolution calculations
    phys_res = fs / float(n_signal)
    bin_res = fs / float(n_points)

    is_real_input = not np.iscomplexobj(x)

    if one_sided and is_real_input:
        # One-sided spectrum for real inputs [0, Fs/2]
        num_bins = (n_points // 2) + 1
        complex_spec = complex_fft[:num_bins]
        freqs = np.fft.rfftfreq(n_points, d=1.0 / fs)

        # One-sided magnitude scaling
        sum_w = win_metrics.sum_w if win_metrics.sum_w > 0 else 1.0
        mag = np.abs(complex_spec) / sum_w

        # Multiply interior positive bins by 2
        if n_points % 2 == 0:
            # Even N_fft: bins 1 to num_bins - 2 are interior
            mag[1:-1] *= 2.0
        else:
            # Odd N_fft: bins 1 to num_bins - 1 are interior
            mag[1:] *= 2.0

        is_onesided_res = True
    else:
        # Two-sided spectrum [0, Fs)
        complex_spec = complex_fft
        freqs = np.fft.fftfreq(n_points, d=1.0 / fs)

        sum_w = win_metrics.sum_w if win_metrics.sum_w > 0 else 1.0
        mag = np.abs(complex_spec) / sum_w
        is_onesided_res = False

    power = mag ** 2
    phase = np.angle(complex_spec)

    return SpectrumResult(
        frequency_hz=freqs,
        magnitude=mag,
        power=power,
        phase_rad=phase,
        complex_spectrum=complex_spec,
        sampling_frequency_hz=fs,
        n_signal_samples=n_signal,
        n_fft_samples=n_points,
        physical_resolution_hz=phys_res,
        bin_resolution_hz=bin_res,
        window_name=win_metrics.name,
        coherent_gain=win_metrics.coherent_gain,
        enbw_bins=win_metrics.enbw_bins,
        is_one_sided=is_onesided_res,
        metadata={"is_real_input": is_real_input},
    )


def compute_ifft(complex_spectrum: np.ndarray, n_samples: Optional[int] = None) -> np.ndarray:
    """Compute Inverse Fast Fourier Transform (IFFT).

    Args:
        complex_spectrum: 1D complex FFT coefficient array.
        n_samples: Output length of time-domain signal.

    Returns:
        1D numpy array of reconstructed time-domain samples.
    """
    return np.fft.ifft(complex_spectrum, n=n_samples)


def compute_rfft(
    x: np.ndarray,
    sampling_rate_hz: float = 1.0,
    window_type: str = "rect",
    n_fft: Optional[int] = None
) -> SpectrumResult:
    """Convenience function computing one-sided Real FFT for real-valued signals."""
    return compute_fft(
        signal_or_array=x,
        sampling_rate_hz=sampling_rate_hz,
        window_type=window_type,
        n_fft=n_fft,
        one_sided=True
    )


def compute_fft_shift(spectrum_res: SpectrumResult) -> SpectrumResult:
    """Shift zero-frequency component to center of spectrum [-Fs/2, Fs/2).

    Args:
        spectrum_res: Input SpectrumResult container (must be two-sided).

    Returns:
        Shifted SpectrumResult container centered at 0 Hz.
    """
    shifted_freqs = np.fft.fftshift(spectrum_res.frequency_hz)
    shifted_complex = np.fft.fftshift(spectrum_res.complex_spectrum)
    shifted_mag = np.fft.fftshift(spectrum_res.magnitude)
    shifted_power = np.fft.fftshift(spectrum_res.power)
    shifted_phase = np.fft.fftshift(spectrum_res.phase_rad)

    return SpectrumResult(
        frequency_hz=shifted_freqs,
        magnitude=shifted_mag,
        power=shifted_power,
        phase_rad=shifted_phase,
        complex_spectrum=shifted_complex,
        sampling_frequency_hz=spectrum_res.sampling_frequency_hz,
        n_signal_samples=spectrum_res.n_signal_samples,
        n_fft_samples=spectrum_res.n_fft_samples,
        physical_resolution_hz=spectrum_res.physical_resolution_hz,
        bin_resolution_hz=spectrum_res.bin_resolution_hz,
        window_name=spectrum_res.window_name,
        coherent_gain=spectrum_res.coherent_gain,
        enbw_bins=spectrum_res.enbw_bins,
        is_one_sided=False,
        metadata={**spectrum_res.metadata, "is_shifted": True},
    )


# ============================================================================
# DOMINANT FREQUENCY & SUB-BIN ESTIMATION
# ============================================================================

def find_dominant_frequency(
    spectrum_res: SpectrumResult,
    ignore_dc: bool = True,
    interpolate_subbin: bool = True
) -> DominantFrequencyResult:
    """Identify the strongest spectral peak frequency with sub-bin parabolic estimation.

    Parabolic Interpolation Formula:
        Given peak bin k* with magnitudes alpha = |X[k*-1]|, beta = |X[k*]|, gamma = |X[k*+1]|:
        delta = 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma)
        f_est = (k* + delta) * Delta f_bin

    Args:
        spectrum_res: Input SpectrumResult container.
        ignore_dc: If True, excludes DC bin (bin 0) from peak search.
        interpolate_subbin: If True, applies 3-point parabolic interpolation.

    Returns:
        DominantFrequencyResult container with estimated peak frequency and magnitude.
    """
    mags = np.copy(spectrum_res.magnitude)
    if len(mags) == 0 or np.all(mags == 0):
        return DominantFrequencyResult(frequency_hz=0.0, magnitude=0.0, interpolated_bin=0.0, raw_bin_index=0, phase_rad=0.0)

    if ignore_dc and len(mags) > 1:
        mags[0] = 0.0

    raw_bin = int(np.argmax(mags))
    peak_mag = float(mags[raw_bin])
    phase_at_peak = float(spectrum_res.phase_rad[raw_bin])
    bin_res = spectrum_res.bin_resolution_hz

    # Check bounds for sub-bin interpolation
    if interpolate_subbin and 0 < raw_bin < len(mags) - 1:
        alpha = mags[raw_bin - 1]
        beta = mags[raw_bin]
        gamma = mags[raw_bin + 1]
        denom = alpha - 2.0 * beta + gamma

        if abs(denom) > 1e-12:
            delta = 0.5 * (alpha - gamma) / denom
            # Clamp delta to [-0.5, 0.5] to prevent instability on degenerate noise peaks
            delta = float(np.clip(delta, -0.5, 0.5))
        else:
            delta = 0.0
    else:
        delta = 0.0

    interp_bin = raw_bin + delta
    f_est = float(interp_bin * bin_res)

    return DominantFrequencyResult(
        frequency_hz=abs(f_est),
        magnitude=peak_mag,
        interpolated_bin=float(interp_bin),
        raw_bin_index=raw_bin,
        phase_rad=phase_at_peak,
    )


# ============================================================================
# SPECTRAL LEAKAGE ANALYSIS
# ============================================================================

def analyze_spectral_leakage(
    signal_frequency_hz: float,
    sampling_rate_hz: float,
    n_samples: int,
    windows: Optional[List[str]] = None
) -> SpectralLeakageResult:
    """Analyze spectral leakage across multiple window functions for a given tone.

    Compares bin-centered vs non-bin-centered tones across Rectangular, Hann, Hamming, and Blackman.

    Args:
        signal_frequency_hz: Test tone frequency f in Hz.
        sampling_rate_hz: Sampling rate Fs in Hz.
        n_samples: Number of samples N.
        windows: List of window names to analyze (default: rect, hann, hamming, blackman).

    Returns:
        SpectralLeakageResult container holding comparison spectra and leakage metrics.
    """
    if windows is None:
        windows = ["rectangular", "hann", "hamming", "blackman"]

    t = np.arange(n_samples, dtype=np.float64) / float(sampling_rate_hz)
    x = np.sin(2.0 * np.pi * signal_frequency_hz * t)

    bin_res = sampling_rate_hz / float(n_samples)
    is_bin_centered = bool(np.isclose((signal_frequency_hz / bin_res) % 1.0, 0.0, atol=1e-5))

    spectra = {}
    peak_mags = {}
    sidelobe_suppressions = {}

    for w_name in windows:
        spec = compute_fft(x, sampling_rate_hz=sampling_rate_hz, window_type=w_name, one_sided=True)
        spectra[w_name] = spec

        dom = find_dominant_frequency(spec, ignore_dc=True, interpolate_subbin=True)
        peak_mags[w_name] = dom.magnitude

        # Estimate sidelobe suppression (ratio of max off-peak bin to peak magnitude)
        sorted_mags = sorted(spec.magnitude, reverse=True)
        if len(sorted_mags) > 2 and sorted_mags[0] > 1e-12:
            sidelobe_ratio_db = float(20.0 * np.log10(sorted_mags[2] / sorted_mags[0]))
        else:
            sidelobe_ratio_db = -100.0
        sidelobe_suppressions[w_name] = sidelobe_ratio_db

    return SpectralLeakageResult(
        signal_frequency_hz=signal_frequency_hz,
        sampling_rate_hz=sampling_rate_hz,
        n_samples=n_samples,
        is_bin_centered=is_bin_centered,
        window_spectra=spectra,
        peak_magnitudes=peak_mags,
        sidelobe_suppression_db=sidelobe_suppressions,
    )


# ============================================================================
# STFT / SPECTROGRAM ENGINE
# ============================================================================

def compute_stft(
    signal_or_array: SignalContainer | np.ndarray,
    sampling_rate_hz: float = 1.0,
    nperseg: int = 256,
    noverlap: Optional[int] = None,
    nfft: Optional[int] = None,
    window_type: str = "hann"
) -> SpectrogramResult:
    """Compute Short-Time Fourier Transform (STFT) for time-frequency spectrogram analysis.

    Args:
        signal_or_array: Input SignalContainer or 1D numerical array.
        sampling_rate_hz: Sampling rate Fs in Hz (> 0) if array is passed.
        nperseg: Segment length in samples (default 256).
        noverlap: Number of overlapping samples between segments (default nperseg // 2).
        nfft: Number of FFT points per segment (default nperseg).
        window_type: Window name used per segment.

    Returns:
        SpectrogramResult container holding time vector, frequency vector, 2D complex STFT matrix,
        2D magnitude matrix, and 2D power spectrogram in dB.

    Raises:
        ValueError: If parameters are invalid.
    """
    if isinstance(signal_or_array, SignalContainer):
        x = np.asarray(signal_or_array.amplitude, dtype=np.float64)
        fs = float(signal_or_array.sampling_rate_hz)
    else:
        x = np.asarray(signal_or_array)
        fs = float(sampling_rate_hz)

    if fs <= 0:
        raise ValueError(f"Sampling frequency Fs must be strictly positive (> 0), got {fs}")

    if len(x) == 0:
        raise ValueError("Input signal array for STFT is empty.")

    if nperseg < 4:
        raise ValueError(f"nperseg must be at least 4, got {nperseg}")

    if noverlap is None:
        noverlap = nperseg // 2

    if noverlap >= nperseg:
        raise ValueError(f"noverlap ({noverlap}) must be strictly smaller than nperseg ({nperseg})")

    # Generate segment window
    w, win_metrics = get_window(window_type, nperseg)

    # Compute STFT using scipy.signal.stft
    freqs, times, stft_mat = scipy.signal.stft(
        x,
        fs=fs,
        window=w,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        boundary=None,
        padded=False
    )

    # Magnitude spectrogram
    sum_w = win_metrics.sum_w if win_metrics.sum_w > 0 else 1.0
    mag_mat = np.abs(stft_mat) / sum_w
    # Scale interior positive bins by 2 for real signals
    mag_mat[1:-1, :] *= 2.0

    # Power spectrogram in dB
    power_mat = mag_mat ** 2
    max_power = np.max(power_mat) if np.max(power_mat) > 0 else 1.0
    power_db_mat = 10.0 * np.log10(np.maximum(power_mat / max_power, 1e-12))

    return SpectrogramResult(
        time_vector=times,
        frequency_vector=freqs,
        stft_matrix=stft_mat,
        magnitude_matrix=mag_mat,
        power_db_matrix=power_db_mat,
        sampling_frequency_hz=fs,
        n_per_seg=nperseg,
        n_overlap=noverlap,
        window_name=win_metrics.name,
    )
