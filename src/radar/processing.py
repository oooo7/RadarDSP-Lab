"""FMCW Radar Range Processing Engine.

Provides de-chirping, 1D Range FFT computation, range-axis mapping,
window selection, zero-padding interpolation, maximum unambiguous range calculation,
beat signal sampling Nyquist validation, and quantitative range estimation.
"""

from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np
import scipy.constants

from src.dsp.transforms import (
    DominantFrequencyResult,
    SpectrumResult,
    compute_fft,
    find_dominant_frequency,
)
from src.radar.chirp import FMCWChirpContainer
from src.radar.propagation import RadarRxPayload
from src.utils.config import RadarConfig

# Speed of light in m/s
SPEED_OF_LIGHT_M_PER_S = float(scipy.constants.c)


@dataclass(frozen=True)
class BeatSamplingValidation:
    """Immutable payload validating ADC sampling rate Fs against the dechirped beat frequency bandwidth.

    Attributes:
        maximum_expected_range_m: Maximum target processing range R_max in meters.
        maximum_beat_frequency_hz: Maximum expected beat frequency f_b_max = 2*S*R_max/c in Hz.
        nyquist_frequency_hz: Nyquist frequency Fs / 2 in Hz.
        minimum_required_beat_sampling_rate_hz: Minimum required sampling rate 2*f_b_max in Hz.
        configured_sampling_rate_hz: Actual ADC sampling rate Fs in Hz.
        sampling_margin_ratio: Ratio Fs / (2*f_b_max).
        sampling_margin_hz: Difference Fs - 2*f_b_max in Hz.
        is_valid: True if Fs > minimum_required_beat_sampling_rate_hz.
    """
    maximum_expected_range_m: float
    maximum_beat_frequency_hz: float
    nyquist_frequency_hz: float
    minimum_required_beat_sampling_rate_hz: float
    configured_sampling_rate_hz: float
    sampling_margin_ratio: float
    sampling_margin_hz: float
    is_valid: bool


@dataclass(frozen=True)
class RangeEstimateResult:
    """Immutable payload holding full FMCW radar range processing and estimation results.

    Attributes:
        estimated_range_m: Estimated target range R_est in meters.
        beat_frequency_hz: Measured beat frequency f_b in Hz.
        peak_magnitude: Peak spectral magnitude in Volts.
        frequency_error_hz: Absolute beat frequency error |f_meas - f_theo| in Hz.
        range_error_m: Absolute range estimation error |R_est - R_true| in meters.
        relative_range_error: Relative range error |R_est - R_true| / R_true.
        theoretical_range_m: True target range R_true in meters.
        theoretical_beat_frequency_hz: Predicted theoretical beat frequency f_b = S * tau in Hz.
        range_resolution_m: Theoretical radar range resolution Delta R = c / (2*B) in meters.
        max_unambiguous_range_m: Maximum processing range R_max in meters.
        fft_bin_spacing_hz: Frequency bin spacing Delta f_bin = Fs / N_fft in Hz.
        range_bin_spacing_m: Range bin spacing Delta R_bin = c * Delta f_bin / (2*S) in meters.
        range_axis_m: 1D float64 array of range values in meters corresponding to spectrum.
        beat_freq_axis_hz: 1D float64 array of positive frequency values in Hz.
        spectrum_magnitude: 1D float64 array of magnitude spectrum |X[k]|.
        spectrum_db: 1D float64 array of magnitude spectrum in dB.
        beat_sampling_validation: BeatSamplingValidation container.
        metadata: Key-value dictionary of processing options.
    """
    estimated_range_m: float
    beat_frequency_hz: float
    peak_magnitude: float
    frequency_error_hz: float
    range_error_m: float
    relative_range_error: float
    theoretical_range_m: float
    theoretical_beat_frequency_hz: float
    range_resolution_m: float
    max_unambiguous_range_m: float
    fft_bin_spacing_hz: float
    range_bin_spacing_m: float
    range_axis_m: np.ndarray
    beat_freq_axis_hz: np.ndarray
    spectrum_magnitude: np.ndarray
    spectrum_db: np.ndarray
    beat_sampling_validation: Optional[BeatSamplingValidation] = None
    metadata: dict[str, Any] = field(default_factory=dict)


def validate_beat_sampling_rate(
    sampling_rate_hz: float,
    chirp_slope_hz_per_sec: float,
    max_expected_range_m: float = 500.0,
    is_complex_beat: bool = False
) -> BeatSamplingValidation:
    """Validate ADC sampling rate Fs against the dechirped beat frequency bandwidth.

    Sampling Architecture (Option B):
        The ADC digitizes the analog dechirped beat signal s_beat(t) at sampling rate Fs.
        Maximum beat frequency for range R_max is: f_b_max = 2 * S * R_max / c.
        For one-sided FFT range processing, the beat frequency must fit strictly inside [0, Fs/2],
        requiring minimum Fs > 2 * f_b_max.

    Args:
        sampling_rate_hz: ADC sampling rate Fs in Hz (> 0).
        chirp_slope_hz_per_sec: Linear chirp slope S = B / T_c in Hz/s (> 0).
        max_expected_range_m: Maximum processing target range R_max in meters.
        is_complex_beat: Reserved parameter for complex spectrum range processing.

    Returns:
        BeatSamplingValidation container.

    Raises:
        ValueError: If sampling rate violates beat signal Nyquist criterion (Fs <= 2*f_b_max).
    """
    fs = float(sampling_rate_hz)
    slope = float(chirp_slope_hz_per_sec)
    r_max = float(max_expected_range_m)

    if fs <= 0:
        raise ValueError(f"Sampling rate Fs must be strictly positive (> 0), got {fs} Hz")
    if slope <= 0:
        raise ValueError(f"Chirp slope S must be strictly positive (> 0), got {slope} Hz/s")
    if r_max <= 0:
        raise ValueError(f"Maximum expected range R_max must be strictly positive (> 0), got {r_max} m")

    fb_max = 2.0 * slope * r_max / SPEED_OF_LIGHT_M_PER_S
    min_fs_req = 2.0 * fb_max
    nyquist_freq = fs / 2.0
    margin_ratio = fs / min_fs_req if min_fs_req > 0 else 1.0
    margin_hz = fs - min_fs_req
    is_satisfied = fs > min_fs_req

    if not is_satisfied:
        raise ValueError(
            f"ADC sampling rate Fs={fs/1e6:.2f} MHz is insufficient for beat frequency bandwidth! "
            f"For R_max={r_max:.1f} m, max beat frequency f_b_max={fb_max/1e6:.4f} MHz requires "
            f"minimum Fs > {min_fs_req/1e6:.4f} MHz (Nyquist={min_fs_req/2e6:.4f} MHz), but Fs={fs/1e6:.2f} MHz was provided."
        )

    return BeatSamplingValidation(
        maximum_expected_range_m=r_max,
        maximum_beat_frequency_hz=fb_max,
        nyquist_frequency_hz=nyquist_freq,
        minimum_required_beat_sampling_rate_hz=min_fs_req,
        configured_sampling_rate_hz=fs,
        sampling_margin_ratio=margin_ratio,
        sampling_margin_hz=margin_hz,
        is_valid=is_satisfied,
    )


def dechirp_signal(
    tx_chirp: FMCWChirpContainer,
    rx_signal: np.ndarray
) -> np.ndarray:
    """Mix transmitted FMCW chirp with received echo to produce de-chirped beat signal.

    Mixing Equation:
        For complex baseband: s_beat(t) = s_tx(t) * conj(s_rx(t))
        For real signals:    s_beat(t) = s_tx(t) * s_rx(t)

    Args:
        tx_chirp: Transmitted FMCWChirpContainer.
        rx_signal: 1D numpy array of received echo signal.

    Returns:
        1D float64 or complex128 numpy array of de-chirped beat signal.

    Raises:
        ValueError: If array lengths mismatch or contain non-finite values.
    """
    tx = tx_chirp.tx_signal
    rx = np.asarray(rx_signal)

    if len(tx) != len(rx):
        raise ValueError(f"Tx length ({len(tx)}) and Rx length ({len(rx)}) must match.")
    if not np.all(np.isfinite(rx)):
        raise ValueError("Received signal contains non-finite NaN or Inf values.")

    if tx_chirp.is_complex_baseband:
        return tx * np.conj(rx)
    else:
        return tx * rx


def calculate_max_unambiguous_range(
    sampling_rate_hz: float,
    chirp_slope_hz_per_sec: float,
    is_complex: bool = True
) -> float:
    """Calculate maximum unambiguous radar range R_max based on sampling rate and chirp slope.

    Equation:
        For complex beat signal (analyzed over [0, Fs]): f_b_max = Fs
        For real beat signal (analyzed over [0, Fs/2]):  f_b_max = Fs / 2
        R_max = c * f_b_max / (2 * S)

    Args:
        sampling_rate_hz: ADC sampling rate Fs in Hz (> 0).
        chirp_slope_hz_per_sec: Linear chirp slope S = B / T_c in Hz/s (> 0).
        is_complex: If True, uses f_b_max = Fs; if False, uses Fs / 2.

    Returns:
        Maximum unambiguous range R_max in meters.

    Raises:
        ValueError: If sampling_rate_hz <= 0 or chirp_slope_hz_per_sec <= 0.
    """
    fs = float(sampling_rate_hz)
    slope = float(chirp_slope_hz_per_sec)

    if fs <= 0:
        raise ValueError(f"Sampling rate Fs must be strictly positive, got {fs} Hz")
    if slope <= 0:
        raise ValueError(f"Chirp slope S must be strictly positive, got {slope} Hz/s")

    fb_max = fs if is_complex else (fs / 2.0)
    return SPEED_OF_LIGHT_M_PER_S * fb_max / (2.0 * slope)


def compute_range_fft(
    beat_signal_or_payload: RadarRxPayload | np.ndarray,
    sampling_rate_hz: float,
    chirp_slope_hz_per_sec: float,
    window_name: str = "hann",
    n_fft: Optional[int] = None
) -> SpectrumResult:
    """Compute 1D Range FFT spectrum on de-chirped beat signal using canonical Phase 3 FFT engine.

    Args:
        beat_signal_or_payload: RadarRxPayload or 1D beat signal array.
        sampling_rate_hz: ADC sampling rate Fs in Hz.
        chirp_slope_hz_per_sec: Chirp slope S in Hz/s.
        window_name: Tapering window name ('rect', 'hann', 'hamming', 'blackman', 'kaiser').
        n_fft: Number of FFT points (zero-padding if n_fft > N_signal).

    Returns:
        SpectrumResult container holding frequency spectrum, magnitude, and metadata.
    """
    if isinstance(beat_signal_or_payload, RadarRxPayload):
        b_sig = beat_signal_or_payload.beat_signal
    else:
        b_sig = np.asarray(beat_signal_or_payload)

    # Use real part if complex baseband beat signal is real-valued sinusoid
    x_input = np.real(b_sig) if np.iscomplexobj(b_sig) else b_sig

    return compute_fft(
        signal_or_array=x_input,
        sampling_rate_hz=sampling_rate_hz,
        window_type=window_name,
        n_fft=n_fft,
        one_sided=True
    )


def estimate_range(
    rx_payload: RadarRxPayload,
    tx_chirp: FMCWChirpContainer,
    window_name: str = "hann",
    n_fft: Optional[int] = None,
    interpolate_subbin: bool = True,
    max_expected_range_m: float = 490.0
) -> RangeEstimateResult:
    """Perform 1D Range FFT processing, range-axis mapping, and quantitative range estimation.

    FMCW Range Processing Pipeline:
        1. Validate ADC beat sampling Nyquist requirement for R_max.
        2. Extract de-chirped beat signal s_beat(t).
        3. Compute windowed one-sided FFT spectrum via Phase 3 compute_fft.
        4. Map frequency axis f to Range axis: R(f) = c * f / (2 * S).
        5. Detect peak beat frequency f_b using parabolic sub-bin interpolation.
        6. Calculate estimated range R_est = c * f_b / (2 * S) and evaluate error metrics.

    Args:
        rx_payload: RadarRxPayload holding beat signal and true target state.
        tx_chirp: Transmitted FMCWChirpContainer holding slope S, Fs, and B.
        window_name: Window function for spectral leakage suppression.
        n_fft: FFT length (supports zero-padding N_fft > N_signal).
        interpolate_subbin: If True, applies 3-point parabolic sub-bin interpolation.
        max_expected_range_m: Maximum expected range R_max for beat Nyquist validation.

    Returns:
        RangeEstimateResult container with estimated range, error metrics, and spectra.
    """
    fs = tx_chirp.sampling_rate_hz
    slope = tx_chirp.chirp_slope_hz_per_sec
    b = tx_chirp.bandwidth_hz

    # Validate Beat Signal Sampling Architecture
    beat_validation = validate_beat_sampling_rate(
        sampling_rate_hz=fs,
        chirp_slope_hz_per_sec=slope,
        max_expected_range_m=max_expected_range_m,
        is_complex_beat=False
    )

    # 1. Range FFT using Phase 3 FFT engine
    spec_res = compute_range_fft(
        beat_signal_or_payload=rx_payload,
        sampling_rate_hz=fs,
        chirp_slope_hz_per_sec=slope,
        window_name=window_name,
        n_fft=n_fft
    )

    # 2. Map beat frequency axis to Range axis R(f) = c * f / (2 * S)
    beat_freqs = spec_res.frequency_hz
    range_axis = SPEED_OF_LIGHT_M_PER_S * beat_freqs / (2.0 * slope)

    # 3. Peak detection with sub-bin parabolic interpolation
    dom_res = find_dominant_frequency(spec_res, ignore_dc=True, interpolate_subbin=interpolate_subbin)
    fb_meas = float(dom_res.frequency_hz)
    peak_mag = float(dom_res.magnitude)

    # 4. Range calculation R_est = c * f_b / (2 * S)
    r_est = SPEED_OF_LIGHT_M_PER_S * fb_meas / (2.0 * slope)

    # 5. Theoretical metrics and error evaluation
    r_true = float(rx_payload.target_state.range_m)
    fb_true = float(rx_payload.target_state.theoretical_beat_frequency_hz)

    err_freq = abs(fb_meas - fb_true)
    err_range = abs(r_est - r_true)
    rel_err_range = err_range / r_true if r_true > 0 else 0.0

    delta_r_theoretical = SPEED_OF_LIGHT_M_PER_S / (2.0 * b)
    r_max = calculate_max_unambiguous_range(fs, slope, is_complex=False)

    bin_spacing_freq = float(spec_res.bin_resolution_hz)
    bin_spacing_range = SPEED_OF_LIGHT_M_PER_S * bin_spacing_freq / (2.0 * slope)

    return RangeEstimateResult(
        estimated_range_m=r_est,
        beat_frequency_hz=fb_meas,
        peak_magnitude=peak_mag,
        frequency_error_hz=err_freq,
        range_error_m=err_range,
        relative_range_error=rel_err_range,
        theoretical_range_m=r_true,
        theoretical_beat_frequency_hz=fb_true,
        range_resolution_m=delta_r_theoretical,
        max_unambiguous_range_m=r_max,
        fft_bin_spacing_hz=bin_spacing_freq,
        range_bin_spacing_m=bin_spacing_range,
        range_axis_m=range_axis,
        beat_freq_axis_hz=beat_freqs,
        spectrum_magnitude=spec_res.magnitude,
        spectrum_db=spec_res.magnitude_db,
        beat_sampling_validation=beat_validation,
        metadata={
            "window_name": window_name,
            "n_signal_samples": spec_res.n_signal_samples,
            "n_fft_samples": spec_res.n_fft_samples,
            "is_zero_padded": spec_res.n_fft_samples > spec_res.n_signal_samples,
            "interpolated_subbin_index": dom_res.interpolated_bin,
            "sampling_architecture": tx_chirp.sampling_architecture,
        },
    )
