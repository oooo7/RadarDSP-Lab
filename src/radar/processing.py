"""FMCW Radar Range Processing Engine.

Provides de-chirping, 1D Range FFT computation, range-axis mapping,
window selection, zero-padding interpolation, maximum unambiguous range calculation,
and quantitative range estimation validation.
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
    metadata: dict[str, Any] = field(default_factory=dict)


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
    interpolate_subbin: bool = True
) -> RangeEstimateResult:
    """Perform 1D Range FFT processing, range-axis mapping, and quantitative range estimation.

    FMCW Range Processing Pipeline:
        1. Extract de-chirped beat signal s_beat(t).
        2. Compute windowed one-sided FFT spectrum via Phase 3 compute_fft.
        3. Map frequency axis f to Range axis: R(f) = c * f / (2 * S).
        4. Detect peak beat frequency f_b using parabolic sub-bin interpolation.
        5. Calculate estimated range R_est = c * f_b / (2 * S) and evaluate error metrics.

    Args:
        rx_payload: RadarRxPayload holding beat signal and true target state.
        tx_chirp: Transmitted FMCWChirpContainer holding slope S, Fs, and B.
        window_name: Window function for spectral leakage suppression.
        n_fft: FFT length (supports zero-padding N_fft > N_signal).
        interpolate_subbin: If True, applies 3-point parabolic sub-bin interpolation.

    Returns:
        RangeEstimateResult container with estimated range, error metrics, and spectra.
    """
    fs = tx_chirp.sampling_rate_hz
    slope = tx_chirp.chirp_slope_hz_per_sec
    b = tx_chirp.bandwidth_hz

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
        metadata={
            "window_name": window_name,
            "n_signal_samples": spec_res.n_signal_samples,
            "n_fft_samples": spec_res.n_fft_samples,
            "is_zero_padded": spec_res.n_fft_samples > spec_res.n_signal_samples,
            "interpolated_subbin_index": dom_res.interpolated_bin,
        },
    )
