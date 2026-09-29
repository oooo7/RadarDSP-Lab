"""Sampling, Nyquist Analysis, and Aliasing Engine.

Provides tools for discrete uniform sampling, Nyquist-rate analysis,
theoretical alias frequency calculation, spectral frequency measurement, and error evaluation.

Pipeline Flow:
    Continuous / ideal signal
            ↓
        Sampling
            ↓
     Sampled signal
            ↓
     Nyquist analysis
            ↓
   Aliasing calculation
            ↓
Theoretical alias frequency
            ↓
 Measured alias frequency
            ↓
    Error calculation
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple
import numpy as np
import scipy.signal

from src.dsp.signals import SignalContainer, create_time_vector, generate_signal
from src.dsp.transforms import compute_fft
from src.utils.config import SignalGenConfig


@dataclass(frozen=True)
class SamplingAnalysisResult:
    """Immutable data payload holding sampling result, Nyquist analysis, aliasing predictions, and errors.

    Attributes:
        continuous_signal: High-resolution reference signal.
        sampled_signal: Signal sampled at target_fs_hz.
        target_fs_hz: Target sampling rate Fs in Hz.
        nyquist_rate_hz: Nyquist rate (2 * f_max) in Hz.
        folding_frequency_hz: Folding frequency (Fs / 2) in Hz.
        is_undersampled: True if Fs < 2 * f_max.
        is_critically_sampled: True if Fs == 2 * f_max.
        is_oversampled: True if Fs > 2 * f_max.
        is_aliased: True if any signal frequency exceeds Fs / 2.
        signal_frequencies_hz: List of fundamental frequencies present in reference signal.
        theoretical_alias_frequencies_hz: List of predicted alias frequencies folded into [0, Fs/2].
        measured_alias_frequencies_hz: List of measured peak spectral frequencies from sampled signal.
        absolute_errors_hz: List of absolute frequency errors |f_measured - f_theoretical| in Hz.
        relative_errors: List of relative frequency errors |f_measured - f_theoretical| / f_theoretical.
        metadata: Key-value dictionary of analysis settings.
    """
    continuous_signal: SignalContainer
    sampled_signal: SignalContainer
    target_fs_hz: float
    nyquist_rate_hz: float
    folding_frequency_hz: float
    is_undersampled: bool
    is_critically_sampled: bool
    is_oversampled: bool
    is_aliased: bool
    signal_frequencies_hz: List[float]
    theoretical_alias_frequencies_hz: List[float]
    measured_alias_frequencies_hz: List[float]
    absolute_errors_hz: List[float]
    relative_errors: List[float]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def alias_frequencies_hz(self) -> List[float]:
        """Backward-compatible alias for theoretical_alias_frequencies_hz."""
        return self.theoretical_alias_frequencies_hz


def calculate_alias_frequency(fundamental_freq_hz: float, sampling_rate_hz: float) -> float:
    """Calculate theoretical folded alias frequency in the principal Nyquist zone [0, Fs/2].

    For an input tone at frequency f >= 0 sampled at Fs > 0:
        remainder = f mod Fs
        f_alias = remainder if remainder <= Fs/2 else (Fs - remainder)

    Args:
        fundamental_freq_hz: Signal fundamental frequency f in Hz (>= 0).
        sampling_rate_hz: Sampling frequency Fs in Hz (> 0).

    Returns:
        Apparent aliased frequency in Hz within [0, Fs/2].

    Raises:
        ValueError: If fundamental_freq_hz < 0 or sampling_rate_hz <= 0.
    """
    if fundamental_freq_hz < 0:
        raise ValueError(f"Fundamental frequency must be non-negative (>= 0), got {fundamental_freq_hz}")
    if sampling_rate_hz <= 0:
        raise ValueError(f"Sampling rate Fs must be strictly positive (> 0), got {sampling_rate_hz}")

    rem = fundamental_freq_hz % sampling_rate_hz
    folding_freq = sampling_rate_hz / 2.0

    if rem <= folding_freq:
        return float(rem)
    else:
        return float(sampling_rate_hz - rem)


def analyze_nyquist(
    signal_frequencies_hz: List[float],
    sampling_rate_hz: float
) -> dict[str, Any]:
    """Perform Nyquist-rate analysis on a set of signal frequencies.

    Args:
        signal_frequencies_hz: List of tone frequencies present in signal in Hz.
        sampling_rate_hz: Target sampling rate Fs in Hz (> 0).

    Returns:
        Dictionary containing max frequency, Nyquist rate, folding frequency, and sampling state.

    Raises:
        ValueError: If sampling_rate_hz <= 0 or frequency < 0.
    """
    if sampling_rate_hz <= 0:
        raise ValueError(f"Sampling rate Fs must be strictly positive (> 0), got {sampling_rate_hz}")

    for f in signal_frequencies_hz:
        if f < 0:
            raise ValueError(f"Signal frequency must be non-negative (>= 0), got {f}")

    if not signal_frequencies_hz:
        f_max = 0.0
    else:
        f_max = float(max(signal_frequencies_hz))

    nyquist_rate = 2.0 * f_max
    folding_freq = sampling_rate_hz / 2.0

    is_undersampled = sampling_rate_hz < nyquist_rate if nyquist_rate > 0 else False
    is_critically_sampled = np.isclose(sampling_rate_hz, nyquist_rate, atol=1e-9) if nyquist_rate > 0 else False
    is_oversampled = sampling_rate_hz > nyquist_rate if nyquist_rate > 0 else True
    is_aliased = any(f > folding_freq for f in signal_frequencies_hz)

    theoretical_aliases = [calculate_alias_frequency(f, sampling_rate_hz) for f in signal_frequencies_hz]

    return {
        "f_max_hz": f_max,
        "nyquist_rate_hz": nyquist_rate,
        "folding_frequency_hz": folding_freq,
        "is_undersampled": is_undersampled,
        "is_critically_sampled": is_critically_sampled,
        "is_oversampled": is_oversampled,
        "is_aliased": is_aliased,
        "signal_frequencies_hz": list(signal_frequencies_hz),
        "theoretical_alias_frequencies_hz": theoretical_aliases,
    }


def measure_alias_frequencies(
    sampled_signal: SignalContainer,
    num_peaks: int = 1
) -> List[float]:
    """Measure apparent frequencies from sampled signal spectrum using canonical transforms engine.

    Employs real FFT and parabolic peak interpolation for sub-bin frequency accuracy.

    Args:
        sampled_signal: Discrete-time sampled signal container.
        num_peaks: Expected number of spectral tone peaks to identify.

    Returns:
        List of measured frequencies in Hz sorted by magnitude prominence.

    Raises:
        ValueError: If num_peaks < 1.
    """
    if num_peaks < 1:
        raise ValueError(f"num_peaks must be at least 1, got {num_peaks}")

    spec = compute_fft(sampled_signal, window_type="rect", one_sided=True)
    mags = np.copy(spec.magnitude)
    n = spec.n_fft_samples
    fs = spec.sampling_frequency_hz

    if n < 4:
        return [0.0] * num_peaks

    # Ignore DC bin 0
    if len(mags) > 1:
        mags[0] = 0.0

    top_indices = np.argsort(mags)[-num_peaks:][::-1]
    bin_res = spec.bin_resolution_hz

    measured_freqs = []
    for bin_idx in top_indices:
        # Parabolic interpolation
        if 0 < bin_idx < len(mags) - 1:
            alpha = mags[bin_idx - 1]
            beta = mags[bin_idx]
            gamma = mags[bin_idx + 1]
            denom = alpha - 2.0 * beta + gamma
            if abs(denom) > 1e-12:
                delta = float(np.clip(0.5 * (alpha - gamma) / denom, -0.5, 0.5))
            else:
                delta = 0.0
        else:
            delta = 0.0

        f_measured = (bin_idx + delta) * bin_res
        measured_freqs.append(float(abs(f_measured)))

    measured_freqs.sort()
    return measured_freqs


def calculate_alias_errors(
    theoretical_freqs_hz: List[float],
    measured_freqs_hz: List[float]
) -> dict[str, Any]:
    """Calculate absolute and relative error metrics between theoretical and measured frequencies.

    Args:
        theoretical_freqs_hz: Expected theoretical alias frequencies in Hz.
        measured_freqs_hz: Measured spectral frequencies in Hz.

    Returns:
        Dictionary holding absolute errors list, relative errors list, MAE, and RMSE.
    """
    if len(theoretical_freqs_hz) != len(measured_freqs_hz):
        # Match lengths by padding or truncating if needed
        min_len = min(len(theoretical_freqs_hz), len(measured_freqs_hz))
        theo = sorted(theoretical_freqs_hz)[:min_len]
        meas = sorted(measured_freqs_hz)[:min_len]
    else:
        theo = sorted(theoretical_freqs_hz)
        meas = sorted(measured_freqs_hz)

    abs_errors = []
    rel_errors = []

    for t_f, m_f in zip(theo, meas):
        abs_err = float(abs(m_f - t_f))
        abs_errors.append(abs_err)

        if t_f > 1e-9:
            rel_err = float(abs_err / t_f)
        else:
            rel_err = float(abs_err)
        rel_errors.append(rel_err)

    mae = float(np.mean(abs_errors)) if abs_errors else 0.0
    rmse = float(np.sqrt(np.mean(np.square(abs_errors)))) if abs_errors else 0.0

    return {
        "absolute_errors_hz": abs_errors,
        "relative_errors": rel_errors,
        "mae_hz": mae,
        "rmse_hz": rmse,
    }


def extract_signal_frequencies(signal: SignalContainer) -> List[float]:
    """Extract known fundamental tone frequencies from signal metadata."""
    meta = signal.metadata
    if "frequency_hz" in meta:
        return [float(meta["frequency_hz"])]
    elif "frequencies_hz" in meta:
        return [float(f) for f in meta["frequencies_hz"]]
    elif "f_start_hz" in meta and "f_end_hz" in meta:
        return [float(meta["f_start_hz"]), float(meta["f_end_hz"])]
    elif "carrier_frequency_hz" in meta:
        return [float(meta["carrier_frequency_hz"])]
    return [100.0]


def sample_signal(
    continuous_signal: SignalContainer,
    target_fs_hz: float,
    signal_frequencies_hz: Optional[List[float]] = None
) -> SamplingAnalysisResult:
    """Execute complete Sampling, Nyquist Analysis, Aliasing Calculation, and Error Evaluation pipeline.

    Pipeline Steps:
        1. Resample high-resolution continuous reference signal at target_fs_hz.
        2. Perform Nyquist rate analysis (check undersampling/oversampling).
        3. Compute theoretical folded alias frequencies in [0, Fs/2].
        4. Measure apparent frequencies via FFT peak estimation.
        5. Evaluate absolute and relative frequency estimation errors.

    Args:
        continuous_signal: Reference high-resolution continuous signal.
        target_fs_hz: Target sampling frequency Fs in Hz (> 0).
        signal_frequencies_hz: Optional list of true component frequencies in Hz.

    Returns:
        SamplingAnalysisResult container with comprehensive analysis payload.

    Raises:
        ValueError: If target_fs_hz <= 0 or target_fs_hz > continuous_signal.sampling_rate_hz.
    """
    if target_fs_hz <= 0:
        raise ValueError(f"Target sampling frequency Fs must be strictly positive (> 0), got {target_fs_hz}")

    if target_fs_hz > continuous_signal.sampling_rate_hz:
        raise ValueError(
            f"Target sampling rate ({target_fs_hz} Hz) cannot exceed reference continuous "
            f"signal sampling rate ({continuous_signal.sampling_rate_hz} Hz)."
        )

    # Extract or use signal frequencies
    if signal_frequencies_hz is None:
        freqs = extract_signal_frequencies(continuous_signal)
    else:
        freqs = list(signal_frequencies_hz)

    # Step 1: Discrete Uniform Sampling
    t_target = create_time_vector(target_fs_hz, continuous_signal.duration_sec)

    # Perform high-accuracy interpolation or direct configuration regeneration
    if "amplitude" in continuous_signal.metadata or "signal_type" in continuous_signal.metadata:
        # Re-generate sampled signal directly at target_fs_hz for exact mathematical ground truth
        stype = continuous_signal.signal_type
        meta = continuous_signal.metadata
        cfg = SignalGenConfig(
            signal_type=stype,
            sampling_rate_hz=target_fs_hz,
            duration_sec=continuous_signal.duration_sec,
            amplitude=meta.get("amplitude", 1.0),
            frequency_hz=meta.get("frequency_hz", freqs[0] if freqs else 100.0),
            phase_rad=meta.get("phase_rad", 0.0),
            dc_offset=meta.get("dc_offset", 0.0),
            duty_cycle=meta.get("duty_cycle", 0.5),
            frequencies_hz=meta.get("frequencies_hz", freqs),
            amplitudes=meta.get("amplitudes", [1.0] * len(freqs)),
            phases_rad=meta.get("phases_rad", [0.0] * len(freqs)),
            f_start_hz=meta.get("f_start_hz", 0.0),
            f_end_hz=meta.get("f_end_hz", 500.0),
            carrier_frequency_hz=meta.get("carrier_frequency_hz", 100.0),
            modulation_frequency_hz=meta.get("modulation_frequency_hz", 10.0),
            modulation_index=meta.get("modulation_index", 0.5),
            frequency_deviation_hz=meta.get("frequency_deviation_hz", 25.0),
            std_dev=meta.get("std_dev", 1.0),
            seed=meta.get("seed", None),
        )
        sampled_sig = generate_signal(cfg)
    else:
        # Interpolate continuous signal vector onto target discrete time points
        sampled_amp = np.interp(t_target, continuous_signal.time_vector, continuous_signal.amplitude)
        sampled_sig = SignalContainer(
            time_vector=t_target,
            amplitude=sampled_amp,
            sampling_rate_hz=target_fs_hz,
            duration_sec=continuous_signal.duration_sec,
            signal_type=continuous_signal.signal_type,
            metadata=continuous_signal.metadata,
        )

    # Step 2: Nyquist Rate Analysis
    nyquist_report = analyze_nyquist(freqs, target_fs_hz)

    # Step 3: Theoretical Alias Calculation
    theoretical_aliases = nyquist_report["theoretical_alias_frequencies_hz"]

    # Step 4: Measured Alias Frequency (via spectral peak analysis)
    num_peaks = max(1, len(freqs))
    measured_aliases = measure_alias_frequencies(sampled_sig, num_peaks=num_peaks)

    # Step 5: Error Calculation
    error_report = calculate_alias_errors(theoretical_aliases, measured_aliases)

    return SamplingAnalysisResult(
        continuous_signal=continuous_signal,
        sampled_signal=sampled_sig,
        target_fs_hz=target_fs_hz,
        nyquist_rate_hz=nyquist_report["nyquist_rate_hz"],
        folding_frequency_hz=nyquist_report["folding_frequency_hz"],
        is_undersampled=nyquist_report["is_undersampled"],
        is_critically_sampled=nyquist_report["is_critically_sampled"],
        is_oversampled=nyquist_report["is_oversampled"],
        is_aliased=nyquist_report["is_aliased"],
        signal_frequencies_hz=freqs,
        theoretical_alias_frequencies_hz=theoretical_aliases,
        measured_alias_frequencies_hz=measured_aliases,
        absolute_errors_hz=error_report["absolute_errors_hz"],
        relative_errors=error_report["relative_errors"],
        metadata={
            "f_max_hz": nyquist_report["f_max_hz"],
            "mae_hz": error_report["mae_hz"],
            "rmse_hz": error_report["rmse_hz"],
        },
    )
