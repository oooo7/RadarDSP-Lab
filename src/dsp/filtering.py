"""Digital Filtering Engine.

Provides FIR and IIR (Butterworth) filter design, Second-Order Sections (SOS) representation,
causal & zero-phase filtering, frequency response analysis, and group delay evaluation.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple, Union
import numpy as np
import scipy.signal

from src.dsp.signals import SignalContainer
from src.utils.config import FilterConfig


@dataclass(frozen=True)
class FilterDesignResult:
    """Immutable data container holding filter coefficients, design specifications, and metadata.

    Attributes:
        filter_type: Filter type ('lowpass', 'highpass', 'bandpass', 'bandstop').
        filter_family: Filter family ('fir', 'butterworth').
        order: Filter order M (Note: FIR filter with N_taps has order M = N_taps - 1).
        n_taps: Number of filter coefficients/taps.
        sampling_rate_hz: Sampling frequency Fs in Hz.
        cutoff_hz: Cutoff frequency or frequencies in Hz.
        b_coefficients: 1D float64 array of numerator coefficients b.
        a_coefficients: 1D float64 array of denominator coefficients a (1.0 for FIR).
        sos: 2D float64 array of Second-Order Sections (for IIR) or None (for FIR).
        window_name: Window name used for FIR design (or None for IIR).
        metadata: Key-value dictionary of design metadata.
    """
    filter_type: str
    filter_family: str
    order: int
    n_taps: int
    sampling_rate_hz: float
    cutoff_hz: List[float]
    b_coefficients: np.ndarray
    a_coefficients: np.ndarray
    sos: Optional[np.ndarray] = None
    window_name: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


# Backward-compatible alias for Phase 0
FilterResponse = FilterDesignResult


@dataclass(frozen=True)
class FilterResponseResult:
    """Container holding frequency response, magnitude in dB, phase, group delay, and cutoff metrics.

    Attributes:
        frequency_hz: 1D float64 array of frequency points from 0 to Fs/2 in Hz.
        magnitude: 1D float64 array of magnitude response |H(f)|.
        magnitude_db: 1D float64 array of magnitude response in dB (20*log10|H(f)|).
        phase_rad: 1D float64 unwrapped phase response in radians.
        group_delay_samples: 1D float64 array of group delay in samples.
        dc_gain: Linear magnitude gain at DC (0 Hz).
        nyquist_gain: Linear magnitude gain at Nyquist (Fs/2 Hz).
        cutoff_3db_hz: List of measured -3 dB cutoff frequencies in Hz.
    """
    frequency_hz: np.ndarray
    magnitude: np.ndarray
    magnitude_db: np.ndarray
    phase_rad: np.ndarray
    group_delay_samples: np.ndarray
    dc_gain: float
    nyquist_gain: float
    cutoff_3db_hz: List[float]


def _validate_filter_params(
    filter_type: str,
    cutoff_hz: Union[float, List[float], np.ndarray],
    order: int,
    sampling_rate_hz: float
) -> Tuple[str, List[float]]:
    """Validate core digital filter parameters and normalize cutoff frequencies.

    Args:
        filter_type: Filter response type ('lowpass', 'highpass', 'bandpass', 'bandstop').
        cutoff_hz: Cutoff frequency or frequencies in Hz.
        order: Filter order (must be >= 1).
        sampling_rate_hz: Sampling frequency Fs in Hz (must be > 0).

    Returns:
        Tuple of (normalized filter_type string, list of float cutoff frequencies in Hz).

    Raises:
        ValueError: If parameters are outside valid physical bounds.
    """
    if sampling_rate_hz <= 0:
        raise ValueError(f"Sampling frequency Fs must be strictly positive (> 0), got {sampling_rate_hz}")

    if order < 1:
        raise ValueError(f"Filter order must be at least 1, got {order}")

    ftype = filter_type.lower().strip()
    if ftype not in ["lowpass", "highpass", "bandpass", "bandstop"]:
        raise ValueError(
            f"Unsupported filter_type '{filter_type}'. Supported types: "
            "lowpass, highpass, bandpass, bandstop."
        )

    nyquist = sampling_rate_hz / 2.0

    if isinstance(cutoff_hz, (int, float, np.number)):
        cutoffs = [float(cutoff_hz)]
    else:
        cutoffs = [float(c) for c in cutoff_hz]

    if ftype in ["lowpass", "highpass"]:
        if len(cutoffs) != 1:
            raise ValueError(f"{ftype.capitalize()} filter requires exactly 1 cutoff frequency, got {len(cutoffs)}")
        fc = cutoffs[0]
        if not (0.0 < fc < nyquist):
            raise ValueError(f"Cutoff frequency ({fc} Hz) must be strictly between 0 and Nyquist ({nyquist} Hz).")
    elif ftype in ["bandpass", "bandstop"]:
        if len(cutoffs) != 2:
            raise ValueError(f"{ftype.capitalize()} filter requires exactly 2 cutoff frequencies [f_low, f_high], got {len(cutoffs)}")
        f_low, f_high = cutoffs[0], cutoffs[1]
        if not (0.0 < f_low < f_high < nyquist):
            raise ValueError(
                f"Band filter cutoffs [{f_low}, {f_high}] Hz must satisfy "
                f"0 < f_low < f_high < Nyquist ({nyquist} Hz)."
            )

    return ftype, cutoffs


def design_fir_filter(
    filter_type: str,
    cutoff_hz: Union[float, List[float]],
    order: int,
    sampling_rate_hz: float,
    window_type: str = "hamming",
    kaiser_beta: float = 8.6
) -> FilterDesignResult:
    """Design a finite impulse response (FIR) digital filter using window method (scipy.signal.firwin).

    FIR Order vs Taps:
        An FIR filter with N_taps coefficients has filter order M = N_taps - 1.

    Args:
        filter_type: Response type ('lowpass', 'highpass', 'bandpass', 'bandstop').
        cutoff_hz: Cutoff frequency or frequencies in Hz.
        order: Filter order M >= 1 (yields N_taps = order + 1).
        sampling_rate_hz: Sampling frequency Fs in Hz (> 0).
        window_type: Window name ('rectangular', 'hann', 'hamming', 'blackman', 'kaiser').
        kaiser_beta: Beta parameter for Kaiser window.

    Returns:
        FilterDesignResult container holding b coefficients (a = [1.0]).

    Raises:
        ValueError: If parameters are physically invalid.
    """
    ftype, cutoffs = _validate_filter_params(filter_type, cutoff_hz, order, sampling_rate_hz)

    # Determine number of taps
    numtaps = order + 1

    # Highpass and Bandstop FIR filters require an ODD number of taps (even order)
    # to maintain a non-zero gain response at the Nyquist frequency (Fs/2).
    if ftype in ["highpass", "bandstop"] and numtaps % 2 == 0:
        numtaps += 1

    w_name = window_type.lower().strip()
    if w_name in ["rect", "rectangular", "boxcar"]:
        win_param = "boxcar"
    elif w_name in ["hann", "hanning"]:
        win_param = "hann"
    elif w_name in ["hamming"]:
        win_param = "hamming"
    elif w_name in ["blackman"]:
        win_param = "blackman"
    elif w_name in ["kaiser"]:
        win_param = ("kaiser", kaiser_beta)
    else:
        win_param = w_name

    pass_zero_dict = {
        "lowpass": True,
        "highpass": False,
        "bandpass": False,
        "bandstop": True,
    }
    pass_zero = pass_zero_dict[ftype]

    # Design FIR using scipy.signal.firwin
    b = scipy.signal.firwin(
        numtaps=numtaps,
        cutoff=cutoffs if len(cutoffs) > 1 else cutoffs[0],
        window=win_param,
        pass_zero=pass_zero,
        fs=sampling_rate_hz
    )

    a = np.array([1.0], dtype=np.float64)
    actual_order = len(b) - 1

    # Check linear phase symmetry b[n] == b[N-1-n]
    is_symmetric = bool(np.allclose(b, b[::-1], atol=1e-10))

    return FilterDesignResult(
        filter_type=ftype,
        filter_family="fir",
        order=actual_order,
        n_taps=len(b),
        sampling_rate_hz=sampling_rate_hz,
        cutoff_hz=cutoffs,
        b_coefficients=b,
        a_coefficients=a,
        sos=None,
        window_name=window_type,
        metadata={
            "is_linear_phase": is_symmetric,
            "theoretical_group_delay_samples": actual_order / 2.0 if is_symmetric else None,
        },
    )


def design_iir_butterworth(
    filter_type: str,
    cutoff_hz: Union[float, List[float]],
    order: int,
    sampling_rate_hz: float
) -> FilterDesignResult:
    """Design an IIR Butterworth digital filter using Second-Order Sections (SOS).

    Numerical Stability:
        Higher-order IIR filters designed in direct transfer function form (b, a) suffer from severe
        coefficient quantization and pole sensitivity. Using Second-Order Sections (SOS) ensures
        optimal numerical stability.

    Args:
        filter_type: Response type ('lowpass', 'highpass', 'bandpass', 'bandstop').
        cutoff_hz: Cutoff frequency or frequencies in Hz.
        order: Filter order N >= 1.
        sampling_rate_hz: Sampling frequency Fs in Hz (> 0).

    Returns:
        FilterDesignResult container holding SOS matrix, b, and a coefficients.

    Raises:
        ValueError: If parameters are physically invalid.
    """
    ftype, cutoffs = _validate_filter_params(filter_type, cutoff_hz, order, sampling_rate_hz)

    # Design SOS and transfer function coefficients using scipy.signal.butter
    sos = scipy.signal.butter(
        N=order,
        Wn=cutoffs if len(cutoffs) > 1 else cutoffs[0],
        btype=ftype,
        analog=False,
        output="sos",
        fs=sampling_rate_hz
    )

    b, a = scipy.signal.butter(
        N=order,
        Wn=cutoffs if len(cutoffs) > 1 else cutoffs[0],
        btype=ftype,
        analog=False,
        output="ba",
        fs=sampling_rate_hz
    )

    return FilterDesignResult(
        filter_type=ftype,
        filter_family="butterworth",
        order=order,
        n_taps=order + 1,
        sampling_rate_hz=sampling_rate_hz,
        cutoff_hz=cutoffs,
        b_coefficients=b,
        a_coefficients=a,
        sos=sos,
        window_name=None,
        metadata={"is_stable_sos": True},
    )


def design_filter(config: FilterConfig, fs_hz: float) -> FilterDesignResult:
    """Unified configuration dispatcher designing an FIR or IIR filter from FilterConfig.

    Args:
        config: Filter configuration dataclass or Pydantic model.
        fs_hz: Sampling rate Fs in Hz (> 0).

    Returns:
        FilterDesignResult container.
    """
    method = config.design_method.lower().strip()
    ftype = config.filter_type.lower().strip()

    if method in ["firwin", "fir"]:
        return design_fir_filter(
            filter_type=ftype,
            cutoff_hz=config.cutoff_hz,
            order=config.order,
            sampling_rate_hz=fs_hz,
            window_type=config.window
        )
    elif method in ["butter", "butterworth", "iir"]:
        return design_iir_butterworth(
            filter_type=ftype,
            cutoff_hz=config.cutoff_hz,
            order=config.order,
            sampling_rate_hz=fs_hz
        )
    else:
        raise ValueError(
            f"Unsupported design_method '{config.design_method}'. Supported methods: firwin, butter."
        )


def apply_filter(
    signal_or_array: SignalContainer | np.ndarray,
    filter_design: FilterDesignResult,
    zero_phase: bool = False
) -> SignalContainer | np.ndarray:
    """Apply designed FIR or IIR filter to a discrete-time signal.

    Filtering Modes:
        - Causal Filtering (zero_phase = False):
          Introduces physical phase delay / group delay.
          Uses lfilter (for FIR) or sosfilt (for IIR SOS).
        - Zero-Phase Filtering (zero_phase = True):
          Forward-backward filtering that removes all phase distortion (non-causal offline processing).
          Uses filtfilt (for FIR) or sosfiltfilt (for IIR SOS).

    Args:
        signal_or_array: Input SignalContainer or 1D numerical array.
        filter_design: FilterDesignResult container.
        zero_phase: If True, executes zero-phase filtfilt/sosfiltfilt.

    Returns:
        Filtered SignalContainer or 1D float64 numpy array (matching input type).

    Raises:
        ValueError: If signal array is empty, contains non-finite values, or is too short for zero-phase filtering.
    """
    if isinstance(signal_or_array, SignalContainer):
        x = np.asarray(signal_or_array.amplitude, dtype=np.float64)
        is_container = True
    else:
        x = np.asarray(signal_or_array, dtype=np.float64)
        is_container = False

    if len(x) == 0:
        raise ValueError("Input signal array is empty.")

    if not np.all(np.isfinite(x)):
        raise ValueError("Input signal contains non-finite NaN or Inf values.")

    # Apply filter algorithm
    if filter_design.filter_family == "fir":
        b = filter_design.b_coefficients
        a = filter_design.a_coefficients
        if zero_phase:
            min_len = 3 * len(b)
            if len(x) < min_len:
                raise ValueError(
                    f"Signal length ({len(x)}) is too short for zero-phase FIR filtering. "
                    f"Minimum required length is {min_len} samples."
                )
            y = scipy.signal.filtfilt(b, a, x)
        else:
            y = scipy.signal.lfilter(b, a, x)
    else:
        # IIR Filter via SOS representation
        sos = filter_design.sos
        if sos is None:
            b, a = filter_design.b_coefficients, filter_design.a_coefficients
            if zero_phase:
                y = scipy.signal.filtfilt(b, a, x)
            else:
                y = scipy.signal.lfilter(b, a, x)
        else:
            if zero_phase:
                min_len = 3 * filter_design.order
                if len(x) < min_len:
                    raise ValueError(
                        f"Signal length ({len(x)}) is too short for zero-phase IIR filtering. "
                        f"Minimum required length is {min_len} samples."
                    )
                y = scipy.signal.sosfiltfilt(sos, x)
            else:
                y = scipy.signal.sosfilt(sos, x)

    if is_container:
        desc = f"Filtered ({filter_design.filter_family.upper()} {filter_design.filter_type}, zero_phase={zero_phase})"
        return SignalContainer(
            time_vector=signal_or_array.time_vector,
            amplitude=y,
            sampling_rate_hz=signal_or_array.sampling_rate_hz,
            duration_sec=signal_or_array.duration_sec,
            signal_type=signal_or_array.signal_type,
            metadata={**signal_or_array.metadata, "filter_description": desc},
        )
    return y


def analyze_filter_response(
    filter_design: FilterDesignResult,
    n_points: int = 512
) -> FilterResponseResult:
    """Analyze frequency response, magnitude in dB, phase, group delay, and -3 dB cutoff frequency.

    Args:
        filter_design: FilterDesignResult container.
        n_points: Number of evaluation frequency grid points from 0 to Fs/2.

    Returns:
        FilterResponseResult container holding detailed spectral characteristics.

    Raises:
        ValueError: If n_points < 16.
    """
    if n_points < 16:
        raise ValueError(f"n_points must be at least 16, got {n_points}")

    fs = filter_design.sampling_rate_hz

    if filter_design.sos is not None:
        w, h = scipy.signal.sosfreqz(filter_design.sos, worN=n_points, fs=fs)
    else:
        w, h = scipy.signal.freqz(
            filter_design.b_coefficients,
            filter_design.a_coefficients,
            worN=n_points,
            fs=fs
        )

    freqs = w
    mag = np.abs(h)
    mag_db = 20.0 * np.log10(np.maximum(mag, 1e-12))
    phase = np.unwrap(np.angle(h))

    # Calculate group delay
    try:
        if filter_design.filter_family == "fir":
            _, gd = scipy.signal.group_delay(
                (filter_design.b_coefficients, filter_design.a_coefficients),
                w=n_points,
                fs=fs
            )
        else:
            # Numerical gradient of phase wrt frequency for IIR
            dw = 2.0 * np.pi * (freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0
            gd = -np.gradient(phase, dw)
    except Exception:
        gd = np.full_like(freqs, fill_value=filter_design.order / 2.0)

    dc_gain = float(mag[0])
    nyquist_gain = float(mag[-1])

    # Find -3 dB cutoff frequency relative to maximum passband gain
    max_passband_gain = float(np.max(mag))
    target_3db = max_passband_gain / np.sqrt(2.0)

    cutoff_3db_list = []
    # Identify crossings where magnitude passes target_3db threshold
    diff_mag = mag - target_3db
    zero_crossings = np.where(np.diff(np.sign(diff_mag)))[0]
    for idx in zero_crossings:
        f_val = float(freqs[idx])
        cutoff_3db_list.append(f_val)

    return FilterResponseResult(
        frequency_hz=freqs,
        magnitude=mag,
        magnitude_db=mag_db,
        phase_rad=phase,
        group_delay_samples=gd,
        dc_gain=dc_gain,
        nyquist_gain=nyquist_gain,
        cutoff_3db_hz=cutoff_3db_list if cutoff_3db_list else None,
    )


def compare_fir_vs_iir(
    signal: SignalContainer,
    cutoff_hz: float,
    fir_order: int = 64,
    iir_order: int = 4
) -> dict[str, Any]:
    """Compare FIR (firwin) vs IIR (Butterworth) lowpass filter performance on a given signal.

    Args:
        signal: Input SignalContainer.
        cutoff_hz: Lowpass cutoff frequency in Hz.
        fir_order: FIR filter order M (default 64).
        iir_order: IIR Butterworth filter order N (default 4).

    Returns:
        Dictionary containing design results, filtered signals, and performance metrics.
    """
    fs = signal.sampling_rate_hz

    fir_design = design_fir_filter("lowpass", cutoff_hz, fir_order, fs)
    iir_design = design_iir_butterworth("lowpass", cutoff_hz, iir_order, fs)

    fir_filtered_causal = apply_filter(signal, fir_design, zero_phase=False)
    iir_filtered_causal = apply_filter(signal, iir_design, zero_phase=False)

    fir_resp = analyze_filter_response(fir_design)
    iir_resp = analyze_filter_response(iir_design)

    return {
        "fir_design": fir_design,
        "iir_design": iir_design,
        "fir_filtered_signal": fir_filtered_causal,
        "iir_filtered_signal": iir_filtered_causal,
        "fir_response": fir_resp,
        "iir_response": iir_resp,
    }


def resample_signal(
    signal: SignalContainer,
    factor_up: int = 1,
    factor_down: int = 1
) -> SignalContainer:
    """Interpolate (upsample) or decimate (downsample) signal placeholder.

    Raises:
        NotImplementedError: Scheduled for future phase.
    """
    raise NotImplementedError("Resampling and rate conversion is scheduled for a future phase.")
