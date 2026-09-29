"""DSP Signal Generation Engine.

Provides deterministic numerical signal generation for 10 core signal types:
Sine, Cosine, Square, Triangle, Sawtooth, Multi-tone, Linear Chirp, Gaussian Noise, AM, and FM.

All signal generators produce an immutable SignalContainer holding the time vector,
amplitude array, sampling rate, duration, and parameter metadata.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional
import numpy as np
import scipy.signal
from src.utils.config import SignalGenConfig


@dataclass(frozen=True)
class SignalContainer:
    """Immutable data container for a discrete-time numerical signal.

    Attributes:
        time_vector: 1D float64 numpy array of sample timestamps in seconds.
        amplitude: 1D float64 numpy array of discrete-time signal values x[n].
        sampling_rate_hz: Sampling frequency Fs in Hz.
        duration_sec: Total signal duration in seconds.
        signal_type: Identifier string for the signal type.
        metadata: Key-value dictionary storing configuration parameters.
    """
    time_vector: np.ndarray
    amplitude: np.ndarray
    sampling_rate_hz: float
    duration_sec: float
    signal_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


def create_time_vector(sampling_rate_hz: float, duration_sec: float) -> np.ndarray:
    """Construct uniform discrete-time vector.

    Sample count N = int(round(duration_sec * sampling_rate_hz)).
    Time values t_n = n / sampling_rate_hz for n = 0, 1, ..., N - 1.

    Endpoint convention:
        The duration endpoint t = duration_sec is EXCLUDED from the array.
        For Fs = 1000 Hz and duration = 1.0 s, N = 1000 samples spanning [0.0, 0.999] s.
        This prevents sample count off-by-one errors and preserves exact sampling interval dt = 1/Fs.

    Args:
        sampling_rate_hz: Sampling frequency Fs in Hz (must be > 0).
        duration_sec: Duration in seconds (must be > 0).

    Returns:
        1D float64 numpy array of sample timestamps.

    Raises:
        ValueError: If sampling_rate_hz <= 0 or duration_sec <= 0.
    """
    if sampling_rate_hz <= 0:
        raise ValueError(f"Sampling rate Fs must be strictly positive (> 0), got {sampling_rate_hz}")
    if duration_sec <= 0:
        raise ValueError(f"Duration must be strictly positive (> 0), got {duration_sec}")

    num_samples = int(np.round(duration_sec * sampling_rate_hz))
    if num_samples <= 0:
        raise ValueError(f"Duration {duration_sec}s at Fs {sampling_rate_hz}Hz produces 0 samples.")

    return np.arange(num_samples, dtype=np.float64) / float(sampling_rate_hz)


def _validate_basic_params(amplitude: float, sampling_rate_hz: float, duration_sec: float) -> None:
    """Validate fundamental signal parameters."""
    if not np.isfinite(amplitude):
        raise ValueError(f"Amplitude must be a finite number, got {amplitude}")
    if sampling_rate_hz <= 0:
        raise ValueError(f"Sampling rate Fs must be strictly positive (> 0), got {sampling_rate_hz}")
    if duration_sec <= 0:
        raise ValueError(f"Duration must be strictly positive (> 0), got {duration_sec}")


def _validate_frequency(frequency_hz: float, param_name: str = "frequency") -> None:
    """Validate non-negative frequency."""
    if not np.isfinite(frequency_hz) or frequency_hz < 0:
        raise ValueError(f"{param_name} must be finite and non-negative (>= 0), got {frequency_hz}")


def generate_sine(
    amplitude: float,
    frequency_hz: float,
    phase_rad: float,
    sampling_rate_hz: float,
    duration_sec: float,
    dc_offset: float = 0.0
) -> SignalContainer:
    """Generate sinusoidal signal x(t) = A * sin(2*pi*f*t + phase) + dc_offset.

    Args:
        amplitude: Peak amplitude A.
        frequency_hz: Frequency f in Hz (>= 0).
        phase_rad: Initial phase in radians.
        sampling_rate_hz: Sampling frequency Fs in Hz (> 0).
        duration_sec: Duration in seconds (> 0).
        dc_offset: Constant DC bias offset.

    Returns:
        SignalContainer with time vector and amplitude values.
    """
    _validate_basic_params(amplitude, sampling_rate_hz, duration_sec)
    _validate_frequency(frequency_hz)

    t = create_time_vector(sampling_rate_hz, duration_sec)
    x = amplitude * np.sin(2.0 * np.pi * frequency_hz * t + phase_rad) + dc_offset

    metadata = {
        "amplitude": amplitude,
        "frequency_hz": frequency_hz,
        "phase_rad": phase_rad,
        "dc_offset": dc_offset,
    }
    return SignalContainer(
        time_vector=t,
        amplitude=x,
        sampling_rate_hz=sampling_rate_hz,
        duration_sec=duration_sec,
        signal_type="sine",
        metadata=metadata,
    )


def generate_cosine(
    amplitude: float,
    frequency_hz: float,
    phase_rad: float,
    sampling_rate_hz: float,
    duration_sec: float,
    dc_offset: float = 0.0
) -> SignalContainer:
    """Generate cosinusoidal signal x(t) = A * cos(2*pi*f*t + phase) + dc_offset.

    Args:
        amplitude: Peak amplitude A.
        frequency_hz: Frequency f in Hz (>= 0).
        phase_rad: Initial phase in radians.
        sampling_rate_hz: Sampling frequency Fs in Hz (> 0).
        duration_sec: Duration in seconds (> 0).
        dc_offset: Constant DC bias offset.

    Returns:
        SignalContainer with time vector and amplitude values.
    """
    _validate_basic_params(amplitude, sampling_rate_hz, duration_sec)
    _validate_frequency(frequency_hz)

    t = create_time_vector(sampling_rate_hz, duration_sec)
    x = amplitude * np.cos(2.0 * np.pi * frequency_hz * t + phase_rad) + dc_offset

    metadata = {
        "amplitude": amplitude,
        "frequency_hz": frequency_hz,
        "phase_rad": phase_rad,
        "dc_offset": dc_offset,
    }
    return SignalContainer(
        time_vector=t,
        amplitude=x,
        sampling_rate_hz=sampling_rate_hz,
        duration_sec=duration_sec,
        signal_type="cosine",
        metadata=metadata,
    )


def generate_square(
    amplitude: float,
    frequency_hz: float,
    phase_rad: float,
    duty_cycle: float,
    sampling_rate_hz: float,
    duration_sec: float,
    dc_offset: float = 0.0
) -> SignalContainer:
    """Generate square wave with specified duty cycle.

    Args:
        amplitude: Peak amplitude A.
        frequency_hz: Fundamental frequency f in Hz (>= 0).
        phase_rad: Initial phase in radians.
        duty_cycle: Duty cycle ratio in open interval (0.0, 1.0).
        sampling_rate_hz: Sampling frequency Fs in Hz (> 0).
        duration_sec: Duration in seconds (> 0).
        dc_offset: Constant DC bias offset.

    Returns:
        SignalContainer with time vector and amplitude values.
    """
    _validate_basic_params(amplitude, sampling_rate_hz, duration_sec)
    _validate_frequency(frequency_hz)

    if not (0.0 < duty_cycle < 1.0):
        raise ValueError(f"Duty cycle must be strictly between 0 and 1 exclusive (0 < duty < 1), got {duty_cycle}")

    t = create_time_vector(sampling_rate_hz, duration_sec)
    sq = scipy.signal.square(2.0 * np.pi * frequency_hz * t + phase_rad, duty=duty_cycle)
    x = amplitude * sq + dc_offset

    metadata = {
        "amplitude": amplitude,
        "frequency_hz": frequency_hz,
        "phase_rad": phase_rad,
        "duty_cycle": duty_cycle,
        "dc_offset": dc_offset,
    }
    return SignalContainer(
        time_vector=t,
        amplitude=x,
        sampling_rate_hz=sampling_rate_hz,
        duration_sec=duration_sec,
        signal_type="square",
        metadata=metadata,
    )


def generate_triangle(
    amplitude: float,
    frequency_hz: float,
    phase_rad: float,
    sampling_rate_hz: float,
    duration_sec: float,
    dc_offset: float = 0.0
) -> SignalContainer:
    """Generate symmetric triangular wave.

    Args:
        amplitude: Peak amplitude A.
        frequency_hz: Fundamental frequency f in Hz (>= 0).
        phase_rad: Initial phase in radians.
        sampling_rate_hz: Sampling frequency Fs in Hz (> 0).
        duration_sec: Duration in seconds (> 0).
        dc_offset: Constant DC bias offset.

    Returns:
        SignalContainer with time vector and amplitude values.
    """
    _validate_basic_params(amplitude, sampling_rate_hz, duration_sec)
    _validate_frequency(frequency_hz)

    t = create_time_vector(sampling_rate_hz, duration_sec)
    tri = scipy.signal.sawtooth(2.0 * np.pi * frequency_hz * t + phase_rad, width=0.5)
    x = amplitude * tri + dc_offset

    metadata = {
        "amplitude": amplitude,
        "frequency_hz": frequency_hz,
        "phase_rad": phase_rad,
        "dc_offset": dc_offset,
    }
    return SignalContainer(
        time_vector=t,
        amplitude=x,
        sampling_rate_hz=sampling_rate_hz,
        duration_sec=duration_sec,
        signal_type="triangle",
        metadata=metadata,
    )


def generate_sawtooth(
    amplitude: float,
    frequency_hz: float,
    phase_rad: float,
    sampling_rate_hz: float,
    duration_sec: float,
    dc_offset: float = 0.0
) -> SignalContainer:
    """Generate linear sawtooth ramp wave.

    Args:
        amplitude: Peak amplitude A.
        frequency_hz: Fundamental frequency f in Hz (>= 0).
        phase_rad: Initial phase in radians.
        sampling_rate_hz: Sampling frequency Fs in Hz (> 0).
        duration_sec: Duration in seconds (> 0).
        dc_offset: Constant DC bias offset.

    Returns:
        SignalContainer with time vector and amplitude values.
    """
    _validate_basic_params(amplitude, sampling_rate_hz, duration_sec)
    _validate_frequency(frequency_hz)

    t = create_time_vector(sampling_rate_hz, duration_sec)
    saw = scipy.signal.sawtooth(2.0 * np.pi * frequency_hz * t + phase_rad, width=1.0)
    x = amplitude * saw + dc_offset

    metadata = {
        "amplitude": amplitude,
        "frequency_hz": frequency_hz,
        "phase_rad": phase_rad,
        "dc_offset": dc_offset,
    }
    return SignalContainer(
        time_vector=t,
        amplitude=x,
        sampling_rate_hz=sampling_rate_hz,
        duration_sec=duration_sec,
        signal_type="sawtooth",
        metadata=metadata,
    )


def generate_multitone(
    frequencies_hz: List[float],
    amplitudes: List[float],
    phases_rad: List[float],
    sampling_rate_hz: float,
    duration_sec: float,
    dc_offset: float = 0.0
) -> SignalContainer:
    """Generate multi-tone sinusoidal sum x(t) = sum(A_k * sin(2*pi*f_k*t + phase_k)) + dc_offset.

    Args:
        frequencies_hz: List of frequencies f_k in Hz (each >= 0).
        amplitudes: List of peak amplitudes A_k.
        phases_rad: List of initial phases in radians.
        sampling_rate_hz: Sampling frequency Fs in Hz (> 0).
        duration_sec: Duration in seconds (> 0).
        dc_offset: Constant DC bias offset.

    Returns:
        SignalContainer with combined multi-tone time vector and amplitude values.
    """
    if sampling_rate_hz <= 0 or duration_sec <= 0:
        _validate_basic_params(1.0, sampling_rate_hz, duration_sec)

    if not (len(frequencies_hz) == len(amplitudes) == len(phases_rad)):
        raise ValueError(
            f"Multi-tone parameters length mismatch: frequencies ({len(frequencies_hz)}), "
            f"amplitudes ({len(amplitudes)}), phases ({len(phases_rad)}) must be equal."
        )

    if len(frequencies_hz) == 0:
        raise ValueError("Multi-tone generation requires at least one tone component.")

    for i, f in enumerate(frequencies_hz):
        _validate_frequency(f, f"frequencies_hz[{i}]")

    for i, a in enumerate(amplitudes):
        if not np.isfinite(a):
            raise ValueError(f"amplitudes[{i}] must be finite, got {a}")

    t = create_time_vector(sampling_rate_hz, duration_sec)
    x = np.full_like(t, fill_value=dc_offset, dtype=np.float64)

    for f, a, p in zip(frequencies_hz, amplitudes, phases_rad):
        x += a * np.sin(2.0 * np.pi * f * t + p)

    metadata = {
        "frequencies_hz": list(frequencies_hz),
        "amplitudes": list(amplitudes),
        "phases_rad": list(phases_rad),
        "dc_offset": dc_offset,
    }
    return SignalContainer(
        time_vector=t,
        amplitude=x,
        sampling_rate_hz=sampling_rate_hz,
        duration_sec=duration_sec,
        signal_type="multi_tone",
        metadata=metadata,
    )


def generate_chirp(
    amplitude: float,
    f_start_hz: float,
    f_end_hz: float,
    duration_sec: float,
    sampling_rate_hz: float,
    phase_rad: float = 0.0,
    dc_offset: float = 0.0
) -> SignalContainer:
    """Generate linear frequency modulated (LFM) chirp.

    Instantaneous frequency: f(t) = f_start + ((f_end - f_start) / duration) * t
    Instantaneous phase: phi(t) = 2*pi * (f_start * t + 0.5 * (f_end - f_start) / duration * t^2) + phase_rad
    Signal: x(t) = A * cos(phi(t)) + dc_offset

    Args:
        amplitude: Peak amplitude A.
        f_start_hz: Start frequency f0 in Hz (>= 0).
        f_end_hz: End frequency f1 in Hz (>= 0).
        duration_sec: Duration in seconds (> 0).
        sampling_rate_hz: Sampling frequency Fs in Hz (> 0).
        phase_rad: Initial phase in radians.
        dc_offset: Constant DC bias offset.

    Returns:
        SignalContainer with time vector and chirp amplitude values.
    """
    _validate_basic_params(amplitude, sampling_rate_hz, duration_sec)
    _validate_frequency(f_start_hz, "f_start_hz")
    _validate_frequency(f_end_hz, "f_end_hz")

    t = create_time_vector(sampling_rate_hz, duration_sec)
    chirp_slope = (f_end_hz - f_start_hz) / float(duration_sec)
    instantaneous_phase = 2.0 * np.pi * (f_start_hz * t + 0.5 * chirp_slope * (t ** 2)) + phase_rad
    x = amplitude * np.cos(instantaneous_phase) + dc_offset

    metadata = {
        "amplitude": amplitude,
        "f_start_hz": f_start_hz,
        "f_end_hz": f_end_hz,
        "chirp_slope_hz_per_sec": chirp_slope,
        "phase_rad": phase_rad,
        "dc_offset": dc_offset,
    }
    return SignalContainer(
        time_vector=t,
        amplitude=x,
        sampling_rate_hz=sampling_rate_hz,
        duration_sec=duration_sec,
        signal_type="chirp",
        metadata=metadata,
    )


def generate_noise(
    std_dev: float,
    sampling_rate_hz: float,
    duration_sec: float,
    seed: Optional[int] = None,
    dc_offset: float = 0.0
) -> SignalContainer:
    """Generate Gaussian (white) noise with specified standard deviation and optional seed.

    x(t) ~ N(dc_offset, std_dev^2)

    Args:
        std_dev: Standard deviation sigma (>= 0).
        sampling_rate_hz: Sampling frequency Fs in Hz (> 0).
        duration_sec: Duration in seconds (> 0).
        seed: Random seed for deterministic reproducibility.
        dc_offset: Mean / constant DC bias offset.

    Returns:
        SignalContainer with time vector and noise sample values.
    """
    _validate_basic_params(1.0, sampling_rate_hz, duration_sec)
    if not np.isfinite(std_dev) or std_dev < 0:
        raise ValueError(f"Standard deviation std_dev must be non-negative (>= 0), got {std_dev}")

    t = create_time_vector(sampling_rate_hz, duration_sec)
    rng = np.random.default_rng(seed)
    raw_noise = rng.normal(loc=0.0, scale=std_dev, size=len(t))
    x = raw_noise + dc_offset

    metadata = {
        "std_dev": std_dev,
        "seed": seed,
        "dc_offset": dc_offset,
    }
    return SignalContainer(
        time_vector=t,
        amplitude=x,
        sampling_rate_hz=sampling_rate_hz,
        duration_sec=duration_sec,
        signal_type="noise",
        metadata=metadata,
    )


def generate_am(
    amplitude: float,
    carrier_frequency_hz: float,
    modulation_frequency_hz: float,
    modulation_index: float,
    sampling_rate_hz: float,
    duration_sec: float,
    phase_rad: float = 0.0,
    dc_offset: float = 0.0
) -> SignalContainer:
    """Generate Amplitude Modulated (AM) signal.

    Equation:
        x(t) = A_c * [1 + m * sin(2*pi*f_m*t)] * cos(2*pi*f_c*t + phase) + dc_offset

    Args:
        amplitude: Carrier peak amplitude A_c.
        carrier_frequency_hz: Carrier frequency f_c in Hz (>= 0).
        modulation_frequency_hz: Modulating signal frequency f_m in Hz (>= 0).
        modulation_index: AM modulation index m (>= 0).
        sampling_rate_hz: Sampling frequency Fs in Hz (> 0).
        duration_sec: Duration in seconds (> 0).
        phase_rad: Initial carrier phase in radians.
        dc_offset: Constant DC bias offset.

    Returns:
        SignalContainer with time vector and AM signal values.
    """
    _validate_basic_params(amplitude, sampling_rate_hz, duration_sec)
    _validate_frequency(carrier_frequency_hz, "carrier_frequency_hz")
    _validate_frequency(modulation_frequency_hz, "modulation_frequency_hz")

    if not np.isfinite(modulation_index) or modulation_index < 0:
        raise ValueError(f"AM modulation index must be non-negative (>= 0), got {modulation_index}")

    t = create_time_vector(sampling_rate_hz, duration_sec)
    modulating_signal = np.sin(2.0 * np.pi * modulation_frequency_hz * t)
    carrier_signal = np.cos(2.0 * np.pi * carrier_frequency_hz * t + phase_rad)

    envelope = amplitude * (1.0 + modulation_index * modulating_signal)
    x = envelope * carrier_signal + dc_offset

    metadata = {
        "amplitude": amplitude,
        "carrier_frequency_hz": carrier_frequency_hz,
        "modulation_frequency_hz": modulation_frequency_hz,
        "modulation_index": modulation_index,
        "phase_rad": phase_rad,
        "dc_offset": dc_offset,
    }
    return SignalContainer(
        time_vector=t,
        amplitude=x,
        sampling_rate_hz=sampling_rate_hz,
        duration_sec=duration_sec,
        signal_type="am",
        metadata=metadata,
    )


def generate_fm(
    amplitude: float,
    carrier_frequency_hz: float,
    modulation_frequency_hz: float,
    frequency_deviation_hz: float,
    sampling_rate_hz: float,
    duration_sec: float,
    phase_rad: float = 0.0,
    dc_offset: float = 0.0
) -> SignalContainer:
    """Generate Frequency Modulated (FM) signal.

    Modulating signal: m(t) = cos(2*pi*f_m*t)
    Instantaneous frequency: f(t) = f_c + delta_f * cos(2*pi*f_m*t)
    FM Modulation index: beta = delta_f / f_m
    Phase formulation: phi(t) = 2*pi*f_c*t + beta * sin(2*pi*f_m*t) + phase_rad
    Signal: x(t) = A_c * cos(phi(t)) + dc_offset

    Args:
        amplitude: Carrier peak amplitude A_c.
        carrier_frequency_hz: Carrier frequency f_c in Hz (>= 0).
        modulation_frequency_hz: Modulating signal frequency f_m in Hz (>= 0).
        frequency_deviation_hz: Frequency deviation delta_f in Hz (>= 0).
        sampling_rate_hz: Sampling frequency Fs in Hz (> 0).
        duration_sec: Duration in seconds (> 0).
        phase_rad: Initial carrier phase in radians.
        dc_offset: Constant DC bias offset.

    Returns:
        SignalContainer with time vector and FM signal values.
    """
    _validate_basic_params(amplitude, sampling_rate_hz, duration_sec)
    _validate_frequency(carrier_frequency_hz, "carrier_frequency_hz")
    _validate_frequency(modulation_frequency_hz, "modulation_frequency_hz")

    if not np.isfinite(frequency_deviation_hz) or frequency_deviation_hz < 0:
        raise ValueError(f"FM frequency deviation must be non-negative (>= 0), got {frequency_deviation_hz}")

    if modulation_frequency_hz == 0.0 and frequency_deviation_hz > 0.0:
        raise ValueError("Modulation frequency f_m cannot be zero when frequency deviation delta_f > 0.")

    t = create_time_vector(sampling_rate_hz, duration_sec)

    if modulation_frequency_hz == 0.0 or frequency_deviation_hz == 0.0:
        instantaneous_phase = 2.0 * np.pi * carrier_frequency_hz * t + phase_rad
        beta = 0.0
    else:
        beta = frequency_deviation_hz / modulation_frequency_hz
        instantaneous_phase = 2.0 * np.pi * carrier_frequency_hz * t + beta * np.sin(2.0 * np.pi * modulation_frequency_hz * t) + phase_rad

    x = amplitude * np.cos(instantaneous_phase) + dc_offset

    metadata = {
        "amplitude": amplitude,
        "carrier_frequency_hz": carrier_frequency_hz,
        "modulation_frequency_hz": modulation_frequency_hz,
        "frequency_deviation_hz": frequency_deviation_hz,
        "fm_modulation_index_beta": beta,
        "phase_rad": phase_rad,
        "dc_offset": dc_offset,
    }
    return SignalContainer(
        time_vector=t,
        amplitude=x,
        sampling_rate_hz=sampling_rate_hz,
        duration_sec=duration_sec,
        signal_type="fm",
        metadata=metadata,
    )


def generate_signal(config: SignalGenConfig) -> SignalContainer:
    """Generate time-domain signal according to SignalGenConfig parameters.

    Acts as the primary unified dispatcher for all signal types.

    Args:
        config: Signal generation configuration object.

    Returns:
        SignalContainer with time vector, amplitude, and metadata.

    Raises:
        ValueError: If config parameters or signal_type are invalid.
    """
    stype = config.signal_type.lower().strip()

    if stype == "sine":
        return generate_sine(
            amplitude=config.amplitude,
            frequency_hz=config.frequency_hz,
            phase_rad=config.phase_rad,
            sampling_rate_hz=config.sampling_rate_hz,
            duration_sec=config.duration_sec,
            dc_offset=config.dc_offset,
        )
    elif stype == "cosine":
        return generate_cosine(
            amplitude=config.amplitude,
            frequency_hz=config.frequency_hz,
            phase_rad=config.phase_rad,
            sampling_rate_hz=config.sampling_rate_hz,
            duration_sec=config.duration_sec,
            dc_offset=config.dc_offset,
        )
    elif stype == "square":
        return generate_square(
            amplitude=config.amplitude,
            frequency_hz=config.frequency_hz,
            phase_rad=config.phase_rad,
            duty_cycle=config.duty_cycle,
            sampling_rate_hz=config.sampling_rate_hz,
            duration_sec=config.duration_sec,
            dc_offset=config.dc_offset,
        )
    elif stype == "triangle":
        return generate_triangle(
            amplitude=config.amplitude,
            frequency_hz=config.frequency_hz,
            phase_rad=config.phase_rad,
            sampling_rate_hz=config.sampling_rate_hz,
            duration_sec=config.duration_sec,
            dc_offset=config.dc_offset,
        )
    elif stype == "sawtooth":
        return generate_sawtooth(
            amplitude=config.amplitude,
            frequency_hz=config.frequency_hz,
            phase_rad=config.phase_rad,
            sampling_rate_hz=config.sampling_rate_hz,
            duration_sec=config.duration_sec,
            dc_offset=config.dc_offset,
        )
    elif stype == "multi_tone":
        return generate_multitone(
            frequencies_hz=config.frequencies_hz,
            amplitudes=config.amplitudes,
            phases_rad=config.phases_rad,
            sampling_rate_hz=config.sampling_rate_hz,
            duration_sec=config.duration_sec,
            dc_offset=config.dc_offset,
        )
    elif stype == "chirp":
        return generate_chirp(
            amplitude=config.amplitude,
            f_start_hz=config.f_start_hz,
            f_end_hz=config.f_end_hz,
            duration_sec=config.duration_sec,
            sampling_rate_hz=config.sampling_rate_hz,
            phase_rad=config.phase_rad,
            dc_offset=config.dc_offset,
        )
    elif stype == "noise":
        return generate_noise(
            std_dev=config.std_dev,
            sampling_rate_hz=config.sampling_rate_hz,
            duration_sec=config.duration_sec,
            seed=config.seed,
            dc_offset=config.dc_offset,
        )
    elif stype == "am":
        return generate_am(
            amplitude=config.amplitude,
            carrier_frequency_hz=config.carrier_frequency_hz,
            modulation_frequency_hz=config.modulation_frequency_hz,
            modulation_index=config.modulation_index,
            sampling_rate_hz=config.sampling_rate_hz,
            duration_sec=config.duration_sec,
            phase_rad=config.phase_rad,
            dc_offset=config.dc_offset,
        )
    elif stype == "fm":
        return generate_fm(
            amplitude=config.amplitude,
            carrier_frequency_hz=config.carrier_frequency_hz,
            modulation_frequency_hz=config.modulation_frequency_hz,
            frequency_deviation_hz=config.frequency_deviation_hz,
            sampling_rate_hz=config.sampling_rate_hz,
            duration_sec=config.duration_sec,
            phase_rad=config.phase_rad,
            dc_offset=config.dc_offset,
        )
    else:
        raise ValueError(
            f"Unsupported signal_type '{config.signal_type}'. Supported types: "
            "sine, cosine, square, triangle, sawtooth, multi_tone, chirp, noise, am, fm."
        )
