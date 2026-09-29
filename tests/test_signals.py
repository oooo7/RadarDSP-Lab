"""Comprehensive unit tests for the DSP Signal Generation Engine (src/dsp/signals.py)."""

import numpy as np
import pytest
from src.dsp.signals import (
    SignalContainer,
    create_time_vector,
    generate_am,
    generate_chirp,
    generate_cosine,
    generate_fm,
    generate_multitone,
    generate_noise,
    generate_sawtooth,
    generate_signal,
    generate_sine,
    generate_square,
    generate_triangle,
)
from src.utils.config import SignalGenConfig


def test_time_vector_length() -> None:
    """Test 1: Verify discrete time vector sample count computation."""
    t1 = create_time_vector(sampling_rate_hz=1000.0, duration_sec=1.0)
    assert len(t1) == 1000

    t2 = create_time_vector(sampling_rate_hz=500.0, duration_sec=0.5)
    assert len(t2) == 250

    t3 = create_time_vector(sampling_rate_hz=100.0, duration_sec=2.5)
    assert len(t3) == 250


def test_time_vector_spacing() -> None:
    """Test 2: Verify time vector sample spacing dt = 1/Fs and endpoint exclusion."""
    fs = 1000.0
    duration = 1.0
    t = create_time_vector(sampling_rate_hz=fs, duration_sec=duration)

    # Sample spacing must equal 1/Fs
    dt = t[1] - t[0]
    assert dt == pytest.approx(1.0 / fs, abs=1e-12)

    # Start time must be 0.0
    assert t[0] == 0.0

    # Endpoint t = duration (1.0s) must be EXCLUDED. Last sample is at (N-1)/Fs = 0.999s.
    assert t[-1] == pytest.approx(0.999, abs=1e-12)
    assert t[-1] < duration


def test_sine_amplitude() -> None:
    """Test 3: Verify peak amplitude of generated sine wave."""
    amp = 3.5
    sig = generate_sine(
        amplitude=amp,
        frequency_hz=10.0,
        phase_rad=0.0,
        sampling_rate_hz=10000.0,
        duration_sec=1.0
    )
    assert np.max(sig.amplitude) == pytest.approx(amp, abs=1e-4)
    assert np.min(sig.amplitude) == pytest.approx(-amp, abs=1e-4)


def test_sine_frequency() -> None:
    """Test 4: Verify sine wave values at exact mathematical timestamp quarter-cycles."""
    f = 25.0
    fs = 1000.0
    sig = generate_sine(
        amplitude=1.0,
        frequency_hz=f,
        phase_rad=0.0,
        sampling_rate_hz=fs,
        duration_sec=1.0
    )
    # At t = 1/(4f) = 0.01s (sample index n = t*Fs = 10), sin(2*pi*25*0.01) = sin(pi/2) = 1.0
    sample_idx = int(np.round(fs / (4.0 * f)))
    assert sig.amplitude[sample_idx] == pytest.approx(1.0, abs=1e-6)


def test_sine_phase() -> None:
    """Test 5: Verify initial phase offset on sine wave."""
    # Phase pi/2 turns sine into cosine: sin(0 + pi/2) = 1.0
    sig = generate_sine(
        amplitude=2.0,
        frequency_hz=10.0,
        phase_rad=np.pi / 2.0,
        sampling_rate_hz=1000.0,
        duration_sec=1.0
    )
    assert sig.amplitude[0] == pytest.approx(2.0, abs=1e-6)


def test_cosine_generation() -> None:
    """Test 6: Verify cosine signal generation at t=0."""
    sig = generate_cosine(
        amplitude=1.5,
        frequency_hz=50.0,
        phase_rad=0.0,
        sampling_rate_hz=1000.0,
        duration_sec=1.0
    )
    assert sig.amplitude[0] == pytest.approx(1.5, abs=1e-6)
    assert sig.signal_type == "cosine"


def test_square_wave_behavior() -> None:
    """Test 7: Verify square wave peak states and duty cycle ratio."""
    fs = 1000.0
    duty = 0.25
    sig = generate_square(
        amplitude=1.0,
        frequency_hz=1.0,
        phase_rad=0.0,
        duty_cycle=duty,
        sampling_rate_hz=fs,
        duration_sec=1.0
    )
    # Square wave should take values +1.0 or -1.0
    assert np.all(np.isin(np.round(sig.amplitude, 4), [1.0, -1.0]))

    # For 1 Hz wave over 1 second with 25% duty cycle, approx 25% of samples are positive
    num_positive = np.sum(sig.amplitude > 0)
    assert num_positive / len(sig.amplitude) == pytest.approx(duty, abs=0.01)


def test_triangle_wave_behavior() -> None:
    """Test 8: Verify symmetric triangle wave min, max, and zero-crossing bounds."""
    sig = generate_triangle(
        amplitude=2.0,
        frequency_hz=10.0,
        phase_rad=0.0,
        sampling_rate_hz=10000.0,
        duration_sec=1.0
    )
    assert np.max(sig.amplitude) == pytest.approx(2.0, abs=1e-3)
    assert np.min(sig.amplitude) == pytest.approx(-2.0, abs=1e-3)
    assert sig.signal_type == "triangle"


def test_sawtooth_behavior() -> None:
    """Test 9: Verify sawtooth linear ramp generation."""
    sig = generate_sawtooth(
        amplitude=1.0,
        frequency_hz=1.0,
        phase_rad=0.0,
        sampling_rate_hz=1000.0,
        duration_sec=1.0
    )
    # At t=0.999s (last sample), discrete sawtooth value is 1.0 - 2/Fs = 0.998
    assert np.max(sig.amplitude) == pytest.approx(1.0, abs=5e-3)
    assert np.min(sig.amplitude) == pytest.approx(-1.0, abs=5e-3)
    assert sig.signal_type == "sawtooth"


def test_multitone_generation() -> None:
    """Test 10: Verify multi-tone generation equals sum of individual sinusoids."""
    freqs = [10.0, 50.0, 120.0]
    amps = [1.0, 0.5, 0.25]
    phases = [0.0, np.pi / 4.0, 0.0]
    fs = 1000.0
    duration = 1.0

    multi_sig = generate_multitone(
        frequencies_hz=freqs,
        amplitudes=amps,
        phases_rad=phases,
        sampling_rate_hz=fs,
        duration_sec=duration
    )

    # Compute expected manual sum
    t = multi_sig.time_vector
    expected = (
        amps[0] * np.sin(2.0 * np.pi * freqs[0] * t + phases[0])
        + amps[1] * np.sin(2.0 * np.pi * freqs[1] * t + phases[1])
        + amps[2] * np.sin(2.0 * np.pi * freqs[2] * t + phases[2])
    )

    np.testing.assert_allclose(multi_sig.amplitude, expected, atol=1e-12)


def test_chirp_frequency_behavior() -> None:
    """Test 11: Verify LFM chirp instantaneous phase and boundary values."""
    f0 = 10.0
    f1 = 100.0
    duration = 2.0
    amp = 1.0
    sig = generate_chirp(
        amplitude=amp,
        f_start_hz=f0,
        f_end_hz=f1,
        duration_sec=duration,
        sampling_rate_hz=10000.0,
        phase_rad=0.0
    )

    # At t=0, phase is 0 => cos(0) = 1.0
    assert sig.amplitude[0] == pytest.approx(amp, abs=1e-6)
    assert sig.signal_type == "chirp"
    assert sig.metadata["chirp_slope_hz_per_sec"] == pytest.approx((f1 - f0) / duration, abs=1e-9)


def test_noise_reproducibility() -> None:
    """Test 12: Verify Gaussian noise determinism with fixed seed vs random seed."""
    sig1 = generate_noise(std_dev=1.0, sampling_rate_hz=1000.0, duration_sec=1.0, seed=42)
    sig2 = generate_noise(std_dev=1.0, sampling_rate_hz=1000.0, duration_sec=1.0, seed=42)
    sig3 = generate_noise(std_dev=1.0, sampling_rate_hz=1000.0, duration_sec=1.0, seed=99)

    # Same seed must yield identical arrays
    np.testing.assert_array_equal(sig1.amplitude, sig2.amplitude)

    # Different seed must yield different arrays
    assert not np.array_equal(sig1.amplitude, sig3.amplitude)


def test_am_generation() -> None:
    """Test 13: Verify AM signal envelope modulation bounds."""
    ac = 2.0
    m = 0.5
    fc = 100.0
    fm = 5.0

    sig = generate_am(
        amplitude=ac,
        carrier_frequency_hz=fc,
        modulation_frequency_hz=fm,
        modulation_index=m,
        sampling_rate_hz=10000.0,
        duration_sec=1.0
    )

    # Upper envelope peak = Ac * (1 + m) = 2.0 * 1.5 = 3.0
    assert np.max(sig.amplitude) == pytest.approx(ac * (1.0 + m), abs=1e-2)
    assert sig.signal_type == "am"


def test_fm_generation() -> None:
    """Test 14: Verify FM signal generation and constant carrier envelope."""
    ac = 1.5
    fc = 200.0
    fm = 10.0
    delta_f = 50.0

    sig = generate_fm(
        amplitude=ac,
        carrier_frequency_hz=fc,
        modulation_frequency_hz=fm,
        frequency_deviation_hz=delta_f,
        sampling_rate_hz=10000.0,
        duration_sec=1.0
    )

    # FM peak envelope remains constant at Ac
    assert np.max(sig.amplitude) == pytest.approx(ac, abs=1e-4)
    assert np.min(sig.amplitude) == pytest.approx(-ac, abs=1e-4)
    assert sig.metadata["fm_modulation_index_beta"] == pytest.approx(delta_f / fm, abs=1e-9)


def test_invalid_sampling_frequency() -> None:
    """Test 15: Verify ValueError on Fs <= 0."""
    with pytest.raises(ValueError, match="Sampling rate Fs must be strictly positive"):
        create_time_vector(sampling_rate_hz=0.0, duration_sec=1.0)

    with pytest.raises(ValueError, match="Sampling rate Fs must be strictly positive"):
        generate_sine(amplitude=1.0, frequency_hz=10.0, phase_rad=0.0, sampling_rate_hz=-100.0, duration_sec=1.0)


def test_invalid_duration() -> None:
    """Test 16: Verify ValueError on duration <= 0."""
    with pytest.raises(ValueError, match="Duration must be strictly positive"):
        create_time_vector(sampling_rate_hz=1000.0, duration_sec=-0.5)

    with pytest.raises(ValueError, match="Duration must be strictly positive"):
        generate_sine(amplitude=1.0, frequency_hz=10.0, phase_rad=0.0, sampling_rate_hz=1000.0, duration_sec=0.0)


def test_invalid_frequency() -> None:
    """Test 17: Verify ValueError on negative frequency."""
    with pytest.raises(ValueError, match="must be finite and non-negative"):
        generate_sine(amplitude=1.0, frequency_hz=-50.0, phase_rad=0.0, sampling_rate_hz=1000.0, duration_sec=1.0)


def test_invalid_duty_cycle() -> None:
    """Test 18: Verify ValueError on duty cycle <= 0 or >= 1."""
    with pytest.raises(ValueError, match="Duty cycle must be strictly between 0 and 1"):
        generate_square(amplitude=1.0, frequency_hz=10.0, phase_rad=0.0, duty_cycle=0.0, sampling_rate_hz=1000.0, duration_sec=1.0)

    with pytest.raises(ValueError, match="Duty cycle must be strictly between 0 and 1"):
        generate_square(amplitude=1.0, frequency_hz=10.0, phase_rad=0.0, duty_cycle=1.2, sampling_rate_hz=1000.0, duration_sec=1.0)


def test_multitone_array_length_mismatch() -> None:
    """Test 19: Verify ValueError on mismatched multi-tone parameter lengths."""
    with pytest.raises(ValueError, match="Multi-tone parameters length mismatch"):
        generate_multitone(
            frequencies_hz=[10.0, 20.0],
            amplitudes=[1.0],
            phases_rad=[0.0, 0.0],
            sampling_rate_hz=1000.0,
            duration_sec=1.0
        )


def test_invalid_modulation_parameters() -> None:
    """Test 20: Verify ValueError on negative AM modulation index and negative FM deviation."""
    with pytest.raises(ValueError, match="AM modulation index must be non-negative"):
        generate_am(
            amplitude=1.0,
            carrier_frequency_hz=100.0,
            modulation_frequency_hz=10.0,
            modulation_index=-0.5,
            sampling_rate_hz=1000.0,
            duration_sec=1.0
        )

    with pytest.raises(ValueError, match="FM frequency deviation must be non-negative"):
        generate_fm(
            amplitude=1.0,
            carrier_frequency_hz=100.0,
            modulation_frequency_hz=10.0,
            frequency_deviation_hz=-10.0,
            sampling_rate_hz=1000.0,
            duration_sec=1.0
        )


def test_unified_generate_signal_dispatcher() -> None:
    """Test unified dispatcher generate_signal(config)."""
    cfg = SignalGenConfig(signal_type="sine", amplitude=2.0, frequency_hz=100.0, sampling_rate_hz=1000.0, duration_sec=1.0)
    sig = generate_signal(cfg)
    assert isinstance(sig, SignalContainer)
    assert sig.signal_type == "sine"
    assert sig.sampling_rate_hz == 1000.0
    assert len(sig.time_vector) == 1000
