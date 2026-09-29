"""Comprehensive unit tests for the Multirate DSP Engine (src/dsp/resampling.py)."""

import numpy as np
import pytest
from src.dsp.resampling import (
    DecimationAliasingAnalysis,
    InterpolationImageAnalysis,
    ResamplingResult,
    analyze_decimation_aliasing,
    analyze_interpolation_images,
    decimate_signal,
    downsample_naive,
    find_rational_resampling_ratio,
    interpolate_signal,
    resample_rational,
    upsample_zero_insertion,
)
from src.dsp.signals import SignalContainer, generate_multitone, generate_sine
from src.dsp.transforms import compute_fft, find_dominant_frequency


def test_find_rational_resampling_ratio() -> None:
    """Test 1: Verify exact integer rational ratio extraction for sampling rates."""
    L, M = find_rational_resampling_ratio(10000.0, 15000.0)
    assert (L, M) == (3, 2)

    L, M = find_rational_resampling_ratio(8000.0, 12000.0)
    assert (L, M) == (3, 2)

    L, M = find_rational_resampling_ratio(44100.0, 48000.0)
    assert (L, M) == (160, 147)


def test_decimation_output_rate_and_length() -> None:
    """Test 2: Verify decimation by M=2 output sampling rate and sample count."""
    fs_in = 10000.0
    sig = generate_sine(amplitude=1.0, frequency_hz=500.0, phase_rad=0.0, sampling_rate_hz=fs_in, duration_sec=1.0)

    res = decimate_signal(sig, factor_M=2)

    assert isinstance(res, ResamplingResult)
    assert res.input_sampling_rate_hz == 10000.0
    assert res.output_sampling_rate_hz == 5000.0
    assert res.input_num_samples == 10000
    assert res.output_num_samples == 5000
    assert res.effective_duration_sec == pytest.approx(1.0, abs=1e-5)
    assert res.processed_signal.sampling_rate_hz == 5000.0
    assert len(res.processed_signal.time_vector) == 5000


def test_decimation_anti_aliasing_suppression() -> None:
    """Test 3: Verify decimation by M=2 retains 500 Hz tone and suppresses 3000 Hz and 4500 Hz."""
    fs_in = 10000.0
    sig = generate_multitone(
        frequencies_hz=[500.0, 3000.0, 4500.0],
        amplitudes=[1.0, 1.0, 1.0],
        phases_rad=[0.0, 0.0, 0.0],
        sampling_rate_hz=fs_in,
        duration_sec=2.0
    )

    res = decimate_signal(sig, factor_M=2, method="polyphase")
    spec = compute_fft(res.processed_signal, window_type="rect", one_sided=True)

    idx_500 = int(np.round(500.0 / spec.bin_resolution_hz))
    idx_2000 = int(np.round(2000.0 / spec.bin_resolution_hz))  # Where 3000 Hz would alias

    # 500 Hz passes
    assert spec.magnitude[idx_500] == pytest.approx(1.0, abs=0.05)
    # Stopband components (> 2500 Hz) suppressed
    assert spec.magnitude[idx_2000] < 0.05


def test_naive_downsampling_aliasing_demonstration() -> None:
    """Test 4: Verify naive downsampling (x[::2]) causes 3000 Hz tone to alias to 2000 Hz."""
    fs_in = 10000.0
    sig = generate_multitone(
        frequencies_hz=[500.0, 3000.0],
        amplitudes=[1.0, 1.0],
        phases_rad=[0.0, 0.0],
        sampling_rate_hz=fs_in,
        duration_sec=1.0
    )

    naive = downsample_naive(sig, factor_M=2)
    spec_naive = compute_fft(naive, window_type="rect", one_sided=True)

    idx_500 = int(np.round(500.0 / spec_naive.bin_resolution_hz))
    idx_2000 = int(np.round(2000.0 / spec_naive.bin_resolution_hz))

    # Naive downsampling folds 3000 Hz tone into 2000 Hz with full amplitude (~1.0 V)
    assert spec_naive.magnitude[idx_500] == pytest.approx(1.0, abs=0.05)
    assert spec_naive.magnitude[idx_2000] == pytest.approx(1.0, abs=0.05)


def test_interpolation_output_rate_and_length() -> None:
    """Test 5: Verify interpolation by L=2 output sampling rate and sample count."""
    fs_in = 5000.0
    sig = generate_sine(amplitude=1.0, frequency_hz=500.0, phase_rad=0.0, sampling_rate_hz=fs_in, duration_sec=1.0)

    res = interpolate_signal(sig, factor_L=2)

    assert res.input_sampling_rate_hz == 5000.0
    assert res.output_sampling_rate_hz == 10000.0
    assert res.input_num_samples == 5000
    assert res.output_num_samples == 10000
    assert res.processed_signal.sampling_rate_hz == 10000.0
    assert len(res.processed_signal.time_vector) == 10000


def test_interpolation_tone_preservation_and_image_suppression() -> None:
    """Test 6: Verify interpolation by L=2 preserves physical tone and suppresses 4500 Hz image."""
    fs_in = 5000.0
    sig = generate_sine(amplitude=1.0, frequency_hz=500.0, phase_rad=0.0, sampling_rate_hz=fs_in, duration_sec=2.0)

    res = interpolate_signal(sig, factor_L=2)
    spec = compute_fft(res.processed_signal, window_type="rect", one_sided=True)

    idx_500 = int(np.round(500.0 / spec.bin_resolution_hz))
    idx_4500 = int(np.round(4500.0 / spec.bin_resolution_hz))  # Spectral image (5000 - 500)

    # Physical 500 Hz tone preserved at 1.0 V
    assert spec.magnitude[idx_500] == pytest.approx(1.0, abs=0.05)
    # Image at 4500 Hz suppressed
    assert spec.magnitude[idx_4500] < 0.05


def test_zero_insertion_spectral_images_demonstration() -> None:
    """Test 7: Verify raw zero insertion retains un-filtered spectral image at 4500 Hz."""
    fs_in = 5000.0
    sig = generate_sine(amplitude=1.0, frequency_hz=500.0, phase_rad=0.0, sampling_rate_hz=fs_in, duration_sec=1.0)

    zero_up = upsample_zero_insertion(sig, factor_L=2)
    spec = compute_fft(zero_up, window_type="rect", one_sided=True)

    idx_500 = int(np.round(500.0 / spec.bin_resolution_hz))
    idx_4500 = int(np.round(4500.0 / spec.bin_resolution_hz))

    # Raw zero insertion splits energy equally between 500 Hz and 4500 Hz (0.5 V each before gain factor L=2)
    assert spec.magnitude[idx_500] == pytest.approx(0.5, abs=0.05)
    assert spec.magnitude[idx_4500] == pytest.approx(0.5, abs=0.05)


def test_rational_resampling_3_2() -> None:
    """Test 8: Verify 10000 Hz -> 15000 Hz (3/2) rational resampling preserves physical frequencies."""
    fs_in = 10000.0
    fs_out = 15000.0
    sig = generate_multitone(
        frequencies_hz=[500.0, 1500.0],
        amplitudes=[1.0, 1.0],
        phases_rad=[0.0, 0.0],
        sampling_rate_hz=fs_in,
        duration_sec=2.0
    )

    res = resample_rational(sig, target_sampling_rate_hz=fs_out)

    assert res.input_sampling_rate_hz == 10000.0
    assert res.output_sampling_rate_hz == 15000.0
    assert res.up_factor == 3
    assert res.down_factor == 2

    # Measure output FFT
    spec = compute_fft(res.processed_signal, window_type="rect", one_sided=True)
    idx_500 = int(np.round(500.0 / spec.bin_resolution_hz))
    idx_1500 = int(np.round(1500.0 / spec.bin_resolution_hz))

    assert spec.magnitude[idx_500] == pytest.approx(1.0, abs=0.05)
    assert spec.magnitude[idx_1500] == pytest.approx(1.0, abs=0.05)


def test_analyze_decimation_aliasing() -> None:
    """Test 9: Verify educational decimation aliasing analysis helper."""
    analysis = analyze_decimation_aliasing(
        signal_frequencies_hz=[500.0, 3000.0, 4500.0],
        fs_in_hz=10000.0,
        decimation_factor_M=2
    )

    assert isinstance(analysis, DecimationAliasingAnalysis)
    assert analysis.output_sampling_rate_hz == 5000.0
    assert analysis.output_nyquist_hz == 2500.0
    assert analysis.surviving_passband_frequencies_hz == [500.0]
    assert analysis.aliasing_components_hz == [3000.0, 4500.0]
    assert analysis.theoretical_aliased_frequencies_hz == [2000.0, 500.0]


def test_analyze_interpolation_images() -> None:
    """Test 10: Verify educational interpolation image analysis helper."""
    analysis = analyze_interpolation_images(
        signal_frequencies_hz=[500.0],
        fs_in_hz=5000.0,
        interpolation_factor_L=2
    )

    assert isinstance(analysis, InterpolationImageAnalysis)
    assert analysis.output_sampling_rate_hz == 10000.0
    assert analysis.output_nyquist_hz == 5000.0
    assert analysis.replicated_image_frequencies_hz == [4500.0]


def test_signal_container_compatibility() -> None:
    """Test 11: Verify SignalContainer metadata, duration, and time vector consistency."""
    sig = generate_sine(amplitude=2.0, frequency_hz=200.0, phase_rad=0.0, sampling_rate_hz=10000.0, duration_sec=1.0)
    res = decimate_signal(sig, factor_M=2)

    processed = res.processed_signal
    assert isinstance(processed, SignalContainer)
    assert processed.sampling_rate_hz == 5000.0
    assert len(processed.amplitude) == len(processed.time_vector) == 5000
    # Time vector endpoint excluded convention: t[last] < duration
    assert processed.time_vector[-1] < processed.duration_sec
    assert processed.metadata["resampling_method"].startswith("decimate_M2")


def test_resampling_edge_cases_and_validations() -> None:
    """Test 12: Verify ValueError / TypeError exceptions on invalid inputs."""
    sig = generate_sine(amplitude=1.0, frequency_hz=100.0, phase_rad=0.0, sampling_rate_hz=10000.0, duration_sec=0.1)

    with pytest.raises(TypeError, match="integer"):
        decimate_signal(sig, factor_M=2.5)  # type: ignore

    with pytest.raises(ValueError, match="at least 1"):
        decimate_signal(sig, factor_M=0)

    with pytest.raises(TypeError, match="integer"):
        interpolate_signal(sig, factor_L=1.5)  # type: ignore

    with pytest.raises(ValueError, match="at least 1"):
        interpolate_signal(sig, factor_L=-2)

    with pytest.raises(ValueError, match="strictly positive"):
        resample_rational(sig, target_sampling_rate_hz=-100.0)

    with pytest.raises(ValueError, match="non-finite"):
        decimate_signal(np.array([1.0, np.nan, 2.0, 3.0]), factor_M=2)

    with pytest.raises(ValueError, match="at least 2 samples"):
        decimate_signal(np.array([1.0]), factor_M=2)
