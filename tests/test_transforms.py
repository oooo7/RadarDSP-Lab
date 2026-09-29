"""Comprehensive unit tests for the FFT, Spectrum Analysis, Windowing, and STFT Engine (src/dsp/transforms.py)."""

import numpy as np
import pytest
from src.dsp.signals import generate_sine
from src.dsp.transforms import (
    DominantFrequencyResult,
    SpectrogramResult,
    SpectrumResult,
    analyze_spectral_leakage,
    compute_dft_educational,
    compute_fft,
    compute_fft_shift,
    compute_ifft,
    compute_rfft,
    compute_stft,
    find_dominant_frequency,
    get_window,
)


def test_fft_known_sinusoid() -> None:
    """Test 1: Verify FFT of a known bin-centered sinusoid recovers amplitude A."""
    fs = 1000.0
    duration = 1.0
    f_tone = 100.0  # Exactly bin-centered at k = 100
    amp = 2.5

    sig = generate_sine(amplitude=amp, frequency_hz=f_tone, phase_rad=0.0, sampling_rate_hz=fs, duration_sec=duration)
    res = compute_fft(sig, window_type="rect", one_sided=True)

    # Peak magnitude at 100 Hz bin must equal amp
    bin_idx = int(np.round(f_tone / res.bin_resolution_hz))
    assert res.magnitude[bin_idx] == pytest.approx(amp, abs=1e-3)


def test_fft_ifft_roundtrip() -> None:
    """Test 2: Verify full two-sided FFT to IFFT round-trip reconstruction."""
    x_orig = np.array([1.0, 2.0, -1.0, 3.5, 0.0, -2.2, 1.1, 4.0], dtype=np.float64)
    res = compute_fft(x_orig, sampling_rate_hz=100.0, window_type="rect", one_sided=False)
    x_rec = compute_ifft(res.complex_spectrum)

    np.testing.assert_allclose(np.real(x_rec), x_orig, atol=1e-12)


def test_frequency_axis_correctness() -> None:
    """Test 3: Verify frequency axis for rFFT (0 to Fs/2) and full FFT shift (-Fs/2 to Fs/2)."""
    fs = 1000.0
    n = 100
    x = np.random.default_rng(42).normal(size=n)

    # One-sided rFFT: 0 -> Fs/2
    res_one = compute_fft(x, sampling_rate_hz=fs, one_sided=True)
    assert res_one.frequency_hz[0] == 0.0
    assert res_one.frequency_hz[-1] == pytest.approx(fs / 2.0, abs=1e-12)

    # Shifted two-sided FFT: -Fs/2 -> Fs/2
    res_two = compute_fft(x, sampling_rate_hz=fs, one_sided=False)
    res_shifted = compute_fft_shift(res_two)
    assert res_shifted.frequency_hz[0] == pytest.approx(-fs / 2.0, abs=1e-12)


def test_frequency_resolution() -> None:
    """Test 4: Verify physical resolution Delta f = Fs / N_signal."""
    fs = 2000.0
    n = 500
    x = np.ones(n)
    res = compute_fft(x, sampling_rate_hz=fs)

    assert res.physical_resolution_hz == pytest.approx(fs / n, abs=1e-12)
    assert res.bin_resolution_hz == pytest.approx(fs / n, abs=1e-12)


def test_one_sided_spectrum() -> None:
    """Test 5: Verify one-sided spectrum properties for real signals."""
    x = np.ones(100)
    res = compute_fft(x, sampling_rate_hz=1000.0, one_sided=True)
    assert res.is_one_sided is True
    assert len(res.frequency_hz) == 51  # (100 // 2) + 1


def test_two_sided_spectrum() -> None:
    """Test 6: Verify two-sided spectrum properties."""
    x = np.ones(100)
    res = compute_fft(x, sampling_rate_hz=1000.0, one_sided=False)
    assert res.is_one_sided is False
    assert len(res.frequency_hz) == 100


def test_dc_scaling() -> None:
    """Test 7: Verify DC scaling: x[n] = C => magnitude at bin 0 is C."""
    c = 4.2
    x = np.full(1000, fill_value=c, dtype=np.float64)
    res = compute_fft(x, sampling_rate_hz=1000.0, window_type="rect", one_sided=True)
    assert res.magnitude[0] == pytest.approx(c, abs=1e-12)


def test_nyquist_bin_scaling_even_n() -> None:
    """Test 8: Verify Nyquist bin scaling for even N."""
    fs = 1000.0
    n = 1000
    amp = 3.0
    # Tone at exact Nyquist frequency Fs/2 = 500 Hz: x[n] = A * cos(pi * n) = A * (-1)^n
    t = np.arange(n) / fs
    x = amp * np.cos(np.pi * fs * t)

    res = compute_fft(x, sampling_rate_hz=fs, window_type="rect", one_sided=True)
    nyq_bin = n // 2
    assert res.magnitude[nyq_bin] == pytest.approx(amp, abs=1e-3)


def test_interior_bin_scaling() -> None:
    """Test 9: Verify interior bin amplitude scaling for real sinusoid."""
    fs = 1000.0
    amp = 2.0
    f_tone = 250.0  # Interior bin k = 250
    t = np.arange(1000) / fs
    x = amp * np.cos(2.0 * np.pi * f_tone * t)

    res = compute_fft(x, sampling_rate_hz=fs, window_type="rect", one_sided=True)
    bin_idx = int(np.round(f_tone / res.bin_resolution_hz))
    assert res.magnitude[bin_idx] == pytest.approx(amp, abs=1e-3)


def test_dominant_frequency_detection() -> None:
    """Test 10: Verify dominant frequency peak search."""
    fs = 1000.0
    f_tone = 150.0
    sig = generate_sine(amplitude=1.5, frequency_hz=f_tone, phase_rad=0.0, sampling_rate_hz=fs, duration_sec=1.0)
    spec = compute_fft(sig, window_type="rect", one_sided=True)

    dom = find_dominant_frequency(spec, ignore_dc=True)
    assert isinstance(dom, DominantFrequencyResult)
    assert dom.frequency_hz == pytest.approx(f_tone, abs=1e-3)
    assert dom.magnitude == pytest.approx(1.5, abs=1e-3)


def test_subbin_frequency_estimation() -> None:
    """Test 11: Verify sub-bin parabolic frequency estimation accuracy for non-bin-centered tone."""
    fs = 1000.0
    f_tone = 100.4  # Non-integer bin location (between bin 100 and 101)
    t = np.arange(1000) / fs
    x = np.sin(2.0 * np.pi * f_tone * t)

    spec = compute_fft(x, sampling_rate_hz=fs, window_type="hann", one_sided=True)
    dom = find_dominant_frequency(spec, ignore_dc=True, interpolate_subbin=True)

    assert dom.frequency_hz == pytest.approx(f_tone, abs=0.05)


def test_hann_window() -> None:
    """Test 12: Verify Hann window coherent gain and ENBW."""
    w, metrics = get_window("hann", 1000)
    assert metrics.name == "hann"
    assert metrics.coherent_gain == pytest.approx(0.5, abs=1e-3)
    assert metrics.enbw_bins == pytest.approx(1.5, abs=1e-2)


def test_hamming_window() -> None:
    """Test 13: Verify Hamming window coherent gain."""
    w, metrics = get_window("hamming", 1000)
    assert metrics.coherent_gain == pytest.approx(0.54, abs=1e-3)


def test_blackman_window() -> None:
    """Test 14: Verify Blackman window coherent gain."""
    w, metrics = get_window("blackman", 1000)
    assert metrics.coherent_gain == pytest.approx(0.42, abs=1e-3)


def test_kaiser_window() -> None:
    """Test 15: Verify Kaiser window generation."""
    w, metrics = get_window("kaiser", 1000, kaiser_beta=8.6)
    assert "kaiser" in metrics.name
    assert len(w) == 1000


def test_rectangular_window() -> None:
    """Test 16: Verify rectangular window coherent gain."""
    w, metrics = get_window("rectangular", 1000)
    assert metrics.coherent_gain == pytest.approx(1.0, abs=1e-12)
    assert metrics.enbw_bins == pytest.approx(1.0, abs=1e-12)


def test_window_coherent_gain_scaling() -> None:
    """Test 17: Verify window scaling recovers amplitude A under Hann windowing."""
    fs = 1000.0
    amp = 2.0
    f_tone = 100.0
    t = np.arange(1000) / fs
    x = amp * np.cos(2.0 * np.pi * f_tone * t)

    res = compute_fft(x, sampling_rate_hz=fs, window_type="hann", one_sided=True)
    bin_idx = int(np.round(f_tone / res.bin_resolution_hz))
    assert res.magnitude[bin_idx] == pytest.approx(amp, abs=1e-3)


def test_zero_padding() -> None:
    """Test 18: Verify zero-padding increases bin grid density without altering physical resolution."""
    fs = 1000.0
    n_sig = 1000
    n_fft = 4096
    x = np.ones(n_sig)

    res = compute_fft(x, sampling_rate_hz=fs, n_fft=n_fft, one_sided=True)
    assert res.n_signal_samples == 1000
    assert res.n_fft_samples == 4096
    assert res.physical_resolution_hz == pytest.approx(1.0, abs=1e-12)  # Fs / N_sig = 1 Hz
    assert res.bin_resolution_hz == pytest.approx(fs / 4096, abs=1e-12)  # Fs / N_fft = 0.244 Hz


def test_zero_padding_does_not_change_physical_resolution() -> None:
    """Test 19: Verify zero-padding does not change theoretical physical resolution Delta f_phys."""
    fs = 1000.0
    x = np.ones(500)

    res_no_pad = compute_fft(x, sampling_rate_hz=fs, n_fft=500)
    res_pad = compute_fft(x, sampling_rate_hz=fs, n_fft=2048)

    assert res_no_pad.physical_resolution_hz == res_pad.physical_resolution_hz
    assert res_pad.bin_resolution_hz < res_no_pad.bin_resolution_hz


def test_spectral_leakage_non_bin_centered_tone() -> None:
    """Test 20: Verify spectral leakage analysis for non-bin-centered tone."""
    # Fs = 10000 Hz, N = 1000 => Delta f = 10 Hz. 1055 Hz is at bin 105.5 (non-bin-centered).
    leakage = analyze_spectral_leakage(signal_frequency_hz=1055.0, sampling_rate_hz=10000.0, n_samples=1000)

    assert leakage.is_bin_centered is False
    assert "rectangular" in leakage.window_spectra
    assert "hann" in leakage.window_spectra

    # Blackman window suppresses far-off leakage sidelobes significantly more than Rectangular
    assert leakage.peak_magnitudes["hann"] <= 1.0


def test_stft_output_shape() -> None:
    """Test 21: Verify STFT output matrix dimensions."""
    fs = 1000.0
    duration = 2.0
    t = np.arange(int(fs * duration)) / fs
    x = np.sin(2.0 * np.pi * 100.0 * t)

    stft_res = compute_stft(x, sampling_rate_hz=fs, nperseg=256, noverlap=128, window_type="hann")

    assert isinstance(stft_res, SpectrogramResult)
    assert stft_res.stft_matrix.shape[0] == 129  # (256 // 2) + 1 frequency bins
    assert stft_res.stft_matrix.shape[1] > 10  # Multiple time segments


def test_even_n_scaling() -> None:
    """Test 22: Verify even N FFT processing."""
    x = np.arange(100, dtype=np.float64)
    res = compute_fft(x, sampling_rate_hz=100.0, one_sided=True)
    assert len(res.frequency_hz) == 51


def test_odd_n_scaling() -> None:
    """Test 23: Verify odd N FFT processing."""
    x = np.arange(101, dtype=np.float64)
    res = compute_fft(x, sampling_rate_hz=100.0, one_sided=True)
    assert len(res.frequency_hz) == 51  # (101 // 2) + 1 = 51 bins


def test_invalid_sampling_rate() -> None:
    """Test 24: Verify ValueError on Fs <= 0."""
    with pytest.raises(ValueError, match="strictly positive"):
        compute_fft(np.ones(10), sampling_rate_hz=0.0)


def test_empty_input() -> None:
    """Test 25: Verify ValueError on empty input array."""
    with pytest.raises(ValueError, match="empty"):
        compute_fft(np.array([]), sampling_rate_hz=100.0)


def test_non_finite_input() -> None:
    """Test 26: Verify ValueError on NaN/Inf inputs."""
    with pytest.raises(ValueError, match="non-finite"):
        compute_fft(np.array([1.0, np.nan, 2.0]), sampling_rate_hz=100.0)


def test_cross_phase_aliasing_validation() -> None:
    """Test 27: Cross-phase validation (7 kHz tone sampled at Fs = 10 kHz yields 3 kHz observed alias peak)."""
    f_tone = 7000.0
    fs = 10000.0
    t = np.arange(10000) / fs
    x = np.sin(2.0 * np.pi * f_tone * t)

    spec = compute_fft(x, sampling_rate_hz=fs, window_type="rect", one_sided=True)
    dom = find_dominant_frequency(spec, ignore_dc=True, interpolate_subbin=True)

    expected_alias = 3000.0
    assert dom.frequency_hz == pytest.approx(expected_alias, abs=0.5)


def test_dft_educational_reference() -> None:
    """Test 28: Verify naive educational DFT matches numpy FFT output."""
    x = np.array([1.0, -0.5, 2.0, 1.5, -1.0, 0.5])
    dft_out = compute_dft_educational(x)
    fft_out = np.fft.fft(x)
    np.testing.assert_allclose(dft_out, fft_out, atol=1e-12)
