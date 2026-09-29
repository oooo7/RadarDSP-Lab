"""Comprehensive audit and numerical unit test suite for src/dsp/transforms.py."""

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


def test_audit_1a_real_cosine_magnitude_scaling() -> None:
    """Audit 1A: A = 2.0 V, Fs = 10000 Hz, N = 10000, f = 1000 Hz => magnitude == 2.0 V."""
    fs = 10000.0
    n = 10000
    amp = 2.0
    f_tone = 1000.0
    t = np.arange(n) / fs
    x = amp * np.cos(2.0 * np.pi * f_tone * t)

    res = compute_fft(x, sampling_rate_hz=fs, window_type="rect", one_sided=True)
    bin_idx = int(np.round(f_tone / res.bin_resolution_hz))
    assert res.magnitude[bin_idx] == pytest.approx(amp, abs=1e-3)


def test_audit_1b_dc_magnitude_scaling() -> None:
    """Audit 1B: Constant DC signal x[n] = 2.0 V => DC magnitude == 2.0 V."""
    c = 2.0
    x = np.full(1000, fill_value=c, dtype=np.float64)
    res = compute_fft(x, sampling_rate_hz=1000.0, window_type="rect", one_sided=True)
    assert res.magnitude[0] == pytest.approx(c, abs=1e-12)


def test_audit_1c_nyquist_cosine_even_n_scaling() -> None:
    """Audit 1C: Nyquist cosine for even N: f = Fs/2, A = 3.0 => Nyquist magnitude == 3.0 V."""
    fs = 1000.0
    n = 1000  # Even N
    amp = 3.0
    t = np.arange(n) / fs
    x = amp * np.cos(np.pi * fs * t)  # 500 Hz tone at Nyquist

    res = compute_fft(x, sampling_rate_hz=fs, window_type="rect", one_sided=True)
    nyq_bin = n // 2
    assert res.magnitude[nyq_bin] == pytest.approx(amp, abs=1e-3)


def test_audit_1d_interior_bin_scaling() -> None:
    """Audit 1D: Interior positive-frequency sinusoid recovers peak amplitude A."""
    fs = 1000.0
    amp = 1.75
    f_tone = 250.0  # Interior bin
    t = np.arange(1000) / fs
    x = amp * np.cos(2.0 * np.pi * f_tone * t)

    res = compute_fft(x, sampling_rate_hz=fs, window_type="rect", one_sided=True)
    bin_idx = int(np.round(f_tone / res.bin_resolution_hz))
    assert res.magnitude[bin_idx] == pytest.approx(amp, abs=1e-3)


def test_audit_1e_twosided_spectrum_scaling() -> None:
    """Audit 1E: Two-sided spectrum positive and negative bins each equal A/2."""
    fs = 1000.0
    amp = 2.0
    f_tone = 200.0
    t = np.arange(1000) / fs
    x = amp * np.cos(2.0 * np.pi * f_tone * t)

    res = compute_fft(x, sampling_rate_hz=fs, window_type="rect", one_sided=False)
    # Bin +200 Hz and bin -200 Hz
    pos_bin = 200
    neg_bin = 1000 - 200
    assert res.magnitude[pos_bin] == pytest.approx(amp / 2.0, abs=1e-3)
    assert res.magnitude[neg_bin] == pytest.approx(amp / 2.0, abs=1e-3)


def test_audit_2_coherent_gain_window_recovery() -> None:
    """Audit 2: Coherent gain correction recovers amplitude A across all supported windows."""
    fs = 1000.0
    amp = 2.5
    f_tone = 100.0
    t = np.arange(1000) / fs
    x = amp * np.cos(2.0 * np.pi * f_tone * t)

    for w_name in ["rectangular", "hann", "hamming", "blackman", "kaiser"]:
        res = compute_fft(x, sampling_rate_hz=fs, window_type=w_name, one_sided=True)
        bin_idx = int(np.round(f_tone / res.bin_resolution_hz))
        assert res.magnitude[bin_idx] == pytest.approx(amp, abs=1e-3), f"Failed for window {w_name}"


def test_audit_3_spectral_leakage_psl() -> None:
    """Audit 3: Peak Sidelobe Level (PSL) metric excludes mainlobe and behaves as expected."""
    leakage = analyze_spectral_leakage(signal_frequency_hz=1055.0, sampling_rate_hz=10000.0, n_samples=1000)

    assert leakage.is_bin_centered is False
    psl = leakage.peak_sidelobe_level_db

    # Blackman window has lower (more negative) Peak Sidelobe Level than Hann, Hamming, and Rectangular
    assert psl["blackman"] < psl["hann"]
    assert psl["hann"] < psl["rectangular"]


def test_audit_4_zero_padding_resolution() -> None:
    """Audit 4: Zero padding increases bin grid density without altering physical resolution."""
    fs = 1000.0
    n_sig = 1000
    n_fft = 4096
    x = np.ones(n_sig)

    res = compute_fft(x, sampling_rate_hz=fs, n_fft=n_fft, one_sided=True)
    assert res.physical_resolution_hz == pytest.approx(fs / n_sig, abs=1e-12)
    assert res.bin_resolution_hz == pytest.approx(fs / n_fft, abs=1e-12)


def test_audit_5_subbin_estimation_various_freqs() -> None:
    """Audit 5: Sub-bin parabolic estimation for non-bin-centered tones (1055 Hz, 1234.5 Hz, 1789.2 Hz)."""
    fs = 10000.0
    n = 2000  # Bin spacing = 5 Hz

    for f_tone in [1055.0, 1234.5, 1789.2]:
        t = np.arange(n) / fs
        x = np.sin(2.0 * np.pi * f_tone * t)

        spec = compute_fft(x, sampling_rate_hz=fs, window_type="hann", one_sided=True)
        dom = find_dominant_frequency(spec, ignore_dc=True, interpolate_subbin=True)

        assert dom.frequency_hz == pytest.approx(f_tone, abs=0.25), f"Failed for {f_tone} Hz"


def test_audit_5b_subbin_edge_cases() -> None:
    """Audit 5B: Sub-bin interpolation safely handles DC, Nyquist, and flat signals without crashing."""
    fs = 1000.0
    n = 100

    # 1. DC Peak
    x_dc = np.ones(n)
    spec_dc = compute_fft(x_dc, sampling_rate_hz=fs, one_sided=True)
    dom_dc = find_dominant_frequency(spec_dc, ignore_dc=False, interpolate_subbin=True)
    assert dom_dc.frequency_hz == 0.0

    # 2. Nyquist Peak
    x_nyq = np.cos(np.pi * np.arange(n))
    spec_nyq = compute_fft(x_nyq, sampling_rate_hz=fs, one_sided=True)
    dom_nyq = find_dominant_frequency(spec_nyq, ignore_dc=True, interpolate_subbin=True)
    assert dom_nyq.frequency_hz == pytest.approx(500.0, abs=1e-2)

    # 3. All Zero Signal
    x_zero = np.zeros(n)
    spec_zero = compute_fft(x_zero, sampling_rate_hz=fs, one_sided=True)
    dom_zero = find_dominant_frequency(spec_zero, ignore_dc=True, interpolate_subbin=True)
    assert dom_zero.frequency_hz == 0.0


def test_audit_6_fft_ifft_roundtrip_all_types() -> None:
    """Audit 6: FFT -> IFFT roundtrip for real, complex, even N, and odd N."""
    rng = np.random.default_rng(42)

    for n in [100, 101]:
        # Real signal
        x_real = rng.normal(size=n)
        spec_real = compute_fft(x_real, sampling_rate_hz=100.0, one_sided=False)
        rec_real = compute_ifft(spec_real.complex_spectrum)
        np.testing.assert_allclose(np.real(rec_real), x_real, atol=1e-12)

        # Complex signal
        x_complex = rng.normal(size=n) + 1j * rng.normal(size=n)
        spec_comp = compute_fft(x_complex, sampling_rate_hz=100.0, one_sided=False)
        rec_comp = compute_ifft(spec_comp.complex_spectrum)
        np.testing.assert_allclose(rec_comp, x_complex, atol=1e-12)


def test_audit_7_rfft_even_odd_n_axes() -> None:
    """Audit 7: rFFT frequency axes for even N (N/2+1 bins ending at Fs/2) and odd N ((N+1)/2 bins ending below Fs/2)."""
    fs = 1000.0

    # Even N = 100 => 51 bins, last bin = 500 Hz
    x_even = np.ones(100)
    res_even = compute_fft(x_even, sampling_rate_hz=fs, one_sided=True)
    assert len(res_even.frequency_hz) == 51
    assert res_even.frequency_hz[-1] == pytest.approx(500.0, abs=1e-12)

    # Odd N = 101 => 51 bins, last bin < 500 Hz (495.0495 Hz)
    x_odd = np.ones(101)
    res_odd = compute_fft(x_odd, sampling_rate_hz=fs, one_sided=True)
    assert len(res_odd.frequency_hz) == 51
    assert res_odd.frequency_hz[-1] < 500.0
    assert res_odd.frequency_hz[-1] == pytest.approx(500.0 * 100.0 / 101.0, abs=1e-12)


def test_audit_8_stft_1k_tone() -> None:
    """Audit 8: STFT for 1 kHz sine wave shows energy concentration at 1 kHz bin."""
    fs = 10000.0
    t = np.arange(10000) / fs
    x = np.sin(2.0 * np.pi * 1000.0 * t)

    stft_res = compute_stft(x, sampling_rate_hz=fs, nperseg=256, noverlap=128, window_type="hann")

    # Find bin closest to 1000 Hz
    freq_bins = stft_res.frequency_vector
    bin_1k = int(np.argmin(np.abs(freq_bins - 1000.0)))

    # Max energy across frequency rows should occur at bin_1k for all time segments
    for time_col in range(stft_res.magnitude_matrix.shape[1]):
        max_row = int(np.argmax(stft_res.magnitude_matrix[:, time_col]))
        assert abs(max_row - bin_1k) <= 1


def test_audit_9_power_spectrum_vs_psd() -> None:
    """Audit 9: Verify distinction between Power Spectrum (V^2) and PSD (V^2/Hz)."""
    fs = 1000.0
    amp = 2.0
    t = np.arange(1000) / fs
    x = amp * np.cos(2.0 * np.pi * 100.0 * t)

    res = compute_fft(x, sampling_rate_hz=fs, window_type="rect", one_sided=True)

    bin_idx = int(np.round(100.0 / res.bin_resolution_hz))
    power_peak = res.power[bin_idx]
    psd_peak = res.psd[bin_idx]

    # Peak Power Spectrum = amp^2 = 4.0 V^2
    assert power_peak == pytest.approx(amp ** 2, abs=1e-2)

    # Integrated PSD over bin width = RMS power = amp^2 / 2 = 2.0 V^2
    assert psd_peak * res.bin_resolution_hz == pytest.approx((amp ** 2) / 2.0, abs=1e-2)



def test_audit_10_phase2_aliasing_integration_multitone() -> None:
    """Audit 10: Verify Phase 2 aliasing integration for 7 kHz, 13 kHz, 17 kHz sampled at 10 kHz (all alias to 3 kHz)."""
    fs = 10000.0
    t = np.arange(10000) / fs

    for f_tone in [7000.0, 13000.0, 17000.0]:
        x = np.sin(2.0 * np.pi * f_tone * t)
        spec = compute_fft(x, sampling_rate_hz=fs, window_type="rect", one_sided=True)
        dom = find_dominant_frequency(spec, ignore_dc=True, interpolate_subbin=True)

        expected_alias = 3000.0
        assert dom.frequency_hz == pytest.approx(expected_alias, abs=0.1), f"Failed for {f_tone} Hz"
