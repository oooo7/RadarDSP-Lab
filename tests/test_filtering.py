"""Comprehensive unit tests for the Digital Filtering Engine (src/dsp/filtering.py)."""

import numpy as np
import pytest
from src.dsp.filtering import (
    FilterDesignResult,
    FilterResponseResult,
    analyze_filter_response,
    apply_filter,
    compare_fir_vs_iir,
    design_fir_filter,
    design_iir_butterworth,
)
from src.dsp.signals import generate_multitone, generate_sine
from src.dsp.transforms import compute_fft, find_dominant_frequency


def test_fir_lowpass_filtering() -> None:
    """Test 1: FIR lowpass retains 500 Hz tone and attenuates 3000 Hz tone at Fs = 10000 Hz."""
    fs = 10000.0
    sig = generate_multitone(
        frequencies_hz=[500.0, 3000.0],
        amplitudes=[1.0, 1.0],
        phases_rad=[0.0, 0.0],
        sampling_rate_hz=fs,
        duration_sec=1.0
    )

    filt = design_fir_filter(filter_type="lowpass", cutoff_hz=1000.0, order=64, sampling_rate_hz=fs)
    filtered = apply_filter(sig, filt, zero_phase=True)

    spec_before = compute_fft(sig, window_type="rect", one_sided=True)
    spec_after = compute_fft(filtered, window_type="rect", one_sided=True)

    idx_500 = int(np.round(500.0 / spec_before.bin_resolution_hz))
    idx_3000 = int(np.round(3000.0 / spec_before.bin_resolution_hz))

    # 500 Hz retained (~1.0 V)
    assert spec_after.magnitude[idx_500] == pytest.approx(1.0, abs=0.05)

    # 3000 Hz strongly attenuated (< 0.05 V)
    assert spec_after.magnitude[idx_3000] < 0.05


def test_fir_highpass_filtering() -> None:
    """Test 2: FIR highpass attenuates 500 Hz tone and retains 3000 Hz tone."""
    fs = 10000.0
    sig = generate_multitone(
        frequencies_hz=[500.0, 3000.0],
        amplitudes=[1.0, 1.0],
        phases_rad=[0.0, 0.0],
        sampling_rate_hz=fs,
        duration_sec=1.0
    )

    filt = design_fir_filter(filter_type="highpass", cutoff_hz=1000.0, order=64, sampling_rate_hz=fs)
    filtered = apply_filter(sig, filt, zero_phase=True)

    spec_after = compute_fft(filtered, window_type="rect", one_sided=True)
    idx_500 = int(np.round(500.0 / spec_after.bin_resolution_hz))
    idx_3000 = int(np.round(3000.0 / spec_after.bin_resolution_hz))

    assert spec_after.magnitude[idx_500] < 0.05
    assert spec_after.magnitude[idx_3000] == pytest.approx(1.0, abs=0.05)


def test_fir_bandpass_filtering() -> None:
    """Test 3: FIR bandpass retains 2000 Hz and attenuates 500 Hz & 3500 Hz."""
    fs = 10000.0
    sig = generate_multitone(
        frequencies_hz=[500.0, 2000.0, 3500.0],
        amplitudes=[1.0, 1.0, 1.0],
        phases_rad=[0.0, 0.0, 0.0],
        sampling_rate_hz=fs,
        duration_sec=1.0
    )

    filt = design_fir_filter(filter_type="bandpass", cutoff_hz=[1500.0, 2500.0], order=80, sampling_rate_hz=fs)
    filtered = apply_filter(sig, filt, zero_phase=True)

    spec_after = compute_fft(filtered, window_type="rect", one_sided=True)
    idx_500 = int(np.round(500.0 / spec_after.bin_resolution_hz))
    idx_2000 = int(np.round(2000.0 / spec_after.bin_resolution_hz))
    idx_3500 = int(np.round(3500.0 / spec_after.bin_resolution_hz))

    assert spec_after.magnitude[idx_500] < 0.05
    assert spec_after.magnitude[idx_2000] == pytest.approx(1.0, abs=0.05)
    assert spec_after.magnitude[idx_3500] < 0.05


def test_fir_bandstop_filtering() -> None:
    """Test 4: FIR bandstop attenuates 2000 Hz and retains 500 Hz & 3500 Hz."""
    fs = 10000.0
    sig = generate_multitone(
        frequencies_hz=[500.0, 2000.0, 3500.0],
        amplitudes=[1.0, 1.0, 1.0],
        phases_rad=[0.0, 0.0, 0.0],
        sampling_rate_hz=fs,
        duration_sec=1.0
    )

    filt = design_fir_filter(filter_type="bandstop", cutoff_hz=[1500.0, 2500.0], order=80, sampling_rate_hz=fs)
    filtered = apply_filter(sig, filt, zero_phase=True)

    spec_after = compute_fft(filtered, window_type="rect", one_sided=True)
    idx_500 = int(np.round(500.0 / spec_after.bin_resolution_hz))
    idx_2000 = int(np.round(2000.0 / spec_after.bin_resolution_hz))
    idx_3500 = int(np.round(3500.0 / spec_after.bin_resolution_hz))

    assert spec_after.magnitude[idx_500] == pytest.approx(1.0, abs=0.05)
    assert spec_after.magnitude[idx_2000] < 0.05
    assert spec_after.magnitude[idx_3500] == pytest.approx(1.0, abs=0.05)


def test_iir_butterworth_lowpass() -> None:
    """Test 5: IIR Butterworth lowpass filtering."""
    fs = 10000.0
    sig = generate_multitone(frequencies_hz=[500.0, 3000.0], amplitudes=[1.0, 1.0], phases_rad=[0.0, 0.0], sampling_rate_hz=fs, duration_sec=1.0)
    filt = design_iir_butterworth(filter_type="lowpass", cutoff_hz=1000.0, order=6, sampling_rate_hz=fs)
    filtered = apply_filter(sig, filt, zero_phase=True)

    spec_after = compute_fft(filtered, window_type="rect", one_sided=True)
    idx_500 = int(np.round(500.0 / spec_after.bin_resolution_hz))
    idx_3000 = int(np.round(3000.0 / spec_after.bin_resolution_hz))

    assert spec_after.magnitude[idx_500] == pytest.approx(1.0, abs=0.05)
    assert spec_after.magnitude[idx_3000] < 0.01


def test_iir_butterworth_highpass() -> None:
    """Test 6: IIR Butterworth highpass filtering."""
    fs = 10000.0
    sig = generate_multitone(frequencies_hz=[500.0, 3000.0], amplitudes=[1.0, 1.0], phases_rad=[0.0, 0.0], sampling_rate_hz=fs, duration_sec=1.0)
    filt = design_iir_butterworth(filter_type="highpass", cutoff_hz=1000.0, order=6, sampling_rate_hz=fs)
    filtered = apply_filter(sig, filt, zero_phase=True)

    spec_after = compute_fft(filtered, window_type="rect", one_sided=True)
    idx_500 = int(np.round(500.0 / spec_after.bin_resolution_hz))
    idx_3000 = int(np.round(3000.0 / spec_after.bin_resolution_hz))

    assert spec_after.magnitude[idx_500] < 0.01
    assert spec_after.magnitude[idx_3000] == pytest.approx(1.0, abs=0.05)


def test_butterworth_cutoff_3db_point() -> None:
    """Test 7: Verify Butterworth lowpass response is -3 dB (0.707) at cutoff frequency."""
    fs = 10000.0
    cutoff = 1000.0
    filt = design_iir_butterworth(filter_type="lowpass", cutoff_hz=cutoff, order=4, sampling_rate_hz=fs)
    resp = analyze_filter_response(filt, n_points=5000)

    # Find magnitude at cutoff frequency (1000 Hz)
    bin_idx = int(np.argmin(np.abs(resp.frequency_hz - cutoff)))
    mag_at_cutoff = resp.magnitude[bin_idx]
    mag_db_at_cutoff = resp.magnitude_db[bin_idx]

    assert mag_at_cutoff == pytest.approx(1.0 / np.sqrt(2.0), abs=1e-2)
    assert mag_db_at_cutoff == pytest.approx(-3.01, abs=0.1)


def test_fir_linear_phase_symmetry() -> None:
    """Test 8: Verify FIR coefficient symmetry b[n] == b[N-1-n] for linear phase."""
    filt = design_fir_filter(filter_type="lowpass", cutoff_hz=1000.0, order=32, sampling_rate_hz=10000.0, window_type="hamming")
    b = filt.b_coefficients
    np.testing.assert_allclose(b, b[::-1], atol=1e-12)
    assert filt.metadata["is_linear_phase"] is True


def test_fir_group_delay() -> None:
    """Test 9: Verify FIR group delay == (N_taps - 1) / 2 samples."""
    order = 40
    filt = design_fir_filter(filter_type="lowpass", cutoff_hz=1000.0, order=order, sampling_rate_hz=10000.0)
    expected_delay = (filt.n_taps - 1) / 2.0
    assert filt.metadata["theoretical_group_delay_samples"] == pytest.approx(expected_delay, abs=1e-12)


def test_causal_vs_zerophase_filtering() -> None:
    """Test 10: Verify causal filtering introduces time delay while zero-phase filtering has no time shift."""
    fs = 10000.0
    sig = generate_sine(amplitude=1.0, frequency_hz=100.0, phase_rad=0.0, sampling_rate_hz=fs, duration_sec=0.1)
    filt = design_fir_filter(filter_type="lowpass", cutoff_hz=1000.0, order=64, sampling_rate_hz=fs)

    causal = apply_filter(sig, filt, zero_phase=False)
    zerophase = apply_filter(sig, filt, zero_phase=True)

    # Peak index for zero-phase filtering aligns with input signal peak
    orig_peak = int(np.argmax(sig.amplitude[:100]))
    zero_peak = int(np.argmax(zerophase.amplitude[:100]))
    causal_peak = int(np.argmax(causal.amplitude[:100]))

    assert zero_peak == orig_peak
    assert causal_peak > orig_peak  # Delayed by group delay


def test_iir_sos_stability() -> None:
    """Test 11: Verify high-order Butterworth IIR filter SOS stability (no NaN, no Inf, bounded output)."""
    fs = 10000.0
    sig = generate_sine(amplitude=1.0, frequency_hz=500.0, phase_rad=0.0, sampling_rate_hz=fs, duration_sec=1.0)
    filt = design_iir_butterworth(filter_type="lowpass", cutoff_hz=1000.0, order=10, sampling_rate_hz=fs)

    filtered = apply_filter(sig, filt, zero_phase=False)
    assert np.all(np.isfinite(filtered.amplitude))
    assert np.max(np.abs(filtered.amplitude)) < 2.0


def test_invalid_filter_inputs() -> None:
    """Test 12: Verify ValueError exceptions on invalid design parameters."""
    with pytest.raises(ValueError, match="strictly positive"):
        design_fir_filter("lowpass", 100.0, 32, -1000.0)

    with pytest.raises(ValueError, match="strictly between 0 and Nyquist"):
        design_fir_filter("lowpass", -50.0, 32, 1000.0)

    with pytest.raises(ValueError, match="strictly between 0 and Nyquist"):
        design_fir_filter("lowpass", 600.0, 32, 1000.0)  # Cutoff >= Nyquist

    with pytest.raises(ValueError, match="must satisfy 0 < f_low < f_high"):
        design_fir_filter("bandpass", [600.0, 200.0], 32, 1000.0)

    with pytest.raises(ValueError, match="Unsupported filter_type"):
        design_fir_filter("invalid_type", 100.0, 32, 1000.0)


def test_invalid_signal_filtering_inputs() -> None:
    """Test 13: Verify ValueError exceptions on invalid signal inputs to apply_filter."""
    filt = design_fir_filter("lowpass", 100.0, 32, 1000.0)

    with pytest.raises(ValueError, match="empty"):
        apply_filter(np.array([]), filt)

    with pytest.raises(ValueError, match="non-finite"):
        apply_filter(np.array([1.0, np.nan, 2.0]), filt)

    with pytest.raises(ValueError, match="too short for zero-phase"):
        apply_filter(np.ones(10), filt, zero_phase=True)


def test_compare_fir_vs_iir_summary() -> None:
    """Test 14: Verify compare_fir_vs_iir comparison helper."""
    sig = generate_multitone(frequencies_hz=[500.0, 3000.0], amplitudes=[1.0, 1.0], phases_rad=[0.0, 0.0], sampling_rate_hz=10000.0, duration_sec=1.0)
    cmp_res = compare_fir_vs_iir(sig, cutoff_hz=1000.0, fir_order=64, iir_order=4)

    assert "fir_design" in cmp_res
    assert "iir_design" in cmp_res
    assert isinstance(cmp_res["fir_design"], FilterDesignResult)
    assert isinstance(cmp_res["iir_design"], FilterDesignResult)
