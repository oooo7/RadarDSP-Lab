"""Comprehensive unit tests for the Sampling, Nyquist Analysis, and Aliasing Engine (src/dsp/sampling.py)."""

import numpy as np
import pytest
from src.dsp.sampling import (
    SamplingAnalysisResult,
    analyze_nyquist,
    calculate_alias_errors,
    calculate_alias_frequency,
    measure_alias_frequencies,
    sample_signal,
)
from src.dsp.signals import generate_multitone, generate_sine


def test_calculate_alias_frequency_no_aliasing() -> None:
    """Test 1: Verify alias frequency when f < Fs / 2 (no aliasing)."""
    # f = 100 Hz, Fs = 1000 Hz => alias = 100 Hz
    alias = calculate_alias_frequency(fundamental_freq_hz=100.0, sampling_rate_hz=1000.0)
    assert alias == pytest.approx(100.0, abs=1e-12)


def test_calculate_alias_frequency_first_zone_foldover() -> None:
    """Test 2: Verify alias frequency foldover when Fs / 2 < f < Fs."""
    # f = 700 Hz, Fs = 1000 Hz => alias = 1000 - 700 = 300 Hz
    alias = calculate_alias_frequency(fundamental_freq_hz=700.0, sampling_rate_hz=1000.0)
    assert alias == pytest.approx(300.0, abs=1e-12)


def test_calculate_alias_frequency_higher_zones() -> None:
    """Test 3: Verify alias frequency for higher Nyquist zone harmonics."""
    # f = 1200 Hz, Fs = 1000 Hz => rem = 200 Hz => alias = 200 Hz
    assert calculate_alias_frequency(1200.0, 1000.0) == pytest.approx(200.0, abs=1e-12)

    # f = 1700 Hz, Fs = 1000 Hz => rem = 700 Hz => alias = 300 Hz
    assert calculate_alias_frequency(1700.0, 1000.0) == pytest.approx(300.0, abs=1e-12)

    # f = 500 Hz (folding frequency), Fs = 1000 Hz => alias = 500 Hz
    assert calculate_alias_frequency(500.0, 1000.0) == pytest.approx(500.0, abs=1e-12)


def test_calculate_alias_frequency_validation() -> None:
    """Test 4: Verify error handling on invalid inputs to calculate_alias_frequency."""
    with pytest.raises(ValueError, match="Fundamental frequency must be non-negative"):
        calculate_alias_frequency(-100.0, 1000.0)

    with pytest.raises(ValueError, match="Sampling rate Fs must be strictly positive"):
        calculate_alias_frequency(100.0, 0.0)


def test_analyze_nyquist_oversampled() -> None:
    """Test 5: Verify Nyquist rate analysis for oversampled signal."""
    res = analyze_nyquist(signal_frequencies_hz=[100.0, 200.0], sampling_rate_hz=1000.0)
    assert res["f_max_hz"] == 200.0
    assert res["nyquist_rate_hz"] == 400.0
    assert res["folding_frequency_hz"] == 500.0
    assert res["is_oversampled"] is True
    assert res["is_undersampled"] is False
    assert res["is_aliased"] is False
    assert res["theoretical_alias_frequencies_hz"] == [100.0, 200.0]


def test_analyze_nyquist_undersampled() -> None:
    """Test 6: Verify Nyquist rate analysis for undersampled/aliased signal."""
    res = analyze_nyquist(signal_frequencies_hz=[700.0], sampling_rate_hz=1000.0)
    assert res["f_max_hz"] == 700.0
    assert res["nyquist_rate_hz"] == 1400.0
    assert res["folding_frequency_hz"] == 500.0
    assert res["is_undersampled"] is True
    assert res["is_aliased"] is True
    assert res["theoretical_alias_frequencies_hz"] == [300.0]


def test_measure_alias_frequencies_clean_tone() -> None:
    """Test 7: Verify spectral FFT measurement of 100 Hz tone at Fs = 1000 Hz."""
    sig = generate_sine(
        amplitude=1.0,
        frequency_hz=100.0,
        phase_rad=0.0,
        sampling_rate_hz=1000.0,
        duration_sec=1.0
    )
    measured = measure_alias_frequencies(sig, num_peaks=1)
    assert len(measured) == 1
    assert measured[0] == pytest.approx(100.0, abs=0.2)


def test_measure_alias_frequencies_aliased_tone() -> None:
    """Test 8: Verify spectral FFT measurement of aliased 700 Hz tone sampled at Fs = 1000 Hz."""
    # 700 Hz tone sampled at 1000 Hz appears at 300 Hz
    sig = generate_sine(
        amplitude=1.0,
        frequency_hz=700.0,
        phase_rad=0.0,
        sampling_rate_hz=1000.0,
        duration_sec=1.0
    )
    measured = measure_alias_frequencies(sig, num_peaks=1)
    assert len(measured) == 1
    assert measured[0] == pytest.approx(300.0, abs=0.2)


def test_calculate_alias_errors() -> None:
    """Test 9: Verify error calculations between theoretical and measured frequencies."""
    theo = [100.0, 300.0]
    meas = [100.05, 299.90]

    errs = calculate_alias_errors(theo, meas)
    assert errs["absolute_errors_hz"][0] == pytest.approx(0.05, abs=1e-4)
    assert errs["absolute_errors_hz"][1] == pytest.approx(0.10, abs=1e-4)
    assert errs["mae_hz"] == pytest.approx(0.075, abs=1e-4)


def test_sample_signal_pipeline_oversampled() -> None:
    """Test 10: Verify end-to-end sampling analysis pipeline for oversampled signal."""
    ref_sig = generate_sine(
        amplitude=1.0,
        frequency_hz=100.0,
        phase_rad=0.0,
        sampling_rate_hz=10000.0,
        duration_sec=1.0
    )

    res = sample_signal(continuous_signal=ref_sig, target_fs_hz=1000.0)

    assert isinstance(res, SamplingAnalysisResult)
    assert res.target_fs_hz == 1000.0
    assert res.nyquist_rate_hz == 200.0
    assert res.is_oversampled is True
    assert res.is_aliased is False
    assert res.theoretical_alias_frequencies_hz == [100.0]
    assert res.measured_alias_frequencies_hz[0] == pytest.approx(100.0, abs=0.5)
    assert res.absolute_errors_hz[0] < 0.5


def test_sample_signal_pipeline_undersampled() -> None:
    """Test 11: Verify end-to-end sampling analysis pipeline for undersampled signal (700 Hz tone at Fs = 1000 Hz)."""
    ref_sig = generate_sine(
        amplitude=1.0,
        frequency_hz=700.0,
        phase_rad=0.0,
        sampling_rate_hz=10000.0,
        duration_sec=1.0
    )

    res = sample_signal(continuous_signal=ref_sig, target_fs_hz=1000.0)

    assert res.is_undersampled is True
    assert res.is_aliased is True
    assert res.theoretical_alias_frequencies_hz == [300.0]
    assert res.measured_alias_frequencies_hz[0] == pytest.approx(300.0, abs=0.5)
    assert res.absolute_errors_hz[0] < 0.5


def test_sample_signal_pipeline_multitone() -> None:
    """Test 12: Verify sampling analysis on multi-tone signal with one aliased and one unaliased component."""
    freqs = [100.0, 750.0]  # 100 Hz (unaliased), 750 Hz (aliased to 250 Hz at Fs = 1000 Hz)
    ref_sig = generate_multitone(
        frequencies_hz=freqs,
        amplitudes=[1.0, 0.8],
        phases_rad=[0.0, 0.0],
        sampling_rate_hz=10000.0,
        duration_sec=1.0
    )

    res = sample_signal(continuous_signal=ref_sig, target_fs_hz=1000.0)

    assert res.is_aliased is True
    assert res.theoretical_alias_frequencies_hz == [100.0, 250.0]
    assert res.measured_alias_frequencies_hz[0] == pytest.approx(100.0, abs=0.5)
    assert res.measured_alias_frequencies_hz[1] == pytest.approx(250.0, abs=0.5)


def test_sample_signal_invalid_target_fs() -> None:
    """Test 13: Verify error when target_fs_hz > continuous_signal.sampling_rate_hz."""
    ref_sig = generate_sine(amplitude=1.0, frequency_hz=100.0, phase_rad=0.0, sampling_rate_hz=1000.0, duration_sec=1.0)

    with pytest.raises(ValueError, match="cannot exceed reference continuous signal sampling rate"):
        sample_signal(continuous_signal=ref_sig, target_fs_hz=5000.0)

    with pytest.raises(ValueError, match="strictly positive"):
        sample_signal(continuous_signal=ref_sig, target_fs_hz=-100.0)
