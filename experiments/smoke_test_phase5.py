"""Phase 5 Multirate DSP Integration & Smoke Test.

Demonstrates:
Experiment 1 — Decimation: Fs = 10000 Hz -> 5000 Hz (M=2) comparing naive downsampling vs proper anti-aliasing decimation.
Experiment 2 — Interpolation: Fs = 5000 Hz -> 10000 Hz (L=2) comparing raw zero-insertion vs polyphase anti-imaging interpolation.
Experiment 3 — Rational Resampling: Fs = 10000 Hz -> 15000 Hz (L=3, M=2) preserving physical frequencies & peak amplitudes.
"""

import numpy as np
from src.dsp.resampling import (
    decimate_signal,
    downsample_naive,
    interpolate_signal,
    resample_rational,
    upsample_zero_insertion,
)
from src.dsp.signals import generate_multitone, generate_sine
from src.dsp.transforms import compute_fft, find_dominant_frequency


def run_smoke_test_phase5() -> None:
    """Run Phase 5 Multirate DSP demonstration pipeline."""
    print("=== RadarDSP Lab — Phase 5 Multirate DSP Engine Smoke Test ===")

    # =========================================================================
    # EXPERIMENT 1 — DECIMATION (M = 2)
    # =========================================================================
    print("\n--------------------------------------------------")
    print("Experiment 1 — Decimation (Fs = 10000 Hz -> 5000 Hz, M = 2)")
    print("--------------------------------------------------")
    fs_in_dec = 10000.0
    M = 2
    fs_out_dec = fs_in_dec / M
    nyq_in_dec = fs_in_dec / 2.0
    nyq_out_dec = fs_out_dec / 2.0

    print(f"  Input Fs:  {fs_in_dec:.1f} Hz | Nyquist: {nyq_in_dec:.1f} Hz")
    print(f"  Output Fs: {fs_out_dec:.1f} Hz | Nyquist: {nyq_out_dec:.1f} Hz")

    sig_dec = generate_multitone(
        frequencies_hz=[500.0, 3000.0],
        amplitudes=[1.0, 1.0],
        phases_rad=[0.0, 0.0],
        sampling_rate_hz=fs_in_dec,
        duration_sec=2.0
    )

    spec_in_dec = compute_fft(sig_dec, window_type="rect", one_sided=True)
    idx_in_500 = int(np.round(500.0 / spec_in_dec.bin_resolution_hz))
    idx_in_3000 = int(np.round(3000.0 / spec_in_dec.bin_resolution_hz))

    in_mag_500 = spec_in_dec.magnitude[idx_in_500]
    in_mag_3000 = spec_in_dec.magnitude[idx_in_3000]

    print(f"  Input 500 Hz Peak Magnitude:  {in_mag_500:.6f} V")
    print(f"  Input 3000 Hz Peak Magnitude: {in_mag_3000:.6f} V (Exceeds Output Nyquist {nyq_out_dec} Hz)")

    # A) Naive Downsampling (x[::2])
    naive_out = downsample_naive(sig_dec, factor_M=M)
    spec_naive = compute_fft(naive_out, window_type="rect", one_sided=True)
    idx_n_500 = int(np.round(500.0 / spec_naive.bin_resolution_hz))
    idx_n_2000 = int(np.round(2000.0 / spec_naive.bin_resolution_hz))  # Aliased 3000 Hz tone

    naive_500 = spec_naive.magnitude[idx_n_500]
    naive_alias_2000 = spec_naive.magnitude[idx_n_2000]

    print("\n  [Naive Downsampling (x[::2]) — ALIASED]:")
    print(f"    Passband 500 Hz Output: {naive_500:.6f} V")
    print(f"    Aliased 2000 Hz Output:  {naive_alias_2000:.6f} V (Unfiltered 3000 Hz folded into 2000 Hz!)")

    # B) Proper Decimation with Anti-Aliasing Filter
    prop_dec_res = decimate_signal(sig_dec, factor_M=M, method="polyphase")
    spec_dec = compute_fft(prop_dec_res.processed_signal, window_type="rect", one_sided=True)
    idx_d_500 = int(np.round(500.0 / spec_dec.bin_resolution_hz))
    idx_d_2000 = int(np.round(2000.0 / spec_dec.bin_resolution_hz))

    dec_500 = spec_dec.magnitude[idx_d_500]
    dec_suppressed_2000 = spec_dec.magnitude[idx_d_2000]

    print("\n  [Proper Decimation — ANTI-ALIASED]:")
    print(f"    Passband 500 Hz Output:     {dec_500:.6f} V (Preserved)")
    print(f"    Stopband 3000 Hz/Alias Out: {dec_suppressed_2000:.6f} V (Strongly Suppressed)")

    # =========================================================================
    # EXPERIMENT 2 — INTERPOLATION (L = 2)
    # =========================================================================
    print("\n--------------------------------------------------")
    print("Experiment 2 — Interpolation (Fs = 5000 Hz -> 10000 Hz, L = 2)")
    print("--------------------------------------------------")
    fs_in_interp = 5000.0
    L = 2
    fs_out_interp = fs_in_interp * L

    print(f"  Input Fs:  {fs_in_interp:.1f} Hz | Output Fs: {fs_out_interp:.1f} Hz")

    sig_interp = generate_sine(
        amplitude=1.0,
        frequency_hz=500.0,
        phase_rad=0.0,
        sampling_rate_hz=fs_in_interp,
        duration_sec=2.0
    )

    spec_in_interp = compute_fft(sig_interp, window_type="rect", one_sided=True)
    idx_in_500_i = int(np.round(500.0 / spec_in_interp.bin_resolution_hz))
    in_tone_mag = spec_in_interp.magnitude[idx_in_500_i]

    print(f"  Input 500 Hz Tone Magnitude: {in_tone_mag:.6f} V")

    # A) Naive Zero Insertion
    zero_up_out = upsample_zero_insertion(sig_interp, factor_L=L)
    spec_zero_up = compute_fft(zero_up_out, window_type="rect", one_sided=True)
    idx_z_500 = int(np.round(500.0 / spec_zero_up.bin_resolution_hz))
    idx_z_4500 = int(np.round(4500.0 / spec_zero_up.bin_resolution_hz))  # Spectral image (5000 - 500)

    zero_up_500 = spec_zero_up.magnitude[idx_z_500]
    zero_up_image_4500 = spec_zero_up.magnitude[idx_z_4500]

    print("\n  [Naive Zero Insertion — UNFILTERED SPECTRAL IMAGES]:")
    print(f"    Physical 500 Hz Tone:   {zero_up_500:.6f} V")
    print(f"    Spectral Image 4500 Hz: {zero_up_image_4500:.6f} V (Unfiltered image at Fs_in - f!)")

    # B) Proper Interpolation with Anti-Imaging Filter
    interp_res = interpolate_signal(sig_interp, factor_L=L)
    spec_interp = compute_fft(interp_res.processed_signal, window_type="rect", one_sided=True)
    idx_p_500 = int(np.round(500.0 / spec_interp.bin_resolution_hz))
    idx_p_4500 = int(np.round(4500.0 / spec_interp.bin_resolution_hz))

    interp_500 = spec_interp.magnitude[idx_p_500]
    interp_image_4500 = spec_interp.magnitude[idx_p_4500]

    print("\n  [Proper Interpolation — ANTI-IMAGING FILTERED]:")
    print(f"    Physical 500 Hz Tone:   {interp_500:.6f} V (Preserved)")
    print(f"    Spectral Image 4500 Hz: {interp_image_4500:.6f} V (Strongly Suppressed)")

    # =========================================================================
    # EXPERIMENT 3 — RATIONAL RESAMPLING (10000 Hz -> 15000 Hz, L/M = 3/2)
    # =========================================================================
    print("\n--------------------------------------------------")
    print("Experiment 3 — Rational Resampling (10000 Hz -> 15000 Hz, L/M = 3/2)")
    print("--------------------------------------------------")
    fs_in_rat = 10000.0
    fs_out_rat = 15000.0

    sig_rat = generate_multitone(
        frequencies_hz=[500.0, 1500.0],
        amplitudes=[1.0, 1.0],
        phases_rad=[0.0, 0.0],
        sampling_rate_hz=fs_in_rat,
        duration_sec=2.0
    )

    rat_res = resample_rational(sig_rat, target_sampling_rate_hz=fs_out_rat)
    spec_rat = compute_fft(rat_res.processed_signal, window_type="rect", one_sided=True)

    idx_r_500 = int(np.round(500.0 / spec_rat.bin_resolution_hz))
    idx_r_1500 = int(np.round(1500.0 / spec_rat.bin_resolution_hz))

    meas_500 = spec_rat.magnitude[idx_r_500]
    meas_1500 = spec_rat.magnitude[idx_r_1500]

    # Measure dominant peak frequency
    dom = find_dominant_frequency(spec_rat, ignore_dc=True, interpolate_subbin=True)

    err_freq_500 = abs(spec_rat.frequency_hz[idx_r_500] - 500.0)
    err_freq_1500 = abs(spec_rat.frequency_hz[idx_r_1500] - 1500.0)
    err_amp_500 = abs(meas_500 - 1.0)
    err_amp_1500 = abs(meas_1500 - 1.0)

    print(f"  Input Fs:  {rat_res.input_sampling_rate_hz:.1f} Hz | Output Fs: {rat_res.output_sampling_rate_hz:.1f} Hz")
    print(f"  Ratio L/M: {rat_res.up_factor}/{rat_res.down_factor}")
    print(f"  Input Samples: {rat_res.input_num_samples} | Output Samples: {rat_res.output_num_samples}")
    print(f"  Tone 1: Target 500 Hz   | Measured Bin: {spec_rat.frequency_hz[idx_r_500]:.2f} Hz (Err: {err_freq_500:.4f} Hz) | Mag: {meas_500:.6f} V (Err: {err_amp_500:.6f} V)")
    print(f"  Tone 2: Target 1500 Hz  | Measured Bin: {spec_rat.frequency_hz[idx_r_1500]:.2f} Hz (Err: {err_freq_1500:.4f} Hz) | Mag: {meas_1500:.6f} V (Err: {err_amp_1500:.6f} V)")

    # Assert finite outputs and accuracy
    assert np.all(np.isfinite(rat_res.processed_signal.amplitude))
    assert err_freq_500 < 0.1
    assert err_freq_1500 < 0.1
    assert err_amp_500 < 0.05
    assert err_amp_1500 < 0.05

    print("\n=== Phase 5 Multirate DSP Smoke Test Completed Successfully ===")


if __name__ == "__main__":
    run_smoke_test_phase5()
