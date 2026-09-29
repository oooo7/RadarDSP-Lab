"""Phase 3 Integration Smoke Test.

Demonstrates:
1. 1 kHz bin-centered sinusoid (Fs = 10 kHz, N = 10,000)
2. 1055 Hz non-bin-centered sinusoid
3. Spectral leakage under Rectangular vs Hann vs Hamming vs Blackman windows
4. Zero-padding effect (N_sig = 1000, N_fft = 4096)
5. Sub-bin dominant-frequency estimation
6. 7 kHz tone sampled at 10 kHz producing a 3 kHz measured alias via transforms.py
"""

import numpy as np
from src.dsp.signals import generate_sine
from src.dsp.transforms import (
    analyze_spectral_leakage,
    compute_fft,
    find_dominant_frequency,
)


def run_smoke_test_phase3() -> None:
    """Run Phase 3 spectral transforms demonstration checks."""
    print("=== RadarDSP Lab — Phase 3 FFT, Spectrum Analysis & Windowing Smoke Test ===")

    fs = 10000.0

    # 1. 1 kHz Bin-Centered Sinusoid
    print("\n[1] 1 kHz Bin-Centered Sinusoid (Fs = 10 kHz, N = 10,000):")
    sig_1k = generate_sine(amplitude=2.0, frequency_hz=1000.0, phase_rad=0.0, sampling_rate_hz=fs, duration_sec=1.0)
    spec_1k = compute_fft(sig_1k, window_type="rect", one_sided=True)
    dom_1k = find_dominant_frequency(spec_1k, ignore_dc=True, interpolate_subbin=True)

    print(f"  Signal Frequency: 1000.0 Hz")
    print(f"  Sampling Frequency: {fs} Hz | N = {spec_1k.n_signal_samples}")
    print(f"  Physical Resolution: Delta f = {spec_1k.physical_resolution_hz:.2f} Hz")
    print(f"  Window: {spec_1k.window_name}")
    print(f"  Estimated Frequency: {dom_1k.frequency_hz:.4f} Hz")
    print(f"  Frequency Error: {abs(dom_1k.frequency_hz - 1000.0):.6f} Hz")
    print(f"  Peak Magnitude: {dom_1k.magnitude:.4f} V (True: 2.0 V)")

    # 2. 1055 Hz Non-Bin-Centered Sinusoid & Window Leakage Comparison
    print("\n[2] 1055 Hz Non-Bin-Centered Sinusoid (Spectral Leakage Analysis, N=1,000, Delta f = 10 Hz):")
    leakage = analyze_spectral_leakage(signal_frequency_hz=1055.0, sampling_rate_hz=fs, n_samples=1000)
    print(f"  Is Bin-Centered: {leakage.is_bin_centered}")
    print("  Window Peak Magnitudes & Sidelobe Suppression:")
    for w_name in ["rectangular", "hann", "hamming", "blackman"]:
        peak = leakage.peak_magnitudes[w_name]
        supp = leakage.sidelobe_suppression_db[w_name]
        print(f"    - {w_name.capitalize():12s} | Peak Mag: {peak:.4f} V | 2nd Peak Suppression: {supp:.2f} dB")

    # 3. Zero-Padding Effect Demonstration
    print("\n[3] Zero-Padding Effect (N_sig = 1000, N_fft = 4096):")
    sig_short = generate_sine(amplitude=1.5, frequency_hz=250.0, phase_rad=0.0, sampling_rate_hz=1000.0, duration_sec=1.0)
    spec_unpadded = compute_fft(sig_short, window_type="hann", n_fft=1000, one_sided=True)
    spec_padded = compute_fft(sig_short, window_type="hann", n_fft=4096, one_sided=True)

    print(f"  Unpadded (N_fft = 1000): Delta f_phys = {spec_unpadded.physical_resolution_hz:.2f} Hz | Bin Spacing = {spec_unpadded.bin_resolution_hz:.2f} Hz")
    print(f"  Padded   (N_fft = 4096): Delta f_phys = {spec_padded.physical_resolution_hz:.2f} Hz | Bin Spacing = {spec_padded.bin_resolution_hz:.4f} Hz")
    print("  Notice: Zero-padding increases display bin density, but physical resolution remains 1.0 Hz.")

    # 4. Cross-Phase Aliasing Measurement (7 kHz sampled at 10 kHz => 3 kHz alias)
    print("\n[4] Cross-Phase Aliasing Validation (7 kHz tone sampled at Fs = 10 kHz):")
    sig_7k = generate_sine(amplitude=1.0, frequency_hz=7000.0, phase_rad=0.0, sampling_rate_hz=10000.0, duration_sec=1.0)
    spec_7k = compute_fft(sig_7k, window_type="rect", one_sided=True)
    dom_7k = find_dominant_frequency(spec_7k, ignore_dc=True, interpolate_subbin=True)

    print(f"  Signal Frequency: 7000.0 Hz | Sampling Rate: 10000.0 Hz")
    print(f"  Theoretical Alias: 3000.00 Hz")
    print(f"  Measured Peak Alias: {dom_7k.frequency_hz:.4f} Hz")
    print(f"  Frequency Error: {abs(dom_7k.frequency_hz - 3000.0):.6f} Hz")
    print(f"  Peak Magnitude: {dom_7k.magnitude:.4f} V")

    print("\n=== Phase 3 Smoke Test Completed Successfully ===")


if __name__ == "__main__":
    run_smoke_test_phase3()
