"""Phase 2 Integration Smoke Test.

Demonstrates full Phase 2 Sampling & Aliasing Pipeline:
1. Continuous / ideal reference signal generation (700 Hz tone sampled at high-resolution Fs_ref = 20 kHz).
2. Discrete uniform sampling at target_fs = 1 kHz (Undersampled scenario).
3. Sampled signal container payload verification.
4. Nyquist rate analysis (f_max = 700 Hz, Nyquist rate = 1400 Hz, Fs = 1000 Hz => Undersampled).
5. Theoretical alias calculation (700 Hz aliases to 300 Hz in [0, 500] Hz).
6. Measured alias frequency from FFT magnitude spectrum via parabolic peak interpolation.
7. Error calculation (|f_measured - f_theoretical| and relative error).
"""

from src.dsp.sampling import sample_signal
from src.dsp.signals import generate_sine


def run_smoke_test_phase2() -> None:
    """Run Phase 2 Sampling & Aliasing demonstration pipeline."""
    print("=== RadarDSP Lab — Phase 2 Sampling & Aliasing Pipeline Smoke Test ===")

    # Step 1: Continuous / ideal reference signal (700 Hz tone at high Fs_ref = 20,000 Hz)
    f_tone = 700.0
    fs_ref = 20000.0
    duration = 1.0

    print(f"\n[Step 1: Continuous/Ideal Signal] Generating {f_tone} Hz tone reference signal (Fs_ref = {fs_ref} Hz)...")
    continuous_sig = generate_sine(
        amplitude=1.0,
        frequency_hz=f_tone,
        phase_rad=0.0,
        sampling_rate_hz=fs_ref,
        duration_sec=duration
    )
    print(f"  Reference signal samples: {len(continuous_sig.amplitude)}")

    # Step 2: Sampling at target_fs = 1000 Hz
    target_fs = 1000.0
    print(f"\n[Step 2 & 3: Sampling & Sampled Signal] Sampling reference signal at target Fs = {target_fs} Hz...")
    result = sample_signal(continuous_signal=continuous_sig, target_fs_hz=target_fs)

    print(f"  Sampled signal samples: {len(result.sampled_signal.amplitude)}")

    # Step 4: Nyquist Analysis
    print(f"\n[Step 4: Nyquist Analysis]")
    print(f"  Max Signal Frequency (f_max): {result.metadata['f_max_hz']} Hz")
    print(f"  Nyquist Rate (2 * f_max): {result.nyquist_rate_hz} Hz")
    print(f"  Folding Frequency (Fs / 2): {result.folding_frequency_hz} Hz")
    print(f"  Is Undersampled: {result.is_undersampled}")
    print(f"  Is Aliased: {result.is_aliased}")

    # Step 5 & 6: Theoretical vs Measured Alias Frequency
    theo_alias = result.theoretical_alias_frequencies_hz[0]
    meas_alias = result.measured_alias_frequencies_hz[0]
    abs_err = result.absolute_errors_hz[0]
    rel_err = result.relative_errors[0]

    print(f"\n[Step 5: Aliasing Calculation & Theoretical Alias Frequency]")
    print(f"  Input Tone Frequency: {f_tone} Hz")
    print(f"  Theoretical Alias Frequency: {theo_alias:.2f} Hz")

    print(f"\n[Step 6: Measured Alias Frequency]")
    print(f"  Measured Spectral Peak Alias Frequency: {meas_alias:.4f} Hz")

    print(f"\n[Step 7: Error Calculation]")
    print(f"  Absolute Error: {abs_err:.6f} Hz")
    print(f"  Relative Error: {rel_err * 100:.4f}%")
    print(f"  Overall MAE: {result.metadata['mae_hz']:.6f} Hz")
    print(f"  Overall RMSE: {result.metadata['rmse_hz']:.6f} Hz")

    print("\n=== Phase 2 Smoke Test Completed Successfully ===")


if __name__ == "__main__":
    run_smoke_test_phase2()
