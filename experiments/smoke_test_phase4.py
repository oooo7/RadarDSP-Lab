"""Phase 4 Integration Smoke Test.

Demonstrates:
1. Dual-tone signal: 500 Hz + 3000 Hz at Fs = 10000 Hz.
2. FIR Lowpass filtering (cutoff = 1000 Hz, order = 64).
3. IIR Butterworth Lowpass filtering (cutoff = 1000 Hz, order = 6).
4. FFT spectral verification before and after filtering.
"""

import numpy as np
from src.dsp.filtering import (
    analyze_filter_response,
    apply_filter,
    design_fir_filter,
    design_iir_butterworth,
)
from src.dsp.signals import generate_multitone
from src.dsp.transforms import compute_fft


def run_smoke_test_phase4() -> None:
    """Run Phase 4 Digital Filtering demonstration & numerical audit pipeline."""
    print("=== RadarDSP Lab — Phase 4 Digital Filtering Engine Smoke Test & Numerical Audit ===")

    fs = 10000.0
    cutoff = 1000.0
    duration = 2.0  # 2.0s duration to allow initial transient cropping

    # 1. Generate Input Signal: 500 Hz (passband) + 3000 Hz (stopband)
    print(f"\n[Step 1: Input Signal] Multi-tone 500 Hz (1.0 V) + 3000 Hz (1.0 V) at Fs = {fs} Hz...")
    sig = generate_multitone(
        frequencies_hz=[500.0, 3000.0],
        amplitudes=[1.0, 1.0],
        phases_rad=[0.0, 0.0],
        sampling_rate_hz=fs,
        duration_sec=duration
    )

    spec_in = compute_fft(sig, window_type="rect", one_sided=True)
    idx_500 = int(np.round(500.0 / spec_in.bin_resolution_hz))
    idx_1000 = int(np.round(1000.0 / spec_in.bin_resolution_hz))
    idx_3000 = int(np.round(3000.0 / spec_in.bin_resolution_hz))

    in_mag_500 = spec_in.magnitude[idx_500]
    in_mag_3000 = spec_in.magnitude[idx_3000]

    print(f"  Input 500 Hz Peak Magnitude:  {in_mag_500:.6f} V")
    print(f"  Input 3000 Hz Peak Magnitude: {in_mag_3000:.6f} V")

    crop_samples = int(0.2 * fs)  # Crop first 0.2s to isolate steady-state response

    # 2. FIR Lowpass Filter Analysis
    print(f"\n[Step 2: FIR Lowpass Filter] Order=64, Cutoff={cutoff} Hz, Window=Hamming...")
    fir_filt = design_fir_filter("lowpass", cutoff_hz=cutoff, order=64, sampling_rate_hz=fs, window_type="hamming")
    fir_resp = analyze_filter_response(fir_filt, n_points=5000)

    # Theoretical sosfreqz/freqz values
    idx_resp_500 = int(np.argmin(np.abs(fir_resp.frequency_hz - 500.0)))
    idx_resp_1000 = int(np.argmin(np.abs(fir_resp.frequency_hz - 1000.0)))
    idx_resp_3000 = int(np.argmin(np.abs(fir_resp.frequency_hz - 3000.0)))

    fir_th_500_mag = fir_resp.magnitude[idx_resp_500]
    fir_th_500_db = fir_resp.magnitude_db[idx_resp_500]
    fir_th_1000_mag = fir_resp.magnitude[idx_resp_1000]
    fir_th_1000_db = fir_resp.magnitude_db[idx_resp_1000]
    fir_th_3000_mag = fir_resp.magnitude[idx_resp_3000]
    fir_th_3000_db = fir_resp.magnitude_db[idx_resp_3000]

    print("  [FIR Theoretical freqz Response]")
    print(f"    500 Hz:  {fir_th_500_mag:.6f} V ({fir_th_500_db:.2f} dB)")
    print(f"    1000 Hz: {fir_th_1000_mag:.6f} V ({fir_th_1000_db:.2f} dB) [Note: firwin design cutoff is -6 dB]")
    print(f"    3000 Hz: {fir_th_3000_mag:.8f} V ({fir_th_3000_db:.2f} dB)")

    # Causal filtering (steady state)
    fir_out_causal = apply_filter(sig, fir_filt, zero_phase=False)
    spec_fir_c = compute_fft(fir_out_causal.amplitude[crop_samples:], sampling_rate_hz=fs, window_type="rect", one_sided=True)
    idx_500_c = int(np.round(500.0 / spec_fir_c.bin_resolution_hz))
    idx_3000_c = int(np.round(3000.0 / spec_fir_c.bin_resolution_hz))
    fir_c_500 = spec_fir_c.magnitude[idx_500_c]
    fir_c_3000 = spec_fir_c.magnitude[idx_3000_c]
    fir_c_500_db = 20.0 * np.log10(max(fir_c_500 / in_mag_500, 1e-12))
    fir_c_3000_db = 20.0 * np.log10(max(fir_c_3000 / in_mag_3000, 1e-12))

    print("  [FIR Causal Steady-State Time-Domain FFT Measurement]")
    print(f"    500 Hz:  {fir_c_500:.6f} V ({fir_c_500_db:.2f} dB) vs Theory: {fir_th_500_db:.2f} dB")
    print(f"    3000 Hz: {fir_c_3000:.8f} V ({fir_c_3000_db:.2f} dB) vs Theory: {fir_th_3000_db:.2f} dB")

    # Zero-phase filtering
    fir_out_zp = apply_filter(sig, fir_filt, zero_phase=True)
    spec_fir_zp = compute_fft(fir_out_zp, window_type="rect", one_sided=True)
    fir_zp_500 = spec_fir_zp.magnitude[idx_500]
    fir_zp_3000 = spec_fir_zp.magnitude[idx_3000]
    fir_zp_500_db = 20.0 * np.log10(max(fir_zp_500 / in_mag_500, 1e-12))
    fir_zp_3000_db = 20.0 * np.log10(max(fir_zp_3000 / in_mag_3000, 1e-12))

    print("  [FIR Zero-Phase Time-Domain FFT Measurement]")
    print(f"    500 Hz:  {fir_zp_500:.6f} V ({fir_zp_500_db:.2f} dB) vs Theory |H(f)|^2: {2.0 * fir_th_500_db:.2f} dB")
    print(f"    3000 Hz: {fir_zp_3000:.8f} V ({fir_zp_3000_db:.2f} dB) [Dynamic Floor from zero-phase boundary transient]")
    if fir_resp.cutoff_3db_hz:
        print(f"  FIR Measured -3 dB Cutoff: {fir_resp.cutoff_3db_hz[0]:.2f} Hz (Actual finite response -3 dB point)")

    # 3. IIR Butterworth Lowpass Filter Analysis
    print(f"\n[Step 3: IIR Butterworth Lowpass Filter] Order=6, Cutoff={cutoff} Hz (SOS Representation)...")
    iir_filt = design_iir_butterworth("lowpass", cutoff_hz=cutoff, order=6, sampling_rate_hz=fs)
    iir_resp = analyze_filter_response(iir_filt, n_points=5000)

    idx_resp_500 = int(np.argmin(np.abs(iir_resp.frequency_hz - 500.0)))
    idx_resp_1000 = int(np.argmin(np.abs(iir_resp.frequency_hz - 1000.0)))
    idx_resp_3000 = int(np.argmin(np.abs(iir_resp.frequency_hz - 3000.0)))

    iir_th_500_mag = iir_resp.magnitude[idx_resp_500]
    iir_th_500_db = iir_resp.magnitude_db[idx_resp_500]
    iir_th_1000_mag = iir_resp.magnitude[idx_resp_1000]
    iir_th_1000_db = iir_resp.magnitude_db[idx_resp_1000]
    iir_th_3000_mag = iir_resp.magnitude[idx_resp_3000]
    iir_th_3000_db = iir_resp.magnitude_db[idx_resp_3000]

    print("  [IIR Theoretical sosfreqz Response]")
    print(f"    500 Hz:  {iir_th_500_mag:.6f} V ({iir_th_500_db:.2f} dB)")
    print(f"    1000 Hz: {iir_th_1000_mag:.6f} V ({iir_th_1000_db:.2f} dB) [Butterworth -3 dB cutoff]")
    print(f"    3000 Hz: {iir_th_3000_mag:.8f} V ({iir_th_3000_db:.2f} dB)")

    # Causal filtering (steady state)
    iir_out_causal = apply_filter(sig, iir_filt, zero_phase=False)
    spec_iir_c = compute_fft(iir_out_causal.amplitude[crop_samples:], sampling_rate_hz=fs, window_type="rect", one_sided=True)
    idx_iir_500_c = int(np.round(500.0 / spec_iir_c.bin_resolution_hz))
    idx_iir_3000_c = int(np.round(3000.0 / spec_iir_c.bin_resolution_hz))
    iir_c_500 = spec_iir_c.magnitude[idx_iir_500_c]
    iir_c_3000 = spec_iir_c.magnitude[idx_iir_3000_c]
    iir_c_500_db = 20.0 * np.log10(max(iir_c_500 / in_mag_500, 1e-12))
    iir_c_3000_db = 20.0 * np.log10(max(iir_c_3000 / in_mag_3000, 1e-12))

    print("  [IIR Causal Steady-State Time-Domain FFT Measurement]")
    print(f"    500 Hz:  {iir_c_500:.6f} V ({iir_c_500_db:.2f} dB) vs Theory: {iir_th_500_db:.2f} dB")
    print(f"    3000 Hz: {iir_c_3000:.8f} V ({iir_c_3000_db:.2f} dB) vs Theory: {iir_th_3000_db:.2f} dB")

    # Zero-phase filtering
    iir_out_zp = apply_filter(sig, iir_filt, zero_phase=True)
    spec_iir_zp = compute_fft(iir_out_zp, window_type="rect", one_sided=True)
    iir_zp_500 = spec_iir_zp.magnitude[idx_500]
    iir_zp_3000 = spec_iir_zp.magnitude[idx_3000]
    iir_zp_500_db = 20.0 * np.log10(max(iir_zp_500 / in_mag_500, 1e-12))
    iir_zp_3000_db = 20.0 * np.log10(max(iir_zp_3000 / in_mag_3000, 1e-12))

    print("  [IIR Zero-Phase Time-Domain FFT Measurement]")
    print(f"    500 Hz:  {iir_zp_500:.6f} V ({iir_zp_500_db:.2f} dB) vs Theory |H(f)|^2: {2.0 * iir_th_500_db:.2f} dB")
    print(f"    3000 Hz: {iir_zp_3000:.8f} V ({iir_zp_3000_db:.2f} dB) [Dynamic Floor from zero-phase boundary transient]")
    if iir_resp.cutoff_3db_hz:
        print(f"  IIR Measured -3 dB Cutoff: {iir_resp.cutoff_3db_hz[0]:.2f} Hz (Matches design cutoff 1000 Hz)")

    # 4. Finite Outputs & Consistency Verification
    assert np.all(np.isfinite(fir_out_causal.amplitude))
    assert np.all(np.isfinite(iir_out_causal.amplitude))
    assert np.all(np.isfinite(fir_out_zp.amplitude))
    assert np.all(np.isfinite(iir_out_zp.amplitude))

    # Causal FFT measurements match theoretical frequency response within 0.05 dB
    assert abs(fir_c_500_db - fir_th_500_db) < 0.05
    assert abs(fir_c_3000_db - fir_th_3000_db) < 0.05
    assert abs(iir_c_500_db - iir_th_500_db) < 0.05
    assert abs(iir_c_3000_db - iir_th_3000_db) < 0.05

    print("\n=== Phase 4 Smoke Test & Numerical Audit Completed Successfully ===")


if __name__ == "__main__":
    run_smoke_test_phase4()

