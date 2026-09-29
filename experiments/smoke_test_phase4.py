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
    """Run Phase 4 Digital Filtering demonstration pipeline."""
    print("=== RadarDSP Lab — Phase 4 Digital Filtering Engine Smoke Test ===")

    fs = 10000.0
    cutoff = 1000.0

    # 1. Generate Input Signal: 500 Hz (passband) + 3000 Hz (stopband)
    print(f"\n[Step 1: Input Signal] Multi-tone 500 Hz (1.0 V) + 3000 Hz (1.0 V) at Fs = {fs} Hz...")
    sig = generate_multitone(
        frequencies_hz=[500.0, 3000.0],
        amplitudes=[1.0, 1.0],
        phases_rad=[0.0, 0.0],
        sampling_rate_hz=fs,
        duration_sec=1.0
    )

    spec_in = compute_fft(sig, window_type="rect", one_sided=True)
    idx_500 = int(np.round(500.0 / spec_in.bin_resolution_hz))
    idx_3000 = int(np.round(3000.0 / spec_in.bin_resolution_hz))

    in_mag_500 = spec_in.magnitude[idx_500]
    in_mag_3000 = spec_in.magnitude[idx_3000]

    print(f"  Input 500 Hz Peak Magnitude: {in_mag_500:.4f} V")
    print(f"  Input 3000 Hz Peak Magnitude: {in_mag_3000:.4f} V")

    # 2. FIR Lowpass Filtering
    print(f"\n[Step 2: FIR Lowpass Filter] Order=64, Cutoff={cutoff} Hz, Window=Hamming...")
    fir_filt = design_fir_filter("lowpass", cutoff_hz=cutoff, order=64, sampling_rate_hz=fs, window_type="hamming")
    fir_resp = analyze_filter_response(fir_filt)

    fir_out = apply_filter(sig, fir_filt, zero_phase=True)
    spec_fir = compute_fft(fir_out, window_type="rect", one_sided=True)

    fir_mag_500 = spec_fir.magnitude[idx_500]
    fir_mag_3000 = spec_fir.magnitude[idx_3000]
    att_500_fir_db = 20.0 * np.log10(max(fir_mag_500 / in_mag_500, 1e-12))
    att_3000_fir_db = 20.0 * np.log10(max(fir_mag_3000 / in_mag_3000, 1e-12))

    print(f"  FIR Output 500 Hz Magnitude: {fir_mag_500:.4f} V (Gain: {att_500_fir_db:.2f} dB)")
    print(f"  FIR Output 3000 Hz Magnitude: {fir_mag_3000:.6f} V (Attenuation: {att_3000_fir_db:.2f} dB)")
    print(f"  FIR Filter Family: {fir_filt.filter_family.upper()} | Taps: {fir_filt.n_taps} | Linear Phase: {fir_filt.metadata['is_linear_phase']}")
    if fir_resp.cutoff_3db_hz:
        print(f"  FIR Measured -3 dB Cutoff: {fir_resp.cutoff_3db_hz[0]:.2f} Hz")

    # 3. IIR Butterworth Lowpass Filtering
    print(f"\n[Step 3: IIR Butterworth Lowpass Filter] Order=6, Cutoff={cutoff} Hz (SOS Representation)...")
    iir_filt = design_iir_butterworth("lowpass", cutoff_hz=cutoff, order=6, sampling_rate_hz=fs)
    iir_resp = analyze_filter_response(iir_filt)

    iir_out = apply_filter(sig, iir_filt, zero_phase=True)
    spec_iir = compute_fft(iir_out, window_type="rect", one_sided=True)

    iir_mag_500 = spec_iir.magnitude[idx_500]
    iir_mag_3000 = spec_iir.magnitude[idx_3000]
    att_500_iir_db = 20.0 * np.log10(max(iir_mag_500 / in_mag_500, 1e-12))
    att_3000_iir_db = 20.0 * np.log10(max(iir_mag_3000 / in_mag_3000, 1e-12))

    print(f"  IIR Output 500 Hz Magnitude: {iir_mag_500:.4f} V (Gain: {att_500_iir_db:.2f} dB)")
    print(f"  IIR Output 3000 Hz Magnitude: {iir_mag_3000:.6f} V (Attenuation: {att_3000_iir_db:.2f} dB)")
    print(f"  IIR Filter Family: {iir_filt.filter_family.upper()} | Order: {iir_filt.order} | SOS Stable: {iir_filt.metadata['is_stable_sos']}")
    if iir_resp.cutoff_3db_hz:
        print(f"  IIR Measured -3 dB Cutoff: {iir_resp.cutoff_3db_hz[0]:.2f} Hz")

    # 4. Finite Outputs Verification
    assert np.all(np.isfinite(fir_out.amplitude))
    assert np.all(np.isfinite(iir_out.amplitude))

    print("\n=== Phase 4 Smoke Test Completed Successfully ===")


if __name__ == "__main__":
    run_smoke_test_phase4()
