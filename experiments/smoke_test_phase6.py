"""Phase 6 FMCW Radar Range Processing Engine Integration Smoke Test.

Demonstrates:
1. FMCW Chirp Generation: 77 GHz carrier, 150 MHz bandwidth, 100 us duration, 10 MHz sampling rate.
2. Target Echo & Propagation Delay: 50 m, 100 m, 150 m targets.
3. Dechirp Mixing & 1D Range FFT Estimation.
4. Zero-Padding Grid Density vs. Fundamental Range Resolution Analysis.
"""

import numpy as np
from src.radar.chirp import generate_fmcw_chirp
from src.radar.processing import estimate_range
from src.radar.propagation import simulate_target_echo
from src.utils.config import RadarConfig, TargetConfig


def run_smoke_test_phase6() -> None:
    """Run Phase 6 FMCW Radar Range Processing demonstration pipeline."""
    print("==================================================")
    print("RadarDSP Lab — Phase 6 FMCW Radar Range Engine")
    print("==================================================")

    # 1. Radar Configuration
    cfg = RadarConfig(
        carrier_frequency_hz=77e9,
        sweep_bandwidth_hz=150e6,
        chirp_duration_sec=100e-6,
        sampling_rate_hz=10e6
    )

    chirp = generate_fmcw_chirp(cfg)

    print("\nRadar Configuration:")
    print(f"  Carrier Frequency: {cfg.carrier_frequency_hz / 1e9:.2f} GHz")
    print(f"  Bandwidth:         {cfg.sweep_bandwidth_hz / 1e6:.2f} MHz")
    print(f"  Chirp Duration:    {cfg.chirp_duration_sec * 1e6:.2f} us")
    print(f"  Sampling Rate:     {cfg.sampling_rate_hz / 1e6:.2f} MHz")
    print(f"  Chirp Slope:       {chirp.chirp_slope_hz_per_sec:.2e} Hz/s")
    print(f"  Range Resolution:  {chirp.range_resolution_m:.4f} m (~1.0 m)")

    target_ranges = [50.0, 100.0, 150.0]

    for r_target in target_ranges:
        print("\n------------------------------------------")
        print(f"TARGET: {r_target:.1f} m")
        print("------------------------------------------")
        target_cfg = TargetConfig(range_m=r_target, velocity_mps=0.0, rcs_sqm=1.0)
        payload = simulate_target_echo(chirp, target_cfg)

        print(f"  Target Range:               {payload.target_state.range_m:.2f} m")
        print(f"  Round Trip Delay:           {payload.round_trip_delay_sec * 1e6:.6f} us")
        print(f"  Theoretical Beat Frequency: {payload.target_state.theoretical_beat_frequency_hz:.2f} Hz")

        print("\n------------------------------------------")
        print("RANGE PROCESSING")
        print("------------------------------------------")
        res = estimate_range(payload, chirp, window_name="hann")

        print("  Window:                 Hann")
        print(f"  FFT Size:               {res.metadata['n_fft_samples']}")
        print(f"  Measured Beat Frequency: {res.beat_frequency_hz:.2f} Hz")
        print(f"  Estimated Range:        {res.estimated_range_m:.4f} m")
        print(f"  Range Error:            {res.range_error_m:.4f} m ({res.range_error_m * 1000:.2f} mm)")
        print(f"  Relative Range Error:   {res.relative_range_error * 100.0:.4f} %")

    # Zero-Padding Comparison for 100 m Target
    print("\n==================================================")
    print("ZERO-PADDING IMPACT ANALYSIS (100 m Target)")
    print("==================================================")
    target_100 = TargetConfig(range_m=100.0, velocity_mps=0.0, rcs_sqm=1.0)
    payload_100 = simulate_target_echo(chirp, target_100)

    for n_fft_val in [1000, 2000, 4000, 8000]:
        res_zp = estimate_range(payload_100, chirp, window_name="hann", n_fft=n_fft_val)
        print(
            f"  N_fft: {n_fft_val:4d} | "
            f"Freq Bin Spacing: {res_zp.fft_bin_spacing_hz:8.2f} Hz | "
            f"Range Bin Spacing: {res_zp.range_bin_spacing_m:6.4f} m | "
            f"Estimated Range: {res_zp.estimated_range_m:8.4f} m | "
            f"Error: {res_zp.range_error_m * 1000:5.2f} mm"
        )

    print("\nExplanation:")
    print("  Zero-padding (N_fft > N_signal) interpolates the continuous spectral envelope,")
    print("  reducing bin quantization spacing (Delta R_bin = c * Delta f / (2*S)). However, zero-padding")
    print("  does NOT alter the fundamental physical range resolution limit (Delta R = c / (2*B) = 0.9993 m),")
    print("  which is strictly governed by the transmitted sweep bandwidth B.")

    print("\n=== Phase 6 FMCW Radar Range Engine Smoke Test Completed Successfully ===")


if __name__ == "__main__":
    run_smoke_test_phase6()
