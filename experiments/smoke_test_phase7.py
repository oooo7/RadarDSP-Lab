"""Phase 7 Multi-Target FMCW Radar + Doppler / Velocity Processing Integration Smoke Test.

Demonstrates:
1. Multi-target FMCW simulation: 3 targets at different ranges and velocities.
2. Fast-time vs Slow-time Multi-Chirp Data Cube synthesis.
3. 2D Range-Doppler matrix processing.
4. Range & Velocity candidate peak detection and resolution validation.
"""

from src.radar.doppler import (
    compute_range_doppler_map,
    extract_range_doppler_peaks,
    generate_multi_chirp_data_cube,
    validate_range_estimate,
    validate_velocity_estimate,
)
from src.utils.config import RadarConfig, TargetConfig


def run_smoke_test_phase7() -> None:
    """Run Phase 7 Multi-Target FMCW Radar + Doppler / Velocity demonstration pipeline."""
    print("==================================================")
    print("=== Phase 7 Multi-Target FMCW + Doppler Smoke Test ===")
    print("==================================================")

    # 1. Radar Configuration
    cfg = RadarConfig(
        carrier_frequency_hz=77e9,
        sweep_bandwidth_hz=150e6,
        chirp_duration_sec=50e-6,  # 50 us -> 20 kHz PRF
        sampling_rate_hz=20e6,
        num_chirps=128
    )

    # 2. Section 13 Targets Configuration:
    # Target A: 50 m, +10 m/s, amp 1.0 (receding)
    # Target B: 120 m, -5 m/s, amp 0.7 (approaching)
    # Target C: 200 m, 0 m/s, amp 0.5 (stationary)
    targets = [
        TargetConfig(range_m=50.0, velocity_mps=10.0, amplitude=1.0, target_id="T1"),
        TargetConfig(range_m=120.0, velocity_mps=-5.0, amplitude=0.7, target_id="T2"),
        TargetConfig(range_m=200.0, velocity_mps=0.0, amplitude=0.5, target_id="T3"),
    ]

    # 3. Data Cube Generation
    data_cube = generate_multi_chirp_data_cube(cfg, targets, num_chirps=128)
    rd_map = compute_range_doppler_map(data_cube)

    print("\nRadar configuration:")
    print(f"  Carrier:              {cfg.carrier_frequency_hz / 1e9:.2f} GHz")
    print(f"  Bandwidth:            {cfg.sweep_bandwidth_hz / 1e6:.2f} MHz")
    print(f"  Chirp duration:       {cfg.chirp_duration_sec * 1e6:.2f} us")
    print(f"  Sampling rate:        {cfg.sampling_rate_hz / 1e6:.2f} MHz")
    print(f"  Number of chirps:     {data_cube.num_chirps}")
    print(f"  Wavelength:           {data_cube.wavelength_m * 1000:.4f} mm")
    print(f"  Range resolution:     {rd_map.range_resolution_m:.4f} m (~1.0 m)")
    print(f"  Velocity resolution:  {rd_map.velocity_resolution_mps:.4f} m/s")
    print(f"  Unambiguous range:    {rd_map.unambiguous_range_m:.2f} m")
    print(f"  Unambiguous velocity: +/- {rd_map.unambiguous_velocity_mps:.2f} m/s")
    print(f"  Range-Doppler matrix shape: {rd_map.magnitude_matrix.shape} (doppler_bins x range_bins)")

    print("\nTargets:")
    for t_state in data_cube.targets:
        print(f"  {t_state.target_id}:")
        print(f"    Range:    {t_state.range_m:.2f} m")
        print(f"    Velocity: {t_state.velocity_mps:.2f} m/s ({'receding' if t_state.velocity_mps > 0 else 'approaching' if t_state.velocity_mps < 0 else 'stationary'})")
        print(f"    Theoretical Beat Freq:    {t_state.theoretical_beat_frequency_hz / 1e3:.2f} kHz")
        print(f"    Theoretical Doppler Freq: {t_state.theoretical_doppler_frequency_hz:.2f} Hz")

    # 4. Peak Extraction
    peaks = extract_range_doppler_peaks(rd_map, threshold_db=-25.0, max_peaks=10)

    print("\nDetected Range-Doppler Peaks:")
    for i, p in enumerate(peaks):
        t_id_str = f" [{p.target_id}]" if p.target_id else ""
        print(f"  Peak {i+1}{t_id_str}: Range = {p.range_m:7.2f} m | Velocity = {p.velocity_mps:6.2f} m/s | Mag = {p.magnitude:6.4f} ({p.magnitude_db:6.2f} dB)")

    print("\nValidation:")
    for t_state in data_cube.targets:
        matched = [p for p in peaks if abs(p.range_m - t_state.range_m) < 3.0 and abs(p.velocity_mps - t_state.velocity_mps) < 2.0]
        if matched:
            best_p = matched[0]
            r_val = validate_range_estimate(t_state.range_m, best_p.range_m, rd_map.range_resolution_m)
            v_val = validate_velocity_estimate(t_state.velocity_mps, best_p.velocity_mps, rd_map.velocity_resolution_mps)

            print(f"  {t_state.target_id} ({t_state.range_m:.1f} m, {t_state.velocity_mps:.1f} m/s):")
            print(f"    Estimated Range:    {best_p.range_m:.4f} m (Error: {r_val.absolute_error * 1000:.2f} mm | Within Cell: {r_val.is_within_resolution_cell})")
            print(f"    Estimated Velocity: {best_p.velocity_mps:.4f} m/s (Error: {v_val.absolute_error:.4f} m/s | Within Cell: {v_val.is_within_resolution_cell})")
        else:
            print(f"  {t_state.target_id}: FAILED TO DETECT PEAK NEAR TARGET REGION!")

    print("\nExpected Target Regions Detection Status:")
    all_detected = len(peaks) >= 3 and all(
        any(abs(p.range_m - t.range_m) < 3.0 and abs(p.velocity_mps - t.velocity_mps) < 2.0 for p in peaks)
        for t in data_cube.targets
    )
    print(f"  All 3 distinct target regions successfully resolved and detected: {all_detected}")

    print("\n=== Phase 7 Multi-Target FMCW + Doppler Smoke Test Completed Successfully ===")


if __name__ == "__main__":
    run_smoke_test_phase7()
