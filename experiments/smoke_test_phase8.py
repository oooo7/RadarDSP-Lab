"""Phase 8 Radar Noise, Statistical Clutter & 2D CA-CFAR Detection Engine Integration Smoke Test.

Demonstrates:
1. Generation of Phase 7 multi-target data cube (T1: 50m/+10m/s, T2: 120m/-5m/s, T3: 200m/0m/s).
2. Addition of complex AWGN noise (SNR = 20.0 dB) & complex Gaussian background clutter (Power = 0.2).
3. 2D Range-Doppler matrix processing.
4. 2D Cell-Averaging CFAR (CA-CFAR) adaptive thresholding and automatic target detection.
"""

from src.radar.cfar import CFARResult, run_ca_cfar
from src.radar.doppler import RadarDataCube, compute_range_doppler_map, generate_multi_chirp_data_cube
from src.radar.noise import compose_radar_environment
from src.utils.config import CFARConfig, RadarConfig, TargetConfig


def run_smoke_test_phase8() -> None:
    """Run Phase 8 Noise, Clutter & CA-CFAR demonstration pipeline."""
    print("==================================================")
    print("=== Phase 8 Noise, Clutter & CA-CFAR Smoke Test ===")
    print("==================================================")

    # 1. Radar & Multi-Target Configuration
    cfg = RadarConfig(
        carrier_frequency_hz=77e9,
        sweep_bandwidth_hz=150e6,
        chirp_duration_sec=50e-6,
        sampling_rate_hz=20e6,
        num_chirps=128
    )

    targets = [
        TargetConfig(range_m=50.0, velocity_mps=10.0, amplitude=1.0, target_id="T1"),
        TargetConfig(range_m=120.0, velocity_mps=-5.0, amplitude=0.7, target_id="T2"),
        TargetConfig(range_m=200.0, velocity_mps=0.0, amplitude=0.5, target_id="T3"),
    ]

    clean_cube = generate_multi_chirp_data_cube(cfg, targets, num_chirps=128)

    # 2. Environmental Signal Composition: Clean Signal + Clutter + AWGN Noise
    requested_snr_db = 20.0
    clutter_pwr = 0.2

    comp_env = compose_radar_environment(
        clean_signal=clean_cube.data,
        snr_db=requested_snr_db,
        clutter_power=clutter_pwr,
        seed=42
    )

    # Create Noisy/Cluttered Data Cube payload
    noisy_cube = RadarDataCube(
        data=comp_env.composite_signal,
        time_vector=clean_cube.time_vector,
        chirp_container=clean_cube.chirp_container,
        num_chirps=clean_cube.num_chirps,
        samples_per_chirp=clean_cube.samples_per_chirp,
        sampling_rate_hz=clean_cube.sampling_rate_hz,
        chirp_duration_sec=clean_cube.chirp_duration_sec,
        carrier_frequency_hz=clean_cube.carrier_frequency_hz,
        bandwidth_hz=clean_cube.bandwidth_hz,
        targets=clean_cube.targets
    )

    print("\nEnvironment & Noise Model Parameters:")
    print(f"  Target Signal Power:       {comp_env.signal_power:.4f}")
    print(f"  Clutter Power (Target):    {clutter_pwr:.4f} (Measured: {comp_env.clutter_power:.4f})")
    print(f"  Requested SNR:             {requested_snr_db:.2f} dB")
    print(f"  Measured SNR:              {comp_env.snr_db:.2f} dB")
    print(f"  Total Interference Power:  {comp_env.total_interference_power:.4f}")

    # 3. 2D Range-Doppler FFT Spectrum
    rd_map = compute_range_doppler_map(noisy_cube)
    print(f"\nRange-Doppler Matrix Dimensions: {rd_map.magnitude_matrix.shape} (doppler_bins x range_bins)")

    # 4. 2D CA-CFAR Adaptive Thresholding
    cfar_cfg = CFARConfig(
        method="CA-CFAR",
        num_guard_range=2,
        num_guard_doppler=2,
        num_train_range=4,
        num_train_doppler=4,
        pfa=1e-4
    )

    cfar_res = run_ca_cfar(rd_map, cfar_cfg=cfar_cfg)

    print("\nCA-CFAR Configuration & Parameters:")
    print(f"  CFAR Algorithm:            {cfar_cfg.method}")
    print(f"  Probability of False Alarm:{cfar_res.pfa:.1e}")
    print(f"  Training Cells (N_train):  {cfar_res.training_cell_count} cells")
    print(f"  CFAR Multiplier Alpha:     {cfar_res.alpha:.4f}")
    print(f"  Guard Window Geometry:     {cfar_cfg.num_guard_doppler}x{cfar_cfg.num_guard_range} per side")
    print(f"  Training Window Geometry:  {cfar_cfg.num_train_doppler}x{cfar_cfg.num_train_range} per side")
    print(f"  Total Automatic Detections:{cfar_res.num_detections}")

    print("\nAutomatic Target Detections Extracted by CA-CFAR:")
    for i, d in enumerate(cfar_res.detections):
        # Match with theoretical target if within range/velocity resolution bounds
        matched_id = None
        for t_state in clean_cube.targets:
            if abs(d.range_m - t_state.range_m) < 3.0 and abs(d.velocity_mps - t_state.velocity_mps) < 2.0:
                matched_id = t_state.target_id
                break

        t_id_str = f" [{matched_id}]" if matched_id else " [Noise/Clutter FA]"
        print(
            f"  Det {i+1:2d}{t_id_str:20s}: Range = {d.range_m:7.2f} m | "
            f"Velocity = {d.velocity_mps:6.2f} m/s | "
            f"Power = {d.power_db:6.2f} dB | "
            f"Threshold = {d.threshold_db:6.2f} dB | "
            f"Local SNR = {d.snr_db:5.2f} dB"
        )

    print("\n=== Phase 8 Noise, Clutter & CA-CFAR Smoke Test Completed Successfully ===")


if __name__ == "__main__":
    run_smoke_test_phase8()
