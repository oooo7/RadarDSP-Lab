"""Phase 9 Validation, Error Analysis & Monte Carlo Experiment Orchestrator.

Executes complete scientific evaluation of RadarDSP Lab:
1. Range estimation accuracy experiment.
2. Velocity estimation accuracy experiment.
3. SNR sensitivity sweep (extended down to -40 dB).
4. Empirical false-alarm rate evaluation.
5. Multi-target Monte Carlo evaluation (100 trials).

Outputs clear text tables to stdout and generates 4 scientific plots in experiments/.
"""

import os
import time
import matplotlib.pyplot as plt
import numpy as np

from src.validation.experiments import (
    run_false_alarm_experiment,
    run_multi_target_monte_carlo_experiment,
    run_range_accuracy_experiment,
    run_snr_sweep_experiment,
    run_velocity_accuracy_experiment,
)


def main():
    print("==========================================================================================================")
    print("                      RADARDSP LAB — PHASE 9 SCIENTIFIC AUDIT & MONTE CARLO EXPERIMENTS                   ")
    print("==========================================================================================================")
    start_total_time = time.time()
    os.makedirs("experiments", exist_ok=True)

    # ------------------------------------------------------------------
    # 1. RANGE ACCURACY EXPERIMENT
    # ------------------------------------------------------------------
    print("\n----------------------------------------------------------------------------------------------------------")
    print(" 1. RANGE ESTIMATION ACCURACY EXPERIMENT")
    print("----------------------------------------------------------------------------------------------------------")
    range_res = run_range_accuracy_experiment()
    print(f"Physical Range Resolution Delta_R: {range_res.range_resolution_m:.4f} m")
    print(f"Max Unambiguous Range R_max:       {range_res.unambiguous_range_m:.2f} m\n")

    print(f"{'True Range (m)':<15} | {'Estimated (m)':<15} | {'Abs Error (m)':<15} | {'Rel Error (%)':<15}")
    print("-" * 68)
    for p in range_res.points:
        print(f"{p.true_range_m:<15.2f} | {p.estimated_range_m:<15.4f} | {p.abs_error_m:<15.4f} | {p.rel_error*100:<15.4f}")

    print("-" * 68)
    print(f"Mean Error: {range_res.aggregate_stats.mean_error:.4f} m | MAE: {range_res.aggregate_stats.mean_absolute_error:.4f} m | RMSE: {range_res.aggregate_stats.rmse:.4f} m | StdDev: {range_res.aggregate_stats.std_dev:.4f} m")

    # Plot 1: Range Accuracy
    fig, ax1 = plt.subplots(figsize=(8, 5))
    true_r = [p.true_range_m for p in range_res.points]
    est_r = [p.estimated_range_m for p in range_res.points]
    abs_e_r = [p.abs_error_m for p in range_res.points]

    ax1.plot(true_r, est_r, 'o-', color='#1f77b4', linewidth=2, label='Estimated Range (m)')
    ax1.plot(true_r, true_r, '--', color='gray', linewidth=1.5, label='Theoretical Ideal')
    ax1.set_xlabel('True Target Range (m)')
    ax1.set_ylabel('Estimated Range (m)', color='#1f77b4')
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax2 = ax1.twinx()
    ax2.plot(true_r, abs_e_r, 's--', color='#d62728', linewidth=1.5, label='Absolute Error (m)')
    ax2.set_ylabel('Absolute Range Error (m)', color='#d62728')
    ax2.tick_params(axis='y', labelcolor='#d62728')

    plt.title('Phase 9 — Range Estimation Accuracy vs True Target Range')
    fig.tight_layout()
    plot1_path = os.path.join("experiments", "phase9_range_accuracy.png")
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f"-> Plot saved to {plot1_path}")

    # ------------------------------------------------------------------
    # 2. VELOCITY ACCURACY EXPERIMENT
    # ------------------------------------------------------------------
    print("\n----------------------------------------------------------------------------------------------------------")
    print(" 2. VELOCITY ESTIMATION ACCURACY EXPERIMENT")
    print("----------------------------------------------------------------------------------------------------------")
    vel_res = run_velocity_accuracy_experiment()
    print(f"Velocity Resolution Delta_v:       {vel_res.velocity_resolution_mps:.4f} m/s")
    print(f"Max Unambiguous Velocity v_max:    ±{vel_res.unambiguous_velocity_mps:.2f} m/s\n")

    print(f"{'True Velocity (m/s)':<20} | {'Estimated (m/s)':<20} | {'Abs Error (m/s)':<18} | {'Rel Error (%)':<15}")
    print("-" * 80)
    for p in vel_res.points:
        print(f"{p.true_velocity_mps:<20.2f} | {p.estimated_velocity_mps:<20.4f} | {p.abs_error_mps:<18.4f} | {p.rel_error*100:<15.4f}")

    print("-" * 80)
    print(f"Mean Error: {vel_res.aggregate_stats.mean_error:.4f} m/s | MAE: {vel_res.aggregate_stats.mean_absolute_error:.4f} m/s | RMSE: {vel_res.aggregate_stats.rmse:.4f} m/s | StdDev: {vel_res.aggregate_stats.std_dev:.4f} m/s")

    # Plot 2: Velocity Accuracy
    fig, ax1 = plt.subplots(figsize=(8, 5))
    true_v = [p.true_velocity_mps for p in vel_res.points]
    est_v = [p.estimated_velocity_mps for p in vel_res.points]
    abs_e_v = [p.abs_error_mps for p in vel_res.points]

    ax1.plot(true_v, est_v, 'o-', color='#2ca02c', linewidth=2, label='Estimated Velocity (m/s)')
    ax1.plot(true_v, true_v, '--', color='gray', linewidth=1.5, label='Theoretical Ideal')
    ax1.set_xlabel('True Target Velocity (m/s)')
    ax1.set_ylabel('Estimated Velocity (m/s)', color='#2ca02c')
    ax1.tick_params(axis='y', labelcolor='#2ca02c')
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax2 = ax1.twinx()
    ax2.plot(true_v, abs_e_v, 's--', color='#ff7f0e', linewidth=1.5, label='Absolute Error (m/s)')
    ax2.set_ylabel('Absolute Velocity Error (m/s)', color='#ff7f0e')
    ax2.tick_params(axis='y', labelcolor='#ff7f0e')

    plt.title('Phase 9 — Velocity Estimation Accuracy vs True Velocity')
    fig.tight_layout()
    plot2_path = os.path.join("experiments", "phase9_velocity_accuracy.png")
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print(f"-> Plot saved to {plot2_path}")

    # ------------------------------------------------------------------
    # 3. SNR SENSITIVITY SWEEP
    # ------------------------------------------------------------------
    print("\n----------------------------------------------------------------------------------------------------------")
    print(" 3. SNR SENSITIVITY SWEEP EXPERIMENT (50 TRIALS / SNR STEP)")
    print("----------------------------------------------------------------------------------------------------------")
    snr_res = run_snr_sweep_experiment(num_trials_per_snr=50)

    print(f"{'Req SNR (dB)':<12} | {'Meas SNR (dB)':<13} | {'Signal Pwr':<12} | {'Noise Pwr':<12} | {'Dets/Trial':<11} | {'Pd':<8} | {'Range MAE (m)':<15} | {'Vel MAE (m/s)':<15}")
    print("-" * 115)
    for i in range(len(snr_res.snr_levels_db)):
        print(f"{snr_res.snr_levels_db[i]:<12.1f} | {snr_res.measured_snr_levels_db[i]:<13.2f} | {snr_res.signal_powers[i]:<12.4e} | {snr_res.noise_powers[i]:<12.4e} | {snr_res.mean_detections_per_trial[i]:<11.1f} | {snr_res.detection_probabilities[i]:<8.4f} | {snr_res.range_maes_m[i]:<15.4f} | {snr_res.velocity_maes_mps[i]:<15.4f}")

    # Plot 3: SNR Sensitivity
    fig, ax1 = plt.subplots(figsize=(8, 5))
    snr_x = snr_res.snr_levels_db
    pd_y = snr_res.detection_probabilities
    rmae_y = snr_res.range_maes_m

    ax1.plot(snr_x, pd_y, 'o-', color='#1f77b4', linewidth=2, label='Detection Probability (Pd)')
    ax1.set_xlabel('Input Signal-to-Noise Ratio (dB)')
    ax1.set_ylabel('Detection Probability (Pd)', color='#1f77b4')
    ax1.set_ylim([-0.05, 1.05])
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax2 = ax1.twinx()
    ax2.plot(snr_x, rmae_y, 's--', color='#d62728', linewidth=1.5, label='Range MAE (m)')
    ax2.set_ylabel('Range MAE (m)', color='#d62728')
    ax2.tick_params(axis='y', labelcolor='#d62728')

    plt.title('Phase 9 — Detection Probability & Range MAE vs Input SNR')
    fig.tight_layout()
    plot3_path = os.path.join("experiments", "phase9_snr_sweep.png")
    plt.savefig(plot3_path, dpi=300)
    plt.close()
    print(f"-> Plot saved to {plot3_path}")

    # ------------------------------------------------------------------
    # 4. CFAR FALSE-ALARM EXPERIMENT
    # ------------------------------------------------------------------
    print("\n----------------------------------------------------------------------------------------------------------")
    print(" 4. CFAR EMPIRICAL FALSE ALARM RATE EXPERIMENT (NO-TARGET SCENARIO)")
    print("----------------------------------------------------------------------------------------------------------")
    fa_res = run_false_alarm_experiment(num_trials=50)

    print(f"{'Configured Pfa':<18} | {'Empirical Pfa (Noise)':<25} | {'Empirical Pfa (Clutter)':<25} | {'Evaluated Cells/Trial':<22} | {'Trials':<8}")
    print("-" * 105)
    for p in fa_res.points:
        print(f"{p.configured_pfa:<18.1e} | {p.empirical_pfa_noise_only:<25.6e} | {p.empirical_pfa_clutter:<25.6e} | {p.total_evaluated_cells_per_trial:<22} | {p.total_trials:<8}")

    # Plot 4: False Alarm
    fig, ax = plt.subplots(figsize=(8, 5))
    cfg_pfa = [p.configured_pfa for p in fa_res.points]
    emp_noise = [p.empirical_pfa_noise_only for p in fa_res.points]
    emp_clutter = [p.empirical_pfa_clutter for p in fa_res.points]

    ax.loglog(cfg_pfa, cfg_pfa, '--', color='gray', linewidth=1.5, label='Theoretical Pfa (Ideal)')
    ax.loglog(cfg_pfa, emp_noise, 'o-', color='#1f77b4', linewidth=2, label='Empirical Pfa (Noise-Only)')
    ax.loglog(cfg_pfa, emp_clutter, 's-', color='#ff7f0e', linewidth=2, label='Empirical Pfa (Noise + Statistical Clutter)')

    ax.set_xlabel('Configured Target Pfa')
    ax.set_ylabel('Measured Empirical False Alarm Rate')
    ax.set_title('Phase 9 — Empirical False Alarm Rate vs Configured Pfa')
    ax.grid(True, which='both', linestyle=':', alpha=0.6)
    ax.legend()
    fig.tight_layout()
    plot4_path = os.path.join("experiments", "phase9_false_alarm.png")
    plt.savefig(plot4_path, dpi=300)
    plt.close()
    print(f"-> Plot saved to {plot4_path}")

    # ------------------------------------------------------------------
    # 5. MULTI-TARGET MONTE CARLO EXPERIMENT
    # ------------------------------------------------------------------
    print("\n----------------------------------------------------------------------------------------------------------")
    print(" 5. MULTI-TARGET MONTE CARLO EVALUATION (100 TRIALS, SNR = 20 dB)")
    print("----------------------------------------------------------------------------------------------------------")
    mc_res = run_multi_target_monte_carlo_experiment(num_trials=100, snr_db=20.0)

    print(f"Total Trials Executed:         {mc_res.num_trials}")
    print(f"Overall Detection Probability: {mc_res.overall_detection_probability*100:.2f}%")
    print(f"Total Missed Detections (FN):  {mc_res.total_missed_detections}")
    print(f"Total Unmatched CFAR Cells (FP):{mc_res.total_false_detections}")
    print(f"Execution Wall-Clock Time:     {mc_res.execution_time_sec:.3f} s\n")

    print(f"{'Target ID':<12} | {'Pd (%)':<10} | {'Range MAE (m)':<16} | {'Range RMSE (m)':<16} | {'Vel MAE (m/s)':<16} | {'Vel RMSE (m/s)':<16} | {'Matched Detections':<20}")
    print("-" * 118)
    for t in mc_res.targets_theoretical:
        tid = t.target_id
        pd_val = mc_res.detection_probability_per_target[tid] * 100.0
        r_s = mc_res.aggregate_range_stats[tid]
        v_s = mc_res.aggregate_velocity_stats[tid]
        tp_count = r_s.num_trials
        print(f"{tid:<12} | {pd_val:<10.1f} | {r_s.mean_absolute_error:<16.4f} | {r_s.rmse:<16.4f} | {v_s.mean_absolute_error:<16.4f} | {v_s.rmse:<16.4f} | {tp_count:<20}")

    r_overall = mc_res.aggregate_range_stats["overall"]
    v_overall = mc_res.aggregate_velocity_stats["overall"]
    print("-" * 118)
    print(f"{'OVERALL':<12} | {mc_res.overall_detection_probability*100:<10.1f} | {r_overall.mean_absolute_error:<16.4f} | {r_overall.rmse:<16.4f} | {v_overall.mean_absolute_error:<16.4f} | {v_overall.rmse:<16.4f} | {r_overall.num_trials:<20}")

    total_elapsed = time.time() - start_total_time
    print("\n==========================================================================================================")
    print(f" PHASE 9 SCIENTIFIC AUDIT & VALIDATION COMPLETE IN {total_elapsed:.3f} SECONDS!")
    print("==========================================================================================================")


if __name__ == "__main__":
    main()
