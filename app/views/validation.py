"""Validation Dashboard View for Streamlit App.

Provides interactive evaluation dashboards for Range Accuracy, Velocity Accuracy,
SNR Sensitivity Sweeps, False Alarm Rate Analysis, and Monte Carlo multi-trial simulations.
Calls validation engines in src.validation.
"""

from typing import List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from app.components.theory import render_theory_panel
from src.validation.experiments import (
    run_false_alarm_experiment,
    run_multi_target_monte_carlo_experiment,
    run_range_accuracy_experiment,
    run_snr_sweep_experiment,
    run_velocity_accuracy_experiment,
)


@st.cache_data
def _cached_range_experiment(test_ranges: List[float]):
    return run_range_accuracy_experiment(test_ranges_m=test_ranges)


@st.cache_data
def _cached_velocity_experiment(test_velocities: List[float]):
    return run_velocity_accuracy_experiment(test_velocities_mps=test_velocities)


@st.cache_data
def _cached_snr_sweep_experiment(snr_levels: List[float]):
    return run_snr_sweep_experiment(snr_levels_db=snr_levels)


@st.cache_data
def _cached_false_alarm_experiment(pfa_levels: List[float]):
    return run_false_alarm_experiment(pfa_levels=pfa_levels)


@st.cache_data
def _cached_monte_carlo_experiment(num_trials: int = 100):
    return run_multi_target_monte_carlo_experiment(num_trials=num_trials, master_seed=42)


def render_validation_dashboard() -> None:
    """Render the Validation Dashboard view in Streamlit."""
    st.header("📊 Scientific Validation & Monte Carlo Evaluation Dashboard")

    sub_nav = st.radio(
        "Select Validation Suite:",
        options=[
            "1. Range Accuracy",
            "2. Velocity Accuracy",
            "3. SNR Sensitivity Sweep",
            "4. False Alarm Analysis",
            "5. Monte Carlo Evaluation",
        ],
        horizontal=True,
    )

    st.markdown("---")

    if sub_nav == "1. Range Accuracy":
        _render_range_accuracy()
    elif sub_nav == "2. Velocity Accuracy":
        _render_velocity_accuracy()
    elif sub_nav == "3. SNR Sensitivity Sweep":
        _render_snr_sweep()
    elif sub_nav == "4. False Alarm Analysis":
        _render_false_alarm_analysis()
    elif sub_nav == "5. Monte Carlo Evaluation":
        _render_monte_carlo()


def _render_range_accuracy() -> None:
    st.subheader("1. Range Estimation Accuracy Across Benchmark Distances")

    test_ranges = [50.0, 100.0, 150.0, 200.0, 300.0, 400.0, 500.0]
    res = _cached_range_experiment(test_ranges)

    col1, col2 = st.columns([1, 1])

    with col1:
        records = []
        for p in res.points:
            within_res = p.abs_error_m <= res.range_resolution_m
            records.append({
                "True Range (m)": p.true_range_m,
                "Estimated Range (m)": f"{p.estimated_range_m:.4f}",
                "Abs Error (m)": f"{p.abs_error_m:.4f}",
                "Relative Error (%)": f"{p.rel_error * 100:.2f}%",
                "Within Resolution Cell": "✅ YES" if within_res else "❌ NO",
            })
        st.dataframe(pd.DataFrame(records), use_container_width=True)

        mae = res.aggregate_stats.mean_absolute_error
        st.markdown(
            r"**Mean Absolute Error (MAE):** `"
            + f"{mae:.4f} m` | "
            + r"**Range Resolution Bound ($\Delta R$):** `"
            + f"{res.range_resolution_m:.4f} m`"
        )

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
        ax.plot([p.true_range_m for p in res.points], [p.estimated_range_m for p in res.points], "bo-", label="Estimated vs True Range")
        ax.plot([0, 500], [0, 500], "r--", label="Ideal 1:1 Identity Line")
        ax.set_title("True Range vs Estimated Range", fontweight="bold")
        ax.set_xlabel("True Range (m)")
        ax.set_ylabel("Estimated Range (m)")
        ax.legend(facecolor="#161B22", edgecolor="#30363D")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig, use_container_width=True)

    render_theory_panel("validation")


def _render_velocity_accuracy() -> None:
    st.subheader("2. Radial Velocity Accuracy Across Kinematic Range")

    test_velocities = [-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0]
    res = _cached_velocity_experiment(test_velocities)

    col1, col2 = st.columns([1, 1])

    with col1:
        records = []
        for p in res.points:
            within_res = p.abs_error_mps <= res.velocity_resolution_mps
            records.append({
                "True Velocity (m/s)": p.true_velocity_mps,
                "Estimated Velocity (m/s)": f"{p.estimated_velocity_mps:+.4f}",
                "Abs Error (m/s)": f"{p.abs_error_mps:.4f}",
                "Relative Error (%)": f"{p.rel_error * 100:.2f}%" if abs(p.true_velocity_mps) >= 1e-3 else "0.00% (Stationary)",
                "Within Resolution Cell": "✅ YES" if within_res else "❌ NO",
            })
        st.dataframe(pd.DataFrame(records), use_container_width=True)

        mae = res.aggregate_stats.mean_absolute_error
        st.markdown(
            r"**Mean Absolute Error (MAE):** `"
            + f"{mae:.4f} m/s` | "
            + r"**Velocity Resolution Bound ($\Delta v$):** `"
            + f"{res.velocity_resolution_mps:.4f} m/s`"
        )

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
        ax.plot([p.true_velocity_mps for p in res.points], [p.estimated_velocity_mps for p in res.points], "go-", label="Estimated vs True Velocity")
        ax.plot([-20, 20], [-20, 20], "r--", label="Ideal 1:1 Identity Line")
        ax.set_title("True Velocity vs Estimated Velocity", fontweight="bold")
        ax.set_xlabel("True Radial Velocity (m/s)")
        ax.set_ylabel("Estimated Radial Velocity (m/s)")
        ax.legend(facecolor="#161B22", edgecolor="#30363D")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig, use_container_width=True)

    render_theory_panel("validation")


def _render_snr_sweep() -> None:
    st.subheader("3. Detection Probability & Range MAE vs SNR Sensitivity Sweep")

    snr_levels = [30.0, 20.0, 10.0, 0.0, -10.0, -20.0, -30.0, -35.0, -40.0]
    res = _cached_snr_sweep_experiment(snr_levels)

    col1, col2 = st.columns([1, 1])

    with col1:
        records = []
        for i in range(len(res.snr_levels_db)):
            records.append({
                "Req SNR (dB)": res.snr_levels_db[i],
                "Meas SNR (dB)": f"{res.measured_snr_levels_db[i]:.2f}",
                "Pd": f"{res.detection_probabilities[i]:.2f}",
                "Range MAE (m)": f"{res.range_maes_m[i]:.4f}",
                "Velocity MAE (m/s)": f"{res.velocity_maes_mps[i]:.4f}",
                "CFAR Detections": res.mean_detections_per_trial[i],
            })
        st.dataframe(pd.DataFrame(records), use_container_width=True)

    with col2:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), dpi=100)
        snrs = res.snr_levels_db
        pds = res.detection_probabilities
        maes = res.range_maes_m

        ax1.plot(snrs, pds, "ro-", lw=1.8)
        ax1.set_title("Detection Probability (Pd) vs SNR", fontweight="bold")
        ax1.set_xlabel("SNR (dB)")
        ax1.set_ylabel("Pd")
        ax1.set_ylim([-0.05, 1.05])
        ax1.grid(True, alpha=0.3)

        ax2.plot(snrs, maes, "bo-", lw=1.8)
        ax2.set_title("Range MAE (m) vs SNR", fontweight="bold")
        ax2.set_xlabel("SNR (dB)")
        ax2.set_ylabel("Range MAE (m)")
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

    render_theory_panel("validation")


def _render_false_alarm_analysis() -> None:
    st.subheader("4. CA-CFAR Empirical False Alarm Rate Analysis")

    pfa_levels = [1e-2, 1e-3, 1e-4]
    res = _cached_false_alarm_experiment(pfa_levels)

    records = []
    for p in res.points:
        records.append({
            "Target Pfa": f"{p.configured_pfa:.0e}",
            "Noise-Only Empirical Pfa": f"{p.empirical_pfa_noise_only:.6f}",
            "Clutter Empirical Pfa": f"{p.empirical_pfa_clutter:.6f}",
            "Evaluated Cells / Trial": p.total_evaluated_cells_per_trial,
        })
    st.dataframe(pd.DataFrame(records), use_container_width=True)

    render_theory_panel("validation")


def _render_monte_carlo() -> None:
    st.subheader("5. Multi-Target Monte Carlo 100-Trial Performance Evaluation")

    mc_res = _cached_monte_carlo_experiment(num_trials=100)

    range_maes = [s.mean_absolute_error for s in mc_res.aggregate_range_stats.values()] if mc_res.aggregate_range_stats else [0.0]
    vel_maes = [s.mean_absolute_error for s in mc_res.aggregate_velocity_stats.values()] if mc_res.aggregate_velocity_stats else [0.0]
    overall_range_mae = float(np.mean(range_maes))
    overall_vel_mae = float(np.mean(vel_maes))

    st.markdown(
        f"**Total Trials:** `{mc_res.num_trials}` | **Overall Detection Probability ($P_d$):** `{mc_res.overall_detection_probability*100:.1f}%` | "
        f"**Missed Detections (FN):** `{mc_res.total_missed_detections}` | **False Detections (FP):** `{mc_res.total_false_detections}` | "
        f"**Range MAE:** `{overall_range_mae:.4f} m` | **Velocity MAE:** `{overall_vel_mae:.4f} m/s`"
    )

    st.info("Unmatched CFAR-positive cells represent adaptive threshold exceedances. Physical target assignment requires 1-to-1 resolution-gated matching.")

    render_theory_panel("validation")

