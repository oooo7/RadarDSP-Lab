"""FMCW Radar Laboratory View for Streamlit App.

Provides interactive UI controls for single-target FMCW stretch-dechirp,
multi-target Range-Doppler matrix processing, AWGN noise & clutter simulation,
and 2D Cell-Averaging CFAR (CA-CFAR) detection.
Calls backend functions in src.radar.
"""

import numpy as np
import pandas as pd
import streamlit as st

from app.components.metrics import format_detection_table
from app.components.plots import (
    plot_cfar_detection_map,
    plot_fmcw_processing_chain,
    plot_range_doppler_heatmap,
    plot_spectrum,
)
from app.components.theory import render_theory_panel
from src.radar.cfar import calculate_ca_cfar_alpha, run_ca_cfar
from src.radar.chirp import generate_fmcw_chirp
from src.radar.clutter import generate_clutter
from src.radar.doppler import (
    compute_range_doppler_map,
    extract_range_doppler_peaks,
    generate_multi_chirp_data_cube,
)
from src.radar.noise import add_awgn_noise
from src.radar.processing import (
    calculate_max_unambiguous_range,
    compute_range_fft,
    dechirp_signal,
    estimate_range,
)
from src.radar.propagation import simulate_target_echo
from src.radar.target import compute_target_state
from src.utils.config import CFARConfig, RadarConfig, TargetConfig


def render_radar_lab() -> None:
    """Render the FMCW Radar Laboratory view in Streamlit."""
    st.header("🚗 FMCW Radar Processing Laboratory")

    sub_nav = st.radio(
        "Select FMCW Experiment:",
        options=[
            "1. Single Target FMCW",
            "2. Multi-Target Kinematics",
            "3. Range-Doppler Map",
            "4. Noise & Clutter Environment",
            "5. CA-CFAR Detection Engine",
        ],
        horizontal=True,
    )

    st.markdown("---")

    if sub_nav == "1. Single Target FMCW":
        _render_single_target()
    elif sub_nav == "2. Multi-Target Kinematics":
        _render_multi_target()
    elif sub_nav == "3. Range-Doppler Map":
        _render_range_doppler()
    elif sub_nav == "4. Noise & Clutter Environment":
        _render_noise_clutter()
    elif sub_nav == "5. CA-CFAR Detection Engine":
        _render_ca_cfar()


def _render_single_target() -> None:
    st.subheader("1. FMCW Stretch-Dechirp Single Target Range Estimation")

    col_ctrl, col_plot = st.columns([1, 2])

    with col_ctrl:
        fc_ghz = st.slider("Carrier Frequency fc (GHz):", 24.0, 79.0, 77.0, 1.0)
        b_mhz = st.slider("Sweep Bandwidth B (MHz):", 50.0, 500.0, 150.0, 10.0)
        tc_us = st.slider("Chirp Duration Tc (µs):", 20.0, 500.0, 100.0, 10.0)
        fs_mhz = st.number_input("ADC Sampling Rate Fs (MHz):", min_value=1.0, max_value=100.0, value=20.0, step=1.0)

        target_r = st.slider("Target Range R (m):", 5.0, 480.0, 100.0, 5.0)

        fc = fc_ghz * 1e9
        b = b_mhz * 1e6
        tc = tc_us * 1e-6
        fs = fs_mhz * 1e6

        radar_cfg = RadarConfig(
            carrier_frequency_hz=fc,
            sweep_bandwidth_hz=b,
            chirp_duration_sec=tc,
            sampling_rate_hz=fs,
        )

        target_cfg = TargetConfig(range_m=target_r, velocity_mps=0.0, amplitude=1.0)

    with col_plot:
        try:
            tx_chirp = generate_fmcw_chirp(radar_cfg)
            target_state = compute_target_state(target_cfg, chirp_slope_hz_per_sec=tx_chirp.chirp_slope_hz_per_sec, carrier_frequency_hz=fc)
            rx_echo = simulate_target_echo(tx_chirp, target_state)
            s_beat = dechirp_signal(tx_chirp, rx_echo)
            spec_res = compute_range_fft(s_beat, sampling_rate_hz=fs, chirp_slope_hz_per_sec=tx_chirp.chirp_slope_hz_per_sec, window_name="hann")
            range_est_res = estimate_range(spec_res, target_state.range_m)

            fig = plot_fmcw_processing_chain(
                t_fast=tx_chirp.time_vector,
                s_beat=s_beat.beat_signal,
                r_axis=spec_res.range_m,
                r_mag=spec_res.magnitude_spectrum,
                est_range_m=range_est_res.estimated_range_m,
                true_range_m=target_r,
                fc=fc,
                b_hz=b,
                tc_sec=tc,
            )
            st.pyplot(fig, use_container_width=True)

            delta_r = radar_cfg.speed_of_light_m_per_s / (2.0 * b)
            r_max = calculate_max_unambiguous_range(fs, tx_chirp.chirp_slope_hz_per_sec)

            st.markdown(
                r"**True Range:** `"
                + f"{target_r:.2f} m` | **Estimated Range:** `{range_est_res.estimated_range_m:.2f} m` | **Abs Error:** `{range_est_res.absolute_error_m:.4f} m` | "
                + r"**Chirp Slope (S):** `"
                + f"{tx_chirp.chirp_slope_hz_per_sec/1e12:.2f} MHz/µs` | **Beat Frequency ($f_b$):** `{target_state.theoretical_beat_frequency_hz/1e3:.2f} kHz` | "
                + r"**Range Resolution ($\Delta R$):** `"
                + f"{delta_r:.4f} m` | **Max Range:** `{r_max:.1f} m`"
            )

        except Exception as e:
            st.error(f"FMCW Processing error: {e}")
            return

    # Compact Theory Box
    st.info(
        r"💡 **FMCW Stretch Processing Basics:** "
        r"Round-trip delay $\tau = \frac{2 R}{c}$ ➔ Chirp Slope $S = \frac{B}{T_c}$ ➔ Beat Frequency $f_b = S \cdot \tau = \frac{2 S R}{c}$ ➔ Range Estimate $R = \frac{c f_b}{2 S}$."
    )

    render_theory_panel("fmcw_single")


def _render_multi_target() -> None:
    st.subheader("2. Multi-Target FMCW Radar Synthesis & Range-Doppler Matrix")

    col_ctrl, col_plot = st.columns([1, 2])

    with col_ctrl:
        st.markdown("**Benchmark Preset:**")
        if st.button("🔄 Reset to Phase 7 Benchmark Targets"):
            st.session_state["targets"] = [
                {"range_m": 50.0, "velocity_mps": 10.0, "amplitude": 1.0, "target_id": "T1"},
                {"range_m": 120.0, "velocity_mps": -5.0, "amplitude": 0.7, "target_id": "T2"},
                {"range_m": 200.0, "velocity_mps": 0.0, "amplitude": 0.5, "target_id": "T3"},
            ]

        if "targets" not in st.session_state:
            st.session_state["targets"] = [
                {"range_m": 50.0, "velocity_mps": 10.0, "amplitude": 1.0, "target_id": "T1"},
                {"range_m": 120.0, "velocity_mps": -5.0, "amplitude": 0.7, "target_id": "T2"},
                {"range_m": 200.0, "velocity_mps": 0.0, "amplitude": 0.5, "target_id": "T3"},
            ]

        st.markdown(r"*(Velocity Sign Convention: Positive $v > 0$ = Receding, Negative $v < 0$ = Approaching)*")

        # Interactive Target Inputs
        t_cfgs = []
        for i, t_dict in enumerate(st.session_state["targets"]):
            st.markdown(f"**Target {i+1} ({t_dict.get('target_id', f'T{i+1}')})**")
            r_val = st.slider(f"Range R (m) - T{i+1}:", 10.0, 400.0, float(t_dict["range_m"]), key=f"r_{i}")
            v_val = st.slider(f"Velocity v (m/s) - T{i+1}:", -20.0, 20.0, float(t_dict["velocity_mps"]), key=f"v_{i}")
            a_val = st.slider(f"Amplitude - T{i+1}:", 0.1, 2.0, float(t_dict["amplitude"]), key=f"a_{i}")
            t_cfgs.append(TargetConfig(range_m=r_val, velocity_mps=v_val, amplitude=a_val, target_id=f"T{i+1}"))

        m_chirps = st.slider("Chirp Count M (Slow-Time):", 16, 128, 64, 16)

    with col_plot:
        radar_cfg = RadarConfig(
            carrier_frequency_hz=77e9,
            sweep_bandwidth_hz=150e6,
            chirp_duration_sec=100e-6,
            sampling_rate_hz=20e6,
            num_chirps=m_chirps,
        )

        try:
            cube = generate_multi_chirp_data_cube(radar_cfg, t_cfgs)
            rd_res = compute_range_doppler_map(cube, range_window="hann", doppler_window="hann")
            peaks = extract_range_doppler_peaks(rd_res, max_peaks=10)

            fig = plot_range_doppler_heatmap(rd_res.range_axis_m, rd_res.velocity_axis_mps, rd_res.magnitude_db, title="Multi-Target Range-Doppler Heatmap")
            st.pyplot(fig, use_container_width=True)

            st.markdown("### 🎯 Detected Target Peak Summary Table")
            if peaks:
                df_peaks = format_detection_table(peaks)
                st.dataframe(df_peaks, use_container_width=True)
            else:
                st.info("No peaks detected above threshold.")

        except Exception as e:
            st.error(f"Multi-target simulation error: {e}")

    render_theory_panel("multi_target")


def _render_range_doppler() -> None:
    st.subheader("3. 2D Range-Doppler Processing Engine")

    col_ctrl, col_plot = st.columns([1, 2])

    with col_ctrl:
        m_chirps = st.select_slider("Slow-Time Chirp Count M:", options=[16, 32, 64, 128, 256], value=64)
        r_win = st.selectbox("Fast-Time Range Window:", options=["hann", "hamming", "blackman", "rect"], index=0)
        d_win = st.selectbox("Slow-Time Doppler Window:", options=["hann", "hamming", "blackman", "rect"], index=0)

    with col_plot:
        radar_cfg = RadarConfig(
            carrier_frequency_hz=77e9,
            sweep_bandwidth_hz=150e6,
            chirp_duration_sec=100e-6,
            sampling_rate_hz=20e6,
            num_chirps=m_chirps,
        )

        targets = [
            TargetConfig(range_m=60.0, velocity_mps=12.0, amplitude=1.0, target_id="T1"),
            TargetConfig(range_m=150.0, velocity_mps=-8.0, amplitude=0.8, target_id="T2"),
        ]

        try:
            cube = generate_multi_chirp_data_cube(radar_cfg, targets)
            rd_res = compute_range_doppler_map(cube, range_window=r_win, doppler_window=d_win)

            fig = plot_range_doppler_heatmap(rd_res.range_axis_m, rd_res.velocity_axis_mps, rd_res.magnitude_db)
            st.pyplot(fig, use_container_width=True)

            st.markdown(
                r"**Range Resolution ($\Delta R$):** `"
                + f"{rd_res.range_resolution_m:.4f} m` | **Velocity Resolution ($\Delta v$):** `{rd_res.velocity_resolution_mps:.4f} m/s` | "
                + r"**Max Range ($R_{{max}}$):** `"
                + f"{rd_res.unambiguous_range_m:.1f} m` | **Max Velocity ($v_{{max}}$):** `{rd_res.unambiguous_velocity_mps:.2f} m/s`"
            )

        except Exception as e:
            st.error(f"Range-Doppler processing error: {e}")

    render_theory_panel("range_doppler")


def _render_noise_clutter() -> None:
    st.subheader("4. Complex AWGN & Statistical Background Clutter")

    col_ctrl, col_plot = st.columns([1, 2])

    with col_ctrl:
        target_snr_db = st.slider("Target SNR (dB):", -20.0, 30.0, 10.0, 2.0)
        seed = st.number_input("RNG Seed:", min_value=0, max_value=9999, value=42)
        enable_clutter = st.checkbox("Enable Background Clutter", value=False)
        clutter_strength = st.slider("Clutter Power Level:", 0.1, 5.0, 1.0) if enable_clutter else 0.0

    with col_plot:
        radar_cfg = RadarConfig(carrier_frequency_hz=77e9, sweep_bandwidth_hz=150e6, chirp_duration_sec=100e-6, sampling_rate_hz=20e6)
        tx_chirp = generate_fmcw_chirp(radar_cfg)
        t_state = compute_target_state(TargetConfig(range_m=100.0, velocity_mps=0.0), chirp_slope_hz_per_sec=tx_chirp.chirp_slope_hz_per_sec, carrier_frequency_hz=77e9)
        rx_echo = simulate_target_echo(tx_chirp, t_state)
        clean_beat = dechirp_signal(tx_chirp, rx_echo).beat_signal

        try:
            noisy_beat, measured_snr = add_awgn_noise(clean_beat, snr_db=target_snr_db, seed=seed)

            if enable_clutter:
                clutter_res = generate_clutter(shape=clean_beat.shape, clutter_power=clutter_strength, seed=seed)
                noisy_beat = noisy_beat + clutter_res.clutter_signal

            spec_clean = compute_range_fft(clean_beat, sampling_rate_hz=20e6, chirp_slope_hz_per_sec=tx_chirp.chirp_slope_hz_per_sec)
            spec_noisy = compute_range_fft(noisy_beat, sampling_rate_hz=20e6, chirp_slope_hz_per_sec=tx_chirp.chirp_slope_hz_per_sec)

            fig = plot_spectrum(spec_noisy.range_m, spec_noisy.magnitude_spectrum, title=f"Range Spectrum with Noise/Clutter (SNR={target_snr_db} dB)", db_scale=True)
            st.pyplot(fig, use_container_width=True)

            st.markdown(
                f"**Requested SNR:** `{target_snr_db:.1f} dB` | **Measured Signal SNR:** `{measured_snr:.2f} dB` | "
                f"**Background Noise/Clutter Active:** `{'YES' if enable_clutter else 'NO'}`"
            )

        except Exception as e:
            st.error(f"Noise/Clutter processing error: {e}")

    render_theory_panel("noise_clutter")


def _render_ca_cfar() -> None:
    st.subheader("5. 2D Cell-Averaging Constant False Alarm Rate (CA-CFAR) Detector")

    col_ctrl, col_plot = st.columns([1, 2])

    with col_ctrl:
        pfa = st.select_slider("Probability of False Alarm (Pfa):", options=[1e-2, 1e-3, 1e-4, 1e-5], value=1e-4)
        g_r = st.slider("Guard Cells Range (G_R):", 0, 5, 2)
        g_d = st.slider("Guard Cells Doppler (G_D):", 0, 5, 2)
        t_r = st.slider("Training Cells Range (T_R):", 1, 8, 4)
        t_d = st.slider("Training Cells Doppler (T_D):", 1, 8, 4)

        n_train = (2 * (t_d + g_d) + 1) * (2 * (t_r + g_r) + 1) - (2 * g_d + 1) * (2 * g_r + 1)
        alpha = calculate_ca_cfar_alpha(pfa, n_train)

    with col_plot:
        radar_cfg = RadarConfig(carrier_frequency_hz=77e9, sweep_bandwidth_hz=150e6, chirp_duration_sec=100e-6, sampling_rate_hz=20e6, num_chirps=64)
        targets = [
            TargetConfig(range_m=50.0, velocity_mps=10.0, amplitude=1.0, target_id="T1"),
            TargetConfig(range_m=120.0, velocity_mps=-5.0, amplitude=0.7, target_id="T2"),
            TargetConfig(range_m=200.0, velocity_mps=0.0, amplitude=0.5, target_id="T3"),
        ]

        try:
            cube = generate_multi_chirp_data_cube(radar_cfg, targets)
            rd_res = compute_range_doppler_map(cube)

            cfar_cfg = CFARConfig(pfa=pfa, num_guard_range=g_r, num_guard_doppler=g_d, num_train_range=t_r, num_train_doppler=t_d)
            cfar_res = run_ca_cfar(rd_res, cfar_cfg=cfar_cfg)

            fig = plot_cfar_detection_map(
                range_axis_m=rd_res.range_axis_m,
                velocity_axis_mps=rd_res.velocity_axis_mps,
                rd_power_db=cfar_res.input_power,
                threshold_db=cfar_res.threshold_db,
                detection_mask=cfar_res.detection_mask,
            )
            st.pyplot(fig, use_container_width=True)

            st.markdown(
                r"**Training Cells ($N_{train}$):** `"
                + f"{n_train}` | **CFAR Multiplier ($\\alpha$):** `{cfar_res.alpha:.4f}` | **Total CFAR-Positive Cells:** `{cfar_res.num_detections}`"
            )

            st.info("Note: CFAR-positive cells mark adaptive power threshold exceedances. Candidate targets require physical 1-to-1 matching/clustering.")

        except Exception as e:
            st.error(f"CFAR processing error: {e}")

    render_theory_panel("cfar")
