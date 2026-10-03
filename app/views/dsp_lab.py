"""DSP Laboratory View for Streamlit App.

Provides interactive UI controls for signal generation, sampling & aliasing,
FFT spectral analysis, digital filtering, and multirate resampling.
Calls backend functions in src.dsp.
"""

import numpy as np
import pandas as pd
import streamlit as st

from app.components.plots import (
    plot_filter_response,
    plot_spectrum,
    plot_time_domain,
)
from app.components.theory import render_theory_panel
from src.dsp.filtering import (
    analyze_filter_response,
    apply_filter,
    design_fir_filter,
    design_iir_butterworth,
)
from src.dsp.resampling import (
    decimate_signal,
    interpolate_signal,
    resample_rational,
)
from src.dsp.sampling import calculate_alias_frequency, sample_signal
from src.dsp.signals import (
    generate_am,
    generate_chirp,
    generate_cosine,
    generate_fm,
    generate_multitone,
    generate_noise,
    generate_sawtooth,
    generate_sine,
    generate_square,
    generate_triangle,
)
from src.dsp.transforms import compute_fft
from src.utils.config import SignalGenConfig


def render_dsp_lab() -> None:
    """Render the DSP Laboratory view in Streamlit."""
    st.header("🔬 Digital Signal Processing Laboratory")

    sub_nav = st.radio(
        "Select DSP Experiment:",
        options=[
            "1. Signal Generator",
            "2. Sampling & Aliasing",
            "3. FFT & Spectrum",
            "4. Digital Filters",
            "5. Resampling",
        ],
        horizontal=True,
    )

    st.markdown("---")

    if sub_nav == "1. Signal Generator":
        _render_signal_generator()
    elif sub_nav == "2. Sampling & Aliasing":
        _render_sampling_aliasing()
    elif sub_nav == "3. FFT & Spectrum":
        _render_fft_spectrum()
    elif sub_nav == "4. Digital Filters":
        _render_digital_filters()
    elif sub_nav == "5. Resampling":
        _render_resampling()


def _render_signal_generator() -> None:
    st.subheader("1. Interactive Signal Generator")

    col_ctrl, col_plot = st.columns([1, 2])

    with col_ctrl:
        sig_type = st.selectbox(
            "Signal Type:",
            options=[
                "Sine",
                "Cosine",
                "Square",
                "Triangle",
                "Sawtooth",
                "Multitone",
                "Chirp",
                "Gaussian Noise",
                "AM",
                "FM",
            ],
            index=0,
        )

        fs = st.number_input("Sampling Frequency Fs (Hz):", min_value=10.0, max_value=100000.0, value=1000.0, step=100.0)
        duration = st.number_input("Duration (seconds):", min_value=0.01, max_value=10.0, value=1.0, step=0.1)

        amp = 1.0
        freq = 10.0
        phase = 0.0

        if sig_type in ["Sine", "Cosine", "Square", "Triangle", "Sawtooth"]:
            amp = st.slider("Amplitude (V):", 0.1, 10.0, 1.0, 0.1)
            freq = st.slider("Frequency (Hz):", 1.0, float(fs / 2), 10.0, 1.0)
            phase = st.slider("Phase (rad):", -np.pi, np.pi, 0.0, 0.1)

        try:
            if sig_type == "Sine":
                container = generate_sine(amplitude=amp, frequency_hz=freq, sampling_rate_hz=fs, duration_sec=duration, phase_rad=phase)
            elif sig_type == "Cosine":
                container = generate_cosine(amplitude=amp, frequency_hz=freq, sampling_rate_hz=fs, duration_sec=duration, phase_rad=phase)
            elif sig_type == "Square":
                container = generate_square(amplitude=amp, frequency_hz=freq, sampling_rate_hz=fs, duration_sec=duration, phase_rad=phase)
            elif sig_type == "Triangle":
                container = generate_triangle(amplitude=amp, frequency_hz=freq, sampling_rate_hz=fs, duration_sec=duration, phase_rad=phase)
            elif sig_type == "Sawtooth":
                container = generate_sawtooth(amplitude=amp, frequency_hz=freq, sampling_rate_hz=fs, duration_sec=duration, phase_rad=phase)
            elif sig_type == "Multitone":
                f_list_str = st.text_input("Frequencies (Hz, comma-separated):", "10, 25, 40")
                a_list_str = st.text_input("Amplitudes (V, comma-separated):", "1.0, 0.5, 0.2")
                f_list = [float(x.strip()) for x in f_list_str.split(",")]
                a_list = [float(x.strip()) for x in a_list_str.split(",")]
                container = generate_multitone(frequencies_hz=f_list, amplitudes=a_list, sampling_rate_hz=fs, duration_sec=duration)
            elif sig_type == "Chirp":
                f0 = st.slider("Start Frequency f0 (Hz):", 1.0, float(fs / 4), 5.0)
                f1 = st.slider("End Frequency f1 (Hz):", 1.0, float(fs / 2), 50.0)
                container = generate_chirp(start_freq_hz=f0, end_freq_hz=f1, duration_sec=duration, sampling_rate_hz=fs)
            elif sig_type == "Gaussian Noise":
                seed = st.number_input("RNG Seed:", min_value=0, max_value=9999, value=42)
                container = generate_noise(std_dev=amp, sampling_rate_hz=fs, duration_sec=duration, seed=seed)
            elif sig_type == "AM":
                fc = st.slider("Carrier Frequency (Hz):", 10.0, float(fs / 2), 50.0)
                fm = st.slider("Modulation Frequency (Hz):", 1.0, 20.0, 5.0)
                mi = st.slider("Modulation Index (m):", 0.1, 1.0, 0.5)
                container = generate_am(carrier_freq_hz=fc, mod_freq_hz=fm, mod_index=mi, sampling_rate_hz=fs, duration_sec=duration)
            elif sig_type == "FM":
                fc = st.slider("Carrier Frequency (Hz):", 10.0, float(fs / 2), 50.0)
                fm = st.slider("Modulation Frequency (Hz):", 1.0, 20.0, 5.0)
                fd = st.slider("Frequency Deviation (Hz):", 1.0, 30.0, 10.0)
                container = generate_fm(carrier_freq_hz=fc, mod_freq_hz=fm, freq_dev_hz=fd, sampling_rate_hz=fs, duration_sec=duration)
        except Exception as e:
            st.error(f"Error generating signal: {e}")
            return

    with col_plot:
        t = container.time_vector
        x = container.amplitude
        st.pyplot(plot_time_domain(t, x, title=f"Time Waveform ({sig_type})"), use_container_width=True)

        # Signal Metrics
        n_samples = len(x)
        rms_val = np.sqrt(np.mean(np.abs(x) ** 2))
        st.markdown(f"**Sample Count:** {n_samples} | **Sampling Rate:** {fs} Hz | **Duration:** {duration} s | **RMS Power:** {rms_val:.3f} V")

        # CSV Download Button
        df_export = pd.DataFrame({"time_sec": t, "amplitude_volts": np.real(x)})
        csv_buffer = df_export.to_csv(index=False)
        st.download_button(
            label="💾 Download Signal CSV",
            data=csv_buffer,
            file_name=f"signal_{sig_type.lower()}.csv",
            mime="text/csv",
        )

    render_theory_panel("signal_gen")


def _render_sampling_aliasing() -> None:
    st.subheader("2. Sampling Theorem & Aliasing Foldover")

    col_ctrl, col_plot = st.columns([1, 2])

    with col_ctrl:
        f_sig = st.slider("Original Signal Frequency f (Hz):", 10.0, 1000.0, 700.0, 10.0)
        fs = st.slider("Sampling Frequency Fs (Hz):", 100.0, 2000.0, 1000.0, 50.0)
        duration = 0.02  # 20 ms display

        # Preset Nyquist Selectors
        preset = st.radio(
            "Quick Nyquist Mode:",
            options=["Custom", "Oversampled (Fs > 2f)", "Critical Nyquist (Fs ≈ 2f)", "Undersampled / Aliased (Fs < 2f)"],
            index=0,
        )

        if preset == "Oversampled (Fs > 2f)":
            fs = 2000.0
        elif preset == "Critical Nyquist (Fs ≈ 2f)":
            fs = 2.0 * f_sig
        elif preset == "Undersampled / Aliased (Fs < 2f)":
            fs = 1000.0
            f_sig = 700.0

        nyquist_rate = 2.0 * f_sig
        nyquist_freq = fs / 2.0
        expected_alias = calculate_alias_frequency(f_sig, fs)

    with col_plot:
        try:
            sig_orig = generate_sine(amplitude=1.0, frequency_hz=f_sig, phase_rad=0.0, sampling_rate_hz=fs, duration_sec=duration)
            res = sample_signal(sig_orig, target_fs_hz=fs)

            # Continuous reference
            sig_cont = generate_sine(amplitude=1.0, frequency_hz=f_sig, phase_rad=0.0, sampling_rate_hz=fs * 20, duration_sec=duration)

            fig, ax1 = plot_time_domain(sig_cont.time_vector, sig_cont.amplitude, title="Continuous Reference vs Discrete Samples").axes[0].figure, None
            ax = fig.axes[0]
            ax.plot(res.sampled_signal.time_vector * 1e3 if max(res.sampled_signal.time_vector) < 0.01 else res.sampled_signal.time_vector, res.sampled_signal.amplitude, "ro-", label="Discrete Samples", ms=4, lw=1)
            ax.legend(loc="upper right", facecolor="#161B22", edgecolor="#30363D")
            st.pyplot(fig, use_container_width=True)

            spec_res = compute_fft(res.sampled_signal.amplitude, sampling_rate_hz=fs)
            peak_f = spec_res.frequency_hz[np.argmax(spec_res.magnitude)]
            fig_spec = plot_spectrum(spec_res.frequency_hz, spec_res.magnitude, title=f"Sampled Spectrum (Peak = {peak_f:.1f} Hz)", db_scale=False)
            fig_spec.axes[0].axvline(nyquist_freq, color="#FF7B72", linestyle="--", label=f"Nyquist Freq (Fs/2 = {nyquist_freq:.1f} Hz)")
            fig_spec.axes[0].legend(loc="upper right", facecolor="#161B22", edgecolor="#30363D")
            st.pyplot(fig_spec, use_container_width=True)

        except Exception as e:
            st.error(f"Sampling pipeline error: {e}")
            return

        st.markdown(
            f"**Nyquist Rate ($2 f$):** `{nyquist_rate:.1f} Hz` | **Nyquist Frequency ($F_s/2$):** `{nyquist_freq:.1f} Hz` | "
            f"**Theoretical Alias ($f_{{alias}}$):** `{expected_alias:.1f} Hz` | **Measured Spectrum Peak:** `{peak_f:.1f} Hz`"
        )

        if f_sig > nyquist_freq:
            st.warning(f"⚠️ UNDERSAMPLED (Aliased!): Original {f_sig:.1f} Hz signal folds over into {expected_alias:.1f} Hz.")
        else:
            st.success(f"✅ OVERSAMPLED: Signal frequency {f_sig:.1f} Hz is safely below Nyquist frequency {nyquist_freq:.1f} Hz.")

    render_theory_panel("nyquist")


def _render_fft_spectrum() -> None:
    st.subheader("3. FFT, Windowing & Spectral Leakage")

    col_ctrl, col_plot = st.columns([1, 2])

    with col_ctrl:
        f_sig = st.slider("Signal Frequency (Hz):", 10.0, 400.0, 100.0, 5.0)
        fs = 1000.0
        n_samples = st.slider("Sample Count N:", 64, 2048, 512, 64)
        win_name = st.selectbox("Window Function:", options=["rect", "hann", "hamming", "blackman", "kaiser"], index=1)
        n_fft = st.select_slider("FFT Size N_fft (Zero Padding):", options=[512, 1024, 2048, 4096, 8192], value=1024)
        scale_db = st.checkbox("Scale Magnitude in dB", value=True)

    with col_plot:
        t = np.arange(n_samples) / fs
        x = np.sin(2 * np.pi * f_sig * t)

        try:
            res = compute_fft(x, sampling_rate_hz=fs, window_type=win_name, n_fft=n_fft)

            fig = plot_spectrum(res.frequency_hz, res.magnitude, title=f"FFT Spectrum (Window: {win_name}, N_fft={n_fft})", db_scale=scale_db)
            st.pyplot(fig, use_container_width=True)

            df_physical = fs / n_samples
            df_grid = fs / n_fft
            peak_f = float(res.frequency_hz[np.argmax(res.magnitude)])
            est_amp = float(np.max(res.magnitude))

            st.markdown(
                r"**Dominant Frequency Peak:** `"
                + f"{peak_f:.2f} Hz` (True: {f_sig:.1f} Hz) | **Estimated Amplitude:** `{est_amp:.3f} V` | "
                + r"**Physical Resolution ($\Delta f = F_s / N$):** `"
                + f"{df_physical:.2f} Hz` | **Zero-Padding Grid Spacing ($F_s / N_{{fft}}$):** `{df_grid:.2f} Hz`"
            )
            st.caption("Note: Zero-padding increases grid density for visual interpolation, but does NOT improve physical frequency resolution.")

            # Spectrum CSV Download Button
            df_spec = pd.DataFrame({"frequency_hz": res.frequency_hz, "magnitude_volts": res.magnitude})
            csv_spec = df_spec.to_csv(index=False)
            st.download_button(
                label="💾 Download Spectrum CSV",
                data=csv_spec,
                file_name="spectrum_analysis.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"FFT processing error: {e}")

    render_theory_panel("fft")


def _render_digital_filters() -> None:
    st.subheader("4. FIR & IIR Digital Filter Design")

    col_ctrl, col_plot = st.columns([1, 2])

    with col_ctrl:
        family = st.radio("Filter Family:", options=["FIR (Windowed)", "IIR Butterworth (SOS)"], index=0)
        ftype = st.selectbox("Filter Type:", options=["lowpass", "highpass", "bandpass", "bandstop"], index=0)
        fs = 1000.0

        if ftype in ["lowpass", "highpass"]:
            cutoff = st.slider("Cutoff Frequency fc (Hz):", 10.0, 450.0, 100.0, 10.0)
        else:
            fc1 = st.slider("Lower Cutoff fc1 (Hz):", 10.0, 200.0, 50.0, 5.0)
            fc2 = st.slider("Upper Cutoff fc2 (Hz):", 210.0, 450.0, 200.0, 5.0)
            cutoff = [fc1, fc2]

        order = st.slider("Filter Order / Taps:", 2, 101, 31 if family.startswith("FIR") else 4, 1)

    with col_plot:
        try:
            if family.startswith("FIR"):
                f_coeffs = design_fir_filter(filter_type=ftype, cutoff_hz=cutoff, order=order, sampling_rate_hz=fs)
                resp = analyze_filter_response(f_coeffs, n_points=1024)
            else:
                f_coeffs = design_iir_butterworth(filter_type=ftype, cutoff_hz=cutoff, order=order, sampling_rate_hz=fs)
                resp = analyze_filter_response(f_coeffs, n_points=1024)

            st.pyplot(plot_filter_response(resp.frequency_hz, resp.magnitude_db, resp.phase_rad, resp.group_delay_samples, cutoff_freq_hz=cutoff), use_container_width=True)

        except Exception as e:
            st.error(f"Filter design error: {e}")

    render_theory_panel("filter")


def _render_resampling() -> None:
    st.subheader("5. Multirate DSP & Resampling Engine")

    col_ctrl, col_plot = st.columns([1, 2])

    with col_ctrl:
        mode = st.radio("Resampling Operation:", options=["Decimation (Downsampling)", "Interpolation (Upsampling)", "Rational Resampling"], index=0)
        fs_in = 1000.0
        f_sig = 50.0
        t = np.arange(500) / fs_in
        x_in = np.sin(2 * np.pi * f_sig * t)

        if mode.startswith("Decimation"):
            factor_m = st.slider("Decimation Factor M:", 2, 8, 2)
            res_obj = decimate_signal(x_in, factor_M=factor_m, sampling_rate_hz=fs_in)
        elif mode.startswith("Interpolation"):
            factor_l = st.slider("Interpolation Factor L:", 2, 8, 2)
            res_obj = interpolate_signal(x_in, factor_L=factor_l, sampling_rate_hz=fs_in)
        else:
            up_l = st.slider("Upsample L:", 2, 5, 3)
            down_m = st.slider("Downsample M:", 2, 5, 2)
            res_obj = resample_rational(x_in, up_L=up_l, down_M=down_m, sampling_rate_hz=fs_in)

    with col_plot:
        y_out = res_obj.processed_signal.amplitude if hasattr(res_obj.processed_signal, "amplitude") else res_obj.processed_signal
        t_out = res_obj.processed_signal.time_vector if hasattr(res_obj.processed_signal, "time_vector") else np.arange(len(y_out)) / res_obj.output_sampling_rate_hz
        st.pyplot(plot_time_domain(t_out, y_out, title=f"Resampled Signal (Fs = {res_obj.output_sampling_rate_hz:.1f} Hz)"), use_container_width=True)

        st.markdown(
            f"**Input Rate:** `{res_obj.input_sampling_rate_hz:.1f} Hz` | **Output Rate:** `{res_obj.output_sampling_rate_hz:.1f} Hz` | "
            f"**Input Samples:** `{res_obj.input_num_samples}` | **Output Samples:** `{res_obj.output_num_samples}`"
        )

    render_theory_panel("resampling")
