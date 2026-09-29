"""Reusable educational theory expander panels for RadarDSP Lab UI.

Provides concise, mathematically accurate explanations for DSP and FMCW radar concepts.
"""

import streamlit as st


def render_theory_panel(concept: str) -> None:
    """Render a standardized expandable theory panel for a specific concept.

    Args:
        concept: Key identifying the concept (e.g., 'signal_gen', 'nyquist', 'fft',
                 'filter', 'resampling', 'fmcw_single', 'multi_target',
                 'range_doppler', 'noise_clutter', 'cfar', 'validation').
    """
    with st.expander("📘 Theory & Mathematical Background", expanded=False):
        if concept == "signal_gen":
            st.markdown(
                """
                **Signal Generation Principles:**
                - Sinusoidal signals are fundamental building blocks: $x(t) = A \\sin(2\\pi f t + \\phi) + D$.
                - Complex chirps vary instantaneous frequency linearly: $f_{\\text{inst}}(t) = f_0 + S t$, where slope $S = \\frac{f_1 - f_0}{T}$.
                - Amplitude Modulation (AM): $s(t) = [1 + m \\cdot x_m(t)] \\cos(2\\pi f_c t)$.
                - Frequency Modulation (FM): $s(t) = \\cos\\left(2\\pi f_c t + 2\\pi k_f \\int x_m(\\tau)d\\tau\\right)$.
                """
            )
        elif concept == "nyquist":
            st.markdown(
                """
                **Sampling Theorem & Aliasing:**
                - **Nyquist Rate:** $F_{\\text{Nyquist, rate}} = 2 f_{\\max}$ — Minimum sampling rate required to reconstruct a band-limited signal without aliasing.
                - **Nyquist Frequency:** $F_{\\text{Nyquist, freq}} = \\frac{F_s}{2}$ — The highest frequency that can be unambiguously represented at sampling rate $F_s$.
                - **Principal Alias Frequency:** When $f > F_s/2$, frequency folds back into $[0, F_s/2]$ via:
                  $$f_{\\text{alias}} = \\left| \\left( (f + F_s / 2) \\bmod F_s \\right) - F_s / 2 \\right|$$
                """
            )
        elif concept == "fft":
            st.markdown(
                """
                **Fast Fourier Transform (FFT) & Spectral Analysis:**
                - **Discrete Fourier Transform (DFT):** $X[k] = \\sum_{n=0}^{N-1} x[n] e^{-j 2\\pi k n / N}$.
                - **Frequency Resolution:** $\\Delta f = \\frac{F_s}{N}$ (determined purely by physical observation time $T_{\\text{obs}} = N / F_s$).
                - **Zero-Padding:** Appending zeros increases frequency grid sampling density, improving visual interpolation, but does **NOT** increase true physical spectral resolution.
                - **Windowing:** Reduces spectral leakage (Peak Sidelobe Level) at the cost of mainlobe widening and Equivalent Noise Bandwidth (ENBW) expansion.
                """
            )
        elif concept == "filter":
            st.markdown(
                """
                **Digital Filter Design:**
                - **FIR Filters:** Always stable, linear phase (symmetric coefficients), zero phase distortion with `filtfilt` zero-phase processing.
                - **IIR Butterworth Filters:** Maximally flat passband response, steep roll-off for low filter order, implemented via Second-Order Sections (SOS) for numerical stability.
                - **Causal vs Zero-Phase Filtering:** Causal filtering introduces group delay $\\tau_g(\\omega)$; zero-phase filtering processes forward and reverse, eliminating phase shift entirely.
                """
            )
        elif concept == "resampling":
            st.markdown(
                """
                **Multirate DSP & Resampling:**
                - **Decimation ($M$):** Lowpass anti-aliasing filtering with cutoff $f_c = \\frac{F_s}{2 M}$, followed by downsampling by factor $M$.
                - **Interpolation ($L$):** Upsampling by zero-insertion by factor $L$, followed by polyphase anti-imaging filtering.
                - **Rational Resampling ($L/M$):** Combined rate conversion achieving target sampling rate $F_{s,\\text{out}} = \\frac{L}{M} F_{s,\\text{in}}$.
                """
            )
        elif concept == "fmcw_single":
            st.markdown(
                """
                **FMCW Radar Single Target Kinematics:**
                - **Chirp Slope:** $S = \\frac{B}{T_c}$ (Hz/s).
                - **Round-Trip Propagation Delay:** $\\tau = \\frac{2 R}{c}$ (seconds).
                - **Dechirp Beat Frequency:** Mixing received echo $s_{\\text{rx}}(t)$ with $s_{\\text{tx}}^*(t)$ yields beat frequency:
                  $$f_b = S \\cdot \\tau = \\frac{2 S R}{c} \\implies R = \\frac{c f_b}{2 S}$$
                - **Physical Range Resolution:** $\\Delta R = \\frac{c}{2 B}$.
                - **Max Unambiguous Range:** $R_{\\max} = \\frac{c F_s}{2 S}$.
                """
            )
        elif concept == "multi_target":
            st.markdown(
                """
                **Multi-Target FMCW Signal Model:**
                - Superposition of $K$ targets with range $R_k$, radial velocity $v_k$, amplitude $A_k$:
                  $$s_{\\text{beat}}[m, n] = \\sum_{k=1}^K A_k \\exp\\left( j \\left( 2\\pi f_{b,k} t_{\text{fast}}[n] + 2\\pi f_{D,k} t_{\text{slow}}[m] + \\phi_{\\text{const},k} \\right) \\right)$$
                - **Doppler Shift:** $f_D = \\frac{2 v}{\\lambda} = \\frac{2 v f_c}{c}$.
                - **Sign Convention:** Positive velocity ($v > 0$) is receding (range increasing); negative velocity ($v < 0$) is approaching.
                """
            )
        elif concept == "range_doppler":
            st.markdown(
                """
                **2D Range-Doppler Matrix Processing:**
                - **Fast-Time (Rows):** 1D Range FFT across samples within each chirp yields range profiles.
                - **Slow-Time (Columns):** 1D Doppler FFT across $M$ chirps with `fftshift` yields radial velocity spectrum.
                - **Velocity Resolution:** $\\Delta v = \\frac{\\lambda}{2 M T_c}$.
                - **Max Unambiguous Velocity:** $v_{\\max} = \\frac{\\lambda}{4 T_c}$.
                """
            )
        elif concept == "noise_clutter":
            st.markdown(
                """
                **Noise & Background Clutter:**
                - **Complex AWGN:** $w[n] \\sim \\mathcal{CN}(0, \\sigma_n^2)$ with equal variance $\\sigma_n^2/2$ on real and imaginary parts.
                - **Signal-to-Noise Ratio (SNR):** $\\text{SNR}_{dB} = 10 \\log_{10} \\left( \\frac{P_{\\text{signal}}}{P_{\\text{noise}}} \\right)$.
                - **Statistical Clutter:** Complex Gaussian background clutter creating Rayleigh magnitude fluctuations.
                """
            )
        elif concept == "cfar":
            st.markdown(
                """
                **2D Cell-Averaging Constant False Alarm Rate (CA-CFAR):**
                - **Threshold Multiplier:** $\\alpha = N_{\\text{train}} \\cdot \\left( P_{\\fa}^{-1 / N_{\\train}} - 1 \\right)$.
                - **Adaptive Thresholding:** Local noise power $P_{\\text{noise}}$ is estimated by averaging training cells surrounding the Cell Under Test (CUT), excluding guard cells.
                - **Boundary Handling:** Outer border cells where the training window extends beyond the matrix edge are marked as invalid non-detections.
                - **Detection vs Target:** CFAR identifies spectral power peaks exceeding local background noise (CFAR-positive cells). Target extraction requires matching candidates to physical kinematics.
                """
            )
        elif concept == "validation":
            st.markdown(
                """
                **Scientific Validation & Error Metrics:**
                - **Mean Absolute Error (MAE):** $\\text{MAE} = \\frac{1}{K} \\sum_{i=1}^K |e_i|$.
                - **Root Mean Square Error (RMSE):** $\\text{RMSE} = \\sqrt{\\frac{1}{K} \\sum_{i=1}^K e_i^2}$.
                - **Idealized Theoretical Limits:** Continuous-time asymptotic reference limits $\\text{CRLB}_R = \\frac{c}{2 B \\sqrt{2 \\cdot \\text{SNR}}}$ and $\\text{CRLB}_v = \\frac{\\lambda}{2 \\pi T_{\\text{frame}} \\sqrt{2 \\cdot \\text{SNR}}}$ represent ideal bounds assuming continuous unwindowed signals without grid quantization loss.
                """
            )
        else:
            st.info("Theoretical details available in `docs/radar_theory.md`.")
