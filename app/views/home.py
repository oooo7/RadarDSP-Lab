"""Home Landing Page View for RadarDSP Lab.

Provides project overview, architecture flow diagram, technology stack badges,
and navigation pointers to DSP Lab, FMCW Radar Lab, and Validation Dashboard.
"""

import streamlit as st


def render_home_page() -> None:
    """Render the main landing page of RadarDSP Lab."""
    st.markdown(
        """
        # 📡 RADARDSP LAB
        ### *Interactive DSP & FMCW Radar Signal Processing Simulator*
        
        A modular academic and engineering laboratory for exploring **Digital Signal Processing (DSP)**
        and **Frequency-Modulated Continuous-Wave (FMCW) Radar Processing** through interactive, real-time experiments.
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    # Technology Badges
    st.markdown(
        """
        **Technology Stack & Reference Frameworks:**  
        `Python 3.10+` | `NumPy` | `SciPy` | `Matplotlib` | `Streamlit` | `MATLAB Golden Reference`
        """
    )

    st.markdown("---")

    # Major Capabilities Cards
    st.subheader("🔬 Core Laboratory Capabilities")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            #### 📡 DSP Engine
            - **Signal Generation:** 10 signal types (Sinusoids, Chirps, AM/FM, Noise).
            - **Sampling & Aliasing:** Nyquist theorem, foldover calculation.
            - **FFT & Spectrum:** Windowing, zero-padding, PSL, parabolic sub-bin.
            - **Digital Filtering:** FIR & IIR Butterworth SOS, zero-phase filtering.
            - **Resampling:** Multirate decimation, interpolation, rational rate conversion.
            """
        )

    with col2:
        st.markdown(
            """
            #### 🚗 FMCW Radar Engine
            - **Analog Stretch Dechirp:** Analog-equivalent beat signal synthesis.
            - **Single-Target Kinematics:** Beat frequency conversion, range estimation.
            - **Multi-Target Data Cube:** Multi-chirp slow-time vs fast-time matrix.
            - **2D Range-Doppler FFT:** Velocity axis mapping ($v = f_D \\lambda / 2$).
            - **Noise & Clutter Environment:** Complex AWGN and Rayleigh clutter.
            """
        )

    with col3:
        st.markdown(
            """
            #### 📊 Detection & Validation
            - **2D CA-CFAR Detector:** Adaptive thresholding with boundary masking.
            - **Error Analysis:** Range & velocity MAE, RMSE, relative error.
            - **SNR Sensitivity Sweeps:** $P_d$ and MAE vs SNR across +30 to -35 dB.
            - **False Alarm Verification:** Empirical vs target $P_{\\text{fa}}$.
            - **Monte Carlo Evaluator:** Multi-trial 100-run target tracking.
            """
        )

    st.markdown("---")

    # Pipeline Flow Sequence
    st.subheader("🔄 Signal Processing Pipeline Architecture")

    st.info(
        "Signal Synthesis ➔ Sampling / Digitization ➔ FFT / Digital Filtering ➔ FMCW Stretch Dechirp ➔ 2D Range-Doppler Matrix ➔ 2D CA-CFAR Detection ➔ Validation Metrics"
    )

    st.markdown("---")

    # Quick Navigation Pointers
    st.subheader("🚀 Getting Started")
    st.markdown(
        """
        Use the **Sidebar Navigation** on the left to select a laboratory module:
        1. **📡 DSP Lab:** Explore signal generation, sampling aliasing, spectral leakage, FIR/IIR filtering, and resampling.
        2. **🚗 FMCW Radar:** Experiment with stretch-dechirp range estimation, Range-Doppler maps, AWGN noise, and CA-CFAR.
        3. **📊 Validation:** Inspect empirical range/velocity accuracy, SNR sweeps, false alarm rates, and Monte Carlo results.
        """
    )
