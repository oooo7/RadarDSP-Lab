"""RadarDSP Lab Main Streamlit Application Entry Point.

Sets up page configuration, navigation sidebar, and renders selected lab module.
"""

import streamlit as st
from app.views.dsp_lab import render_dsp_lab
from app.views.radar_lab import render_radar_lab


def main() -> None:
    """Main application launcher."""
    st.set_page_config(
        page_title="RadarDSP Lab",
        page_icon="📡",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.sidebar.title("📡 RadarDSP Lab")
    st.sidebar.markdown("**Interactive DSP & FMCW Radar Simulator**")
    st.sidebar.markdown("---")

    lab_mode = st.sidebar.radio(
        "Select Laboratory Module:",
        options=["🔬 DSP Laboratory", "📡 FMCW Radar Laboratory"],
        index=0
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("Phase 0 — Core Architecture Initialized")

    if lab_mode == "🔬 DSP Laboratory":
        render_dsp_lab()
    elif lab_mode == "📡 FMCW Radar Laboratory":
        render_radar_lab()


if __name__ == "__main__":
    main()
