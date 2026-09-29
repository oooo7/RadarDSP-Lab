"""FMCW Radar Laboratory View for Streamlit App.

Provides user interface controls for FMCW chirp generation, propagation & targets,
beat signal generation, Range/Doppler processing, and CA-CFAR detection.
Signal processing logic is strictly decoupled and called from src.radar.
"""

import streamlit as st


def render_radar_lab() -> None:
    """Render the FMCW Radar Laboratory view in Streamlit."""
    st.header("📡 FMCW Radar Laboratory")
    st.markdown(
        """
        Interactive FMCW Radar Signal Processing simulator for target detection,
        range-Doppler map generation, CA-CFAR detection, and theoretical performance analysis.
        """
    )
    st.info("FMCW Radar Lab modules will be implemented incrementally starting in Phase 2.")
