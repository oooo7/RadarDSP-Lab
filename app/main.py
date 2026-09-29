"""RadarDSP Lab Main Streamlit Application Entry Point.

Sets up page configuration, navigation sidebar, and renders selected lab module:
Home, DSP Laboratory, FMCW Radar Laboratory, or Validation Dashboard.
"""

import streamlit as st
from app.views.dsp_lab import render_dsp_lab
from app.views.home import render_home_page
from app.views.radar_lab import render_radar_lab
from app.views.validation import render_validation_dashboard


def main() -> None:
    """Main application launcher."""
    st.set_page_config(
        page_title="RadarDSP Lab",
        page_icon="📡",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Sidebar Header & Branding
    st.sidebar.title("📡 RadarDSP Lab")
    st.sidebar.caption("Interactive DSP & FMCW Radar Simulator")
    st.sidebar.markdown("---")

    # Main Sidebar Navigation
    nav_option = st.sidebar.radio(
        "Navigation:",
        options=[
            "🏠 Home",
            "📡 DSP Laboratory",
            "🚗 FMCW Radar Laboratory",
            "📊 Validation Dashboard",
        ],
        index=0
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Phase 11 — Complete Interactive UI**")
    st.sidebar.caption("Scientific Core: Phases 1–10 Verified")

    # View Router
    if nav_option == "🏠 Home":
        render_home_page()
    elif nav_option == "📡 DSP Laboratory":
        render_dsp_lab()
    elif nav_option == "🚗 FMCW Radar Laboratory":
        render_radar_lab()
    elif nav_option == "📊 Validation Dashboard":
        render_validation_dashboard()


if __name__ == "__main__":
    main()
