"""Compact metric display and dataframe formatting helpers for RadarDSP Lab UI.
"""

from typing import Any, Dict, List, Optional
import pandas as pd
import streamlit as st


def render_metric_row(metrics: List[Dict[str, Any]]) -> None:
    """Render a row of metric cards.

    Args:
        metrics: List of dicts with keys: 'label', 'value', optional 'delta', optional 'help'.
    """
    cols = st.columns(len(metrics))
    for idx, m in enumerate(metrics):
        with cols[idx]:
            st.metric(
                label=m["label"],
                value=m["value"],
                delta=m.get("delta"),
                help=m.get("help")
            )


def format_detection_table(detections: List[Any]) -> pd.DataFrame:
    """Format a list of detection objects into a clean Pandas DataFrame for display.

    Args:
        detections: List of RangeDopplerPeak or CFARDetection or TargetState objects.

    Returns:
        Formatted Pandas DataFrame.
    """
    records = []
    for idx, d in enumerate(detections):
        rec = {}
        if hasattr(d, "target_id") and d.target_id:
            rec["Target ID"] = d.target_id
        else:
            rec["Target ID"] = f"T{idx+1}"

        if hasattr(d, "range_m"):
            rec["Range (m)"] = f"{d.range_m:.2f}"
        if hasattr(d, "velocity_mps"):
            rec["Velocity (m/s)"] = f"{d.velocity_mps:+.2f}"
        if hasattr(d, "magnitude_db"):
            rec["Magnitude (dB)"] = f"{d.magnitude_db:.1f}"
        elif hasattr(d, "snr_db"):
            rec["SNR (dB)"] = f"{d.snr_db:.1f}"
        elif hasattr(d, "power_db"):
            rec["Power (dB)"] = f"{d.power_db:.1f}"

        if hasattr(d, "range_bin"):
            rec["Range Bin"] = int(d.range_bin)
        if hasattr(d, "doppler_bin"):
            rec["Doppler Bin"] = int(d.doppler_bin)

        records.append(rec)

    return pd.DataFrame(records)
