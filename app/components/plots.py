"""Standardized Matplotlib visualization components for RadarDSP Lab UI.

Provides clean, publication-quality plotting utilities for time-domain signals,
spectral analysis, filter responses, FMCW radar processing chains, Range-Doppler heatmaps,
and 2D CFAR detection maps.
"""

from typing import List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np


def set_plot_style() -> None:
    """Apply clean scientific dark/slate plot theme."""
    plt.rcParams.update({
        "figure.facecolor": "#0E1117",
        "axes.facecolor": "#161B22",
        "axes.edgecolor": "#30363D",
        "axes.labelcolor": "#E6EDE3",
        "axes.grid": True,
        "grid.color": "#21262D",
        "grid.linestyle": "--",
        "grid.alpha": 0.6,
        "text.color": "#E6EDE3",
        "xtick.color": "#8B949E",
        "ytick.color": "#8B949E",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
    })


def plot_time_domain(
    t: np.ndarray,
    x: np.ndarray,
    title: str = "Time-Domain Waveform",
    xlabel: str = "Time (s)",
    ylabel: str = "Amplitude (V)",
    color: str = "#58A6FF"
) -> plt.Figure:
    """Plot 1D time-domain waveform."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 3.2), dpi=100)
    
    if np.iscomplexobj(x):
        ax.plot(t * 1e3 if max(t) < 0.01 else t, np.real(x), label="Re", color="#58A6FF", lw=1.5)
        ax.plot(t * 1e3 if max(t) < 0.01 else t, np.imag(x), label="Im", color="#FF7B72", lw=1.2, linestyle="--")
        ax.legend(loc="upper right", facecolor="#161B22", edgecolor="#30363D")
    else:
        ax.plot(t * 1e3 if max(t) < 0.01 else t, x, color=color, lw=1.5)
        
    time_unit = "ms" if max(t) < 0.01 else "s"
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(f"Time ({time_unit})" if xlabel == "Time (s)" else xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig


def plot_spectrum(
    freqs: np.ndarray,
    mag: np.ndarray,
    title: str = "Frequency Spectrum",
    db_scale: bool = True,
    color: str = "#7EE787"
) -> plt.Figure:
    """Plot 1D frequency magnitude spectrum."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 3.2), dpi=100)

    # Autoscale frequency axis (Hz vs kHz vs MHz)
    max_f = np.max(freqs)
    if max_f >= 1e6:
        f_scale = freqs / 1e6
        f_unit = "MHz"
    elif max_f >= 1e3:
        f_scale = freqs / 1e3
        f_unit = "kHz"
    else:
        f_scale = freqs
        f_unit = "Hz"

    if db_scale:
        p_max = np.max(mag) if np.max(mag) > 0 else 1.0
        mag_db = 20.0 * np.log10(np.maximum(mag, 1e-12) / p_max)
        ax.plot(f_scale, mag_db, color=color, lw=1.5)
        ax.set_ylabel("Magnitude (dB)")
        ax.set_ylim([-80, 5])
    else:
        ax.plot(f_scale, mag, color=color, lw=1.5)
        ax.set_ylabel("Magnitude (Volts)")

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(f"Frequency ({f_unit})")
    fig.tight_layout()
    return fig


def plot_time_and_frequency(
    t: np.ndarray,
    x: np.ndarray,
    freqs: np.ndarray,
    mag: np.ndarray,
    title_prefix: str = "Signal Analysis"
) -> plt.Figure:
    """Plot dual-panel time domain waveform and frequency spectrum side by side."""
    set_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5), dpi=100)

    # Panel 1: Time domain
    time_unit = "ms" if max(t) < 0.01 else "s"
    t_scale = t * 1e3 if max(t) < 0.01 else t
    if np.iscomplexobj(x):
        ax1.plot(t_scale, np.real(x), label="Re", color="#58A6FF", lw=1.5)
        ax1.plot(t_scale, np.imag(x), label="Im", color="#FF7B72", lw=1.2, linestyle="--")
        ax1.legend(loc="upper right", facecolor="#161B22", edgecolor="#30363D")
    else:
        ax1.plot(t_scale, x, color="#58A6FF", lw=1.5)
    ax1.set_title(f"{title_prefix}: Time Waveform", fontweight="bold")
    ax1.set_xlabel(f"Time ({time_unit})")
    ax1.set_ylabel("Amplitude (V)")

    # Panel 2: Frequency spectrum
    max_f = np.max(freqs)
    if max_f >= 1e6:
        f_scale = freqs / 1e6
        f_unit = "MHz"
    elif max_f >= 1e3:
        f_scale = freqs / 1e3
        f_unit = "kHz"
    else:
        f_scale = freqs
        f_unit = "Hz"

    ax2.plot(f_scale, mag, color="#7EE787", lw=1.5)
    ax2.set_title(f"{title_prefix}: Spectrum", fontweight="bold")
    ax2.set_xlabel(f"Frequency ({f_unit})")
    ax2.set_ylabel("Magnitude (V)")

    fig.tight_layout()
    return fig


def plot_filter_response(
    freqs_hz: np.ndarray,
    mag_db: np.ndarray,
    phase_rad: Optional[np.ndarray] = None,
    group_delay: Optional[np.ndarray] = None,
    cutoff_freq_hz: Union[float, List[float]] = 0.0
) -> plt.Figure:
    """Plot digital filter magnitude response, phase, and group delay."""
    set_plot_style()
    num_panels = 2 if phase_rad is not None else 1
    fig, axes = plt.subplots(num_panels, 1, figsize=(8, 3.2 * num_panels), dpi=100)
    
    ax_mag = axes[0] if num_panels > 1 else axes

    max_f = np.max(freqs_hz)
    if max_f >= 1e6:
        f_scale = freqs_hz / 1e6
        f_unit = "MHz"
    elif max_f >= 1e3:
        f_scale = freqs_hz / 1e3
        f_unit = "kHz"
    else:
        f_scale = freqs_hz
        f_unit = "Hz"

    ax_mag.plot(f_scale, mag_db, color="#D2A8FF", lw=1.8, label="Magnitude (dB)")
    ax_mag.axhline(-3.0, color="#FF7B72", linestyle=":", label="-3 dB Cutoff Line")
    
    # Mark cutoffs
    if isinstance(cutoff_freq_hz, (int, float)) and cutoff_freq_hz > 0:
        c_val = cutoff_freq_hz / (1e6 if max_f >= 1e6 else (1e3 if max_f >= 1e3 else 1.0))
        ax_mag.axvline(c_val, color="#FFA657", linestyle="--", label=f"fc = {cutoff_freq_hz:.1f} Hz")
    elif isinstance(cutoff_freq_hz, list):
        for fc in cutoff_freq_hz:
            c_val = fc / (1e6 if max_f >= 1e6 else (1e3 if max_f >= 1e3 else 1.0))
            ax_mag.axvline(c_val, color="#FFA657", linestyle="--", label=f"fc = {fc:.1f} Hz")

    ax_mag.set_title("Filter Frequency Magnitude Response |H(f)|", fontweight="bold")
    ax_mag.set_xlabel(f"Frequency ({f_unit})")
    ax_mag.set_ylabel("Magnitude (dB)")
    ax_mag.set_ylim([-80, 5])
    ax_mag.legend(loc="lower left", facecolor="#161B22", edgecolor="#30363D")

    if num_panels > 1 and phase_rad is not None:
        ax_phase = axes[1]
        ax_phase.plot(f_scale, np.degrees(phase_rad), color="#79C0FF", lw=1.5)
        ax_phase.set_title("Filter Phase Response (Degrees)", fontweight="bold")
        ax_phase.set_xlabel(f"Frequency ({f_unit})")
        ax_phase.set_ylabel("Phase (deg)")

    fig.tight_layout()
    return fig


def plot_fmcw_processing_chain(
    t_fast: np.ndarray,
    s_beat: np.ndarray,
    r_axis: np.ndarray,
    r_mag: np.ndarray,
    est_range_m: float,
    true_range_m: float,
    fc: float = 77e9,
    b_hz: float = 150e6,
    tc_sec: float = 100e-6
) -> plt.Figure:
    """Plot 4-panel FMCW radar single-target stretch-dechirp processing chain."""
    set_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), dpi=100)

    # Panel 1: Transmit FMCW Chirp frequency slope concept
    t_chirp_us = np.linspace(0, tc_sec * 1e6, 200)
    f_inst_ghz = (fc + (b_hz / tc_sec) * (t_chirp_us * 1e-6)) / 1e9
    axes[0, 0].plot(t_chirp_us, f_inst_ghz, color="#FFA657", lw=2)
    axes[0, 0].set_title("1. Transmit LFM Chirp Sweep f(t)", fontweight="bold")
    axes[0, 0].set_xlabel("Time (µs)")
    axes[0, 0].set_ylabel("Frequency (GHz)")

    # Panel 2: Dechirped Beat Signal (Fast-Time Waveform)
    t_fast_us = t_fast * 1e6
    axes[0, 1].plot(t_fast_us, np.real(s_beat), color="#58A6FF", lw=1.2, label="Re{s_beat}")
    axes[0, 1].set_title("2. Dechirped Beat Signal s_beat(t)", fontweight="bold")
    axes[0, 1].set_xlabel("Fast-Time (µs)")
    axes[0, 1].set_ylabel("Amplitude (V)")
    axes[0, 1].legend(loc="upper right", facecolor="#161B22", edgecolor="#30363D")

    # Panel 3: Beat Spectrum (Linear Magnitude)
    # Estimate beat frequency from peak
    peak_idx = np.argmax(r_mag)
    r_max_range = r_axis[-1]
    axes[1, 0].plot(r_axis, r_mag, color="#7EE787", lw=1.5)
    axes[1, 0].set_title("3. Range FFT Spectrum (Linear Magnitude)", fontweight="bold")
    axes[1, 0].set_xlabel("Range (m)")
    axes[1, 0].set_ylabel("Magnitude (V)")
    axes[1, 0].set_xlim([0, r_max_range])

    # Panel 4: Range Estimate dB Peak
    r_mag_db = 20.0 * np.log10(np.maximum(r_mag, 1e-12) / np.max(r_mag))
    axes[1, 1].plot(r_axis, r_mag_db, color="#D2A8FF", lw=1.5)
    axes[1, 1].axvline(true_range_m, color="#FF7B72", linestyle="--", label=f"True R = {true_range_m:.1f} m")
    axes[1, 1].axvline(est_range_m, color="#79C0FF", linestyle=":", label=f"Est R = {est_range_m:.2f} m")
    axes[1, 1].set_title("4. Estimated Target Range Spectrum (dB)", fontweight="bold")
    axes[1, 1].set_xlabel("Range (m)")
    axes[1, 1].set_ylabel("Magnitude (dB)")
    axes[1, 1].set_xlim([0, r_max_range])
    axes[1, 1].set_ylim([-60, 5])
    axes[1, 1].legend(loc="upper right", facecolor="#161B22", edgecolor="#30363D")

    fig.tight_layout()
    return fig


def plot_range_doppler_heatmap(
    range_axis_m: np.ndarray,
    velocity_axis_mps: np.ndarray,
    rd_magnitude_db: np.ndarray,
    title: str = "2D Range-Doppler Matrix Spectrum"
) -> plt.Figure:
    """Plot professional 2D Range-Doppler heatmap."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(9, 5), dpi=100)

    # Extent bounds: [xmin, xmax, ymin, ymax] -> [r_min, r_max, v_min, v_max]
    extent = [range_axis_m[0], range_axis_m[-1], velocity_axis_mps[0], velocity_axis_mps[-1]]

    im = ax.imshow(
        rd_magnitude_db,
        extent=extent,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=-50,
        vmax=0
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Magnitude (dB)", color="#E6EDE3")
    cbar.ax.yaxis.set_tick_params(color="#8B949E")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#8B949E")

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Radial Velocity (m/s)")

    # Draw 0 m/s velocity line
    ax.axhline(0.0, color="#FF7B72", linestyle=":", alpha=0.7, label="0 m/s Stationary Baseline")
    ax.legend(loc="upper right", facecolor="#161B22", edgecolor="#30363D")

    fig.tight_layout()
    return fig


def plot_cfar_detection_map(
    range_axis_m: np.ndarray,
    velocity_axis_mps: np.ndarray,
    rd_power_db: np.ndarray,
    threshold_db: np.ndarray,
    detection_mask: np.ndarray
) -> plt.Figure:
    """Plot 3-panel CA-CFAR evaluation: Power Map, Adaptive Threshold, Detection Mask."""
    set_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), dpi=100)
    extent = [range_axis_m[0], range_axis_m[-1], velocity_axis_mps[0], velocity_axis_mps[-1]]

    # Panel 1: Input Power Map
    im1 = axes[0].imshow(rd_power_db, extent=extent, origin="lower", aspect="auto", cmap="magma", vmin=-50, vmax=0)
    axes[0].set_title("1. RD Input Power (dB)", fontweight="bold")
    axes[0].set_xlabel("Range (m)")
    axes[0].set_ylabel("Velocity (m/s)")
    fig.colorbar(im1, ax=axes[0])

    # Panel 2: Adaptive Threshold Map
    im2 = axes[1].imshow(threshold_db, extent=extent, origin="lower", aspect="auto", cmap="plasma")
    axes[1].set_title("2. Adaptive Threshold (dB)", fontweight="bold")
    axes[1].set_xlabel("Range (m)")
    axes[1].set_ylabel("Velocity (m/s)")
    fig.colorbar(im2, ax=axes[1])

    # Panel 3: Binary Detection Mask
    axes[2].imshow(detection_mask, extent=extent, origin="lower", aspect="auto", cmap="binary")
    axes[2].set_title("3. CFAR Detection Mask", fontweight="bold")
    axes[2].set_xlabel("Range (m)")
    axes[2].set_ylabel("Velocity (m/s)")

    fig.tight_layout()
    return fig
