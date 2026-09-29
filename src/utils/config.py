"""Configuration models and schema definitions for RadarDSP Lab.

Provides type-safe, validated Pydantic models for both the DSP Laboratory
and FMCW Radar Laboratory parameter configurations. Supports serialization to
and deserialization from YAML and JSON configs.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class SignalGenConfig(BaseModel):
    """Configuration for DSP signal generation."""
    signal_type: str = Field(
        default="sine",
        description="Type of signal: sine, cosine, square, triangle, sawtooth, multi_tone, chirp, noise, am, fm"
    )
    amplitude: float = Field(default=1.0, description="Peak signal amplitude")
    frequency_hz: float = Field(default=100.0, description="Fundamental frequency in Hz")
    phase_rad: float = Field(default=0.0, description="Initial phase offset in radians")
    sampling_rate_hz: float = Field(default=1000.0, gt=0, description="Sampling frequency Fs in Hz")
    duration_sec: float = Field(default=1.0, gt=0, description="Signal duration in seconds")
    dc_offset: float = Field(default=0.0, description="DC bias offset")
    duty_cycle: float = Field(default=0.5, description="Duty cycle for square wave (0 to 1)")

    # Multi-tone parameters
    frequencies_hz: List[float] = Field(
        default_factory=lambda: [100.0],
        description="Tone frequencies in Hz for multi-tone signal"
    )
    amplitudes: List[float] = Field(
        default_factory=lambda: [1.0],
        description="Amplitudes for each tone in multi-tone signal"
    )
    phases_rad: List[float] = Field(
        default_factory=lambda: [0.0],
        description="Phase offsets in radians for each tone in multi-tone signal"
    )

    # Chirp parameters
    f_start_hz: float = Field(default=0.0, description="Chirp start frequency in Hz")
    f_end_hz: float = Field(default=500.0, description="Chirp end frequency in Hz")

    # AM / FM parameters
    carrier_frequency_hz: float = Field(default=100.0, description="Carrier frequency in Hz")
    modulation_frequency_hz: float = Field(default=10.0, description="Modulating signal frequency in Hz")
    modulation_index: float = Field(default=0.5, description="AM modulation index m")
    frequency_deviation_hz: float = Field(default=25.0, description="FM frequency deviation delta_f in Hz")

    # Noise parameters
    std_dev: float = Field(default=1.0, description="Standard deviation for Gaussian noise")
    seed: Optional[int] = Field(default=None, description="Random seed for deterministic noise generation")
    snr_db: Optional[float] = Field(default=None, description="Signal-to-Noise ratio in dB (None for clean signal)")


class FilterConfig(BaseModel):
    """Configuration for DSP filtering."""
    filter_type: str = Field(default="lowpass", description="Filter response type: lowpass, highpass, bandpass, bandstop")
    design_method: str = Field(default="firwin", description="Design method: firwin, butter, cheby1, ellip")
    cutoff_hz: List[float] = Field(default_factory=lambda: [200.0], description="Cutoff frequency/frequencies in Hz")
    order: int = Field(default=32, ge=1, description="Filter order / number of taps")
    window: str = Field(default="hamming", description="Window type for FIR design")


class TargetConfig(BaseModel):
    """Configuration for a single radar target."""
    range_m: float = Field(..., ge=0, description="Target range in meters")
    velocity_mps: float = Field(default=0.0, description="Target radial velocity in m/s (positive = receding)")
    rcs_sqm: float = Field(default=1.0, gt=0, description="Radar Cross Section (RCS) in m^2")
    amplitude: float = Field(default=1.0, gt=0, description="Linear reflection amplitude scale factor")
    phase_rad: float = Field(default=0.0, description="Target reflection phase offset in radians")
    target_id: Optional[str] = Field(default=None, description="Target identifier (e.g. 'T1')")


class CFARConfig(BaseModel):
    """Configuration for Constant False Alarm Rate (CFAR) detection."""
    method: str = Field(default="CA-CFAR", description="CFAR algorithm: CA-CFAR, GO-CFAR, SO-CFAR, OS-CFAR")
    num_guard_cells: int = Field(default=2, ge=0, description="Number of guard cells per side")
    num_reference_cells: int = Field(default=8, ge=1, description="Number of reference cells per side")
    pfa: float = Field(default=1e-5, gt=0, lt=1, description="Probability of False Alarm")


class RadarConfig(BaseModel):
    """Configuration for FMCW Radar Laboratory simulation."""
    carrier_frequency_hz: float = Field(default=77e9, gt=0, description="Carrier frequency fc in Hz (e.g., 77 GHz)")
    sweep_bandwidth_hz: float = Field(default=150e6, gt=0, description="Chirp sweep bandwidth B in Hz")
    chirp_duration_sec: float = Field(default=100e-6, gt=0, description="Chirp duration T_c in seconds")
    sampling_rate_hz: float = Field(default=20e6, gt=0, description="ADC sampling rate fs in Hz (default 20 MHz for robust beat Nyquist margin)")
    num_chirps: int = Field(default=64, ge=1, description="Number of chirps per frame for Doppler processing")
    tx_power_dbm: float = Field(default=10.0, description="Transmit power in dBm")
    noise_figure_db: float = Field(default=10.0, description="Receiver noise figure in dB")
    targets: List[TargetConfig] = Field(
        default_factory=lambda: [TargetConfig(range_m=50.0, velocity_mps=15.0, rcs_sqm=1.0)],
        description="List of radar targets"
    )
    cfar: CFARConfig = Field(default_factory=CFARConfig, description="CFAR detector configuration")


class LabConfig(BaseModel):
    """Top-level unified laboratory configuration container."""
    dsp: SignalGenConfig = Field(default_factory=SignalGenConfig)
    filter: FilterConfig = Field(default_factory=FilterConfig)
    radar: RadarConfig = Field(default_factory=RadarConfig)
