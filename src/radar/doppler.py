"""FMCW Radar Multi-Target Doppler & Range-Doppler Processing Engine.

Provides multi-chirp FMCW data cube generation, multi-target echo synthesis,
2D Range-Doppler FFT matrix computation, velocity axis generation, peak extraction,
and theoretical-vs-estimated range and velocity validation helpers.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple
import numpy as np
import scipy.constants
import scipy.ndimage

from src.dsp.transforms import compute_fft
from src.radar.chirp import FMCWChirpContainer, generate_fmcw_chirp
from src.radar.processing import calculate_max_unambiguous_range, compute_range_fft
from src.radar.target import TargetState, compute_target_state
from src.utils.config import RadarConfig, TargetConfig

# Speed of light in m/s
SPEED_OF_LIGHT_M_PER_S = float(scipy.constants.c)


@dataclass(frozen=True)
class RadarDataCube:
    """Immutable data container for multi-chirp FMCW radar data cube.

    Dimensions:
        Axis 0 (slow-time): Chirp index m in [0, num_chirps - 1].
        Axis 1 (fast-time): ADC sample index n in [0, samples_per_chirp - 1].

    Attributes:
        data: 2D complex128 array of shape [num_chirps, samples_per_chirp].
        time_vector: 1D float64 fast-time vector t[n] in seconds.
        chirp_container: Reference transmit FMCWChirpContainer object.
        num_chirps: Total number of chirps M in slow-time dimension.
        samples_per_chirp: Number of ADC samples N in fast-time dimension.
        sampling_rate_hz: ADC sampling rate Fs in Hz.
        chirp_duration_sec: Single chirp duration T_c in seconds.
        carrier_frequency_hz: Operating carrier frequency fc in Hz.
        bandwidth_hz: Sweep bandwidth B in Hz.
        targets: List of TargetState objects present in the data cube.
        metadata: Additional metadata dictionary.
    """
    data: np.ndarray
    time_vector: np.ndarray
    chirp_container: FMCWChirpContainer
    num_chirps: int
    samples_per_chirp: int
    sampling_rate_hz: float
    chirp_duration_sec: float
    carrier_frequency_hz: float
    bandwidth_hz: float
    targets: List[TargetState]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def prf_hz(self) -> float:
        """Pulse Repetition Frequency PRF = 1 / T_c in Hz."""
        return 1.0 / self.chirp_duration_sec

    @property
    def wavelength_m(self) -> float:
        """Operating wavelength lambda = c / fc in meters."""
        return SPEED_OF_LIGHT_M_PER_S / self.carrier_frequency_hz

    @property
    def max_unambiguous_velocity_mps(self) -> float:
        """Maximum unambiguous Doppler velocity v_max = lambda / (4 * T_c) in m/s."""
        return self.wavelength_m / (4.0 * self.chirp_duration_sec)


@dataclass(frozen=True)
class RangeDopplerResult:
    """Immutable container holding 2D Range-Doppler processing matrix and axes.

    Matrix Orientation:
        matrix[doppler_bin, range_bin] of shape (num_doppler_bins, num_range_bins).

    Attributes:
        range_axis_m: 1D float64 array of range values in meters [0, R_max].
        velocity_axis_mps: 1D float64 array of velocity values in m/s [-v_max, +v_max].
        doppler_freq_axis_hz: 1D float64 array of Doppler frequencies in Hz [-PRF/2, +PRF/2].
        complex_matrix: 2D complex128 array of Range-Doppler spectrum.
        magnitude_matrix: 2D float64 array of linear magnitude spectrum |S(v, R)|.
        magnitude_db: 2D float64 array of magnitude spectrum in dB.
        range_resolution_m: Physical range resolution Delta R = c / (2*B) in meters.
        velocity_resolution_mps: Velocity resolution Delta v = lambda / (2 * M * T_c) in m/s.
        unambiguous_range_m: Maximum unambiguous range bound R_max in meters.
        unambiguous_velocity_mps: Maximum unambiguous velocity bound v_max in m/s.
        num_range_bins: Number of range bins (columns).
        num_doppler_bins: Number of Doppler bins (rows).
        targets_theoretical: List of theoretical TargetState objects simulated.
        metadata: Processing metadata dictionary.
    """
    range_axis_m: np.ndarray
    velocity_axis_mps: np.ndarray
    doppler_freq_axis_hz: np.ndarray
    complex_matrix: np.ndarray
    magnitude_matrix: np.ndarray
    magnitude_db: np.ndarray
    range_resolution_m: float
    velocity_resolution_mps: float
    unambiguous_range_m: float
    unambiguous_velocity_mps: float
    num_range_bins: int
    num_doppler_bins: int
    targets_theoretical: List[TargetState]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RangeDopplerPeak:
    """Immutable payload holding a detected Range-Doppler candidate peak.

    Attributes:
        range_m: Estimated target range in meters.
        velocity_mps: Estimated target velocity in m/s.
        magnitude: Spectral peak magnitude in linear Volts.
        magnitude_db: Spectral peak magnitude in dB.
        range_bin: Integer column index in Range-Doppler matrix.
        doppler_bin: Integer row index in Range-Doppler matrix.
        target_id: Optional target identifier string if matched.
    """
    range_m: float
    velocity_mps: float
    magnitude: float
    magnitude_db: float
    range_bin: int
    doppler_bin: int
    target_id: Optional[str] = None


@dataclass(frozen=True)
class ValidationEstimateResult:
    """Immutable validation metric result comparing theoretical value against estimated value.

    Attributes:
        parameter_name: 'range' or 'velocity'.
        theoretical_value: True physical value (m or m/s).
        estimated_value: Measured value from processing pipeline (m or m/s).
        absolute_error: Absolute error |R_est - R_true| or |v_est - v_true|.
        relative_error: Relative error |est - true| / |true|.
        is_within_resolution_cell: True if absolute error <= resolution cell size.
        resolution_cell_size: Physical resolution Delta R (m) or Delta v (m/s).
    """
    parameter_name: str
    theoretical_value: float
    estimated_value: float
    absolute_error: float
    relative_error: float
    is_within_resolution_cell: bool
    resolution_cell_size: float


def generate_multi_chirp_data_cube(
    config: RadarConfig,
    targets: List[TargetConfig],
    num_chirps: Optional[int] = None,
    sampling_architecture: str = "stretch_dechirp_analog"
) -> RadarDataCube:
    """Generate multi-chirp multi-target FMCW radar data cube via vectorized signal synthesis.

    Monostatic FMCW Multi-Target Model:
        For target k with initial range R_0k, radial velocity v_k, amplitude A_k, phase phi_0k:
        Target range at chirp m (slow-time m * T_c) is: R_k(m) = R_0k + v_k * m * T_c.
        Under the slow-motion narrowband approximation:
        - Beat frequency: f_bk = 2 * S * R_0k / c
        - Doppler frequency: f_Dk = 2 * v_k * f_c / c = 2 * v_k / lambda
        
        The de-chirped beat signal for chirp m at fast-time t[n] is the coherent sum:
        s_beat[m, n] = SUM_k A_k * exp( j * ( 2*pi*f_bk*t[n] + 2*pi*f_Dk*m*T_c + phi_const_k ) )

    Velocity Convention:
        Positive velocity (v > 0) means target is moving away (receding, range increasing).
        Negative velocity (v < 0) means target is approaching (closing, range decreasing).

    Args:
        config: FMCW RadarConfig object (carrier_frequency_hz, sweep_bandwidth_hz, chirp_duration_sec, etc.).
        targets: List of TargetConfig objects specifying range_m, velocity_mps, amplitude, rcs_sqm, target_id.
        num_chirps: Number of chirps M (slow-time dimension). Overrides config.num_chirps if specified.
        sampling_architecture: 'stretch_dechirp_analog' or 'direct_sampled_chirp'.

    Returns:
        RadarDataCube payload with data matrix of shape [num_chirps, samples_per_chirp].

    Raises:
        ValueError: If configuration or target list is invalid.
    """
    if not targets:
        raise ValueError("At least one target must be provided in targets list.")

    m_chirps = int(num_chirps if num_chirps is not None else config.num_chirps)
    if m_chirps < 1:
        raise ValueError(f"Number of chirps must be at least 1, got {m_chirps}")

    # Generate single transmit chirp container
    tx_chirp = generate_fmcw_chirp(config, sampling_architecture=sampling_architecture)
    fc = tx_chirp.carrier_frequency_hz
    s = tx_chirp.chirp_slope_hz_per_sec
    tc = tx_chirp.chirp_duration_sec
    fs = tx_chirp.sampling_rate_hz
    t_fast = tx_chirp.time_vector  # 1D array shape (N,)
    n_samples = len(t_fast)

    # Slow-time time vector: m * T_c
    m_indices = np.arange(m_chirps, dtype=np.float64)
    t_slow = m_indices * tc  # 1D array shape (M,)

    # Meshgrid vectors for vectorized 2D computation:
    # t_fast_2d shape (1, N), t_slow_2d shape (M, 1)
    t_fast_2d = t_fast[np.newaxis, :]
    t_slow_2d = t_slow[:, np.newaxis]

    data_cube = np.zeros((m_chirps, n_samples), dtype=np.complex128)
    target_states: List[TargetState] = []

    for idx, t_cfg in enumerate(targets):
        t_id = t_cfg.target_id if (hasattr(t_cfg, "target_id") and t_cfg.target_id) else f"T{idx+1}"
        # Attach default target_id if needed
        t_cfg_with_id = TargetConfig(
            range_m=t_cfg.range_m,
            velocity_mps=t_cfg.velocity_mps,
            rcs_sqm=t_cfg.rcs_sqm,
            amplitude=t_cfg.amplitude,
            phase_rad=t_cfg.phase_rad,
            target_id=t_id
        )

        t_state = compute_target_state(
            target_cfg=t_cfg_with_id,
            chirp_slope_hz_per_sec=s,
            amplitude_scaling=t_cfg.amplitude,
            phase_offset_rad=t_cfg.phase_rad,
            carrier_frequency_hz=fc
        )
        target_states.append(t_state)

        r0 = t_state.range_m
        v = t_state.velocity_mps
        alpha = t_state.amplitude_scaling
        phi0 = t_state.phase_offset_rad

        fb = t_state.theoretical_beat_frequency_hz
        fd = t_state.theoretical_doppler_frequency_hz
        tau0 = t_state.propagation_delay_sec

        # Constant residual phase term: 2*pi*fc*tau0 - pi*S*tau0^2 + phi0
        phi_const = 2.0 * np.pi * fc * tau0 - np.pi * s * (tau0**2) + phi0

        # Vectorized 2D beat signal synthesis across slow-time (rows) and fast-time (cols)
        # s_beat[m, n] = alpha * exp( j * ( 2*pi*fb*t_fast + 2*pi*fd*t_slow + phi_const ) )
        phase_matrix = 2.0 * np.pi * fb * t_fast_2d + 2.0 * np.pi * fd * t_slow_2d + phi_const
        s_beat_k = alpha * np.exp(1j * phase_matrix)

        data_cube += s_beat_k

    return RadarDataCube(
        data=data_cube,
        time_vector=t_fast,
        chirp_container=tx_chirp,
        num_chirps=m_chirps,
        samples_per_chirp=n_samples,
        sampling_rate_hz=fs,
        chirp_duration_sec=tc,
        carrier_frequency_hz=fc,
        bandwidth_hz=tx_chirp.bandwidth_hz,
        targets=target_states,
        metadata={
            "num_chirps": m_chirps,
            "samples_per_chirp": n_samples,
            "sampling_architecture": tx_chirp.sampling_architecture,
            "speed_of_light_m_per_s": SPEED_OF_LIGHT_M_PER_S,
        }
    )


def compute_range_doppler_map(
    data_cube: RadarDataCube,
    range_window: str = "hann",
    doppler_window: str = "hann",
    n_fft_range: Optional[int] = None,
    n_fft_doppler: Optional[int] = None
) -> RangeDopplerResult:
    """Compute 2D Range-Doppler matrix spectrum via 2D FFT processing.

    2D Signal Processing Chain:
        1. Fast-time Range FFT:
           For each chirp row m in [0, M-1], apply fast-time window and compute 1D FFT along fast-time samples.
           Produces complex range profiles matrix of shape (M, N_range).
        2. Slow-time Doppler FFT:
           For each range bin column r in [0, N_range-1], apply slow-time window and compute 1D FFT across chirps.
           Apply np.fft.fftshift to center 0 Hz / 0 m/s Doppler velocity in the middle.
           Produces complex 2D Range-Doppler matrix of shape (N_doppler, N_range).

    Args:
        data_cube: RadarDataCube input payload containing 2D data matrix [M, N].
        range_window: Window type for fast-time Range FFT ('rect', 'hann', 'hamming', 'blackman').
        doppler_window: Window type for slow-time Doppler FFT ('rect', 'hann', 'hamming', 'blackman').
        n_fft_range: Range FFT size (defaults to fast-time sample count N).
        n_fft_doppler: Doppler FFT size (defaults to chirp count M).

    Returns:
        RangeDopplerResult container holding 2D complex matrix, magnitude dB, and axes.
    """
    raw_data = data_cube.data  # shape (M, N)
    m_chirps, n_samples = raw_data.shape
    fs = data_cube.sampling_rate_hz
    s = data_cube.chirp_container.chirp_slope_hz_per_sec
    tc = data_cube.chirp_duration_sec
    fc = data_cube.carrier_frequency_hz
    b = data_cube.bandwidth_hz
    wavelength = data_cube.wavelength_m

    n_range = n_fft_range if n_fft_range is not None else n_samples
    m_doppler = n_fft_doppler if n_fft_doppler is not None else m_chirps

    if n_range < n_samples:
        n_range = n_samples
    if m_doppler < m_chirps:
        m_doppler = m_chirps

    # Step 1: Fast-time Range FFT along rows (axis=1) using Phase 3 FFT engine
    # Matrix shape after Range FFT: (M, N_range_bins)
    range_profiles = np.zeros((m_chirps, n_range // 2 + 1 if True else n_range), dtype=np.complex128)

    for m in range(m_chirps):
        spec_res = compute_range_fft(
            beat_signal_or_payload=raw_data[m, :],
            sampling_rate_hz=fs,
            chirp_slope_hz_per_sec=s,
            window_name=range_window,
            n_fft=n_range
        )
        range_profiles[m, :] = spec_res.spectrum

    # Frequency and Range Axis (one-sided positive frequency spectrum)
    range_freqs = spec_res.frequency_hz
    num_r_bins = len(range_freqs)
    range_axis = SPEED_OF_LIGHT_M_PER_S * range_freqs / (2.0 * s)

    # Step 2: Slow-time Doppler FFT along columns (axis=0)
    # Extract window function for slow time
    if doppler_window.lower() == "hann":
        win_doppler = np.hanning(m_chirps)
    elif doppler_window.lower() == "hamming":
        win_doppler = np.hamming(m_chirps)
    elif doppler_window.lower() == "blackman":
        win_doppler = np.blackman(m_chirps)
    elif doppler_window.lower() == "rect":
        win_doppler = np.ones(m_chirps, dtype=np.float64)
    else:
        win_doppler = np.hanning(m_chirps)

    # Normalize window for coherent gain preservation
    win_doppler = win_doppler / np.mean(win_doppler)
    win_doppler_2d = win_doppler[:, np.newaxis]  # shape (M, 1)

    # Apply slow-time windowing across chirps
    windowed_range_profiles = range_profiles * win_doppler_2d

    # Compute 1D FFT along slow-time axis (axis=0) and shift 0 Hz to center
    rd_complex = np.fft.fft(windowed_range_profiles, n=m_doppler, axis=0)
    rd_complex = np.fft.fftshift(rd_complex, axes=0)
    # Normalize by M chirps
    rd_complex = rd_complex / float(m_chirps)

    # Velocity and Doppler Frequency Axes
    # Doppler frequency axis centered around 0 Hz
    prf = 1.0 / tc
    doppler_freqs = np.fft.fftshift(np.fft.fftfreq(m_doppler, d=tc))  # in Hz [-PRF/2, +PRF/2]
    velocity_axis = doppler_freqs * wavelength / 2.0  # v = f_D * lambda / 2 in m/s

    magnitude_matrix = np.abs(rd_complex)
    # Prevent log10(0)
    mag_max = np.max(magnitude_matrix) if np.max(magnitude_matrix) > 0 else 1.0
    magnitude_db = 20.0 * np.log10(np.maximum(magnitude_matrix, 1e-12) / mag_max)

    delta_r = SPEED_OF_LIGHT_M_PER_S / (2.0 * b)
    delta_v = wavelength / (2.0 * m_chirps * tc)
    r_max = calculate_max_unambiguous_range(fs, s, is_complex=False)
    v_max = wavelength / (4.0 * tc)

    return RangeDopplerResult(
        range_axis_m=range_axis,
        velocity_axis_mps=velocity_axis,
        doppler_freq_axis_hz=doppler_freqs,
        complex_matrix=rd_complex,
        magnitude_matrix=magnitude_matrix,
        magnitude_db=magnitude_db,
        range_resolution_m=delta_r,
        velocity_resolution_mps=delta_v,
        unambiguous_range_m=r_max,
        unambiguous_velocity_mps=v_max,
        num_range_bins=num_r_bins,
        num_doppler_bins=len(velocity_axis),
        targets_theoretical=data_cube.targets,
        metadata={
            "range_window": range_window,
            "doppler_window": doppler_window,
            "num_chirps": m_chirps,
            "samples_per_chirp": n_samples,
            "n_fft_range": n_range,
            "n_fft_doppler": m_doppler,
            "matrix_orientation": "matrix[doppler_bin, range_bin] -> (num_doppler_bins, num_range_bins)",
        }
    )


def extract_range_doppler_peaks(
    rd_result: RangeDopplerResult,
    threshold_db: float = -25.0,
    min_range_bin_sep: int = 3,
    min_doppler_bin_sep: int = 3,
    max_peaks: int = 10
) -> List[RangeDopplerPeak]:
    """Extract candidate target peaks from Range-Doppler matrix using 2D local max detection.

    Note:
        This is a simple deterministic peak detector for Phase 7 demonstration.
        Constant False Alarm Rate (CFAR) detection will be introduced in Phase 8.

    Args:
        rd_result: RangeDopplerResult object.
        threshold_db: Detection threshold relative to peak magnitude in dB (e.g. -25 dB).
        min_range_bin_sep: Minimum separation between peaks in range bins.
        min_doppler_bin_sep: Minimum separation between peaks in Doppler bins.
        max_peaks: Maximum number of peaks to extract.

    Returns:
        List of RangeDopplerPeak objects sorted by magnitude in descending order.
    """
    mag_db = rd_result.magnitude_db
    mag_lin = rd_result.magnitude_matrix
    v_axis = rd_result.velocity_axis_mps
    r_axis = rd_result.range_axis_m

    # 2D local maximum filter footprint
    size_v = max(1, 2 * min_doppler_bin_sep + 1)
    size_r = max(1, 2 * min_range_bin_sep + 1)
    local_max = scipy.ndimage.maximum_filter(mag_db, size=(size_v, size_r))

    # Mask for local maxima above threshold
    is_peak = (mag_db == local_max) & (mag_db >= threshold_db)

    # Exclude edge bins to avoid boundary artifacts
    is_peak[0, :] = False
    is_peak[-1, :] = False
    is_peak[:, 0] = False
    is_peak[:, -1] = False

    peak_indices = np.argwhere(is_peak)  # list of [d_idx, r_idx]

    peaks: List[RangeDopplerPeak] = []
    for d_idx, r_idx in peak_indices:
        p_mag = float(mag_lin[d_idx, r_idx])
        p_db = float(mag_db[d_idx, r_idx])
        p_range = float(r_axis[r_idx])
        p_vel = float(v_axis[d_idx])

        # Match with theoretical target ID if close
        matched_id = None
        for t_theo in rd_result.targets_theoretical:
            if abs(p_range - t_theo.range_m) <= rd_result.range_resolution_m * 1.5 and \
               abs(p_vel - t_theo.velocity_mps) <= rd_result.velocity_resolution_mps * 2.0:
                matched_id = t_theo.target_id
                break

        peaks.append(
            RangeDopplerPeak(
                range_m=p_range,
                velocity_mps=p_vel,
                magnitude=p_mag,
                magnitude_db=p_db,
                range_bin=int(r_idx),
                doppler_bin=int(d_idx),
                target_id=matched_id
            )
        )

    # Sort by magnitude descending
    peaks.sort(key=lambda p: p.magnitude, reverse=True)
    return peaks[:max_peaks]


def validate_range_estimate(
    theoretical_range_m: float,
    estimated_range_m: float,
    range_resolution_m: float
) -> ValidationEstimateResult:
    """Validate estimated target range against physical theoretical range.

    Args:
        theoretical_range_m: True target range R_true in meters.
        estimated_range_m: Measured target range R_est in meters.
        range_resolution_m: Physical range resolution Delta R = c / (2*B) in meters.

    Returns:
        ValidationEstimateResult container.
    """
    r_true = float(theoretical_range_m)
    r_est = float(estimated_range_m)
    dr = float(range_resolution_m)

    abs_err = abs(r_est - r_true)
    rel_err = abs_err / r_true if r_true > 0 else 0.0
    within_cell = abs_err <= dr

    return ValidationEstimateResult(
        parameter_name="range",
        theoretical_value=r_true,
        estimated_value=r_est,
        absolute_error=abs_err,
        relative_error=rel_err,
        is_within_resolution_cell=within_cell,
        resolution_cell_size=dr
    )


def validate_velocity_estimate(
    theoretical_velocity_mps: float,
    estimated_velocity_mps: float,
    velocity_resolution_mps: float
) -> ValidationEstimateResult:
    """Validate estimated target velocity against physical theoretical velocity.

    Args:
        theoretical_velocity_mps: True target radial velocity v_true in m/s.
        estimated_velocity_mps: Measured target velocity v_est in m/s.
        velocity_resolution_mps: Velocity bin resolution Delta v in m/s.

    Returns:
        ValidationEstimateResult container.
    """
    v_true = float(theoretical_velocity_mps)
    v_est = float(estimated_velocity_mps)
    dv = float(velocity_resolution_mps)

    abs_err = abs(v_est - v_true)
    rel_err = abs_err / abs(v_true) if abs(v_true) > 0 else 0.0
    within_cell = abs_err <= dv

    return ValidationEstimateResult(
        parameter_name="velocity",
        theoretical_value=v_true,
        estimated_value=v_est,
        absolute_error=abs_err,
        relative_error=rel_err,
        is_within_resolution_cell=within_cell,
        resolution_cell_size=dv
    )
