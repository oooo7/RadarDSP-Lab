"""FMCW Radar 2D Cell-Averaging Constant False Alarm Rate (CA-CFAR) Detection Engine.

Provides adaptive thresholding on 2D Range-Doppler power maps, alpha threshold multiplier calculation,
boundary masking, and candidate detection extraction.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple, Union
import numpy as np
import scipy.ndimage

from src.radar.doppler import RangeDopplerResult
from src.utils.config import CFARConfig


@dataclass(frozen=True)
class CFARDetection:
    """Immutable payload holding a single target detection extracted from CFAR processing.

    Attributes:
        range_m: Target range in meters.
        velocity_mps: Target velocity in m/s.
        power: CUT signal power in linear units |S|^2.
        power_db: CUT signal power in dB (10 * log10(power / P_ref)).
        threshold_power: Adaptive local threshold power.
        threshold_db: Adaptive threshold in dB.
        snr_db: Local signal-to-noise ratio in dB: 10 * log10(power / local_noise_power).
        range_bin: Range bin index (column).
        doppler_bin: Doppler bin index (row).
    """
    range_m: float
    velocity_mps: float
    power: float
    power_db: float
    threshold_power: float
    threshold_db: float
    snr_db: float
    range_bin: int
    doppler_bin: int


@dataclass(frozen=True)
class CFARResult:
    """Immutable payload holding complete 2D CA-CFAR detection result matrix and metadata.

    Matrix Orientation:
        Arrays have shape (num_doppler_bins, num_range_bins).

    Attributes:
        detection_mask: 2D boolean numpy array (True for detected cells).
        threshold_power: 2D float64 array of local adaptive power thresholds.
        threshold_db: 2D float64 array of adaptive thresholds in dB.
        input_power: 2D float64 array of CUT input power values |S|^2.
        training_cell_count: Total training cells N_train used per window.
        pfa: Configured probability of false alarm Pfa.
        alpha: Computed CFAR multiplier alpha = N_train * (Pfa^(-1/N_train) - 1).
        num_detections: Total integer count of detected cells.
        range_axis_m: 1D float64 array of range values in meters.
        velocity_axis_mps: 1D float64 array of velocity values in m/s.
        detections: List of extracted CFARDetection objects.
        metadata: CFAR processing metadata dictionary.
    """
    detection_mask: np.ndarray
    threshold_power: np.ndarray
    threshold_db: np.ndarray
    input_power: np.ndarray
    training_cell_count: int
    pfa: float
    alpha: float
    num_detections: int
    range_axis_m: np.ndarray
    velocity_axis_mps: np.ndarray
    detections: List[CFARDetection]
    metadata: dict[str, Any] = field(default_factory=dict)


def calculate_ca_cfar_alpha(pfa: float, num_training_cells: int) -> float:
    """Calculate 2D Cell-Averaging CFAR threshold multiplier alpha.

    Equation:
        alpha = N * (Pfa^(-1 / N) - 1)
        where N is total number of training cells and Pfa is target false alarm probability.

    Args:
        pfa: Probability of false alarm Pfa (0 < Pfa < 1).
        num_training_cells: Total number of training cells N (> 0).

    Returns:
        CFAR multiplier alpha as a float64.

    Raises:
        ValueError: If pfa <= 0, pfa >= 1, or num_training_cells <= 0.
    """
    p_fa = float(pfa)
    n_train = int(num_training_cells)

    if p_fa <= 0.0 or p_fa >= 1.0 or not np.isfinite(p_fa):
        raise ValueError(f"Probability of false alarm Pfa must be strictly between 0 and 1, got {p_fa}")
    if n_train <= 0:
        raise ValueError(f"Number of training cells must be strictly positive (> 0), got {n_train}")

    return float(n_train * (p_fa**(-1.0 / n_train) - 1.0))


def run_ca_cfar(
    rd_result_or_matrix: Union[RangeDopplerResult, np.ndarray],
    cfar_cfg: Optional[CFARConfig] = None,
    pfa: float = 1e-4,
    num_guard_range: int = 2,
    num_guard_doppler: int = 2,
    num_train_range: int = 4,
    num_train_doppler: int = 4,
    min_threshold_floor_db: Optional[float] = None,
    range_axis_m: Optional[np.ndarray] = None,
    velocity_axis_mps: Optional[np.ndarray] = None
) -> CFARResult:
    """Run 2D Cell-Averaging CFAR (CA-CFAR) detection engine on Range-Doppler spectrum.

    Vectorized 2D Window Mechanics:
        Let outer window size = (2*T_D + 2*G_D + 1, 2*T_R + 2*G_R + 1).
        Let inner window size = (2*G_D + 1, 2*G_R + 1).
        N_train = N_outer - N_inner.
        Training sum = Outer window sum - Inner window sum.
        Local background noise power P_noise = Training sum / N_train.
        Threshold power P_thresh = alpha * P_noise.
        CUT is detected if CUT_power > P_thresh.

    Boundary Handling:
        Boundary cells where the full training/guard window extends past matrix edges
        are marked as INVALID / NON-DETECTION (detection_mask = False, threshold_db = NaN).
        No wrap-around (np.roll) is permitted.

    Args:
        rd_result_or_matrix: RangeDopplerResult container or 2D complex/magnitude numpy array.
        cfar_cfg: Optional CFARConfig pydantic object overriding parameters.
        pfa: Probability of false alarm Pfa (0 < Pfa < 1).
        num_guard_range: Guard cells per side in Range direction (G_R >= 0).
        num_guard_doppler: Guard cells per side in Doppler direction (G_D >= 0).
        num_train_range: Training cells per side in Range direction (T_R >= 1).
        num_train_doppler: Training cells per side in Doppler direction (T_D >= 1).
        min_threshold_floor_db: Optional minimum threshold floor in dB.
        range_axis_m: 1D float64 range axis if input is raw 2D array.
        velocity_axis_mps: 1D float64 velocity axis if input is raw 2D array.

    Returns:
        CFARResult container holding detection mask, threshold matrices, and detected candidates.

    Raises:
        ValueError: If input dimensions or CFAR window geometry are invalid.
    """
    # Extract inputs and options
    if isinstance(rd_result_or_matrix, RangeDopplerResult):
        rd_res = rd_result_or_matrix
        c_matrix = rd_res.complex_matrix
        r_axis = rd_res.range_axis_m
        v_axis = rd_res.velocity_axis_mps
    else:
        c_matrix = np.asarray(rd_result_or_matrix)
        r_axis = range_axis_m if range_axis_m is not None else np.arange(c_matrix.shape[1], dtype=np.float64)
        v_axis = velocity_axis_mps if velocity_axis_mps is not None else np.arange(c_matrix.shape[0], dtype=np.float64)

    if c_matrix.ndim != 2:
        raise ValueError(f"Range-Doppler matrix must be 2D, got {c_matrix.ndim}D array of shape {c_matrix.shape}")

    # Read config if provided
    if cfar_cfg is not None:
        target_pfa = float(cfar_cfg.pfa)
        g_r = int(cfar_cfg.num_guard_range)
        g_d = int(cfar_cfg.num_guard_doppler)
        t_r = int(cfar_cfg.num_train_range)
        t_d = int(cfar_cfg.num_train_doppler)
        floor_db = cfar_cfg.min_threshold_floor_db if cfar_cfg.min_threshold_floor_db is not None else min_threshold_floor_db
    else:
        target_pfa = float(pfa)
        g_r = int(num_guard_range)
        g_d = int(num_guard_doppler)
        t_r = int(num_train_range)
        t_d = int(num_train_doppler)
        floor_db = min_threshold_floor_db

    num_d_bins, num_r_bins = c_matrix.shape

    if g_r < 0 or g_d < 0:
        raise ValueError(f"Guard cells must be non-negative (>= 0), got G_R={g_r}, G_D={g_d}")
    if t_r < 1 or t_d < 1:
        raise ValueError(f"Training cells must be positive (>= 1), got T_R={t_r}, T_D={t_d}")

    size_outer_d = 2 * (t_d + g_d) + 1
    size_outer_r = 2 * (t_r + g_r) + 1
    size_inner_d = 2 * g_d + 1
    size_inner_r = 2 * g_r + 1

    if size_outer_d > num_d_bins or size_outer_r > num_r_bins:
        raise ValueError(
            f"CFAR window size ({size_outer_d}x{size_outer_r}) is larger than Range-Doppler map ({num_d_bins}x{num_r_bins})!"
        )

    n_outer = size_outer_d * size_outer_r
    n_inner = size_inner_d * size_inner_r
    n_train = n_outer - n_inner

    if n_train <= 0:
        raise ValueError(f"Total training cells must be positive, got N_train={n_train}")

    # Compute CFAR multiplier alpha
    alpha = calculate_ca_cfar_alpha(target_pfa, n_train)

    # Power domain processing: P[d, r] = |S[d, r]|^2
    if np.iscomplexobj(c_matrix):
        power_matrix = np.abs(c_matrix)**2
    else:
        # If real magnitude matrix was passed, square it to power domain
        power_matrix = (c_matrix.astype(np.float64))**2

    # Vectorized 2D window sum using SciPy constant boundary mode
    # Uniform filter computes sum(W) / N_w, so multiply by N_w to get raw sum
    sum_outer = scipy.ndimage.uniform_filter(
        power_matrix,
        size=(size_outer_d, size_outer_r),
        mode='constant',
        cval=0.0
    ) * float(n_outer)

    sum_inner = scipy.ndimage.uniform_filter(
        power_matrix,
        size=(size_inner_d, size_inner_r),
        mode='constant',
        cval=0.0
    ) * float(n_inner)

    sum_train = sum_outer - sum_inner
    local_noise_power = sum_train / float(n_train)
    # Prevent divide by zero / negative noise
    local_noise_power = np.maximum(local_noise_power, 1e-15)

    threshold_power = alpha * local_noise_power

    # Apply minimum threshold floor if requested
    if floor_db is not None:
        p_ref = np.max(power_matrix) if np.max(power_matrix) > 0 else 1.0
        floor_power = p_ref * (10.0**(float(floor_db) / 10.0))
        threshold_power = np.maximum(threshold_power, floor_power)

    # Raw detection mask
    raw_mask = power_matrix > threshold_power

    # Boundary handling: invalidate cells where full window does not fit
    border_d = t_d + g_d
    border_r = t_r + g_r

    detection_mask = np.zeros_like(raw_mask, dtype=bool)
    # Valid interior region
    detection_mask[border_d : num_d_bins - border_d, border_r : num_r_bins - border_r] = \
        raw_mask[border_d : num_d_bins - border_d, border_r : num_r_bins - border_r]

    # Convert threshold to dB relative to peak power
    p_peak = np.max(power_matrix) if np.max(power_matrix) > 0 else 1.0
    threshold_db = 10.0 * np.log10(np.maximum(threshold_power, 1e-15) / p_peak)
    input_power_db = 10.0 * np.log10(np.maximum(power_matrix, 1e-15) / p_peak)

    # Set NaN for boundary threshold dB visualization
    threshold_db_vis = threshold_db.copy()
    threshold_db_vis[:border_d, :] = np.nan
    threshold_db_vis[num_d_bins - border_d:, :] = np.nan
    threshold_db_vis[:, :border_r] = np.nan
    threshold_db_vis[:, num_r_bins - border_r:] = np.nan

    # Extract detected candidates
    det_indices = np.argwhere(detection_mask)
    detections: List[CFARDetection] = []

    for d_idx, r_idx in det_indices:
        p_cut = float(power_matrix[d_idx, r_idx])
        p_thresh = float(threshold_power[d_idx, r_idx])
        p_noise = float(local_noise_power[d_idx, r_idx])

        p_cut_db = float(input_power_db[d_idx, r_idx])
        p_thresh_db = float(threshold_db[d_idx, r_idx])
        snr_cell_db = 10.0 * np.log10(p_cut / p_noise) if p_noise > 0 else 0.0

        detections.append(
            CFARDetection(
                range_m=float(r_axis[r_idx]),
                velocity_mps=float(v_axis[d_idx]),
                power=p_cut,
                power_db=p_cut_db,
                threshold_power=p_thresh,
                threshold_db=p_thresh_db,
                snr_db=snr_cell_db,
                range_bin=int(r_idx),
                doppler_bin=int(d_idx),
            )
        )

    # Sort detections by power descending
    detections.sort(key=lambda d: d.power, reverse=True)

    return CFARResult(
        detection_mask=detection_mask,
        threshold_power=threshold_power,
        threshold_db=threshold_db_vis,
        input_power=power_matrix,
        training_cell_count=n_train,
        pfa=target_pfa,
        alpha=alpha,
        num_detections=len(detections),
        range_axis_m=r_axis,
        velocity_axis_mps=v_axis,
        detections=detections,
        metadata={
            "num_guard_range": g_r,
            "num_guard_doppler": g_d,
            "num_train_range": t_r,
            "num_train_doppler": t_d,
            "border_range": border_r,
            "border_doppler": border_d,
            "threshold_floor_db": floor_db,
        }
    )


def extract_cfar_detections(cfar_result: CFARResult) -> List[CFARDetection]:
    """Extract list of CFARDetection objects from CFARResult container.

    Args:
        cfar_result: CFARResult payload.

    Returns:
        List of CFARDetection objects.
    """
    return cfar_result.detections
