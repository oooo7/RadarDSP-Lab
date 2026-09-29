"""Validation Metrics, Target Matching & Theoretical Limit Evaluation Engine.

Provides quantitative radar error metrics (Absolute Error, Relative Error, MAE, RMSE, StdDev),
Cramér-Rao Lower Bounds (CRLB), target-to-detection matching, and detection performance metrics.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple, Union
import numpy as np
import scipy.constants

from src.radar.cfar import CFARDetection
from src.radar.target import TargetState
from src.utils.config import RadarConfig, TargetConfig

# Speed of light in m/s
SPEED_OF_LIGHT_M_PER_S = float(scipy.constants.c)


@dataclass(frozen=True)
class TheoreticalLimits:
    """Immutable payload holding theoretical radar resolutions and Cramér-Rao Lower Bounds (CRLB).

    Attributes:
        range_resolution_m: Physical range resolution Delta R = c / (2*B) in meters.
        max_unambiguous_range_m: Maximum unambiguous range R_max = c*Fs / (4*S) in meters.
        velocity_resolution_mps: Velocity resolution Delta v = lambda / (2*M*T_c) in m/s.
        max_unambiguous_velocity_mps: Maximum unambiguous velocity v_max = lambda / (4*T_c) in m/s.
        snr_db: Signal-to-Noise Ratio in dB used for CRLB calculation.
        snr_linear: Linear Signal-to-Noise Ratio (10^(snr_db / 10)).
        range_crlb_m: Theoretical minimum range estimation standard deviation CRLB_R in meters.
        velocity_crlb_mps: Theoretical minimum velocity estimation standard deviation CRLB_v in m/s.
    """
    range_resolution_m: float
    max_unambiguous_range_m: float
    velocity_resolution_mps: float
    max_unambiguous_velocity_mps: float
    snr_db: float
    snr_linear: float
    range_crlb_m: float
    velocity_crlb_mps: float


@dataclass(frozen=True)
class EstimationErrorMetrics:
    """Immutable payload comparing true ground truth target against radar estimate.

    Attributes:
        true_range_m: Theoretical target range R_true in meters.
        estimated_range_m: Estimated target range R_est in meters.
        range_error_m: Signed range error (R_est - R_true) in meters.
        abs_range_error_m: Absolute range error |R_est - R_true| in meters.
        rel_range_error: Relative range error |R_est - R_true| / R_true (or 0.0 if R_true <= 0).
        true_velocity_mps: Theoretical target velocity v_true in m/s.
        estimated_velocity_mps: Estimated target velocity v_est in m/s.
        velocity_error_mps: Signed velocity error (v_est - v_true) in m/s.
        abs_velocity_error_mps: Absolute velocity error |v_est - v_true| in m/s.
        rel_velocity_error: Relative velocity error |v_est - v_true| / |v_true| (or 0.0 if |v_true| < 1e-3).
    """
    true_range_m: float
    estimated_range_m: float
    range_error_m: float
    abs_range_error_m: float
    rel_range_error: float
    true_velocity_mps: float
    estimated_velocity_mps: float
    velocity_error_mps: float
    abs_velocity_error_mps: float
    rel_velocity_error: float


@dataclass(frozen=True)
class AggregateErrorStats:
    """Immutable payload holding statistical summary over multiple trials (Mean, MAE, RMSE, Std, Max).

    Attributes:
        num_trials: Total number of evaluated trials/samples.
        mean_error: Signed arithmetic mean error.
        mean_absolute_error: Mean Absolute Error (MAE).
        rmse: Root Mean Square Error (RMSE).
        std_dev: Standard deviation of errors.
        max_absolute_error: Maximum absolute error observed.
    """
    num_trials: int
    mean_error: float
    mean_absolute_error: float
    rmse: float
    std_dev: float
    max_absolute_error: float


@dataclass(frozen=True)
class TargetMatch:
    """Immutable pair linking a ground truth TargetState to its matched CFARDetection candidate.

    Attributes:
        target: True TargetState object.
        detection: Matched CFARDetection candidate object.
        range_error_m: Absolute range error |R_est - R_true| in meters.
        velocity_error_mps: Absolute velocity error |v_est - v_true| in m/s.
        normalized_distance: Normalized Euclidean distance in resolution units.
    """
    target: TargetState
    detection: CFARDetection
    range_error_m: float
    velocity_error_mps: float
    normalized_distance: float


@dataclass(frozen=True)
class TargetMatchResult:
    """Immutable container holding one-to-one target matching results.

    Attributes:
        matches: List of TargetMatch pairs (True Positives).
        unmatched_targets: List of TargetState objects not detected (Missed Detections / False Negatives).
        unmatched_detections: List of CFARDetection candidates not matching targets (False Alarms / False Positives).
        total_targets: Total ground truth targets count.
        total_detections: Total CFAR detections count.
        num_true_positives: Count of correctly matched targets (TP).
        num_false_negatives: Count of missed targets (FN).
        num_false_positives: Count of spurious detections (FP).
        probability_of_detection: Pd = TP / (TP + FN).
    """
    matches: List[TargetMatch]
    unmatched_targets: List[TargetState]
    unmatched_detections: List[CFARDetection]
    total_targets: int
    total_detections: int
    num_true_positives: int
    num_false_negatives: int
    num_false_positives: int
    probability_of_detection: float


def compute_theoretical_limits(config: RadarConfig, snr_db: float) -> TheoreticalLimits:
    """Compute theoretical radar resolutions and Cramér-Rao Lower Bounds (CRLB).

    Formulas:
        Range Resolution: Delta R = c / (2 * B)
        Velocity Resolution: Delta v = lambda / (2 * M * T_c)
        Range CRLB: CRLB_R = c / (2 * B * sqrt(2 * SNR_linear))
        Velocity CRLB: CRLB_v = lambda / (2 * pi * T_frame * sqrt(2 * SNR_linear))
        where T_frame = M * T_c is total coherent observation time.

    Args:
        config: RadarConfig baseline parameters.
        snr_db: Signal-to-Noise Ratio in dB (> 0 for CRLB calculation).

    Returns:
        TheoreticalLimits container.
    """
    fc = float(config.carrier_frequency_hz)
    b = float(config.sweep_bandwidth_hz)
    tc = float(config.chirp_duration_sec)
    fs = float(config.sampling_rate_hz)
    m = int(config.num_chirps)
    snr_val = float(snr_db)

    wavelength = SPEED_OF_LIGHT_M_PER_S / fc
    slope = b / tc
    t_frame = m * tc
    snr_lin = 10.0**(snr_val / 10.0) if snr_val > -50 else 1e-5

    delta_r = SPEED_OF_LIGHT_M_PER_S / (2.0 * b)
    r_max = SPEED_OF_LIGHT_M_PER_S * fs / (4.0 * slope)
    delta_v = wavelength / (2.0 * t_frame)
    v_max = wavelength / (4.0 * tc)

    # CRLB bounds
    crlb_r = SPEED_OF_LIGHT_M_PER_S / (2.0 * b * np.sqrt(2.0 * snr_lin))
    crlb_v = wavelength / (2.0 * np.pi * t_frame * np.sqrt(2.0 * snr_lin))

    return TheoreticalLimits(
        range_resolution_m=delta_r,
        max_unambiguous_range_m=r_max,
        velocity_resolution_mps=delta_v,
        max_unambiguous_velocity_mps=v_max,
        snr_db=snr_val,
        snr_linear=snr_lin,
        range_crlb_m=crlb_r,
        velocity_crlb_mps=crlb_v
    )


def calculate_absolute_error(estimated: float, true_val: float) -> float:
    """Calculate signed error (estimated - true_val)."""
    return float(estimated) - float(true_val)


def calculate_relative_error(estimated: float, true_val: float) -> float:
    """Calculate relative error |estimated - true_val| / |true_val|.

    Safely handles true_val ≈ 0 by returning 0.0 to prevent division instability.
    """
    est = float(estimated)
    tru = float(true_val)
    abs_tru = abs(tru)
    if abs_tru < 1e-4:
        return 0.0
    return abs(est - tru) / abs_tru


def calculate_aggregate_metrics(errors: Union[np.ndarray, List[float]]) -> AggregateErrorStats:
    """Compute summary error metrics: Mean, MAE, RMSE, StdDev, Max Absolute Error.

    Args:
        errors: 1D array or list of signed error values (est - true).

    Returns:
        AggregateErrorStats container.
    """
    arr = np.asarray(errors, dtype=np.float64)
    if arr.size == 0:
        return AggregateErrorStats(
            num_trials=0,
            mean_error=0.0,
            mean_absolute_error=0.0,
            rmse=0.0,
            std_dev=0.0,
            max_absolute_error=0.0
        )

    mean_err = float(np.mean(arr))
    mae = float(np.mean(np.abs(arr)))
    rmse = float(np.sqrt(np.mean(arr**2)))
    std_err = float(np.std(arr))
    max_abs = float(np.max(np.abs(arr)))

    return AggregateErrorStats(
        num_trials=len(arr),
        mean_error=mean_err,
        mean_absolute_error=mae,
        rmse=rmse,
        std_dev=std_err,
        max_absolute_error=max_abs
    )


def evaluate_estimation_errors(
    ground_truth: TargetConfig,
    estimated_range_m: float,
    estimated_velocity_mps: float
) -> EstimationErrorMetrics:
    """Calculate exact range and velocity error metrics against ground truth target.

    Args:
        ground_truth: True TargetConfig object.
        estimated_range_m: Measured target range in meters.
        estimated_velocity_mps: Measured target velocity in m/s.

    Returns:
        EstimationErrorMetrics payload.
    """
    r_true = float(ground_truth.range_m)
    r_est = float(estimated_range_m)
    v_true = float(ground_truth.velocity_mps)
    v_est = float(estimated_velocity_mps)

    e_r = r_est - r_true
    abs_e_r = abs(e_r)
    rel_e_r = calculate_relative_error(r_est, r_true)

    e_v = v_est - v_true
    abs_e_v = abs(e_v)
    rel_e_v = calculate_relative_error(v_est, v_true)

    return EstimationErrorMetrics(
        true_range_m=r_true,
        estimated_range_m=r_est,
        range_error_m=e_r,
        abs_range_error_m=abs_e_r,
        rel_range_error=rel_e_r,
        true_velocity_mps=v_true,
        estimated_velocity_mps=v_est,
        velocity_error_mps=e_v,
        abs_velocity_error_mps=abs_e_v,
        rel_velocity_error=rel_e_v
    )


def match_targets_to_detections(
    targets: List[TargetState],
    detections: List[CFARDetection],
    range_gate_m: float,
    velocity_gate_mps: float
) -> TargetMatchResult:
    """Perform one-to-one optimal matching between theoretical targets and CFAR detections.

    Matching Algorithm:
        For each target k, candidate detections falling within the physical resolution gate:
        |R_det - R_target| <= range_gate_m AND |v_det - v_target| <= velocity_gate_mps
        are evaluated. The detection minimizing normalized distance:
        d = sqrt( ( (R_det - R_target) / range_gate_m )^2 + ( (v_det - v_target) / velocity_gate_mps )^2 )
        is assigned in a one-to-one fashion (greedy assignment by distance).

    Args:
        targets: List of theoretical TargetState objects.
        detections: List of CFARDetection candidates extracted from CFAR.
        range_gate_m: Gating tolerance in range direction (m).
        velocity_gate_mps: Gating tolerance in velocity direction (m/s).

    Returns:
        TargetMatchResult payload with matched pairs, missed targets, and false alarms.
    """
    if not targets:
        return TargetMatchResult(
            matches=[],
            unmatched_targets=[],
            unmatched_detections=list(detections),
            total_targets=0,
            total_detections=len(detections),
            num_true_positives=0,
            num_false_negatives=0,
            num_false_positives=len(detections),
            probability_of_detection=0.0
        )

    r_gate = float(range_gate_m)
    v_gate = float(velocity_gate_mps)

    # Compute candidate distance matrix between targets (rows) and detections (cols)
    candidate_list: List[Tuple[float, float, float, int, int]] = []

    for t_idx, t in enumerate(targets):
        for d_idx, d in enumerate(detections):
            e_r = abs(d.range_m - t.range_m)
            e_v = abs(d.velocity_mps - t.velocity_mps)

            if e_r <= r_gate and e_v <= v_gate:
                norm_dist = np.sqrt((e_r / r_gate)**2 + (e_v / v_gate)**2)
                candidate_list.append((norm_dist, e_r, e_v, t_idx, d_idx))

    # Sort candidates by normalized distance ascending
    candidate_list.sort(key=lambda x: x[0])

    matched_targets_set = set()
    matched_dets_set = set()
    matches: List[TargetMatch] = []

    for norm_dist, e_r, e_v, t_idx, d_idx in candidate_list:
        if t_idx not in matched_targets_set and d_idx not in matched_dets_set:
            matched_targets_set.add(t_idx)
            matched_dets_set.add(d_idx)
            matches.append(
                TargetMatch(
                    target=targets[t_idx],
                    detection=detections[d_idx],
                    range_error_m=e_r,
                    velocity_error_mps=e_v,
                    normalized_distance=norm_dist
                )
            )

    unmatched_targets = [targets[i] for i in range(len(targets)) if i not in matched_targets_set]
    unmatched_dets = [detections[j] for j in range(len(detections)) if j not in matched_dets_set]

    tp = len(matches)
    fn = len(unmatched_targets)
    fp = len(unmatched_dets)
    pd_val = tp / float(len(targets)) if targets else 0.0

    return TargetMatchResult(
        matches=matches,
        unmatched_targets=unmatched_targets,
        unmatched_detections=unmatched_dets,
        total_targets=len(targets),
        total_detections=len(detections),
        num_true_positives=tp,
        num_false_negatives=fn,
        num_false_positives=fp,
        probability_of_detection=pd_val
    )
