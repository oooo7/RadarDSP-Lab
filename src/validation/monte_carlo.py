"""FMCW Radar Monte Carlo Simulation & Validation Engine.

Provides automated multi-trial stochastic simulations across variable SNR,
clutter density, and multi-target scenarios to evaluate radar detection
performance, estimation accuracy (RMSE, MAE), false alarm behavior, and Monte Carlo variability.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import time
import numpy as np
import scipy.constants

from src.radar.cfar import CFARConfig, CFARResult, run_ca_cfar
from src.radar.doppler import (
    RadarDataCube,
    RangeDopplerResult,
    compute_range_doppler_map,
    generate_multi_chirp_data_cube,
)
from src.radar.noise import compose_radar_environment
from src.radar.target import TargetState
from src.utils.config import RadarConfig, TargetConfig
from src.validation.metrics import (
    AggregateErrorStats,
    TargetMatchResult,
    calculate_aggregate_metrics,
    compute_theoretical_limits,
    match_targets_to_detections,
)

SPEED_OF_LIGHT_M_PER_S = float(scipy.constants.c)


@dataclass
class MonteCarloConfig:
    """Configuration payload for Monte Carlo radar simulations.

    Attributes:
        num_trials: Total number of independent stochastic trials (e.g. 100).
        master_seed: Master random seed for deterministic reproducibility.
        snr_db: Signal-to-Noise Ratio in dB (or None for noise-free baseline).
        enable_clutter: If True, adds statistical background clutter.
        clutter_power: Clutter power P_clutter (or None if disabled).
        range_gate_mult: Range matching tolerance multiplier (default 1.5 * Delta_R).
        velocity_gate_mult: Velocity matching tolerance multiplier (default 1.5 * Delta_v).
        cfar_config: CFAR algorithm parameters (Pfa, window sizes).
    """
    num_trials: int = 100
    master_seed: int = 42
    snr_db: Optional[float] = None
    enable_clutter: bool = False
    clutter_power: Optional[float] = None
    range_gate_mult: float = 1.5
    velocity_gate_mult: float = 1.5
    cfar_config: Optional[CFARConfig] = None


@dataclass
class MonteCarloResult:
    """Immutable payload holding aggregated multi-trial Monte Carlo results.

    Attributes:
        config: MonteCarloConfig used for run.
        radar_config: Baseline RadarConfig.
        targets_theoretical: List of ground-truth TargetState objects.
        num_trials: Number of executed trials.
        per_trial_matches: List of TargetMatchResult payloads per trial.
        aggregate_range_stats: Dictionary mapping target_id -> AggregateErrorStats for range.
        aggregate_velocity_stats: Dictionary mapping target_id -> AggregateErrorStats for velocity.
        detection_probability_per_target: Dictionary mapping target_id -> Pd.
        overall_detection_probability: Mean Pd across all targets and trials.
        total_missed_detections: Count of missed target opportunities (False Negatives).
        total_false_detections: Count of spurious detections (False Positives).
        execution_time_sec: Wall-clock execution time in seconds.
        metadata: Metadata dictionary.
    """
    config: MonteCarloConfig
    radar_config: RadarConfig
    targets_theoretical: List[TargetState]
    num_trials: int
    per_trial_matches: List[TargetMatchResult]
    aggregate_range_stats: Dict[str, AggregateErrorStats]
    aggregate_velocity_stats: Dict[str, AggregateErrorStats]
    detection_probability_per_target: Dict[str, float]
    overall_detection_probability: float
    total_missed_detections: int
    total_false_detections: int
    execution_time_sec: float
    metadata: Dict[str, Any] = field(default_factory=dict)


def run_monte_carlo_simulation(
    config: RadarConfig,
    targets: List[TargetConfig],
    mc_config: Optional[MonteCarloConfig] = None,
    num_chirps: Optional[int] = None
) -> MonteCarloResult:
    """Execute reproducible Monte Carlo simulation over multi-chirp FMCW radar trials.

    Reproducibility:
        Trial i uses deterministic trial seed:
        trial_seed = master_seed + 1000 * i + 1

    Args:
        config: Baseline RadarConfig.
        targets: List of TargetConfig ground-truth target specifications.
        mc_config: Optional MonteCarloConfig parameters (defaults to 100 trials, seed 42).
        num_chirps: Optional override for chirp count M.

    Returns:
        MonteCarloResult container with aggregate statistics and matching records.

    Raises:
        ValueError: If configuration or parameters are invalid.
    """
    if not targets:
        raise ValueError("Must provide at least one target in targets list.")

    mc_cfg = mc_config if mc_config is not None else MonteCarloConfig()
    if mc_cfg.num_trials < 1:
        raise ValueError(f"Number of trials must be >= 1, got {mc_cfg.num_trials}")

    start_time = time.time()

    # Step 1: Synthesize reference clean data cube once
    clean_cube = generate_multi_chirp_data_cube(
        config=config,
        targets=targets,
        num_chirps=num_chirps
    )
    ground_truth_targets = clean_cube.targets

    # Determine matching gates based on physical resolution
    b = clean_cube.bandwidth_hz
    tc = clean_cube.chirp_duration_sec
    m_chirps = clean_cube.num_chirps
    wavelength = clean_cube.wavelength_m

    res_r = SPEED_OF_LIGHT_M_PER_S / (2.0 * b)
    res_v = wavelength / (2.0 * m_chirps * tc)

    gate_r = mc_cfg.range_gate_mult * res_r
    gate_v = mc_cfg.velocity_gate_mult * res_v

    cfar_cfg = mc_cfg.cfar_config if mc_cfg.cfar_config is not None else CFARConfig()

    per_trial_matches: List[TargetMatchResult] = []
    
    # Track errors per target ID
    target_ids = [t.target_id for t in ground_truth_targets]
    target_range_errors: Dict[str, List[float]] = {tid: [] for tid in target_ids}
    target_vel_errors: Dict[str, List[float]] = {tid: [] for tid in target_ids}
    target_detections_count: Dict[str, int] = {tid: 0 for tid in target_ids}

    total_tp = 0
    total_fn = 0
    total_fp = 0

    for i in range(mc_cfg.num_trials):
        # Deterministic per-trial seed
        trial_seed = mc_cfg.master_seed + 1000 * i + 1

        # Add noise and/or clutter if specified
        if mc_cfg.snr_db is not None or (mc_cfg.enable_clutter and mc_cfg.clutter_power is not None):
            clutter_p = mc_cfg.clutter_power if mc_cfg.enable_clutter else None
            env_res = compose_radar_environment(
                clean_signal=clean_cube.data,
                snr_db=mc_cfg.snr_db,
                clutter_power=clutter_p,
                seed=trial_seed
            )
            trial_data = env_res.composite_signal
        else:
            trial_data = clean_cube.data

        # Construct data cube payload for this trial
        trial_cube = RadarDataCube(
            data=trial_data,
            time_vector=clean_cube.time_vector,
            chirp_container=clean_cube.chirp_container,
            num_chirps=clean_cube.num_chirps,
            samples_per_chirp=clean_cube.samples_per_chirp,
            sampling_rate_hz=clean_cube.sampling_rate_hz,
            chirp_duration_sec=clean_cube.chirp_duration_sec,
            carrier_frequency_hz=clean_cube.carrier_frequency_hz,
            bandwidth_hz=clean_cube.bandwidth_hz,
            targets=ground_truth_targets
        )

        # Compute Range-Doppler map
        rd_res = compute_range_doppler_map(trial_cube)

        # Run 2D CA-CFAR detection
        cfar_res = run_ca_cfar(rd_res, cfar_cfg=cfar_cfg)

        # Match detections to theoretical targets
        match_res = match_targets_to_detections(
            targets=ground_truth_targets,
            detections=cfar_res.detections,
            range_gate_m=gate_r,
            velocity_gate_mps=gate_v
        )

        per_trial_matches.append(match_res)

        total_tp += match_res.num_true_positives
        total_fn += match_res.num_false_negatives
        total_fp += match_res.num_false_positives

        # Record matched target errors
        for m in match_res.matches:
            tid = m.target.target_id
            if tid in target_range_errors:
                target_range_errors[tid].append(m.detection.range_m - m.target.range_m)
                target_vel_errors[tid].append(m.detection.velocity_mps - m.target.velocity_mps)
                target_detections_count[tid] += 1

    # Aggregate statistics per target
    agg_r_stats: Dict[str, AggregateErrorStats] = {}
    agg_v_stats: Dict[str, AggregateErrorStats] = {}
    pd_per_target: Dict[str, float] = {}

    all_r_errors: List[float] = []
    all_v_errors: List[float] = []

    for tid in target_ids:
        r_errs = target_range_errors[tid]
        v_errs = target_vel_errors[tid]

        agg_r_stats[tid] = calculate_aggregate_metrics(r_errs)
        agg_v_stats[tid] = calculate_aggregate_metrics(v_errs)
        pd_per_target[tid] = target_detections_count[tid] / float(mc_cfg.num_trials)

        all_r_errors.extend(r_errs)
        all_v_errors.extend(v_errs)

    # Overall aggregates
    agg_r_stats["overall"] = calculate_aggregate_metrics(all_r_errors)
    agg_v_stats["overall"] = calculate_aggregate_metrics(all_v_errors)

    total_target_opportunities = len(ground_truth_targets) * mc_cfg.num_trials
    overall_pd = total_tp / float(total_target_opportunities) if total_target_opportunities > 0 else 0.0

    exec_time = time.time() - start_time

    return MonteCarloResult(
        config=mc_cfg,
        radar_config=config,
        targets_theoretical=ground_truth_targets,
        num_trials=mc_cfg.num_trials,
        per_trial_matches=per_trial_matches,
        aggregate_range_stats=agg_r_stats,
        aggregate_velocity_stats=agg_v_stats,
        detection_probability_per_target=pd_per_target,
        overall_detection_probability=overall_pd,
        total_missed_detections=total_fn,
        total_false_detections=total_fp,
        execution_time_sec=exec_time,
        metadata={
            "range_resolution_m": res_r,
            "velocity_resolution_mps": res_v,
            "range_gate_m": gate_r,
            "velocity_gate_mps": gate_v,
            "total_true_positives": total_tp,
        }
    )
