"""Radar Validation & Statistical Experiment Suite.

Executes scientific validation experiments quantifying:
1. Range estimation accuracy across unambiguous range bounds.
2. Doppler/velocity estimation accuracy across negative, zero, and positive velocities.
3. Signal-to-Noise Ratio (SNR) sensitivity sweep (Pd vs SNR, MAE vs SNR).
4. CFAR empirical false alarm rate vs configured Pfa (noise-only and statistical clutter).
5. Multi-target Monte Carlo performance metrics.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import time
import numpy as np
import scipy.constants

from src.dsp.transforms import find_dominant_frequency
from src.radar.cfar import CFARConfig, CFARResult, run_ca_cfar
from src.radar.doppler import (
    RadarDataCube,
    RangeDopplerResult,
    compute_range_doppler_map,
    extract_range_doppler_peaks,
    generate_multi_chirp_data_cube,
)
from src.radar.noise import add_awgn_noise, calculate_noise_power, calculate_signal_power, compose_radar_environment
from src.radar.processing import compute_range_fft
from src.radar.target import TargetState
from src.utils.config import CFARConfig as PydanticCFARConfig, RadarConfig, TargetConfig
from src.validation.metrics import (
    AggregateErrorStats,
    EstimationErrorMetrics,
    calculate_absolute_error,
    calculate_aggregate_metrics,
    calculate_relative_error,
    evaluate_estimation_errors,
)
from src.validation.monte_carlo import (
    MonteCarloConfig,
    MonteCarloResult,
    run_monte_carlo_simulation,
)

SPEED_OF_LIGHT_M_PER_S = float(scipy.constants.c)


@dataclass(frozen=True)
class RangeValidationPoint:
    """Single data point in range validation experiment."""
    true_range_m: float
    estimated_range_m: float
    abs_error_m: float
    rel_error: float


@dataclass(frozen=True)
class RangeValidationResult:
    """Result payload for range accuracy validation experiment."""
    points: List[RangeValidationPoint]
    aggregate_stats: AggregateErrorStats
    range_resolution_m: float
    unambiguous_range_m: float
    radar_config: RadarConfig


@dataclass(frozen=True)
class VelocityValidationPoint:
    """Single data point in velocity validation experiment."""
    true_velocity_mps: float
    estimated_velocity_mps: float
    abs_error_mps: float
    rel_error: float


@dataclass(frozen=True)
class VelocityValidationResult:
    """Result payload for velocity accuracy validation experiment."""
    points: List[VelocityValidationPoint]
    aggregate_stats: AggregateErrorStats
    velocity_resolution_mps: float
    unambiguous_velocity_mps: float
    radar_config: RadarConfig


@dataclass(frozen=True)
class SNRExperimentResult:
    """Result payload for SNR sensitivity sweep experiment."""
    snr_levels_db: List[float]
    measured_snr_levels_db: List[float]
    signal_powers: List[float]
    noise_powers: List[float]
    mean_detections_per_trial: List[float]
    detection_probabilities: List[float]
    missed_detection_rates: List[float]
    range_maes_m: List[float]
    velocity_maes_mps: List[float]
    range_rmses_m: List[float]
    velocity_rmses_mps: List[float]
    num_trials_per_snr: int
    target_config: TargetConfig


@dataclass(frozen=True)
class FalseAlarmPoint:
    """Empirical false alarm result for a single configured Pfa."""
    configured_pfa: float
    empirical_pfa_noise_only: float
    empirical_pfa_clutter: float
    total_trials: int
    total_evaluated_cells_per_trial: int
    total_false_detections_noise: int
    total_false_detections_clutter: int


@dataclass(frozen=True)
class FalseAlarmResult:
    """Result payload for CFAR false alarm experiment."""
    points: List[FalseAlarmPoint]
    cfar_config: CFARConfig
    map_shape: Tuple[int, int]


def run_range_accuracy_experiment(
    config: Optional[RadarConfig] = None,
    test_ranges_m: Optional[List[float]] = None
) -> RangeValidationResult:
    """Execute range accuracy experiment evaluating estimated range across multiple target ranges.

    Args:
        config: Baseline RadarConfig (defaults to 500m max range config).
        test_ranges_m: List of test target ranges (defaults to [50, 100, 150, 200, 300, 400] m).

    Returns:
        RangeValidationResult container.
    """
    cfg = config if config is not None else RadarConfig(
        carrier_frequency_hz=77e9,
        sweep_bandwidth_hz=150e6,
        chirp_duration_sec=100e-6,
        sampling_rate_hz=20e6,
        num_chirps=64
    )

    ranges = test_ranges_m if test_ranges_m is not None else [50.0, 100.0, 150.0, 200.0, 300.0, 400.0]
    points: List[RangeValidationPoint] = []
    signed_errors: List[float] = []

    for r_true in ranges:
        target = TargetConfig(range_m=r_true, velocity_mps=0.0, rcs_sqm=1.0, amplitude=1.0)
        cube = generate_multi_chirp_data_cube(config=cfg, targets=[target], num_chirps=1)
        slope = cube.chirp_container.chirp_slope_hz_per_sec

        # Range FFT processing on chirp row 0
        beat = cube.data[0, :]
        spec_res = compute_range_fft(
            beat,
            sampling_rate_hz=cfg.sampling_rate_hz,
            chirp_slope_hz_per_sec=slope
        )
        dom_res = find_dominant_frequency(spec_res, ignore_dc=True, interpolate_subbin=True)
        fb_meas = dom_res.frequency_hz
        r_est = SPEED_OF_LIGHT_M_PER_S * fb_meas / (2.0 * slope)

        e_r = r_est - r_true
        abs_e_r = abs(e_r)
        rel_e_r = calculate_relative_error(r_est, r_true)

        points.append(RangeValidationPoint(
            true_range_m=r_true,
            estimated_range_m=r_est,
            abs_error_m=abs_e_r,
            rel_error=rel_e_r
        ))
        signed_errors.append(e_r)

    agg_stats = calculate_aggregate_metrics(signed_errors)
    res_r = SPEED_OF_LIGHT_M_PER_S / (2.0 * cfg.sweep_bandwidth_hz)
    slope = cfg.sweep_bandwidth_hz / cfg.chirp_duration_sec
    r_max = SPEED_OF_LIGHT_M_PER_S * cfg.sampling_rate_hz / (4.0 * slope)

    return RangeValidationResult(
        points=points,
        aggregate_stats=agg_stats,
        range_resolution_m=res_r,
        unambiguous_range_m=r_max,
        radar_config=cfg
    )


def run_velocity_accuracy_experiment(
    config: Optional[RadarConfig] = None,
    test_velocities_mps: Optional[List[float]] = None
) -> VelocityValidationResult:
    """Execute velocity accuracy experiment evaluating estimated velocity across negative, zero, and positive velocities.

    Note:
        Uses Tc = 50 us so unambiguous velocity v_max ≈ 19.47 m/s.

    Args:
        config: Baseline RadarConfig.
        test_velocities_mps: List of velocities (defaults to [-15, -10, -5, 0, 5, 10, 15] m/s).

    Returns:
        VelocityValidationResult container.
    """
    cfg = config if config is not None else RadarConfig(
        carrier_frequency_hz=77e9,
        sweep_bandwidth_hz=150e6,
        chirp_duration_sec=50e-6,
        sampling_rate_hz=20e6,
        num_chirps=128
    )

    velocities = test_velocities_mps if test_velocities_mps is not None else [-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0]
    points: List[VelocityValidationPoint] = []
    signed_errors: List[float] = []

    for v_true in velocities:
        target = TargetConfig(range_m=100.0, velocity_mps=v_true, rcs_sqm=1.0, amplitude=1.0)
        cube = generate_multi_chirp_data_cube(config=cfg, targets=[target])
        rd_res = compute_range_doppler_map(cube)

        # Extract peak
        peaks = extract_range_doppler_peaks(rd_res, threshold_db=-20.0, max_peaks=1)
        if peaks:
            v_est = peaks[0].velocity_mps
        else:
            # Fallback to argmax in magnitude matrix
            d_idx, r_idx = np.unravel_index(np.argmax(rd_res.magnitude_matrix), rd_res.magnitude_matrix.shape)
            v_est = float(rd_res.velocity_axis_mps[d_idx])

        e_v = v_est - v_true
        abs_e_v = abs(e_v)
        rel_e_v = calculate_relative_error(v_est, v_true)

        points.append(VelocityValidationPoint(
            true_velocity_mps=v_true,
            estimated_velocity_mps=v_est,
            abs_error_mps=abs_e_v,
            rel_error=rel_e_v
        ))
        signed_errors.append(e_v)

    agg_stats = calculate_aggregate_metrics(signed_errors)
    wavelength = SPEED_OF_LIGHT_M_PER_S / cfg.carrier_frequency_hz
    res_v = wavelength / (2.0 * cfg.num_chirps * cfg.chirp_duration_sec)
    v_max = wavelength / (4.0 * cfg.chirp_duration_sec)

    return VelocityValidationResult(
        points=points,
        aggregate_stats=agg_stats,
        velocity_resolution_mps=res_v,
        unambiguous_velocity_mps=v_max,
        radar_config=cfg
    )


def run_snr_sweep_experiment(
    config: Optional[RadarConfig] = None,
    target: Optional[TargetConfig] = None,
    snr_levels_db: Optional[List[float]] = None,
    num_trials_per_snr: int = 50,
    master_seed: int = 42
) -> SNRExperimentResult:
    """Execute SNR sensitivity sweep measuring detection probability and MAE/RMSE across SNR levels.

    Args:
        config: Baseline RadarConfig.
        target: Ground truth TargetConfig (defaults to 100m range, 10m/s velocity).
        snr_levels_db: List of SNR values in dB (defaults to [30, 20, 10, 0, -10, -20, -30, -35, -40] dB).
        num_trials_per_snr: Trials per SNR step (default 50).
        master_seed: Master seed for reproducibility.

    Returns:
        SNRExperimentResult payload.
    """
    cfg = config if config is not None else RadarConfig(
        carrier_frequency_hz=77e9,
        sweep_bandwidth_hz=150e6,
        chirp_duration_sec=50e-6,
        sampling_rate_hz=20e6,
        num_chirps=64
    )

    tgt = target if target is not None else TargetConfig(
        range_m=100.0, velocity_mps=10.0, rcs_sqm=1.0, amplitude=1.0, target_id="T1"
    )

    snr_list = snr_levels_db if snr_levels_db is not None else [30.0, 20.0, 10.0, 0.0, -10.0, -20.0, -30.0, -35.0, -40.0]

    meas_snr_list: List[float] = []
    sig_power_list: List[float] = []
    noise_power_list: List[float] = []
    mean_dets_list: List[float] = []
    p_d_list: List[float] = []
    p_miss_list: List[float] = []
    r_mae_list: List[float] = []
    v_mae_list: List[float] = []
    r_rmse_list: List[float] = []
    v_rmse_list: List[float] = []

    # Synthesize clean cube once to get base signal power
    clean_cube = generate_multi_chirp_data_cube(config=cfg, targets=[tgt])
    p_sig = calculate_signal_power(clean_cube.data)

    for step_idx, snr_val in enumerate(snr_list):
        step_master_seed = master_seed + step_idx * 10000
        mc_cfg = MonteCarloConfig(
            num_trials=num_trials_per_snr,
            master_seed=step_master_seed,
            snr_db=snr_val,
            enable_clutter=False
        )

        mc_res = run_monte_carlo_simulation(config=cfg, targets=[tgt], mc_config=mc_cfg)

        # Measure noise power and resulting SNR from representative sample
        env_sample = compose_radar_environment(clean_signal=clean_cube.data, snr_db=snr_val, seed=step_master_seed)
        p_noise = env_sample.noise_power
        meas_snr = env_sample.snr_db if env_sample.snr_db is not None else snr_val

        total_dets = sum(len(match.unmatched_detections) + match.num_true_positives for match in mc_res.per_trial_matches)
        mean_dets = total_dets / float(num_trials_per_snr)

        pd_val = mc_res.overall_detection_probability
        pmiss_val = 1.0 - pd_val

        meas_snr_list.append(meas_snr)
        sig_power_list.append(p_sig)
        noise_power_list.append(p_noise)
        mean_dets_list.append(mean_dets)

        p_d_list.append(pd_val)
        p_miss_list.append(pmiss_val)

        r_stats = mc_res.aggregate_range_stats.get("T1", mc_res.aggregate_range_stats["overall"])
        v_stats = mc_res.aggregate_velocity_stats.get("T1", mc_res.aggregate_velocity_stats["overall"])

        r_mae_list.append(r_stats.mean_absolute_error)
        v_mae_list.append(v_stats.mean_absolute_error)
        r_rmse_list.append(r_stats.rmse)
        v_rmse_list.append(v_stats.rmse)

    return SNRExperimentResult(
        snr_levels_db=snr_list,
        measured_snr_levels_db=meas_snr_list,
        signal_powers=sig_power_list,
        noise_powers=noise_power_list,
        mean_detections_per_trial=mean_dets_list,
        detection_probabilities=p_d_list,
        missed_detection_rates=p_miss_list,
        range_maes_m=r_mae_list,
        velocity_maes_mps=v_mae_list,
        range_rmses_m=r_rmse_list,
        velocity_rmses_mps=v_rmse_list,
        num_trials_per_snr=num_trials_per_snr,
        target_config=tgt
    )


def run_false_alarm_experiment(
    config: Optional[RadarConfig] = None,
    cfar_config: Optional[PydanticCFARConfig] = None,
    pfa_levels: Optional[List[float]] = None,
    num_trials: int = 50,
    master_seed: int = 42
) -> FalseAlarmResult:
    """Execute CFAR false-alarm experiment evaluating empirical Pfa in NO-TARGET environment.

    Evaluates:
        A. Noise-only baseline
        B. Noise + statistical clutter baseline

    Args:
        config: Baseline RadarConfig.
        cfar_config: Base Pydantic CFARConfig.
        pfa_levels: List of target Pfa values (defaults to [1e-2, 1e-3, 1e-4]).
        num_trials: Trials per Pfa level (default 50).
        master_seed: Master seed for reproducibility.

    Returns:
        FalseAlarmResult container.
    """
    cfg = config if config is not None else RadarConfig(
        carrier_frequency_hz=77e9,
        sweep_bandwidth_hz=150e6,
        chirp_duration_sec=100e-6,
        sampling_rate_hz=20e6,
        num_chirps=64
    )

    pfa_list = pfa_levels if pfa_levels is not None else [1e-2, 1e-3, 1e-4]
    
    # Synthesize zero-target dummy cube structure to get dimension metadata
    dummy_target = TargetConfig(range_m=100.0, velocity_mps=0.0, amplitude=1e-12)
    dummy_cube = generate_multi_chirp_data_cube(config=cfg, targets=[dummy_target])
    shape = dummy_cube.data.shape  # (M, N)

    points: List[FalseAlarmPoint] = []

    for pfa_val in pfa_list:
        cfar_cfg = PydanticCFARConfig(
            pfa=pfa_val,
            num_guard_range=2,
            num_guard_doppler=2,
            num_train_range=4,
            num_train_doppler=4
        )

        total_false_noise = 0
        total_false_clutter = 0
        total_valid_cells_per_trial = 0

        for trial_idx in range(num_trials):
            seed_noise = master_seed + 1000 * trial_idx + 1
            seed_clutter = master_seed + 1000 * trial_idx + 2

            # 1. Noise-only scenario
            zero_signal = np.zeros(shape, dtype=np.complex128)
            n_res = add_awgn_noise(signal=zero_signal, target_noise_power=1.0, seed=seed_noise)
            noise_matrix = n_res.noisy_signal

            cube_noise = RadarDataCube(
                data=noise_matrix,
                time_vector=dummy_cube.time_vector,
                chirp_container=dummy_cube.chirp_container,
                num_chirps=dummy_cube.num_chirps,
                samples_per_chirp=dummy_cube.samples_per_chirp,
                sampling_rate_hz=dummy_cube.sampling_rate_hz,
                chirp_duration_sec=dummy_cube.chirp_duration_sec,
                carrier_frequency_hz=dummy_cube.carrier_frequency_hz,
                bandwidth_hz=dummy_cube.bandwidth_hz,
                targets=[]
            )
            rd_noise = compute_range_doppler_map(cube_noise)
            cfar_noise = run_ca_cfar(rd_noise, cfar_cfg=cfar_cfg)

            total_false_noise += cfar_noise.num_detections

            # Count valid interior cells evaluated
            border_d = cfar_noise.metadata["border_doppler"]
            border_r = cfar_noise.metadata["border_range"]
            valid_rows = rd_noise.num_doppler_bins - 2 * border_d
            valid_cols = rd_noise.num_range_bins - 2 * border_r
            n_valid_cells = max(0, valid_rows * valid_cols)
            total_valid_cells_per_trial = n_valid_cells

            # 2. Noise + Statistical Clutter scenario
            env_res = compose_radar_environment(
                clean_signal=zero_signal,
                clutter_power=1.0,
                seed=seed_clutter
            )
            cube_clutter = RadarDataCube(
                data=env_res.composite_signal,
                time_vector=dummy_cube.time_vector,
                chirp_container=dummy_cube.chirp_container,
                num_chirps=dummy_cube.num_chirps,
                samples_per_chirp=dummy_cube.samples_per_chirp,
                sampling_rate_hz=dummy_cube.sampling_rate_hz,
                chirp_duration_sec=dummy_cube.chirp_duration_sec,
                carrier_frequency_hz=dummy_cube.carrier_frequency_hz,
                bandwidth_hz=dummy_cube.bandwidth_hz,
                targets=[]
            )
            rd_clutter = compute_range_doppler_map(cube_clutter)
            cfar_clutter = run_ca_cfar(rd_clutter, cfar_cfg=cfar_cfg)

            total_false_clutter += cfar_clutter.num_detections

        total_evaluated_cells_all_trials = total_valid_cells_per_trial * num_trials
        emp_pfa_noise = total_false_noise / float(total_evaluated_cells_all_trials) if total_evaluated_cells_all_trials > 0 else 0.0
        emp_pfa_clutter = total_false_clutter / float(total_evaluated_cells_all_trials) if total_evaluated_cells_all_trials > 0 else 0.0

        points.append(
            FalseAlarmPoint(
                configured_pfa=pfa_val,
                empirical_pfa_noise_only=emp_pfa_noise,
                empirical_pfa_clutter=emp_pfa_clutter,
                total_trials=num_trials,
                total_evaluated_cells_per_trial=total_valid_cells_per_trial,
                total_false_detections_noise=total_false_noise,
                total_false_detections_clutter=total_false_clutter
            )
        )

    base_cfar_cfg = PydanticCFARConfig()
    return FalseAlarmResult(
        points=points,
        cfar_config=base_cfar_cfg,
        map_shape=shape
    )


def run_multi_target_monte_carlo_experiment(
    config: Optional[RadarConfig] = None,
    num_trials: int = 100,
    snr_db: float = 20.0,
    master_seed: int = 42
) -> MonteCarloResult:
    """Execute multi-target Monte Carlo benchmark experiment (Phase 7/8 3-target scenario).

    Targets:
        T1: R = 50 m, v = +10 m/s, amp = 1.0
        T2: R = 120 m, v = -5 m/s, amp = 0.7
        T3: R = 200 m, v = 0 m/s, amp = 0.5

    Args:
        config: Baseline RadarConfig.
        num_trials: Number of trials (default 100).
        snr_db: Requested SNR in dB (default 20 dB).
        master_seed: Master seed for reproducibility.

    Returns:
        MonteCarloResult payload.
    """
    cfg = config if config is not None else RadarConfig(
        carrier_frequency_hz=77e9,
        sweep_bandwidth_hz=150e6,
        chirp_duration_sec=50e-6,
        sampling_rate_hz=20e6,
        num_chirps=64
    )

    targets = [
        TargetConfig(range_m=50.0, velocity_mps=10.0, rcs_sqm=1.0, amplitude=1.0, target_id="T1"),
        TargetConfig(range_m=120.0, velocity_mps=-5.0, rcs_sqm=0.7, amplitude=0.7, target_id="T2"),
        TargetConfig(range_m=200.0, velocity_mps=0.0, rcs_sqm=0.5, amplitude=0.5, target_id="T3"),
    ]

    mc_cfg = MonteCarloConfig(
        num_trials=num_trials,
        master_seed=master_seed,
        snr_db=snr_db,
        enable_clutter=False
    )

    return run_monte_carlo_simulation(config=cfg, targets=targets, mc_config=mc_cfg)
