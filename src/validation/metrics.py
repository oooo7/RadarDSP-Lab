"""Validation Metrics & Error Evaluation Module.

Computes theoretical limits (Cramér-Rao Lower Bound, range/velocity resolutions)
and quantitative error metrics (RMSE, Absolute Error, Peak-to-Sidelobe Ratio).
Actual logic will be implemented in Phase 3.
"""

from dataclasses import dataclass
from src.utils.config import RadarConfig, TargetConfig


@dataclass
class TheoreticalLimits:
    """Theoretical resolution limits and Cramér-Rao Lower Bounds (CRLB)."""
    range_resolution_m: float
    max_unambiguous_range_m: float
    velocity_resolution_mps: float
    max_unambiguous_velocity_mps: float
    range_crlb_m: float
    velocity_crlb_mps: float


@dataclass
class EstimationErrorMetrics:
    """Payload comparing ground truth values against estimated radar/DSP targets."""
    true_range_m: float
    estimated_range_m: float
    range_error_m: float
    true_velocity_mps: float
    estimated_velocity_mps: float
    velocity_error_mps: float


def compute_theoretical_limits(config: RadarConfig, snr_db: float) -> TheoreticalLimits:
    """Compute theoretical range/velocity resolutions and CRLB bounds.

    Args:
        config: Radar configuration options.
        snr_db: Signal-to-noise ratio in dB.

    Returns:
        TheoreticalLimits container.

    Raises:
        NotImplementedError: Implementation scheduled for Phase 3.
    """
    raise NotImplementedError("Theoretical radar limit equations will be implemented in Phase 3.")


def evaluate_estimation_errors(
    ground_truth: TargetConfig,
    estimated_range_m: float,
    estimated_velocity_mps: float
) -> EstimationErrorMetrics:
    """Calculate absolute error metrics between ground truth target and radar estimates.

    Args:
        ground_truth: True target configuration.
        estimated_range_m: Measured target range.
        estimated_velocity_mps: Measured target velocity.

    Returns:
        EstimationErrorMetrics payload.

    Raises:
        NotImplementedError: Implementation scheduled for Phase 3.
    """
    raise NotImplementedError("Error metric calculation will be implemented in Phase 3.")
