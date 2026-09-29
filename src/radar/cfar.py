"""Constant False Alarm Rate (CFAR) Detection Module.

Implements 1D and 2D CA-CFAR (Cell-Averaging CFAR), OS-CFAR, GO-CFAR, and SO-CFAR detectors.
Actual detection algorithms will be implemented in Phase 2.
"""

from dataclasses import dataclass
import numpy as np
from src.utils.config import CFARConfig


@dataclass
class CFARDetectionResult:
    """Payload holding CFAR adaptive threshold map, binary detection mask, and detected target indices/coordinates."""
    threshold_map: np.ndarray
    detection_mask: np.ndarray
    detected_ranges_m: np.ndarray
    detected_velocities_mps: np.ndarray | None = None


def run_ca_cfar_1d(
    signal_powers_db: np.ndarray,
    range_axis_m: np.ndarray,
    cfar_config: CFARConfig
) -> CFARDetectionResult:
    """Execute 1D Cell-Averaging Constant False Alarm Rate (CA-CFAR) detection.

    Args:
        signal_powers_db: 1D power spectrum or Range FFT magnitude array in dB.
        range_axis_m: Range axis in meters.
        cfar_config: CFAR parameters (guard cells, reference cells, Pfa).

    Returns:
        CFARDetectionResult containing threshold curve and detected target indices.

    Raises:
        NotImplementedError: Implementation scheduled for Phase 2.
    """
    raise NotImplementedError("1D CA-CFAR algorithm will be implemented in Phase 2.")


def run_ca_cfar_2d(
    rd_map_db: np.ndarray,
    range_axis_m: np.ndarray,
    velocity_axis_mps: np.ndarray,
    cfar_config: CFARConfig
) -> CFARDetectionResult:
    """Execute 2D Cell-Averaging CFAR detection on Range-Doppler map.

    Args:
        rd_map_db: 2D matrix of Range-Doppler power magnitudes in dB.
        range_axis_m: Range axis in meters.
        velocity_axis_mps: Velocity axis in m/s.
        cfar_config: CFAR parameters.

    Returns:
        CFARDetectionResult containing 2D threshold matrix and detected (range, velocity) targets.

    Raises:
        NotImplementedError: Implementation scheduled for Phase 2.
    """
    raise NotImplementedError("2D CA-CFAR algorithm will be implemented in Phase 2.")
