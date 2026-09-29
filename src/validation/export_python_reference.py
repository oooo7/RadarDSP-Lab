"""Export Python Reference Results for MATLAB Cross-Validation.

Synthesizes exact Python reference datasets for:
1. DSP aliasing calculation (f = 700 Hz, Fs = 1000 Hz -> 300 Hz alias)
2. DSP FFT spectrum calculation (100 Hz tone, Fs = 1000 Hz)
3. FMCW single-target range estimation (50 m, 100 m, 150 m, 200 m, 300 m, 400 m, 500 m)
4. FMCW multi-target Range-Doppler benchmark (T1, T2, T3)
5. Complex AWGN noise power measurement (20, 10, 0, -10 dB)
6. 2D CA-CFAR multiplier alpha and detection metrics

Exports results to experiments/results/python_reference_results.json.
"""

from dataclasses import asdict
import json
import os
import numpy as np
import scipy.constants

from src.dsp.sampling import calculate_alias_frequency
from src.dsp.signals import generate_sine
from src.dsp.transforms import compute_fft, find_dominant_frequency
from src.radar.cfar import CFARConfig, calculate_ca_cfar_alpha, run_ca_cfar
from src.radar.doppler import compute_range_doppler_map, extract_range_doppler_peaks, generate_multi_chirp_data_cube
from src.radar.noise import add_awgn_noise, calculate_noise_power, calculate_signal_power
from src.radar.processing import compute_range_fft
from src.radar.target import TargetConfig
from src.utils.config import RadarConfig, TargetConfig as PydanticTargetConfig
from src.validation.metrics import match_targets_to_detections

SPEED_OF_LIGHT_M_PER_S = float(scipy.constants.c)


def export_python_reference() -> dict:
    """Compute and return exact Python reference metrics dictionary."""
    results = {}

    # ------------------------------------------------------------------
    # 1. DSP ALIASING
    # ------------------------------------------------------------------
    f_in = 700.0
    fs_in = 1000.0
    alias_val = calculate_alias_frequency(f_in, fs_in)
    results["dsp_aliasing"] = {
        "input_frequency_hz": f_in,
        "sampling_rate_hz": fs_in,
        "expected_alias_frequency_hz": alias_val,
        "is_aliased": f_in > (fs_in / 2.0),
        "nyquist_frequency_hz": fs_in / 2.0,
    }

    # ------------------------------------------------------------------
    # 2. DSP FFT SPECTRUM
    # ------------------------------------------------------------------
    sig_res = generate_sine(amplitude=1.0, frequency_hz=100.0, phase_rad=0.0, sampling_rate_hz=1000.0, duration_sec=1.0)
    spec_res = compute_fft(sig_res, window_type="rect", one_sided=True)
    dom_res = find_dominant_frequency(spec_res, ignore_dc=True, interpolate_subbin=True)
    results["dsp_fft"] = {
        "frequency_hz": 100.0,
        "sampling_rate_hz": 1000.0,
        "n_samples": len(sig_res.amplitude),
        "dominant_frequency_hz": float(dom_res.frequency_hz),
        "peak_magnitude": float(dom_res.magnitude),
        "frequency_error_hz": float(abs(dom_res.frequency_hz - 100.0)),
    }

    # ------------------------------------------------------------------
    # 3. FMCW SINGLE-TARGET RANGES (50m to 500m)
    # ------------------------------------------------------------------
    cfg_fmcw = RadarConfig(
        carrier_frequency_hz=77e9,
        sweep_bandwidth_hz=150e6,
        chirp_duration_sec=100e-6,
        sampling_rate_hz=20e6,
        num_chirps=1
    )
    test_ranges = [50.0, 100.0, 150.0, 200.0, 300.0, 400.0, 500.0]
    single_target_results = []
    slope = cfg_fmcw.sweep_bandwidth_hz / cfg_fmcw.chirp_duration_sec

    for r_true in test_ranges:
        tgt = PydanticTargetConfig(range_m=r_true, velocity_mps=0.0, amplitude=1.0)
        cube = generate_multi_chirp_data_cube(config=cfg_fmcw, targets=[tgt], num_chirps=1)
        beat = cube.data[0, :]
        r_spec = compute_range_fft(beat, sampling_rate_hz=cfg_fmcw.sampling_rate_hz, chirp_slope_hz_per_sec=slope)
        r_dom = find_dominant_frequency(r_spec, ignore_dc=True, interpolate_subbin=True)
        r_est = SPEED_OF_LIGHT_M_PER_S * float(r_dom.frequency_hz) / (2.0 * slope)
        fb_true = 2.0 * slope * r_true / SPEED_OF_LIGHT_M_PER_S

        single_target_results.append({
            "true_range_m": r_true,
            "estimated_range_m": float(r_est),
            "theoretical_beat_frequency_hz": float(fb_true),
            "measured_beat_frequency_hz": float(r_dom.frequency_hz),
            "abs_error_m": float(abs(r_est - r_true)),
        })

    results["fmcw_single_target"] = {
        "carrier_frequency_hz": cfg_fmcw.carrier_frequency_hz,
        "sweep_bandwidth_hz": cfg_fmcw.sweep_bandwidth_hz,
        "chirp_duration_sec": cfg_fmcw.chirp_duration_sec,
        "sampling_rate_hz": cfg_fmcw.sampling_rate_hz,
        "chirp_slope_hz_per_sec": slope,
        "targets": single_target_results,
    }

    # ------------------------------------------------------------------
    # 4. MULTI-TARGET DOPPLER (T1, T2, T3)
    # ------------------------------------------------------------------
    cfg_multi = RadarConfig(
        carrier_frequency_hz=77e9,
        sweep_bandwidth_hz=150e6,
        chirp_duration_sec=50e-6,
        sampling_rate_hz=20e6,
        num_chirps=64
    )
    targets_multi = [
        PydanticTargetConfig(range_m=50.0, velocity_mps=10.0, amplitude=1.0, target_id="T1"),
        PydanticTargetConfig(range_m=120.0, velocity_mps=-5.0, amplitude=0.7, target_id="T2"),
        PydanticTargetConfig(range_m=200.0, velocity_mps=0.0, amplitude=0.5, target_id="T3"),
    ]
    cube_multi = generate_multi_chirp_data_cube(config=cfg_multi, targets=targets_multi)
    rd_res = compute_range_doppler_map(cube_multi)
    peaks = extract_range_doppler_peaks(rd_res, threshold_db=-25.0, max_peaks=10)

    multi_target_peaks = []
    for p in peaks:
        multi_target_peaks.append({
            "target_id": p.target_id,
            "range_m": float(p.range_m),
            "velocity_mps": float(p.velocity_mps),
            "magnitude_db": float(p.magnitude_db),
            "range_bin": int(p.range_bin),
            "doppler_bin": int(p.doppler_bin),
        })

    results["fmcw_multi_target"] = {
        "range_resolution_m": float(rd_res.range_resolution_m),
        "velocity_resolution_mps": float(rd_res.velocity_resolution_mps),
        "unambiguous_range_m": float(rd_res.unambiguous_range_m),
        "unambiguous_velocity_mps": float(rd_res.unambiguous_velocity_mps),
        "targets_theoretical": [
            {"id": "T1", "range_m": 50.0, "velocity_mps": 10.0, "amplitude": 1.0},
            {"id": "T2", "range_m": 120.0, "velocity_mps": -5.0, "amplitude": 0.7},
            {"id": "T3", "range_m": 200.0, "velocity_mps": 0.0, "amplitude": 0.5},
        ],
        "extracted_peaks": multi_target_peaks,
    }

    # ------------------------------------------------------------------
    # 5. AWGN NOISE POWER & SNR
    # ------------------------------------------------------------------
    sig_dummy = np.ones((64, 1000), dtype=np.complex128)
    p_dummy_sig = calculate_signal_power(sig_dummy)
    awgn_results = []
    for snr_req in [20.0, 10.0, 0.0, -10.0]:
        n_res = add_awgn_noise(signal=sig_dummy, snr_db=snr_req, seed=42)
        awgn_results.append({
            "requested_snr_db": snr_req,
            "measured_snr_db": float(n_res.snr_db),
            "signal_power": float(n_res.signal_power),
            "measured_noise_power": float(n_res.noise_power),
        })

    results["awgn_noise"] = {
        "signal_power": float(p_dummy_sig),
        "trials": awgn_results,
    }

    # ------------------------------------------------------------------
    # 6. 2D CA-CFAR DETECTOR & ALPHA
    # ------------------------------------------------------------------
    pfa_test = 1e-4
    cfar_cfg_pydantic = CFARConfig(
        pfa=pfa_test,
        num_guard_range=2,
        num_guard_doppler=2,
        num_train_range=4,
        num_train_doppler=4
    )
    # Calculate actual training cells: outer (2*4+2*2+1)^2 = 13^2 = 169, inner (2*2+1)^2 = 25 -> 144 cells
    alpha_calc = calculate_ca_cfar_alpha(pfa=pfa_test, num_training_cells=144)
    cfar_res = run_ca_cfar(rd_res, cfar_cfg=cfar_cfg_pydantic)
    match_res = match_targets_to_detections(
        targets=cube_multi.targets,
        detections=cfar_res.detections,
        range_gate_m=1.5 * rd_res.range_resolution_m,
        velocity_gate_mps=1.5 * rd_res.velocity_resolution_mps
    )

    results["ca_cfar"] = {
        "pfa": pfa_test,
        "num_training_cells": 144,
        "alpha_multiplier": float(alpha_calc),
        "num_detections": int(cfar_res.num_detections),
        "true_positives": int(match_res.num_true_positives),
        "false_negatives": int(match_res.num_false_negatives),
        "false_positives": int(match_res.num_false_positives),
        "border_range": int(cfar_res.metadata["border_range"]),
        "border_doppler": int(cfar_res.metadata["border_doppler"]),
    }

    return results


def main():
    os.makedirs("experiments/results", exist_ok=True)
    res = export_python_reference()
    out_path = os.path.join("experiments", "results", "python_reference_results.json")
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Successfully exported Python reference results to {out_path}")


if __name__ == "__main__":
    main()
