"""Phase 1 Integration Smoke Test.

Demonstrates generation of:
1. 1 kHz sine wave sampled at 10 kHz
2. Multi-tone signal
3. Linear chirp
4. Seeded Gaussian noise
"""

import numpy as np
from src.dsp.signals import (
    generate_chirp,
    generate_multitone,
    generate_noise,
    generate_sine,
)


def run_smoke_test() -> None:
    """Run Phase 1 demonstration checks."""
    print("=== RadarDSP Lab — Phase 1 Smoke Test ===")

    # 1. Generate 1 kHz sine wave sampled at 10 kHz for 0.1s
    sine_sig = generate_sine(
        amplitude=1.0,
        frequency_hz=1000.0,
        phase_rad=0.0,
        sampling_rate_hz=10000.0,
        duration_sec=0.1
    )
    print(f"[1] Sine Signal: type={sine_sig.signal_type}, samples={len(sine_sig.amplitude)}, max={np.max(sine_sig.amplitude):.4f}")

    # 2. Multi-tone signal (100 Hz, 250 Hz, 400 Hz)
    multi_sig = generate_multitone(
        frequencies_hz=[100.0, 250.0, 400.0],
        amplitudes=[1.0, 0.5, 0.25],
        phases_rad=[0.0, 0.0, 0.0],
        sampling_rate_hz=5000.0,
        duration_sec=0.5
    )
    print(f"[2] Multi-tone Signal: type={multi_sig.signal_type}, samples={len(multi_sig.amplitude)}, tones={len(multi_sig.metadata['frequencies_hz'])}")

    # 3. Linear Chirp (10 Hz -> 500 Hz over 1.0s sampled at 2000 Hz)
    chirp_sig = generate_chirp(
        amplitude=1.0,
        f_start_hz=10.0,
        f_end_hz=500.0,
        duration_sec=1.0,
        sampling_rate_hz=2000.0
    )
    print(f"[3] Linear Chirp: type={chirp_sig.signal_type}, samples={len(chirp_sig.amplitude)}, slope={chirp_sig.metadata['chirp_slope_hz_per_sec']:.1f} Hz/s")

    # 4. Seeded Gaussian noise (std=0.5, seed=42)
    noise_sig = generate_noise(
        std_dev=0.5,
        sampling_rate_hz=1000.0,
        duration_sec=1.0,
        seed=42
    )
    print(f"[4] Seeded Noise: type={noise_sig.signal_type}, samples={len(noise_sig.amplitude)}, std={np.std(noise_sig.amplitude):.4f}")

    print("=== Phase 1 Smoke Test Completed Successfully ===")


if __name__ == "__main__":
    run_smoke_test()
