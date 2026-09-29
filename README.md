# RadarDSP Lab — Interactive DSP & FMCW Radar Signal Processing Simulator

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/Streamlit-1.30%2B-red.svg)](https://streamlit.io/)

A modular academic and engineering simulator for **Digital Signal Processing (DSP)** and **Frequency-Modulated Continuous-Wave (FMCW) Radar Processing**. Designed for algorithm validation, visual education, and quantitative radar performance analysis.

---

## 🔬 Project Overview

`RadarDSP Lab` contains two primary laboratory environments:

### 1. DSP Laboratory
- **Signal Generation (Phase 1 Completed):** Multi-tone synthesis, sine, cosine, square, triangle, sawtooth, linear chirps, seeded Gaussian noise, AM, and FM signals.
- **Sampling & Nyquist Theory (Phase 2 Completed):** Discrete uniform sampling, Nyquist-rate analysis ($F_{\text{Nyquist}} = 2 f_{\max}$), theoretical alias frequency foldover calculation ($f_{\text{alias}} \in [0, F_s/2]$), parabolic FFT peak estimation, and quantitative error metrics (MAE/RMSE).
- **Spectral Analysis & STFT (Phase 3 Completed):** FFT/DFT computation, physical magnitude scaling, Power Spectrum ($V^2$) vs PSD ($V^2/\text{Hz}$), window functions (Rectangular, Hann, Hamming, Blackman, Kaiser), Peak Sidelobe Level (PSL), zero-padding resolution invariants, sub-bin parabolic frequency estimation, and STFT spectrogram engine.
- **Digital Filtering (Phase 4 Completed):** FIR (lowpass, highpass, bandpass, bandstop) and IIR Butterworth filters (SOS representation), frequency response analysis, -3 dB cutoff measurement, group delay, linear phase symmetry, causal vs. zero-phase filtering.
- **Multirate DSP & Resampling (Phase 5 Completed):** Decimation (anti-aliasing filtering + downsampling), interpolation (zero-insertion + polyphase anti-imaging filtering), rational resampling ($F_{s,\text{out}}/F_{s,\text{in}} = L/M$), polyphase rate conversion, naive vs. proper comparison, and analytical aliasing/spectral image analysis.

### 2. FMCW Radar Laboratory
- **Waveform Synthesis (Phase 6 Completed):** Configurable carrier frequency ($f_c$), bandwidth ($B$), chirp duration ($T_c$), and sampling rate ($f_s$). Scientifically validated analog stretch dechirp processing architecture ($F_s > 2 f_{b,\max}$).
- **Multi-Target Kinematics & Doppler Engine (Phase 7 Completed):** Multi-target synthesis ($R, v, A, \phi_0$), fast-time vs slow-time data cube generation ($M \times N$), 2D Range-Doppler matrix processing (2D FFT, slow-time windowing, `fftshift`), velocity axis mapping ($v = f_D \cdot \lambda / 2$), range/velocity resolution bounds, and candidate peak detection.
- **Detection Engine (Phase 8 Completed):** Complex AWGN noise module, Rayleigh magnitude statistical background clutter model, signal environment composition, and 2D Cell-Averaging CFAR (CA-CFAR) adaptive thresholding ($\alpha = N(P_{\text{fa}}^{-1/N}-1)$, boundary masking, detection extraction).
- **Validation Suite:** Theoretical resolution limits, Cramér-Rao Lower Bounds (CRLB), and Monte-Carlo performance evaluation.

---

## 🏗️ Repository Architecture

```
RadarDSP-Lab/
├── README.md
├── requirements.txt
├── pyproject.toml
├── app/                  # Streamlit application entry point & view modules
├── src/                  # Core signal processing & radar domain packages
│   ├── dsp/              # Sampling, FFT, Filtering, Resampling, Signals Engine
│   │   ├── signals.py    # Production-quality signal generation engine (Phase 1)
│   │   ├── sampling.py   # Sampling, Nyquist rate analysis, aliasing engine (Phase 2)
│   │   ├── transforms.py # FFT, PSD, Windowing, Parabolic Sub-bin & STFT Engine (Phase 3)
│   │   ├── filtering.py  # FIR & IIR Butterworth Digital Filtering Engine (Phase 4)
│   │   └── resampling.py # Multirate DSP, Decimation, Interpolation & Rational Resampling (Phase 5)
│   ├── radar/            # Chirp generation, Target model, Dechirp, 2D Range-Doppler, CFAR Engine
│   │   ├── chirp.py      # FMCW chirp generation & sampling architecture validator (Phase 6)
│   │   ├── target.py     # Stationary & moving radar target kinematics (Phase 6 & 7)
│   │   ├── propagation.py# Round-trip delay synthesis & dechirp mixing (Phase 6 & 7)
│   │   ├── processing.py # 1D Range FFT engine & beat sampling Nyquist validator (Phase 6)
│   │   ├── doppler.py    # Multi-chirp data cube, 2D Range-Doppler FFT engine (Phase 7)
│   │   ├── noise.py      # Complex AWGN noise generator, SNR calculation & environment composition (Phase 8)
│   │   ├── clutter.py    # Complex Gaussian statistical background clutter model (Phase 8)
│   │   └── cfar.py       # 2D Cell-Averaging CFAR adaptive threshold detector (Phase 8)
│   ├── validation/       # Cramér-Rao Lower Bounds, Error metrics, Monte-Carlo sweeps
│   └── utils/            # Pydantic configuration schemas & logging
├── tests/                # PyTest suite for unit and integration testing
│   ├── test_signals.py   # DSP signal engine test suite (Phase 1)
│   ├── test_sampling.py  # Sampling & aliasing test suite (Phase 2)
│   ├── test_transforms.py# FFT, Spectral Analysis, Windowing test suite (Phase 3)
│   ├── test_filtering.py # FIR & IIR Digital Filtering test suite (Phase 4)
│   ├── test_resampling.py# Multirate DSP & Resampling test suite (Phase 5)
│   ├── test_radar_chirp.py# FMCW Chirp & sampling architecture test suite (Phase 6)
│   ├── test_radar_processing.py# 1D Range FFT & beat sampling test suite (Phase 6)
│   ├── test_radar_doppler.py   # Multi-target 2D Range-Doppler test suite (Phase 7)
│   ├── test_radar_noise.py     # AWGN noise & SNR engine test suite (Phase 8)
│   ├── test_radar_clutter.py   # Statistical clutter background test suite (Phase 8)
│   └── test_radar_cfar.py      # 2D CA-CFAR detection engine test suite (Phase 8)
├── matlab/               # Independent MATLAB reference implementations
├── configs/              # Preserved YAML parameter files
├── experiments/          # Saved experiment runs and benchmark visual outputs
└── docs/                 # Architectural & mathematical documentation
    ├── architecture.md   # Architectural Specifications
    ├── dsp_theory.md     # Mathematical Signal Models, Multirate DSP & Filtering Theory
    └── radar_theory.md   # FMCW Radar Principles, Stretch Processing, Range-Doppler & CFAR Theory
```

---

## 🚀 Quick Start & Installation

### 1. Clone & Setup Environment

```bash
git clone https://github.com/oooo7/RadarDSP-Lab.git
cd RadarDSP-Lab

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies in editable mode
pip install -e .
```

### 2. Run Tests

```bash
pytest
```

---

## 📜 Development Roadmap

- [x] **Phase 0:** Core Clean Architecture setup, Pydantic schemas, package stubs, test suite initialization.
- [x] **Phase 1:** Core DSP Signal Generation Engine (`src/dsp/signals.py`, 10 signal types, determinism, test suite).
- [x] **Phase 2:** Sampling & Nyquist-rate analysis, undersampling, aliasing calculation (`src/dsp/sampling.py`, test suite).
- [x] **Phase 3:** Spectral analysis (FFT/DFT, physical scaling, windowing, PSL, STFT spectrograms, test suite).
- [x] **Phase 4:** Digital filtering engine (`src/dsp/filtering.py`, FIR & IIR Butterworth SOS, causal/zero-phase, test suite).
- [x] **Phase 5:** Multirate DSP Engine (`src/dsp/resampling.py`, decimation, interpolation, rational resampling, test suite).
- [x] **Phase 6:** FMCW Radar Range-Processing Engine (`src/radar/chirp.py`, `processing.py`, analog stretch dechirp model, beat sampling Nyquist validator, test suite).
- [x] **Phase 7:** Multi-Target FMCW Radar + Doppler / Velocity Engine (`src/radar/doppler.py`, 2D Range-Doppler map, velocity axis, peak extraction, test suite).
- [x] **Phase 8:** Noise, Clutter & CA-CFAR Detection Engine (`src/radar/noise.py`, `clutter.py`, `cfar.py`, 2D CA-CFAR detector, test suite).

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for details.
