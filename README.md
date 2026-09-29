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
- **Sampling & Nyquist Theory:** Aliasing, undersampling, and signal reconstruction.
- **Spectral Analysis:** FFT/DFT computation, windowing functions, and STFT spectrograms.
- **Digital Filtering:** FIR & IIR filter design (butterworth, firwin), phase response, decimation, and interpolation.

### 2. FMCW Radar Laboratory
- **Waveform Synthesis:** Configurable carrier frequency ($f_c$), bandwidth ($B$), chirp duration ($T_c$), and sampling rate ($f_s$).
- **Channel Dynamics:** Propagation delay, multi-target echoes, radial velocity, AWGN, and clutter.
- **Signal Processing:** Beat frequency de-chirping, 1D Range FFT, and 2D Range-Doppler maps.
- **Detection Engine:** 1D and 2D CA-CFAR adaptive thresholding.
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
│   ├── dsp/              # Sampling, FFT, Filtering, STFT, Signals Engine
│   │   └── signals.py    # Production-quality signal generation engine (Phase 1)
│   ├── radar/            # Chirp generation, Propagation, 2D Range-Doppler, CFAR
│   ├── validation/       # Cramér-Rao Lower Bounds, Error metrics, Monte-Carlo sweeps
│   └── utils/            # Pydantic configuration schemas & logging
├── tests/                # PyTest suite for unit and integration testing
│   └── test_signals.py   # Comprehensive DSP signal engine test suite (Phase 1)
├── matlab/               # Independent MATLAB reference implementations
├── configs/              # Preserved YAML parameter files
├── experiments/          # Saved experiment runs and benchmark visual outputs
└── docs/                 # Architectural & mathematical documentation
    ├── architecture.md   # Architectural Specifications
    └── dsp_theory.md     # Phase 1 Mathematical Signal Models & Theories
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
- [ ] **Phase 2:** Sampling & Nyquist-rate analysis, undersampling, aliasing prediction.
- [ ] **Phase 3:** Spectral analysis (FFT/DFT, windowing, spectral leakage, STFT spectrograms).
- [ ] **Phase 4:** Digital filtering (FIR/IIR design, decimation, interpolation, anti-aliasing).
- [ ] **Phase 5:** FMCW Radar Laboratory (chirps, target channel, 2D Range-Doppler, CA-CFAR).
- [ ] **Phase 6:** Interactive Streamlit dashboard completion and quantitative reporting exports.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for details.
