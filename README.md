# RadarDSP Lab — Interactive DSP & FMCW Radar Signal Processing Simulator

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/Streamlit-1.30%2B-red.svg)](https://streamlit.io/)

A modular academic and engineering simulator for **Digital Signal Processing (DSP)** and **Frequency-Modulated Continuous-Wave (FMCW) Radar Processing**. Designed for algorithm validation, visual education, and quantitative radar performance analysis.

---

## 🔬 Project Overview

`RadarDSP Lab` contains two primary laboratory environments:

### 1. DSP Laboratory
- **Signal Generation:** Multi-tone synthesis, chirps, square waves, and noise injection.
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
│   ├── dsp/              # Sampling, FFT, Filtering, STFT
│   ├── radar/            # Chirp generation, Propagation, 2D Range-Doppler, CFAR
│   ├── validation/       # Cramér-Rao Lower Bounds, Error metrics, Monte-Carlo sweeps
│   └── utils/            # Pydantic configuration schemas & logging
├── tests/                # PyTest suite for unit and integration testing
├── matlab/               # Independent MATLAB reference implementations
├── configs/              # Preserved YAML parameter files
├── experiments/          # Saved experiment runs and benchmark visual outputs
└── docs/                 # Architectural specifications (architecture.md)
```

---

## 🚀 Quick Start & Installation

### 1. Clone & Setup Environment

```bash
git clone https://github.com/your-username/RadarDSP-Lab.git
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

### 3. Launch Interactive Application

```bash
streamlit run app/main.py
```

---

## 🧪 Testing & Validation Strategy

The project employs a three-tier validation strategy:
1. **PyTest Suite:** Validates mathematical invariants (Parseval's theorem, zero-phase response, Pydantic bounds).
2. **Theoretical Closed-Form Verification:** Benchmarks measured range/doppler estimates against analytical CRLB limits.
3. **MATLAB Golden Reference:** Cross-validates Python outputs against independent MATLAB Phased Array reference scripts.

---

## 📜 Development Roadmap

- [x] **Phase 0:** Core Clean Architecture setup, Pydantic schemas, package stubs, test suite initialization.
- [ ] **Phase 1:** Core DSP Laboratory implementation (signals, sampling, FFT, windowing, FIR/IIR, STFT).
- [ ] **Phase 2:** FMCW Radar Laboratory implementation (chirps, target channel, 2D Range-Doppler, CA-CFAR).
- [ ] **Phase 3:** Validation suite, Monte-Carlo simulations, and MATLAB cross-verification.
- [ ] **Phase 4:** Interactive Streamlit dashboard completion and quantitative reporting exports.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for details.
