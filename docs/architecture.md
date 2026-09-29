# RadarDSP Lab — Software Architecture Specification

## 1. System Architecture Overview

**RadarDSP Lab** is an interactive, high-precision Digital Signal Processing (DSP) and Frequency-Modulated Continuous-Wave (FMCW) Radar simulator designed for academic research, algorithm prototyping, and education.

The codebase adheres strictly to **Clean Architecture** principles, enforcing separation of concerns into distinct layers:

```
+-----------------------------------------------------------------------+
|                           Streamlit Application                       |
|        (app/main.py, app/views/dsp_lab.py, app/views/radar_lab.py)    |
+-----------------------------------------------------------------------+
                                   |
                                   v (Calls domain functions & APIs)
+-----------------------------------------------------------------------+
|                          Domain Core Engine                           |
|        (src/dsp/, src/radar/, src/validation/, src/utils/)            |
+-----------------------------------------------------------------------+
                                   |
                                   v (Cross-validated against)
+-----------------------------------------------------------------------+
|                    MATLAB Ground-Truth Reference                      |
|                  (matlab/ reference script suite)                     |
+-----------------------------------------------------------------------+
```

### Key Architectural Rules
1. **Zero UI Coupling:** The signal processing core (`src/`) has zero dependencies on Streamlit, Matplotlib UI widgets, or browser components. It operates purely on NumPy arrays, SciPy algorithms, and Pydantic configuration models.
2. **Immutable Data Payloads:** Inter-module data flow uses typed Python `@dataclass` containers (`SignalContainer`, `SpectrumContainer`, `FMCWChirpContainer`, `RangeDopplerMap`). This eliminates dynamic dictionary lookups and prevents accidental state mutations across pipeline stages.
3. **Strict Validation & Typing:** Pydantic models enforce physical boundaries (e.g., non-negative sampling frequencies, carrier frequencies > 0, probability of false alarm within $(0, 1)$).

---

## 2. Module Responsibilities

The codebase is organized under `src/` into specialized sub-packages:

```
RadarDSP-Lab/
├── app/                  # Presentation Layer: Streamlit UI components & routing
│   ├── main.py           # Entry point and sidebar navigation
│   ├── views/            # Dedicated laboratory views (DSP & FMCW Radar)
│   └── components/       # Reusable UI widgets, plotters, and input controls
├── src/                  # Domain Engine Layer
│   ├── dsp/              # Core DSP algorithms (sampling, filtering, STFT, windows)
│   │   ├── signals.py    # Signal generation engine (sine, cosine, chirp, square, multi-tone)
│   │   ├── sampling.py   # Nyquist sampling, undersampling, aliasing prediction
│   │   ├── transforms.py # FFT/DFT spectral analysis, windowing, STFT spectrograms
│   │   └── filtering.py  # FIR/IIR design, frequency response, decimation/interpolation
│   ├── radar/            # FMCW Radar signal processing pipeline
│   │   ├── chirp.py      # LFM waveform synthesis (carrier, bandwidth, chirp duration)
│   │   ├── target.py     # Kinematics engine (range delay, Doppler frequency shift, RCS)
│   │   ├── propagation.py# Tx/Rx channel simulation, AWGN, clutter, and de-chirping/mixing
│   │   ├── processing.py # 1D Range FFT, 2D Range-Doppler processing across multi-chirp frames
│   │   └── cfar.py       # CA-CFAR (1D/2D), OS-CFAR, GO-CFAR adaptive thresholding
│   ├── validation/       # Theoretical verification & quantitative error engine
│   │   ├── metrics.py    # Theoretical limits (resolution, CRLB bounds, RMSE, bias)
│   │   └── monte_carlo.py# Automated multi-trial stochastic simulations vs SNR
│   └── utils/            # Shared utilities
│       ├── config.py     # Pydantic configuration schema & serialization
│       └── logging.py    # Structured logging infrastructure
├── tests/                # Test suite (PyTest)
├── matlab/               # Independent MATLAB reference implementation scripts
├── configs/              # Preserved default YAML parameter files
├── experiments/          # Stored benchmark runs, plot outputs, and exported datasets
└── docs/                 # Architectural specifications and documentation
```

---

## 3. Signal Processing Pipelines

### 3.1 DSP Laboratory Pipeline

```
[SignalGenConfig] ---> (signals.generate_signal) ---> [SignalContainer (Time domain)]
                                                             |
                 +-------------------------------------------+-------------------------------------------+
                 |                                           |                                           |
                 v                                           v                                           v
    (sampling.sample_signal)                  (transforms.compute_fft)                    (filtering.design_filter)
                 |                                           |                                           |
                 v                                           v                                           v
     [SamplingAnalysisResult]                   [SpectrumContainer]                          [FilterResponse]
   (Nyquist/Aliasing Metrics)               (Magnitude/Phase Spectra)                (Frequency & Phase Response)
                                                             |                                           |
                                                             v                                           v
                                                  (transforms.compute_stft)                  (filtering.apply_filter)
                                                             |                                           |
                                                             v                                           v
                                                  [SpectrogramContainer]                     [Filtered Signal]
```

### 3.2 FMCW Radar Laboratory Pipeline

```
[RadarConfig]
     |
     v
(chirp.generate_fmcw_chirp) ----> [FMCWChirpContainer (Tx LFM Chirp)]
     |                                          |
     |                                          v
(target.compute_target_state) ---> (propagation.simulate_radar_channel)
                                                |
                                                v
                                  [RadarRxPayload (Rx Echo + Beat Signal)]
                                                |
                      +-------------------------+-------------------------+
                      |                                                   |
                      v                                                   v
          (processing.compute_range_fft)                    (processing.compute_range_doppler_map)
                      |                                                   |
                      v                                                   v
              [RangeFFTResult]                                     [RangeDopplerMap]
                      |                                                   |
                      v                                                   v
            (cfar.run_ca_cfar_1d)                               (cfar.run_ca_cfar_2d)
                      |                                                   |
                      +-------------------------+-------------------------+
                                                |
                                                v
                                   [CFARDetectionResult]
                                                |
                                                v
                              (validation.metrics / monte_carlo)
                                                |
                                                v
                            [Estimation Error & CRLB Bounds Analysis]
```

---

## 4. Theoretical Validation & Testing Strategy

### 4.1 Theoretical Benchmarks
Every empirical measurement is benchmarked against exact physical equations:
- **Range Resolution:** $\Delta R = \frac{c}{2B}$
- **Maximum Unambiguous Range:** $R_{\max} = \frac{f_s c T_c}{2 B}$
- **Velocity Resolution:** $\Delta v = \frac{\lambda}{2 N T_c}$
- **Maximum Unambiguous Velocity:** $v_{\max} = \frac{\lambda}{4 T_c}$
- **Cramér-Rao Lower Bound (CRLB):** Theoretical variance limit for unbiased range and velocity estimators under Gaussian noise.

### 4.2 MATLAB Interoperability & Golden Reference
- MATLAB scripts in `matlab/` serve as an independent reference implementation.
- Python algorithms will be cross-validated against MATLAB numerical outputs by exporting target data vectors to CSV/HDF5 and running automated regression checks.

### 4.3 Automated PyTest Strategy
- **Unit Tests:** Fast execution testing mathematical invariants (e.g., Parseval's theorem in FFTs, zero-phase delay in symmetric FIR filters).
- **Configuration Boundary Tests:** Validating input constraints via Pydantic schema validation.
- **Pipeline Integration Tests:** End-to-end execution checks from Tx chirp generation down to CFAR target detection.

---

## 5. Streamlit Architecture & User Experience

- **Decoupled Controller Logic:** The Streamlit app renders parameter sliders and delegates logic to `src/`.
- **Session State Caching:** Expensive calculations (such as 2D Range-Doppler maps or 100-trial Monte-Carlo sweeps) are cached using `@st.cache_data` to ensure fluid UI interaction at 60 FPS.
- **Interactive Multi-Panel Visualizations:** Visualizations are isolated into modular functions under `app/components/` rendering responsive Matplotlib and Plotly interactive figures.

---

## 6. Architectural Rationale & Design Trade-offs

1. **Why Pydantic over plain dataclasses for configs?**
   Pydantic provides automatic type casting, strict validation error tracebacks, and native YAML/JSON serialization out of the box, preventing invalid physical inputs (e.g., negative speed of light, invalid Pfa).
2. **Why separate `src/dsp` and `src/radar` packages?**
   `src/dsp` focuses on foundational signal processing techniques applicable to any domain (audio, communications, biomedical), whereas `src/radar` specializes in FMCW radar kinematics, de-chirping, 2D Range-Doppler processing, and CFAR detection.
3. **Why raise `NotImplementedError` in Phase 0 instead of dummy return values?**
   Returning mock data creates a false impression of completed logic, hides interface bugs, and risks silent degradation. Explicitly throwing `NotImplementedError` guarantees that missing functionality is surfaced instantly during testing.
