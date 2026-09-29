# RadarDSP Lab — FMCW Radar Mathematical Signal Processing Theory

## 1. Frequency-Modulated Continuous-Wave (FMCW) Radar Principles

Frequency-Modulated Continuous-Wave (FMCW) radar transmits a continuous signal whose frequency varies linearly over time. By mixing the received echo reflected from a target with the transmitted chirp signal, FMCW radar generates a low-frequency **beat signal** whose frequency is directly proportional to the target's radial distance (range).

---

## 2. Linear Frequency Modulated (LFM) Chirp Model

A transmitted LFM FMCW chirp is defined by:
- **Carrier Frequency ($f_c$):** Operating center frequency in Hz (e.g. $77\text{ GHz}$).
- **Sweep Bandwidth ($B$):** Total frequency excursion in Hz (e.g. $150\text{ MHz}$).
- **Chirp Duration ($T_c$):** Sweep duration in seconds (e.g. $100\,\mu\text{s}$).
- **ADC Sampling Rate ($F_s$):** Receiver baseband digitizer rate in Hz (e.g. $10\text{ MHz}$).

### Chirp Slope ($S$)
The linear chirp sweep rate $S$ is defined as:

$$S = \frac{B}{T_c} \quad (\text{Hz/s})$$

### Transmitted Signal Formulations

1. **Complex Baseband Representation (Standard Simulation Model):**

   $$s_{\text{tx}}(t) = A \cdot \exp\left( j \left( \pi S t^2 + \phi_0 \right) \right), \quad 0 \le t < T_c$$

2. **Real Passband Representation:**

   $$s_{\text{tx,RF}}(t) = A \cdot \cos\left( 2\pi \left( f_c t + \frac{1}{2} S t^2 \right) + \phi_0 \right), \quad 0 \le t < T_c$$

### Instantaneous Baseband Frequency
The instantaneous baseband frequency sweeps linearly over time:

$$f_{\text{inst}}(t) = S \cdot t \quad (\text{Hz}), \quad 0 \le t < T_c$$

---

## 3. Propagation Delay & Target Echo Synthesis

For a stationary monostatic radar target at range $R$ (meters):

### Round-Trip Propagation Delay ($\tau$)
The total two-way electromagnetic propagation time is:

$$\tau = \frac{2 R}{c}$$

where $c = 299,792,458\text{ m/s}$ is the speed of light in vacuum.

### Received Target Echo Signal ($s_{\text{rx}}(t)$)
The received echo signal is an attenuated, delayed replica of the transmitted chirp:

$$s_{\text{rx}}(t) = \alpha \cdot s_{\text{tx}}(t - \tau) = \alpha A \cdot \exp\left( j \left( \pi S (t - \tau)^2 + \phi_0 + \phi_{\text{target}} \right) \right)$$

where $\alpha$ represents reflection amplitude attenuation and $\phi_{\text{target}}$ is target phase shift.

---

## 4. De-Chirping (Mixing) & Beat Signal Generation

De-chirping multiplies the transmitted chirp $s_{\text{tx}}(t)$ by the complex conjugate of the received echo $s_{\text{rx}}^*(t)$:

$$s_{\text{beat}}(t) = s_{\text{tx}}(t) \cdot s_{\text{rx}}^*(t)$$

Substituting the complex baseband chirp equations:

$$s_{\text{beat}}(t) = A e^{j \pi S t^2} \cdot \alpha A e^{-j \left( \pi S (t - \tau)^2 + \phi_{\text{target}} \right)}$$

$$s_{\text{beat}}(t) = \alpha A^2 \cdot \exp\left( j \left( 2\pi (S \tau) t - \pi S \tau^2 - \phi_{\text{target}} \right) \right)$$

### Beat Frequency Formulation ($f_b$)
The de-chirped beat signal $s_{\text{beat}}(t)$ is a pure complex sinusoid at a single constant frequency $f_b$:

$$f_b = S \cdot \tau = \frac{2 S R}{c} = \frac{2 B R}{c T_c} \quad (\text{Hz})$$

---

## 5. FMCW Range Estimation & Range Axis Mapping

### The Fundamental Range Equation
Rearranging the beat frequency formula yields target range $R$:

$$R = \frac{c \cdot f_b}{2 S} = \frac{c T_c \cdot f_b}{2 B}$$

### Range-Axis Mapping
When a 1D FFT is computed on $N_{\text{fft}}$ samples of the beat signal, the discrete frequency bin axis $f[k] = k \cdot \Delta f_{\text{bin}}$ maps directly to physical range:

$$R[k] = \frac{c \cdot f[k]}{2 S} = \frac{c \cdot k \cdot F_s}{2 S N_{\text{fft}}} \quad (\text{meters})$$

---

## 6. Physical Range Resolution vs. FFT Bin Spacing

### 6.1 Theoretical Radar Range Resolution ($\Delta R$)
Range resolution $\Delta R$ is the minimum physical separation required between two targets for them to be resolved as distinct spectral peaks. It is **strictly determined by transmitted sweep bandwidth $B$**:

$$\Delta R = \frac{c}{2 B}$$

*Example:* For $B = 150\text{ MHz}$, $\Delta R = \frac{299,792,458}{2 \times 150 \times 10^6} \approx 0.9993\text{ m} \approx 1.0\text{ m}$.

### 6.2 FFT Frequency & Range Bin Spacing ($\Delta R_{\text{bin}}$)
The FFT bin grid resolution is determined by sampling rate $F_s$ and total FFT points $N_{\text{fft}}$:

$$\Delta f_{\text{bin}} = \frac{F_s}{N_{\text{fft}}} \implies \Delta R_{\text{bin}} = \frac{c \cdot \Delta f_{\text{bin}}}{2 S} = \frac{c F_s}{2 S N_{\text{fft}}}$$

### 6.3 Critical Distinction: Range Resolution vs. Bin Spacing
- **Physical Range Resolution ($\Delta R$):** Set by $B$. Cannot be improved by post-processing or zero-padding.
- **FFT Range Bin Spacing ($\Delta R_{\text{bin}}$):** Set by $N_{\text{fft}}$. Zero-padding ($N_{\text{fft}} > N_{\text{signal}}$) decreases $\Delta R_{\text{bin}}$, interpolating the spectral envelope and enabling millimeter-level peak location, but **does not resolve two targets closer than $\Delta R$**.

---

## 7. Windowing & Zero-Padding Trade-offs

- **Windowing (Hann / Hamming / Blackman):** Suppresses spectral leakage sidelobes at the expense of slight mainlobe broadening.
- **Parabolic Sub-bin Interpolation:** Estimates peak location between FFT bin centers to achieve sub-millimeter range estimation accuracy.

---

## 8. Maximum Unambiguous Range ($R_{\max}$)

The maximum range $R_{\max}$ that can be processed without aliasing the beat frequency beyond the ADC Nyquist bandwidth:

$$f_{b,\max} = \frac{F_s}{2} \implies R_{\max} = \frac{c \cdot F_s}{4 S} = \frac{c F_s T_c}{4 B}$$

---

## 9. Complex Baseband vs. Carrier RF Simulation Rationale

Simulating RF carrier oscillations directly at $77\text{ GHz}$ would require a sampling rate $F_s > 154\text{ GHz}$, generating over $1.5 \times 10^7$ samples per microsecond. Complex baseband envelope modeling captures all phase, frequency, delay, and amplitude dynamics at standard ADC baseband rates ($10\text{ MHz}$), maintaining machine-precision accuracy while reducing compute overhead by $> 10,000 \times$.

---

## 10. Sampling Architecture: Chirp vs. Dechirped Beat Signal

An important scientific sampling question arises when evaluating FMCW radar parameters: *How can a baseband ADC sampling rate of 20 MHz faithfully represent a 150 MHz sweep bandwidth chirp?*

To answer this, `RadarDSP Lab` explicitly distinguishes between two receiver architectures:

### Option A — Directly Sampled Chirp Receiver Architecture
In a receiver that digitizes the raw transmit/receive chirp waveform directly before mixing, the ADC digitizer samples a signal with sweep bandwidth $B = 150\text{ MHz}$. By the Nyquist-Shannon sampling theorem:

$$F_{s,\text{ADC}} \ge B \quad (\text{Complex Baseband}) \quad \text{or} \quad F_{s,\text{ADC}} \ge 2 B \quad (\text{Real Passband})$$

Digitizing a $150\text{ MHz}$ chirp directly before mixing requires $F_s \ge 150\text{ MHz}$.

### Option B — Analog Stretch Processing / Dechirp Mixer Architecture (Default Simulator Model)
In standard FMCW radar hardware (e.g. 77 GHz automotive radars), de-chirping occurs in the **analog domain** before digitizer ADC sampling. An analog RF mixer multiplies the analog transmit chirp with the analog target echo:

$$s_{\text{beat}}(t) = s_{\text{tx}}(t) \cdot s_{\text{rx}}^*(t) = \alpha A^2 \exp\left( j \left( 2\pi S \tau t - \pi S \tau^2 - \phi_{\text{target}} \right) \right)$$

The resulting analog beat signal $s_{\text{beat}}(t)$ is a low-frequency sinusoid whose maximum frequency $f_{b,\max}$ depends on the maximum target processing range $R_{\max}$:

$$f_{b,\max} = \frac{2 S R_{\max}}{c}$$

For $R_{\max} = 500\text{ m}$ and $S = 1.5 \times 10^{12}\text{ Hz/s}$, $f_{b,\max} \approx 5.0035\text{ MHz}$.

### Why the Default ADC Rate is 20 MHz
The simulator processes the dechirped beat signal using a conventional one-sided Range FFT, mapping positive frequencies to the spectrum $[0, F_s / 2]$. For the maximum beat frequency $f_{b,\max} \approx 5.0035\text{ MHz}$ at $R_{\max} = 500\text{ m}$, the minimum required sampling rate to keep the beat frequency inside $[0, F_s / 2]$ is:

$$F_{s,\text{min}} > 2 \cdot f_{b,\max} \approx 10.0069\text{ MHz}$$

An ADC rate of $F_s = 10\text{ MHz}$ places the maximum beat frequency slightly above the $F_s / 2 = 5\text{ MHz}$ positive Nyquist limit. By selecting $F_s = 20\text{ MHz}$:

- **Nyquist Limit ($F_s / 2$):** $10.0\text{ MHz}$
- **Maximum Beat Frequency ($f_{b,\max}$):** $5.0035\text{ MHz}$
- **Sampling Margin Ratio ($F_s / (2 f_{b,\max})$):** $1.9986$ ($99.86\%$ headroom)

This guarantees that all dechirped beat signals up to $R_{\max} = 500\text{ m}$ lie strictly within the unambiguous positive-frequency Nyquist zone.

---

## 11. Multi-Target FMCW Model & Fast-Time vs. Slow-Time

In a multi-target environment, the radar receives echoes from $K$ simultaneous targets. The simulator uses a two-dimensional grid:

- **Fast-time ($t$):** Time samples $n \in [0, N-1]$ within a single chirp ($t[n] = n / F_s$). Governs **range estimation**.
- **Slow-time ($t_{\text{slow}}$):** Chirp repetition index $m \in [0, M-1]$ across $M$ consecutive chirps ($t_{\text{slow}}[m] = m \cdot T_c$). Governs **Doppler velocity estimation**.

### Multi-Target Dechirped Beat Signal
The dechirped beat signal across chirps $m$ and fast-time $t$ is the coherent sum of all target echoes:

$$s_{\text{beat}}[m, n] = \sum_{k=1}^K A_k \cdot \exp\left(j \left( 2\pi f_{b,k} t[n] + 2\pi f_{D,k} (m T_c) + \phi_{\text{const},k} \right)\right)$$

where $f_{b,k} = \frac{2 S R_{0,k}}{c}$ is the beat frequency and $f_{D,k} = \frac{2 v_k}{\lambda}$ is the Doppler frequency for target $k$.

---

## 12. Doppler Effect & Velocity Sign Convention

### Radial Velocity Convention
- **Positive Velocity ($v > 0$):** Target is moving **away** from the radar (receding).
- **Negative Velocity ($v < 0$):** Target is **approaching** the radar (closing).

### Monostatic Doppler Frequency ($f_D$)
The Doppler frequency shift generated by radial velocity $v$ at carrier wavelength $\lambda = c / f_c$ is:

$$f_D = \frac{2 v}{\lambda} = \frac{2 v f_c}{c}$$

Across repeated chirps $m$, target movement alters the phase by $\Delta \phi_D = 2\pi f_D m T_c$.

---

## 13. 2D Range-Doppler Matrix Processing

Range-Doppler processing extracts 2D target maps using a sequential two-dimensional FFT:

1. **Fast-time Range FFT (Axis 1):** Computed across ADC samples $n$ for each chirp $m$, producing complex range profiles of shape $[M, N_{\text{range}}]$.
2. **Slow-time Doppler FFT (Axis 0):** Computed across chirps $m$ for each range bin $r$, windowed in slow-time and centered using `fftshift`.

The output is a 2D Range-Doppler matrix $S[v, R]$ of shape $[N_{\text{doppler}}, N_{\text{range}}]$.

### 13.1 Idealized Coherent 2D FFT Processing Gain

For a data cube of $M$ chirps and $N$ fast-time samples, the **idealized coherent 2D FFT processing gain** is:

$$\text{Processing Gain}_{\text{ideal}} = 10 \log_{10}(M \cdot N) \quad (\text{dB})$$

For example, a data cube of $M = 64$ chirps and $N = 2000$ samples yields an idealized processing gain of $10 \log_{10}(128,000) \approx +51.07\text{ dB}$.

#### Explicit Processing Gain Assumptions & Limitations:
1. **Rectangular Windowing:** Assumes unwindowed fast-time and slow-time FFT processing.
2. **Perfect Bin Alignment:** Assumes target range and velocity coincide exactly with FFT bin centers (zero straddle / scalloping loss).
3. **Uncorrelated AWGN Noise:** Assumes complex AWGN noise is independent and identically distributed across all fast-time and slow-time samples.
4. **Coherent Integration:** Assumes perfect phase coherence across all $M \times N$ samples.

*Windowing Loss Note:* When windowing (e.g. Hann tapering) is applied in fast-time and slow-time, coherent signal gain is scaled ($0.5 \times 0.5 = 0.25$ in 2D, introducing a $6\text{ dB}$ signal power reduction), while noise variance is scaled by the noise power gain factor ($1.5$ per dimension). This introduces a net Equivalent Noise Bandwidth (ENBW) SNR processing loss of $\approx 1.76\text{ dB}$ per dimension ($3.52\text{ dB}$ total 2D windowing loss).

---

## 14. Velocity Resolution & Unambiguous Velocity Bounds

### 14.1 Velocity Resolution ($\Delta v$)
Velocity resolution $\Delta v$ is the minimum velocity separation required to distinguish two targets in the same range cell. It depends on the total coherent observation duration $M \cdot T_c$:

$$\Delta v = \frac{\lambda}{2 M T_c} = \frac{\lambda \cdot \text{PRF}}{2 M}$$

### 14.2 Maximum Unambiguous Velocity ($v_{\max}$)
The maximum radial velocity that can be measured without Doppler phase aliasing across chirps spaced by $T_c$:

$$f_{D,\max} = \frac{\text{PRF}}{2} = \frac{1}{2 T_c} \implies v_{\max} = \frac{\lambda}{4 T_c} = \frac{c}{4 f_c T_c}$$

---

## 15. Noise, Statistical Background Clutter & CA-CFAR Detection Engine

To transition the radar simulation from an ideal noiseless model to a realistic detection environment, `RadarDSP Lab` introduces complex additive noise, background clutter, and adaptive Constant False Alarm Rate (CFAR) detection.

### 15.1 Additive White Gaussian Noise (AWGN) & SNR
The complex baseband noise $w[n] = w_I[n] + j w_Q[n]$ consists of independent real and imaginary Gaussian components:

$$w_I \sim \mathcal{N}\left(0, \frac{P_{\text{noise}}}{2}\right), \quad w_Q \sim \mathcal{N}\left(0, \frac{P_{\text{noise}}}{2}\right)$$

Total average complex noise power is $P_{\text{noise}} = \mathbb{E}[|w|^2] = \mathbb{E}[w_I^2] + \mathbb{E}[w_Q^2]$.

The Signal-to-Noise Ratio (SNR) in decibels is defined as:

$$\text{SNR}_{\text{dB}} = 10 \log_{10}\left( \frac{P_{\text{signal}}}{P_{\text{noise}}} \right) = 10 \log_{10}\left( \frac{\frac{1}{N} \sum_{n=1}^N |x[n]|^2}{\frac{1}{N} \sum_{n=1}^N |w[n]|^2} \right)$$

### 15.2 Statistical Background Clutter Model
Radar clutter represents unwanted environmental reflections (e.g. ground, sea, weather). In Phase 8, background clutter is modeled as a complex Gaussian random process $c[n] = c_I[n] + j c_Q[n]$ with configurable power $P_{\text{clutter}}$. The magnitude envelope $|c[n]|$ follows a Rayleigh distribution:

$$p(|c|) = \frac{2 |c|}{P_{\text{clutter}}} \exp\left(-\frac{|c|^2}{P_{\text{clutter}}}\right)$$

*Scope Note:* This model provides an algorithmic statistical background baseline for CFAR validation. It does not model physical electromagnetic terrain/sea scattering.

### 15.3 2D Cell-Averaging Constant False Alarm Rate (CA-CFAR) Engine

A fixed power threshold cannot maintain a constant false alarm rate when background noise/clutter fluctuates spatially. CA-CFAR dynamically estimates local background power and computes an adaptive threshold.

#### Window Geometry & Definitions
For a Cell Under Test (CUT) at index $(d, r)$ in the Range-Doppler map:
- **Cell Under Test (CUT):** The candidate cell being evaluated for target presence.
- **Guard Cells ($G_D \times G_R$):** Immediate neighboring region surrounding the CUT. Guard cells prevent target mainlobe energy spillover from contaminating the local noise estimate.
- **Training Cells ($T_D \times T_R$):** Outer region surrounding the guard window used to estimate background noise power.

$$\text{Outer Window Size} = (2 T_D + 2 G_D + 1) \times (2 T_R + 2 G_R + 1)$$
$$\text{Inner Window Size} = (2 G_D + 1) \times (2 G_R + 1)$$
$$\text{Total Training Cells } N_{\text{train}} = \text{Outer Size} - \text{Inner Size}$$

#### Power-Domain Processing & Threshold Factor ($\alpha$)
CA-CFAR operates in the **power domain** ($P[d, r] = |S[d, r]|^2$). For $N_{\text{train}}$ independent Rayleigh-fading training cells and target false alarm probability $P_{\text{fa}}$, the analytical threshold factor $\alpha$ is:

$$\alpha = N_{\text{train}} \cdot \left( P_{\text{fa}}^{-1 / N_{\text{train}}} - 1 \right)$$

#### Local Noise Power Estimation & Detection Criterion
The local background noise power estimate $\hat{P}_{\text{noise}}(d, r)$ is the arithmetic mean of power across all $N_{\text{train}}$ training cells:

$$\hat{P}_{\text{noise}}(d, r) = \frac{1}{N_{\text{train}}} \sum_{\text{training cells}} P[i, j]$$

The adaptive power threshold is:

$$P_{\text{threshold}}(d, r) = \alpha \cdot \hat{P}_{\text{noise}}(d, r)$$

A target detection is declared if:

$$P_{\text{CUT}}(d, r) > P_{\text{threshold}}(d, r)$$

#### Boundary Handling
CUT cells near the matrix edges where the full training/guard window extends past array boundaries are marked as **INVALID / NON-DETECTION** (`detection_mask = False`, `threshold_db = NaN`). Edge wrap-around (`np.roll`) is strictly avoided to prevent Doppler/range contamination across opposite boundaries.

---

## 16. Scientific Validation, Error Analysis & Monte Carlo Evaluation Engine

Phase 9 establishes a scientific validation and experimentation layer designed to quantify estimation accuracy, detection probability ($P_d$), empirical false alarm rate ($P_{\text{fa}}$), SNR sensitivity, and Monte Carlo variability across RadarDSP Lab.

### 16.1 Quantitative Error Metrics

For a target with ground-truth range $R_{\text{true}}$ and velocity $v_{\text{true}}$, and estimated values $R_{\text{est}}$ and $v_{\text{est}}$:

#### Signed Absolute Error
$$e_R = R_{\text{est}} - R_{\text{true}} \quad (\text{m}), \qquad e_v = v_{\text{est}} - v_{\text{true}} \quad (\text{m/s})$$

#### Magnitude Error
$$|e_R| = |R_{\text{est}} - R_{\text{true}}| \quad (\text{m}), \qquad |e_v| = |v_{\text{est}} - v_{\text{true}}| \quad (\text{m/s})$$

#### Relative Percentage Error
$$\text{Error}_R(\%) = 100 \times \frac{|R_{\text{est}} - R_{\text{true}}|}{R_{\text{true}}}$$

$$\text{Error}_v(\%) = 
\begin{cases} 
100 \times \frac{|v_{\text{est}} - v_{\text{true}}|}{|v_{\text{true}}|}, & |v_{\text{true}}| \ge 10^{-3}\text{ m/s} \\
0.0, & |v_{\text{true}}| < 10^{-3}\text{ m/s} \quad (\text{Stationary Targets})
\end{cases}$$

#### Aggregate Statistical Summaries across $K$ Trials
- **Mean Error:** $\bar{e} = \frac{1}{K} \sum_{i=1}^K e_i$
- **Mean Absolute Error (MAE):** $\text{MAE} = \frac{1}{K} \sum_{i=1}^K |e_i|$
- **Root Mean Square Error (RMSE):** $\text{RMSE} = \sqrt{\frac{1}{K} \sum_{i=1}^K e_i^2}$
- **Standard Deviation ($\sigma$):** $\sigma = \sqrt{\frac{1}{K} \sum_{i=1}^K (e_i - \bar{e})^2}$
- **Maximum Absolute Error:** $\text{Max}|e| = \max_i |e_i|$

### 16.2 Theoretical Reference Limits & Cramér-Rao Bounds (CRLB)

The formulas:

$$\text{CRLB}_R = \frac{c}{2 B \sqrt{2 \cdot \text{SNR}_{\text{linear}}}} \quad (\text{m})$$

$$\text{CRLB}_v = \frac{\lambda}{2 \pi T_{\text{frame}} \sqrt{2 \cdot \text{SNR}_{\text{linear}}}} \quad (\text{m/s})$$

where $T_{\text{frame}} = M \cdot T_c$ is the coherent processing interval (CPI), represent **idealized continuous-time theoretical reference limits** rather than exact bounds for discrete windowed estimators.

#### Explicit Assumptions for Theoretical Reference Bounds:
1. **Single Isolated Target:** Single target in un-cluttered complex AWGN noise.
2. **Asymptotic High SNR:** High SNR assumption ($\text{SNR}_{\text{linear}} \gg 1$).
3. **Continuous Signal Model:** Continuous-time unwindowed (rectangular window) signal representation.
4. **No Discrete Binning / Windowing Loss:** Continuous parameter estimation without discrete FFT bin grid quantization ($\Delta R / \sqrt{12}$) or windowing taper ENBW losses.

In discrete sampled systems with windowed FFT processing, discrete bin grid quantization and windowing losses modify the exact estimator variance.

### 16.3 Physical Resolution vs Estimator Accuracy

It is critical to distinguish **physical resolution** from **estimator accuracy**:
- **Physical Resolution ($\Delta R, \Delta v$):** The minimum separation required to resolve two closely spaced targets as distinct spectral peaks ($\Delta R = c / (2B), \Delta v = \lambda / (2 M T_c)$).
- **Estimator Accuracy (MAE/RMSE):** The precision with which a single isolated target peak can be estimated using sub-bin interpolation or peak finding. High SNR can yield sub-resolution estimation accuracy ($|e_R| \ll \Delta R$), but this does NOT imply the physical resolution of the radar has changed.

### 16.4 Resolution-Gated 1-to-1 Target Matching

For multi-target detection, raw CFAR detection masks cannot be directly compared to ordered target lists. RadarDSP Lab implements a greedy 1-to-1 target-to-detection matching algorithm:
1. Candidate detections are gated by physical resolution bounds:
   $$|R_{\text{det}} - R_{\text{true}}| \le 1.5 \cdot \Delta R \quad \text{AND} \quad |v_{\text{det}} - v_{\text{true}}| \le 1.5 \cdot \Delta v$$
2. Candidate pairs are assigned by minimizing normalized distance:
   $$d = \sqrt{ \left( \frac{R_{\text{det}} - R_{\text{true}}}{1.5 \Delta R} \right)^2 + \left( \frac{v_{\text{det}} - v_{\text{true}}}{1.5 \Delta v} \right)^2 }$$
3. Each detection matches at most one target (True Positive), unmatched targets are counted as Missed Detections (False Negatives), and remaining detections are counted as Spurious Alarms (False Positives).

### 16.5 Reproducible Monte Carlo Methodology

To guarantee exact numerical reproducibility across multi-trial simulations, trial $i$ ($0 \le i < K$) derives its random seed deterministically from master seed $S_0$:

$$\text{Seed}_i = S_0 + 1000 \cdot i + 1$$

This ensures that independent trials receive distinct, non-overlapping pseudo-random streams while producing identical numerical results across execution environments.

---

## 17. Independent MATLAB Golden Reference Implementation & Cross-Validation Methodology

Phase 10 introduces an independent MATLAB reference implementation for core DSP and FMCW calculations to perform numerical cross-validation against the Python implementation.

### 17.1 Purpose & Scientific Scope
The objective is NOT to duplicate the full Python framework, but to independently formulate selected core mathematical models in MATLAB and verify agreement within documented physical tolerances.

The cross-validation pipeline operates as:
$$\text{Python Implementation} \longrightarrow \text{Independent Math Model} \longrightarrow \text{MATLAB Reference} \longrightarrow \text{Numerical Cross-Validation} \longrightarrow \text{Documented Tolerances}$$

### 17.2 Mathematical Equivalence vs Bit-for-Bit Reproducibility
- **Mathematical Equivalence:** Formulating identical underlying physical models (e.g. stretch-dechirp beat frequency $f_b = 2 S R / c$, Doppler shift $f_D = 2 v / \lambda$, CA-CFAR multiplier $\alpha = N(P_{\text{fa}}^{-1/N}-1)$).
- **Numerical Agreement:** Comparing calculated quantities within physical resolution tolerances ($\Delta R = c/(2B)$, $\Delta v = \lambda/(2 M T_c)$). Bit-for-bit identity is NOT claimed due to differences in floating-point libraries (FFTW vs NumPy/SciPy FFT implementations) and pseudo-random sequence generators.

### 17.3 Executed vs Implemented Status
When running in environments where MATLAB/Octave CLI is unavailable, the repository clearly distinguishes:
- **Implemented:** All MATLAB mathematical reference routines (`matlab/dsp/`, `matlab/radar/`, `matlab/validation/`) exist and are syntactically and mathematically verified.
- **Executed at Runtime:** Reported as unavailable when the MATLAB binary is absent from the host system environment.






