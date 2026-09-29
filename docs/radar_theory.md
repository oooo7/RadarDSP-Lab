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
