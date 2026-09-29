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

---

## 14. Velocity Resolution & Unambiguous Velocity Bounds

### 14.1 Velocity Resolution ($\Delta v$)
Velocity resolution $\Delta v$ is the minimum velocity separation required to distinguish two targets in the same range cell. It depends on the total coherent observation duration $M \cdot T_c$:

$$\Delta v = \frac{\lambda}{2 M T_c} = \frac{\lambda \cdot \text{PRF}}{2 M}$$

### 14.2 Maximum Unambiguous Velocity ($v_{\max}$)
The maximum radial velocity that can be measured without Doppler phase aliasing across chirps spaced by $T_c$:

$$f_{D,\max} = \frac{\text{PRF}}{2} = \frac{1}{2 T_c} \implies v_{\max} = \frac{\lambda}{4 T_c} = \frac{c}{4 f_c T_c}$$



