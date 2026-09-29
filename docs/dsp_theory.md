# RadarDSP Lab — Digital Signal Processing Theory & Mathematical Models

## 1. Continuous-Time vs. Discrete-Time Signals

In continuous-time signal processing, a physical signal $x(t)$ is defined over a continuous real domain $t \in \mathbb{R}$.

In digital signal processing (DSP), signals are represented as discrete-time sequences $x[n] = x(n \cdot T_s)$ obtained by sampling a continuous signal $x(t)$ at uniform temporal intervals $T_s = \frac{1}{F_s}$, where:
- $F_s$ is the **sampling frequency** (or sampling rate) in Hertz (Hz).
- $T_s$ is the **sampling period** (or time step $\Delta t$) in seconds.
- $n \in \{0, 1, 2, \dots, N-1\}$ is the integer sample index.

---

## 2. Discrete Time Vector Construction & Endpoint Convention

To avoid floating-point accumulation drift during time-step summation ($t_{n+1} = t_n + \Delta t$), discrete time vectors in `RadarDSP Lab` are constructed via direct integer index division:

$$t[n] = \frac{n}{F_s}, \quad n = 0, 1, \dots, N-1$$

where the total sample count $N$ is defined by:

$$N = \left\lfloor F_s \cdot T \right\rceil = \text{round}(F_s \cdot T)$$

### Endpoint Convention
- **Excluded Endpoint:** The duration endpoint $t = T$ is **EXCLUDED** from the time vector array.
- **Rationale:** For a signal of duration $T = 1.0\text{ s}$ sampled at $F_s = 1000\text{ Hz}$, exactly $N = 1000$ discrete samples are generated spanning the interval $t \in [0.0, 0.999]\text{ s}$. Including the endpoint $t = 1.0\text{ s}$ would yield $N + 1 = 1001$ samples, creating an off-by-one sample count mismatch and corrupting discrete Fourier transforms.

---

## 3. Signal Types & Mathematical Formulations

### 3.1 Sinusoidal Signals (`sine` and `cosine`)

Sinusoidal tones form the fundamental orthogonal basis for spectral analysis.

$$\text{Sine: } x(t) = A \cdot \sin(2\pi f t + \phi) + V_{\text{dc}}$$

$$\text{Cosine: } x(t) = A \cdot \cos(2\pi f t + \phi) + V_{\text{dc}}$$

Where:
- $A$: Peak signal amplitude.
- $f$: Fundamental frequency in Hz ($f \ge 0$).
- $\phi$: Initial phase offset in radians.
- $V_{\text{dc}}$: Constant DC bias offset.

---

### 3.2 Non-Sinusoids: Square, Triangle, and Sawtooth

#### Square Wave (`square`)
Generates a periodic rectangular wave alternating between $+A$ and $-A$:

$$x(t) = A \cdot \text{square}(2\pi f t + \phi, \text{duty}=D) + V_{\text{dc}}$$

where duty cycle $D \in (0, 1)$ specifies the fraction of each period in which the signal is positive.

#### Triangle Wave (`triangle`)
Generates a symmetric triangular wave with linear ramps between $-A$ and $+A$:

$$x(t) = A \cdot \text{sawtooth}(2\pi f t + \phi, \text{width}=0.5) + V_{\text{dc}}$$

#### Sawtooth Wave (`sawtooth`)
Generates a linear ramp wave rising from $-A$ to $+A$ over each full period:

$$x(t) = A \cdot \text{sawtooth}(2\pi f t + \phi, \text{width}=1.0) + V_{\text{dc}}$$

---

### 3.3 Multi-Tone Sinusoidal Sum (`multi_tone`)

A composite signal formed by superimposing $M$ independent harmonic or non-harmonic sinusoidal components:

$$x(t) = \sum_{k=0}^{M-1} A_k \cdot \sin(2\pi f_k t + \phi_k) + V_{\text{dc}}$$

**Constraint:** The lists of frequencies $\{f_k\}$, amplitudes $\{A_k\}$, and phases $\{\phi_k\}$ must have matching lengths $M \ge 1$.

---

### 3.4 Linear Frequency Modulated (LFM) Chirp (`chirp`)

An LFM chirp is a signal whose instantaneous frequency $f_{\text{inst}}(t)$ sweeps linearly over time from a start frequency $f_0$ to an end frequency $f_1$ across duration $T$:

$$f_{\text{inst}}(t) = f_0 + K \cdot t, \quad \text{where } K = \frac{f_1 - f_0}{T}$$

The instantaneous phase $\phi_{\text{inst}}(t)$ is the time integral of instantaneous frequency:

$$\phi_{\text{inst}}(t) = 2\pi \int_{0}^{t} f_{\text{inst}}(\tau) \, d\tau + \phi_0 = 2\pi \left( f_0 t + \frac{K}{2} t^2 \right) + \phi_0$$

The discrete chirp signal is generated as:

$$x(t) = A \cdot \cos\left( 2\pi \left( f_0 t + \frac{f_1 - f_0}{2T} t^2 \right) + \phi_0 \right) + V_{\text{dc}}$$

---

### 3.5 Gaussian White Noise (`noise`)

Generates independent and identically distributed (i.i.d.) Gaussian random variables:

$$x[n] \sim \mathcal{N}(V_{\text{dc}}, \sigma^2)$$

Where $\sigma = \text{std\_dev} \ge 0$ is the standard deviation.

#### Determinism & Reproducibility
Random sequences are generated using NumPy's modern `np.random.default_rng(seed)` generator API. Supplying an integer `seed` guarantees exact numerical reproducibility across runs.

---

### 3.6 Amplitude Modulation (`am`)

In Amplitude Modulation, a high-frequency carrier wave $c(t) = A_c \cos(2\pi f_c t + \phi)$ has its amplitude modulated by a lower-frequency modulating signal $m(t) = \sin(2\pi f_m t)$:

$$x(t) = A_c \cdot \left[ 1 + \mu \cdot \sin(2\pi f_m t) \right] \cdot \cos(2\pi f_c t + \phi) + V_{\text{dc}}$$

Where:
- $A_c$: Carrier amplitude.
- $f_c$: Carrier frequency in Hz ($f_c \ge 0$).
- $f_m$: Modulating signal frequency in Hz ($f_m \ge 0$).
- $\mu$: AM modulation index ($\mu \ge 0$). When $\mu \le 1.0$, envelope distortion is avoided.

---

### 3.7 Frequency Modulation (`fm`)

In Frequency Modulation, the instantaneous frequency of the carrier is varied proportionally to the modulating signal:

$$f_{\text{inst}}(t) = f_c + \Delta f \cdot \cos(2\pi f_m t)$$

Where $\Delta f \ge 0$ is the peak **frequency deviation** in Hz.

The FM modulation index is defined as:

$$\beta = \frac{\Delta f}{f_m}$$

The instantaneous phase is:

$$\phi_{\text{inst}}(t) = 2\pi f_c t + \beta \cdot \sin(2\pi f_m t) + \phi_0$$

The resulting FM signal is:

$$x(t) = A_c \cdot \cos\left( 2\pi f_c t + \frac{\Delta f}{f_m} \sin(2\pi f_m t) + \phi_0 \right) + V_{\text{dc}}$$

---

## 4. Frequency Validation vs. Nyquist Aliasing Policy

In `RadarDSP Lab`, a crucial architectural separation exists between **Signal Generation** and **Sampling Analysis**:

1. **Signal Generation Policy (`src/dsp/signals.py`):**
   The signal generator computes discrete sample evaluations $x[n] = f(n/F_s)$ for requested mathematical parameters. It enforces physical validity ($f \ge 0$, $F_s > 0$, $T > 0$, finite $A$), but does **NOT** restrict frequencies to $f < \frac{F_s}{2}$.
2. **Sampling Analysis Policy (`src/dsp/sampling.py` - Phase 2):**
   The evaluation of Nyquist rate ($F_{\text{Nyquist}} = 2 f_{\max}$), spectral aliasing foldover, undersampling, and signal reconstruction belongs strictly to the downstream sampling module.

This design enables explicit intentional undersampling and aliasing experiments without artificial generator errors.
