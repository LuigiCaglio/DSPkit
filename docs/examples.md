# Examples

The [`examples/`](https://github.com/LuigiCaglio/DSPkit/tree/main/examples) folder contains eight self-contained scripts. Each one generates a multi-panel matplotlib figure you can run directly:

```bash
python examples/example_spectral.py
python examples/example_filters_utils.py
python examples/example_timefreq.py
python examples/example_instantaneous.py
python examples/example_emd.py
python examples/example_peaks_indicators.py
python examples/example_fdd.py
python examples/example_multisensor_stats.py
```

---

## example_spectral.py

**2DOF spectral analysis.**

Simulates 60 s of 2DOF acceleration under white noise, then shows:

- Time signal (first 2 s), both channels
- FFT amplitude spectrum (semi-log)
- Welch PSD of both channels
- Magnitude-squared coherence between channels
- Autocorrelation with 95 % confidence bands

---

## example_filters_utils.py

**Filtering, decimation, signal metrics.**

Adds 50 Hz mains hum to the 2DOF signal, then demonstrates:

- Bandpass filter to isolate mode 1
- Notch filter to remove the 50 Hz tone
- Decimation from 1 kHz → 100 Hz with anti-aliasing
- Numerical integration: acceleration → velocity → displacement
- Bar chart of RMS, peak, and crest factor across signal variants

---

## example_timefreq.py

**Time-frequency comparison on a chirp and a 2DOF segment.**

Runs four TF methods side by side (chirp on the left, 2DOF on the right):

| Row | Method | Notes |
|---|---|---|
| 0 | STFT | 128-sample window, 94 % overlap |
| 1 | CWT (Morlet) | Log frequency axis, 80 analysis frequencies |
| 2 | Wigner-Ville | Cross-terms visible, N = 1000 |
| 3 | Smoothed Pseudo WVD | Hann lag + time smoothing reduce cross-terms |

---

## example_instantaneous.py

**Hilbert transform on three signal types.**

Three rows, three columns (waveform + envelope, instantaneous frequency, instantaneous phase):

1. AM sine at 100 Hz — envelope recovers the 3 Hz modulation exactly
2. Linear chirp 10 → 200 Hz — instantaneous frequency tracks the sweep
3. 2DOF ring-down, mode 1 isolated — log-envelope gives damping ratio ζ

---

## example_emd.py

**EMD and Hilbert-Huang Transform on 20 s of 2DOF acceleration.**

- **Figure 1 — IMF waterfall:** each IMF plotted with its Hilbert envelope
- **Figure 2 — HHT analysis:**
    - Time-frequency scatter coloured by instantaneous energy
    - Marginal spectrum vs Welch PSD
    - Log-envelope fit → damping ratio per modal IMF

---

## example_peaks_indicators.py

**Peak detection and SHM indicators on a 2DOF system.**

- **Figure 1 — Peak Detection:**
    - FFT peak detection with prominence filtering
    - PSD peak detection in dB
    - Peak bandwidth and Q-factor estimation
    - Harmonic series identification (f0 = 25 Hz)

- **Figure 2 — SHM Indicators:**
    - Scalar indicators: spectral entropy, kurtosis, skewness
    - RMS variation over time (10 s segments)
    - Energy variation over time
    - Dominant frequency tracking
    - Spectral entropy evolution

---

## example_fdd.py

**Frequency Domain Decomposition (FDD) on a 2DOF system.**

Full OMA workflow:

- Singular values of the PSD matrix (peak picking)
- Natural frequency identification
- Mode shape extraction and visualisation
- Enhanced FDD (EFDD) damping estimation

---

## example_multisensor_stats.py

**Multi-sensor analysis and probability statistics.**

- **Figure 1 — Multi-Sensor:**
    - Correlation matrix heatmap
    - Coherence between channels vs frequency
    - PSD matrix singular values (FDD-style)
    - Covariance matrix

- **Figure 2 — Probability Statistics:**
    - PDF estimation (KDE + histogram) for each channel
    - Joint distribution (2D histogram)
    - Mahalanobis distance for outlier detection
