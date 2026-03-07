# Quick Start

This guide walks through the main DSPkit workflows on a simulated 2DOF structural system — the same test signal used throughout the example scripts.

## The test signal

DSPkit ships with a built-in 2DOF spring-mass-damper simulator:

```python
from dspkit._testing import generate_2dof, natural_frequencies_2dof

fs = 1000.0   # Hz
t, a1, a2 = generate_2dof(
    duration=60.0,
    fs=fs,
    noise_std=1.0,
    output="acceleration",   # "displacement" | "velocity" | "acceleration"
    seed=42,
)

fn1, fn2 = natural_frequencies_2dof()
# fn1 ≈ 8.6 Hz,  fn2 ≈ 20.8 Hz
```

## Spectral analysis

```python
import dspkit as dsp

# Single-sided FFT amplitude spectrum
freqs, amp = dsp.fft_spectrum(a1, fs, window="hann", scaling="amplitude")
dsp.plot_fft(freqs, amp, xlim=(0, 80), title="FFT spectrum")

# Welch PSD
freqs, Pxx = dsp.psd(a1, fs, nperseg=4096)
dsp.plot_psd(freqs, Pxx, xlim=(0, 80))

# Magnitude-squared coherence between channels
freqs, Cxy = dsp.coherence(a1, a2, fs, nperseg=4096)
dsp.plot_coherence(freqs, Cxy, xlim=(0, 80))

# Autocorrelation
lags, acf = dsp.autocorrelation(a1, fs=fs, normalize=True, max_lag=0.5)
dsp.plot_autocorrelation(lags, acf, n_samples=len(a1), xlabel="Lag [s]")
```

## Filtering

```python
# Zero-phase Butterworth bandpass around first mode
a1_bp = dsp.bandpass(a1, fs, low=fn1 - 3, high=fn1 + 3, order=4)

# Notch filter to remove 50 Hz mains hum
a1_notched = dsp.notch(a1, fs, freq=50.0, q=30.0)

# Decimate to 100 Hz with anti-aliasing
a1_dec, fs_dec = dsp.decimate(a1, fs, target_fs=100.0)
```

## Time-frequency analysis

```python
import numpy as np

# Short-Time Fourier Transform
f, t, Zxx = dsp.stft(a1[:4096], fs, nperseg=256)
dsp.plot_spectrogram(f, t, Zxx, ylim=(0, 80))

# CWT scalogram (analytic Morlet, FFT-based)
analysis_freqs = np.geomspace(2.0, fs / 4, num=100)
f, t, W = dsp.cwt_scalogram(a1[:4096], fs, freqs=analysis_freqs)
dsp.plot_scalogram(f, t, W, ylim=(2, 80))

# Wigner-Ville (O(N²) — keep N small)
f, t, WVD = dsp.wigner_ville(a1[:512], fs)
dsp.plot_wvd(f, t, WVD, ylim=(0, 80))
```

## Hilbert transform

```python
# All three attributes in one call
env, phase, fi = dsp.hilbert_attributes(a1_bp, fs)

import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 1, figsize=(10, 5), constrained_layout=True)
dsp.plot_signal(t, a1_bp, ax=axes[0], envelope=env, title="Bandpass + envelope")
axes[1].plot(t, fi); axes[1].set_ylabel("Instantaneous freq [Hz]")
plt.show()
```

## EMD and HHT

```python
# Decompose into IMFs
imfs, residue = dsp.emd(a1, max_imfs=8)

# Hilbert-Huang Transform
envs, inst_freqs = dsp.hht(imfs, fs)

# Marginal spectrum (adaptive PSD analogue)
freq_bins, marginal = dsp.hht_marginal_spectrum(envs, inst_freqs, fs, n_bins=512)
```

## Embedding plots in your own figures

Every `plot_*` function accepts an `ax` parameter:

```python
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(12, 8), constrained_layout=True)
gs = gridspec.GridSpec(2, 2, figure=fig)

dsp.plot_psd(freqs, Pxx, ax=fig.add_subplot(gs[0, 0]))
dsp.plot_coherence(freqs, Cxy, ax=fig.add_subplot(gs[0, 1]))
dsp.plot_spectrogram(f, t, Zxx, ax=fig.add_subplot(gs[1, :]))
plt.show()
```
