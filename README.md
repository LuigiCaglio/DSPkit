# DSPkit

**[Documentation](https://LuigiCaglio.github.io/DSPkit)**

A lightweight Digital Signal Processing (DSP) toolkit for time series data. Mostly for plotting and visual analysis.
Based on NumPy + SciPy + Matplotlib.

---

## Features

| Module | Functions |
|---|---|
| `spectral` | FFT amplitude spectrum, Welch PSD, CSD, coherence, autocorrelation |
| `filters` | Lowpass / highpass / bandpass / bandstop / notch filters, decimation |
| `utils` | Detrend, RMS, peak, crest factor, integration, differentiation |
| `timefreq` | STFT spectrogram, CWT scalogram (Morlet), Wigner-Ville, Smoothed Pseudo WVD |
| `instantaneous` | Hilbert envelope, instantaneous phase & frequency |
| `emd` | Empirical Mode Decomposition, Hilbert-Huang Transform, marginal spectrum |
| `plots` | Thin matplotlib wrappers for every analysis output |

---

## Installation

```bash
pip install git+https://github.com/LuigiCaglio/DSPkit.git
```

**Requirements:** Python ≥ 3.10, NumPy ≥ 1.24, SciPy ≥ 1.10, Matplotlib ≥ 3.7

---

## Quick start

```python
import numpy as np
import dspkit as dsp

# Simulate a 2DOF structural response
from dspkit._testing import generate_2dof, natural_frequencies_2dof

fs = 1000.0
t, a1, a2 = generate_2dof(duration=60.0, fs=fs, noise_std=1.0,
                           output="acceleration", seed=42)

# Welch PSD
freqs, Pxx = dsp.psd(a1, fs, nperseg=4096)
dsp.plot_psd(freqs, Pxx, title="Welch PSD — mass 1")

# Bandpass filter around first mode
fn1, fn2 = natural_frequencies_2dof()
a1_bp = dsp.bandpass(a1, fs, low=fn1 - 3, high=fn1 + 3)

# Hilbert envelope
env, phase, fi = dsp.hilbert_attributes(a1_bp, fs)

# EMD
imfs, residue = dsp.emd(a1)
envs, inst_freqs = dsp.hht(imfs, fs)
```

---

## Examples

Runnable scripts with plots are in the [`examples/`](examples/) folder:

| Script | Demonstrates |
|---|---|
| `example_spectral.py` | FFT, Welch PSD, coherence, autocorrelation |
| `example_filters_utils.py` | Filtering, notch, decimation, integration, signal metrics |
| `example_timefreq.py` | STFT, CWT, Wigner-Ville, Smoothed Pseudo WVD |
| `example_instantaneous.py` | Hilbert envelope, instantaneous frequency & phase |
| `example_emd.py` | EMD, HHT time-frequency scatter, marginal spectrum, damping |

```bash
python examples/example_spectral.py
```

---

## Documentation

Full API reference and narrative guides:
**[https://LuigiCaglio.github.io/DSPkit](https://LuigiCaglio.github.io/DSPkit)**

To build docs locally:

```bash
pip install -e ".[docs]"
mkdocs serve
```

---

## Development

```bash
pip install -e ".[dev]"
pytest
```

---

## License

[MIT](LICENSE)
