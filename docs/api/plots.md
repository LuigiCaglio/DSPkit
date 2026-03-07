# Plots

Thin matplotlib wrappers for all DSPkit analysis outputs.

## Design

- Every function accepts `ax=None`. When `None`, a new figure is created automatically.
- Every function **returns the `Axes` object** so calls can be chained or the axes customised further.
- Extra `**kwargs` are forwarded to the underlying `ax.plot` / `ax.pcolormesh` call.

## Embedding in multi-panel figures

```python
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import dspkit as dsp

fig = plt.figure(figsize=(12, 8), constrained_layout=True)
gs = gridspec.GridSpec(2, 2, figure=fig)

dsp.plot_psd(freqs, Pxx,   ax=fig.add_subplot(gs[0, 0]))
dsp.plot_coherence(f, Cxy, ax=fig.add_subplot(gs[0, 1]))
dsp.plot_spectrogram(f, t, Zxx, ax=fig.add_subplot(gs[1, :]))
plt.show()
```

---

## Time-domain

::: dspkit.plots.plot_signal

---

## Spectral

::: dspkit.plots.plot_fft

---

::: dspkit.plots.plot_psd

---

::: dspkit.plots.plot_csd

---

::: dspkit.plots.plot_coherence

---

::: dspkit.plots.plot_autocorrelation

---

::: dspkit.plots.plot_cross_correlation

---

## Time-frequency

::: dspkit.plots.plot_spectrogram

---

::: dspkit.plots.plot_scalogram

---

::: dspkit.plots.plot_wvd
