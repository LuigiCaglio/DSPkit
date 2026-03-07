# Conventions

## Input signals

All functions expect a **1D NumPy array** (or anything that `np.asarray` can promote to one):

```python
x : array_like, shape (N,)   # N samples, single channel
fs : float                    # sampling frequency in Hz
```

Multi-channel data is **not** handled natively. Process each channel independently:

```python
freqs, Pxx1 = dsp.psd(channel_1, fs)
freqs, Pxx2 = dsp.psd(channel_2, fs)
```

---

## Output shapes

### Spectral functions

| Function | Returns |
|---|---|
| `fft_spectrum(x, fs)` | `freqs (N//2+1,)`, `amplitude (N//2+1,)` |
| `psd(x, fs)` | `freqs (M,)`, `Pxx (M,)` — M depends on `nperseg` |
| `csd(x, y, fs)` | `freqs (M,)`, `Pxy (M,)` **complex** |
| `coherence(x, y, fs)` | `freqs (M,)`, `Cxy (M,)` real ∈ [0, 1] |
| `autocorrelation(x, fs)` | `lags (L,)`, `acf (L,)` |

### Filters

All filter functions return an array of the **same shape as the input** `(N,)`. `decimate` additionally returns the new sampling frequency:

```python
x_out = dsp.lowpass(x, fs, cutoff=50.0)        # shape (N,)
x_dec, fs_new = dsp.decimate(x, fs, target_fs=100.0)  # shape (N//ratio,)
```

### Time-frequency

!!! warning "Axis order"
    STFT output is **(n_freqs, n_times)** — frequency first.
    WVD / SPWVD output is **(n_times, n_freqs)** — time first.
    This matches the conventions of `scipy.signal.stft` and the WVD literature respectively. `plot_wvd` handles the transpose internally.

| Function | freqs | times | Data shape |
|---|---|---|---|
| `stft` | `(nperseg//2+1,)` | `(n_frames,)` | `(n_freqs, n_frames)` complex |
| `cwt_scalogram` | `(n_freqs,)` user-defined | `(N,)` | `(n_freqs, N)` complex |
| `wigner_ville` | `(N//2+1,)` | `(N,)` | `(N, N//2+1)` real |
| `smoothed_pseudo_wv` | `(N//2+1,)` | `(N,)` | `(N, N//2+1)` real |

### Instantaneous

All functions return arrays of shape `(N,)` — same length as the input:

```python
env, phase, fi = dsp.hilbert_attributes(x, fs)
# env   : (N,)  instantaneous amplitude  ≥ 0
# phase : (N,)  unwrapped phase [rad]
# fi    : (N,)  instantaneous frequency [Hz]
```

### EMD

```python
imfs, residue = dsp.emd(x)
# imfs    : (n_imfs, N)  each row is one IMF
# residue : (N,)         monotone trend

envs, inst_freqs = dsp.hht(imfs, fs)
# envs       : (n_imfs, N)
# inst_freqs : (n_imfs, N)  [Hz]

freq_bins, spectrum = dsp.hht_marginal_spectrum(envs, inst_freqs, fs)
# freq_bins : (n_bins,)
# spectrum  : (n_bins,)
```

Reconstruction is always exact (within floating-point tolerance):

```python
assert np.allclose(imfs.sum(axis=0) + residue, x, atol=1e-8)
```

---

## Units

DSPkit is unit-agnostic. Whatever units `x` is in, outputs carry those units consistently:

| Input unit | `fft_spectrum` | `psd` | `rms` / `peak` |
|---|---|---|---|
| m/s² | m/s² | (m/s²)²/Hz | m/s² |
| µε | µε | µε²/Hz | µε |

The time axis is always in **seconds** when `fs` is in Hz.

---

## dtype

All functions cast inputs to `float64` internally via `np.asarray(x, dtype=float)`. Passing integer or float32 arrays is safe; the original array is never modified.
