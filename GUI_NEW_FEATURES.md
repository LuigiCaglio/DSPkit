# GUI Integration — New DSPkit Features

This document lists all new DSPkit backend features that need GUI integration. Each section describes the backend API, the suggested GUI workflow, and the plot functions available.

---

## 1. Peak Detection (`dspkit.peaks`)

### Backend API
- `find_peaks(freqs, spectrum, prominence, height, distance_hz, max_peaks)` → peak_freqs, peak_values, prominences
- `peak_bandwidth(freqs, spectrum, peak_freqs, rel_height)` → peak_freqs, bandwidths, q_factors
- `find_harmonics(freqs, spectrum, fundamental, n_harmonics, tolerance_hz)` → harmonic_freqs, values, orders

### GUI Integration
- **Add a "Detect Peaks" button/checkbox** on any spectral plot (FFT, PSD). When enabled:
  - Run `find_peaks` on the displayed spectrum
  - Overlay peak markers (triangles) on the plot
  - Show a table/panel listing: frequency, amplitude, prominence, bandwidth, Q-factor
  - Allow the user to adjust: prominence threshold, min distance, max peaks (sliders or spinboxes)
- **Harmonic detection panel**: user selects a fundamental frequency (click on plot or type in), DSPkit finds harmonics and marks them
- **Plot function**: `plot_peaks(freqs, spectrum, peak_freqs, peak_values, db=True)`

### Parameters for GUI Controls
| Parameter | Type | Default | Widget |
|---|---|---|---|
| prominence | float | None | Slider / spinbox |
| distance_hz | float | None | Spinbox |
| max_peaks | int | None | Spinbox |
| fundamental (harmonics) | float | — | Spinbox or click-to-select |
| n_harmonics | int | 5 | Spinbox |

---

## 2. SHM Indicators (`dspkit.indicators`)

### Backend API
- `spectral_entropy(freqs, Pxx)` → float (0-1)
- `kurtosis(x, excess=True)` → float
- `skewness(x)` → float
- `rms_variation(x, fs, segment_duration)` → times, rms_values
- `frequency_shift(x, fs, segment_duration, nperseg)` → times, dominant_freqs
- `energy_variation(x, fs, segment_duration)` → times, energies

### GUI Integration
- **New "SHM Indicators" tab/panel** in the analysis section:
  - **Scalar indicators row**: show spectral entropy, kurtosis, skewness as labelled values (update when signal/channel changes)
  - **Time-varying indicators**: three sub-plots stacked vertically:
    1. RMS variation
    2. Energy variation
    3. Dominant frequency tracking
  - User controls: segment duration (slider, default = signal_length / 10)
- **Plot function**: `plot_indicators(times, values, title, ylabel)`

### Parameters for GUI Controls
| Parameter | Type | Default | Widget |
|---|---|---|---|
| segment_duration | float | auto (N/fs/10) | Slider |
| excess (kurtosis) | bool | True | Checkbox |

---

## 3. Multi-Sensor Analysis (`dspkit.multisensor`)

### Backend API
- `correlation_matrix(data)` → R (n_ch × n_ch)
- `coherence_matrix(data, fs, nperseg, ...)` → freqs, C (n_ch × n_ch × M)
- `psd_matrix(data, fs, nperseg, ...)` → freqs, G (n_ch × n_ch × M, complex)

### GUI Integration
- **New "Multi-Sensor" tab** (visible when ≥ 2 channels loaded):
  - **Correlation matrix heatmap**: `plot_correlation_matrix(R, labels)`
  - **Coherence matrix viewer**: dropdown to select channel pair, shows coherence vs frequency. Or a full n×n grid of coherence plots.
  - **PSD matrix**: primarily used as input to FDD (see below), but could show diagonal entries (auto-PSD) in a multi-line PSD plot
- Requires: user loads multi-channel data (CSV/MAT with multiple columns)

### Plot Functions
- `plot_correlation_matrix(R, labels, cmap)`
- Coherence: reuse `plot_coherence` for individual pairs

### Parameters for GUI Controls
| Parameter | Type | Default | Widget |
|---|---|---|---|
| nperseg | int | min(N, 1024) | Spinbox |
| window | str | "hann" | Dropdown |

---

## 4. FDD — Frequency Domain Decomposition (`dspkit.fdd`)

### Backend API
- `fdd_svd(data, fs, nperseg, ...)` → freqs, S (M × n_ch), U (M × n_ch × n_ch)
- `fdd_peak_picking(freqs, S, prominence, distance_hz, max_peaks, freq_range)` → peak_freqs, peak_indices
- `fdd_mode_shapes(U, peak_indices, normalize)` → modes (n_modes × n_ch)
- `efdd_damping(freqs, S, U, peak_indices, fs, mac_threshold, n_crossings)` → damping_ratios, natural_freqs

### GUI Integration
This is the most interactive new feature. Suggested workflow:

1. **User loads multi-channel data** (≥ 2 channels)
2. **"FDD Analysis" tab opens**:
   - **Top panel**: Singular value plot (`plot_singular_values`). Shows SV1, SV2, ... in dB vs frequency.
   - **Peak picking**: User can either:
     - Click peaks on the SV plot (interactive picking) — preferred for GUI
     - Or use automatic detection with controls for prominence/distance/max_peaks
   - **Below the SV plot**: Results table showing:
     | Mode | Frequency [Hz] | Damping [%] | Mode Shape |
   - **Mode shape visualisation**: For each selected peak, show `plot_mode_shape` as a bar chart. If sensor positions are known, could show a geometry-based plot.
3. **EFDD damping**: after peak picking, run `efdd_damping` and display results in the table.

### Plot Functions
- `plot_singular_values(freqs, S, n_sv, db, peak_freqs, xlim)` — main SV plot
- `plot_mode_shape(mode, sensor_labels, title)` — bar chart per mode

### Parameters for GUI Controls
| Parameter | Type | Default | Widget |
|---|---|---|---|
| nperseg | int | min(N, 1024) | Spinbox (affects frequency resolution) |
| window | str | "hann" | Dropdown |
| prominence (dB) | float | None | Slider |
| distance_hz | float | None | Spinbox |
| max_peaks | int | None | Spinbox |
| freq_range | (float, float) | full | Range slider or text inputs |
| mac_threshold (EFDD) | float | 0.8 | Slider (0-1) |
| n_crossings (EFDD) | int | 10 | Spinbox |

### Interactive Features (Priority)
- **Click-to-pick peaks** on the SV plot is the killer feature for FDD in a GUI
- Hovering over a peak could preview the mode shape
- User should be able to add/remove peaks and re-run EFDD

---

## 5. Probability & Statistics (`dspkit.statistics`)

### Backend API
- `pdf_estimate(x, n_points, bandwidth)` → xi, density
- `histogram(x, bins, density)` → bin_centres, counts
- `joint_histogram(x, y, bins, density)` → x_centres, y_centres, H
- `covariance_matrix(data, bias)` → C (n_ch × n_ch)
- `mahalanobis(data, reference)` → distances (N,)

### GUI Integration
- **New "Statistics" tab**:
  - **PDF panel**: show KDE curve overlaid on histogram for selected channel
    - User controls: number of bins, KDE bandwidth (auto/manual)
  - **Joint distribution**: user selects two channels → 2D heatmap
  - **Covariance matrix**: heatmap similar to correlation matrix
  - **Mahalanobis distance**: time series plot highlighting outliers above a threshold
    - User controls: percentile threshold (slider, default 99th)
- **Plot functions**:
  - `plot_pdf(xi, density, hist_data, hist_bins)`
  - `plot_joint_histogram(x_centres, y_centres, H, xlabel, ylabel)`
  - `plot_correlation_matrix` (reuse for covariance with different colormap)
  - `plot_indicators` (reuse for Mahalanobis time series)

### Parameters for GUI Controls
| Parameter | Type | Default | Widget |
|---|---|---|---|
| bins | int | 50 | Slider |
| bandwidth (KDE) | float | auto | Spinbox (optional) |
| channel_x, channel_y (joint) | str | — | Dropdowns |
| outlier_percentile | float | 99 | Slider |

---

## 6. New Plot Functions Summary

All new plotting functions follow the existing convention: accept `ax=None` (create figure) or `ax=Axes` (embed in existing layout). They all return the `Axes` object.

| Function | Module | Used For |
|---|---|---|
| `plot_peaks` | plots | Spectrum with peak markers |
| `plot_singular_values` | plots | FDD singular value curves |
| `plot_mode_shape` | plots | Mode shape bar chart |
| `plot_pdf` | plots | KDE + histogram overlay |
| `plot_joint_histogram` | plots | 2D heatmap |
| `plot_correlation_matrix` | plots | Correlation/covariance heatmap |
| `plot_indicators` | plots | SHM indicator time series |

---

## 7. Suggested GUI Tab Structure

```
Data Loading
├── Load CSV / MAT / HDF5
├── Channel selection
└── Time range selection

Analysis (existing tabs extended)
├── Time Domain
├── Spectral Analysis
│   └── [NEW] Peak Detection overlay
├── Filtering
├── Time-Frequency
├── Hilbert / Instantaneous
├── EMD / HHT
├── [NEW] SHM Indicators
├── [NEW] Multi-Sensor
├── [NEW] FDD (Operational Modal Analysis)
└── [NEW] Statistics / Probability
```

---

## 8. Data Requirements

| Feature | Min Channels | Notes |
|---|---|---|
| Peak detection | 1 | Works on any spectrum |
| SHM indicators | 1 | Works on any single channel |
| Multi-sensor tools | 2+ | Requires multi-channel data |
| FDD | 2+ | Requires multi-channel data |
| Statistics (PDF) | 1 | Per-channel |
| Statistics (joint) | 2 | Requires 2 channels |
| Mahalanobis | 2+ | Requires multi-channel data |
