"""
Example: peak detection and SHM indicators on a 2DOF structural system.

Demonstrates:
- Peak detection on FFT and PSD spectra
- Peak bandwidth and Q-factor estimation
- Harmonic identification
- SHM indicators: spectral entropy, kurtosis, skewness
- Time-varying indicators: RMS variation, frequency shift, energy variation

Run:
    python examples/example_peaks_indicators.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from dspkit._testing import generate_2dof, natural_frequencies_2dof
from dspkit.spectral import fft_spectrum, psd
from dspkit.peaks import find_peaks, peak_bandwidth, find_harmonics
from dspkit.indicators import (
    spectral_entropy, kurtosis, skewness,
    rms_variation, frequency_shift, energy_variation,
)
from dspkit.plots import plot_peaks, plot_indicators

# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

FS = 1000.0
DURATION = 120.0
fn1, fn2 = natural_frequencies_2dof()

t, a1, a2 = generate_2dof(duration=DURATION, fs=FS, noise_std=1.0,
                            output="acceleration", seed=42)

# ===========================================================================
# Figure 1: Peak Detection
# ===========================================================================

fig = plt.figure(figsize=(14, 10), constrained_layout=True)
fig.suptitle("Peak Detection & Harmonic Analysis", fontsize=14, fontweight="bold")
gs = gridspec.GridSpec(2, 2, figure=fig)

# --- FFT peak detection ---
freqs_fft, amp = fft_spectrum(a1, FS)
peak_freqs, peak_vals, proms = find_peaks(
    freqs_fft, amp, distance_hz=3.0, max_peaks=5,
)

ax = fig.add_subplot(gs[0, 0])
plot_peaks(freqs_fft, amp, peak_freqs, peak_vals, ax=ax,
           xlim=(0, 80), title="FFT — Peak Detection")
for f in peak_freqs:
    ax.annotate(f"{f:.1f} Hz", (f, amp[np.argmin(np.abs(freqs_fft - f))]),
                textcoords="offset points", xytext=(5, 10), fontsize=7)

# --- PSD peak detection ---
freqs_psd, Pxx = psd(a1, FS, nperseg=4096)
pf_psd, pv_psd, _ = find_peaks(freqs_psd, Pxx, distance_hz=5.0, max_peaks=3)

ax = fig.add_subplot(gs[0, 1])
plot_peaks(freqs_psd, Pxx, pf_psd, pv_psd, ax=ax, db=True,
           xlim=(0, 80), title="PSD — Peak Detection (dB)")
ax.axvline(fn1, color="cyan", ls="--", lw=1, label=f"fn1={fn1:.1f} Hz")
ax.axvline(fn2, color="lime", ls="--", lw=1, label=f"fn2={fn2:.1f} Hz")
ax.legend(fontsize=7)

# --- Bandwidth estimation ---
pf_bw, bw, Q = peak_bandwidth(freqs_psd, Pxx, peak_freqs=pf_psd)

ax = fig.add_subplot(gs[1, 0])
ax.barh(range(len(pf_bw)), bw, color="steelblue", edgecolor="black")
ax.set_yticks(range(len(pf_bw)))
ax.set_yticklabels([f"{f:.1f} Hz" for f in pf_bw])
ax.set_xlabel("Bandwidth [Hz]")
ax.set_title("Peak Bandwidth (half-power)")
for i, (b, q) in enumerate(zip(bw, Q)):
    ax.text(b + 0.05, i, f"Q = {q:.1f}", va="center", fontsize=9)
ax.grid(True, axis="x", alpha=0.3)

# --- Harmonic detection ---
# Add a fake harmonic signal for demonstration
t_h = np.arange(int(10 * FS)) / FS
x_harm = (np.sin(2 * np.pi * 25 * t_h) +
           0.5 * np.sin(2 * np.pi * 50 * t_h) +
           0.3 * np.sin(2 * np.pi * 75 * t_h) +
           0.15 * np.sin(2 * np.pi * 100 * t_h))
freqs_h, amp_h = fft_spectrum(x_harm, FS)
hf, hv, ho = find_harmonics(freqs_h, amp_h, fundamental=25.0, n_harmonics=5)

ax = fig.add_subplot(gs[1, 1])
ax.semilogy(freqs_h, amp_h, lw=0.8, color="gray")
ax.plot(hf, hv, "rv", ms=8, label="harmonics")
for f, v, n in zip(hf, hv, ho):
    ax.annotate(f"H{n} ({f:.0f} Hz)", (f, v),
                textcoords="offset points", xytext=(5, 8), fontsize=8)
ax.set_xlim(0, 150)
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Amplitude")
ax.set_title("Harmonic Detection (f0 = 25 Hz)")
ax.legend(fontsize=8)
ax.grid(True, which="both", alpha=0.3)

plt.show()


# ===========================================================================
# Figure 2: SHM Indicators
# ===========================================================================

fig2 = plt.figure(figsize=(14, 10), constrained_layout=True)
fig2.suptitle("SHM Indicators", fontsize=14, fontweight="bold")
gs2 = gridspec.GridSpec(3, 2, figure=fig2)

# --- Scalar indicators ---
se = spectral_entropy(freqs_psd, Pxx)
kurt = kurtosis(a1)
skew = skewness(a1)

ax = fig2.add_subplot(gs2[0, :])
names = ["Spectral\nEntropy", "Excess\nKurtosis", "Skewness"]
values = [se, kurt, skew]
colors = ["steelblue", "coral", "mediumseagreen"]
bars = ax.bar(names, values, color=colors, edgecolor="black", width=0.5)
for bar, v in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02 * max(abs(v), 0.1),
            f"{v:.4f}", ha="center", fontsize=10, fontweight="bold")
ax.set_title(f"Signal Statistics — mass 1 ({DURATION:.0f} s)")
ax.axhline(0, color="black", lw=0.6)
ax.grid(True, axis="y", alpha=0.3)

# --- RMS variation ---
times_rms, rms_vals = rms_variation(a1, FS, segment_duration=10.0)
ax = fig2.add_subplot(gs2[1, 0])
plot_indicators(times_rms, rms_vals, ax=ax,
                title="RMS Variation (10 s segments)", ylabel="RMS [m/s²]")

# --- Energy variation ---
times_e, energies = energy_variation(a1, FS, segment_duration=10.0)
ax = fig2.add_subplot(gs2[1, 1])
plot_indicators(times_e, energies, ax=ax,
                title="Energy Variation (10 s segments)", ylabel="Energy [(m/s²)²]")

# --- Frequency shift ---
times_f, dom_freqs = frequency_shift(a1, FS, segment_duration=10.0)
ax = fig2.add_subplot(gs2[2, 0])
plot_indicators(times_f, dom_freqs, ax=ax,
                title="Dominant Frequency Tracking", ylabel="Frequency [Hz]")
ax.axhline(fn1, color="red", ls="--", lw=1, label=f"fn1={fn1:.1f} Hz")
ax.legend(fontsize=8)

# --- Comparison: entropy across segments ---
times_se = []
se_vals = []
seg_dur = 10.0
seg_len = int(seg_dur * FS)
for i in range(len(a1) // seg_len):
    chunk = a1[i * seg_len : (i + 1) * seg_len]
    f, P = psd(chunk, FS, nperseg=min(seg_len, 1024))
    times_se.append((i + 0.5) * seg_dur)
    se_vals.append(spectral_entropy(f, P))

ax = fig2.add_subplot(gs2[2, 1])
plot_indicators(np.array(times_se), np.array(se_vals), ax=ax,
                title="Spectral Entropy over Time", ylabel="Entropy [-]")

plt.show()
