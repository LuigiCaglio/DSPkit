"""
Example: spectral analysis of a simulated 2DOF structural system.

Generates acceleration responses from a 2DOF spring-mass-damper chain
under white noise excitation, then runs through the main spectral tools.

Layout:
    ground --[k1,c1]-- m1 --[k2,c2]-- m2
    fn1 ~ 8.6 Hz,  fn2 ~ 20.8 Hz  (default parameters)

Run:
    python examples/example_spectral.py
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from dspkit._testing import generate_2dof, natural_frequencies_2dof
from dspkit.spectral import fft_spectrum, psd, csd, coherence, autocorrelation, cross_correlation
from dspkit.plots import (
    plot_signal, plot_fft, plot_psd, plot_csd,
    plot_coherence, plot_autocorrelation, plot_cross_correlation,
)

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------

FS = 1000.0       # Hz
DURATION = 60.0   # s  (longer → smoother Welch estimate)
NPERSEG = 4096    # frequency resolution = FS / NPERSEG ~ 0.24 Hz

fn1, fn2 = natural_frequencies_2dof()
print(f"Theoretical natural frequencies:  fn1 = {fn1:.2f} Hz,  fn2 = {fn2:.2f} Hz")

t, a1, a2 = generate_2dof(
    duration=DURATION,
    fs=FS,
    noise_std=1.0,
    output="acceleration",
    seed=42,
)

# ---------------------------------------------------------------------------
# Figure layout  (4 rows × 2 columns)
# ---------------------------------------------------------------------------

fig = plt.figure(figsize=(14, 14), constrained_layout=True)
fig.suptitle("2DOF System — Spectral Analysis", fontsize=14, fontweight="bold")

gs = gridspec.GridSpec(4, 2, figure=fig)

ax_time = fig.add_subplot(gs[0, :])   # full-width top row
ax_fft  = fig.add_subplot(gs[1, 0])
ax_psd  = fig.add_subplot(gs[1, 1])
ax_csd  = fig.add_subplot(gs[2, 0])
ax_coh  = fig.add_subplot(gs[2, 1])
ax_acf  = fig.add_subplot(gs[3, 0])
ax_ccf  = fig.add_subplot(gs[3, 1])

# ---------------------------------------------------------------------------
# Time signal (first 2 s)
# ---------------------------------------------------------------------------

mask = t <= 2.0
plot_signal(t[mask], a1[mask], ax=ax_time, label="mass 1",
            title="Time signal (first 2 s)", ylabel="Acceleration [m/s²]")
ax_time.plot(t[mask], a2[mask], label="mass 2", lw=0.8, alpha=0.8)
ax_time.legend()

# ---------------------------------------------------------------------------
# FFT amplitude spectrum (mass 1, full signal)
# ---------------------------------------------------------------------------

freqs_fft, amp = fft_spectrum(a1, FS, window="hann", scaling="amplitude")

plot_fft(freqs_fft, amp, ax=ax_fft, xlim=(0, 80),
         title="FFT spectrum — mass 1", ylabel="Amplitude [m/s²]")
ax_fft.axvline(fn1, color="red",    ls="--", lw=1.2, label=f"fn1 = {fn1:.1f} Hz")
ax_fft.axvline(fn2, color="orange", ls="--", lw=1.2, label=f"fn2 = {fn2:.1f} Hz")
ax_fft.legend(fontsize=8)

# ---------------------------------------------------------------------------
# PSD via Welch (both channels)
# ---------------------------------------------------------------------------

freqs_psd, Pxx1 = psd(a1, FS, nperseg=NPERSEG)
freqs_psd, Pxx2 = psd(a2, FS, nperseg=NPERSEG)

plot_psd(freqs_psd, Pxx1, ax=ax_psd, db=False, xlim=(0, 80),
         title=f"Welch PSD  (nperseg={NPERSEG}, Δf={FS/NPERSEG:.2f} Hz)",
         ylabel="PSD [(m/s²)²/Hz]", label="mass 1")
ax_psd.semilogy(freqs_psd, Pxx2, label="mass 2", lw=0.9, alpha=0.85)
ax_psd.axvline(fn1, color="red",    ls="--", lw=1.2, label=f"fn1 = {fn1:.1f} Hz")
ax_psd.axvline(fn2, color="orange", ls="--", lw=1.2, label=f"fn2 = {fn2:.1f} Hz")
ax_psd.legend(fontsize=8)

# ---------------------------------------------------------------------------
# Cross-spectral density (mass 1 → mass 2)
# ---------------------------------------------------------------------------

freqs_csd, Pxy = csd(a1, a2, FS, nperseg=NPERSEG)

plot_csd(freqs_csd, Pxy, ax=ax_csd, db=False, xlim=(0, 80),
         title="Cross-spectral density (mass 1 → mass 2)",
         ylabel="|CSD| [(m/s²)²/Hz]")
ax_csd.axvline(fn1, color="red",    ls="--", lw=1.2, label=f"fn1 = {fn1:.1f} Hz")
ax_csd.axvline(fn2, color="orange", ls="--", lw=1.2, label=f"fn2 = {fn2:.1f} Hz")
ax_csd.legend(fontsize=8)

# ---------------------------------------------------------------------------
# Coherence between the two channels
# ---------------------------------------------------------------------------

freqs_coh, Cxy = coherence(a1, a2, FS, nperseg=NPERSEG)

plot_coherence(freqs_coh, Cxy, ax=ax_coh, threshold=0.8, xlim=(0, 80),
               title="Magnitude-squared coherence (mass 1 vs mass 2)")
ax_coh.axvline(fn1, color="red",    ls="--", lw=1.2, label=f"fn1 = {fn1:.1f} Hz")
ax_coh.axvline(fn2, color="orange", ls="--", lw=1.2, label=f"fn2 = {fn2:.1f} Hz")
ax_coh.legend(fontsize=8)

# ---------------------------------------------------------------------------
# Autocorrelation (mass 1, up to 0.5 s lag)
# ---------------------------------------------------------------------------

lags_acf, acf = autocorrelation(a1, fs=FS, normalize=True, max_lag=0.5)

plot_autocorrelation(lags_acf, acf, ax=ax_acf, n_samples=len(a1),
                     xlabel="Lag [s]", title="Autocorrelation — mass 1")

# ---------------------------------------------------------------------------
# Cross-correlation (mass 1 vs mass 2, ±0.2 s)
# ---------------------------------------------------------------------------

lags_ccf, ccf = cross_correlation(a1, a2, fs=FS, normalize=True, max_lag=0.2)

plot_cross_correlation(lags_ccf, ccf, ax=ax_ccf, n_samples=len(a1),
                       xlabel="Lag [s]",
                       title="Cross-correlation (mass 1 vs mass 2)")

plt.show()
