"""
Example: filtering, decimation, and signal metrics on a 2DOF system.

Demonstrates:
  - Bandpass filtering to isolate a resonance
  - Notch filter removing a tonal disturbance
  - Decimation with anti-aliasing
  - Integration: acceleration -> velocity -> displacement
  - Scalar metrics: RMS, peak, crest factor

Run:
    python examples/example_filters_utils.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from dspkit._testing import generate_2dof, natural_frequencies_2dof
from dspkit.spectral import psd
from dspkit.filters import bandpass, notch, decimate
from dspkit.utils import integrate, rms, peak, crest_factor
from dspkit.plots import plot_signal, plot_psd

# ---------------------------------------------------------------------------
# Simulate 2DOF system + tonal disturbance at 50 Hz (mains hum)
# ---------------------------------------------------------------------------

FS = 1000.0
DURATION = 30.0

fn1, fn2 = natural_frequencies_2dof()
print(f"Natural frequencies:  fn1 = {fn1:.2f} Hz,  fn2 = {fn2:.2f} Hz")

t, a1, _ = generate_2dof(duration=DURATION, fs=FS, noise_std=1.0,
                          output="acceleration", seed=0)

# Add 50 Hz mains hum on top
hum = 0.5 * np.sin(2 * np.pi * 50.0 * t)
a1_contaminated = a1 + hum

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

fig = plt.figure(figsize=(15, 12), constrained_layout=True)
fig.suptitle("Filtering & Utilities — 2DOF System", fontsize=14, fontweight="bold")

gs = gridspec.GridSpec(4, 2, figure=fig)

ax_raw       = fig.add_subplot(gs[0, :])
ax_bp_time   = fig.add_subplot(gs[1, 0])
ax_bp_psd    = fig.add_subplot(gs[1, 1])
ax_notch_psd = fig.add_subplot(gs[2, 0])
ax_integ     = fig.add_subplot(gs[2, 1])
ax_dec_psd   = fig.add_subplot(gs[3, 0])
ax_metrics   = fig.add_subplot(gs[3, 1])

SHOW = slice(0, int(3 * FS))   # first 3 s for time plots

# ---------------------------------------------------------------------------
# Raw signal (first 3 s)
# ---------------------------------------------------------------------------

plot_signal(t[SHOW], a1_contaminated[SHOW], ax=ax_raw,
            label="raw (with 50 Hz hum)",
            title="Raw signal (first 3 s)", ylabel="Acceleration [m/s²]")
ax_raw.plot(t[SHOW], a1[SHOW], lw=0.7, alpha=0.7, label="clean (no hum)")
ax_raw.legend()

# ---------------------------------------------------------------------------
# Bandpass around fn1 to isolate first mode
# ---------------------------------------------------------------------------

bp_low, bp_high = fn1 - 3.0, fn1 + 3.0
a1_bp = bandpass(a1_contaminated, FS, low=bp_low, high=bp_high, order=4)

plot_signal(t[SHOW], a1_bp[SHOW], ax=ax_bp_time,
            title=f"Bandpass [{bp_low:.1f}–{bp_high:.1f} Hz] — isolates mode 1",
            ylabel="Acceleration [m/s²]")

freqs, Pxx_raw = psd(a1_contaminated, FS, nperseg=2048)
freqs, Pxx_bp  = psd(a1_bp, FS, nperseg=2048)

plot_psd(freqs, Pxx_raw, ax=ax_bp_psd, db=False, xlim=(0, 60),
         title="PSD: raw vs bandpassed", ylabel="PSD [(m/s²)²/Hz]",
         label="raw", alpha=0.7)
ax_bp_psd.semilogy(freqs, Pxx_bp, lw=0.9, label=f"bandpass [{bp_low:.1f}–{bp_high:.1f} Hz]")
ax_bp_psd.axvline(fn1, color="red", ls="--", lw=1.2, label=f"fn1={fn1:.1f} Hz")
ax_bp_psd.legend(fontsize=8)

# ---------------------------------------------------------------------------
# Notch filter at 50 Hz to remove mains hum
# ---------------------------------------------------------------------------

a1_notched = notch(a1_contaminated, FS, freq=50.0, q=30.0)
freqs, Pxx_notched = psd(a1_notched, FS, nperseg=2048)

plot_psd(freqs, Pxx_raw, ax=ax_notch_psd, db=False, xlim=(0, 100),
         title="PSD: raw vs notch-filtered (50 Hz hum removed)",
         ylabel="PSD [(m/s²)²/Hz]", label="raw (with hum)", alpha=0.7)
ax_notch_psd.semilogy(freqs, Pxx_notched, lw=0.9, label="after notch @ 50 Hz")
ax_notch_psd.axvline(50,  color="orange", ls="--", lw=1.2, label="50 Hz notch")
ax_notch_psd.axvline(fn1, color="red",    ls="--", lw=1.0, alpha=0.7)
ax_notch_psd.axvline(fn2, color="green",  ls="--", lw=1.0, alpha=0.7)
ax_notch_psd.legend(fontsize=8)

# ---------------------------------------------------------------------------
# Integration: acceleration -> velocity -> displacement
# ---------------------------------------------------------------------------

vel  = integrate(a1_notched, FS, detrend_after=True)
disp = integrate(vel, FS, detrend_after=True)

plot_signal(t[SHOW], a1_notched[SHOW] / np.std(a1_notched), ax=ax_integ,
            label="accel (normalised)",
            title="Integration: acceleration → velocity → displacement",
            ylabel="Normalised amplitude")
ax_integ.plot(t[SHOW], vel[SHOW]  / np.std(vel),  lw=0.8, label="velocity (normalised)")
ax_integ.plot(t[SHOW], disp[SHOW] / np.std(disp), lw=0.8, label="displacement (normalised)")
ax_integ.legend(fontsize=8)

# ---------------------------------------------------------------------------
# Decimation: 1000 Hz -> 100 Hz
# ---------------------------------------------------------------------------

a1_dec, fs_dec = decimate(a1_notched, FS, target_fs=100.0)
freqs_dec, Pxx_dec = psd(a1_dec, fs_dec, nperseg=256)

plot_psd(freqs, Pxx_notched, ax=ax_dec_psd, db=False, xlim=(0, 50),
         title=f"Decimation {FS:.0f} → {fs_dec:.0f} Hz (new Nyquist = {fs_dec/2:.0f} Hz)",
         ylabel="PSD [(m/s²)²/Hz]", label=f"original ({FS:.0f} Hz)", alpha=0.8)
ax_dec_psd.semilogy(freqs_dec, Pxx_dec, lw=0.9, label=f"decimated ({fs_dec:.0f} Hz)")
ax_dec_psd.axvline(fn1, color="red",   ls="--", lw=1.2, label=f"fn1={fn1:.1f} Hz")
ax_dec_psd.axvline(fn2, color="green", ls="--", lw=1.2, label=f"fn2={fn2:.1f} Hz")
ax_dec_psd.legend(fontsize=8)

# ---------------------------------------------------------------------------
# Scalar metrics bar chart
# ---------------------------------------------------------------------------

signals = {
    "raw":      a1_contaminated,
    "notched":  a1_notched,
    "bandpass": a1_bp,
    "velocity": vel,
    "displ.":   disp,
}
names = list(signals.keys())
rms_vals  = [rms(s)          for s in signals.values()]
peak_vals = [peak(s)         for s in signals.values()]
cf_vals   = [crest_factor(s) for s in signals.values()]

x_pos = np.arange(len(names))
width = 0.28

ax_metrics.bar(x_pos - width, rms_vals,  width, label="RMS")
ax_metrics.bar(x_pos,         peak_vals, width, label="Peak")
ax_metrics.bar(x_pos + width, cf_vals,   width, label="Crest factor")
ax_metrics.set_yscale("log")
ax_metrics.set_xticks(x_pos)
ax_metrics.set_xticklabels(names, fontsize=9)
ax_metrics.set_ylabel("Value (log scale)")
ax_metrics.set_title("Signal metrics: RMS, Peak, Crest factor")
ax_metrics.legend(fontsize=8)
ax_metrics.grid(True, axis="y", alpha=0.3)

# Print metrics table to terminal
print("\nSignal metrics:")
print(f"{'Signal':<12} {'RMS':>10} {'Peak':>10} {'Crest':>8}")
print("-" * 44)
for name, r, p, cf in zip(names, rms_vals, peak_vals, cf_vals):
    print(f"{name:<12} {r:>10.4f} {p:>10.4f} {cf:>8.3f}")

plt.show()
