"""
Example: time-frequency analysis of a chirp + 2DOF structural response.

Two signals are analysed side by side:
  1. Linear chirp (f = 10 → 150 Hz over 1 s) — ideal for comparing TF methods,
     since the ground truth is a straight line in the TF plane.
  2. Short segment of the 2DOF acceleration response — shows how resonances
     appear as horizontal ridges in a real structural signal.

Methods compared: STFT, CWT scalogram, Wigner-Ville, Smoothed Pseudo WVD.

Run:
    python examples/example_timefreq.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp

from dspkit._testing import generate_2dof
from dspkit.timefreq import stft, cwt_scalogram, wigner_ville, smoothed_pseudo_wv
from dspkit.plots import plot_spectrogram, plot_scalogram, plot_wvd

# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

FS = 1000.0

# --- Chirp (1 s) ---
t_chirp = np.arange(int(FS)) / FS
f0_chirp, f1_chirp = 10.0, 150.0
x_chirp = chirp(t_chirp, f0=f0_chirp, f1=f1_chirp, t1=1.0, method="linear")

# --- 2DOF response: take a 1 s segment from the middle ---
t_long, a1_long, _ = generate_2dof(duration=10.0, fs=FS, noise_std=0.5,
                                    output="acceleration", seed=7)
seg = slice(int(4.5 * FS), int(5.5 * FS))
t_2dof = t_long[seg] - t_long[seg][0]
x_2dof = a1_long[seg]

# ---------------------------------------------------------------------------
# Compute TF representations
# ---------------------------------------------------------------------------

print("Computing STFT ...", end=" ", flush=True)
f_stft,  t_stft,  Z_chirp = stft(x_chirp, FS, nperseg=128, noverlap=120)
f_stft2, t_stft2, Z_2dof  = stft(x_2dof,  FS, nperseg=128, noverlap=120)
print("done")

print("Computing CWT  ...", end=" ", flush=True)
analysis_freqs = np.geomspace(2.0, FS / 4, num=80)
f_cwt,  t_cwt,  W_chirp = cwt_scalogram(x_chirp, FS, freqs=analysis_freqs, w=6.0)
f_cwt2, t_cwt2, W_2dof  = cwt_scalogram(x_2dof,  FS, freqs=analysis_freqs, w=6.0)
print("done")

print("Computing WVD  ...", end=" ", flush=True)
f_wvd,  t_wvd,  WVD_chirp = wigner_ville(x_chirp, FS)
f_wvd2, t_wvd2, WVD_2dof  = wigner_ville(x_2dof,  FS)
print("done")

print("Computing SPWVD...", end=" ", flush=True)
N = len(x_chirp)
f_spwvd,  t_spwvd,  SPWVD_chirp = smoothed_pseudo_wv(x_chirp, FS,
                                                       lag_samples=N // 6,
                                                       time_samples=N // 16)
f_spwvd2, t_spwvd2, SPWVD_2dof  = smoothed_pseudo_wv(x_2dof,  FS,
                                                       lag_samples=N // 6,
                                                       time_samples=N // 16)
print("done")

# ---------------------------------------------------------------------------
# Figure: 4 rows × 2 columns (method × signal)
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(4, 2, figsize=(14, 12), constrained_layout=True)
fig.suptitle("Time-Frequency Analysis Comparison", fontsize=13, fontweight="bold")

col_labels = [
    f"Chirp  {f0_chirp:.0f} → {f1_chirp:.0f} Hz",
    "2DOF structural response",
]

# ---- Row 0: STFT ----
plot_spectrogram(f_stft,  t_stft,  Z_chirp, ax=axes[0, 0], ylim=(0, 200),
                 title=f"STFT — {col_labels[0]}")
plot_spectrogram(f_stft2, t_stft2, Z_2dof,  ax=axes[0, 1], ylim=(0, 200),
                 title=f"STFT — {col_labels[1]}")

# ---- Row 1: CWT scalogram (log frequency axis) ----
plot_scalogram(f_cwt,  t_cwt,  W_chirp, ax=axes[1, 0], log_freq=True, ylim=(2, 200),
               title=f"CWT (Morlet) — {col_labels[0]}")
plot_scalogram(f_cwt2, t_cwt2, W_2dof,  ax=axes[1, 1], log_freq=True, ylim=(2, 200),
               title=f"CWT (Morlet) — {col_labels[1]}")
for ax in axes[1]:
    ax.set_ylabel("Frequency [Hz] (log)")

# ---- Row 2: WVD ----
plot_wvd(f_wvd,  t_wvd,  WVD_chirp, ax=axes[2, 0], ylim=(0, 200),
         title=f"WVD — {col_labels[0]}")
plot_wvd(f_wvd2, t_wvd2, WVD_2dof,  ax=axes[2, 1], ylim=(0, 200),
         title=f"WVD — {col_labels[1]}")

# ---- Row 3: SPWVD ----
plot_wvd(f_spwvd,  t_spwvd,  SPWVD_chirp, ax=axes[3, 0], ylim=(0, 200),
         title=f"SPWVD — {col_labels[0]}")
plot_wvd(f_spwvd2, t_spwvd2, SPWVD_2dof,  ax=axes[3, 1], ylim=(0, 200),
         title=f"SPWVD — {col_labels[1]}")

# Add row labels on the left
for ax, label in zip(axes[:, 0], ["STFT", "CWT (Morlet)", "WVD", "SPWVD"]):
    ax.set_ylabel(f"[{label}]\nFrequency [Hz]")

plt.show()
