"""
Example: multi-sensor analysis and probability statistics.

Demonstrates:
- Correlation matrix across channels
- Coherence matrix
- PSD matrix (cross-spectral density)
- PDF estimation (KDE and histogram)
- Joint distributions
- Covariance matrix and Mahalanobis distance

Run:
    python examples/example_multisensor_stats.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from dspkit._testing import generate_2dof, natural_frequencies_2dof
from dspkit.multisensor import correlation_matrix, coherence_matrix, psd_matrix
from dspkit.statistics import (
    pdf_estimate, histogram, joint_histogram,
    covariance_matrix, mahalanobis,
)
from dspkit.plots import (
    plot_correlation_matrix, plot_pdf, plot_joint_histogram,
)

# ---------------------------------------------------------------------------
# Simulation — 2DOF with 2 sensors
# ---------------------------------------------------------------------------

FS = 1000.0
DURATION = 60.0
fn1, fn2 = natural_frequencies_2dof()

t, a1, a2 = generate_2dof(duration=DURATION, fs=FS, noise_std=1.0,
                            output="acceleration", seed=42)
data = np.vstack([a1, a2])  # shape (2, N)

# ===========================================================================
# Figure 1: Multi-sensor Analysis
# ===========================================================================

fig = plt.figure(figsize=(14, 10), constrained_layout=True)
fig.suptitle("Multi-Sensor Analysis — 2DOF System", fontsize=14, fontweight="bold")
gs = gridspec.GridSpec(2, 2, figure=fig)

# --- Correlation matrix ---
R = correlation_matrix(data)
ax = fig.add_subplot(gs[0, 0])
plot_correlation_matrix(R, ax=ax, labels=["Mass 1", "Mass 2"],
                        title="Correlation Matrix")

# --- Coherence matrix at selected frequencies ---
freqs, C = coherence_matrix(data, FS, nperseg=4096)
ax = fig.add_subplot(gs[0, 1])
ax.plot(freqs, C[0, 1, :], lw=0.9, color="steelblue")
ax.axhline(0.8, color="gray", ls=":", lw=1, label="threshold = 0.8")
ax.axvline(fn1, color="red", ls="--", lw=1, label=f"fn1 = {fn1:.1f} Hz")
ax.axvline(fn2, color="orange", ls="--", lw=1, label=f"fn2 = {fn2:.1f} Hz")
ax.set_xlim(0, 80)
ax.set_ylim(0, 1.05)
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Coherence [-]")
ax.set_title("Coherence: Mass 1 vs Mass 2")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- PSD matrix: singular values ---
freqs_g, G = psd_matrix(data, FS, nperseg=4096)
# SVD at each frequency to show singular values
sv1 = np.zeros(len(freqs_g))
sv2 = np.zeros(len(freqs_g))
for k in range(len(freqs_g)):
    _, s, _ = np.linalg.svd(G[:, :, k])
    sv1[k] = s[0]
    sv2[k] = s[1] if len(s) > 1 else 0

ax = fig.add_subplot(gs[1, 0])
ax.semilogy(freqs_g, sv1, lw=0.9, label="SV1")
ax.semilogy(freqs_g, sv2, lw=0.9, label="SV2")
ax.set_xlim(0, 80)
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Singular Value")
ax.set_title("PSD Matrix — Singular Values")
ax.axvline(fn1, color="red", ls="--", lw=1, label=f"fn1 = {fn1:.1f} Hz")
ax.axvline(fn2, color="orange", ls="--", lw=1, label=f"fn2 = {fn2:.1f} Hz")
ax.legend(fontsize=8)
ax.grid(True, which="both", alpha=0.3)

# --- Covariance matrix ---
cov = covariance_matrix(data)
ax = fig.add_subplot(gs[1, 1])
im = ax.imshow(cov, cmap="coolwarm", aspect="equal")
plt.colorbar(im, ax=ax, label="Covariance")
ax.set_xticks([0, 1])
ax.set_xticklabels(["Mass 1", "Mass 2"])
ax.set_yticks([0, 1])
ax.set_yticklabels(["Mass 1", "Mass 2"])
ax.set_title("Covariance Matrix")
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cov[i, j]:.3f}", ha="center", va="center", fontsize=11)

plt.show()


# ===========================================================================
# Figure 2: Probability & Joint Statistics
# ===========================================================================

fig2 = plt.figure(figsize=(14, 10), constrained_layout=True)
fig2.suptitle("Probability Distributions & Joint Statistics",
              fontsize=14, fontweight="bold")
gs2 = gridspec.GridSpec(2, 2, figure=fig2)

# --- PDF of mass 1 ---
xi1, density1 = pdf_estimate(a1)
ax = fig2.add_subplot(gs2[0, 0])
plot_pdf(xi1, density1, ax=ax, hist_data=a1, hist_bins=80,
         title="PDF — Mass 1 Acceleration", xlabel="Acceleration [m/s²]")

# --- PDF of mass 2 ---
xi2, density2 = pdf_estimate(a2)
ax = fig2.add_subplot(gs2[0, 1])
plot_pdf(xi2, density2, ax=ax, hist_data=a2, hist_bins=80,
         title="PDF — Mass 2 Acceleration", xlabel="Acceleration [m/s²]")

# --- Joint distribution ---
# Downsample for visual clarity
step = 10
xc, yc, H = joint_histogram(a1[::step], a2[::step], bins=60)
ax = fig2.add_subplot(gs2[1, 0])
plot_joint_histogram(xc, yc, H, ax=ax,
                     xlabel="Mass 1 [m/s²]", ylabel="Mass 2 [m/s²]",
                     title="Joint Distribution (Mass 1 vs Mass 2)")

# --- Mahalanobis distance ---
distances = mahalanobis(data[:, ::step])
t_sub = t[::step]
ax = fig2.add_subplot(gs2[1, 1])
sc = ax.scatter(t_sub, distances, c=distances, s=1, cmap="hot_r", alpha=0.5)
plt.colorbar(sc, ax=ax, label="Distance")
threshold = np.percentile(distances, 99)
ax.axhline(threshold, color="red", ls="--", lw=1.2,
           label=f"99th percentile = {threshold:.1f}")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Mahalanobis Distance")
ax.set_title("Mahalanobis Distance (outlier detection)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.show()
