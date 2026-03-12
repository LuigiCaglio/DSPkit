"""
Example: Frequency Domain Decomposition (FDD) on a 2DOF structural system.

Demonstrates the full FDD workflow:
1. Compute singular values of the PSD matrix
2. Pick natural frequency peaks
3. Extract mode shapes
4. Enhanced FDD damping estimation

Run:
    python examples/example_fdd.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from dspkit._testing import generate_2dof, natural_frequencies_2dof
from dspkit.fdd import fdd_svd, fdd_peak_picking, fdd_mode_shapes, efdd_damping
from dspkit.plots import plot_singular_values, plot_mode_shape

# ---------------------------------------------------------------------------
# Simulation — 2DOF system with 2 sensors
# ---------------------------------------------------------------------------

FS = 1000.0
DURATION = 120.0
fn1, fn2 = natural_frequencies_2dof()

print(f"Theoretical natural frequencies: fn1 = {fn1:.2f} Hz, fn2 = {fn2:.2f} Hz")

t, a1, a2 = generate_2dof(duration=DURATION, fs=FS, noise_std=1.0,
                            output="acceleration", seed=42)
data = np.vstack([a1, a2])  # shape (2, N)

# ---------------------------------------------------------------------------
# Step 1: SVD of the PSD matrix
# ---------------------------------------------------------------------------

freqs, S, U = fdd_svd(data, FS, nperseg=4096)

# ---------------------------------------------------------------------------
# Step 2: Peak picking on first singular value
# ---------------------------------------------------------------------------

peak_freqs, peak_idx = fdd_peak_picking(
    freqs, S,
    distance_hz=5.0,
    max_peaks=2,
)

print(f"\nDetected natural frequencies:")
for i, f in enumerate(peak_freqs):
    print(f"  Mode {i + 1}: {f:.2f} Hz")

# ---------------------------------------------------------------------------
# Step 3: Mode shapes
# ---------------------------------------------------------------------------

modes = fdd_mode_shapes(U, peak_idx, normalize=True)

print(f"\nMode shapes (normalised):")
for i, (f, mode) in enumerate(zip(peak_freqs, modes)):
    print(f"  Mode {i + 1} ({f:.2f} Hz): {mode.real}")

# ---------------------------------------------------------------------------
# Step 4: Enhanced FDD damping
# ---------------------------------------------------------------------------

zeta, fn_refined = efdd_damping(freqs, S, U, peak_idx, FS,
                                 mac_threshold=0.8, n_crossings=20)

print(f"\nDamping estimation (EFDD):")
for i, (f, z) in enumerate(zip(fn_refined, zeta)):
    if np.isnan(z):
        print(f"  Mode {i + 1}: fn = {f:.2f} Hz, damping = N/A")
    else:
        print(f"  Mode {i + 1}: fn = {f:.2f} Hz, zeta = {z:.4f} ({z*100:.2f}%)")

# ===========================================================================
# Figure: FDD results
# ===========================================================================

fig = plt.figure(figsize=(14, 10), constrained_layout=True)
fig.suptitle("Frequency Domain Decomposition (FDD) — 2DOF System",
             fontsize=14, fontweight="bold")
gs = gridspec.GridSpec(2, 2, figure=fig)

# --- Singular values ---
ax = fig.add_subplot(gs[0, :])
plot_singular_values(freqs, S, ax=ax, db=True, peak_freqs=peak_freqs,
                     xlim=(0, 80), title="Singular Values of PSD Matrix")
ax.axvline(fn1, color="cyan", ls=":", lw=1, alpha=0.7, label=f"fn1 = {fn1:.1f} Hz (theory)")
ax.axvline(fn2, color="lime", ls=":", lw=1, alpha=0.7, label=f"fn2 = {fn2:.1f} Hz (theory)")
ax.legend(fontsize=8)

# --- Mode shapes ---
for i, (f, mode) in enumerate(zip(peak_freqs, modes)):
    ax = fig.add_subplot(gs[1, i])
    plot_mode_shape(mode, ax=ax,
                    sensor_labels=["Mass 1", "Mass 2"],
                    title=f"Mode {i + 1} — {f:.2f} Hz")
    if not np.isnan(zeta[i]):
        ax.text(0.95, 0.95, f"zeta = {zeta[i]:.4f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round", fc="wheat", alpha=0.8))

plt.show()
