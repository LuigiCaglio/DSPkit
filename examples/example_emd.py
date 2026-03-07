"""
Example: EMD and Hilbert-Huang Transform on a 2DOF structural response.

Demonstrates:
  1. EMD decomposition of a multi-component signal into IMFs.
  2. Instantaneous amplitude and frequency of each IMF via HHT.
  3. HHT marginal spectrum compared against the Welch PSD.
  4. Damping estimation from the log-envelope of each modal IMF.

The signal is a 2DOF acceleration response: two resonances at fn1 and fn2
plus broadband noise — a realistic structural health monitoring scenario.

Run:
    python examples/example_emd.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from dspkit._testing import generate_2dof, natural_frequencies_2dof
from dspkit.spectral import psd
from dspkit.emd import emd, hht, hht_marginal_spectrum
from dspkit.plots import plot_signal, plot_psd

# ---------------------------------------------------------------------------
# Simulate 2DOF response (use moderate duration for speed)
# ---------------------------------------------------------------------------

FS = 500.0          # Hz — decimate to keep EMD tractable
DURATION = 20.0

fn1, fn2 = natural_frequencies_2dof()
print(f"Theoretical natural frequencies:  fn1 = {fn1:.2f} Hz,  fn2 = {fn2:.2f} Hz")

t, a1, _ = generate_2dof(duration=DURATION, fs=FS, noise_std=0.5,
                          output="acceleration", seed=0)
N = len(t)

# ---------------------------------------------------------------------------
# EMD
# ---------------------------------------------------------------------------
print("Running EMD ...", end=" ", flush=True)
imfs, residue = emd(a1, max_imfs=8, max_sifting=10, sd_threshold=0.2)
n_imfs = imfs.shape[0]
print(f"done — {n_imfs} IMFs extracted")

# HHT
envs, inst_freqs = hht(imfs, FS)

# Marginal spectrum
freq_bins, marginal = hht_marginal_spectrum(envs, inst_freqs, FS, n_bins=512)

# Welch PSD for comparison
freqs_psd, Pxx = psd(a1, FS, nperseg=1024)

# ---------------------------------------------------------------------------
# Figure 1: IMFs waterfall
# ---------------------------------------------------------------------------

fig1, axes = plt.subplots(n_imfs + 2, 1, figsize=(13, 2.2 * (n_imfs + 2)),
                          constrained_layout=True)
fig1.suptitle("EMD — Intrinsic Mode Functions", fontsize=12, fontweight="bold")

show = slice(0, int(5 * FS))  # first 5 s

plot_signal(t[show], a1[show], ax=axes[0],
            title="Input signal (first 5 s)", ylabel="Original")

for i, (imf, env) in enumerate(zip(imfs, envs)):
    ax = axes[i + 1]
    sl = slice(N // 10, 9 * N // 10)
    fi_mean = inst_freqs[i, sl].mean()
    plot_signal(t[show], imf[show], ax=ax, envelope=env[show],
                title=f"IMF {i+1}   (mean fi = {fi_mean:.1f} Hz)",
                ylabel=f"IMF {i+1}")

axes[-1].plot(t[show], residue[show], lw=0.8, color="saddlebrown")
axes[-1].set_ylabel("Residue")
axes[-1].set_title("Residue (trend)")
axes[-1].set_xlabel("Time [s]")
axes[-1].grid(True, alpha=0.3)

# ---------------------------------------------------------------------------
# Figure 2: HHT spectrum, marginal spectrum, damping
# ---------------------------------------------------------------------------

fig2 = plt.figure(figsize=(14, 9), constrained_layout=True)
fig2.suptitle("HHT Analysis", fontsize=12, fontweight="bold")
gs = gridspec.GridSpec(2, 2, figure=fig2)

# ---- HHT time-frequency scatter (instantaneous freq coloured by energy) ----
ax_tf = fig2.add_subplot(gs[0, :])
cmap = plt.cm.inferno
sl_tf = slice(N // 20, 19 * N // 20)  # trim edges

for i, (env, fi) in enumerate(zip(envs, inst_freqs)):
    sc = ax_tf.scatter(
        t[sl_tf], fi[sl_tf],
        c=env[sl_tf] ** 2,
        s=1, alpha=0.4, cmap=cmap,
        vmin=0, vmax=np.percentile(envs ** 2, 95),
    )

ax_tf.axhline(fn1, color="cyan",  ls="--", lw=1.2, label=f"fn1={fn1:.1f} Hz")
ax_tf.axhline(fn2, color="lime",  ls="--", lw=1.2, label=f"fn2={fn2:.1f} Hz")
ax_tf.set_ylim(0, FS / 2)
ax_tf.set_xlabel("Time [s]")
ax_tf.set_ylabel("Instantaneous frequency [Hz]")
ax_tf.set_title("HHT time-frequency representation (colour = instantaneous energy)")
ax_tf.legend(fontsize=9)
fig2.colorbar(sc, ax=ax_tf, label="A²(t)")

# ---- Marginal spectrum vs Welch PSD ----
ax_spec = fig2.add_subplot(gs[1, 0])
plot_psd(freq_bins, marginal, ax=ax_spec, db=False, xlim=(0, 60),
         title="Marginal HHT spectrum vs Welch PSD",
         ylabel="Amplitude² / Hz equivalent",
         label="HHT marginal spectrum")
ax_spec.semilogy(freqs_psd, Pxx, lw=1.0, alpha=0.8, label="Welch PSD")
ax_spec.axvline(fn1, color="red",    ls="--", lw=1.2, label=f"fn1={fn1:.1f} Hz")
ax_spec.axvline(fn2, color="orange", ls="--", lw=1.2, label=f"fn2={fn2:.1f} Hz")
ax_spec.legend(fontsize=8)

# ---- Damping estimation from log-envelope of each IMF ----
ax_damp = fig2.add_subplot(gs[1, 1])
sl_d = slice(N // 10, 9 * N // 10)
t_d = t[sl_d]

for i, (env, fi) in enumerate(zip(envs, inst_freqs)):
    fi_mean = fi[sl_d].mean()
    if fi_mean < 1.0 or fi_mean > FS / 4:
        continue  # skip noise/trend IMFs
    log_env = np.log(np.maximum(env[sl_d], 1e-12))
    slope, intercept = np.polyfit(t_d, log_env, 1)
    zeta = -slope / (2.0 * np.pi * fi_mean)
    if 0 < zeta < 0.5:  # only plausible damping values
        ax_damp.plot(t_d, log_env, lw=0.8, label=f"IMF {i+1}  fi={fi_mean:.1f} Hz  ζ={zeta:.4f}")
        ax_damp.plot(t_d, slope * t_d + intercept, lw=1.5, ls="--", color="gray")
        print(f"IMF {i+1}: mean fi = {fi_mean:.2f} Hz,  ζ ≈ {zeta:.4f}")

ax_damp.set_xlabel("Time [s]")
ax_damp.set_ylabel("ln(envelope)")
ax_damp.set_title("Log-envelope → damping ratio per IMF\n(slope = −ζ ωn)")
ax_damp.legend(fontsize=7)
ax_damp.grid(True, alpha=0.3)

plt.show()
