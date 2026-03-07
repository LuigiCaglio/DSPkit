"""
Example: Hilbert transform & instantaneous signal attributes.

Three signals are analysed:
  1. Amplitude-modulated (AM) sine — envelope should recover the modulation.
  2. Linear chirp — instantaneous frequency should track the sweep.
  3. 2DOF ring-down (bandpass filtered around mode 1) — envelope reveals
     the exponential decay; instantaneous frequency should be ~fn1.

Run:
    python examples/example_instantaneous.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import chirp

from dspkit._testing import generate_2dof, natural_frequencies_2dof
from dspkit.filters import bandpass
from dspkit.instantaneous import hilbert_attributes
from dspkit.plots import plot_signal

FS = 1000.0

# ---------------------------------------------------------------------------
# Signal 1: Amplitude-modulated sine
# ---------------------------------------------------------------------------
DURATION_AM = 2.0
t_am = np.arange(int(FS * DURATION_AM)) / FS
f_carrier = 100.0
f_mod = 3.0
A_mod = 1.0 + 0.6 * np.sin(2.0 * np.pi * f_mod * t_am)  # 0.4 to 1.6
x_am = A_mod * np.sin(2.0 * np.pi * f_carrier * t_am)
env_am, phase_am, fi_am = hilbert_attributes(x_am, FS)

# ---------------------------------------------------------------------------
# Signal 2: Linear chirp 10 → 200 Hz
# ---------------------------------------------------------------------------
DURATION_CH = 2.0
t_ch = np.arange(int(FS * DURATION_CH)) / FS
x_ch = chirp(t_ch, f0=10.0, f1=200.0, t1=DURATION_CH, method="linear")
env_ch, phase_ch, fi_ch = hilbert_attributes(x_ch, FS)
fi_ch_theory = 10.0 + (200.0 - 10.0) / DURATION_CH * t_ch  # ground truth

# ---------------------------------------------------------------------------
# Signal 3: 2DOF ring-down, mode 1 isolated
# ---------------------------------------------------------------------------
fn1, fn2 = natural_frequencies_2dof()
t_long, a1_long, _ = generate_2dof(duration=30.0, fs=FS, noise_std=0.3,
                                    output="acceleration", seed=42)
x_mode1 = bandpass(a1_long, FS, low=fn1 - 2.0, high=fn1 + 2.0, order=6)
env_m1, phase_m1, fi_m1 = hilbert_attributes(x_mode1, FS)
seg = slice(int(3 * FS), int(8 * FS))
t_m1 = t_long[seg] - t_long[seg][0]

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(14, 10), constrained_layout=True)
fig.suptitle("Hilbert Transform — Instantaneous Signal Attributes", fontsize=13, fontweight="bold")

gs = gridspec.GridSpec(3, 3, figure=fig)

# ---- Row 0: AM signal ----
ax00 = fig.add_subplot(gs[0, 0])
plot_signal(t_am, x_am, ax=ax00, label="signal", envelope=env_am,
            title="AM signal: envelope vs true modulation",
            ylabel="Amplitude")
ax00.plot(t_am, A_mod,  lw=1.5, color="green", ls="--", label="true A(t)")
ax00.plot(t_am, -A_mod, lw=1.5, color="green", ls="--")
ax00.legend(fontsize=8)

ax01 = fig.add_subplot(gs[0, 1])
ax01.plot(t_am, fi_am, lw=0.8, color="purple")
ax01.axhline(f_carrier, color="red", ls="--", lw=1.2, label=f"carrier = {f_carrier} Hz")
ax01.set_ylim(80, 120)
ax01.set_title("AM signal: instantaneous frequency")
ax01.set_xlabel("Time [s]"); ax01.set_ylabel("Frequency [Hz]")
ax01.legend(fontsize=8); ax01.grid(True, alpha=0.3)

ax02 = fig.add_subplot(gs[0, 2])
ax02.plot(t_am, phase_am, lw=0.8, color="darkorange")
ax02.set_title("AM signal: instantaneous phase (unwrapped)")
ax02.set_xlabel("Time [s]"); ax02.set_ylabel("Phase [rad]")
ax02.grid(True, alpha=0.3)

# ---- Row 1: Chirp ----
ax10 = fig.add_subplot(gs[1, 0])
plot_signal(t_ch, x_ch, ax=ax10, label="chirp", envelope=env_ch,
            title="Chirp 10→200 Hz: signal & envelope",
            ylabel="Amplitude")

ax11 = fig.add_subplot(gs[1, 1])
ax11.plot(t_ch, fi_ch,        lw=0.9, color="purple", label="instantaneous freq")
ax11.plot(t_ch, fi_ch_theory, lw=1.5, color="red", ls="--", label="theoretical")
ax11.set_title("Chirp: instantaneous frequency vs theory")
ax11.set_xlabel("Time [s]"); ax11.set_ylabel("Frequency [Hz]")
ax11.legend(fontsize=8); ax11.grid(True, alpha=0.3)

ax12 = fig.add_subplot(gs[1, 2])
ax12.plot(t_ch, phase_ch, lw=0.8, color="darkorange")
ax12.set_title("Chirp: instantaneous phase (quadratic)")
ax12.set_xlabel("Time [s]"); ax12.set_ylabel("Phase [rad]")
ax12.grid(True, alpha=0.3)

# ---- Row 2: 2DOF ring-down mode 1 ----
ax20 = fig.add_subplot(gs[2, 0])
plot_signal(t_m1, x_mode1[seg], ax=ax20, label="mode 1 (filtered)",
            envelope=env_m1[seg],
            title=f"2DOF mode 1 (fn1={fn1:.1f} Hz): envelope",
            ylabel="Acceleration [m/s²]")

ax21 = fig.add_subplot(gs[2, 1])
ax21.plot(t_m1, fi_m1[seg], lw=0.8, color="purple")
ax21.axhline(fn1, color="red", ls="--", lw=1.2, label=f"fn1 = {fn1:.1f} Hz")
ax21.set_ylim(fn1 - 5, fn1 + 5)
ax21.set_title("2DOF mode 1: instantaneous frequency")
ax21.set_xlabel("Time [s]"); ax21.set_ylabel("Frequency [Hz]")
ax21.legend(fontsize=8); ax21.grid(True, alpha=0.3)

ax22 = fig.add_subplot(gs[2, 2])
env_log = np.log(np.maximum(env_m1[seg], 1e-12))
slope, intercept = np.polyfit(t_m1, env_log, 1)
zeta_est = -slope / (2.0 * np.pi * fn1)
ax22.plot(t_m1, env_log, lw=0.9, color="steelblue", label="log(envelope)")
ax22.plot(t_m1, slope * t_m1 + intercept, lw=1.5, color="red", ls="--",
          label=f"fit  ζ ≈ {zeta_est:.4f}")
ax22.set_title("Log envelope → damping ratio estimate")
ax22.set_xlabel("Time [s]"); ax22.set_ylabel("ln(envelope)")
ax22.legend(fontsize=8); ax22.grid(True, alpha=0.3)

print(f"Damping ratio estimate for mode 1: ζ ≈ {zeta_est:.4f}")

plt.show()
