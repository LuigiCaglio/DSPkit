"""
Generate documentation images from DSPkit examples.

Run from the project root:
    python docs/gen_images.py

Saves PNGs to docs/images/.  Uses the non-interactive Agg backend so it
works on headless CI machines — no display required.
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.signal import chirp

OUT = Path(__file__).parent / "images"
OUT.mkdir(exist_ok=True)

DPI = 150


def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")


# ---------------------------------------------------------------------------
# Imports (after backend is set)
# ---------------------------------------------------------------------------
from dspkit._testing import generate_2dof, natural_frequencies_2dof
from dspkit.spectral import fft_spectrum, psd, csd, coherence, autocorrelation, cross_correlation
from dspkit.filters import bandpass, notch, decimate
from dspkit.utils import integrate, rms, peak, crest_factor
from dspkit.timefreq import stft, cwt_scalogram, wigner_ville, smoothed_pseudo_wv
from dspkit.instantaneous import hilbert_attributes
from dspkit.emd import emd, hht, hht_marginal_spectrum
from dspkit.plots import (
    plot_signal, plot_fft, plot_psd, plot_csd,
    plot_coherence, plot_autocorrelation, plot_cross_correlation,
    plot_spectrogram, plot_scalogram, plot_wvd,
)

fn1, fn2 = natural_frequencies_2dof()

# ===========================================================================
# spectral.png
# ===========================================================================
print("spectral.png ...")

FS = 1000.0
NPERSEG = 4096
t, a1, a2 = generate_2dof(duration=60.0, fs=FS, noise_std=1.0,
                           output="acceleration", seed=42)

fig = plt.figure(figsize=(14, 14), constrained_layout=True)
fig.suptitle("2DOF System — Spectral Analysis", fontsize=14, fontweight="bold")
gs = gridspec.GridSpec(4, 2, figure=fig)

ax_time = fig.add_subplot(gs[0, :])
ax_fft  = fig.add_subplot(gs[1, 0])
ax_psd  = fig.add_subplot(gs[1, 1])
ax_csd  = fig.add_subplot(gs[2, 0])
ax_coh  = fig.add_subplot(gs[2, 1])
ax_acf  = fig.add_subplot(gs[3, 0])
ax_ccf  = fig.add_subplot(gs[3, 1])

mask = t <= 2.0
plot_signal(t[mask], a1[mask], ax=ax_time, label="mass 1",
            title="Time signal (first 2 s)", ylabel="Acceleration [m/s²]")
ax_time.plot(t[mask], a2[mask], label="mass 2", lw=0.8, alpha=0.8)
ax_time.legend()

freqs_fft, amp = fft_spectrum(a1, FS, window="hann", scaling="amplitude")
plot_fft(freqs_fft, amp, ax=ax_fft, xlim=(0, 80),
         title="FFT spectrum — mass 1", ylabel="Amplitude [m/s²]")
ax_fft.axvline(fn1, color="red",    ls="--", lw=1.2, label=f"fn1 = {fn1:.1f} Hz")
ax_fft.axvline(fn2, color="orange", ls="--", lw=1.2, label=f"fn2 = {fn2:.1f} Hz")
ax_fft.legend(fontsize=8)

freqs_psd, Pxx1 = psd(a1, FS, nperseg=NPERSEG)
freqs_psd, Pxx2 = psd(a2, FS, nperseg=NPERSEG)
plot_psd(freqs_psd, Pxx1, ax=ax_psd, db=False, xlim=(0, 80),
         title=f"Welch PSD  (nperseg={NPERSEG})", ylabel="PSD [(m/s²)²/Hz]", label="mass 1")
ax_psd.semilogy(freqs_psd, Pxx2, label="mass 2", lw=0.9, alpha=0.85)
ax_psd.axvline(fn1, color="red",    ls="--", lw=1.2, label=f"fn1 = {fn1:.1f} Hz")
ax_psd.axvline(fn2, color="orange", ls="--", lw=1.2, label=f"fn2 = {fn2:.1f} Hz")
ax_psd.legend(fontsize=8)

freqs_csd, Pxy = csd(a1, a2, FS, nperseg=NPERSEG)
plot_csd(freqs_csd, Pxy, ax=ax_csd, db=False, xlim=(0, 80),
         title="Cross-spectral density", ylabel="|CSD| [(m/s²)²/Hz]")
ax_csd.axvline(fn1, color="red",    ls="--", lw=1.2, label=f"fn1 = {fn1:.1f} Hz")
ax_csd.axvline(fn2, color="orange", ls="--", lw=1.2, label=f"fn2 = {fn2:.1f} Hz")
ax_csd.legend(fontsize=8)

freqs_coh, Cxy = coherence(a1, a2, FS, nperseg=NPERSEG)
plot_coherence(freqs_coh, Cxy, ax=ax_coh, threshold=0.8, xlim=(0, 80),
               title="Coherence (mass 1 vs mass 2)")
ax_coh.axvline(fn1, color="red",    ls="--", lw=1.2, label=f"fn1 = {fn1:.1f} Hz")
ax_coh.axvline(fn2, color="orange", ls="--", lw=1.2, label=f"fn2 = {fn2:.1f} Hz")
ax_coh.legend(fontsize=8)

lags_acf, acf = autocorrelation(a1, fs=FS, normalize=True, max_lag=0.5)
plot_autocorrelation(lags_acf, acf, ax=ax_acf, n_samples=len(a1),
                     xlabel="Lag [s]", title="Autocorrelation — mass 1")

lags_ccf, ccf = cross_correlation(a1, a2, fs=FS, normalize=True, max_lag=0.2)
plot_cross_correlation(lags_ccf, ccf, ax=ax_ccf, n_samples=len(a1),
                       xlabel="Lag [s]", title="Cross-correlation (mass 1 vs mass 2)")

save(fig, "spectral.png")

# ===========================================================================
# filters_utils.png
# ===========================================================================
print("filters_utils.png ...")

t, a1, _ = generate_2dof(duration=30.0, fs=FS, noise_std=1.0,
                          output="acceleration", seed=0)
hum = 0.5 * np.sin(2 * np.pi * 50.0 * t)
a1_contaminated = a1 + hum

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

SHOW = slice(0, int(3 * FS))
plot_signal(t[SHOW], a1_contaminated[SHOW], ax=ax_raw,
            label="raw (with 50 Hz hum)",
            title="Raw signal (first 3 s)", ylabel="Acceleration [m/s²]")
ax_raw.plot(t[SHOW], a1[SHOW], lw=0.7, alpha=0.7, label="clean")
ax_raw.legend()

bp_low, bp_high = fn1 - 3.0, fn1 + 3.0
a1_bp = bandpass(a1_contaminated, FS, low=bp_low, high=bp_high, order=4)
plot_signal(t[SHOW], a1_bp[SHOW], ax=ax_bp_time,
            title=f"Bandpass [{bp_low:.1f}–{bp_high:.1f} Hz] — mode 1",
            ylabel="Acceleration [m/s²]")

freqs, Pxx_raw = psd(a1_contaminated, FS, nperseg=2048)
freqs, Pxx_bp  = psd(a1_bp, FS, nperseg=2048)
plot_psd(freqs, Pxx_raw, ax=ax_bp_psd, db=False, xlim=(0, 60),
         title="PSD: raw vs bandpassed", ylabel="PSD [(m/s²)²/Hz]",
         label="raw", alpha=0.7)
ax_bp_psd.semilogy(freqs, Pxx_bp, lw=0.9, label=f"bandpass [{bp_low:.1f}–{bp_high:.1f} Hz]")
ax_bp_psd.axvline(fn1, color="red", ls="--", lw=1.2, label=f"fn1={fn1:.1f} Hz")
ax_bp_psd.legend(fontsize=8)

a1_notched = notch(a1_contaminated, FS, freq=50.0, q=30.0)
freqs, Pxx_notched = psd(a1_notched, FS, nperseg=2048)
plot_psd(freqs, Pxx_raw, ax=ax_notch_psd, db=False, xlim=(0, 100),
         title="Notch filter — 50 Hz hum removed",
         ylabel="PSD [(m/s²)²/Hz]", label="raw (with hum)", alpha=0.7)
ax_notch_psd.semilogy(freqs, Pxx_notched, lw=0.9, label="after notch @ 50 Hz")
ax_notch_psd.axvline(50,  color="orange", ls="--", lw=1.2, label="50 Hz notch")
ax_notch_psd.legend(fontsize=8)

vel  = integrate(a1_notched, FS, detrend_after=True)
disp = integrate(vel, FS, detrend_after=True)
plot_signal(t[SHOW], a1_notched[SHOW] / np.std(a1_notched), ax=ax_integ,
            label="accel (normalised)",
            title="Integration: accel → velocity → displacement",
            ylabel="Normalised amplitude")
ax_integ.plot(t[SHOW], vel[SHOW]  / np.std(vel),  lw=0.8, label="velocity")
ax_integ.plot(t[SHOW], disp[SHOW] / np.std(disp), lw=0.8, label="displacement")
ax_integ.legend(fontsize=8)

a1_dec, fs_dec = decimate(a1_notched, FS, target_fs=100.0)
freqs_dec, Pxx_dec = psd(a1_dec, fs_dec, nperseg=256)
plot_psd(freqs, Pxx_notched, ax=ax_dec_psd, db=False, xlim=(0, 50),
         title=f"Decimation {FS:.0f} → {fs_dec:.0f} Hz",
         ylabel="PSD [(m/s²)²/Hz]", label=f"original ({FS:.0f} Hz)", alpha=0.8)
ax_dec_psd.semilogy(freqs_dec, Pxx_dec, lw=0.9, label=f"decimated ({fs_dec:.0f} Hz)")
ax_dec_psd.legend(fontsize=8)

signals = {"raw": a1_contaminated, "notched": a1_notched,
           "bandpass": a1_bp, "velocity": vel, "displ.": disp}
names = list(signals.keys())
rms_vals  = [rms(s)          for s in signals.values()]
peak_vals = [peak(s)         for s in signals.values()]
cf_vals   = [crest_factor(s) for s in signals.values()]
x_pos = np.arange(len(names)); width = 0.28
ax_metrics.bar(x_pos - width, rms_vals,  width, label="RMS")
ax_metrics.bar(x_pos,         peak_vals, width, label="Peak")
ax_metrics.bar(x_pos + width, cf_vals,   width, label="Crest factor")
ax_metrics.set_yscale("log"); ax_metrics.set_xticks(x_pos)
ax_metrics.set_xticklabels(names, fontsize=9)
ax_metrics.set_ylabel("Value (log scale)")
ax_metrics.set_title("Signal metrics: RMS, Peak, Crest factor")
ax_metrics.legend(fontsize=8); ax_metrics.grid(True, axis="y", alpha=0.3)

save(fig, "filters_utils.png")

# ===========================================================================
# timefreq.png
# ===========================================================================
print("timefreq.png ...")

t_chirp = np.arange(int(FS)) / FS
x_chirp = chirp(t_chirp, f0=10.0, f1=150.0, t1=1.0, method="linear")

t_long, a1_long, _ = generate_2dof(duration=10.0, fs=FS, noise_std=0.5,
                                    output="acceleration", seed=7)
seg = slice(int(4.5 * FS), int(5.5 * FS))
t_2dof = t_long[seg] - t_long[seg][0]
x_2dof = a1_long[seg]

f_stft,  t_stft,  Z_chirp = stft(x_chirp, FS, nperseg=128, noverlap=120)
f_stft2, t_stft2, Z_2dof  = stft(x_2dof,  FS, nperseg=128, noverlap=120)

analysis_freqs = np.geomspace(2.0, FS / 4, num=80)
f_cwt,  t_cwt,  W_chirp = cwt_scalogram(x_chirp, FS, freqs=analysis_freqs, w=6.0)
f_cwt2, t_cwt2, W_2dof  = cwt_scalogram(x_2dof,  FS, freqs=analysis_freqs, w=6.0)

f_wvd,  t_wvd,  WVD_chirp = wigner_ville(x_chirp, FS)
f_wvd2, t_wvd2, WVD_2dof  = wigner_ville(x_2dof,  FS)

N = len(x_chirp)
f_spwvd,  t_spwvd,  SPWVD_chirp = smoothed_pseudo_wv(
    x_chirp, FS, lag_samples=N // 6, time_samples=N // 16)
f_spwvd2, t_spwvd2, SPWVD_2dof  = smoothed_pseudo_wv(
    x_2dof,  FS, lag_samples=N // 6, time_samples=N // 16)

fig, axes = plt.subplots(4, 2, figsize=(14, 12), constrained_layout=True)
fig.suptitle("Time-Frequency Analysis Comparison", fontsize=13, fontweight="bold")

plot_spectrogram(f_stft,  t_stft,  Z_chirp, ax=axes[0, 0], ylim=(0, 200),
                 title="STFT — Chirp 10→150 Hz")
plot_spectrogram(f_stft2, t_stft2, Z_2dof,  ax=axes[0, 1], ylim=(0, 200),
                 title="STFT — 2DOF structural response")

plot_scalogram(f_cwt,  t_cwt,  W_chirp, ax=axes[1, 0], log_freq=True, ylim=(2, 200),
               title="CWT (Morlet) — Chirp")
plot_scalogram(f_cwt2, t_cwt2, W_2dof,  ax=axes[1, 1], log_freq=True, ylim=(2, 200),
               title="CWT (Morlet) — 2DOF")
for ax in axes[1]:
    ax.set_ylabel("Frequency [Hz] (log)")

plot_wvd(f_wvd,  t_wvd,  WVD_chirp, ax=axes[2, 0], ylim=(0, 200),
         title="Wigner-Ville — Chirp")
plot_wvd(f_wvd2, t_wvd2, WVD_2dof,  ax=axes[2, 1], ylim=(0, 200),
         title="Wigner-Ville — 2DOF")

plot_wvd(f_spwvd,  t_spwvd,  SPWVD_chirp, ax=axes[3, 0], ylim=(0, 200),
         title="Smoothed Pseudo WVD — Chirp")
plot_wvd(f_spwvd2, t_spwvd2, SPWVD_2dof,  ax=axes[3, 1], ylim=(0, 200),
         title="Smoothed Pseudo WVD — 2DOF")

for ax, label in zip(axes[:, 0], ["STFT", "CWT (Morlet)", "WVD", "SPWVD"]):
    ax.set_ylabel(f"[{label}]  Frequency [Hz]")

save(fig, "timefreq.png")

# ===========================================================================
# instantaneous.png
# ===========================================================================
print("instantaneous.png ...")

DURATION_AM = 2.0
t_am = np.arange(int(FS * DURATION_AM)) / FS
f_carrier = 100.0; f_mod = 3.0
A_mod = 1.0 + 0.6 * np.sin(2.0 * np.pi * f_mod * t_am)
x_am = A_mod * np.sin(2.0 * np.pi * f_carrier * t_am)
env_am, phase_am, fi_am = hilbert_attributes(x_am, FS)

DURATION_CH = 2.0
t_ch = np.arange(int(FS * DURATION_CH)) / FS
x_ch = chirp(t_ch, f0=10.0, f1=200.0, t1=DURATION_CH, method="linear")
env_ch, phase_ch, fi_ch = hilbert_attributes(x_ch, FS)
fi_ch_theory = 10.0 + (200.0 - 10.0) / DURATION_CH * t_ch

t_long, a1_long, _ = generate_2dof(duration=30.0, fs=FS, noise_std=0.3,
                                    output="acceleration", seed=42)
x_mode1 = bandpass(a1_long, FS, low=fn1 - 2.0, high=fn1 + 2.0, order=6)
env_m1, phase_m1, fi_m1 = hilbert_attributes(x_mode1, FS)
seg = slice(int(3 * FS), int(8 * FS))
t_m1 = t_long[seg] - t_long[seg][0]

fig = plt.figure(figsize=(14, 10), constrained_layout=True)
fig.suptitle("Hilbert Transform — Instantaneous Signal Attributes",
             fontsize=13, fontweight="bold")
gs = gridspec.GridSpec(3, 3, figure=fig)

ax00 = fig.add_subplot(gs[0, 0])
plot_signal(t_am, x_am, ax=ax00, label="signal", envelope=env_am,
            title="AM signal: envelope vs true modulation", ylabel="Amplitude")
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

ax10 = fig.add_subplot(gs[1, 0])
plot_signal(t_ch, x_ch, ax=ax10, label="chirp", envelope=env_ch,
            title="Chirp 10→200 Hz: signal & envelope", ylabel="Amplitude")

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

save(fig, "instantaneous.png")

# ===========================================================================
# emd_imfs.png  and  emd_hht.png
# ===========================================================================
print("emd_imfs.png + emd_hht.png ...")

FS_EMD = 500.0
t_e, a1_e, _ = generate_2dof(duration=20.0, fs=FS_EMD, noise_std=0.5,
                               output="acceleration", seed=0)
N_e = len(t_e)

imfs, residue = emd(a1_e, max_imfs=8, max_sifting=10, sd_threshold=0.2)
n_imfs = imfs.shape[0]
envs, inst_freqs = hht(imfs, FS_EMD)
freq_bins, marginal = hht_marginal_spectrum(envs, inst_freqs, FS_EMD, n_bins=512)
freqs_psd_e, Pxx_e = psd(a1_e, FS_EMD, nperseg=1024)

# IMFs waterfall
show = slice(0, int(5 * FS_EMD))
fig1, axes = plt.subplots(n_imfs + 2, 1,
                           figsize=(13, 2.2 * (n_imfs + 2)),
                           constrained_layout=True)
fig1.suptitle("EMD — Intrinsic Mode Functions", fontsize=12, fontweight="bold")

plot_signal(t_e[show], a1_e[show], ax=axes[0],
            title="Input signal (first 5 s)", ylabel="Original")

for i, (imf, env) in enumerate(zip(imfs, envs)):
    sl = slice(N_e // 10, 9 * N_e // 10)
    fi_mean = inst_freqs[i, sl].mean()
    plot_signal(t_e[show], imf[show], ax=axes[i + 1], envelope=env[show],
                title=f"IMF {i+1}   (mean fi = {fi_mean:.1f} Hz)",
                ylabel=f"IMF {i+1}")

axes[-1].plot(t_e[show], residue[show], lw=0.8, color="saddlebrown")
axes[-1].set_ylabel("Residue"); axes[-1].set_title("Residue (trend)")
axes[-1].set_xlabel("Time [s]"); axes[-1].grid(True, alpha=0.3)

save(fig1, "emd_imfs.png")

# HHT analysis
fig2 = plt.figure(figsize=(14, 9), constrained_layout=True)
fig2.suptitle("HHT Analysis", fontsize=12, fontweight="bold")
gs = gridspec.GridSpec(2, 2, figure=fig2)

ax_tf = fig2.add_subplot(gs[0, :])
cmap = plt.cm.inferno
sl_tf = slice(N_e // 20, 19 * N_e // 20)
for env, fi in zip(envs, inst_freqs):
    sc = ax_tf.scatter(
        t_e[sl_tf], fi[sl_tf],
        c=env[sl_tf] ** 2, s=1, alpha=0.4, cmap=cmap,
        vmin=0, vmax=np.percentile(envs ** 2, 95),
    )
ax_tf.axhline(fn1, color="cyan",  ls="--", lw=1.2, label=f"fn1={fn1:.1f} Hz")
ax_tf.axhline(fn2, color="lime",  ls="--", lw=1.2, label=f"fn2={fn2:.1f} Hz")
ax_tf.set_ylim(0, FS_EMD / 2)
ax_tf.set_xlabel("Time [s]"); ax_tf.set_ylabel("Instantaneous frequency [Hz]")
ax_tf.set_title("HHT time-frequency representation (colour = instantaneous energy)")
ax_tf.legend(fontsize=9); fig2.colorbar(sc, ax=ax_tf, label="A²(t)")

ax_spec = fig2.add_subplot(gs[1, 0])
plot_psd(freq_bins, marginal, ax=ax_spec, db=False, xlim=(0, 60),
         title="Marginal HHT spectrum vs Welch PSD",
         ylabel="Amplitude² / Hz equivalent", label="HHT marginal")
ax_spec.semilogy(freqs_psd_e, Pxx_e, lw=1.0, alpha=0.8, label="Welch PSD")
ax_spec.axvline(fn1, color="red",    ls="--", lw=1.2, label=f"fn1={fn1:.1f} Hz")
ax_spec.axvline(fn2, color="orange", ls="--", lw=1.2, label=f"fn2={fn2:.1f} Hz")
ax_spec.legend(fontsize=8)

ax_damp = fig2.add_subplot(gs[1, 1])
sl_d = slice(N_e // 10, 9 * N_e // 10)
t_d = t_e[sl_d]
for i, (env, fi) in enumerate(zip(envs, inst_freqs)):
    fi_mean = fi[sl_d].mean()
    if fi_mean < 1.0 or fi_mean > FS_EMD / 4:
        continue
    log_env = np.log(np.maximum(env[sl_d], 1e-12))
    slope, intercept = np.polyfit(t_d, log_env, 1)
    zeta = -slope / (2.0 * np.pi * fi_mean)
    if 0 < zeta < 0.5:
        ax_damp.plot(t_d, log_env, lw=0.8,
                     label=f"IMF {i+1}  fi={fi_mean:.1f} Hz  ζ={zeta:.4f}")
        ax_damp.plot(t_d, slope * t_d + intercept, lw=1.5, ls="--", color="gray")
ax_damp.set_xlabel("Time [s]"); ax_damp.set_ylabel("ln(envelope)")
ax_damp.set_title("Log-envelope → damping ratio per IMF")
ax_damp.legend(fontsize=7); ax_damp.grid(True, alpha=0.3)

save(fig2, "emd_hht.png")

print("Done. All images saved to", OUT)
