"""
Tests for dspkit.timefreq.

Strategy: use chirp signals and pure sines with known time-frequency structure
to verify that each distribution concentrates energy at the right location.
"""

import warnings
import numpy as np
import pytest

from dspkit.timefreq import cwt_scalogram, smoothed_pseudo_wv, stft, wigner_ville

FS = 1000.0
DURATION = 1.0  # short to keep WVD tests fast
N = int(FS * DURATION)
t = np.arange(N) / FS


def _sine(freq: float, amplitude: float = 1.0) -> np.ndarray:
    return amplitude * np.sin(2.0 * np.pi * freq * t)


def _chirp(f0: float, f1: float) -> np.ndarray:
    """Linear chirp from f0 to f1 over DURATION."""
    from scipy.signal import chirp
    return chirp(t, f0=f0, f1=f1, t1=DURATION, method="linear")


# ---------------------------------------------------------------------------
# STFT
# ---------------------------------------------------------------------------

class TestStft:
    def test_output_shapes(self):
        x = _sine(50.0)
        freqs, times, Zxx = stft(x, FS, nperseg=128)
        assert freqs.shape[0] == 128 // 2 + 1
        assert Zxx.shape == (len(freqs), len(times))
        assert np.iscomplexobj(Zxx)

    def test_peak_at_sine_frequency(self):
        freq = 100.0
        x = _sine(freq)
        freqs, times, Zxx = stft(x, FS, nperseg=256)
        amp = np.abs(Zxx)
        # Average over time, find peak frequency
        mean_amp = amp.mean(axis=1)
        assert abs(freqs[np.argmax(mean_amp)] - freq) < 5.0

    def test_frequency_axis_range(self):
        x = _sine(50.0)
        freqs, _, _ = stft(x, FS, nperseg=256)
        assert freqs[0] == 0.0
        assert abs(freqs[-1] - FS / 2) < 1e-6

    def test_default_noverlap_is_75_percent(self):
        x = _sine(50.0)
        nperseg = 128
        freqs, times_default, _ = stft(x, FS, nperseg=nperseg)
        freqs, times_explicit, _ = stft(x, FS, nperseg=nperseg, noverlap=nperseg * 3 // 4)
        np.testing.assert_array_equal(times_default, times_explicit)

    def test_chirp_energy_moves_in_time(self):
        """For a chirp, early time bins should peak at lower frequencies."""
        x = _chirp(f0=50.0, f1=200.0)
        freqs, times, Zxx = stft(x, FS, nperseg=128)
        amp = np.abs(Zxx)
        # Peak frequency in first quarter vs last quarter
        n_t = len(times)
        peak_early = freqs[np.argmax(amp[:, : n_t // 4].mean(axis=1))]
        peak_late  = freqs[np.argmax(amp[:, 3 * n_t // 4 :].mean(axis=1))]
        assert peak_early < peak_late


# ---------------------------------------------------------------------------
# CWT scalogram
# ---------------------------------------------------------------------------

class TestCwtScalogram:
    def test_output_shapes(self):
        x = _sine(50.0)
        freqs_out, times_out, W = cwt_scalogram(x, FS)
        assert len(freqs_out) == 50  # default
        assert len(times_out) == N
        assert W.shape == (50, N)
        assert np.iscomplexobj(W)

    def test_custom_freqs(self):
        x = _sine(50.0)
        analysis_freqs = np.array([10.0, 50.0, 100.0, 200.0])
        freqs_out, _, W = cwt_scalogram(x, FS, freqs=analysis_freqs)
        np.testing.assert_array_equal(freqs_out, analysis_freqs)
        assert W.shape[0] == 4

    def test_peak_at_sine_frequency(self):
        freq = 80.0
        x = _sine(freq)
        analysis_freqs = np.linspace(10.0, 200.0, 100)
        freqs_out, times_out, W = cwt_scalogram(x, FS, freqs=analysis_freqs)
        # Average energy over time
        mean_power = np.abs(W).mean(axis=1)
        peak_freq = freqs_out[np.argmax(mean_power)]
        assert abs(peak_freq - freq) < 5.0

    def test_chirp_energy_moves_in_time(self):
        x = _chirp(f0=20.0, f1=150.0)
        freqs_out, times_out, W = cwt_scalogram(x, FS)
        power = np.abs(W) ** 2
        n_t = len(times_out)
        peak_freq_early = freqs_out[np.argmax(power[:, : n_t // 4].mean(axis=1))]
        peak_freq_late  = freqs_out[np.argmax(power[:, 3 * n_t // 4:].mean(axis=1))]
        assert peak_freq_early < peak_freq_late


# ---------------------------------------------------------------------------
# Wigner-Ville Distribution
# ---------------------------------------------------------------------------

class TestWignerVille:
    def test_output_shapes(self):
        x = _sine(50.0)
        freqs, times, WVD = wigner_ville(x, FS)
        assert len(freqs) == N // 2 + 1
        assert len(times) == N
        assert WVD.shape == (N, N // 2 + 1)
        assert WVD.dtype == float

    def test_frequency_axis(self):
        x = _sine(50.0)
        freqs, _, _ = wigner_ville(x, FS)
        assert freqs[0] == 0.0
        # Half-lag WVD covers 0 to fs/4 (Nyquist of the half-lag domain)
        assert abs(freqs[-1] - FS / 4) < 1.0

    def test_real_output(self):
        x = _sine(50.0)
        _, _, WVD = wigner_ville(x, FS)
        assert np.isrealobj(WVD)

    def test_peak_at_sine_frequency(self):
        """WVD of a sine should concentrate energy at its frequency."""
        freq = 100.0
        x = _sine(freq)
        freqs, times, WVD = wigner_ville(x, FS)
        # Integrate over time (avoid edges)
        sl = slice(N // 8, 7 * N // 8)
        mean_wvd = WVD[sl, :].mean(axis=0)
        peak_freq = freqs[np.argmax(mean_wvd)]
        assert abs(peak_freq - freq) < 5.0

    def test_warn_for_large_signal(self):
        x = np.zeros(3000)
        with pytest.warns(UserWarning, match="O\\(N²\\)"):
            wigner_ville(x, FS, warn_above=2048)

    def test_no_warn_below_threshold(self):
        x = _sine(50.0)  # N=1000 < 2048
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            wigner_ville(x, FS, warn_above=2048)

    def test_instantaneous_power_marginal(self):
        """
        Integrating WVD over frequency should give instantaneous power.
        WVD marginal: ∫ W(t,f) df ≈ |z(t)|² = A² for a real sine of
        amplitude A (analytic signal has |z| = A).
        For interior samples (avoid edge effects), check proportionality.
        """
        freq = 80.0
        A = 2.0
        x = A * np.sin(2.0 * np.pi * freq * t)
        freqs, times, WVD = wigner_ville(x, FS)
        df = freqs[1] - freqs[0]
        power_marginal = WVD.sum(axis=1) * df
        # Interior only (edges have fewer valid lags)
        sl = slice(N // 4, 3 * N // 4)
        expected_power = A ** 2  # |z(t)|² for analytic signal of amplitude A
        assert abs(power_marginal[sl].mean() / expected_power - 1.0) < 0.05


# ---------------------------------------------------------------------------
# Smoothed Pseudo WVD
# ---------------------------------------------------------------------------

class TestSmoothedPseudoWv:
    def test_output_shapes(self):
        x = _sine(50.0)
        freqs, times, S = smoothed_pseudo_wv(x, FS)
        assert len(freqs) == N // 2 + 1
        assert len(times) == N
        assert S.shape == (N, N // 2 + 1)

    def test_peak_at_sine_frequency(self):
        freq = 120.0
        x = _sine(freq)
        freqs, _, S = smoothed_pseudo_wv(x, FS)
        sl = slice(N // 8, 7 * N // 8)
        mean_s = S[sl, :].mean(axis=0)
        peak_freq = freqs[np.argmax(mean_s)]
        assert abs(peak_freq - freq) < 5.0

    def test_more_smoothing_reduces_cross_terms(self):
        """
        Two-component signal: SPWVD with more smoothing should have
        lower energy at the midpoint frequency (cross-term location).
        """
        f1, f2 = 80.0, 200.0
        x = _sine(f1) + _sine(f2)
        freqs, _, S_less = smoothed_pseudo_wv(x, FS, lag_samples=8,  time_samples=8)
        freqs, _, S_more = smoothed_pseudo_wv(x, FS, lag_samples=64, time_samples=64)

        # Cross-term sits between f1 and f2
        mid = (f1 + f2) / 2.0
        mid_idx = np.argmin(np.abs(freqs - mid))
        sl = slice(N // 8, 7 * N // 8)

        cross_less = S_less[sl, mid_idx].mean()
        cross_more = S_more[sl, mid_idx].mean()
        # More smoothing → smaller (or equal) cross-term magnitude
        assert abs(cross_more) <= abs(cross_less) + 1e-10

    def test_warn_for_large_signal(self):
        x = np.zeros(3000)
        with pytest.warns(UserWarning, match="O\\(N²\\)"):
            smoothed_pseudo_wv(x, FS, warn_above=2048)
