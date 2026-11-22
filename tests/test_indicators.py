"""Tests for dspkit.indicators."""

import numpy as np
import pytest

from dspkit.indicators import (
    spectral_entropy,
    kurtosis,
    skewness,
    rms_variation,
    frequency_shift,
    energy_variation,
)
from dspkit.spectral import psd


FS = 1000.0
DURATION = 10.0
N = int(FS * DURATION)


class TestSpectralEntropy:
    def test_white_noise_high_entropy(self):
        """White noise should have spectral entropy close to 1."""
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, N)
        freqs, Pxx = psd(x, FS, nperseg=1024)
        se = spectral_entropy(freqs, Pxx)
        assert se > 0.9

    def test_pure_tone_low_entropy(self):
        """A pure sine should have low spectral entropy."""
        t = np.arange(N) / FS
        x = np.sin(2 * np.pi * 100 * t)
        freqs, Pxx = psd(x, FS, nperseg=1024)
        se = spectral_entropy(freqs, Pxx)
        assert se < 0.3

    def test_zero_spectrum(self):
        freqs = np.linspace(0, 500, 513)
        Pxx = np.zeros(513)
        assert spectral_entropy(freqs, Pxx) == 0.0

    def test_range(self):
        """Spectral entropy must be in [0, 1]."""
        rng = np.random.default_rng(1)
        x = rng.normal(0, 1, N)
        freqs, Pxx = psd(x, FS)
        se = spectral_entropy(freqs, Pxx)
        assert 0.0 <= se <= 1.0


class TestKurtosis:
    def test_gaussian_excess_near_zero(self):
        """Gaussian noise has excess kurtosis ~ 0."""
        rng = np.random.default_rng(2)
        x = rng.normal(0, 1, 100_000)
        k = kurtosis(x, excess=True)
        assert abs(k) < 0.1

    def test_gaussian_regular_near_three(self):
        rng = np.random.default_rng(3)
        x = rng.normal(0, 1, 100_000)
        k = kurtosis(x, excess=False)
        assert abs(k - 3.0) < 0.1

    def test_uniform_negative_excess(self):
        """Uniform distribution has excess kurtosis = -1.2."""
        rng = np.random.default_rng(4)
        x = rng.uniform(-1, 1, 100_000)
        k = kurtosis(x, excess=True)
        assert abs(k - (-1.2)) < 0.15


class TestSkewness:
    def test_symmetric_near_zero(self):
        rng = np.random.default_rng(5)
        x = rng.normal(0, 1, 100_000)
        assert abs(skewness(x)) < 0.05

    def test_positive_skew(self):
        """Exponential distribution is positively skewed."""
        rng = np.random.default_rng(6)
        x = rng.exponential(1.0, 100_000)
        assert skewness(x) > 1.0


class TestRmsVariation:
    def test_constant_amplitude(self):
        """Constant-amplitude sine should have constant RMS."""
        t = np.arange(N) / FS
        x = np.sin(2 * np.pi * 10 * t)
        times, rms_vals = rms_variation(x, FS)
        # All RMS values should be similar
        assert np.std(rms_vals) / np.mean(rms_vals) < 0.05

    def test_output_shape(self):
        t = np.arange(N) / FS
        x = np.sin(2 * np.pi * 10 * t)
        times, rms_vals = rms_variation(x, FS, segment_duration=1.0)
        assert len(times) == len(rms_vals)
        assert len(times) == 10  # 10s / 1s = 10 segments


class TestFrequencyShift:
    def test_constant_frequency(self):
        """A pure tone should show constant dominant frequency."""
        t = np.arange(N) / FS
        x = np.sin(2 * np.pi * 50 * t)
        times, freqs = frequency_shift(x, FS, segment_duration=2.0)
        np.testing.assert_allclose(freqs, 50.0, atol=2.0)


class TestEnergyVariation:
    def test_output_shape(self):
        rng = np.random.default_rng(7)
        x = rng.normal(0, 1, N)
        times, energies = energy_variation(x, FS, segment_duration=1.0)
        assert len(times) == 10
        assert len(energies) == 10
        assert np.all(energies > 0)


# --- windowed_stats ----------------------------------------------------------

from dspkit.indicators import windowed_stats, spectral_entropy as _se
from scipy import signal as _sig


def test_windowed_stats_matches_direct_computation():
    rng = np.random.default_rng(3)
    fs = 200.0
    x = rng.normal(size=20_123) + 0.5
    w = windowed_stats(x, fs, window_duration=5.0)
    L = 1000
    assert w["window_samples"] == L and len(w["times"]) == 20_123 // L
    for i in (0, 7, len(w["times"]) - 1):
        c = x[i * L:(i + 1) * L]
        assert np.isclose(w["rms"][i], np.sqrt(np.mean(c ** 2)))
        assert np.isclose(w["mean"][i], c.mean())
        assert np.isclose(w["crest_factor"][i], np.max(np.abs(c)) / np.sqrt(np.mean(c ** 2)))
        assert np.isclose(w["kurtosis"][i], kurtosis(c, excess=True))
        f, P = _sig.welch(c, fs=fs, nperseg=1000)
        assert np.isclose(w["dominant_freq"][i], f[np.argmax(P)])
        assert np.isclose(w["spectral_entropy"][i], _se(f, P))
    assert np.isclose(w["times"][0], 2.5)


def test_windowed_stats_reads_signals_right():
    fs = 100.0
    t = np.arange(0, 200, 1 / fs)
    x = np.sin(2 * np.pi * 4 * t)
    x[10_050:10_060] += 30                       # impact in window 10 (100-110 s)
    w = windowed_stats(x, fs, window_duration=10.0)
    assert int(np.argmax(w["kurtosis"])) == 10 and w["kurtosis"][10] > 10
    quiet = np.delete(np.arange(20), 10)
    assert np.allclose(w["kurtosis"][quiet], -1.5, atol=0.01)       # sine
    assert np.allclose(w["crest_factor"][quiet], np.sqrt(2), atol=0.01)
    assert np.allclose(w["dominant_freq"][quiet], 4.0, atol=w["df"])
    assert w["spectral_entropy"][quiet].max() < 0.2                   # one line
    noise = windowed_stats(np.random.default_rng(0).normal(size=40_000), fs, 20.0)
    assert noise["spectral_entropy"].min() > 0.9
    assert np.allclose(noise["kurtosis"], 0, atol=0.3)


def test_windowed_stats_defaults_and_edges():
    x = np.random.default_rng(1).normal(size=50_000)
    w = windowed_stats(x, 100.0)
    assert w["window_samples"] == 500 and len(w["times"]) == 100
    assert windowed_stats(x[:3000], 100.0)["window_samples"] == 64     # floor
    one = windowed_stats(x[:500], 100.0, window_duration=60.0)          # longer than record
    assert len(one["times"]) == 1 and one["window_samples"] == 500
    z = windowed_stats(np.zeros(1000), 100.0, 1.0)
    assert np.all(np.isnan(z["crest_factor"])) and np.all(np.isnan(z["kurtosis"]))
    with pytest.raises(ValueError):
        windowed_stats(x, 100.0, window_duration=0)
