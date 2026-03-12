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
