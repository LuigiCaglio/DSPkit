"""Tests for dspkit.utils."""

import numpy as np
import pytest

from dspkit.utils import (
    crest_factor,
    detrend,
    differentiate,
    integrate,
    peak,
    rms,
)

FS = 1000.0
DURATION = 5.0
N = int(FS * DURATION)
t = np.arange(N) / FS


# ---------------------------------------------------------------------------
# detrend
# ---------------------------------------------------------------------------

class TestDetrend:
    def test_removes_mean(self):
        x = np.ones(N) * 5.0 + np.random.default_rng(0).normal(0, 0.1, N)
        y = detrend(x, order=0)
        assert abs(y.mean()) < 1e-10

    def test_removes_linear_trend(self):
        trend = np.linspace(0, 10, N)
        noise = np.random.default_rng(1).normal(0, 0.01, N)
        y = detrend(trend + noise, order=1)
        # Residual should be small
        assert np.std(y) < 0.05

    def test_removes_quadratic_trend(self):
        trend = t ** 2
        y = detrend(trend, order=2)
        assert np.max(np.abs(y)) < 1e-8

    def test_output_length_unchanged(self):
        x = np.random.default_rng(2).normal(size=N)
        assert len(detrend(x, order=1)) == N


# ---------------------------------------------------------------------------
# rms
# ---------------------------------------------------------------------------

class TestRms:
    def test_sine_rms(self):
        """RMS of a sine with amplitude A should be A / sqrt(2)."""
        A = 3.0
        x = A * np.sin(2 * np.pi * 50 * t)
        assert abs(rms(x) - A / np.sqrt(2)) < 1e-3

    def test_constant_rms(self):
        assert abs(rms(np.full(N, 4.0)) - 4.0) < 1e-10

    def test_zero_signal(self):
        assert rms(np.zeros(N)) == 0.0


# ---------------------------------------------------------------------------
# peak
# ---------------------------------------------------------------------------

class TestPeak:
    def test_sine_peak(self):
        A = 2.5
        x = A * np.sin(2 * np.pi * 10 * t)
        assert abs(peak(x) - A) < 1e-6

    def test_negative_peak(self):
        x = np.array([-5.0, 1.0, 2.0])
        assert peak(x) == 5.0


# ---------------------------------------------------------------------------
# crest_factor
# ---------------------------------------------------------------------------

class TestCrestFactor:
    def test_sine_crest_factor(self):
        """Sine crest factor = sqrt(2)."""
        x = 3.0 * np.sin(2 * np.pi * 20 * t)
        assert abs(crest_factor(x) - np.sqrt(2)) < 1e-2

    def test_constant_crest_factor(self):
        """Constant signal has crest factor = 1."""
        assert abs(crest_factor(np.ones(N)) - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# integrate
# ---------------------------------------------------------------------------

class TestIntegrate:
    def test_integrate_cosine_gives_sine(self):
        """
        Integral of cos(2*pi*f*t) should be sin(2*pi*f*t) / (2*pi*f).
        Check that the dominant frequency of the result matches.
        """
        f = 10.0
        x = np.cos(2 * np.pi * f * t)
        y = integrate(x, FS, detrend_after=True)
        # Check correlation with expected sine
        expected = np.sin(2 * np.pi * f * t) / (2 * np.pi * f)
        # Normalise both to remove amplitude scale
        y_n = y / np.std(y)
        e_n = expected / np.std(expected)
        corr = np.corrcoef(y_n, e_n)[0, 1]
        assert corr > 0.99

    def test_output_length_unchanged(self):
        x = np.random.default_rng(3).normal(size=N)
        assert len(integrate(x, FS)) == N

    def test_first_sample_is_zero(self):
        x = np.ones(N)
        y = integrate(x, FS, detrend_after=False)
        assert y[0] == 0.0


# ---------------------------------------------------------------------------
# differentiate
# ---------------------------------------------------------------------------

class TestDifferentiate:
    def test_differentiate_sine_gives_cosine(self):
        """
        d/dt sin(2*pi*f*t) = 2*pi*f * cos(2*pi*f*t).
        Check amplitude at interior points (edges use lower-order stencils).
        """
        f = 5.0
        x = np.sin(2 * np.pi * f * t)
        dxdt = differentiate(x, FS)
        expected = 2 * np.pi * f * np.cos(2 * np.pi * f * t)
        # Compare interior only (avoid edge effects)
        sl = slice(10, -10)
        np.testing.assert_allclose(dxdt[sl], expected[sl], atol=1e-3)

    def test_output_length_unchanged(self):
        x = np.random.default_rng(4).normal(size=N)
        assert len(differentiate(x, FS)) == N

    def test_constant_derivative_is_zero(self):
        x = np.full(N, 7.0)
        np.testing.assert_allclose(differentiate(x, FS), 0.0, atol=1e-10)

    def test_differentiate_integrate_roundtrip(self):
        """Differentiating the integral of a signal should recover the original."""
        f = 8.0
        x = np.sin(2 * np.pi * f * t)
        y = integrate(x, FS, detrend_after=False)
        dxdt = differentiate(y, FS)
        sl = slice(20, -20)
        corr = np.corrcoef(x[sl], dxdt[sl])[0, 1]
        assert corr > 0.999
