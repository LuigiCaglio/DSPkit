"""Tests for dspkit.utils."""

import numpy as np
import pytest

from dspkit.utils import (
    crest_factor,
    detrend,
    differentiate,
    differentiate_fft,
    integrate,
    integrate_fft,
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
        # np.gradient uses O(h²) central differences; with h=1/fs and
        # f=5 Hz the truncation error is ~(h·2πf)²/6·amplitude ≈ 0.005.
        np.testing.assert_allclose(dxdt[sl], expected[sl], atol=0.01)

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


# ── frequency-domain integration and differentiation ─────────────────────────
# The operators are exact for a band-limited signal, so these check against
# analytic truth rather than against another implementation.

def _tones(fs=200.0, n=8000):
    t = np.arange(n) / fs
    f1, f2 = 1.3, 4.7
    disp = 0.02 * np.sin(2 * np.pi * f1 * t) + 0.008 * np.sin(2 * np.pi * f2 * t + 0.7)
    vel = (0.02 * 2 * np.pi * f1 * np.cos(2 * np.pi * f1 * t)
           + 0.008 * 2 * np.pi * f2 * np.cos(2 * np.pi * f2 * t + 0.7))
    acc = (-0.02 * (2 * np.pi * f1) ** 2 * np.sin(2 * np.pi * f1 * t)
           - 0.008 * (2 * np.pi * f2) ** 2 * np.sin(2 * np.pi * f2 * t + 0.7))
    return fs, t, disp, vel, acc


def _rel_err(est, ref, trim=200):
    sl = slice(trim, len(est) - trim)
    e = est[sl] - est[sl].mean()
    r = ref[sl] - ref[sl].mean()
    return float(np.linalg.norm(e - r) / np.linalg.norm(r))


def test_integrate_fft_matches_analytic_velocity():
    fs, _, _, vel, acc = _tones()
    assert _rel_err(integrate_fft(acc, fs), vel) < 0.01


def test_differentiate_fft_matches_analytic_velocity():
    fs, _, disp, vel, _ = _tones()
    assert _rel_err(differentiate_fft(disp, fs), vel) < 0.01


def test_double_integration_beats_trapezoid_on_biased_noisy_data():
    """The reason this function exists: trapezoid turns a DC bias into a parabola."""
    fs, _, disp, _, acc = _tones()
    rng = np.random.default_rng(0)
    measured = acc + 0.02 * np.max(np.abs(acc)) * rng.normal(size=acc.size) + 0.35

    trap = integrate(integrate(measured, fs), fs)
    spec = integrate_fft(measured, fs, order=2, hp_cutoff=0.5)

    assert _rel_err(spec, disp) < 0.15
    # Not a marginal improvement: trapezoid is off by several hundred percent.
    assert _rel_err(trap, disp) > 10 * _rel_err(spec, disp)


def test_high_pass_does_not_touch_a_signal_above_the_cutoff():
    """Filtering happens before zero-padding, so the padding envelope is safe."""
    fs, _, _, vel, acc = _tones()
    without = integrate_fft(acc, fs)
    with_hp = integrate_fft(acc, fs, hp_cutoff=0.5)
    assert _rel_err(with_hp, vel) == pytest.approx(_rel_err(without, vel), abs=1e-3)


def test_cutoffs_outside_the_usable_range_are_refused():
    fs, _, _, _, acc = _tones()
    for bad in (0.0, -1.0, fs / 2, fs):
        if bad == 0.0:
            continue          # falsy: means "no cutoff", not an error
        with pytest.raises(ValueError, match="Nyquist"):
            integrate_fft(acc, fs, hp_cutoff=bad)


def test_order_must_be_at_least_one():
    fs, _, _, _, acc = _tones()
    with pytest.raises(ValueError, match="order"):
        integrate_fft(acc, fs, order=0)
