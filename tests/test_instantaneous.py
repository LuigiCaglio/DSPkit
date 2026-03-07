"""Tests for dspkit.instantaneous."""

import numpy as np
import pytest

from dspkit.instantaneous import (
    analytic_signal,
    hilbert_attributes,
    hilbert_envelope,
    instantaneous_freq,
    instantaneous_phase,
)

FS = 2000.0
DURATION = 2.0
N = int(FS * DURATION)
t = np.arange(N) / FS

# Slice that avoids Hilbert edge effects (first/last ~5 %)
INTERIOR = slice(N // 20, 19 * N // 20)


def _sine(freq: float, amplitude: float = 1.0, phase: float = 0.0) -> np.ndarray:
    return amplitude * np.sin(2.0 * np.pi * freq * t + phase)


# ---------------------------------------------------------------------------
# analytic_signal
# ---------------------------------------------------------------------------

class TestAnalyticSignal:
    def test_real_part_equals_input(self):
        x = _sine(50.0)
        z = analytic_signal(x)
        np.testing.assert_allclose(z.real, x, atol=1e-10)

    def test_output_is_complex(self):
        z = analytic_signal(_sine(50.0))
        assert np.iscomplexobj(z)

    def test_output_length(self):
        x = _sine(50.0)
        assert len(analytic_signal(x)) == N

    def test_imaginary_part_is_cosine(self):
        """H{sin(2π f t)} = -cos(2π f t)."""
        freq = 40.0
        x = np.sin(2.0 * np.pi * freq * t)
        z = analytic_signal(x)
        expected_imag = -np.cos(2.0 * np.pi * freq * t)
        np.testing.assert_allclose(z.imag[INTERIOR], expected_imag[INTERIOR], atol=1e-3)


# ---------------------------------------------------------------------------
# hilbert_envelope
# ---------------------------------------------------------------------------

class TestHilbertEnvelope:
    def test_constant_amplitude_sine(self):
        """Envelope of a pure sine with amplitude A should be ~A everywhere."""
        A = 3.5
        x = _sine(60.0, amplitude=A)
        env = hilbert_envelope(x)
        np.testing.assert_allclose(env[INTERIOR], A, rtol=1e-3)

    def test_am_signal_tracks_modulation(self):
        """Envelope should track the amplitude modulation A(t)."""
        f_carrier = 200.0
        f_mod = 5.0
        A_mod = 1.0 + 0.5 * np.sin(2.0 * np.pi * f_mod * t)  # 0.5 to 1.5
        x = A_mod * np.sin(2.0 * np.pi * f_carrier * t)
        env = hilbert_envelope(x)
        # Interior away from edges AND averaged over carrier cycles
        sl = INTERIOR
        np.testing.assert_allclose(env[sl], A_mod[sl], atol=0.05)

    def test_nonnegative(self):
        x = _sine(50.0) + np.random.default_rng(0).normal(0, 0.1, N)
        assert np.all(hilbert_envelope(x) >= 0.0)

    def test_output_length(self):
        assert len(hilbert_envelope(_sine(50.0))) == N


# ---------------------------------------------------------------------------
# instantaneous_phase
# ---------------------------------------------------------------------------

class TestInstantaneousPhase:
    def test_linear_phase_for_pure_sine(self):
        """
        x(t) = sin(2π f t + φ₀)  →  phase(t) = 2π f t + φ₀ - π/2
        (sin is cos shifted by -π/2, so the analytic phase has a -π/2 offset)
        Actually phase = 2π f t + φ₀ - π/2; what matters is linearity.
        """
        freq = 30.0
        x = _sine(freq)
        phase = instantaneous_phase(x)
        # Phase should grow linearly at rate 2π*freq
        dphi = np.diff(phase[INTERIOR])
        expected_step = 2.0 * np.pi * freq / FS
        np.testing.assert_allclose(dphi, expected_step, rtol=1e-3)

    def test_unwrapped_monotone(self):
        """Unwrapped phase of a positive-frequency sine must be monotone."""
        phase = instantaneous_phase(_sine(50.0))
        assert np.all(np.diff(phase[INTERIOR]) > 0)

    def test_output_length(self):
        assert len(instantaneous_phase(_sine(50.0))) == N


# ---------------------------------------------------------------------------
# instantaneous_freq
# ---------------------------------------------------------------------------

class TestInstantaneousFreq:
    def test_pure_sine_constant_frequency(self):
        """Instantaneous frequency of a pure sine should equal its frequency."""
        freq = 80.0
        x = _sine(freq)
        fi = instantaneous_freq(x, FS)
        np.testing.assert_allclose(fi[INTERIOR], freq, rtol=5e-3)

    def test_linear_chirp_increasing_frequency(self):
        """For a linear chirp f0→f1, fi(t) should increase linearly."""
        from scipy.signal import chirp as _chirp
        f0, f1 = 20.0, 150.0
        x = _chirp(t, f0=f0, f1=f1, t1=DURATION, method="linear")
        fi = instantaneous_freq(x, FS)
        sl = INTERIOR
        # Should be approximately linear between f0 and f1
        assert fi[sl].min() > f0 * 0.85
        assert fi[sl].max() < f1 * 1.15
        # Slope should be positive (frequency increases)
        slope = np.polyfit(t[sl], fi[sl], 1)[0]
        assert slope > 0

    def test_output_length(self):
        assert len(instantaneous_freq(_sine(50.0), FS)) == N


# ---------------------------------------------------------------------------
# hilbert_attributes — consistency with individual functions
# ---------------------------------------------------------------------------

class TestHilbertAttributes:
    def test_consistent_with_individual_functions(self):
        """hilbert_attributes must return same values as the individual calls."""
        x = _sine(60.0, amplitude=2.0)
        env_a, phase_a, freq_a = hilbert_attributes(x, FS)

        np.testing.assert_array_equal(env_a,   hilbert_envelope(x))
        np.testing.assert_array_equal(phase_a, instantaneous_phase(x))
        np.testing.assert_array_equal(freq_a,  instantaneous_freq(x, FS))

    def test_output_shapes(self):
        x = _sine(50.0)
        env, phase, freq = hilbert_attributes(x, FS)
        assert env.shape == phase.shape == freq.shape == (N,)
