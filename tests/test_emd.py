"""Tests for dspkit.emd."""

import numpy as np
import pytest

from dspkit.emd import emd, hht, hht_marginal_spectrum

FS = 500.0
DURATION = 4.0
N = int(FS * DURATION)
t = np.arange(N) / FS


def _sine(freq: float, amplitude: float = 1.0) -> np.ndarray:
    return amplitude * np.sin(2.0 * np.pi * freq * t)


# ---------------------------------------------------------------------------
# emd — reconstruction and basic properties
# ---------------------------------------------------------------------------

class TestEmd:
    def test_reconstruction_single_sine(self):
        """IMFs + residue must sum back to the original signal exactly."""
        x = _sine(20.0)
        imfs, residue = emd(x)
        reconstructed = imfs.sum(axis=0) + residue
        np.testing.assert_allclose(reconstructed, x, atol=1e-8)

    def test_reconstruction_two_sines(self):
        x = _sine(10.0) + _sine(50.0, amplitude=0.5)
        imfs, residue = emd(x)
        np.testing.assert_allclose(imfs.sum(axis=0) + residue, x, atol=1e-8)

    def test_reconstruction_with_noise(self):
        rng = np.random.default_rng(0)
        x = _sine(20.0) + rng.normal(0, 0.2, N)
        imfs, residue = emd(x)
        np.testing.assert_allclose(imfs.sum(axis=0) + residue, x, atol=1e-8)

    def test_output_shapes(self):
        x = _sine(10.0) + _sine(40.0)
        imfs, residue = emd(x)
        assert imfs.ndim == 2
        assert imfs.shape[1] == N
        assert residue.shape == (N,)

    def test_max_imfs_respected(self):
        x = _sine(10.0) + _sine(30.0) + _sine(80.0)
        imfs, residue = emd(x, max_imfs=2)
        assert imfs.shape[0] == 2

    def test_monotone_input_returns_no_imfs(self):
        """A monotone (e.g. linear ramp) has no extrema → empty IMF array."""
        x = np.linspace(0, 1, N)
        imfs, residue = emd(x)
        assert imfs.shape[0] == 0
        np.testing.assert_allclose(residue, x, atol=1e-10)

    def test_two_sines_separated_in_frequency(self):
        """
        A two-component signal at well-separated frequencies should yield
        at least 2 IMFs, with the first carrying more high-frequency energy.
        """
        f_low, f_high = 5.0, 60.0
        x = _sine(f_low) + _sine(f_high)
        imfs, _ = emd(x)
        assert imfs.shape[0] >= 2

        # First IMF should have higher mean instantaneous frequency than second
        from dspkit.instantaneous import instantaneous_freq
        fi_0 = instantaneous_freq(imfs[0], FS)
        fi_1 = instantaneous_freq(imfs[1], FS)
        sl = slice(N // 10, 9 * N // 10)  # avoid edges
        assert fi_0[sl].mean() > fi_1[sl].mean()

    def test_imfs_orthogonal_approximately(self):
        """
        IMFs should be approximately orthogonal (a property of proper EMD).
        The inner product of IMF_i and IMF_j (i≠j) should be small relative
        to their individual energies.
        """
        x = _sine(8.0) + _sine(40.0, amplitude=0.8)
        imfs, _ = emd(x, max_imfs=3)
        if imfs.shape[0] < 2:
            pytest.skip("Not enough IMFs extracted")
        dot = np.dot(imfs[0], imfs[1])
        e0 = np.dot(imfs[0], imfs[0])
        e1 = np.dot(imfs[1], imfs[1])
        # Normalised inner product should be small
        assert abs(dot) / np.sqrt(e0 * e1) < 0.3


# ---------------------------------------------------------------------------
# hht
# ---------------------------------------------------------------------------

class TestHht:
    def test_output_shapes(self):
        x = _sine(20.0) + _sine(60.0)
        imfs, _ = emd(x, max_imfs=3)
        envs, freqs = hht(imfs, FS)
        assert envs.shape == imfs.shape
        assert freqs.shape == imfs.shape

    def test_envelopes_nonnegative(self):
        x = _sine(20.0) + _sine(60.0)
        imfs, _ = emd(x, max_imfs=3)
        envs, _ = hht(imfs, FS)
        assert np.all(envs >= 0.0)

    def test_instantaneous_freq_of_imf_near_true_freq(self):
        """
        For a well-separated two-component signal, the first IMF's mean
        instantaneous frequency should be close to the high-frequency component.
        """
        f_high = 60.0
        x = _sine(5.0) + _sine(f_high)
        imfs, _ = emd(x, max_imfs=2)
        if imfs.shape[0] < 1:
            pytest.skip("No IMFs extracted")
        _, freqs = hht(imfs, FS)
        sl = slice(N // 8, 7 * N // 8)
        assert abs(freqs[0, sl].mean() - f_high) < 10.0


# ---------------------------------------------------------------------------
# hht_marginal_spectrum
# ---------------------------------------------------------------------------

class TestHhtMarginalSpectrum:
    def test_output_shapes(self):
        x = _sine(20.0) + _sine(60.0)
        imfs, _ = emd(x, max_imfs=3)
        envs, freqs = hht(imfs, FS)
        fb, spec = hht_marginal_spectrum(envs, freqs, FS, n_bins=256)
        assert len(fb) == 256
        assert len(spec) == 256

    def test_frequency_axis_range(self):
        x = _sine(20.0)
        imfs, _ = emd(x, max_imfs=2)
        envs, freqs = hht(imfs, FS)
        fb, _ = hht_marginal_spectrum(envs, freqs, FS)
        assert fb[0] == 0.0
        assert abs(fb[-1] - FS / 2) < 1.0

    def test_spectrum_nonnegative(self):
        x = _sine(20.0) + _sine(60.0)
        imfs, _ = emd(x, max_imfs=3)
        envs, freqs = hht(imfs, FS)
        _, spec = hht_marginal_spectrum(envs, freqs, FS)
        assert np.all(spec >= 0.0)

    def test_peak_near_dominant_frequency(self):
        """Marginal spectrum should peak near the dominant signal frequency."""
        f_dom = 30.0
        x = 2.0 * _sine(f_dom) + 0.3 * _sine(100.0)
        imfs, _ = emd(x, max_imfs=4)
        envs, freqs = hht(imfs, FS)
        fb, spec = hht_marginal_spectrum(envs, freqs, FS, n_bins=1024)
        peak_freq = fb[np.argmax(spec)]
        assert abs(peak_freq - f_dom) < 5.0
