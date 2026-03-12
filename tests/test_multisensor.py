"""Tests for dspkit.multisensor."""

import numpy as np
import pytest

from dspkit.multisensor import correlation_matrix, coherence_matrix, psd_matrix


FS = 1000.0
N = 10_000


class TestCorrelationMatrix:
    def test_self_correlation_is_one(self):
        rng = np.random.default_rng(0)
        data = rng.normal(0, 1, (3, N))
        R = correlation_matrix(data)
        np.testing.assert_allclose(np.diag(R), 1.0, atol=1e-10)

    def test_symmetric(self):
        rng = np.random.default_rng(1)
        data = rng.normal(0, 1, (4, N))
        R = correlation_matrix(data)
        np.testing.assert_allclose(R, R.T, atol=1e-10)

    def test_correlated_signals(self):
        """Identical signals should have correlation = 1."""
        rng = np.random.default_rng(2)
        x = rng.normal(0, 1, N)
        data = np.vstack([x, x, x])
        R = correlation_matrix(data)
        np.testing.assert_allclose(R, np.ones((3, 3)), atol=1e-10)

    def test_uncorrelated_signals(self):
        """Independent noise channels should have off-diagonal ≈ 0."""
        rng = np.random.default_rng(3)
        data = rng.normal(0, 1, (3, N))
        R = correlation_matrix(data)
        off_diag = R[np.triu_indices(3, k=1)]
        assert np.all(np.abs(off_diag) < 0.05)


class TestCoherenceMatrix:
    def test_self_coherence_is_one(self):
        rng = np.random.default_rng(4)
        data = rng.normal(0, 1, (2, N))
        freqs, C = coherence_matrix(data, FS)
        # Diagonal should be 1 at all frequencies
        for i in range(2):
            np.testing.assert_allclose(C[i, i, :], 1.0, atol=1e-10)

    def test_output_shape(self):
        rng = np.random.default_rng(5)
        data = rng.normal(0, 1, (3, N))
        freqs, C = coherence_matrix(data, FS, nperseg=256)
        assert C.shape[0] == 3
        assert C.shape[1] == 3
        assert C.shape[2] == len(freqs)

    def test_symmetric(self):
        rng = np.random.default_rng(6)
        data = rng.normal(0, 1, (3, N))
        _, C = coherence_matrix(data, FS)
        for k in range(C.shape[2]):
            np.testing.assert_allclose(C[:, :, k], C[:, :, k].T, atol=1e-10)


class TestPsdMatrix:
    def test_hermitian(self):
        """PSD matrix should be Hermitian at each frequency."""
        rng = np.random.default_rng(7)
        data = rng.normal(0, 1, (3, N))
        _, G = psd_matrix(data, FS)
        for k in range(G.shape[2]):
            np.testing.assert_allclose(G[:, :, k], G[:, :, k].conj().T, atol=1e-10)

    def test_diagonal_real_nonneg(self):
        """Diagonal entries (auto-PSD) should be real and non-negative."""
        rng = np.random.default_rng(8)
        data = rng.normal(0, 1, (2, N))
        _, G = psd_matrix(data, FS)
        for i in range(2):
            auto_psd = G[i, i, :]
            assert np.allclose(auto_psd.imag, 0, atol=1e-12)
            assert np.all(auto_psd.real >= -1e-12)

    def test_output_shape(self):
        rng = np.random.default_rng(9)
        data = rng.normal(0, 1, (4, N))
        freqs, G = psd_matrix(data, FS, nperseg=512)
        assert G.shape == (4, 4, len(freqs))
