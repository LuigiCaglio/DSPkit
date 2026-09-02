"""Tests for dspkit.multisensor."""

import warnings

import numpy as np
import pytest

from dspkit.multisensor import (
    _normalised_csd_inverse,
    correlation_matrix,
    coherence_matrix,
    psd_matrix,
    multiple_coherence,
    partial_coherence,
)
from dspkit.spectral import coherence


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


# ---------------------------------------------------------------------------
# Segment-count guard (the single-segment trap)
#
# The guard itself, and the pin on the 1.0-everywhere behaviour it prevents,
# live in tests/test_spectral.py next to coherence(). Here: the matrix form.
# ---------------------------------------------------------------------------

class TestSegmentGuard:
    def test_coherence_matrix_refuses_single_segment(self):
        rng = np.random.default_rng(21)
        data = rng.normal(0, 1, (3, N))
        with pytest.raises(ValueError, match="identically 1.0"):
            coherence_matrix(data, FS, nperseg=N)

    def test_coherence_matrix_warns_below_min_segments(self):
        rng = np.random.default_rng(22)
        data = rng.normal(0, 1, (3, N))
        with pytest.warns(UserWarning, match="Welch segments"):
            coherence_matrix(data, FS, nperseg=N // 3)

    def test_default_nperseg_is_quiet(self):
        rng = np.random.default_rng(23)
        data = rng.normal(0, 1, (3, N))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            coherence_matrix(data, FS)


# ---------------------------------------------------------------------------
# multiple_coherence
# ---------------------------------------------------------------------------

class TestMultipleCoherence:
    def test_output_shape(self):
        rng = np.random.default_rng(30)
        data = rng.normal(0, 1, (4, N))
        freqs, g2 = multiple_coherence(data, FS, nperseg=256)
        assert g2.shape == (4, len(freqs))

    def test_independent_channels_are_low(self):
        """Two independent noise channels: nothing to explain."""
        rng = np.random.default_rng(31)
        data = rng.normal(0, 1, (2, N))
        _, g2 = multiple_coherence(data, FS, nperseg=256)
        assert np.mean(g2) < 0.1

    def test_independent_channels_sit_at_the_bias_floor(self):
        """
        Not at zero — at about q / n_segments, which is what the docstring
        tells the caller to compare against.
        """
        rng = np.random.default_rng(32)
        n_ch = 4
        data = rng.normal(0, 1, (n_ch, N))
        nperseg = 256
        n_seg = (N - nperseg // 2) // (nperseg // 2)
        _, g2 = multiple_coherence(data, FS, nperseg=nperseg)
        floor = (n_ch - 1) / n_seg
        assert 0.5 * floor < np.mean(g2) < 2.0 * floor

    def test_exact_linear_combination_is_one_across_the_band(self):
        """
        The case the ridge exists for: channel 2 is x + y exactly, so the CSD
        matrix is exactly rank-deficient and the true answer is 1.
        """
        rng = np.random.default_rng(33)
        x, y = rng.normal(0, 1, (2, N))
        data = np.vstack([x, y, x + y])
        _, g2 = multiple_coherence(data, FS, nperseg=256)
        assert np.all(g2[2] > 0.999)

    def test_redundant_and_independent_channels_are_separated(self):
        rng = np.random.default_rng(34)
        x, y, w = rng.normal(0, 1, (3, N))
        data = np.vstack([x, y, x + y, w])
        _, g2 = multiple_coherence(data, FS, nperseg=256)
        assert np.median(g2[2]) > 0.999      # exactly redundant
        assert np.median(g2[3]) < 0.1        # unrelated to everything

    def test_pinv_would_fail_the_redundant_case(self):
        """
        Documents why the ridge was chosen. A pseudo-inverse truncates the
        singular direction that carries the answer, so 1 - 1/(G_ii·inv_ii)
        comes out negative for a perfectly redundant channel — which the
        [0, 1] clip would then show as 0.0, the most redundant array possible
        reported as perfectly independent.
        """
        rng = np.random.default_rng(35)
        x, y = rng.normal(0, 1, (2, N))
        data = np.vstack([x, y, x + y])
        freqs, G = psd_matrix(data, FS, nperseg=256)
        Gt = np.moveaxis(G, -1, 0)
        idx = np.arange(3)
        d = np.sqrt(np.real(Gt[:, idx, idx]))
        Ghat = Gt / (d[:, :, None] * d[:, None, :])
        Gpinv = np.linalg.pinv(Ghat)
        pinv_gamma2 = 1.0 - 1.0 / np.real(Gpinv[:, idx, idx])
        assert np.median(pinv_gamma2[:, 2]) < 0.0      # pinv: badly wrong

        _, g2 = multiple_coherence(data, FS, nperseg=256)
        assert np.median(g2[2]) > 0.999                # ridge: right

    def test_two_channels_reduce_to_ordinary_coherence(self):
        rng = np.random.default_rng(36)
        x = rng.normal(0, 1, N)
        y = 0.5 * x + rng.normal(0, 1, N)
        _, g2 = multiple_coherence(np.vstack([x, y]), FS, nperseg=256)
        _, Cxy = coherence(x, y, FS, nperseg=256)
        np.testing.assert_allclose(g2[0], Cxy, atol=1e-8)

    def test_values_stay_in_range(self):
        rng = np.random.default_rng(37)
        x, y = rng.normal(0, 1, (2, N))
        data = np.vstack([x, y, x + y, x - 2 * y])
        _, g2 = multiple_coherence(data, FS, nperseg=256)
        assert np.all(g2 >= 0.0) and np.all(g2 <= 1.0)

    def test_too_few_segments_raises(self):
        rng = np.random.default_rng(38)
        data = rng.normal(0, 1, (4, N))
        with pytest.raises(ValueError, match="no residual degrees of freedom"):
            multiple_coherence(data, FS, nperseg=N // 2)

    def test_boundary_is_n_channels_not_below_it(self):
        """
        n_segments == n_channels is refused too: the matrix is invertible
        again, but the fit has no residual degrees of freedom.
        """
        rng = np.random.default_rng(41)
        data = rng.normal(0, 1, (4, N))
        with pytest.raises(ValueError):
            multiple_coherence(data, FS, nperseg=4000)      # 4 segments
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            multiple_coherence(data, FS, nperseg=3332)      # 5 segments: ok

    def test_rank_deficient_matrix_reads_one_everywhere(self):
        """
        The behaviour the raise prevents, pinned. Four independent channels
        over three segments give a singular CSD matrix, and every multiple
        coherence comes back at 1.0 — the bias floor q/n_d has reached 1.
        """
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, (4, N))
        _, G = psd_matrix(data, FS, nperseg=5000)           # 3 segments
        Ghat_reg, Ghat_inv = _normalised_csd_inverse(G, 1e-10)
        idx = np.arange(4)
        gamma2 = 1.0 - 1.0 / (
            np.real(Ghat_reg[:, idx, idx]) * np.real(Ghat_inv[:, idx, idx])
        )
        assert np.all(gamma2 > 0.999)

    def test_raise_message_names_channels_and_segments(self):
        rng = np.random.default_rng(39)
        data = rng.normal(0, 1, (5, N))
        with pytest.raises(ValueError) as exc:
            multiple_coherence(data, FS, nperseg=N)
        message = str(exc.value)
        assert "n_channels (5)" in message
        assert "1 Welch segment" in message
        assert "Fix: shorten nperseg" in message

    def test_zero_power_channel_does_not_blow_up(self):
        """A dead channel has no relationship with anything, and no NaNs."""
        rng = np.random.default_rng(40)
        x, y = rng.normal(0, 1, (2, N))
        data = np.vstack([x, y, np.zeros(N)])
        _, g2 = multiple_coherence(data, FS, nperseg=256)
        assert np.all(np.isfinite(g2))
        assert np.median(g2[2]) < 0.1


# ---------------------------------------------------------------------------
# partial_coherence
# ---------------------------------------------------------------------------

class TestPartialCoherence:
    def test_output_shape_and_symmetry(self):
        rng = np.random.default_rng(50)
        data = rng.normal(0, 1, (3, N))
        freqs, C = partial_coherence(data, FS, nperseg=256)
        assert C.shape == (3, 3, len(freqs))
        for k in range(0, C.shape[2], 10):
            np.testing.assert_allclose(C[:, :, k], C[:, :, k].T, atol=1e-10)

    def test_diagonal_is_one(self):
        rng = np.random.default_rng(51)
        data = rng.normal(0, 1, (3, N))
        _, C = partial_coherence(data, FS, nperseg=256)
        for i in range(3):
            np.testing.assert_allclose(C[i, i, :], 1.0, atol=1e-12)

    def test_two_channels_reduce_to_coherence_matrix(self):
        """Nothing to condition on, so it must agree with the pairwise form."""
        rng = np.random.default_rng(52)
        x = rng.normal(0, 1, N)
        y = 0.5 * x + rng.normal(0, 1, N)
        data = np.vstack([x, y])
        _, C = partial_coherence(data, FS, nperseg=256)
        _, Cm = coherence_matrix(data, FS, nperseg=256)
        np.testing.assert_allclose(C, Cm, atol=1e-8)

    def test_mediated_relationship_is_separated_from_a_direct_one(self):
        """
        x -> y -> z. Ordinary coherence of x and z is high because z carries
        a filtered copy of x; partial coherence given y falls to the floor,
        because y is where the path goes.
        """
        rng = np.random.default_rng(53)
        x = rng.normal(0, 1, N)
        y = np.convolve(x, np.ones(8) / 8, "same") + 0.02 * rng.normal(0, 1, N)
        z = np.convolve(y, np.ones(8) / 8, "same") + 0.02 * rng.normal(0, 1, N)
        data = np.vstack([x, y, z])

        _, Cm = partial_coherence(data, FS, nperseg=256)
        _, Cord = coherence_matrix(data, FS, nperseg=256)
        band = slice(1, 60)          # below the smoother's first null

        assert np.median(Cord[0, 2, band]) > 0.8      # x and z look related
        assert np.median(Cm[0, 2, band]) < 0.1        # ... only through y
        assert np.median(Cm[0, 1, band]) > 0.8        # x and y: direct, stays

    def test_values_stay_in_range(self):
        rng = np.random.default_rng(54)
        x, y = rng.normal(0, 1, (2, N))
        data = np.vstack([x, y, x + y, x - 2 * y])
        _, C = partial_coherence(data, FS, nperseg=256)
        assert np.all(C >= 0.0) and np.all(C <= 1.0)

    def test_too_few_segments_raises(self):
        rng = np.random.default_rng(55)
        data = rng.normal(0, 1, (4, N))
        with pytest.raises(ValueError, match="no residual degrees of freedom"):
            partial_coherence(data, FS, nperseg=N // 2)

    def test_few_segments_warns(self):
        rng = np.random.default_rng(56)
        data = rng.normal(0, 1, (3, N))
        with pytest.warns(UserWarning, match="Welch segments"):
            partial_coherence(data, FS, nperseg=N // 6)
