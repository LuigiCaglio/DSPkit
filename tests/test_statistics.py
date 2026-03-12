"""Tests for dspkit.statistics."""

import numpy as np
import pytest

from dspkit.statistics import (
    pdf_estimate,
    histogram,
    joint_histogram,
    covariance_matrix,
    mahalanobis,
)


N = 50_000


class TestPdfEstimate:
    def test_gaussian_peak_near_mean(self):
        """KDE of Gaussian data should peak near the mean."""
        rng = np.random.default_rng(0)
        x = rng.normal(5.0, 1.0, N)
        xi, density = pdf_estimate(x)
        peak_x = xi[np.argmax(density)]
        assert abs(peak_x - 5.0) < 0.2

    def test_integrates_to_one(self):
        """PDF should integrate to approximately 1."""
        rng = np.random.default_rng(1)
        x = rng.normal(0, 1, N)
        xi, density = pdf_estimate(x, n_points=512)
        dx = xi[1] - xi[0]
        integral = np.sum(density) * dx
        assert abs(integral - 1.0) < 0.05

    def test_nonnegative(self):
        rng = np.random.default_rng(2)
        x = rng.normal(0, 1, N)
        _, density = pdf_estimate(x)
        assert np.all(density >= 0)


class TestHistogram:
    def test_density_integrates_to_one(self):
        rng = np.random.default_rng(3)
        x = rng.normal(0, 1, N)
        centres, counts = histogram(x, bins=100, density=True)
        dx = centres[1] - centres[0]
        assert abs(np.sum(counts) * dx - 1.0) < 0.05

    def test_output_length(self):
        rng = np.random.default_rng(4)
        x = rng.normal(0, 1, 1000)
        centres, counts = histogram(x, bins=30)
        assert len(centres) == 30
        assert len(counts) == 30


class TestJointHistogram:
    def test_output_shapes(self):
        rng = np.random.default_rng(5)
        x = rng.normal(0, 1, N)
        y = rng.normal(0, 1, N)
        xc, yc, H = joint_histogram(x, y, bins=40)
        assert len(xc) == 40
        assert len(yc) == 40
        assert H.shape == (40, 40)

    def test_nonnegative(self):
        rng = np.random.default_rng(6)
        x = rng.normal(0, 1, N)
        y = rng.normal(0, 1, N)
        _, _, H = joint_histogram(x, y)
        assert np.all(H >= 0)


class TestCovarianceMatrix:
    def test_diagonal_is_variance(self):
        rng = np.random.default_rng(7)
        data = rng.normal(0, 2.0, (3, N))
        C = covariance_matrix(data)
        for i in range(3):
            assert abs(C[i, i] - 4.0) < 0.2  # var = σ² = 4

    def test_symmetric(self):
        rng = np.random.default_rng(8)
        data = rng.normal(0, 1, (4, N))
        C = covariance_matrix(data)
        np.testing.assert_allclose(C, C.T, atol=1e-10)

    def test_uncorrelated_offdiag_near_zero(self):
        rng = np.random.default_rng(9)
        data = rng.normal(0, 1, (3, N))
        C = covariance_matrix(data)
        off_diag = C[np.triu_indices(3, k=1)]
        assert np.all(np.abs(off_diag) < 0.05)


class TestMahalanobis:
    def test_outlier_has_large_distance(self):
        """An outlier should have a larger Mahalanobis distance."""
        rng = np.random.default_rng(10)
        normal_data = rng.normal(0, 1, (2, N))
        # Add an outlier at position 0
        test_data = normal_data.copy()
        test_data[:, 0] = [10.0, 10.0]  # far from the mean
        distances = mahalanobis(test_data)
        # The outlier should have the largest distance
        assert np.argmax(distances) == 0

    def test_centred_gaussian_chi_distributed(self):
        """Mahalanobis distances of standard Gaussian should follow chi distribution."""
        rng = np.random.default_rng(11)
        data = rng.normal(0, 1, (2, N))
        distances = mahalanobis(data)
        # Mean of chi distribution with k=2 DOF is sqrt(2) * Γ(3/2)/Γ(1) ≈ 1.25
        assert abs(distances.mean() - 1.41) < 0.20
