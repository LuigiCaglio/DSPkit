"""Tests for dspkit.statistics."""

import numpy as np
import pytest

from dspkit.statistics import (
    pdf_estimate,
    histogram,
    joint_histogram,
    covariance_matrix,
    mahalanobis,
    qq_normal,
    normality,
    mutual_information,
    mi_significance,
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


# ---------------------------------------------------------------------------
# qq_normal
# ---------------------------------------------------------------------------

class TestQQNormal:
    def test_line_recovers_mean_and_std(self):
        x = np.random.default_rng(200).normal(2.0, 3.0, 5000)
        theoretical, ordered, slope, intercept = qq_normal(x)
        assert abs(slope - 3.0) < 0.15
        assert abs(intercept - 2.0) < 0.15

    def test_output_shapes_and_ordering(self):
        x = np.random.default_rng(201).normal(0, 1, 500)
        theoretical, ordered, _, _ = qq_normal(x)
        assert theoretical.shape == ordered.shape == (500,)
        assert np.all(np.diff(theoretical) > 0)
        assert np.all(np.diff(ordered) >= 0)

    def test_normal_sample_is_straight(self):
        x = np.random.default_rng(202).normal(0, 1, 4000)
        theoretical, ordered, slope, intercept = qq_normal(x)
        residual = ordered - (slope * theoretical + intercept)
        assert np.max(np.abs(residual)) < 0.5

    def test_heavy_tails_curve_away_from_the_line(self):
        x = np.random.default_rng(203).standard_t(df=3, size=4000)
        theoretical, ordered, slope, intercept = qq_normal(x)
        residual = ordered - (slope * theoretical + intercept)
        # Tails high on the right, low on the left: the classic heavy-tail S.
        assert residual[-1] > 1.0
        assert residual[0] < -1.0

    def test_quartile_line_ignores_the_tails(self):
        """
        The reason the option exists: with heavy tails the OLS line is levered
        by the extremes, so its slope exceeds the bulk's spread.
        """
        x = np.random.default_rng(204).standard_t(df=3, size=4000)
        _, _, slope_ols, _ = qq_normal(x, line="ols")
        _, _, slope_q, _ = qq_normal(x, line="quartile")
        assert slope_ols > slope_q * 1.2

    def test_bad_line_raises(self):
        with pytest.raises(ValueError, match="ols"):
            qq_normal(np.arange(100.0), line="nonsense")

    def test_non_finite_dropped(self):
        x = np.array([1.0, 2.0, np.nan, 3.0, np.inf, 4.0])
        theoretical, ordered, _, _ = qq_normal(x)
        assert ordered.size == 4


# ---------------------------------------------------------------------------
# normality
# ---------------------------------------------------------------------------

class TestNormality:
    def test_reports_n_and_all_indicators(self):
        x = np.random.default_rng(210).normal(0, 1, 3000)
        r = normality(x)
        assert r["n"] == 3000
        for key in (
            "skewness", "excess_kurtosis", "dagostino_k2", "jarque_bera",
            "anderson_darling", "shapiro_wilk",
        ):
            assert key in r
            assert isinstance(r[key]["interpretation"], str)
            assert r[key]["interpretation"]

    def test_kurtosis_is_excess_not_raw(self):
        """0 for a normal, not 3 — the single most common misreading."""
        x = np.random.default_rng(211).normal(0, 1, 20000)
        r = normality(x)
        assert abs(r["excess_kurtosis"]["value"]) < 0.15
        assert "excess" in r["excess_kurtosis"]["interpretation"].lower()

    def test_effect_sizes_track_the_distribution(self):
        rng = np.random.default_rng(212)
        heavy = normality(rng.standard_t(df=4, size=20000))
        skewed = normality(rng.exponential(size=20000))
        assert heavy["excess_kurtosis"]["value"] > 1.0
        assert skewed["skewness"]["value"] > 1.0

    def test_large_n_marks_every_p_value_unreliable(self):
        """
        The point of the whole function: at 20480 samples a p-value is not a
        verdict, and the structure must say so rather than leaving it in the
        docstring.
        """
        x = np.random.default_rng(213).normal(0, 1, 20480)
        r = normality(x)
        for key in ("dagostino_k2", "jarque_bera", "anderson_darling"):
            assert r[key]["reliable"] is False
            assert "20480" in r[key]["interpretation"]
        assert "not informative" in r["summary"]

    def test_small_n_keeps_the_tests_reliable(self):
        x = np.random.default_rng(214).normal(0, 1, 800)
        r = normality(x)
        assert r["dagostino_k2"]["reliable"] is True
        assert r["anderson_darling"]["reliable"] is True
        assert r["shapiro_wilk"]["reliable"] is True

    def test_jarque_bera_marked_unreliable_below_its_asymptotic_range(self):
        x = np.random.default_rng(215).normal(0, 1, 300)
        r = normality(x)
        assert r["jarque_bera"]["reliable"] is False
        assert "asymptotic" in r["jarque_bera"]["interpretation"]

    def test_shapiro_subsampled_above_its_limit_and_says_so(self):
        x = np.random.default_rng(216).normal(0, 1, 20480)
        r = normality(x)
        sw = r["shapiro_wilk"]
        assert sw["subsampled"] is True
        assert sw["n_used"] == 5000
        assert "subsample" in sw["interpretation"]

    def test_shapiro_not_subsampled_when_short_enough(self):
        x = np.random.default_rng(217).normal(0, 1, 2000)
        sw = normality(x)["shapiro_wilk"]
        assert sw["subsampled"] is False
        assert sw["n_used"] == 2000

    def test_shapiro_subsample_is_deterministic_by_default(self):
        x = np.random.default_rng(218).normal(0, 1, 20000)
        assert (
            normality(x)["shapiro_wilk"]["pvalue"]
            == normality(x)["shapiro_wilk"]["pvalue"]
        )

    def test_shapiro_can_be_skipped(self):
        x = np.random.default_rng(219).normal(0, 1, 20000)
        sw = normality(x, shapiro_max_n=0)["shapiro_wilk"]
        assert sw["pvalue"] is None
        assert "skipped" in sw["interpretation"]

    def test_summary_carries_n_and_both_effect_sizes(self):
        x = np.random.default_rng(220).normal(0, 1, 4000)
        summary = normality(x)["summary"]
        assert "n = 4000" in summary
        assert "skew" in summary and "excess kurtosis" in summary

    def test_json_friendly_types(self):
        """The app serialises this; nothing exotic may leak out."""
        x = np.random.default_rng(221).normal(0, 1, 6000)
        r = normality(x)
        for key, value in r.items():
            if isinstance(value, dict):
                for field, item in value.items():
                    assert item is None or isinstance(
                        item, (float, int, bool, str, dict)
                    ), (key, field, type(item))
            else:
                assert isinstance(value, (int, str)), (key, type(value))

    def test_too_short_raises(self):
        with pytest.raises(ValueError, match="at least 8"):
            normality(np.arange(5.0))


# ---------------------------------------------------------------------------
# mutual_information
# ---------------------------------------------------------------------------

class TestMutualInformation:
    def test_independent_signals_are_near_zero(self):
        rng = np.random.default_rng(230)
        assert mutual_information(rng.normal(size=4000),
                                  rng.normal(size=4000)) < 0.05

    def test_dependence_increases_it(self):
        rng = np.random.default_rng(231)
        x = rng.normal(size=4000)
        weak = 0.3 * x + rng.normal(size=4000)
        strong = x + 0.1 * rng.normal(size=4000)
        assert (mutual_information(x, strong)
                > mutual_information(x, weak)
                > mutual_information(x, rng.normal(size=4000)))

    def test_catches_a_relationship_coherence_cannot(self):
        """
        y = x², which has essentially zero correlation and, being static and
        nonlinear, no coherence to speak of — but is fully determined by x.
        """
        rng = np.random.default_rng(232)
        x = rng.normal(size=4000)
        y = x ** 2 + 0.05 * rng.normal(size=4000)
        assert abs(np.corrcoef(x, y)[0, 1]) < 0.06
        assert mutual_information(x, y) > 0.5

    def test_symmetric(self):
        rng = np.random.default_rng(233)
        x = rng.normal(size=3000)
        y = 0.6 * x + rng.normal(size=3000)
        a = mutual_information(x, y)
        b = mutual_information(y, x)
        assert abs(a - b) < 0.05

    def test_lag_scan_finds_the_delay_with_the_library_sign_convention(self):
        """
        y[n] = x[n-25], so y is the delayed copy and the peak must sit at
        +25 — the same pairing as spectral.cross_correlation.
        """
        rng = np.random.default_rng(234)
        x = rng.normal(size=6000)
        y = np.roll(x, 25) + 0.05 * rng.normal(size=6000)
        lags = np.arange(-40, 41, 5)
        mi = mutual_information(x, y, lags=lags)
        assert mi.shape == lags.shape
        assert lags[int(np.argmax(mi))] == 25
        assert mutual_information(x, y, lags=0) < 0.1 * mi.max()

    def test_scalar_lag_returns_a_float(self):
        rng = np.random.default_rng(235)
        value = mutual_information(rng.normal(size=2000),
                                   rng.normal(size=2000), lags=3)
        assert isinstance(value, float)

    def test_lag_scan_uses_one_sample_count(self):
        """
        Every lag must be estimated from the same N, or the curve is not
        comparable with itself — the estimator's bias depends on N.
        """
        rng = np.random.default_rng(236)
        x = rng.normal(size=4000)
        y = rng.normal(size=4000)
        flat = mutual_information(x, y, lags=np.arange(0, 401, 100))
        assert np.ptp(flat) < 0.03

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="same length"):
            mutual_information(np.arange(100.0), np.arange(50.0))

    def test_lags_too_wide_for_the_record_raise(self):
        rng = np.random.default_rng(237)
        with pytest.raises(ValueError, match="usable samples"):
            mutual_information(rng.normal(size=100), rng.normal(size=100),
                               lags=np.arange(0, 99))

    def test_tied_values_do_not_break_the_estimator(self):
        """Quantised signals produce exact ties; jitter is what handles them."""
        rng = np.random.default_rng(238)
        x = np.round(rng.normal(size=3000))          # heavily quantised
        y = np.round(0.5 * x + rng.normal(size=3000))
        value = mutual_information(x, y)
        assert np.isfinite(value) and value > 0.0


# ---------------------------------------------------------------------------
# mi_significance
# ---------------------------------------------------------------------------

class TestMISignificance:
    def test_independent_signals_are_not_significant(self):
        rng = np.random.default_rng(240)
        r = mi_significance(rng.normal(size=2000), rng.normal(size=2000),
                            n_surrogates=19)
        assert r["p_value"] > 0.05
        assert "independence" in r["interpretation"]

    def test_real_dependence_is_significant(self):
        rng = np.random.default_rng(241)
        x = rng.normal(size=2000)
        y = x ** 2 + 0.1 * rng.normal(size=2000)
        r = mi_significance(x, y, n_surrogates=19)
        assert r["p_value"] <= 0.05
        assert r["mi"] > r["null_p95"]

    def test_null_floor_is_not_zero(self):
        """
        The reason the surrogate test exists: the estimator's own floor is a
        positive number, so a bare MI value cannot be read against zero.
        """
        rng = np.random.default_rng(242)
        r = mi_significance(rng.normal(size=2000), rng.normal(size=2000),
                            n_surrogates=19)
        assert r["null_mean"] > 0.0

    def test_shift_surrogates_keep_the_autocorrelation(self):
        """
        Permutation whitens the signal and lowers the null floor, so it
        declares dependence on an autocorrelated pair that shift surrogates
        correctly clear. Pinning the difference keeps 'shift' the default.
        """
        rng = np.random.default_rng(243)
        smoother = np.ones(20) / 20
        x = np.convolve(rng.normal(size=4000), smoother, "same")
        y = np.convolve(rng.normal(size=4000), smoother, "same")
        shift = mi_significance(x, y, n_surrogates=19, method="shift")
        perm = mi_significance(x, y, n_surrogates=19, method="permutation")
        assert shift["null_mean"] > perm["null_mean"]

    def test_p_value_cannot_be_zero(self):
        rng = np.random.default_rng(244)
        x = rng.normal(size=1500)
        r = mi_significance(x, x + 0.01 * rng.normal(size=1500),
                            n_surrogates=9)
        assert r["p_value"] == pytest.approx(1.0 / 10.0)

    def test_reports_the_lag_of_the_peak(self):
        rng = np.random.default_rng(245)
        x = rng.normal(size=3000)
        y = np.roll(x, 10) + 0.05 * rng.normal(size=3000)
        r = mi_significance(x, y, lags=np.arange(-20, 21, 5), n_surrogates=9)
        assert r["lag"] == 10
        assert r["p_value"] <= 0.1

    def test_bad_method_raises(self):
        rng = np.random.default_rng(246)
        with pytest.raises(ValueError, match="shift"):
            mi_significance(rng.normal(size=500), rng.normal(size=500),
                            n_surrogates=2, method="bootstrap")
