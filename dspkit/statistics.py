"""
Probability density and joint statistics for signal analysis.

Functions for estimating probability distributions and joint relationships
between signals — useful for characterising response distributions,
detecting non-Gaussianity, and understanding inter-channel dependencies.

Functions
---------
pdf_estimate        -- kernel density estimate of a signal's PDF
histogram           -- normalised histogram (empirical PDF)
joint_histogram     -- 2D histogram (empirical joint PDF)
covariance_matrix   -- covariance matrix for multi-channel data
mahalanobis         -- Mahalanobis distance (outlier detection)
qq_normal           -- data for a normal Q-Q plot
normality           -- normality indicators, with how to read them
mutual_information  -- nonlinear / lagged dependence (KSG estimator)
mi_significance     -- surrogate test for a mutual-information value
"""

import warnings
from typing import Literal

import numpy as np
from scipy import stats as _stats
from scipy.spatial import cKDTree as _cKDTree
from scipy.special import digamma as _digamma


def pdf_estimate(
    x: np.ndarray,
    n_points: int = 256,
    bandwidth: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Kernel density estimate (KDE) of a signal's probability density function.

    Uses a Gaussian kernel with automatic or user-specified bandwidth.

    Parameters
    ----------
    x : array_like, shape (N,)
        Signal samples.
    n_points : int
        Number of evaluation points (default 256).
    bandwidth : float or None
        KDE bandwidth (standard deviation of the Gaussian kernel).
        If ``None``, uses Scott's rule of thumb.

    Returns
    -------
    xi : ndarray, shape (n_points,)
        Evaluation points (range of ``x`` extended by 10 %).
    density : ndarray, shape (n_points,)
        Estimated PDF values.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.array([0.0]), np.array([0.0])

    kde = _stats.gaussian_kde(x, bw_method=bandwidth)
    margin = 0.1 * (x.max() - x.min()) if x.max() > x.min() else 1.0
    xi = np.linspace(x.min() - margin, x.max() + margin, n_points)
    density = kde(xi)
    return xi, density


def histogram(
    x: np.ndarray,
    bins: int | np.ndarray = 50,
    density: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalised histogram (empirical PDF approximation).

    Parameters
    ----------
    x : array_like, shape (N,)
        Signal samples.
    bins : int or array_like
        Number of bins or bin edges.
    density : bool
        If ``True`` (default), normalise so the histogram integrates to 1.

    Returns
    -------
    bin_centres : ndarray
        Centre of each bin.
    counts : ndarray
        Histogram values (probability density if ``density=True``).
    """
    x = np.asarray(x, dtype=float)
    counts, edges = np.histogram(x, bins=bins, density=density)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres, counts


def joint_histogram(
    x: np.ndarray,
    y: np.ndarray,
    bins: int | tuple[int, int] = 50,
    density: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    2D histogram (empirical joint PDF) of two signals.

    Parameters
    ----------
    x, y : array_like, shape (N,)
        Signal samples (must have equal length).
    bins : int or (int, int)
        Number of bins in each dimension.
    density : bool
        If ``True`` (default), normalise so the histogram integrates to 1.

    Returns
    -------
    x_centres : ndarray, shape (nx,)
        Bin centres along x.
    y_centres : ndarray, shape (ny,)
        Bin centres along y.
    H : ndarray, shape (nx, ny)
        Joint histogram values.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if isinstance(bins, int):
        bins = (bins, bins)

    H, xedges, yedges = np.histogram2d(x, y, bins=bins, density=density)
    x_centres = 0.5 * (xedges[:-1] + xedges[1:])
    y_centres = 0.5 * (yedges[:-1] + yedges[1:])
    return x_centres, y_centres, H


def covariance_matrix(
    data: np.ndarray,
    bias: bool = False,
) -> np.ndarray:
    """
    Covariance matrix for multi-channel data.

    Parameters
    ----------
    data : array_like, shape (n_channels, N)
        Each row is a time series from one sensor.
    bias : bool
        If ``False`` (default), normalise by N-1 (unbiased estimator).
        If ``True``, normalise by N (biased / maximum-likelihood estimator).

    Returns
    -------
    C : ndarray, shape (n_channels, n_channels)
        Covariance matrix. ``C[i, j]`` is the covariance between
        channels ``i`` and ``j``. Diagonal entries are variances.
    """
    data = np.atleast_2d(np.asarray(data, dtype=float))
    return np.cov(data, bias=bias)


def mahalanobis(
    data: np.ndarray,
    reference: np.ndarray | None = None,
) -> np.ndarray:
    """
    Mahalanobis distance of each time sample from the distribution centre.

    Useful for multivariate outlier detection in multi-channel SHM data.

    D_M(x) = sqrt( (x - μ)^T · Σ^{-1} · (x - μ) )

    Parameters
    ----------
    data : array_like, shape (n_channels, N)
        Multi-channel time series. Each column is one observation.
    reference : array_like, shape (n_channels, N_ref) or None
        Reference data to compute the mean and covariance from.
        If ``None``, uses ``data`` itself.

    Returns
    -------
    distances : ndarray, shape (N,)
        Mahalanobis distance of each time sample.
    """
    data = np.atleast_2d(np.asarray(data, dtype=float))
    n_ch, N = data.shape

    if reference is not None:
        ref = np.atleast_2d(np.asarray(reference, dtype=float))
    else:
        ref = data

    mu = ref.mean(axis=1, keepdims=True)
    cov = np.cov(ref)

    # Ensure cov is 2D for single-channel case
    if cov.ndim == 0:
        cov = cov.reshape(1, 1)

    # Regularise for numerical stability
    cov += np.eye(n_ch) * 1e-10 * np.trace(cov) / n_ch

    cov_inv = np.linalg.inv(cov)
    diff = data - mu  # (n_ch, N)

    # D² = diag( diff^T · Σ^{-1} · diff )
    d_sq = np.einsum("ij,ik,kj->j", diff, cov_inv, diff)
    return np.sqrt(np.maximum(d_sq, 0.0))


# ---------------------------------------------------------------------------
# Normality: how Gaussian is this signal?
# ---------------------------------------------------------------------------


def qq_normal(
    x: np.ndarray,
    line: Literal["ols", "quartile"] = "ols",
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Data for a normal Q-Q plot: theoretical quantiles, sample quantiles, line.

    Returns the numbers, not a plot, in keeping with the rest of the library —
    ``dspkit.plots`` wraps it for drawing. Plotting ``ordered`` against
    ``theoretical`` gives a straight line if the sample is normal; curvature
    at the ends is a tail departure, an S-shape is skew.

    Parameters
    ----------
    x : array_like, shape (N,)
        Signal samples. Non-finite values are dropped.
    line : {'ols', 'quartile'}
        How the reference line is fitted.
        ``'ols'`` (default) is least squares through every point, matching
        ``scipy.stats.probplot``.
        ``'quartile'`` passes through the first and third quartile points,
        matching R's ``qqline``.

    Returns
    -------
    theoretical : ndarray, shape (N,)
        Standard-normal quantiles at the plotting positions, ascending.
    ordered : ndarray, shape (N,)
        The sample sorted ascending — the matching empirical quantiles.
    slope : float
        Slope of the reference line. For a normal sample it estimates the
        standard deviation.
    intercept : float
        Intercept of the reference line. For a normal sample it estimates
        the mean.

    Notes
    -----
    Plotting positions are ``(k - a) / (N + 1 - 2a)`` with ``a = 3/8`` for
    ``N <= 10`` and ``a = 1/2`` above, the same convention scipy uses.

    **The reference line is not neutral, and this is the thing to know before
    reading a Q-Q plot as evidence.** A least-squares line is fitted to all N
    points, and the extreme order statistics are the most spread out along the
    x-axis, so they carry the most leverage: heavy tails tilt the line towards
    themselves and thereby hide part of their own departure. ``'quartile'``
    fits the bulk and lets the tails fall where they fall, which is what you
    want when the tails are the question. Neither line is a fit "to the
    normal distribution" — both are fitted to the data.

    What this will not tell you: whether a departure is large enough to
    matter, or whether it is real rather than sampling scatter. Q-Q plots of
    normal samples wander visibly at the ends even at large N, because the
    variance of an extreme order statistic is large. Read it alongside
    ``normality``, which puts numbers on the same departures.

    See Also
    --------
    normality

    Examples
    --------
    >>> import numpy as np
    >>> from dspkit.statistics import qq_normal
    >>> x = np.random.default_rng(0).normal(2.0, 3.0, 5000)
    >>> t, o, slope, intercept = qq_normal(x)
    >>> bool(abs(slope - 3.0) < 0.2), bool(abs(intercept - 2.0) < 0.2)
    (True, True)
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    ordered = np.sort(x)
    n = ordered.size
    if n < 2:
        return np.zeros(n), ordered, np.nan, np.nan

    a = 3.0 / 8.0 if n <= 10 else 0.5
    positions = (np.arange(1, n + 1) - a) / (n + 1 - 2 * a)
    theoretical = _stats.norm.ppf(positions)

    if line == "quartile":
        # Through the two quartile points, as R's qqline does.
        xq = _stats.norm.ppf([0.25, 0.75])
        yq = np.quantile(ordered, [0.25, 0.75])
        slope = float((yq[1] - yq[0]) / (xq[1] - xq[0]))
        intercept = float(yq[0] - slope * xq[0])
    elif line == "ols":
        slope, intercept = np.polyfit(theoretical, ordered, 1)
        slope, intercept = float(slope), float(intercept)
    else:
        raise ValueError(f"line must be 'ols' or 'quartile', got {line!r}")

    return theoretical, ordered, slope, intercept


def _skew_note(s: float, se: float) -> str:
    """One factual sentence on a skewness value."""
    side = "a longer left tail" if s < 0 else "a longer right tail"
    mag = abs(s)
    if mag < 0.5:
        size = "close to symmetric"
    elif mag < 1.0:
        size = "moderately skewed"
    else:
        size = "strongly skewed"
    detected = "larger than" if mag > 2 * se else "within"
    return (
        f"Skew {s:+.3f} ({size}); the sign means {side}. "
        f"That is {detected} twice the sampling standard error ({se:.3f}), "
        f"which assumes independent samples and so understates the error for "
        f"a serially correlated record."
    )


def _kurtosis_note(k: float, se: float) -> str:
    """One factual sentence on an excess-kurtosis value."""
    if k > 3.0:
        size = (
            "strongly impulsive: a few large excursions carry much of the "
            "variance, as with impacts, rattle, or a developing fault"
        )
    elif k > 1.0:
        size = "heavier tails than a normal, i.e. some impulsive content"
    elif k > 0.5:
        size = "slightly heavier tails than a normal"
    elif k < -1.0:
        size = (
            "much flatter than a normal, which usually means a bounded or "
            "clipped response rather than a physical effect"
        )
    elif k < -0.5:
        size = "slightly flatter tails than a normal"
    else:
        size = "tails close to a normal"
    return (
        f"Excess kurtosis {k:+.3f} — excess, so 0 is normal and 3 is not "
        f"({size}). Sampling standard error {se:.3f} under independence, "
        f"and kurtosis is the more fragile of the two moments: it is driven "
        f"by the few largest samples, so one spike or dropout moves it."
    )


def _test_note(name: str, p: float | None, n: int, reliable: bool, why: str) -> str:
    """One factual sentence on a normality test result at this sample size."""
    if not reliable:
        return f"{name}: {why}"
    if p is None:
        return f"{name}: {why}"
    if p < 0.05:
        return (
            f"{name}: p = {p:.3g} at n = {n}, so the departure from normality "
            f"is larger than sampling scatter. The test does not say how "
            f"large — read the skewness and excess kurtosis for that."
        )
    return (
        f"{name}: p = {p:.3g} at n = {n}, so no departure is detectable at "
        f"this sample size. That is not evidence of normality, only absence "
        f"of evidence against it."
    )


def normality(
    x: np.ndarray,
    shapiro_max_n: int = 5000,
    large_n: int = 5000,
    seed: int | None = 0,
) -> dict:
    """
    Normality indicators with the guidance needed to read them.

    Returns effect sizes (skewness, excess kurtosis) and the four standard
    tests — D'Agostino K², Jarque-Bera, Anderson-Darling, Shapiro-Wilk — each
    with a plain-language interpretation and an explicit statement of whether
    it can be trusted at this sample size.

    Parameters
    ----------
    x : array_like, shape (N,)
        Signal samples. Non-finite values are dropped.
    shapiro_max_n : int
        Shapiro-Wilk is computed on a random subsample of at most this many
        points (default 5000); above it the test's own approximation degrades.
        Pass 0 to skip Shapiro-Wilk entirely.
    large_n : int
        Sample size above which p-values are reported as uninformative
        (default 5000). See Notes.
    seed : int or None
        Seed for the Shapiro-Wilk subsample. Defaults to 0 so that repeated
        calls on the same record agree; pass ``None`` for a fresh draw.

    Returns
    -------
    result : dict
        ``{'n': int, 'summary': str}`` plus one sub-dict per indicator, each
        carrying an ``'interpretation'`` string::

            'skewness'         {'value', 'se', 'interpretation'}
            'excess_kurtosis'  {'value', 'se', 'interpretation'}
            'dagostino_k2'     {'statistic', 'pvalue', 'reliable',
                                'interpretation'}
            'jarque_bera'      {'statistic', 'pvalue', 'reliable',
                                'interpretation'}
            'anderson_darling' {'statistic', 'pvalue', 'critical_values',
                                'reject_5pct', 'reliable', 'interpretation'}
            'shapiro_wilk'     {'statistic', 'pvalue', 'n_used',
                                'subsampled', 'reliable', 'interpretation'}

        Anderson-Darling reports ``'critical_values'`` with ``'pvalue'``
        ``None`` where scipy still provides them (it has critical values
        rather than a p-value); on scipy versions that have dropped them it
        reports an interpolated ``'pvalue'`` and an empty
        ``'critical_values'``. Every Shapiro-Wilk field is ``None`` when it
        was skipped. All values are plain floats, ints, bools, strings and
        dicts.

    Notes
    -----
    **At a realistic record length every one of these tests rejects, and that
    is a property of the test, not of the signal.** A normality test asks "is
    the departure larger than sampling scatter?", and sampling scatter shrinks
    as 1/sqrt(N). With 20 000 samples a skew of 0.03 — visually and physically
    nothing — is detected with certainty, so p < 1e-16 is the expected result
    for real data and carries no information about how non-Gaussian the signal
    is. Above ``large_n`` every test is therefore marked
    ``reliable=False``, and the effect sizes are what remain worth reading:
    they are estimates of a fixed property of the distribution, and they get
    *better*, not more damning, as N grows.

    The tests also assume independent samples, which a vibration record is
    not. Serial correlation reduces the effective sample size, so the true
    scatter is wider than any of these p-values or the reported standard
    errors assume. The direction of that error is towards over-rejection,
    on top of the large-N effect above.

    **Shapiro-Wilk** is the most powerful of the four at small N and the
    least usable at large N: its approximation is documented as unreliable
    much beyond 5000 samples. Rather than skip it or report a number known to
    be wrong, it is computed on a random subsample of ``shapiro_max_n``
    points, and ``'subsampled'`` and ``'n_used'`` say so. Subsampling costs
    power and does not repair the independence assumption.

    What this will not tell you: whether the signal is stationary (a record
    that switches between two Gaussian regimes is non-Gaussian overall while
    every part of it is Gaussian), whether non-normality means damage, or
    which samples caused it. For a signal that is Gaussian in the bulk and
    not in the tails, ``qq_normal`` shows where the departure sits, which no
    scalar here does.

    See Also
    --------
    qq_normal, dspkit.indicators.kurtosis, dspkit.indicators.skewness

    Examples
    --------
    >>> import numpy as np
    >>> from dspkit.statistics import normality
    >>> x = np.random.default_rng(0).standard_t(df=4, size=4000)
    >>> r = normality(x)
    >>> r['n']
    4000
    >>> bool(r['excess_kurtosis']['value'] > 1.0)     # t(4) has heavy tails
    True
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n < 8:
        raise ValueError(
            f"normality: n={n} after dropping non-finite samples; at least 8 "
            f"are needed for any of these indicators to be defined."
        )

    skew = float(_stats.skew(x, bias=False))
    exkurt = float(_stats.kurtosis(x, fisher=True, bias=False))

    # Sampling standard errors under independence and normality.
    se_skew = float(np.sqrt(6.0 * n * (n - 1) / ((n - 2) * (n + 1) * (n + 3))))
    se_kurt = float(
        2.0 * se_skew * np.sqrt((n * n - 1.0) / ((n - 3.0) * (n + 5.0)))
    )

    big = n > large_n
    big_why = (
        f"n = {n} is above large_n = {large_n}. At this length the test "
        f"rejects on departures too small to matter, so the p-value is not "
        f"a verdict on normality — read the effect sizes above."
    )

    result: dict = {
        "n": n,
        "skewness": {
            "value": skew,
            "se": se_skew,
            "interpretation": _skew_note(skew, se_skew),
        },
        "excess_kurtosis": {
            "value": exkurt,
            "se": se_kurt,
            "interpretation": _kurtosis_note(exkurt, se_kurt),
        },
    }

    # --- D'Agostino K² (skew + kurtosis combined) --------------------------
    if n >= 20:
        k2, p_k2 = _stats.normaltest(x)
        k2_reliable, k2_why = (not big), big_why
    else:
        k2, p_k2 = np.nan, np.nan
        k2_reliable = False
        k2_why = f"n = {n} < 20; scipy's K² approximation is not valid here."
    result["dagostino_k2"] = {
        "statistic": float(k2),
        "pvalue": float(p_k2),
        "reliable": bool(k2_reliable),
        "interpretation": _test_note(
            "D'Agostino K²", float(p_k2), n, k2_reliable, k2_why
        ),
    }

    # --- Jarque-Bera -------------------------------------------------------
    jb, p_jb = _stats.jarque_bera(x)
    jb_reliable = (not big) and n >= 2000
    if big:
        jb_why = big_why
    else:
        jb_why = (
            f"n = {n}: the chi² null distribution Jarque-Bera assumes is "
            f"asymptotic and is poor below a few thousand samples, where the "
            f"test rejects too rarely."
        )
    result["jarque_bera"] = {
        "statistic": float(jb),
        "pvalue": float(p_jb),
        "reliable": bool(jb_reliable),
        "interpretation": _test_note(
            "Jarque-Bera", float(p_jb), n, jb_reliable, jb_why
        ),
    }

    # --- Anderson-Darling --------------------------------------------------
    # Critical values are preferred over scipy's interpolated p-value, which
    # is clamped to the tabulated range (0.01 to 0.15) and so saturates on
    # exactly the records where the answer is interesting. SciPy 1.17
    # deprecated the critical-value form; fall back when it goes.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            ad = _stats.anderson(x, dist="norm")
        crit = {
            float(level): float(value)
            for level, value in zip(ad.significance_level, ad.critical_values)
        }
        p_ad = None
    except (AttributeError, TypeError, ValueError):
        ad = _stats.anderson(x, dist="norm", method="interpolate")
        crit = {}
        p_ad = float(ad.pvalue)

    if crit:
        crit_5 = crit.get(5.0, float("nan"))
        reject_5 = bool(ad.statistic > crit_5)
        detail = (
            f"A² = {float(ad.statistic):.3f} against the 5 % critical value "
            f"{crit_5:.3f} (mean and variance estimated from the data)."
        )
    else:
        reject_5 = bool(p_ad < 0.05)
        detail = (
            f"A² = {float(ad.statistic):.3f}, p = {p_ad:.3g} — interpolated "
            f"from scipy's tables and clamped to their range, so the extreme "
            f"values mean 'at least this far out', not the number shown."
        )
    ad_why = big_why if big else (
        detail + " Anderson-Darling weights the tails, so it responds to the "
        "departure that matters most for peak and fatigue estimates."
    )
    result["anderson_darling"] = {
        "statistic": float(ad.statistic),
        "pvalue": p_ad,
        "critical_values": crit,
        "reject_5pct": reject_5,
        "reliable": bool(not big),
        "interpretation": _test_note(
            "Anderson-Darling", None, n, not big, ad_why
        ),
    }

    # --- Shapiro-Wilk ------------------------------------------------------
    if shapiro_max_n and shapiro_max_n > 2:
        if n > shapiro_max_n:
            rng = np.random.default_rng(seed)
            sample = x[rng.choice(n, size=shapiro_max_n, replace=False)]
            subsampled = True
        else:
            sample = x
            subsampled = False
        sw, p_sw = _stats.shapiro(sample)
        n_used = int(sample.size)
        sw_reliable = n_used <= large_n
        if subsampled:
            sw_why = (
                f"Computed on a random subsample of {n_used} of {n} samples, "
                f"because the Shapiro-Wilk approximation degrades beyond "
                f"about 5000. The p-value therefore describes the subsample, "
                f"and the subsample is not independent — it is drawn from a "
                f"serially correlated record."
            )
            if sw_reliable:
                sw_note = _test_note(
                    "Shapiro-Wilk", float(p_sw), n_used, True, sw_why
                ) + " " + sw_why
            else:
                sw_note = _test_note(
                    "Shapiro-Wilk", float(p_sw), n_used, False, sw_why
                )
        else:
            sw_note = _test_note(
                "Shapiro-Wilk", float(p_sw), n_used, sw_reliable, big_why
            )
        result["shapiro_wilk"] = {
            "statistic": float(sw),
            "pvalue": float(p_sw),
            "n_used": n_used,
            "subsampled": bool(subsampled),
            "reliable": bool(sw_reliable),
            "interpretation": sw_note,
        }
    else:
        result["shapiro_wilk"] = {
            "statistic": None,
            "pvalue": None,
            "n_used": 0,
            "subsampled": False,
            "reliable": False,
            "interpretation": (
                "Shapiro-Wilk: skipped (shapiro_max_n=0)."
            ),
        }

    # --- one-line summary --------------------------------------------------
    if abs(skew) < 0.5 and abs(exkurt) < 0.5:
        shape = "close to Gaussian in shape"
    elif abs(exkurt) >= abs(skew):
        shape = (
            "heavier-tailed than Gaussian" if exkurt > 0
            else "flatter-tailed than Gaussian"
        )
    else:
        shape = "skewed relative to a Gaussian"

    p_values = [
        d["pvalue"] for d in (
            result["dagostino_k2"], result["jarque_bera"],
            result["shapiro_wilk"],
        )
        if d["pvalue"] is not None and np.isfinite(d["pvalue"])
    ]
    n_reject = int(sum(p < 0.05 for p in p_values)) + int(reject_5)
    n_tests = len(p_values) + 1
    if big:
        tests_line = (
            f"{n_reject} of {n_tests} tests reject at 5 %, which at n = {n} "
            f"is expected whatever the data and is not informative."
        )
    else:
        tests_line = f"{n_reject} of {n_tests} tests reject at 5 %."

    result["summary"] = (
        f"n = {n}; skew {skew:+.3f}, excess kurtosis {exkurt:+.3f} — "
        f"{shape}. {tests_line}"
    )
    return result


# ---------------------------------------------------------------------------
# Mutual information — dependence that coherence cannot see
# ---------------------------------------------------------------------------


def _ksg_mi(x: np.ndarray, y: np.ndarray, k: int) -> float:
    """
    Kraskov-Stogbauer-Grassberger mutual-information estimate (algorithm 1).

    ``I = psi(k) + psi(N) - <psi(nx + 1) + psi(ny + 1)>``, with neighbour
    counts taken in the max norm and the k-th joint neighbour distance as the
    per-point radius. Returns nats, floored at 0 — the estimator has no
    non-negativity guarantee and returns small negative values for
    independent variables.
    """
    n = int(x.size)
    z = np.column_stack([x, y])
    tree = _cKDTree(z)
    # k + 1 because the first neighbour of every point is itself.
    dist, _ = tree.query(z, k=k + 1, p=np.inf, workers=-1)
    radius = np.nextafter(dist[:, -1], 0)  # strict inequality, as KSG defines

    counts = []
    for v in (x, y):
        marginal = _cKDTree(v[:, None])
        c = marginal.query_ball_point(
            v[:, None], radius, p=np.inf, return_length=True
        )
        counts.append(np.asarray(c, dtype=float) - 1.0)  # drop the self-match

    mi = (
        _digamma(k) + _digamma(n)
        - np.mean(_digamma(counts[0] + 1.0) + _digamma(counts[1] + 1.0))
    )
    return float(max(mi, 0.0))


def _prepare_mi(
    x: np.ndarray,
    y: np.ndarray,
    standardize: bool,
    jitter: bool,
    seed: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Standardise and jitter a pair of signals for the KSG estimator."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size:
        raise ValueError(
            f"x and y must have the same length, got {x.size} and {y.size}"
        )

    out = []
    rng = np.random.default_rng(seed)
    for v in (x, y):
        v = v - v.mean()
        scale = v.std()
        if standardize and scale > 0:
            v = v / scale
            scale = 1.0
        if jitter:
            # KSG assumes continuous variables; exact ties (quantised ADC
            # output, a clipped signal) break the neighbour counts. The noise
            # is 1e-10 of the spread — far below any measurement resolution,
            # far above float64 spacing.
            v = v + 1e-10 * max(scale, 1e-300) * rng.standard_normal(v.size)
        out.append(v)
    return out[0], out[1]


def _lag_windows(n: int, lags: np.ndarray) -> tuple[int, int]:
    """Common start index and length so every lag uses the same sample count."""
    lo = int(min(lags.min(), 0))
    hi = int(max(lags.max(), 0))
    start = -lo
    length = n - hi + lo
    if length < 16:
        raise ValueError(
            f"mutual_information: lags spanning {lo}..{hi} leave only "
            f"{length} usable samples of {n}; use smaller lags or a longer "
            f"record."
        )
    return start, length


def mutual_information(
    x: np.ndarray,
    y: np.ndarray,
    k: int = 3,
    lags: int | np.ndarray = 0,
    standardize: bool = True,
    jitter: bool = True,
    seed: int | None = 0,
) -> float | np.ndarray:
    """
    Mutual information between two signals, by k-nearest-neighbour estimation.

    I(X;Y) in nats — the dependence between two channels *of any shape*, not
    only the linear, same-frequency dependence coherence measures. It is zero
    if and only if the two are independent, which makes it the right tool for
    a relationship that is nonlinear (a rectifying joint, a rattle that
    responds to amplitude rather than to phase) or that lives at a lag.

    Uses the Kraskov-Stogbauer-Grassberger estimator (algorithm 1): binned
    estimators are badly biased at realistic sample sizes because the answer
    depends on a bin width nobody can choose defensibly.

    Parameters
    ----------
    x, y : array_like, shape (N,)
        The two signals, equal length.
    k : int
        Number of nearest neighbours (default 3). Small k means low bias and
        high variance; k = 3 or 4 is the usual compromise.
    lags : int or array_like
        Lag(s) in samples. MI is computed between ``x[n]`` and ``y[n + lag]``,
        the same pairing as ``dspkit.spectral.cross_correlation``: a peak at
        lag ``+d`` means ``y`` is the delayed copy (``y[n] = x[n-d]``), so x
        leads y by d samples.
    standardize : bool
        Scale each signal to unit variance first (default True). MI is
        invariant to this in theory; the estimator is not, because its
        neighbourhoods are squares in the joint (x, y) plane, so without it
        the answer depends on the relative units of the two channels.
    jitter : bool
        Add noise at 1e-10 of the signal spread to break exact ties
        (default True). Quantised or clipped signals otherwise produce
        degenerate neighbour counts.
    seed : int or None
        Seed for the jitter, so repeated calls agree. ``None`` for a fresh
        draw.

    Returns
    -------
    mi : float or ndarray
        Mutual information in nats — a float if ``lags`` is a scalar, an
        array of shape ``(len(lags),)`` otherwise. Divide by ``log(2)`` for
        bits. Floored at 0.

    Notes
    -----
    **Lags.** MI has no frequency decomposition, so a lagged relationship is
    invisible unless you look for it: at lag 0 a pure delay of half a period
    can read as independence. Pass a range of ``lags`` and read the curve.
    Every lag in a scan is estimated from the *same* number of samples — the
    window is shrunk once, by the full span of the requested lags — because
    the estimator's bias depends on N, and a curve whose sample count varies
    along it is not comparable with itself.

    Two things the lag scan does not fix. The maximum over a scan of L lags
    is a biased estimate of the maximum: scanning inflates it, and the
    inflation grows with L, so ``max(mi)`` from a wide scan is not comparable
    with a single-lag value. And a lag scan finds a *fixed* delay; a
    relationship spread over many lags (any filtered path) shows up weakly at
    each one.

    **Significance.** The number this returns is not a verdict, and it has no
    natural scale to compare against: unlike coherence there is no [0, 1]
    range and no fixed value that means "independent". The estimate for truly
    independent signals is a small positive number that depends on N and k,
    so "MI = 0.02 nats" is meaningless on its own. Establishing whether a
    value is more than that floor requires a null distribution built from the
    same data — use ``mi_significance``, which does it with time-shifted
    surrogates. **A bare MI value with no surrogate test establishes
    nothing.**

    What this will not tell you: direction or causality (MI is symmetric —
    ``I(X;Y) = I(Y;X)``, and a lag peak is evidence of order, not of cause),
    which frequency the dependence lives at, or the shape of the
    relationship. It is also O(N log N) per lag with a KD-tree, so a wide
    scan on a long record is seconds, not milliseconds.

    See Also
    --------
    mi_significance, dspkit.multisensor.partial_coherence,
    dspkit.spectral.coherence

    Examples
    --------
    >>> import numpy as np
    >>> from dspkit.statistics import mutual_information
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(size=4000)
    >>> quad = x ** 2 + 0.1 * rng.normal(size=4000)   # zero correlation
    >>> bool(abs(np.corrcoef(x, quad)[0, 1]) < 0.05)
    True
    >>> bool(mutual_information(x, quad) > 0.3)       # but not independent
    True
    """
    scalar = np.isscalar(lags)
    lag_array = np.atleast_1d(np.asarray(lags, dtype=int))
    xs, ys = _prepare_mi(x, y, standardize, jitter, seed)
    start, length = _lag_windows(xs.size, lag_array)

    out = np.empty(lag_array.size, dtype=float)
    for i, lag in enumerate(lag_array):
        xi = xs[start:start + length]
        yi = ys[start + lag:start + lag + length]
        out[i] = _ksg_mi(xi, yi, k)

    return float(out[0]) if scalar else out


def mi_significance(
    x: np.ndarray,
    y: np.ndarray,
    k: int = 3,
    lags: int | np.ndarray = 0,
    n_surrogates: int = 199,
    method: Literal["shift", "permutation"] = "shift",
    standardize: bool = True,
    jitter: bool = True,
    seed: int | None = 0,
) -> dict:
    """
    Is a mutual-information value more than the estimator's own floor?

    Builds a null distribution by recomputing MI between ``x`` and surrogate
    versions of ``y`` that carry no relationship to ``x``, and reports where
    the observed value falls in it. This is what turns an MI number into a
    statement; without it, ``mutual_information`` returns a quantity with no
    reference point.

    Parameters
    ----------
    x, y : array_like, shape (N,)
        The two signals, equal length.
    k : int
        Neighbours for the KSG estimator (default 3).
    lags : int or array_like
        Lag or lags in samples, as in ``mutual_information``. If a range is
        given, the statistic is the maximum over the scan, and **each
        surrogate is scanned the same way** so the null carries the same
        selection bias as the observation.
    n_surrogates : int
        Number of surrogates (default 199, giving a smallest attainable
        p-value of 1/200 = 0.005). Cost is ``n_surrogates × len(lags)``
        estimator calls.
    method : {'shift', 'permutation'}
        How the surrogates are built. See Notes — the default matters.
    standardize, jitter, seed
        As in ``mutual_information``. ``seed`` also seeds the surrogates.

    Returns
    -------
    result : dict
        ``'mi'`` (float, observed value in nats; the max over the scan),
        ``'lag'`` (int, the lag it occurred at),
        ``'p_value'`` (float),
        ``'null_mean'``, ``'null_p95'`` (float, the estimator's floor and its
        95th percentile under the null),
        ``'null_distribution'`` (ndarray, shape ``(n_surrogates,)``),
        ``'n_samples'``, ``'n_surrogates'``, ``'k'``, ``'method'``,
        ``'interpretation'`` (str).

    Notes
    -----
    **Time-shifted surrogates are the default, and permutation is not.**
    Permuting y destroys its autocorrelation as well as its relationship to
    x, so the null it builds is the null for "y is white noise", not for "y
    is unrelated to x". Real signals are strongly autocorrelated, the KSG
    floor rises with autocorrelation, and testing against a white-noise null
    therefore declares dependence far too readily. A circular time shift
    keeps each signal's own spectrum, distribution and autocorrelation
    exactly, and only destroys the alignment between them, which is the one
    thing under test. Shifts are drawn to exceed the widest requested lag, so
    a surrogate cannot accidentally re-align the pair.

    ``method='permutation'`` is available for the case where the samples
    genuinely are independent draws rather than a time series. Do not use it
    on a vibration record.

    ``p = (1 + #{null >= observed}) / (1 + n_surrogates)``, the standard
    Monte-Carlo p-value, which cannot return 0: with 199 surrogates the
    smallest value it can report is 0.005, and that means "not exceeded",
    not "certain".

    What this will not tell you: it tests one null — that the two signals
    are unrelated given each one's own autocorrelation. Rejecting it does not
    identify the shape of the relationship, its direction, or whether a third
    channel drives both. A shared excitation makes every sensor pair
    dependent, and this will report exactly that, correctly and
    uninformatively.

    See Also
    --------
    mutual_information

    Examples
    --------
    >>> import numpy as np
    >>> from dspkit.statistics import mi_significance
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(size=2000)
    >>> r = mi_significance(x, x ** 2 + 0.1 * rng.normal(size=2000),
    ...                     n_surrogates=19)
    >>> bool(r['p_value'] < 0.06)
    True
    """
    lag_array = np.atleast_1d(np.asarray(lags, dtype=int))
    xs, ys = _prepare_mi(x, y, standardize, jitter, seed)
    n = xs.size
    start, length = _lag_windows(n, lag_array)

    def _scan(y_signal: np.ndarray) -> np.ndarray:
        return np.array([
            _ksg_mi(
                xs[start:start + length],
                y_signal[start + lag:start + lag + length],
                k,
            )
            for lag in lag_array
        ])

    observed = _scan(ys)
    best = int(np.argmax(observed))

    rng = np.random.default_rng(seed)
    guard = int(np.abs(lag_array).max()) + 1
    null = np.empty(n_surrogates, dtype=float)
    for s in range(n_surrogates):
        if method == "shift":
            # Beyond the widest lag at both ends, so no surrogate can happen
            # to sit back on the true alignment.
            shift = int(rng.integers(guard, max(n - guard, guard + 1)))
            surrogate = np.roll(ys, shift)
        elif method == "permutation":
            surrogate = rng.permutation(ys)
        else:
            raise ValueError(
                f"method must be 'shift' or 'permutation', got {method!r}"
            )
        null[s] = float(np.max(_scan(surrogate)))

    mi_obs = float(observed[best])
    p_value = float((1 + np.sum(null >= mi_obs)) / (1 + n_surrogates))
    null_mean = float(null.mean())
    null_p95 = float(np.percentile(null, 95))

    if p_value <= 0.05:
        verdict = (
            f"MI = {mi_obs:.4f} nats at lag {int(lag_array[best])} against a "
            f"null floor of {null_mean:.4f} (95th percentile {null_p95:.4f}); "
            f"p = {p_value:.3f}. The dependence is more than the estimator's "
            f"own bias, but its size in nats has no absolute meaning."
        )
    else:
        verdict = (
            f"MI = {mi_obs:.4f} nats at lag {int(lag_array[best])} is inside "
            f"the null distribution (mean {null_mean:.4f}, 95th percentile "
            f"{null_p95:.4f}); p = {p_value:.3f}. That is what independence "
            f"looks like at this N and k — the raw MI value is floor, not "
            f"signal."
        )

    return {
        "mi": mi_obs,
        "lag": int(lag_array[best]),
        "p_value": p_value,
        "null_mean": null_mean,
        "null_p95": null_p95,
        "null_distribution": null,
        "n_samples": int(length),
        "n_surrogates": int(n_surrogates),
        "k": int(k),
        "method": method,
        "interpretation": verdict,
    }
