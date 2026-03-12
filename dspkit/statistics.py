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
"""

import numpy as np
from scipy import stats as _stats


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
