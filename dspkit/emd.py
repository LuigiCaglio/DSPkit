"""
Empirical Mode Decomposition (EMD) and Hilbert-Huang Transform (HHT).

EMD adaptively decomposes a signal into Intrinsic Mode Functions (IMFs)
ordered from highest to lowest instantaneous frequency, plus a monotone
residue. Unlike Fourier or wavelet bases, IMFs are data-driven and
require no a-priori frequency resolution choice.

The HHT applies the Hilbert transform to each IMF to obtain time-varying
amplitude and frequency — a fully adaptive time-frequency representation
without the cross-term problem of the WVD.

Functions
---------
emd                     -- decompose signal into IMFs + residue
hht                     -- apply Hilbert to each IMF → envelopes, inst. freqs
hht_marginal_spectrum   -- time-averaged HHT energy as function of frequency

Reference
---------
Huang et al. (1998), "The empirical mode decomposition and the Hilbert
spectrum for nonlinear and non-stationary time series analysis",
Proc. R. Soc. London A, 454, 903-995.
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import argrelmax, argrelmin

from dspkit.instantaneous import hilbert_attributes


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _envelope(t: np.ndarray, x: np.ndarray, ext_idx: np.ndarray) -> np.ndarray:
    """
    Cubic spline through local extrema with mirror-boundary extension.

    The first and last extremum are reflected about the signal endpoints
    to keep the spline well-conditioned at the edges.
    """
    n = len(ext_idx)
    if n == 0:
        return np.zeros_like(x)
    if n == 1:
        return np.full_like(x, x[ext_idx[0]])

    te = t[ext_idx]
    xe = x[ext_idx]

    # Mirror the outermost extrema to anchor the spline at the boundaries
    t_l = 2.0 * t[0]  - te[0];   x_l = xe[0]
    t_r = 2.0 * t[-1] - te[-1];  x_r = xe[-1]

    te_ext = np.r_[t_l, te, t_r]
    xe_ext = np.r_[x_l, xe, x_r]

    cs = CubicSpline(te_ext, xe_ext)
    return cs(t)


def _sift(
    t: np.ndarray,
    x: np.ndarray,
    max_sifting: int,
    sd_threshold: float,
) -> np.ndarray:
    """
    Sifting loop: extract a single IMF candidate from x.

    Iterates until the SD stopping criterion is met or max_sifting
    iterations are exhausted.
    """
    h = x.copy()

    for _ in range(max_sifting):
        h_prev = h.copy()

        max_idx = argrelmax(h, order=1)[0]
        min_idx = argrelmin(h, order=1)[0]

        if len(max_idx) < 2 or len(min_idx) < 2:
            break  # not enough extrema — accept current h as IMF

        upper = _envelope(t, h, max_idx)
        lower = _envelope(t, h, min_idx)
        mean_env = (upper + lower) / 2.0

        h = h - mean_env

        # SD criterion (Huang 1998): stop when successive iterations agree
        denom = np.dot(h_prev, h_prev)
        if denom > 0 and np.dot(h_prev - h, h_prev - h) / denom < sd_threshold:
            break

    return h


def _has_enough_extrema(r: np.ndarray, min_extrema: int = 3) -> bool:
    """Return True if r has at least min_extrema local extrema."""
    return len(argrelmax(r, order=1)[0]) + len(argrelmin(r, order=1)[0]) >= min_extrema


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def emd(
    x: np.ndarray,
    max_imfs: int | None = None,
    max_sifting: int = 10,
    sd_threshold: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Empirical Mode Decomposition.

    Iteratively extracts Intrinsic Mode Functions (IMFs) from x using the
    sifting process. Each IMF satisfies:

    1. The number of extrema and zero-crossings differ by at most one.
    2. The mean of the upper and lower envelopes is everywhere near zero.

    IMFs are ordered from highest to lowest instantaneous frequency.
    Their sum plus the residue exactly reconstructs the original signal.

    Parameters
    ----------
    x : array_like, shape (N,)
        Input signal. Should be detrended beforehand if it has a strong
        polynomial trend (EMD handles mild trends via the residue).
    max_imfs : int or None
        Maximum number of IMFs to extract. ``None`` extracts all possible.
    max_sifting : int
        Maximum sifting iterations per IMF (default 10). Higher values
        enforce stricter IMF symmetry at the cost of computation.
    sd_threshold : float
        Sifting stops when the normalised squared difference between
        successive iterates falls below this threshold (default 0.2).

    Returns
    -------
    imfs : ndarray, shape (n_imfs, N)
        Extracted IMFs, ordered highest to lowest frequency.
    residue : ndarray, shape (N,)
        Monotone (or near-monotone) residue. Represents the overall trend.

    Notes
    -----
    Reconstruction is exact: ``imfs.sum(axis=0) + residue ≈ x``.
    """
    x = np.asarray(x, dtype=float)
    N = len(x)
    t = np.arange(N, dtype=float)

    imfs = []
    r = x.copy()

    while _has_enough_extrema(r):
        if max_imfs is not None and len(imfs) >= max_imfs:
            break
        imf = _sift(t, r, max_sifting=max_sifting, sd_threshold=sd_threshold)
        imfs.append(imf)
        r = r - imf

    if not imfs:
        # Signal is already monotone — return empty IMF array
        return np.empty((0, N)), r

    return np.array(imfs), r


def hht(
    imfs: np.ndarray,
    fs: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Hilbert-Huang Transform: apply the Hilbert transform to each IMF.

    Each IMF is a narrow-band AM-FM signal, so the Hilbert transform gives
    physically meaningful instantaneous amplitude and frequency.

    Parameters
    ----------
    imfs : array_like, shape (n_imfs, N)
        IMFs returned by ``emd``.
    fs : float
        Sampling frequency [Hz].

    Returns
    -------
    envelopes : ndarray, shape (n_imfs, N)
        Instantaneous amplitude of each IMF.
    inst_freqs : ndarray, shape (n_imfs, N)
        Instantaneous frequency of each IMF [Hz].
    """
    imfs = np.atleast_2d(np.asarray(imfs, dtype=float))
    envelopes  = np.zeros_like(imfs)
    inst_freqs = np.zeros_like(imfs)

    for i, imf in enumerate(imfs):
        envelopes[i], _, inst_freqs[i] = hilbert_attributes(imf, fs)

    return envelopes, inst_freqs


def hht_marginal_spectrum(
    envelopes: np.ndarray,
    inst_freqs: np.ndarray,
    fs: float,
    n_bins: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Marginal Hilbert spectrum.

    Accumulates instantaneous energy (amplitude²) at each instantaneous
    frequency across all IMFs and all time samples:

        H(f) = (1/N) Σ_{imfs} Σ_{t: f_i(t)≈f} A²(t)

    This is an adaptive analogue of the PSD — it does not assume stationarity
    and its frequency axis is not fixed by a window length.

    Parameters
    ----------
    envelopes : ndarray, shape (n_imfs, N)
        Instantaneous amplitudes from ``hht``.
    inst_freqs : ndarray, shape (n_imfs, N)
        Instantaneous frequencies from ``hht`` [Hz].
    fs : float
        Sampling frequency [Hz]. Used to set the Nyquist limit.
    n_bins : int
        Number of frequency bins (default 512).

    Returns
    -------
    freq_bins : ndarray, shape (n_bins,)
        Frequency axis [Hz], from 0 to fs / 2.
    spectrum : ndarray, shape (n_bins,)
        Marginal Hilbert spectrum [amplitude² per Hz equivalent].
    """
    envelopes  = np.atleast_2d(np.asarray(envelopes,  dtype=float))
    inst_freqs = np.atleast_2d(np.asarray(inst_freqs, dtype=float))

    nyq = fs / 2.0
    freq_bins = np.linspace(0.0, nyq, n_bins)
    spectrum  = np.zeros(n_bins)
    N = envelopes.shape[1]

    for env, fi in zip(envelopes, inst_freqs):
        # Clip to valid range, convert to bin index
        fi_clipped = np.clip(fi, 0.0, nyq * (1.0 - 1e-9))
        idx = (fi_clipped / nyq * (n_bins - 1)).astype(int)
        np.add.at(spectrum, idx, env ** 2)

    spectrum /= N  # time-average
    return freq_bins, spectrum
