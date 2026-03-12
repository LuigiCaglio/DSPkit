"""
Multi-sensor analysis tools.

Functions for analysing relationships between arrays of sensors, commonly
used in SHM systems with multiple measurement channels.

Functions
---------
correlation_matrix      -- pairwise Pearson correlation matrix
coherence_matrix        -- pairwise magnitude-squared coherence matrix
psd_matrix              -- cross-spectral density matrix (for FDD / OMA)
"""

from typing import Literal

import numpy as np
from scipy import signal as _signal


def correlation_matrix(
    data: np.ndarray,
) -> np.ndarray:
    """
    Pairwise Pearson correlation matrix for multi-channel data.

    Parameters
    ----------
    data : array_like, shape (n_channels, N)
        Each row is a time series from one sensor.

    Returns
    -------
    R : ndarray, shape (n_channels, n_channels)
        Correlation matrix with values in [-1, 1].
        ``R[i, j]`` is the Pearson correlation coefficient between
        channels ``i`` and ``j``.
    """
    data = np.atleast_2d(np.asarray(data, dtype=float))
    return np.corrcoef(data)


def coherence_matrix(
    data: np.ndarray,
    fs: float,
    window: str = "hann",
    nperseg: int | None = None,
    noverlap: int | None = None,
    detrend: str | Literal[False] = "constant",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pairwise magnitude-squared coherence matrix.

    Parameters
    ----------
    data : array_like, shape (n_channels, N)
        Each row is a time series from one sensor.
    fs : float
        Sampling frequency [Hz].
    window : str
        Window function (default ``'hann'``).
    nperseg : int or None
        Welch segment length. Defaults to ``min(N, 1024)``.
    noverlap : int or None
        Overlap between segments.
    detrend : str or False
        Per-segment detrending.

    Returns
    -------
    freqs : ndarray, shape (M,)
        Frequency vector [Hz].
    C : ndarray, shape (n_channels, n_channels, M)
        Coherence matrix. ``C[i, j, :]`` is the magnitude-squared coherence
        between channels ``i`` and ``j``. Diagonal entries are 1.0.
    """
    data = np.atleast_2d(np.asarray(data, dtype=float))
    n_ch, N = data.shape

    if nperseg is None:
        nperseg = min(N, 1024)

    # Compute one coherence to get the frequency vector length
    freqs, _ = _signal.coherence(
        data[0], data[0], fs=fs, window=window,
        nperseg=nperseg, noverlap=noverlap, detrend=detrend,
    )
    M = len(freqs)
    C = np.ones((n_ch, n_ch, M), dtype=float)

    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            _, Cij = _signal.coherence(
                data[i], data[j], fs=fs, window=window,
                nperseg=nperseg, noverlap=noverlap, detrend=detrend,
            )
            C[i, j, :] = Cij
            C[j, i, :] = Cij

    return freqs, C


def psd_matrix(
    data: np.ndarray,
    fs: float,
    window: str = "hann",
    nperseg: int | None = None,
    noverlap: int | None = None,
    detrend: str | Literal[False] = "constant",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cross-spectral density matrix (power spectral density matrix).

    Computes the full n_channels × n_channels CSD matrix at each frequency.
    This is the input required for Frequency Domain Decomposition (FDD).

    The matrix is Hermitian at each frequency: ``G[i,j,f] = conj(G[j,i,f])``.
    Diagonal entries ``G[i,i,f]`` are real-valued (auto-PSD).

    Parameters
    ----------
    data : array_like, shape (n_channels, N)
        Each row is a time series from one sensor.
    fs : float
        Sampling frequency [Hz].
    window : str
        Window function (default ``'hann'``).
    nperseg : int or None
        Welch segment length. Defaults to ``min(N, 1024)``.
    noverlap : int or None
        Overlap between segments.
    detrend : str or False
        Per-segment detrending.

    Returns
    -------
    freqs : ndarray, shape (M,)
        Frequency vector [Hz].
    G : ndarray, shape (n_channels, n_channels, M), complex
        Cross-spectral density matrix. ``G[i, j, k]`` is the CSD between
        channels ``i`` and ``j`` at frequency ``freqs[k]``.
    """
    data = np.atleast_2d(np.asarray(data, dtype=float))
    n_ch, N = data.shape

    if nperseg is None:
        nperseg = min(N, 1024)

    # Get frequency vector length
    freqs, _ = _signal.csd(
        data[0], data[0], fs=fs, window=window,
        nperseg=nperseg, noverlap=noverlap, detrend=detrend,
    )
    M = len(freqs)
    G = np.zeros((n_ch, n_ch, M), dtype=complex)

    for i in range(n_ch):
        for j in range(i, n_ch):
            _, Gij = _signal.csd(
                data[i], data[j], fs=fs, window=window,
                nperseg=nperseg, noverlap=noverlap, detrend=detrend,
            )
            G[i, j, :] = Gij
            if i != j:
                G[j, i, :] = np.conj(Gij)

    return freqs, G
