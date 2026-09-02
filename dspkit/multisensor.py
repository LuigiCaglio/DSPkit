"""
Multi-sensor analysis tools.

Functions for analysing relationships between arrays of sensors, commonly
used in SHM systems with multiple measurement channels.

Functions
---------
correlation_matrix      -- pairwise Pearson correlation matrix
coherence_matrix        -- pairwise magnitude-squared coherence matrix
psd_matrix              -- cross-spectral density matrix (for FDD / OMA)
multiple_coherence      -- one channel against all the others, per frequency
partial_coherence       -- pairwise, with the remaining channels conditioned out
"""

from typing import Literal

import numpy as np
from scipy import signal as _signal

from dspkit.spectral import _check_welch_segments


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
    min_segments: int = 8,
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
    min_segments : int
        Warn below this many Welch segments (default 8). Fewer than two
        segments raises instead — see Notes.

    Returns
    -------
    freqs : ndarray, shape (M,)
        Frequency vector [Hz].
    C : ndarray, shape (n_channels, n_channels, M)
        Coherence matrix. ``C[i, j, :]`` is the magnitude-squared coherence
        between channels ``i`` and ``j``. Diagonal entries are 1.0.

    Raises
    ------
    ValueError
        If the parameters give fewer than two Welch segments, in which case
        every off-diagonal entry would be identically 1.0 by construction.
        See ``dspkit.spectral.coherence`` for the full explanation.

    Notes
    -----
    Every entry here is a *pairwise* coherence: channels i and j are compared
    with the rest of the array ignored, so a pair that only looks related
    because both follow a third channel looks exactly like a pair that is
    directly related. ``partial_coherence`` is the same matrix with the other
    channels conditioned out, and is the one to read when the array has more
    than two sensors.

    See Also
    --------
    partial_coherence, multiple_coherence
    """
    data = np.atleast_2d(np.asarray(data, dtype=float))
    n_ch, N = data.shape

    if nperseg is None:
        nperseg = min(N, 1024)

    _check_welch_segments(
        "coherence_matrix", N, nperseg, noverlap,
        hard_min=2, min_segments=min_segments,
        hard_reason=(
            "A single-segment estimate is identically 1.0 at every frequency "
            "by construction and says nothing about the signals."
        ),
    )

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


# ---------------------------------------------------------------------------
# Conditioning: everything below is a function of the inverse CSD matrix
# ---------------------------------------------------------------------------


def _normalised_csd_inverse(
    G: np.ndarray,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Ridge-regularised inverse of the CSD matrix, one inverse per frequency.

    ``G`` is first normalised to the complex coherence matrix
    ``Ghat = D⁻¹ G D⁻¹`` with ``D = diag(sqrt(G_ii))``, so the diagonal is 1
    and ``|Ghat_ij|²`` is the ordinary magnitude-squared coherence. Multiple
    and partial coherence are both invariant under this normalisation (the
    ``D`` factors cancel), but the ridge is not: added to the raw ``G`` it is
    an absolute quantity, and would swamp a low-power channel while leaving a
    high-power one untouched. On the normalised matrix it is relative and
    treats every channel alike.

    Parameters
    ----------
    G : ndarray, shape (n_ch, n_ch, M), complex
        Cross-spectral density matrix from ``psd_matrix``.
    ridge : float
        Ridge added to the unit diagonal.

    Returns
    -------
    Ghat_reg : ndarray, shape (M, n_ch, n_ch), complex
        Normalised, regularised matrix. Diagonal is ``1 + ridge``.
    Ghat_inv : ndarray, shape (M, n_ch, n_ch), complex
        Its inverse.
    """
    Gt = np.moveaxis(np.asarray(G), -1, 0)  # (M, n_ch, n_ch)
    n_ch = Gt.shape[-1]
    idx = np.arange(n_ch)

    d = np.sqrt(np.clip(np.real(Gt[:, idx, idx]), 0.0, None))  # (M, n_ch)
    # A channel with no power at this frequency (d = 0) would divide by zero.
    # Cauchy-Schwarz makes its whole row zero anyway, so dividing by 1 leaves
    # an all-zero row, and the unit diagonal below turns it into an isolated
    # channel: related to nothing, which is the honest reading of "no power".
    d_safe = np.where(d > 0.0, d, 1.0)
    Ghat = Gt / (d_safe[:, :, None] * d_safe[:, None, :])

    # Hermitian by construction; enforce it so the eigenvalues stay real and
    # the inverse stays positive definite where float error has nudged the two
    # triangles apart.
    Ghat = 0.5 * (Ghat + np.conj(np.swapaxes(Ghat, -1, -2)))
    Ghat[:, idx, idx] = 1.0

    Ghat_reg = Ghat + ridge * np.eye(n_ch)
    return Ghat_reg, np.linalg.inv(Ghat_reg)


def _conditioned_inverse_from_data(
    func_name: str,
    data: np.ndarray,
    fs: float,
    window: str,
    nperseg: int | None,
    noverlap: int | None,
    detrend: str | Literal[False],
    ridge: float,
    min_segments: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Shared front end for ``multiple_coherence`` and ``partial_coherence``.

    Resolves the defaults, refuses a rank-deficient CSD matrix, and returns
    ``(freqs, Ghat_reg, Ghat_inv, n_ch)``.
    """
    data = np.atleast_2d(np.asarray(data, dtype=float))
    n_ch, N = data.shape

    if nperseg is None:
        nperseg = min(N, 1024)
    if min_segments is None:
        # Under independence the estimate sits at about (n_ch - 1) / n_seg,
        # so 5 * n_ch segments is where that floor drops near 0.2.
        min_segments = 5 * n_ch

    _check_welch_segments(
        func_name, N, nperseg, noverlap,
        hard_min=n_ch + 1,
        min_segments=max(min_segments, n_ch + 1),
        hard_reason=(
            f"The CSD matrix averages n_segments rank-1 terms: below "
            f"n_channels ({n_ch}) it is singular at every frequency and every "
            f"coherence is identically 1.0, and at exactly n_channels the fit "
            f"has no residual degrees of freedom — independent channels still "
            f"read about {(n_ch - 1) / n_ch:.2f}."
        ),
        stacklevel=4,
    )

    freqs, G = psd_matrix(
        data, fs, window=window, nperseg=nperseg,
        noverlap=noverlap, detrend=detrend,
    )
    Ghat_reg, Ghat_inv = _normalised_csd_inverse(G, ridge)
    return freqs, Ghat_reg, Ghat_inv, n_ch


def multiple_coherence(
    data: np.ndarray,
    fs: float,
    window: str = "hann",
    nperseg: int | None = None,
    noverlap: int | None = None,
    detrend: str | Literal[False] = "constant",
    ridge: float = 1e-10,
    min_segments: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Multiple coherence: each channel against all the others, per frequency.

    ``gamma_i²(f) = 1 - 1 / (G_ii(f) · inv(G)(f)_ii)`` — the fraction of
    channel *i*'s power at frequency *f* that a linear combination of every
    other channel accounts for. It is the frequency-domain coefficient of
    determination of channel *i* regressed on the rest of the array, and it
    answers "is this sensor redundant?" in the only form that question has a
    defensible answer: as a curve, not a score.

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
        Overlap between segments. Defaults to ``nperseg // 2``.
    detrend : str or False
        Per-segment detrending.
    ridge : float
        Ridge added to the unit diagonal of the normalised CSD matrix before
        inversion (default 1e-10). See Notes — this is a numerical floor, not
        a statistical shrinkage.
    min_segments : int or None
        Warn below this many Welch segments. Defaults to ``5 * n_channels``.
        At or below ``n_channels`` segments the function raises instead.

    Returns
    -------
    freqs : ndarray, shape (M,)
        Frequency vector [Hz].
    gamma2 : ndarray, shape (n_channels, M)
        Multiple coherence of each channel against all the others, in [0, 1].

    Raises
    ------
    ValueError
        If the Welch parameters give ``n_segments <= n_channels``, where the
        CSD matrix is singular at every frequency and the result would be
        identically 1.0.

    Notes
    -----
    **Regularisation.** The matrix is inverted with a ridge, not with
    ``numpy.linalg.pinv``. The reason is the case this function exists to
    detect: when a channel is an exact linear combination of the others, G is
    exactly rank-deficient, and the true multiple coherence is 1 — which
    requires ``inv(G)_ii -> inf``. A pseudo-inverse truncates precisely that
    direction and returns a *finite* ``inv(G)_ii``, small enough that the
    formula goes **negative** (measured: -0.6 to -2.9 on three channels where
    the third is the sum of the first two), which a clip to [0, 1] would then
    present as 0.0 — the most redundant array that can exist, reported as
    perfectly independent. A ridge keeps the limit intact and returns
    ``1 - O(ridge)``. The default 1e-10 bounds the condition number at about
    1e10, which float64 absorbs; raise it if the inverse looks noisy, at the
    cost of biasing values away from 1 by roughly that amount.

    **The bias floor is the number to compare against, not zero.** With
    ``q = n_channels - 1`` other channels and ``n_d`` Welch segments, the
    expected multiple coherence of *independent* channels is about
    ``q / n_d``. Measured on four independent noise channels, N = 20480::

        n_d =  3   mean 1.000     (q / n_d = 1.00, and G is singular here)
        n_d =  4   mean 0.759     (0.75)
        n_d =  9   mean 0.343     (0.33)
        n_d = 19   mean 0.167     (0.16)
        n_d = 39   mean 0.079     (0.077)

    That one formula covers both the noise floor and the failure mode: it
    reaches 1.0 exactly at ``n_d = q``, which is where the matrix goes
    singular, so "every coherence comes back at 1.0" is the same statement as
    "the floor has risen to 1". At ``n_d = n_channels`` the matrix is
    invertible again but the fit has no residual degrees of freedom, so the
    values are still meaningless — hence the refusal at ``n_d <= n_channels``
    rather than only below it. This function reports the curve, not the excess
    over the floor; the floor is yours to subtract, and ``min_segments`` warns
    while it is large.

    What this will not tell you: which of the other channels does the
    explaining (use ``partial_coherence``), whether the relationship is
    causal, or whether a low value means independence or a nonlinear
    relationship that no linear predictor can reach (see
    ``dspkit.statistics.mutual_information``). It is also blind to any
    redundancy that only appears outside the analysis band, and it says
    nothing about whether a redundant sensor is worth keeping — a duplicate
    channel is redundant and also the only thing that will catch the other
    one failing.

    See Also
    --------
    partial_coherence, coherence_matrix, psd_matrix

    Examples
    --------
    >>> import numpy as np
    >>> from dspkit.multisensor import multiple_coherence
    >>> rng = np.random.default_rng(0)
    >>> x, y, w = rng.normal(size=(3, 20000))
    >>> data = np.vstack([x, y, x + y, w])   # channel 2 is x + y, channel 3 is not
    >>> f, g2 = multiple_coherence(data, 1000.0, nperseg=256)
    >>> bool(np.median(g2[2]) > 0.99), bool(np.median(g2[3]) < 0.1)
    (True, True)
    """
    freqs, Ghat_reg, Ghat_inv, n_ch = _conditioned_inverse_from_data(
        "multiple_coherence", data, fs, window, nperseg, noverlap,
        detrend, ridge, min_segments,
    )
    idx = np.arange(n_ch)
    diag_reg = np.real(Ghat_reg[:, idx, idx])   # (M, n_ch), equals 1 + ridge
    diag_inv = np.real(Ghat_inv[:, idx, idx])   # (M, n_ch)

    gamma2 = 1.0 - 1.0 / (diag_reg * diag_inv)

    # Cauchy-Schwarz makes G_ii · inv(G)_ii >= 1 for any positive-definite G,
    # so gamma2 cannot truly leave [0, 1]. Only float error in the inverse can
    # push it out, by ~1e-12; this clip absorbs that and hides nothing real.
    return freqs, np.clip(gamma2, 0.0, 1.0).T


def partial_coherence(
    data: np.ndarray,
    fs: float,
    window: str = "hann",
    nperseg: int | None = None,
    noverlap: int | None = None,
    detrend: str | Literal[False] = "constant",
    ridge: float = 1e-10,
    min_segments: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Partial coherence: pairwise, with every other channel conditioned out.

    ``gamma_ij²(f) = |inv(G)_ij|² / (inv(G)_ii · inv(G)_jj)`` — the coherence
    that survives after the linear contribution of the remaining channels has
    been removed from both *i* and *j*. Where ``coherence_matrix`` cannot tell
    a direct relationship from one mediated by a third sensor, this can: for a
    chain x -> y -> z the ordinary coherence of x and z is high, while their
    partial coherence given y falls to the noise floor.

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
        Overlap between segments. Defaults to ``nperseg // 2``.
    detrend : str or False
        Per-segment detrending.
    ridge : float
        Ridge added to the unit diagonal of the normalised CSD matrix before
        inversion (default 1e-10). See ``multiple_coherence`` for why a ridge
        rather than a pseudo-inverse.
    min_segments : int or None
        Warn below this many Welch segments. Defaults to ``5 * n_channels``.
        At or below ``n_channels`` segments the function raises instead.

    Returns
    -------
    freqs : ndarray, shape (M,)
        Frequency vector [Hz].
    C : ndarray, shape (n_channels, n_channels, M)
        Partial coherence matrix, mirroring ``coherence_matrix``.
        ``C[i, j, :]`` is the partial coherence of channels ``i`` and ``j``
        given all the others; the diagonal is 1.0 and the matrix is symmetric.

    Raises
    ------
    ValueError
        If the Welch parameters give ``n_segments <= n_channels``.

    Notes
    -----
    With exactly two channels there is nothing to condition on, and this
    reduces to ``coherence_matrix`` — a useful check, not a use case.

    Conditioning is **linear and simultaneous at each frequency**: the other
    channels are removed as a frequency-domain linear predictor, which is the
    right operation for a linear time-invariant structure and the wrong one if
    the mediation is nonlinear, or if the mediating channel was not measured.
    An unmeasured common cause is invisible here, exactly as it is in ordinary
    coherence — conditioning removes what is in ``data``, and nothing else.

    The bias floor applies here too: with ``n_d`` segments and ``n_ch - 2``
    conditioning channels, independent pairs sit near
    ``1 / (n_d - n_ch + 2)`` rather than at 0. Conditioning also spends
    degrees of freedom, so partial coherence is noisier than the pairwise
    coherence it refines; the more channels are conditioned out, the more
    segments are needed to see the same contrast.

    See Also
    --------
    multiple_coherence, coherence_matrix, psd_matrix

    Examples
    --------
    >>> import numpy as np
    >>> from dspkit.multisensor import partial_coherence
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(size=20000)
    >>> y = np.convolve(x, np.ones(8) / 8, "same") + 0.02 * rng.normal(size=20000)
    >>> z = np.convolve(y, np.ones(8) / 8, "same") + 0.02 * rng.normal(size=20000)
    >>> f, C = partial_coherence(np.vstack([x, y, z]), 1000.0, nperseg=256)
    >>> bool(np.median(C[0, 2]) < 0.1)          # x and z, given y
    True
    """
    freqs, _, Ghat_inv, n_ch = _conditioned_inverse_from_data(
        "partial_coherence", data, fs, window, nperseg, noverlap,
        detrend, ridge, min_segments,
    )
    idx = np.arange(n_ch)
    diag_inv = np.real(Ghat_inv[:, idx, idx])   # (M, n_ch), real and positive

    C = np.abs(Ghat_inv) ** 2 / (diag_inv[:, :, None] * diag_inv[:, None, :])

    # Same Cauchy-Schwarz argument as in multiple_coherence: the ratio is in
    # [0, 1] exactly, so this clip only absorbs float error in the inverse.
    C = np.clip(C, 0.0, 1.0)
    C[:, idx, idx] = 1.0
    return freqs, np.moveaxis(C, 0, -1)
