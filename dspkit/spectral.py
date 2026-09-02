"""
Spectral analysis functions.

All functions accept plain NumPy arrays and return NumPy arrays.
No hidden state, no side effects.
"""

import warnings
from typing import Literal

import numpy as np
from scipy import signal as _signal


# ---------------------------------------------------------------------------
# Welch segment bookkeeping (shared with dspkit.multisensor)
#
# Every Welch-averaged estimate in the library gets noisier as the segment
# count falls, but coherence-type estimates do something worse: below a
# minimum count they become identically 1.0 by construction, independently of
# the data. These helpers exist so that case can be detected and named.
# ---------------------------------------------------------------------------


def _welch_segment_count(
    n_samples: int,
    nperseg: int,
    noverlap: int | None = None,
) -> int:
    """
    Number of segments ``scipy.signal.welch`` and friends will average over.

    Mirrors scipy's own bookkeeping: ``nperseg`` is clipped to the record
    length, ``noverlap`` defaults to ``nperseg // 2``, and the trailing
    partial segment is discarded.
    """
    nperseg = int(min(nperseg, n_samples))
    if noverlap is None:
        noverlap = nperseg // 2
    noverlap = int(noverlap)
    step = nperseg - noverlap
    if step <= 0:
        return 0
    return max(int((n_samples - noverlap) // step), 0)


def _suggest_nperseg(
    n_samples: int,
    target_segments: int,
    nperseg: int | None = None,
    noverlap: int | None = None,
) -> tuple[int, int]:
    """
    Largest ``nperseg`` reaching ``target_segments``, at the caller's overlap
    *fraction*.

    Returns ``(nperseg, n_segments)`` so an error message can quote a fix that
    has actually been checked rather than one derived from a closed form that
    ignores integer truncation.
    """
    if nperseg and noverlap is not None and nperseg > 0:
        frac = min(float(noverlap) / float(nperseg), 0.95)
    else:
        frac = 0.5

    # n_seg = (N - nperseg*frac) / (nperseg*(1 - frac))  ->  invert for nperseg
    denom = target_segments * (1.0 - frac) + frac
    guess = int(n_samples / denom) if denom > 0 else n_samples
    guess = max(min(guess, n_samples), 8)

    count = _welch_segment_count(n_samples, guess, int(round(guess * frac)))
    while guess > 8 and count < target_segments:
        guess -= max(guess // 64, 1)
        count = _welch_segment_count(n_samples, guess, int(round(guess * frac)))
    return guess, count


def _check_welch_segments(
    func_name: str,
    n_samples: int,
    nperseg: int,
    noverlap: int | None,
    hard_min: int,
    min_segments: int,
    hard_reason: str,
    stacklevel: int = 3,
) -> int:
    """
    Raise below ``hard_min`` segments, warn below ``min_segments``.

    ``hard_reason`` says why the hard limit exists; it is quoted in the error
    so the caller learns what the returned numbers would have meant.
    """
    n_seg = _welch_segment_count(n_samples, nperseg, noverlap)
    ov = "nperseg // 2" if noverlap is None else str(noverlap)

    if n_seg < hard_min:
        fix_nperseg, fix_count = _suggest_nperseg(
            n_samples, max(min_segments, hard_min), nperseg, noverlap
        )
        raise ValueError(
            f"{func_name}: N={n_samples} with nperseg={nperseg}, "
            f"noverlap={ov} gives {n_seg} Welch segment(s); "
            f"at least {hard_min} are required. {hard_reason} "
            f"Fix: shorten nperseg — nperseg={fix_nperseg} gives "
            f"{fix_count} segments at the same overlap fraction."
        )

    if n_seg < min_segments:
        warnings.warn(
            f"{func_name}: only {n_seg} Welch segments "
            f"(N={n_samples}, nperseg={nperseg}, noverlap={ov}). "
            f"Coherence is biased upward by roughly 1/{n_seg} = "
            f"{1.0 / n_seg:.2f} for unrelated signals; read values against "
            f"that floor, not against zero. Shorten nperseg or pass "
            f"min_segments to silence this.",
            stacklevel=stacklevel,
        )
    return n_seg


def fft_spectrum(
    x: np.ndarray,
    fs: float,
    window: str | None = "hann",
    scaling: Literal["amplitude", "rms"] = "amplitude",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Single-sided FFT amplitude spectrum with window amplitude correction.

    For a pure sine of amplitude A at frequency f, the returned spectrum will
    show A at that frequency bin (with ``scaling='amplitude'``).

    Parameters
    ----------
    x : array_like, shape (N,)
        Time-domain signal.
    fs : float
        Sampling frequency [Hz].
    window : str or None
        Window function name accepted by ``scipy.signal.get_window``.
        ``None`` uses a rectangular window (no windowing).
    scaling : {'amplitude', 'rms'}
        ``'amplitude'`` returns peak amplitude per bin.
        ``'rms'`` returns RMS amplitude (peak / sqrt(2)), useful for
        comparing sinusoidal components with broadband levels.

    Returns
    -------
    freqs : ndarray, shape (N//2 + 1,)
        Frequency vector [Hz].
    amplitude : ndarray, shape (N//2 + 1,)
        Amplitude spectrum in the same units as ``x``.
    """
    x = np.asarray(x, dtype=float)
    N = len(x)

    if window is not None:
        win = _signal.get_window(window, N)
        # Amplitude correction: scale so that a sine's peak is preserved
        acf = N / win.sum()
        x = x * win
    else:
        acf = 1.0

    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)

    amplitude = np.abs(X) * acf / N
    # Double interior bins to account for the discarded negative-frequency mirror
    amplitude[1:-1] *= 2.0

    if scaling == "rms":
        amplitude /= np.sqrt(2.0)

    return freqs, amplitude


def psd(
    x: np.ndarray,
    fs: float,
    window: str = "hann",
    nperseg: int | None = None,
    noverlap: int | None = None,
    scaling: Literal["density", "spectrum"] = "density",
    detrend: str | Literal[False] = "constant",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Power spectral density (or power spectrum) via Welch's method.

    Parameters
    ----------
    x : array_like, shape (N,)
        Time-domain signal.
    fs : float
        Sampling frequency [Hz].
    window : str
        Window function (default ``'hann'``).
    nperseg : int or None
        Segment length. Defaults to ``min(len(x), 1024)``.
    noverlap : int or None
        Number of overlapping samples between segments.
        Defaults to ``nperseg // 2`` (50 % overlap).
    scaling : {'density', 'spectrum'}
        ``'density'`` → PSD [units²/Hz].
        ``'spectrum'`` → power spectrum [units²].
    detrend : str or False
        Detrending applied to each segment before windowing.
        ``'constant'`` removes the mean, ``'linear'`` removes a linear trend,
        ``False`` skips detrending.

    Returns
    -------
    freqs : ndarray
        Frequency vector [Hz].
    Pxx : ndarray
        One-sided PSD or power spectrum (real, non-negative).
    """
    x = np.asarray(x, dtype=float)
    if nperseg is None:
        nperseg = min(len(x), 1024)

    freqs, Pxx = _signal.welch(
        x,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling=scaling,
        detrend=detrend,
    )
    return freqs, Pxx


def csd(
    x: np.ndarray,
    y: np.ndarray,
    fs: float,
    window: str = "hann",
    nperseg: int | None = None,
    noverlap: int | None = None,
    detrend: str | Literal[False] = "constant",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cross-spectral density via Welch's method.

    Gxy(f) = E[X*(f) Y(f)] / Hz,  where X, Y are the DFTs of x and y.

    Parameters
    ----------
    x, y : array_like, shape (N,)
        Input signals. They must have the same sampling frequency.
    fs : float
        Sampling frequency [Hz].
    window : str
        Window function (default ``'hann'``).
    nperseg : int or None
        Segment length. Defaults to ``min(len(x), len(y), 1024)``.
    noverlap : int or None
        Overlapping samples. Defaults to ``nperseg // 2``.
    detrend : str or False
        Per-segment detrending (see `psd`).

    Returns
    -------
    freqs : ndarray
        Frequency vector [Hz].
    Pxy : ndarray (complex)
        One-sided cross-spectral density [units_x · units_y / Hz].
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if nperseg is None:
        nperseg = min(len(x), len(y), 1024)

    freqs, Pxy = _signal.csd(
        x,
        y,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=detrend,
    )
    return freqs, Pxy


def coherence(
    x: np.ndarray,
    y: np.ndarray,
    fs: float,
    window: str = "hann",
    nperseg: int | None = None,
    noverlap: int | None = None,
    detrend: str | Literal[False] = "constant",
    min_segments: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Magnitude-squared coherence between x and y.

    Cxy(f) = |Gxy(f)|² / (Gxx(f) · Gyy(f)),  values in [0, 1].

    A value near 1 means the two signals are linearly related at that frequency.
    A value near 0 indicates noise or nonlinearity.

    Parameters
    ----------
    x, y : array_like, shape (N,)
    fs : float
        Sampling frequency [Hz].
    window : str
        Window function (default ``'hann'``).
    nperseg : int or None
        Segment length. Defaults to ``min(len(x), len(y), 1024)``.
    noverlap : int or None
        Overlapping samples. Defaults to ``nperseg // 2``.
    detrend : str or False
        Per-segment detrending.
    min_segments : int
        Warn below this many Welch segments (default 8). Fewer than two
        segments raises instead — see Notes.

    Returns
    -------
    freqs : ndarray
    Cxy : ndarray
        Magnitude-squared coherence, values in [0, 1].

    Raises
    ------
    ValueError
        If the parameters give fewer than two Welch segments.

    Notes
    -----
    **Coherence is made by the averaging, not by the formula.** Within a
    single segment |Gxy|² = Gxx·Gyy identically, so a one-segment estimate is
    exactly 1.0 at every frequency whatever the two signals are.
    ``nperseg = len(x)`` is precisely that case, and it is rejected rather
    than returned. Measured on the library's 2-DOF example (N = 20480,
    fs = 1024 Hz): mean coherence 0.0675 at ``nperseg=1024`` (39 segments),
    and exactly 1.0000 everywhere at ``nperseg=N``.

    The bias does not stop at one segment, it only becomes finite. For two
    independent signals averaged over ``n_d`` segments the expected coherence
    is about ``1 / n_d``, so a value of 0.2 from 5 segments is what
    independence looks like, not evidence of a relationship. ``min_segments``
    warns while that floor is still large; it does not correct for it.

    What this will not tell you: whether the relationship is causal, which
    channel leads (use the cross-spectrum phase), whether a low value means
    noise or nonlinearity, or whether a high value at one frequency survives
    conditioning on the other channels in an array — for that see
    ``dspkit.multisensor.partial_coherence``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if nperseg is None:
        nperseg = min(len(x), len(y), 1024)

    _check_welch_segments(
        "coherence",
        min(len(x), len(y)),
        nperseg,
        noverlap,
        hard_min=2,
        min_segments=min_segments,
        hard_reason=(
            "A single-segment estimate is identically 1.0 at every frequency "
            "by construction and says nothing about the signals."
        ),
    )

    freqs, Cxy = _signal.coherence(
        x,
        y,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=detrend,
    )
    return freqs, Cxy


def cross_correlation(
    x: np.ndarray,
    y: np.ndarray,
    fs: float | None = None,
    normalize: bool = True,
    max_lag: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Biased cross-correlation function (CCF) via FFT.

    Computes the full two-sided CCF with lags from -(N-1) to +(N-1):

        CCF[k] = (1/N) Σ_n x[n] · y[n + k]

    A positive peak at lag k > 0 means **y is the delayed copy**: the pairing
    is x[n] with y[n+k], so if y[n] = x[n-d] the peak sits at k = +d and x
    leads y by d samples. (This docstring said the opposite until 2026-09-02;
    the formula above was always right. ``test_cross_correlation_lag_sign``
    pins the direction.)

    Parameters
    ----------
    x, y : array_like, shape (N,)
        Input signals. Must have the same length.
    fs : float or None
        Sampling frequency [Hz]. If provided, the lag axis is in seconds;
        otherwise it is in samples.
    normalize : bool
        If ``True`` (default), normalise so that ``max |CCF| ≤ 1``.
        Specifically divides by ``sqrt(Rxx[0] · Ryy[0])``, giving the
        cross-correlation coefficient — the same convention as
        ``numpy.corrcoef``.
    max_lag : float or None
        Maximum absolute lag to return. Interpreted in seconds if ``fs``
        is given, otherwise in samples.

    Returns
    -------
    lags : ndarray
        Symmetric lag axis, running from ``-max_lag`` to ``+max_lag``
        (seconds if ``fs`` given, else samples).
    ccf : ndarray
        Cross-correlation values.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    N = len(x)

    Xf = np.fft.fft(x, n=2 * N)
    Yf = np.fft.fft(y, n=2 * N)
    ccf_raw = np.fft.ifft(np.conj(Xf) * Yf).real

    # Rearrange to two-sided: lags -(N-1) … 0 … +(N-1)
    ccf_two = np.concatenate([ccf_raw[-(N - 1):], ccf_raw[:N]])

    if normalize:
        norm = np.sqrt(np.dot(x, x) * np.dot(y, y))
        if norm > 0:
            ccf_two = ccf_two / norm
    else:
        ccf_two = ccf_two / N

    lags_samples = np.arange(-(N - 1), N, dtype=float)
    lags = lags_samples / fs if fs is not None else lags_samples

    if max_lag is not None:
        cutoff = int(max_lag * fs) if fs is not None else int(max_lag)
        mask = np.abs(lags_samples) <= cutoff
        lags = lags[mask]
        ccf_two = ccf_two[mask]

    return lags, ccf_two


def autocorrelation(
    x: np.ndarray,
    fs: float | None = None,
    normalize: bool = True,
    max_lag: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Biased autocorrelation function (ACF) via FFT.

    Uses the biased estimator (divides by N, not N-k) for better
    variance behaviour at large lags.

    Parameters
    ----------
    x : array_like, shape (N,)
        Input signal (zero-mean recommended; detrend first if needed).
    fs : float or None
        Sampling frequency [Hz]. If provided, the lag axis is in seconds;
        otherwise it is in samples.
    normalize : bool
        If ``True`` (default), normalise so that ACF[0] = 1.
    max_lag : float or None
        Maximum lag to return. Interpreted in seconds if ``fs`` is given,
        otherwise in samples. Defaults to the full one-sided ACF.

    Returns
    -------
    lags : ndarray
        Lag axis (seconds if ``fs`` given, else samples).
    acf : ndarray
        Autocorrelation values.
    """
    x = np.asarray(x, dtype=float)
    N = len(x)

    # FFT-based circular correlation then truncate to causal (positive) lags
    Xf = np.fft.rfft(x, n=2 * N)
    acf_full = np.fft.irfft(Xf * np.conj(Xf))[:N]

    if normalize:
        acf_full = acf_full / acf_full[0]
    else:
        acf_full = acf_full / N

    lags = np.arange(N) / fs if fs is not None else np.arange(N, dtype=float)

    if max_lag is not None:
        cutoff = int(max_lag * fs) + 1 if fs is not None else int(max_lag) + 1
        lags = lags[:cutoff]
        acf_full = acf_full[:cutoff]

    return lags, acf_full
