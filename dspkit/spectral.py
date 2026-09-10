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


# Both guardrail messages used to end at "shorten nperseg", which is right for
# one of the two failure modes and actively wrong for the other. A segment too
# short to resolve the sharpest peak lets leakage smear peaks and notches, which
# biases coherence *down* and manufactures residual power where the signal is
# strongest — an error growing as the square of the resolution bandwidth.
# Measured on a noise-free single-input system whose true error spectrum is
# exactly zero, the apparent unexplained variance ran 3.8e-5, 1.6e-4, 6.3e-4 and
# 2.4e-3 of the target as Be/Br went 0.06, 0.12, 0.24, 0.49 — entirely an
# artefact of the segment length.
_BOTH_DIRECTIONS = (
    "a segment too short to resolve the sharpest peak smears it, which biases "
    "coherence down and fabricates residual power"
)


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
            f"nperseg={fix_nperseg} would give {fix_count} segments at the "
            f"same overlap fraction — but check what that costs in resolution "
            f"before doing it ({_BOTH_DIRECTIONS}). If the record is simply "
            f"too short to do both, lengthening it is the only real fix."
        )

    if n_seg < min_segments:
        warnings.warn(
            f"{func_name}: only {n_seg} Welch segments "
            f"(N={n_samples}, nperseg={nperseg}, noverlap={ov}). "
            f"Coherence is biased upward by roughly 1/{n_seg} = "
            f"{1.0 / n_seg:.2f} for unrelated signals; read values against "
            f"that floor, not against zero. Shortening nperseg buys averages "
            f"and spends resolution ({_BOTH_DIRECTIONS}); "
            f"dspkit.spectral.segment_advice weighs the two on your record. "
            f"Pass min_segments to silence this.",
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


# ─── lag windows and Blackman-Tukey ───────────────────────────────────────────

#: Lag windows whose own Fourier transform is non-negative. Only these keep the
#: Blackman-Tukey estimate non-negative; see :func:`lag_window` for why.
NONNEGATIVE_LAG_WINDOWS = ("bartlett", "parzen", "exponential")

LAG_WINDOWS = ("none", "bartlett", "parzen", "exponential", "hann", "hamming")


def lag_window(name: str, m: int, decay: float = 3.0) -> np.ndarray:
    """
    One-sided lag window of length ``m``, for tapering an autocorrelation.

    Parameters
    ----------
    name : str
        One of ``'none'``, ``'bartlett'``, ``'parzen'``, ``'exponential'``,
        ``'hann'``, ``'hamming'``.
    m : int
        Number of lags, including lag zero.
    decay : float
        For ``'exponential'``, how many time constants fit in ``m`` lags.
        Larger decays faster.

    Returns
    -------
    ndarray, shape (m,)
        Window values for lags 0..m-1, starting at 1.0.

    Notes
    -----
    **A lag window is not a data window, and the usual advice inverts.** A data
    window multiplies the signal before transforming; a lag window multiplies
    the autocorrelation. The biased autocorrelation is positive semi-definite,
    so its transform cannot be negative -- you cannot get negative power out of
    it. Tapering preserves that only if the *window's own transform* is
    non-negative, because the estimate is then the true spectrum convolved with
    a non-negative kernel.

    - ``bartlett`` -- transform is the Fejer kernel, ``|Dirichlet|**2``.
      Non-negative. Safe.
    - ``parzen`` -- non-negative transform with better sidelobe decay. The usual
      default.
    - ``exponential`` -- transform is a Lorentzian, non-negative. Safe, and it
      is what modal testing uses to add known artificial damping.
    - ``none`` -- rectangular truncation, whose transform is the Dirichlet
      kernel with negative sidelobes. **Unsafe**: can return negative power.
    - ``hann``, ``hamming`` -- negative sidelobes as *lag* windows. **Unsafe**,
      which is exactly backwards from their role as data windows. Same name,
      opposite verdict, different domain.
    """
    key = str(name).lower()
    if key not in LAG_WINDOWS:
        raise ValueError(
            "Unknown lag window {!r}. Choose one of: {}.".format(
                name, ", ".join(LAG_WINDOWS))
        )
    m = int(m)
    if m < 2:
        raise ValueError("Need at least 2 lags.")

    k = np.arange(m, dtype=float)
    if key == "none":
        return np.ones(m)
    if key == "bartlett":
        return 1.0 - k / m
    if key == "exponential":
        return np.exp(-decay * k / m)
    if key in ("hann", "hamming"):
        a = 0.5 if key == "hann" else 0.54
        b = 0.5 if key == "hann" else 0.46
        return a + b * np.cos(np.pi * k / m)
    # Parzen, the standard piecewise cubic.
    w = np.empty(m)
    half = m / 2.0
    lo = k <= half
    r = k[lo] / m
    w[lo] = 1.0 - 6.0 * r ** 2 + 6.0 * r ** 3
    r = k[~lo] / m
    w[~lo] = 2.0 * (1.0 - r) ** 3
    return w


def blackman_tukey_psd(
    x: np.ndarray,
    fs: float,
    lag_window_name: str = "parzen",
    max_lag: int | None = None,
    decay: float = 3.0,
    detrend_first: bool = True,
):
    """
    Power spectral density as the Fourier transform of the autocorrelation.

    The third route to a spectrum, alongside the periodogram and Welch. What
    makes it worth having is the knob: the autocorrelation at lag k is estimated
    from N-k sample pairs, so long lags are progressively noisier, and a lag
    window trades resolution against variance directly and explicitly.

    Be honest about the gain. Blackman-Tukey with a lag window of length M and
    Welch with segments of length about M land in much the same place on bias
    and variance. This is not a better estimator, it is a differently
    parameterised one, and sometimes that parameterisation is the point.

    Parameters
    ----------
    x : array_like, shape (N,)
        Input signal.
    fs : float
        Sampling frequency [Hz].
    lag_window_name : str
        See :func:`lag_window`. Defaults to ``'parzen'``, which is safe and
        well behaved.
    max_lag : int or None
        Number of lags to keep, including zero. Defaults to ``N // 10``.
        Effective resolution is roughly ``fs / max_lag``, times a
        window-dependent factor.
    decay : float
        Passed to the exponential window.
    detrend_first : bool
        Remove the mean before estimating, so a DC offset does not dominate.

    Returns
    -------
    freqs : ndarray
    psd : ndarray
    negative_fraction : float
        Share of the spectrum that came out negative. Zero for a non-negative
        window; above zero it is a warning that the estimate is not a valid
        spectrum, not merely a noisy one.

    Notes
    -----
    Negative power is not possible, so if ``negative_fraction`` is above zero
    the window is at fault rather than the data. Only the windows in
    :data:`NONNEGATIVE_LAG_WINDOWS` guarantee it cannot happen.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("Expected a one-dimensional signal.")
    n = x.size
    if n < 16:
        raise ValueError("Signal is too short for this estimate.")
    if detrend_first:
        x = x - x.mean()

    m = int(max_lag) if max_lag else max(8, n // 10)
    m = max(4, min(m, n - 1))

    # Biased autocorrelation: divide by N, not N-k. That is what makes the
    # sequence positive semi-definite, which is what makes the untapered
    # transform non-negative in the first place.
    full = np.correlate(x, x, mode="full") / n
    acf = full[n - 1: n - 1 + m]

    w = lag_window(lag_window_name, m, decay=decay)
    tapered = acf * w

    # Rebuild the two-sided sequence, which is what has a real transform.
    two_sided = np.concatenate([tapered, tapered[-1:0:-1]])
    spec = np.fft.rfft(two_sided).real / fs
    freqs = np.fft.rfftfreq(two_sided.size, d=1.0 / fs)

    neg = float(np.mean(spec < 0))
    return freqs, spec, neg


def resolution_bandwidth(fs: float, nperseg: int, window: str = "hann") -> float:
    """
    Effective resolution bandwidth of a Welch estimate [Hz].

    The equivalent noise bandwidth of the window, ``N sum(w^2) / (sum w)^2``
    bins, converted to Hz. It is the width of the rectangle that would pass the
    same noise power, and it is what decides whether a sharp peak is resolved
    or smeared -- not the bin spacing ``fs / nperseg``, which is finer and
    flatters the estimate.

    Measured: Hann gives exactly 1.5 bins, Hamming 1.363, rectangular 1.0,
    Blackman 1.727 and flat-top 3.770, independent of length.

    Parameters
    ----------
    fs : float
        Sampling frequency [Hz].
    nperseg : int
        Welch segment length in samples.
    window : str
        Window name, as passed to the estimators.
    """
    w = _signal.get_window(window, int(nperseg))
    enbw_bins = len(w) * np.sum(w ** 2) / (np.sum(w) ** 2)
    return float(enbw_bins * fs / nperseg)


def segment_advice(
    x: np.ndarray,
    fs: float,
    nperseg: int | None = None,
    noverlap: int | None = None,
    window: str = "hann",
    n_inputs: int = 1,
    target_segments: int = 20,
    peak_prominence_db: float = 6.0,
    dynamic_range_db: float = 20.0,
    max_peaks: int = 8,
) -> dict:
    """
    Whether ``nperseg`` is long enough to resolve the peaks and short enough to
    average.

    **The two failure modes pull in opposite directions and only one of them is
    obvious.** Too few averages is the familiar one: coherence is biased up by
    about ``q / n_d`` and spectra are noisy, and the cure is a shorter segment.
    The other is that a segment too short to resolve the narrowest peak lets
    leakage smear peaks and notches, which biases coherence *down* and
    manufactures a residual where none exists -- an error that grows as the
    square of the resolution bandwidth and is worst exactly where the signal is
    strongest. Shortening ``nperseg`` to win averages walks straight into it.

    This returns both numbers together so the trade can be made deliberately
    rather than one side at a time.

    Parameters
    ----------
    x : array_like, shape (N,)
        The record. Used to find the sharpest peak; pass the channel whose
        resonances matter.
    fs : float
        Sampling frequency [Hz].
    nperseg : int or None
        Segment length to assess. Defaults to ``min(N, 1024)``, the library's
        own default, which is what makes this useful before choosing one.
    noverlap : int or None
        Overlap in samples, default ``nperseg // 2``.
    window : str
        Window name.
    n_inputs : int
        Number of predictors, for the coherence bias floor ``q / n_d``. 1 for
        an ordinary spectrum or a single-input coherence.
    target_segments : int
        Averages considered adequate, used only to size the record this would
        need. 20 puts the bias floor at 0.05.
    peak_prominence_db : float
        How far a peak must stand above its surroundings, in dB, to count as a
        resonance worth resolving (default 6, a factor of four in power).
        Lower it on a record whose modes barely clear the noise.
    dynamic_range_db : float
        Ignore peaks more than this far below the tallest (default 20 dB).
        Widens the net on a flat spectrum; narrow it on one with a huge
        dynamic range where only the top mode is of interest.
    max_peaks : int
        Consider only the tallest this many peaks. "The sharpest peak" means
        the sharpest one that matters, not the narrowest wiggle on the floor.

    Returns
    -------
    dict with keys
        ``nperseg``, ``n_segments``, ``bias_floor``, ``resolution_bw`` [Hz],
        ``peak_freq`` and ``peak_bandwidth`` [Hz] of the sharpest peak found
        (``None`` if none was), ``ratio`` -- ``resolution_bw / peak_bandwidth``,
        the quantity the classical rule bounds at 1/4 -- ``resolved``,
        ``nperseg_for_resolution``, ``duration_for_both`` [s], ``verdict``
        (one of ``"ok"``, ``"unresolved"``, ``"marginal"``, ``"too_few"``,
        ``"squeezed"``) and ``advice``, a sentence naming what to change.

    Notes
    -----
    **An unresolved peak cannot report its own width.** The bandwidth measured
    off a smeared peak is roughly the resolution bandwidth itself, whatever the
    true width is, so ``ratio`` saturates near 1 rather than growing. That is
    why the verdict for ``ratio`` above about 0.9 is "unresolved" rather than a
    number: the honest answer is that the record cannot yet say how narrow the
    peak is, and the test is to lengthen ``nperseg`` until the measured width
    stops shrinking.

    ``"squeezed"`` means both cannot be satisfied on this record: resolving the
    peak leaves too few segments. That is a statement about the *record*, not
    about the parameters, and the fix is more data. ``duration_for_both`` says
    how much.

    See Also
    --------
    resolution_bandwidth, coherence, dspkit.frf.error_spectrum,
    dspkit.peaks.peak_bandwidth
    """
    from .peaks import find_peaks as _find_peaks, peak_bandwidth as _peak_bw

    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if nperseg is None:
        nperseg = min(n, 1024)
    nperseg = int(min(nperseg, n))
    step = nperseg // 2 if noverlap is None else nperseg - int(noverlap)
    n_seg = _welch_segment_count(n, nperseg, noverlap)

    be = resolution_bandwidth(fs, nperseg, window)
    floor = n_inputs / n_seg if n_seg > 0 else float("inf")

    # The sharpest peak, measured at this very resolution -- which is the point:
    # if it comes back at the resolution bandwidth, it is not resolved.
    #
    # Detected on the dB spectrum with a prominence threshold, because on a
    # linear PSD every noise wiggle is a peak and the *narrowest* of those is
    # always about one resolution bandwidth wide. Taking it would report
    # "unresolved" on any record whatsoever, which is a diagnostic that never
    # says anything. A resonance worth resolving stands well clear of the floor,
    # so prominence in dB is the right filter, and only the tallest few are
    # considered -- "the sharpest peak" means the sharpest one that matters.
    peak_f = peak_bw = None
    try:
        freqs, pxx = psd(x, fs, window=window, nperseg=nperseg, noverlap=noverlap)
        db = 10.0 * np.log10(np.maximum(pxx, np.finfo(float).tiny))
        pf, pv, _ = _find_peaks(freqs, db, prominence=peak_prominence_db,
                                max_peaks=max_peaks)
        # And within `dynamic_range_db` of the tallest. With few averages a PSD
        # is so noisy that 6 dB of prominence is nothing -- measured on a 2-DOF
        # record at 4 averages, the narrowest "significant" peak was a noise
        # spike at 182 Hz, fifty-odd dB below the modes. A resonance that
        # matters sits near the top of the spectrum, and saying so is what keeps
        # this from tracking the noise floor as the resolution improves.
        if len(pf):
            keep = pv >= (np.max(pv) - dynamic_range_db)
            pf, pv = pf[keep], pv[keep]
        if len(pf):
            # Width is measured on the linear PSD: half-power is a factor of two
            # in power, which is not what half-height means in dB.
            pfs, bws, _ = _peak_bw(freqs, pxx, pf)
            good = np.isfinite(bws) & (bws > 0)
            if np.any(good):
                i = int(np.argmin(bws[good]))
                peak_f = float(pfs[good][i])
                peak_bw = float(bws[good][i])
    except (ValueError, IndexError):
        pass

    ratio = (be / peak_bw) if peak_bw else None
    resolved = bool(ratio is not None and ratio <= 0.25)

    # What length would resolve it, and what record length would then still
    # leave enough averages at the caller's overlap fraction.
    nperseg_res = None
    duration_both = None
    if peak_bw:
        enbw_bins = be * nperseg / fs
        nperseg_res = int(np.ceil(4.0 * enbw_bins * fs / peak_bw))
        frac = 1.0 - step / nperseg
        samples = nperseg_res * (target_segments * (1.0 - frac) + frac)
        duration_both = float(samples / fs)

    if peak_bw is None:
        verdict = "ok" if n_seg >= target_segments else "too_few"
    elif ratio > 0.9:
        verdict = "unresolved"
    elif ratio > 0.25:
        verdict = "marginal"
    elif n_seg < target_segments:
        verdict = "too_few"
    else:
        verdict = "ok"

    if verdict in ("unresolved", "marginal") and duration_both and \
            duration_both > n / fs * 1.05:
        verdict = "squeezed"

    if verdict == "ok":
        advice = (
            f"nperseg={nperseg} resolves the sharpest peak "
            f"(Be/Br = {ratio:.2f}) and leaves {n_seg} segments "
            f"(bias floor {floor:.3f}). Nothing to change."
            if peak_bw else
            f"No clear peak to resolve; {n_seg} segments is adequate "
            f"(bias floor {floor:.3f})."
        )
    elif verdict == "unresolved":
        advice = (
            f"The sharpest peak near {peak_f:.4g} Hz is not resolved: its "
            f"apparent width {peak_bw:.4g} Hz is the resolution bandwidth "
            f"{be:.4g} Hz, so its true width could be anything smaller. "
            f"Raise nperseg to at least {nperseg_res} and see whether the width "
            f"keeps shrinking. Do not shorten it to win averages — leakage from "
            f"an unresolved peak fabricates residual power where the signal is "
            f"strongest."
        )
    elif verdict == "marginal":
        advice = (
            f"Be/Br = {ratio:.2f} at nperseg={nperseg}; the classical rule asks "
            f"for 0.25 or less. Raise nperseg to about {nperseg_res}, which "
            f"leaves {_welch_segment_count(n, nperseg_res, None)} segment(s) "
            f"here."
        )
    elif verdict == "too_few":
        fix, count = _suggest_nperseg(n, target_segments, nperseg, noverlap)
        advice = (
            f"{n_seg} segment(s) puts the coherence bias floor at "
            f"{floor:.2f}. nperseg={fix} would give {count}, but check the "
            f"resolution before shortening: "
            + (f"Be/Br would rise from {ratio:.2f} to "
               f"{resolution_bandwidth(fs, fix, window) / peak_bw:.2f}."
               if peak_bw else "no peak was found to check against.")
        )
    else:  # squeezed
        advice = (
            f"This record cannot satisfy both. Resolving the peak at "
            f"{peak_f:.4g} Hz needs nperseg>={nperseg_res}, which leaves "
            f"{_welch_segment_count(n, nperseg_res, None)} segment(s) of the "
            f"{n} samples available. About {duration_both:.4g} s of record "
            f"would give both, against the {n / fs:.4g} s you have. Until then "
            f"the choice is which error to prefer, and it should be made "
            f"knowingly."
        )

    return {
        "nperseg": nperseg,
        "n_segments": n_seg,
        "bias_floor": float(floor),
        "resolution_bw": be,
        "peak_freq": peak_f,
        "peak_bandwidth": peak_bw,
        "ratio": ratio,
        "resolved": resolved,
        "nperseg_for_resolution": nperseg_res,
        "duration_for_both": duration_both,
        "record_duration": float(n / fs),
        "verdict": verdict,
        "advice": advice,
    }


AR_METHODS = ("burg", "yule_walker")


def _yule_walker(x: np.ndarray, order: int,
                 history: bool = False) -> tuple[np.ndarray, float]:
    """
    AR coefficients by Levinson-Durbin on the biased autocorrelation.

    With ``history=True`` also returns the prediction error at every order up
    to `order`. The recursion computes them all on the way, so order selection
    costs one pass rather than one fit per candidate.
    """
    n = x.size
    r = np.correlate(x, x, mode="full")[n - 1: n + order] / n
    if r[0] <= 0:
        raise ValueError("The signal has no power; an AR fit is undefined.")
    a = np.zeros(order + 1)
    a[0] = 1.0
    e = float(r[0])
    errs = np.full(order + 1, np.nan)
    errs[0] = e
    for m in range(1, order + 1):
        acc = r[m] + np.dot(a[1:m], r[m - 1:0:-1]) if m > 1 else r[m]
        k = -acc / e
        a[1:m + 1] = a[1:m + 1] + k * a[m - 1::-1][:m]
        e *= (1.0 - k * k)
        if e <= 0:
            if history:
                return a, e, errs
            raise ValueError(
                f"The Levinson recursion lost positivity at order {m}. The "
                f"autocorrelation is not positive definite, which usually means "
                f"the order is far too high for the record."
            )
        errs[m] = e
    return (a, e, errs) if history else (a, e)


def _burg(x: np.ndarray, order: int,
          history: bool = False) -> tuple[np.ndarray, float]:
    """
    AR coefficients by Burg's method.

    Minimises the forward *and* backward prediction error together, subject to
    the Levinson recursion. It does not window the data and cannot produce an
    unstable model, which is why it is the default here: on a short record
    Yule-Walker's implicit windowing broadens and shifts peaks, and that is
    exactly the case an AR spectrum is reached for.
    """
    n = x.size
    f = x.astype(float).copy()
    b = x.astype(float).copy()
    a = np.zeros(order + 1)
    a[0] = 1.0
    e = float(np.dot(x, x) / n)
    errs = np.full(order + 1, np.nan)
    errs[0] = e
    for m in range(1, order + 1):
        # Copies, not views. `f[m:n] = ...` below overwrites the forward error
        # in place, and the backward update on the next line still needs the
        # *old* values — reading them through a view silently uses the new ones
        # and the fit comes out wrong but plausible.
        fn = f[m:n].copy()
        bn = b[m - 1:n - 1].copy()
        den = float(np.dot(fn, fn) + np.dot(bn, bn))
        if den <= 0:
            if history:
                return a, e, errs
            raise ValueError(
                f"Burg's recursion ran out of prediction error at order {m}; "
                f"the record is too short or too nearly deterministic for an "
                f"order this high."
            )
        k = -2.0 * float(np.dot(fn, bn)) / den
        prev = a.copy()
        for i in range(1, m + 1):
            a[i] = prev[i] + k * prev[m - i]
        # The two error sequences advance together, and `b` must end up shifted
        # by one for the next pass — which is what writing the new backward
        # error at [m:n] achieves, since the next iteration reads [m:n-1].
        f[m:n] = fn + k * bn
        b[m:n] = bn + k * fn
        e *= (1.0 - k * k)
        errs[m] = e
    return (a, e, errs) if history else (a, e)


def ar_order_selection(
    x: np.ndarray,
    max_order: int,
    method: str = "burg",
    criterion: str = "aic",
) -> tuple[int, np.ndarray]:
    """
    Choose an AR order by an information criterion.

    Returns ``(best_order, scores)`` where ``scores[p]`` is the criterion at
    order ``p``. ``"aic"`` is the Akaike criterion ``N ln(e_p) + 2p``,
    ``"bic"`` swaps the penalty for ``p ln N`` and so picks lower orders.

    Order selection on a *spectrum* is not the same problem as on a forecast,
    and no criterion settles it: too low and peaks merge, too high and the
    estimate grows spurious ones. Treat the answer as a starting point and look
    at the spectrum.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    fit = _burg if method == "burg" else _yule_walker
    # One pass. Both recursions produce the order-p error on their way to
    # order `max_order`, so refitting per candidate did the same arithmetic
    # `max_order` times over -- measurable on any real record, and the reason
    # the app's AR tab used to sit there computing.
    _, _, errs = fit(x, int(max_order), history=True)
    p_idx = np.arange(max_order + 1, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_e = np.log(np.where(errs > 0, errs, np.nan))
    penalty = 2.0 * p_idx if criterion == "aic" else p_idx * np.log(n)
    scores = n * log_e + penalty
    scores[0] = np.inf                      # order 0 is not a model
    scores = np.where(np.isfinite(scores), scores, np.inf)
    best = int(np.argmin(scores))
    return max(best, 1), scores


def ar_psd(
    x: np.ndarray,
    fs: float,
    order: int | None = None,
    method: str = "burg",
    n_freqs: int = 2048,
    max_order: int | None = None,
    criterion: str = "aic",
    detrend: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Power spectral density from an autoregressive model.

    The third classical spectral estimator, alongside Welch (`psd`) and the
    correlogram (`blackman_tukey_psd`). Instead of averaging periodograms it
    fits an all-pole model to the record and evaluates its transfer function::

        S(f) = e / | 1 + sum_k a_k exp(-2 pi i f k / fs) |^2 / fs

    **Reach for it when the record is too short for Welch.** Welch's resolution
    is set by the segment length and its variance by the number of segments,
    and a short record cannot give both. An AR model is not segmented at all:
    it spends the whole record on one fit, so it can resolve two close modes
    where Welch would need more data than exists.

    Measured on two tones at 10.0 and 10.8 Hz in noise, 2 s at 100 Hz -- 200
    samples in total::

        Welch, nperseg=200 (Be = 0.75 Hz)   one peak at 10.50 Hz
        Welch, nperseg=128 (Be = 1.17 Hz)   one peak at 10.16 Hz
        Welch, nperseg= 64 (Be = 2.34 Hz)   one peak at 10.94 Hz
        AR Burg, order 30                   9.99 and 10.85 Hz

    Welch merges them at every usable segment length; the AR fit separates them
    to within 0.05 Hz. Note the order needed, which is the next point.

    Parameters
    ----------
    x : array_like, shape (N,)
        The record.
    fs : float
        Sampling frequency [Hz].
    order : int or None
        Model order. ``None`` selects one by `criterion` up to `max_order`.
    method : {'burg', 'yule_walker'}
        Burg by default -- it does not window the data and always returns a
        stable model. Yule-Walker is offered because it is what most textbooks
        derive and is useful for comparison.
    n_freqs : int
        Frequencies at which to evaluate, from 0 to Nyquist. This is a drawing
        resolution, not an information one: the model has `order` poles however
        finely it is sampled.
    max_order : int or None
        Ceiling for automatic selection. Defaults to ``min(N // 2, 100)``.
    criterion : {'aic', 'bic'}
        Used only when `order` is None.
    detrend : bool
        Remove the mean first (default True). A DC offset is a pole at zero
        frequency and will otherwise dominate the fit.

    Returns
    -------
    freqs : ndarray
        Frequency vector [Hz].
    pxx : ndarray
        One-sided PSD [units^2/Hz].
    info : dict
        ``order``, ``method``, ``criterion``, ``variance`` (the residual
        prediction error), ``reflection_stable`` and, when the order was
        chosen, ``scores``.

    Notes
    -----
    **The order is the estimate.** Unlike Welch, where a bad `nperseg` gives a
    blurred but honest picture, a bad AR order changes what the spectrum *says*.
    Too low and close modes merge into one broad peak; too high and the model
    spends poles on noise. Measured on the library's 2-DOF example, displacement
    output (N = 20480, fs = 1024 Hz, true modes 8.613 Hz at 1.22% damping and
    20.795 Hz at 2.94%)::

        order   2    one peak, at 10.5 Hz — the two modes merged
        order   4    one peak, at 10.5 Hz
        order  22    8.504 and 21.761 Hz
        order 100    8.504 and 20.760 Hz
        AIC -> 83    8.504 and 20.760 Hz
        BIC -> 46    8.504 and 21.010 Hz

    An order of 4 is *not* enough for a 2-DOF system, which is the trap: the
    rule of thumb "two poles per mode" describes a noiseless model, not a fit to
    a finite noisy record. AIC and BIC both landed somewhere workable here, but
    AIC will happily run to whatever `max_order` allows on a long record — it
    chose the ceiling on the acceleration channel of the same fixture.

    **It is a model, not a measurement.** An AR spectrum is smooth and confident
    everywhere, including where the data says nothing, and it has no equivalent
    of a confidence interval falling out of the segment count. Cross-check
    against `psd` before believing a peak that only the AR estimate shows.

    Peaks are also biased: an all-pole model represents a resonance exactly and
    an antiresonance only by cancellation, so notches come out shallower than
    they are. Use `psd` where the notch matters.

    See Also
    --------
    psd, blackman_tukey_psd, segment_advice, ar_order_selection

    Examples
    --------
    >>> import numpy as np
    >>> from dspkit.spectral import ar_psd
    >>> from dspkit._testing import generate_2dof
    >>> t, a1, a2 = generate_2dof(duration=20.0, fs=1024.0, seed=0)
    >>> f, p, info = ar_psd(a1, 1024.0, order=40)
    >>> info['order'], info['reflection_stable']
    (40, True)
    """
    if method not in AR_METHODS:
        raise ValueError(
            f"method must be one of {AR_METHODS}, not {method!r}."
        )
    x = np.asarray(x, dtype=float).ravel()
    if x.size < 8:
        raise ValueError(f"An AR fit needs more than {x.size} samples.")
    if detrend:
        x = x - x.mean()

    scores = None
    if order is None:
        if max_order is None:
            max_order = int(min(x.size // 2, 100))
        order, scores = ar_order_selection(x, max_order, method, criterion)
    order = int(order)
    if order < 1:
        raise ValueError("order must be at least 1.")
    if order >= x.size:
        raise ValueError(
            f"order={order} needs more than {x.size} samples to fit."
        )

    a, e = (_burg if method == "burg" else _yule_walker)(x, order)

    freqs = np.linspace(0.0, fs / 2.0, int(n_freqs))
    k = np.arange(order + 1)[:, None]
    denom = a[:, None] * np.exp(-2j * np.pi * freqs[None, :] * k / fs)
    h = np.abs(denom.sum(axis=0)) ** 2
    # One-sided: the two-sided density e/(fs |A|^2) doubled everywhere except
    # DC and Nyquist, which have no mirror image to fold in.
    pxx = e / (fs * np.maximum(h, np.finfo(float).tiny))
    pxx[1:-1] *= 2.0

    roots = np.roots(a) if order > 0 else np.array([])
    info = {
        "order": order,
        "method": method,
        "criterion": criterion if scores is not None else None,
        "variance": float(e),
        "reflection_stable": bool(np.all(np.abs(roots) < 1.0)) if roots.size else True,
    }
    if scores is not None:
        info["scores"] = scores
    return freqs, pxx, info
