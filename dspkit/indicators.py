"""
Signal indicators.

Signal-derived features that summarise a record or follow how it changes over
time: level, distribution shape, and where the power sits in frequency. All
functions accept plain NumPy arrays.

Functions
---------
spectral_entropy    -- normalised Shannon entropy of a power spectrum
kurtosis            -- fourth standardised moment (impulsiveness indicator)
skewness            -- third standardised moment (asymmetry indicator)
rms_variation       -- RMS variation across signal segments
frequency_shift     -- track dominant frequency changes across segments
energy_variation    -- signal energy variation across segments
windowed_stats      -- level, distribution shape and spectrum, per window
"""

import numpy as np
from scipy import signal as _signal


def spectral_entropy(
    freqs: np.ndarray,
    Pxx: np.ndarray,
) -> float:
    """
    Normalised Shannon entropy of a power spectrum.

    A value near 1.0 indicates a flat (white noise-like) spectrum.
    A value near 0.0 indicates energy concentrated at very few frequencies
    (highly tonal / narrow-band).

    Parameters
    ----------
    freqs : array_like, shape (M,)
        Frequency vector [Hz] (used only for validation; not consumed).
    Pxx : array_like, shape (M,)
        Power spectral density or power spectrum (non-negative).

    Returns
    -------
    float
        Spectral entropy in [0, 1].
    """
    Pxx = np.asarray(Pxx, dtype=float)
    Pxx = np.maximum(Pxx, 0.0)
    total = Pxx.sum()
    if total == 0:
        return 0.0
    p = Pxx / total
    # Avoid log(0) by masking zeros
    nonzero = p > 0
    H = -np.sum(p[nonzero] * np.log(p[nonzero]))
    H_max = np.log(len(p))
    return float(H / H_max) if H_max > 0 else 0.0


def kurtosis(x: np.ndarray, excess: bool = True) -> float:
    """
    Kurtosis (fourth standardised moment) of a signal.

    High kurtosis indicates heavy tails / impulsive content.
    Normal distribution has excess kurtosis = 0 (regular kurtosis = 3).

    Parameters
    ----------
    x : array_like, shape (N,)
    excess : bool
        If ``True`` (default), return excess kurtosis (subtract 3).
        If ``False``, return the regular (non-excess) kurtosis.

    Returns
    -------
    float
    """
    x = np.asarray(x, dtype=float)
    m = x.mean()
    s = x.std()
    if s == 0:
        return 0.0
    k = float(np.mean(((x - m) / s) ** 4))
    return k - 3.0 if excess else k


def skewness(x: np.ndarray) -> float:
    """
    Skewness (third standardised moment) of a signal.

    Positive skewness means the tail on the right side is longer.
    Zero skewness for a symmetric distribution.

    Parameters
    ----------
    x : array_like, shape (N,)

    Returns
    -------
    float
    """
    x = np.asarray(x, dtype=float)
    m = x.mean()
    s = x.std()
    if s == 0:
        return 0.0
    return float(np.mean(((x - m) / s) ** 3))


def rms_variation(
    x: np.ndarray,
    fs: float,
    segment_duration: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    RMS level computed over consecutive non-overlapping segments.

    Useful for tracking amplitude changes over time (e.g. damage progression).

    Parameters
    ----------
    x : array_like, shape (N,)
    fs : float
        Sampling frequency [Hz].
    segment_duration : float or None
        Duration of each segment [s]. Defaults to ``len(x) / fs / 10``
        (ten segments).

    Returns
    -------
    times : ndarray
        Centre time of each segment [s].
    rms_values : ndarray
        RMS value per segment.
    """
    x = np.asarray(x, dtype=float)
    N = len(x)

    if segment_duration is None:
        segment_duration = N / fs / 10.0
    seg_len = max(1, int(segment_duration * fs))

    n_segments = N // seg_len
    if n_segments == 0:
        return np.array([N / (2.0 * fs)]), np.array([np.sqrt(np.mean(x**2))])

    x_trimmed = x[: n_segments * seg_len].reshape(n_segments, seg_len)
    rms_vals = np.sqrt(np.mean(x_trimmed**2, axis=1))
    times = (np.arange(n_segments) + 0.5) * seg_len / fs

    return times, rms_vals


def frequency_shift(
    x: np.ndarray,
    fs: float,
    segment_duration: float | None = None,
    nperseg: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Track the dominant PSD frequency across consecutive segments.

    A shift in the dominant frequency over time may indicate stiffness
    degradation or damage.

    Parameters
    ----------
    x : array_like, shape (N,)
    fs : float
        Sampling frequency [Hz].
    segment_duration : float or None
        Duration of each analysis segment [s]. Defaults to ten segments.
    nperseg : int or None
        Welch PSD segment length within each analysis segment.
        Defaults to ``min(segment_samples, 1024)``.

    Returns
    -------
    times : ndarray
        Centre time of each segment [s].
    dominant_freqs : ndarray
        Dominant (peak PSD) frequency per segment [Hz].
    """
    x = np.asarray(x, dtype=float)
    N = len(x)

    if segment_duration is None:
        segment_duration = N / fs / 10.0
    seg_len = max(1, int(segment_duration * fs))

    n_segments = N // seg_len
    if n_segments == 0:
        n_segments = 1
        seg_len = N

    times = np.zeros(n_segments)
    dominant_freqs = np.zeros(n_segments)

    for i in range(n_segments):
        chunk = x[i * seg_len: (i + 1) * seg_len]
        nps = min(len(chunk), 1024) if nperseg is None else nperseg
        freqs, Pxx = _signal.welch(chunk, fs=fs, nperseg=nps)
        dominant_freqs[i] = freqs[np.argmax(Pxx)]
        times[i] = (i + 0.5) * seg_len / fs

    return times, dominant_freqs


def energy_variation(
    x: np.ndarray,
    fs: float,
    segment_duration: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Signal energy (mean squared value) over consecutive segments.

    Parameters
    ----------
    x : array_like, shape (N,)
    fs : float
        Sampling frequency [Hz].
    segment_duration : float or None
        Duration of each segment [s]. Defaults to ten segments.

    Returns
    -------
    times : ndarray
        Centre time of each segment [s].
    energies : ndarray
        Mean squared value per segment.
    """
    x = np.asarray(x, dtype=float)
    N = len(x)

    if segment_duration is None:
        segment_duration = N / fs / 10.0
    seg_len = max(1, int(segment_duration * fs))

    n_segments = N // seg_len
    if n_segments == 0:
        return np.array([N / (2.0 * fs)]), np.array([np.mean(x**2)])

    x_trimmed = x[: n_segments * seg_len].reshape(n_segments, seg_len)
    energies = np.mean(x_trimmed**2, axis=1)
    times = (np.arange(n_segments) + 0.5) * seg_len / fs

    return times, energies


def windowed_stats(
    x: np.ndarray,
    fs: float,
    window_duration: float | None = None,
    nperseg: int | None = None,
) -> dict:
    """
    Summary statistics of a signal over consecutive windows.

    A general first look at how a record changes over time: level, shape of
    the amplitude distribution, and where the power sits in frequency, one
    value per window. Useful to find transients, handling or impacts, changes
    of operating condition, a sensor that drops out, or a tone that comes and
    goes, before choosing what to analyse.

    Each window is ``window_duration`` long, non-overlapping; a trailing
    partial window is dropped. Values are plotted at the window's centre.

    Parameters
    ----------
    x : array_like, shape (N,)
    fs : float
        Sampling frequency [Hz].
    window_duration : float or None
        Window length [s]. Defaults to the record length / 100, and to at
        least 64 samples.
    nperseg : int or None
        Welch segment length for the per-window spectrum. Defaults to
        ``min(window_samples, 1024)``. Sets the frequency resolution of
        ``dominant_freq``: ``fs / nperseg``.

    Returns
    -------
    dict with arrays of one value per window:

    ``times`` [s]
        Window centres.
    ``rms`` [signal units]
        Root mean square, about zero (so an offset counts).
    ``mean`` [signal units]
        Mean; shows drift or an offset change.
    ``peak`` [signal units]
        Largest absolute value.
    ``crest_factor`` [-]
        ``peak / rms``. About 1.41 for a sine, 3 to 4 for Gaussian noise
        over a few thousand samples; much higher means isolated spikes.
    ``kurtosis`` [-]
        Excess kurtosis of the window about its own mean: 0 for Gaussian,
        -1.5 for a sine, positive for impulsive content (impacts, bursts).
    ``dominant_freq`` [Hz]
        Frequency of the highest bin of the window's Welch PSD.
    ``spectral_entropy`` [-]
        Normalised entropy of that PSD (see ``spectral_entropy``): near 0
        when the power is in one narrow line, near 1 for white noise.

    and the scalars ``window_duration`` [s], ``window_samples``,
    ``nperseg`` and ``df`` [Hz].

    Examples
    --------
    >>> import numpy as np
    >>> from dspkit.indicators import windowed_stats
    >>> fs = 100.0
    >>> t = np.arange(0, 100, 1 / fs)
    >>> x = np.sin(2 * np.pi * 5 * t)
    >>> x[5000:5010] += 20                     # an impact at t = 50 s
    >>> w = windowed_stats(x, fs, window_duration=10.0)
    >>> int(np.argmax(w["kurtosis"])), float(np.round(w["dominant_freq"][0], 1))
    (5, 5.0)
    """
    x = np.asarray(x, dtype=float)
    N = len(x)
    if N < 2:
        raise ValueError("windowed_stats needs at least 2 samples")
    if window_duration is None:
        L = max(64, N // 100)
    else:
        if window_duration <= 0:
            raise ValueError("window_duration must be positive")
        L = int(round(window_duration * fs))
    L = int(min(max(L, 2), N))
    n = N // L
    X = x[: n * L].reshape(n, L)

    rms = np.sqrt(np.mean(X ** 2, axis=1))
    mean = X.mean(axis=1)
    peak = np.max(np.abs(X), axis=1)
    d = X - mean[:, None]
    var = np.mean(d ** 2, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        crest = np.where(rms > 0, peak / rms, np.nan)
        kurt = np.where(var > 0, np.mean(d ** 4, axis=1) / var ** 2 - 3.0, np.nan)

    nps = int(min(L, 1024) if nperseg is None else min(nperseg, L))
    f, P = _signal.welch(X, fs=fs, nperseg=nps, axis=-1)
    dom = f[np.argmax(P, axis=1)]
    Pn = np.maximum(P, 0.0)
    tot = Pn.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(tot > 0, Pn / tot, 0.0)
        H = -np.sum(np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0), axis=1)
    ent = H / np.log(P.shape[1]) if P.shape[1] > 1 else np.zeros(n)

    return {
        "times": (np.arange(n) + 0.5) * L / fs,
        "rms": rms, "mean": mean, "peak": peak,
        "crest_factor": crest, "kurtosis": kurt,
        "dominant_freq": dom, "spectral_entropy": ent,
        "window_duration": L / fs, "window_samples": L,
        "nperseg": nps, "df": fs / nps,
    }
