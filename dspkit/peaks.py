"""
Peak detection and characterisation for spectral data.

Functions
---------
find_peaks          -- detect peaks in a spectrum with prominence filtering
peak_bandwidth      -- half-power (-3 dB) bandwidth of each detected peak
find_harmonics      -- identify harmonic series from a fundamental frequency
"""

import numpy as np
from scipy import signal as _signal


def find_peaks(
    freqs: np.ndarray,
    spectrum: np.ndarray,
    prominence: float | None = None,
    height: float | None = None,
    distance_hz: float | None = None,
    max_peaks: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect peaks in a frequency spectrum.

    Wraps ``scipy.signal.find_peaks`` with frequency-aware defaults suitable
    for PSD / FFT spectra.

    Parameters
    ----------
    freqs : array_like, shape (M,)
        Frequency vector [Hz].
    spectrum : array_like, shape (M,)
        Spectral amplitude or PSD values (real, non-negative).
    prominence : float or None
        Minimum peak prominence (in the same units as ``spectrum``).
        Peaks below this prominence are discarded.
        Default ``None`` keeps all peaks.
    height : float or None
        Minimum absolute peak height.
    distance_hz : float or None
        Minimum horizontal distance between peaks [Hz].
        Converted to samples internally.
    max_peaks : int or None
        Return only the ``max_peaks`` most prominent peaks.

    Returns
    -------
    peak_freqs : ndarray
        Frequencies of detected peaks [Hz].
    peak_values : ndarray
        Spectrum values at the detected peaks.
    prominences : ndarray
        Prominence of each peak (useful for ranking).
    """
    freqs = np.asarray(freqs, dtype=float)
    spectrum = np.asarray(spectrum, dtype=float)

    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    distance = max(1, int(round(distance_hz / df))) if distance_hz is not None else None

    kwargs: dict = {}
    if height is not None:
        kwargs["height"] = height
    if distance is not None:
        kwargs["distance"] = distance
    if prominence is not None:
        kwargs["prominence"] = prominence

    idx, properties = _signal.find_peaks(spectrum, **kwargs)

    if len(idx) == 0:
        return np.array([]), np.array([]), np.array([])

    # Compute prominences if not already computed via the prominence kwarg
    if "prominences" in properties:
        proms = properties["prominences"]
    else:
        proms, _, _ = _signal.peak_prominences(spectrum, idx)

    # Sort by prominence (descending) and optionally limit
    order = np.argsort(proms)[::-1]
    if max_peaks is not None:
        order = order[:max_peaks]

    idx = idx[order]
    proms = proms[order]

    return freqs[idx], spectrum[idx], proms


def peak_bandwidth(
    freqs: np.ndarray,
    spectrum: np.ndarray,
    peak_freqs: np.ndarray | None = None,
    rel_height: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate the bandwidth of spectral peaks at a given relative height.

    By default measures the half-power (-3 dB) bandwidth, i.e. the width
    at 50 % of peak height.

    Parameters
    ----------
    freqs : array_like, shape (M,)
        Frequency vector [Hz].
    spectrum : array_like, shape (M,)
        Spectral amplitude or PSD values.
    peak_freqs : array_like or None
        Frequencies of peaks to measure. If ``None``, peaks are detected
        automatically via ``find_peaks``.
    rel_height : float
        Relative height at which to measure width. 0.5 = half-power (-3 dB).

    Returns
    -------
    peak_freqs : ndarray
        Frequencies of the measured peaks [Hz].
    bandwidths : ndarray
        Bandwidth of each peak [Hz].
    q_factors : ndarray
        Quality factor Q = f_peak / bandwidth for each peak.
    """
    freqs = np.asarray(freqs, dtype=float)
    spectrum = np.asarray(spectrum, dtype=float)
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0

    if peak_freqs is None:
        peak_freqs, _, _ = find_peaks(freqs, spectrum)

    peak_freqs = np.atleast_1d(np.asarray(peak_freqs, dtype=float))
    # Map peak frequencies to nearest indices
    idx = np.array([np.argmin(np.abs(freqs - f)) for f in peak_freqs])

    if len(idx) == 0:
        return np.array([]), np.array([]), np.array([])

    widths, _, _, _ = _signal.peak_widths(spectrum, idx, rel_height=rel_height)
    bandwidths = widths * df

    with np.errstate(divide="ignore", invalid="ignore"):
        q_factors = np.where(bandwidths > 0, freqs[idx] / bandwidths, np.inf)

    return freqs[idx], bandwidths, q_factors


def find_harmonics(
    freqs: np.ndarray,
    spectrum: np.ndarray,
    fundamental: float,
    n_harmonics: int = 5,
    tolerance_hz: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Identify harmonic peaks (f0, 2*f0, 3*f0, ...) in a spectrum.

    Parameters
    ----------
    freqs : array_like, shape (M,)
        Frequency vector [Hz].
    spectrum : array_like, shape (M,)
        Spectral amplitude or PSD.
    fundamental : float
        Fundamental frequency [Hz].
    n_harmonics : int
        Number of harmonics to look for (including the fundamental).
    tolerance_hz : float or None
        Search tolerance around each expected harmonic [Hz].
        Defaults to ``2 * df`` where ``df`` is the frequency resolution.

    Returns
    -------
    harmonic_freqs : ndarray, shape (n_found,)
        Detected harmonic frequencies [Hz].
    harmonic_values : ndarray, shape (n_found,)
        Spectrum values at those frequencies.
    harmonic_orders : ndarray, shape (n_found,)
        Harmonic order (1 = fundamental, 2 = second harmonic, ...).
    """
    freqs = np.asarray(freqs, dtype=float)
    spectrum = np.asarray(spectrum, dtype=float)
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0

    if tolerance_hz is None:
        tolerance_hz = 2.0 * df

    found_freqs = []
    found_vals = []
    found_orders = []

    for n in range(1, n_harmonics + 1):
        target = n * fundamental
        if target > freqs[-1]:
            break
        mask = np.abs(freqs - target) <= tolerance_hz
        if not np.any(mask):
            continue
        local_idx = np.where(mask)[0]
        best = local_idx[np.argmax(spectrum[local_idx])]
        found_freqs.append(freqs[best])
        found_vals.append(spectrum[best])
        found_orders.append(n)

    return (
        np.array(found_freqs),
        np.array(found_vals),
        np.array(found_orders, dtype=int),
    )
