"""
Digital filter design and application.

All filters use second-order sections (SOS) internally for numerical stability.
Zero-phase filtering (sosfiltfilt) is the default — suitable for offline analysis.
For causal (real-time) filtering, pass ``zero_phase=False``.

Functions
---------
lowpass     -- Butterworth lowpass
highpass    -- Butterworth highpass
bandpass    -- Butterworth bandpass
bandstop    -- Butterworth bandstop
notch       -- IIR notch (single frequency)
decimate    -- downsample with anti-aliasing lowpass
"""

import numpy as np
from scipy import signal as _signal


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply(sos: np.ndarray, x: np.ndarray, zero_phase: bool) -> np.ndarray:
    if zero_phase:
        return _signal.sosfiltfilt(sos, x)
    return _signal.sosfilt(sos, x)


def _validate_cutoff(cutoff: float, fs: float, label: str = "cutoff") -> None:
    nyq = fs / 2.0
    if not (0 < cutoff < nyq):
        raise ValueError(
            f"{label} must be in (0, {nyq}) Hz for fs={fs} Hz, got {cutoff}"
        )


# ---------------------------------------------------------------------------
# Filter functions
# ---------------------------------------------------------------------------

def lowpass(
    x: np.ndarray,
    fs: float,
    cutoff: float,
    order: int = 4,
    zero_phase: bool = True,
) -> np.ndarray:
    """
    Butterworth lowpass filter.

    Parameters
    ----------
    x : array_like, shape (N,)
        Input signal.
    fs : float
        Sampling frequency [Hz].
    cutoff : float
        -3 dB cutoff frequency [Hz].
    order : int
        Filter order. With ``zero_phase=True`` the effective order is doubled
        (forward + reverse pass).
    zero_phase : bool
        If ``True`` (default), use zero-phase filtering (``sosfiltfilt``).
        If ``False``, use causal filtering (``sosfilt``).

    Returns
    -------
    ndarray, shape (N,)
    """
    _validate_cutoff(cutoff, fs, "cutoff")
    sos = _signal.butter(order, cutoff, btype="low", fs=fs, output="sos")
    return _apply(sos, np.asarray(x, dtype=float), zero_phase)


def highpass(
    x: np.ndarray,
    fs: float,
    cutoff: float,
    order: int = 4,
    zero_phase: bool = True,
) -> np.ndarray:
    """
    Butterworth highpass filter.

    Parameters
    ----------
    x : array_like, shape (N,)
    fs : float
        Sampling frequency [Hz].
    cutoff : float
        -3 dB cutoff frequency [Hz].
    order : int
    zero_phase : bool

    Returns
    -------
    ndarray, shape (N,)
    """
    _validate_cutoff(cutoff, fs, "cutoff")
    sos = _signal.butter(order, cutoff, btype="high", fs=fs, output="sos")
    return _apply(sos, np.asarray(x, dtype=float), zero_phase)


def bandpass(
    x: np.ndarray,
    fs: float,
    low: float,
    high: float,
    order: int = 4,
    zero_phase: bool = True,
) -> np.ndarray:
    """
    Butterworth bandpass filter.

    Parameters
    ----------
    x : array_like, shape (N,)
    fs : float
        Sampling frequency [Hz].
    low : float
        Lower -3 dB cutoff [Hz].
    high : float
        Upper -3 dB cutoff [Hz].
    order : int
        Order of each lowpass/highpass section (total order = 2 * order).
    zero_phase : bool

    Returns
    -------
    ndarray, shape (N,)
    """
    _validate_cutoff(low, fs, "low")
    _validate_cutoff(high, fs, "high")
    if low >= high:
        raise ValueError(f"low ({low}) must be less than high ({high})")
    sos = _signal.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return _apply(sos, np.asarray(x, dtype=float), zero_phase)


def bandstop(
    x: np.ndarray,
    fs: float,
    low: float,
    high: float,
    order: int = 4,
    zero_phase: bool = True,
) -> np.ndarray:
    """
    Butterworth bandstop (band-reject) filter.

    Parameters
    ----------
    x : array_like, shape (N,)
    fs : float
        Sampling frequency [Hz].
    low : float
        Lower edge of the stop band [Hz].
    high : float
        Upper edge of the stop band [Hz].
    order : int
    zero_phase : bool

    Returns
    -------
    ndarray, shape (N,)
    """
    _validate_cutoff(low, fs, "low")
    _validate_cutoff(high, fs, "high")
    if low >= high:
        raise ValueError(f"low ({low}) must be less than high ({high})")
    sos = _signal.butter(order, [low, high], btype="bandstop", fs=fs, output="sos")
    return _apply(sos, np.asarray(x, dtype=float), zero_phase)


def notch(
    x: np.ndarray,
    fs: float,
    freq: float,
    q: float = 30.0,
    zero_phase: bool = True,
) -> np.ndarray:
    """
    IIR notch filter at a single frequency.

    Common uses: remove mains hum (50 or 60 Hz) or a known harmonic.

    Parameters
    ----------
    x : array_like, shape (N,)
    fs : float
        Sampling frequency [Hz].
    freq : float
        Notch frequency [Hz].
    q : float
        Quality factor. Higher Q → narrower notch.
        Typical values: 10 (broad) to 50 (narrow). Default 30.
    zero_phase : bool

    Returns
    -------
    ndarray, shape (N,)
    """
    _validate_cutoff(freq, fs, "freq")
    b, a = _signal.iirnotch(freq, q, fs=fs)
    x = np.asarray(x, dtype=float)
    if zero_phase:
        return _signal.filtfilt(b, a, x)
    return _signal.lfilter(b, a, x)


def decimate(
    x: np.ndarray,
    fs: float,
    target_fs: float,
    zero_phase: bool = True,
) -> tuple[np.ndarray, float]:
    """
    Downsample a signal with an anti-aliasing lowpass filter.

    The decimation factor must be a positive integer (fs / target_fs).
    For non-integer ratios, use ``scipy.signal.resample_poly`` directly.

    Parameters
    ----------
    x : array_like, shape (N,)
    fs : float
        Original sampling frequency [Hz].
    target_fs : float
        Target sampling frequency [Hz]. Must satisfy ``fs / target_fs``
        is close to a positive integer.
    zero_phase : bool
        If ``True`` (default), use a FIR anti-aliasing filter with
        zero phase delay. If ``False``, use an IIR (Chebyshev type I)
        filter (matches ``scipy.signal.decimate`` default).

    Returns
    -------
    x_decimated : ndarray
        Downsampled signal.
    target_fs : float
        Actual output sampling frequency (same as ``target_fs`` input,
        returned for convenience when chaining calls).
    """
    ratio = fs / target_fs
    q = int(round(ratio))
    if q < 1:
        raise ValueError(f"target_fs ({target_fs}) must be less than fs ({fs})")
    if not np.isclose(ratio, q, rtol=1e-3):
        raise ValueError(
            f"fs / target_fs = {ratio:.4f} is not close to an integer. "
            "For non-integer ratios use scipy.signal.resample_poly."
        )
    ftype = "fir" if zero_phase else "iir"
    x_dec = _signal.decimate(np.asarray(x, dtype=float), q, ftype=ftype, zero_phase=zero_phase)
    return x_dec, float(target_fs)
