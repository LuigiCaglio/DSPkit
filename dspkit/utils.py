"""
Signal pre-processing and scalar metrics.

Functions
---------
detrend             -- remove polynomial trend (mean, linear, or arbitrary order)
rms                 -- root mean square
peak                -- maximum absolute value
crest_factor        -- peak / RMS
integrate           -- cumulative time integration (accel -> vel, vel -> disp)
differentiate       -- numerical differentiation (vel -> accel, disp -> vel)
"""

import numpy as np
from scipy import signal as _signal
from scipy.integrate import cumulative_trapezoid


# ---------------------------------------------------------------------------
# Trend removal
# ---------------------------------------------------------------------------

def detrend(x: np.ndarray, order: int = 1) -> np.ndarray:
    """
    Remove a polynomial trend from a signal.

    Parameters
    ----------
    x : array_like, shape (N,)
        Input signal.
    order : int
        Polynomial order.
        ``0`` removes the mean, ``1`` removes a linear trend,
        higher orders fit and subtract a polynomial of that degree.

    Returns
    -------
    ndarray
        Detrended signal with the same shape as ``x``.
    """
    x = np.asarray(x, dtype=float)
    if order == 0:
        return x - x.mean()
    if order == 1:
        return _signal.detrend(x, type="linear")
    # General polynomial detrending
    t = np.arange(len(x))
    coeffs = np.polyfit(t, x, order)
    return x - np.polyval(coeffs, t)


# ---------------------------------------------------------------------------
# Scalar metrics
# ---------------------------------------------------------------------------

def rms(x: np.ndarray) -> float:
    """
    Root mean square of a signal.

    For a pure sine of amplitude A, RMS = A / sqrt(2).
    """
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x ** 2)))


def peak(x: np.ndarray) -> float:
    """Maximum absolute value of a signal (0-to-peak amplitude)."""
    return float(np.max(np.abs(x)))


def crest_factor(x: np.ndarray) -> float:
    """
    Crest factor: peak / RMS.

    For a pure sine: sqrt(2) ~ 1.414.
    For white Gaussian noise: typically 3–4.
    High crest factor indicates impulsive content.
    """
    return peak(x) / rms(x)


# ---------------------------------------------------------------------------
# Integration and differentiation
# ---------------------------------------------------------------------------

def integrate(
    x: np.ndarray,
    fs: float,
    detrend_after: bool = True,
    detrend_order: int = 1,
) -> np.ndarray:
    """
    Cumulative time integration using the trapezoidal rule.

    Typical use: acceleration → velocity, velocity → displacement.

    Real sensor signals contain a small DC bias that grows unboundedly
    when integrated. ``detrend_after=True`` (default) removes a linear
    trend from the result, which suppresses this drift while preserving
    the physically meaningful AC content.

    Parameters
    ----------
    x : array_like, shape (N,)
        Input signal.
    fs : float
        Sampling frequency [Hz].
    detrend_after : bool
        If ``True``, apply ``detrend(result, order=detrend_order)`` after
        integration to remove integration drift.
    detrend_order : int
        Polynomial order for post-integration detrending (default 1 = linear).

    Returns
    -------
    ndarray, shape (N,)
        Integrated signal. The first sample is set to zero (initial condition).
    """
    x = np.asarray(x, dtype=float)
    result = cumulative_trapezoid(x, dx=1.0 / fs, initial=0.0)
    if detrend_after:
        result = detrend(result, order=detrend_order)
    return result


def differentiate(x: np.ndarray, fs: float) -> np.ndarray:
    """
    Numerical differentiation using central differences (numpy.gradient).

    Uses second-order accurate central differences at interior points and
    first-order forward/backward differences at the edges. The output has
    the same length as the input.

    Typical use: displacement → velocity, velocity → acceleration.

    Parameters
    ----------
    x : array_like, shape (N,)
        Input signal.
    fs : float
        Sampling frequency [Hz].

    Returns
    -------
    ndarray, shape (N,)
        Derivative, in units of [x_units * Hz].
    """
    x = np.asarray(x, dtype=float)
    return np.gradient(x, 1.0 / fs)
