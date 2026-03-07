"""
Instantaneous signal attributes via the analytic signal (Hilbert transform).

For a real narrow-band signal  x(t) = A(t) cos(φ(t)),  the analytic signal is:

    z(t) = x(t) + j H{x}(t) = A(t) exp(j φ(t))

where H{x} is the Hilbert transform (90° phase-shifted version of x).
From z(t) we extract three physically meaningful quantities:

    - Envelope (instantaneous amplitude):  A(t) = |z(t)|
    - Instantaneous phase:                 φ(t) = angle(z(t))  [unwrapped]
    - Instantaneous frequency:             f(t) = (1/2π) dφ/dt

These are meaningful only for narrow-band or single-component signals.
For multi-component signals, apply a bandpass filter or EMD first.

Functions
---------
analytic_signal         -- z(t) = x(t) + j H{x(t)}
hilbert_envelope        -- A(t) = |z(t)|
instantaneous_phase     -- φ(t) = unwrap(∠z(t))   [rad]
instantaneous_freq      -- f(t) = (1/2π) dφ/dt    [Hz]
hilbert_attributes      -- compute all three in one pass (single Hilbert call)
"""

import numpy as np
from scipy import signal as _signal


def analytic_signal(x: np.ndarray) -> np.ndarray:
    """
    Compute the analytic signal via the Hilbert transform.

    z(t) = x(t) + j · H{x}(t)

    where H{x} is the Hilbert transform (all frequency components of x
    phase-shifted by -90°).

    Parameters
    ----------
    x : array_like, shape (N,)

    Returns
    -------
    z : ndarray, complex, shape (N,)
    """
    return _signal.hilbert(np.asarray(x, dtype=float))


def hilbert_envelope(x: np.ndarray) -> np.ndarray:
    """
    Instantaneous amplitude (signal envelope).

    A(t) = |z(t)| = sqrt(x(t)² + H{x}(t)²)

    For a pure tone x(t) = A cos(2π f t), the envelope is the constant A.
    For a modulated signal, it tracks the slow amplitude variation.

    Parameters
    ----------
    x : array_like, shape (N,)

    Returns
    -------
    envelope : ndarray, shape (N,), non-negative
    """
    return np.abs(_signal.hilbert(np.asarray(x, dtype=float)))


def instantaneous_phase(x: np.ndarray) -> np.ndarray:
    """
    Instantaneous phase of the analytic signal (unwrapped).

    φ(t) = unwrap( arctan2( H{x}(t), x(t) ) )

    Unwrapping removes 2π discontinuities so the phase is continuous.
    For a pure tone at frequency f, φ(t) = 2π f t + φ₀ (linear).

    Parameters
    ----------
    x : array_like, shape (N,)

    Returns
    -------
    phase : ndarray, shape (N,)
        Instantaneous phase [rad], unwrapped and continuous.
    """
    z = _signal.hilbert(np.asarray(x, dtype=float))
    return np.unwrap(np.angle(z))


def instantaneous_freq(x: np.ndarray, fs: float) -> np.ndarray:
    """
    Instantaneous frequency via the derivative of the unwrapped phase.

    f_i(t) = (1 / 2π) · dφ/dt

    For a pure sinusoid, this returns the carrier frequency everywhere
    (except at the edges where the derivative stencil degrades).
    For a chirp, it tracks the smoothly varying frequency.
    For broadband noise, the result is meaningless — filter or decompose first.

    Uses ``numpy.gradient`` (central differences), so the output has the
    same length as the input.

    Parameters
    ----------
    x : array_like, shape (N,)
    fs : float
        Sampling frequency [Hz].

    Returns
    -------
    fi : ndarray, shape (N,)
        Instantaneous frequency [Hz].
    """
    phase = instantaneous_phase(x)
    return np.gradient(phase, 1.0 / fs) / (2.0 * np.pi)


def hilbert_attributes(
    x: np.ndarray,
    fs: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute envelope, phase, and instantaneous frequency in a single pass.

    Equivalent to calling ``hilbert_envelope``, ``instantaneous_phase``, and
    ``instantaneous_freq`` separately, but performs the Hilbert transform only
    once.

    Parameters
    ----------
    x : array_like, shape (N,)
    fs : float
        Sampling frequency [Hz].

    Returns
    -------
    envelope : ndarray, shape (N,)
        Instantaneous amplitude [same units as x].
    phase : ndarray, shape (N,)
        Instantaneous phase [rad], unwrapped.
    freq : ndarray, shape (N,)
        Instantaneous frequency [Hz].
    """
    z = _signal.hilbert(np.asarray(x, dtype=float))
    envelope = np.abs(z)
    phase = np.unwrap(np.angle(z))
    freq = np.gradient(phase, 1.0 / fs) / (2.0 * np.pi)
    return envelope, phase, freq
