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


def envelope_spectrum(
    x: np.ndarray,
    fs: float,
    band: tuple[float, float] | None = None,
    filter_order: int = 4,
    nperseg: int | None = None,
    window: str = "hann",
    detrend_envelope: bool = True,
):
    """
    Spectrum of the signal's envelope, optionally within a band.

    The standard tool for finding a repeating impact buried in a resonance --
    a spalled bearing race, a chipped gear tooth. The impacts themselves are
    weak and broadband, but they *excite* a structural resonance far above them,
    and they amplitude-modulate it at the repetition rate. So the fault does not
    appear in the spectrum of the signal, where the resonance dominates; it
    appears in the spectrum of the signal's *envelope*, at the repetition rate
    and its harmonics.

    Parameters
    ----------
    x : array_like, shape (N,)
        Input signal.
    fs : float
        Sampling frequency [Hz].
    band : (low, high) or None
        Band-pass applied before the envelope is taken, in hertz. Choosing it
        around the excited resonance is the whole method -- see the notes.
    filter_order : int
        Butterworth order for that band-pass, applied zero-phase.
    nperseg, window
        Welch parameters for the spectrum of the envelope.
    detrend_envelope : bool
        Remove the envelope's mean before transforming. On by default: the mean
        is a large DC term that otherwise dominates the plot and hides the
        modulation lines, which are what you are looking for.

    Returns
    -------
    freqs : ndarray
        Frequency axis [Hz], spanning modulation rates rather than the
        carrier frequencies of the original signal.
    spectrum : ndarray
        Envelope spectrum.
    envelope : ndarray
        The envelope itself, for plotting against time.

    Notes
    -----
    **The band matters more than anything else here.** Pick it around the
    resonance the impacts excite, which is usually a broad hump well above the
    shaft rate -- often found by looking for the band whose kurtosis is highest,
    since impulsiveness is what marks it. Too wide and the modulation is diluted
    by everything else in the signal; too narrow and the sidebands carrying the
    modulation are filtered out along with the noise.

    Without a band this returns the envelope spectrum of the whole signal, which
    is occasionally what you want and usually not.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("Expected a one-dimensional signal.")

    from .filters import bandpass as _bandpass
    from .spectral import psd as _psd

    sig = x
    if band is not None:
        low, high = float(band[0]), float(band[1])
        nyq = fs / 2.0
        if not 0 < low < high < nyq:
            raise ValueError(
                "Band must satisfy 0 < low < high < {:g} Hz (Nyquist); "
                "got {:g}-{:g}.".format(nyq, low, high)
            )
        sig = _bandpass(x, fs, low, high, order=filter_order, zero_phase=True)

    env = hilbert_envelope(sig)
    env_for_spectrum = env - env.mean() if detrend_envelope else env

    freqs, spectrum = _psd(env_for_spectrum, fs=fs, window=window, nperseg=nperseg)
    return freqs, spectrum, env
