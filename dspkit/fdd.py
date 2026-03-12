"""
Frequency Domain Decomposition (FDD) for Operational Modal Analysis.

FDD identifies natural frequencies and mode shapes from output-only
(ambient vibration) data. It is the simplest and most widely used
OMA technique, requiring only multi-channel response measurements.

Algorithm
---------
1. Estimate the cross-spectral density (CSD/PSD) matrix G(f) from all
   sensor channels using Welch's method.
2. At each frequency line, perform a Singular Value Decomposition (SVD)
   of G(f):  G(f) = U(f) S(f) U(f)^H
3. Plot the first (largest) singular value vs frequency. Peaks in this
   curve correspond to natural frequencies.
4. At each peak frequency, the first left singular vector U[:,0] is an
   estimate of the mode shape.

The Enhanced FDD (EFDD) extension estimates damping ratios by inverse-FFT
of the singular value bell around each peak, then fitting the free-decay
envelope with the logarithmic decrement.

Functions
---------
fdd_svd             -- compute singular values and vectors of the PSD matrix
fdd_peak_picking    -- automatic or manual peak picking on singular values
fdd_mode_shapes     -- extract mode shapes at given frequencies
efdd_damping        -- Enhanced FDD damping estimation

References
----------
Brincker, R., Zhang, L., Andersen, P. (2001). "Modal identification of
output-only systems using frequency domain decomposition." Smart Materials
and Structures, 10(3), 441.
"""

import numpy as np
from scipy import signal as _signal

from dspkit.multisensor import psd_matrix


def fdd_svd(
    data: np.ndarray,
    fs: float,
    window: str = "hann",
    nperseg: int | None = None,
    noverlap: int | None = None,
    detrend: str | bool = "constant",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the SVD of the PSD matrix at each frequency line.

    This is the core FDD computation. The first singular value curve
    (``S[:, 0]``) is the primary tool for identifying natural frequencies.

    Parameters
    ----------
    data : array_like, shape (n_channels, N)
        Multi-channel time series. Each row is one sensor.
    fs : float
        Sampling frequency [Hz].
    window : str
        Window function for Welch estimation (default ``'hann'``).
    nperseg : int or None
        Welch segment length. Defaults to ``min(N, 1024)``.
    noverlap : int or None
        Overlap between Welch segments.
    detrend : str or False
        Per-segment detrending.

    Returns
    -------
    freqs : ndarray, shape (M,)
        Frequency vector [Hz].
    S : ndarray, shape (M, n_channels)
        Singular values at each frequency. ``S[:, 0]`` is the first
        (largest) singular value — peaks indicate natural frequencies.
    U : ndarray, shape (M, n_channels, n_channels), complex
        Left singular vectors. ``U[k, :, 0]`` is the first mode shape
        estimate at frequency ``freqs[k]``.
    """
    freqs, G = psd_matrix(
        data, fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=detrend,
    )
    n_ch = G.shape[0]
    M = len(freqs)

    S = np.zeros((M, n_ch))
    U = np.zeros((M, n_ch, n_ch), dtype=complex)

    for k in range(M):
        Uk, Sk, _ = np.linalg.svd(G[:, :, k], full_matrices=True)
        S[k, :] = Sk
        U[k, :, :] = Uk

    return freqs, S, U


def fdd_peak_picking(
    freqs: np.ndarray,
    S: np.ndarray,
    prominence: float | None = None,
    distance_hz: float | None = None,
    max_peaks: int | None = None,
    freq_range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pick natural frequency peaks from the first singular value curve.

    Parameters
    ----------
    freqs : ndarray, shape (M,)
        Frequency vector [Hz] (from ``fdd_svd``).
    S : ndarray, shape (M, n_channels)
        Singular values (from ``fdd_svd``).
    prominence : float or None
        Minimum peak prominence in dB. Peaks with less prominence are
        discarded. Default ``None`` keeps all peaks.
    distance_hz : float or None
        Minimum distance between peaks [Hz].
    max_peaks : int or None
        Return at most this many peaks (most prominent first).
    freq_range : (float, float) or None
        Restrict peak search to this frequency range [Hz].

    Returns
    -------
    peak_freqs : ndarray
        Natural frequencies [Hz].
    peak_indices : ndarray of int
        Indices into ``freqs`` / ``S`` for each detected peak.
    """
    sv1 = S[:, 0].copy()

    # Restrict to frequency range
    if freq_range is not None:
        mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        search_indices = np.where(mask)[0]
    else:
        search_indices = np.arange(len(freqs))

    if len(search_indices) == 0:
        return np.array([]), np.array([], dtype=int)

    # Work in dB for peak picking (more natural for prominence thresholds)
    sv1_db = 10.0 * np.log10(np.maximum(sv1[search_indices], sv1.max() * 1e-15))

    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    distance = max(1, int(round(distance_hz / df))) if distance_hz is not None else None

    kwargs: dict = {}
    if prominence is not None:
        kwargs["prominence"] = prominence
    if distance is not None:
        kwargs["distance"] = distance

    local_idx, properties = _signal.find_peaks(sv1_db, **kwargs)

    if len(local_idx) == 0:
        return np.array([]), np.array([], dtype=int)

    # Map back to global indices
    global_idx = search_indices[local_idx]

    # Sort by prominence (descending)
    if "prominences" in properties:
        proms = properties["prominences"]
    else:
        proms, _, _ = _signal.peak_prominences(sv1_db, local_idx)
    order = np.argsort(proms)[::-1]
    if max_peaks is not None:
        order = order[:max_peaks]

    global_idx = global_idx[order]
    return freqs[global_idx], global_idx


def fdd_mode_shapes(
    U: np.ndarray,
    peak_indices: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """
    Extract mode shapes at the identified natural frequencies.

    Parameters
    ----------
    U : ndarray, shape (M, n_channels, n_channels), complex
        Left singular vectors from ``fdd_svd``.
    peak_indices : array_like of int
        Indices of the natural frequencies (from ``fdd_peak_picking``).
    normalize : bool
        If ``True`` (default), normalise each mode shape so that the
        component with the largest magnitude equals 1.0.

    Returns
    -------
    modes : ndarray, shape (n_modes, n_channels), complex
        Mode shapes. ``modes[i, :]`` is the mode shape for the i-th
        natural frequency. Real-valued for well-separated modes.
    """
    peak_indices = np.atleast_1d(np.asarray(peak_indices, dtype=int))
    modes = U[peak_indices, :, 0]  # first singular vector at each peak

    if normalize and modes.size > 0:
        for i in range(len(modes)):
            max_idx = np.argmax(np.abs(modes[i]))
            if np.abs(modes[i, max_idx]) > 0:
                modes[i] = modes[i] / modes[i, max_idx]

    return modes


def efdd_damping(
    freqs: np.ndarray,
    S: np.ndarray,
    U: np.ndarray,
    peak_indices: np.ndarray,
    fs: float,
    mac_threshold: float = 0.8,
    n_crossings: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Enhanced FDD damping estimation via inverse FFT of the SDOF bell.

    For each identified mode:
    1. Extract the singular value bell around the peak by checking the
       Modal Assurance Criterion (MAC) with the peak mode shape.
    2. Inverse-FFT the bell to get the free-decay autocorrelation.
    3. Count zero-crossings and fit the logarithmic decrement to estimate
       the damping ratio.

    Parameters
    ----------
    freqs : ndarray, shape (M,)
        Frequency vector [Hz].
    S : ndarray, shape (M, n_channels)
        Singular values from ``fdd_svd``.
    U : ndarray, shape (M, n_channels, n_channels), complex
        Left singular vectors from ``fdd_svd``.
    peak_indices : array_like of int
        Indices of the natural frequencies.
    fs : float
        Sampling frequency [Hz].
    mac_threshold : float
        MAC threshold for including frequency lines in the SDOF bell.
        Default 0.8.
    n_crossings : int
        Number of zero-crossings to use for damping estimation.
        Default 10.

    Returns
    -------
    damping_ratios : ndarray, shape (n_modes,)
        Estimated damping ratios (fraction of critical).
    natural_freqs : ndarray, shape (n_modes,)
        Refined natural frequency estimates [Hz] from zero-crossing counting.
    """
    peak_indices = np.atleast_1d(np.asarray(peak_indices, dtype=int))
    n_modes = len(peak_indices)
    M = len(freqs)
    df = freqs[1] - freqs[0] if M > 1 else 1.0

    damping_ratios = np.zeros(n_modes)
    natural_freqs = np.zeros(n_modes)

    for m, pk in enumerate(peak_indices):
        phi_ref = U[pk, :, 0]  # reference mode shape at peak

        # Build SDOF bell: include frequency lines where MAC > threshold
        bell = np.zeros(M)
        for k in range(M):
            phi_k = U[k, :, 0]
            mac = _mac(phi_ref, phi_k)
            if mac >= mac_threshold:
                bell[k] = S[k, 0]

        # If the bell is empty or too narrow, skip
        nonzero = np.where(bell > 0)[0]
        if len(nonzero) < 3:
            natural_freqs[m] = freqs[pk]
            damping_ratios[m] = np.nan
            continue

        # Inverse FFT of the one-sided SDOF bell → free-decay autocorrelation
        # Mirror to create a two-sided spectrum for real-valued output
        n_ifft = 2 * (M - 1)
        bell_twosided = np.zeros(n_ifft)
        bell_twosided[:M] = bell
        bell_twosided[M:] = bell[-2:0:-1]
        autocorr = np.fft.ifft(bell_twosided).real
        autocorr = autocorr[:M]  # keep positive lags only

        # Normalise
        if autocorr[0] != 0:
            autocorr = autocorr / autocorr[0]

        # Find zero-crossings for frequency and damping estimation
        crossings = []
        for i in range(len(autocorr) - 1):
            if autocorr[i] * autocorr[i + 1] < 0:
                # Linear interpolation for sub-sample accuracy
                frac = autocorr[i] / (autocorr[i] - autocorr[i + 1])
                crossings.append(i + frac)
                if len(crossings) >= n_crossings:
                    break

        if len(crossings) < 2:
            natural_freqs[m] = freqs[pk]
            damping_ratios[m] = np.nan
            continue

        crossings = np.array(crossings)
        # Period = 2 * (average half-period between consecutive crossings)
        half_periods = np.diff(crossings) / (fs * df)  # time between crossings
        T_avg = 2.0 * np.mean(half_periods)
        fn = 1.0 / T_avg
        natural_freqs[m] = fn

        # Logarithmic decrement from the envelope at zero-crossings
        # Envelope at crossing ≈ magnitude of peaks between crossings
        peaks_env = []
        for i in range(len(crossings) - 1):
            i0 = int(np.floor(crossings[i]))
            i1 = int(np.ceil(crossings[i + 1])) + 1
            i1 = min(i1, len(autocorr))
            if i1 > i0:
                peaks_env.append(np.max(np.abs(autocorr[i0:i1])))

        if len(peaks_env) >= 2:
            peaks_env = np.array(peaks_env)
            # Fit exponential decay: log(env) = -δ·n + const
            # where δ = 2πζ is the logarithmic decrement per half-cycle
            n_env = np.arange(len(peaks_env), dtype=float)
            valid = peaks_env > 0
            if np.sum(valid) >= 2:
                log_env = np.log(peaks_env[valid])
                n_valid = n_env[valid]
                # Linear fit
                coeffs = np.polyfit(n_valid, log_env, 1)
                delta = -coeffs[0]  # logarithmic decrement per half-cycle
                # ζ = δ / (2π) for small damping (per full cycle → divide by 2)
                zeta = delta / (2.0 * np.pi) * 2.0
                damping_ratios[m] = max(zeta, 0.0)
            else:
                damping_ratios[m] = np.nan
        else:
            damping_ratios[m] = np.nan

    return damping_ratios, natural_freqs


def _mac(phi_a: np.ndarray, phi_b: np.ndarray) -> float:
    """
    Modal Assurance Criterion between two mode shape vectors.

    MAC = |φ_a^H · φ_b|² / (|φ_a|² · |φ_b|²)

    Returns a value in [0, 1]. MAC ≈ 1 means the shapes are consistent.
    """
    num = np.abs(np.vdot(phi_a, phi_b)) ** 2
    denom = np.vdot(phi_a, phi_a).real * np.vdot(phi_b, phi_b).real
    if denom == 0:
        return 0.0
    return float(num / denom)
