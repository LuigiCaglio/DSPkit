"""
Time-frequency analysis.

Functions
---------
stft                -- Short-Time Fourier Transform
cwt_scalogram       -- Continuous Wavelet Transform (complex Morlet)
wigner_ville        -- Wigner-Ville Distribution (WVD)
smoothed_pseudo_wv  -- Smoothed Pseudo Wigner-Ville Distribution (SPWVD)
synchrosqueeze_stft -- Fourier-based synchrosqueezing transform (FSST)

Notes
-----
WVD and SPWVD are O(N²) in both time and memory.  For long signals,
decimate first or analyse a representative short segment.
"""

from typing import Literal
import warnings

import numpy as np
from scipy import signal as _signal
from scipy import ndimage as _ndimage


# ---------------------------------------------------------------------------
# STFT
# ---------------------------------------------------------------------------

def stft(
    x: np.ndarray,
    fs: float,
    window: str = "hann",
    nperseg: int = 256,
    noverlap: int | None = None,
    scaling: Literal["spectrum", "psd"] = "spectrum",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Short-Time Fourier Transform.

    Parameters
    ----------
    x : array_like, shape (N,)
    fs : float
        Sampling frequency [Hz].
    window : str
        Window function (default ``'hann'``).
    nperseg : int
        Segment length [samples]. Larger → better frequency resolution,
        worse time resolution. Default 256.
    noverlap : int or None
        Overlapping samples between segments. Defaults to
        ``nperseg * 3 // 4`` (75 %) for smooth output.
    scaling : {'spectrum', 'psd'}
        ``'spectrum'`` normalises as amplitude; ``'psd'`` as power density.

    Returns
    -------
    freqs : ndarray, shape (nperseg // 2 + 1,)
        Frequency vector [Hz].
    times : ndarray
        Time vector [s], centre of each segment.
    Zxx : ndarray, shape (len(freqs), len(times)), complex
        STFT coefficients. ``np.abs(Zxx)`` is the spectrogram amplitude.
    """
    x = np.asarray(x, dtype=float)
    if noverlap is None:
        noverlap = nperseg * 3 // 4
    freqs, times, Zxx = _signal.stft(
        x, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, scaling=scaling
    )
    return freqs, times, Zxx


# ---------------------------------------------------------------------------
# CWT scalogram
# ---------------------------------------------------------------------------

def cwt_scalogram(
    x: np.ndarray,
    fs: float,
    freqs: np.ndarray | None = None,
    w: float = 6.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Continuous Wavelet Transform scalogram using the complex Morlet wavelet.

    The Morlet wavelet provides excellent time-frequency localisation
    for oscillatory signals and is standard in structural dynamics.

    The parameter ``w`` sets the number of oscillations in the wavelet
    (its centre frequency in rad):

    - Higher ``w`` → better frequency resolution, worse time resolution.
    - Lower  ``w`` → better time resolution, worse frequency resolution.

    Parameters
    ----------
    x : array_like, shape (N,)
    fs : float
        Sampling frequency [Hz].
    freqs : array_like or None
        Analysis frequencies [Hz]. Defaults to 50 log-spaced values from
        1 Hz to ``fs / 4``.
    w : float
        Morlet centre-frequency parameter (default 6.0).

    Returns
    -------
    freqs : ndarray
        Analysis frequencies [Hz].
    times : ndarray, shape (N,)
        Time vector [s].
    W : ndarray, shape (len(freqs), N), complex
        CWT coefficients. ``np.abs(W)`` is the scalogram amplitude;
        ``np.abs(W) ** 2`` is the energy density.
    """
    x = np.asarray(x, dtype=float)
    N = len(x)
    if freqs is None:
        freqs = np.geomspace(1.0, fs / 4.0, num=50)
    else:
        freqs = np.asarray(freqs, dtype=float)

    # Scale-to-frequency: f = w * fs / (2π * a) → a = w * fs / (2π * f)
    scales = w * fs / (2.0 * np.pi * freqs)

    # FFT-based CWT — no external wavelet library required.
    #
    # The analytic complex Morlet wavelet:
    #   ψ(t) = π^(-1/4) * exp(j*w*t) * exp(-t²/2)
    # Its Fourier transform (one-sided, analytic):
    #   Ψ(ω) = π^(-1/4) * sqrt(2π) * exp(-(ω-w)²/2)  for ω > 0,  0 otherwise
    # Scaled version (scale = a):
    #   Ψ_a(ω) = π^(-1/4) * sqrt(2π·a) * exp(-(a·ω - w)²/2)
    #
    # CWT via Fourier convolution theorem:
    #   W(a, b) = IFFT{ X(ω) · Ψ_a*(ω) }
    # Since Ψ_a is real (Gaussian), Ψ_a* = Ψ_a.
    #
    # Zero-padding to 2N reduces circular convolution wrap-around at edges.
    Nfft = 2 * N
    X = np.fft.fft(x, n=Nfft)
    xi = 2.0 * np.pi * np.fft.fftfreq(Nfft)  # angular frequency [rad/sample]

    W = np.zeros((len(scales), N), dtype=complex)
    c = np.pi ** -0.25  # normalisation constant

    for i, a in enumerate(scales):
        psi_hat = np.where(
            xi > 0,
            c * np.sqrt(2.0 * np.pi * a) * np.exp(-0.5 * (a * xi - w) ** 2),
            0.0,
        )
        W[i, :] = np.fft.ifft(X * psi_hat)[:N]

    times = np.arange(N) / fs
    return freqs, times, W


# ---------------------------------------------------------------------------
# Wigner-Ville Distribution
# ---------------------------------------------------------------------------

def wigner_ville(
    x: np.ndarray,
    fs: float,
    warn_above: int = 2048,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Wigner-Ville Distribution (WVD).

    Computed via the analytic signal (Hilbert transform) to suppress
    aliasing artifacts. The result is real-valued.

    The WVD achieves the highest joint time-frequency resolution of any
    bilinear distribution and satisfies both the time and frequency
    marginals exactly for single-component signals. However, it produces
    oscillatory cross-terms between signal components. For noisy or
    multi-component signals, prefer ``smoothed_pseudo_wv``.

    .. warning::
        Computation is O(N²) in time and memory.

    Parameters
    ----------
    x : array_like, shape (N,)
    fs : float
        Sampling frequency [Hz].
    warn_above : int
        Emit a warning when ``len(x)`` exceeds this value (default 2048).

    Returns
    -------
    freqs : ndarray, shape (N // 2 + 1,)
        Frequency vector [Hz], from 0 to fs / 4.
        The half-lag autocorrelation limits the representable frequency
        range to fs/4 (Nyquist of the half-lag domain).
    times : ndarray, shape (N,)
        Time vector [s].
    WVD : ndarray, shape (N, N // 2 + 1)
        Wigner-Ville distribution. May contain negative values (cross-terms
        or edge effects). Integrating over frequency yields instantaneous power.
    """
    x = np.asarray(x, dtype=float)
    N = len(x)

    if N > warn_above:
        warnings.warn(
            f"wigner_ville: N={N} > {warn_above}. WVD is O(N²) — "
            "consider decimating or passing a shorter segment.",
            stacklevel=2,
        )

    z = _signal.hilbert(x)  # analytic signal

    # Build instantaneous autocorrelation R[n, m] = z[n+m] * conj(z[n-m])
    # stored as an N×N complex matrix using circular (Hermitian) indexing:
    #   column 0            → half-lag m=0
    #   columns 1..N//2     → positive half-lags
    #   columns N//2+1..N-1 → negative half-lags (conjugate of positive)
    #
    # Because R[n,m] = A²·exp(j·4π·f₀·m/fs) for a pure tone at f₀,
    # the N-point DFT peaks at bin k = 2·N·f₀/fs.  The correct frequency
    # mapping is therefore  f_k = k·fs/(2N), i.e. rfftfreq(N, d=2/fs).
    # This covers 0 to fs/4 (the Nyquist limit of the half-lag domain).
    # Dividing by fs normalises the power marginal:
    #   ∑_k W[n, k] · Δf  ≈  A²/2   (RMS power of a real sine of amplitude A).
    R = np.zeros((N, N), dtype=complex)
    R[:, 0] = np.abs(z) ** 2  # m=0

    for m in range(1, N // 2 + 1):
        ns = np.arange(m, N - m)  # valid time indices for this lag
        if len(ns) == 0:
            continue
        vals = z[ns + m] * np.conj(z[ns - m])
        R[ns, m] = vals
        R[ns, N - m] = np.conj(vals)  # Hermitian symmetry → real output

    WVD_full = 2.0 * np.real(np.fft.fft(R, axis=1)) / fs
    freqs = np.fft.rfftfreq(N, d=2.0 / fs)   # 0 … fs/4, length N//2+1
    times = np.arange(N) / fs
    return freqs, times, WVD_full[:, : len(freqs)]


# ---------------------------------------------------------------------------
# Smoothed Pseudo Wigner-Ville Distribution
# ---------------------------------------------------------------------------

def smoothed_pseudo_wv(
    x: np.ndarray,
    fs: float,
    lag_samples: int | None = None,
    time_samples: int | None = None,
    warn_above: int = 2048,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Smoothed Pseudo Wigner-Ville Distribution (SPWVD).

    Suppresses the cross-term interference of the plain WVD by applying:

    - A **Hann lag window** of half-length ``lag_samples``: limits the
      autocorrelation lag, smoothing the frequency axis and attenuating
      cross-terms between components at different frequencies.
    - A **Hann time window** of half-length ``time_samples``: smooths the
      time axis and attenuates cross-terms between components at different
      times.

    Increasing either window suppresses more cross-terms but reduces the
    corresponding resolution. Choose window sizes smaller than the signal's
    characteristic time/frequency separation.

    .. warning::
        Computation is O(N²) in time and memory.

    Parameters
    ----------
    x : array_like, shape (N,)
    fs : float
        Sampling frequency [Hz].
    lag_samples : int or None
        Half-length of the Hann lag window [samples].
        Full window = ``2 * lag_samples + 1``. Defaults to ``max(N // 8, 4)``.
    time_samples : int or None
        Half-length of the Hann time window [samples].
        Defaults to ``max(N // 8, 4)``.
    warn_above : int
        Emit a warning when ``len(x)`` exceeds this value (default 2048).

    Returns
    -------
    freqs : ndarray, shape (N // 2 + 1,)
        Frequency vector [Hz], from 0 to fs / 4.
    times : ndarray, shape (N,)
        Time vector [s].
    SPWVD : ndarray, shape (N, N // 2 + 1)
        Smoothed distribution. Tends to non-negative after sufficient smoothing.
    """
    x = np.asarray(x, dtype=float)
    N = len(x)

    if N > warn_above:
        warnings.warn(
            f"smoothed_pseudo_wv: N={N} > {warn_above}. SPWVD is O(N²) — "
            "consider decimating or passing a shorter segment.",
            stacklevel=2,
        )

    if lag_samples is None:
        lag_samples = max(N // 8, 4)
    if time_samples is None:
        time_samples = max(N // 8, 4)

    z = _signal.hilbert(x)

    # Lag (frequency) window: Hann of length 2*L+1
    L = lag_samples
    lag_win = _signal.windows.hann(2 * L + 1)  # lag_win[L + m] = weight at lag m

    R = np.zeros((N, N), dtype=complex)
    R[:, 0] = np.abs(z) ** 2 * lag_win[L]  # m=0

    for m in range(1, min(L + 1, N // 2 + 1)):
        g_m = lag_win[L + m]
        if g_m == 0.0:
            continue
        ns = np.arange(m, N - m)
        if len(ns) == 0:
            continue
        vals = g_m * z[ns + m] * np.conj(z[ns - m])
        R[ns, m] = vals
        R[ns, N - m] = np.conj(vals)

    # Time smoothing via Hann window convolved along axis 0 (time).
    h = _signal.windows.hann(2 * time_samples + 1)
    h = h / h.sum()
    R_real = _ndimage.convolve1d(R.real, h, axis=0, mode="constant", cval=0.0)
    R_imag = _ndimage.convolve1d(R.imag, h, axis=0, mode="constant", cval=0.0)
    R = R_real + 1j * R_imag

    SPWVD_full = 2.0 * np.real(np.fft.fft(R, axis=1)) / fs
    freqs = np.fft.rfftfreq(N, d=2.0 / fs)   # 0 … fs/4, length N//2+1
    times = np.arange(N) / fs
    return freqs, times, SPWVD_full[:, : len(freqs)]


# ---------------------------------------------------------------------------
# Synchrosqueezing (Fourier-based)
# ---------------------------------------------------------------------------

def synchrosqueeze_stft(
    x: np.ndarray,
    fs: float,
    window: str = "hann",
    nperseg: int = 256,
    noverlap: int | None = None,
    threshold: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fourier-based synchrosqueezing transform (FSST).

    A spectrogram smears: a pure tone appears as a ridge as wide as the
    window's bandwidth, and no choice of ``nperseg`` escapes that — it only
    trades smear in frequency for smear in time.

    The magnitude is smeared, but the *phase* is not. At every bin the phase
    derivative still encodes the true instantaneous frequency of whatever
    component dominates there. Synchrosqueezing estimates that frequency and
    **moves** each bin's energy to it, so energy the window scattered across
    many bins collapses back onto the one frequency it came from.

    Reassignment happens in frequency only, never in time. That restriction is
    what makes the transform invertible in principle — unlike classic
    reassignment, which moves energy in both directions and cannot be undone —
    and it is why the coefficients below are summed as complex numbers rather
    than as magnitudes, so an inverse remains possible.

    **The inverse is not implemented here yet.** Mode extraction (integrating
    one ridge back into a time-domain signal, to get a single mode's damping or
    to track a drifting frequency) is the payoff that invertibility buys, and
    it needs a reconstruction routine this module does not currently provide.

    The instantaneous frequency is obtained without numerical differentiation,
    by computing a second STFT with the window's derivative::

        omega(t, eta) = eta - Im( S_dg(t, eta) / (2 pi S_g(t, eta)) )

    which is both cheaper and far better conditioned than differencing phase.

    Parameters
    ----------
    x : array_like, shape (N,)
    fs : float
        Sampling frequency [Hz].
    window : str
        Window function (default ``'hann'``). Must be one scipy recognises.
    nperseg : int
        Segment length [samples]. Default 256.
    noverlap : int or None
        Overlapping samples. Defaults to ``nperseg * 3 // 4``.
    threshold : float
        Bins whose ``|S_g|`` falls below ``threshold * max(|S_g|)`` are
        discarded rather than reassigned. Where the magnitude is near zero the
        phase derivative is noise, and reassigning it scatters speckle across
        the whole plane. Default 1e-3. Set to 0 to keep every bin.

    Returns
    -------
    freqs : ndarray, shape (nperseg // 2 + 1,)
        Frequency vector [Hz] — the same grid the STFT uses, since energy is
        squeezed onto it rather than onto a new one.
    times : ndarray
        Time vector [s], centre of each segment. Unlike :func:`stft` this does
        not start at zero: frames lie wholly inside the signal, so the first
        centre sits half a window in.
    Tx : ndarray, shape (len(freqs), len(times)), complex
        Synchrosqueezed coefficients. ``np.abs(Tx)`` is the sharpened
        time-frequency representation.

    See Also
    --------
    stft : the transform this sharpens, and the one to compare it against.

    Notes
    -----
    This is the **first-order** transform. It assumes each component's
    amplitude and frequency vary slowly within one window, so the estimate is
    biased for strongly chirping components — a fast chirp reassigns to a
    frequency that lags the true one. Second-order variants correct this.

    It sharpens; it does not create resolution. Two components closer together
    than the window bandwidth are not separated — they merge into a single
    sharp ridge, which is arguably more misleading than a blurry one, because
    it looks confident. Choose ``nperseg`` so the components you care about are
    resolved *before* squeezing.

    References
    ----------
    Daubechies, Lu & Wu (2011), "Synchrosqueezed wavelet transforms: an
    empirical mode decomposition-like tool", Appl. Comput. Harmon. Anal. 30(2).
    Oberlin, Meignen & Perrier (2014), "The Fourier-based synchrosqueezing
    transform", ICASSP.

    Examples
    --------
    >>> import numpy as np
    >>> from dspkit.timefreq import synchrosqueeze_stft
    >>> fs = 1000.0
    >>> t = np.arange(fs) / fs
    >>> x = np.sin(2 * np.pi * 100 * t)
    >>> freqs, times, Tx = synchrosqueeze_stft(x, fs, nperseg=128)
    >>> # energy concentrates in the bin nearest 100 Hz
    >>> int(round(freqs[np.argmax(np.abs(Tx).sum(axis=1))]))
    100
    """
    x = np.asarray(x, dtype=float)
    if nperseg <= 0:
        raise ValueError("nperseg must be positive")
    if noverlap is None:
        noverlap = nperseg * 3 // 4
    if not 0 <= noverlap < nperseg:
        raise ValueError("noverlap must satisfy 0 <= noverlap < nperseg")
    if threshold < 0:
        raise ValueError("threshold must be non-negative")

    g = _signal.get_window(window, nperseg, fftbins=True).astype(float)

    # Derivative of the window, in the units the frequency axis uses (per
    # second, hence the factor fs). np.gradient is second-order accurate in the
    # interior, which is ample for the smooth windows in use here.
    dg = np.gradient(g) * fs

    # The two STFTs are framed here rather than by scipy, because scipy's
    # ``scaling='spectrum'`` divides by ``window.sum()`` — and a symmetric
    # window's derivative sums to zero. That normalisation would divide S_dg by
    # ~1e-16 and destroy the ratio the whole method rests on. Framing both with
    # the *same* unnormalised convention is what makes S_dg / S_g meaningful.
    hop = nperseg - noverlap
    if len(x) < nperseg:
        raise ValueError(
            f"signal is shorter than one segment ({len(x)} < {nperseg})")

    # Deliberately *not* zero-padded at the boundaries, unlike ``stft`` above.
    # Padding invents a step discontinuity where the data stops, and a
    # discontinuity is broadband: the phase derivative across it is meaningless,
    # so those frames reassign to arbitrary rows and pile up there. Measured on
    # a pure tone, the padded edge frames came out 64x larger than the real
    # ridge and swamped it. Frames therefore lie wholly inside the signal —
    # there is no instantaneous frequency to report where there is no data.
    n_frames = 1 + (len(x) - nperseg) // hop
    starts = np.arange(n_frames) * hop
    frames = np.lib.stride_tricks.sliding_window_view(x, nperseg)[starts]

    S_g = np.fft.rfft(frames * g, axis=1).T
    S_dg = np.fft.rfft(frames * dg, axis=1).T
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / fs)

    # Reference the exponential to the window's *centre* rather than the frame
    # start, which for bin j is a factor (-1)**j.
    #
    # This does not change the instantaneous frequency — the factor is common
    # to S_g and S_dg and cancels in their ratio — but it matters enormously to
    # the squeeze. Squeezing sums complex coefficients, so the bins straddling
    # a ridge only reinforce each other if their phases are referenced to a
    # common instant. Referenced to the frame start they alternate in sign
    # across bins and partially cancel, which made |Tx| oscillate by a factor
    # of 30 as a tone's phase drifted against the hop.
    centre_phase = ((-1.0) ** np.arange(len(freqs)))[:, None]
    S_g = S_g * centre_phase
    S_dg = S_dg * centre_phase
    # A frame describes its own centre, which is the instant its estimate
    # belongs to. The first centre therefore sits half a window in, rather than
    # at t=0 where no full window exists.
    times = (starts + nperseg / 2.0) / fs

    # Instantaneous frequency at every bin. Bins with no energy are masked out
    # rather than divided: where |S_g| is near zero the phase derivative is
    # noise, and reassigning it scatters speckle over the whole plane.
    mag = np.abs(S_g)
    peak = mag.max() if mag.size else 0.0
    keep = mag > threshold * peak if peak > 0 else np.zeros(mag.shape, dtype=bool)

    with np.errstate(divide="ignore", invalid="ignore"):
        correction = np.imag(S_dg / S_g) / (2.0 * np.pi)
    omega = np.zeros(mag.shape)
    omega[keep] = (freqs[:, None] - correction)[keep]

    # Squeeze: accumulate each surviving bin into the output row nearest its
    # estimated frequency. Coefficients are summed rather than magnitudes, so
    # the result stays complex and therefore invertible.
    Tx = np.zeros(S_g.shape, dtype=complex)
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    with np.errstate(invalid="ignore"):
        rows = np.rint(np.nan_to_num(omega, nan=-1.0) / df).astype(int)
    valid = keep & (rows >= 0) & (rows < len(freqs))

    src_f, src_t = np.nonzero(valid)
    np.add.at(Tx, (rows[src_f, src_t], src_t), S_g[src_f, src_t])

    return freqs, times, Tx
