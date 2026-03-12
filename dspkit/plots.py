"""
Plotting functions for DSP results.

Each function accepts an optional ``ax`` argument.  When ``ax=None`` a new
figure is created automatically.  Every function returns the axes object so
calls can be chained or the axes passed to further customisation.

Functions
---------
plot_signal             -- time-domain waveform (+ optional envelope overlay)
plot_fft                -- single-sided amplitude spectrum
plot_psd                -- power spectral density
plot_coherence          -- magnitude-squared coherence
plot_autocorrelation    -- ACF with optional confidence band
plot_spectrogram        -- STFT magnitude (time × frequency heatmap)
plot_scalogram          -- CWT scalogram  (time × frequency heatmap)
plot_wvd                -- Wigner-Ville or SPWVD distribution
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_ax(ax: Axes | None, figsize: tuple = (9, 4)) -> Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


def _to_db(Z: np.ndarray, floor_db: float) -> np.ndarray:
    """Convert power (or complex amplitude) to dB, clipped at floor_db."""
    power = np.abs(Z) ** 2 if np.iscomplexobj(Z) else np.asarray(Z, dtype=float)
    ref = power.max()
    return 10.0 * np.log10(np.maximum(power, ref * 10.0 ** (floor_db / 10.0)))


def _tf_image(
    ax: Axes,
    times: np.ndarray,
    freqs: np.ndarray,
    C: np.ndarray,        # shape (n_freqs, n_times)
    db: bool,
    floor_db: float,
    cmap: str,
    cb_label: str,
) -> None:
    """Shared pcolormesh helper for time-frequency plots."""
    Z = _to_db(C, floor_db) if db else (np.abs(C) if np.iscomplexobj(C) else C)
    im = ax.pcolormesh(times, freqs, Z, shading="auto", cmap=cmap)
    plt.colorbar(im, ax=ax, label=cb_label, pad=0.01)


# ---------------------------------------------------------------------------
# Time-domain
# ---------------------------------------------------------------------------

def plot_signal(
    t: np.ndarray,
    x: np.ndarray,
    *,
    ax: Axes | None = None,
    label: str | None = None,
    envelope: np.ndarray | None = None,
    title: str = "Signal",
    xlabel: str = "Time [s]",
    ylabel: str = "Amplitude",
    **kwargs,
) -> Axes:
    """
    Plot a time-domain signal.

    Parameters
    ----------
    t, x : array_like
        Time vector [s] and signal values.
    ax : Axes or None
        Target axes; a new figure is created when ``None``.
    label : str or None
        Legend label for the waveform.
    envelope : array_like or None
        Instantaneous envelope (e.g. from ``hilbert_envelope``).
        Drawn as a symmetric shaded band if provided.
    title, xlabel, ylabel : str
    **kwargs
        Forwarded to ``ax.plot``.

    Returns
    -------
    ax : Axes
    """
    ax = _ensure_ax(ax)
    kwargs.setdefault("lw", 0.8)
    ax.plot(t, x, label=label, **kwargs)
    if envelope is not None:
        env = np.asarray(envelope)
        ax.fill_between(t, -env, env, alpha=0.25, color="red", label="envelope")
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid(True, alpha=0.3)
    if label is not None or envelope is not None:
        ax.legend(fontsize=8)
    return ax


# ---------------------------------------------------------------------------
# Spectral
# ---------------------------------------------------------------------------

def plot_fft(
    freqs: np.ndarray,
    amplitude: np.ndarray,
    *,
    ax: Axes | None = None,
    db: bool = False,
    xlim: tuple[float, float] | None = None,
    title: str = "FFT Amplitude Spectrum",
    ylabel: str | None = None,
    **kwargs,
) -> Axes:
    """
    Plot a single-sided amplitude spectrum (output of ``fft_spectrum``).

    Parameters
    ----------
    freqs, amplitude : array_like
        Output of ``fft_spectrum``.
    db : bool
        If ``True``, display as 20·log10(amplitude). Default ``False``.
    xlim : (float, float) or None
        Frequency axis limits.
    **kwargs
        Forwarded to ``ax.plot`` / ``ax.semilogy``.
    """
    ax = _ensure_ax(ax)
    kwargs.setdefault("lw", 0.9)
    if db:
        y = 20.0 * np.log10(np.maximum(amplitude, amplitude.max() * 1e-6))
        ax.plot(freqs, y, **kwargs)
        ax.set_ylabel(ylabel or "Amplitude [dB]")
    else:
        ax.semilogy(freqs, np.maximum(amplitude, amplitude.max() * 1e-12), **kwargs)
        ax.set_ylabel(ylabel or "Amplitude")
    ax.set(xlabel="Frequency [Hz]", title=title)
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.grid(True, which="both", alpha=0.3)
    return ax


def plot_psd(
    freqs: np.ndarray,
    Pxx: np.ndarray,
    *,
    ax: Axes | None = None,
    db: bool = True,
    xlim: tuple[float, float] | None = None,
    title: str = "Power Spectral Density",
    ylabel: str | None = None,
    **kwargs,
) -> Axes:
    """
    Plot a power spectral density estimate (output of ``psd``).

    Parameters
    ----------
    freqs, Pxx : array_like
        Output of ``psd`` or ``csd`` (magnitude used for complex input).
    db : bool
        If ``True`` (default), display in dB relative to the peak.
    xlim : (float, float) or None
    **kwargs
        Forwarded to ``ax.plot`` / ``ax.semilogy``.
    """
    ax = _ensure_ax(ax)
    kwargs.setdefault("lw", 0.9)
    Pxx_real = np.abs(Pxx)
    if db:
        y = 10.0 * np.log10(np.maximum(Pxx_real, Pxx_real.max() * 1e-10))
        ax.plot(freqs, y, **kwargs)
        ax.set_ylabel(ylabel or "PSD [dB re peak]")
    else:
        ax.semilogy(freqs, np.maximum(Pxx_real, Pxx_real.max() * 1e-12), **kwargs)
        ax.set_ylabel(ylabel or "PSD")
    ax.set(xlabel="Frequency [Hz]", title=title)
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.grid(True, which="both", alpha=0.3)
    return ax


def plot_csd(
    freqs: np.ndarray,
    Pxy: np.ndarray,
    *,
    ax: Axes | None = None,
    db: bool = True,
    xlim: tuple[float, float] | None = None,
    title: str = "Cross-Spectral Density",
    ylabel: str | None = None,
    **kwargs,
) -> Axes:
    """
    Plot the magnitude of a cross-spectral density estimate (output of ``csd``).

    Parameters
    ----------
    freqs, Pxy : array_like
        Output of ``csd``. The complex ``Pxy`` is reduced to ``|Pxy|`` before
        plotting.
    db : bool
        If ``True`` (default), display in dB relative to the peak.
    xlim : (float, float) or None
    **kwargs
        Forwarded to ``ax.plot`` / ``ax.semilogy``.
    """
    ax = _ensure_ax(ax)
    kwargs.setdefault("lw", 0.9)
    mag = np.abs(Pxy)
    if db:
        y = 10.0 * np.log10(np.maximum(mag, mag.max() * 1e-10))
        ax.plot(freqs, y, **kwargs)
        ax.set_ylabel(ylabel or "|CSD| [dB re peak]")
    else:
        ax.semilogy(freqs, np.maximum(mag, mag.max() * 1e-12), **kwargs)
        ax.set_ylabel(ylabel or "|CSD|")
    ax.set(xlabel="Frequency [Hz]", title=title)
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.grid(True, which="both", alpha=0.3)
    return ax


def plot_cross_correlation(
    lags: np.ndarray,
    ccf: np.ndarray,
    *,
    ax: Axes | None = None,
    n_samples: int | None = None,
    title: str = "Cross-Correlation",
    xlabel: str = "Lag",
    **kwargs,
) -> Axes:
    """
    Plot the cross-correlation function (output of ``cross_correlation``).

    Parameters
    ----------
    lags, ccf : array_like
        Output of ``cross_correlation``.
    n_samples : int or None
        If provided, overlays ±1.96/sqrt(N) 95 % confidence bands
        (white-noise / independence null hypothesis).
    xlabel : str
        Use ``'Lag [s]'`` or ``'Lag [samples]'`` as appropriate.
    """
    ax = _ensure_ax(ax)
    kwargs.setdefault("lw", 0.8)
    ax.plot(lags, ccf, **kwargs)
    ax.axhline(0, color="black", lw=0.6)
    ax.axvline(0, color="black", lw=0.6, ls="--", alpha=0.4)
    if n_samples is not None:
        ci = 1.96 / np.sqrt(n_samples)
        ax.axhline( ci, color="red", ls="--", lw=0.9, label="95 % CI")
        ax.axhline(-ci, color="red", ls="--", lw=0.9)
        ax.legend(fontsize=8)
    ax.set(xlabel=xlabel, ylabel="CCF [-]", title=title)
    ax.grid(True, alpha=0.3)
    return ax


def plot_coherence(
    freqs: np.ndarray,
    Cxy: np.ndarray,
    *,
    ax: Axes | None = None,
    threshold: float | None = 0.8,
    xlim: tuple[float, float] | None = None,
    title: str = "Magnitude-Squared Coherence",
    **kwargs,
) -> Axes:
    """
    Plot magnitude-squared coherence (output of ``coherence``).

    Parameters
    ----------
    freqs, Cxy : array_like
    threshold : float or None
        Horizontal reference line. Default 0.8.
    xlim : (float, float) or None
    """
    ax = _ensure_ax(ax)
    kwargs.setdefault("lw", 0.9)
    ax.plot(freqs, Cxy, **kwargs)
    if threshold is not None:
        ax.axhline(threshold, color="gray", ls=":", lw=1.0,
                   label=f"threshold = {threshold}")
        ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set(xlabel="Frequency [Hz]", ylabel="Coherence [-]", title=title)
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.grid(True, alpha=0.3)
    return ax


def plot_autocorrelation(
    lags: np.ndarray,
    acf: np.ndarray,
    *,
    ax: Axes | None = None,
    n_samples: int | None = None,
    title: str = "Autocorrelation",
    xlabel: str = "Lag",
    **kwargs,
) -> Axes:
    """
    Plot the autocorrelation function.

    Parameters
    ----------
    lags, acf : array_like
        Output of ``autocorrelation``.
    n_samples : int or None
        If provided, overlays ±1.96/sqrt(N) 95 % confidence bands
        (white-noise null hypothesis).
    xlabel : str
        Use ``'Lag [s]'`` or ``'Lag [samples]'`` as appropriate.
    """
    ax = _ensure_ax(ax)
    kwargs.setdefault("lw", 0.8)
    ax.plot(lags, acf, **kwargs)
    ax.axhline(0, color="black", lw=0.6)
    if n_samples is not None:
        ci = 1.96 / np.sqrt(n_samples)
        ax.axhline( ci, color="red", ls="--", lw=0.9, label="95 % CI")
        ax.axhline(-ci, color="red", ls="--", lw=0.9)
        ax.legend(fontsize=8)
    ax.set(xlabel=xlabel, ylabel="ACF [-]", title=title)
    ax.grid(True, alpha=0.3)
    return ax


# ---------------------------------------------------------------------------
# Time-frequency
# ---------------------------------------------------------------------------

def plot_spectrogram(
    freqs: np.ndarray,
    times: np.ndarray,
    Zxx: np.ndarray,
    *,
    ax: Axes | None = None,
    db: bool = True,
    floor_db: float = -60.0,
    ylim: tuple[float, float] | None = None,
    cmap: str = "inferno",
    title: str = "Spectrogram (STFT)",
    colorbar_label: str | None = None,
) -> Axes:
    """
    Plot an STFT spectrogram.

    Parameters
    ----------
    freqs, times, Zxx : array_like
        Output of ``stft``.
    db : bool
        Display in dB (default ``True``).
    floor_db : float
        Colour scale floor relative to peak [dB]. Default -60.
    ylim : (float, float) or None
        Frequency axis limits.
    cmap : str
        Matplotlib colormap. Default ``'inferno'``.
    """
    ax = _ensure_ax(ax, figsize=(10, 4))
    cb = colorbar_label or ("Power [dB]" if db else "Amplitude")
    _tf_image(ax, times, freqs, Zxx, db, floor_db, cmap, cb)
    ax.set(xlabel="Time [s]", ylabel="Frequency [Hz]", title=title)
    if ylim is not None:
        ax.set_ylim(ylim)
    return ax


def plot_scalogram(
    freqs: np.ndarray,
    times: np.ndarray,
    W: np.ndarray,
    *,
    ax: Axes | None = None,
    db: bool = True,
    floor_db: float = -40.0,
    log_freq: bool = True,
    ylim: tuple[float, float] | None = None,
    cmap: str = "inferno",
    title: str = "CWT Scalogram",
    colorbar_label: str | None = None,
) -> Axes:
    """
    Plot a CWT scalogram (output of ``cwt_scalogram``).

    Parameters
    ----------
    freqs, times, W : array_like
        Output of ``cwt_scalogram``.
    db : bool
        Display in dB (default ``True``).
    floor_db : float
        Colour scale floor relative to peak [dB]. Default -40.
    log_freq : bool
        Log-scale frequency axis (default ``True``).
    ylim : (float, float) or None
    cmap : str
    """
    ax = _ensure_ax(ax, figsize=(10, 4))
    cb = colorbar_label or ("Power [dB]" if db else "Amplitude")
    _tf_image(ax, times, freqs, W, db, floor_db, cmap, cb)
    ax.set(xlabel="Time [s]", ylabel="Frequency [Hz]", title=title)
    if log_freq:
        ax.set_yscale("log")
    if ylim is not None:
        ax.set_ylim(ylim)
    return ax


def plot_peaks(
    freqs: np.ndarray,
    spectrum: np.ndarray,
    peak_freqs: np.ndarray,
    peak_values: np.ndarray,
    *,
    ax: Axes | None = None,
    db: bool = False,
    xlim: tuple[float, float] | None = None,
    title: str = "Spectrum with Detected Peaks",
    ylabel: str | None = None,
    **kwargs,
) -> Axes:
    """
    Plot a spectrum with detected peaks marked.

    Parameters
    ----------
    freqs, spectrum : array_like
        Frequency vector and spectrum values.
    peak_freqs, peak_values : array_like
        Detected peak positions (from ``find_peaks``).
    db : bool
        Display in dB.
    """
    ax = _ensure_ax(ax)
    kwargs.setdefault("lw", 0.9)
    if db:
        y = 10.0 * np.log10(np.maximum(spectrum, spectrum.max() * 1e-10))
        y_peaks = 10.0 * np.log10(np.maximum(peak_values, spectrum.max() * 1e-10))
        ax.plot(freqs, y, **kwargs)
        ax.plot(peak_freqs, y_peaks, "rv", ms=8, label="peaks")
        ax.set_ylabel(ylabel or "Power [dB]")
    else:
        ax.semilogy(freqs, np.maximum(spectrum, spectrum.max() * 1e-12), **kwargs)
        ax.plot(peak_freqs, peak_values, "rv", ms=8, label="peaks")
        ax.set_ylabel(ylabel or "Amplitude")
    ax.set(xlabel="Frequency [Hz]", title=title)
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    return ax


def plot_singular_values(
    freqs: np.ndarray,
    S: np.ndarray,
    *,
    ax: Axes | None = None,
    n_sv: int | None = None,
    db: bool = True,
    peak_freqs: np.ndarray | None = None,
    xlim: tuple[float, float] | None = None,
    title: str = "FDD — Singular Values",
    **kwargs,
) -> Axes:
    """
    Plot FDD singular value curves.

    Parameters
    ----------
    freqs : array_like, shape (M,)
    S : array_like, shape (M, n_channels)
        Singular values from ``fdd_svd``.
    n_sv : int or None
        Number of singular values to plot. Default: all.
    db : bool
        Display in dB (default ``True``).
    peak_freqs : array_like or None
        If provided, mark natural frequency peaks on the first SV curve.
    """
    ax = _ensure_ax(ax)
    kwargs.setdefault("lw", 0.9)
    n_ch = S.shape[1]
    n_sv = n_sv or n_ch

    for i in range(min(n_sv, n_ch)):
        sv = S[:, i]
        label = f"SV{i + 1}"
        if db:
            y = 10.0 * np.log10(np.maximum(sv, sv.max() * 1e-15))
            ax.plot(freqs, y, label=label, **kwargs)
        else:
            ax.semilogy(freqs, np.maximum(sv, sv.max() * 1e-15), label=label, **kwargs)

    if peak_freqs is not None and len(peak_freqs) > 0:
        sv1 = S[:, 0]
        if db:
            sv1_db = 10.0 * np.log10(np.maximum(sv1, sv1.max() * 1e-15))
            peak_idx = [np.argmin(np.abs(freqs - f)) for f in peak_freqs]
            ax.plot(peak_freqs, sv1_db[peak_idx], "rv", ms=8, label="modes")
        else:
            peak_idx = [np.argmin(np.abs(freqs - f)) for f in peak_freqs]
            ax.plot(peak_freqs, sv1[peak_idx], "rv", ms=8, label="modes")

    ax.set(xlabel="Frequency [Hz]",
           ylabel="Singular Value [dB]" if db else "Singular Value",
           title=title)
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    return ax


def plot_mode_shape(
    mode: np.ndarray,
    *,
    ax: Axes | None = None,
    sensor_labels: list[str] | None = None,
    title: str = "Mode Shape",
    **kwargs,
) -> Axes:
    """
    Plot a single mode shape as a bar chart.

    Parameters
    ----------
    mode : array_like, shape (n_channels,)
        Mode shape vector (from ``fdd_mode_shapes``).
        Real part is plotted.
    sensor_labels : list of str or None
        Labels for each sensor/channel.
    """
    ax = _ensure_ax(ax)
    mode = np.asarray(mode)
    n_ch = len(mode)
    positions = np.arange(n_ch)
    values = mode.real

    colors = ["tab:blue" if v >= 0 else "tab:red" for v in values]
    ax.bar(positions, values, color=colors, edgecolor="black", lw=0.6, **kwargs)
    ax.axhline(0, color="black", lw=0.6)

    if sensor_labels is not None:
        ax.set_xticks(positions)
        ax.set_xticklabels(sensor_labels, rotation=45, ha="right")
    else:
        ax.set_xticks(positions)
        ax.set_xticklabels([f"Ch{i + 1}" for i in range(n_ch)])

    ax.set(ylabel="Amplitude", title=title)
    ax.grid(True, axis="y", alpha=0.3)
    return ax


def plot_pdf(
    xi: np.ndarray,
    density: np.ndarray,
    *,
    ax: Axes | None = None,
    hist_data: np.ndarray | None = None,
    hist_bins: int = 50,
    title: str = "Probability Density Function",
    xlabel: str = "Value",
    **kwargs,
) -> Axes:
    """
    Plot a PDF estimate, optionally with a histogram overlay.

    Parameters
    ----------
    xi, density : array_like
        KDE output from ``pdf_estimate``.
    hist_data : array_like or None
        Raw signal data to overlay as a normalised histogram.
    hist_bins : int
        Number of histogram bins.
    """
    ax = _ensure_ax(ax)
    kwargs.setdefault("lw", 1.2)
    if hist_data is not None:
        ax.hist(hist_data, bins=hist_bins, density=True, alpha=0.3,
                color="steelblue", label="histogram")
    ax.plot(xi, density, color="tab:red", label="KDE", **kwargs)
    ax.set(xlabel=xlabel, ylabel="Density", title=title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return ax


def plot_joint_histogram(
    x_centres: np.ndarray,
    y_centres: np.ndarray,
    H: np.ndarray,
    *,
    ax: Axes | None = None,
    cmap: str = "viridis",
    title: str = "Joint Distribution",
    xlabel: str = "X",
    ylabel: str = "Y",
    colorbar_label: str = "Density",
) -> Axes:
    """
    Plot a 2D joint histogram as a heatmap.

    Parameters
    ----------
    x_centres, y_centres, H : array_like
        Output of ``joint_histogram``.
    """
    ax = _ensure_ax(ax, figsize=(7, 6))
    im = ax.pcolormesh(x_centres, y_centres, H.T, shading="auto", cmap=cmap)
    plt.colorbar(im, ax=ax, label=colorbar_label, pad=0.01)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    return ax


def plot_correlation_matrix(
    R: np.ndarray,
    *,
    ax: Axes | None = None,
    labels: list[str] | None = None,
    cmap: str = "RdBu_r",
    title: str = "Correlation Matrix",
) -> Axes:
    """
    Plot a correlation matrix as a heatmap.

    Parameters
    ----------
    R : array_like, shape (n, n)
        Correlation matrix (from ``correlation_matrix``).
    labels : list of str or None
        Channel labels.
    """
    ax = _ensure_ax(ax, figsize=(7, 6))
    n = R.shape[0]
    im = ax.imshow(R, cmap=cmap, vmin=-1, vmax=1, aspect="equal")
    plt.colorbar(im, ax=ax, label="Correlation", pad=0.01)
    ticks = np.arange(n)
    if labels is not None:
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)
    ax.set_title(title)
    return ax


def plot_indicators(
    times: np.ndarray,
    values: np.ndarray,
    *,
    ax: Axes | None = None,
    title: str = "SHM Indicator",
    ylabel: str = "Value",
    **kwargs,
) -> Axes:
    """
    Plot an SHM indicator time series (RMS variation, energy, frequency shift).

    Parameters
    ----------
    times, values : array_like
        Output of ``rms_variation``, ``energy_variation``, or ``frequency_shift``.
    """
    ax = _ensure_ax(ax)
    kwargs.setdefault("marker", "o")
    kwargs.setdefault("lw", 1.0)
    kwargs.setdefault("ms", 4)
    ax.plot(times, values, **kwargs)
    ax.set(xlabel="Time [s]", ylabel=ylabel, title=title)
    ax.grid(True, alpha=0.3)
    return ax


def plot_wvd(
    freqs: np.ndarray,
    times: np.ndarray,
    WVD: np.ndarray,
    *,
    ax: Axes | None = None,
    clip_negative: bool = True,
    db: bool = True,
    floor_db: float = -40.0,
    ylim: tuple[float, float] | None = None,
    cmap: str = "inferno",
    title: str = "Wigner-Ville Distribution",
    colorbar_label: str | None = None,
) -> Axes:
    """
    Plot a Wigner-Ville or Smoothed Pseudo WVD.

    Parameters
    ----------
    freqs, times, WVD : array_like
        Output of ``wigner_ville`` or ``smoothed_pseudo_wv``.
        WVD is expected in shape ``(n_times, n_freqs)`` as returned by those
        functions; this function transposes internally.
    clip_negative : bool
        Clip negative values to zero before display (default ``True``).
        Negative values are cross-term artifacts; clipping avoids misleading
        colours.
    db : bool
        Display in dB (default ``True``).
    floor_db : float
        Colour scale floor relative to peak [dB]. Default -40.
    ylim : (float, float) or None
    cmap : str
    """
    ax = _ensure_ax(ax, figsize=(10, 4))
    # WVD is (n_times, n_freqs); transpose to (n_freqs, n_times) for pcolormesh
    Z = WVD.T
    if clip_negative:
        Z = np.maximum(Z, 0.0)
    cb = colorbar_label or ("Power [dB]" if db else "Amplitude")
    _tf_image(ax, times, freqs, Z, db, floor_db, cmap, cb)
    ax.set(xlabel="Time [s]", ylabel="Frequency [Hz]", title=title)
    if ylim is not None:
        ax.set_ylim(ylim)
    return ax
