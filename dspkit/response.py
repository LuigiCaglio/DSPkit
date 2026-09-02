"""
Single-degree-of-freedom response, and the spectra built from it.

The solver is the Nigam-Jennings piecewise-linear recurrence: a 2x2 state
transition that is *exact* when the input varies linearly between samples,
which is the standard assumption for a sampled record. It is an IIR filter, so
`scipy.signal.lfilter` runs a whole period family in milliseconds rather than
stepping in Python.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import signal as _signal

__all__ = [
    "sdof_response",
    "response_spectrum",
    "log_decrement",
    "random_decrement",
]


def sdof_response(
    accel: np.ndarray,
    fs: float,
    period: float,
    zeta: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Relative response of one SDOF oscillator to a base acceleration record.

    Parameters
    ----------
    accel : array_like, shape (N,)
        Base acceleration history.
    fs : float
        Sampling frequency [Hz].
    period : float
        Undamped natural period T [s].
    zeta : float
        Damping ratio (0.05 = 5% of critical).

    Returns
    -------
    u : ndarray
        Relative displacement.
    v : ndarray
        Relative velocity.
    a_abs : ndarray
        Absolute acceleration of the mass, ``-2*zeta*wn*v - wn**2*u``.

    Notes
    -----
    The recurrence is exact for an input interpolated linearly between samples.
    That interpolation is the approximation, and it degrades at short periods on
    a coarsely sampled record -- see :func:`response_spectrum`, which warns when
    ``dt`` is not small against ``T``.
    """
    accel = np.asarray(accel, dtype=float)
    if accel.ndim != 1:
        raise ValueError("Expected a one-dimensional acceleration record.")
    if period <= 0:
        raise ValueError("period must be positive.")
    if not 0 <= zeta < 1:
        raise ValueError("zeta must be in [0, 1).")

    dt = 1.0 / fs
    wn = 2.0 * np.pi / period
    wd = wn * np.sqrt(1.0 - zeta ** 2)

    e = np.exp(-zeta * wn * dt)
    s = np.sin(wd * dt)
    c = np.cos(wd * dt)

    # State transition A and load operator B (Nigam & Jennings 1969).
    a11 = e * (c + zeta / np.sqrt(1 - zeta ** 2) * s)
    a12 = e / wd * s
    a21 = -wn / np.sqrt(1 - zeta ** 2) * e * s
    a22 = e * (c - zeta / np.sqrt(1 - zeta ** 2) * s)

    zz = zeta / np.sqrt(1 - zeta ** 2)
    b11 = e * (((2 * zeta ** 2 - 1) / (wn ** 2 * dt) + zeta / wn) * s / wd
               + (2 * zeta / (wn ** 3 * dt) + 1 / wn ** 2) * c) - 2 * zeta / (wn ** 3 * dt)
    b12 = -e * (((2 * zeta ** 2 - 1) / (wn ** 2 * dt)) * s / wd
                + (2 * zeta / (wn ** 3 * dt)) * c) - 1 / wn ** 2 + 2 * zeta / (wn ** 3 * dt)
    b21 = e * (((2 * zeta ** 2 - 1) / (wn ** 2 * dt) + zeta / wn) * (c - zz * s)
               - (2 * zeta / (wn ** 3 * dt) + 1 / wn ** 2) * (wd * s + zeta * wn * c)) \
        + 1 / (wn ** 2 * dt)
    b22 = -e * (((2 * zeta ** 2 - 1) / (wn ** 2 * dt)) * (c - zz * s)
                - (2 * zeta / (wn ** 3 * dt)) * (wd * s + zeta * wn * c)) - 1 / (wn ** 2 * dt)

    n = accel.size
    u = np.zeros(n)
    v = np.zeros(n)
    # The load is -accel: base excitation enters as an inertial force.
    p = -accel
    for i in range(1, n):
        u[i] = a11 * u[i - 1] + a12 * v[i - 1] + b11 * p[i - 1] + b12 * p[i]
        v[i] = a21 * u[i - 1] + a22 * v[i - 1] + b21 * p[i - 1] + b22 * p[i]

    a_abs = -(2.0 * zeta * wn * v + wn ** 2 * u)
    return u, v, a_abs


def response_spectrum(
    accel: np.ndarray,
    fs: float,
    periods: np.ndarray | None = None,
    zeta: float | np.ndarray = 0.05,
    warn_short_period: bool = True,
) -> dict:
    """
    Peak SDOF response against period, for one or more damping ratios.

    Parameters
    ----------
    accel : array_like, shape (N,)
        Base acceleration history, in whatever units the results should carry.
    fs : float
        Sampling frequency [Hz].
    periods : array_like or None
        Oscillator periods [s]. Defaults to 100 points log-spaced over
        0.02-10 s, clipped so the shortest period stays resolvable at ``fs``.
    zeta : float or array_like
        One or more damping ratios.
    warn_short_period : bool
        Warn when ``dt > T/10`` for any requested period.

    Returns
    -------
    dict with keys
        ``periods``, ``zeta`` (list), and per damping ratio the arrays
        ``Sd``, ``Sv``, ``Sa`` (true peak relative displacement, relative
        velocity, absolute acceleration) and ``PSv``, ``PSa`` (pseudo-velocity
        and pseudo-acceleration).

    Notes
    -----
    **Pseudo is not the same as true.** ``PSv = w*Sd`` and ``PSa = w**2*Sd`` are
    *definitions*, not the oscillator's actual peak velocity and acceleration.
    They coincide for light damping and separate as damping rises, so both are
    returned rather than one being labelled ambiguously.

    Accuracy at short periods is set by the linear interpolation of the input
    between samples, not by the solver. The usual guidance is ``dt < T/10``;
    below that the curve is not to be trusted, and a warning says so.
    """
    accel = np.asarray(accel, dtype=float)
    dt = 1.0 / fs
    if periods is None:
        periods = np.logspace(np.log10(max(0.02, 10 * dt)), np.log10(10.0), 100)
    periods = np.atleast_1d(np.asarray(periods, dtype=float))
    if np.any(periods <= 0):
        raise ValueError("periods must all be positive.")

    zetas = np.atleast_1d(np.asarray(zeta, dtype=float))

    too_short = periods[periods < 10 * dt]
    if warn_short_period and too_short.size:
        warnings.warn(
            "response_spectrum: {} period(s) are shorter than 10*dt "
            "({:.4g} s). The solver is exact for the interpolated input, but "
            "the interpolation is not, so those points understate the peak. "
            "Resample higher or drop periods below {:.4g} s.".format(
                too_short.size, 10 * dt, 10 * dt),
            stacklevel=2,
        )

    out = {"periods": periods, "zeta": [float(z) for z in zetas]}
    for z in zetas:
        Sd = np.empty(periods.size)
        Sv = np.empty(periods.size)
        Sa = np.empty(periods.size)
        for i, T in enumerate(periods):
            u, v, a_abs = sdof_response(accel, fs, T, float(z))
            Sd[i] = np.max(np.abs(u))
            Sv[i] = np.max(np.abs(v))
            Sa[i] = np.max(np.abs(a_abs))
        w = 2.0 * np.pi / periods
        out[float(z)] = {
            "Sd": Sd, "Sv": Sv, "Sa": Sa,
            "PSv": w * Sd, "PSa": w ** 2 * Sd,
        }
    return out


def log_decrement(
    x: np.ndarray,
    fs: float,
    n_peaks: int | None = None,
    min_prominence: float | None = None,
    floor_fraction: float = 0.05,
) -> dict:
    """
    Damping from the decay of a free-vibration record, by log decrement.

    Fits a straight line to ``ln(peak amplitude)`` against peak index. Using all
    the peaks rather than just the first and last is the point: two peaks give
    an answer that any single noisy peak can move, while the slope of the fit
    averages over the decay and exposes how well it actually fits.

    Parameters
    ----------
    x : array_like, shape (N,)
        Free-decay record. The mean is removed first.
    fs : float
        Sampling frequency [Hz].
    n_peaks : int or None
        Use at most this many peaks from the start of the decay. ``None`` uses
        all of them, which usually means including peaks that have decayed into
        the noise -- see ``r_squared`` before trusting the result.
    min_prominence : float or None
        Passed to peak finding; defaults to 1% of the record's peak amplitude.
    floor_fraction : float
        Stop fitting once peaks fall below this fraction of the first peak.
        Past that point the peaks are noise rather than decay, and including
        them flattens the fit towards zero damping.

    Returns
    -------
    dict with keys
        ``delta`` (log decrement per cycle), ``zeta`` (damping ratio),
        ``fd`` (damped natural frequency [Hz], from the mean peak spacing),
        ``fn`` (undamped natural frequency [Hz]), ``n_peaks_used``,
        ``r_squared`` (of the straight-line fit), ``peak_times``,
        ``peak_amplitudes``.

    Notes
    -----
    This assumes a single decaying mode. On a record with two close modes the
    peak amplitudes beat rather than decay cleanly and the fit degrades, which
    ``r_squared`` will show.

    A high ``r_squared`` is not on its own proof the answer is right. Before
    peak spacing was constrained, a noisy record produced a damping estimate
    half the true value with ``r_squared`` of 0.99: the fit through the peaks
    was excellent, but they were not the peaks of the decay. Check
    ``n_peaks_used`` against the cycles you expect to see in the record.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("Expected a one-dimensional record.")
    # Deliberately NOT mean-removed. A decay is asymmetric in time so its sample
    # mean is not zero, and subtracting it shifts every peak by a constant --
    # a large fraction of the small late peaks and a tiny one of the early ones.
    # That steepens the log slope and inflated the damping by ~5% at every level
    # tested. Remove a genuine DC offset before calling if the record has one.
    if x.size < 8:
        raise ValueError("Record is too short to find a decay.")

    if min_prominence is None:
        min_prominence = 0.01 * np.max(np.abs(x))

    # Require peaks to be at least most of a cycle apart, at the record's own
    # dominant frequency. Without this, noise inserts maxima between the real
    # peaks -- 1550 of them instead of 96 on a 1%-noise decay, median spacing 4
    # samples against an expected 83. Because delta is proportional to the mean
    # peak spacing, that halved the damping estimate while leaving the
    # straight-line fit looking excellent (R-squared 0.99). A confident wrong
    # answer, which is the failure worth engineering against.
    spec = np.abs(np.fft.rfft(x - x.mean()))
    fpk = np.fft.rfftfreq(x.size, d=1.0 / fs)[int(np.argmax(spec))]
    min_distance = max(1, int(0.7 * fs / fpk)) if fpk > 0 else 1

    idx, _ = _signal.find_peaks(x, prominence=min_prominence, distance=min_distance)
    if idx.size < 3:
        raise ValueError(
            "Found fewer than 3 peaks. Log decrement needs a free decay with "
            "several visible cycles; check the record is a decay and not a "
            "forced response."
        )
    amps = x[idx]
    good = amps > 0
    idx, amps = idx[good], amps[good]

    # Stop where the decay reaches the noise. Past that the "peaks" are noise
    # maxima, roughly constant in amplitude, which flattens the fit towards zero
    # damping -- with 1% noise on a 2% damped decay the estimate collapsed to
    # 0.0002 with an R-squared of 0.15. Truncating at a floor keeps the part of
    # the record that is still signal.
    floor = floor_fraction * amps[0]
    keep = np.argmax(amps < floor) if np.any(amps < floor) else amps.size
    if keep >= 3:
        idx, amps = idx[:keep], amps[:keep]

    if n_peaks is not None:
        idx = idx[:max(3, int(n_peaks))]
        amps = amps[:idx.size]
    if idx.size < 3:
        raise ValueError("Fewer than 3 usable peaks above the noise floor.")

    # Fitted against peak *time*, not ordinal index: if a peak is missed the
    # ordinal no longer counts cycles, and the slope comes out too steep.
    t_peak = idx / fs
    ln_a = np.log(amps)
    slope, intercept = np.polyfit(t_peak, ln_a, 1)

    dt = 1.0 / fs
    # Median, not mean: one missed or extra peak should not move it.
    period_d = float(np.median(np.diff(idx))) * dt
    # slope = -zeta*wn, and the peaks are spaced by the damped period.
    delta = float(-slope) * period_d            # log decrement per cycle
    # zeta from delta exactly, not the small-damping delta/(2*pi) shortcut.
    zeta = delta / np.sqrt(4.0 * np.pi ** 2 + delta ** 2)

    fitted = slope * t_peak + intercept
    ss_res = float(np.sum((ln_a - fitted) ** 2))
    ss_tot = float(np.sum((ln_a - ln_a.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    fd = 1.0 / period_d if period_d > 0 else float("nan")
    fn = fd / np.sqrt(1.0 - zeta ** 2) if zeta < 1 else fd

    return {
        "delta": delta,
        "zeta": float(zeta),
        "zeta_pct": float(100 * zeta),
        "fd": float(fd),
        "fn": float(fn),
        "n_peaks_used": int(idx.size),
        "r_squared": float(r2),
        "peak_times": idx * dt,
        "peak_amplitudes": amps,
    }


def random_decrement(
    x: np.ndarray,
    fs: float,
    trigger_level: float | None = None,
    segment_length: int | None = None,
    condition: str = "level_up",
    max_segments: int | None = None,
    y: np.ndarray | None = None,
) -> dict:
    """
    Free-decay signature from an ambient record, by random decrement.

    Log decrement needs a free decay, and ambient vibration is not one. This
    manufactures one. Take many short segments that all begin from the same
    condition -- a level crossing, say -- and average them. The random forcing
    is uncorrelated with the trigger and averages towards zero; the structure's
    own response to that condition is the same every time and survives. What is
    left is proportional to the free decay, and can be handed to
    :func:`log_decrement`.

    Parameters
    ----------
    x : array_like, shape (N,)
        Ambient response record.
    fs : float
        Sampling frequency [Hz].
    trigger_level : float or None
        The level a segment must start from. Defaults to the record's standard
        deviation, which is the usual choice -- see the notes.
    segment_length : int or None
        Samples per segment. Defaults to ``N // 20``, capped at 2000.
    condition : {'level_up', 'level', 'positive_point', 'local_extremum'}
        What counts as a trigger. ``level_up`` requires an upward crossing of
        ``trigger_level`` and is the standard choice: fixing both the level and
        the slope makes every segment start from the same state, so the average
        is a genuine free-decay shape rather than a mixture.
    max_segments : int or None
        Stop after this many triggers.
    y : array_like or None
        Optional second channel. When given, the trigger is taken from ``x``
        and the segments are averaged from ``y``, giving the cross random
        decrement signature -- the equivalent of a cross-correlation and what
        you need for mode shapes across an array.

    Returns
    -------
    dict with keys
        ``tau`` (lag axis [s]), ``signature``, ``n_segments``,
        ``trigger_level``, ``condition``.

    Notes
    -----
    **The trigger level is a trade.** Higher levels give a cleaner starting
    state but fewer segments, so the random part averages out less; lower levels
    give more segments that are individually less alike. One standard deviation
    is the usual compromise, and the segment count returned is what says whether
    it worked -- below roughly 100 the signature is still visibly noisy.

    This assumes the response is stationary and the excitation broadband and
    roughly white. A narrowband or harmonic excitation -- rotating machinery,
    say -- is *correlated* with the trigger and does not average away, so the
    signature then contains the forcing rather than the structure.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("Expected a one-dimensional record.")
    target = x if y is None else np.asarray(y, dtype=float)
    if target.shape != x.shape:
        raise ValueError("Both channels must be the same length.")

    n = x.size
    if segment_length is None:
        segment_length = int(min(2000, max(64, n // 20)))
    segment_length = int(segment_length)
    if segment_length < 8 or segment_length >= n:
        raise ValueError(
            "segment_length must be between 8 and the record length "
            "({}).".format(n)
        )

    sd = float(np.std(x))
    if sd == 0:
        raise ValueError("The record is constant; there is nothing to trigger on.")
    level = sd if trigger_level is None else float(trigger_level)

    cond = condition.lower()
    if cond == "level_up":
        # Upward crossings only: fixing the slope as well as the level means
        # every segment starts from the same state, not from two mirrored ones.
        hits = np.where((x[:-1] < level) & (x[1:] >= level))[0] + 1
    elif cond == "level":
        up = (x[:-1] < level) & (x[1:] >= level)
        down = (x[:-1] > level) & (x[1:] <= level)
        hits = np.where(up | down)[0] + 1
    elif cond == "positive_point":
        hits = np.where(x >= level)[0]
    elif cond == "local_extremum":
        hits = np.where((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]) & (x[1:-1] >= level))[0] + 1
    else:
        raise ValueError(
            "condition must be one of 'level_up', 'level', 'positive_point', "
            "'local_extremum'."
        )

    hits = hits[hits + segment_length < n]
    if max_segments is not None:
        hits = hits[: int(max_segments)]

    if hits.size < 10:
        raise ValueError(
            "Only {} trigger(s) at level {:.4g}. Random decrement needs many "
            "segments to average -- lower the trigger level, or use a longer "
            "record.".format(hits.size, level)
        )

    seg = np.empty((hits.size, segment_length))
    for i, h in enumerate(hits):
        seg[i] = target[h:h + segment_length]
    signature = seg.mean(axis=0)

    return {
        "tau": np.arange(segment_length) / fs,
        "signature": signature,
        "n_segments": int(hits.size),
        "trigger_level": level,
        "trigger_level_sd": level / sd,
        "condition": cond,
        "segment_length": segment_length,
    }
