"""
Spectral cube: the cross-spectral matrix of a record, computed once.

For ``n`` channels the cross-spectral matrix at frequency ``f`` is the
``n x n`` Hermitian matrix ``G(f)`` with the auto-PSDs on the diagonal and the
cross-spectra off it. Stacked over frequency it is the "cube". Almost every
multichannel spectral quantity is cheap arithmetic on it:

==========================  ==============================================
PSD of channel i            ``G_ii``
CSD of i and j              ``G_ij``
coherence of i and j        ``|G_ij|^2 / (G_ii G_jj)``
multiple / partial coh.     inverse of the normalised ``G``
FDD                         SVD of ``G(f)`` at each frequency
==========================  ==============================================

The expensive part, the FFT of every Welch segment of every channel, is done
once per channel here. ``psd_matrix`` and ``coherence_matrix`` call scipy once
per channel *pair*, so each channel is transformed ``n`` times over.

The estimate is the same as scipy's Welch (``scipy.signal.csd``) for the same
window, ``nperseg``, ``noverlap`` and ``detrend``: one-sided, mean-averaged,
``G_ij = E[conj(X_i) X_j]`` scaled to a density (or a spectrum). The tests
check every method against the function it replaces.

Functions and classes
---------------------
spectral_cube   -- estimate the cube from data, optionally over chosen spans
SpectralCube    -- the cube and the quantities derived from it
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np
from scipy import signal as _signal

from dspkit.spectral import _check_welch_segments
from dspkit.multisensor import _normalised_csd_inverse

__all__ = ["SpectralCube", "spectral_cube"]

# Welch segments transformed at once. Bounds memory to about
# _BATCH * n_channels * (nperseg / 2 + 1) * 16 bytes.
_BATCH = 64


@dataclass
class SpectralCube:
    """
    Cross-spectral matrix ``G`` of a record, and what derives from it.

    Build it with ``spectral_cube``. The methods return the same shapes as
    the functions they replace, so code using those can switch directly.

    Attributes
    ----------
    freqs : ndarray, shape (M,)
        Frequency vector [Hz].
    G : ndarray, shape (M, n_channels, n_channels), complex
        ``G[k, i, j]`` is the cross-spectrum of channels ``i`` and ``j`` at
        ``freqs[k]``, as ``scipy.signal.csd(data[i], data[j])``. Note the
        frequency axis is first here; ``psd_matrix`` puts it last.
    fs, window, nperseg, noverlap, detrend, scaling
        The estimation parameters. Two cubes combine only if all match.
    n_segments : int
        Welch segments averaged. Sets the bias floor of every coherence.
    n_samples : int
        Samples used, summed over spans.
    labels : list of str or None
        Channel names, if given.
    """

    freqs: np.ndarray
    G: np.ndarray
    fs: float
    window: str
    nperseg: int
    noverlap: int
    detrend: str | Literal[False]
    scaling: str
    n_segments: int
    n_samples: int
    labels: list[str] | None = None
    _svd: tuple | None = field(default=None, repr=False, compare=False)

    # -- shape -------------------------------------------------------------

    @property
    def n_channels(self) -> int:
        return self.G.shape[1]

    @property
    def df(self) -> float:
        """Bin spacing [Hz]."""
        return self.fs / self.nperseg

    def _idx(self, ch) -> int:
        if isinstance(ch, str):
            if self.labels is None or ch not in self.labels:
                raise KeyError(f"no channel named {ch!r}")
            return self.labels.index(ch)
        return int(ch)

    # -- direct reads ------------------------------------------------------

    def psd(self, ch=None) -> np.ndarray:
        """
        Auto-PSD, real. One channel gives shape (M,); ``None`` gives every
        channel, shape (n_channels, M), as ``np.vstack`` of ``dspkit.psd``.
        """
        n = np.arange(self.n_channels)
        P = np.real(self.G[:, n, n]).T
        return P if ch is None else P[self._idx(ch)]

    def csd(self, i, j) -> np.ndarray:
        """Cross-spectrum of ``i`` and ``j``, shape (M,), as ``dspkit.csd``."""
        return self.G[:, self._idx(i), self._idx(j)]

    def psd_matrix(self) -> np.ndarray:
        """``G`` with frequency last, shape (n, n, M), as ``psd_matrix``."""
        return np.moveaxis(self.G, 0, -1)

    # -- coherence ---------------------------------------------------------

    def _check(self, name, hard_min, min_segments, reason):
        _check_welch_segments(
            name, self.n_segments * self.nperseg - (self.n_segments - 1) * self.noverlap,
            self.nperseg, self.noverlap,
            hard_min=hard_min, min_segments=min_segments,
            hard_reason=reason, stacklevel=4,
        )

    def coherence_matrix(self, min_segments: int = 8) -> np.ndarray:
        """
        Pairwise magnitude-squared coherence, shape (n, n, M), diagonal 1,
        as ``coherence_matrix``. Same segment checks.
        """
        self._check(
            "SpectralCube.coherence_matrix", 2, min_segments,
            "A single-segment estimate is identically 1.0 at every frequency "
            "by construction and says nothing about the signals.",
        )
        d = np.real(np.diagonal(self.G, axis1=1, axis2=2))      # (M, n)
        with np.errstate(divide="ignore", invalid="ignore"):
            C = np.abs(self.G) ** 2 / (d[:, :, None] * d[:, None, :])
        C = np.nan_to_num(C, nan=0.0, posinf=0.0)
        n = np.arange(self.n_channels)
        C[:, n, n] = 1.0
        return np.moveaxis(C, 0, -1)

    def coherence(self, i, j, min_segments: int = 8) -> np.ndarray:
        """Magnitude-squared coherence of ``i`` and ``j``, shape (M,)."""
        self._check(
            "SpectralCube.coherence", 2, min_segments,
            "A single-segment estimate is identically 1.0 at every frequency "
            "by construction and says nothing about the signals.",
        )
        i, j = self._idx(i), self._idx(j)
        with np.errstate(divide="ignore", invalid="ignore"):
            c = np.abs(self.G[:, i, j]) ** 2 / (
                np.real(self.G[:, i, i]) * np.real(self.G[:, j, j]))
        return np.nan_to_num(c, nan=0.0, posinf=0.0)

    def _conditioned(self, name, ridge, min_segments):
        n = self.n_channels
        if min_segments is None:
            min_segments = 5 * n
        self._check(
            name, n + 1, max(min_segments, n + 1),
            f"The CSD matrix averages n_segments rank-1 terms: below "
            f"n_channels ({n}) it is singular at every frequency and every "
            f"coherence is identically 1.0, and at exactly n_channels the fit "
            f"has no residual degrees of freedom.",
        )
        return _normalised_csd_inverse(self.psd_matrix(), ridge)

    def multiple_coherence(self, ridge: float = 1e-10,
                           min_segments: int | None = None) -> np.ndarray:
        """
        Each channel against all others, shape (n, M), as
        ``multiple_coherence`` (same ridge and segment checks).
        """
        reg, inv = self._conditioned(
            "SpectralCube.multiple_coherence", ridge, min_segments)
        n = np.arange(self.n_channels)
        g2 = 1.0 - 1.0 / (np.real(reg[:, n, n]) * np.real(inv[:, n, n]))
        return np.clip(g2, 0.0, 1.0).T

    def partial_coherence(self, ridge: float = 1e-10,
                          min_segments: int | None = None) -> np.ndarray:
        """
        Pairwise with the other channels conditioned out, shape (n, n, M),
        as ``partial_coherence``.
        """
        _, inv = self._conditioned(
            "SpectralCube.partial_coherence", ridge, min_segments)
        n = np.arange(self.n_channels)
        d = np.real(inv[:, n, n])
        C = np.clip(np.abs(inv) ** 2 / (d[:, :, None] * d[:, None, :]), 0.0, 1.0)
        C[:, n, n] = 1.0
        return np.moveaxis(C, 0, -1)

    # -- FDD ---------------------------------------------------------------

    def svd(self) -> tuple[np.ndarray, np.ndarray]:
        """
        SVD of ``G(f)`` at each frequency, as ``fdd_svd``: ``S`` shape
        (M, n), ``U`` shape (M, n, n). Computed once, then reused.
        """
        if self._svd is None:
            U, S, _ = np.linalg.svd(self.G)
            self._svd = (S, U)
        return self._svd

    # -- subsets and combination -------------------------------------------

    def select(self, channels: Sequence) -> "SpectralCube":
        """Sub-cube on the given channels (indices or labels), in that order."""
        idx = [self._idx(c) for c in channels]
        return SpectralCube(
            self.freqs, self.G[:, idx][:, :, idx], self.fs, self.window,
            self.nperseg, self.noverlap, self.detrend, self.scaling,
            self.n_segments, self.n_samples,
            None if self.labels is None else [self.labels[i] for i in idx],
        )

    def band(self, fmin: float = 0.0, fmax: float | None = None) -> "SpectralCube":
        """Sub-cube on ``fmin <= f <= fmax``."""
        fmax = self.freqs[-1] if fmax is None else fmax
        m = (self.freqs >= fmin) & (self.freqs <= fmax)
        return SpectralCube(
            self.freqs[m], self.G[m], self.fs, self.window, self.nperseg,
            self.noverlap, self.detrend, self.scaling, self.n_segments,
            self.n_samples, self.labels,
        )

    def _key(self):
        return (self.fs, self.window, self.nperseg, self.noverlap,
                self.detrend, self.scaling, self.n_channels, len(self.freqs))

    @staticmethod
    def average(cubes: Sequence["SpectralCube"]) -> "SpectralCube":
        """
        Combine cubes from separate spans of data (setups, segments, records)
        into one, weighted by their Welch segment counts. The result equals
        averaging every Welch segment of every span together. Refuses cubes
        estimated with different parameters or channel sets.
        """
        cubes = list(cubes)
        if not cubes:
            raise ValueError("SpectralCube.average: no cubes given")
        k0 = cubes[0]._key()
        for c in cubes[1:]:
            if c._key() != k0:
                raise ValueError(
                    "SpectralCube.average: cubes differ in fs, window, "
                    f"nperseg, noverlap, detrend, scaling or channels: "
                    f"{k0} vs {c._key()}")
            if c.labels != cubes[0].labels:
                raise ValueError(
                    "SpectralCube.average: channel labels differ: "
                    f"{cubes[0].labels} vs {c.labels}")
        w = np.array([c.n_segments for c in cubes], dtype=float)
        G = sum(wi * c.G for wi, c in zip(w, cubes)) / w.sum()
        c0 = cubes[0]
        return SpectralCube(
            c0.freqs, G, c0.fs, c0.window, c0.nperseg, c0.noverlap,
            c0.detrend, c0.scaling, int(w.sum()),
            sum(c.n_samples for c in cubes), c0.labels,
        )

    # -- storage -----------------------------------------------------------

    def save(self, path) -> None:
        """Write to a compressed ``.npz`` (for an on-disk cache)."""
        np.savez_compressed(
            path, freqs=self.freqs, G=self.G, fs=self.fs, window=self.window,
            nperseg=self.nperseg, noverlap=self.noverlap,
            detrend="__none__" if self.detrend is False else self.detrend,
            scaling=self.scaling, n_segments=self.n_segments,
            n_samples=self.n_samples,
            labels=np.array([] if self.labels is None else self.labels),
            has_labels=self.labels is not None,
        )

    @classmethod
    def load(cls, path) -> "SpectralCube":
        """Read a cube written by ``save``."""
        z = np.load(path, allow_pickle=False)
        det = str(z["detrend"])
        return cls(
            z["freqs"], z["G"], float(z["fs"]), str(z["window"]),
            int(z["nperseg"]), int(z["noverlap"]),
            False if det == "__none__" else det, str(z["scaling"]),
            int(z["n_segments"]), int(z["n_samples"]),
            [str(s) for s in z["labels"]] if bool(z["has_labels"]) else None,
        )


def _detrend_segments(seg: np.ndarray, detrend) -> np.ndarray:
    """seg shape (batch, n_ch, nperseg); same detrend as scipy's Welch."""
    if detrend is False or detrend is None:
        return seg
    if detrend == "constant":
        return seg - seg.mean(axis=-1, keepdims=True)
    if detrend == "linear":
        return _signal.detrend(seg, axis=-1, type="linear")
    raise ValueError(
        f"detrend must be 'constant', 'linear' or False, got {detrend!r}")


def _span_cube_sum(x, win, nperseg, step, detrend):
    """Sum over Welch segments of conj(X_i) X_j for one span; and the count."""
    n_ch, N = x.shape
    n_seg = (N - (nperseg - step)) // step
    acc = np.zeros((nperseg // 2 + 1, n_ch, n_ch), dtype=complex)
    if n_seg <= 0:
        return acc, 0
    # strided view: (n_ch, n_seg, nperseg), no copy
    view = np.lib.stride_tricks.sliding_window_view(x, nperseg, axis=1)[:, ::step][:, :n_seg]
    for a in range(0, n_seg, _BATCH):
        seg = np.swapaxes(view[:, a:a + _BATCH], 0, 1)        # (b, n_ch, nperseg)
        X = np.fft.rfft(_detrend_segments(seg, detrend) * win, axis=-1)  # (b, n_ch, M)
        acc += np.einsum("bif,bjf->fij", np.conj(X), X, optimize=True)
    return acc, n_seg


def spectral_cube(
    data: np.ndarray,
    fs: float,
    window: str = "hann",
    nperseg: int | None = None,
    noverlap: int | None = None,
    detrend: str | Literal[False] = "constant",
    scaling: Literal["density", "spectrum"] = "density",
    spans: Sequence[tuple[int, int]] | None = None,
    labels: Sequence[str] | None = None,
) -> SpectralCube:
    """
    Estimate the cross-spectral matrix of a multichannel record.

    Parameters
    ----------
    data : array_like, shape (n_channels, N)
        Each row is one channel (DSPkit's convention).
    fs : float
        Sampling frequency [Hz].
    window : str
        Window name for ``scipy.signal.get_window`` (default ``'hann'``).
    nperseg : int or None
        Welch segment length. Defaults to ``min(N, 1024)``, as
        ``psd_matrix``.
    noverlap : int or None
        Overlap. Defaults to ``nperseg // 2``.
    detrend : {'constant', 'linear', False}
        Per-segment detrending, as in scipy.
    scaling : {'density', 'spectrum'}
        ``'density'`` [units^2/Hz] or ``'spectrum'`` [units^2].
    spans : list of (start, stop) sample indices, optional
        Estimate only over these spans, e.g. the setups of a roving test or
        the clean parts of a record. Welch segments never cross a span
        boundary, so a gap or a handling burst between spans does not leak
        in. The result is the average over all segments of all spans. Spans
        shorter than ``nperseg`` contribute nothing.
    labels : list of str, optional
        Channel names, so methods accept names as well as indices.

    Returns
    -------
    SpectralCube

    Examples
    --------
    >>> import numpy as np
    >>> from dspkit.cube import spectral_cube
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(size=(3, 20000))
    >>> cube = spectral_cube(x, 100.0, nperseg=1024)
    >>> cube.G.shape, cube.n_segments
    ((513, 3, 3), 38)
    >>> S, U = cube.svd()          # FDD without re-estimating anything
    """
    x = np.atleast_2d(np.asarray(data, dtype=float))
    n_ch, N = x.shape
    if spans is None:
        spans = [(0, N)]
    spans = [(max(0, int(a)), min(N, int(b))) for a, b in spans]
    longest = max((b - a for a, b in spans), default=0)
    if longest <= 0:
        raise ValueError("spectral_cube: no data in the given spans")

    if nperseg is None:
        nperseg = min(N, 1024)
    nperseg = int(min(nperseg, longest))
    if noverlap is None:
        noverlap = nperseg // 2
    noverlap = int(noverlap)
    step = nperseg - noverlap
    if step <= 0:
        raise ValueError(
            f"spectral_cube: noverlap ({noverlap}) must be below nperseg ({nperseg})")
    if labels is not None and len(labels) != n_ch:
        raise ValueError(
            f"spectral_cube: {len(labels)} labels for {n_ch} channels")

    win = _signal.get_window(window, nperseg)
    acc = np.zeros((nperseg // 2 + 1, n_ch, n_ch), dtype=complex)
    n_seg = n_used = 0
    for a, b in spans:
        s, k = _span_cube_sum(x[:, a:b], win, nperseg, step, detrend)
        acc += s
        n_seg += k
        if k:
            n_used += b - a
    if n_seg == 0:
        raise ValueError(
            f"spectral_cube: no span is at least nperseg ({nperseg}) samples long")

    if scaling == "density":
        scale = 1.0 / (fs * np.sum(win ** 2))
    elif scaling == "spectrum":
        scale = 1.0 / np.sum(win) ** 2
    else:
        raise ValueError(f"scaling must be 'density' or 'spectrum', got {scaling!r}")
    G = acc * (scale / n_seg)
    # one-sided: double everything but DC and (for even nperseg) Nyquist
    if nperseg % 2:
        G[1:] *= 2.0
    else:
        G[1:-1] *= 2.0
    freqs = np.fft.rfftfreq(nperseg, 1.0 / fs)

    return SpectralCube(
        freqs, G, float(fs), window, nperseg, noverlap, detrend, scaling,
        n_seg, n_used, None if labels is None else list(labels),
    )
