"""Tests for dspkit.cube: every derived quantity against the function it replaces."""

import warnings

import numpy as np
import pytest

from dspkit.cube import SpectralCube, spectral_cube
from dspkit.fdd import fdd_svd
from dspkit.multisensor import (
    coherence_matrix,
    multiple_coherence,
    partial_coherence,
    psd_matrix,
)
from dspkit.spectral import coherence, csd, psd

FS = 100.0


def _data(n_ch=4, N=20_000, seed=0):
    """Correlated channels: two shared narrowband sources plus noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(N) / FS
    s1 = np.sin(2 * np.pi * 3.2 * t + rng.uniform(0, 6)) * (1 + 0.1 * rng.normal(size=N))
    s2 = np.convolve(rng.normal(size=N), np.ones(5) / 5, "same")
    mix = rng.normal(size=(n_ch, 2))
    return mix @ np.vstack([s1, s2]) + 0.3 * rng.normal(size=(n_ch, N))


def _close(a, b, rtol=1e-9):
    scale = np.max(np.abs(b))
    np.testing.assert_allclose(a, b, rtol=0, atol=rtol * scale)


@pytest.mark.parametrize("nperseg,noverlap", [(512, None), (1000, 300), (777, 0)])
@pytest.mark.parametrize("detrend", ["constant", "linear", False])
def test_psd_matrix_matches(nperseg, noverlap, detrend):
    x = _data()
    f0, G0 = psd_matrix(x, FS, nperseg=nperseg, noverlap=noverlap, detrend=detrend)
    c = spectral_cube(x, FS, nperseg=nperseg, noverlap=noverlap, detrend=detrend)
    np.testing.assert_allclose(c.freqs, f0)
    _close(c.psd_matrix(), G0)


@pytest.mark.parametrize("window", ["hann", "hamming", "boxcar"])
@pytest.mark.parametrize("scaling", ["density", "spectrum"])
def test_psd_and_csd_match(window, scaling):
    x = _data(3)
    c = spectral_cube(x, FS, window=window, nperseg=1024, scaling=scaling)
    for i in range(3):
        _, p = psd(x[i], FS, window=window, nperseg=1024, scaling=scaling)
        _close(c.psd(i), p)
    if scaling == "density":
        _, g = csd(x[0], x[2], FS, window=window, nperseg=1024)
        _close(c.csd(0, 2), g)


def test_coherence_matches():
    x = _data()
    c = spectral_cube(x, FS, nperseg=1024)
    _, C0 = coherence_matrix(x, FS, nperseg=1024)
    _close(c.coherence_matrix(), C0, 1e-8)
    _, c01 = coherence(x[0], x[1], FS, nperseg=1024)
    _close(c.coherence(0, 1), c01, 1e-8)


def test_multiple_and_partial_coherence_match():
    x = _data(4, 40_000)
    c = spectral_cube(x, FS, nperseg=512)
    _, g0 = multiple_coherence(x, FS, nperseg=512)
    _close(c.multiple_coherence(), g0, 1e-7)
    _, P0 = partial_coherence(x, FS, nperseg=512)
    _close(c.partial_coherence(), P0, 1e-7)


def test_svd_matches_fdd():
    x = _data()
    c = spectral_cube(x, FS, nperseg=1024)
    _, S0, U0 = fdd_svd(x, FS, nperseg=1024)
    S, U = c.svd()
    _close(S, S0)
    # singular vectors are defined up to a phase: compare the first one by MAC
    k = np.argmax(S0[:, 0])
    u, u0 = U[k, :, 0], U0[k, :, 0]
    mac = np.abs(np.vdot(u, u0)) ** 2 / (np.vdot(u, u).real * np.vdot(u0, u0).real)
    assert mac > 1 - 1e-10
    assert c.svd() is c.svd()  # computed once


def test_spans_equal_concatenated_segments():
    """Spans never share a Welch segment; average weights by segment count."""
    x = _data(3, 30_000)
    spans = [(0, 9000), (12_000, 20_000), (22_500, 30_000)]
    c = spectral_cube(x, FS, nperseg=1000, noverlap=500, spans=spans)
    parts = [spectral_cube(x[:, a:b], FS, nperseg=1000, noverlap=500) for a, b in spans]
    avg = SpectralCube.average(parts)
    _close(c.G, avg.G)
    assert c.n_segments == sum(p.n_segments for p in parts) == avg.n_segments
    assert c.n_samples == 9000 + 8000 + 7500
    # manual check of the weighting: segment-weighted mean of the parts
    w = np.array([p.n_segments for p in parts], float)
    _close(c.G, sum(wi * p.G for wi, p in zip(w, parts)) / w.sum())


def test_short_span_contributes_nothing():
    x = _data(2, 10_000)
    a = spectral_cube(x, FS, nperseg=1000, spans=[(0, 8000), (9500, 10_000)])
    b = spectral_cube(x[:, :8000], FS, nperseg=1000)
    _close(a.G, b.G)
    assert a.n_samples == 8000


def test_average_refuses_mismatch():
    x = _data(2)
    a = spectral_cube(x, FS, nperseg=512)
    b = spectral_cube(x, FS, nperseg=1024)
    with pytest.raises(ValueError, match="differ"):
        SpectralCube.average([a, b])
    c = spectral_cube(x, FS, nperseg=512, labels=["a", "b"])
    d = spectral_cube(x, FS, nperseg=512, labels=["a", "c"])
    with pytest.raises(ValueError, match="labels"):
        SpectralCube.average([c, d])


def test_labels_select_and_band():
    x = _data(4)
    c = spectral_cube(x, FS, nperseg=512, labels=["A", "B", "C", "D"])
    _close(c.psd("C"), c.psd(2))
    s = c.select(["D", "B"])
    assert s.labels == ["D", "B"]
    _close(s.csd(0, 1), c.csd(3, 1))
    b = c.band(2.0, 5.0)
    assert b.freqs.min() >= 2.0 and b.freqs.max() <= 5.0
    _close(b.psd(0), c.psd(0)[(c.freqs >= 2) & (c.freqs <= 5)])
    with pytest.raises(KeyError):
        c.psd("Z")


def test_save_load_roundtrip(tmp_path):
    x = _data(3)
    for labels, det in [(["x", "y", "z"], "constant"), (None, False)]:
        c = spectral_cube(x, FS, nperseg=512, labels=labels, detrend=det)
        c.save(tmp_path / "c.npz")
        d = SpectralCube.load(tmp_path / "c.npz")
        np.testing.assert_array_equal(d.G, c.G)
        assert (d.labels, d.detrend, d.nperseg, d.noverlap, d.n_segments, d.fs) == \
               (c.labels, c.detrend, c.nperseg, c.noverlap, c.n_segments, c.fs)


def test_segment_checks_follow_the_originals():
    x = _data(4, 3000)
    c = spectral_cube(x, FS, nperseg=1024)          # 4 segments
    with pytest.warns(UserWarning, match="Welch segments"):
        c.coherence_matrix()
    with pytest.raises(ValueError, match="at least 5"):
        c.multiple_coherence()
    one = spectral_cube(x, FS, nperseg=3000)        # 1 segment
    with pytest.raises(ValueError):
        one.coherence(0, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        spectral_cube(_data(4, 40_000), FS, nperseg=512).coherence_matrix()


def test_bad_inputs():
    x = _data(2, 1000)
    with pytest.raises(ValueError, match="noverlap"):
        spectral_cube(x, FS, nperseg=100, noverlap=100)
    with pytest.raises(ValueError, match="labels"):
        spectral_cube(x, FS, labels=["one"])
    with pytest.raises(ValueError, match="no data"):
        spectral_cube(x, FS, spans=[(500, 500)])
