"""Tests for FRF estimation, SDOF response and log decrement."""

import numpy as np
import pytest
from scipy import signal

import dspkit as dsp


def _sdof_filter(x, fs, fn, zeta):
    wn = 2 * np.pi * fn
    z = signal.cont2discrete(([1.0], [1.0, 2 * zeta * wn, wn ** 2]), 1 / fs,
                             method="bilinear")
    return signal.lfilter(np.asarray(z[0]).ravel(), np.asarray(z[1]).ravel(), x)


# ── SDOF response and response spectrum ──────────────────────────────────────

def test_sdof_matches_steady_state_theory():
    """Harmonic base motion has a closed-form response; the solver must hit it."""
    fs, n = 500.0, 20000
    t = np.arange(n) / fs
    T, zeta = 0.5, 0.05
    wn = 2 * np.pi / T
    for ratio in (0.3, 1.0, 2.5):
        w = ratio * wn
        u, _, _ = dsp.sdof_response(np.sin(w * t), fs, T, zeta)
        theory = 1.0 / np.sqrt((wn ** 2 - w ** 2) ** 2 + (2 * zeta * wn * w) ** 2)
        assert np.max(np.abs(u[-int(5 * fs):])) == pytest.approx(theory, rel=1e-3)


def test_pseudo_and_true_are_returned_separately():
    """They are defined differently and diverge with damping; both must be there."""
    fs, n = 200.0, 4000
    rng = np.random.default_rng(0)
    out = dsp.response_spectrum(rng.normal(size=n), fs,
                                periods=np.array([0.2, 0.5, 1.0]), zeta=0.20)
    r = out[0.20]
    assert set(("Sd", "Sv", "Sa", "PSv", "PSa")) <= set(r)
    # At 20% damping pseudo-velocity is not the true peak velocity.
    assert not np.allclose(r["PSv"], r["Sv"], rtol=0.05)


def test_short_periods_warn_rather_than_return_a_confident_curve():
    fs, n = 50.0, 2000
    rng = np.random.default_rng(0)
    with pytest.warns(UserWarning, match="shorter than"):
        dsp.response_spectrum(rng.normal(size=n), fs,
                              periods=np.array([0.05, 1.0]), zeta=0.05)


# ── log decrement ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("zeta_true", [0.01, 0.02, 0.05])
def test_log_decrement_recovers_damping_from_a_clean_decay(zeta_true):
    fs, n = 1000.0, 8000
    t = np.arange(n) / fs
    wn = 2 * np.pi * 12
    x = np.exp(-zeta_true * wn * t) * np.sin(wn * np.sqrt(1 - zeta_true ** 2) * t)
    r = dsp.log_decrement(x, fs)
    assert r["zeta"] == pytest.approx(zeta_true, rel=0.03)
    assert r["fn"] == pytest.approx(12.0, rel=0.01)
    assert r["r_squared"] > 0.99


def test_log_decrement_is_not_fooled_by_noise_peaks():
    """
    Noise inserts maxima between the real peaks. Because the log decrement is
    proportional to the mean peak spacing, unconstrained peak finding halved the
    damping while leaving an excellent straight-line fit -- a confident wrong
    answer. Peaks are now required to be most of a cycle apart.
    """
    fs, n = 1000.0, 8000
    t = np.arange(n) / fs
    wn = 2 * np.pi * 12
    rng = np.random.default_rng(0)
    x = np.exp(-0.02 * wn * t) * np.sin(wn * t) + 0.01 * rng.normal(size=n)
    r = dsp.log_decrement(x, fs)
    assert r["zeta"] == pytest.approx(0.02, rel=0.2)
    # The peak count should be near the true cycle count, not 10x it.
    assert r["n_peaks_used"] < 100


def test_log_decrement_needs_an_actual_decay():
    fs = 100.0
    with pytest.raises(ValueError):
        dsp.log_decrement(np.ones(500), fs)


# ── FRF ──────────────────────────────────────────────────────────────────────

def test_frf_recovers_a_known_transfer_function():
    fs, n = 1000.0, 200000
    rng = np.random.default_rng(0)
    fn, zeta = 30.0, 0.03
    x = rng.normal(size=n)
    y = _sdof_filter(x, fs, fn, zeta)
    y = y + 0.02 * np.std(y) * rng.normal(size=n)

    r = dsp.frf(x, y, fs, nperseg=4096)
    peak = r["freqs"][int(np.argmax(r["magnitude"]))]
    assert peak == pytest.approx(fn, abs=0.5)
    assert r["coherence"][int(np.argmax(r["magnitude"]))] > 0.9


def test_frf_rejects_an_unknown_estimator():
    with pytest.raises(ValueError, match="estimator"):
        dsp.frf(np.zeros(100), np.zeros(100), 100.0, estimator="H9")


def test_mimo_attributes_each_independent_input_to_its_own_mode():
    fs, n = 1000.0, 200000
    rng = np.random.default_rng(1)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    y = _sdof_filter(x1, fs, 30, 0.03) + 0.5 * _sdof_filter(x2, fs, 55, 0.04)
    y = y + 0.02 * np.std(y) * rng.normal(size=n)

    r = dsp.frf_mimo(np.vstack([x1, x2]), y, fs, nperseg=4096)
    f = r["freqs"]
    assert f[int(np.argmax(r["magnitude"][0]))] == pytest.approx(30, abs=1.5)
    assert f[int(np.argmax(r["magnitude"][1]))] == pytest.approx(55, abs=1.5)
    band = (f > 10) & (f < 80)
    assert r["multiple_coherence"][band].mean() > 0.9


def test_correlated_inputs_show_in_the_condition_number_not_the_coherence():
    """
    The failure this guards: at 0.95 input correlation both the multiple and the
    ordinary coherences stay above 0.99 while the individual FRFs are not
    separable. Only the conditioning of the input matrix reveals it.
    """
    fs, n = 1000.0, 200000
    rng = np.random.default_rng(1)
    conds, cohs = [], []
    for corr in (0.0, 0.95):
        x1 = rng.normal(size=n)
        x2 = corr * x1 + np.sqrt(1 - corr ** 2) * rng.normal(size=n)
        y = _sdof_filter(x1, fs, 30, 0.03) + 0.5 * _sdof_filter(x2, fs, 55, 0.04)
        y = y + 0.02 * np.std(y) * rng.normal(size=n)
        r = dsp.frf_mimo(np.vstack([x1, x2]), y, fs, nperseg=4096)
        band = (r["freqs"] > 10) & (r["freqs"] < 80)
        conds.append(float(np.median(r["input_condition"][band])))
        cohs.append(float(r["multiple_coherence"][band].mean()))

    assert conds[1] > 10 * conds[0]          # conditioning reacts
    assert abs(cohs[1] - cohs[0]) < 0.02     # coherence does not


def test_mimo_refuses_too_few_segments():
    fs, n = 1000.0, 8000
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="segment"):
        dsp.frf_mimo(rng.normal(size=(2, n)), rng.normal(size=n), fs, nperseg=4096)


# ── envelope spectrum ────────────────────────────────────────────────────────

def test_envelope_spectrum_finds_a_modulation_hidden_under_a_carrier():
    """
    The bearing-fault case: weak impacts at 7 Hz modulating a 300 Hz resonance.
    The impact rate is absent from the signal's own spectrum and present in the
    spectrum of its envelope.
    """
    fs, n = 5000.0, 50000
    t = np.arange(n) / fs
    f_mod, f_carrier = 7.0, 300.0
    rng = np.random.default_rng(0)
    sig = (1.0 + 0.8 * np.sign(np.sin(2 * np.pi * f_mod * t))) \
        * np.sin(2 * np.pi * f_carrier * t) + 0.05 * rng.normal(size=n)

    freqs, spec, env = dsp.envelope_spectrum(sig, fs, band=(200, 400), nperseg=8192)
    low = (freqs > 2) & (freqs < 20)
    assert freqs[low][int(np.argmax(spec[low]))] == pytest.approx(f_mod, abs=0.5)
    assert env.size == sig.size


def test_envelope_spectrum_rejects_a_band_outside_nyquist():
    with pytest.raises(ValueError, match="Nyquist"):
        dsp.envelope_spectrum(np.zeros(1000), 100.0, band=(10, 60))
