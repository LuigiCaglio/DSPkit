"""Tests for FRF estimation, SDOF response and log decrement."""

import numpy as np
import pytest
from scipy import signal

import dspkit as dsp
from dspkit._testing import generate_shear4


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


# ── random decrement ─────────────────────────────────────────────────────────

def _ambient_sdof(fs, n, fn, zeta, seed=0):
    """Ambient response of an SDOF to white excitation."""
    rng = np.random.default_rng(seed)
    wn = 2 * np.pi * fn
    q = signal.cont2discrete(([1.0], [1.0, 2 * zeta * wn, wn ** 2]), 1 / fs,
                             method="bilinear")
    return signal.lfilter(np.asarray(q[0]).ravel(), np.asarray(q[1]).ravel(),
                          rng.normal(size=n))


def test_random_decrement_recovers_damping_from_ambient_data():
    """
    The point of the method: log decrement needs a decay, ambient data is not
    one, and RDT manufactures a decay signature from it.
    """
    fs, n = 200.0, 400000
    x = _ambient_sdof(fs, n, 3.0, 0.02)
    rd = dsp.random_decrement(x, fs, segment_length=600)
    assert rd["n_segments"] > 500
    ld = dsp.log_decrement(rd["signature"], fs)
    assert ld["zeta"] == pytest.approx(0.02, rel=0.15)
    assert ld["fn"] == pytest.approx(3.0, rel=0.02)


def test_autocorrelation_route_agrees_with_random_decrement():
    """Both are proportional to the free decay, so they should broadly agree."""
    fs, n = 200.0, 400000
    x = _ambient_sdof(fs, n, 3.0, 0.02)
    rd = dsp.log_decrement(dsp.random_decrement(x, fs, segment_length=600)["signature"], fs)
    lags, acf = dsp.autocorrelation(x, fs=fs, normalize=True)
    ac = dsp.log_decrement(acf[lags >= 0], fs)
    assert ac["zeta"] == pytest.approx(rd["zeta"], rel=0.25)


def test_random_decrement_needs_enough_triggers():
    fs, n = 200.0, 4000
    x = _ambient_sdof(fs, n, 3.0, 0.02)
    with pytest.raises(ValueError, match="trigger"):
        dsp.random_decrement(x, fs, trigger_level=100 * np.std(x))


def test_cross_random_decrement_uses_the_second_channel():
    fs, n = 200.0, 100000
    x = _ambient_sdof(fs, n, 3.0, 0.02, seed=0)
    y = _ambient_sdof(fs, n, 3.0, 0.02, seed=1)
    auto = dsp.random_decrement(x, fs, segment_length=400)
    cross = dsp.random_decrement(x, fs, segment_length=400, y=y)
    # Triggered identically, but averaging a different channel.
    assert cross["n_segments"] == auto["n_segments"]
    assert not np.allclose(cross["signature"], auto["signature"])


# ── error spectrum ───────────────────────────────────────────────────────────
#
# `generate_shear4` is the only fixture in the suite where the *true* answer to
# "how much of this signal is unexplainable" is known: a second, deliberately
# unmeasured force is injected, so the residual has a floor no amount of
# estimation can remove. Its natural frequencies are 0.375, 0.984, 1.516 and
# 1.985 Hz, which the first test pins so the numbers below cannot drift.

def _reconstruct(H, welch_freqs, preds, fs, trim=0.02):
    """Apply estimated filters to a record; both steps here are load-bearing."""
    n = preds.shape[1]
    grid = np.fft.rfftfreq(n, 1 / fs)
    Dhat = np.zeros(grid.size, dtype=complex)
    for i in range(preds.shape[0]):
        # Real and imaginary parts separately -- interpolating magnitude and
        # phase corrupts the filter wherever the phase wraps.
        re = np.interp(grid, welch_freqs, H[i].real)
        im = np.interp(grid, welch_freqs, H[i].imag)
        Dhat += (re + 1j * im) * np.fft.rfft(preds[i])
    dhat = np.fft.irfft(Dhat, n=n)
    # Multiplying in frequency is a *circular* convolution, so the ends carry
    # wrap-around transients. Keeping them inflated var(e) by 2x in testing.
    cut = int(trim * n)
    return dhat, slice(cut, n - cut)


@pytest.fixture(scope="module")
def shear4():
    return generate_shear4()


def test_shear_chain_fixture_has_the_intended_modes(shear4):
    """If the fixture drifts, every number below drifts with it silently."""
    assert shear4["fn"] == pytest.approx([0.375, 0.984, 1.516, 1.985], abs=1e-3)


def test_error_spectrum_reduces_to_ordinary_coherence_for_one_predictor(shear4):
    """The q=1 case must collapse to |Sad|^2/(Saa Sdd), not merely resemble it."""
    fs = shear4["fs"]
    d, a = shear4["d"], shear4["A"]
    r = dsp.error_spectrum(d, a, fs, nperseg=8192)
    _, coh = dsp.coherence(a, d, fs, nperseg=8192)
    assert np.max(np.abs(r["coherence"] - coh)) < 1e-8


def test_error_spectrum_splits_the_target_power_exactly(shear4):
    """Coherent plus error must return the target spectrum, bin by bin."""
    r = dsp.error_spectrum(shear4["d"], shear4["A"], shear4["fs"], nperseg=8192)
    total = r["coherent_power"] + r["error_spectrum"]
    assert np.allclose(total, r["target_psd"], rtol=1e-12, atol=0.0)
    assert np.all(r["error_spectrum"] >= 0.0)


def test_error_spectrum_predicts_a_held_out_residual(shear4):
    """The claim worth testing: this is the residual you actually get.

    Fit on one half, filter the other, compare. In-sample would be optimistic
    by construction -- the filter has q complex coefficients per frequency
    line -- so the comparison is made out of sample.
    """
    fs = shear4["fs"]
    d = shear4["d"]
    half = d.size // 2
    tr, te = slice(0, half), slice(half, d.size)

    for preds in ([shear4["A"]], [shear4["B"]],
                  [shear4["A"], shear4["B"], shear4["C"]]):
        P = np.vstack(preds)
        r = dsp.error_spectrum(d[tr], P[:, tr], fs, nperseg=8192)
        dhat, keep = _reconstruct(r["H"], r["freqs"], P[:, te], fs)
        measured = np.var(d[te][keep] - dhat[keep])
        assert r["unexplained_variance"] == pytest.approx(measured, rel=0.15)


def test_low_coherence_band_can_be_irrelevant(shear4):
    """Why the error spectrum exists: the coherence plot alone cries wolf."""
    fs = shear4["fs"]
    r = dsp.error_spectrum(shear4["d"], shear4["A"], fs, nperseg=8192)
    f = r["freqs"]
    band = (f > 0.5) & (f < 3.0)
    # Coherence says the model is worthless over most of that band ...
    assert r["coherence"][band].mean() < 0.7
    # ... while under 0.5% of the target's variance lives there.
    assert r["unexplained_fraction"] < 0.005


def test_more_predictors_lower_the_error_spectrum(shear4):
    """Each sensor fails at different frequencies, so together they do better."""
    fs = shear4["fs"]
    d = shear4["d"]
    one = dsp.error_spectrum(d, shear4["A"], fs, nperseg=8192)
    three = dsp.error_spectrum(
        d, np.vstack([shear4["A"], shear4["B"], shear4["C"]]), fs, nperseg=8192)
    assert three["unexplained_variance"] < 0.5 * one["unexplained_variance"]
    # Multiple coherence is bounded below by the best ordinary one, everywhere.
    assert np.all(three["coherence"] >= three["ordinary_coherence"].max(axis=0)
                  - 1e-9)


def test_short_segments_manufacture_an_error_spectrum():
    """The trap: on a noise-free single-input system the truth is exactly zero.

    With one measured force and no noise the map from acceleration to
    displacement is a single exact transfer function, so anything the
    estimator reports as unexplained is resolution bias. It scales as
    (Be/Br)^2, which is what pins the cause.
    """
    s = generate_shear4(noise_frac=0.0, unmeasured_frac=0.0)
    fs = s["fs"]
    fracs = [dsp.error_spectrum(s["d"], s["A"], fs, nperseg=n)["unexplained_fraction"]
             for n in (8192, 4096, 2048)]
    # Nowhere near zero, and growing fast, despite a true residual of zero.
    assert fracs[0] < 1e-3
    for lo, hi in zip(fracs, fracs[1:]):
        assert 3.0 < hi / lo < 5.0        # halving nperseg quadruples it


def test_error_spectrum_warns_when_the_bias_floor_is_large(shear4):
    """Optimistic by roughly q/n_d, and the user has to be told which way."""
    short = shear4["d"][:60000]
    preds = np.vstack([shear4["A"][:60000], shear4["B"][:60000],
                       shear4["C"][:60000]])
    with pytest.warns(UserWarning, match="bias floor"):
        r = dsp.error_spectrum(short, preds, shear4["fs"], nperseg=16384)
    assert r["bias_floor"] > 0.1


def test_error_spectrum_rejects_mismatched_lengths(shear4):
    with pytest.raises(ValueError, match="same length"):
        dsp.error_spectrum(shear4["d"], shear4["A"][:-10], shear4["fs"])
