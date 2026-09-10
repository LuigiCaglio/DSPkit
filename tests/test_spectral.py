"""
Tests for dspkit.spectral.

Strategy: use analytically known results (pure sines, white noise properties)
so tests are deterministic and self-documenting.
"""

import re
import warnings

import numpy as np
import pytest

import dspkit as dsp
from scipy import signal as _signal

from dspkit.spectral import autocorrelation, coherence, cross_correlation, csd, fft_spectrum, psd
from dspkit._testing import generate_2dof, generate_sine, natural_frequencies_2dof


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FS = 2000.0          # Hz
DURATION = 10.0      # s
N = int(FS * DURATION)


def _sine(freq: float, amplitude: float = 1.0) -> np.ndarray:
    t = np.arange(N) / FS
    return amplitude * np.sin(2.0 * np.pi * freq * t)


# ---------------------------------------------------------------------------
# fft_spectrum
# ---------------------------------------------------------------------------

class TestFftSpectrum:
    def test_peak_at_correct_frequency(self):
        """The dominant bin must be at the sine's frequency."""
        freq = 100.0
        x = _sine(freq)
        freqs, amp = fft_spectrum(x, FS, window="hann")
        peak_freq = freqs[np.argmax(amp)]
        assert abs(peak_freq - freq) < 1.0  # within 1 Hz

    def test_amplitude_recovery(self):
        """Peak amplitude at the sine frequency should recover the true amplitude."""
        freq = 50.0
        A = 3.7
        x = _sine(freq, amplitude=A)
        freqs, amp = fft_spectrum(x, FS, window="hann")
        recovered = amp[np.argmax(amp)]
        assert abs(recovered - A) / A < 0.01  # within 1 %

    def test_rms_scaling(self):
        """RMS-scaled amplitude should be peak / sqrt(2)."""
        freq = 80.0
        A = 2.0
        x = _sine(freq, amplitude=A)
        _, amp_peak = fft_spectrum(x, FS, window="hann", scaling="amplitude")
        _, amp_rms = fft_spectrum(x, FS, window="hann", scaling="rms")
        ratio = amp_peak[np.argmax(amp_peak)] / amp_rms[np.argmax(amp_rms)]
        assert abs(ratio - np.sqrt(2.0)) < 0.02

    def test_dc_not_doubled(self):
        """DC bin should not be doubled (only interior bins are)."""
        x = np.ones(N)  # pure DC signal, amplitude 1
        _, amp = fft_spectrum(x, FS, window=None)  # rectangular for clean DC
        assert abs(amp[0] - 1.0) < 1e-10

    def test_output_length(self):
        x = _sine(10.0)
        freqs, amp = fft_spectrum(x, FS)
        assert len(freqs) == N // 2 + 1
        assert len(amp) == len(freqs)

    def test_freqs_positive(self):
        x = _sine(10.0)
        freqs, _ = fft_spectrum(x, FS)
        assert freqs[0] == 0.0
        assert np.all(freqs >= 0.0)


# ---------------------------------------------------------------------------
# psd
# ---------------------------------------------------------------------------

class TestPsd:
    def test_output_shapes_match(self):
        x = _sine(50.0)
        freqs, Pxx = psd(x, FS)
        assert freqs.shape == Pxx.shape
        assert Pxx.dtype == float

    def test_psd_nonnegative(self):
        x = _sine(50.0) + np.random.default_rng(0).normal(0, 0.1, N)
        _, Pxx = psd(x, FS)
        assert np.all(Pxx >= 0.0)

    def test_sine_peak_frequency(self):
        """PSD of a sine should peak at the sine's frequency."""
        freq = 120.0
        x = _sine(freq)
        freqs, Pxx = psd(x, FS, nperseg=1024)
        assert abs(freqs[np.argmax(Pxx)] - freq) < 2.0

    def test_parseval_density_scaling(self):
        """Parseval's theorem: ∑(Pxx * df) ≈ mean(x²) for density scaling."""
        x = _sine(100.0, amplitude=1.0)
        freqs, Pxx = psd(x, FS, scaling="density")
        df = freqs[1] - freqs[0]
        # Signal power = A² / 2 = 0.5
        total_power = (Pxx * df).sum()
        assert abs(total_power - 0.5) < 0.05

    def test_default_nperseg(self):
        """Default nperseg should be min(N, 1024)."""
        x = _sine(50.0)
        freqs, Pxx = psd(x, FS)
        # With N=20000 and nperseg=1024, freq resolution = FS/1024 ~ 1.95 Hz
        df = freqs[1] - freqs[0]
        assert abs(df - FS / 1024) < 0.01


# ---------------------------------------------------------------------------
# csd
# ---------------------------------------------------------------------------

class TestCsd:
    def test_output_is_complex(self):
        x = _sine(50.0)
        y = _sine(50.0, amplitude=2.0)
        _, Pxy = csd(x, y, FS)
        assert np.iscomplexobj(Pxy)

    def test_csd_of_identical_signals_equals_psd(self):
        """CSD(x, x) magnitude should equal PSD(x)."""
        x = _sine(50.0) + np.random.default_rng(1).normal(0, 0.5, N)
        freqs_psd, Pxx = psd(x, FS, nperseg=512)
        freqs_csd, Pxy = csd(x, x, FS, nperseg=512)
        np.testing.assert_allclose(freqs_psd, freqs_csd)
        np.testing.assert_allclose(np.abs(Pxy), Pxx, rtol=1e-10)


# ---------------------------------------------------------------------------
# coherence
# ---------------------------------------------------------------------------

class TestCoherence:
    def test_coherence_range(self):
        """Coherence must be in [0, 1]."""
        x = _sine(50.0) + np.random.default_rng(2).normal(0, 1.0, N)
        y = _sine(50.0) + np.random.default_rng(3).normal(0, 1.0, N)
        _, Cxy = coherence(x, y, FS)
        assert np.all(Cxy >= -1e-12)
        assert np.all(Cxy <= 1.0 + 1e-12)

    def test_self_coherence_is_one(self):
        """A signal is perfectly coherent with itself."""
        x = _sine(50.0) + np.random.default_rng(4).normal(0, 0.5, N)
        _, Cxy = coherence(x, x, FS, nperseg=512)
        # All bins should be ~1 (small numerical errors aside)
        assert np.all(Cxy > 0.999)

    def test_unrelated_noise_low_coherence(self):
        """Two independent noise signals should have low mean coherence."""
        rng = np.random.default_rng(5)
        x = rng.normal(0, 1, N)
        y = rng.normal(0, 1, N)
        _, Cxy = coherence(x, y, FS, nperseg=256)
        assert np.mean(Cxy) < 0.2


# ---------------------------------------------------------------------------
# autocorrelation
# ---------------------------------------------------------------------------

class TestAutocorrelation:
    def test_zero_lag_is_one_normalized(self):
        x = _sine(50.0) + np.random.default_rng(6).normal(0, 0.5, N)
        _, acf = autocorrelation(x)
        assert abs(acf[0] - 1.0) < 1e-10

    def test_lag_axis_in_seconds(self):
        x = _sine(50.0)
        lags, _ = autocorrelation(x, fs=FS)
        assert lags[0] == 0.0
        assert abs(lags[1] - 1.0 / FS) < 1e-12

    def test_lag_axis_in_samples_when_no_fs(self):
        x = _sine(50.0)
        lags, _ = autocorrelation(x)
        assert lags[0] == 0.0
        assert lags[1] == 1.0

    def test_max_lag_truncation(self):
        x = _sine(50.0)
        max_lag = 1.0  # second
        lags, acf = autocorrelation(x, fs=FS, max_lag=max_lag)
        assert lags[-1] <= max_lag + 1.0 / FS

    def test_sine_acf_is_cosine(self):
        """ACF of a sine wave is a cosine at the same frequency."""
        freq = 50.0
        x = _sine(freq)
        lags, acf = autocorrelation(x, fs=FS, max_lag=0.5)
        expected = np.cos(2.0 * np.pi * freq * lags)
        # Ignore the first and last few samples (edge effects from the biased estimator)
        np.testing.assert_allclose(acf[10:-10], expected[10:-10], atol=0.06)


# ---------------------------------------------------------------------------
# cross_correlation
# ---------------------------------------------------------------------------

class TestCrossCorrelation:
    def test_symmetric_lag_axis(self):
        """Lag axis should be symmetric around zero."""
        x = _sine(50.0)
        lags, _ = cross_correlation(x, x)
        assert lags[0] == -(N - 1)
        assert lags[-1] == N - 1
        assert lags[len(lags) // 2] == 0.0

    def test_lag_axis_in_seconds(self):
        x = _sine(50.0)
        lags, _ = cross_correlation(x, x, fs=FS)
        assert abs(lags[len(lags) // 2]) < 1e-12   # zero lag at centre
        assert abs(lags[1] - lags[0] - 1.0 / FS) < 1e-12

    def test_max_lag_truncation(self):
        x = _sine(50.0)
        lags, ccf = cross_correlation(x, x, fs=FS, max_lag=0.5)
        assert lags[0] >= -0.5 - 1.0 / FS
        assert lags[-1] <= 0.5 + 1.0 / FS
        assert len(lags) == len(ccf)

    def test_normalized_range(self):
        """Normalised CCF should have |values| <= 1."""
        rng = np.random.default_rng(10)
        x = _sine(30.0) + rng.normal(0, 0.3, N)
        y = _sine(30.0) + rng.normal(0, 0.3, N)
        _, ccf = cross_correlation(x, y, normalize=True)
        assert np.all(np.abs(ccf) <= 1.0 + 1e-10)

    def test_self_correlation_peak_at_zero(self):
        """CCF(x, x) should peak at lag=0 (same signal, no delay)."""
        x = _sine(40.0)
        lags, ccf = cross_correlation(x, x)
        zero_idx = len(lags) // 2
        assert np.argmax(ccf) == zero_idx

    def test_delayed_signal_peak_at_delay(self):
        """If y is x delayed by k samples, peak should be at lag +k."""
        delay = 50   # samples
        x = _sine(20.0)
        y = np.roll(x, delay)
        y[:delay] = 0.0   # clear wrap-around
        lags, ccf = cross_correlation(x, y)
        assert lags[np.argmax(ccf)] == delay

    def test_ccf_equals_acf_for_identical_signals(self):
        """CCF(x, x) should equal ACF(x) at positive lags."""
        x = _sine(30.0)
        lags_acf, acf = autocorrelation(x, normalize=True)
        lags_ccf, ccf = cross_correlation(x, x, normalize=True)
        zero = len(lags_ccf) // 2
        # Positive half of CCF should match ACF
        np.testing.assert_allclose(
            ccf[zero:zero + len(acf)], acf, atol=1e-10
        )


# ---------------------------------------------------------------------------
# _testing helpers
# ---------------------------------------------------------------------------

class TestGenerators:
    def test_2dof_output_shapes(self):
        t, x1, x2 = generate_2dof(duration=5.0, fs=500.0, seed=0)
        N = 5 * 500
        assert t.shape == (N,) == x1.shape == x2.shape

    def test_2dof_natural_frequencies(self):
        """Simulated 2DOF PSD should show peaks near the theoretical fn."""
        fn1, fn2 = natural_frequencies_2dof()
        t, x1, _ = generate_2dof(duration=120.0, fs=1000.0, seed=42)
        freqs, Pxx = psd(x1, 1000.0, nperseg=4096)
        # Find two largest peaks
        from scipy.signal import find_peaks
        peaks_idx, _ = find_peaks(Pxx, height=np.max(Pxx) * 0.01)
        peak_freqs = freqs[peaks_idx]
        # At least one peak should be within 1 Hz of each natural frequency
        assert any(abs(pf - fn1) < 1.0 for pf in peak_freqs), f"fn1={fn1:.2f} not found in {peak_freqs}"
        assert any(abs(pf - fn2) < 1.0 for pf in peak_freqs), f"fn2={fn2:.2f} not found in {peak_freqs}"

    def test_generate_sine_amplitude(self):
        _, x = generate_sine(freqs=100.0, amplitudes=2.5, duration=5.0, fs=FS)
        assert abs(np.max(np.abs(x)) - 2.5) < 1e-10

    def test_natural_frequencies_default(self):
        fn1, fn2 = natural_frequencies_2dof()
        assert 8.0 < fn1 < 10.0
        assert 19.0 < fn2 < 23.0


# ---------------------------------------------------------------------------
# coherence: the single-segment trap
# ---------------------------------------------------------------------------

class TestCoherenceSegmentGuard:
    def test_single_segment_would_be_exactly_one_everywhere(self):
        """
        The behaviour being prevented, pinned so it stays prevented.

        With one Welch segment |Gxy|² = Gxx·Gyy identically, so scipy returns
        1.0 at every frequency for two unrelated noise records. Measured on
        the 2-DOF example: 0.0675 mean at nperseg=1024, 1.0000 at nperseg=N.
        """
        rng = np.random.default_rng(60)
        x = rng.normal(0, 1, N)
        y = rng.normal(0, 1, N)

        _, raw = _signal.coherence(x, y, fs=FS, nperseg=N)
        np.testing.assert_allclose(raw, 1.0, atol=1e-9)

        with pytest.raises(ValueError, match="identically 1.0"):
            coherence(x, y, FS, nperseg=N)

    def test_error_names_the_numbers_and_a_fix(self):
        """Named numbers, a candidate fix, and what that fix costs.

        This used to assert the literal string "Fix: shorten nperseg", which is
        the defect rather than the contract: shortening buys averages and spends
        resolution, and a segment too short to resolve the sharpest peak
        fabricates residual power. The message now has to offer the candidate
        *and* name the cost, so neither direction is presented as free.
        """
        rng = np.random.default_rng(61)
        x = rng.normal(0, 1, N)
        with pytest.raises(ValueError) as exc:
            coherence(x, x, FS, nperseg=N)
        message = str(exc.value)
        assert f"N={N}" in message
        assert f"nperseg={N}" in message
        assert re.search(r"nperseg=\d+ (?:would give|gives)", message)
        assert "resolution" in message

    def test_suggested_nperseg_actually_works(self):
        """The fix quoted in the error must survive being followed."""
        rng = np.random.default_rng(62)
        x = rng.normal(0, 1, N)
        y = rng.normal(0, 1, N)
        with pytest.raises(ValueError) as exc:
            coherence(x, y, FS, nperseg=N)
        suggested = int(re.search(r"nperseg=(\d+) (?:would give|gives)",
                                  str(exc.value)).group(1))
        _, Cxy = coherence(x, y, FS, nperseg=suggested)   # must not raise
        assert np.mean(Cxy) < 0.5

    def test_few_segments_warns_and_names_the_bias_floor(self):
        rng = np.random.default_rng(63)
        x = rng.normal(0, 1, N)
        y = rng.normal(0, 1, N)
        with pytest.warns(UserWarning, match="biased upward"):
            _, Cxy = coherence(x, y, FS, nperseg=N // 3)
        # The warning is earned: unrelated noise sits well above zero here.
        assert np.mean(Cxy) > 0.15

    def test_min_segments_can_be_lowered_to_silence_the_warning(self):
        rng = np.random.default_rng(64)
        x = rng.normal(0, 1, N)
        y = rng.normal(0, 1, N)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            coherence(x, y, FS, nperseg=N // 3, min_segments=2)

    def test_default_nperseg_is_quiet(self):
        rng = np.random.default_rng(65)
        x = rng.normal(0, 1, N)
        y = rng.normal(0, 1, N)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            coherence(x, y, FS)


class TestCrossCorrelationLagSign:
    def test_cross_correlation_lag_sign(self):
        """
        Pins the direction of the lag axis, which the docstring had backwards
        until 2026-09-02.

        CCF[k] pairs x[n] with y[n+k]. If y is a delayed copy of x — that is,
        x leads y — the peak sits at *positive* k.
        """
        rng = np.random.default_rng(66)
        x = rng.normal(0, 1, 4000)
        delay = 20

        y_delayed = np.roll(x, delay)        # y[n] = x[n - 20]; x leads y
        lags, ccf = cross_correlation(x, y_delayed)
        assert lags[int(np.argmax(ccf))] == delay

        y_advanced = np.roll(x, -delay)      # y leads x
        lags, ccf = cross_correlation(x, y_advanced)
        assert lags[int(np.argmax(ccf))] == -delay


# ── Blackman-Tukey and lag windows ───────────────────────────────────────────

def _narrowband(fs=200.0, n=40000, fn=10.0, zeta=0.03, seed=0):
    from scipy import signal as _sig
    rng = np.random.default_rng(seed)
    wn = 2 * np.pi * fn
    q = _sig.cont2discrete(([1.0], [1.0, 2 * zeta * wn, wn ** 2]), 1 / fs,
                           method="bilinear")
    return _sig.lfilter(np.asarray(q[0]).ravel(), np.asarray(q[1]).ravel(),
                        rng.normal(size=n))


def test_blackman_tukey_finds_the_peak():
    fs = 200.0
    x = _narrowband(fs)
    for w in ("bartlett", "parzen", "exponential"):
        f, p, _ = dsp.blackman_tukey_psd(x, fs, lag_window_name=w, max_lag=2000)
        band = (f > 1) & (f < 40)
        assert f[band][int(np.argmax(p[band]))] == pytest.approx(10.0, abs=0.3)


def test_only_the_safe_lag_windows_keep_the_spectrum_non_negative():
    """
    The whole reason to care which window is used. Negative power is not a
    possible value, and the windows whose own transform has negative sidelobes
    produce it.
    """
    fs = 200.0
    x = _narrowband(fs)
    for w in dsp.NONNEGATIVE_LAG_WINDOWS:
        _, _, neg = dsp.blackman_tukey_psd(x, fs, lag_window_name=w, max_lag=2000)
        assert neg == 0.0, f"{w} should never produce negative power"

    # Rectangular truncation is the clearest offender.
    _, _, neg_none = dsp.blackman_tukey_psd(x, fs, lag_window_name="none",
                                            max_lag=2000)
    assert neg_none > 0.1


def test_lag_window_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="Unknown lag window"):
        dsp.lag_window("blackman-harris", 100)


def test_lag_windows_start_at_one_and_do_not_grow():
    for w in dsp.LAG_WINDOWS:
        win = dsp.lag_window(w, 128)
        assert win[0] == pytest.approx(1.0)
        assert np.all(np.diff(win) <= 1e-12), f"{w} should be non-increasing"


# ── AR spectral estimation ───────────────────────────────────────────────────

def _ar2(n=20000, a1=1.6, a2=-0.9, seed=0):
    """A process whose true coefficients are known exactly."""
    rng = np.random.default_rng(seed)
    e = rng.normal(size=n)
    x = np.zeros(n)
    for i in range(2, n):
        x[i] = a1 * x[i - 1] + a2 * x[i - 2] + e[i]
    return x


def test_both_ar_methods_recover_known_coefficients():
    """The check that catches an off-by-one in the recursion, which nothing else does.

    A wrong Levinson update still produces a plausible-looking spectrum, so the
    only honest test is against a process whose coefficients are known. Both
    estimators must land on a = [1, -1.6, 0.9] and an innovation variance of 1.
    """
    x = _ar2()
    for method in dsp.AR_METHODS:
        a, var = (dsp.spectral._burg if method == "burg"
                  else dsp.spectral._yule_walker)(x, 2)
        assert a[0] == 1.0
        assert a[1] == pytest.approx(-1.6, abs=0.01), method
        assert a[2] == pytest.approx(0.9, abs=0.01), method
        assert var == pytest.approx(1.0, rel=0.05), method


def test_ar_psd_peaks_at_the_true_resonance():
    x = _ar2()
    f, p, info = dsp.ar_psd(x, 1.0, order=2)          # fs = 1 sample/s
    true = np.angle(np.roots([1.0, -1.6, 0.9])[0]) / (2 * np.pi)
    assert f[np.argmax(p)] == pytest.approx(true, abs=0.002)
    assert info["reflection_stable"] is True


def test_ar_psd_resolves_what_welch_cannot_on_a_short_record():
    """The reason the function exists, as a measurement rather than a claim."""
    fs, n = 100.0, 200                                 # two seconds
    t = np.arange(n) / fs
    rng = np.random.default_rng(3)
    x = (np.sin(2 * np.pi * 10.0 * t) + np.sin(2 * np.pi * 10.8 * t)
         + 0.1 * rng.normal(size=n))

    def peaks_between(f, p, lo=8.0, hi=13.0):
        idx, _ = _signal.find_peaks(p, prominence=p.max() * 0.02)
        return [float(f[i]) for i in idx if lo < f[i] < hi]

    # Welch merges them at every segment length the record allows.
    for nperseg in (200, 128, 64):
        fw, pw = dsp.psd(x, fs, nperseg=nperseg)
        assert len(peaks_between(fw, pw)) == 1, f"nperseg={nperseg}"

    fa, pa, _ = dsp.ar_psd(x, fs, order=30, n_freqs=4096)
    found = sorted(peaks_between(fa, pa))
    assert len(found) == 2, found
    assert found[0] == pytest.approx(10.0, abs=0.1)
    assert found[1] == pytest.approx(10.8, abs=0.1)


def test_ar_psd_conserves_power_against_welch():
    """Different estimators of the same thing must integrate to the same power."""
    rng = np.random.default_rng(1)
    x = _signal.lfilter([1.0], [1.0, -0.7], rng.normal(size=8000))
    fa, pa, _ = dsp.ar_psd(x, 100.0, order=8, n_freqs=4096)
    fw, pw = dsp.psd(x, 100.0, nperseg=1024)
    from scipy.integrate import trapezoid
    assert trapezoid(pa, fa) == pytest.approx(trapezoid(pw, fw), rel=0.1)
    assert trapezoid(pa, fa) == pytest.approx(np.var(x), rel=0.1)


def test_ar_order_selection_prefers_the_true_order_for_a_clean_ar2():
    x = _ar2(n=4000)
    order, scores = dsp.ar_order_selection(x, max_order=20, criterion="bic")
    # BIC penalises harder and should not wander far above the truth.
    assert 2 <= order <= 6, order
    assert np.isfinite(scores[2])


def test_ar_psd_rejects_impossible_requests():
    x = _ar2(n=200)
    with pytest.raises(ValueError, match="method must be one of"):
        dsp.ar_psd(x, 100.0, order=4, method="nonsense")
    with pytest.raises(ValueError, match="more than"):
        dsp.ar_psd(x[:4], 100.0, order=2)
    with pytest.raises(ValueError, match="samples to fit"):
        dsp.ar_psd(x, 100.0, order=500)


# ── resolution bandwidth and segment advice ──────────────────────────────────

def test_resolution_bandwidth_matches_the_known_window_constants():
    """These are textbook values; a wrong formula would drift from all of them."""
    fs, n = 1000.0, 1024
    expected = {"hann": 1.5, "hamming": 1.3628, "boxcar": 1.0,
                "blackman": 1.7268, "flattop": 3.7702}
    for name, bins in expected.items():
        be = dsp.resolution_bandwidth(fs, n, name)
        assert be == pytest.approx(bins * fs / n, rel=1e-3), name


def test_resolution_bandwidth_is_wider_than_the_bin_spacing():
    # The trap it exists to close: fs/nperseg flatters the estimate.
    fs, n = 1024.0, 4096
    assert dsp.resolution_bandwidth(fs, n, "hann") > fs / n


def test_segment_advice_finds_the_length_that_resolves_a_light_mode():
    """A 1.2% damped mode needs a long segment, and the advice must say so.

    The fixture's first mode is at 8.613 Hz with 1.218% damping, so its
    half-power width is 0.2098 Hz and the classical Be/Br <= 1/4 rule needs a
    resolution near 0.05 Hz. At fs = 1024 that is nperseg of about 32768 —
    thirty-two second segments — which is exactly the answer a user would not
    guess and would never reach by shortening nperseg for averages.
    """
    _, x, _ = generate_2dof(duration=600.0, fs=1024.0, seed=0)

    short = dsp.segment_advice(x, 1024.0, nperseg=1024)
    assert short["verdict"] in ("marginal", "unresolved", "squeezed")
    assert short["ratio"] > 0.25
    assert short["peak_freq"] == pytest.approx(8.6, abs=1.0)

    good = dsp.segment_advice(x, 1024.0, nperseg=32768)
    assert good["verdict"] == "ok"
    assert good["ratio"] <= 0.25
    # The measured width must be converging on the true 0.2098 Hz.
    assert good["peak_bandwidth"] == pytest.approx(0.21, abs=0.06)


def test_segment_advice_reports_both_failure_modes():
    """Too few averages and too coarse a resolution are different diagnoses."""
    _, x, _ = generate_2dof(duration=600.0, fs=1024.0, seed=0)

    # Very long segments: resolution is ample, averages are not.
    few = dsp.segment_advice(x, 1024.0, nperseg=65536, target_segments=20)
    assert few["verdict"] == "too_few"
    assert few["ratio"] < 0.25
    assert few["bias_floor"] > 0.05
    # And crucially it does not simply say "shorten nperseg".
    assert "resolution" in few["advice"].lower()

    coarse = dsp.segment_advice(x, 1024.0, nperseg=256)
    assert coarse["ratio"] > coarse["ratio"] * 0 + 0.25
    assert coarse["n_segments"] > few["n_segments"]


def test_segment_advice_says_when_the_record_is_simply_too_short():
    _, x, _ = generate_2dof(duration=20.0, fs=1024.0, seed=0)
    r = dsp.segment_advice(x, 1024.0, nperseg=1024, target_segments=20)
    assert r["verdict"] == "squeezed"
    assert r["duration_for_both"] > r["record_duration"]
    assert "record" in r["advice"]


def test_the_welch_guardrail_no_longer_advises_only_shortening():
    """The defect: every message ended at 'shorten nperseg'.

    That is right for one failure mode and wrong for the other, and the warning
    is the only place most users will ever read about either.
    """
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=4000), rng.normal(size=4000)
    with pytest.warns(UserWarning) as rec:
        dsp.coherence(a, b, 100.0, nperseg=1024)
    msg = str(rec[0].message)
    assert "resolution" in msg
    assert "segment_advice" in msg

    with pytest.raises(ValueError, match="resolution"):
        dsp.coherence(a, b, 100.0, nperseg=4000)
