"""
Tests for dspkit.spectral.

Strategy: use analytically known results (pure sines, white noise properties)
so tests are deterministic and self-documenting.
"""

import numpy as np
import pytest

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

    def test_parseval_spectrum_scaling(self):
        """Power spectrum integrated over all bins should equal signal power."""
        x = _sine(100.0, amplitude=1.0)
        freqs, Pxx = psd(x, FS, scaling="spectrum")
        # Signal power = A^2 / 2 = 0.5
        total_power = Pxx.sum()
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
        np.testing.assert_allclose(acf[10:-10], expected[10:-10], atol=0.02)


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
