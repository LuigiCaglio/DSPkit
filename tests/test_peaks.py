"""Tests for dspkit.peaks."""

import numpy as np
import pytest

from dspkit.peaks import find_peaks, peak_bandwidth, find_harmonics
from dspkit.spectral import fft_spectrum, psd
from dspkit._testing import generate_sine


FS = 2000.0
DURATION = 10.0


class TestFindPeaks:
    def test_single_tone_detected(self):
        """A single sine should produce one dominant peak."""
        _, x = generate_sine(freqs=100.0, duration=DURATION, fs=FS)
        freqs, amp = fft_spectrum(x, FS)
        pf, pv, proms = find_peaks(freqs, amp, max_peaks=1)
        assert len(pf) == 1
        assert abs(pf[0] - 100.0) < 1.0

    def test_two_tones_detected(self):
        """Two sines at different frequencies should produce two peaks."""
        _, x = generate_sine(freqs=[50.0, 200.0], duration=DURATION, fs=FS)
        freqs, amp = fft_spectrum(x, FS)
        pf, pv, proms = find_peaks(freqs, amp, distance_hz=10.0, max_peaks=2)
        assert len(pf) == 2
        detected = sorted(pf)
        assert abs(detected[0] - 50.0) < 2.0
        assert abs(detected[1] - 200.0) < 2.0

    def test_prominence_filter(self):
        """High prominence threshold should filter out small peaks."""
        _, x = generate_sine(
            freqs=[50.0, 200.0],
            amplitudes=[1.0, 0.01],
            duration=DURATION, fs=FS,
        )
        freqs, amp = fft_spectrum(x, FS)
        # Only the large peak should survive a high prominence threshold
        pf, _, _ = find_peaks(freqs, amp, prominence=0.1)
        assert len(pf) >= 1
        assert any(abs(f - 50.0) < 2.0 for f in pf)

    def test_empty_spectrum(self):
        """All-zero spectrum should return no peaks."""
        freqs = np.linspace(0, 100, 512)
        spectrum = np.zeros(512)
        pf, pv, proms = find_peaks(freqs, spectrum)
        assert len(pf) == 0


class TestPeakBandwidth:
    def test_narrow_peak_has_small_bandwidth(self):
        """A long pure tone should have very narrow bandwidth."""
        _, x = generate_sine(freqs=100.0, duration=DURATION, fs=FS)
        freqs, Pxx = psd(x, FS, nperseg=2048)
        pf, bw, Q = peak_bandwidth(freqs, Pxx, peak_freqs=np.array([100.0]))
        assert len(bw) == 1
        assert bw[0] < 5.0  # very narrow

    def test_q_factor_positive(self):
        _, x = generate_sine(freqs=100.0, duration=DURATION, fs=FS)
        freqs, Pxx = psd(x, FS, nperseg=2048)
        _, _, Q = peak_bandwidth(freqs, Pxx, peak_freqs=np.array([100.0]))
        assert Q[0] > 0


class TestFindHarmonics:
    def test_harmonic_series(self):
        """A signal with harmonics should have them detected."""
        t = np.arange(int(DURATION * FS)) / FS
        x = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 100 * t) + 0.3 * np.sin(2 * np.pi * 150 * t)
        freqs, amp = fft_spectrum(x, FS)
        hf, hv, ho = find_harmonics(freqs, amp, fundamental=50.0, n_harmonics=3)
        assert len(hf) == 3
        assert 1 in ho and 2 in ho and 3 in ho
        np.testing.assert_allclose(sorted(hf), [50.0, 100.0, 150.0], atol=1.0)

    def test_missing_harmonic(self):
        """Only present harmonics should be returned."""
        t = np.arange(int(DURATION * FS)) / FS
        x = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 150 * t)
        freqs, amp = fft_spectrum(x, FS)
        hf, hv, ho = find_harmonics(freqs, amp, fundamental=50.0, n_harmonics=3, tolerance_hz=1.0)
        # 2nd harmonic (100 Hz) is missing — should still find 1st and 3rd
        assert 1 in ho
        assert 3 in ho
