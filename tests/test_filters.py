"""Tests for dspkit.filters."""

import numpy as np
import pytest

from dspkit.filters import bandpass, bandstop, decimate, highpass, lowpass, notch
from dspkit.spectral import psd

FS = 2000.0
DURATION = 10.0
N = int(FS * DURATION)
t = np.arange(N) / FS


def _sine(freq: float, amplitude: float = 1.0) -> np.ndarray:
    return amplitude * np.sin(2.0 * np.pi * freq * t)


def _power_at(freqs, Pxx, target, bw=5.0) -> float:
    """Total PSD power within ±bw Hz of target frequency."""
    mask = np.abs(freqs - target) <= bw
    return float(Pxx[mask].sum())


# ---------------------------------------------------------------------------
# lowpass
# ---------------------------------------------------------------------------

class TestLowpass:
    def test_passes_low_freq(self):
        x = _sine(10.0)
        y = lowpass(x, FS, cutoff=100.0)
        assert np.std(y) / np.std(x) > 0.99

    def test_attenuates_high_freq(self):
        x = _sine(800.0)
        y = lowpass(x, FS, cutoff=100.0, order=8)
        assert np.std(y) / np.std(x) < 0.01

    def test_output_length(self):
        x = _sine(10.0)
        assert len(lowpass(x, FS, cutoff=100.0)) == N

    def test_invalid_cutoff_raises(self):
        with pytest.raises(ValueError):
            lowpass(_sine(10.0), FS, cutoff=FS)  # above Nyquist

    def test_causal_same_length(self):
        x = _sine(10.0)
        y = lowpass(x, FS, cutoff=100.0, zero_phase=False)
        assert len(y) == N


# ---------------------------------------------------------------------------
# highpass
# ---------------------------------------------------------------------------

class TestHighpass:
    def test_passes_high_freq(self):
        x = _sine(500.0)
        y = highpass(x, FS, cutoff=100.0)
        assert np.std(y) / np.std(x) > 0.99

    def test_attenuates_low_freq(self):
        x = _sine(5.0)
        y = highpass(x, FS, cutoff=100.0, order=8)
        assert np.std(y) / np.std(x) < 0.01

    def test_removes_dc(self):
        x = np.ones(N) * 10.0
        y = highpass(x, FS, cutoff=1.0)
        assert np.max(np.abs(y)) < 0.1


# ---------------------------------------------------------------------------
# bandpass
# ---------------------------------------------------------------------------

class TestBandpass:
    def test_passes_target_freq(self):
        x = _sine(200.0)
        y = bandpass(x, FS, low=100.0, high=400.0)
        assert np.std(y) / np.std(x) > 0.95

    def test_attenuates_out_of_band_low(self):
        x = _sine(10.0)
        y = bandpass(x, FS, low=100.0, high=400.0, order=6)
        assert np.std(y) / np.std(x) < 0.02

    def test_attenuates_out_of_band_high(self):
        x = _sine(900.0)
        y = bandpass(x, FS, low=100.0, high=400.0, order=6)
        assert np.std(y) / np.std(x) < 0.02

    def test_invalid_band_raises(self):
        with pytest.raises(ValueError):
            bandpass(_sine(50.0), FS, low=400.0, high=100.0)  # low > high

    def test_output_length(self):
        x = _sine(200.0)
        assert len(bandpass(x, FS, low=100.0, high=400.0)) == N


# ---------------------------------------------------------------------------
# bandstop
# ---------------------------------------------------------------------------

class TestBandstop:
    def test_attenuates_stop_band(self):
        x = _sine(200.0)
        y = bandstop(x, FS, low=100.0, high=400.0, order=6)
        assert np.std(y) / np.std(x) < 0.05

    def test_passes_outside_stop_band(self):
        x = _sine(20.0)
        y = bandstop(x, FS, low=100.0, high=400.0)
        assert np.std(y) / np.std(x) > 0.95


# ---------------------------------------------------------------------------
# notch
# ---------------------------------------------------------------------------

class TestNotch:
    def test_attenuates_notch_freq(self):
        x = _sine(50.0)   # mains hum
        y = notch(x, FS, freq=50.0, q=30.0)
        assert np.std(y) < 0.05 * np.std(x)

    def test_preserves_other_freqs(self):
        x = _sine(100.0)
        y = notch(x, FS, freq=50.0, q=30.0)
        assert np.std(y) / np.std(x) > 0.99

    def test_output_length(self):
        x = _sine(50.0)
        assert len(notch(x, FS, freq=50.0)) == N

    def test_invalid_freq_raises(self):
        with pytest.raises(ValueError):
            notch(_sine(50.0), FS, freq=0.0)


# ---------------------------------------------------------------------------
# decimate
# ---------------------------------------------------------------------------

class TestDecimate:
    def test_output_length(self):
        x = _sine(10.0)
        x_dec, fs_new = decimate(x, FS, target_fs=500.0)
        assert len(x_dec) == N // 4
        assert fs_new == 500.0

    def test_returned_fs(self):
        x = _sine(10.0)
        _, fs_new = decimate(x, FS, target_fs=200.0)
        assert fs_new == 200.0

    def test_non_integer_ratio_raises(self):
        x = _sine(10.0)
        with pytest.raises(ValueError):
            decimate(x, FS, target_fs=300.0)  # 2000/300 is not integer

    def test_target_above_fs_raises(self):
        x = _sine(10.0)
        with pytest.raises(ValueError):
            decimate(x, FS, target_fs=5000.0)

    def test_low_freq_content_preserved(self):
        """After decimation, PSD peak at a low frequency should be preserved."""
        freq = 20.0
        x = _sine(freq)
        x_dec, fs_new = decimate(x, FS, target_fs=200.0)
        freqs, Pxx = psd(x_dec, fs_new, nperseg=512)
        peak_freq = freqs[np.argmax(Pxx)]
        assert abs(peak_freq - freq) < 2.0

    def test_high_freq_above_nyquist_removed(self):
        """Content above target Nyquist should be removed by anti-aliasing filter."""
        # Signal at 90 Hz, decimate to 100 Hz (Nyquist = 50 Hz) → should be attenuated
        x = _sine(90.0)
        x_dec, fs_new = decimate(x, FS, target_fs=100.0)
        assert np.std(x_dec) < 0.1 * np.std(x)
