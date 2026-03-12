"""Tests for dspkit.fdd."""

import numpy as np
import pytest

from dspkit.fdd import fdd_svd, fdd_peak_picking, fdd_mode_shapes, efdd_damping, _mac
from dspkit._testing import generate_2dof, natural_frequencies_2dof


FS = 1000.0
DURATION = 120.0


@pytest.fixture(scope="module")
def two_dof_data():
    """Generate 2DOF response data (cached across tests)."""
    t, x1, x2 = generate_2dof(duration=DURATION, fs=FS, seed=42)
    data = np.vstack([x1, x2])
    return data


@pytest.fixture(scope="module")
def fdd_results(two_dof_data):
    """Run FDD SVD (cached across tests)."""
    freqs, S, U = fdd_svd(two_dof_data, FS, nperseg=4096)
    return freqs, S, U


class TestFddSvd:
    def test_output_shapes(self, fdd_results):
        freqs, S, U = fdd_results
        M = len(freqs)
        assert S.shape == (M, 2)
        assert U.shape == (M, 2, 2)

    def test_singular_values_nonneg(self, fdd_results):
        _, S, _ = fdd_results
        assert np.all(S >= -1e-12)

    def test_sv1_peaks_near_natural_freqs(self, fdd_results):
        """First singular value should peak near the 2DOF natural frequencies."""
        freqs, S, _ = fdd_results
        fn1, fn2 = natural_frequencies_2dof()
        sv1 = S[:, 0]
        # Find the two largest peaks in SV1
        from scipy.signal import find_peaks as sp_find_peaks
        idx, _ = sp_find_peaks(sv1, height=sv1.max() * 0.01)
        peak_freqs = freqs[idx]
        assert any(abs(f - fn1) < 1.5 for f in peak_freqs), \
            f"fn1={fn1:.1f} not found in {peak_freqs}"
        assert any(abs(f - fn2) < 1.5 for f in peak_freqs), \
            f"fn2={fn2:.1f} not found in {peak_freqs}"


class TestFddPeakPicking:
    def test_detects_two_modes(self, fdd_results):
        freqs, S, _ = fdd_results
        fn1, fn2 = natural_frequencies_2dof()
        peak_freqs, peak_idx = fdd_peak_picking(
            freqs, S, distance_hz=5.0, max_peaks=2,
        )
        assert len(peak_freqs) == 2
        detected = sorted(peak_freqs)
        assert abs(detected[0] - fn1) < 2.0, f"Expected ~{fn1:.1f}, got {detected[0]:.1f}"
        assert abs(detected[1] - fn2) < 2.0, f"Expected ~{fn2:.1f}, got {detected[1]:.1f}"

    def test_freq_range_filter(self, fdd_results):
        freqs, S, _ = fdd_results
        fn1, _ = natural_frequencies_2dof()
        peak_freqs, _ = fdd_peak_picking(
            freqs, S, freq_range=(0, 15.0), max_peaks=1,
        )
        assert len(peak_freqs) >= 1
        assert abs(peak_freqs[0] - fn1) < 2.0


class TestFddModeShapes:
    def test_normalised_mode_shapes(self, fdd_results):
        freqs, S, U = fdd_results
        _, peak_idx = fdd_peak_picking(freqs, S, distance_hz=5.0, max_peaks=2)
        modes = fdd_mode_shapes(U, peak_idx, normalize=True)
        assert modes.shape == (2, 2)
        # Each mode should have max component = 1
        for i in range(2):
            assert abs(np.max(np.abs(modes[i])) - 1.0) < 1e-10

    def test_mode_shapes_different(self, fdd_results):
        """The two mode shapes should be distinguishable (low MAC)."""
        freqs, S, U = fdd_results
        _, peak_idx = fdd_peak_picking(freqs, S, distance_hz=5.0, max_peaks=2)
        modes = fdd_mode_shapes(U, peak_idx)
        mac_val = _mac(modes[0], modes[1])
        assert mac_val < 0.5, f"Modes too similar, MAC={mac_val:.3f}"


class TestMac:
    def test_identical_vectors(self):
        phi = np.array([1.0 + 0j, 2.0, 3.0])
        assert abs(_mac(phi, phi) - 1.0) < 1e-10

    def test_orthogonal_vectors(self):
        phi_a = np.array([1.0, 0.0])
        phi_b = np.array([0.0, 1.0])
        assert abs(_mac(phi_a, phi_b)) < 1e-10

    def test_range(self):
        rng = np.random.default_rng(0)
        for _ in range(10):
            a = rng.normal(0, 1, 5) + 1j * rng.normal(0, 1, 5)
            b = rng.normal(0, 1, 5) + 1j * rng.normal(0, 1, 5)
            m = _mac(a, b)
            assert 0.0 <= m <= 1.0 + 1e-10


class TestEfddDamping:
    def test_positive_damping(self, fdd_results):
        """Estimated damping ratios should be positive (underdamped system)."""
        freqs, S, U = fdd_results
        _, peak_idx = fdd_peak_picking(freqs, S, distance_hz=5.0, max_peaks=2)
        zeta, fn = efdd_damping(freqs, S, U, peak_idx, FS)
        for i in range(len(zeta)):
            if not np.isnan(zeta[i]):
                assert zeta[i] > 0, f"Mode {i}: ζ={zeta[i]} should be > 0"
                assert zeta[i] < 0.2, f"Mode {i}: ζ={zeta[i]} unreasonably high"
