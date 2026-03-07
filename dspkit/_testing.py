"""
Signal generators for testing and examples.

These are not part of the public API — they exist to produce physically
meaningful test signals so examples and unit tests don't rely on external data.
"""

from typing import Literal

import numpy as np
from scipy import signal as _signal


def generate_2dof(
    duration: float = 30.0,
    fs: float = 1000.0,
    m1: float = 1.0,
    m2: float = 1.0,
    k1: float = 10_000.0,
    k2: float = 5_000.0,
    c1: float = 5.0,
    c2: float = 2.0,
    noise_std: float = 1.0,
    output: Literal["displacement", "velocity", "acceleration"] = "acceleration",
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate a 2DOF spring-mass-damper chain under white noise force excitation.

    Layout::

        ground --[k1,c1]-- m1 --[k2,c2]-- m2

    Independent white noise forces are applied to both masses.

    Default natural frequencies (undamped):
        fn1 ~ 8.6 Hz,  fn2 ~ 20.8 Hz

    Use `natural_frequencies_2dof()` to compute exact values for any parameters.

    Parameters
    ----------
    duration : float
        Signal duration [s].
    fs : float
        Sampling frequency [Hz].
    m1, m2 : float
        Masses [kg].
    k1, k2 : float
        Stiffnesses [N/m].
    c1, c2 : float
        Viscous damping coefficients [N·s/m].
    noise_std : float
        Standard deviation of the white noise force applied to each mass [N].
    output : {'displacement', 'velocity', 'acceleration'}
        Physical quantity to return.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    t : ndarray, shape (N,)
        Time vector [s].
    x1 : ndarray, shape (N,)
        Response of mass 1.
    x2 : ndarray, shape (N,)
        Response of mass 2.
    """
    rng = np.random.default_rng(seed)
    N = int(duration * fs)
    t = np.arange(N) / fs

    # State vector: q = [x1, x2, v1, v2]
    # Equations of motion:
    #   m1*x1'' = -k1*x1 - c1*v1 - k2*(x1-x2) - c2*(v1-v2) + f1
    #   m2*x2'' =                   k2*(x1-x2) + c2*(v1-v2)          + f2
    A = np.array([
        [0.0,               0.0,              1.0,              0.0],
        [0.0,               0.0,              0.0,              1.0],
        [-(k1 + k2) / m1,   k2 / m1,        -(c1 + c2) / m1,   c2 / m1],
        [  k2 / m2,        -k2 / m2,           c2 / m2,        -c2 / m2],
    ])
    B = np.array([
        [0.0,      0.0  ],
        [0.0,      0.0  ],
        [1.0 / m1, 0.0  ],
        [0.0,      1.0 / m2],
    ])

    if output == "displacement":
        C_out = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        D_out = np.zeros((2, 2))
    elif output == "velocity":
        C_out = np.array([[0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        D_out = np.zeros((2, 2))
    else:  # acceleration
        # a = M^{-1}(F - K*x - C*v)  =>  C_out = A[2:,:],  D_out = B[2:,:]
        C_out = A[2:, :]
        D_out = B[2:, :]

    sys = _signal.StateSpace(A, B, C_out, D_out)

    F = rng.normal(0.0, noise_std, size=(N, 2))
    _, y, _ = _signal.lsim(sys, U=F, T=t)

    return t, y[:, 0], y[:, 1]


def natural_frequencies_2dof(
    m1: float = 1.0,
    m2: float = 1.0,
    k1: float = 10_000.0,
    k2: float = 5_000.0,
) -> tuple[float, float]:
    """
    Compute undamped natural frequencies of the 2DOF chain system.

    Returns
    -------
    fn1, fn2 : float
        Natural frequencies [Hz], sorted ascending.
    """
    M_inv = np.diag([1.0 / m1, 1.0 / m2])
    K = np.array([[k1 + k2, -k2], [-k2, k2]])
    eigvals = np.linalg.eigvalsh(M_inv @ K)
    fn = np.sqrt(np.maximum(eigvals, 0.0)) / (2.0 * np.pi)
    return float(fn[0]), float(fn[1])


def generate_sine(
    freqs: float | list[float],
    amplitudes: float | list[float] = 1.0,
    duration: float = 5.0,
    fs: float = 1000.0,
    noise_std: float = 0.0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a multi-tone sine signal with optional additive white noise.

    Parameters
    ----------
    freqs : float or list of float
        Frequency or frequencies [Hz].
    amplitudes : float or list of float
        Amplitude(s) of each tone. A scalar applies to all tones.
    duration : float
        Signal duration [s].
    fs : float
        Sampling frequency [Hz].
    noise_std : float
        Standard deviation of additive white Gaussian noise.
    seed : int or None
        Random seed.

    Returns
    -------
    t : ndarray
    x : ndarray
    """
    freqs = [freqs] if np.isscalar(freqs) else list(freqs)
    if np.isscalar(amplitudes):
        amplitudes = [amplitudes] * len(freqs)

    t = np.arange(int(duration * fs)) / fs
    x = sum(a * np.sin(2.0 * np.pi * f * t) for f, a in zip(freqs, amplitudes))

    if noise_std > 0.0:
        x = x + np.random.default_rng(seed).normal(0.0, noise_std, size=len(t))

    return t, np.asarray(x)
