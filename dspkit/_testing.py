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


def jonswap_spectrum(
    freqs: np.ndarray,
    Hs: float = 4.0,
    Tp: float = 9.0,
    gamma: float = 3.3,
) -> np.ndarray:
    """
    JONSWAP wave spectrum, one-sided, in m^2/Hz.

    Parameters
    ----------
    freqs : array_like
        Frequencies [Hz]. The zero bin is returned as zero.
    Hs : float
        Significant wave height [m].
    Tp : float
        Peak period [s].
    gamma : float
        Peak enhancement factor (3.3 is the standard mean value).
    """
    freqs = np.asarray(freqs, dtype=float)
    fp = 1.0 / Tp
    S = np.zeros_like(freqs)
    m = freqs > 0
    f = freqs[m]
    sigma = np.where(f <= fp, 0.07, 0.09)
    r = np.exp(-((f - fp) ** 2) / (2.0 * sigma ** 2 * fp ** 2))
    S[m] = ((1.0 - 0.287 * np.log(gamma)) * 5.0 / 16.0 * Hs ** 2 * fp ** 4
            * f ** -5 * np.exp(-1.25 * (fp / f) ** 4) * gamma ** r)
    return S


def generate_shear4(
    fs: float = 10.0,
    duration: float = 25_200.0,
    burn_in: float = 600.0,
    noise_frac: float = 0.01,
    unmeasured_frac: float = 0.05,
    force_rms: float = 0.3e6,
    seed: int | None = 0,
) -> dict:
    """
    Four-storey shear chain under JONSWAP wave loading, with a hidden force.

    A fixed-base chain of four 100-tonne masses on stiffnesses
    ``[5, 5, 4, 3] MN/m``, Rayleigh-damped to 2% at the first mode and 3% at
    the fourth, giving natural frequencies 0.375, 0.984, 1.516 and 1.985 Hz.
    Wave loading drives DOF 1; almost all of the response power therefore sits
    in the wave band, well below the first mode.

    **What makes this fixture useful is the second force.** A broadband force
    also acts at DOF 3 and is *not* returned, so no linear model built from
    the channels here can be exact and the residual has a floor that is a
    property of the system rather than of the estimator. That is what lets
    ``dspkit.frf.error_spectrum`` be tested against a known answer instead of
    against itself. Measurement noise is added to every channel on top.

    Parameters
    ----------
    fs : float
        Sampling frequency [Hz].
    duration : float
        Record length [s] after the burn-in is discarded (default 7 hours).
    burn_in : float
        Discarded settling time [s].
    noise_frac : float
        Gaussian measurement noise on each channel, as a fraction of its own
        rms. Set to 0 for a noise-free system.
    unmeasured_frac : float
        The hidden force at DOF 3, as a fraction of the wave force rms. Set to
        0 to make the map from any acceleration to the displacement exactly
        linear -- the configuration in which any reported error spectrum is
        pure estimation bias.
    force_rms : float
        Rms of the wave force [N].
    seed : int or None
        Seed for the realisation and the noise.

    Returns
    -------
    dict with keys
        ``fs``, ``fn`` (the four natural frequencies [Hz]), ``d`` --
        displacement at DOF 1 [m], the natural target -- and ``A``, ``B``,
        ``C``, accelerations at DOF 4, 2 and 3 [m/s^2].

    Notes
    -----
    Seven hours at 10 Hz is 252,000 samples and the simulation is a Python
    loop over the state, so a call takes a few seconds. It is worth caching in
    a module-scoped fixture rather than regenerating per test.
    """
    rng = np.random.default_rng(seed)

    M = np.eye(4) * 1e5
    k = np.array([5.0, 5.0, 4.0, 3.0]) * 1e6
    K = np.zeros((4, 4))
    for i in range(4):
        K[i, i] += k[i]
        if i + 1 < 4:
            K[i, i] += k[i + 1]
            K[i, i + 1] -= k[i + 1]
            K[i + 1, i] -= k[i + 1]
    wn = np.sqrt(np.linalg.eigvalsh(np.linalg.solve(M, K)))

    # Rayleigh damping fitted to 2% at mode 1 and 3% at mode 4.
    a, b = np.linalg.solve(
        np.array([[1.0 / (2 * wn[0]), wn[0] / 2.0],
                  [1.0 / (2 * wn[3]), wn[3] / 2.0]]),
        np.array([0.02, 0.03]))
    C = a * M + b * K

    n_burn = int(burn_in * fs)
    n = n_burn + int(duration * fs)

    # Spectral representation: random phases on the JONSWAP amplitude spectrum.
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    amp = np.sqrt(2.0 * jonswap_spectrum(freqs) * (freqs[1] - freqs[0]))
    X = amp * np.exp(1j * rng.uniform(0.0, 2 * np.pi, size=freqs.size))
    X[0] = 0.0
    X[-1] = np.abs(X[-1])
    f_wave = np.fft.irfft(X * n / 2.0, n=n)
    f_wave *= force_rms / np.std(f_wave)

    f_hidden = rng.normal(size=n)
    if unmeasured_frac:
        f_hidden *= unmeasured_frac * force_rms / np.std(f_hidden)
    else:
        f_hidden *= 0.0

    Minv = np.linalg.inv(M)
    Bsel = np.zeros((4, 2))
    Bsel[0, 0] = 1.0          # wave force at DOF 1
    Bsel[2, 1] = 1.0          # hidden force at DOF 3
    Ac = np.block([[np.zeros((4, 4)), np.eye(4)], [-Minv @ K, -Minv @ C]])
    Bc = np.vstack([np.zeros((4, 2)), Minv @ Bsel])
    Ad, Bd, _, _, _ = _signal.cont2discrete(
        (Ac, Bc, np.eye(8), np.zeros((8, 2))), 1.0 / fs, method="zoh")

    u = np.vstack([f_wave, f_hidden])
    x = np.zeros((8, n))
    for t in range(n - 1):
        x[:, t + 1] = Ad @ x[:, t] + Bd @ u[:, t]

    q, qd = x[:4], x[4:]
    acc = -Minv @ K @ q - Minv @ C @ qd + Minv @ Bsel @ u

    out = [q[0, n_burn:], acc[3, n_burn:], acc[1, n_burn:], acc[2, n_burn:]]
    if noise_frac:
        out = [s + rng.normal(size=s.size) * noise_frac * np.std(s) for s in out]

    return {"fs": fs, "fn": wn / (2 * np.pi),
            "d": out[0], "A": out[1], "B": out[2], "C": out[3]}
