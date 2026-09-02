"""
Frequency response function estimation, for one input or several.

The FRF is the primary measurement of input-output testing, in the way that
Frequency Domain Decomposition is for output-only. Which estimator to use is
decided by where the noise is, and coherence is how you tell.
"""

from __future__ import annotations

import numpy as np

from .multisensor import psd_matrix
from .spectral import coherence as _coherence, csd as _csd, psd as _psd

__all__ = ["frf", "frf_mimo"]


def frf(
    x: np.ndarray,
    y: np.ndarray,
    fs: float,
    estimator: str = "H1",
    window: str = "hann",
    nperseg: int | None = None,
    noverlap: int | None = None,
) -> dict:
    """
    Frequency response function between one input and one output.

    Parameters
    ----------
    x : array_like, shape (N,)
        Input (excitation), e.g. a measured force.
    y : array_like, shape (N,)
        Output (response), e.g. an acceleration.
    fs : float
        Sampling frequency [Hz].
    estimator : {'H1', 'H2', 'H3'}
        Which estimator to form. See the notes.
    window, nperseg, noverlap
        Welch parameters, as elsewhere in the library.

    Returns
    -------
    dict with keys
        ``freqs``, ``H`` (complex), ``magnitude``, ``phase_deg``,
        ``coherence``, and ``estimator``.

    Notes
    -----
    The three estimators differ only in which spectrum sits where, and they
    answer to different noise:

    - ``H1 = Gxy / Gxx`` assumes the noise is on the **output**. It is the usual
      default, and it is biased *down* at resonance, where the response is large
      and any input noise matters most.
    - ``H2 = Gyy / Gyx`` assumes the noise is on the **input**. It is biased *up*
      at resonance and is the better choice at anti-resonances, where the output
      is small and output noise dominates.
    - ``H3`` is their geometric mean, a compromise with no cleaner justification
      than that.

    Coherence is returned alongside because it is what says whether either
    number is worth reading. Where coherence is near 1, H1 and H2 agree and the
    choice does not matter; where they disagree, coherence has already dropped
    and is telling you why.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("Input and output must be the same length.")

    est = estimator.upper()
    if est not in ("H1", "H2", "H3"):
        raise ValueError("estimator must be one of 'H1', 'H2', 'H3'.")

    kw = dict(fs=fs, window=window, nperseg=nperseg, noverlap=noverlap)
    freqs, Gxy = _csd(x, y, **kw)
    _, Gxx = _psd(x, **kw)
    _, Gyy = _psd(y, **kw)
    _, coh = _coherence(x, y, **kw)

    eps = np.finfo(float).tiny
    H1 = Gxy / np.maximum(Gxx, eps)
    # Gyx is the conjugate of Gxy; forming it that way avoids a second transform.
    H2 = Gyy / np.maximum(np.abs(np.conj(Gxy)), eps) * np.exp(1j * np.angle(Gxy))

    if est == "H1":
        H = H1
    elif est == "H2":
        H = H2
    else:
        H = np.sqrt(np.abs(H1) * np.abs(H2)) * np.exp(1j * np.angle(H1))

    return {
        "freqs": freqs,
        "H": H,
        "magnitude": np.abs(H),
        "phase_deg": np.angle(H, deg=True),
        "coherence": coh,
        "estimator": est,
    }


def frf_mimo(
    inputs: np.ndarray,
    output: np.ndarray,
    fs: float,
    window: str = "hann",
    nperseg: int | None = None,
    noverlap: int | None = None,
    ridge: float = 1e-10,
    min_segments: int | None = None,
) -> dict:
    """
    Frequency response functions from several simultaneous inputs to one output.

    Solves ``H = inv(Gxx) @ Gxy`` at each frequency, where ``Gxx`` is the
    inputs' own cross-spectral matrix and ``Gxy`` the input-output
    cross-spectrum. This is the multi-input generalisation of ``H1``.

    Parameters
    ----------
    inputs : array_like, shape (n_inputs, N)
        Simultaneous excitations.
    output : array_like, shape (N,)
        The response.
    fs : float
        Sampling frequency [Hz].
    window, nperseg, noverlap
        Welch parameters.
    ridge : float
        Regularisation added to the normalised input matrix before inversion.
    min_segments : int or None
        Refuse below this many Welch averages. Defaults to ``n_inputs + 2``.

    Returns
    -------
    dict with keys
        ``freqs``, ``H`` (complex, shape ``(n_inputs, M)``), ``magnitude``,
        ``phase_deg``, ``multiple_coherence`` (shape ``(M,)``),
        ``ordinary_coherence`` (shape ``(n_inputs, M)``), and
        ``input_condition`` -- the condition number of the normalised input
        matrix at each frequency, which is the diagnostic that matters.

    Notes
    -----
    **Correlated inputs are the thing to watch, and coherence will not warn
    you.** If two shakers drive the structure in a correlated way, ``Gxx`` is
    near-singular and the split of credit between the inputs is arbitrary --
    the individual FRFs can be meaningless while together they still predict
    the output perfectly. Measured on two inputs at 0.95 correlation, both the
    multiple coherence (0.996) and the ordinary coherences (0.994, 0.993) stayed
    high while the FRFs were not separable. Neither number tells you anything is
    wrong.

    ``input_condition`` is what does. It is the condition number of the
    normalised input cross-spectral matrix: near 1 the inputs are distinguishable
    and the FRFs mean something individually; large (say above 100) they are not,
    and only their combined effect is identifiable. Drive the shakers with
    uncorrelated signals if you need the FRFs separately.

    ``Gxx`` is only invertible with more Welch averages than inputs. Below that
    the result is arbitrary rather than merely noisy, so it is refused.
    """
    inputs = np.atleast_2d(np.asarray(inputs, dtype=float))
    output = np.asarray(output, dtype=float).ravel()
    n_in, n = inputs.shape
    if output.size != n:
        raise ValueError("Inputs and output must be the same length.")

    if nperseg is None:
        nperseg = min(n, 1024)
    step = nperseg // 2 if noverlap is None else nperseg - noverlap
    n_seg = 1 + max(0, (n - nperseg)) // max(1, step)
    floor = (n_in + 2) if min_segments is None else int(min_segments)
    if n_seg < floor:
        raise ValueError(
            "Only {} Welch segment(s) for {} inputs. The input cross-spectral "
            "matrix is not invertible with fewer averages than inputs, so the "
            "result would be arbitrary rather than noisy. Use nperseg below "
            "{} to get more segments.".format(n_seg, n_in, n // floor)
        )

    # One matrix over inputs and output together: its top-left block is Gxx and
    # its last column the input-output cross-spectrum, so a single Welch pass
    # gives everything.
    stacked = np.vstack([inputs, output[None, :]])
    freqs, G = psd_matrix(stacked, fs, window=window, nperseg=nperseg,
                          noverlap=noverlap)

    M = freqs.size
    H = np.zeros((n_in, M), dtype=complex)
    mult_coh = np.zeros(M)
    cond = np.zeros(M)

    Gyy = np.real(G[n_in, n_in, :])
    eps = np.finfo(float).tiny
    for k in range(M):
        Gxx = G[:n_in, :n_in, k]
        Gxy = G[:n_in, n_in, k]

        # Regularise on the normalised matrix: the ridge is scale-dependent and
        # a raw one would swamp a low-power input while barely touching a
        # strong one.
        d = np.sqrt(np.maximum(np.abs(np.diag(Gxx)), eps))
        D = np.outer(d, d)
        Gn = Gxx / D
        Gn_inv = np.linalg.inv(Gn + ridge * np.eye(n_in))
        Gxx_inv = Gn_inv / D

        # Conditioning of the *normalised* input matrix: this is the honest
        # measure of whether the inputs can be told apart, and it is scale-free.
        cond[k] = float(np.linalg.cond(Gn))

        H[:, k] = Gxx_inv @ Gxy
        # Share of the output explained by all inputs together.
        num = np.real(np.conj(Gxy) @ (Gxx_inv @ Gxy))
        mult_coh[k] = num / max(Gyy[k], eps)

    # Float error can push this a hair outside [0, 1]; that is what is clipped,
    # not a genuinely out-of-range value.
    mult_coh = np.clip(mult_coh, 0.0, 1.0)

    ordinary = np.zeros((n_in, M))
    for i in range(n_in):
        Gxx_i = np.real(G[i, i, :])
        ordinary[i] = np.clip(
            np.abs(G[i, n_in, :]) ** 2 / np.maximum(Gxx_i * Gyy, eps), 0.0, 1.0)

    return {
        "freqs": freqs,
        "H": H,
        "magnitude": np.abs(H),
        "phase_deg": np.angle(H, deg=True),
        "multiple_coherence": mult_coh,
        "ordinary_coherence": ordinary,
        "input_condition": cond,
        "n_segments": int(n_seg),
    }
