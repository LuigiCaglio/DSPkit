"""
Frequency response function estimation, for one input or several.

The FRF is the primary measurement of input-output testing, in the way that
Frequency Domain Decomposition is for output-only. Which estimator to use is
decided by where the noise is, and coherence is how you tell.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.integrate import trapezoid as _trapezoid

from .multisensor import psd_matrix
from .spectral import coherence as _coherence, csd as _csd, psd as _psd

__all__ = ["frf", "frf_mimo", "error_spectrum"]


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
        "output_psd": Gyy,
        "n_segments": int(n_seg),
    }


def error_spectrum(
    target: np.ndarray,
    predictors: np.ndarray,
    fs: float,
    window: str = "hann",
    nperseg: int | None = None,
    noverlap: int | None = None,
    ridge: float = 1e-10,
    min_segments: int | None = None,
) -> dict:
    """
    How much of a signal the other channels cannot account for, per frequency.

    Given a target ``d`` and one or more predictors ``x``, the best linear
    time-invariant model has a residual whose spectrum is fixed entirely by
    the coherence::

        S_ee(f) = [1 - gamma^2(f)] * S_dd(f)

    with ``gamma^2`` the ordinary coherence for one predictor and the multiple
    coherence for several. The target spectrum splits into the part any linear
    model driven by ``x`` can reproduce and the part none can::

        S_dd = gamma^2 * S_dd  +  (1 - gamma^2) * S_dd
                coherent            error spectrum

    This is the frequency-domain form of ``var_resid = (1 - r^2) var_y``, with
    ``gamma^2(f)`` playing the role of a frequency-local R-squared.

    Parameters
    ----------
    target : array_like, shape (N,)
        The signal to be explained.
    predictors : array_like, shape (N,) or (q, N)
        The signals to explain it with.
    fs : float
        Sampling frequency [Hz].
    window, nperseg, noverlap
        Welch parameters. ``nperseg`` is the parameter that decides whether
        the answer is meaningful -- see Notes.
    ridge : float
        Regularisation of the normalised predictor matrix before inversion.
    min_segments : int or None
        Refuse below this many Welch averages. Defaults to ``q + 2``.

    Returns
    -------
    dict with keys
        ``freqs``, ``error_spectrum`` -- ``(1 - gamma^2) S_dd``, in the
        target's units squared per Hz -- ``coherent_power``, ``target_psd``,
        ``coherence`` (multiple, or ordinary when ``q = 1``),
        ``ordinary_coherence`` (shape ``(q, M)``), ``H`` (the optimal filters),
        ``input_condition``, ``bias_floor``, ``n_segments``,
        ``unexplained_variance`` -- the error spectrum integrated over
        frequency -- and ``unexplained_fraction``, that as a share of the
        target's variance.

    Notes
    -----
    **Report the error spectrum, not the coherence alone.** Coherence is
    dimensionless and says *where* a model fails; the error spectrum carries
    the target's units and says *how much that matters*. A band where
    ``gamma^2`` collapses to zero is harmless if ``S_dd`` is negligible there,
    and that is the usual case, because coherence is worst exactly where there
    is no signal to be coherent about. Judging a model by the coherence plot
    alone tends to produce needless alarm. Measured on the four-storey shear chain
    in ``tests/test_frf_response.py``, a single accelerometer averages a
    coherence of 0.50 above the wave band, which looks like a broken model,
    and still accounts for 99.8% of the target's variance -- because that
    band carries two and a half decades less power than the wave band does.

    **Compare the coherence against ``bias_floor``, not against zero.** With
    ``q`` predictors and ``n_d`` Welch averages, genuinely unrelated signals
    give an expected coherence of about ``q / n_d``. Adding predictors always
    raises the apparent coherence, part of which is nothing but extra fitting
    freedom, so the error spectrum returned here is biased *downward* --
    optimistic. Models with different numbers of predictors cannot be ranked
    on this number alone; hold out part of the record instead. A warning is
    issued once the floor passes 0.1.

    **A too-short segment manufactures an error spectrum that is not there.**
    This is the failure mode to watch, and it is the opposite of the one
    ``dspkit.spectral.coherence`` warns about. Let ``B_e`` be the resolution
    bandwidth of the estimate (about ``1.5 fs / nperseg`` for a Hann window)
    and ``B_r = 2 zeta f_r`` the half-power bandwidth of the sharpest feature.
    When ``B_e`` is not small next to ``B_r``, leakage smears peaks and
    notches, coherence is biased downward, and the spurious residual scales as
    ``(B_e / B_r)^2``. Keep ``B_e / B_r <= 1/4``. Because the spurious part is
    proportional to ``S_dd``, it is worst where the target is strong, and it
    can exceed the genuine residual. On a noise-free
    single-input version of the same shear chain, where the true error
    spectrum is exactly zero, the apparent unexplained variance runs
    3.8e-5, 1.6e-4, 6.3e-4 and 2.4e-3 of the target as ``B_e / B_r`` goes
    0.06, 0.12, 0.24, 0.49 -- a clean square law, and every bit of it an
    artefact of the segment length.
    Choose ``nperseg`` from the damping bandwidth of the sharpest mode and
    then lengthen the *record* until the averages are adequate. Do not trade
    resolution for averages when ``S_dd`` is large and sharply peaked.

    **What the residual contains cannot be separated here.** Three distinct
    mechanisms push ``gamma^2`` below one and this decomposition sees only
    their sum: measurement noise on either channel, inputs driving the system
    that are not reflected in the predictors, and nonlinearity, which by
    construction no ``H(f)`` can represent. Noise on a *predictor* is the
    awkward one -- it inflates the denominator and biases the identified gain
    towards zero, so the model is attenuated rather than merely noisy.
    Telling the three apart needs more information: partial coherences,
    repeated tests, or a physical model.

    **This is a lower bound for any causal predictor.** The filters returned
    are the non-causal Wiener solution and use the whole record, past and
    future. A causal finite-order model is constrained by the Wiener-Hopf
    equations and cannot do better, so a fitted ARX or state-space predictor
    sitting well above this curve is limited by its model order, not by the
    physics.

    See Also
    --------
    frf_mimo, dspkit.multisensor.multiple_coherence,
    dspkit.multisensor.partial_coherence

    Examples
    --------
    >>> import numpy as np
    >>> from dspkit.frf import error_spectrum
    >>> rng = np.random.default_rng(0)
    >>> a = rng.normal(size=20000)
    >>> d = np.convolve(a, np.ones(8) / 8, mode="same")
    >>> r = error_spectrum(d, a, 100.0, nperseg=1024)
    >>> bool(r["unexplained_fraction"] < 0.01)
    True
    """
    target = np.asarray(target, dtype=float).ravel()
    predictors = np.atleast_2d(np.asarray(predictors, dtype=float))
    q = predictors.shape[0]

    if predictors.shape[1] != target.size:
        raise ValueError(
            f"Target has {target.size} samples and the predictors have "
            f"{predictors.shape[1]}; they must be simultaneous records of the "
            f"same length."
        )

    r = frf_mimo(
        predictors,
        target,
        fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        ridge=ridge,
        min_segments=min_segments,
    )

    freqs = r["freqs"]
    Sdd = r["output_psd"]
    g2 = r["multiple_coherence"]
    n_seg = int(r["n_segments"])

    err = (1.0 - g2) * Sdd
    coherent = g2 * Sdd

    floor = q / n_seg
    if floor > 0.1:
        warnings.warn(
            f"error_spectrum: {q} predictor(s) over {n_seg} Welch segments "
            f"puts the coherence bias floor at q/n_d = {floor:.2f}. "
            f"Unrelated signals would already show that much coherence, so "
            f"the error spectrum returned here is biased low by roughly the "
            f"same fraction of the target's power. Lengthen the record, or "
            f"validate on data held out from the fit.",
            stacklevel=2,
        )

    var_d = float(_trapezoid(Sdd, freqs))
    var_e = float(_trapezoid(err, freqs))

    return {
        "freqs": freqs,
        "error_spectrum": err,
        "coherent_power": coherent,
        "target_psd": Sdd,
        "coherence": g2,
        "ordinary_coherence": r["ordinary_coherence"],
        "H": r["H"],
        "input_condition": r["input_condition"],
        "bias_floor": float(floor),
        "n_segments": n_seg,
        "unexplained_variance": var_e,
        "unexplained_fraction": (var_e / var_d) if var_d > 0 else float("nan"),
    }
