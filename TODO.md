# dspkit — what's left

State as of 2026-08-07. Library work only; the GUI's own list lives in
`../DSPkit-app/TODO.md`, which points here for anything algorithmic.

Run the tests with `pytest tests/` — 189 passing.

`ideas_DSPkit_chat.md` is the original scoping document and is now partly
out of date (it says the toolkit avoids "heavy algorithms like full OMA",
which stopped being true when `fdd.py` landed). Treat it as history.

---

## 1. Tapering the autocorrelation

Nothing here is implemented. It is one idea with three separate consequences,
and they are worth keeping apart because only the third is a live defect.

### 1.1 Blackman–Tukey: the spectrum as the transform of the ACF
By Wiener–Khinchin the PSD *is* the Fourier transform of the autocorrelation,
so estimating the spectrum that way is a legitimate third route alongside the
periodogram and Welch. What makes it useful is the knob: the ACF at lag k is
estimated from N−k sample pairs, so long lags are progressively noisier, and
downweighting them with a **lag window** trades resolution for variance
directly and explicitly.

Be honest about the gain. Blackman–Tukey with a lag window of length M and
Welch with segments of length ≈M land in much the same place on bias and
variance — this is not a better estimator, it is a *differently parameterised*
one, and sometimes the parameterisation is the point. Do not ship it claiming
more.

Effective resolution is roughly `1 / (M·dt)`, times a window-dependent
equivalent-bandwidth factor.

### 1.2 The non-negativity trap, which is the reason to be careful
`autocorrelation()` uses the **biased** estimator (divides by N). The docstring
gives the reason as variance at large lags, which is right, but it has a second
consequence worth writing down: the biased ACF is a positive semi-definite
sequence, so its full-length DFT is non-negative *everywhere*. You cannot get
negative power out of it.

**Tapering breaks that guarantee unless the lag window's own Fourier transform
is non-negative.** Then the estimate is the true PSD convolved with a
non-negative kernel, and stays non-negative. Otherwise it does not, and the
estimator can return **negative power** — a number that cannot be true.

- **Bartlett / triangular** — transform is the Fejér kernel, `|Dirichlet|²`,
  non-negative. Safe.
- **Parzen** — non-negative transform, and better sidelobe decay than Bartlett.
  The usual default.
- **Rectangular truncation** — Dirichlet kernel, negative sidelobes. Unsafe.
- **Hann / Hamming** — negative sidelobes as *lag* windows. Unsafe.

That last one is the trap worth stating in the docstring, because it is exactly
backwards from the intuition the rest of the library builds: Hann is the right
default as a **data** window applied to x before the FFT, and the wrong choice
as a **lag** window applied to the ACF. Same name, opposite verdict, different
domain.

Note also that `autocorrelation(max_lag=...)` already performs a rectangular
truncation. That is harmless while the result stays an ACF, and stops being
harmless the moment someone transforms it — which is the natural next thing to
do with it. Worth a line in that docstring even before Blackman–Tukey exists.

### 1.3 The OMA case, where a taper silently changes the answer
In operational modal analysis the correlation of ambient response is
proportional to a free-decay response, and it gets fitted for frequency and
damping. The noisy tail invites a taper, and an **exponential window**
`exp(−βt)` is the common choice.

An exponential window **adds known damping**: the fitted value is
`ζ_fit = ζ_true + β/ωn`, so it must be subtracted afterwards. Forgetting is a
classic and entirely silent error — the damping simply comes out too high and
nothing about the result looks wrong.

If an exponential taper is offered anywhere near a damping estimate, the
correction should be applied by the library rather than left to the caller, or
refused outright. This is the same class of failure the app's TODO §2 is about:
a number that has not been earned.

### 1.4 A live instance of this, in `fdd.py`
`fdd_damping` builds its SDOF bell by MAC-gating the first singular value —
`bell[k] = S[k,0]` where `MAC ≥ threshold`, and zero elsewhere (`fdd.py:275`)
— then inverse-transforms it to a free decay and takes a log decrement.

**That gate is a rectangular window, applied in frequency.** Its inverse
transform convolves the true free decay with a sinc, so the recovered
autocorrelation rings, the envelope is distorted, and the log decrement is
taken on a biased envelope. It is the same problem as §1.2 in the dual domain,
and it is already shipping. Narrow bells — few frequency lines above the MAC
threshold — should make it worse, which is a testable prediction, so start
there rather than assuming.

Likely remedies, cheapest first: taper the bell edges instead of cutting them
square, or weight by MAC continuously rather than thresholding it. Both are
small changes; the work is measuring whether the damping estimate actually
moves, and on what.

---

## 2. Response spectrum

Not started. The peak response of a family of SDOF oscillators against period
T, at one or more damping ratios, for a base-excitation record.

**Use Nigam–Jennings, not Newmark.** The exact piecewise-linear recurrence is a
2×2 state transition, exact when the input is linearly interpolated between
samples — which is the standard assumption anyway. It drops into
`scipy.signal.lfilter` as an IIR filter, so 200 periods over a 20 000-sample
record is milliseconds.

Sensible home is a new `response.py` holding SDOF simulation generally, since
the same machinery gives the shock response spectrum (SRS) used in mechanical
and aerospace testing — same solver, different conventions (maximax / primary /
residual).

**The trap:** pseudo-velocity and pseudo-acceleration are *defined* as
`Sv = ω·Sd` and `Sa = ω²·Sd`. They are **not** the oscillator's peak velocity
and acceleration. Close for light damping, diverging as damping rises. Name
them pseudo, or return both.

Second: the recurrence is exact for the *interpolated* input, but the
interpolation is the approximation, and it fails at short periods on a coarsely
sampled record. The usual guidance is `dt < T/10` — warn or refuse below it
rather than return a confident wrong curve.

---

## 3. Synchrosqueezing, continued

`synchrosqueeze_stft` landed 2026-08-07 (see its docstring for the three things
that had to be got right).

### 3.1 The inverse — the highest-value item here
Coefficients are summed as complex numbers precisely so an inverse stays
possible, but it is not written, and the docstring says so rather than implying
mode extraction works.

That is the payoff: integrate one ridge back into a time-domain signal and you
can take damping off a single isolated mode by log decrement, or follow a mode
whose frequency drifts with temperature or damage.

The window-centre phase reference `(-1)**j` already in the forward transform is
a **prerequisite** — reconstruction sums coefficients across frequency, and
that only works if they share a phase reference. Verify reconstruction error
numerically against the original signal rather than trusting the derivation;
the forward transform needed three separate corrections that only measurement
caught.

### 3.2 Second-order
First-order is biased for strongly chirping components — the instantaneous
frequency estimate lags a fast chirp, because it assumes amplitude and
frequency vary slowly within one window. Second-order corrects it. Natural
follow-up now the machinery exists.

### 3.3 Lower priority
Multitaper spectrogram, Stockwell (S) transform. Both add coverage rather than
capability, and the last several sessions have all shown the value being in the
seams between existing things rather than in new ones.
