# dspkit — what's left

State as of **2026-09-10**, at **v0.5.0**. Library work only; the app's own list
lives in `../DSPkit-app/TODO.md`, which points here for anything algorithmic.

Run the tests with `pytest tests/` — **306 passing**.

`ideas_DSPkit_chat.md` is the original scoping document and is now largely out
of date; treat it as history.

## Where to pick up

Nothing here is blocking. In rough order of value:

1. **§3.1, the FSST inverse.** Still the highest-value item. The coefficients
   are summed as complex numbers precisely so an inverse stays possible, and it
   is what mode extraction needs — integrate one ridge back to a time signal and
   you can take damping off a single isolated mode, or follow one that drifts.
2. **§1.3 and §1.4**, the OMA consequences of lag-window tapering, now that
   §1.1/§1.2 have landed. §1.4 names a live defect in `fdd.py`; it is the only
   item here that is a possible correctness problem rather than a feature.
3. **§6.3, conditional mutual information.** The app has a natural home for it
   already — it is the MI analogue of partial coherence, and the coherence pair
   is there.
4. **Shock response spectrum conventions** (maximax / primary / residual). The
   solver in `response.py` supports it; only the peak-taking differs.

**A habit worth keeping**, since it caught four real defects in this round:
check claims by measuring, not by reading. The lag-window non-negativity, the
log-decrement peak spacing, the MIMO condition number, and the mean-removal
bias in `log_decrement` were all things the code or docs asserted confidently
and the measurement contradicted.

---

## 1. Tapering the autocorrelation — done 2026-09-02

`blackman_tukey_psd` and `lag_window` in `spectral.py`, with
`NONNEGATIVE_LAG_WINDOWS` naming the safe set.

§1.2 was right, and now measured rather than argued. On a narrowband record,
rectangular truncation puts **48% of the spectrum below zero** and Hamming 38%,
while Bartlett, Parzen and exponential give exactly 0%. Negative power is not a
possible value, so that is a defect of the window and not of the data. Hann
happened to give 0% on this record but is still unsafe in principle, and is
labelled as such rather than by whether one dataset caught it out.

Exponential is included as a fourth safe window -- its transform is a
Lorentzian, and it is what modal testing uses to add known artificial damping.

The docstring states the trap as §1.2 asked: Hann is the right default as a
*data* window and the wrong choice as a *lag* window. Same name, opposite
verdict, different domain.

Still open from §1.3 and §1.4: the OMA case, and whether `fdd.py` is affected.

## 2. Response spectrum — done 2026-09-02

`response.py` holds `sdof_response`, `response_spectrum` and `log_decrement`.
Nigam-Jennings as specified; validated against the closed-form steady-state
response at three frequency ratios including resonance, all within 0.1%.

Both traps in the original note are handled. Pseudo and true are returned
separately (`PSv`/`PSa` alongside `Sv`/`Sa`) rather than one being labelled
ambiguously, and periods below `10*dt` raise a warning naming the limit.

Not done: the shock response spectrum conventions (maximax / primary /
residual). The solver supports it; only the peak-taking differs.

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

---

## 4. Coherence beyond pairs — is a sensor set predictive?  *(done 2026-09-02)*

The question: given a set of signals, do they carry enough information to
predict the rest.

### 4.1 Multiple and partial coherence — done 2026-09-02

Shipped in `multisensor.py`, both returning per-frequency curves:

```
multiple_coherence(data, fs, window="hann", nperseg=None, noverlap=None,
                   detrend="constant", ridge=1e-10, min_segments=None)
    -> (freqs (M,), gamma2 (n_channels, M))

partial_coherence(data, fs, window="hann", nperseg=None, noverlap=None,
                  detrend="constant", ridge=1e-10, min_segments=None)
    -> (freqs (M,), C (n_channels, n_channels, M))
```

**Ridge, not pseudo-inverse — and they are not interchangeable.** §4.1 said the
conditioning "needs a pseudo-inverse or ridge regularisation", as if either
would do. It measurably would not. When a channel is an exact linear
combination of the others, G is exactly rank-deficient and the true multiple
coherence is 1, which needs `inv(G)_ii → ∞`. `pinv` truncates precisely that
direction and returns a *finite* `inv(G)_ii`, small enough that
`1 - 1/(G_ii·inv_ii)` comes out **negative** — measured −0.6 to −2.9 on three
channels where the third is the sum of the first two. Clipped to [0, 1] that
becomes 0.0: the most redundant array that can exist, reported as perfectly
independent. It fails hardest on the case the function exists to detect. A
ridge keeps the limit and returns `1 - O(ridge)`.
`test_pinv_would_fail_the_redundant_case` pins the difference so the choice
cannot be quietly reversed.

The ridge is applied to the **normalised** matrix `D⁻¹ G D⁻¹`,
`D = diag(sqrt(G_ii))`, not to G. Both coherences are invariant under that
normalisation, but a ridge is not: added to raw G it is an absolute quantity
and would swamp a low-power channel while leaving a high-power one untouched.
On the normalised matrix it is relative and treats every channel alike.
Default 1e-10, a numerical floor bounding the condition number near 1e10 — the
statistical control on near-singularity is the segment count, not this knob.

**The rank claim above was nearly right, and the correction is useful.** §4.1
said "with `n_segments <= n_channels` every coherence comes back at 1.0". The
measured behaviour, four independent noise channels, N = 20480:

| n_segments | mean multiple coherence | q / n_d |
|-----------:|------------------------:|--------:|
|          3 |                   1.000 |    1.00 |
|          4 |                   0.759 |    0.75 |
|          9 |                   0.343 |    0.33 |
|         19 |                   0.167 |    0.16 |
|         39 |                   0.079 |    0.077 |

One formula explains the whole column: independent channels sit at the bias
floor `q / n_d` with `q = n_channels - 1`, and that floor *reaches* 1.0 at
`n_d = q`, which is exactly where the matrix goes singular. So "everything
comes back at 1.0" is not a separate failure mode, it is the noise floor
hitting the ceiling. At `n_d = n_channels` the matrix is invertible again and
the values are **not** 1.0 — they are 0.75 for four channels — but the fit has
no residual degrees of freedom and means nothing either. Both are refused:
the guard is `n_segments > n_channels`, and it names the numbers and a working
`nperseg` in the message.

Deliberately not added: band-integrated scores and "most redundant sensor"
rankings. Collapsing frequency needs a band the user must choose, and it
destroys the thing that makes coherence diagnostic. The app derives its own
summaries on top of the curves.

### 4.2 Mutual information — done 2026-09-02

Shipped in `statistics.py` rather than `multisensor.py`: it is a pairwise
estimator over samples, next to `joint_histogram`, not a matrix operation.

```
mutual_information(x, y, k=3, lags=0, standardize=True, jitter=True, seed=0)
    -> float, or ndarray (len(lags),) — nats

mi_significance(x, y, k=3, lags=0, n_surrogates=199, method="shift",
                standardize=True, jitter=True, seed=0)
    -> dict: mi, lag, p_value, null_mean, null_p95, null_distribution,
             n_samples, n_surrogates, k, method, interpretation
```

Kraskov-Stögbauer-Grassberger algorithm 1, max-norm neighbours via
`scipy.spatial.cKDTree`. The two positions the scoping asked for, both stated
in the docstrings:

- **Lags.** MI has no frequency decomposition, so a lagged relationship is
  invisible unless scanned for; at lag 0 a pure half-period delay can read as
  independence. Every lag in a scan uses the *same* sample count (the window
  is shrunk once, by the full lag span), because the estimator's bias depends
  on N and a curve whose N varies along it is not comparable with itself. The
  max over a scan of L lags is still a biased estimate of the max, and the
  docstring says so.
- **Significance.** Time-shifted surrogates by default, not permutation.
  Permuting destroys autocorrelation as well as coupling, so it builds the
  null for "y is white noise" rather than "y is unrelated to x"; the KSG floor
  rises with autocorrelation, so a permutation null declares dependence far
  too readily on real records. `test_shift_surrogates_keep_the_autocorrelation`
  pins that the two nulls differ in the expected direction.

The estimator's floor for independent signals is a positive number that
depends on N and k (measured ~0.005 nats at N = 2000, k = 3), so a bare MI
value has no reference point. Both docstrings say plainly that MI without a
surrogate test establishes nothing.

### 4.3 The design question — settled 2026-09-02

Per-frequency curves, no scalars. Settled before implementing, as this item
asked. See the "deliberately not added" paragraph in §4.1.

### 4.4 The single-segment trap in ordinary `coherence` — done 2026-09-02

`spectral.coherence` and `multisensor.coherence_matrix` now refuse fewer than
two Welch segments and warn below `min_segments` (default 8). The refusal is
unconditional because there is nothing to salvage: within one segment
`|Gxy|² = Gxx·Gyy` identically, so the answer is 1.0 at every frequency for
any two signals. `test_single_segment_would_be_exactly_one_everywhere` pins
scipy producing exactly that, next to the raise that prevents it.

The warning quotes the bias floor `1 / n_d` rather than just complaining about
the count, since that floor is the number a caller has to read the result
against.

---

## 5. Normality assessment — done 2026-09-02

Not previously in this list. Added to `statistics.py`:

```
qq_normal(x, line="ols"|"quartile")
    -> (theoretical (N,), ordered (N,), slope, intercept)

normality(x, shapiro_max_n=5000, large_n=5000, seed=0)
    -> dict: n, summary, and one sub-dict per indicator, each with an
             'interpretation' string — skewness, excess_kurtosis,
             dagostino_k2, jarque_bera, anderson_darling, shapiro_wilk
```

The reason it is worth having rather than four scipy calls: **at realistic
record lengths every normality test rejects, and that is a property of the
test.** Sampling scatter shrinks as 1/sqrt(N), so at 20 480 samples a skew of
0.03 is detected with certainty and p < 1e-16 is the expected result for real
data. `normality` therefore reports `n`, marks every p-value
`reliable=False` above `large_n`, and says so in the returned structure rather
than only in the docstring. Shapiro-Wilk is subsampled to 5000 points with
`subsampled` and `n_used` reported. Excess kurtosis is labelled *excess* in
its own interpretation string, because reading it as 3-for-normal is the
common failure.

Also fixed here: `spectral.cross_correlation`'s docstring said a positive peak
means "y leads x", which is backwards. The formula `CCF[k] = Σ x[n]·y[n+k]`
was always right; with `y[n] = x[n-d]` the peak sits at `k = +d`, so **x**
leads y. Measured, corrected, and pinned by
`test_cross_correlation_lag_sign`. `mutual_information`'s lag axis follows the
same pairing. Anything built on the old sentence has its sign inverted.

---

## 6. Deferred, with reasons

### 6.1 `plots.py` wrappers for the new outputs
`plot_qq`, and a partial-coherence matrix plot. Every other estimator in the
library has a thin plotting wrapper and these do not, so the set is
inconsistent. Left out because the app renders these itself and a wrapper
written without a consumer tends to fix the wrong presentation choices.

### 6.2 Bias-floor correction on the coherence estimates
The floor `q / n_d` is quoted in the docstrings and the warnings but never
subtracted. Subtracting it would make the curves look cleaner and would be
wrong in a familiar way: the floor is an expectation, not a per-frequency
value, so removing it turns a known upward bias into an unknown two-sided one
and puts negative coherences on the plot. Only worth revisiting alongside a
proper confidence interval on the estimate, which is the thing actually
wanted.

### 6.3 Conditional mutual information
`mi_significance` answers "are these two related"; it cannot answer "are they
related other than through channel 3", which is what `partial_coherence` does
in the linear case. Conditional MI needs a higher-dimensional KSG variant and
many more samples for the same variance. Do it only if a case turns up where
the linear picture is visibly wrong.

### 6.4 Multiple coherence against a *subset* of channels
Currently every other channel is used as a predictor. Choosing the subset is
how you find out *which* sensors make one redundant, but the number of subsets
is exponential and the answer is a model-selection problem, not a transform.
`partial_coherence` covers the single-mediator case, which is most of the
value. Partly answered since: `error_spectrum` (§7) takes an explicit target
and an explicit predictor set, so a named subset can now be scored directly —
what is still missing is the search over subsets, which is the expensive half.

---

## 7. The error spectrum — landed 2026-09-10

`error_spectrum` in `frf.py`, exported from the package, documented at
`docs/api/frf.md` (a page the FRF module did not have at all before). Source:
`P:\Postdoc_2025_2027\2026_09_September\coherence_error_spectrum.pdf`.

It returns `(1 - gamma^2) S_dd` — the residual spectrum of the best linear
predictor of one channel from the others — plus the coherent/error split of the
target's power, the bias floor, the filters, the input conditioning, and the
integrated unexplained variance. `frf_mimo` gained an `output_psd` key to feed
it; nothing else changed.

**Most of the theory was already here.** The optimal filter is H1, which `frf`
returns. The multiple coherence against a designated target is what `frf_mimo`
already computes. Partial coherence, the `q / n_d` bias floor and the
per-frequency condition number were all in place. What was missing was the one
thing the source note leads with: report the residual in the target's units,
not the dimensionless coherence.

Tests are in `test_frf_response.py`, nine of them, on `generate_shear4` in
`_testing.py` — a four-storey shear chain under JONSWAP loading with a
deliberately unmeasured second force, the only fixture in the suite where the
true unexplainable share is known by construction. Its natural frequencies are
pinned so the numbers below cannot drift silently. `docs/gen_images.py` draws
`frf_prediction.png` from the same generator, so the figure on the docs page
and the test numbers cannot disagree.

Measured, not asserted:

- Theory matches a **held-out** residual within 15% for one and three
  predictors. In-sample is optimistic by construction and is not what is
  checked.
- A single accelerometer averages coherence 0.50 above the wave band, which
  looks like a broken model, and still accounts for 99.8% of the target's
  variance. That is the whole argument for the function.
- The `q = 1` case reduces to ordinary coherence to 1e-10.

### 7.1 The one-sided nperseg advice — fixed 2026-09-10, see §8

`_check_welch_segments` in `spectral.py` warns only about **too few segments**,
and every fix it suggests is "shorten nperseg". Three functions route through
it, `coherence` among them.

The opposite failure is worse and goes unmentioned. On a **noise-free**
single-input version of the shear chain, where the true error spectrum is
exactly zero, the apparent unexplained variance runs 3.8e-5, 1.6e-4, 6.3e-4 and
2.4e-3 of the target as `B_e / B_r` goes 0.06, 0.12, 0.24, 0.49 — a clean
square law and entirely an artefact of segment length. Because the spurious
part is proportional to `S_dd` it is worst where the target is strong, so it
can swamp the genuine residual. Shortening `nperseg` to win averages is
therefore sometimes exactly the wrong move.

`error_spectrum`'s docstring states this, but the advice one module over still
contradicts it. The fix is a resolution-bandwidth check — `B_e ≈ 1.5 fs /
nperseg` against the half-power bandwidth of the sharpest peak, with the
classical `B_e / B_r <= 1/4` — and a message that names both directions. It
needs the sharpest peak's bandwidth, which `peaks.peak_bandwidth` already
computes, so the pieces exist.

### 7.2 The predictor path — open

`error_spectrum` returns `H` but nothing applies it. Reconstructing `d_hat(t)`
and taking a real residual currently lives in `_reconstruct` in the test file,
where two traps are recorded and worth promoting to the library: interpolate
the filter's real and imaginary parts separately, since interpolating magnitude
and phase corrupts it wherever the phase wraps; and trim the ends, because
multiplying in frequency is a circular convolution and keeping the wrap-around
inflated `var(e)` by a factor of two while leaving the Welch estimate of the
residual PSD almost unchanged.

That discrepancy is itself a useful diagnostic, which is an argument for
shipping the function rather than leaving everyone to rediscover it.

### 7.3 App integration — done 2026-09-10

Landed as a third mode on the Predictability tab. See `../DSPkit-app/TODO.md`
§8 for the app side. One thing there belongs here: the app's own example file
turns out to be a live demonstration of §7.1, which is the strongest argument
yet for fixing it.

---

---

## 8. The nperseg trade, and a third spectral estimator — done 2026-09-10

Closes §7.1 and adds the one piece of parametric estimation that belongs here.
`spectral.py` gained `resolution_bandwidth`, `segment_advice`, `ar_psd` and
`ar_order_selection`; all four are exported and documented on the Spectral API
page. 306 tests pass, up from 294.

### 8.1 §7.1 is fixed — the guardrail no longer advises one direction

Every message from `_check_welch_segments` used to end at "Fix: shorten
nperseg". That is right for too few averages and actively wrong for the other
failure mode. Both messages now name the cost of shortening and point at
`segment_advice`. Two tests asserted the old string and have been rewritten —
they were encoding the defect, which is worth remembering when one of them
"fails" after a message is improved.

`segment_advice(x, fs, nperseg)` weighs both on the actual record and returns a
verdict: `ok`, `marginal`, `unresolved`, `too_few` or `squeezed`. The last means
the record cannot satisfy both, which is a statement about the *data* rather
than the parameters, and it says how many seconds would.

**Two things had to be got right before it said anything useful.**

Peaks are detected on the dB spectrum with a prominence threshold *and* a
dynamic-range window, because on a linear PSD every noise wiggle is a peak and
the narrowest of those is always about one resolution bandwidth wide — the
first version reported "unresolved" on every record ever, which is a diagnostic
that never says anything. With few averages a PSD is noisy enough that 6 dB of
prominence is nothing: on the 2-DOF fixture at 4 averages the narrowest
"significant" peak was a noise spike at 182 Hz, fifty-odd dB below the modes.
Peaks must also sit within 20 dB of the tallest.

`resolution_bandwidth` uses the window's equivalent noise bandwidth,
`N sum(w²)/(sum w)²`, not the bin spacing `fs/nperseg`, which is finer and
flatters the estimate. Measured: Hann 1.5 bins exactly, Hamming 1.363,
rectangular 1.0, Blackman 1.727, flat-top 3.770, independent of length.

**A caution for anyone reading the numbers.** The 2-DOF fixture's first mode is
at 8.613 Hz with 1.218% damping, so its half-power width is 0.2098 Hz and the
classical `Be/Br <= 1/4` rule needs nperseg near 32768 at fs = 1024 — 32-second
segments. That is not a number anyone guesses, and it is the opposite of what
"shorten nperseg for more averages" would lead you to. It took a wrong hand
derivation of the damping (0.79 Hz, from `c1/(2 m wn)`, which ignores the mode
shape) to make the estimator look broken when it was converging correctly.
Compute modal damping from the state matrix eigenvalues, not from a damper.

### 8.2 `ar_psd` — the third classical estimator

Burg by default, Yule-Walker offered for comparison, order by AIC or BIC.
Deliberately *not* system identification: it returns a spectrum, which is why it
belongs beside `psd` and `blackman_tukey_psd` rather than in omakit. ARX, ARMA
and state-space models stay out — omakit's SSI-COV and p-LSCF already cover
parametric identification for structures, kalmpy owns state-space, and a second
weaker implementation here would give two answers to one question.

What it earns: two tones at 10.0 and 10.8 Hz in noise, 2 s at 100 Hz, 200
samples total. Welch merges them at every usable segment length (one peak, at
10.50, 10.16 and 10.94 Hz for nperseg 200, 128 and 64); AR order 30 returns 9.99
and 10.85 Hz.

**Two bugs are worth recording because neither looks like a bug.**

The Levinson update produced a plausible spectrum while being wrong. Only a
test against a process with *known* coefficients caught it — `test_spectral.py`
now fits an AR(2) with a = [1, -1.6, 0.9] and checks both estimators land on it.

Burg's forward and backward error update aliased: `f[m:n] = fn + k*bn` overwrote
the forward error in place, and the next line still needed the *old* values, but
`fn` was a view. Copies, not views. The fit came out wrong and plausible.

Order selection originally refit from scratch at every candidate order. Both
recursions produce the order-p error on the way to order `max_order`, so one
pass does it: 100 orders on 20480 samples went from seconds to 40 ms. That is
what made the app's AR tab sit there computing.

### 8.3 Left open

- **A causal ARX one-step predictor.** `error_spectrum`'s docstring says the
  coherence bound is what a causal model is judged against, and you still
  cannot compute the thing being judged. Worth doing when that comparison is
  actually wanted; it is a nicer argument than it is a need.
- **AR spectra for several channels share nothing.** Each channel is fitted
  independently, which is right for a spectrum and wrong for a multivariate
  model. If a multichannel AR is ever wanted, that is omakit's ground.
- **`segment_advice` looks at one channel.** The sharpest resonance in an array
  usually appears in more than one, so this is nearly always fine, but a
  sensor at a node of the sharpest mode would not show it.
