# Spectral

Spectral analysis functions: FFT amplitude spectrum, power spectral density, cross-spectral density, coherence, autocorrelation, and cross-correlation.

Three estimators of the same quantity live here, and they fail differently.
`psd` averages periodograms and is the default. `blackman_tukey_psd` transforms
a tapered autocorrelation and trades resolution for variance through the lag
window, explicitly. `ar_psd` fits an all-pole model to the whole record and can
separate two close peaks on a record too short for either of the others — at
the cost of an order that decides what the answer says.

`segment_advice` answers the question the first of those poses. `nperseg` can be
wrong in two opposite directions, and only one of them is obvious.

All estimators use `scipy.signal` under the hood with engineering-friendly defaults (Hann window, density scaling, mean detrending).

![Spectral analysis example](../images/spectral.png)

---

::: dspkit.spectral.fft_spectrum

---

::: dspkit.spectral.psd

---

::: dspkit.spectral.ar_psd

---

::: dspkit.spectral.ar_order_selection

---

::: dspkit.spectral.csd

---

::: dspkit.spectral.coherence

---

::: dspkit.spectral.resolution_bandwidth

---

::: dspkit.spectral.segment_advice

---

::: dspkit.spectral.autocorrelation

---

::: dspkit.spectral.cross_correlation
