# Instantaneous

Hilbert transform and instantaneous signal attributes.

For a narrow-band or IMF signal `x(t)`, the analytic signal `z(t) = x(t) + j·H{x}(t)` gives physically meaningful:

- **Envelope** `A(t) = |z(t)|` — instantaneous amplitude
- **Phase** `φ(t) = ∠z(t)` — instantaneous (unwrapped) phase [rad]
- **Frequency** `f(t) = (1/2π) dφ/dt` — instantaneous frequency [Hz]

!!! tip "Use `hilbert_attributes` for efficiency"
    `hilbert_attributes` computes all three in a single Hilbert call. Use the individual functions only when you need one attribute.

---

::: dspkit.instantaneous.hilbert_attributes

---

::: dspkit.instantaneous.analytic_signal

---

::: dspkit.instantaneous.hilbert_envelope

---

::: dspkit.instantaneous.instantaneous_phase

---

::: dspkit.instantaneous.instantaneous_freq
