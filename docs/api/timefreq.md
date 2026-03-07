# Time-Frequency

Joint time-frequency representations for non-stationary signals.

| Function | Resolution | Cross-terms | Cost |
|---|---|---|---|
| `stft` | Window-limited | None | O(N log N) |
| `cwt_scalogram` | Adaptive (log-freq) | None | O(N log N) |
| `wigner_ville` | Optimal | Yes | O(N²) |
| `smoothed_pseudo_wv` | Tunable | Reduced | O(N²) |

!!! warning "WVD/SPWVD computation time"
    Both Wigner-Ville functions are O(N²). For signals longer than ~2048 samples a `UserWarning` is raised. Use `warn_above` to adjust the threshold.

---

## STFT

::: dspkit.timefreq.stft

---

## CWT Scalogram

::: dspkit.timefreq.cwt_scalogram

---

## Wigner-Ville Distribution

::: dspkit.timefreq.wigner_ville

---

## Smoothed Pseudo WVD

::: dspkit.timefreq.smoothed_pseudo_wv
