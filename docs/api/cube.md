# Spectral cube

The cross-spectral matrix of a multichannel record, estimated once and reused. PSD, CSD, coherence, multiple and partial coherence and the FDD singular values are all cheap arithmetic on it, so an interactive tool can switch between them, or between segments of a record, without going back to the time series.

The estimate is scipy's Welch (`scipy.signal.csd`) for the same parameters; each method matches the function it replaces to float precision. Each channel is transformed once, where `psd_matrix` and `coherence_matrix` transform it once per channel pair.

Measured on a 15-channel record of 480,000 samples at 100 Hz (nperseg 4096): `spectral_cube` 1.0 s, then SVD, coherence matrix and PSDs from it 0.3 s; `fdd_svd` alone 26 s and `coherence_matrix` alone 39 s on the same data.

```python
import dspkit as dsp

cube = dsp.spectral_cube(data, fs, nperseg=4096, labels=names)
S, U = cube.svd()                       # FDD
C = cube.coherence_matrix()             # pairwise coherence
g2 = cube.multiple_coherence()

# only the clean spans (e.g. the setups of a roving test);
# no Welch segment crosses a span boundary
setups = dsp.spectral_cube(data, fs, nperseg=4096, spans=[(0, 63600), (79800, 142300)])

# or one cube per setup, combined later
both = dsp.SpectralCube.average([cube_a, cube_b])
```

---

::: dspkit.cube.spectral_cube

---

::: dspkit.cube.SpectralCube
