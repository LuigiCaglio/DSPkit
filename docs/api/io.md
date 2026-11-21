# File I/O

Reading measurement files into channels, a sample rate and channel names.

```python
rec = dsp.load_mat("ambient_20190101_000000.mat")
rec.data            # (n_channels, n_samples)
rec.fs              # Hz, or None if the file does not say
rec.channel_names   # one per channel
rec.sources         # which variable supplied each of the above
```

---

::: dspkit.io.load_mat

---

::: dspkit.io.MatRecord
