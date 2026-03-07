"""
DSPkit — DSP toolkit for structural health monitoring.
"""

from dspkit.spectral import (
    fft_spectrum,
    psd,
    csd,
    coherence,
    autocorrelation,
    cross_correlation,
)
from dspkit.filters import (
    lowpass,
    highpass,
    bandpass,
    bandstop,
    notch,
    decimate,
)
from dspkit.utils import (
    detrend,
    rms,
    peak,
    crest_factor,
    integrate,
    differentiate,
)
from dspkit.timefreq import (
    stft,
    cwt_scalogram,
    wigner_ville,
    smoothed_pseudo_wv,
)
from dspkit.instantaneous import (
    analytic_signal,
    hilbert_envelope,
    instantaneous_phase,
    instantaneous_freq,
    hilbert_attributes,
)
from dspkit.emd import (
    emd,
    hht,
    hht_marginal_spectrum,
)
from dspkit.plots import (
    plot_signal,
    plot_fft,
    plot_psd,
    plot_csd,
    plot_coherence,
    plot_autocorrelation,
    plot_cross_correlation,
    plot_spectrogram,
    plot_scalogram,
    plot_wvd,
)

__all__ = [
    # spectral
    "fft_spectrum",
    "psd",
    "csd",
    "coherence",
    "autocorrelation",
    "cross_correlation",
    # filters
    "lowpass",
    "highpass",
    "bandpass",
    "bandstop",
    "notch",
    "decimate",
    # utils
    "detrend",
    "rms",
    "peak",
    "crest_factor",
    "integrate",
    "differentiate",
    # timefreq
    "stft",
    "cwt_scalogram",
    "wigner_ville",
    "smoothed_pseudo_wv",
    # instantaneous
    "analytic_signal",
    "hilbert_envelope",
    "instantaneous_phase",
    "instantaneous_freq",
    "hilbert_attributes",
    # emd / hht
    "emd",
    "hht",
    "hht_marginal_spectrum",
    # plots
    "plot_signal",
    "plot_fft",
    "plot_psd",
    "plot_csd",
    "plot_coherence",
    "plot_autocorrelation",
    "plot_cross_correlation",
    "plot_spectrogram",
    "plot_scalogram",
    "plot_wvd",
]
