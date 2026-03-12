"""
DSPkit — DSP toolkit for structural health monitoring.
"""

__version__ = "0.1.0"

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
from dspkit.peaks import (
    find_peaks,
    peak_bandwidth,
    find_harmonics,
)
from dspkit.indicators import (
    spectral_entropy,
    kurtosis,
    skewness,
    rms_variation,
    frequency_shift,
    energy_variation,
)
from dspkit.multisensor import (
    correlation_matrix,
    coherence_matrix,
    psd_matrix,
)
from dspkit.fdd import (
    fdd_svd,
    fdd_peak_picking,
    fdd_mode_shapes,
    efdd_damping,
)
from dspkit.statistics import (
    pdf_estimate,
    histogram,
    joint_histogram,
    covariance_matrix,
    mahalanobis,
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
    plot_peaks,
    plot_singular_values,
    plot_mode_shape,
    plot_pdf,
    plot_joint_histogram,
    plot_correlation_matrix,
    plot_indicators,
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
    # peaks
    "find_peaks",
    "peak_bandwidth",
    "find_harmonics",
    # indicators
    "spectral_entropy",
    "kurtosis",
    "skewness",
    "rms_variation",
    "frequency_shift",
    "energy_variation",
    # multisensor
    "correlation_matrix",
    "coherence_matrix",
    "psd_matrix",
    # fdd
    "fdd_svd",
    "fdd_peak_picking",
    "fdd_mode_shapes",
    "efdd_damping",
    # statistics
    "pdf_estimate",
    "histogram",
    "joint_histogram",
    "covariance_matrix",
    "mahalanobis",
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
    "plot_peaks",
    "plot_singular_values",
    "plot_mode_shape",
    "plot_pdf",
    "plot_joint_histogram",
    "plot_correlation_matrix",
    "plot_indicators",
]
