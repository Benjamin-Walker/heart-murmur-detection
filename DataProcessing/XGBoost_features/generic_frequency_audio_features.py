import numpy as np
import scipy.stats as stats
from scipy.fft import fft


def get_generic_frequency_audio_features(X):
    ft = fft(X)
    S = np.abs(ft**2) / len(X)
    return (
        np.max(S),
        np.sum(S),
        np.mean(S),
        np.var(S),
        np.max(np.abs(S)),
        stats.skew(S),
        stats.kurtosis(S),
    )
