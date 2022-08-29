import librosa
import numpy as np
import scipy.stats as stats


def get_spectral_bandwidth(X):
    af = librosa.feature.spectral_bandwidth(y=X + 0.01, sr=4000)[0]
    return (
        np.max(af),
        np.sum(af),
        np.mean(af),
        np.var(af),
        np.max(np.abs(af)),
        stats.skew(af),
        stats.kurtosis(af),
    )
