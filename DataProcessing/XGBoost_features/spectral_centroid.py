import librosa
import numpy as np
import scipy.stats as stats


def get_spectral_centroid(X):
    af = librosa.feature.spectral_centroid(y=X, sr=4000)[0]
    return (
        np.max(af),
        np.sum(af),
        np.mean(af),
        np.var(af),
        np.max(np.abs(af)),
        stats.skew(af),
        stats.kurtosis(af),
    )
