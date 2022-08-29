import numpy as np
import scipy.stats as stats


def get_generic_time_audio_features(X):
    return (
        np.min(X),
        np.max(X),
        np.mean(X),
        np.sqrt(np.mean(X**2)),
        np.var(X),
        np.std(X),
        np.mean(X**2),
        np.max(np.abs(X)),
        np.ptp(X),
        np.max(np.abs(X)) / np.sqrt(np.mean(X**2)),
        stats.skew(X),
        stats.kurtosis(X),
        np.sqrt(np.mean(X**2)) / np.mean(X),
        np.max(np.abs(X)) / np.mean(X),
    )
