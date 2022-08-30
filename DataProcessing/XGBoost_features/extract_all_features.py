import librosa
import numpy as np

from DataProcessing.helper_code import compare_strings, get_locations
from DataProcessing.XGBoost_features.generic_frequency_audio_features import (
    get_generic_frequency_audio_features,
)
from DataProcessing.XGBoost_features.generic_time_audio_features import (
    get_generic_time_audio_features,
)
from DataProcessing.XGBoost_features.metadata import get_metadata
from DataProcessing.XGBoost_features.spectral_bandwidth import get_spectral_bandwidth
from DataProcessing.XGBoost_features.spectral_centroid import get_spectral_centroid
from DataProcessing.XGBoost_features.spectral_rolloff import get_spectral_rolloff


# Extract all features from the data.
def extract_all_features(data, recordings):

    metadata = get_metadata(data)

    # Extract recording locations and data. Identify when a location is present,
    # and compute the mean, variance, and skewness of each recording. If there
    # are multiple recordings for one location, then extract features from the last recording.
    locations = get_locations(data)

    recording_locations = ["AV", "MV", "PV", "TV", "PhC"]
    num_recording_locations = len(recording_locations)
    audio_features = np.zeros((num_recording_locations, 44), dtype=float)
    num_locations = len(locations)
    num_recordings = len(recordings)
    recording_length = 0.0
    assert num_locations == num_recordings
    for i in range(num_locations):
        for j in range(num_recording_locations):
            if (
                compare_strings(locations[i], recording_locations[j])
                and np.size(recordings[i]) > 0
            ):
                X = recordings[i]
                recording_length += len(X) / 4000
                audio_features[j, 0] = 1
                audio_features[j, 1:15] = get_generic_time_audio_features(X)
                audio_features[j, 15:22] = get_generic_frequency_audio_features(X)
                audio_features[j, 22:29] = get_spectral_centroid(X)
                audio_features[j, 29:36] = get_spectral_rolloff(X)
                audio_features[j, 36:43] = get_spectral_bandwidth(X)
                audio_features[j, 43] = sum(librosa.zero_crossings(X, pad=False))

    metadata_features = np.hstack((metadata, [recording_length], [num_recordings]))
    return np.asarray(metadata_features, dtype=np.float32), np.asarray(
        audio_features, dtype=np.float32
    )
