"""Compute log mel spectrograms."""

import numpy as np
import resampy
import torch

import HumBugDB.LogMelSpecs.mel_features as mel_features
from Config import hyperparameters


def waveform_to_examples(data, sample_rate, return_tensor=True):
    """Converts audio waveform into an array of log mel spectrograms.

    Args:
      data: np.array of either one dimension (mono) or two dimensions
        (multi-channel, with the outer dimension representing channels).
        Each sample is generally expected to lie in the range [-1.0, +1.0],
        although this is not required.
      sample_rate: Sample rate of data.
      return_tensor: Return data as a Pytorch tensor.

    Returns:
      3-D np.array of shape [num_examples, num_frames, num_bands] which represents
      a sequence of examples, each of which contains a patch of log mel
      spectrogram, covering num_frames frames of audio and num_bands mel frequency
      bands, where the frame length is hyperparameters.STFT_HOP_LENGTH_SECONDS.

    """

    # Convert to mono.
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    # Resample.
    if sample_rate != hyperparameters.SAMPLE_RATE:
        data = resampy.resample(data, sample_rate, hyperparameters.SAMPLE_RATE)

    # Compute log mel spectrogram features.
    log_mel = mel_features.log_mel_spectrogram(
        data,
        audio_sample_rate=hyperparameters.SAMPLE_RATE,
        log_offset=hyperparameters.LOG_OFFSET,
        window_length_secs=hyperparameters.STFT_WINDOW_LENGTH_SECONDS,
        hop_length_secs=hyperparameters.STFT_HOP_LENGTH_SECONDS,
        num_mel_bins=hyperparameters.NUM_MEL_BINS,
        lower_edge_hertz=hyperparameters.MEL_MIN_HZ,
        upper_edge_hertz=hyperparameters.MEL_MAX_HZ,
    )

    # Frame features into examples.
    features_sample_rate = 1.0 / hyperparameters.STFT_HOP_LENGTH_SECONDS
    example_window_length = int(
        round(hyperparameters.EXAMPLE_WINDOW_SECONDS * features_sample_rate)
    )
    example_hop_length = int(
        round(hyperparameters.EXAMPLE_HOP_SECONDS * features_sample_rate)
    )
    log_mel_examples = mel_features.frame(
        log_mel, window_length=example_window_length, hop_length=example_hop_length
    )

    if return_tensor:
        log_mel_examples = torch.tensor(log_mel_examples, requires_grad=True)[
            :, None, :, :
        ].float()

    return log_mel_examples
