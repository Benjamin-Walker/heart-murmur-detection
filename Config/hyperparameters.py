# Config file containing all of the model hyperparameters.

# ResNet parameters
dropout = 0.3
# Learning rate
lr = 3e-5
max_overrun = 10
epochs = 200
batch_size = 128
pretrained = True
# Number of classes for multi class classification
n_classes = 3

# Parameters for generating the log mel spectrograms used during training.
# Architectural constants.
NUM_BANDS = 64  # Frequency bands in input mel-spectrogram patch.
EMBEDDING_SIZE = 128  # Size of embedding layer.
# Hyperparameters used in feature and example generation.
SAMPLE_RATE = 4000 # 8000 for Yaseen, else 4000
STFT_WINDOW_LENGTH_SECONDS = 0.025
STFT_HOP_LENGTH_SECONDS = 0.010
NUM_MEL_BINS = NUM_BANDS
MEL_MIN_HZ = 10
MEL_MAX_HZ = 2000
LOG_OFFSET = 0.01  # Offset used for stabilized log of input mel-spectrogram.
EXAMPLE_WINDOW_SECONDS = 4.0 # 0.5 for Yaseen, else 4.0
EXAMPLE_HOP_SECONDS = 1.0 # 0.01 for Yaseen, else 1.0
