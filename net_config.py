dropout = 0.3

# Learning rate:
lr = 0.00003  # Learning rate may need adjusting for multiclass VGGish.

max_overrun = 10
epochs = 200
batch_size = 128  # Increased batch size for VGGish (DEBUG)
pretrained = True

# Settings for multi-class classification with 8 species for species_classification.ipynb
n_classes = 3
