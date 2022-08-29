import os

import numpy as np
import scipy.io
import scipy.io.wavfile
import scipy.signal


# Check if a variable is a number or represents a number.
def is_number(x):
    try:
        float(x)
        return True
    except (ValueError, TypeError):
        return False


# Check if a variable is an integer or represents an integer.
def is_integer(x):
    if is_number(x):
        return float(x).is_integer()
    else:
        return False


# Check if a variable is a finite number or represents a finite number.
def is_finite_number(x):
    if is_number(x):
        return np.isfinite(float(x))
    else:
        return False


# Compare normalized strings.
def compare_strings(x, y):
    try:
        return str(x).strip().casefold() == str(y).strip().casefold()
    except AttributeError:  # For Python 2.x compatibility
        return str(x).strip().lower() == str(y).strip().lower()


# Load a WAV file.
def load_wav_file(filename):
    frequency, recording = scipy.io.wavfile.read(filename)
    return recording, frequency


# Load recordings.
def load_recordings(data_folder, data, get_frequencies=False):
    num_locations = get_num_locations(data)
    recording_information = data.split("\n")[1 : num_locations + 1]

    recordings = list()
    frequencies = list()
    for i in range(num_locations):
        entries = recording_information[i].split(" ")
        recording_file = entries[2]
        filename = os.path.join(data_folder, recording_file)
        recording, frequency = load_wav_file(filename)
        recordings.append(recording)
        frequencies.append(frequency)

    if get_frequencies:
        return recordings, frequencies
    else:
        return recordings


# Get number of recording locations from patient data.
def get_num_locations(data):
    num_locations = None
    for i, l in enumerate(data.split("\n")):
        if i == 0:
            num_locations = int(l.split(" ")[1])
        else:
            break
    return num_locations


# Get recording locations from patient data.
def get_locations(data):
    num_locations = get_num_locations(data)
    locations = list()
    for i, text in enumerate(data.split("\n")):
        entries = text.split(" ")
        if i == 0:
            pass
        elif 1 <= i <= num_locations:
            locations.append(entries[0])
        else:
            break
    return locations


# Sanitize binary values from Challenge outputs.
def sanitize_binary_value(x):
    x = (
        str(x).replace('"', "").replace("'", "").strip()
    )  # Remove any quotes or invisible characters.
    if (is_finite_number(x) and float(x) == 1) or (x in ("True", "true", "T", "t")):
        return 1
    else:
        return 0
