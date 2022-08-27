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


# Get patient ID from patient data.
def get_patient_id(data):
    patient_id = None
    for i, l in enumerate(data.split("\n")):
        if i == 0:
            patient_id = l.split(" ")[0]
        else:
            break
    return patient_id


# Get number of recording locations from patient data.
def get_num_locations(data):
    num_locations = None
    for i, l in enumerate(data.split("\n")):
        if i == 0:
            num_locations = int(l.split(" ")[1])
        else:
            break
    return num_locations


# Get frequency from patient data.
def get_frequency(data):
    frequency = None
    for i, l in enumerate(data.split("\n")):
        if i == 0:
            frequency = float(l.split(" ")[2])
        else:
            break
    return frequency


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


# Santize scalar values from Challenge outputs.
def sanitize_scalar_value(x):
    x = (
        str(x).replace('"', "").replace("'", "").strip()
    )  # Remove any quotes or invisible characters.
    if is_finite_number(x) or (is_number(x) and np.isinf(float(x))):
        return float(x)
    else:
        return 0.0


# Save Challenge outputs.
def save_challenge_outputs(filename, patient_id, classes, labels, probabilities):
    # Format Challenge outputs.
    patient_string = "#{}".format(patient_id)
    class_string = ",".join(str(c) for c in classes)
    label_string = ",".join(str(label) for label in labels)
    probabilities_string = ",".join(str(p) for p in probabilities)
    output_string = (
        patient_string
        + "\n"
        + class_string
        + "\n"
        + label_string
        + "\n"
        + probabilities_string
        + "\n"
    )

    # Write the Challenge outputs.
    with open(filename, "w") as f:
        f.write(output_string)


# Load Challenge outputs.
def load_challenge_outputs(filename):
    with open(filename, "r") as f:
        for i, text in enumerate(f):
            if i == 0:
                patient_id = text.replace("#", "").strip()
            elif i == 1:
                classes = tuple(entry.strip() for entry in text.split(","))
            elif i == 2:
                labels = tuple(
                    sanitize_binary_value(entry) for entry in text.split(",")
                )
            elif i == 3:
                probabilities = tuple(
                    sanitize_scalar_value(entry) for entry in text.split(",")
                )
            else:
                break
    return patient_id, classes, labels, probabilities
