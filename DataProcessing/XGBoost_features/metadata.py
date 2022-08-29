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


# Find patient data files.
def find_patient_files(data_folder):
    # Find patient files.
    filenames = list()
    for f in sorted(os.listdir(data_folder)):
        root, extension = os.path.splitext(f)
        if not root.startswith(".") and extension == ".txt":
            filename = os.path.join(data_folder, f)
            filenames.append(filename)

    # To help with debugging, sort numerically if the filenames are integers.
    roots = [os.path.split(filename)[1][:-4] for filename in filenames]
    if all(is_integer(root) for root in roots):
        filenames = sorted(
            filenames, key=lambda filename: int(os.path.split(filename)[1][:-4])
        )

    return filenames


# Load patient data as a string.
def load_patient_data(filename):
    with open(filename, "r") as f:
        data = f.read()
    return data


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
    for i, text in enumerate(data.split("\n")):
        if i == 0:
            num_locations = int(text.split(" ")[1])
        else:
            break
    return num_locations


# Get frequency from patient data.
def get_frequency(data):
    frequency = None
    for i, text in enumerate(data.split("\n")):
        if i == 0:
            frequency = float(text.split(" ")[2])
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


# Get age from patient data.
def get_age(data):
    age = None
    for text in data.split("\n"):
        if text.startswith("#Age:"):
            age = text.split(": ")[1].strip()
    return age


# Get sex from patient data.
def get_sex(data):
    sex = None
    for text in data.split("\n"):
        if text.startswith("#Sex:"):
            sex = text.split(": ")[1].strip()
    return sex


# Get height from patient data.
def get_height(data):
    height = None
    for text in data.split("\n"):
        if text.startswith("#Height:"):
            height = float(text.split(": ")[1].strip())
    return height


# Get weight from patient data.
def get_weight(data):
    weight = None
    for text in data.split("\n"):
        if text.startswith("#Weight:"):
            weight = float(text.split(": ")[1].strip())
    return weight


# Get pregnancy status from patient data.
def get_pregnancy_status(data):
    is_pregnant = None
    for text in data.split("\n"):
        if text.startswith("#Pregnancy status:"):
            is_pregnant = bool(sanitize_binary_value(text.split(": ")[1].strip()))
    return is_pregnant


# Sanitize binary values from Challenge outputs.
def sanitize_binary_value(x):
    x = (
        str(x).replace('"', "").replace("'", "").strip()
    )  # Remove any quotes or invisible characters.
    if (is_finite_number(x) and float(x) == 1) or (x in ("True", "true", "T", "t")):
        return 1
    else:
        return 0


# Extract features from the data.
def get_metadata(data):
    # Extract the age group and replace with the (approximate) number of months for the middle of the age group.
    age_group = get_age(data)

    if compare_strings(age_group, "Neonate"):
        age = 0.5
    elif compare_strings(age_group, "Infant"):
        age = 6
    elif compare_strings(age_group, "Child"):
        age = 6 * 12
    elif compare_strings(age_group, "Adolescent"):
        age = 15 * 12
    elif compare_strings(age_group, "Young Adult"):
        age = 20 * 12
    else:
        age = float("nan")

    # Extract sex. Use one-hot encoding.
    sex = get_sex(data)

    sex_features = np.zeros(2, dtype=int)
    if compare_strings(sex, "Female"):
        sex_features[0] = 1
    elif compare_strings(sex, "Male"):
        sex_features[1] = 1

    # Extract height and weight.
    height = get_height(data)
    weight = get_weight(data)

    # Extract pregnancy status.
    is_pregnant = get_pregnancy_status(data)

    features = np.hstack(([age], sex_features, [height], [weight], [is_pregnant]))

    return np.asarray(features, dtype=np.float32)
