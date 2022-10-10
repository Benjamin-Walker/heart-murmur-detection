import numpy as np

from DataProcessing.helper_code import compare_strings, sanitize_binary_value


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


# Extract features from the data.
def get_metadata(data):

    # Extract the age group and replace with the (approximate) number of months
    # for the middle of the age group.
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
