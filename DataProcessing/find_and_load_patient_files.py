import os

from DataProcessing.helper_code import is_integer


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
