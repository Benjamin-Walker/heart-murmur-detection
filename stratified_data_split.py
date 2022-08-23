import os
import shutil
import sys

import numpy as np
import pandas as pd
from helper_code import (
    compare_strings,
    find_patient_files,
    get_age,
    get_height,
    get_murmur,
    get_outcome,
    get_pregnancy_status,
    get_sex,
    get_weight,
    is_integer,
    load_patient_data,
)
from sklearn.model_selection import KFold, train_test_split
from team_code import get_features_labels
from tqdm import tqdm


def split_features(
    feature_folder: str, k: int, output_folder: str, random_state: int
) -> None:

    features, murmurs, outcomes = get_features_labels(feature_folder, 0)

    # Get k folds.
    kf = KFold(n_splits=k, random_state=random_state, shuffle=True)

    # Copy files
    fold = 0
    for train_index, vali_index in kf.split(features):

        train_features = features[train_index]
        vali_features = features[vali_index]
        train_murmurs = murmurs[train_index]
        vali_murmurs = murmurs[vali_index]
        train_outcomes = outcomes[train_index]
        vali_outcomes = outcomes[vali_index]

        os.makedirs(output_folder + f"training_data_cv/fold_{fold}/", exist_ok=True)
        os.makedirs(output_folder + f"vali_data_cv/fold_{fold}/", exist_ok=True)

        np.save(
            output_folder + f"training_data_cv/fold_{fold}/features.npy", train_features
        )
        np.save(
            output_folder + f"training_data_cv/fold_{fold}/murmurs.npy", train_murmurs
        )
        np.save(
            output_folder + f"training_data_cv/fold_{fold}/outcomes.npy", train_outcomes
        )
        np.save(output_folder + f"vali_data_cv/fold_{fold}/features.npy", vali_features)
        np.save(output_folder + f"vali_data_cv/fold_{fold}/murmurs.npy", vali_murmurs)
        np.save(output_folder + f"vali_data_cv/fold_{fold}/outcomes.npy", vali_outcomes)
        fold += 1


def stratisfied_test_split(
    data_folder: str,
    output_data_folder: str,
    test_size: float,
    stratify_columns: list,
    random_state: int,
    verbose: int,
):

    # Get metadata
    patient_files = find_patient_files(data_folder)
    num_patient_files = len(patient_files)

    murmur_classes = ["Present", "Unknown", "Absent"]
    num_murmur_classes = len(murmur_classes)
    outcome_classes = ["Abnormal", "Normal"]
    num_outcome_classes = len(outcome_classes)
    features = list()
    murmurs = list()
    outcomes = list()
    for i in tqdm(range(num_patient_files)):

        # Load the current patient data and recordings.
        current_patient_data = load_patient_data(patient_files[i])

        # Extract features.
        current_features = get_features(current_patient_data)
        current_features = np.insert(
            current_features, 0, current_patient_data.split(" ")[0]
        )
        current_features = np.insert(
            current_features, 1, current_patient_data.split(" ")[2][:-3]
        )
        features.append(current_features)

        # Extract labels and use one-hot encoding.
        # Murmur
        current_murmur = np.zeros(num_murmur_classes, dtype=int)
        murmur = get_murmur(current_patient_data)
        if murmur in murmur_classes:
            j = murmur_classes.index(murmur)
            current_murmur[j] = 1
        murmurs.append(current_murmur)
        # Outcome
        current_outcome = np.zeros(num_outcome_classes, dtype=int)
        outcome = get_outcome(current_patient_data)
        if outcome in outcome_classes:
            j = outcome_classes.index(outcome)
            current_outcome[j] = 1
        outcomes.append(current_outcome)

    features = np.vstack(features)
    murmurs = np.vstack(murmurs)
    outcomes = np.vstack(outcomes)

    # Combine dataframes
    features_pd = pd.DataFrame(
        features,
        columns=[
            "id",
            "hz",
            "age",
            "female",
            "male",
            "height",
            "weight",
            "is_pregnant",
        ],
    )
    murmurs_pd = pd.DataFrame(murmurs, columns=murmur_classes)
    outcomes_pd = pd.DataFrame(outcomes, columns=outcome_classes)
    complete_pd = pd.concat([features_pd, murmurs_pd, outcomes_pd], axis=1)
    complete_pd["id"] = complete_pd["id"].astype(int).astype(str)

    # Split data
    complete_pd["stratify_column"] = (
        complete_pd[stratify_columns].astype(str).agg("-".join, axis=1)
    )
    complete_pd_train, complete_pd_test = train_test_split(
        complete_pd,
        test_size=test_size,
        random_state=random_state,
        stratify=complete_pd["stratify_column"],
    )

    # Save the files.
    for f in complete_pd_train["id"]:
        copy_files(
            data_folder,
            f,
            output_data_folder
            + f'balanced_{"_".join(stratify_columns)}/'
            + "train_data/",
        )
    for f in complete_pd_test["id"]:
        copy_files(
            data_folder,
            f,
            output_data_folder
            + f'balanced_{"_".join(stratify_columns)}/'
            + "test_data/",
        )


# Extract features from the data.
def get_features(data):
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


def split_data(
    data_folder: str, k: int, output_folder: str, random_state: int, verbose: int
) -> None:
    """Split data in data_folder based on number in file name into k folds of train and vali data and
    save them into output_folder/training_data_cv/ and output_folder/test_data_cv/.

    Parameters
    ----------
    data_folder : str
        Folder containing the Challenge data.
    k : int
        Number of folds to split the data into.
    output_folder : str
        Folder to save the data folds into.
    random_state : int
        Random seed to use for splitting the data.
    verbose : int
        Level of verbosity. If bigger than 0, print out the progress.

    Returns
    -------
    None
    """

    # Get the list of files in the data folder.
    files = os.listdir(data_folder)

    # Split all strings in files into a list of strings and get unique strings.
    files = [f.split("_")[0] for f in files]
    files = [f.split(".")[0] for f in files]
    files = list(set(files))
    files.sort(key=int)

    # Get k folds.
    kf = KFold(n_splits=k, random_state=random_state, shuffle=True)

    # Copy files
    fold = 0
    for train_index, vali_index in kf.split(files):
        if verbose >= 1:
            print(f"Copying the files for fold {fold}.")

        # Get the training and vali files.
        train_files = [files[i] for i in train_index]
        vali_files = [files[i] for i in vali_index]

        # Save the files.
        for f in train_files:
            copy_files(data_folder, f, output_folder + f"training_data_cv/fold_{fold}/")
        for f in vali_files:
            copy_files(data_folder, f, output_folder + f"vali_data_cv/fold_{fold}/")

        fold += 1


# Copy all files in data_folder that start with f to output_folder
def copy_files(data_folder: str, start_with: str, output_folder: str) -> None:
    """Copy all files in data_folder that start with f to output_folder

    Parameters
    ----------
    data_folder : str
        Folder containing the Challenge data.
    start_with : str
        String to start with.
    output_folder : str
        Folder to save the data folds into.

    Returns
    -------
    None
    """

    # Get the list of files in the data folder.
    files = os.listdir(data_folder)

    # Check if output_folder directory exists, otherwise create it.
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Copy all files in data_folder that start with f to output_folder
    for f in files:
        if f.startswith(start_with):
            _ = shutil.copy(data_folder + f, output_folder)


if __name__ == "__main__":

    # python cv_data.py data k cv_data test_size random_state verbose
    data_folder = sys.argv[1]
    feature_folder = sys.argv[2]
    k_fold = int(sys.argv[3])
    output_data_folder = sys.argv[4]
    output_feature_folder = sys.argv[5]
    test_size = float(sys.argv[6])
    random_state = int(sys.argv[7])

    create_test_split = False
    stratify_columns = ["Normal", "Abnormal", "Absent", "Present", "Unknown"]
    resplit_files = False
    resplit_features = True

    # Change the level of verbosity; helpful for debugging.
    if len(sys.argv) == 8 and is_integer(sys.argv[7]):
        verbose = int(sys.argv[7])
    else:
        verbose = 1

    if create_test_split:
        # Create the test split.
        stratisfied_test_split(
            data_folder,
            output_data_folder,
            test_size,
            stratify_columns,
            random_state,
            verbose,
        )

    if resplit_files:
        split_data(data_folder, k_fold, output_data_folder, random_state, verbose)

    if resplit_features:
        split_features(feature_folder, k_fold, output_feature_folder, random_state)
