import argparse
import os
import shutil

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from DataProcessing.find_and_load_patient_files import (
    find_patient_files,
    load_patient_data,
)
from DataProcessing.label_extraction import get_murmur, get_outcome
from DataProcessing.metadata_extraction import get_features


def stratified_test_vali_split(
    stratified_features: list,
    data_directory: str,
    out_directory: str,
    test_size: float,
    vali_size: float,
    random_state: int,
):
    """Split the data in data_directory into train and test, stratifying over
    the specified columns.

        Parameters
        ----------
        stratified_features : list
            A list of the features over which to stratify the split.
        data_directory : str
            Folder containing the data to split.
        out_directory : str
            Folder to save the split data into.
        test_size : float
            Fraction of the data to place into the test set.
        random_state : int
            Random state used for the data split.

        Returns
        -------
        None
    """

    # Check if out_directory directory exists, otherwise create it.
    if not os.path.exists(out_directory):
        os.makedirs(out_directory)
    else:
        shutil.rmtree(out_directory)
        os.makedirs(out_directory)
    # Get metadata
    patient_files = find_patient_files(data_directory)
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
        complete_pd[stratified_features].astype(str).agg("-".join, axis=1)
    )
    complete_pd_train, complete_pd_test = train_test_split(
        complete_pd,
        test_size=test_size,
        random_state=random_state,
        stratify=complete_pd["stratify_column"],
    )
    vali_split = vali_size / (1 - test_size)
    complete_pd_train, complete_pd_val = train_test_split(
        complete_pd_train,
        test_size=vali_split,
        random_state=random_state + 1,
        stratify=complete_pd_train["stratify_column"],
    )
    with open(os.path.join(out_directory, "split_details.txt"), "w") as text_file:
        text_file.write("This data split is stratified over the following features: \n")
        for feature in stratified_features:
            text_file.write(feature + ", ")
    # Save the files.
    os.makedirs(os.path.join(out_directory, "train_data"))
    os.makedirs(os.path.join(out_directory, "vali_data"))
    os.makedirs(os.path.join(out_directory, "test_data"))
    for f in complete_pd_train["id"]:
        copy_files(
            data_directory,
            f,
            os.path.join(out_directory, "train_data/"),
        )
    for f in complete_pd_val["id"]:
        copy_files(
            data_directory,
            f,
            os.path.join(out_directory, "vali_data/"),
        )
    for f in complete_pd_test["id"]:
        copy_files(
            data_directory,
            f,
            os.path.join(out_directory, "test_data/"),
        )


def copy_files(data_directory: str, ident: str, out_directory: str) -> None:
    """Copy file in data_directory beginning with ident to out_directory

    Parameters
    ----------
    data_directory : str
        Folder containing the data to split.
    ident : str
        Identifier for the file to be copied.
    out_directory : str
        Folder to save the copied file into.

    Returns
    -------
    None
    """

    # Get the list of files in the data folder.
    files = os.listdir(data_directory)
    # Copy all files in data_directory that start with f to out_directory
    for f in files:
        if f.startswith(ident):
            _ = shutil.copy(os.path.join(data_directory, f), out_directory)


if __name__ == "__main__":

    my_parser = argparse.ArgumentParser(prog="StratifiedDataSplit")
    my_parser.add_argument(
        "--data_directory",
        type=str,
        help="The directory containing the data you wish to split.",
        default="physionet.org/files/circor-heart-sound/1.0.3/training_data",
    )
    my_parser.add_argument(
        "--out_directory",
        type=str,
        help="The directory to store the split data.",
        default="data/stratified_data",
    )
    my_parser.add_argument(
        "--vali_size", type=float, default=0.16, help="The size of the test split."
    )
    my_parser.add_argument(
        "--test_size", type=float, default=0.2, help="The size of the test split."
    )
    my_parser.add_argument(
        "--random_state", type=int, default=5678, help="The random state for the split."
    )
    args = my_parser.parse_args()

    stratified_features = ["Normal", "Abnormal", "Absent", "Present", "Unknown"]

    # Create the test split.
    stratified_test_vali_split(stratified_features, **vars(args))
