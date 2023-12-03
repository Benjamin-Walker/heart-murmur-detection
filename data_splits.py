import argparse
import os
import shutil

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from tqdm import tqdm

from DataProcessing.find_and_load_patient_files import (
    find_patient_files,
    load_patient_data,
)
from DataProcessing.label_extraction import get_murmur, get_outcome
from DataProcessing.XGBoost_features.metadata import get_metadata


def stratified_test_vali_split(
    stratified_features: list,
    data_directory: str,
    stratified_directory: str,
    test_size: float,
    vali_size: float,
    random_states: list = [42],
    cv: bool = False,
    n_splits: int = 10,
    stratified_cv: bool = False,
):
    # Check if stratified_directory directory exists, otherwise create it.
    if not os.path.exists(stratified_directory):
        os.makedirs(stratified_directory)

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
        current_features = get_metadata(current_patient_data)
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
    complete_pd["stratify_column"] = (
        complete_pd[stratified_features].astype(str).agg("-".join, axis=1)
    )

    # Split data
    complete_pd_train_list = list()
    complete_pd_val_list = list()
    complete_pd_test_list = list()
    cnums = list()
    if cv:
        if stratified_cv:
            print("Performing stratified cross-validation")
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            for i, (train_index, test_index) in enumerate(
                skf.split(complete_pd, complete_pd["stratify_column"])
            ):
                cnums.append(f"split_{i}")
                complete_pd_train, complete_pd_test = complete_pd.iloc[train_index], complete_pd.iloc[test_index]
                vali_split = vali_size / (1 - test_size)
                complete_pd_train, complete_pd_val = train_test_split(
                    complete_pd_train,
                    test_size=vali_split,
                    random_state=42,
                    stratify=complete_pd_train["stratify_column"],
                )
                complete_pd_train_list.append(complete_pd_train)
                complete_pd_val_list.append(complete_pd_val)
                complete_pd_test_list.append(complete_pd_test)
        else:
            print("Performing random cross-validation")
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            for i, (train_index, test_index) in enumerate(
                kf.split(complete_pd)
            ):
                cnums.append(f"split_{i}")
                complete_pd_train, complete_pd_test = complete_pd.iloc[train_index], complete_pd.iloc[test_index]
                vali_split = vali_size / (1 - test_size)
                complete_pd_train, complete_pd_val = train_test_split(
                    complete_pd_train,
                    test_size=vali_split,
                    random_state=42,
                )
                complete_pd_train_list.append(complete_pd_train)
                complete_pd_val_list.append(complete_pd_val)
                complete_pd_test_list.append(complete_pd_test)
    else:
        print("Performing statified split")
        for random_state in random_states:
            cnums.append(f"seed_{random_state}")
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
            complete_pd_train_list.append(complete_pd_train)
            complete_pd_val_list.append(complete_pd_val)
            complete_pd_test_list.append(complete_pd_test)

    # Save the files.
    for cnum, complete_pd_train, complete_pd_val, complete_pd_test in zip(
        cnums, complete_pd_train_list, complete_pd_val_list, complete_pd_test_list
    ):
        print(f"Saving split {cnum} with cv {cv} from {len(cnums)} splits...")
        if cv:
            save_folder = os.path.join(stratified_directory, f"cv_{cv}_stratified_{stratified_cv}", cnum)
        else:
            save_folder = os.path.join(stratified_directory, f"cv_{cv}", cnum)
        os.makedirs(os.path.join(save_folder, "train_data"))
        os.makedirs(os.path.join(save_folder, "vali_data"))
        os.makedirs(os.path.join(save_folder, "test_data"))
        with open(os.path.join(save_folder, "split_details.txt"), "w") as text_file:
            text_file.write("This data split is stratified over the following features: \n")
            for feature in stratified_features:
                text_file.write(feature + ", ")
        for f in complete_pd_train["id"]:
            copy_files(
                data_directory,
                f,
                os.path.join(save_folder, "train_data/"),
            )
        for f in complete_pd_val["id"]:
            copy_files(
                data_directory,
                f,
                os.path.join(save_folder, "vali_data/"),
            )
        for f in complete_pd_test["id"]:
            copy_files(
                data_directory,
                f,
                os.path.join(save_folder, "test_data/"),
            )


def copy_files(data_directory: str, ident: str, stratified_directory: str) -> None:
    # Get the list of files in the data folder.
    files = os.listdir(data_directory)
    # Copy all files in data_directory that start with f to stratified_directory
    for f in files:
        if f.startswith(ident):
            _ = shutil.copy(os.path.join(data_directory, f), stratified_directory)


if __name__ == "__main__":

    print("---------------- Starting data_splits.py to split the data ----------------")

    parser = argparse.ArgumentParser(prog="StratifiedDataSplit")
    parser.add_argument(
        "--data_directory",
        type=str,
        help="The directory containing the data you wish to split.",
        default="physionet.org/files/circor-heart-sound/1.0.3/training_data",
    )
    parser.add_argument(
        "--stratified_directory",
        type=str,
        help="The directory to store the split data.",
        default="data/a_splits",
    )
    parser.add_argument(
        "--vali_size", type=float, default=0.16, help="The size of the test split."
    )
    parser.add_argument(
        "--test_size", type=float, default=0.2, help="The size of the test split."
    )
    parser.add_argument(
        "--cv", type=bool, default=False, help="Whether to run cv."
    )
    parser.add_argument(
        "--stratified_cv", type=bool, default=False, help="Whether to run cv."
    )
    args = parser.parse_args()

    stratified_features = ["Normal", "Abnormal", "Absent", "Present", "Unknown"]

    # Create the test split.
    stratified_test_vali_split(stratified_features, **vars(args))
