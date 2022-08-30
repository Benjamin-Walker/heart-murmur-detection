import argparse

import numpy as np
import xgboost as xgb
from sklearn.impute import SimpleImputer
from tqdm import tqdm

from DataProcessing.find_and_load_patient_files import (
    find_patient_files,
    load_patient_data,
)
from DataProcessing.helper_code import load_recordings
from DataProcessing.label_extraction import get_murmur
from DataProcessing.XGBoost_features.extract_all_features import extract_all_features
from dbres import calculate_dbres_output
from ModelEvaluation.evaluate_model import evaluate_model


def get_murmurs_features(
    data_directory,
    recalc_dbres_output,
    dbres_output_directory,
    model_binary_present_pth,
    model_binary_unknown_pth,
):
    patient_files = find_patient_files(data_directory)
    num_patient_files = len(patient_files)

    # Extract the features and labels.
    murmur_classes = ["Present", "Unknown", "Absent"]
    num_murmur_classes = len(murmur_classes)

    features = list()
    murmurs = list()
    for i in tqdm(range(num_patient_files)):

        # Load the current patient data and recordings.
        current_patient_data = load_patient_data(patient_files[i])
        current_recordings = load_recordings(data_directory, current_patient_data)
        current_recordings = [r / 32768 for r in current_recordings]

        # Extract features.
        metadata_features, audio_features = extract_all_features(
            current_patient_data, current_recordings
        )
        audio_features_reshaped = audio_features.reshape(1, -1)[0]
        current_features = np.hstack((metadata_features, audio_features_reshaped))
        features.append(current_features)

        # Extract labels and use one-hot encoding.
        current_murmur = np.zeros(num_murmur_classes, dtype=int)
        murmur = get_murmur(current_patient_data)
        if murmur in murmur_classes:
            j = murmur_classes.index(murmur)
            current_murmur[j] = 1
        murmurs.append(current_murmur)

    features = np.vstack(features)
    murmurs = np.vstack(murmurs)

    imputer = SimpleImputer().fit(features)
    features = imputer.transform(features)

    _, spectrogram_outputs = calculate_dbres_output(
        recalc_dbres_output,
        data_directory,
        dbres_output_directory,
        model_binary_present_pth,
        model_binary_unknown_pth,
    )

    features_combined = np.vstack(
        [np.concatenate((f, s)) for f, s in zip(features, spectrogram_outputs)]
    )

    return murmurs, features_combined


def train_xgboost_integration(
    train_data_directory,
    dbres_output_directory,
    model_binary_present_pth,
    model_binary_unknown_pth,
):

    murmurs, features_combined = get_murmurs_features(
        train_data_directory,
        True,
        dbres_output_directory,
        model_binary_present_pth,
        model_binary_unknown_pth,
    )

    murmur_classifier = xgb.XGBClassifier()
    murmur_classifier.fit(
        features_combined, murmurs, sample_weight=-2 * np.argmax(murmurs, axis=1) + 5
    )

    return murmur_classifier


def text_xgboost_integration(
    murmur_classifier,
    test_data_directory,
    dbres_output_directory,
    model_binary_present_pth,
    model_binary_unknown_pth,
):

    _, features_combined = get_murmurs_features(
        test_data_directory,
        True,
        dbres_output_directory,
        model_binary_present_pth,
        model_binary_unknown_pth,
    )

    murmur_probabilities = murmur_classifier.predict_proba(features_combined)
    murmur_outputs = np.zeros(murmur_probabilities.shape, dtype=np.int_)
    idx = np.argmax(murmur_probabilities, axis=1)
    for i in range(len(murmur_outputs)):
        murmur_outputs[i][idx[i]] = 1

    return murmur_probabilities, murmur_outputs


def calculate_xgboost_integration_scores(
    train_data_directory,
    test_data_directory,
    dbres_output_directory,
    model_binary_present_pth,
    model_binary_unknown_pth,
):
    murmur_classifier = train_xgboost_integration(
        train_data_directory,
        dbres_output_directory,
        model_binary_present_pth,
        model_binary_unknown_pth,
    )
    murmur_probabilities, murmur_outputs = text_xgboost_integration(
        murmur_classifier,
        test_data_directory,
        dbres_output_directory,
        model_binary_present_pth,
        model_binary_unknown_pth,
    )
    scores = evaluate_model(test_data_directory, murmur_probabilities, murmur_outputs)
    print(f"{scores}")

    return scores


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="XGBoostIntegration")
    parser.add_argument(
        "--train_data_directory",
        type=str,
        help="The directory of the training data.",
        default="data/stratified_data/train_data",
    )
    parser.add_argument(
        "--test_data_directory",
        type=str,
        help="The directory of the test data.",
        default="data/stratified_data/test_data",
    )
    parser.add_argument(
        "--dbres_output_directory",
        type=str,
        help="The directory in which DBRes's output is saved.",
        default="data/dbres_outputs",
    )
    parser.add_argument(
        "--model_binary_present_pth",
        type=str,
        help="The path of binary ResNet trained to classify present vs not present.",
        default="models/model_BinaryPresent.pth",
    )
    parser.add_argument(
        "--model_binary_unknown_pth",
        type=str,
        help="The path of binary ResNet trained to classify unknown vs not unknown.",
        default="models/model_BinaryUnknown.pth",
    )

    args = parser.parse_args()

    scores = calculate_xgboost_integration_scores(**vars(args))
