import argparse
import os
import numpy as np
import xgboost as xgb
from sklearn.impute import SimpleImputer
from tqdm import tqdm

from DataProcessing.find_and_load_patient_files import (
    find_patient_files,
    load_patient_data,
)
from DataProcessing.helper_code import load_recordings
from DataProcessing.label_extraction import get_murmur, get_outcome
from DataProcessing.XGBoost_features.extract_all_features import extract_all_features
from dbres import calculate_dbres_output
from ModelEvaluation.evaluate_model import evaluate_model


def get_murmurs_features(
    model_name,
    data_directory,
    recalc_dbres_output,
    dbres_output_directory,
    model_binary_pth,
    model_binary_present_pth,
    model_binary_unknown_pth,
    model_type,
    bayesian
):
    patient_files = find_patient_files(data_directory)
    num_patient_files = len(patient_files)

    # Extract the features and labels.
    if model_type == "murmur":
        label_classes = ["Present", "Unknown", "Absent"]
    elif model_type == "murmur_binary":
        label_classes = ["Present", "Absent"]
    elif model_type == "outcome_binary":
        label_classes = ["Abnormal", "Normal"]
    num_label_classes = len(label_classes)

    features = list()
    labels = list()
    for i in range(num_patient_files):

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
        current_label = np.zeros(num_label_classes, dtype=int)
        if model_type == "murmur":
            label = get_murmur(current_patient_data)
        elif model_type == "murmur_binary":
            label = get_murmur(current_patient_data)
            if label == "Unknown":
                label = "Present"
        elif model_type == "outcome_binary":
            label = get_outcome(current_patient_data)
        if label in label_classes:
            j = label_classes.index(label)
            current_label[j] = 1
        labels.append(current_label)

    features = np.vstack(features)
    labels = np.vstack(labels)

    imputer = SimpleImputer().fit(features)
    features = imputer.transform(features)

    _, spectrogram_outputs = calculate_dbres_output(
        model_name,
        recalc_dbres_output,
        data_directory,
        dbres_output_directory,
        model_binary_pth,
        model_binary_present_pth,
        model_binary_unknown_pth,
        bayesian
    )

    features_combined = np.vstack(
        [np.concatenate((f, s)) for f, s in zip(features, spectrogram_outputs)]
    )

    return labels, features_combined


def train_xgboost_integration(
    model_name,
    train_data_directory,
    dbres_output_directory,
    model_binary_pth,
    model_binary_present_pth,
    model_binary_unknown_pth,
    model_type,
    use_weights=False,
    bayesian=True,
):
    
    if os.path.exists(dbres_output_directory):
        recalculated_dbres_output = False
    else:
        recalculated_dbres_output = True
        print(f"DBRes output directory {dbres_output_directory} does not exist. Recalculating DBRes output.")

    murmurs, features_combined = get_murmurs_features(
        model_name,
        train_data_directory,
        recalculated_dbres_output,
        dbres_output_directory,
        model_binary_pth,
        model_binary_present_pth,
        model_binary_unknown_pth,
        model_type=model_type,
        bayesian=bayesian
    )

    if use_weights:
        w_pos = 5
        print(f"Using postivie class sample weight {w_pos}.")
        sample_weights = np.ones(len(murmurs))
        for i in range(len(murmurs)):
            if murmurs[i][0] == 1:
                sample_weights[i] = w_pos
    else:
        print("Not using sample weights.")
        sample_weights = None

    murmur_classifier = xgb.XGBClassifier()
    murmur_classifier.fit(features_combined, murmurs, sample_weight = sample_weights)

    return murmur_classifier


def test_xgboost_integration(
    model_name,
    murmur_classifier,
    test_data_directory,
    dbres_output_directory,
    model_binary_pth,
    model_binary_present_pth,
    model_binary_unknown_pth,
    model_type,
    recordings_file,
    bayesian=True,
):

    if os.path.exists(dbres_output_directory):
        recalculated_dbres_output = False
    else:
        recalculated_dbres_output = True
        print(f"DBRes output directory {dbres_output_directory} does not exist. Recalculating DBRes output.")
    
    # TODO: Add code to load the recordings file.
    _, features_combined = get_murmurs_features(
        model_name,
        test_data_directory,
        recalculated_dbres_output,
        dbres_output_directory,
        model_binary_pth,
        model_binary_present_pth,
        model_binary_unknown_pth,
        model_type=model_type,
        bayesian=bayesian
    )

    murmur_probabilities = murmur_classifier.predict_proba(features_combined)
    murmur_outputs = np.zeros(murmur_probabilities.shape, dtype=np.int_)
    idx = np.argmax(murmur_probabilities, axis=1)
    for i in range(len(murmur_outputs)):
        murmur_outputs[i][idx[i]] = 1

    return murmur_probabilities, murmur_outputs


def calculate_xgboost_integration_scores(
    model_name,
    train_data_directory,
    test_data_directory,
    model_xgb_pth,
    dbres_output_directory,
    model_binary_pth,
    model_binary_present_pth,
    model_binary_unknown_pth,
    output_directory,
    recordings_file,
    use_weights,
    bayesian
):
    
    if (model_binary_present_pth is not None) and (model_binary_unknown_pth is not None):
        model_type = "murmur"
    elif model_binary_pth is not None:
        if "MurmurBinary" in model_binary_pth:
            model_type = "murmur_binary"
        elif "OutcomeBinary" in model_binary_pth:
            model_type = "outcome_binary"
        else:
            raise Exception("No binary murmur or outcome model was provided.")
    else:
        raise Exception("No model was provided.")
    
    print(f"--- Using {model_type} model ---")

    # Train
    if train_data_directory is not None:
        print("Training the model.")
        murmur_classifier = train_xgboost_integration(
            model_name,
            train_data_directory,
            os.path.join(dbres_output_directory, "train"),
            model_binary_pth,
            model_binary_present_pth,
            model_binary_unknown_pth,
            model_type=model_type,
            use_weights=use_weights,
            bayesian=bayesian
        )
        # Save the model.
        if "binary" in model_type:
            model_path = ("/").join(model_binary_pth.split("/")[:-1]) + f"/XGB_{model_type}.json"
        else:
            model_path = ("/").join(model_binary_present_pth.split("/")[:-1]) + f"/XGB_{model_type}.json"
        murmur_classifier.save_model(model_path)
        print(f"Model saved to {model_path}")
    else:
        print("No training data was provided. Loading the model.")
        assert model_xgb_pth is not None, "No model or training data were provided."
        murmur_classifier = xgb.XGBClassifier()
        murmur_classifier.load_model(model_xgb_pth)

    # Test
    murmur_probabilities, murmur_outputs = test_xgboost_integration(
        model_name,
        murmur_classifier,
        test_data_directory,
        os.path.join(dbres_output_directory, "test"),
        model_binary_pth,
        model_binary_present_pth,
        model_binary_unknown_pth,
        model_type=model_type,
        recordings_file = recordings_file,
        bayesian=bayesian
    )
    
    if (model_binary_present_pth is not None) and (model_binary_unknown_pth is not None):
        model_type = "murmur"
    elif model_binary_pth is not None:
        if "MurmurBinary" in model_binary_pth:
            model_type = "murmur_binary"
        elif "OutcomeBinary" in model_binary_pth:
            model_type = "outcome_binary"
        else:
            raise Exception("No binary murmur or outcome model was provided.")
    else:
        raise Exception("No model was provided.")
    print(f"--- Evaluating {model_type} model ---")
    scores = evaluate_model(test_data_directory, murmur_probabilities, murmur_outputs, model_type=model_type, recordings_file = recordings_file, output_directory=output_directory)
    print("---- XGBoost Integration Scores ----")
    print(f"{scores}")
    with open(os.path.join(output_directory, "DBRes_score.npy"), "w") as text_file:
        text_file.write(scores)

    return scores


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="XGBoostIntegration")
    parser.add_argument(
        "--model_name",
        type=str,
        help="The ResNet to train. Current options are resnet50 or resnet50dropout.",
        choices=["resnet50", "resnet50dropout"],
        default="resnet50dropout",
    )
    parser.add_argument(
        "--train_data_directory",
        type=str,
        help="The directory of the training data.",
        default=None,
    )
    parser.add_argument(
        "--test_data_directory",
        type=str,
        help="The directory of the test data.",
        default=None,
    )
    parser.add_argument(
        "--model_xgb_pth",
        type=str,
        help="The path of the xgb model. Must be set if no training data are provided.",
        default=None,
    )
    parser.add_argument(
        "--dbres_output_directory",
        type=str,
        help="The directory in which DBRes's output will be saved.",
        default="data/dbres_outputs",
    )
    parser.add_argument(
        "--model_binary_pth",
        type=str,
        help="The path of binary ResNet trained to classify present vs not present.",
        default=None,
    )
    parser.add_argument(
        "--model_binary_present_pth",
        type=str,
        help="The path of binary ResNet trained to classify present vs not present.",
        default=None,
    )
    parser.add_argument(
        "--model_binary_unknown_pth",
        type=str,
        help="The path of binary ResNet trained to classify unknown vs not unknown.",
        default=None,
    )
    parser.add_argument(
        "--output_directory",
        type=str,
        help="The directory in which to save DBRes's output.",
        default="data/dbres_outputs",
    )
    parser.add_argument(
        "--recordings_file",
        type=str,
        help="The path to a recordings file.",
        default="",
    )
    parser.add_argument(
        "--use_weights",
        type=bool,
        help="Whether to use weights in the training data.",
        default=False,
    )
    parser.add_argument(
        '--disable-bayesian', 
        dest='bayesian', 
        action='store_false', 
        default=True,
        help='Disable Bayesian features (default: Bayesian is enabled)'
    )

    args = parser.parse_args()

    print("---------------- Starting xgboost_integration.py for training ----------------")
    print(f"---------------- Using data from {args.train_data_directory}")

    scores = calculate_xgboost_integration_scores(**vars(args))
