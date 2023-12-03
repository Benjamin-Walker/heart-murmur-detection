import argparse
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from DataProcessing.find_and_load_patient_files import (
    find_patient_files,
    load_patient_data,
)
from DataProcessing.helper_code import get_num_locations, load_recordings, load_wav_file
from DataProcessing.net_feature_extractor import load_spectrograms_yaseen, load_spectrograms_yaseen
from HumBugDB.LogMelSpecs.compute_LogMelSpecs import waveform_to_examples
from HumBugDB.runTorch import load_model
from ModelEvaluation.evaluate_model import evaluate_model
from train_resnet import create_model

from Config import hyperparameters


def list_wav_files(data_directory):
    wav_files = []
    subfolder_names = []

    for root, dirs, files in os.walk(data_directory):
        for file in files:
            if file.endswith('.wav'):
                wav_files.append(os.path.join(root, file))
                subfolder_names.append(os.path.basename(root))
    
    return wav_files, subfolder_names


def get_binary_spectrogram_outputs(
    spectrograms,
    model_binary_present,
    model_binary_unknown,
    model_binary
):
    if (model_binary_present is not None) and (model_binary_unknown is not None):
        model_outputs_unknown = []
        model_outputs_present = []
        unknown_probabilities = []
        present_probabilities = []
        for spectrogram in spectrograms:
            output_present = (
                calc_patient_output(model_binary_present, [spectrogram], repeats=30)
                .cpu()
                .numpy()
            )
            output_unknown = (
                calc_patient_output(model_binary_unknown, [spectrogram], repeats=30)
                .cpu()
                .numpy()
            )
            model_outputs_present.append(output_present)
            model_outputs_unknown.append(output_unknown)
            present_probabilities.append(
                np.array([1 - output_present[0], output_present[0]])
            )
            unknown_probabilities.append(
                np.array([1 - output_unknown[0], output_unknown[0]])
            )
        present_probability = np.mean(np.array(present_probabilities), axis=0)
        unknown_probability = np.mean(np.array(unknown_probabilities), axis=0)
        outputs = []
        idx_unknown = (np.mean(np.array(model_outputs_unknown)) > 0.5).astype(float)
        idx_present = (np.mean(np.array(model_outputs_present)) > 0.5).astype(float)
        if idx_present == 0:
            outputs.append(np.array([1, 0, 0]))
        elif idx_unknown == 0:
            outputs.append(np.array([0, 1, 0]))
        else:
            outputs.append(np.array([0, 0, 1]))

        probabilities = [
            present_probability[0],
            present_probability[1] * unknown_probability[0],
            present_probability[1] * unknown_probability[1],
        ]
    elif model_binary is not None:
        model_outputs = []
        probabilities = []
        for spectrogram in spectrograms:
            output = (
                calc_patient_output(model_binary, [spectrogram], repeats=30)
                .cpu()
                .numpy()
            )
            model_outputs.append(output)
            probabilities.append(np.array([1 - output[0], output[0]]))
        probability = np.mean(np.array(probabilities), axis=0)
        outputs = []
        idx = (np.mean(np.array(model_outputs)) > 0.5).astype(float)
        if idx == 0:
            outputs.append(np.array([1, 0]))
        else:
            outputs.append(np.array([0, 1]))
        probabilities = [probability[0], probability[1]]

    return outputs[0].tolist(), probabilities


def calc_patient_output(model, recording_spectrograms, repeats):
    model.eval()
    outputs = []
    for location in recording_spectrograms:
        input = location.repeat(1, 3, 1, 1)
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        input = input.to(device)
        model_out = []
        for _ in range(repeats):
            out = model(input)
            out = out.cpu().detach().unsqueeze(2)
            model_out.append(out)
        model_out = torch.mean(torch.cat(model_out, dim=2), dim=2)
        outputs.append(torch.mean(model_out, axis=0).unsqueeze(dim=0))
    output = torch.mean(torch.cat(outputs), axis=0).detach()
    return output


def calculate_dbres_output(
    model_name,
    recalc_output,
    data_directory,
    output_directory,
    model_binary_pth,
    model_binary_present_pth,
    model_binary_unknown_pth,
    recordings_file: str = "",
    bayesian: bool = True
):

    if recalc_output:

        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # Get model
        model_binary_present = create_model(model_name, 2, bayesian)
        model_binary_unknown = create_model(model_name, 2, bayesian)
        model_binary = create_model(model_name, 2, bayesian)

        # Load model
        if (model_binary_present_pth is not None) and (model_binary_unknown_pth is not None):
            print("Loading multiclass model")
            model_binary_present = load_model(
                model_binary_present_pth, model=model_binary_present[0]
            )
            model_binary_unknown = load_model(
                model_binary_unknown_pth, model=model_binary_unknown[0]
            )
            model_binary = None
        elif model_binary_pth is not None:
            print("Loading binary model")
            model_binary = load_model(model_binary_pth, model=model_binary[0])
            model_binary_present = None
            model_binary_unknown = None
        else:
            raise Exception("No model was provided.")

        # Get data
        murmur_probabilities = list()
        murmur_outputs = list()
        labels = None
        if len(recordings_file) > 0:
            patient_files = pd.read_csv(recordings_file)
        else:
            if "yaseen" in data_directory:
                outcome_classes = [f.name for f in os.scandir(data_directory) if f.is_dir()]
                murmur_classes = outcome_classes
                patient_files, labels = list_wav_files(data_directory)
            else:
                patient_files = find_patient_files(data_directory)

        # Get count of patient files
        num_patient_files = len(patient_files)
        if num_patient_files == 0:
            print(f"No data was provided in {data_directory} for recordings_file {recordings_file}.")
            raise Exception("No data was provided.")
        
        # Get spectrograms and predictions
        for i in tqdm(range(num_patient_files)):
            if len(recordings_file) > 0:
                current_patient_data = patient_files.iloc[i]
                current_recordings = list()
                recording, frequency = load_wav_file(patient_files["path"].iloc[i])
                current_recordings.append(recording)
                sample_rate = frequency
                num_locations = 1
            else:
                if "yaseen" in data_directory:
                    pass
                else:
                    sample_rate=hyperparameters.SAMPLE_RATE
                    current_patient_data = load_patient_data(patient_files[i])
                    current_recordings = load_recordings(data_directory, current_patient_data)
                    num_locations = get_num_locations(current_patient_data)

            # Get spectrograms
            if "yaseen" in data_directory:
                spectrograms = load_spectrograms_yaseen(patient_files[i])
            else:
                current_recordings = [r / 32768 for r in current_recordings]
                spectrograms = list()
                for j in range(num_locations):
                    mel_spec = waveform_to_examples(
                        data=current_recordings[j], sample_rate=sample_rate
                    )
                    spectrograms.append(mel_spec)

            # Get predictions
            murmur_output, murmur_probability = get_binary_spectrogram_outputs(
                spectrograms, model_binary_present, model_binary_unknown, model_binary
            )
            murmur_probabilities.append(murmur_probability)
            murmur_outputs.append(murmur_output)

        # Store
        murmur_probabilities = np.vstack(murmur_probabilities)
        np.save(
            os.path.join(output_directory, "probabilities.npy"),
            murmur_probabilities,
        )
        murmur_outputs = np.vstack(murmur_outputs)
        np.save(os.path.join(output_directory, "outputs.npy"), murmur_outputs)
    else:
        murmur_probabilities = np.load(
            os.path.join(output_directory, "probabilities.npy")
        )
        murmur_outputs = np.load(os.path.join(output_directory, "outputs.npy"))

    return murmur_probabilities, murmur_outputs, labels


def calculate_dbres_scores(
    model_name,
    recalc_output,
    data_directory,
    output_directory,
    model_binary_pth,
    model_binary_present_pth,
    model_binary_unknown_pth,
    recordings_file: str = "",
    bayesian: bool = True
):

    probabilities, outputs, labels = calculate_dbres_output(
        model_name,
        recalc_output,
        data_directory,
        output_directory,
        model_binary_pth,
        model_binary_present_pth,
        model_binary_unknown_pth,
        recordings_file,
        bayesian
    )

    if (model_binary_present_pth is not None) and (model_binary_unknown_pth is not None):
        model_type = "murmur"
    elif model_binary_pth is not None:
        if ("MurmurBinary" in model_binary_pth) or ("Murmur_Binary" in model_binary_pth):
            model_type = "murmur_binary"
        elif ("OutcomeBinary" in model_binary_pth) or ("Outcome_Binary" in model_binary_pth):
            model_type = "outcome_binary"
        else:
            raise Exception("No binary murmur or outcome model was provided.")
    else:
        raise Exception("No model was provided.")
    
    print(f"--- Evaluating {model_type} model ---")
    if "yaseen" in data_directory:
        scores = evaluate_model(data_directory, probabilities, outputs, model_type, recordings_file = recordings_file, output_directory = output_directory, true_labels = labels)
    else:
        scores = evaluate_model(data_directory, probabilities, outputs, model_type, recordings_file = recordings_file, output_directory = output_directory)

    print("--- DBRes scores ---")
    print(f"{scores}")
    with open(os.path.join(output_directory, "DBRes_score.npy"), "w") as text_file:
        text_file.write(scores)

    if model_type == "murmur":
        print(f"--- Evaluating {model_type} model as binary ---")
        # Combine element at position 0 and 1 to get binary output, but keep position 2
        outputs_binary = np.vstack(
            [np.logical_or(outputs[:, 0], outputs[:, 1]), outputs[:, 2]]
        ).T
        probabilities_binary = np.vstack(
            [np.max(probabilities[:, :2], axis=1), probabilities[:, 2]]
        ).T
        scores_binary = evaluate_model(
            data_directory, probabilities_binary, outputs_binary, "murmur_binary", recordings_file = recordings_file, output_directory = output_directory
        )
        print("--- DBRes scores binary ---")
        print(f"{scores_binary}")
        with open(os.path.join(output_directory, "DBRes_score_binary.npy"), "w") as text_file:
            text_file.write(scores_binary)

    return scores


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="DBRes")
    parser.add_argument(
        "--model_name",
        type=str,
        help="The ResNet to train. Current options are resnet50 or resnet50dropout.",
        choices=["resnet50", "resnet50dropout"],
        default="resnet50dropout",
    )
    parser.add_argument(
        "--recalc_output",
        action="store_true",
        help="Whether or not to recalculate the output from DBRes.",
    )
    parser.add_argument(
        "--no-recalc_output", dest="recalc_output", action="store_false"
    )
    parser.set_defaults(recalc_output=True)
    parser.add_argument(
        "--data_directory",
        type=str,
        help="The directory of the data.",
        default="data/stratified_data/test_data",
    )
    parser.add_argument(
        "--output_directory",
        type=str,
        help="The directory in which to save DBRes's output.",
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
        "--recordings_file",
        type=str,
        help="The path to a recordings file.",
        default="",
    )
    parser.add_argument(
        '--disable-bayesian', 
        dest='bayesian', 
        action='store_false', 
        default=True,
        help='Disable Bayesian features (default: Bayesian is enabled)'
    )

    args = parser.parse_args()

    print("---------------- Starting dbres.py for predictions and evaluations ----------------")
    if len(args.recordings_file) > 0:
        print(f"---------------- Using data from {args.recordings_file}")
    else:
        print(f"---------------- Using data from {args.data_directory}")

    scores = calculate_dbres_scores(**vars(args))
