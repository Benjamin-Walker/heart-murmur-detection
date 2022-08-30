import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from DataProcessing.find_and_load_patient_files import (
    find_patient_files,
    load_patient_data,
)
from DataProcessing.helper_code import get_num_locations, load_recordings
from HumBugDB.LogMelSpecs.compute_LogMelSpecs import waveform_to_examples
from HumBugDB.runTorch import load_model
from ModelEvaluation.evaluate_model import evaluate_model
from train_resnet import create_model


def get_binary_spectrogram_outputs(
    spectrograms,
    model_binary_present,
    model_binary_unknown,
):
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

    return outputs[0].tolist(), probabilities


def calc_patient_output(model, recording_spectrograms, repeats):
    model.eval()
    outputs = []
    for location in recording_spectrograms:
        input = location.repeat(1, 3, 1, 1)
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
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
    recalc_output,
    data_directory,
    output_directory,
    model_binary_present_pth,
    model_binary_unknown_pth,
):

    if recalc_output:

        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        model_binary_present = create_model("resnet50dropout", 2)
        model_binary_unknown = create_model("resnet50dropout", 2)
        model_binary_present = load_model(
            model_binary_present_pth, model=model_binary_present[0]
        )
        model_binary_unknown = load_model(
            model_binary_unknown_pth, model=model_binary_unknown[0]
        )
        patient_files = find_patient_files(data_directory)
        num_patient_files = len(patient_files)

        if num_patient_files == 0:
            raise Exception("No data was provided.")

        murmur_probabilities = list()
        murmur_outputs = list()
        for i in tqdm(range(num_patient_files)):
            current_patient_data = load_patient_data(patient_files[i])
            current_recordings = load_recordings(data_directory, current_patient_data)
            current_recordings = [r / 32768 for r in current_recordings]
            num_locations = get_num_locations(current_patient_data)
            spectrograms = list()
            for j in range(num_locations):
                mel_spec = waveform_to_examples(
                    data=current_recordings[j], sample_rate=4000
                )
                spectrograms.append(mel_spec)
            murmur_output, murmur_probability = get_binary_spectrogram_outputs(
                spectrograms, model_binary_present, model_binary_unknown
            )
            murmur_probabilities.append(murmur_probability)
            murmur_outputs.append(murmur_output)

        murmur_probabilities = np.vstack(murmur_probabilities)
        np.save(
            os.path.join(output_directory, "murmur_probabilities.npy"),
            murmur_probabilities,
        )
        murmur_outputs = np.vstack(murmur_outputs)
        np.save(os.path.join(output_directory, "murmur_outputs.npy"), murmur_outputs)
    else:
        murmur_probabilities = np.load(
            os.path.join(output_directory, "murmur_probabilities.npy")
        )
        murmur_outputs = np.load(os.path.join(output_directory, "murmur_outputs.npy"))

    return murmur_probabilities, murmur_outputs


def calculate_dbres_scores(
    recalc_output,
    data_directory,
    output_directory,
    model_binary_present_pth,
    model_binary_unknown_pth,
):

    murmur_probabilities, murmur_outputs = calculate_dbres_output(
        recalc_output,
        data_directory,
        output_directory,
        model_binary_present_pth,
        model_binary_unknown_pth,
    )
    scores = evaluate_model(data_directory, murmur_probabilities, murmur_outputs)
    print(f"{scores}")

    return scores


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="DBRes")
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

    scores = calculate_dbres_scores(**vars(args))
