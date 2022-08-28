import sys


sys.path.insert(0, "/home/walkerb1/project/physio")
sys.path.insert(0, "/")
import numpy as np
import torch
import torch.nn as nn
from evaluate_model import compute_cost, compute_weighted_accuracy
from tqdm import tqdm

from DataProcessing.net_feature_extractor import patient_feature_loader
from HumBugDB.runTorch import load_model
from train_net import create_model


def calc_patient_output(model, recording_spectrograms, repeats):
    model.eval()
    outputs = []
    for location in recording_spectrograms:
        input = location.repeat(1, 3, 1, 1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input = input.to(device)
        softmax = nn.Softmax(dim=1)
        model_out = []
        for _ in range(repeats):
            out = softmax(model(input)).detach().unsqueeze(2)
            model_out.append(out)
            del out
        model_out = torch.mean(torch.cat(model_out, dim=2), dim=2)
        outputs.append(torch.mean(model_out))
    output = torch.mean(torch.cat(outputs), axis=0).detach()
    return output


def calc_model_scores(
    recalc_features, data_folder, model_name, model_label, classes_name
):

    if classes_name == "murmur":
        classes = ["Present", "Unknown", "Absent"]
        score_fn = compute_weighted_accuracy
    elif classes_name == "outcome":
        classes = ["Abnormal", "Normal"]
        score_fn = compute_cost
    else:
        raise ValueError("classes_name must be one of outcome or murmur.")

    model_path = f"DeepNet/outputs/models/pytorch/model_{model_label}.pth"
    # model_name = resnet50 or resnet50dropout
    model = create_model(model_name, len(classes))
    model = load_model(model_path, model=model[0])

    spectrograms, murmurs, outcomes = patient_feature_loader(
        recalc_features, data_folder, "DeepNet/PatientData/"
    )
    model_out = []
    outputs = []
    labels = []
    repeats = 30
    for spectrogram, murmur, outcome in tqdm(zip(spectrograms, murmurs, outcomes)):
        softmax_ = calc_patient_output(model, spectrogram, repeats)
        model_out.append(softmax_.detach().cpu().numpy())
        output = np.zeros(len(classes), dtype=np.int_)
        idx = np.argmax(softmax_.cpu().numpy())
        output[idx] = 1
        outputs.append(output)
        if classes_name == "murmur":
            labels.append(murmur)
        elif classes_name == "outcome":
            labels.append(outcome)

    return (
        score_fn(np.array(labels), np.array(outputs), classes),
        np.array(model_out),
        np.array(labels),
    )


if __name__ == "__main__":

    recalc_features = sys.argv[1]
    data_folder = sys.argv[2]
    model_name = sys.argv[3]
    model_label = sys.argv[4]
    classes_name = sys.argv[5]

    score, model_out, labels = calc_model_scores(
        recalc_features, data_folder, model_name, model_label, classes_name
    )

    breakpoint()
