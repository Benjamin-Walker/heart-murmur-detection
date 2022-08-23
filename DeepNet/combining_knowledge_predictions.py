import sys


sys.path.insert(0, "/home/walkerb1/project/physio")
sys.path.insert(0, "/Users/benwalker/PycharmProjects/PhysionetChallenge2022")
import numpy as np
import torch
from evaluate_model import compute_weighted_accuracy
from tqdm import tqdm

from DeepNet.HumBugDB.lib.PyTorch.runTorch import load_model
from DeepNet.net_feature_extractor import patient_feature_loader
from DeepNet.train_net import create_model


def calc_patient_output_know(model, recording_spectrograms, repeats, weighted):
    model.eval()
    outputs = []
    for location in recording_spectrograms:
        input = location.repeat(1, 3, 1, 1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input = input.to(device)
        model_out = []
        for _ in range(repeats):
            out = model(input).detach().unsqueeze(2)
            model_out.append(out)
            del out
        model_out = torch.mean(torch.cat(model_out, dim=2), dim=2)
        outputs.append(torch.mean(model_out, axis=0).unsqueeze(dim=0))
    output = torch.mean(torch.cat(outputs), axis=0).detach()
    return output


if __name__ == "__main__":

    recalc_features = sys.argv[1]
    data_folder = sys.argv[2]
    model_path_knowledge = sys.argv[3]
    model_path_murmur = sys.argv[4]

    spectrograms, murmurs, outcomes = patient_feature_loader(
        recalc_features, data_folder, "DeepNet/PatientData/test_4_1/"
    )

    knowledge = []
    for i in range(len(murmurs)):
        if np.argmax(murmurs[i]) == 1:
            knowledge.append(np.array([0, 1]))
        else:
            knowledge.append(np.array([1, 0]))
    # model_path_knowledge = 'DeepNet/outputs/models/pytorch/model_knowledge_unknown_submit.pth'
    # model_path_murmur = 'DeepNet/outputs/models/pytorch/model_knowledge_present_submit.pth'
    # model_name = resnet50 or resnet50dropout
    model_know = create_model("resnet50dropout", 2)
    model_mur = create_model("resnet50dropout", 2)
    model_unknown = load_model(model_path_knowledge, model=model_know[0])
    model_present = load_model(model_path_murmur, model=model_mur[0])

    model_outputs_unknown = []
    model_outputs_present = []
    outputs_unknown = []
    outputs_present = []
    labels = []
    num_spectrograms = []

    for spectrogram, murmur in tqdm(zip(spectrograms, murmurs)):
        output_unknown = (
            calc_patient_output_know(
                model_unknown, spectrogram, repeats=10, weighted=False
            )
            .cpu()
            .numpy()
        )
        output_present = (
            calc_patient_output_know(
                model_present, spectrogram, repeats=10, weighted=False
            )
            .cpu()
            .numpy()
        )
        model_outputs_unknown.append(output_unknown)
        output = np.zeros(2, dtype=np.int_)
        idx = (output_unknown > 0.5).astype(int)
        output[idx] = 1
        outputs_unknown.append(output)
        model_outputs_present.append(output_present)
        output = np.zeros(2, dtype=np.int_)
        idx = (output_present > 0.5).astype(int)
        output[idx] = 1
        outputs_present.append(output)
        labels.append(murmur)

    classes = ["Present", "Unknown", "Absent"]

    outputs = []

    idx_unknown = (np.array(model_outputs_unknown) > 0.5).astype(float).T[0]
    idx_present = (np.array(model_outputs_present) > 0.5).astype(float).T[0]
    for i in range(len(np.array(model_outputs_unknown))):
        if idx_present[i] == 0:
            outputs.append(np.array([1, 0, 0]))
        elif idx_unknown[i] == 0:
            outputs.append(np.array([0, 1, 0]))
        else:
            outputs.append(np.array([0, 0, 1]))

    score = compute_weighted_accuracy(np.array(labels), np.array(outputs), classes)

    breakpoint()
