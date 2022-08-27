import sys

import torch

from DeepNet.HumBugDB.lib.PyTorch.runTorch import (
    ResnetDropoutFull as ResnetDropoutBinary,
)
from DeepNet.HumBugDB.lib.PyTorch.runTorch import ResnetFull as ResnetBinary
from DeepNet.HumBugDB.lib.PyTorch.runTorch import train_model as train_model_binary
from DeepNet.HumBugDB.lib.PyTorch.runTorchMultiClass import (
    ResnetDropoutFull as ResnetDropoutMulti,
)
from DeepNet.HumBugDB.lib.PyTorch.runTorchMultiClass import ResnetFull as ResnetMulti
from DeepNet.HumBugDB.lib.PyTorch.runTorchMultiClass import (
    train_model as train_model_multi,
)
from DeepNet.net_feature_extractor import net_feature_loader


def create_model(model_name, num_classes):
    if model_name == "resent50":
        if num_classes == 2:
            model = ResnetBinary()
            training = train_model_binary
        else:
            model = ResnetMulti(num_classes)
            training = train_model_multi
    elif model_name == "resnet50dropout":
        if num_classes == 2:
            model = ResnetDropoutBinary(dropout=0.3)
            training = train_model_binary
        else:
            model = ResnetDropoutMulti(num_classes)
            training = train_model_multi
    else:
        raise NotImplementedError("Only implemented resnet50 and resnet50dropout")

    return model, training


def run_model_training(
    recalc_features,
    train_data_folder,
    test_data_folder,
    model_name,
    model_label,
    classes_name,
    weights,
):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    (
        spectrograms_train,
        murmurs_train,
        outcomes_train,
        spectrograms_test,
        murmurs_test,
        outcomes_test,
    ) = net_feature_loader(recalc_features, train_data_folder, test_data_folder)

    X_train = spectrograms_train.to(device)
    X_test = spectrograms_test.to(device)

    if classes_name == "murmur":
        y_train = murmurs_train.to(device)
        y_test = murmurs_test.to(device)
        model, training = create_model(model_name, 3)
        training(
            X_train,
            y_train,
            clas_weight=weights,
            x_val=X_test,
            y_val=y_test,
            model=model,
            model_name=model_label,
        )
    elif classes_name == "murmur_binary":
        X_train = spectrograms_train[(murmurs_train[:, 1] == 0)].to(device)
        X_test = spectrograms_test[(murmurs_test[:, 1] == 0)].to(device)
        y_train = murmurs_train[(murmurs_train[:, 1] == 0)][:, [0, 2]].to(device)
        y_test = murmurs_test[(murmurs_test[:, 1] == 0)][:, [0, 2]].to(device)
        model, training = create_model(model_name, 2)
        training(
            X_train,
            y_train,
            clas_weight=weights,
            x_val=X_test,
            y_val=y_test,
            model=model,
            model_name=model_label,
        )
    elif classes_name == "outcome":
        y_train = outcomes_train.to(device)
        y_test = outcomes_test.to(device)
        model, training = create_model(model_name, 2)
        training(
            X_train,
            y_train,
            clas_weight=weights,
            x_val=X_test,
            y_val=y_test,
            model=model,
            model_name=model_label,
        )
    elif classes_name == "knowledge_present":

        knowledge_train = torch.zeros((murmurs_train.shape[0], 2))
        for i in range(len(murmurs_train)):
            if (
                torch.argmax(murmurs_train[i]) == 1
                or torch.argmax(murmurs_train[i]) == 2
            ):
                knowledge_train[i, 1] = 1
            else:
                knowledge_train[i, 0] = 1

        knowledge_test = torch.zeros((murmurs_test.shape[0], 2))
        for i in range(len(murmurs_test)):
            if torch.argmax(murmurs_test[i]) == 1 or torch.argmax(murmurs_test[i]) == 2:
                knowledge_test[i, 1] = 1
            else:
                knowledge_test[i, 0] = 1

        y_train = knowledge_train.to(device)
        y_test = knowledge_test.to(device)
        model, training = create_model(model_name, 2)
        training(
            X_train,
            y_train,
            clas_weight=weights,
            x_val=X_test,
            y_val=y_test,
            model=model,
            model_name=model_label,
            sampler=True,
        )
    elif classes_name == "knowledge_unknown":

        knowledge_train = torch.zeros((murmurs_train.shape[0], 2))
        for i in range(len(murmurs_train)):
            if (
                torch.argmax(murmurs_train[i]) == 0
                or torch.argmax(murmurs_train[i]) == 2
            ):
                knowledge_train[i, 1] = 1
            else:
                knowledge_train[i, 0] = 1

        knowledge_test = torch.zeros((murmurs_test.shape[0], 2))
        for i in range(len(murmurs_test)):
            if torch.argmax(murmurs_test[i]) == 0 or torch.argmax(murmurs_test[i]) == 2:
                knowledge_test[i, 1] = 1
            else:
                knowledge_test[i, 0] = 1

        y_train = knowledge_train.to(device)
        y_test = knowledge_test.to(device)
        model, training = create_model(model_name, 2)
        training(
            X_train,
            y_train,
            clas_weight=weights,
            x_val=X_test,
            y_val=y_test,
            model=model,
            model_name=model_label,
            sampler=True,
        )
    else:
        raise ValueError("classes_name must be one of outcome, murmur or knowledge.")


if __name__ == "__main__":

    """
    recalc_features: whether to recalculate the spectrogram patches
    train_data_folder: folder where the training patient data is located, only
                       used if recalculating features
    vali_data_folder: folder where the validation patient data is located, only
                      used if recalculating features
    model_name: the name of the model, currently either resnet50 or
                resnet50dropout
    model_label: name used when saving the model
    classes_name: either outcome or murmur
    weights: comma separated weights for the classes, e.g. 5,3,1
    """

    recalc_features = sys.argv[1]
    train_data_folder = sys.argv[2]
    vali_data_folder = sys.argv[3]
    model_name = sys.argv[4]
    model_label = sys.argv[5]
    classes_name = sys.argv[6]
    if len(sys.argv) == 8:
        str_weight = sys.argv[7]
        weights = [int(x) for x in str_weight.split(",")]
    else:
        weights = None

    run_model_training(
        recalc_features,
        train_data_folder,
        vali_data_folder,
        model_name,
        model_label,
        classes_name,
        weights,
    )
