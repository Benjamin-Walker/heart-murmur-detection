import argparse

import torch

from Config import hyperparameters
from DataProcessing.net_feature_extractor import net_feature_loader
from HumBugDB.runTorch import ResnetDropoutFull as ResnetDropoutBinary
from HumBugDB.runTorch import ResnetFull as ResnetBinary
from HumBugDB.runTorch import train_model as train_model_binary
from HumBugDB.runTorchMultiClass import ResnetDropoutFull as ResnetDropoutMulti
from HumBugDB.runTorchMultiClass import ResnetFull as ResnetMulti
from HumBugDB.runTorchMultiClass import train_model as train_model_multi


def create_model(model_name, num_classes, bayesian):
    if model_name == "resnet50":
        print("Running resnet without dropout")
        if num_classes == 2:
            model = ResnetBinary()
            training = train_model_binary
        else:
            model = ResnetMulti(num_classes)
            training = train_model_multi
    elif model_name == "resnet50dropout":
        print(f"Creating dropout model with bayesian: {bayesian}")
        if num_classes == 2:
            model = ResnetDropoutBinary(dropout=hyperparameters.dropout, bayesian=bayesian)
            training = train_model_binary
        else:
            model = ResnetDropoutMulti(n_classes=num_classes, bayesian=bayesian)
            training = train_model_multi
    else:
        raise NotImplementedError("Only implemented resnet50 and resnet50dropout")

    return model, training


def run_model_training(
    recalc_features,
    train_data_directory,
    vali_data_directory,
    spectrogram_directory,
    model_name,
    model_label,
    model_dir,
    classes_name,
    bayesian,
    weights,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    (
        spectrograms_train,
        murmurs_train,
        outcomes_train,
        spectrograms_test,
        murmurs_test,
        outcomes_test,
    ) = net_feature_loader(
        recalc_features,
        train_data_directory,
        vali_data_directory,
        spectrogram_directory,
    )
    print("Data loaded")
    
    X_train = spectrograms_train.to(device)
    X_test = spectrograms_test.to(device)
    if classes_name == "murmur":
        y_train = murmurs_train.to(device)
        y_test = murmurs_test.to(device)
        model, training = create_model(model_name, 3, bayesian)
        training(
            X_train,
            y_train,
            clas_weight=weights,
            x_val=X_test,
            y_val=y_test,
            model=model,
            model_name=model_label,
            model_dir=model_dir,
        )
    elif classes_name == "outcome_binary":
        y_train = outcomes_train.to(device)
        y_test = outcomes_test.to(device)
        model, training = create_model(model_name, 2, bayesian)
        training(
            X_train,
            y_train,
            clas_weight=weights,
            x_val=X_test,
            y_val=y_test,
            model=model,
            model_name=model_label,
            model_dir=model_dir,
        )
    elif classes_name == "murmur_binary":
        knowledge_train = torch.zeros((murmurs_train.shape[0], 2))
        for i in range(len(murmurs_train)):
            if (
                torch.argmax(murmurs_train[i]) == 0
                or torch.argmax(murmurs_train[i]) == 1
            ):
                knowledge_train[i, 0] = 1
            else:
                knowledge_train[i, 1] = 1
        knowledge_test = torch.zeros((murmurs_test.shape[0], 2))
        for i in range(len(murmurs_test)):
            if torch.argmax(murmurs_test[i]) == 0 or torch.argmax(murmurs_test[i]) == 1:
                knowledge_test[i, 0] = 1
            else:
                knowledge_test[i, 1] = 1
        y_train = knowledge_train.to(device)
        y_test = knowledge_test.to(device)
        model, training = create_model(model_name, 2, bayesian)
        training(
            X_train,
            y_train,
            clas_weight=weights,
            x_val=X_test,
            y_val=y_test,
            model=model,
            model_name=model_label,
            model_dir=model_dir,
        )
    elif classes_name == "binary_present":
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
        model, training = create_model(model_name, 2, bayesian)
        training(
            X_train,
            y_train,
            clas_weight=weights,
            x_val=X_test,
            y_val=y_test,
            model=model,
            model_name=model_label,
            model_dir=model_dir,
        )
    elif classes_name == "binary_unknown":
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
        model, training = create_model(model_name, 2, bayesian)
        training(
            X_train,
            y_train,
            clas_weight=weights,
            x_val=X_test,
            y_val=y_test,
            model=model,
            model_name=model_label,
            model_dir=model_dir,
            sampler=True,
        )
    else:
        raise ValueError("classes_name must be one of outcome, murmur or knowledge.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="TrainResNet")
    parser.add_argument(
        "--recalc_features",
        action="store_true",
        help="Whether or not to recalculate the log mel spectrograms used as "
        "input to the ResNet.",
    )
    parser.add_argument(
        "--no-recalc_features", dest="recalc_features", action="store_false"
    )
    parser.set_defaults(recalc_features=True)
    parser.add_argument(
        "--train_data_directory",
        type=str,
        help="The directory of the training data.",
        default="data/stratified_data/train_data",
    )
    parser.add_argument(
        "--vali_data_directory",
        type=str,
        help="The directory of the validation data.",
        default="data/stratified_data/vali_data",
    )
    parser.add_argument(
        "--spectrogram_directory",
        type=str,
        help="The directory in which to save the spectrogram training data.",
        default="data/spectrograms",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="The ResNet to train. Current options are resnet50 or resnet50dropout.",
        choices=["resnet50", "resnet50dropout"],
        default="resnet50dropout",
    )
    parser.add_argument(
        "--model_label",
        type=str,
        help="The label to use when saving the model.",
        default="ResNetDropout",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        help="The directory to use when saving the model.",
        default="data/models",
    )
    parser.add_argument(
        "--classes_name",
        type=str,
        help="The name of the classes to train the model on.",
        choices=["murmur", "outcome_binary", "murmur_binary", "binary_present", "binary_unknown"],
        default="murmur",
    )
    parser.add_argument(
        '--disable-bayesian', 
        dest='bayesian', 
        action='store_false', 
        default=True,
        help='Disable Bayesian features (default: Bayesian is enabled)'
    )
    parser.add_argument(
        "--weights_str",
        type=str,
        help="String containing the class weights for a weighted loss function, "
        "e.g.5,3,1.",
        default=None,
    )

    args = parser.parse_args()

    weights = None
    if args.weights_str:
        weights = [int(x) for x in args.weights_str.split(",")]
    vars(args).popitem()

    print("---------------- Starting train_resnet.py for training ----------------")
    print(f"---------------- Using data from {args.train_data_directory}")

    run_model_training(**vars(args), weights=weights)
