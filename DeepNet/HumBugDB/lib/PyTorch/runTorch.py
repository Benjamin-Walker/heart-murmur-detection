import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import DeepNet.config as config
from DeepNet import net_config as config_pytorch
from DeepNet.HumBugDB.lib.PyTorch.ResNetDropoutSource import resnet18, resnet50dropout
from DeepNet.HumBugDB.lib.PyTorch.ResNetSource import resnet50
from DeepNet.HumBugDB.lib.PyTorch.vggish.vggish import VGGish


def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[torch.argmax(item[1])] += 1
    weight_per_class = [0.0] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[torch.argmax(val[1])]
    return weight


class ResnetFull(nn.Module):
    def __init__(self):
        #     def __init__(self):
        super(ResnetFull, self).__init__()
        self.resnet = resnet50(pretrained=config_pytorch.pretrained)
        # self.resnet = resnet18(pretrained=config_pytorch.pretrained)
        self.n_channels = 3  # For building data correctly with dataloaders. Check if 1 works with pretrained=False
        # Remove final linear layer
        self.resnet = nn.Sequential(*(list(self.resnet.children())[:-1]))
        self.fc1 = nn.Linear(
            2048, 1
        )  # 512 for resnet18, resnet34, 2048 for resnet50. Determine from x.shape() before fc1 layer

    #         self.apply(_weights_init)
    def forward(self, x):
        x = self.resnet(x).squeeze()
        #         x = self.fc1(x)
        # print(x.shape)
        x = self.fc1(x)
        x = torch.sigmoid(x)
        return x


class ResnetDropoutFull(nn.Module):
    def __init__(self, dropout=0.2):
        #     def __init__(self):
        super(ResnetDropoutFull, self).__init__()
        self.dropout = dropout
        self.resnet = resnet50dropout(
            pretrained=config_pytorch.pretrained, dropout_p=self.dropout
        )
        # self.resnet = resnet18(pretrained=config_pytorch.pretrained)
        self.n_channels = 3  # For building data correctly with dataloaders. Check if 1 works with pretrained=False
        # Remove final linear layer
        self.resnet = nn.Sequential(*(list(self.resnet.children())[:-1]))
        self.fc1 = nn.Linear(
            2048, 1
        )  # 512 for resnet18, resnet34, 2048 for resnet50. Determine from x.shape() before fc1 layer

    #         self.apply(_weights_init)
    def forward(self, x):
        x = self.resnet(x).squeeze()
        #         x = self.fc1(x)
        # print(x.shape)
        x = self.fc1(F.dropout(x, p=self.dropout))
        x = torch.sigmoid(x)
        return x


class Resnet18DropoutFull(nn.Module):
    def __init__(self, dropout=0.2):
        #     def __init__(self):
        super(Resnet18DropoutFull, self).__init__()
        # self.resnet = resnet50dropout(pretrained=config_pytorch.pretrained, dropout_p=0.2)
        self.resnet = resnet18(pretrained=config_pytorch.pretrained)
        self.dropout = dropout
        self.n_channels = 3  # For building data correctly with dataloaders. Check if 1 works with pretrained=False
        # Remove final linear layer
        self.resnet = nn.Sequential(*(list(self.resnet.children())[:-1]))
        self.fc1 = nn.Linear(
            512, 1
        )  # 512 for resnet18, resnet34, 2048 for resnet50. Determine from x.shape() before fc1 layer

    #         self.apply(_weights_init)
    def forward(self, x):
        x = self.resnet(x).squeeze()
        #         x = self.fc1(x)
        # print(x.shape)
        x = self.fc1(F.dropout(x, p=self.dropout))
        x = torch.sigmoid(x)
        return x


class VGGishDropout(nn.Module):
    def __init__(self, preprocess=False, dropout=0.2):
        super(VGGishDropout, self).__init__()
        self.model_urls = config_pytorch.vggish_model_urls
        self.vggish = VGGish(self.model_urls, postprocess=False, preprocess=preprocess)
        self.dropout = dropout
        self.n_channels = 1  # For building data correctly with dataloaders
        self.fc1 = nn.Linear(128, 1)

    def forward(self, x):
        # (Batch, Segments, C, H, W) -> (Batch*Segments, C, H, W)
        x = x.view(-1, 1, 96, 64)
        x = self.vggish.forward(x)
        x = self.fc1(F.dropout(x, p=self.dropout))
        x = torch.sigmoid(x)
        return x


class VGGishDropoutFeatB(nn.Module):
    def __init__(self, preprocess=False, dropout=0.2):
        super(VGGishDropoutFeatB, self).__init__()
        self.model_urls = config_pytorch.vggish_model_urls
        self.vggish = VGGish(
            self.model_urls,
            pretrained=config_pytorch.pretrained,
            postprocess=False,
            preprocess=preprocess,
        )
        # self.vggish = nn.Sequential(*(list(self.vggish.children())[2:])) # skip layers
        self.vggish.embeddings = nn.Sequential(
            *(list(self.vggish.embeddings.children())[2:])
        )  # skip layers
        self.dropout = dropout
        self.n_channels = 1  # For building data correctly with dataloaders
        self.fc1 = nn.Linear(128, 1)  # for multiclass
        # For application to embeddings, see:
        # https://github.com/tensorflow/models/blob/master/research/audioset/vggish/vggish_train_demo.py

    def forward(self, x):
        x = x.view(-1, 1, 30, 128)  # Feat B
        x = self.vggish.forward(x)
        x = self.fc1(F.dropout(x, p=self.dropout))
        x = torch.sigmoid(x)
        return x


def build_dataloader(
    x_train, y_train, x_val=None, y_val=None, shuffle=True, sampler=None, n_channels=1
):
    x_train = torch.tensor(x_train).float()
    y_train = torch.tensor(y_train).float()
    train_dataset = TensorDataset(x_train, y_train)
    if sampler is None:
        train_loader = DataLoader(
            train_dataset, batch_size=config_pytorch.batch_size, shuffle=shuffle
        )
    else:
        weights = make_weights_for_balanced_classes(train_dataset, 2)
        weights = torch.DoubleTensor(weights)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
        train_loader = DataLoader(
            train_dataset, batch_size=config_pytorch.batch_size, sampler=sampler
        )

    if x_val is not None:
        x_val = torch.tensor(x_val).float()
        if n_channels == 3:
            x_val = x_val.repeat(1, 3, 1, 1)
        y_val = torch.tensor(y_val).float()
        val_dataset = TensorDataset(x_val, y_val)
        val_loader = DataLoader(
            val_dataset, batch_size=config_pytorch.batch_size, shuffle=shuffle
        )

        return train_loader, val_loader
    return train_loader


def train_model(
    x_train,
    y_train,
    clas_weight=None,
    x_val=None,
    y_val=None,
    model=Resnet18DropoutFull(),
    model_name="test",
    sampler=None,
):
    if x_val is not None:  # TODO: check dimensions when supplying validation data.
        train_loader, val_loader = build_dataloader(
            x_train, y_train, x_val, y_val, sampler=sampler, n_channels=model.n_channels
        )

    else:
        train_loader = build_dataloader(
            x_train, y_train, sampler=sampler, n_channels=model.n_channels
        )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(f'Training on {device}')

    if torch.cuda.device_count() > 1:
        print("Using data parallel")
        model = nn.DataParallel(
            model, device_ids=list(range(torch.cuda.device_count()))
        )

    model = model.to(device)
    # Change compatibility to other loss function, cross-test with main.
    criterion = nn.BCELoss()
    optimiser = optim.Adam(model.parameters(), lr=config_pytorch.lr)

    all_train_loss = []
    all_train_metric = []
    all_val_loss = []
    all_val_metric = []
    best_val_acc = -np.inf

    # best_train_loss = np.inf
    best_train_acc = -np.inf

    overrun_counter = 0
    for e in range(config_pytorch.epochs):
        train_loss = 0.0
        model.train()

        all_y = []
        all_y_pred = []
        for batch_i, inputs in enumerate(train_loader):

            x = inputs[:-1][0].repeat(1, 3, 1, 1)
            y = torch.argmax(inputs[1], dim=1, keepdim=True).float()

            optimiser.zero_grad()
            y_pred = model(x)

            if clas_weight is not None:
                criterion.weight = (clas_weight[1] - clas_weight[0]) * y + clas_weight[
                    0
                ]
                loss = criterion.forward(y_pred, y)
            else:
                loss = criterion(y_pred, y)

            loss.backward()
            optimiser.step()

            train_loss += loss.item()
            all_y.append(y.cpu().detach())
            all_y_pred.append(y_pred.cpu().detach())

            del x
            del y

        all_train_loss.append(train_loss / len(train_loader))

        all_y = torch.cat(all_y)
        all_y_pred = torch.cat(all_y_pred)
        # train_metric = balanced_accuracy_score(all_y.numpy(), (all_y_pred.numpy() > 0.5).astype(float))
        train_metric = (
            all_y[all_y == 0] == torch.round(all_y_pred[all_y == 0])
        ).sum() / (all_y == 0).sum()
        all_train_metric.append(train_metric)

        # Can add more conditions to support loss instead of accuracy. Use *-1 for loss inequality instead of acc
        if x_val is not None:
            val_loss, val_metric = test_model(
                model, val_loader, clas_weight, criterion, 0.5, device=device
            )
            all_val_loss.append(val_loss)
            all_val_metric.append(val_metric)

            acc_metric = val_metric
            best_acc_metric = best_val_acc
        else:
            acc_metric = train_metric
            best_acc_metric = best_train_acc
        if acc_metric > best_acc_metric:
            # if checkpoint_name is not None:
            # os.path.join(os.path.pardir, 'models', 'pytorch', checkpoint_name)

            checkpoint_name = f"model_{model_name}.pth"

            torch.save(
                model.state_dict(),
                os.path.join(config.model_dir, "pytorch", checkpoint_name),
            )
            print(
                "Saving model to:",
                os.path.join(config.model_dir, "pytorch", checkpoint_name),
            )
            best_train_acc = train_metric
            if x_val is not None:
                best_val_acc = val_metric
            overrun_counter = -1

        overrun_counter += 1
        if x_val is not None:
            print(
                "Epoch: %d, Train Loss: %.8f, Train Acc: %.8f, Val Loss: %.8f, Val Acc: %.8f, overrun_counter %i"
                % (
                    e,
                    train_loss / len(train_loader),
                    train_metric,
                    val_loss,
                    val_metric,
                    overrun_counter,
                )
            )
        else:
            print(
                "Epoch: %d, Train Loss: %.8f, Train Acc: %.8f, overrun_counter %i"
                % (e, train_loss / len(train_loader), train_metric, overrun_counter)
            )
        if overrun_counter > config_pytorch.max_overrun:
            break
    return model


def test_model(
    model, test_loader, clas_weight, criterion, class_threshold=0.5, device=None
):
    with torch.no_grad():
        if device is None:
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        test_loss = 0.0
        model.eval()

        all_y = []
        all_y_pred = []
        counter = 1
        for inputs in test_loader:

            x = [xi.to(device).detach() for xi in inputs[:-1]]
            y = torch.argmax(inputs[1], dim=1, keepdim=True).float()

            if len(x) == 1:
                x = x[0]

            y_pred = model(x)

            if clas_weight is not None:
                criterion.weight = (clas_weight[1] - clas_weight[0]) * y + clas_weight[
                    0
                ]
                loss = criterion.forward(y_pred, y)
            else:
                loss = criterion(y_pred, y)
            test_loss += loss.item()
            all_y.append(y.cpu().detach())
            all_y_pred.append(y_pred.cpu().detach())

            del x
            del y
            del y_pred

            counter += 1

        all_y = torch.cat(all_y)
        all_y_pred = torch.cat(all_y_pred)
        test_metric = (
            all_y[all_y == 0] == torch.round(all_y_pred[all_y == 0])
        ).sum() / (all_y == 0).sum()
        # test_metric = balanced_accuracy_score(all_y.numpy(), (all_y_pred.numpy() > class_threshold).astype(float))

        # all_y = torch.nn.functional.one_hot(all_y.squeeze().long(), num_classes=2)
        # all_y_pred = torch.nn.functional.one_hot((all_y_pred > 0.5).squeeze().long(), num_classes=2)

        # test_metric = compute_weighted_accuracy(all_y, all_y_pred, ['Abnormal', 'Normal'])
        test_loss = test_loss / len(test_loader)

    return test_loss, test_metric


def load_model(filepath, model=Resnet18DropoutFull()):
    # Instantiate model to inspect
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else torch.device("cpu")
    )
    # print(f'Training on {device}')

    if torch.cuda.device_count() > 1:
        print("Using data parallel")
        model = nn.DataParallel(
            model, device_ids=list(range(torch.cuda.device_count()))
        )
    model = model.to(device)
    # Load trained parameters from checkpoint (may need to download from S3 first)

    if torch.cuda.is_available():
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = torch.device("cpu")
    model.load_state_dict(torch.load(filepath, map_location=map_location))

    return model


def evaluate_model(model, X_test, y_test, n_samples):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on {device}")

    x_test = torch.tensor(X_test).float()
    if model.n_channels == 3:
        x_test = x_test.repeat(1, 3, 1, 1)

    y_test = torch.tensor(y_test).float()
    test_dataset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    y_preds_all = np.zeros([n_samples, len(y_test), 2])
    model.eval()  # Important to not leak info from batch norm layers and cause other issues

    for n in range(n_samples):
        all_y_pred = []
        all_y = []
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            y_pred = model(x).squeeze()
            all_y.append(y.cpu().detach())

            all_y_pred.append(y_pred.cpu().detach())

            del x
            del y
            del y_pred

        all_y_pred = torch.cat(all_y_pred)
        all_y = torch.cat(all_y)

        y_preds_all[n, :, 1] = np.array(all_y_pred)
        y_preds_all[n, :, 0] = 1 - np.array(
            all_y_pred
        )  # Check ordering of classes (yes/no)
    return y_preds_all


def evaluate_model_aggregated(model, X_test, y_test, n_samples):
    """Generate predictions for VGGish features (Feat. A) rescaled to time window of 1.92 second features (Feat. B)"""
    preds_aggregated_by_mean = []
    y_aggregated_prediction_by_mean = []
    y_target_aggregated = []

    for idx, recording in enumerate(X_test):
        n_target_windows = (
            len(recording) // 2
        )  # Calculate expected length: discard edge
        y_target = np.repeat(
            y_test[idx], n_target_windows
        )  # Create y array of correct length
        preds = evaluate_model(
            model, recording, np.repeat(y_test[idx], len(recording)), n_samples
        )  # Sample BNN
        #         preds = np.mean(preds, axis=0) # Average across BNN samples
        #         print(np.shape(preds))
        preds = preds[:, : n_target_windows * 2, :]  # Discard edge case
        #         print(np.shape(preds))
        #         print('reshaping')
        preds = np.mean(
            preds.reshape(len(preds), -1, 2, 2), axis=2
        )  # Average every 2 elements, keep samples in first dim
        #         print(np.shape(preds))
        preds_y = np.argmax(preds)  # Append argmax prediction (label output)
        y_aggregated_prediction_by_mean.append(preds_y)
        preds_aggregated_by_mean.append(preds)  # Append prob (or log-prob/other space)
        y_target_aggregated.append(y_target)  # Append y_target
    #     return preds_aggregated_by_mean, y_aggregated_prediction_by_mean, y_target_aggregated
    return np.hstack(preds_aggregated_by_mean), np.concatenate(y_target_aggregated)
