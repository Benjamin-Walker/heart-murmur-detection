import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import net_config as config_pytorch
from HumBugDB.ResNetDropoutSource import resnet50dropout
from HumBugDB.ResNetSource import resnet50


class ResnetFull(nn.Module):
    def __init__(self):
        super(ResnetFull, self).__init__()
        self.resnet = resnet50(pretrained=config_pytorch.pretrained)
        self.n_channels = 3  # For building data correctly with dataloaders. Check if 1 works with pretrained=False
        # Remove final linear layer
        self.resnet = nn.Sequential(*(list(self.resnet.children())[:-1]))
        self.fc1 = nn.Linear(
            2048, 1
        )  # 512 for resnet18, resnet34, 2048 for resnet50. Determine from x.shape() before fc1 layer

    def forward(self, x):
        x = self.resnet(x).squeeze()
        x = self.fc1(x)
        x = torch.sigmoid(x)
        return x


class ResnetDropoutFull(nn.Module):
    def __init__(self, dropout=0.2):
        super(ResnetDropoutFull, self).__init__()
        self.dropout = dropout
        self.resnet = resnet50dropout(
            pretrained=config_pytorch.pretrained, dropout_p=self.dropout
        )
        self.n_channels = 3  # For building data correctly with dataloaders. Check if 1 works with pretrained=False
        # Remove final linear layer
        self.resnet = nn.Sequential(*(list(self.resnet.children())[:-1]))
        self.fc1 = nn.Linear(
            2048, 1
        )  # 512 for resnet18, resnet34, 2048 for resnet50. Determine from x.shape() before fc1 layer

    def forward(self, x):
        x = self.resnet(x).squeeze()
        x = self.fc1(F.dropout(x, p=self.dropout))
        x = torch.sigmoid(x)
        return x


def build_dataloader(
    x_train, y_train, x_val=None, y_val=None, shuffle=True, n_channels=1
):
    train_dataset = TensorDataset(x_train, y_train)

    train_loader = DataLoader(
        train_dataset, batch_size=config_pytorch.batch_size, shuffle=shuffle
    )

    if x_val is not None:
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
    model=ResnetDropoutFull(),
    model_name="test",
    model_dir="models",
):

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    if x_val is not None:  # TODO: check dimensions when supplying validation data.
        train_loader, val_loader = build_dataloader(
            x_train, y_train, x_val, y_val, n_channels=model.n_channels
        )

    else:
        train_loader = build_dataloader(x_train, y_train, n_channels=model.n_channels)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        train_metric = (
            all_y[all_y == 0] == torch.round(all_y_pred[all_y == 0])
        ).sum() / (all_y == 0).sum()
        all_train_metric.append(train_metric)

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

            checkpoint_name = f"model_{model_name}.pth"

            torch.save(
                model.state_dict(),
                os.path.join(model_dir, checkpoint_name),
            )
            print(
                "Saving model to:",
                os.path.join(model_dir, checkpoint_name),
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


def test_model(model, test_loader, clas_weight, criterion, device=None):
    with torch.no_grad():
        if device is None:
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        test_loss = 0.0
        model.eval()

        all_y = []
        all_y_pred = []
        counter = 1
        for inputs in test_loader:

            x = inputs[:-1][0].repeat(1, 3, 1, 1)
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
        test_loss = test_loss / len(test_loader)

    return test_loss, test_metric


def load_model(filepath, model=ResnetDropoutFull()):
    # Instantiate model to inspect
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else torch.device("cpu")
    )

    if torch.cuda.device_count() > 1:
        print("Using data parallel")
        model = nn.DataParallel(
            model, device_ids=list(range(torch.cuda.device_count()))
        )
    model = model.to(device)

    if torch.cuda.is_available():
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = torch.device("cpu")
    model.load_state_dict(torch.load(filepath, map_location=map_location))

    return model
