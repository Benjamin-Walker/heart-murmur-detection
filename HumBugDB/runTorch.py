import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from Config import hyperparameters
from HumBugDB.ResNetDropoutSource import resnet50dropout
from HumBugDB.ResNetSource import resnet50


class ResnetFull(nn.Module):
    def __init__(self):
        super(ResnetFull, self).__init__()
        self.resnet = resnet50(pretrained=hyperparameters.pretrained)
        self.n_channels = 3
        # Remove final linear layer
        self.resnet = nn.Sequential(*(list(self.resnet.children())[:-1]))
        self.fc1 = nn.Linear(2048, 1)

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
            pretrained=hyperparameters.pretrained, dropout_p=self.dropout
        )
        self.n_channels = 3
        # Remove final linear layer
        self.resnet = nn.Sequential(*(list(self.resnet.children())[:-1]))
        self.fc1 = nn.Linear(2048, 1)

    def forward(self, x):
        x = self.resnet(x).squeeze()
        x = self.fc1(F.dropout(x, p=self.dropout))
        x = torch.sigmoid(x)
        return x


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


def build_dataloader(
    x_train, y_train, x_val=None, y_val=None, shuffle=True, sampler=None
):
    x_train = torch.tensor(x_train).float()
    y_train = torch.tensor(y_train).float()
    train_dataset = TensorDataset(x_train, y_train)
    if sampler is None:
        train_loader = DataLoader(
            train_dataset, batch_size=hyperparameters.batch_size, shuffle=shuffle
        )
    else:
        weights = make_weights_for_balanced_classes(train_dataset, 2)
        weights = torch.DoubleTensor(weights)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
        train_loader = DataLoader(
            train_dataset, batch_size=hyperparameters.batch_size, sampler=sampler
        )

    if x_val is not None:
        x_val = torch.tensor(x_val).float()
        y_val = torch.tensor(y_val).float()
        val_dataset = TensorDataset(x_val, y_val)
        val_loader = DataLoader(
            val_dataset, batch_size=hyperparameters.batch_size, shuffle=shuffle
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
    sampler=None,
):

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    if x_val is not None:
        train_loader, val_loader = build_dataloader(
            x_train, y_train, x_val, y_val, sampler=sampler
        )

    else:
        train_loader = build_dataloader(x_train, y_train, sampler=sampler)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        print("Using data parallel")
        model = nn.DataParallel(
            model, device_ids=list(range(torch.cuda.device_count()))
        )

    model = model.to(device)
    criterion = nn.BCELoss()
    optimiser = optim.Adam(model.parameters(), lr=hyperparameters.lr)

    all_train_loss = []
    all_train_metric = []
    all_val_loss = []
    all_val_metric = []
    best_val_acc = -np.inf

    best_train_acc = -np.inf

    overrun_counter = 0
    for e in range(hyperparameters.epochs):
        train_loss = 0.0
        model.train()

        all_y = []
        all_y_pred = []
        for batch_i, inputs in tqdm(enumerate(train_loader), total=len(train_loader)):

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
        train_metric = balanced_accuracy_score(
            all_y.numpy(), (all_y_pred.numpy() > 0.5).astype(float)
        )
        all_train_metric.append(train_metric)

        if x_val is not None:
            val_loss, val_metric = test_model(
                model, val_loader, clas_weight, criterion, device=device
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
                "Epoch: %d, Train Loss: %.8f, Train Acc: %.8f, Val Loss: %.8f, "
                "Val Acc: %.8f, overrun_counter %i"
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
        if overrun_counter > hyperparameters.max_overrun:
            break
    return model


def test_model(model, test_loader, clas_weight, criterion, device=None):
    with torch.no_grad():
        if device is None:
            torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        test_metric = balanced_accuracy_score(
            all_y.numpy(), (all_y_pred.numpy() > 0.5).astype(float)
        )
        test_loss = test_loss / len(test_loader)

    return test_loss, test_metric


def load_model(filepath, model=ResnetDropoutFull()):

    device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))

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
