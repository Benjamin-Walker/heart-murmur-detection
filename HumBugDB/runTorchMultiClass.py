import os

import numpy as np
import time
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
    def __init__(self, n_classes):
        super(ResnetFull, self).__init__()
        self.resnet = resnet50(pretrained=hyperparameters.pretrained)
        self.n_channels = 3
        # Remove final linear layer
        self.resnet = nn.Sequential(*(list(self.resnet.children())[:-1]))
        self.fc1 = nn.Linear(2048, n_classes)

    def forward(self, x):
        x = self.resnet(x).squeeze()
        x = self.fc1(x)
        return x


class ResnetDropoutFull(nn.Module):
    def __init__(self, n_classes, bayesian=True, dropout=0.2):
        super(ResnetDropoutFull, self).__init__()
        self.dropout = dropout
        self.bayesian = bayesian
        self.resnet = resnet50dropout(
            pretrained=hyperparameters.pretrained, dropout_p=self.dropout, bayesian=bayesian
        )

        self.n_channels = 3
        # Remove final linear layer
        self.resnet = nn.Sequential(*(list(self.resnet.children())[:-1]))
        self.fc1 = nn.Linear(2048, n_classes)

    def forward(self, x):
        if self.bayesian == True:
            training = True
        else:
            training = self.training
        x = self.resnet(x).squeeze()
        x = self.fc1(F.dropout(x, p=self.dropout, training=training))
        return x


def build_dataloader(x_train, y_train, x_val=None, y_val=None, shuffle=True):
    x_train = torch.tensor(x_train)
    y_train = torch.tensor(y_train)
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(
        train_dataset, batch_size=hyperparameters.batch_size, shuffle=shuffle
    )

    if x_val is not None:
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
    model=ResnetDropoutFull(hyperparameters.n_classes),
    model_name="test",
    model_dir="models",
):
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    if x_val is not None:
        train_loader, val_loader = build_dataloader(x_train, y_train, x_val, y_val)

    else:
        train_loader = build_dataloader(x_train, y_train)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        print("Using data parallel")
        model = nn.DataParallel(
            model, device_ids=list(range(torch.cuda.device_count()))
        )

    model = model.to(device)

    if clas_weight is not None:
        print("Applying class weights:", clas_weight)
        clas_weight = torch.tensor([clas_weight]).squeeze().float().to(device)
    criterion = nn.CrossEntropyLoss(weight=clas_weight)

    optimiser = optim.Adam(model.parameters(), lr=hyperparameters.lr)

    all_train_loss = []
    all_train_acc = []
    all_val_loss = []
    all_val_acc = []
    best_val_acc = -np.inf
    best_train_acc = -np.inf
    overrun_counter = 0

    for e in range(hyperparameters.epochs):
        start_time = time.time()
        train_loss = 0.0
        model.train()

        all_y = []
        all_y_pred = []
        for batch_i, inputs in tqdm(enumerate(train_loader), total=len(train_loader)):
            x = inputs[:-1][0].repeat(1, 3, 1, 1)
            y = inputs[-1].to(device).detach()
            if len(x) == 1:
                x = x[0]
            optimiser.zero_grad()
            y_pred = model(x)
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
        train_acc = balanced_accuracy_score(
            all_y.numpy(), (all_y_pred.numpy() > 0.5).astype(float)
        )
        all_train_acc.append(train_acc)

        if x_val is not None:
            val_loss, val_acc = test_model(model, val_loader, criterion, device=device)
            all_val_loss.append(val_loss)
            all_val_acc.append(val_acc)

            acc_metric = val_acc
            best_acc_metric = best_val_acc
        else:
            acc_metric = train_acc
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
            best_train_acc = train_acc
            if x_val is not None:
                best_val_acc = val_acc
            overrun_counter = -1

        overrun_counter += 1
        if x_val is not None:
            print(
                "Epoch: %d, Train Loss: %.8f, Train Acc: %.8f, Val Loss: %.8f, "
                "Val Acc: %.8f, overrun_counter %i"
                % (
                    e,
                    train_loss / len(train_loader),
                    train_acc,
                    val_loss,
                    val_acc,
                    overrun_counter,
                )
            )
        else:
            print(
                "Epoch: %d, Train Loss: %.8f, Train Acc: %.8f, overrun_counter %i"
                % (e, train_loss / len(train_loader), train_acc, overrun_counter)
            )
        print(f"Training epoch {e} took {round((time.time()-start_time)/60,4)} min.")
        if overrun_counter > hyperparameters.max_overrun:
            break
    return model


def test_model(model, test_loader, criterion, device=None):
    with torch.no_grad():
        if device is None:
            torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

        test_loss = 0.0
        model.eval()

        all_y = []
        all_y_pred = []
        counter = 1
        for inputs in test_loader:
            x = inputs[:-1][0].repeat(1, 3, 1, 1)
            y = inputs[1].float()

            y_pred = model(x)

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

        test_loss = test_loss / len(test_loader)
        test_acc = balanced_accuracy_score(
            all_y.numpy(), (all_y_pred.numpy() > 0.5).astype(float)
        )

        return test_loss, test_acc
