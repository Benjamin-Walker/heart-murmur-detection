import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset

import net_config as config_pytorch
from HumBugDB.ResNetDropoutSource import resnet50dropout
from HumBugDB.ResNetSource import resnet50
from HumBugDB.vggish.vggish import VGGish


# Resnet with full dropout


class ResnetFull(nn.Module):
    def __init__(self, n_classes):
        super(ResnetDropoutFull, self).__init__()
        self.resnet = resnet50(pretrained=config_pytorch.pretrained)
        self.n_channels = 3
        # Remove final linear layer
        self.resnet = nn.Sequential(*(list(self.resnet.children())[:-1]))
        self.fc1 = nn.Linear(
            2048, n_classes
        )  # 512 for resnet18, resnet34, 2048 for resnet50. Determine from x.shape() before fc1 layer

    def forward(self, x):
        x = self.resnet(x).squeeze()
        x = self.fc1(x)
        return x


class ResnetDropoutFull(nn.Module):
    def __init__(self, n_classes, dropout=0.2):
        super(ResnetDropoutFull, self).__init__()
        self.dropout = dropout
        self.resnet = resnet50dropout(
            pretrained=config_pytorch.pretrained, dropout_p=self.dropout
        )

        self.n_channels = 3
        # Remove final linear layer
        self.resnet = nn.Sequential(*(list(self.resnet.children())[:-1]))
        self.fc1 = nn.Linear(
            2048, n_classes
        )  # 512 for resnet18, resnet34, 2048 for resnet50. Determine from x.shape() before fc1 layer

    def forward(self, x):
        x = self.resnet(x).squeeze()
        x = self.fc1(F.dropout(x, p=self.dropout))
        return x


# Resnet with dropout on last layer only
class Resnet(nn.Module):
    def __init__(self, n_classes, dropout=0.2):
        super(Resnet, self).__init__()
        self.resnet = resnet50(pretrained=config_pytorch.pretrained)
        self.dropout = dropout
        self.n_channels = 3
        # Remove final linear layer
        self.resnet = nn.Sequential(*(list(self.resnet.children())[:-1]))
        self.fc1 = nn.Linear(
            2048, n_classes
        )  # 512 for resnet18, resnet34, 2048 for resnet50. Determine from x.shape() before fc1 layer

    #         self.apply(_weights_init)
    def forward(self, x):
        x = self.resnet(x).squeeze()
        #         x = self.fc1(x)
        # print(x.shape)
        x = self.fc1(F.dropout(x, p=self.dropout))
        # x = torch.sigmoid(x)  # Warning on this: XENT loss doesn't need sigmoid whereas BCELoss does
        return x


class VGGishDropout(nn.Module):
    def __init__(self, n_classes, preprocess=False, dropout=0.2):
        super(VGGishDropout, self).__init__()
        self.model_urls = config_pytorch.vggish_model_urls
        self.vggish = VGGish(
            self.model_urls,
            pretrained=config_pytorch.pretrained,
            postprocess=False,
            preprocess=preprocess,
        )
        self.dropout = dropout
        self.n_channels = 1  # For building data correctly with dataloaders
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        # (Batch, Segments, C, H, W) -> (Batch*Segments, C, H, W)
        x = x.view(-1, 1, 96, 64)
        x = self.vggish.forward(x)
        x = self.fc2(F.dropout(x, p=self.dropout))
        return x


class VGGishDropoutFeatB(nn.Module):
    def __init__(self, n_classes, preprocess=False, dropout=0.2):
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
        self.fc1 = nn.Linear(128, n_classes)  # for multiclass
        # For application to embeddings, see:
        # https://github.com/tensorflow/models/blob/master/research/audioset/vggish/vggish_train_demo.py

    def forward(self, x):
        x = x.view(-1, 1, 30, 128)  # Feat B
        x = self.vggish.forward(x)
        x = self.fc1(F.dropout(x, p=self.dropout))
        x = torch.sigmoid(x)
        return x


def build_dataloader(x_train, y_train, x_val=None, y_val=None, shuffle=True):
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
    model=Resnet(config_pytorch.n_classes),
    model_name="test",
    model_dir="models",
):
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    if x_val is not None:  # TODO: check dimensions when supplying validation data.
        train_loader, val_loader = build_dataloader(x_train, y_train, x_val, y_val)

    else:
        train_loader = build_dataloader(x_train, y_train, n_channels=model.n_channels)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(f'Training on {device}')

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

    optimiser = optim.Adam(model.parameters(), lr=config_pytorch.lr)

    all_train_loss = []
    all_train_acc = []
    all_val_loss = []
    all_val_acc = []
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
            y = inputs[1].float()
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
        train_acc = accuracy_score(
            np.argmax(all_y.numpy(), axis=1), np.argmax(all_y_pred.numpy(), axis=1)
        )
        all_train_acc.append(train_acc)

        # Can add more conditions to support loss instead of accuracy. Use *-1 for loss inequality instead of acc
        if x_val is not None:
            val_loss, val_acc = test_model(
                model, val_loader, criterion, 0.5, device=device
            )  # This might not work multi.c.
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
                os.path.join(model_dir, "pytorch", checkpoint_name),
            )
            print(
                "Saving model to:",
                os.path.join(model_dir, "pytorch", checkpoint_name),
            )
            best_train_acc = train_acc
            if x_val is not None:
                best_val_acc = val_acc
            overrun_counter = -1

        overrun_counter += 1
        if x_val is not None:
            print(
                "Epoch: %d, Train Loss: %.8f, Train Acc: %.8f, Val Loss: %.8f, Val Acc: %.8f, overrun_counter %i"
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
        if overrun_counter > config_pytorch.max_overrun:
            break
    return model


def test_model(model, test_loader, criterion, device=None):
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
        test_acc = accuracy_score(
            np.argmax(all_y.numpy(), axis=1), np.argmax(all_y_pred.numpy(), axis=1)
        )

        return test_loss, test_acc


def load_model(filepath, model=Resnet(config_pytorch.n_classes)):
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
    # Load trained parameters from checkpoint (may need to download from S3 first)

    if torch.cuda.is_available():
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = torch.device("cpu")
    model.load_state_dict(torch.load(filepath, map_location=map_location))

    return model
