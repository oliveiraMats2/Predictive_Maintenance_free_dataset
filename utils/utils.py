import torch
import os

import yaml
from tqdm import tqdm
import numpy as np
from typing import List
from sklearn.metrics import confusion_matrix


def or_operation(list_one_hot: List[int]) -> int:
    or_result = 0
    for i in range(len(list_one_hot)):
        or_result = or_result or list_one_hot[i]

    return or_result


def create_context(X: np.ndarray, Y: np.ndarray, context: int) -> (list, list):
    samples = X.shape[1]  # modify

    data_context = []
    context_labels = []

    for i in range(samples - context):
        data_context.append(X[:, i:i + context])
        context_labels.append(Y[i:i + context])

    context_labels = [or_operation(context_) for context_ in context_labels]

    return np.array(data_context), np.array(context_labels)


def set_device():
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)
    print('Using {}'.format(device))

    return device


def read_yaml(file: str) -> yaml.loader.FullLoader:
    with open(file, "r") as yaml_file:
        configurations = yaml.load(yaml_file, Loader=yaml.FullLoader)

    return configurations





def evaluate_matrix_confusion(model, loader, context_size, device: str):
    y_list = []
    y_hat_list = []

    with torch.no_grad():
        total = 0
        acc = 0
        for x, y in tqdm(loader):
            y_hat = model(x.to(device))
            y_hat = torch.argmax(y_hat, dim=1)

            y = y.to(device)
            y_hat = y_hat.to(device)

            y = np.squeeze(y.numpy())
            y_hat = np.squeeze(y_hat.numpy())

            y_list.append(y)
            y_hat_list.append(y_hat)

            acc += y[y == y_hat].shape[0]
            total += 1 * x.shape[0]

        mean_accuracy = acc / total

        y_array = np.concatenate(y_list)
        y_hat_array = np.concatenate(y_hat_list)

        cm = confusion_matrix(y_array, y_hat_array)
        return cm / context_size, mean_accuracy


def evaluate_batch(model, valid_loader, criterion, device):
    with torch.no_grad():
        total = 0
        acc = 0
        x, y = next(iter(valid_loader))
        y_hat = model(x.to(device))

        y = y.to(device)
        loss = criterion(y_hat, y[:, 0])

        y_hat = torch.argmax(y_hat, dim=1)

        acc += y[y.squeeze(1) == y_hat].shape[0]
        total += 1 * x.shape[0]

        mean_accuracy = acc / total

    return mean_accuracy, loss


def evaluate(model, loader, device: str) -> float:
    with torch.no_grad():
        total = 0
        acc = 0
        for x, y in tqdm(loader):
            y_hat = model(x.to(device))
            y_hat = torch.argmax(y_hat, dim=1)

            y = y.to(device)
            y_hat = y_hat.to(device)

            acc += y[y.squeeze(1) == y_hat].shape[0]
            total += 1 * x.shape[0]

        mean_accuracy = int(acc / total)

        return mean_accuracy


def train(model, train_dataloader, valid_dataloader, optimizer, criterion, lr,
          batch_size_train, name_models, num_epochs, save_scores, device):
    train_loss = 0
    list_loss_valid = []
    accuracy_list_valid = []
    list_loss_train = []

    for epoch in range(num_epochs):
        for x, y in tqdm(train_dataloader):
            optimizer.zero_grad()

            x_ids = x[0].to(device)
            attention_mask = x[1].to(device)
            labels = y.to(device)
            logits = model(x_ids, attention_mask=attention_mask).logits
            loss = criterion(logits, labels)

            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        list_loss_train.append(train_loss / len(train_dataloader))

        num_examples = len(valid_dataloader.dataset)

        accuracy_valid, loss_valid = evaluate(model, num_examples, valid_dataloader, criterion, device)

        last_loss = loss_valid

        save_scores.save_loss_in_file(loss_valid,
                                      f"valid_lr_{lr}_batch_size_{batch_size_train}_name_{name_models}.txt".replace("/",
                                                                                                                    ""))
        save_scores.save_loss_in_file(accuracy_valid,
                                      f"acc_lr_{lr}_batch_size_{batch_size_train}_name_{name_models}.txt".replace("/",
                                                                                                                  ""))

        save_scores.save_loss_in_file((train_loss / len(train_dataloader)),
                                      f"train_lr_{lr}_batch_size_{batch_size_train}_name_{name_models}.txt".replace("/",
                                                                                                                    ""))

        print(
            f"Epoch: {epoch} | accuracy {accuracy_valid}| valid loss {loss_valid} | exp_loss {torch.Tensor([last_loss])}")

        accuracy_list_valid.append(accuracy_valid)
        list_loss_valid.append(last_loss)

    return list_loss_train, accuracy_list_valid, list_loss_valid
