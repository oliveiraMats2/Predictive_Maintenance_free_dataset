import torch
import os
from tqdm import tqdm


def set_device():
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)
    print('Using {}'.format(device))

    return device


def evaluate(model, loader, device):
    with torch.no_grad():
        total = 0
        acertos = 0
        for x, y in tqdm(loader):
            y_hat = model(x.to(device))
            y_hat = torch.argmax(y_hat, dim=1)

            y = y.to(device)
            y_hat = y_hat.to(device)

            acertos += y[y.squeeze(1) == y_hat].shape[0]
            total += 1 * x.shape[0]

        mean_accuracy = acertos / total

        print(f'Accuracy: {mean_accuracy}')


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
