from utils.read_dataset import read_h5
from DataLoaders.data_loaders import DatasetWileC
from utils.utils import set_device
from models import LSTM, LSTMattn
from tqdm import tqdm
import torch


def evaluation(model, loader):
    with torch.no_grad():
        total = 0
        acertos = 0
        for x, y in loader:
            y_hat = model(x.to(device))
            y_hat = torch.argmax(y_hat, dim=1)

            y = y.to(device)
            y_hat = y_hat.to(device)

            acertos += y[y.squeeze(1) == y_hat].shape[0]
            total += 1 * x.shape[0]

        mean_accuracy = acertos / total

        print(f'Accuracy: {mean_accuracy}')


input_dim = 100
lr = 1e-4
epochs = 5
device = set_device()

data_normal_train = read_h5('dataset_free/X_train_normal.h5')[:1]
data_failure_train = read_h5('dataset_free/X_train_failure.h5')[:1]

data_normal_valid = read_h5('dataset_free/X_val_normal.h5')[:1]
data_failure_valid = read_h5('dataset_free/X_val_failure.h5')[:1]

dataset_train = DatasetWileC(data_normal_train, data_failure_train, 100)
dataset_valid = DatasetWileC(data_normal_valid, data_failure_valid, 100)

train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=4096, shuffle=True)
valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=512, shuffle=False)

# model = LSTM(input_dim).to(device)
model = LSTMattn(input_dim, 20).to(device)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

list_loss_train = []

for epoch in range(epochs):
    model.train()
    train_loss = 0
    correct = 0
    epoch_loss = 0

    for i, sample_batch in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()

        inputs, labels = sample_batch[0], sample_batch[1]

        inputs = inputs.to(device)
        labels = labels.to(device)

        pred_labels = model(inputs)

        loss = criterion(pred_labels, labels)
        epoch_loss += loss.item()

        if i % 10 == 0:
            print(f'epoch = {epoch + 1:d}, iteration = {i:d}/{len(train_loader):d}, loss = {loss.item():.5f}')
            evaluation(model, valid_loader)

        loss.backward()
        optimizer.step()

    # print(f'Epoch {epoch + 1} finished! Loss: {epoch_loss / i}. Accuracy: {correct.item() / len(train_loader.dataset)}')
