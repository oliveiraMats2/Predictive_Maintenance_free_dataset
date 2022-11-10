from utils.read_dataset import read_h5
from DataLoaders.data_loaders import DatasetWileC
from models import LSTM
import numpy as np
import torch

sequence = 30

data_normal = read_h5('X_train_normal.h5')[:1]
data_failure = read_h5('X_train_failure.h5')[:1]

dataset_train = DatasetWileC(data_normal, data_failure, 10)
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=6, shuffle=False)

input_dim = 8
lr = 1e-4
epochs = 5

model = LSTM(input_dim)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    for i, sample_batch in enumerate(train_loader):

        inputs, labels = sample_batch[0], sample_batch[0]

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        # multi-task loss
        loss = 0
        for lx in range(len(outputs)):
            loss += criterion(outputs[lx], labels[:, lx])
        epoch_loss += loss.item()

        if i % 10 == 0:
            print(f'epoch = {epoch + 1:d}, iteration = {i:d}/{len(train_loader):d}, loss = {loss.item():.5f}')
            writer.add_scalar('train_loss_iter', loss.item(), i + len(train_loader) * epoch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch finished ! Loss: {epoch_loss / i}')
    # training set accuracy
    accuracy = eval_batch(model, train_loader, n_labels=len(targets), gpu=gpu)
    print(f'Accuracy = {accuracy}')