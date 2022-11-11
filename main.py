from utils.read_dataset import read_h5, read_csv_uci
from DataLoaders.data_loaders import DatasetWileC
from utils.utils import set_device, evaluate
from models import LSTM, LSTMattn
from tqdm import tqdm
import torch

configs = {'context_size': 15,
           'batch_size': 128,
           'batch_size_valid': 64,
           'learning_rate': 1e-5,
           'epochs': 2,
           'LSTM_config': {'hidden_dim': 128,
                           'num_layers': 15,
                           'output_dim': 2}}

device = set_device()

data_normal_train = read_h5('dataset_free/X_train_normal.h5')
data_failure_train = read_h5('dataset_free/X_train_failure.h5')

data_normal_valid = read_h5('dataset_free/X_val_normal.h5')[:1]
data_failure_valid = read_h5('dataset_free/X_val_failure.h5')[:1]

dataset_train = DatasetWileC(data_normal_train, data_failure_train, context=configs['context_size'])
dataset_valid = DatasetWileC(data_normal_valid, data_failure_valid, context=configs['context_size'])

train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=configs['batch_size'], shuffle=True)
valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=configs['batch_size_valid'], shuffle=False)
raise('pause')

model = LSTM(configs['context_size'],
             hidden_dim=configs['LSTM_config']['hidden_dim'],
             num_layers=configs['LSTM_config']['num_layers'],
             output_dim=configs['LSTM_config']['output_dim']
             ).to(device)

# model = LSTMattn(input_dim, 1024).to(device)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=configs['learning_rate'], weight_decay=1e-5)

list_loss_train = []

for epoch in range(configs['epochs']):
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

        if (i + 1) % 100 == 0:
            print(f'epoch = {epoch + 1:d}, iteration = {i:d}/{len(train_loader):d}, loss = {loss.item():.5f}')
            evaluate(model, valid_loader, device)

        loss.backward()
        optimizer.step()

    # print(f'Epoch {epoch + 1} finished! Loss: {epoch_loss / i}. Accuracy: {correct.item() / len(train_loader.dataset)}')
