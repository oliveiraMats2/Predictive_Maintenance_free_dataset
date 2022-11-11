from utils.read_dataset import read_h5, read_csv_uci
from DataLoaders.data_loaders import DatasetWileC, Dataset_UCI
from utils.utils import set_device, evaluate, create_context, read_yaml
from sklearn.model_selection import train_test_split
from models import LSTM, LSTMattn
from tqdm import tqdm
from tqdm import trange
import torch

configs = read_yaml('configs/config_model.yaml')

device = set_device()

X, y = read_csv_uci('dataset_free/uci_base_machine_learning.csv')

X, y = create_context(X,
                      y,
                      configs['context_size'])

x_train, x_valid, y_train, y_valid = train_test_split(X,
                                                      y,
                                                      test_size=configs['train_test_split']['test_size'],
                                                      random_state=configs['train_test_split']['random_state'])

dataset_train = Dataset_UCI(x_train, y_train)
dataset_valid = Dataset_UCI(x_valid, y_valid)

train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=configs['batch_size_train'], shuffle=True)
valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=configs['batch_size_valid'], shuffle=False)

model = LSTMattn(configs['context_size'],
                 hidden_dim=configs['LSTM_config']['hidden_dim'],
                 num_layers=configs['LSTM_config']['num_layers'],
                 output_dim=configs['LSTM_config']['output_dim']
                 ).to(device)

# criterion = torch.nn.MSELoss()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=configs['learning_rate'], weight_decay=1e-5)

list_loss_train = []

for epoch in range(configs['epochs']):
    model.train()
    train_loss = 0
    correct = 0
    epoch_loss = 0
    with trange(len(train_loader), desc='Train Loop') as progress_bar:
        for batch_idx, sample_batch in zip(progress_bar, train_loader):
            optimizer.zero_grad()

            inputs, labels = sample_batch[0], sample_batch[1]

            inputs = inputs.to(device)
            labels = labels.to(device)

            pred_labels = model(inputs)

            _, predicted = torch.max(pred_labels, 1)
            correct += (predicted == labels).float().sum()

            loss = criterion(pred_labels, labels[:, 0])
            epoch_loss += loss.item()

            progress_bar.set_postfix(
                desc=f'[epoch: {epoch + 1:d}], iteration: {batch_idx:d}/{len(train_loader):d}, loss: {loss.item():.5f}'
            )

            loss.backward()
            optimizer.step()

evaluate(model, valid_loader, device)
