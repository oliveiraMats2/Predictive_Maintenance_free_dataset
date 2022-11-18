from utils.read_dataset import read_h5
from DataLoaders.data_loaders import DatasetWileC
from matrix_confusion import plot_confusion_matrix
from utils.utils import set_device, evaluate, create_context, read_yaml, config_flatten, evaluate_matrix_confusion
from models import LSTM, LSTMattn
from tqdm import trange
from save_models import SaveBestModel
import torch, wandb

configs = read_yaml('configs/config_model.yaml')

device = set_device()

save_best_models = SaveBestModel('moodels_h5/')

f_configurations = {}
f_configurations = config_flatten(configs, f_configurations)
if configs['wandb']:
    run = wandb.init(project="wileC_free_datasets",
                     reinit=True,
                     config=f_configurations,
                     notes="Testing wandb implementation",
                     entity="oliveira_mats")

data_normal_train = read_h5('dataset_free/X_train_normal.h5')
data_failure_train = read_h5('dataset_free/X_train_failure.h5')

data_normal_valid = read_h5('dataset_free/X_val_normal.h5')[:1]
data_failure_valid = read_h5('dataset_free/X_val_failure.h5')[:1]

dataset_train = DatasetWileC(data_normal_train, data_failure_train, context=configs['context_size'])
dataset_valid = DatasetWileC(data_normal_valid, data_failure_valid, context=configs['context_size'])

train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=configs['batch_size_train'], shuffle=True)
valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=configs['batch_size_valid'], shuffle=False)

model = LSTMattn(configs['context_size'],
                 hidden_dim=configs['LSTM_config']['hidden_dim'],
                 num_layers=configs['LSTM_config']['num_layers'],
                 output_dim=configs['LSTM_config']['output_dim']
                 ).to(device)

parameters = sum(p.numel() for p in model.parameters())


criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=configs['learning_rate'],
                             weight_decay=1e-5)

list_loss_train = []
evaluate_step = 5000

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

            save_best_models(loss, epoch, model,
                             optimizer, criterion, configs['name_model'])

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

                mean_accuracy = acc / total

            loss.backward()
            optimizer.step()

            if configs['wandb']:
                wandb.log({'train_loss': loss})

            if (batch_idx + 1) % evaluate_step == 0:
                epoch_acc = evaluate(model, valid_loader, device)

                print(f"acc{epoch_acc}")

                if configs['wandb']:
                    wandb.log({'valid_acc': epoch_acc})

from utils.utils import evaluate

epoch_acc = evaluate(model, valid_loader, device)
cm = evaluate_matrix_confusion(model.to('cpu'), valid_loader, 'cpu')  # use estrategies class staticmethods

print(f"epoca acuraccy {epoch_acc}")

plot_confusion_matrix(cm, ["não defeito", "defeito"], dir_artifacts='artifacts/matrix_confusion.png')

if configs['wandb']:
    artifact = wandb.Artifact('artifacts', type='dataset')
    artifact.add_dir('artifacts')  # Adds multiple files to artifact
    wandb.log_artifact(artifact)

    wandb.finish()
