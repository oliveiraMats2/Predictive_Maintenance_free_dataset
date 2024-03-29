import argparse
from datasets import DatasetWileC, Dataset_UCI
from models.supervised.models import LSTM, LSTMattn
from save_models import SaveBestModel
from utils.utils import *
from tools_wandb import ToolsWandb
from tqdm import trange
import wandb

save_best_model = SaveBestModel()

DEVICE = set_device()

FACTORY_DICT = {
    "model": {
        "LSTM": LSTM,
        "LSTMattn": LSTMattn,
    },
    "dataset": {
        "DatasetWileC": DatasetWileC,
        "DatasetUCI": Dataset_UCI
    },
    "optimizer": {
        "Adam": torch.optim.Adam
    },
    "loss": {
        "CrossEntropyLoss": torch.nn.CrossEntropyLoss(),
    },
}


def get_dataset(dataset_configs):
    dataset = FACTORY_DICT["dataset"][list(dataset_configs)[0]](
        **dataset_configs[list(dataset_configs.keys())[0]]
    )

    return dataset


def experiment_factory(configs):
    train_dataset_configs = configs["train_dataset"]
    validation_dataset_configs = configs["valid_dataset"]
    test_dataset_configs = configs["test_dataset"]
    model_configs = configs["model"]
    optimizer_configs = configs["optimizer"]
    criterion_configs = configs["loss"]

    # Construct the dataloaders with any given transformations (if any)
    train_dataset = get_dataset(train_dataset_configs)
    validation_dataset = get_dataset(validation_dataset_configs)
    # test_dataset = get_dataset(test_dataset_configs)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=configs["train"]["batch_size"], shuffle=True
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=configs["valid"]["batch_size"], shuffle=False
    )
    # test_loader = torch.utils.data.DataLoader(
    #     test_dataset, batch_size=configs["test"]["batch_size"], shuffle=False
    # )

    # Build model
    if type(model_configs) == dict:
        model = FACTORY_DICT["model"][list(model_configs.keys())[0]](
            **model_configs[list(model_configs.keys())[0]]
        )
    else:
        model = FACTORY_DICT["model"][model_configs]()

    optimizer = FACTORY_DICT["optimizer"][list(optimizer_configs.keys())[0]](
        model.parameters(), **optimizer_configs[list(optimizer_configs.keys())[0]]
    )
    criterion = FACTORY_DICT["loss"][list(criterion_configs.keys())[0]]

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min'
    )

    return model, train_loader, validation_loader, optimizer, \
           criterion, scheduler


def run_train_epoch(model, optimizer, criterion, loader,
                    monitoring_metrics, epoch, valid_loader):
    model.to(DEVICE)
    model.train()
    correct = 0
    epoch_loss = 0
    acc = 0
    total = 0

    with trange(len(loader), desc='Train Loop') as progress_bar:
        for batch_idx, sample_batch in zip(progress_bar, loader):
            optimizer.zero_grad()

            inputs, labels = sample_batch[0], sample_batch[1]

            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            pred_labels = model(inputs)

            _, predicted = torch.max(pred_labels, 1)
            correct += (predicted == labels).float().sum()

            loss = criterion(pred_labels, labels[:, 0])
            epoch_loss += loss.item()

            acc = labels[labels.squeeze(1) == torch.argmax(pred_labels, dim=1)].shape[0]

            batch_size_len = labels.shape[0]
            # total += 1 * inputs.shape[0]

            progress_bar.set_postfix(
                desc=f'[epoch: {epoch + 1:d}], iteration: {batch_idx:d}/{len(train_loader):d}, loss: {loss.item():.5f} acc: {(acc / batch_size_len):.5f} '
            )

            loss.backward()
            optimizer.step()

            if configs['wandb']:
                wandb.log({'train_loss': loss})

        epoch_loss = (epoch_loss / len(loader))
        monitoring_metrics["loss"]["train"].append(epoch_loss)

    return epoch_loss


def run_training_experiment(model, train_loader, validation_loader, optimizer,
                            criterion, scheduler, configs
                            ):
    os.makedirs(configs["path_to_save_model"], exist_ok=True)

    monitoring_metrics = {
        "loss": {"train": [], "validation": []},
        "accuracy": {"train": [], "validation": []}
    }

    for epoch in range(1, configs["epochs"] + 1):
        train_loss = run_train_epoch(
            model, optimizer, criterion, train_loader, monitoring_metrics,
            epoch, validation_loader
        )
        # valid_loss = run_validation(
        #     model, optimizer, criterion, validation_loader, monitoring_metrics,
        #     epoch, batch_size=configs["batch_size"]
        # )
        scheduler.step(monitoring_metrics["loss"]["train"][-1])

    savingName = f'{configs["network"]}_epoch_{epoch}.pth'
    savingPath = os.path.join(configs["path_to_save_model"], savingName)
    torch.save(model.state_dict(), savingPath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="unsupervised main WileC")

    parser.add_argument(
        "config_file", type=str, help="Path to YAML configuration file"
    )

    args = parser.parse_args()

    configs = read_yaml(args.config_file)

    f_configurations = {}
    f_configurations = ToolsWandb.config_flatten(configs, f_configurations)

    model, train_loader, validation_loader, \
    optimizer, criterion, scheduler = experiment_factory(configs)

    if configs['wandb']:
        run = wandb.init(project="wileC_free_datasets",
                         reinit=True,
                         config=f_configurations,
                         notes="Testing wandb implementation",
                         entity="oliveira_mats")

    run_training_experiment(
        model, train_loader, validation_loader, optimizer,
        criterion, scheduler, configs
    )

    torch.cuda.empty_cache()
    wandb.finish()
