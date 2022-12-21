import numpy as np
import argparse
from utils.utils import read_yaml
import torch
from tools_wandb import ToolsWandb
from datasets import DatasetWileC, Dataset_UCI, DatasetUnsupervisedMafaulda
from models.unsupervised.models import TimeSeriesTransformers
from losses import smape_loss
from utils.utils import set_device
from tqdm import trange

DEVICE = set_device()

FACTORY_DICT = {
    "model": {
        "TimeSeriesTransformers": TimeSeriesTransformers
    },
    "dataset": {
        "DatasetWileC": DatasetWileC,
        "DatasetUCI": Dataset_UCI,
        "DatasetUnsupervisedMafaulda": DatasetUnsupervisedMafaulda
    },
    "optimizer": {
        "Adam": torch.optim.Adam
    },
    "loss": {
        "CrossEntropyLoss": torch.nn.CrossEntropyLoss(),
        "smape_loss": smape_loss
    },
}


def get_dataset(dataset_configs):
    dataset = FACTORY_DICT["dataset"][list(dataset_configs)[0]](
        **dataset_configs[list(dataset_configs.keys())[0]]
    )

    return dataset


def eval_epoch(model, criterion, loader, epoch):
    model.to(DEVICE)
    model.eval()

    epoch_loss = 0

    with trange(len(loader), desc='Test Loop') as progress_bar:
        for batch_idx, sample_batch in zip(progress_bar, loader):

            inputs, labels = sample_batch[0], sample_batch[1]

            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            pred_labels = model(inputs)

            loss = criterion(pred_labels, labels)
            epoch_loss += loss.item()

            progress_bar.set_postfix(
                desc=f'[epoch: {epoch + 1:d}], iteration: {batch_idx:d}/{len(loader):d}, loss: {loss.item():.5f}'
            )

        print(f"\n --- Mean loss {epoch_loss/len(loader)}")


def experiment_factory(configs):
    test_dataset_configs = configs["test_dataset"]
    model_configs = configs["model"]
    criterion_configs = configs["loss"]

    # Construct the dataloaders with any given transformations (if any)
    test_dataset = get_dataset(test_dataset_configs)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=configs["test"]["batch_size"], shuffle=False
    )

    # Build model
    if type(model_configs) == dict:
        model = FACTORY_DICT["model"][list(model_configs.keys())[0]](
            **model_configs[list(model_configs.keys())[0]]
        )
    else:
        model = FACTORY_DICT["model"][model_configs]()

    criterion = FACTORY_DICT["loss"][list(criterion_configs.keys())[0]]

    return model, test_loader, criterion


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="unsupervised main WileC")

    parser.add_argument(
        "config_file", type=str, help="Path to YAML configuration file"
    )

    args = parser.parse_args()

    configs = read_yaml(args.config_file)

    f_configurations = {}
    f_configurations = ToolsWandb.config_flatten(configs, f_configurations)

    model, test_loader, criterion = experiment_factory(configs)

    name_model = f"{configs['path_to_save_model']}{configs['network']}.pt"

    load_dict = torch.load(name_model)

    model.load_state_dict(load_dict['model_state_dict'])

    eval_epoch(model, criterion, test_loader, 0)


