import argparse
import os
import shutil
from tqdm import trange

import wandb
from datasets import DatasetUnsupervisedMafaulda, DatasetSinteticUnsupervised, DatasetSinteticUnsupervisedLSTM
from datasets import DatasetWileC, Dataset_UCI
from losses import smape_loss, soft_dtw
from models.unsupervised.models import TimeSeriesTransformers, LstmModel, LstmModelConv
from save_models import SaveBestModel
from tools_wandb import ToolsWandb
from utils.utils import *

save_best_model = SaveBestModel()

DEVICE = set_device()

FACTORY_DICT = {
    "model": {
        "TimeSeriesTransformers": TimeSeriesTransformers,
        "LstmModel": LstmModel,
        "LstmModelConv": LstmModelConv
    },
    "dataset": {
        "DatasetWileC": DatasetWileC,
        "DatasetUCI": Dataset_UCI,
        "DatasetUnsupervisedMafaulda": DatasetUnsupervisedMafaulda,
        "DatasetSinteticUnsupervised": DatasetSinteticUnsupervised,
        "DatasetSinteticUnsupervisedLSTM": DatasetSinteticUnsupervisedLSTM
    },
    "optimizer": {
        "Adam": torch.optim.Adam
    },
    "loss": {
        "CrossEntropyLoss": torch.nn.CrossEntropyLoss(),
        "smape_loss": smape_loss,
        "MSELoss": torch.nn.MSELoss(),
        "HingeLoss": torch.nn.HingeEmbeddingLoss(),
        "KullbackLeibler": torch.nn.KLDivLoss(reduction='batchmean'),
        "soft_dtw": soft_dtw
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
    # test_dataset_configs = configs["test_dataset"]
    model_configs = configs["model"]
    optimizer_configs = configs["optimizer"]
    criterion_configs = configs["loss"]

    # Construct the dataloaders with any given transformations (if any)
    train_dataset = get_dataset(train_dataset_configs)
    validation_dataset = get_dataset(validation_dataset_configs)
    # test_dataset = get_dataset(test_dataset_configs)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=configs["train_batch_size"], shuffle=False
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=configs["valid_batch_size"], shuffle=False
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
                    monitoring_metrics, epoch, valid_loader, scheduler, run):
    model.to(DEVICE)
    model.train()

    epoch_loss = 0

    with trange(len(loader), desc='Train Loop') as progress_bar:
        for batch_idx, sample_batch in zip(progress_bar, loader):
            optimizer.zero_grad()

            inputs, labels = sample_batch[0], sample_batch[1]

            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            pred_labels = model(inputs)

            try:# concertar mais tarde.
                loss = criterion(pred_labels[:, -1, :], labels)
            except:
                loss = criterion(pred_labels, labels[:, 0].unsqueeze(1))

            epoch_loss += loss.item()

            progress_bar.set_postfix(
                desc=f'[epoch: {epoch + 1:d}], iteration: {batch_idx:d}/{len(train_loader):d}, loss: {loss.item():.5f}'
            )

            loss.backward()
            optimizer.step()

            if configs['wandb']:
                wandb.log({'train_loss': loss})

            if (batch_idx + 1) % 10000 == 0:
                print("Atualizar schedule loss")
                scheduler.step(loss)

            name_model = f"{configs['path_to_save_model']}{configs['network']}_{configs['reload_model']['data']}.pt"

            save_best_model(loss,
                            batch_idx,
                            model, optimizer, criterion, name_model, run)

            # if (batch_idx + 1) % configs['evaluate_step'] == 0:
            #     epoch_acc = evaluate(model, valid_loader, DEVICE)

        epoch_loss = (epoch_loss / len(loader))
        monitoring_metrics["loss"]["train"].append(epoch_loss)

    return epoch_loss


def calculate_parameters(model):
    qtd_model = sum(p.numel() for p in model.parameters())
    print(f"quantidade de parametros: {qtd_model}")
    return


def run_training_experiment(model, train_loader, validation_loader, optimizer,
                            criterion, scheduler, configs, run
                            ):
    os.makedirs(configs["path_to_save_model"], exist_ok=True)

    monitoring_metrics = {
        "loss": {"train": [], "validation": []},
        "accuracy": {"train": [], "validation": []}
    }

    calculate_parameters(model)

    for epoch in range(0, configs["epochs"]):
        train_loss = run_train_epoch(
            model, optimizer, criterion, train_loader, monitoring_metrics,
            epoch, validation_loader, scheduler, run
        )

        # valid_loss = run_validation(
        #     model, optimizer, criterion, validation_loader, monitoring_metrics,
        #     epoch, batch_size=configs["batch_size"]
        # )
        scheduler.step(monitoring_metrics["loss"]["train"][-1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="unsupervised main WileC")

    parser.add_argument(
        "config_file", type=str, help="Path to YAML configuration file"
    )

    print("============ Delete .wandb path ============")
    try:
        shutil.rmtree("wandb/")
    except:
        raise ValueError("especific directory .wandb")



    args = parser.parse_args()

    configs = read_yaml(args.config_file)

    f_configurations = {}
    f_configurations = ToolsWandb.config_flatten(configs, f_configurations)

    model, train_loader, validation_loader, \
        optimizer, criterion, scheduler = experiment_factory(configs)

    if configs['reload_model']['type']:
        name_model = f"{configs['path_to_save_model']}backup/{configs['network']}_{configs['reload_model']['data']}.pt"

        load_dict = torch.load(name_model)

        model.load_state_dict(load_dict['model_state_dict'])

    run = None

    if configs['wandb']:
        run = wandb.init(project="wile_c_machine_learning_team",
                         reinit=True,
                         config=f_configurations,
                         notes="Testing wandb implementation",
                         entity="oliveira_mats",
                         dir=None)

    run_training_experiment(
        model, train_loader, validation_loader, optimizer,
        criterion, scheduler, configs, run
    )

    torch.cuda.empty_cache()
    wandb.finish()
