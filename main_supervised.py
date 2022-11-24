from tqdm import trange
import argparse
from DataLoaders.data_loaders import DatasetWileC
from matrix_confusion import plot_confusion_matrix
from models.supervised.models import LSTM, LSTMattn
from save_models import SaveBestModel
import torch
from utils.read_dataset import read_h5
from utils.utils import *
from tools_wandb import ToolsWandb
import wandb

save_best_model = SaveBestModel()

DEVICE = set_device()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="unsupervised main WileC")

    parser.add_argument(
        "config_file", type=str, help="Path to YAML configuration file"
    )

    args = parser.parse_args()

    configs = read_yaml(args.config_file)

    f_configurations = {}
    f_configurations = ToolsWandb.config_flatten(configs, f_configurations)

    if configs['wandb']:
        run = ToolsWandb.init_wandb_run(f_configurations)

