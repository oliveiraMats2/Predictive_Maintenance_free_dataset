import numpy as np
from typing import List

import pandas as pd
import torch
from tqdm import tqdm
from utils.read_dataset import ReadDatasets


class Dataset_UCI:
    def __init__(self, data: np.ndarray, labels: np.ndarray):
        self.data_context, self.context_labels = data, labels

    def __len__(self) -> int:
        return len(self.context_labels)

    def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor):
        return torch.Tensor(self.data_context[idx]), torch.LongTensor([self.context_labels[idx]])


class DatasetSinteticUnsupervised:
    def __init__(self, **kargs):
        dir_data = kargs["dir_data"]
        context = kargs["context"]
        stride = kargs["stride"]

        self.data = ReadDatasets.read_h5(dir_data)

        self.data = np.array(self.data).reshape(-1, 4)

        self.len_data = self.data.shape[0]

        self.context_data = []
        self.labels_data = []

        for i in tqdm(range(self.len_data - context)):
            self.context_data.append(self.data[i:i + context, :])
            self.labels_data.append(self.data[i + stride:i + context + stride])

        print(f"len dataset:{len(self.context_data)}")

    def __len__(self) -> int:
        return len(self.context_data)

    def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor):
        # return torch.Tensor((self.context_data[idx])), torch.Tensor([self.labels_data[idx]])
        return torch.Tensor((self.context_data[idx])), torch.Tensor(self.labels_data[idx])


class DatasetSinteticUnsupervisedLSTM:
    def __init__(self, **kargs):
        dir_data = kargs["dir_data"]
        context = kargs["context"]
        stride = kargs["stride"]

        self.data = ReadDatasets.read_h5(dir_data)

        self.data = np.array(self.data).reshape(-1, 1)

        self.len_data = self.data.shape[0]

        self.context_data = []
        self.labels_data = []

        for i in tqdm(range(self.len_data - context)):
            self.context_data.append(self.data[i:i + context, :])

        print(f"len dataset:{len(self.context_data)}")

    def __len__(self) -> int:
        return len(self.context_data)

    def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor):
        # return torch.Tensor((self.context_data[idx])), torch.Tensor([self.labels_data[idx]])
        return torch.Tensor((self.context_data[idx])), torch.Tensor(self.context_data[idx][-1])


class DatasetUnsupervisedMafaulda:
    def __init__(self, **kargs):
        data_normal = kargs['data_normal']
        data_failure = kargs['data_failure']
        context = kargs['context']

        data_normal = ReadDatasets.read_h5(data_normal)[:1]  # debug
        data_failure = ReadDatasets.read_h5(data_failure)[:1]  # debug

        self.data_normal = np.array(data_normal).reshape(-1, 8)
        self.data_failure = np.array(data_failure).reshape(-1, 8)

        self.len_data_normal = self.data_normal.shape[0]
        self.len_data_failure = self.data_failure.shape[0]

        self.len_data = self.len_data_normal + self.len_data_failure

        self.data = np.concatenate([self.data_normal, self.data_failure])
        # self.data = self.data_normal + self.data_failure

        self.context_data = []
        self.labels_data = []

        stride = 1

        for i in tqdm(range(self.len_data - context - stride)):
            self.context_data.append(self.data[i:i + context])
            self.labels_data.append(self.data[i + stride:i + context + stride])

    def __len__(self) -> int:
        return len(self.context_data)

    def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor):
        # return torch.Tensor((self.context_data[idx])), torch.Tensor([self.labels_data[idx]])
        return torch.Tensor((self.context_data[idx])), torch.Tensor(self.labels_data[idx])


class DatasetWileC:
    def __init__(self, **kargs):
        data_normal = kargs['data_normal']
        data_failure = kargs['data_failure']
        context = kargs['context']

        data_normal = ReadDatasets.read_h5(data_normal)
        data_failure = ReadDatasets.read_h5(data_failure)

        self.data_normal = np.array(data_normal).reshape(-1, 8)
        self.data_failure = np.array(data_failure).reshape(-1, 8)

        self.len_data_normal = self.data_normal.shape[0]
        self.len_data_failure = self.data_failure.shape[0]

        data_normal_target = np.array([0] * self.len_data_normal)
        data_failure_target = np.array([1] * self.len_data_failure)

        self.len_data = self.len_data_normal + self.len_data_failure

        self.data = np.concatenate([self.data_normal, self.data_failure])

        self.data_target = np.concatenate([data_normal_target, data_failure_target])

        self.context_data = []

        for i in tqdm(range(self.len_data - context)):
            self.context_data.append(self.data[i:i + context, :])

    def __len__(self) -> int:
        return len(self.context_data)

    def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor):
        return torch.Tensor(np.array(self.context_data[idx])), torch.LongTensor([self.data_target[idx]])


class DatasetProphetTransformer:
    def __init__(self, **kargs):
        dir_data = kargs["dir_data"]
        self.context = kargs["context"]
        stride = kargs["stride"]

        df_prophet = pd.read_csv(dir_data)
        self.data = np.array(df_prophet["yhat_upper"].tolist())

        self.context_data = []
        self.labels_data = []

        self.len_data = len(self.data)

        for i in tqdm(range(self.len_data - self.context)):
            self.context_data.append(self.data[i:i + self.context])
            self.labels_data.append(self.data[i + stride:i + self.context + stride])

        print(f"len dataset:{len(self.context_data)}")

    def __len__(self) -> int:
        return len(self.context_data)

    def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor):
        # return torch.Tensor((self.context_data[idx])), torch.Tensor([self.labels_data[idx]])
        return torch.Tensor((self.context_data[idx])).unsqueeze(1).view(-1, self.context),\
            torch.Tensor(self.labels_data[idx])


class DatasetProphetTest:
    def __init__(self, **kargs):
        dir_data = kargs["dir_data"]
        context = kargs["context"]
        stride = kargs["stride"]

        df_prophet = pd.read_csv(dir_data)
        self.data = np.array(df_prophet["yhat_upper"].tolist())

        self.context_data = []
        self.labels_data = []

        self.len_data = len(self.data)

        for i in tqdm(range(self.len_data - context)):
            self.context_data.append(self.data[i:i + context])
            self.labels_data.append(self.data[i + stride:i + context + stride])

        print(f"len dataset:{len(self.context_data)}")

    def __len__(self) -> int:
        return len(self.context_data)

    def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor):
        # return torch.Tensor((self.context_data[idx])), torch.Tensor([self.labels_data[idx]])
        return torch.Tensor((self.context_data[idx])).unsqueeze(1), torch.Tensor(self.labels_data[idx])


class DatasetTest:
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass
