import numpy as np
from typing import List
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


class DatasetUnsupervisedMafaulda:
    def __init__(self, **kargs):
        data_normal = kargs['data_normal']
        data_failure = kargs['data_failure']
        context = kargs['context']

        data_normal = ReadDatasets.read_h5(data_normal)[:1] #debug
        data_failure = ReadDatasets.read_h5(data_failure)[:1] #debug

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
        #return torch.Tensor((self.context_data[idx])), torch.Tensor([self.labels_data[idx]])
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


class DatasetTest:
    def __init__(self):
        pass
    def __len__(self):
        pass
    def __getitem__(self, item):
        pass