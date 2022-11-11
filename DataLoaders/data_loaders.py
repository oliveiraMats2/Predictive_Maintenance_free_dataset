import numpy as np
from typing import List
import torch
from tqdm import tqdm


class Dataset_UCI:
    def __init__(self, data, labels, context: int = 10):
        self.samples = data.shape[1]

        self.data_context = []
        self.context_labels = []

        for i in range(self.samples - context):
            self.data_context.append(data[:, i:i + context])

        for i in range(self.samples - context):
            self.context_labels.append(labels[i:i + context])

    def __len__(self) -> int:
        return len(self.context_labels)

    def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor):
        return torch.Tensor(np.array(self.data_context[idx])), torch.Tensor([self.context_labels[idx]])



class DatasetWileC:
    def __init__(self, data_normal: List[np.ndarray], data_failure: List[np.ndarray], context: int = 10):
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

    def __len__(self):
        return len(self.context_data)

    def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor):
        return torch.Tensor(np.array(self.context_data[idx])), torch.Tensor([self.data_target[idx]])
