import h5py
import numpy as np
from typing import List

filename = "X_train_normal.h5"


def read_h5(filename: str) -> List[np.ndarray]:
    with h5py.File(filename, "r") as f:
        a_group_key = list(f.keys())[0]

        data = list(f[a_group_key])

        return data
