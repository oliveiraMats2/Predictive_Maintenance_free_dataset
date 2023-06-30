# import torch
import matplotlib.pyplot as plt
import os

import yaml
from tqdm import tqdm
import numpy as np
from typing import List
from sklearn.metrics import confusion_matrix

def read_yaml(file: str) -> yaml.loader.FullLoader:
    with open(file, "r") as yaml_file:
        configurations = yaml.load(yaml_file, Loader=yaml.FullLoader)

    return configurations
