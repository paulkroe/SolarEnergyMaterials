# taken mostly from microsofts graphormer implementaion

from torch_geometric.data import Dataset
import torch
import numpy as np
import copy
from functools import lru_cache
from sklearn.model_selection import train_test_split

class GraphormerPYGDataset(Dataset):
    def __init__(
            self,
            dataset: Dataset,
            seed: int = 0,
            train_ind = None,
            valid_ind = None,
            test_ind = None,
            train_set = None,
            valid_set = None,
            test_set = None,
            ):

        self.dataset = dataset #either MyPygGraphPropPredDataset, MyPygPCQM4MDataset or MyPygPCQM4Mv2Dataset