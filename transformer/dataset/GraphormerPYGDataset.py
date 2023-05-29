# taken mostly from microsofts graphormer implementaion

from torch_geometric.data import Dataset
import torch
import numpy as np
import copy
from functools import lru_cache
from sklearn.model_selection import train_test_split
import transformer.dataset.algorithms
from transformer.dataset.wrapper import preprocess_item

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
        if self.dataset is not None:
            self.num_data = len(self.dataset)
        self.seed = seed
        
        if train_ind is None and train_set is None: # checks if no train spilt is given
            train_valid_ind, test_ind = train_test_split(
                np.arange(self.num_data),
                test_size=self.num_data//10,
                random_state=self.seed
            )
            train_ind, valid_ind = train_test_split(
                np.arrange(train_valid_ind),
                test_size=train_valid_ind//5,
                random_state=self.seed
            )
            self.train_ind = torch.from_numpy(train_ind)
            self.valid_ind = torch.from_numpy(valid_ind)
            self.test_ind = torch.from_numpy(test_ind)
            self.train_data = self.index_select(self.train_ind)
            self.valid_data = self.index_select(self.valid_ind)
            self.test_data = self.index_select(self.test_ind)
        elif train_set is not None:
            self.num_data = len(train_set) + len(valid_set) + len(test_set)
            self.train_data = self.create_subset(train_set)
            self.valid_data = self.create_subset(valid_set)
            self.test_data = self.create_subset(test_set)
            self.train_ind = None
            self.valid_ind = None
            self.test_ind = None
        else: #!
            self.num_data = len(train_ind) + len(valid_ind) + len(test_ind)
            self.train_ind = train_ind
            self.valid_ind = valid_ind
            self.test_ind = test_ind
            self.train_data = self.index_select(self.train_ind)
            self.valid_data = self.index_select(self.valid_ind)
            self.test_data = self.index_select(self.test_ind)
        self.__indices__ = None

    def create_subset(self, subset):
        dataset = copy.copy(self)
        dataset.dataset = subset
        dataset.num_data = len(subset)
        dataset.train_data = None
        dataset.valid_data = None
        dataset.test_data = None
        dataset.train_ind = None
        dataset.valid_ind = None
        dataset.test_ind = None
        return dataset

    def index_select(self, ind):
        dataset = copy.copy(self)
        dataset.dataset = self.dataset.index_select(ind) # using index_select from Dataset
        if isinstance(ind, torch.Tensor):
            dataset.num_data = ind.size(0)
        else:
            dataset.num_data = ind.shape[0]
        dataset.__indices__ = ind
        dataset.train_data = None
        dataset.valid_data = None
        dataset.test_data = None
        dataset.train_ind = None
        dataset.valid_ind = None
        dataset.test_ind = None
        return dataset

    @lru_cache(maxsize=16)
    def __getitem__(self, ind):
        if isinstance(ind, int):
            item = self.dataset[ind]
            item.idx = ind
            item.y = item.y.reshape(-1)
            return preprocess_item(item)
        else:
            raise IndexError("Only integer indices supported")
    def __len__(self):
        return self.num_data