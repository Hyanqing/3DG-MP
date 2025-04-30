import torch
from torch_geometric.data import (Data, InMemoryDataset)
from torch_geometric.data.collate import collate
from itertools import repeat
import os
from os.path import join
import pandas as pd

class MoleculeDatasetnew(InMemoryDataset):
    def __init__(self, data_list):
        super().__init__()
        self.data_list = data_list
        self.data, self.slices = self.collate()
        self._indices = list(range(len(data_list)))
        # super(Dataset, self)

    def collate(self):
        data, slices, _ = collate(
            self.data_list[0].__class__,
            data_list=self.data_list,
            increment=False,
            add_batch=False,
        )
        return data, slices

    def get(self, idx):
        data = Data()

        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]
        return data

class MoleculeDatasetComplete(InMemoryDataset):
    def __init__(self, root, path, dataset='zinc250k', transform=None,
                 pre_transform=None, pre_filter=None, empty=False):

        self.root = root
        self.dataset = dataset
        self.transform = transform
        self.pre_filter = pre_filter
        self.pre_transform = pre_transform

        super(MoleculeDatasetComplete, self).__init__(root, transform, pre_transform, pre_filter)

        if not empty:
            self.data, self.slices = torch.load(os.path.join(self.root, path))

            # self.data.x = torch.hstack((self.data.x, self.data.positions * 10 + 500))
            self.data.x = self.data.x.long()
        print('Dataset: {}\nData: {}'.format(self.dataset, self.data))

    def get(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]
        return data