"""
Datasets implementation
"""

import torch
from torch.utils.data import Dataset


class SimpleDataset(Dataset):
    """
    Dataset for simply generated data
    """

    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return (
            torch.tensor(self.data[index])[..., None],
            torch.tensor(self.targets[index])[..., None],
        )
