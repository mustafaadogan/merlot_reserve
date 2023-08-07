import os
import os.path as osp
import json
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import Dataset
from .utils import process_path


class BaseDataset(Dataset):
    """
    Only loads the JSON annotations.
    """
    def __init__(self, json_path):
        json_path = process_path(json_path)
        assert osp.isfile(json_path)
        with open(json_path, 'r') as f:
            self.json_data = json.load(f)
        self.ids = list(self.json_data.keys())
        self.ids.sort()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        item_id = self.ids[index]
        return item_id, self.json_data[item_id]

