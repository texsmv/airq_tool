import numpy as np
import torch
from torch.utils.data import Dataset
import random

class ContrastiveDataset(Dataset):
    def __init__(self, X, y, use_label = False):
        self.X = X
        self.y = y
        self.classes = np.unique(y)
        self.classes_samples = {}
        self.use_label = use_label
        for i in range(len(self.classes)):
            c = self.classes[i]
            samples = np.where(c == y)[0]
            self.classes_samples[c] = samples
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.use_label:
            idx_c = self.y[idx][0]
            o_c = random.choice(np.delete(self.classes, idx_c))
            
            samples = self.classes_samples[o_c]
            
            rand_id = random.randint(0, len(samples) - 1)
            negative = self.X[samples[rand_id]]
            negative_label = self.y[samples[rand_id]]
        else:
            rand_id = random.randint(0, len(self.X) - 1)
            negative = self.X[rand_id]
            negative_label = self.y[rand_id]
        return self.X[idx], negative, self.y[idx], negative_label