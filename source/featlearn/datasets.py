import numpy as np
import torch
import random
from torch.utils.data import Dataset
from .augmentations import *


def getSubsequence(x, size):
    D, T = x.shape
    b = random.randint(0, T - size) 
    return x[:, b: b + size]

    # elif mode == 'shape':
    #     slides = []
    #     for i in range(B):
    #         slide = batch[i,:, b[i]: b[i] + size]
    #         for j in range(D):
    #             slide[j] = slide[j] + random.uniform(-0.1, 0.1)
    #         slides.append(slide)
    #     return np.array(slides).astype(np.float32)

# # mode = subsequences or shape
# def getViews(batch, size, isNumpy = False, mode = 'subsequences'):
#     originalSlides = getRandomSlides(batch, size, isNumpy, mode = mode)
    
#     fslides = originalSlides.transpose((0, 2, 1))
#     # scaled = aug.scaling(fslides, sigma=0.1).transpose((0, 2, 1))
#     scaled = fslides.transpose((0, 2, 1))
    
#     originalSlides = getRandomSlides(batch, size, isNumpy, mode = mode)
#     fslides = originalSlides.transpose((0, 2, 1))
#     # flipped = aug.rotation(fslides).transpose((0, 2, 1))
#     flipped = fslides.transpose((0, 2, 1))
    
#     originalSlides = getRandomSlides(batch, size, isNumpy, mode = mode)
#     fslides = originalSlides.transpose((0, 2, 1))
#     # magWarped =  aug.magnitude_warp(fslides, sigma=0.2, knot=4).transpose((0, 2, 1))
#     magWarped =  fslides.transpose((0, 2, 1))
    
#     originalSlides = getRandomSlides(batch, size, isNumpy, mode = mode)
#     # fslides = originalSlides.transpose((0, 2, 1))
    
#     # return torch.from_numpy(originalSlides.astype(np.float32))
#     return [
#         torch.from_numpy(originalSlides.astype(np.float32)),
#         torch.from_numpy(scaled.astype(np.float32)),
#         torch.from_numpy(flipped.astype(np.float32)),
#         torch.from_numpy(magWarped.astype(np.float32)),
#     ]

class SubsequencesDataset(Dataset):
    def __init__(self, X, subsequence_size, n_views=2):
        self.X = X
        self.n_views = n_views
        self.subsequence_size = subsequence_size
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # return self.X[idx] + [getSubsequence(self.X, self.subsequence_size) for i in range(self.n_views)]
        return [getSubsequence(self.X[idx], self.subsequence_size) for i in range(self.n_views)]
    

class AugmentationDataset(Dataset):
    def __init__(self, X):
        self.X = X.transpose([0, 2, 1])
    def __len__(self):
        return len(self.X)

    def weak_transformation(self, x):
        return jitter(x, sigma=0.03).transpose([1, 0])

    def strong_transformation(self, x):
        return permutation(x).transpose([1, 0])

    def __getitem__(self, idx):
        return (self.X[idx].transpose([1, 0]).astype(np.float32), self.weak_transformation(self.X[idx]).astype(np.float32), self.strong_transformation(self.X[idx]).astype(np.float32))