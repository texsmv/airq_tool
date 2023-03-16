import numpy as np
import torch
import random
from torch.utils.data import Dataset
from .maugmentations import *
from scipy.fft import fft
from .augmentations import gen_aug

def getSubsequence(x, size, y = None):
    # y is one dimensional
    D, T = x.shape
    # x = x_in + np.random.uniform(-0.1, 0.1, (D, 1))
    b = random.randint(0, T - size)
    if y is None:
        return x[:, b: b + size]
    else:
        return x[:, b: b + size] , y[b: b + size]

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
    def __init__(self, X, y, subsequence_size, n_views=2):
        self.X = X
        self.y = y
        self.n_views = n_views
        self.subsequence_size = subsequence_size
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # return self.X[idx] + [getSubsequence(self.X, self.subsequence_size) for i in range(self.n_views)]
        return [getSubsequence(self.X[idx], self.subsequence_size) for i in range(self.n_views)], self.y[idx]
    

class SubsequencesFreqDataset(Dataset):
    def __init__(self, X, Acc, y, subsequence_size, n_views=2):
        self.X = X
        self.y = y
        self.n_views = n_views
        self.subsequence_size = subsequence_size
        self.Acc_res = np.linalg.norm(Acc, axis=1)
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        subsequences = []
        ffts = []
        for i in range(self.n_views):
            sub, acc = getSubsequence(self.X[idx], self.subsequence_size, y=self.Acc_res[idx])
            subsequences.append(sub.astype(np.float32))
            ffts.append(np.expand_dims(np.absolute(fft(acc)), axis=0).astype(np.float32))
        return subsequences, ffts, self.y[idx]


class AugmentationsFreqDataset(Dataset):
    def __init__(self, X, subsequence_size, n_views=2, aug_type=None, test=False):
        self.X = X
        # self.y = y
        self.n_views = n_views
        self.subsequence_size = subsequence_size
        # self.Acc_res = np.linalg.norm(Acc, axis=1)
        self.aug_type = aug_type
        self.test = test
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        subsequences = []
        ffts = []
        for i in range(self.n_views):
            # if self.aug_type == None or self.test:
            #     sub = getSubsequence(self.X[idx], self.subsequence_size)
            # else:
            sub = self.X[idx]
            sub = np.expand_dims(sub, axis=0)
            sub = torch.Tensor(sub)
            # print(sub.shape)
            # print(sub.shape)
            if self.aug_type != None:
                sub = gen_aug(sub, ssh_type = self.aug_type).numpy().squeeze()
            else:
                sub = sub.numpy().squeeze()
                # print(sub.shape)
                
                # acc = self.Acc_res[idx]
            
            subsequences.append(sub.astype(np.float32))
            # ffts.append(np.expand_dims(np.absolute(fft(acc)), axis=0).astype(np.float32))
        return subsequences

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
        return (
            self.X[idx].transpose([1, 0]).astype(np.float32), 
            self.weak_transformation(self.X[idx]).astype(np.float32), 
            self.strong_transformation(self.X[idx]).astype(np.float32)
        )