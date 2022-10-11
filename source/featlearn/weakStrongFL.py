import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from .datasets import AugmentationDataset
from torch.utils.data import DataLoader
from torch_snippets import *
from ..models.contrastive.losses import TripletLoss
from ..utils import ValueLogger
import matplotlib.pyplot as plt

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride = 1, padding_mode='replicate', padding = padding)
        self.bn = nn.BatchNorm1d(out_channels)
    def forward(self, x):
        x = self.conv(x)
        x = F.gelu(x)
        x = self.bn(x)
        return x


class EncoderCNN(nn.Module):
    def __init__(self, in_channels, filters = [16, 16, 16], kernels = [5, 5, 5]):
        super(EncoderCNN, self).__init__()
        
        convs  = []
        curr_channels = in_channels
        self.n_conv = len(filters)
        for i in range(self.n_conv):
            k = kernels[i]
            p = k // 2
            convs.append(ConvBlock(curr_channels, filters[i], k, p))
            curr_channels = filters[i]
        self.convs = nn.ModuleList(convs)
        self.m = nn.MaxPool1d(2)
        
    def forward(self, x):
        for i in range(self.n_conv):
            x  = self.convs[i](x)
            x  = self.m(x)
        return x


class HeadModel(nn.Module):
    def __init__(self, dim_in, head='mlp',  feat_dim=128):
        super(HeadModel, self).__init__()
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(dim_in),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        x = self.head(x)
        x = F.normalize(x, dim=1)
        return x

class SiameseNetwork(nn.Module):
    def __init__(self, in_channels, time_length, filters = [16, 16, 16], kernels = [5, 5, 5], feature_size=1024, encoding_size = 8, head='linear'):
        super(SiameseNetwork, self).__init__()
        
        self.features = EncoderCNN(in_channels, filters=filters, kernels=kernels)
        
        self.encoder_out_size = (time_length// 2 ** len(filters)) * filters[-1]
        self.dense = nn.Linear(self.encoder_out_size, feature_size)
        self.head = HeadModel(feature_size, head = head, feat_dim=encoding_size)
        
    def forward(self, x):
        x = self.features(x)
        
        # Get Representations
        x = torch.flatten(x, start_dim=1)
        x = self.dense(x)
        x = F.relu(x)
        x = F.normalize(x, dim=1)
        
        # Get Encondings
        x = self.head(x)
        x = F.normalize(x, dim=1)
        return x

    def encode(self, x):
        x = self.features(x)
        
        # Get Representations
        x = torch.flatten(x, start_dim=1)
        x = self.dense(x)
        x = F.relu(x)
        x = F.normalize(x, dim=1)
        return x
        

class WeakStrongFL():
    def __init__(self, in_channels, in_time, conv_filters = [16, 16, 16], conv_kernels = [5, 5, 5]):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.net = SiameseNetwork(in_channels, in_time, conv_filters, conv_kernels).to(self.device)
        
    def reviewAugmentations(self, X, batch_size = 32, ): 
        X = X.astype(np.float32)
        dataset = AugmentationDataset(X)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for i, views in enumerate(dataloader):
            anchor, weak, strong = views
            print(anchor.shape)
            plt.plot(anchor[0])
            return
            
                
    
    def fit(self, X, batch_size = 32, epochs = 2):
        X = X.astype(np.float32)
        dataset = AugmentationDataset(X)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer  = optim.Adam(self.net.parameters(),lr = 0.0005)
        criterion = TripletLoss().to(self.device)
        logs = ValueLogger("Train loss   ", epoch_freq=10)
        
        for epoch in range(epochs):
            for i, views in enumerate(dataloader):
                optimizer.zero_grad()
                anchor, weak, strong = views
                anchor, weak, strong = anchor.to(self.device), weak.to(self.device), strong.to(self.device) 
                codesAnchor = self.net(anchor)
                codesPositive = self.net(weak)
                codesNegative = self.net(strong)
                
                loss = criterion(codesAnchor, codesPositive, codesNegative)

                loss.backward()
                optimizer.step()
                
                logs.update(loss.item())
            
            logs.end_epoch()
                
                
    def encode(self, X, batch_size = 32):
        X = X.astype(np.float32)
        dataset = AugmentationDataset(X)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        output = []
        with torch.torch.no_grad():
            for i, views in enumerate(dataloader):
                view = views[0].to(self.device)
                
                repr = self.net.encode(view)
                output.append(repr)
            output = torch.cat(output, dim=0)
        return output.cpu().numpy()
                
            
        
    