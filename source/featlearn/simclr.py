import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from .datasets import SubsequencesDataset
from torch.utils.data import DataLoader
from torch_snippets import *
from ..models.contrastive.losses import SupConLoss
from ..utils import ValueLogger
from torchsummary import summary

from ..models.contrastive.models import SiameseNetwork

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
        

# class SimClrFLMH():
#     def __init__(self, in_channels, in_time, filters = [16, 16, 16], kernels = [5, 5, 5], feature_size = 1024, encoding_size = 8):
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         self.time_length = int(in_time * 0.9)
#         self.net = SiameseNetwork(in_channels, self.time_length, filters, kernels, feature_size = feature_size, encoding_size = encoding_size).to(self.device)
        
    
#     def fit(self, X, batch_size = 32, epochs = 2):
#         X = X.astype(np.float32)
#         dataset = SubsequencesDataset(X, self.time_length, n_views=4)
#         dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#         optimizer  = optim.Adam(self.net.parameters(),lr = 0.0005)
#         criterion = SupConLoss().to(self.device)
#         logs = ValueLogger("Train loss   ", epoch_freq=10)
        
#         for epoch in range(epochs):
#             for i, views in enumerate(dataloader):
#                 optimizer.zero_grad()
#                 views = [view.to(self.device) for view in views]
#                 codes = [self.net(view) for view in views]
#                 codes = torch.stack(codes, 1)
#                 loss = criterion(codes)
                
#                 loss.backward()
#                 optimizer.step()
                
#                 logs.update(loss.item())
            
#             logs.end_epoch()
                
                
#     def encode(self, X, batch_size = 32):
#         X = X.astype(np.float32)
#         dataset = SubsequencesDataset(X, self.time_length, n_views=1)
#         dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
#         output = []
#         with torch.torch.no_grad():
#             for i, views in enumerate(dataloader):
#                 view = views[0].to(self.device)
                
#                 repr = self.net.encode(view)
#                 output.append(repr)
#             output = torch.cat(output, dim=0)
#         return output.cpu().numpy()
                
            
        
class SimClrFL():
    def __init__(self, in_channels, in_time, filters = [16, 16, 16], kernels = [5, 5, 5], feature_size = 1024, encoding_size = 8):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.time_length = int(in_time * 0.8)
        # self.net = SiameseNetwork(in_channels, self.time_length, self.device, head='linear',conv_filters= filters,conv_kernels= kernels, feat_size = feature_size, encoding_size = encoding_size, use_KL_regularizer=False).to(self.device)
        self.net = SiameseNetwork(in_channels, self.time_length, filters, kernels, feature_size = feature_size, encoding_size = encoding_size).to(self.device)
        
    
    def fit(self, X, batch_size = 32, epochs = 32, X_val=None):
        X = X.astype(np.float32)
        dataset = SubsequencesDataset(X, self.time_length, n_views=4)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        if X_val is not None:
            X_val = X_val.astype(np.float32)
            dataset_val = SubsequencesDataset(X_val, self.time_length, n_views=4)
            dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)
        optimizer  = optim.Adam(self.net.parameters(),lr = 0.0005, weight_decay=0)
        criterion = SupConLoss().to(self.device)
        logs = ValueLogger("Train loss   ", epoch_freq=10)
        val_logs = ValueLogger("Val loss   ", epoch_freq=10)
        
        
        for epoch in range(epochs):
            for i, views in enumerate(dataloader):
                self.net.train()
                optimizer.zero_grad()
                
                views = [view.to(self.device) for view in views]
                
                codes = [self.net(view) for view in views]
                
                codes = torch.stack(codes, 1)
                
                loss = criterion(codes)
                
                loss.backward()
                optimizer.step()
                
                logs.update(loss.item())
            
            logs.end_epoch()
            
            if X_val is not None:
                with torch.no_grad():
                    for i, views in enumerate(dataloader_val):
                        views = [view.to(self.device) for view in views]
                        codes = [self.net(view) for view in views]
                        
                        codes = torch.stack(codes, 1)
                        
                        loss = criterion(codes)
                        
                        val_logs.update(loss.item())
                    
                    val_logs.end_epoch()
                           
    def encode(self, X, batch_size = 32):
        X = X.astype(np.float32)
        dataset = SubsequencesDataset(X, self.time_length, n_views=1)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        output = []
        with torch.torch.no_grad():
            for i, views in enumerate(dataloader):
                view = views[0].to(self.device)
                
                repr = self.net.encode(view)
                output.append(repr)
            output = torch.cat(output, dim=0)
        return output.cpu().numpy()