import torch
from torch import nn
import numpy as np
import copy
import torch.nn.functional as F
# from ..models.feat .datasets import SubsequencesDataset, SubsequencesFreqDataset
from .mdatasets import SubsequencesDataset, SubsequencesFreqDataset, AugmentationsFreqDataset
from torch.utils.data import DataLoader
from torch_snippets import *
from ..utils import ValueLogger
from math import *
import torch.nn.functional as tf

EPOCH_FREQ = 20
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    

class WFEncoder(nn.Module):
    def __init__(self, in_channels, out_size, time_length, classify=False, n_classes=None):
        # Input x is (batch, in_channels, time_length)
        super(WFEncoder, self).__init__()

        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=4, stride=1, padding=2, padding_mode='replicate'),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(64, eps=0.001),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(64, eps=0.001),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(128, eps=0.001),
            # nn.Dropout(),
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(128, eps=0.001),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # nn.Dropout(0.5),
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(256, eps=0.001),
            # nn.Dropout(0.5),
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(256, eps=0.001),
            nn.MaxPool1d(kernel_size=2, stride=2)
            )
        features_size = (time_length // (2 ** 3)) * 256
        self.dense = nn.Linear(features_size, out_size)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dense(x)
        x = F.relu(x)
        x = F.normalize(x, dim=1)
        return x

class FreqEncoder(nn.Module):
    def __init__(self, in_channels, out_size, freq_size, classify=False, n_classes=None):
        # Input x is (batch, in_channels, out_size)
        super(FreqEncoder, self).__init__()

        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, stride=1, padding=2, padding_mode='replicate'),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(64, eps=0.001),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(64, eps=0.001),
            # nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(128, eps=0.001),
        )
        features_size = (freq_size ) * 128
        self.dense = nn.Linear(features_size, out_size)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dense(x)
        x = F.relu(x)
        x = F.normalize(x, dim=1)
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

# class TimeFreq_Net(nn.Module):
#     def __init__(self, in_channels, time_length, 
#                  feature_size=64, 
#                  freq_feature_size=32, 
#                  encoding_size = 8, 
#                  head='linear', 
#                  device ='cpu',
#                  use_head=True,
#                  encoder = 'CNN', # CNN, LSTM, TRANSFORMER, CONV_LSTM
#                  ):
#         super(TimeFreq_Net, self).__init__()
#         self.device= device
#         self.encoder = encoder
#         if encoder == 'CNN':
#             print('Using CNN backbone')
#             self.features = WFEncoder(in_channels, feature_size, time_length)
#         # elif encoder == 'TRANSFORMER':
#         #     print('Using Transformer backbone')
#         #     self.features = MTransformer(in_channels, time_length, dim=feature_size)
#         # elif encoder == 'LSTM':
#         #     print('Using LSTM backbone')
#         #     self.features = MLSTM(in_channels, LSTM_units=feature_size)
#         # elif encoder == 'CONV_LSTM':
#         #     print('Using DeepConvLSTM backbone')
#         #     self.features = MConvLSTM(in_channels, LSTM_units=feature_size)
#         # else:
#         #     encoder = None
#         # self.freqFeatures = FreqEncoder(1, freq_feature_size, time_length)
#         # self.head = HeadModel(feature_size + freq_feature_size, head = head, feat_dim=encoding_size)
#         self.head = HeadModel(feature_size , head = head, feat_dim=encoding_size)
#         # self.head = HeadModel(freq_feature_size, head = head, feat_dim=encoding_size)
#         self.dropout = nn.Dropout(0.1)
#         self.use_head = use_head
        
#     def forward(self, x):
#         # Get Representations
#         # print('---')
#         # print(x.shape)
#         if self.encoder == 'CNN':
#             a = self.features(x)
#         else:
#             a = self.features(x.transpose(1, 2))
#         # print(a.shape)
#         # print(a)
#         # b = self.freqFeatures(x_freq)
#         # return a
#         x = self.head(a)
#         x = F.normalize(x, dim=1)
#         return x
    
#         x = torch.cat([a, b], dim=1)
        
#         if not self.use_head:
#             return x
        
#         # Get Encondings

#     def encode(self, x):
#         if self.encoder == 'CNN':
#             a = self.features(x)
#         else:
#             a = self.features(x.transpose(1, 2))
#         # b = self.freqFeatures(x_freq)
#         return a
#         x = torch.cat([a, b], dim=1)
#         return x     

class EncodernNet(nn.Module):
    def __init__(self, 
                 in_channels, 
                 time_length, 
                 feature_size=64, 
                 freq_feature_size=32, 
                 encoding_size = 8, 
                 head='linear', 
                 device ='cpu',
                #  use_head=True, # TODO delete this
                 encoder = 'CNN', # CNN, LSTM, TRANSFORMER, CONV_LSTM
                 use_frequency=True,
                 ):
        super(EncodernNet, self).__init__()
        self.device= device
        self.encoder = encoder
        if encoder == 'CNN':
            print('Using CNN backbone')
            self.features = WFEncoder(in_channels, feature_size, time_length)
        # elif encoder == 'TRANSFORMER':
        #     print('Using Transformer backbone')
        #     self.features = MTransformer(in_channels, time_length, dim=feature_size)
        # elif encoder == 'LSTM':
        #     print('Using LSTM backbone')
        #     self.features = MLSTM(in_channels, LSTM_units=feature_size)
        # elif encoder == 'CONV_LSTM':
        #     print('Using DeepConvLSTM backbone')
        #     self.features = MConvLSTM(in_channels, LSTM_units=feature_size)
        else:
            encoder = None
        self.freqFeatures = FreqEncoder(in_channels, freq_feature_size, time_length)
        
        if use_frequency:
            self.head = HeadModel(feature_size + freq_feature_size, head=head, feat_dim=encoding_size)
        else:
            self.head = HeadModel(feature_size , head = head, feat_dim=encoding_size)
        # TODO to only use frequency
        # self.head = HeadModel(freq_feature_size, head = head, feat_dim=encoding_size)
        # self.use_head = use_head
        self.use_frequency = use_frequency
        
    def forward(self, x, x_freq):
        # Get Representations
        if self.encoder == 'CNN':
            a = self.features(x)
        else:
            a = self.features(x.transpose(1, 2))
        
        if self.use_frequency:
            b = self.freqFeatures(x_freq)
            x = torch.cat([a, b], dim=1)
            x = F.normalize(x, dim=1)
        else:
            x = a
        
        # Get Encondings
        x = self.head(x)
        x = F.normalize(x, dim=1)
        return x

    def encode(self, x, x_freq):
        if self.encoder == 'CNN':
            a = self.features(x)
        else:
            a = self.features(x.transpose(1, 2))
            
        if self.use_frequency:
            b = self.freqFeatures(x_freq)
            x = torch.cat([a, b], dim=1)
            x = F.normalize(x, dim=1)
        else:
            x = a
        
        return x

       
    
class MLPHead(nn.Module):
    def __init__(self, in_dim, hidden_size=4096, projection_size=256):
        super(MLPHead, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)


def loss_fn(q1,q2, z1t,z2t):
    
    l1 = - tf.cosine_similarity(q1, z1t.detach(), dim=-1).mean()
    l2 = - tf.cosine_similarity(q2, z2t.detach(), dim=-1).mean()
    
    return (l1+l2)/2
class BYOL():
    def __init__(self, in_channels, in_time, base_target_ema=0.996, encoder = 'CNN', feature_size = 128, freq_feature_size = 32, encoding_size = 8, aug_type=None, use_frequency=True,):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.net = SiameseNetwork(in_channels, self.time_length, self.device, head='linear',conv_filters= filters,conv_kernels= kernels, feat_size = feature_size, encoding_size = encoding_size, use_KL_regularizer=False).to(self.device)
        # if aug_type == None:
        #     self.time_length = int(in_time * 0.94)
        # else:
        self.time_length = in_time
        self.aug_type = aug_type
        self.path = 'byol.pt'
        self.best_epoch = None
        self.base_ema = base_target_ema
        
        
        # self.online_encoder = EncodernNet(in_channels, self.time_length, encoder=encoder, feature_size = feature_size, freq_feature_size=freq_feature_size, encoding_size = encoding_size, device=self.device)
        self.online_encoder = EncodernNet(
            in_channels, 
            self.time_length, 
            encoder=encoder, 
            feature_size = feature_size, 
            freq_feature_size=freq_feature_size, 
            encoding_size = encoding_size, 
            device=self.device,
            use_frequency=use_frequency,
        )
        self.target_encoder = copy.deepcopy(self.online_encoder).to(self.device)
        self.online_encoder = self.online_encoder.to(self.device)
        self.online_predictor = MLPHead(in_dim=encoding_size,hidden_size=1024, projection_size=encoding_size).to(self.device)
    
    @torch.no_grad()
    def update_moving_average(self, global_step, max_steps):
        
        tau = 1- ((1 - self.base_ema)* (cos(pi*global_step/max_steps)+1)/2) 
        
        for online, target in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target.data = tau * target.data + (1 - tau) * online.data  
      
    def fit(self, X, batch_size = 32, epochs = 32, X_val=None, y_val=None, Acc_val=None):
        X = X.astype(np.float32)
        # dataset = SubsequencesFreqDataset(X, Acc, y, self.time_length, n_views=n_views)
        dataset = AugmentationsFreqDataset(X, self.time_length, n_views=2, aug_type=self.aug_type)
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        if X_val is not None:
            X_val = X_val.astype(np.float32)
            dataset_val = AugmentationsFreqDataset(X_val, self.time_length, n_views=2, aug_type=self.aug_type)
            dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)
        params = list(self.online_encoder.parameters()) + list(self.online_predictor.parameters())
        optimizer  = optim.Adam(params,lr = 0.0005)
        # criterion = SupConLoss().to(self.device)
        logs = ValueLogger("Train loss   ", epoch_freq=5)
        val_logs = ValueLogger("Val loss   ", epoch_freq=5)
        
        early_stopper = EarlyStopper(patience=20, min_delta=0.0002)
        best_loss = np.inf
        for epoch in range(epochs):
            for i, data in enumerate(dataloader):
                self.online_encoder.train()
                self.online_predictor.train()
                optimizer.zero_grad()
                
                views, views_freq  = data
                views = torch.stack([view.to(self.device) for view in views])
                views_freq = torch.stack([view.to(self.device) for view in views_freq])
                zs = [self.online_encoder(views[i], views_freq[i]) for i in range(len(views))]
                qs = [self.online_predictor(zs[i]) for i in range(len(views))]
                
                with torch.no_grad():
                    zs_t = [self.target_encoder(views[i], views_freq[i]) for i in range(len(views))]
                
                loss = loss_fn(qs[0], qs[1], zs_t[0], zs_t[1])
                
                loss.backward()
                optimizer.step()
                self.update_moving_average(epoch, epochs)
                
                logs.update(loss.item())
                
            
            logs.end_epoch()
            # break
            if X_val is not None:
                with torch.no_grad():
                    for i, data in enumerate(dataloader_val):
                        self.online_encoder.eval()
                        self.online_predictor.eval()
                        optimizer.zero_grad()
                        
                        views, views_freq = data
                        views = torch.stack([view.to(self.device) for view in views])
                        views_freq = torch.stack([view.to(self.device) for view in views_freq])
                        
                        zs = [self.online_encoder(views[i], views_freq[i]) for i in range(len(views))]
                        qs = [self.online_predictor(zs[i]) for i in range(len(views))]
                        
                        with torch.no_grad():
                            zs_t = [self.target_encoder(views[i], views_freq[i]) for i in range(len(views))]
                        
                        loss = loss_fn(qs[0], qs[1], zs_t[0], zs_t[1])
                        val_loss = loss.item()
                        
                        val_logs.update(val_loss)
                    if val_logs.end_epoch():
                        # print('Saving model of epoch {}'.format(epoch))
                        self.best_epoch = epoch
                        torch.save(self.online_encoder.state_dict(), self.path)
                    
                    if early_stopper.early_stop(val_logs.avg):             
                        print('Early Stop!!!')
                        break
        if X_val is not None:
            return logs, val_logs
        else:
            return logs
                           
    def encode(self, X, batch_size = 32):
        # print('Loading model of epoch {}'.format(self.best_epoch))
        self.online_encoder.load_state_dict(torch.load(self.path))
        self.online_encoder.eval()
        self.online_predictor.eval()
        
        X = X.astype(np.float32)
        dataset = AugmentationsFreqDataset(X,self.time_length, n_views=1, test=True)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        output = []
        with torch.torch.no_grad():
            for i, data in enumerate(dataloader):
                views, views_freq = data
                view = views[0].to(self.device)
                view_freq = views_freq[0].to(self.device)
                
                repr = self.online_encoder.encode(view, view_freq)
                # repr = self.net.forward(view)
                output.append(repr)
            output = torch.cat(output, dim=0)
        return output.cpu().numpy()





# def off_diagonal(x):
#     # return a flattened view of the off-diagonal elements of a square matrix
#     n, m = x.shape
#     assert n == m
#     return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

# # class BarlowTwins():
#     def __init__(self, 
#                  in_channels,
#                  in_time, 
#                  encoder = 'CNN', 
#                  feature_size = 128, 
#                  freq_feature_size = 32, 
#                  encoding_size = 8, 
#                  aug_type=None,
#                  use_frequency=True,
#         ):
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         if aug_type == None:
#             self.time_length = int(in_time * 0.8)
#         else:
#             self.time_length = in_time
#         self.aug_type = aug_type
#         self.path = 'barlow.pt'
#         self.best_epoch = None

        
#         self.net = EncodernNet(
#             in_channels, 
#             self.time_length, 
#             encoder=encoder, 
#             feature_size = feature_size, 
#             freq_feature_size=freq_feature_size, 
#             encoding_size = encoding_size, 
#             device=self.device,
#             use_frequency=use_frequency,
#         ).to(self.device)
        
#         self.bn = nn.BatchNorm1d(encoding_size, affine=False).to(self.device)
#         self.lambd = 0.0051
        
#     def loss(self, z1, z2, bs):
#         # empirical cross-correlation matrix
#         c = self.bn(z1).T @ self.bn(z2)

#         # sum the cross-correlation matrix between all gpus
#         c.div_(bs)
#         # torch.distributed.all_reduce(c)

#         on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
#         off_diag = off_diagonal(c).pow_(2).sum()
#         loss = on_diag + self.lambd * off_diag
#         return loss
      
#     def fit(self, X,  batch_size = 32, epochs = 32, X_val=None, y_val=None,
#             vis=None, model_name='model',
#         ):
#         X = X.astype(np.float32)
#         # dataset = SubsequencesFreqDataset(X, Acc, y, self.time_length, n_views=n_views)
#         dataset = AugmentationsFreqDataset(X, self.time_length, n_views=2, aug_type=self.aug_type)
        
#         dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#         if X_val is not None:
#             X_val = X_val.astype(np.float32)
#             dataset_val = AugmentationsFreqDataset(X_val, self.time_length, n_views=2, aug_type=self.aug_type)
#             dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)
#         optimizer  = optim.Adam(self.net.parameters(),lr = 0.0005)
#         logs = ValueLogger("Train loss   ", epoch_freq=EPOCH_FREQ)
#         val_logs = ValueLogger("Val loss   ", epoch_freq=EPOCH_FREQ)
        
#         early_stopper = EarlyStopper(patience=20, min_delta=0.0002)
#         best_loss = np.inf
#         for epoch in range(epochs):
#             for i, data in enumerate(dataloader):
#                 self.net.train()
#                 optimizer.zero_grad()
                
#                 views, views_freq = data
#                 views = torch.stack([view.to(self.device) for view in views])
#                 views_freq = torch.stack([view.to(self.device) for view in views_freq])
#                 zs = [self.net(views[i], views_freq[i]) for i in range(len(views))]
                
#                 bs = zs[0].shape[0]
#                 loss = self.loss(zs[0], zs[1], bs)
                
#                 loss.backward()
#                 optimizer.step()
                
#                 logs.update(loss.item())
                
            
#             logs.end_epoch()
#             # break
#             if X_val is not None:
#                 with torch.no_grad():
#                     for i, data in enumerate(dataloader_val):
#                         self.net.eval()
#                         optimizer.zero_grad()
                        
#                         views, views_freq = data
#                         views = torch.stack([view.to(self.device) for view in views])
#                         views_freq = torch.stack([view.to(self.device) for view in views_freq])
#                         zs = [self.net(views[i], views_freq[i]) for i in range(len(views))]
#                         bs = zs[0].shape[0]
                        
#                         loss = self.loss(zs[0], zs[1], bs)
#                         val_loss = loss.item()
                        
#                         val_logs.update(val_loss)
#                     if val_logs.end_epoch():
#                         # print('Saving model of epoch {}'.format(epoch))
#                         self.best_epoch = epoch
#                         torch.save(self.net.state_dict(), self.path)
                    
#                     if early_stopper.early_stop(val_logs.avg):             
#                         print('Early Stop!!!')
#                         break
#             if vis != None:
#                 x = np.arange(len(logs.avgs))
#                 title = 'Loss - {}'.format(model_name)
#                 vis.line(logs.avgs, x, name='Train', win=title, 
#                          opts=dict(
#                             title=title,
#                             legend=["Train","Val"],
#                         )
#                 )
#                 vis.line(val_logs.avgs, x, name='Validation', win=title, update="append")
#         if X_val is not None:
#             return logs, val_logs
#         else:
#             return logs
                           
#     def encode(self, X, batch_size = 32):
#         # print('Loading model of epoch {}'.format(self.best_epoch))
#         self.net.load_state_dict(torch.load(self.path))
#         self.net.eval()
        
#         X = X.astype(np.float32)
#         dataset = AugmentationsFreqDataset(X,  self.time_length, n_views=1, test=True)
#         dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
#         output = []
#         with torch.torch.no_grad():
#             for i, data in enumerate(dataloader):
#                 views, views_freq = data
#                 view = views[0].to(self.device)
#                 view_freq = views_freq[0].to(self.device)
                
#                 repr = self.net.encode(view, view_freq)
#                 output.append(repr)
#             output = torch.cat(output, dim=0)
#         return output.cpu().numpy()
