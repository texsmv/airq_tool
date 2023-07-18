import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from math import *
import copy
from lightly.models.modules.heads import TiCoProjectionHead
from pytorch_optimizer import *
from lightly.models import utils
from lightly.models.utils import deactivate_requires_grad, update_momentum
from torch.utils.data import DataLoader
from torch_snippets import *
from ..utils import ValueLogger

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride = 1, padding_mode='replicate', padding = padding)
        self.bn = nn.BatchNorm1d(out_channels, eps=0.001)
    def forward(self, x):
        x = self.conv(x)
        # x = torch.nn.ReLU(x, inplace=True)
        x = F.relu(x)
        x = self.bn(x)
        return x

class Conv1DLSTM(nn.Module):
    def __init__(self, in_channels, out_size, time_length):
        super(Conv1DLSTM, self).__init__()

        filters = [16, 32, 64] 
        # filters = [32, 64, 128]
        kernels = [3, 5, 7]
        
        # filters = [16, 32, 64, 128]
        # kernels = [1, 3, 5, 7]
        # self.conv_features = DilatedCATEN2(in_channels, out_size, time_length, filters)
        
        # filters = [16, 32, 64, 128]
        # kernels = [1, 3, 5, 7]
        self.n_conv = len(filters)
        
        convs  = []
        curr_channels = in_channels
        
        for i in range(self.n_conv ):
            k = kernels[i]
            p = k // 2
            convs.append(
                ConvBlock(curr_channels, filters[i], k, p)
            )
            curr_channels = filters[i]
        
        self.m = nn.MaxPool1d(2)
        
        self.convs = nn.ModuleList(convs)
        
        # self.lstm_1 = nn.LSTM(filters[-1], 32, num_layers=1, batch_first=True, bidirectional=False)
        # self.lstm_2 = nn.LSTM(32, 128, num_layers=1, batch_first=True, bidirectional=False)
        
        self.lstm_1 = nn.GRU(filters[-1], 32, num_layers=1, batch_first=True, bidirectional=True)
        # self.lstm_1 = nn.GRU(sum(filters), 64, num_layers=1, batch_first=True, bidirectional=True)
        self.lstm_2 = nn.GRU(32*2, 128, num_layers=1, batch_first=True, bidirectional=True)
        
        self.dropout_1 = nn.Dropout(0.2)
        self.dropout_2 = nn.Dropout(0.2)
        
        self.dropout_in = nn.Dropout(0.5)
        
        self.dense = nn.Linear(128*2, out_size)
        
    def forward(self, x):
        x = self.dropout_in(x)
        
        self.lstm_1.flatten_parameters()
        self.lstm_2.flatten_parameters()
        
        # x = self.conv_features(x)
        for i in range(self.n_conv):
            x = self.convs[i](x)
            x = self.m(x)
        
        x = x.transpose(1, 2)
        
        x = self.dropout_1(x)
        
        x, h = self.lstm_1(x)
        
        x = self.dropout_2(x)
        
        x, h = self.lstm_2(x)
        
        
        x = x[:, -1, :]
        
        x = self.dense(x)
        return x
    
class CNN(nn.Module):
    def __init__(self, in_channels, out_size, time_length):
        super(CNN, self).__init__()
        
        # self.use_dense = False
        
        # if self.use_dense:
        #     size = 16
        #     self.input_fc = nn.Linear(in_channels, size)
        #     in_channels = size
        
        
        # filters = [16, 32, 64, 128]
        filters = [32, 64, 128]
        # filters = [16, 32, 64, 128, 256]
        
        self.n_conv = len(filters)
        
        branch_1  = []
        curr_channels = in_channels
        for i in range(self.n_conv ):
            k = 3
            p = k // 2
            branch_1.append(
                ConvBlock(curr_channels, filters[i], k, p)
            )
            curr_channels = filters[i]
            # branch_1.append(nn.Dropout(0.5))
        
        self.branch_1 = nn.Sequential(*branch_1)
        self.m = nn.MaxPool1d(2)
        
        features_size = (time_length // (2 ** len(filters))) * filters[-1]
        self.dense = nn.Linear(features_size, out_size)
        self.dropout_1 = nn.Dropout(0.1)
        # self.dropout_in = nn.Dropout(0.5)
        
        
        
        
    def forward(self, x):
        # if self.use_dense:
        #     x = self.input_fc(x)
        
        # if self.use_channel_attention:
        #     x = x * self.c_attention(x) 
        
        first = True
        for i, conv in enumerate(self.branch_1):
            if first:
                x_1 = self.dropout_1(conv(x))
                first = False
            else:
                x_1 = conv(x_1)
            x_1 = self.m(x_1)
            
        convs = x_1
        
        x = torch.flatten(convs, start_dim=1)
        x = self.dense(x)
        x = F.normalize(x, dim=1)
        
        return x

    

class MyDataset(Dataset):
    def __init__(self, X):
        self.X = X
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx]


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

def generate_continuous_mask(B, T, n=5, l=0.1):
    res = torch.full((B, T), True, dtype=torch.bool)
    if isinstance(n, float):
        n = int(n * T)
    n = max(min(n, T // 2), 1)
    
    if isinstance(l, float):
        l = int(l * T)
    l = max(l, 1)
    
    for i in range(B):
        for _ in range(n):
            t = np.random.randint(T-l+1)
            res[i, t:t+l] = False
    return res
def generate_binomial_mask(B, T, p=0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)

class MEC_FL(nn.Module):
    def __init__(self, in_channels, in_time, feature_size = 128):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.time_length = in_time
        self.path = 'mec.pt'
        self.best_epoch = None
        
        #  in_channels, out_size, time_length
        # self.backbone = Conv1DLSTM(in_channels, feature_size, self.time_length).to(self.device)
        
        
        self.backbone = CNN(in_channels, feature_size, self.time_length).to(self.device)
        
        
        # self.projection_head = MSNProjectionHead(feature_size, 512, 64)
        self.predictor = TiCoProjectionHead(feature_size, 512, feature_size).to(self.device)
        
        
        self.teacher = copy.deepcopy(self.backbone)

        deactivate_requires_grad(self.teacher)
        self.feature_size = feature_size
        

    def init_optimizer(self):
        params = [
            *list(self.backbone.parameters()),
            *list(self.predictor.parameters()),
        ]

        # optimizer = torch.optim.SGD(params, lr=0.03)
        optim = torch.optim.SGD(
            params,
            lr=0.005,
            momentum=0.9,
            weight_decay=1e-6,
        )
        # optim = Lion(params, lr=0.00005)
        # optimizer = torch.optim.AdamW(params, lr=1.5e-4)
        return optim

    def forward(self, x1, x2, args = None):
        # if mask == 'binomial':
        mask1 = generate_binomial_mask(x1.size(0), x1.size(1)).to(self.device)
        # mask1 = generate_continuous_mask(x1.size(0), x1.size(1)).to(self.device)
        
        mask2 = generate_binomial_mask(x1.size(0), x1.size(1)).to(self.device)
        # mask2 = generate_continuous_mask(x1.size(0), x1.size(1)).to(self.device)
        
        x1[~mask1] = 0
        x2[~mask2] = 0
        # elif mask == 'all_true':
        #     mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        
        update_momentum(self.backbone, self.teacher, 0.996)

        feat1 = self.backbone(x1)
        feat2 = self.backbone(x2)
        
        z1 = self.predictor(feat1) # NxC
        z2 = self.predictor(feat2) # NxC

        with torch.no_grad():
            p1 = self.teacher(x1)
            p1 = p1.detach()
            p2 = self.teacher(x2)
            p2 = p2.detach()

        lamda_inv = self.lamda_schedule[args['curr_epoch']]
        mec_loss = (loss_func(p1, z2, lamda_inv) + loss_func(p2, z1, lamda_inv)) * 0.5 / self.m
        ssl_loss = -1 * mec_loss * lamda_inv
        
        return ssl_loss

    def fit(self, X, batch_size = 32, epochs = 32, freq = 10, X_val=None):
        
        self.m = batch_size
        d = self.feature_size
        # args.mu = (args.m + d) / 2
        # eps_d = args.eps / d
        eps_d = 64 / d
        lamda = 1 / (self.m * eps_d)
        N = X.shape[0]
        
        self.lamda_schedule = lamda_scheduler(8/lamda, 1/lamda, epochs, N, warmup_epochs=10)
        
        X = X.astype(np.float32)
        dataset = MyDataset(X)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        if X_val is not None:
            X_val = X_val.astype(np.float32)
            dataset_val = MyDataset(X_val)
            dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)

        optimizer = self.init_optimizer()
        
        logs = ValueLogger("Train loss   ", epoch_freq=freq)
        val_logs = ValueLogger("Val loss   ", epoch_freq=freq)
        
        early_stopper = EarlyStopper(patience=6, min_delta=0.0001)
        
        for epoch in range(epochs):
            for i, data in enumerate(dataloader):
                x = data.to(self.device)
                
                self.train()
                optimizer.zero_grad()
                # x_o = self.net(x)
                
                loss = self.forward(x, x, args={'curr_epoch': epoch})
                    
                loss.backward()
                optimizer.step()
                logs.update(loss.item())
                
            if logs.end_epoch():
                self.best_epoch = epoch
                torch.save(self.state_dict(), self.path)
                    
            if early_stopper.early_stop(val_logs.avg):             
                print('Early Stop!!!')
                break
                
            
            # break
            # if X_val is not None:
            #     with torch.no_grad():
            #         for i, data in enumerate(dataloader_val):
            #             views, views_freq, labels = data
            #             # views = [view.to(self.device) for view in views]
            #             self.net.eval()
            #             views = torch.stack([view.to(self.device) for view in views])
            #             views_freq = torch.stack([view.to(self.device) for view in views_freq])
            #             labels = labels.to(self.device)
            #             # views = torch.transpose(views, 0, 1)
                        
            #             codes = [self.net(views[i], views_freq[i]) for i in range(len(views))]
                        
            #             codes = torch.stack(codes, 1)
                        
            #             if self.supervised:
            #                 loss = criterion(codes, labels)
            #             else:
            #                 loss = criterion(codes)
            #             val_loss = loss.item()
                        
            #             val_logs.update(val_loss)
            #         if val_logs.end_epoch():
            #             # print('Saving model of epoch {}'.format(epoch))
            #             self.best_epoch = epoch
            #             torch.save(self.net.state_dict(), self.path)
                    
            #         if early_stopper.early_stop(val_logs.avg):             
            #             print('Early Stop!!!')
            #             break
        if X_val is not None:
            return logs, val_logs
        else:
            return logs
                           
    def encode(self, X, batch_size = 32):
        # print('Loading model of epoch {}'.format(self.best_epoch))
        self.load_state_dict(torch.load(self.path))
        self.eval()
        
        X = X.astype(np.float32)
        dataset = MyDataset(X)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        encoded = []
        with torch.torch.no_grad():
            for i, data in enumerate(dataloader):
                x = data.to(self.device)
                enc = self.backbone(x)
                encoded.append(enc)
            encoded = torch.cat(encoded, dim=0)
        return encoded.cpu().numpy()



def lamda_scheduler(start_warmup_value, base_value, epochs, niter_per_ep, warmup_epochs=5):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    schedule = np.ones(epochs * niter_per_ep - warmup_iters) * base_value
    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

def loss_func(p, z, lamda_inv, order=4):

    # p = gather_from_all(p)
    # z = gather_from_all(z)

    p = F.normalize(p)
    z = F.normalize(z)

    c = p @ z.T

    c = c / lamda_inv 

    power_matrix = c
    sum_matrix = torch.zeros_like(power_matrix)

    for k in range(1, order+1):
        if k > 1:
            power_matrix = torch.matmul(power_matrix, c)
        if (k + 1) % 2 == 0:
            sum_matrix += power_matrix / k
        else: 
            sum_matrix -= power_matrix / k

    trace = torch.trace(sum_matrix)

    return trace


