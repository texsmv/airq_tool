import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_snippets import *
from ..utils import ValueLogger

# class AE(nn.Module):
#     def __init__(self, n_channels, len_sw, n_classes, outdim=128, backbone=True):
#         super(AE, self).__init__()

#         self.backbone = backbone
#         self.len_sw = len_sw

#         self.e1 = nn.Linear(n_channels, 8)
#         self.e2 = nn.Linear(8 * len_sw, 2 * len_sw)
#         self.e3 = nn.Linear(2 * len_sw, outdim)

#         self.d1 = nn.Linear(outdim, 2 * len_sw)
#         self.d2 = nn.Linear(2 * len_sw, 8 * len_sw)
#         self.d3 = nn.Linear(8, n_channels)

#         self.out_dim = outdim

#         if backbone == False:
#             self.classifier = nn.Linear(outdim, n_classes)

#     def forward(self, x):
#         x_e1 = self.e1(x)
#         x_e1 = x_e1.reshape(x_e1.shape[0], -1)
#         x_e2 = self.e2(x_e1)
#         x_encoded = self.e3(x_e2)

#         x_d1 = self.d1(x_encoded)
#         x_d2 = self.d2(x_d1)
#         x_d2 = x_d2.reshape(x_d2.shape[0], self.len_sw, 8)
#         x_decoded = self.d3(x_d2)

#         if self.backbone:
#             return x_decoded, x_encoded
#         else:
#             out = self.classifier(x_encoded)
#             return out, x_decoded
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


class CNN_AE(nn.Module):
    def __init__(self, n_channels, length, out_channels=128, repr_size = 16):
        super(CNN_AE, self).__init__()
        self.out_channels = out_channels
        self.length = length
        self.n_channels = n_channels
        
        curr_length = length
        if curr_length % 2 != 0:
            self.padding3 = True
        else:
            self.padding3 = False
        curr_length = curr_length // 2 
        self.e_conv1 = nn.Sequential(nn.Conv1d(n_channels, 32, kernel_size=5, stride=1, bias=False, padding=2),
                                         nn.BatchNorm1d(32),
                                         nn.ReLU())
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0, return_indices=True)
        self.dropout = nn.Dropout(0.35)

        if curr_length % 2 != 0:
            self.padding2 = True
        else:
            self.padding2 = False
        curr_length = curr_length // 2
        self.e_conv2 = nn.Sequential(nn.Conv1d(32, 64, kernel_size=5, stride=1, bias=False, padding=2),
                                         nn.BatchNorm1d(64),
                                         nn.ReLU())
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0, return_indices=True)

        if curr_length % 2 != 0:
            self.padding1 = True
        else:
            self.padding1 = False
        curr_length = curr_length // 2
        self.e_conv3 = nn.Sequential(nn.Conv1d(64, out_channels, kernel_size=5, stride=1, bias=False, padding=2),
                                         nn.BatchNorm1d(out_channels),
                                         nn.ReLU())
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0, return_indices=True)

        self.unpool1 = nn.MaxUnpool1d(kernel_size=2, stride=2, padding=0)
        
        self.d_conv1 = nn.Sequential(nn.ConvTranspose1d(out_channels, 64, kernel_size=5, stride=1, bias=False, padding=2),
                                     nn.BatchNorm1d(64),
                                     nn.ReLU())

        self.lin1 = nn.Linear( length//8 * out_channels, repr_size)
        self.lin2 = nn.Linear(repr_size, length//8 * out_channels)

        self.unpool2 = nn.MaxUnpool1d(kernel_size=2, stride=2, padding=0)
        self.d_conv2 = nn.Sequential(nn.ConvTranspose1d(64, 32, kernel_size=5, stride=1, bias=False, padding=2),
                                     nn.BatchNorm1d(32),
                                     nn.ReLU())

        self.unpool3 = nn.MaxUnpool1d(kernel_size=2, stride=2, padding=0)
        self.d_conv3 = nn.Sequential(nn.ConvTranspose1d(32, n_channels, kernel_size=5, stride=1, bias=False, padding=2),
                                     nn.BatchNorm1d(n_channels),
                                     nn.ReLU())



    def forward(self, x):
        x = x.permute(0, 2, 1)
        
        x, indice1 = self.pool1(self.e_conv1(x))
        x = self.dropout(x)
        x, indice2 = self.pool2(self.e_conv2(x))
        x, indice3 = self.pool3(self.e_conv3(x))
        
        x = x.reshape(x.shape[0], -1)
        x_encoded = self.lin1(x)
        x_encoded = F.normalize(x_encoded, dim=1)
        # x_encoded
        x = self.lin2(x_encoded)
        x = x.reshape(-1, self.out_channels, self.length//8)
        
        x = self.d_conv1(self.unpool1(x, indice3))
        # x = self.d_conv1(self.unpool1(x_encoded, indice3))
        if self.padding1:
            # m = nn.ConstantPad1d((0, 1), 1)
            # x  = m(x)
            x = nn.functional.pad(x, (0, 1), mode = 'reflect')
        x = self.d_conv2(self.unpool2(x, indice2))
        if self.padding2:
            # m = nn.ConstantPad1d((0, 1), 1)
            # x  = m(x)
            x = nn.functional.pad(x, (0, 1), mode = 'reflect')
        x = self.d_conv3(self.unpool3(x, indice1))
        if self.padding3:
            # m = nn.ConstantPad1d((0, 1), 1)
            # x  = m(x)
            x = nn.functional.pad(x, (0, 1), mode = 'reflect')
        x_decoded = x.permute(0, 2, 1)
            
        # x_encoded = x_encoded.reshape(x_encoded.shape[0], -1)
        
        return x_decoded, x_encoded

class MyDataset(Dataset):
    def __init__(self, X):
        self.X = X
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx]



class AutoencoderFL():
    def __init__(self, in_channels, in_time, feature_size = 128):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.time_length = in_time
        self.net = CNN_AE(in_channels, self.time_length, repr_size=feature_size).to(self.device)
        self.path = 'cae.pt'
        self.best_epoch = None
    
    def fit(self, X, batch_size = 32, epochs = 32, X_val=None):
        X = X.astype(np.float32)
        dataset = MyDataset(X)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        if X_val is not None:
            X_val = X_val.astype(np.float32)
            dataset_val = MyDataset(X_val)
            dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)
        optimizer  = optim.Adam(self.net.parameters(),lr = 0.0005)
        criterion = nn.MSELoss()
        logs = ValueLogger("Train loss   ", epoch_freq=4)
        val_logs = ValueLogger("Val loss   ", epoch_freq=4)
        
        early_stopper = EarlyStopper(patience=6, min_delta=0.0001)
        
        for epoch in range(epochs):
            for i, data in enumerate(dataloader):
                x = data.to(self.device)
                self.net.train()
                optimizer.zero_grad()
                x_o, _ = self.net(x)
                
                loss = criterion(x, x_o)
                    
                loss.backward()
                optimizer.step()
                logs.update(loss.item())
                
            if logs.end_epoch():
                self.best_epoch = epoch
                torch.save(self.net.state_dict(), self.path)
                    
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
        self.net.load_state_dict(torch.load(self.path))
        self.net.eval()
        
        X = X.astype(np.float32)
        dataset = MyDataset(X)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        decoded = []
        encoded = []
        with torch.torch.no_grad():
            for i, data in enumerate(dataloader):
                x = data.to(self.device)
                dec, enc = self.net(x)
                decoded.append(dec)
                encoded.append(enc)
            decoded = torch.cat(decoded, dim=0)
            encoded = torch.cat(encoded, dim=0)
        return decoded.cpu().numpy(), encoded.cpu().numpy()
        