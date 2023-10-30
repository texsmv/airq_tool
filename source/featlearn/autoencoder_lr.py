import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from torch_snippets import *
import torch.optim as optim
from torch.utils.data import Dataset
from ..utils import ValueLogger
import copy
from sklearn.cluster import KMeans

from scipy.optimize import linear_sum_assignment as linear_assignment
import torch.nn.functional as F
import sklearn.metrics
import numpy as np

def KL_loss(z_mu, z_var):
    # recon_loss = F.mse_loss(X_sample, X, reduction='sum') / mb_size
    kl_loss = torch.mean(
        0.5 * torch.sum(torch.exp(z_var) + z_mu ** 2 - 1. - z_var, 1))
    # Loss = recon_loss + kl_loss
    return kl_loss
class metrics:
    nmi = sklearn.metrics.normalized_mutual_info_score
    ari = sklearn.metrics.adjusted_rand_score

    @staticmethod
    def acc(labels_true, labels_pred):
        labels_true = labels_true.astype(np.int64)
        assert labels_pred.size == labels_true.size
        D = max(labels_pred.max(), labels_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(labels_pred.size):
            w[labels_pred[i], labels_true[i]] += 1
        ind = linear_assignment(w.max() - w)
        return sum([w[i, j] for i, j in ind]) * 1.0 / labels_pred.size


# Clustering layer definition (see DCEC article for equations)
class ClusterlingLayer(nn.Module):
    def __init__(self, in_features=10, out_features=10, alpha=1.0):
        super(ClusterlingLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.weight = nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        self.weight = nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        x = x.unsqueeze(1) - self.weight
        x = torch.mul(x, x)
        x = torch.sum(x, dim=2)
        x = 1.0 + (x / self.alpha)
        x = 1.0 / x
        x = x ** ((self.alpha +1.0) / 2.0)
        x = torch.t(x) / torch.sum(x, dim=1)
        x = torch.t(x)
        return x

    def extra_repr(self):
        return 'in_features={}, out_features={}, alpha={}'.format(
            self.in_features, self.out_features, self.alpha
        )

    def set_weight(self, tensor):
        self.weight = nn.Parameter(tensor)



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

class WSense(nn.Module):
    def __init__(self, in_channels, out_size):
        super(WSense, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_size, 5, 
            stride = 1, 
            padding_mode='replicate', 
            padding = 'same'
        )
        self.elu = nn.ELU()
        self.sig = nn.Sigmoid()
        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.conv2 = nn.Conv1d(
            out_size, out_size, 1, 
            stride = 1, 
            padding_mode='replicate', 
            padding = 'same'
        )
    def forward(self, x):
        x = self.elu(self.conv1(x))
        feat = self.pooling(x)
        att = self.sig(self.conv2(feat))
        
        x = feat * att
        x = x.squeeze(dim=2)
        return x
    
class GroupConvs(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(GroupConvs, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding='same', padding_mode='replicate'),
            nn.BatchNorm1d(out_channels, eps=0.001),
            nn.ELU(inplace=True),
        )
    def forward(self, x):
        return self.features(x)

class EncoderBranch(nn.Module):
    def __init__(self, input_size, encoding_size):
        super(EncoderBranch, self).__init__()
        
        # self.wsense = WSense(input_size, encoding_size)
        
        # self.encoder = nn.Sequential(
        #     nn.Linear(input_size, input_size // 2),  # Reduce input dimensions
        #     nn.ReLU(),
        #     nn.Linear(input_size // 2, encoding_size),  # Reduce to the specified encoding size
        #     nn.ReLU(),
        # )
        self.encoder = nn.Sequential(
            GroupConvs(1, 16, 3),
            nn.MaxPool1d(2),
            GroupConvs(16, 32, 3),
            nn.MaxPool1d(2),
            WSense(32, encoding_size),
        )
        
    def forward(self, x):
        x = self.encoder(torch.unsqueeze(x, dim=1))
        return x


class MainEncoder(nn.Module):
    def __init__(self, in_channels, length, feature_size):
        super(MainEncoder, self).__init__()
        self.in_channels = in_channels
        self.heads = nn.ModuleList([EncoderBranch(length, feature_size) for i in range(in_channels)])
        self.feat_mixture = nn.Conv1d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        
        feats = []
        for i in range(self.in_channels):
            v = self.heads[i](x[:, :, i])
            v = nn.functional.normalize(v, dim=1)
            feats.append(v)
        
        feats = torch.stack(feats, dim=1)
        emb = self.feat_mixture(feats)
        emb = nn.functional.normalize(emb, dim=1)
        
        return emb.squeeze(dim=1)


class MainAutoEncoder(nn.Module):
    def __init__(self, in_channels, length, feature_size):
        super(MainAutoEncoder, self).__init__()
        self.encoder = MainEncoder(in_channels, length, feature_size)
        self.D = in_channels
        self.T = length
        
        self.decoder = nn.Sequential(
            nn.Linear(feature_size, (in_channels * length) // 2),  # Increase dimensions
            nn.ReLU(),
            nn.Linear((in_channels * length) // 2, in_channels * length)  # Reconstruct original dimensions
        )
        

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = decoded.reshape([x.shape[0], self.T, self.D])
        
        return decoded, encoded

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
                                     nn.BatchNorm1d(n_channels)
                                     )



    def forward(self, x):
        x = x.permute(0, 2, 1)
        
        x, indice1 = self.pool1(self.e_conv1(x))
        x = self.dropout(x)
        x, indice2 = self.pool2(self.e_conv2(x))
        x, indice3 = self.pool3(self.e_conv3(x))
        
        x = x.reshape(x.shape[0], -1)
        x_encoded = self.lin1(x)
        
        # x_encoded = F.normalize(x_encoded, dim=1)
        # x_encoded
        x = self.lin2(x_encoded)
        x = x.reshape(-1, self.out_channels, self.length//8)
        
        x = self.d_conv1(self.unpool1(x, indice3))
        # x = self.d_conv1(x)
        # x = self.d_conv1(self.unpool1(x_encoded, indice3))
        if self.padding1:
            # m = nn.ConstantPad1d((0, 1), 1)
            # x  = m(x)
            x = nn.functional.pad(x, (0, 1), mode = 'reflect')
        x = self.d_conv2(self.unpool2(x, indice2))
        # x = self.d_conv2(x)
        if self.padding2:
            # m = nn.ConstantPad1d((0, 1), 1)
            # x  = m(x)
            x = nn.functional.pad(x, (0, 1), mode = 'reflect')
        x = self.d_conv3(self.unpool3(x, indice1))
        # x = self.d_conv3(x)
        if self.padding3:
            # m = nn.ConstantPad1d((0, 1), 1)
            # x  = m(x)
            x = nn.functional.pad(x, (0, 1), mode = 'reflect')
        x_decoded = x.permute(0, 2, 1)
            
        # x_encoded = x_encoded.reshape(x_encoded.shape[0], -1)
        
        return x_decoded, x_encoded

class Autoencoder(nn.Module):
    def __init__(self, input_size, encoding_size):
        super(Autoencoder, self).__init__()
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_size, input_size // 2),  # Reduce input dimensions
            nn.ReLU(),
            nn.Linear(input_size // 2, encoding_size),  # Reduce to the specified encoding size
            nn.ReLU(),
        )
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(encoding_size, input_size // 2),  # Increase dimensions
            nn.ReLU(),
            nn.Linear(input_size // 2, input_size),  # Reconstruct original dimensions
            nn.Sigmoid()  # Sigmoid for output between 0 and 1 (if your data is normalized)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded


class CNN_VAE(nn.Module):
    def __init__(self, n_channels, length, out_channels=128, repr_size = 16, device='cpu'):
        super(CNN_VAE, self).__init__()
        self.out_channels = out_channels
        self.length = length
        self.n_channels = n_channels
        self.device = device
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

        # self.lin1 = nn.Linear( length//8 * out_channels, repr_size)
        self.lin2 = nn.Linear(repr_size, length//8 * out_channels)

        self.unpool2 = nn.MaxUnpool1d(kernel_size=2, stride=2, padding=0)
        self.d_conv2 = nn.Sequential(nn.ConvTranspose1d(64, 32, kernel_size=5, stride=1, bias=False, padding=2),
                                     nn.BatchNorm1d(32),
                                     nn.ReLU())

        self.unpool3 = nn.MaxUnpool1d(kernel_size=2, stride=2, padding=0)
        self.d_conv3 = nn.Sequential(nn.ConvTranspose1d(32, n_channels, kernel_size=5, stride=1, bias=False, padding=2),
                                     nn.BatchNorm1d(n_channels),
                                     nn.ReLU())
        
        self.z_dim = repr_size
        # self.linear_mu = nn.Linear(repr_size, self.z_dim)
        # self.linear_var = nn.Linear(repr_size, self.z_dim)
        self.linear_mu = nn.Linear(length//8 * out_channels, self.z_dim)
        self.linear_var = nn.Linear(length//8 * out_channels, self.z_dim)
        
        self.N = torch.distributions.Normal(0, 1)

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        B = x.shape[0]
        
        x = x.permute(0, 2, 1)
        
        x, indice1 = self.pool1(self.e_conv1(x))
        x = self.dropout(x)
        x, indice2 = self.pool2(self.e_conv2(x))
        x, indice3 = self.pool3(self.e_conv3(x))
        
        x = x.reshape(x.shape[0], -1)
        # x_encoded = self.lin1(x)
        
        z_mu = self.linear_mu(x)
        
        z_var = self.linear_var(x)
        
        # eps = torch.randn((B, self.z_dim)).to(self.device)
        # z_sample = z_mu + torch.exp(z_var / 2) * eps
        z_sample = self.reparameterize(z_mu, z_var)
        
        
        # x_encoded = F.normalize(x_encoded, dim=1)
        
        x = self.lin2(z_sample)
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
        
        
        
        return x_decoded, z_sample, z_mu, z_var


class CNN_DCEC(nn.Module):
    def __init__(self, n_channels, length, out_channels=128, repr_size = 16, n_clusters = 5):
        super(CNN_DCEC, self).__init__()
        self.out_channels = out_channels
        self.length = length
        self.n_channels = n_channels
        self.n_clusters = n_clusters
        
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
        
        self.clustering = ClusterlingLayer(repr_size, n_clusters)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        
        x, indice1 = self.pool1(self.e_conv1(x))
        x = self.dropout(x)
        x, indice2 = self.pool2(self.e_conv2(x))
        x, indice3 = self.pool3(self.e_conv3(x))
        
        x = x.reshape(x.shape[0], -1)
        x_encoded = self.lin1(x)
        # x_encoded = F.normalize(x_encoded, dim=1)
        x_encoded = self.sig(x_encoded)
        
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
        
        
        clustering_out = self.clustering(x_encoded)
        
        return x_decoded, x_encoded, clustering_out


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
        # self.net = MainAutoEncoder(in_channels, self.time_length, feature_size).to(self.device)
        # self.net = CNN_AE(self.time_length * in_channels, feature_size).to(self.device)
        self.net = CNN_AE(in_channels, self.time_length, out_channels=128, repr_size = feature_size).to(self.device)
        
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
        # criterion = nn.MSELoss()
        criterion = nn.L1Loss()
        logs = ValueLogger("Train loss   ", epoch_freq=50)
        val_logs = ValueLogger("Val loss   ", epoch_freq=50)
        
        early_stopper = EarlyStopper(patience=6, min_delta=0.0001)
        
        for epoch in range(epochs):
            for i, data in enumerate(dataloader):
                x = data.to(self.device)
                self.net.train()
                optimizer.zero_grad()
                # x_o, _ = self.net(x.reshape([x.shape[0], -1]))
                x_o, _ = self.net(x)
                
                # loss = criterion(x.reshape([x.shape[0], -1]) , x_o.reshape([x.shape[0], -1]))
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
                # dec, enc = self.net(x.reshape([x.shape[0], -1]))
                dec, enc = self.net(x)
                decoded.append(dec)
                encoded.append(enc)
            decoded = torch.cat(decoded, dim=0)
            encoded = torch.cat(encoded, dim=0)
        return decoded.cpu().numpy(), encoded.cpu().numpy()





def target(out_distr):
    tar_dist = out_distr ** 2 / np.sum(out_distr, axis=0)
    tar_dist = np.transpose(np.transpose(tar_dist) / np.sum(tar_dist, axis=1))
    return tar_dist


def calculate_predictions(model, dataloader, device):
    # print("Calculating predictions")
    output_array = None
    label_array = None
    model.eval()
    for data in dataloader:
        inputs = data.to(device)
        # labels = labels.to(device)
        _, _, outputs = model(inputs)
        if output_array is not None:
            output_array = np.concatenate((output_array, outputs.cpu().detach().numpy()), 0)
            # label_array = np.concatenate((label_array, labels.cpu().detach().numpy()), 0)
        else:
            output_array = outputs.cpu().detach().numpy()
            # label_array = labels.cpu().detach().numpy()

    preds = np.argmax(output_array.data, axis=1)
    # print(output_array.shape)
    return output_array, preds

def kmeans(model, dataloader, device):
    km = KMeans(n_clusters=model.n_clusters, n_init=20)
    output_array = None
    model.eval()
    # Itarate throught the data and concatenate the latent space representations of images
    for data in dataloader:
        inputs = data
        inputs = inputs.to(device)
        _, outputs, _ = model(inputs)
        if output_array is not None:
            output_array = np.concatenate((output_array, outputs.cpu().detach().numpy()), 0)
        else:
            output_array = outputs.cpu().detach().numpy()
        # print(output_array.shape)
        if output_array.shape[0] > 50000: break

    # Perform K-means
    outs = km.fit_predict(output_array)
    # print(np.unique(outs))
    # Update clustering layer weights
    weights = torch.from_numpy(km.cluster_centers_).to(device)
    # print('KMEANS weights initialized!')
    # print(weights.shape)
    model.clustering.set_weight(weights.to(device))
        
class DCEC():
    def __init__(self, in_channels, in_time, feature_size = 128, n_clusters = 5):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.time_length = in_time
        self.net = CNN_DCEC(in_channels, self.time_length, repr_size=feature_size, n_clusters = n_clusters).to(self.device)
        self.path = 'cae.pt'
        self.best_epoch = None
        self.n_clusters = n_clusters
    
    def fit(self, X, batch_size = 32, epochs = 32, X_val=None, gamma=0):
        # print('FIT!')
        X = X.astype(np.float32)
        dataset = MyDataset(X)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # if X_val is not None:
        #     X_val = X_val.astype(np.float32)
        #     dataset_val = MyDataset(X_val)
            # dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
        print('Optimizer')
        optimizer  = optim.Adam(self.net.parameters(),lr = 0.0005)
        criterion = nn.MSELoss()
        criterion_2 = nn.KLDivLoss(size_average=False)
        # gamma = 0.1
        batch = batch_size
        
        logs = ValueLogger("Train loss   ", epoch_freq=50)
        logs_rec = ValueLogger("Rec loss   ", epoch_freq=50)
        logs_clust = ValueLogger("Train clust loss   ", epoch_freq=50)
        val_logs = ValueLogger("Val loss   ", epoch_freq=50)
        
        early_stopper = EarlyStopper(patience=6, min_delta=0.0001)
        update_interval = 50
        tol = 1e-2
        
        kmeans(self.net, copy.deepcopy(dataloader), self.device)
        
        output_distribution, preds_prev = calculate_predictions(self.net, copy.deepcopy(dataloader), self.device)
        target_distribution = target(output_distribution)
        
        print(target_distribution)
        print(target_distribution.shape)
        
        for epoch in range(epochs):
            batch_num = 1
            for i, data in enumerate(dataloader):
                x = data.to(self.device)
                self.net.train()
                optimizer.zero_grad()
                
                if (batch_num - 1) % update_interval == 0 and not (batch_num == 1 and epoch == 0):
                    # print('TARGET distribution')
                    # kmeans(self.net, copy.deepcopy(dataloader), self.device)
                    output_distribution, preds = calculate_predictions(self.net, dataloader, self.device)
                    target_distribution = target(output_distribution)
                    # print(target_distribution)
                    # print(target_distribution.shape)
                    
                    # nmi = metrics.nmi(labels, preds)
                    # ari = metrics.ari(labels, preds)
                    # acc = metrics.acc(labels, preds)
                    
                    # utils.print_both(txt_file,
                    #                 'NMI: {0:.5f}\tARI: {1:.5f}\tAcc {2:.5f}\t'.format(nmi, ari, acc))
                    # if board:
                    #     niter = update_iter
                    #     writer.add_scalar('/NMI', nmi, niter)
                    #     writer.add_scalar('/ARI', ari, niter)
                    #     writer.add_scalar('/Acc', acc, niter)
                    #     update_iter += 1

                    # check stop criterion
                    delta_label = np.sum(preds != preds_prev).astype(np.float32) / preds.shape[0]
                    preds_prev = np.copy(preds)
                    # print('Delta label')
                    # print(delta_label)
                    # if delta_label < tol:
                    #     # utils.print_both(txt_file, 'Label divergence ' + str(delta_label) + '< tol ' + str(tol))
                    #     # utils.print_both(txt_file, 'Reached tolerance threshold. Stopping training.')
                    #     finished = True
                    #     break

                tar_dist = target_distribution[((batch_num - 1) * batch):(batch_num*batch), :]
                tar_dist = torch.from_numpy(tar_dist).to(device)
                
                x_o, _, clusters = self.net(x)
                
                loss_rec = criterion(x, x_o)
                loss_clust = gamma * criterion_2(torch.log(clusters), tar_dist) / batch
                
                loss = loss_rec + loss_clust
                # loss = loss_rec
                # loss = loss_clust
                loss.backward()
                optimizer.step()
                
                logs.update(loss.item())
                logs_clust.update(loss_clust.item())
                logs_rec.update(loss_rec.item())
                
                batch_num = batch_num + 1
                
            if logs.end_epoch():
                self.best_epoch = epoch
                torch.save(self.net.state_dict(), self.path)
            logs_clust.end_epoch()
            logs_rec.end_epoch()
                    
            if early_stopper.early_stop(val_logs.avg):             
                print('Early Stop!!!')
                break
            
        if X_val is not None:
            return logs, val_logs
        else:
            return logs
                           
    def encode(self, X, batch_size = 32):
        # print('Loading model of epoch {}'.format(self.best_epoch))
        # self.net.load_state_dict(torch.load(self.path))
        self.net.eval()
        
        X = X.astype(np.float32)
        dataset = MyDataset(X)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        decoded = []
        encoded = []
        clusters = []
        with torch.torch.no_grad():
            for i, data in enumerate(dataloader):
                x = data.to(self.device)
                dec, enc, clus = self.net(x)
                decoded.append(dec)
                encoded.append(enc)
                clusters.append(clus)
            decoded = torch.cat(decoded, dim=0)
            encoded = torch.cat(encoded, dim=0)
            clusters = torch.cat(clusters, dim=0)
        return decoded.cpu().numpy(), encoded.cpu().numpy(), clusters.cpu().numpy()
        
        
class VAE_FL():
    def __init__(self, in_channels, in_time, feature_size = 128):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.time_length = in_time
        self.net = CNN_VAE(in_channels, self.time_length, repr_size=feature_size, device=self.device).to(self.device)
        self.path = 'cae.pt'
        self.best_epoch = None
    
    def kl_loss_function(self,
                      mu,
                      log_var,
                      kld_weight
                      ):
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        # recons = args[0]
        # input = args[1]
        # mu = args[2]
        # log_var = args[3]

        # kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        # recons_loss =F.mse_loss(recons, input)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = kld_weight * kld_loss
        return loss
        # return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

    def fit(self, X, batch_size = 32, epochs = 32, X_val=None):
        X = X.astype(np.float32)
        dataset = MyDataset(X)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        if X_val is not None:
            X_val = X_val.astype(np.float32)
            dataset_val = MyDataset(X_val)
            dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)
        optimizer  = optim.Adam(self.net.parameters(),lr = 0.0005, weight_decay=1e-8)
        criterion = nn.MSELoss()
        # kl_criterion = KL_loss
        logs = ValueLogger("Train loss   ", epoch_freq=50)
        logs_kl = ValueLogger("Train KL loss   ", epoch_freq=50)
        val_logs = ValueLogger("Val loss   ", epoch_freq=50)
        
        early_stopper = EarlyStopper(patience=6, min_delta=0.0001)
        
        for epoch in range(epochs):
            for i, data in enumerate(dataloader):
                x = data.to(self.device)
                self.net.train()
                optimizer.zero_grad()
                x_o, _, z_mu, z_var = self.net(x)
                
                rec_loss = criterion(x, x_o)
                # kl_loss = KL_loss(z_mu, z_var)
                kl_loss = self.kl_loss_function(z_mu, z_var, 1 * (epochs / (epoch + 1)))
                # kl_loss.item = 0
                
                loss = rec_loss + kl_loss
                # loss = rec_loss
                
                loss.backward()
                optimizer.step()
                logs.update(loss.item())
                logs_kl.update(kl_loss.item())
                
            if logs.end_epoch():
                self.best_epoch = epoch
                torch.save(self.net.state_dict(), self.path)
            logs_kl.end_epoch()
                    
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
                dec, enc, z_mu, z_var = self.net(x)
                decoded.append(dec)
                encoded.append(enc)
            decoded = torch.cat(decoded, dim=0)
            encoded = torch.cat(encoded, dim=0)
        return decoded.cpu().numpy(), encoded.cpu().numpy()
