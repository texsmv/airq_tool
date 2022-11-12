# import sys
# sys.path.insert(0, '/home/texs/Documentos/Repositories/mts_feature_learning/source')

from torch_snippets import *
from source.models.contrastive.models import SiameseNetwork, SiameseNetworkMH, train_batch, eval_batch, train_batch_KL, eval_batch_KL
from source.models.contrastive.losses import ContrastiveLoss, SupConLoss, TripletLoss
from source.models.contrastive.datasets import ContrastiveDataset
from torch.utils.data import DataLoader
from source.utils import create_dir
from torch.utils.data import Dataset
import torch
import random

SUBSEC_PORC = 0.8 # Porcentage of the window to be used as a sub-sequence
EXP_DIR = 'experiments'
EXPERIMENT_NAME = 'test'
EXP_PATH = os.path.join(EXP_DIR, EXPERIMENT_NAME)
create_dir(EXP_DIR)
create_dir(EXP_PATH)



class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride = 1, padding_mode='replicate', padding = padding)
        self.bn = nn.BatchNorm1d(out_channels)
    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
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


class CNNFeaturesCust(nn.Module):
    def __init__(self, in_features, device, length, feat_size, use_batch_norm = True, conv_filters = [16, 16, 16], conv_kernels = [5, 5, 5], use_KL_regularizer=True):
        super(CNNFeaturesCust, self).__init__()
        self.use_batch_norm = use_batch_norm
        self.n_conv = len(conv_filters)
        self.convs = []
        self.bns = []
        in_size = in_features
        for i in range(self.n_conv):
            k = conv_kernels[i]
            p = k // 2
            self.convs.append(nn.Conv1d(in_size, conv_filters[i], k, stride = 1, padding_mode='replicate', padding = p, device=device))
            in_size = conv_filters[i]
            if self.use_batch_norm:
                self.bns.append(nn.BatchNorm1d(conv_filters[i], device =device))
        self.dropout = nn.Dropout(p=0.2)
        self.m = nn.MaxPool1d(2)
        
        self.feat_size = (length// 2 ** len(conv_filters)) * conv_filters[-1]
        self.dense = nn.Linear(self.feat_size, feat_size)
        
        self.z_dim = feat_size 
        self.linear_mu = nn.Linear(self.z_dim, self.z_dim)
        self.linear_var = nn.Linear(self.z_dim, self.z_dim)
        self.device = device
        self.use_KL_regularizer = use_KL_regularizer
    def forward(self,x):
        B = x.shape[0]
        
        for i in range(self.n_conv):
            x = self.convs[i](x)
            x = F.relu(x)
            if self.use_batch_norm:
                x = self.bns[i](x)
            else:
                x = self.dropout(x)
            x = self.m(x)
        
        x = torch.flatten(x, start_dim=1)
        x = self.dense(x)
        x = F.relu(x)
        
        if self.use_KL_regularizer:
            z_mu = self.linear_mu(x)
            
            z_var = self.linear_var(x)
            
            eps = torch.randn(B, self.z_dim).to(self.device)
            
            z_sample = z_mu + torch.exp(z_var / 2) * eps
            
            
            # return x
            return z_sample, z_mu, z_var
        else:
            x = F.normalize(x, dim=1)
            return x
    


        

class SiameseNetwork(nn.Module):
    def __init__(self, in_channels, time_length, filters = [16, 16, 16], kernels = [5, 5, 5], feature_size=1024, encoding_size = 8, head='linear'):
        super(SiameseNetwork, self).__init__()
        
        self.features = EncoderCNN(in_channels, filters=filters, kernels=kernels)
        # self.features = CNNFeaturesCust(in_channels, 'gpu: 0', time_length, feature_size, conv_filters=conv_filters, conv_kernels=conv_kernels, use_KL_regularizer=use_KL_regularizer)
        
        self.encoder_out_size = (time_length// 2 ** len(filters)) * filters[-1]
        self.dense = nn.Linear(self.encoder_out_size, feature_size)
        self.head = HeadModel(feature_size, head = head, feat_dim=encoding_size)
        self.time_length = time_length
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

    def uencode(self, x):
        x = self.features(x)
        
        # Get Representations
        x = torch.flatten(x, start_dim=1)
        x = self.dense(x)
        x = F.relu(x)
        x = F.normalize(x, dim=1)
        return x
    
    def encode(self, X, y, subsequence_length, batch_size = 32, device = 'cuda'):
        X = X.astype(np.float32)
        # dataset = SubsequencesDataset(X, self.time_length, n_views=1)
        dataset = ContrastiveDataset(X.astype(np.float32), y, use_label=False)

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        output = []
        with torch.torch.no_grad():
            for i, views in enumerate(dataloader):
                # view = views[0].to(device)
                
                xA, xB, lA, lB = views
                view1, view2, view3, view4 = getViews(xA, subsequence_length, mode='subsequences')
                views = [view1, view2, view3, view4]
                views = [view.to(device) for view in views]
                
                repr = self.uencode(views[0])
                output.append(repr)
            output = torch.cat(output, dim=0)
        return output.cpu().numpy()


def getSubsequence(x, size):
    D, T = x.shape
    b = random.randint(0, T - size) 
    # print(b)
    return x[:, b: b + size]

# from source.torch_utils import getRandomSlides, getViews

# Batch of shape BxDxT
# mode = subsequences or shape
def getRandomSlides(batch, size, isNumpy = False, mode = 'subsequences'):
    if not isNumpy:
        batch = batch.numpy()
    B, D, T = batch.shape
    b = np.array([random.randint(0, T - size) for i in range(B)])
    
    if mode == 'subsequences':
        slides = np.array([ batch[i,:, b[i]: b[i] + size] for i in range(B)]).astype(np.float32)
        return slides

    elif mode == 'shape':
        slides = []
        for i in range(B):
            slide = batch[i,:, b[i]: b[i] + size]
            for j in range(D):
                slide[j] = slide[j] + random.uniform(-0.1, 0.1)
            slides.append(slide)
        return np.array(slides).astype(np.float32)

# mode = subsequences or shape
def getViews(batch, size, isNumpy = False, mode = 'subsequences'):
    originalSlides = getRandomSlides(batch, size, isNumpy, mode = mode)
    
    fslides = originalSlides.transpose((0, 2, 1))
    # scaled = aug.scaling(fslides, sigma=0.1).transpose((0, 2, 1))
    scaled = fslides.transpose((0, 2, 1))
    
    originalSlides = getRandomSlides(batch, size, isNumpy, mode = mode)
    fslides = originalSlides.transpose((0, 2, 1))
    # flipped = aug.rotation(fslides).transpose((0, 2, 1))
    flipped = fslides.transpose((0, 2, 1))
    
    originalSlides = getRandomSlides(batch, size, isNumpy, mode = mode)
    fslides = originalSlides.transpose((0, 2, 1))
    # magWarped =  aug.magnitude_warp(fslides, sigma=0.2, knot=4).transpose((0, 2, 1))
    magWarped =  fslides.transpose((0, 2, 1))
    
    originalSlides = getRandomSlides(batch, size, isNumpy, mode = mode)
    # fslides = originalSlides.transpose((0, 2, 1))
    
    # return torch.from_numpy(originalSlides.astype(np.float32))
    return [
        torch.from_numpy(originalSlides.astype(np.float32)),
        torch.from_numpy(scaled.astype(np.float32)),
        torch.from_numpy(flipped.astype(np.float32)),
        torch.from_numpy(magWarped.astype(np.float32)),
    ]
    # return torch.from_numpy(np.stack([originalSlides, scaled, flipped, magWarped], axis=1).astype(np.float32))






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
    

# loss # "Contrastive" or "Triplet"  or "SupConLoss" or "SimCLR"
def getContrastiveFeatures(X, y, epochs = 100, batch_size = 32, head='linear', loss_metric = 'SimCLR', feat_size = 1024, encoding_size = 8, mode = 'subsequences', X_test=[], y_test =[], conv_filters = [16, 16, 16], conv_kernels = [5, 5, 5], use_KL_regularizer = True, X_val=None, y_val=None):

    if mode=='subsequences':
        subsequence_length = int(X.shape[2] * SUBSEC_PORC)
        print("Subsequence length: {}".format(subsequence_length))
    else:
        subsequence_length = X.shape[2]
        
    train_dataset = ContrastiveDataset(X.astype(np.float32), y, use_label=False)
    # train_dataset = SubsequencesDataset(X.astype(np.float32), subsequence_length, n_views=4)
            
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = ContrastiveDataset(X.astype(np.float32), y_val, use_label=False)
    # val_dataset = SubsequencesDataset(X_val.astype(np.float32), subsequence_length, n_views=4)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'

    
    model      = SiameseNetwork(
        X.shape[1], 
        subsequence_length, 
        head = head, 
        encoding_size=encoding_size, 
        filters = conv_filters, 
        kernels = conv_kernels,
        feature_size = feat_size
    ).to(device)

    model_path = os.path.join(EXP_PATH, 'model.pt')
    loss_path = os.path.join(EXP_PATH, 'loss.png')

    
    if loss_metric == "SupConLoss":
        criterion = SupConLoss().to(device)
        supervised = True
    else:
        criterion = SupConLoss().to(device)
        supervised = False

    trainLogs = ValueLogger("Train loss   ", epoch_freq=10)
    trainKlLogs = ValueLogger("Train KL loss   ", epoch_freq=10)
    # testLogs = ValueLogger( "Test loss    ", epoch_freq=10)
    # testKlLogs = ValueLogger("Test KL loss   ", epoch_freq=10)
    valLogs = ValueLogger(  "Val loss     ", epoch_freq=10)
    valKlLogs = ValueLogger("Val Kl loss   ", epoch_freq=10)

    # optimizer  = optim.AdamW(model.parameters(),lr = 0.0005, )
    # optimizer  = optim.AdamW(model.parameters(),lr = 0.00001) #Triplet
    # optimizer  = optim.SGD(model.parameters(),lr = 0.001)
    optimizer  = optim.Adam(model.parameters(),lr = 0.0005, weight_decay=0)

    for epoch in range(epochs):
        N = len(train_dataloader)
        for i, views in enumerate(train_dataloader):
            model.train()
            optimizer.zero_grad()
            
            xA, xB, lA, lB = views
            view1, view2, view3, view4 = getViews(xA, subsequence_length, mode='subsequences')
            views = [view1, view2, view3, view4]
            views = [view.to(device) for view in views]
            # print(views[0].shape)
            
            codes = [model(view) for view in views]
            
            codes = torch.stack(codes, 1)
            
            loss = criterion(codes)
            
            loss.backward()
            optimizer.step()
            
            trainLogs.update(loss.item())
        trainLogs.end_epoch()
        with torch.no_grad():
            N = len(val_dataloader)
            for i, views in enumerate(val_dataloader):
                # views = [view.to(device) for view in views]
                xA, xB, lA, lB = views
                view1, view2, view3, view4 = getViews(xA, subsequence_length, mode='subsequences')
                views = [view1, view2, view3, view4]
                views = [view.to(device) for view in views]
                codes = [model(view) for view in views]
                
                codes = torch.stack(codes, 1)
                
                loss = criterion(codes)
                
                
                valLogs.update(loss.item())
            
            if  valLogs.end_epoch():
                print('[Log] Saving model with loss: {}'.format(valLogs.bestAvg))
                torch.save(model, model_path) 

    # fig = plt.figure()
    # ax0 = fig.add_subplot(111, title="loss")
    # ax0.plot(trainLogs.avgs, 'bo-', label='train')
    # ax0.plot(valLogs.avgs, 'ro-', label='val')

    # ax0.legend()
    # fig.savefig(loss_path)
    # model = torch.load(model_path)
    if len(X_test) != 0:
        return model.encode(X, y, subsequence_length), model.encode(X_test, y_test, subsequence_length)
    return model.encode(X, y, subsequence_length)
class ValueLogger(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, epoch_freq = 5):
        self.name = name
        self.epoch_freq = epoch_freq
        self.reset()
    
    def reset(self):
        self.avgs = []
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0.0
        self.bestAvg = np.inf
        
        
    def end_epoch(self):
        
        self.avgs = self.avgs + [self.avg]
        self.val = 0
        self.sum = 0
        self.count = 0.0
        if len(self.avgs) == 1 or len(self.avgs) % self.epoch_freq == 0:
            print("Epoch[{}] {} {}: {}".format(len(self.avgs), self.name, "avg", self.avg))
    
        if self.bestAvg > self.avg:
            self.bestAvg = self.avg
            return True
        else:
            return False

    # Updates de value history
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count





# def getContrastiveFeatures(X, y, epochs = 100, batch_size = 32, head='linear', loss_metric = 'SimCLR', feat_size = 1024, encoding_size = 8, mode = 'subsequences', X_test=[], conv_filters = [16, 16, 16], conv_kernels = [5, 5, 5], use_KL_regularizer = True, X_val=None, y_val=None):
#     train_dataset = ContrastiveDataset(X.astype(np.float32), y, use_label=False)
#     train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
#     val_dataset = ContrastiveDataset(X.astype(np.float32), y_val, use_label=False)
#     val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     # device = 'cpu'

#     if mode=='subsequences':
#         subsequence_length = int(X.shape[2] * SUBSEC_PORC)
#         print("Subsequence length: {}".format(subsequence_length))
#     else:
#         subsequence_length = X.shape[2]

#     model      = SiameseNetwork(
#         X.shape[1], 
#         subsequence_length, 
#         device,
#         head = head, 
#         encoding_size=encoding_size, 
#         conv_filters = conv_filters, 
#         conv_kernels = conv_kernels,
#         feat_size = feat_size,
#         use_KL_regularizer=use_KL_regularizer
#     ).to(device)

#     model_path = os.path.join(EXP_PATH, 'model.pt')
#     loss_path = os.path.join(EXP_PATH, 'loss.png')

#     if loss_metric == "Contrastive":
#         criterion  = ContrastiveLoss().to(device)
#     elif loss_metric == "Triplet":
#         criterion  = TripletLoss(margin=4.0).to(device)
#     elif loss_metric == "SupConLoss":
#         criterion = SupConLoss().to(device)
#         supervised = True
#     else:
#         criterion = SupConLoss().to(device)
#         supervised = False

#     trainLogs = ValueLogger("Train loss   ", epoch_freq=10)
#     trainKlLogs = ValueLogger("Train KL loss   ", epoch_freq=10)
#     # testLogs = ValueLogger( "Test loss    ", epoch_freq=10)
#     # testKlLogs = ValueLogger("Test KL loss   ", epoch_freq=10)
#     valLogs = ValueLogger(  "Val loss     ", epoch_freq=10)
#     valKlLogs = ValueLogger("Val Kl loss   ", epoch_freq=10)

#     # optimizer  = optim.AdamW(model.parameters(),lr = 0.0005, )
#     # optimizer  = optim.AdamW(model.parameters(),lr = 0.00001) #Triplet
#     # optimizer  = optim.SGD(model.parameters(),lr = 0.001)
#     optimizer  = optim.Adam(model.parameters(),lr = 0.0005, weight_decay=0)

#     for epoch in range(epochs):
#         N = len(train_dataloader)
#         for i, data in enumerate(train_dataloader):
#             loss = None
#             if loss_metric == "Contrastive":
#                 loss = train_batch_contrastive(model, data, optimizer, criterion, device, subsequence_length)
#             elif loss_metric == "Triplet":
#                 loss = train_batch_triplet(model, data, optimizer, criterion, device, subsequence_length)
#             elif loss_metric == "SupConLoss":
#                 if use_KL_regularizer:
#                     loss = train_batch_KL(model, data, optimizer, criterion, device, subsequence_length, supervised=supervised, mode=mode )
#                 else:
#                     loss = train_batch(model, data, optimizer, criterion, device, subsequence_length, supervised=supervised, mode=mode )
#             else:
#                 if use_KL_regularizer:
#                     loss = train_batch_KL(model, data, optimizer, criterion, device, subsequence_length, supervised=supervised, mode=mode)
#                 else:
#                     loss = train_batch(model, data, optimizer, criterion, device, subsequence_length, supervised=supervised, mode=mode)
#             trainLogs.update(loss)
#         trainLogs.end_epoch()
#         with torch.no_grad():
#             N = len(val_dataloader)
#             for i, data in enumerate(val_dataloader):
#                 # if LOSS == "Contrastive":
#                 #     loss = eval_batch_contrastive(model, data,  criterion, device, subsequence_length)
#                 # elif LOSS == "Triplet":
#                 #     loss = eval_batch_triplet(model, data,  criterion, device, subsequence_length)
#                 if loss_metric == "SupConLoss":
#                     loss = eval_batch(model, data,  criterion, device, subsequence_length, supervised=supervised)
#                 else:
#                     loss = eval_batch(model, data,  criterion, device, subsequence_length, supervised=supervised)
#                 valLogs.update(loss)
            
#             if  valLogs.end_epoch():
#                 print('[Log] Saving model with loss: {}'.format(valLogs.bestAvg))
#                 torch.save(model, model_path) 

#     # fig = plt.figure()
#     # ax0 = fig.add_subplot(111, title="loss")
#     # ax0.plot(trainLogs.avgs, 'bo-', label='train')
#     # ax0.plot(valLogs.avgs, 'ro-', label='val')

#     # ax0.legend()
#     # fig.savefig(loss_path)
#     # model = torch.load(model_path)
#     if len(X_test) != 0:
#         return model.encode(X, device), model.encode(X_test, device)
#     return model.encode(X, device)