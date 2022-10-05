# import sys
# sys.path.insert(0, '/home/texs/Documentos/Repositories/mts_feature_learning/source')

from torch_snippets import *
from source.models.contrastive.models import SiameseNetwork, SiameseNetworkMH, train_batch, eval_batch, train_batch_KL, eval_batch_KL
from source.models.contrastive.losses import ContrastiveLoss, SupConLoss, TripletLoss
from source.models.contrastive.datasets import ContrastiveDataset
from torch.utils.data import DataLoader
from source.utils import create_dir
import torch
import random

SUBSEC_PORC = 0.95 # Porcentage of the window to be used as a sub-sequence
EXP_DIR = 'experiments'
EXPERIMENT_NAME = 'test'
EXP_PATH = os.path.join(EXP_DIR, EXPERIMENT_NAME)
create_dir(EXP_DIR)
create_dir(EXP_PATH)

# loss # "Contrastive" or "Triplet"  or "SupConLoss" or "SimCLR"
def getContrastiveFeatures(X, y, epochs = 100, batch_size = 32, head='linear', loss_metric = 'SimCLR', feat_size = 1024, encoding_size = 8, mode = 'subsequences', X_test=[], conv_filters = [16, 16, 16], conv_kernels = [5, 5, 5], use_KL_regularizer = True):
    train_dataset = ContrastiveDataset(X.astype(np.float32), y, use_label=False)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'

    if mode=='subsequences':
        subsequence_length = int(X.shape[2] * SUBSEC_PORC)
        print("Subsequence length: {}".format(subsequence_length))
    else:
        subsequence_length = X.shape[2]

    model      = SiameseNetworkMH(
        X.shape[1], 
        subsequence_length, 
        device,
        head = head, 
        encoding_size=encoding_size, 
        conv_filters = conv_filters, 
        conv_kernels = conv_kernels,
        feat_size = feat_size,
        use_KL_regularizer=use_KL_regularizer
    ).to(device)

    model_path = os.path.join(EXP_PATH, 'model.pt')
    loss_path = os.path.join(EXP_PATH, 'loss.png')

    if loss_metric == "Contrastive":
        criterion  = ContrastiveLoss().to(device)
    elif loss_metric == "Triplet":
        criterion  = TripletLoss(margin=4.0).to(device)
    elif loss_metric == "SupConLoss":
        criterion = SupConLoss().to(device)
        supervised = True
    else:
        criterion = SupConLoss().to(device)
        supervised = False

    trainLogs = ValueLogger("Train loss   ", epoch_freq=10)
    trainKlLogs = ValueLogger("Train KL loss   ", epoch_freq=10)
    # testLogs = ValueLogger( "Test loss    ", epoch_freq=10)
    # testKlLogs = ValueLogger("Test KL loss   ", epoch_freq=10)
    # valLogs = ValueLogger(  "Val loss     ", epoch_freq=10)
    # valKlLogs = ValueLogger("Val Kl loss   ", epoch_freq=10)

    # optimizer  = optim.AdamW(model.parameters(),lr = 0.0005, )
    # optimizer  = optim.AdamW(model.parameters(),lr = 0.00001) #Triplet
    # optimizer  = optim.SGD(model.parameters(),lr = 0.001)
    optimizer  = optim.Adam(model.parameters(),lr = 0.0005, weight_decay=0)

    for epoch in range(epochs):
        N = len(train_dataloader)
        for i, data in enumerate(train_dataloader):
            loss = None
            if loss_metric == "Contrastive":
                loss = train_batch_contrastive(model, data, optimizer, criterion, device, subsequence_length)
            elif loss_metric == "Triplet":
                loss = train_batch_triplet(model, data, optimizer, criterion, device, subsequence_length)
            elif loss_metric == "SupConLoss":
                if use_KL_regularizer:
                    loss = train_batch_KL(model, data, optimizer, criterion, device, subsequence_length, supervised=supervised, mode=mode )
                else:
                    loss = train_batch(model, data, optimizer, criterion, device, subsequence_length, supervised=supervised, mode=mode )
            else:
                if use_KL_regularizer:
                    loss = train_batch_KL(model, data, optimizer, criterion, device, subsequence_length, supervised=supervised, mode=mode)
                else:
                    loss = train_batch(model, data, optimizer, criterion, device, subsequence_length, supervised=supervised, mode=mode)
            trainLogs.update(loss)
        trainLogs.end_epoch()
        # with torch.no_grad():
        #     N = len(val_dataloader)
        #     for i, data in enumerate(val_dataloader):
        #         if LOSS == "Contrastive":
        #             loss = eval_batch_contrastive(model, data,  criterion, device, subsequence_length)
        #         elif LOSS == "Triplet":
        #             loss = eval_batch_triplet(model, data,  criterion, device, subsequence_length)
        #         elif LOSS == "SupConLoss":
        #             loss = eval_batch(model, data,  criterion, device, subsequence_length, supervised=supervised)
        #         else:
        #             loss = eval_batch(model, data,  criterion, device, subsequence_length, supervised=supervised)
        #         valLogs.update(loss)
            
        #     if  valLogs.end_epoch():
        #         print('[Log] Saving model with loss: {}'.format(valLogs.bestAvg))
        #         torch.save(model, model_path) 

    # fig = plt.figure()
    # ax0 = fig.add_subplot(111, title="loss")
    # ax0.plot(trainLogs.avgs, 'bo-', label='train')
    # ax0.plot(valLogs.avgs, 'ro-', label='val')

    # ax0.legend()
    # fig.savefig(loss_path)
    # model = torch.load(model_path)
    if len(X_test) != 0:
        return model.encode(X, device), model.encode(X_test, device)
    return model.encode(X, device)
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




