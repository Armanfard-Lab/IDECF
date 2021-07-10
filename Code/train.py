from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms

from IDECF import IDECF
from Networks import AutoEncoder, FCMNet



# dataset
dataset_size = 70000
batch_size=500
train_set = dset.MNIST(root='/home/mrsadeghi/Spectral_clustering_network', train=True,
                       transform=transforms.ToTensor(),download=True)
test_set = dset.MNIST(root='/home/mrsadeghi/Spectral_clustering_network', train=False,
                      transform=transforms.ToTensor(),download=True)
kwargs = {'num_workers': 1}
train1 = torch.utils.data.ConcatDataset([train_set,test_set])

data_loader = torch.utils.data.DataLoader(
                 dataset=train1,
                 batch_size=batch_size,
                 shuffle=True, **kwargs)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias.data)

        # torch.nn.init.xavier_uniform(m.bias.data)
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias.data)



if __name__ == '__main__':
    FCM_Net = FCMNet().cuda()
    FCM_Net.apply(weights_init)

    AE = AutoEncoder().cuda()
    AE.apply(weights_init)

    pretraining = True

    IDECF = IDECF(AE, FCM_Net, data_loader, dataset_size= dataset_size, batch_size=batch_size, pretraining_epoch = 200, max_iteration = 60, num_cluster = 10, m = 1.5, T1=10
                  , landa =0.1, feature_size =1*28*28)
    if pretraining ==True:
        IDECF.train_IDECF()
    else:
        AE = IDECF.pretraining()
        IDECF.train_IDECF()






