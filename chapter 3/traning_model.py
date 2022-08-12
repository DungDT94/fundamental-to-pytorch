from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
from torchvision import datasets
data_folder = 'D:\machine_learning_sach_code\chapter 3\data' # This can be any directory you
# want to download FMNIST to
fmnist = datasets.FashionMNIST(data_folder, download=True, train=True)
tr_images = fmnist.data
tr_targets = fmnist.targets

'''
class FMNISTDataset(Dataset):
    def __init__(self,x,y):
        x = x.float()
        x = x.view(-1, 28*28)
    def __getitem__(self, ix):
        x,y = self.x[ix], self.y[ix]
        return x.to(device), y.to(device)
    def __len__(self):
        return len(self.x)   

def get_data():
    train = FMNISTDataset(tr_images, tr_targets)
    trn_dl = DataLoader(train, batch_size =32, shuffle = True)
    return trn_dl   '''           