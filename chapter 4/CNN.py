import torch
from torch import nn
from torch.utils.data import TensorDataset, Dataset,DataLoader
from torch.optim import SGD, Adam
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary

X_train = torch.tensor([[[[1,2,3,4],[2,3,4,5],[5,6,7,8],[1,3,4,5]]],  
                        [[[-1,2,3,-4],[2,-3,4,5], [-5,6,-7,8],[-1,-3,-4,-5]]]]).to(device).float()
print(X_train.shape)                        
X_train /= 8
y_train = torch.tensor([0,1]).to(device).float()

def get_model():
    model = nn.Sequential(
        nn.Conv2d(1,1, kernel_size =3),  #1 filter with 3x3 kernel size
        nn.MaxPool2d(2),  # pooling size 2 
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(1,1),
        nn.Sigmoid(),

    ).to(device)
    loss_fn = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr = 1e-3 )
    return model, loss_fn, optimizer

def traing_batch(x,y,model, opt, loss_fn):
    model.train()
    prediction = model(x)
    batch_loss = loss_fn(prediction.squeeze(0), y)
    batch_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return batch_loss.item()


model, loss_fn, optimizer = get_model()
trn_dl = DataLoader(TensorDataset(X_train, y_train))
#summary(model, X_train)   
#help(nn.Conv2d) 

for epoch in range(20):
    for ix, batch in enumerate(iter(trn_dl)):
        x, y = batch
        batch_loss = traing_batch(x,y, model, optimizer, loss_fn) 
        

print(model(torch.tensor([[[[1,2,5,4],[2,3,56,5],[5,6,76,8],[1,3,64,5]]]]).float().to(device))) 
#print(model(X_train[:1]))
print(list(model.children()))