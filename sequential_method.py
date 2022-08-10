from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import code_torch_2 as ct
from torch.optim import SGD
from torchsummary import summary


x = [[1,2],[3,4],[5,6],[7,8]]
y = [[3],[7],[11],[15]]
X = torch.tensor(x).float()
Y = torch.tensor(y).float()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
X = X.to(device)
Y = Y.to(device)
class MyDataset(Dataset):
    def __init__(self,x,y):
        self.x = torch.tensor(x).float()
        self.y = torch.tensor(y).float()
    def __len__(self):
        return len(self.x)    
    def __getitem__(self, ix):
        return self.x[ix], self.y[ix]

ds = MyDataset(X, Y) 
dl = DataLoader(ds, batch_size =3, shuffle = True)  # chon batch_size = 3, moi 1 lan training no se chon 3 phan tu cua x
model = nn.Sequential(
    nn.Linear(2,8),
    nn.ReLU(),
    nn.Linear(8, 1)
).to(device)

#summary(model, torch.zeros(1,2))

loss_func = nn.MSELoss()
opt = SGD(model.parameters(), lr = 0.001)
import time 
loss_history = []
start = time.time()
for _ in range(50):
    for ix, iy in dl:
        opt.zero_grad()
        loss_value = loss_func(model(ix),iy)
        loss_value.backward()
        opt.step()
        loss_history.append(loss_value)
end = time.time()
print(end - start)

val = [[1,2],[10, 11], [1.5, 2.5]]
print(model(torch.tensor(val).float().to(device))) 