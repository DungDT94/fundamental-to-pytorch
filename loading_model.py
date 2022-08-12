import torch
import torch.nn as nn
#khi load model co san phai tao mot model trong
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 1)).to(device)

state_dict = torch.load('mymodel.pth')
model.load_state_dict(state_dict)
model.to(device)


val = [[2,8], [3,9], [8,13]]
print(model(torch.tensor(val).float().to(device))) 