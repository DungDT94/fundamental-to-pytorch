import torch
import torch.nn as nn
from torch.optim import SGD

x = [[1.0,2.0],[3.0,4.0],[5.0,6.0],[7.0,8.0]]
y = [[3.0],[7.0],[11.0],[15.0]]


X = torch.tensor(x, requires_grad=True).float()
Y = torch.tensor(y).float()



device = 'cuda' if torch.cuda.is_available() else 'cpu'

X = X.to(device)
Y = Y.to(device)

class MyNeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_to_hidden_layer = nn.Linear(2,8)
        self.hidden_layer_activation = nn.ReLU()
        self.hidden_to_output_layer = nn.Linear(8,1)

    def forward(self, x):
        x = self.input_to_hidden_layer = nn.Linear(x)
        x = self.hidden_layer_activation = nn.ReLU(x)
        x = self.hidden_to_output_layer = nn.Linear(x)
        return x



myNet = MyNeuralNet().to(device)   
#print(myNet.input_to_hidden_layer.weight)
#print(myNet.input_to_hidden_layer.bias)
#print(myNet.hidden_to_output_layer.weight)
#for par in mynet.parameters():
    #print(par)

loss_func = nn.MSELoss()
_Y = myNet(X)    #tinhs toan output khi cho input chay qua network
print(_Y)
loss_value = loss_func(_Y, Y) # tinh loss value cua gia tri duj doan va gia tri thuc
print(loss_value)
opt = SGD(myNet.parameters(), lr = 0.001) # ham toi uu voi learning rate 0.001

loss_history = []


for _ in range(50):
    opt.zero_grad() #loai bo nhung gradient cua epoch truoc
    loss_value = loss_func(myNet(X),Y) # toan toan loss 
    loss_value.backward() # perform back-propagation
    opt.step() #update weight
    loss_history.append(loss_value) # cho loss vao mang
    
    

for j in range(len(loss_history)):
    print(loss_history[j])










