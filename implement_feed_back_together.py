from copy import deepcopy
import feed_forward as ff
import numpy as np
import matplotlib.pyplot as plt




x = np.array([[1,1]])
y = np.array([[0]])

W = [
     np.array([[-0.0053, 0.3793],
               [-0.5820, -0.5204],
               [-0.2723, 0.1896]], dtype=np.float32).T,            #6 weight of 2 input and 3 hidden layer
     np.array([-0.0140, 0.5607, -0.0628], dtype=np.float32),       # 3 bias 
     np.array([[ 0.1528,-0.1745,-0.1135]],dtype=np.float32).T,     # 3 weight of 3 hidden layer and output    
     np.array([-0.5516], dtype=np.float32)                         # bias of output
]

losses = []
for epoch in range(100):
    W, loss = ff.update_weights(x,y,W,0.01)
    losses.append(loss)

print(losses[:5])
print(W)

plt.plot(losses)
plt.title('Loss over increasing number of epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss value')
plt.show()


