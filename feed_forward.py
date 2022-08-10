import numpy as np
from copy import deepcopy

def feed_forward(inputs, outputs, weights):
    pre_hidden = np.dot(inputs, weights[0]) + weights[1]
    hidden = 1/(1 + np.exp(-pre_hidden))
    pred_out = np.dot(hidden, weights[2]) + weights[3]
    mean_squared_error = np.mean(np.square(pred_out-outputs))
    return mean_squared_error

def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def relu(x):
    return np.where(x>0,x,0)

def linear(x):
    return x

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))    

def mse(p, y):
    return np.mean(np.square(p - y))

def mae(p, y):
    return np.mean(np.abs(p-y))

def binary_cross_entropy(p, y):
    return -np.mean(np.sum((y*np.log(p)+(1-y)*np.log(1-p))))

def categorical_cross_entropy(p, y):
    return -np.mean(np.sum(y*np.log(p)))    
 
def update_weights(inputs, outputs, weights, lr):
    original_weights = deepcopy(weights)
    temp_weights = deepcopy(weights)
    updated_weights = deepcopy(weights)
    original_loss = feed_forward(inputs, outputs, original_weights)
    for i, layer in enumerate(original_weights):
        for index, weight in np.ndenumerate(layer):
            temp_weights = deepcopy(weights)
            temp_weights[i][index] += 0.0001
            _loss_plus = feed_forward(inputs, outputs, temp_weights)
            grad =  (_loss_plus - original_loss)/(0.0001)
            updated_weights[i][index] -= grad*lr
    return updated_weights, original_loss        




 