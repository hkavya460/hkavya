import numpy as np
import matplotlib.pyplot as plt

x = np.random.rand(4,1).round(3)
w1 = np.random.rand(4,4)
w2 = np.random.rand(3,4)
w3 = np.random.rand(2,3)
b1 = np.random.rand(4,1)

# dropout
drop_prob = 0.25
def dropout(x,drop_prob):
    keep_prob = 1 - drop_prob
    mask = np.random.uniform(0,1.0,x.shape) < keep_prob
    if keep_prob > 0.0:
        scale = 1/keep_prob
    else:
        scale=0
    return mask * x * scale
def relu(x):
    return np.maximum(0,x)
import torch.nn as nn
h1_linear = np.dot(w1,x) + b1
h1 = relu(h1_linear)
h1 = dropout(h1,0.25)
print(h1)
b2  = np.random.rand(3,1)
h2_linear = np.dot(w2,h1) + b2
h2 =  relu(h2_linear)
h2 = dropout(h2,0.25)
print(h2)
b3 = np.random.rand(2,3)
yhat_linear = np.dot(w3,h2) +b3
# yhat = nn.Softmax(yhat_linear)
yhat = relu(yhat_linear)
print(yhat)

#dropout using dataset

