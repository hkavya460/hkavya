import numpy as np
from sklearn.datasets import make_classification,make_circles
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


x1 = np.array([[0],[1],[1],[0]])
print(x1.shape)
x2 = np.array([[0],[1],[0],[1]])
x3 = np.array([[1],[1],[1],[1]])
y = np.array([[0],[1],[1],[0]])
w1 = np.random.rand(4,4)
b1 = np.random.rand(4,1)
w2 = np.random.rand(2,4)
b2 = np.random.rand(2,1)
#feed foreward
def relu(x):
    return np.max(0,x)
def forward_propgation():
    h1 = np.dot(w1,x1) + b1
    act1 = relu(h1)
    #2nd layer
    h2 = np.dot(act1,w2) + b2
    y_pred = relu(h2)


forward_propgation()

