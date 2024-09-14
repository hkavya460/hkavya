import numpy as np
import matplotlib.pyplot as plt

x = np.array([[0,0,1,1],[0,1,0,1]])
print(x.shape)
y = np.array([0,1,1,0])
print(y.shape)
n_x = x.shape[1]
print(n_x)
n_h = 2
n_o = 1
w1 = np.random.rand(n_h,n_x)
w2 = np.random.rand(n_o,n_h)
b1 = np.random.rand(n_h,1)
b2 = np.random.rand(n_o,1)
#
def relu(x):
    return np.maximum(0,x)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def forward():
    z1 = np.dot(w1,x.T) +b1
    a1 = relu(z1)
    z2 = np.dot(w2,a1) + b2
    a2 = sigmoid(z2)
    return z1,a1,z2,a2
def backward(z1,a1,z2,a2):

    dl_da2 = -1 *  (y - a2.flatten())
    dl_dz2 =  dl_da2 *   (z2 >0 ).astype(float)
    dl_dw2 = np.dot(dl_dz2,a1.T)
    dl_a1 = np.dot(w2.T ,dl_dz2 )
    dl_dz1 = dl_a1 *( z1 > 0) .astype(float)
    dl_dw1 = np.dot(dl_dz1,x.T)
    return dl_da2 ,dl_dz2,dl_a1,dl_dz1 ,dl_dw2,dl_dw1

iterations = 1000
loss =[]
lr = 0.001
for i in range(iterations):
    z1, a1, z2, a2 = forward()
    l = -(1 / 2) * np.sum(y * np.log(a2) + (1 - y) * np.log(1 - a2))
    # l = np.sum(-(y * np.log(a2.flatten()) + (1 - y) * np.log(1 - a2.flatten())))

    # l = 1/2 * ((y - a2.flatten()) ** 2)
    loss.append(l)
    dl_dw1, dl_db1, dl_dw2, dl_db2  = backward(z1,a1,z2,a2)
    w2  -=  lr * dl_dw2
    w1 -= lr * dl_dw1
plt.plot(loss)
plt.show()







