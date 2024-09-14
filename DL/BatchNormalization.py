#Batch normaaization and layer normalization

import  torch
from torch.utils.data  import DataLoader
from torchvision import datasets

import torch.nn as nn
import   torch.nn.functional as f
from torchvision.transforms import ToTensor
from torch.nn import Conv2d ,MaxPool2d ,ReLU
import matplotlib.pyplot as plt
import numpy as np
def relu(x):
    return np.maximum(0,x)
#batch normalization from scratch

def normalization():
    inputs = np.random.rand(6,1).round(3)
    num_inputs = 6
    h1,h2,h3 = 4 ,2,1

    w1 = np.random.rand(h1,num_inputs)
    w2 = np.random.rand(h2,h1)
    w3 = np.random.rand(h3,h2)
    epislon = 0.01
    gamma = 1
    beta = 0

   #normalization
    x_norm = (inputs - np.mean(inputs)) / np.std(inputs)

    #feed forward neural network for first layer
    b1 = np.random.rand(h1,1)
    a1 = relu(np.dot(w1,x_norm) + b1 )
    h1_norm = gamma * ( (a1 - np.mean(a1)) / (np.std(a1) + epislon) + beta )
    print(h1_norm)
    #for second layer
    b2 = np.random.rand(h2,1)
    a2 = relu(np.dot( w2,h1_norm) + b2)
    h2_norm = gamma * ((a2- np.mean(a2)) / (np.std(a2) + epislon) + beta)
    print(h2_norm.shape)
    # for third layer 
    b3 = np.random.rand(h3)
    a3 = relu(np.dot(w3, h2_norm) + b3)
    h3_norm = gamma * ((a3- np.mean(a3)) / (np.std(a3) + epislon) + beta)
    # print(h3_norm.shape)

normalization()

#batch normalization for training data
import tensorflow as tf
torch.manual_seed(41)
import torch.nn.init
data = datasets.MNIST(root='data',download=True)
print(len(data))
# plottting the datasets images

image,labels = data[0]

plt.imshow(image,cmap='gray')
print("Label",labels)
plt.show()

train_data = datasets.MNIST(root='mnist',download=True,train=True,transform=ToTensor())
image,labels = train_data[0]
image_tensor = image
print(image_tensor.shape,labels)
val_data= datasets.MNIST(root='mnist',download=True,train=False,transform=ToTensor())
test_data = datasets.MNIST(root='mnist',download=True,train=False,transform=ToTensor())
#finding the number of targets and size of the data
train_labels = train_data.targets
print(train_labels)
n_classes = len(torch.unique(train_labels))
print(n_classes)
#
# image,labels = train_data[0]
image,labels = val_data[0]
plt.imshow(image[0],cmap='gray')
print("Label",labels)
plt.show()

print(image.shape)
print(len(train_data))
print(len(val_data))

#polotting the  images


batch_size = 64
train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(val_data,batch_size=batch_size)
def normalize(train_loader):
    for data in enumerate(train_loader):
        train_normalize = (data - torch.mean(data)) / torch.std(data)
    return train_normalize

keep_prob = 0.25
class BatchNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3),nn.BatchNorm2d(32),nn.ReLU(),
                                nn.MaxPool2d(2,2),nn.Dropout(p=keep_prob))
        # self.norm = normalize(self.l1)
        self.l2 = nn.Sequential(nn.Conv2d(32,32,3),nn.BatchNorm2d(32),
                                nn.ReLU(),nn.MaxPool2d(2,2),nn.Dropout(p=keep_prob))

        self.l3 = nn.Sequential(nn.Conv2d(32,64,3),nn.BatchNorm2d(64),nn.ReLU(),
                                nn.MaxPool2d(2,2),nn.Dropout(p=keep_prob))


        self.fc1 = nn.Sequential(nn.Linear(64 ,128),nn.BatchNorm1d(128),nn.ReLU())
        # self.norm = normalize(self.l4)
        self.fc2 = nn.Linear(128,10)

    def forward(self,x):
        out = self.l1(x)
        # out  = normalize(out)

        out = self.l2(out)
        # out = normalize(out)
        out = self.l3(out)
        out = out.view(out.size(0),-1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out



model = BatchNorm()
model

from torch.autograd import Variable
lr = 0.01
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=lr)
#tarining model loss ,accuarcy calculation
train_cost =[]
train_acc =[]
n_epochs = 10
total_batch  = len(train_data )
for epoch in range (n_epochs):
    avg_cost =0
    for i,(batch_x,batch_y) in enumerate(train_loader):
        x = Variable(batch_x)
        y = Variable(batch_y)
        optimizer.zero_grad()
        hypothesis = model(x)
        cost = criterion(hypothesis,y)
        cost.backward()
        optimizer.step()
        #printing the accuracy ,cost
        avg_cost += cost.item() / total_batch
    # print(f"epochs :[{epoch+1} /{n_epochs}] ,loss:{cost.item():3f}")
with torch.no_grad():
    correct =0
    total = 0

    for i,(image,labels) in enumerate(train_loader):
        output  = model(image)
        _,predicted = torch.max(output.data,1)
        total += labels.size(0)
        correct += (predicted==labels).sum().item()
    print("the accuracy of the train images is ",correct/total * 100)

# plotting
import torchvision
dataiter = iter(train_loader)
image,labels = next(dataiter)
npimg = torchvision.utils.make_grid(image).numpy()
plt.imshow(np.transpose(npimg))
plt.show()



def predict_image(img,model):
    xb = img.unsqueeze(0)
    yb = model(xb)
    _,pred = torch.max(yb,dim=1)
    return (pred[0].item())
img,labels = test_data[30]
plt.imshow(img[0],cmap='gray')
print('Label:',labels,'predicted_image :',predict_image(img,model))
plt.show()













