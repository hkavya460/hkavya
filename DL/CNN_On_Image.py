import numpy as np
import torch
import torchvision.utils
from torch.utils.data import dataset ,DataLoader
from torchvision import datasets
import torch.nn  as  nn
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

torch.random.seed()
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
train_data = datasets.CIFAR10(root='images',train=True,download=True,transform=transform)
test_data = datasets.CIFAR10(root='images',train=False,download=True,transform=transform)

print(test_data)

batchsize = 64
train_loader = DataLoader(train_data,batch_size=batchsize,shuffle=True,num_workers=2)
test_loader = DataLoader(test_data,batch_size=batchsize,shuffle=True ,num_workers=2)
classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

def imagshow(img):
    img = img / 2 + 0.5
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img,(1,2,0)))
    plt.show()

#get some random training images
datailer = iter(train_loader)
images,labels = next(datailer)
imagshow(torchvision.utils.make_grid(images))



