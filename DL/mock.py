
# (i) Implement the following functions in Python from scratch. Generate 100 equally spaced values between -10 and 10. Call this list as  z. Implement the following functions and its derivative. Use z as input and plot both the function outputs and its derivative outputs.  (10 Marks)
# Sigmoid
# Tanh
# ReLU (Rectified Linear Unit)
import numpy as np
import matplotlib.pyplot as plt
#IMPLEMENTINFG THE SIGMOID FUNCTION AND ITS DERIVATIVE
def sigmoid_function(z):
    sigm = 1 / (1+ np.exp(-z))
    #Derivative the sigmoid function
    derv_sigm = sigm *  (1 - sigm)
    plt.plot(z, sigm, 'g', label='Sigmoid')
    plt.xlabel('Z values')
    plt.title('sigmoid function and its derivative ')
    plt.plot(z, derv_sigm, 'r', label='Derivative')
    plt.legend()
    return sigm,derv_sigm

def tanh_function(z):
    tanh = (np.exp(z) - np.exp(-z)) / (np.exp(z) +  np.exp(-z))
    derv_tanh = 1 - (( np.exp(z) - np.exp(-z))  **2 / (np.exp(z) + np.exp(-z))**2 )
    plt.plot(z, tanh, 'r', label='Tanh function')
    plt.plot(z, derv_tanh, 'b', label='derivative of Tanh')
    plt.xlabel('z values')
    plt.title('Tanh and its derivative function')
    plt.legend()
    plt.show()
    return tanh,derv_tanh

def relu_function(z):
    relu = np.maximum(0,z)
    deriv_relu = np.where(z>0,1,0)
    plt.plot(z,relu,'b',label='Relu function')
    plt.plot(z,deriv_relu,'r',label='Derivative of relu function')
    plt.xlabel('Z values')
    plt.ylabel('Relu function')
    plt.title('Relu and its derivative function')
    plt.legend()
    plt.show()
    return relu,deriv_relu

def main():
    z = np.linspace(-10,10,100)
    sigm,derv_sigm  = sigmoid_function(z)
    tanh ,derv_tanh = tanh_function(z)
    relu ,deriv_relu = relu_function(z)

# main()
######################################################################
#2) EMNNIST DATASET  AND CONSTRUCTING THE FEED FORWARD NEURAL NETWORK
import torch
import  torchvision
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
from torch.nn import Module ,functional,Sequential
torch.manual_seed(0)
train_data = datasets.EMNIST(root ='EMNIST',split='letters',train=True,download=True,transform=transforms.ToTensor())
test_data = datasets.EMNIST(root='EMNIST',split='letters',train=False ,download=True,transform=transforms.ToTensor())
print(len(train_data))
print(len(test_data))
images,labels = train_data[0]
print(images.shape)
#DATA LOADING WITH DATALOADER
batch_size =64
# batch_size = 128
train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(test_data,batch_size=batch_size)
image,label =  next(iter(train_loader))
print(image.shape)
train_targets = train_data.classes
print("The number of classes in the train data is :",len(train_targets))
#PLOTTING THE TRAIN_DATA IMAGES
grid = torchvision.utils.make_grid(image,nrow=10,padding=2,normalize=True)
plt.figure(figsize=(10,10))
plt.imshow(grid.permute(1,2,0))
plt.axis('off')
# plt.show()
#ITERATE THE DATA
n_iter = 3000
# n_epoch = n_iter / (len(train_loader) / batch_size)
# n_epoch = int(n_epoch)
# print(n_epoch)
# BUILDING THE CLASS
class EMNISTFNN(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super(EMNISTFNN,self).__init__()
        self.l1 = nn.Linear(input_dim,hidden_dim)
        #Non linera function
        self.act = nn.ReLU()
        #hidden layer
        self.l2 = nn.Linear(hidden_dim,hidden_dim)
        self.l3 = nn.Linear(hidden_dim,output_dim)
    def forward(self,x):  #Feed forward neural network
        out = self.l1(x)
        out = self.act(out)
        out = self.l2(out)
        out = self.act(out)
        out = self.l3(out)
        return out
input_dim = 28*28
hidden_dim = 100
output_dim = 27
model = EMNISTFNN(input_dim,hidden_dim,output_dim)

#INSTANTIATE THE LOSS FUNCTION
criterion = nn.CrossEntropyLoss()

#INSTANTIATE THE OPTIMIZER FOR GRADIENT OPTIMIZATIOON
optimizer = torch.optim.SGD(model.parameters(),lr = 0.01)

# FINDING THE LOSS AND ACCURACY
n_epoch =10
for epoch in range(n_epoch):
    for i,(images,labels) in enumerate(train_loader):
        images = images.view(-1,28*28)
        optimizer.zero_grad()
        output = model(images)
        # loss function
        loss = criterion(output,labels)
        #backpropagtion
        loss.backward()
        optimizer.step()


    # print(f"{epoch +1} /{n_epoch},loss:{loss.item()}")


#accuracy checking
total = 0
correct =0
for i,(test_img,test_label) in enumerate(test_loader):
    test_img = test_img.view(-1,28*28)
    output = model(test_img)
    _,predicated = torch.max(output.data,1)
    correct += (predicated==test_label).sum().item()
    total  += test_label.size(0)
    accuracy = correct / total * 100

print(f"the accuracy of the test data is :", accuracy )
print(f"the accuracy  with {batch_size} is ",accuracy)


# DOWNLOADING THE FLOWERS DATASET FROM VISION PACKAGES
import torch
import torchvision
from torchvision import datasets
from torchvision import  transforms
from torch.utils.data import DataLoader



#apply the train_transform and test_transform
train_transform = transforms.Compose([transforms.Resize((224,224)),transforms.RandomHorizontalFlip(),transforms.RandomRotation(10),transforms.ToTensor(),
                                      transforms.Normalize((0.45,0.46,0.47),(0.21,0.22,0.24))])
test_transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
                                      transforms.Normalize((0.45,0.46,0.47),(0.21,0.22,0.24))])
train_data = datasets.Flowers102(root='flower_data',split='train',download=True,transform= train_transform)
test_data = datasets.Flowers102(root='flower_data',split='test',download=True,transform= test_transform)

print("the size of the daatset is:",len(train_data))
print(len(test_data))
batch_size = 64
train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(test_data,batch_size=batch_size)
#KNOW ABOUT THE IMAGE DIMENSION INFORMATION
# print(train_data.__dict__)
all_labels = torch.tensor(train_data._labels)
num_class = torch.unique(all_labels)
print(f"the number of classes in the dataset :",len(num_class))


images,labels = next(iter(train_loader))
print(images.shape)
grid = torchvision.utils.make_grid(images,nrow=8,normalize=True)
plt.figure(figsize=(10,10))
plt.imshow(grid.permute(1,2,0))
plt.show()
import torch.nn as nn
#BUILD THE MODEL
class FlowerCNN(nn.Module):
    def __init__(self):
        super(FlowerCNN,self).__init__()
        self.conv1 = nn.Conv2d(3,224,3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(224,224,3)
        self.conv3 = nn.Conv2d(224,320,3)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(320 *26 *26 ,160)
        self.fc2 = nn.Linear(160,128)
        self.fc3 = nn.Linear(128,102)
    def forward(self,x):
        out = self.pool(self.relu(self.conv1(x)))
        out = self.pool(self.relu(self.conv2(out)))
        out = self.pool(self.relu(self.conv3(out)))
        out = out.view(out.size(0),-1,)
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out

model = FlowerCNN()
#INSTANTIATE THE LOSS FUNCTION
criterion = nn.CrossEntropyLoss()
#instantiate the optimizer
optimizer = torch.optim.SGD(model.parameters(),lr=0.1)
# print(model.parameters())
# print(len(list(model.parameters())))
# print(len(list(model.parameters())))

n_epoch = 10
for epoch in range(n_epoch):
    for i,(images,labels)in enumerate(train_loader):
        output = model(images)
        loss = criterion(output,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"epoches:{epoch+1}/{n_epoch},Loss:{loss.item()}")

#ACCURACY CALCULATION
total = 0
correct =0
with torch.no_grad():
    for i,(image,label) in enumerate(train_loader):
        output = model(image)
        _,predicted = torch.max(output.data,1)
        total += label.size(0)
        correct += (predicted==label).sum().item()
        accuracy = correct / total * 100
    print(f"the accuracy of the test datset with {batch_size} is :",accuracy)

