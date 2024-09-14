import  torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets
import  torchvision.transforms  as transforms

torch.manual_seed(42)
batch_size = 64
num_epoches = 10
learning_rate = 0.0001
num_classes = 10

#checking for the device run if the device contains GPU otherwise run in thee CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# downloading the datasets
transformer =  transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor(),transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])
train_data = datasets.CIFAR10(root='images',train=True,transform=transformer)
test_data = datasets.CIFAR10(root='images',train=False,transform=transformer)
# print(test_data)
# print(train_data)

#loading the dataa using the DataLoader
train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(test_data,batch_size=batch_size,shuffle=True)

#creating thr convulational layer

class ConvNeuralNet(nn.Module):
    def __init__(self,num_classes):
        super(ConvNeuralNet,self).__init__()
        self.convlayer1 = nn.Conv2d(3,32,3)  #in_channels = channel(gray or rgb)
        self.convlayer2 = nn.Conv2d(32,32,3)
        self.pool1 = nn.MaxPool2d(2,2)

        self.convlayer3 = nn.Conv2d(32,64,3)
        self.convlayer4 = nn.Conv2d(64,64,3)
        self.pool2 = nn.MaxPool2d(2,2)
        self.act = nn.ReLU()
        self.fc = nn.Linear(1600,128)
        self.relu1 = nn.ReLU()
        self.fc2 =nn.Linear(128,10)
    def forward(self,x):  #forward propagation
        out = self.convlayer1(x)
        out = self.convlayer2(out)
        out = self.pool1(out)
        out = self.convlayer3(out)
        out = self.convlayer4(out)
        out = self.pool2(out)

        out = out.reshape(out.size(0),-1)
        out = self.fc(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out

model = ConvNeuralNet(num_classes=10).to(device)


criterion = nn. CrossEntropyLoss()
optimizer= torch.optim.SGD(model.parameters(),lr =learning_rate,weight_decay=0.005,momentum=0.9)
total_step = len(train_loader)

#epoches for checking the loss in each step
for epoch in range (num_epoches):
    for i , (images,labels) in enumerate (train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'epoaches [{epoch+1}/{num_epoches}],Loss:{loss.item():.4f}')

#checking for the accuracy

with torch.no_grad():
    correct = 0
    total = 0
    for images,labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _,predicted = torch.max(outputs.data,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('accuracy of the network on the {} train images:{}% '.format(50000,100 * correct / total))


