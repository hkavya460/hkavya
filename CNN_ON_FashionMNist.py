import torch
import torchvision.utils
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as f
from torchvision import datasets
import torchvision.transforms as transforms
import  numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)
images_size = 28
def imageshow(data):
    plt.imshow(data[0].numpy().reshape(images_size,images_size),cmap='gray')
    plt.title('y = ' +  str(data[1]))
    plt.show()

##############################################################
transformer = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.5,),(0.5,),inplace=True)])

batch_size = 100
training_data = datasets.FashionMNIST(root="data",train=True ,transform=transformer)
train_loader = DataLoader(training_data,batch_size=batch_size)
test_data = datasets.FashionMNIST(root="data",train=False,transform=transformer)
test_loader = DataLoader(test_data,batch_size=batch_size)

class ConvuNeurNet(nn.Module):
    def __init__(self):
        super(ConvuNeurNet,self).__init__()
        self.conv1 = nn.Conv2d(1,16,5,padding=2)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(16 ,32,5,padding=2)
        self.fc1 = nn.Linear(32 * 7 * 7 ,10)
        # self.fc2 = nn.Linear(120,8)
        # self.fc3 = nn.Linear(84,10)

    def forward(self,x):
        out = self.pool(f.relu(self.conv1(x)))
        out = self.pool(f.relu(self.conv2(out)))
        #flatten the matrices
        out = out.view(out.size(0),-1)
        out = f.relu(self.fc1(out))
        # out = f.relu(self.fc2(out))
        # out = self.fc3(out)

        return out

model = ConvuNeurNet()

#optimization function
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr = 0.01,momentum=0.9)



num_epochs = 10

for epoch in range(num_epochs):
    for i, data in enumerate(train_loader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # print(f" epochs [{epoch + 1} / {num_epochs}] ,loss :{loss.item()}")

#model evaluation

correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        input,labels = data
        output = model(input)
        _,yhat = torch.max(output.data,1)
        total += labels.size(0)
        correct += (yhat == labels ).sum().item()
    print(f'accuracy of the  test data is : {100 * correct / total} % ')


dataiter = iter(train_loader)
image,labels = next(dataiter)
#unnormalizing the image pixel values
# x * 0.5 + 0.5 for unnormalization
image = image /2 + 0.5
npimg = torchvision.utils.make_grid(image).numpy()
plt.imshow(np.transpose(npimg,(1,2,0)))  # imshow(plot by h * w * c but transposing leads to c*h*w
plt.show()







