import numpy as np
import torch
from sklearn.datasets import load_diabetes
from sklearn.model_selection import  train_test_split
import matplotlib.pyplot as plt

data = load_diabetes()
x = data['data']
y = data['target']

# print(f"the shape of x  is ", x.shape)
# print(f" shape of y is ",y.shape)


from torch.utils.data import  Dataset  ,DataLoader

class diabetes(Dataset):
    def __init__(self,x,y):
        self.x = torch.tensor(x,dtype=torch.float32)
        self.y = torch.tensor(y,dtype=torch.float32)
        self.length = self.x.shape[0]
    def __getitem__(self, idx):
        return self.x[idx],self.y[idx]

    def __len__(self):
        return self.length
dataset = diabetes(x,y)

dataloader = DataLoader(dataset=dataset,shuffle=True,batch_size=100)
from torch import nn
class net(nn.Module):
    def __init__(self,input_size,output_size):
        super(net,self).__init__()
        self.l1 = nn.Linear(input_size,5)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(5,output_size)
    def forward(self,x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out
model = net(x.shape[1],1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr = 0.001)
epochs = 200

#training
costval =[]

for   epoch in range (epochs):
    for i,(x_train,y_train) in enumerate(dataloader):
        y_pred = model(x_train)
        cost  = criterion(y_pred,y_train.reshape(-1,1))

        #optimizer
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        costval.append(cost)
print(costval)



