import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import  train_test_split
import torch
from torch import nn
import  torch.nn.functional as  f




iris  = datasets.load_iris()
df = pd.DataFrame(iris.data)
df['class'] = iris.target

df['class'] = df['class'].replace('virginica',0.0)
df['class'] = df['class'].replace('versicolor',1.0)
df['class'] = df['class'].replace('setosa',2.0)



print(df.shape)
x = df.iloc[:,:-1]
print(x.shape)
y = df.iloc[:,-1]
#converting into numpy values
x = x.values
y = y.values

class simpleneuralNetwork(nn.Module):
    def __init__(self,input_features,hidden_layer1=8,hidden_layer2=9,output_feature=3):
        super().__init__()
        self.fc1 = nn.Linear(input_features,hidden_layer1)
        self.fc2 = nn.Linear(hidden_layer1,hidden_layer2)
        self.out = nn.Linear(hidden_layer2,output_feature)

        #forward
    def forward(self,x):
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.out(x)
        return x

model = simpleneuralNetwork(input_features=x.shape[1])

#splitting the model

torch.manual_seed(41)
x_train ,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30,random_state=41)
x_train  = torch.FloatTensor(x_train)
x_test = torch.FloatTensor(x_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

#
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
from sklearn.metrics import accuracy_score
epochs = 100
size = len(x)
losses =[]
correct =0
for i in range(epochs):
    y_pred = model.forward(x_train)
    loss = criterion(y_pred,y_train )
    losses.append(loss.detach().numpy())
    correct += (y_pred.argmax(1)==y_train).type(torch.float).sum().item()


    #print evey 10 epochs
    if i% 10 ==0 :
        print(f"epoch:{i},loss:{loss}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    correct /= size
print(f"accuracy {(100 * correct):>0.1f}%")
#plotting
plt.plot(range(epochs),losses)
plt.title('N_epochs vs loss')
plt.xlabel('number of epochs')
plt.ylabel('Loss')
plt.show()

# accuracy calculation


