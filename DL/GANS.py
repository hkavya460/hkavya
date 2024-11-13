import numpy as np
import torch
import torchvision
from torchvision import  transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn as nn
#simualting the sinsodial function
train_data_length  = 1024
train_data = torch.zeros((train_data_length,2))
train_data[:,0] = 2 * np.pi * (torch.rand(train_data_length))
train_data[:,1] =  torch.sin(train_data[:,0])
train_data_labels = torch.zeros((train_data_length))
train_set = [(train_data[i] ,train_data_labels[i]) for i in range (train_data_length)]

plt.plot(train_data[:,0],train_data[:,1],".")
plt.xlabel("x1")
plt.ylabel("x2")

plt.show()

#load the data with toch dataloader
batch_size =32
train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True)

#implement discrimnator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.model = nn.Sequential(nn.Linear(2,256),nn.ReLU(),nn.Dropout(0.3),nn.Linear(256,128),
                                   nn.ReLU(),nn.Dropout(0.3),nn.Linear(128,64),nn.ReLU(),
                                   nn.Dropout(0.3),nn.Linear(64,1),nn.Sigmoid())


    def forward(self,x):
        out = self.model(x)
        return out

discriminator = Discriminator()

#implement the class Genertor
class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.model = nn.Sequential(nn.Linear(2,16),nn.ReLU(),nn.Dropout(0.3),
                                   nn.Linear(16,32),nn.ReLU(),nn.Dropout(0.3),
                                   nn.Linear(32,2),nn.ReLU(),nn.Dropout(0.3))
    def forward(self,x):
        out = self.model(x)
        return out

generator = Generator()




lr = 0.001
num_epochs =300
criterion = nn.BCELoss()
optimizer_discriminator  = torch.optim.Adam(discriminator.parameters(),lr=lr)
optimizer_generator = torch.optim.Adam(generator.parameters(),lr=lr)

#training the model
for epoch in range(num_epochs):
    for i,(real_samples,_) in enumerate(train_loader):
        real_sample_labels = torch.ones((batch_size,1))
        latent_samples = torch.randn((batch_size,2))
        generated_samples = generator(latent_samples)
        generator_sample_output = torch.zeros((batch_size,1))
        all_samples = torch.cat((generated_samples,real_samples))
        all_samples_labels = torch.cat((real_sample_labels,generator_sample_output))

    #training the discriminator
    discriminator.zero_grad()
    output_discriminator = discriminator(all_samples)
    loss_discrimnator  = criterion(output_discriminator,all_samples_labels)
    loss_discrimnator.backward()
    optimizer_discriminator.step()

    #DATA FOR THE GENERATOR
    latent_samples = torch.rand((batch_size,2))
    #training the generator

    generator.zero_grad()
    generated_samples = generator(latent_samples)
    output_dis_generated = discriminator(generated_samples)
    loss_generator = criterion(output_dis_generated,real_sample_labels)
    loss_generator.backward()
    optimizer_generator.step()
    #show loss
    if epoch % 10 ==0 and  i == batch_size -1:
        print(f"Epoch:{epoch} Loss D.:{loss_discrimnator.item()}")
        print(f"Epoch:{epoch} Loss G.:{loss_generator.item()}")

