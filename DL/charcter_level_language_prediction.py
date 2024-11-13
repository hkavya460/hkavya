#chracter level prediction Steps to follow
#1.preprocees
#a)Tokenization
#data cleaning - lower,spaces removal,non ascii charcter conversion
#2.chracter encoding
#3.sequence prepartion
#4.Model building - Input_dim,output_dim,num_layer for lstm
#Input layer ,output layer ,lstm layer
#5.train the model

import re
import string
from string import punctuation

import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
import numpy as np
torch.manual_seed(0)
with open('/home/ibab/PycharmProjects/python_project/DL/data/data/file.txt','r') as f:
    text = f.read()
print("before preprocessing",text)
#preprocess the text remove puntcutations and spaces

text = ''.join(ch for ch in text if ch not in string.punctuation).lower()
print(text)
char = sorted(set(text))
#charcter to integer
ch_to_int = {ch:i for i,ch in enumerate(char)}
int_to_char = {i:ch for ch ,i in enumerate(char) }
print(ch_to_int)

#customdataset

class TextCustomDataset(Dataset):
    def __init__(self,text,seq_len):
        self.text = text
        self.seq_len = seq_len
        self.ch_to_int = ch_to_int
        self.int_to_char = int_to_char
    def __len__(self):
        return len(self.text) - self.seq_len
    def __getitem__(self, idx):
        seq = self.text[idx:idx+self.seq_len]
        label = self.text[idx+self.seq_len]
        seq_idx = [self.ch_to_int[char] for char in seq]
        label_idx = [self.ch_to_int[label]]
        return torch.tensor(seq_idx),torch.tensor(label_idx)
seq_dataset = TextCustomDataset(text,seq_len=10)
batch_size = 32

seq_dataloader = DataLoader(seq_dataset,batch_size=batch_size,shuffle=True)
#build the model

class RNN(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super(RNN,self).__init__()
        self.embedded = nn.Embedding(input_dim,hidden_dim)
        self.lstm = nn.LSTM(hidden_dim,hidden_dim,num_layers=1,batch_first=True)
        self.fc = nn.Linear(hidden_dim,output_dim)
    def forward(self,x):
        embed = self.embedded(x)
        out,_ = self.lstm(embed)
        out = self.fc(out[:,-1,:])
        return out

input_dim = len(char)
output_dim =len(char)
hidden_dim = 128
model =  RNN(input_dim,hidden_dim,output_dim)
print(model)

#loss function
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)

n_epochs = 10
loss = 0
for epoch in range (n_epochs):
    model.train()

    for seq_batch,target in seq_dataloader:

        optimizer.zero_grad()
        output = model(seq_batch)
        batch_loss  = criterion(output,target.squeeze())
        batch_loss.backward()
        optimizer.step()
        loss += batch_loss.item()
    print(f"Epochs : {epoch}/{n_epochs},Loss:{batch_loss:.4f}")

#model evaluation
loss_eval = 0
total =0
correct =0
model.eval()
with torch.no_grad():
    for batch_seq, label in seq_dataloader:
        output = model(batch_seq)

        loss_eval += criterion(output,label.squeeze())
        _,pred = torch.max(output,dim=1)
        correct += (pred == label.squeeze()).sum().item()
        total +=  label.size(0)
        acc = correct / total * 100

    print(f"Test Loss :{loss_eval.item():.4f},accuracy :{acc:4f}")












