import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader ,Dataset
from torchvision.transforms import ToTensor
from torchvision import datasets
from torchvision.io import read_image
import os
import matplotlib.pyplot as plt

training_data = datasets.FashionMNIST(root='data',train=True,download=True,transform=ToTensor())
test_data = datasets.FashionMNIST(root='data',train=False,download=True,transform=ToTensor())
label_maps = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot"
}

df = pd.DataFrame(list(label_maps.items()), columns=['Label', 'Category'])
# # Save the DataFrame to a CSV file
df.to_csv('annotation_file.csv', index=False)
class CustomImagedataset(Dataset):
    def __init__(self,data,annotation_file,transform=None,target_transform=None):
        self.data = data
        self.img_labels = pd.read_csv(annotation_file)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)
    def __getitem__(self, idx):
        img_path = os.path.join(self.data,self.img_labels[idx,0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx,1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image,label
custom_dataset = CustomImagedataset(data='data',annotation_file='annotation_file.csv',transform=ToTensor(),target_transform=ToTensor())
custom_loadet = DataLoader(custom_dataset,batch_size=64,shuffle=True)

batch_size = 64
train_loader =DataLoader(training_data,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(test_data,batch_size=batch_size,shuffle=True)

train_feature,train_label = next(iter(train_loader))
print(f"the training feature size is{train_feature.size()}")
print(f"the training label size is {train_label.size()}")
img = train_feature[0].squeeze()
label = train_label[0]
print(f"label :",label)
plt.imshow(img,cmap = 'gray')
plt.title(label_maps[label.item()])
plt.show()





