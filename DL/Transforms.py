#transform - manipulation of data to make suitable for training
import torch
from torchvision.transforms import ToTensor ,Lambda
from torchvision import datasets
from torch import nn
#ToTensor (converts the image .pil format or numpy array to pytorch tensors and normalized in scale [0,1]

dataset = datasets.FashionMNIST(root ='data',train=True,download=True,transform=ToTensor(),
                                target_transform=Lambda(lambda y : torch.zeros(10,dtype=torch.float).scatter_(torch.tensor(y),value=1)))


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"

)
print(f"the device used to run is {device} device ")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(nn.Linear(28*28,512),
                                                nn.ReLU(),nn.Linear(512,512),
                                                nn.ReLU(),nn.Linear(512,10))

    def forward(self,x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
# print(model)

# x = torch.rand(1,28,28,device=device)
# logits = model(x)
# pred_prob =nn.Softmax(dim=1)(logits)
# y_pred = pred_prob.argmax(1)
# print(y_pred)

input_image = torch.rand(3,28,28)
# print(input_image.size())

flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())