import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import  numpy as np
import matplotlib.pyplot as plt

#training the data
def loading():
    #downloading the training data
    training_data = datasets.FashionMNIST(root="data",train=True ,download=True,transform=ToTensor())
    # downloading the testing  data
    test_data = datasets.FashionMNIST(root="data",train=False,download=True,transform=ToTensor())
    #batchwise training so initalize the batchsize to 64
    batch_size = 64
    train_loader = DataLoader(training_data,batch_size=batch_size)
    test_loader = DataLoader(test_data,batch_size=batch_size)
    return training_data ,train_loader,test_loader


#creating the models
#finding file to run whether to in gpu or cpu

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu")
print(f" using {device} device")

# define model
def neural_network():
    #flatten will faltten the size
    #flattening the 2D to 1D using the flatten
    model = nn.Sequential(nn.Flatten(),
                          nn.Linear(28*28,512),
                          nn.ReLU(),nn.Linear(512,512),
                          nn.ReLU(),nn.Linear(512,10))

    return  model

def gradinet(model):
    #finding the loss function and optimizing using the stochastic gradient function
    loss_fun = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return loss_fun,optimizer

#Trainig function

def train(dataloader, model, loss_fun, optimizer):
    size = len(dataloader.dataset)
    model.train()

    for batch ,(x,y) in enumerate(dataloader) :
        x,y =x.to(device),y.to(device)
        #predict the error
        pred = model(x)
        loss = loss_fun(pred,y)

        #backpropogation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


        if batch % 100 == 0:
            loss ,current = loss.item(),(batch+1) * len(x)
            # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")




def test(dataloader,model,loss_fun):
    size = len(dataloader.dataset)
    num_batches =len(dataloader)
    model.eval()
    test_loss ,correct = 0,0
    with torch.no_grad():
        for x ,y in dataloader:
            x,y = x.to(device),y.to(device)

            pred = model(x)
            test_loss += loss_fun(pred,y).item()
            correct += (pred.argmax(1)==y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    # print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def main():
    training_data ,train_loader, test_loader = loading()
    model = neural_network().to(device)
    loss_fun ,optimizer = gradinet(model)
    train(train_loader, model, loss_fun, optimizer)
    test(test_loader,model,loss_fun)
    torch.save(model.state_dict(),"model.pth")
    model.load_state_dict(torch.load("model.pth"))

    # epoches = 5
    # for t in range(epoches):
    #     # print(f"Epoch {t + 1}\n-------------------------------")
    #     train(train_loader, model, loss_fun, optimizer)
    #     test(test_loader, model, loss_fun)
    # print("Done!")
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

    fig, axes = plt.subplots(3, 3, figsize=(8, 8))

    for i in range(3):
        for j in range(3):
            sample_idx = torch.randint(len(training_data), size=(1,)).item()
            img, label = training_data[sample_idx]
            ax = axes[i, j]
            ax.imshow(img.squeeze(), cmap='gray')
            ax.set_title(label_maps[label])
            ax.axis('off')  # Hide the axis
    #
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()  # Display all images at once
    #
    # fig ,ax = plt.subplots(3,3)
    # ax = plt.figure(figsize=(8,8))
    # #looping
    # cols ,rows = 3 ,3
    # for i in range (1,cols * rows +1):
    #     sample_idx = torch.randint(len(training_data), size=(1,)).item()  # Random index
    #     img, label = training_data[sample_idx]  # Get image and label
    #     ax.add_subplot(rows, cols, i)  # Add subplot at position i
    #     plt.title(label_maps[label])  # Set the title to the class name
    #     plt.axis("off")  # Hide the axis
    #     plt.imshow(img.squeeze(),cmap ='BrBG')
    #     plt.tight_layout()
    #     plt.show()









if __name__=="__main__":
    main()





