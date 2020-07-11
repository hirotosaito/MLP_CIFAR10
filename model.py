import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from matplotlib import pyplot as plt
import os

print("prepare the data set")

if(not os.path.exists('./data')):
    print("downloading data set")
    train_dataset = torchvision.datasets.CIFAR10(root       = './data',\
                                                 train      = True,\
                                                 transform  = transforms.ToTensor(),\
                                                 download   = True   )

    test_dataset  = torchvision.datasets.CIFAR10(root       = './data',\
                                                 train      = False  ,\
                                                 transform  = transforms.ToTensor,\
                                                 download   = True)
else:
    print("data already exists. no need to download")
    train_dataset = torchvision.datasets.CIFAR10(root       = './data',\
                                                 train      = True,\
                                                 transform  = transforms.ToTensor(),\
                                                 download   = False )

    test_dataset  = torchvision.datasets.CIFAR10(root       = './data',\
                                                 train      = False  ,\
                                                 transform  = transforms.ToTensor(),\
                                                 download   = False)

image , label = train_dataset[0]
print(type(train_dataset))
print(label)

train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = 64, shuffle = True,  num_workers = 2)
test_loader  = torch.utils.data.DataLoader(dataset = test_dataset,  batch_size = 64, shuffle = False, num_workers = 2)
num_classes  = 10 # CIFAR10 has 10 categories

class MLPNet(nn.Module):

    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 600)
        self.fc2 = nn.Linear(600,600)
        self.fc3 = nn.Linear(600, num_classes)
        self.dropout1 = nn.Dropout2d(0.2)
        self.dropout2 = nn.Dropout2d(0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        return F.relu(self.fc3(x))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
net    = MLPNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.01, momentum = 0.9, weight_decay = 5e-4)

num_epochs = 50

train_loss_list = []
train_acc_list  = []
val_loss_list   = []
val_acc_list    = []

for epoch in range(num_epochs):
    train_loss = 0
    train_acc  = 0
    val_loss   = 0
    val_acc    = 0

    net.train()
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.view(-1, 32*32*3).to(device), labels.to(device)
        optimizer.zero_grad()
        outputs     = net(images)
        loss        = criterion(outputs, labels)
        train_loss  += loss.item()
        train_acc   += (outputs.max(1)[1] == labels).sum().item()
        loss.backward()
        optimizer.step()
        print(i)
        print(outputs.max(1)[1])
        print(images.shape, labels, outputs.shape)

    avg_train_loss = train_loss / len(train_loader.dataset)
    avg_train_acc  = train_acc  / len(train_loader.dataset)

    net.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.view(-1, 32*32*3).to(device), labels.to(device)
            outputs   = net(images)
            loss      = criterion(outputs, labels)
            val_loss  += loss.item()
            val_acc   += (outputs.max(1)[1] == labels).sum().item()

    avg_val_loss = val_loss / len(test_loader.dataset)
    avg_val_acc  = val_acc  / len(test_loader.dataset)

    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    val_loss_list.append(val_loss)
    val_acc_list.append(val_acc)


fig = plt.figure()
ax  = fig.add_subplot()
ax.scatter(x,y)
ax.plot(range(num_epochs),train_acc_list)
ax.plot(range(num_epochs),val_acc_list)
fig.savefig('acc')


