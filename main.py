import pandas as pd
import torch
from torch.utils.data import dataset
from torchvision import datasets,transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5),(0.5))
    ])

training_set = datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
        )
test_set = datasets.FashionMNIST(
        root="./data
        train=False,
        download=True,
        transform=transform
        )
image,label = training_set[0]
batch norm layers
train_loader = torch.utils.data.DataLoader(
        dataset=training_set,
        batch_size = 64,
        shuffle=True
        )
test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=64,
        shuffle=False
        )

EPOCH_COUNT = 8

for item in range(0,EPOCH_COUNT):
    print(item)
print(image.shape)

