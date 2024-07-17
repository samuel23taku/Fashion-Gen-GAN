import pandas as pd
import torch
from torch.utils.data import dataset
from torchvision import datasets, transforms
import matplotlib
import torch.nn as nn
import torch.optim as optim
from models.discriminator import Discriminator
from models.generator import Generator
import utils.constants as Constants

matplotlib.use("Agg")
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])

training_set = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)
test_set = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)
image, label = training_set[0]

train_loader = torch.utils.data.DataLoader(
    dataset=training_set,
    batch_size=64,
    shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_set,
    batch_size=64,
    shuffle=False
)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


fixed_noise = torch.randn(64, Constants.NOISE_VECTOR_SIZE, 1, 1)

generator = Generator(Constants.NOISE_VECTOR_SIZE, Constants.NGF, Constants.NC)
discriminator = Discriminator(Constants.NC, Constants.NGF)

optimizerD = optim.Adam(discriminator.parameters(), lr=Constants.LEARNING_RATE)

'''
Weight initialization
'''
generator.apply(weights_init)
discriminator.apply(weights_init)

criterion = nn.BCELoss()  # for binary cross entropy for the discriminator

# for item in range(0,EPOCH_COUNT):
#     print(item)
# print(image.shape)
