import os
import time

import matplotlib
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import dataset
from torchvision import datasets, transforms

import utils.constants as Constants
from models.discriminator import Discriminator
from models.generator import Generator

matplotlib.use("Agg")

output_dir = './generated_images'
os.makedirs(output_dir, exist_ok=True)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])

training_set = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
test_set = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
image, label = training_set[0]

train_loader = torch.utils.data.DataLoader(dataset=training_set, batch_size=Constants.BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=500, shuffle=False)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


generator = Generator(Constants.NOISE_VECTOR_SIZE, Constants.NGF, Constants.NC)
discriminator = Discriminator(Constants.NC, Constants.NDF)
generator.apply(weights_init)

optimizerD = optim.Adam(discriminator.parameters(), lr=Constants.LEARNING_RATE)
optimizerG = optim.Adam(generator.parameters(), lr=Constants.LEARNING_RATE)

criterion = nn.BCELoss()  # for binary cross entropy for the discriminator
fixed_noise = torch.randn(64, Constants.NOISE_VECTOR_SIZE, 1, 1)

img_list = []
G_losses = []
D_losses = []
iters = 0

start_time = time.time()
for epoch in range(Constants.EPOCH_COUNT):
    epoch_start_time = time.time()
    for i, data in enumerate(train_loader):
        discriminator.zero_grad()
        batch_real_images = data[0]
        batch_size = batch_real_images.size(0)

        label = torch.full((batch_size,), Constants.LABEL_REAL_IMAGE, dtype=torch.float)
        output = discriminator(batch_real_images)
        output = output.view(batch_size, -1).mean(1)

        errorD_real = criterion(output, label)

        errorD_real.backward()

        D_x = output.mean().item()

        # Train disc on fake images

        noise = torch.randn(batch_size, Constants.NOISE_VECTOR_SIZE, 1, 1)
        fake = generator(noise)
        label.fill_(Constants.LABEL_FAKE_IMAGE)
        output = discriminator(fake.detach()).view(batch_size, -1).mean(1)
        errorD_fake = criterion(output, label)
        # Propagate gradients
        errorD_fake.backward()
        D_x_fake = output.mean().item()
        errorD = errorD_real + errorD_fake  # Add up errors from fake n real classications
        optimizerD.step()

        # Train Generator
        generator.zero_grad()
        label.fill_(Constants.LABEL_REAL_IMAGE)
        output = discriminator(fake).view(batch_size, -1).mean(1)
        errorG = criterion(output, label)
        errorG.backward()
        D_g_average_error = output.mean().item()
        optimizerG.step()

        # Logging
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f' % (
                epoch, Constants.EPOCH_COUNT, i, len(train_loader), errorD.item(), errorG.item(), D_x, D_x_fake,
                D_g_average_error))

        G_losses.append(errorG.item())
        D_losses.append(errorD.item())

        if (iters % 500 == 0) or ((epoch == Constants.EPOCH_COUNT - 1) and (i == len(train_loader) - 1)):
            with torch.no_grad():
                fake = generator(fixed_noise).detach().cpu()
                img_grid = (vutils.make_grid(fake, padding=2, normalize=True))
                output_path = os.path.join(output_dir, f'fake_images_epoch_{epoch}.png')

                # Save the image
                vutils.save_image(img_grid, output_path)

        iters += 1

    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time
    total_duration = epoch_end_time - start_time

    print('Epoch %d completed in %.2f seconds. Total training time: %.2f seconds.' % (
        epoch + 1, epoch_duration, total_duration))

total_training_time = time.time() - start_time
print('Total training completed in %.2f seconds.' % total_training_time)
