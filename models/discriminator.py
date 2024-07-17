import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(nn.Conv2d(nc, ndf, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.LeakyReLU(0.2, inplace=True),
                                  nn.Conv2d(ndf, ndf * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, inplace=True),
                                  nn.Conv2d(ndf * 2, ndf * 4, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, inplace=True),
                                  nn.Conv2d(ndf * 4, ndf * 8, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, inplace=True),
                                  nn.Conv2d(ndf * 8, 1, kernel_size=3, stride=1, padding=0, bias=False),
                                  nn.Sigmoid()
                                  )

    def forward(self, x):
        return self.main(x)
