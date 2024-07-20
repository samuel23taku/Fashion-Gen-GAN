import torch.nn as nn
import torchvision.models as models

from torchvision.models import ResNet18_Weights


class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        self.resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.conv1 = nn.Conv2d(in_channels=nc,out_channels=64,kernel_size=4,stride=2,padding=1,bias=False)
        num_features = self.resnet.fc.in_features
        for param in self.resnet.parameters():
            param.requires_grad = False

        self.resnet.fc = nn.Identity()
        self.fc = nn.Sequential(
                nn.Linear(num_features,1)
                ,nn.Sigmoid()
                )
    def forward(self, x):
        out = self.resnet(x)
        return self.fc(out)
