import torch.nn as nn
import torchvision.models as models


class Generator(nn.Module):
    def __init__(self, input_noise_vector_size, feature_map_depth, number_of_channels) -> None:
        super(Generator, self).__init__()
        
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels=input_noise_vector_size,
                               out_channels=feature_map_depth * 8,
                               kernel_size=8,
                               stride=4,
                               padding=2, bias=False),
            nn.BatchNorm2d(feature_map_depth * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=feature_map_depth * 8, out_channels=feature_map_depth * 4, kernel_size=6,
                               stride=2, padding=0, bias=False),
            nn.BatchNorm2d(feature_map_depth * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=feature_map_depth * 4, out_channels=feature_map_depth * 2, kernel_size=4,
                               stride=2, padding=0, bias=False),
            nn.BatchNorm2d(feature_map_depth * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=feature_map_depth * 2, out_channels=number_of_channels, kernel_size=2,
                               stride=2, padding=0, bias=False),
            nn.Tanh()
        )
        


    def forward(self, x):
        return self.main(x)
